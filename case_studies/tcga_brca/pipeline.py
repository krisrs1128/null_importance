"""RF_concat pipeline For the multi-omics analysis from Novoloaca et al. 2024

https://doi.org/10.1093/bib/bbae331
https://github.com/bioaster/benchmark-integrative-methods

Follows the logic from benchmark-integrative-methods/RF.R + nestedCV.R:
  - Split data according to 5-fold stratified CV
  - OOB MCC to tune (min_samples_leaf, max_features) hyperparameters
  - Matthews Correlation Coefficient as final evaluation metric

Make sure to call download_tcga.R first so that we get all the data downloaded.

Run with,
    python pipeline.py
    python pipeline.py --n-reps 2 # run with two seeds
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

from data import load_data


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def mtry_grid(p):
    """Analog of getMtryGrid(p) from nestedCV.R: log10, sqrt, p/5, p/2."""
    return sorted({
        max(1, round(np.log10(p))),
        max(1, round(np.sqrt(p))),
        max(1, round(p / 5)),
        max(1, round(p / 2)),
    })


def nested_cv_rf(X, y, config, rng):
    """5-fold nested CV (tune then report tuned performance on held out fold)

    Returns (predictions, mcc, fold_params).
    """
    n_folds = config["cv"]["n_folds"]
    n_trees = config["rf"]["n_trees"]
    msl_grid = config["rf"]["min_samples_leaf"]
    mf_grid = mtry_grid(X.shape[1])

    param_grid = [(msl, mf) for msl in msl_grid for mf in mf_grid]

    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True,
        random_state=int(rng.integers(1, 2**31)),
    )
    X_arr, y_arr = X.values, y.values
    predictions = np.empty(len(y_arr), dtype=int)
    fold_params = []

    for train_idx, test_idx in skf.split(X_arr, y_arr):
        X_train, y_train = X_arr[train_idx], y_arr[train_idx]

        best_mcc, best_model, best_p = -np.inf, None, param_grid[0]
        for msl, mf in param_grid:
            rf = RandomForestClassifier(
                n_estimators=n_trees,
                max_features=mf,
                min_samples_leaf=msl,
                oob_score=True,
                n_jobs=-1,
                random_state=int(rng.integers(1, 2**31)),
            )
            rf.fit(X_train, y_train)
            oob_pred = rf.classes_[rf.oob_decision_function_.argmax(axis=1)]
            mcc = matthews_corrcoef(y_train, oob_pred)
            if mcc > best_mcc:
                best_mcc, best_model, best_p = mcc, rf, (msl, mf)

        predictions[test_idx] = best_model.predict(X_arr[test_idx])
        fold_params.append(best_p)

    return predictions, matthews_corrcoef(y_arr, predictions), fold_params


def fit_final_model(X, y, config, rng):
    """Train RF on all data with OOB-selected hyperparams (for interpretability use)."""
    msl_grid = config["rf"]["min_samples_leaf"]
    mf_grid = mtry_grid(X.shape[1])
    X_arr, y_arr = X.values, y.values

    best_mcc, best_model = -np.inf, None
    for msl in msl_grid:
        for mf in mf_grid:
            rf = RandomForestClassifier(
                n_estimators=config["rf"]["n_trees"],
                max_features=mf,
                min_samples_leaf=msl,
                oob_score=True,
                n_jobs=-1,
                random_state=int(rng.integers(1, 2**31)),
            )
            rf.fit(X_arr, y_arr)
            oob_pred = rf.classes_[rf.oob_decision_function_.argmax(axis=1)]
            mcc = matthews_corrcoef(y_arr, oob_pred)
            if mcc > best_mcc:
                best_mcc, best_model = mcc, rf

    print(f"Final model OOB MCC: {best_mcc:.4f}")
    return best_model


def main(n_reps=1):
    config = load_config()
    seeds = config["seeds"]

    # Load preprocessed data (or download + preprocess on first run)
    data_dir = Path(__file__).parent / "data"
    X_path, y_path = data_dir / "X.parquet", data_dir / "y.parquet"
    if X_path.exists() and y_path.exists():
        X = pd.read_parquet(X_path)
        y = pd.read_parquet(y_path)["y"]
        y_labels = config["outcome"]["classes"]
    else:
        print("Preprocessing raw data (first run) …")
        X, y, y_labels, _ = load_data(config)
        data_dir.mkdir(exist_ok=True)
        X.to_parquet(X_path)
        y.to_frame().to_parquet(y_path)

    print(f"Dataset: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Classes: {y_labels}  counts: {dict(y.value_counts())}\n")

    # Nested CV across requested repetitions
    mccs = []
    all_preds = []
    for i, seed in enumerate(seeds[:n_reps]):
        rng = np.random.default_rng(seed)
        preds, mcc, fold_params = nested_cv_rf(X, y, config, rng)
        mccs.append(mcc)
        all_preds.append(preds)
        print(f"Rep {i+1:2d}  seed={seed}  MCC={mcc:.4f}  fold_params={fold_params}")

    print(f"\nMean MCC over {n_reps} rep(s): {np.mean(mccs):.4f} ± {np.std(mccs):.4f}")

    # Save CV predictions from first rep
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    pd.DataFrame({"y_true": y.values, "y_pred": all_preds[0]}).to_csv(
        results_dir / "cv_predictions.csv", index=False
    )
    pd.Series(mccs, name="mcc").to_csv(results_dir / "mccs.csv", index=False)

    # Final model on all data (for SHAP / interpretability analysis)
    rng_final = np.random.default_rng(seeds[0])
    final_model = fit_final_model(X, y, config, rng_final)
    with open(results_dir / "final_model.pkl", "wb") as f:
        pickle.dump(
            {"model": final_model, "feature_names": list(X.columns), "y_labels": y_labels},
            f,
        )
    print(f"Saved final model → {results_dir / 'final_model.pkl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int, default=1,
                        help="Number of seeds/repetitions")
    args = parser.parse_args()
    main(n_reps=args.n_reps)
