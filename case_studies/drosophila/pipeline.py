"""Random forest prediction pipeline for Drosophila enhancer activity.

This is the same prediction problem as S. Basu, ... B. Yu "Iterative Random
Forest" paper. We're using a simpler SHAP-based interpretation. The overall
implementation mimics the tcga_brca case study.

  - 5-fold stratified nested CV
  - OOB hyperparameter tuning (min_samples_leaf, max_features)
  - final model refit on all samples. This is what we explain later.

Run with:
    python pipeline.py
    python pipeline.py --n-reps 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import load_config
from rf import nested_cv, fit_final
from results import save_results

from data import load_data

BASE = Path(__file__).parent


def main(n_reps=1):
    config = load_config(BASE)
    seeds = config["seeds"]

    data_dir = BASE / "data"
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

    mccs, all_preds = [], []
    for i, seed in enumerate(seeds[:n_reps]):
        rng = np.random.default_rng(seed)
        preds, mcc, fold_params = nested_cv(X, y, config, rng)
        mccs.append(mcc)
        all_preds.append(preds)
        print(f"Rep {i + 1:2d}  seed={seed}  MCC={mcc:.4f}  fold_params={fold_params}")

    print(f"\nMean MCC over {n_reps} rep(s): {np.mean(mccs):.4f} ± {np.std(mccs):.4f}")

    rng_final = np.random.default_rng(seeds[0])
    final_model, best_params = fit_final(X, y, config, rng_final)
    print(f"Final model OOB MCC  params={best_params}")

    save_results(BASE / "results", X, y, all_preds, mccs, final_model,
                 best_params, y_labels)
    print(f"Saved final model → {BASE / 'results' / 'final_model.pkl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int, default=1,
                        help="Number of seeds/repetitions")
    args = parser.parse_args()
    main(n_reps=args.n_reps)
