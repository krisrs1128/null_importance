"""I/O for model training and explanation."""

import pickle
from pathlib import Path
import pandas as pd


def save_results(results_dir, X, y, all_preds, mccs, aucs, final_model,
                 best_params, y_labels):
    """Write standard pipeline outputs.

    Parameters
    ----------
    results_dir : Path
    X : DataFrame
        Used for sample IDs (index) and feature names (columns).
    y : Series
    all_preds : list of ndarray
        Per-repetition predictions; first rep written to CSV.
    mccs : list of float
    aucs : list of float
    final_model : RandomForestClassifier
    best_params : tuple of (min_samples_leaf, max_features)
    y_labels : list of str
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    pd.DataFrame(
        {"y_true": y.values, "y_pred": all_preds[0]},
        index=X.index,
    ).to_csv(results_dir / "cv_predictions.csv", index_label="sample_id")

    pd.Series(mccs, name="mcc").to_csv(results_dir / "mccs.csv", index=False)
    pd.Series(aucs, name="auc").to_csv(results_dir / "auc.csv", index=False)

    with open(results_dir / "final_model.pkl", "wb") as f:
        pickle.dump({
            "model": final_model,
            "feature_names": list(X.columns),
            "sample_ids": list(X.index.astype(str)),
            "best_params": best_params,
            "y_labels": y_labels,
        }, f)


def load_model(results_dir):
    """Load the pickled model bundle.

    Parameters
    ----------
    results_dir : Path

    Returns
    -------
    dict
        Keys: model, feature_names, sample_ids, y_labels.
    """
    with open(Path(results_dir) / "final_model.pkl", "rb") as f:
        return pickle.load(f)
