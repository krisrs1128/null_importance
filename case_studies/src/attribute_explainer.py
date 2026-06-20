"""Compute SHAP and minSHAP attributions for a fitted classifier."""

import sys
from pathlib import Path

# explain module imports axiom_interp from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd

from config import load_config
from results import load_model
from explain import select_samples, attribute


def run(base_dir, quick=False):
    cfg = load_config(base_dir)
    ecfg = cfg["explain"]
    seed = cfg["seeds"][0]
    rng = np.random.default_rng(seed)

    bundle = load_model(base_dir / "results")
    model = bundle["model"]
    feat_names = bundle["feature_names"]

    X_df = pd.read_parquet(base_dir / "data/X.parquet")
    y = pd.read_parquet(base_dir / "data/y.parquet")["y"].values
    X = X_df.values
    sample_ids = X_df.index.astype(str).to_numpy()
    n = X.shape[0]

    y_pred_prob = model.predict_proba(X)[:, 1]

    if quick:
        sample_idx = select_samples(y, y_pred_prob)
    else:
        sample_idx = np.arange(n)

    meta = pd.DataFrame({
        "sample_idx": sample_idx,
        "sample_id": sample_ids[sample_idx],
        "y_true": y[sample_idx],
        "y_pred_prob": y_pred_prob[sample_idx],
    })

    shap_df, minshap_df = attribute(
        X, model, feat_names, sample_idx, ecfg, seed, rng
    )

    results = base_dir / "results"
    meta.to_csv(results / "patient_meta.csv", index=False)
    shap_df.to_csv(results / "shap_attributions.csv", index=False)
    minshap_df.to_csv(results / "minshap_attributions.csv", index=False)
    print(f"Saved {len(sample_idx)} samples to results/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    run(args.study_dir.resolve(), quick=args.quick)
