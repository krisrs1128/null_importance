"""SHAP and minSHAP attributions for the TCGA BRCA random forest model.

Run with:
    python explain.py
    python explain.py --quick
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import load_config
from results import load_model
from explain import select_samples, attribute

BASE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Explain only a handful of patients")
    args = parser.parse_args()

    cfg = load_config(BASE)
    ecfg = cfg["explain"]
    seed = cfg["seeds"][0]
    rng = np.random.default_rng(seed)

    bundle = load_model(BASE / "results")
    model = bundle["model"]
    feat_names = bundle["feature_names"]

    X_df = pd.read_parquet(BASE / "data/X.parquet")
    y = pd.read_parquet(BASE / "data/y.parquet")["y"].values
    X = X_df.values
    sample_ids = X_df.index.astype(str).to_numpy()
    n = X.shape[0]

    y_pred_prob = model.predict_proba(X)[:, 1]

    if args.quick:
        sample_idx = select_samples(y, y_pred_prob)
    else:
        sample_idx = np.arange(n)

    meta = pd.DataFrame({
        "sample_idx": sample_idx,
        "sample_id": sample_ids[sample_idx],
        "y_true": y[sample_idx],
        "y_pred_prob": y_pred_prob[sample_idx],
    })

    shap_df, minshap_df = attribute(X, model, feat_names, sample_idx, ecfg,
                                    seed, rng)

    results = BASE / "results"
    meta.to_csv(results / "patient_meta.csv", index=False)
    shap_df.to_csv(results / "shap_attributions.csv", index=False)
    minshap_df.to_csv(results / "minshap_attributions.csv", index=False)
    print(f"Saved {len(sample_idx)} patients to results/")


if __name__ == "__main__":
    main()
