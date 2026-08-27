"""Per-feature vignette panels for the TCGA BRCA importance sweep.

Picks the features whose importance profiles disagree most across methods and
saves the raw values and partial dependence curves needed to draw them. Run
after sweep.py:

    python case_studies/tcga_brca/vignettes.py
"""

import logging
import sys
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.inspection import partial_dependence as pd_func

log = logging.getLogger(__name__)
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir.parent / "src"))
sys.path.insert(0, str(_script_dir.parents[1] / "src"))
from evaluation import load_scores
from importance import METHODS
from results import load_model


def disagreement_features(scores, feature_names, n_features):
    """Features whose importance varies most across methods.

    Scores are averaged over seeds, standardized within method so the methods'
    different units are comparable, then ranked by their variance across
    methods.
    """
    matrix = (
        scores.groupby(["feature", "method"], sort=False)["importance"]
        .mean()
        .unstack("method")
        .reindex(feature_names)
    )
    values = matrix.to_numpy(dtype=float)
    standardized = (values - values.mean(axis=0)) / (values.std(axis=0) + 1e-12)
    spread = np.nanvar(standardized, axis=1)
    order = np.argsort(spread)[::-1][:n_features]
    return [matrix.index[i] for i in order]


def pdp_curves(model, X_df, features, grid_resolution):
    """Partial dependence of the fitted model along each feature's grid."""
    X = X_df.to_numpy(dtype=float)
    feature_names = list(X_df.columns)
    rows = []
    for feature in features:
        j = feature_names.index(feature)
        result = pd_func(
            model, X, features=[j],
            grid_resolution=grid_resolution, kind="average",
        )
        for grid_value, pdp_value in zip(
            result["grid_values"][0], result["average"][0]
        ):
            rows.append({
                "feature": feature,
                "grid_value": grid_value,
                "pdp_value": pdp_value,
            })
    return pd.DataFrame(rows)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    data_dir = _script_dir / "data"
    results_dir = _script_dir / "results"

    X_df = pd.read_parquet(data_dir / "X.parquet")
    y = pd.read_parquet(data_dir / "y.parquet").iloc[:, 0].values
    bundle = load_model(results_dir)
    feature_names = bundle["feature_names"]
    X_df = X_df[feature_names]

    methods = [
        name for name in METHODS if cfg_dict["methods"].get(name, False)
    ]
    scores = load_scores(results_dir, methods)
    features = disagreement_features(
        scores, feature_names, cfg_dict["vignettes"]["n_features"]
    )

    feature_frame = X_df[features].copy()
    feature_frame.insert(0, "y", y)
    feature_frame.to_csv(results_dir / "vignette_features.csv", index=False)

    curves = pdp_curves(
        bundle["model"], X_df, features, cfg_dict["pdp"]["grid_resolution"]
    )
    curves.to_csv(results_dir / "vignette_pdp.csv", index=False)
    log.info(f"Saved vignette data for {len(features)} features")


if __name__ == "__main__":
    main()
