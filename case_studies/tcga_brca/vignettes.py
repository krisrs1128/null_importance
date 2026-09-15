"""Per-feature vignette panels for the TCGA BRCA importance sweep.

Find the features with the largest absolute PC2 values and generate the partial
Run after sweep.py:

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


def _has_nonzero_variance(values):
    """Whether the finite entries have an estimable, nonzero variance."""
    finite = values[np.isfinite(values)]
    return len(finite) > 1 and np.var(finite, ddof=1) > 0


def pc2_loading_features(
    scores, feature_names, n_features, normalize_mass=True
):
    """Features with the largest absolute method-profile PC2 loadings.

    This mirrors ``importance_pca()`` in the R visualization helpers. We first
    average over seeds, remove constant methods, normalize each method's
    absolute scores to unit mass, and perform PCA with methods as rows and
    features as columns.
    """
    matrix = (
        scores.groupby(["feature", "method"], sort=False)["importance"]
        .mean()
        .unstack("method")
        .reindex(feature_names)
    )
    values = matrix.to_numpy(dtype=float)

    keep_methods = np.apply_along_axis(_has_nonzero_variance, 0, values)
    values = values[:, keep_methods]
    keep_features = np.apply_along_axis(_has_nonzero_variance, 1, values)
    values = values[keep_features]
    kept_names = matrix.index.to_numpy()[keep_features]

    if normalize_mass:
        totals = np.nansum(np.abs(values), axis=0)
        pca_values = np.abs(values) / totals
    else:
        means = np.nanmean(values, axis=0)
        scales = np.nanstd(values, axis=0, ddof=1)
        pca_values = (values - means) / scales
    pca_values[~np.isfinite(pca_values)] = 0

    # R's prcomp(t(mat), center = FALSE, scale. = FALSE) is this SVD. The
    # rows of vh are the feature loadings for successive principal components.
    _, _, vh = np.linalg.svd(pca_values.T, full_matrices=False)
    if vh.shape[0] < 2:
        raise ValueError("At least two nonconstant method profiles are required")
    order = np.argsort(-np.abs(vh[1]), kind="stable")[:n_features]
    return kept_names[order].tolist()


def _partial_dependence(model, X, feature_index, grid_resolution):
    """Return both the average and individual partial dependence curves."""
    return pd_func(
        model, X, features=[feature_index],
        grid_resolution=grid_resolution, kind="both",
        response_method="predict_proba",
    )


def pdp_and_ice_curves(model, X_df, features, grid_resolution):
    """Return mean PDP and individual conditional expectation data frames."""
    X = X_df.to_numpy(dtype=float)
    feature_names = list(X_df.columns)
    pdp_rows = []
    ice_rows = []
    for feature in features:
        j = feature_names.index(feature)
        result = _partial_dependence(model, X, j, grid_resolution)
        grid_values = result["grid_values"][0]
        average = np.asarray(result["average"])[0]
        individual = np.asarray(result["individual"])[0]

        for grid_value, pdp_value in zip(grid_values, average):
            pdp_rows.append({
                "feature": feature,
                "grid_value": grid_value,
                "pdp_value": pdp_value,
            })
        for sample_index, curve in enumerate(individual):
            for grid_value, ice_value in zip(grid_values, curve):
                ice_rows.append({
                    "feature": feature,
                    "sample_index": sample_index,
                    "grid_value": grid_value,
                    "ice_value": ice_value,
                })

    return pd.DataFrame(pdp_rows), pd.DataFrame(ice_rows)


def pdp_curves(model, X_df, features, grid_resolution):
    """Partial dependence of the fitted model along each feature's grid."""
    pdp, _ = pdp_and_ice_curves(model, X_df, features, grid_resolution)
    return pdp


def prediction_points(model, X_df, y, features):
    """Observed feature values and fitted class-1 probabilities."""
    X = X_df.to_numpy(dtype=float)
    probability = model.predict_proba(X)[:, 1]
    predicted_class = model.predict(X)
    rows = []
    for feature in features:
        values = zip(X_df[feature], y, probability, predicted_class)
        for sample_index, (
            feature_value, actual, probability_value, hard_prediction
        ) in enumerate(values):
            rows.append({
                "feature": feature,
                "sample_index": sample_index,
                "feature_value": feature_value,
                "prediction": probability_value,
                "predicted_class": hard_prediction,
                "actual_class": actual,
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
    scores = scores.loc[
        (scores["dataset"] == cfg_dict["dataset"])
        & (scores["n"] == len(X_df))
        & (scores["response_type"] == cfg_dict["response_type"])
        & (scores["seed"].isin(cfg_dict["seeds"]))
    ]
    normalize_mass = cfg_dict["visualization"].get(
        "pca_normalize_mass", True
    )
    features = pc2_loading_features(
        scores, feature_names, cfg_dict["vignettes"]["n_features"],
        normalize_mass,
    )

    feature_frame = X_df[features].copy()
    feature_frame.insert(0, "y", y)
    feature_frame.to_csv(results_dir / "vignette_features.csv", index=False)

    curves, ice = pdp_and_ice_curves(
        bundle["model"], X_df, features, cfg_dict["pdp"]["grid_resolution"]
    )
    curves.to_csv(results_dir / "vignette_pdp.csv", index=False)
    ice.to_csv(results_dir / "vignette_ice.csv", index=False)

    points = prediction_points(bundle["model"], X_df, y, features)
    points.to_csv(results_dir / "vignette_predictions.csv", index=False)
    log.info(f"Saved vignette data for {len(features)} features")


if __name__ == "__main__":
    main()
