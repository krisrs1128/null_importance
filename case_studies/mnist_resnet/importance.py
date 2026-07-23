"""Compute local pixel importances for selected MNIST samples.

Run from this case-study directory, after download.py and select_samples.py:
    python importance.py
"""

import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "src"))

from axiom_interp import compute_count, presets, reset_compute_count
from local_ttest import local_ttest_scores
from model import (
    MNISTResNetFlat,
    N_PIXELS,
    case_path,
    load_mnist_flat,
    pixel_feature_names,
)

log = logging.getLogger(__name__)
META_COLS = [
        "sample_index",
        "true_label",
        "predicted_label",
        "predicted_probability",
        "correct",
        "target_label",
    ]

def attribution_frame(sample_meta: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    meta = sample_meta[META_COLS].reset_index(drop=True)
    return pd.concat([meta, pd.DataFrame(scores, columns=pixel_feature_names())], axis=1)


def load_inputs(cfg: DictConfig, rng: np.random.Generator):
    """Load the 100 selected test rows plus a random flattened training background."""
    sample_meta = pd.read_csv(case_path(cfg.paths.sample_meta))
    if cfg.run.sample_limit is not None:
        sample_meta = sample_meta.head(int(cfg.run.sample_limit)).copy()
    sample_meta["target_label"] = sample_meta[str(cfg.explain.target)].astype(int)

    data_dir = case_path(cfg.paths.data_dir)
    X_test, _ = load_mnist_flat(data_dir, train=False)
    X_train, _ = load_mnist_flat(data_dir, train=True)
    n_background = min(int(cfg.explain.n_background), len(X_train))
    background_idx = rng.choice(len(X_train), n_background, replace=False)

    sample_meta = sample_meta.reset_index(drop=True)
    sample_idx = sample_meta["sample_index"].to_numpy(dtype=int)
    return sample_meta, X_test[sample_idx], X_train[background_idx], background_idx


def attribute(
    X: np.ndarray,
    sample_meta: pd.DataFrame,
    background: np.ndarray,
    model: MNISTResNetFlat,
    n_orderings: int,
    ig_steps: int,
    ig_eps: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Adapt case_studies/src/explain.py::attribute to flat MNIST pixels."""
    shap_explainer = presets.shap(background, n_orderings, seed)
    minshap_explainer = presets.minshap(background, n_orderings, seed)
    ig_explainer = presets.integrated_gradients(np.zeros(N_PIXELS), ig_steps, ig_eps)
    functions_by_label = {}
    scores = {"shap": [], "minshap": [], "integrated_gradients": []}

    for i, row in sample_meta.iterrows():
        target = int(row["target_label"])
        if target not in functions_by_label:
            functions_by_label[target] = model.class_probability(target)

        # minSHAP changes only the aggregator, so it should reuse SHAP's tensor.
        reset_compute_count()
        f = functions_by_label[target]
        scores["shap"].append(shap_explainer.explain(f, X[i]).as_array())
        scores["minshap"].append(minshap_explainer.explain(f, X[i]).as_array())
        assert compute_count() == 1, "minSHAP should reuse the cached SHAP tensor"
        scores["integrated_gradients"].append(ig_explainer.explain(f, X[i]).as_array())
        log.info("[%s/%s] sample_index=%s target=%s", i + 1, len(X), row["sample_index"], target)

    return {name: attribution_frame(sample_meta, values) for name, values in scores.items()}


@hydra.main(version_base=None, config_path=".", config_name="importance")
def main(cfg: DictConfig) -> None:
    if str(cfg.local_ttest.distance) != "pixel_l2":
        raise ValueError("local_ttest.distance currently supports only 'pixel_l2'")

    rng = np.random.default_rng(int(cfg.seed))
    output_dir = case_path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_meta, X, background, background_idx = load_inputs(cfg, rng)
    model = MNISTResNetFlat.from_downloaded(
        model_source_dir=case_path(cfg.paths.model_source_dir),
        model_path=case_path(cfg.paths.model_path),
        batch_size=int(cfg.model.batch_size),
    )

    background_probs = model.predict_proba(background)
    background_pred = background_probs.argmax(axis=1)

    attributions = attribute(
        X=X,
        sample_meta=sample_meta,
        background=background,
        model=model,
        n_orderings=int(cfg.explain.n_orderings),
        ig_steps=int(cfg.integrated_gradients.n_steps),
        ig_eps=float(cfg.integrated_gradients.eps),
        seed=int(cfg.seed),
    )
    ttest_scores, neighbor_df = local_ttest_scores(
        X=X,
        background=background,
        background_predicted_label=background_pred,
        sample_predicted_label=sample_meta["predicted_label"].to_numpy(dtype=int),
        sample_index=sample_meta["sample_index"].to_numpy(dtype=int),
        background_index=background_idx,
        n_neighbors=int(cfg.local_ttest.n_neighbors),
        min_group_size=int(cfg.local_ttest.min_group_size),
    )
    attributions["local_ttest"] = attribution_frame(sample_meta, ttest_scores)

    for name, df in attributions.items():
        df.to_csv(output_dir / f"{name}_attributions.csv", index=False)

    pixels = np.arange(N_PIXELS)
    pd.DataFrame({"pixel_index": pixels, "row": pixels // 28, "col": pixels % 28}).to_csv(
        output_dir / "pixel_feature_map.csv", index=False
    )
    neighbor_df.to_csv(output_dir / "local_ttest_neighbors.csv", index=False)
    pd.DataFrame(
        {
            "background_index": background_idx,
            "predicted_label": background_pred,
            "predicted_probability": background_probs.max(axis=1),
        }
    ).to_csv(output_dir / "background_meta.csv", index=False)

    methods = list(attributions)
    metadata = {
        "seed": int(cfg.seed),
        "n_samples": int(len(sample_meta)),
        "n_background": int(len(background)),
        "n_orderings": int(cfg.explain.n_orderings),
        "methods": methods,
        "integrated_gradients": {
            "baseline": "zero_image",
            "n_steps": int(cfg.integrated_gradients.n_steps),
            "eps": float(cfg.integrated_gradients.eps),
        },
        "local_ttest": {
            "distance": str(cfg.local_ttest.distance),
            "n_neighbors": int(cfg.local_ttest.n_neighbors),
            "min_group_size": int(cfg.local_ttest.min_group_size),
            "split": "background predicted label equals selected sample predicted label vs other labels",
        },
        "input_space": "flattened raw MNIST pixels in row-major order, values in [0, 1]",
        "target": f"probability of {cfg.explain.target}",
    }
    OmegaConf.save(OmegaConf.create(metadata), output_dir / "importance_metadata.yaml")
    log.info("Saved attribution files to %s", output_dir)


if __name__ == "__main__":
    main()
