"""Compute local SHAP and minSHAP pixel importances for selected MNIST samples.

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
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Adapt case_studies/src/explain.py::attribute to flat MNIST pixels."""
    explainers = {
        "shap": presets.shap(background, n_orderings, seed),
        "minshap": presets.minshap(background, n_orderings, seed),
    }
    functions_by_label = {}
    scores = {name: [] for name in explainers}

    for i, row in sample_meta.iterrows():
        target = int(row["target_label"])
        if target not in functions_by_label:
            functions_by_label[target] = model.class_probability(target)

        # minSHAP changes only the aggregator, so it should reuse SHAP's tensor.
        reset_compute_count()
        for name, explainer in explainers.items():
            scores[name].append(explainer.explain(functions_by_label[target], X[i]).as_array())
        assert compute_count() == 1, "minSHAP should reuse the cached SHAP tensor"
        log.info("[%s/%s] sample_index=%s target=%s", i + 1, len(X), row["sample_index"], target)

    meta = sample_meta[META_COLS].reset_index(drop=True)
    return {
        name: pd.concat(
            [meta, pd.DataFrame(values, columns=pixel_feature_names())],
            axis=1,
        )
        for name, values in scores.items()
    }


@hydra.main(version_base=None, config_path=".", config_name="importance")
def main(cfg: DictConfig) -> None:
    rng = np.random.default_rng(int(cfg.seed))
    output_dir = case_path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_meta, X, background, background_idx = load_inputs(cfg, rng)
    model = MNISTResNetFlat.from_downloaded(
        model_source_dir=case_path(cfg.paths.model_source_dir),
        model_path=case_path(cfg.paths.model_path),
        batch_size=int(cfg.model.batch_size),
    )

    attributions = attribute(
        X=X,
        sample_meta=sample_meta,
        background=background,
        model=model,
        n_orderings=int(cfg.explain.n_orderings),
        seed=int(cfg.seed),
    )

    for name, df in attributions.items():
        df.to_csv(output_dir / f"{name}_attributions.csv", index=False)

    pixels = np.arange(N_PIXELS)
    pd.DataFrame({"pixel_index": pixels, "row": pixels // 28, "col": pixels % 28}).to_csv(
        output_dir / "pixel_feature_map.csv", index=False
    )
    pd.DataFrame({"background_index": background_idx}).to_csv(
        output_dir / "background_meta.csv", index=False
    )

    metadata = {
        "seed": int(cfg.seed),
        "n_samples": int(len(sample_meta)),
        "n_background": int(len(background)),
        "n_orderings": int(cfg.explain.n_orderings),
        "methods": ["shap", "minshap"],
        "input_space": "flattened raw MNIST pixels in row-major order, values in [0, 1]",
        "target": f"probability of {cfg.explain.target}",
    }
    OmegaConf.save(OmegaConf.create(metadata), output_dir / "importance_metadata.yaml")
    log.info("Saved SHAP/minSHAP attributions to %s", output_dir)


if __name__ == "__main__":
    main()
