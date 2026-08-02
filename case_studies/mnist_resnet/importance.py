"""Reusable attribution helpers for selected MNIST samples."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from omegaconf import DictConfig

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "src"))

from axiom_interp import presets
from model import (
    MNISTResNetFlat,
    N_PIXELS,
    case_path,
    load_mnist_flat,
    pixel_feature_names,
)

log = logging.getLogger(__name__)
logging.getLogger("shap").setLevel(logging.WARNING)
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


def _sample_training_rows(
    X_train: np.ndarray,
    rng: np.random.Generator,
    n_rows: int,
) -> np.ndarray:
    n_rows = min(int(n_rows), len(X_train))
    row_idx = rng.choice(len(X_train), n_rows, replace=False)
    return X_train[row_idx]


def load_inputs(cfg: DictConfig, rng: np.random.Generator):
    """Load selected test rows plus separate training pools for each method."""
    sample_meta = pd.read_csv(case_path(cfg.paths.sample_meta))
    if cfg.run.sample_limit is not None:
        sample_meta = sample_meta.head(int(cfg.run.sample_limit)).copy()
    sample_meta["target_label"] = sample_meta[str(cfg.explain.target)].astype(int)

    data_dir = case_path(cfg.paths.data_dir)
    X_test, _ = load_mnist_flat(data_dir, train=False)
    X_train, _ = load_mnist_flat(data_dir, train=True)
    shap_background = _sample_training_rows(X_train, rng, int(cfg.explain.n_background))
    ttest_pool = _sample_training_rows(X_train, rng, int(cfg.local_ttest.n_pool))

    sample_meta = sample_meta.reset_index(drop=True)
    sample_idx = sample_meta["sample_index"].to_numpy(dtype=int)
    return (
        sample_meta,
        X_test[sample_idx],
        shap_background,
        ttest_pool,
    )


def attribute(
    X: np.ndarray,
    sample_meta: pd.DataFrame,
    background: np.ndarray,
    model: MNISTResNetFlat,
    n_shap_samples: int | str,
    ig_steps: int,
    ig_eps: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Adapt case_studies/src/explain.py::attribute to flat MNIST pixels."""
    np.random.seed(seed)

    ig_explainer = presets.integrated_gradients(np.zeros(N_PIXELS), ig_steps, ig_eps)
    functions_by_label = {}
    shap_explainers_by_label = {}
    scores = {"shap": [], "integrated_gradients": []}

    for i, row in sample_meta.iterrows():
        target = int(row["target_label"])
        if target not in functions_by_label:
            functions_by_label[target] = model.class_probability(target)
            shap_explainers_by_label[target] = shap.KernelExplainer(
                functions_by_label[target], background
            )

        f = functions_by_label[target]
        shap_values = shap_explainers_by_label[target].shap_values(
            X[i : i + 1], nsamples=n_shap_samples, silent=True
        )
        scores["shap"].append(np.asarray(shap_values).reshape(-1))
        scores["integrated_gradients"].append(ig_explainer.explain(f, X[i]).as_array())
        log.info("[%s/%s] sample_index=%s target=%s", i + 1, len(X), row["sample_index"], target)

    return {name: attribution_frame(sample_meta, values) for name, values in scores.items()}
