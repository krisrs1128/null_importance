"""Reusable attribution helpers for selected MNIST samples."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "src"))

from axiom_interp import presets
from axiom_interp.streaming import (
    fixed_background_sample,
    marginal_minshap_batched,
)
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
    saliency_eps: float,
    seed: int,
    methods: list[str] | None = None,
    gradient_x_input_eps: float = 1e-5,
) -> dict[str, pd.DataFrame]:
    """Adapt case_studies/src/explain.py::attribute to flat MNIST pixels."""
    np.random.seed(seed)

    enabled = (
        {"shap", "integrated_gradients", "gradient_x_input", "saliency"}
        if methods is None
        else set(methods)
    )
    shap_module = None
    if "shap" in enabled:
        import shap as shap_module

        logging.getLogger("shap").setLevel(logging.WARNING)

    baseline = np.zeros(N_PIXELS)
    ig_explainer = (
        presets.integrated_gradients(baseline, ig_steps, ig_eps)
        if "integrated_gradients" in enabled
        else None
    )
    gradient_x_input_explainer = (
        presets.gradient_x_input(baseline, gradient_x_input_eps)
        if "gradient_x_input" in enabled
        else None
    )
    saliency_explainer = (
        presets.saliency(saliency_eps) if "saliency" in enabled else None
    )
    functions_by_label = {}
    shap_explainers_by_label = {}
    scores = {
        name: []
        for name in [
            "shap",
            "integrated_gradients",
            "gradient_x_input",
            "saliency",
        ]
        if name in enabled
    }

    for i, row in sample_meta.iterrows():
        target = int(row["target_label"])
        if target not in functions_by_label:
            functions_by_label[target] = model.class_probability(target)
            if "shap" in enabled:
                shap_explainers_by_label[target] = shap_module.KernelExplainer(
                    functions_by_label[target], background
                )

        f = functions_by_label[target]
        if "shap" in enabled:
            shap_values = shap_explainers_by_label[target].shap_values(
                X[i : i + 1], nsamples=n_shap_samples, silent=True, l1_reg=0
            )
            scores["shap"].append(np.asarray(shap_values).reshape(-1))
        if ig_explainer is not None:
            scores["integrated_gradients"].append(
                ig_explainer.explain(f, X[i]).as_array()
            )
        if gradient_x_input_explainer is not None:
            scores["gradient_x_input"].append(
                gradient_x_input_explainer.explain(f, X[i]).as_array()
            )
        if saliency_explainer is not None:
            scores["saliency"].append(saliency_explainer.explain(f, X[i]).as_array())
        log.info(
            "[%s/%s] sample_index=%s target=%s",
            i + 1,
            len(X),
            row["sample_index"],
            target,
        )

    return {
        name: attribution_frame(sample_meta, values)
        for name, values in scores.items()
    }


def attribute_marginalminshap(
    X: np.ndarray,
    sample_meta: pd.DataFrame,
    background: np.ndarray,
    model: MNISTResNetFlat,
    n_orderings: int,
    seed: int,
    background_eval_size: int | str | None = None,
    prefix_batch_size: int = 32,
) -> pd.DataFrame:
    """Marginal minSHAP with fixed-background batched prefix evaluation."""
    n_orderings = int(n_orderings)
    prefix_batch_size = int(prefix_batch_size)
    background_eval = fixed_background_sample(background, background_eval_size, seed)
    log.info(
        "Running marginalminshap with n_orderings=%s, background_eval_rows=%s, "
        "prefix_batch_size=%s",
        n_orderings,
        len(background_eval),
        prefix_batch_size,
    )

    scores = []

    for i, row in sample_meta.iterrows():
        target = int(row["target_label"])
        f = model.class_probability(target)
        sample_scores = marginal_minshap_batched(
            f,
            X[i],
            background_eval,
            n_orderings=n_orderings,
            seed=seed,
            prefix_batch_size=prefix_batch_size,
        )
        scores.append(sample_scores)
        log.info(
            "[marginalminshap %s/%s] sample_index=%s target=%s",
            i + 1, len(X), row["sample_index"], target,
        )

    return attribution_frame(sample_meta, np.asarray(scores))
