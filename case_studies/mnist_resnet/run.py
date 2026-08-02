"""Run enabled MNIST local-attribution methods against the already-downloaded
model and already-selected samples (see download.py, select_samples.py).
"""

import logging
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import importance
from local_ttest import local_ttest_scores
from model import MNISTResNetFlat, case_path
from reproducibility import capture_run_metadata

log = logging.getLogger(__name__)

INPUT_SPACE = "flattened raw MNIST pixels in row-major order, values in [0, 1]"
NEIGHBORHOOD_SPACES = {
    "pixel_l2": "flattened raw MNIST pixels",
    "resnet_final_embedding_l2": (
        "64-dimensional adaptive-average-pooled ResNet features immediately "
        "before the linear classifier"
    ),
}


def enabled_methods(cfg: DictConfig) -> list[str]:
    configured = OmegaConf.select(cfg, "methods.enabled")
    if configured is not None:
        return [str(name) for name in configured]

    methods = ["shap"]
    if OmegaConf.select(cfg, "integrated_gradients") is not None:
        methods.append("integrated_gradients")
    if OmegaConf.select(cfg, "local_ttest") is not None:
        methods.append("local_ttest")
    return methods


def neighborhood_space(distance: str) -> str:
    try:
        return NEIGHBORHOOD_SPACES[distance]
    except KeyError as exc:
        supported = ", ".join(sorted(NEIGHBORHOOD_SPACES))
        raise ValueError(
            f"Unsupported local_ttest.distance={distance!r}; expected one of: {supported}"
        ) from exc


def artifact_path(path: Path) -> str:
    """Represent an artifact relative to this case study when possible."""
    try:
        return str(path.resolve().relative_to(SCRIPT_DIR))
    except ValueError:
        return str(path.resolve())


def build_metadata(
    cfg: DictConfig,
    *,
    methods: list[str],
    n_samples: int,
    n_background: int,
    n_local_ttest_pool: int,
    outputs: dict,
    run_metadata: dict,
) -> dict:
    """Combine resolved configuration with facts observed during the run."""
    local_ttest_enabled = "local_ttest" in methods
    return {
        "schema_version": 1,
        "run": {"seed": int(cfg.seed), **run_metadata},
        "config": OmegaConf.to_container(cfg, resolve=True),
        "inputs": {
            "n_samples": n_samples,
            "n_shap_background": n_background,
            "n_local_ttest_pool": (
                n_local_ttest_pool if local_ttest_enabled else None
            ),
            "feature_space": INPUT_SPACE,
        },
        "attribution": {
            "enabled_methods": methods,
            "target": {
                "quantity": "model class probability",
                "class_column": str(cfg.explain.target),
            },
            "local_ttest_neighborhood_space": (
                neighborhood_space(str(cfg.local_ttest.distance))
                if local_ttest_enabled
                else None
            ),
        },
        "outputs": outputs,
    }


@hydra.main(version_base=None, config_path=".", config_name="importance")
def main(cfg: DictConfig) -> None:
    seed = int(cfg.seed)
    run_metadata = capture_run_metadata(SCRIPT_DIR)
    log.info(
        "Running with seed=%s, git_commit=%s; hydra config at %s",
        seed, run_metadata["git_commit"], run_metadata["hydra_output_dir"],
    )

    methods = enabled_methods(cfg)
    local_ttest_distance = str(cfg.local_ttest.distance)
    if "local_ttest" in methods:
        neighborhood_space(local_ttest_distance)  # validate before expensive work
        if int(cfg.local_ttest.n_pool) < int(cfg.local_ttest.n_neighbors):
            raise ValueError(
                "local_ttest.n_pool must be at least local_ttest.n_neighbors"
            )

    rng = np.random.default_rng(seed)
    results_dir = case_path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    sample_meta, X, shap_background, ttest_pool = importance.load_inputs(cfg, rng)
    model = MNISTResNetFlat.from_downloaded(
        model_source_dir=case_path(cfg.paths.model_source_dir),
        model_path=case_path(cfg.paths.model_path),
        batch_size=int(cfg.model.batch_size),
    )

    attributions = importance.attribute(
        X=X,
        sample_meta=sample_meta,
        background=shap_background,
        model=model,
        n_shap_samples=cfg.explain.n_samples,
        ig_steps=int(cfg.integrated_gradients.n_steps),
        ig_eps=float(cfg.integrated_gradients.eps),
        seed=seed,
    )

    if "local_ttest" in methods:
        ttest_pool_probs = model.predict_proba(ttest_pool)
        ttest_pool_pred = ttest_pool_probs.argmax(axis=1)
        if local_ttest_distance == "resnet_final_embedding_l2":
            X_neighbor_features = model.final_embeddings(X)
            pool_neighbor_features = model.final_embeddings(ttest_pool)
        else:
            X_neighbor_features = X
            pool_neighbor_features = ttest_pool

        ttest_scores = local_ttest_scores(
            X=X,
            background=ttest_pool,
            background_predicted_label=ttest_pool_pred,
            sample_predicted_label=sample_meta["predicted_label"].to_numpy(dtype=int),
            X_neighbor_features=X_neighbor_features,
            background_neighbor_features=pool_neighbor_features,
            n_neighbors=int(cfg.local_ttest.n_neighbors),
            min_group_size=int(cfg.local_ttest.min_group_size),
        )
        attributions["local_ttest"] = importance.attribution_frame(
            sample_meta, ttest_scores
        )

    attribution_files = {}
    for method in methods:
        path = results_dir / f"{method}_attributions.csv"
        attributions[method].to_csv(path, index=False)
        attribution_files[method] = artifact_path(path)
        log.info("Saved %s", path)

    raw_pixels_path = results_dir / "raw_pixels.csv"
    importance.attribution_frame(sample_meta, X).to_csv(raw_pixels_path, index=False)
    log.info("Saved %s", raw_pixels_path)

    metadata = build_metadata(
        cfg,
        methods=methods,
        n_samples=len(sample_meta),
        n_background=len(shap_background),
        n_local_ttest_pool=len(ttest_pool),
        outputs={
            "attributions": attribution_files,
            "raw_pixels": artifact_path(raw_pixels_path),
        },
        run_metadata=run_metadata,
    )
    metadata_path = results_dir / "importance_metadata.yaml"
    OmegaConf.save(OmegaConf.create(metadata), metadata_path)
    log.info("Wrote %s", metadata_path)


if __name__ == "__main__":
    main()
