"""Run download, sample selection, and enabled MNIST importance methods."""

import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import importance
import select_samples
import download
from local_ttest import local_ttest_scores
from model import MNISTResNetFlat, N_PIXELS, case_path, pixel_feature_names
from reproducibility import capture_run_metadata

log = logging.getLogger(__name__)


def metadata_path(path: Path) -> str:
    try:
        return str(path.relative_to(SCRIPT_DIR))
    except ValueError:
        return str(path)


def enabled_methods(cfg: DictConfig) -> list[str]:
    configured = OmegaConf.select(cfg, "methods.enabled")
    if configured is not None:
        return [str(name) for name in configured]

    methods = ["shap", "minshap"]
    if OmegaConf.select(cfg, "integrated_gradients") is not None:
        methods.append("integrated_gradients")
    if OmegaConf.select(cfg, "local_ttest") is not None:
        methods.append("local_ttest")
    return methods


@hydra.main(version_base=None, config_path=".", config_name="importance")
def main(cfg: DictConfig) -> None:
    seed = int(cfg.seed)
    run_metadata = capture_run_metadata(SCRIPT_DIR)
    log.info(
        "Running with seed=%s, git_commit=%s; hydra config at %s",
        seed,
        run_metadata["git_commit"],
        run_metadata["hydra_output_dir"],
    )

    download_cfg = OmegaConf.load(SCRIPT_DIR / "config.yaml")
    for key in ("data_dir", "results_dir", "model_path", "model_source_dir"):
        download_cfg.paths[key] = cfg.paths[key]
    download_cfg.seed = seed
    if "download" in cfg:
        download_cfg.download = OmegaConf.merge(download_cfg.download, cfg.download)

    download._set_seed(seed)
    download_metadata = {
        "model": download.download_model(download_cfg),
        "data": download.download_mnist_test(download_cfg),
    }

    sample_meta = select_samples.select_samples(select_samples.predict_test_set(download_cfg))
    sample_path = case_path(cfg.paths.sample_meta)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_meta.to_csv(sample_path, index=False)
    log.info("Saved %s selected samples to %s", len(sample_meta), sample_path)

    methods = enabled_methods(cfg)
    if "local_ttest" in methods and str(cfg.local_ttest.distance) != "pixel_l2":
        raise ValueError("local_ttest.distance currently supports only 'pixel_l2'")

    rng = np.random.default_rng(seed)
    results_dir = case_path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    sample_meta, X, background, background_idx = importance.load_inputs(cfg, rng)
    model = MNISTResNetFlat.from_downloaded(
        model_source_dir=case_path(cfg.paths.model_source_dir),
        model_path=case_path(cfg.paths.model_path),
        batch_size=int(cfg.model.batch_size),
    )

    attributions = importance.attribute(
        X=X,
        sample_meta=sample_meta,
        background=background,
        model=model,
        n_orderings=int(cfg.explain.n_orderings),
        ig_steps=int(cfg.integrated_gradients.n_steps),
        ig_eps=float(cfg.integrated_gradients.eps),
        seed=seed,
    )

    background_probs = model.predict_proba(background)
    background_pred = background_probs.argmax(axis=1)
    if "local_ttest" in methods:
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
        attributions["local_ttest"] = importance.attribution_frame(sample_meta, ttest_scores)
        neighbor_df.to_csv(results_dir / "local_ttest_neighbors.csv", index=False)

    pixel_cols = pixel_feature_names()
    output_files = {}
    for method in methods:
        path = results_dir / f"{method}_attributions.csv"
        attributions[method][pixel_cols].to_csv(path, index=False)
        output_files[method] = metadata_path(path)
        log.info("Saved %s", path)

    pixels = np.arange(N_PIXELS)
    pd.DataFrame({"pixel_index": pixels, "row": pixels // 28, "col": pixels % 28}).to_csv(
        results_dir / "pixel_feature_map.csv", index=False
    )
    pd.DataFrame(
        {
            "background_index": background_idx,
            "predicted_label": background_pred,
            "predicted_probability": background_probs.max(axis=1),
        }
    ).to_csv(results_dir / "background_meta.csv", index=False)

    metadata = {
        "seed": seed,
        "config_paths": {"download": "config.yaml", "importance": "importance.yaml"},
        "sample_meta": metadata_path(sample_path),
        "download": download_metadata,
        "importance": {
            "methods": methods,
            "output_files": output_files,
            "n_samples": int(len(sample_meta)),
            "n_features": len(pixel_cols),
            "n_background": int(len(background)),
            "n_orderings": int(cfg.explain.n_orderings),
            "integrated_gradients": OmegaConf.to_container(
                cfg.integrated_gradients, resolve=True
            ),
            "local_ttest": OmegaConf.to_container(cfg.local_ttest, resolve=True),
        },
        **run_metadata,
    }
    metadata_path_out = results_dir / "run_metadata.yaml"
    OmegaConf.save(OmegaConf.create(metadata), metadata_path_out)
    log.info("Wrote %s", metadata_path_out)


if __name__ == "__main__":
    main()
