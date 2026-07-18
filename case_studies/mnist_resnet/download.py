"""Download the pretrained MNIST ResNet checkpoint and MNIST test split.

Run from this case-study directory:
    cd case_studies/mnist_resnet
    python download.py
"""

import logging
import random
import shutil
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from torchvision.datasets import MNIST

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir.parent / "src"))

from reproducibility import capture_run_metadata

log = logging.getLogger(__name__)


# Keep configured paths relative to this case-study directory by default.
def _resolve_path(path: str) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return (_script_dir / resolved).resolve()


# Store portable relative paths in metadata when outputs live in this folder.
def _metadata_path(path: Path) -> str:
    try:
        return str(path.relative_to(_script_dir))
    except ValueError:
        return str(path)


def _copy_file(source: Path, target: Path, overwrite: bool) -> None:
    if target.exists() and not overwrite:
        log.info("Using cached %s", target)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, tmp_target)
    tmp_target.replace(target)
    log.info("Wrote %s", target)


def _download_hf_file(cfg: DictConfig, filename: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=cfg.model.repo_id,
            filename=filename,
            revision=cfg.model.revision,
            cache_dir=str(_resolve_path(cfg.paths.hf_cache_dir)),
            local_files_only=cfg.download.local_files_only,
        )
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def download_model(cfg: DictConfig) -> dict:
    model_path = _resolve_path(cfg.paths.model_path)
    source_dir = _resolve_path(cfg.paths.model_source_dir)

    # Pull the pinned public checkpoint, then copy it to the stable path used by
    # later prediction and importance scripts.
    checkpoint_source = _download_hf_file(cfg, cfg.model.checkpoint_file)

    _copy_file(checkpoint_source, model_path, cfg.download.overwrite)
    _copy_file(
        checkpoint_source,
        source_dir / cfg.model.checkpoint_file,
        cfg.download.overwrite,
    )

    copied_auxiliary_files = []
    for filename in cfg.model.auxiliary_files:
        # Keep source config files next to the checkpoint so the architecture and
        # preprocessing can be reconstructed without changing config.yaml.
        source = _download_hf_file(cfg, filename)
        target = source_dir / filename
        _copy_file(source, target, cfg.download.overwrite)
        copied_auxiliary_files.append(_metadata_path(target))

    return {
        "repo_id": cfg.model.repo_id,
        "revision": cfg.model.revision,
        "checkpoint_file": cfg.model.checkpoint_file,
        "source_url": cfg.model.source_url,
        "model_path": _metadata_path(model_path),
        "model_source_dir": _metadata_path(source_dir),
        "auxiliary_files": copied_auxiliary_files,
    }


def download_mnist_test(cfg: DictConfig) -> dict:
    data_dir = _resolve_path(cfg.paths.data_dir)
    labels_path = _resolve_path(cfg.data.labels_path)
    data_dir.mkdir(parents=True, exist_ok=True)

    # TorchVision downloads the established MNIST files under data/MNIST.
    dataset = MNIST(root=str(data_dir), train=cfg.data.train, download=True)
    targets = dataset.targets
    if hasattr(targets, "detach"):
        labels = targets.detach().cpu().numpy()
    else:
        labels = np.asarray(targets)

    # Save labels explicitly; the next stage selects correct/incorrect examples
    # by joining this table with model predictions.
    label_df = pd.DataFrame(
        {
            "mnist_index": np.arange(len(labels), dtype=int),
            "label": labels.astype(int),
        }
    )
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    label_df.to_csv(labels_path, index=False)
    log.info("Wrote %s", labels_path)

    class_counts = label_df["label"].value_counts().sort_index()
    return {
        "name": cfg.data.name,
        "split": cfg.data.split,
        "data_dir": _metadata_path(data_dir),
        "labels_path": _metadata_path(labels_path),
        "n_examples": int(len(dataset)),
        "class_counts": {int(k): int(v) for k, v in class_counts.items()},
    }


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    _set_seed(int(cfg.seed))

    # Record the exact code/config context that produced model.pt and data/.
    run_metadata = capture_run_metadata(_script_dir)
    log.info(
        "Running with git_commit=%s; hydra config at %s",
        run_metadata["git_commit"],
        run_metadata["hydra_output_dir"],
    )

    model_metadata = download_model(cfg)
    data_metadata = download_mnist_test(cfg)

    metadata = {
        "seed": int(cfg.seed),
        "model": model_metadata,
        "data": data_metadata,
        **run_metadata,
    }
    metadata_path = _resolve_path(cfg.paths.download_metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(metadata), metadata_path)
    log.info("Wrote %s", metadata_path)


if __name__ == "__main__":
    main()
