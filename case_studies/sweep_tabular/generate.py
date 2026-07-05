"""Generate the Sweep 1 synthetic null-importance datasets.

Run from the repo root:
    python case_studies/sweep_tabular/generate.py
"""

import logging
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Add case_studies/src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from reproducibility import capture_run_metadata
from datasets import DATASETS

log = logging.getLogger(__name__)
_script_dir = Path(__file__).resolve().parent


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Capture git commit and hydra output directory for reproducibility
    run_metadata = capture_run_metadata(_script_dir)
    log.info(
        f"Running with git_commit={run_metadata['git_commit']}; "
        f"hydra config at {run_metadata['hydra_output_dir']}"
    )

    # set seed and create output directories
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    rng = np.random.default_rng(cfg.seed)

    data_dir = _script_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # loop over datasets, response types, and sample sizes to generate data
    for name, fn in DATASETS.items():
        base_cfg = {**cfg_dict["dimensions"], **cfg_dict["datasets"][name]}
        for rt in cfg.response_types:
            dataset_cfg = {**base_cfg, "response_type": rt}
            for n in cfg.sample_sizes:
                X, y, _, meta = fn(n, rng, dataset_cfg)
                out = X.copy()
                out["y"] = y
                path = data_dir / f"{name}_{n}_{rt}.csv"
                out.to_csv(path, index=False)
                log.info(f"Wrote {path} ({n} rows, {rt}); null_type={meta['null_type']}")

    # Write run metadata for reproducibility
    metadata = {
        "seed": cfg.seed,
        **run_metadata,
    }
    metadata_path = data_dir / "run_metadata.yaml"
    OmegaConf.save(OmegaConf.create(metadata), metadata_path)
    log.info(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
