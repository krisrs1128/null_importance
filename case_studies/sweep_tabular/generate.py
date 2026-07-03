"""Generate the Sweep 1 synthetic null-importance datasets.

Run from the repo root:
    python case_studies/sweep_tabular/generate.py
"""

import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from datasets import DATASETS

log = logging.getLogger(__name__)
_script_dir = Path(__file__).resolve().parent


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    rng = np.random.default_rng(cfg.seed)

    data_dir = _script_dir / "data"
    data_dir.mkdir(exist_ok=True)

    for name, fn in DATASETS.items():
        # Merge global dimensions into per-dataset cfg; per-dataset keys win on conflict.
        dataset_cfg = {**cfg_dict["dimensions"], **cfg_dict["datasets"][name]}
        for n in cfg.sample_sizes:
            X, y, _, meta = fn(n, rng, dataset_cfg)
            out = X.copy()
            out["y"] = y
            path = data_dir / f"{name}_{n}.csv"
            out.to_csv(path, index=False)
            log.info(f"Wrote {path} ({n} rows); null_type={meta['null_type']}")


if __name__ == "__main__":
    main()
