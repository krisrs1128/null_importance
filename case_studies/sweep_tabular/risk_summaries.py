"""Save the sampled conditional contributions behind minSHAP and Shapley.

Record the ``VI_j(S)`` terms producing the SHAP and minSHAP values, for the
datasets named under ``risk_summaries`` in config.yaml. sweep.py computes the
minSHAP and Shapley values themselves.  This command recomputes every (dataset,
response type, sample size, seed) combination and overwrites the output with new
orderings.

    python case_studies/sweep_tabular/risk_summaries.py
"""

import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "src"))

from importance import risk_importance_details

log = logging.getLogger(__name__)


def save_risk_summaries(*, results_dir, stem, X, y, feature_names, risk_cfg, rng):
    """Save one shared sample of orderings, from which both values are reduced."""
    _, _, contributions = risk_importance_details(
        X, y, feature_names, risk_cfg, rng
    )
    contributions.to_csv(
        Path(results_dir) / f"{stem}_risk_contributions.csv", index=False
    )


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    if not (cfg.risk_summaries.enabled and cfg.risk_summaries.save_contributions):
        log.info("Risk summary collection is disabled.")
        return

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    data_dir = SCRIPT_DIR / "data"
    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    for seed in cfg.seeds:
        rng = np.random.default_rng(seed)
        for dataset in cfg.risk_summaries.contribution_datasets:
            for response_type in cfg.response_types:
                for n in cfg.sample_sizes:
                    stem = f"{dataset}_{n}_{response_type}_{seed}"
                    log.info("Computing risk summaries for %s...", stem)

                    df = pd.read_csv(data_dir / f"{stem}.csv")
                    X_df = df.drop(columns=["y"])
                    save_risk_summaries(
                        results_dir=results_dir,
                        stem=stem,
                        X=X_df.to_numpy(),
                        y=df["y"].to_numpy(),
                        feature_names=list(X_df.columns),
                        risk_cfg=cfg_dict["risk"],
                        rng=rng,
                    )


if __name__ == "__main__":
    main()
