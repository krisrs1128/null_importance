"""Save minSHAP/Shapley summaries and sampled conditional contributions.

Record the ``VI_j(S)`` terms producing the SHAP and minSHAP values

Run from the repository root:

    python case_studies/sweep_tabular/risk_summaries.py


Whether this runs in sweep.py can be configured using the ``risk_summaries``
group in ``config.yaml``.
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


def _save_importance(results_dir: Path, stem: str, values: pd.Series) -> None:
    pd.DataFrame({
        "feature": values.index,
        "importance": values.values,
    }).to_csv(results_dir / f"{stem}.csv", index=False)


def save_risk_summaries(
    *,
    results_dir,
    stem,
    X,
    y,
    feature_names,
    risk_cfg,
    rng,
    save_minshap=True,
    save_shapley=True,
    save_contributions=False,
):
    """Compute and save summaries from one shared collection of orderings."""
    minshap, shapley, contributions = risk_importance_details(
        X, y, feature_names, risk_cfg, rng
    )

    results_dir = Path(results_dir)
    if save_minshap:
        _save_importance(results_dir, f"{stem}_minshap", minshap)
    if save_shapley:
        # Preserve the existing case-study method name for risk-Shapley.
        _save_importance(results_dir, f"{stem}_sage", shapley)
    if save_contributions:
        contributions.to_csv(
            results_dir / f"{stem}_risk_contributions.csv",
            index=False,
        )


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.risk_summaries.enabled:
        log.info("Risk summary collection is disabled.")
        return

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    data_dir = SCRIPT_DIR / "data"
    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    contribution_datasets = set(cfg.risk_summaries.contribution_datasets)

    for seed in cfg.seeds:
        rng = np.random.default_rng(seed)
        for dataset in cfg.datasets:
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
                        save_minshap=cfg.methods.minshap,
                        save_shapley=cfg.methods.sage,
                        save_contributions=(
                            cfg.risk_summaries.save_contributions
                            and dataset in contribution_datasets
                        ),
                    )


if __name__ == "__main__":
    main()
