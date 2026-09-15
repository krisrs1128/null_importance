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


def save_risk_summaries(
    *, results_dir, stem, X, y, feature_names, risk_cfg, rng, response_type
):
    """Save one shared sample of orderings, from which both values are reduced."""
    _, _, contributions = risk_importance_details(
        X, y, feature_names, risk_cfg, rng, response_type
    )
    output_path = Path(results_dir) / f"{stem}_risk_contributions.csv"
    contributions.to_csv(output_path, index=False)


def _risk_config(cfg, dataset):
    """The model class always agrees with the risk class."""
    return {
        **cfg["risk"],
        "model_class": cfg["datasets"][dataset].get(
            "risk_class", cfg["risk"]["default_class"]
        ),
    }


def _load_dataset(data_dir, stem):
    """Load a generated dataset and separate its response from its features."""
    df = pd.read_csv(data_dir / f"{stem}.csv")
    X_df = df.drop(columns=["y"])
    return X_df.to_numpy(), df["y"].to_numpy(), list(X_df.columns)


def _run_task(
    *, data_dir, results_dir, dataset, response_type, sample_size,
    seed, risk_cfg, rng
):
    """Compute and save contributions for one dataset configuration."""
    stem = f"{dataset}_{sample_size}_{response_type}_{seed}"
    log.info("Computing risk summaries for %s...", stem)
    X, y, feature_names = _load_dataset(data_dir, stem)
    save_risk_summaries(
        results_dir=results_dir,
        stem=stem,
        X=X,
        y=y,
        feature_names=feature_names,
        risk_cfg=risk_cfg,
        rng=rng,
        response_type=response_type,
    )


def _run_seed(cfg, seed, data_dir, results_dir):
    """Run all configured contribution tasks for one seed."""
    rng = np.random.default_rng(seed)
    for dataset in cfg["risk_summaries"]["contribution_datasets"]:
        risk_cfg = _risk_config(cfg, dataset)
        for response_type in cfg["response_types"]:
            for sample_size in cfg["sample_sizes"]:
                _run_task(
                    data_dir=data_dir,
                    results_dir=results_dir,
                    dataset=dataset,
                    response_type=response_type,
                    sample_size=sample_size,
                    seed=seed,
                    risk_cfg=risk_cfg,
                    rng=rng,
                )


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    summaries_cfg = cfg_dict["risk_summaries"]
    if not (summaries_cfg["enabled"] and summaries_cfg["save_contributions"]):
        log.info("Risk summary collection is disabled.")
        return

    data_dir = SCRIPT_DIR / "data"
    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for seed in cfg_dict["seeds"]:
        _run_seed(cfg_dict, seed, data_dir, results_dir)


if __name__ == "__main__":
    main()
