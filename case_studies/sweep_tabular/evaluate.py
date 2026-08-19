"""Evaluate Sweep 1 importance scores as null-importance tests.

Using the results saved by ``sweep.py``, call features null or nonnull based on
the returned feature scores, and compare those calls with the ground truth saved
in our configuration. Write 2x2 tables, error rates, and an empirical version
of Table 1.

You can run this the repository root after ``sweep.py`` saves CSVs into
``results/``:

    python case_studies/sweep_tabular/evaluate.py
"""

import logging
import sys
from functools import partial
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from null_truth import coarse_null_label, null_truth_frame

log = logging.getLogger(__name__)
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir.parent / "src"))
sys.path.insert(0, str(_script_dir.parents[1] / "src"))
from evaluation import (  # noqa: E402
    BLOCK_KEYS, call_nonnull, confusion, empirical_table, error_rates,
    load_scores, pool,
)


def _write(frame, out_dir, name, dataset_ids=None):
    """Write one output and add the paper's D1--D7 id when it has a ``dataset``."""
    if dataset_ids is not None and "dataset" in frame.columns:
        frame = frame.copy()
        frame.insert(
            frame.columns.get_loc("dataset") + 1,
            "dataset_id",
            frame["dataset"].map(dataset_ids),
        )
    path = out_dir / name
    frame.to_csv(path, index=False)
    log.info(f"Wrote {path} ({len(frame)} rows)")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    eval_cfg = cfg_dict["evaluation"]

    # setup the output directories
    write = partial(_write, dataset_ids=cfg_dict["dataset_ids"])
    results_dir = _script_dir / "results"
    out_dir = results_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [name for name, on in cfg_dict["methods"].items() if on]

    # Get ground truth from the data generating process metadata.
    truth = null_truth_frame(cfg_dict)
    write(truth, out_dir, "null_truth.csv")
    write(coarse_null_label(truth), out_dir, "null_labels.csv")

    # Apply the decision rule used to call features null or nonnull
    scores = load_scores(results_dir, methods)
    labeled = call_nonnull(scores, truth, alpha=eval_cfg["alpha"])
    write(labeled, out_dir, "scores_long.csv")

    # compute error rates
    counts = confusion(labeled, truth, BLOCK_KEYS)
    write(counts, out_dir, "confusion.csv")
    per_seed = error_rates(counts)
    pooled = error_rates(pool(counts, over=("seed",)))
    write(per_seed, out_dir, "error_rates.csv")
    write(pooled, out_dir, "error_rates_pooled.csv")

    # Build the empirical Table 1, pooled and split by sample size, restricted
    # to one response type (error_rates.qmd interprets a single type at a time).
    rates = pooled if eval_cfg["pool_seeds"] else per_seed
    rates = rates[rates["response_type"] == eval_cfg["response_type"]]
    table = empirical_table(
        rates,
        alpha=eval_cfg["alpha"],
        power_target=eval_cfg["power_target"],
        bands=tuple(eval_cfg["bands"]),
        categories=cfg_dict["categories"],
    )
    write(table, out_dir, "table1_empirical.csv")

    table_by_n = empirical_table(
        rates,
        alpha=eval_cfg["alpha"],
        power_target=eval_cfg["power_target"],
        bands=tuple(eval_cfg["bands"]),
        cell_keys=("dataset", "response_type"),
        categories=cfg_dict["categories"],
        group_keys=("method", "notion", "n"),
    )
    write(table_by_n, out_dir, "table1_empirical_by_n.csv")


if __name__ == "__main__":
    main()
