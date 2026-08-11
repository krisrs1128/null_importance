"""Evaluate Sweep 1 importance scores as null-importance tests.

Read the per-method CSVs written by ``sweep.py``, call features null or
non-null, and compare those calls with the per-feature ground truth. Write
2x2 tables, error rates, threshold sweeps, ranking summaries, and an empirical
version of Table 1.

Run from the repository root after ``sweep.py`` has populated ``results/``:

    python case_studies/sweep_tabular/evaluate.py

The command overwrites its evaluation outputs. It does not regenerate data.
"""

import logging
import sys
from functools import partial
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from null_truth import coarse_null_label, null_truth_frame

log = logging.getLogger(__name__)
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir.parent / "src"))
sys.path.insert(0, str(_script_dir.parents[1] / "src"))
from evaluation import (  # noqa: E402
    BLOCK_KEYS, call_nonnull, confusion, curve_summary, empirical_table,
    error_rates, load_scores, pool, threshold_sweep,
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

    write = partial(_write, dataset_ids=cfg_dict["dataset_ids"])

    results_dir = _script_dir / "results"
    out_dir = results_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [name for name, on in cfg_dict["methods"].items() if on]

    # Ground truth, taken directly from the DGP metadata.
    truth = null_truth_frame(cfg_dict)
    write(truth, out_dir, "null_truth.csv")
    write(coarse_null_label(truth), out_dir, "null_labels.csv")

    # Apply the decision rule used for the headline tables.
    scores = load_scores(results_dir, methods)
    labeled = call_nonnull(scores, threshold=eval_cfg["threshold"])
    write(labeled, out_dir, "scores_long.csv")

    counts = pd.concat(
        [confusion(labeled, truth, BLOCK_KEYS, scope)
         for scope in eval_cfg["null_scopes"]],
        ignore_index=True,
    )
    write(counts, out_dir, "confusion.csv")

    per_seed = error_rates(counts)
    pooled = error_rates(pool(counts, over=("seed",)))
    write(per_seed, out_dir, "error_rates.csv")
    write(pooled, out_dir, "error_rates_pooled.csv")

    # Record how the threshold trades type I error against power.
    write(
        threshold_sweep(
            scores, truth, eval_cfg["thresholds"],
            null_scopes=eval_cfg["null_scopes"],
        ),
        out_dir, "threshold_sweep.csv",
    )
    write(
        curve_summary(scores, truth, null_scopes=eval_cfg["null_scopes"]),
        out_dir, "curve_summary.csv",
    )

    # Build the empirical Table 1 from structural nulls only.
    scope = eval_cfg["table_null_scope"]
    source = pooled if eval_cfg["pool_seeds"] else per_seed
    rates = source[source["null_scope"] == scope]
    table = empirical_table(
        rates,
        alpha=eval_cfg["alpha"],
        power_target=eval_cfg["power_target"],
        bands=tuple(eval_cfg["bands"]),
        categories=cfg_dict["categories"],
    )
    write(table, out_dir, "table1_empirical.csv")


if __name__ == "__main__":
    main()
