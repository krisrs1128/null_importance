"""Summarize simulation attribution metric CSV files.

This is a light post-processing step: it groups per-row metric files by
`dataset_type` and `method`, then averages numeric fields that are present.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


# Column order for the aggregate benchmark summary.
SUMMARY_FIELDS = [
    "dataset_type",
    "method",
    "n_rows",
    "mean_top_k_recall",
    "mean_abs_relevant",
    "mean_abs_noise",
    "mean_max_abs_noise",
    "mean_baseline_efficiency_gap",
    "mean_injected_null",
    "mean_stability_lipschitz",
]


def summarize_metrics(metrics_dir: str | Path, output_path: str | Path) -> Path:
    """Aggregate metric CSV files into one dataset-by-method summary."""

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for path in sorted(Path(metrics_dir).glob("*.csv")):
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                groups[(row["dataset_type"], row["method"])].append(row)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for (dataset_type, method), rows in sorted(groups.items()):
            writer.writerow(
                {
                    "dataset_type": dataset_type,
                    "method": method,
                    "n_rows": len(rows),
                    "mean_top_k_recall": _mean_field(rows, "top_k_recall"),
                    "mean_abs_relevant": _mean_field(rows, "mean_abs_relevant"),
                    "mean_abs_noise": _mean_field(rows, "mean_abs_noise"),
                    "mean_max_abs_noise": _mean_field(rows, "max_abs_noise"),
                    "mean_baseline_efficiency_gap": _mean_field(rows, "baseline_efficiency_gap"),
                    "mean_injected_null": _mean_field(rows, "injected_null"),
                    "mean_stability_lipschitz": _mean_field(rows, "stability_lipschitz"),
                }
            )
    return output


def _mean_field(rows: list[dict[str, str]], field: str) -> float | str:
    """Average a metric column while preserving blank values for unsupported metrics."""

    values = []
    for row in rows:
        raw = row.get(field, "")
        if raw == "":
            continue
        values.append(float(raw))
    return "" if not values else mean(values)
