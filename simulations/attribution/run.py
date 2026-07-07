"""Run attribution methods over generated simulation datasets.

This module owns the benchmark execution loop:

1. discover generated dataset CSVs;
2. rebuild the oracle function from each dataset's metadata sidecar;
3. call `Explainer.explain(f, x)` for selected rows and methods;
4. write feature-level scores and row-level metrics.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from axiom_interp import properties
from simulation_data import load_csv_dataset
from simulations.attribution.methods import build_explainers
from simulations.attribution.metrics import (
    METRIC_FIELDNAMES,
    SCORE_FIELDNAMES,
    attribution_metric_row,
    dataset_context,
    score_rows,
)
from simulations.attribution.oracle import function_from_metadata, max_oracle_error


BASELINE_EFFICIENT_METHODS = {"baseline_shap", "integrated_gradients"}
INJECTED_NULL_METHODS = {
    "baseline_shap",
    "baseline_minshap",
    "marginal_shap",
    "marginal_minshap",
}


def run_sweep(config: dict[str, Any], *, overwrite: bool | None = None) -> list[tuple[Path, Path]]:
    """Run the configured attribution benchmark over all matching datasets."""

    data_dir = ROOT / config.get("data_dir", "data/simulations")
    output_dir = ROOT / config.get("output_dir", "results/simulation_attribution")
    dataset_types = config.get("dataset_types")
    dataset_limit = config.get("dataset_limit")
    paths = discover_datasets(data_dir, dataset_types=dataset_types)
    if dataset_limit is not None:
        paths = paths[: int(dataset_limit)]

    metrics_config = config.get("metrics", {})
    effective_overwrite = bool(config.get("overwrite", False)) if overwrite is None else overwrite
    outputs: list[tuple[Path, Path]] = []
    for dataset_path in paths:
        outputs.append(
            run_single_dataset(
                dataset_path=dataset_path,
                output_dir=output_dir,
                methods=config.get("methods"),
                row_ids=config.get("row_ids"),
                row_limit=int(config.get("row_limit", 5)),
                background_size=int(config.get("background_size", 64)),
                n_orderings=int(config.get("n_orderings", 200)),
                n_steps=int(config.get("n_steps", 64)),
                seed=int(config.get("seed", 0)),
                baseline_strategy=config.get("baseline_strategy", "zeros"),
                include_unsupported=bool(config.get("include_unsupported", False)),
                overwrite=effective_overwrite,
                compute_injected_null=bool(metrics_config.get("injected_null", False)),
                compute_stability=bool(metrics_config.get("stability", False)),
                stability_neighbors=int(metrics_config.get("stability_neighbors", 16)),
                stability_radius=float(metrics_config.get("stability_radius", 0.05)),
            )
        )
    return outputs


def run_single_dataset(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    methods: list[str] | None = None,
    row_ids: list[int] | None = None,
    row_limit: int = 5,
    background_size: int = 64,
    n_orderings: int = 200,
    n_steps: int = 64,
    seed: int = 0,
    baseline_strategy: str = "zeros",
    include_unsupported: bool = False,
    overwrite: bool = False,
    compute_injected_null: bool = False,
    compute_stability: bool = False,
    stability_neighbors: int = 16,
    stability_radius: float = 0.05,
    oracle_tolerance: float = 1e-8,
) -> tuple[Path, Path]:
    """Run configured attribution methods on one generated dataset CSV.

    The oracle check is intentionally before attribution: downstream metrics are
    only meaningful if metadata reconstruction exactly matches stored y_mean.
    """

    dataset_path = Path(dataset_path)
    score_path = score_output_path(output_dir, dataset_path)
    metric_path = metric_output_path(output_dir, dataset_path)
    if not overwrite and score_path.exists() and metric_path.exists():
        return score_path, metric_path

    X, _, y_mean, metadata = load_csv_dataset(dataset_path)
    f = function_from_metadata(metadata)
    oracle_error = max_oracle_error(f, X, y_mean)
    if oracle_error > oracle_tolerance:
        raise ValueError(
            f"Rebuilt oracle does not match y_mean for {dataset_path}: "
            f"max_abs_error={oracle_error:.3e}"
        )

    selected_rows = _select_rows(n_rows=len(X), row_ids=row_ids, row_limit=row_limit)
    specs = build_explainers(
        method_names=methods,
        X=X,
        metadata=metadata,
        background_size=background_size,
        n_orderings=n_orderings,
        n_steps=n_steps,
        seed=seed,
        baseline_strategy=baseline_strategy,
        include_unsupported=include_unsupported,
    )

    score_out: list[dict[str, Any]] = []
    metric_out: list[dict[str, Any]] = []
    context = dataset_context(str(dataset_path), metadata)
    rng = np.random.default_rng(seed)

    for row_id in selected_rows:
        x = np.asarray(X[row_id], dtype=float)
        for spec in specs:
            obj = spec.explainer.explain(f, x)
            score_out.extend(
                score_rows(
                    scores=obj.scores,
                    provenance=obj.provenance,
                    metadata=metadata,
                    context=context,
                    method=spec.name,
                    row_id=row_id,
                )
            )
            metric_out.append(
                attribution_metric_row(
                    scores=obj.scores,
                    metadata=metadata,
                    context=context,
                    method=spec.name,
                    row_id=row_id,
                    baseline_efficiency_gap=_baseline_efficiency_gap(
                        spec.name, obj, f, x, spec.baseline
                    ),
                    injected_null=_injected_null(
                        spec.name,
                        spec.explainer,
                        f,
                        x,
                        spec.background,
                        compute=compute_injected_null,
                    ),
                    stability_lipschitz=_stability(
                        spec.explainer,
                        f,
                        x,
                        rng,
                        compute=compute_stability,
                        n_neighbors=stability_neighbors,
                        radius=stability_radius,
                    ),
                    oracle_max_abs_error=oracle_error,
                )
            )

    write_rows(score_path, score_out, SCORE_FIELDNAMES)
    write_rows(metric_path, metric_out, METRIC_FIELDNAMES)
    return score_path, metric_path


def discover_datasets(
    data_dir: str | Path,
    *,
    dataset_types: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    """Find generated dataset CSVs, optionally restricted by dataset type."""

    root = Path(data_dir)
    if dataset_types:
        paths: list[Path] = []
        for dataset_type in dataset_types:
            paths.extend((root / dataset_type).glob("*.csv"))
        return sorted(paths)
    return sorted(root.glob("*/*.csv"))


def score_output_path(output_dir: str | Path, dataset_path: str | Path) -> Path:
    """Return the raw feature-score CSV path for one dataset."""

    return ensure_result_dirs(output_dir)["raw_scores"] / f"{Path(dataset_path).stem}.csv"


def metric_output_path(output_dir: str | Path, dataset_path: str | Path) -> Path:
    """Return the row-level metric CSV path for one dataset."""

    return ensure_result_dirs(output_dir)["metrics"] / f"{Path(dataset_path).stem}.csv"


def ensure_result_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    dirs = {
        "raw_scores": root / "raw_scores",
        "metrics": root / "metrics",
        "summaries": root / "summaries",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_rows(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write CSV rows with a fixed schema, even when the row list is empty."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _select_rows(*, n_rows: int, row_ids: list[int] | None, row_limit: int) -> list[int]:
    if row_ids is not None:
        selected = [int(v) for v in row_ids]
    else:
        selected = list(range(min(int(row_limit), n_rows)))
    bad = [idx for idx in selected if idx < 0 or idx >= n_rows]
    if bad:
        raise ValueError(f"row_ids outside dataset bounds: {bad}")
    return selected


def _baseline_efficiency_gap(method: str, obj, f, x, baseline) -> float | None:
    """Compute completeness only for methods with a fixed baseline target."""

    if method not in BASELINE_EFFICIENT_METHODS:
        return None
    return properties.efficiency_gap(obj, f, x, baseline)


def _injected_null(method: str, explainer, f, x, background, *, compute: bool) -> float | None:
    """Optionally measure attribution assigned to an injected null feature."""

    if not compute or method not in INJECTED_NULL_METHODS:
        return None
    return properties.injected_null(explainer, f, x, background, n=1)


def _stability(
    explainer,
    f,
    x,
    rng: np.random.Generator,
    *,
    compute: bool,
    n_neighbors: int,
    radius: float,
) -> float | None:
    """Optionally measure local explanation sensitivity around one row."""

    if not compute:
        return None
    neighbors = x + rng.normal(scale=float(radius), size=(int(n_neighbors), len(x)))
    return properties.perturbation_stability(explainer, f, x, neighbors, metric="lipschitz")
