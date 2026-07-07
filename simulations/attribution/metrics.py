"""Metrics for generated-data attribution runs.

The benchmark writes two CSV layers:

- raw score rows: one row per dataset, method, input row, and feature;
- metric rows: one row per dataset, method, and input row.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


# Column order for feature-level attribution scores.
SCORE_FIELDNAMES = [
    "dataset_path",
    "dataset_type",
    "n",
    "p",
    "seed",
    "method",
    "row_id",
    "feature",
    "score",
    "abs_score",
    "is_relevant",
    "is_noise",
    "is_additive",
    "is_interaction_member",
    "provenance",
]

# Column order for row-level attribution quality metrics.
METRIC_FIELDNAMES = [
    "dataset_path",
    "dataset_type",
    "n",
    "p",
    "seed",
    "method",
    "row_id",
    "mean_abs_relevant",
    "mean_abs_noise",
    "max_abs_noise",
    "relevant_to_noise_ratio",
    "top_k_recall",
    "baseline_efficiency_gap",
    "injected_null",
    "stability_lipschitz",
    "oracle_max_abs_error",
]


def dataset_context(dataset_path: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Return fields repeated across score and metric rows."""

    return {
        "dataset_path": dataset_path,
        "dataset_type": metadata.get("dataset_type", ""),
        "n": metadata.get("n", ""),
        "p": metadata.get("p", ""),
        "seed": metadata.get("seed", ""),
    }


def score_rows(
    *,
    scores: dict[int, float],
    provenance: dict[str, Any],
    metadata: dict[str, Any],
    context: dict[str, Any],
    method: str,
    row_id: int,
) -> list[dict[str, Any]]:
    """Convert one explanation object into feature-level score rows."""

    relevant = set(_int_list(metadata.get("relevant_features", [])))
    noise = set(_int_list(metadata.get("noise_features", [])))
    additive = set(_int_list(metadata.get("additive_features", [])))
    interaction = _interaction_members(metadata)
    provenance_json = json.dumps(provenance, sort_keys=True)

    rows = []
    for feature in sorted(scores):
        score = float(scores[feature])
        rows.append(
            {
                **context,
                "method": method,
                "row_id": int(row_id),
                "feature": int(feature),
                "score": score,
                "abs_score": abs(score),
                "is_relevant": int(feature in relevant),
                "is_noise": int(feature in noise),
                "is_additive": int(feature in additive),
                "is_interaction_member": int(feature in interaction),
                "provenance": provenance_json,
            }
        )
    return rows


def attribution_metric_row(
    *,
    scores: dict[int, float],
    metadata: dict[str, Any],
    context: dict[str, Any],
    method: str,
    row_id: int,
    baseline_efficiency_gap: float | None,
    injected_null: float | None,
    stability_lipschitz: float | None,
    oracle_max_abs_error: float,
) -> dict[str, Any]:
    """Summarize one method's feature scores against known dataset metadata."""

    relevant = _int_list(metadata.get("relevant_features", []))
    noise = _int_list(metadata.get("noise_features", []))
    abs_scores = {int(k): abs(float(v)) for k, v in scores.items()}

    mean_abs_relevant = _mean_abs(abs_scores, relevant)
    mean_abs_noise = _mean_abs(abs_scores, noise)
    max_abs_noise = _max_abs(abs_scores, noise)
    top_k_recall = _top_k_recall(abs_scores, relevant)
    ratio = _ratio(mean_abs_relevant, mean_abs_noise)

    return {
        **context,
        "method": method,
        "row_id": int(row_id),
        "mean_abs_relevant": _blank_if_none(mean_abs_relevant),
        "mean_abs_noise": _blank_if_none(mean_abs_noise),
        "max_abs_noise": _blank_if_none(max_abs_noise),
        "relevant_to_noise_ratio": _blank_if_none(ratio),
        "top_k_recall": _blank_if_none(top_k_recall),
        "baseline_efficiency_gap": _blank_if_none(baseline_efficiency_gap),
        "injected_null": _blank_if_none(injected_null),
        "stability_lipschitz": _blank_if_none(stability_lipschitz),
        "oracle_max_abs_error": float(oracle_max_abs_error),
    }


def _int_list(values: list[Any]) -> list[int]:
    return [int(v) for v in values]


def _interaction_members(metadata: dict[str, Any]) -> set[int]:
    members: set[int] = set()
    for group in metadata.get("interaction_features", []):
        members.update(int(v) for v in group)
    return members


def _mean_abs(abs_scores: dict[int, float], features: list[int]) -> float | None:
    present = [abs_scores[j] for j in features if j in abs_scores]
    if not present:
        return None
    return float(np.mean(present))


def _max_abs(abs_scores: dict[int, float], features: list[int]) -> float | None:
    present = [abs_scores[j] for j in features if j in abs_scores]
    if not present:
        return None
    return float(np.max(present))


def _top_k_recall(abs_scores: dict[int, float], relevant: list[int]) -> float | None:
    """Recall of relevant features among the top-k absolute attribution scores."""

    if not relevant:
        return None
    k = len(relevant)
    ranked = sorted(abs_scores, key=lambda j: (-abs_scores[j], j))
    selected = set(ranked[:k])
    return len(selected.intersection(relevant)) / float(k)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0.0:
        return math.inf if numerator > 0.0 else 0.0
    return numerator / denominator


def _blank_if_none(value: float | None) -> float | str:
    return "" if value is None else float(value)
