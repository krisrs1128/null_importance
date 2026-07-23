"""Nearest-neighbor marginal t-statistics for flattened MNIST pixels."""

import numpy as np
import pandas as pd


def _welch_tstat(a: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    """Per-pixel Welch t-statistic comparing two local predicted-class groups."""
    denom = a.var(axis=0, ddof=1) / len(a) + b.var(axis=0, ddof=1) / len(b)
    return (a.mean(axis=0) - b.mean(axis=0)) / np.sqrt(denom + eps)


def local_ttest_scores(
    X: np.ndarray,
    background: np.ndarray,
    background_predicted_label: np.ndarray,
    sample_predicted_label: np.ndarray,
    sample_index: np.ndarray,
    background_index: np.ndarray,
    n_neighbors: int = 20,
    min_group_size: int = 2,
    eps: float = 1e-8,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compare same-predicted-class vs other-class neighbors near each sample."""
    X = np.asarray(X, dtype=float)
    background = np.asarray(background, dtype=float)
    background_predicted_label = np.asarray(background_predicted_label, dtype=int)
    k = min(int(n_neighbors), len(background))

    scores = np.zeros_like(X, dtype=float)
    neighbor_rows = []
    for i, x in enumerate(X):
        distances = np.linalg.norm(background - x, axis=1)
        nearest = np.argsort(distances)[:k]
        same_class = background_predicted_label[nearest] == int(sample_predicted_label[i])

        for rank, j in enumerate(nearest, start=1):
            neighbor_rows.append(
                {
                    "sample_index": int(sample_index[i]),
                    "neighbor_rank": rank,
                    "background_position": int(j),
                    "background_index": int(background_index[j]),
                    "distance": float(distances[j]),
                    "background_predicted_label": int(background_predicted_label[j]),
                    "same_predicted_class": bool(same_class[rank - 1]),
                }
            )

        if same_class.sum() >= min_group_size and (~same_class).sum() >= min_group_size:
            scores[i] = _welch_tstat(
                background[nearest][same_class],
                background[nearest][~same_class],
                eps,
            )

    return scores, pd.DataFrame(neighbor_rows)
