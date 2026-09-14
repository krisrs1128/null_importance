"""Batched estimators for expensive marginal-contribution games."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def fixed_background_sample(
    background: np.ndarray,
    n_rows: int | str | None,
    seed: int,
) -> np.ndarray:
    """Return a deterministic background subset for all coalition evaluations."""
    bg = np.asarray(background)
    if bg.ndim != 2:
        raise ValueError(f"Expected 2D background, got shape {bg.shape}")
    if len(bg) == 0:
        raise ValueError("background must contain at least one row")

    if n_rows is None or str(n_rows).lower() == "all":
        return bg.copy()

    n_rows = int(n_rows)
    if n_rows <= 0:
        raise ValueError("n_rows must be positive, null, or 'all'")
    if n_rows >= len(bg):
        return bg.copy()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(bg), n_rows, replace=False)
    return bg[idx].copy()


def prefix_values_batched(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
    ordering: np.ndarray,
    prefix_batch_size: int,
) -> np.ndarray:
    """Evaluate v(ordering[:p]) for p=0..d using fixed background rows.

    ``prefix_batch_size`` controls how many coalition prefixes are materialized
    before calling ``f``. Model-internal batching still decides the final
    inference batch size.
    """
    x = np.asarray(x)
    bg = np.asarray(background)
    ordering = np.asarray(ordering, dtype=int)
    d = len(x)

    if bg.ndim != 2 or bg.shape[1] != d:
        raise ValueError(
            f"Expected background with shape (n, {d}), got {bg.shape}"
        )
    if len(ordering) != d or set(ordering.tolist()) != set(range(d)):
        raise ValueError("ordering must be a permutation of feature indices")
    if prefix_batch_size <= 0:
        raise ValueError("prefix_batch_size must be positive")

    dtype = np.result_type(x.dtype, bg.dtype)
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float32
    x = x.astype(dtype, copy=False)
    bg = bg.astype(dtype, copy=False)

    values = np.empty(d + 1, dtype=float)
    current = bg.copy()
    position = 0

    while position <= d:
        stop = min(d + 1, position + prefix_batch_size)
        n_prefixes = stop - position
        masked = np.empty((n_prefixes, len(bg), d), dtype=dtype)

        for offset, prefix_size in enumerate(range(position, stop)):
            masked[offset] = current
            if prefix_size < d:
                current[:, ordering[prefix_size]] = x[ordering[prefix_size]]

        predictions = np.asarray(f(masked.reshape(-1, d)), dtype=float).reshape(-1)
        expected = n_prefixes * len(bg)
        if len(predictions) != expected:
            raise ValueError(
                f"f returned {len(predictions)} predictions for {expected} rows"
            )
        values[position:stop] = predictions.reshape(n_prefixes, len(bg)).mean(
            axis=1
        )
        position = stop

    return values


def marginal_minshap_batched(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
    *,
    n_orderings: int,
    seed: int,
    prefix_batch_size: int,
) -> np.ndarray:
    """Compute marginal minSHAP over sampled orderings."""
    x = np.asarray(x)
    d = len(x)
    n_orderings = int(n_orderings)
    prefix_batch_size = int(prefix_batch_size)

    if n_orderings <= 0:
        raise ValueError("n_orderings must be positive")
    rng = np.random.default_rng(seed)
    running_min = np.full(d, np.inf, dtype=float)

    for _ in range(n_orderings):
        ordering = rng.permutation(d)
        prefix_values = prefix_values_batched(
            f, x, background, ordering, prefix_batch_size
        )
        contributions = np.diff(prefix_values)
        running_min[ordering] = np.minimum(running_min[ordering], contributions)

    return running_min
