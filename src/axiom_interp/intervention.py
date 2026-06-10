"""Feature-removed/Intervened versions of f evaluation

The explainer needs a version of f with some
coordinates present (kept at their x-value) and the rest absent (filled in by a
reference rule). In SHAP, the reference rule replaces f with the marginal
E[f(x[S], X[-S])].  Changing this to conditional gives causal SHAP without
changing any of the other method components.

We can use the same interface for mechanistic methods, but the masks are applied
in a latent space.
"""

from __future__ import annotations

import numpy as np


class Intervention:
    """value(f, x, active) = E[ f(point with `active` set to x, rest to reference) ]."""

    def value(self, f, x, active) -> float:
        raise NotImplementedError


class BaselineMask(Intervention):
    """Absent coordinates take a fixed baseline x0 (functional v(S))."""

    def __init__(self, baseline: np.ndarray):
        self.baseline = np.asarray(baseline, dtype=float)

    def value(self, f, x, active) -> float:
        row = self.baseline.copy()
        idx = np.asarray(list(active), dtype=int)
        if idx.size:
            row[idx] = np.asarray(x)[idx]
        return float(f(row[None, :])[0])

    def __repr__(self):
        return f"BaselineMask({np.round(self.baseline, 4).tolist()})"


class ZeroMask(Intervention):
    """Absent coordinates take 0, regardless of dimension.

    The dimension-agnostic reference used by latent ablation: f(z) vs f(z_{-j}).
    """

    def value(self, f, x, active) -> float:
        x = np.asarray(x, dtype=float)
        row = np.zeros_like(x)
        idx = np.asarray(list(active), dtype=int)
        if idx.size:
            row[idx] = x[idx]
        return float(f(row[None, :])[0])

    def __repr__(self):
        return "ZeroMask()"


class MarginalMask(Intervention):
    """Absent coordinates are drawn from a background sample (statistical v(S)).

    v(S) = E_{X~bg}[ f(x_S, X_{-S}) ], estimated by averaging over rows of `bg`.
    """

    def __init__(self, background: np.ndarray, n_samples: int = 64, seed: int = 0):
        self.background = np.asarray(background, dtype=float)
        self.n_samples = min(n_samples, len(self.background))
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def value(self, f, x, active) -> float:
        rows = self.background[
            self._rng.choice(len(self.background), self.n_samples, replace=True)
        ].copy()
        idx = np.asarray(list(active), dtype=int)
        if idx.size:
            rows[:, idx] = np.asarray(x)[idx]
        return float(np.mean(f(rows)))

    def __repr__(self):
        return f"MarginalMask(n={self.n_samples}, seed={self.seed})"
