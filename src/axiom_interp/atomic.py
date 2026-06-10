"""Atomic Importance Statistics I_j
"""

import numpy as np


def _grad(f, point: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central finite-difference gradient of scalar f at a single point."""
    d = len(point)
    g = np.zeros(d)
    for j in range(d):
        plus = point.copy(); plus[j] += eps
        minus = point.copy(); minus[j] -= eps
        g[j] = (f(plus[None, :])[0] - f(minus[None, :])[0]) / (2 * eps)
    return g


class AtomicStatistic:
    def compute_all(self, f, intervention, elements, x) -> dict:
        raise NotImplementedError


class MarginalContribution(AtomicStatistic):
    """v(S u {j}) - v(S), where S = predecessors of j in an ordering element."""

    def compute_all(self, f, intervention, elements, x) -> dict:
        d = len(x)
        data = {j: [] for j in range(d)}
        for order in elements:
            # prefix values v(order[:p]) for p = 0..d
            prefix = [intervention.value(f, x, order[:p]) for p in range(d + 1)]
            for p, j in enumerate(order):
                data[j].append(prefix[p + 1] - prefix[p])
        return {j: np.asarray(v) for j, v in data.items()}

    def __repr__(self):
        return "MarginalContribution()"


class PathIntegratedGradient(AtomicStatistic):
    """(x_j - x0_j) * df/dx_j along the straight path from baseline x0 to x."""

    def __init__(self, baseline: np.ndarray):
        self.baseline = np.asarray(baseline, dtype=float)

    def compute_all(self, f, intervention, elements, x) -> dict:
        x = np.asarray(x, dtype=float)
        diff = x - self.baseline
        data = {j: [] for j in range(len(x))}
        for alpha in elements:
            g = _grad(f, self.baseline + alpha * diff)
            for j in range(len(x)):
                data[j].append(diff[j] * g[j])
        return {j: np.asarray(v) for j, v in data.items()}

    def __repr__(self):
        return f"PathIntegratedGradient(baseline={np.round(self.baseline, 4).tolist()})"


class AblationDelta(AtomicStatistic):
    """v(all units) - v(all units except j): the drop from removing unit j.

    With f = downstream . decode and a zero-reference intervention in latent
    space this is exactly the sparse-autoencoder feature attribution
    f(Dz) - f(Dz_{-j}) -- the same skeleton as input attribution.
    """

    def compute_all(self, f, intervention, elements, x) -> dict:
        d = len(x)
        full = intervention.value(f, x, range(d))
        data = {}
        for j in elements:  # UnitIndices -> j is the unit id
            without_j = [k for k in range(d) if k != j]
            data[j] = np.asarray([full - intervention.value(f, x, without_j)])
        return data

    def __repr__(self):
        return "AblationDelta()"
