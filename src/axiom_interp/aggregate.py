"""A: reduce a unit's atomic vector to one score.
"""

import numpy as np


class Aggregator:
    def reduce(self, atomic: np.ndarray) -> float:
        raise NotImplementedError


class ShapleyWeights(Aggregator):
    """Mean over orderings = the Shapley value (additive, efficient, stable)."""

    def reduce(self, atomic: np.ndarray) -> float:
        return float(np.mean(atomic))

    def __repr__(self):
        return "ShapleyWeights()"


class Mean(Aggregator):
    def reduce(self, atomic: np.ndarray) -> float:
        return float(np.mean(atomic))

    def __repr__(self):
        return "Mean()"


class Min(Aggregator):
    """minSHAP: a unit scores high only if it contributes across all contexts.
    """

    def reduce(self, atomic: np.ndarray) -> float:
        return float(np.min(atomic))

    def __repr__(self):
        return "Min()"


class Identity(Aggregator):
    """For statistics that that do not aggregate across samples (LOCO, SAE)."""

    def reduce(self, atomic: np.ndarray) -> float:
        return float(atomic.reshape(-1)[0])

    def __repr__(self):
        return "Identity()"
