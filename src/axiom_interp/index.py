"""The index axis Omega: what we enumerate/sample to probe f.

Each IndexSpace yields a list of *elements* omega. The atomic statistic decides
what an element means (a coalition ordering, a path location, a unit id), so the
index and the statistic are chosen together but stay independently swappable.
"""

import numpy as np


class IndexSpace:
    def elements(self, d: int):
        raise NotImplementedError


class FeatureCoalitions(IndexSpace):
    """IME / Monte-Carlo SHAP: each element is a random feature ordering.

    A single ordering yields one marginal-contribution sample per feature
    (predecessors = the coalition). Averaging over orderings is the unbiased
    Shapley estimate (Strumbelj-Kononenko). We do NOT enumerate the powerset.
    """

    def __init__(self, n_orderings: int = 200, seed: int = 0):
        self.n_orderings = n_orderings
        self.seed = seed

    def elements(self, d: int):
        rng = np.random.default_rng(self.seed)
        return [rng.permutation(d) for _ in range(self.n_orderings)]

    def __repr__(self):
        return f"FeatureCoalitions(n={self.n_orderings}, seed={self.seed})"


class PathSteps(IndexSpace):
    """Integrated-gradients path: each element is an interpolation point alpha."""

    def __init__(self, n_steps: int = 64):
        self.n_steps = n_steps

    def elements(self, d: int):
        # midpoint rule on (0, 1]
        return list((np.arange(1, self.n_steps + 1) - 0.5) / self.n_steps)

    def __repr__(self):
        return f"PathSteps(n={self.n_steps})"


class UnitIndices(IndexSpace):
    """One element per unit: used by leave-one-out / latent-ablation statistics."""

    def elements(self, d: int):
        return list(range(d))

    def __repr__(self):
        return "UnitIndices()"
