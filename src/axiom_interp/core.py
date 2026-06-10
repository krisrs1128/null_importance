"""Compose Explainers from Parts

    E = A( { I_j(f, x, omega) }_omega )

The atomic-statistics { I_j(.) } are computed once and saved. Swapping only the
aggregator (SHAP -> minSHAP) reuses those summaries.
"""

from dataclasses import dataclass, replace as _dc_replace

import numpy as np

# Module-level cache shared across explainers.
_TENSOR_CACHE: dict = {}
_COMPUTE_COUNT = [0]


def reset_compute_count():
    _COMPUTE_COUNT[0] = 0


def compute_count() -> int:
    return _COMPUTE_COUNT[0]


@dataclass(frozen=True)
class ExplanatoryObject:
    scores: dict          # unit -> float
    provenance: dict      # the four components
    tensor: dict          # unit -> atomic array (for the property layer)

    def as_array(self) -> np.ndarray:
        return np.array([self.scores[j] for j in sorted(self.scores)])


@dataclass(frozen=True)
class Explainer:
    index: object
    intervention: object
    atomic: object
    aggregator: object

    def _tensor_key(self, f, x):
        return (
            id(f),
            np.asarray(x, dtype=float).tobytes(),
            repr(self.index),
            repr(self.intervention),
            repr(self.atomic),
        )

    def atomic_tensor(self, f, x) -> dict:
        """Compute (or fetch from cache) the atomic-statistics tensor."""
        key = self._tensor_key(f, x)
        if key not in _TENSOR_CACHE:
            elements = self.index.elements(len(x))
            _TENSOR_CACHE[key] = self.atomic.compute_all(
                f, self.intervention, elements, np.asarray(x, dtype=float)
            )
            _COMPUTE_COUNT[0] += 1
        return _TENSOR_CACHE[key]

    def explain(self, f, x, P=None) -> ExplanatoryObject:
        tensor = self.atomic_tensor(f, x)
        scores = {j: self.aggregator.reduce(v) for j, v in tensor.items()}
        return ExplanatoryObject(scores=scores, provenance=self._provenance(), tensor=tensor)

    def replace(self, **changes) -> "Explainer":
        """Return a new Explainer with some axes swapped (the recombination op)."""
        return _dc_replace(self, **changes)

    def _provenance(self) -> dict:
        return {
            "index": repr(self.index),
            "intervention": repr(self.intervention),
            "atomic": repr(self.atomic),
            "aggregator": repr(self.aggregator),
        }
