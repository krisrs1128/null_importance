"""Grammar of Interpretability

Every method is determined by (index x intervention x atomic x aggregator).

    >>> e = presets.shap(background)
    >>> e_min = e.replace(aggregator=aggregate.Min())   # minSHAP
"""

from . import index, intervention, atomic, aggregate, presets, properties
from .core import Explainer, ExplanatoryObject, compute_count, reset_compute_count

__all__ = [
    "index",
    "intervention",
    "atomic",
    "aggregate",
    "presets",
    "properties",
    "Explainer",
    "ExplanatoryObject",
    "compute_count",
    "reset_compute_count",
]
