"""axiom_interp: a grammar of interpretability.

Every method is a point in (index x intervention x atomic x aggregator). Build
one, move a single axis to get another:

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
