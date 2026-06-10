"""The evaluation layer (minimal stub).

Properties take ExplanatoryObject's as inputs and return whether the object
satisfies the property.
"""

import numpy as np


def efficiency_gap(obj, f, x, baseline) -> float:
    """|sum_j score_j - (f(x) - f(baseline))|. ~0 for additive methods."""
    x = np.asarray(x, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    total = sum(obj.scores.values())
    target = float(f(x[None, :])[0] - f(baseline[None, :])[0])
    return abs(total - target)


def null_importance(obj, null_units) -> float:
    """Max |score| over units that should be irrelevant. Smaller is better."""
    return max(abs(obj.scores[j]) for j in null_units)
