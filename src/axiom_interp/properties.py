"""Trust-Building Properties

We have two metrics,

  * perturbation stability (input-neighborhood): the Lipschitz constant of the
    explanation map Phi(x), d_E(Phi(f,x), Phi(f,x')) <= L d_X(x,x').
  * null importance
      - functional null: f(x) = f(x_{-j})  -- a property of f alone, probed.
      - injected dummy: a feature we construct to be null, then re-explain.
"""

import numpy as np
from .intervention import MarginalMask, BaselineMask


# --------------------------------------------------------------------------- #
# Additivity                                                                  #
# --------------------------------------------------------------------------- #
def efficiency_gap(obj, f, x, baseline) -> float:
    """|sum_j score_j - (f(x) - f(baseline))|

    Should be 0 if efficiency is satisfied.
    """
    x = np.asarray(x, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    total = sum(obj.scores.values())
    target = float(f(x[None, :])[0] - f(baseline[None, :])[0])
    return abs(total - target)


def null_importance(obj, null_set) -> float:
    """Max |score| over features that should be irrelevant."""
    return max(abs(obj.scores[j]) for j in null_set)


# --------------------------------------------------------------------------- #
# 1. Perturbation stability (input-neighborhood)                              #
# --------------------------------------------------------------------------- #
def stability_replicates(explainer, f, neighbors) -> np.ndarray:
    """Phi(x') for each neighbor x', stacked into an (n, d) array."""
    return np.array([explainer.explain(f, xp).as_array() for xp in np.asarray(neighbors, float)])


def perturbation_stability(explainer, f, x, neighbors, *, metric="lipschitz") -> float:
    """Sensitivity of the explanation map over a random neighborhood.

    `neighbors` is an (n, d) array of points x' (built by the caller, e.g.
    x + rng.normal(scale=radius, size=(n, d)))

      metric="lipschitz"   -> max ||Phi(x') - Phi(x)|| / ||x' - x||
      metric="sensitivity" -> mean ||Phi(x') - Phi(x)||
    """
    x = np.asarray(x, dtype=float)
    neighbors = np.asarray(neighbors, dtype=float)
    phi0 = explainer.explain(f, x).as_array()
    phis = stability_replicates(explainer, f, neighbors)

    dE = np.linalg.norm(phis - phi0, axis=1)
    if metric == "sensitivity":
        return float(np.mean(dE))
    if metric == "lipschitz":
        dX = np.linalg.norm(neighbors - x, axis=1)
        moved = dX > 0
        return float(np.max(dE[moved] / dX[moved])) if np.any(moved) else 0.0

    raise ValueError(f"unknown metric {metric!r}")


# --------------------------------------------------------------------------- #
# 2. Functional null importance: f(x) = f(x_{-j}) probed directly             #
# --------------------------------------------------------------------------- #
def functional_null_strength(f, x, background, j) -> float:
    """How much f at x responds to feature j, others held at x.

    Varies coordinate j across its observed values, holding other coordinates
    fixed, and report std(f). ~0 means f is functionally null in j.
    """
    x = np.asarray(x, dtype=float)
    background = np.asarray(background, dtype=float)
    points = np.tile(x, (len(background), 1))
    points[:, j] = background[:, j]
    return float(np.std(f(points)))


def functional_null_set(f, x, background, *, tol) -> list:
    """Features whose functional strength falls below `tol`

    These variables are considered functionally null."""
    d = len(np.asarray(x, dtype=float))
    return [j for j in range(d) if functional_null_strength(f, x, background, j) < tol]


# --------------------------------------------------------------------------- #
# 3. Injected dummy null                                                      #
# --------------------------------------------------------------------------- #
def with_injected_null(f, x, background, *, baseline=None, n=1):
    """Append `n` features known to be null

    This is the basic idea of https://doi.org/10.18637/jss.v036.i11
    The null variables are permuted versions of existing columns. Doesn't
    account for correlation across features X though. f_aug defined below
    completely ignores the columns after :d.

    Returns (f_aug, x_aug, background_aug, baseline_aug, injected_idx).
    """
    x = np.asarray(x, dtype=float)
    background = np.asarray(background, dtype=float)
    d = len(x)

    dummy = np.column_stack([background[:, i % d][::-1] for i in range(n)])  # (m, n)
    background_aug = np.hstack([background, dummy])
    x_aug = np.concatenate([x, dummy.mean(axis=0)])

    def f_aug(X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return f(X[:, :d])

    baseline_aug = (
        None if baseline is None
        else np.concatenate([np.asarray(baseline, dtype=float), np.zeros(n)])
    )
    return f_aug, x_aug, background_aug, baseline_aug, list(range(d, d + n))


def injected_null(explainer, f, x, background, *, n=1) -> float:
    """Max |score| an explainer assigns to `n` injected null features.
    """
    # build the augmented data/response
    iv = explainer.intervention
    baseline = getattr(iv, "baseline", None)
    f_aug, x_aug, bg_aug, base_aug, injected_idx = with_injected_null(
        f, x, background, baseline=baseline, n=n
    )

    # the intervention needs to be aware of the updated input shape
    if isinstance(iv, MarginalMask):
        new_iv = MarginalMask(bg_aug, n_samples=iv.n_samples, seed=iv.seed)
    elif isinstance(iv, BaselineMask):
        new_iv = BaselineMask(base_aug)
    else:
        new_iv = iv

    # run the explainer on the modified null
    obj = explainer.replace(intervention=new_iv).explain(f_aug, x_aug)
    return max(abs(obj.scores[j]) for j in injected_idx)
