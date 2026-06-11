"""Evaluating trust-building properties

Run: python simulations/demo_properties.py

Toy model: f(x) = x0*x1 + x2   (x3 is null).

  1. Functional null importance -- check (f(x)=f(x_{-j})) to recover the null
     feature set, then check each explainer scores it ~0.
  2. Injected dummy null -- append a feature known to be null and confirm ~0 score.
  3. Perturbation stability -- the Lipschitz constant L of the explanation map
     over an input neighborhood.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from axiom_interp import presets, aggregate, properties


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def f(X):
    X = np.atleast_2d(X)
    return X[:, 0] * X[:, 1] + X[:, 2]


# --- config: generate data and spcify randomness---------------------------- #
rng = np.random.default_rng(0)
background = rng.normal(size=(256, 4))
x = np.array([1.0, 2.0, 1.5, 3.0])     # x3 irrelevant
baseline = np.zeros(4)

shap = presets.shap(background, n_orderings=400)          # marginal v (MC)
minshap = presets.baseline_shap(baseline, n_orderings=400).replace(aggregator=aggregate.Min())
bshap = presets.baseline_shap(baseline, n_orderings=400)  # baseline v (exact)
ig = presets.integrated_gradients(baseline, n_steps=128)


# use different tolerances for different methods
methods = [("marginal-SHAP", shap, 5e-2), ("minSHAP", minshap, 1e-8),
           ("baseline-SHAP", bshap, 1e-8)]


# --- 1: functional null set recovered from f, then null importance ---------- #
banner("1: functional null importance -- recover the null feature, score it ~0")
null_set = properties.functional_null_set(f, x, background, tol=1e-6)
print("functional null set (from f alone):", null_set)
assert null_set == [3], null_set

for name, e, tol in methods:
    obj = e.explain(f, x)
    ni = properties.null_importance(obj, null_set)
    print(f"  {name:13s} score[x3]={obj.scores[3]:+.3e}  null_importance={ni:.3e}")
    assert ni < tol, (name, ni, tol)


# --- 2: injected dummy null ------------------------------------------------ #
banner("2: injected dummy null -- append a known-null feature, expect ~0 score")
for name, e, tol in methods:
    score = properties.injected_null(e, f, x, background, n=1)
    print(f"  {name:13s} |score[injected]| = {score:.3e}")
    assert score < tol, (name, score, tol)


# --- 3: perturbation stability (Lipschitz L over an input neighborhood) ----- #
banner("3: perturbation stability -- L of the explanation map (smaller = stabler)")
neighbors = x + rng.normal(scale=0.05, size=(64, 4))

L = {}
for name, e in [("baseline-SHAP", bshap), ("minSHAP", minshap),
                ("IG", ig), ("marginal-SHAP", shap)]:
    L[name] = properties.perturbation_stability(e, f, x, neighbors, metric="lipschitz")
    print(f"  {name:13s} L = {L[name]:.3f}")

assert L["marginal-SHAP"] > L["baseline-SHAP"], L

# spot-check: sensitivity -> 0 as the neighborhood collapses
tiny = x + rng.normal(scale=1e-9, size=(16, 4))
s0 = properties.perturbation_stability(bshap, f, x, tiny, metric="sensitivity")
assert s0 < 1e-6, s0
print(f"\nsensitivity at radius~1e-9: {s0:.2e}")

print("\nAll demonstrations passed.")
