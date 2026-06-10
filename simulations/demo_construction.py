"""Demonstrate the construction grammar on a toy model.

Run: python simulations/demo_construction.py

Toy model: f(x) = x0*x1 + x2.

This demo shows:
  1. SHAP and minSHAP can reuse atomic importance statistics (caching)
  2. minSHAP gives an interacting feature 0 importance (strong null importance) while
     ordinary SHAP splits the interaction.
  3. Swapping the intervention axis (baseline -> marginal v(S)) changes scores.
  4. Same framework gives us Integrated gradients.
  5. Same framework gives us SAEs.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import axiom_interp as ai
from axiom_interp import presets, aggregate


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def f(X):
    X = np.atleast_2d(X)
    return X[:, 0] * X[:, 1] + X[:, 2]


rng = np.random.default_rng(0)
background = rng.normal(size=(256, 4))
x = np.array([1.0, 2.0, 1.5, 3.0])     # x3 irrelevant; x0*x1 = 2; x2 = 1.5
baseline = np.zeros(4)


# --- 1 & 2: SHAP -> minSHAP by one swap, sharing the (exact) cached tensor --
banner("1+2: SHAP and minSHAP share one tensor; min vs mean disagree on x0,x1")
ai.reset_compute_count()
shap = presets.baseline_shap(baseline, n_orderings=400)   # exact => no MC noise
minshap = shap.replace(aggregator=aggregate.Min())        # the recombination

e_shap = shap.explain(f, x)
n1 = ai.compute_count()
e_min = minshap.explain(f, x)
n2 = ai.compute_count()

print("SHAP    scores:", np.round(e_shap.as_array(), 3), " (splits interaction)")
print("minSHAP scores:", np.round(e_min.as_array(), 3), " (interacting features -> 0)")
print(f"tensor computes: after SHAP={n1}, after minSHAP={n2}")
assert n1 == 1 and n2 == 1, "tensor was not reused!"
assert np.allclose(e_min.as_array(), [0, 0, 1.5, 0], atol=1e-6)
print("OK: one swap, zero recompute -- minSHAP reused SHAP's machinery.")


# --- 3: swap the intervention (baseline v(S) -> marginal v(S)) -------------
banner("3: swapping the intervention axis changes the scores")
e_marg = presets.shap(background, n_orderings=400).explain(f, x)
print("baseline-SHAP:", np.round(e_shap.as_array(), 3),
      " gap:", round(ai.properties.efficiency_gap(e_shap, f, x, baseline), 4))
print("marginal-SHAP:", np.round(e_marg.as_array(), 3))


# --- 4: integrated gradients -- different index+atomic, same idea ---------
banner("4: integrated gradients (path index + gradient atomic), completeness")
e_ig = presets.integrated_gradients(baseline, n_steps=128).explain(f, x)
print("IG scores:", np.round(e_ig.as_array(), 3))
print("sum(IG) =", round(sum(e_ig.scores.values()), 4),
      " f(x)-f(baseline) =", round(float(f(x[None])[0] - f(baseline[None])[0]), 4))


# --- 5: mechanistic SAE attribution through the same Explainer -------------
banner("5: SAE feature attribution -- same idea, latent space")
D = np.array([[1.0, 0.0, 0.0, 0.5],
              [0.0, 1.0, 0.0, 0.5],
              [0.0, 0.0, 1.0, 0.0]])
w = np.array([2.0, -1.0, 0.5])


def f_latent(Z):                       # downstream readout . decoder
    Z = np.atleast_2d(Z)
    return (Z @ D.T) @ w


z = np.array([1.0, 1.0, 1.0, 0.0])     # latent 3 inactive -> ~0 attribution
e_sae = presets.sae_attribution().explain(f_latent, z)
print("latent attributions:", np.round(e_sae.as_array(), 3))

print("\nAll demonstrations passed.")
