"""Wrap SHAP and minSHAP across samples."""

import numpy as np
import pandas as pd
from axiom_interp import presets, compute_count, reset_compute_count


def select_samples(y, y_pred_prob):
    """Pick ~5 representative samples: 2 per class + 1 near the decision boundary.

    Parameters
    ----------
    y : ndarray of int
        True labels in {0, 1}.
    y_pred_prob : ndarray of float
        Predicted P(class 1).

    Returns
    -------
    ndarray of int
        Unique sample indices.
    """
    neg = np.where(y == 0)[0]
    pos = np.where(y == 1)[0]
    borderline = np.argmin(np.abs(y_pred_prob - 0.5))

    picks = [
        neg[0], neg[min(1, len(neg) - 1)],
        pos[0], pos[min(1, len(pos) - 1)],
        borderline,
    ]
    return np.unique(np.array(picks, dtype=int))


def attribute(X, model, feat_names, sample_idx, ecfg, seed, rng):
    """Compute SHAP and minSHAP attributions for selected samples.

    Parameters
    ----------
    X : ndarray, shape (n, d)
    model : classifier
        Must expose ``predict_proba``.
    feat_names : list of str
    sample_idx : ndarray of int
        Row indices into X to explain.
    ecfg : dict
        Keys ``n_background``, ``n_orderings``.
    seed : int
    rng : numpy.random.Generator

    Returns
    -------
    shap_df : DataFrame, shape (len(sample_idx), d)
    minshap_df : DataFrame, shape (len(sample_idx), d)
    """
    n, d = X.shape

    def f(batch):
        return model.predict_proba(batch)[:, 1]

    n_bg = min(ecfg["n_background"], n)
    bg_idx = rng.choice(n, n_bg, replace=False)
    bg = X[bg_idx]

    explainer_shap = presets.shap(bg, ecfg["n_orderings"], seed)
    explainer_min = presets.minshap(bg, ecfg["n_orderings"], seed)

    shap_arr = np.zeros((len(sample_idx), d))
    min_arr = np.zeros((len(sample_idx), d))

    for i, idx in enumerate(sample_idx):
        xi = X[idx]
        reset_compute_count()

        res_shap = explainer_shap.explain(f, xi)
        res_min = explainer_min.explain(f, xi)
        assert compute_count() == 1, "minSHAP should reuse cached tensor"

        shap_arr[i] = res_shap.as_array()
        min_arr[i] = res_min.as_array()

        if i == 0:
            expected = f(xi[None, :])[0] - np.mean(f(bg))
            actual = sum(res_shap.scores.values())
            print(f"Efficiency check: sum(SHAP)={actual:.4f}, f(x)-E[f(bg)]={expected:.4f}")

        print(f"[{i + 1}/{len(sample_idx)}] sample {idx} done")

    shap_df = pd.DataFrame(shap_arr, columns=feat_names)
    minshap_df = pd.DataFrame(min_arr, columns=feat_names)
    return shap_df, minshap_df
