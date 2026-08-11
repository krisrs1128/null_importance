"""Evaluate importance scores as tests of null importance.

The null-importance framework (Definitions 2.2--2.9) are defined at the
population level.  Definitions 4.1 and 4.2 ask whether an importance statistic
respects ``phi_j = 0 => N_j``. We test this through simulations.

    H0(j, notion): feature j is null under `notion`
    reject:        the method's score for j exceeds a fraction `threshold` of
                   the largest score in its block

A false positive is a null feature called relevant (type I error). A false
negative is a missed relevant feature (type II error).
"""

import logging
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

log = logging.getLogger(__name__)

#: Keys identifying one comparable set of scores.
BLOCK_KEYS = ["dataset", "n", "response_type", "seed", "method"]

_RESULT_PATTERN = re.compile(
    r"^(?P<dataset>.+)"
    r"_(?P<n>\d+)"
    r"_(?P<response_type>classification|regression)"
    r"_(?P<seed>\d+)"
    r"_(?P<method>.+)$"
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_scores(results_dir, methods):
    """Read the importance files from each method

    Files must follow the ``{dataset}_{n}_{response_type}_{seed}_{method}.csv``
    format used by ``sweep.py``.

    Args:
        results_dir: Directory containing the result files.
        methods: Iterable of recognised method names.

    Returns:
        A frame with columns ``dataset``, ``n``, ``response_type``, ``seed``,
        ``method``, ``feature``, and ``importance``.
    """
    methods = set(methods)
    frames = []
    for path in sorted(results_dir.glob("*.csv")):
        match = _RESULT_PATTERN.match(path.stem)
        if match is None:
            log.debug(f"Skipping unparsable result file {path.name}")
            continue

        fields = match.groupdict()
        if fields["method"] not in methods:
            continue
        frame = pd.read_csv(path)
        frame["dataset"] = fields["dataset"]
        frame["n"] = int(fields["n"])
        frame["response_type"] = fields["response_type"]
        frame["seed"] = int(fields["seed"])
        frame["method"] = fields["method"]
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)[
        BLOCK_KEYS + ["feature", "importance"]
    ]


def coverage(scores, datasets, sample_sizes, response_types, seeds, methods):
    """Compare observed scores for the current data block

    Returns:
        A frame with one row per configured block and columns ``dataset``,
        ``n``, ``response_type``, ``seed``, ``method``, and ``present``.
    """
    grid = pd.MultiIndex.from_product(
        [list(datasets), list(sample_sizes), list(response_types),
         list(seeds), list(methods)],
        names=BLOCK_KEYS,
    ).to_frame(index=False)
    observed = scores[BLOCK_KEYS].drop_duplicates()
    observed["present"] = True
    merged = grid.merge(observed, on=BLOCK_KEYS, how="left")
    merged["present"] = merged["present"].fillna(False).astype(bool)
    return merged


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------

def normalize_scores(importance):
    """Map scores to ``[0, 1]``, treating zero as no evidence of relevance.

    Negative scores count as evidence against relevance: examples include a
    knockoff statistic favouring the knockoff copy and a minSHAP contribution
    for which adding the feature raises risk. They are clipped at zero.
    """
    clipped = np.clip(np.asarray(importance, dtype=float), 0.0, None)
    largest = clipped.max() if clipped.size else 0.0
    if not np.isfinite(largest) or largest <= 0.0:
        return np.zeros_like(clipped)
    return clipped / largest


def call_nonnull(scores, threshold=0.1, block_keys=BLOCK_KEYS):
    """Add normalized scores and non-null calls within each block.

    A feature is called non-null when its normalized score exceeds
    ``threshold``. With infinite data, we would force threshold = 0 to declare a
    feature nonnull.

    Args:
        scores: Frame returned by :func:`load_scores`.
        threshold: Fraction of the block maximum above which to reject.
        block_keys: Columns defining one comparable set of scores.

    Returns:
        A copy of ``scores`` with float column ``normalized`` and Boolean
        column ``called_nonnull``.
    """
    out = scores.copy()
    out["normalized"] = np.nan
    for _, block in out.groupby(block_keys, sort=False):
        out.loc[block.index, "normalized"] = normalize_scores(
            block["importance"].values
        )
    out["called_nonnull"] = out["normalized"] > threshold
    return out


# ---------------------------------------------------------------------------
# 2x2 tables and error rates
# ---------------------------------------------------------------------------

def _scope(labeled, null_scope):
    """Keep either all features or only the structural signal features.

    Random noise "pads" are null under every notion and easy to classify.
    Pooling them with structural nulls (e.g., conditional null features) makes
    methods look better than they are. ``null_scope="signal"`` removes pads.
    """
    if null_scope == "all":
        return labeled
    if null_scope == "signal":
        return labeled[labeled["null_kind"] != "pad"]
    raise ValueError(f"null_scope must be 'all' or 'signal'; got {null_scope!r}")


def confusion(labeled, truth, group_keys=BLOCK_KEYS, null_scope="all"):
    """Compare calls with the ground truth for each notion.

    The null hypothesis is that feature ``j`` is null under the notion.
    Rejecting it means calling the feature non-null; therefore ``fp`` is the
    type I count and ``fn`` is the type II count.

    Args:
        labeled: Output of :func:`call_nonnull`.
        truth: Long ground-truth frame with ``dataset``, ``feature``,
            ``notion``, ``is_null``, and ``null_kind``.
        group_keys: Columns defining each table, in addition to ``notion``.
        null_scope: ``"all"`` or ``"signal"``; see :func:`_scope`.

    Returns:
        A frame with ``group_keys``, ``notion``, ``null_scope``, ``tn``, ``fp``,
        ``fn``, ``tp``, ``n_null``, and ``n_nonnull``.
    """
    joined = labeled.merge(truth, on=["dataset", "feature"], how="inner")
    joined = _scope(joined, null_scope)

    joined["tp"] = ~joined["is_null"] & joined["called_nonnull"]
    joined["fn"] = ~joined["is_null"] & ~joined["called_nonnull"]
    joined["fp"] = joined["is_null"] & joined["called_nonnull"]
    joined["tn"] = joined["is_null"] & ~joined["called_nonnull"]

    keys = list(group_keys) + ["notion"]
    counts = joined.groupby(keys, sort=False)[["tn", "fp", "fn", "tp"]].sum()
    counts = counts.reset_index()
    counts["n_null"] = counts["tn"] + counts["fp"]
    counts["n_nonnull"] = counts["tp"] + counts["fn"]
    counts["null_scope"] = null_scope
    return counts[keys + ["null_scope", "tn", "fp", "fn", "tp",
                          "n_null", "n_nonnull"]]


def _ratio(numerator, denominator):
    """Divide elementwise, returning NaN when the denominator is zero."""
    denominator = denominator.astype(float)
    return numerator.astype(float).div(denominator).where(denominator > 0)


def error_rates(counts):
    """Add standard testing rates to a confusion table.

    The main rates are ``fpr = fp / (fp + tn)`` and ``power = tp / (tp + fn)``;
    precision, false-discovery proportion, and F1 are also included.
    """
    out = counts.copy()
    out["fpr"] = _ratio(out["fp"], out["fp"] + out["tn"])
    out["power"] = _ratio(out["tp"], out["tp"] + out["fn"])
    out["precision"] = _ratio(out["tp"], out["tp"] + out["fp"])
    out["fdp"] = _ratio(out["fp"], out["fp"] + out["tp"])
    out["f1"] = _ratio(2 * out["tp"], 2 * out["tp"] + out["fp"] + out["fn"])
    return out


def pool(counts, over=("seed",)):
    """Sum 2x2 counts over ``over`` before calculating rates.

    For example, pooling ten seeds gives a more stable false-positive estimate
    than calculating a rate from six null features at one seed.
    """
    count_cols = ["tn", "fp", "fn", "tp", "n_null", "n_nonnull"]
    keys = [c for c in counts.columns if c not in count_cols and c not in over]
    return counts.groupby(keys, sort=False)[count_cols].sum().reset_index()


def threshold_sweep(scores, truth, thresholds, group_keys=BLOCK_KEYS,
                    null_scopes=("all", "signal")):
    """Calculate error and classification rates at several thresholds."""
    frames = []
    for threshold in thresholds:
        labeled = call_nonnull(scores, threshold=threshold,
                               block_keys=group_keys)
        for null_scope in null_scopes:
            rates = error_rates(
                confusion(labeled, truth, group_keys, null_scope)
            )
            rates["threshold"] = threshold
            frames.append(rates)
    return pd.concat(frames, ignore_index=True)


def curve_summary(scores, truth, group_keys=BLOCK_KEYS,
                  null_scopes=("all", "signal")):
    """Summarise ROC AUC and average precision by block and notion.

    This can be used to summarize method performance without thresholding.
    """
    joined = scores.merge(truth, on=["dataset", "feature"], how="inner")
    keys = list(group_keys) + ["notion"]

    rows = []
    for null_scope in null_scopes:
        scoped = _scope(joined, null_scope)
        for key_values, block in scoped.groupby(keys, sort=False):
            y_true = (~block["is_null"]).astype(int).values
            y_score = block["importance"].values.astype(float)
            if y_true.min() == y_true.max() or not np.isfinite(y_score).all():
                roc, average_precision = np.nan, np.nan
            else:
                roc = roc_auc_score(y_true, y_score)
                average_precision = average_precision_score(y_true, y_score)
            rows.append({
                **dict(zip(keys, key_values)),
                "null_scope": null_scope,
                "roc_auc": roc,
                "average_precision": average_precision,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Empirical version of Table 1
# ---------------------------------------------------------------------------

def empirical_table(rates, alpha=0.1, power_target=0.5, bands=(0.4, 0.8),
                    cell_keys=("dataset", "response_type", "n"),
                    categories=None):
    """Reduce cell-level rates to one verdict per method and null type.

    Each ``(dataset, response_type, n)`` is a simulated cell. A method controls
    a null_type when its type I rate is at most ``alpha``; it is powered when
    its power is at least ``power_target``. The verdict records the fraction of
    cells satisfying each condition:

        check -- control holds in at least bands[1] of cells and, when power
                 is defined, power does too
        tilde -- control holds in at least bands[1] but power does not, or
                 control holds in a fraction between bands[0] and bands[1]
        cross -- control holds in fewer than bands[0] of cells

    Power is required for a checkmark. Without that requirement, a method that
    never rejects a feature would pass every null_type.

    Args:
        rates: Output of :func:`error_rates`, for one ``null_scope``.
        alpha: Target type I error rate.
        power_target: Minimum power counted as non-trivial.
        bands: Lower and upper control fractions.
        cell_keys: Columns identifying one simulated cell.
        categories: Optional mapping ``{method: [null_type, ...]}`` identifying
            each method's nominal target null_types.

    Returns:
        A frame with the verdict, cell counts, fractions, mean rates, and
        ``is_target_null_type`` for each method and null_type.
    """
    low, high = bands
    cell_keys = list(cell_keys)
    rows = []
    for (method, notion), group in rates.groupby(["method", "notion"], sort=False):
        cells = group.groupby(cell_keys, sort=False)[["fpr", "power"]].mean()
        defined_fpr = cells["fpr"].dropna()
        defined_power = cells["power"].dropna()

        controlled = (
            float((defined_fpr <= alpha).mean()) if len(defined_fpr) else np.nan
        )
        powered = (
            float((defined_power >= power_target).mean())
            if len(defined_power) else np.nan
        )

        if np.isnan(controlled) or controlled < low:
            symbol = "cross"
        elif controlled >= high and (np.isnan(powered) or powered >= high):
            symbol = "check"
        else:
            symbol = "tilde"

        target = notion in (categories or {}).get(method, [])
        rows.append({
            "method": method,
            "notion": notion,
            "n_cells": len(defined_fpr),
            "n_power_cells": len(defined_power),
            "controlled_fraction": controlled,
            "powered_fraction": powered,
            "mean_fpr": float(defined_fpr.mean()) if len(defined_fpr) else np.nan,
            "mean_power": float(defined_power.mean()) if len(defined_power) else np.nan,
            "symbol": symbol,
            "is_target_notion": target,
        })
    return pd.DataFrame(rows)
