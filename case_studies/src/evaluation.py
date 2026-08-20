"""Evaluate importance scores as tests of null importance.

The null-importance framework (Definitions 2.2--2.9) defines population level
importance statistics ``phi_j`` and articulate differents kinds of null
importance hypotheses that these statistics might be sensitive to.

    H0(j, notion): feature j is null under `notion`
    reject:        the method's score for j is too large to be null, judged
                   against the reference its statistic admits

A false positive is a null feature called relevant (type I error). A false
negative is a missed relevant feature (type II error).

For many of these statistics, it's not entirely clear what decision threshold to
be using. We've considered two strategies.

:func:`zero_reference_calls` - This rejects if the null importance statistic is
    larger than zero. It's the most reliable strategy and doesn't require any
    external knowledge. However, we have so far only applied it to minSHAP.

:func:`pad_pvalues` - This rejects if the important statistics are above a
    quantile of the statistics observed in the noise features. This requires
    external knowledge about which features are known not to be related to the
    response.  This is our default strategy for most methods.

We also write :func:`null_mass` It measures the proportion of importance that
goes to the null features. This can be computed without a threshold and is a
simple diagnostic that doesn't require the two-by-two table decisions.
"""

import logging
import re
import numpy as np
import pandas as pd

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


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------

def pad_pvalues(importance, is_pad):
    """One-sided p-values against the pad scores in the same block.

    Args:
        importance: Scores for one block, in feature order.
        is_pad: Boolean mask marking the iid noise features.

    Returns:
        Array of p-values. The smallest attainable value is
        ``1 / (1 + n_pads)``, so ``alpha`` below that rejects nothing.
    """
    importance = np.asarray(importance, dtype=float)
    importance = np.where(np.isfinite(importance), importance, -np.inf)
    pads = importance[np.asarray(is_pad, dtype=bool)]
    if pads.size == 0:
        raise ValueError("a block with no pad features has no null reference")

    exceed = (pads[None, :] >= importance[:, None]).sum(axis=1)
    return (1.0 + exceed) / (1.0 + pads.size)


def zero_reference_calls(importance):
    """Detect nonnull features for statistics assumed to be zero under the null

    For methods like a minSHAP, we can reject if the importance is above zero.
    """
    return np.asarray(importance, dtype=float) > 0


def call_nonnull(scores, truth, alpha=0.1, block_keys=BLOCK_KEYS,
                 zero_reference_methods=()):
    """Decide whether a feature is nonnull

    A block includes all features from a single combination of:
        - One dataset (e.g., "linear_additive", "parity")
        - One sample size (n)
        - One response type (classification or regression)
        - One random seed
        - One method (e.g., "minshap", "sage", "knockoffs")

    We refer only to the features within this block when flagging features as
    null or non-null (we might refer to the other features in the same block if
    we're calibrating using noise features like in func::`pad_pvalues`, for
    example. )

    Args:
        scores: Frame returned by :func:`load_scores`.
        truth: Long ground-truth frame; supplies ``null_kind``, which marks the
            pads.
        alpha: Level at which to reject, for the pad-referenced methods.
        block_keys: Columns defining one comparable set of scores.
        zero_reference_methods: Methods whose statistic is zero under the null.
          Uses a simpler rejection rule but doesn't return p-values.

    Returns:
        A copy of ``scores`` with float column ``p_value``, Boolean column
        ``called_nonnull``, and string column ``calibration`` recording which
        reference produced the call.
    """
    # initialize results
    kinds = truth[["dataset", "feature", "null_kind"]].drop_duplicates()
    out = scores.merge(kinds, on=["dataset", "feature"], how="left")

    zero_reference_methods = set(zero_reference_methods)
    out["p_value"] = np.nan
    out["called_nonnull"] = False
    out["calibration"] = "pad"

    # call both zero reference and padding approaches
    for _, block in out.groupby(block_keys, sort=False):
        if block["method"].iat[0] in zero_reference_methods:
            calls = zero_reference_calls(block["importance"].values)
            out.loc[block.index, "calibration"] = "zero"
        else:
            p_value = pad_pvalues(
                block["importance"].values, block["null_kind"].values == "pad"
            )
            out.loc[block.index, "p_value"] = p_value
            calls = p_value <= alpha
        out.loc[block.index, "called_nonnull"] = calls
    return out


# ---------------------------------------------------------------------------
# 2x2 tables and error rates
# ---------------------------------------------------------------------------

def confusion(labeled, truth, group_keys=BLOCK_KEYS):
    """Compare calls with the ground truth for each notion.

    The null hypothesis is that feature ``j`` is null under the notion.  ``fp``
    is the type I error and ``fn`` is the type II error.

    Args:
        labeled: Output of :func:`call_nonnull`.
        truth: Long ground-truth frame with ``dataset``, ``feature``,
            ``notion``, ``is_null``, and ``null_kind``.
        group_keys: Columns defining each table, in addition to ``notion``.

    Returns:
        A frame with ``group_keys``, ``notion``, ``tn``, ``fp``, ``fn``,
        ``tp``, ``n_null``, and ``n_nonnull``.
    """
    joined = labeled.drop(columns="null_kind", errors="ignore").merge(
        truth, on=["dataset", "feature"], how="inner"
    )

    joined["tp"] = ~joined["is_null"] & joined["called_nonnull"]
    joined["fn"] = ~joined["is_null"] & ~joined["called_nonnull"]
    joined["fp"] = joined["is_null"] & joined["called_nonnull"]
    joined["tn"] = joined["is_null"] & ~joined["called_nonnull"]

    keys = list(group_keys) + ["notion"]
    counts = joined.groupby(keys, sort=False)[["tn", "fp", "fn", "tp"]].sum()
    counts = counts.reset_index()
    counts["n_null"] = counts["tn"] + counts["fp"]
    counts["n_nonnull"] = counts["tp"] + counts["fn"]
    return counts[keys + ["tn", "fp", "fn", "tp", "n_null", "n_nonnull"]]


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


def null_mass(labeled, truth, group_keys=BLOCK_KEYS):
    """What fraction of total variable importance is assigned to null features?

    ``sum |phi| over null features / (sum |phi| over all features)
    Zero means the method never assigns any importance to the null features, one
    means it assigns all of its mass to them.  Notice that this can get
    influenced by the total and original features in the dataset.

    Args:
        labeled: Output of :func:`call_nonnull`.
        truth: Long ground-truth frame with ``dataset``, ``feature``,
            ``notion``, ``is_null``, and ``null_kind``.
        group_keys: Columns defining each proportion, in addition to
            ``notion``.

    Returns:
        A frame with ``group_keys``, ``notion``, ``null_magnitude``,
        ``nonnull_magnitude``, and their proportion ``null_mass``. Blocks with
        no mass anywhere give NaN.
    """
    joined = labeled.drop(columns="null_kind", errors="ignore").merge(
        truth, on=["dataset", "feature"], how="inner"
    )
    joined["magnitude"] = joined["importance"].abs()

    keys = list(group_keys) + ["notion"]
    sums = (
        joined.groupby(keys + ["is_null"], sort=False)["magnitude"]
        .sum()
        .unstack("is_null")
        .reindex(columns=[False, True])
        .rename(columns={False: "nonnull_magnitude", True: "null_magnitude"})
        .reset_index()
    )
    sums["null_mass"] = _ratio(
        sums["null_magnitude"],
        sums["null_magnitude"] + sums["nonnull_magnitude"],
    )
    return sums[keys + ["null_magnitude", "nonnull_magnitude", "null_mass"]]


def pool(counts, over=("seed",)):
    """Sum 2x2 counts over ``over`` before calculating rates.

    For example, pooling ten seeds gives a more stable false-positive estimate
    than calculating a rate from six null features at one seed.
    """
    count_cols = ["tn", "fp", "fn", "tp", "n_null", "n_nonnull"]
    keys = [c for c in counts.columns if c not in count_cols and c not in over]
    return counts.groupby(keys, sort=False)[count_cols].sum().reset_index()


# ---------------------------------------------------------------------------
# Empirical version of Table 1
# ---------------------------------------------------------------------------

def empirical_table(rates, alpha=0.1, power_target=0.5, bands=(0.4, 0.8),
                    cell_keys=("dataset", "response_type", "n"),
                    categories=None, group_keys=("method", "notion")):
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
        rates: Output of :func:`error_rates`.
        alpha: Target type I error rate.
        power_target: Minimum power counted as non-trivial.
        bands: Lower and upper control fractions.
        cell_keys: Columns identifying one simulated cell.
        categories: Optional mapping ``{method: [null_type, ...]}`` identifying
            each method's nominal target null_types.
        group_keys: Columns identifying what to pool over.

    Returns:
        A frame with the cell counts, fractions, error rates, and
        ``is_target_null_type`` for each method and null_type (and any extra
        ``group_keys``).
    """
    low, high = bands
    group_keys = list(group_keys)
    cell_keys = [key for key in cell_keys if key not in group_keys]
    rows = []
    for group_values, group in rates.groupby(group_keys, sort=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        cells = group.groupby(cell_keys, sort=False)[["fpr", "power"]].mean()
        defined_fpr = cells["fpr"].dropna()
        defined_power = cells["power"].dropna()

        # Compare against false positive and negative thresholds.
        controlled = (
            float((defined_fpr <= alpha).mean()) if len(defined_fpr) else np.nan
        )
        powered = (
            float((defined_power >= power_target).mean())
            if len(defined_power) else np.nan
        )

        # Assign a symbol to the different categories
        if np.isnan(controlled) or controlled < low:
            symbol = "cross"
        elif controlled >= high and (np.isnan(powered) or powered >= high):
            symbol = "check"
        else:
            symbol = "tilde"

        # Return summary statistics of the method and its overall assignment.
        group_dict = dict(zip(group_keys, group_values))
        target = group_dict["notion"] in (categories or {}).get(group_dict["method"], [])
        rows.append({
            **group_dict,
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
