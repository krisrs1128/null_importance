"""Write each DGP's per-feature null-importance labels as a table.

Each dataset is associated with ``meta["null_type"]`` which says which type of
null importance each feature satisfies. These functions organize that annotation
into a table used for the reporting and figures.
"""

import numpy as np
import pandas as pd

from datasets import DATASETS, NULL_NOTIONS

PAD_PREFIX = "noise_"


def null_truth_frame(cfg_dict, n=8, response_type="regression", seed=0):
    """Return one long ground-truth row for each dataset, feature, and notion.

    Args:
        cfg_dict: Resolved config. Uses ``dimensions`` and the per-DGP blocks
            in ``datasets``.
        n: Number of rows in the throwaway DGP draw.
        response_type: Passed to the DGP; it does not change the labels.
        seed: Seed for the throwaway draw.

    Returns:
        A frame with ``dataset``, ``feature``, ``notion``, ``is_null``, and
        ``null_kind``. It has ``n_datasets * n_features *
        len(NULL_NOTIONS)`` rows.
    """
    rows = []
    for name in cfg_dict["datasets"]:
        if name not in DATASETS:
            raise KeyError(
                f"config lists dataset {name!r} but datasets.py has no such DGP"
            )
        dataset_cfg = {
            **cfg_dict["dimensions"],
            **cfg_dict["datasets"][name],
            "response_type": response_type,
        }
        rng = np.random.default_rng(seed)
        _, _, feature_names, meta = DATASETS[name](n, rng, dataset_cfg)

        for feature in feature_names:
            null_notions = set(meta["null_type"][feature])
            unknown = null_notions - set(NULL_NOTIONS)
            if unknown:
                raise ValueError(
                    f"{name}/{feature} tagged with unknown notions {sorted(unknown)}"
                )
            kind = "pad" if feature.startswith(PAD_PREFIX) else "signal"
            for notion in NULL_NOTIONS:
                rows.append({
                    "dataset": name,
                    "feature": feature,
                    "notion": notion,
                    "is_null": notion in null_notions,
                    "null_kind": kind,
                })
    return pd.DataFrame(rows)


def coarse_null_label(truth):
    """Return one display label per dataset-feature pair.

    Let ``S_j`` be the set of notions under which feature ``j`` is null. The
    label is ``Signal`` if ``S_j`` is empty, ``Null`` if ``S_j`` equals
    ``NULL_NOTIONS``, and otherwise ``<first notion> null``, where "first" uses
    ``NULL_NOTIONS`` order.

    Args:
        truth: Frame returned by :func:`null_truth_frame`.

    Returns:
        A frame with ``dataset``, ``feature``, and ``null_label``.
    """
    def label(group):
        null_notions = set(group.loc[group["is_null"], "notion"])
        if not null_notions:
            return "Signal"
        if len(null_notions) == len(NULL_NOTIONS):
            return "Null"
        first = next(n for n in NULL_NOTIONS if n in null_notions)
        return f"{first.capitalize()} null"

    return (
        truth.groupby(["dataset", "feature"], sort=False)
        .apply(label, include_groups=False)
        .rename("null_label")
        .reset_index()
    )
