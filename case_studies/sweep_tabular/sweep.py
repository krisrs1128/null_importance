"""Feature importance methods on the Sweep 1 synthetic datasets.

Explains each dataset's data-generating function (see SCORES in model.py),
treating its output as predictions, rather than training a secondary model to
use in the explanation.

Run from the repo root, after generate.py creates data/{dataset}_{n}_{response_type}_{seed}.csv:
    python case_studies/sweep_tabular/sweep.py

To save additional summaries about the risk-based methods, see risk_summaries.py.
"""

import hashlib
import inspect
import logging
import sys
import warnings
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from omegaconf import DictConfig, OmegaConf
from datasets import DATASETS, dataset_response_types
from model import (
    SCORES,
    FunctionClassifier,
    FunctionRegressor,
    fit_explanation_tree,
)

# setup logging and imports from local scripts
log = logging.getLogger(__name__)
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir.parent / "src"))
sys.path.insert(0, str(_script_dir.parents[1] / "src"))
from importance import FITTED_MODEL_METHODS, METHODS


def _save_method(results_dir, name, series):
    pd.DataFrame({"feature": series.index, "importance": series.values}).to_csv(
        results_dir / f"{name}.csv", index=False
    )


def _task_rng(seed, dataset, response_type, n, method):
    """Set a new RNG for each task

    This is more robust in the case that we resume a sweep and only run a subset
    of tasks the second time around.
    """
    label = f"{dataset}|{response_type}|{n}|{method}".encode()
    digest = hashlib.blake2b(label, digest_size=8).digest()
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int.from_bytes(digest, "big")])
    )


def _run_task(cfg_dict, dataset, response_type, n, seed, method, data_dir, results_dir):
    """Save one method on one dataset "block".
    """
    warnings.simplefilter("ignore")

    stem = f"{dataset}_{n}_{response_type}_{seed}_{method}"
    if (results_dir / f"{stem}.csv").exists():
        return f"skipped {stem}"

    # The risk depends on the function class, and each data generating process
    # specifies what type of risk to use in its config (e.g. linear vs. rich
    # xgboost model)
    base_cfg = {**cfg_dict["dimensions"], **cfg_dict["datasets"][dataset]}
    risk_cfg = {
        **cfg_dict["risk"],
        "model_class": base_cfg.get("risk_class", cfg_dict["risk"]["default_class"]),
    }
    dataset_cfg = {**base_cfg, "response_type": response_type}

    df = pd.read_csv(data_dir / f"{dataset}_{n}_{response_type}_{seed}.csv")
    X_df = df.drop(columns=["y"])
    y = df["y"].values
    X = X_df.values
    feature_names = list(X_df.columns)
    rng = _task_rng(seed, dataset, response_type, n, method)

    # the hypothetical model (or a CART for methods that need a tree)
    ModelCls = FunctionClassifier if response_type == "classification" else FunctionRegressor
    model = ModelCls(SCORES[dataset], dataset_cfg).fit(X_df)
    if method in FITTED_MODEL_METHODS:
        model = fit_explanation_tree(
            X, y, response_type, cfg_dict["explanation_tree"],
            int(rng.integers(1, 2**31)),
        )

    ctx = {
        "model": model, "X": X, "y": y,
        "X_df": X_df,
        "feature_names": feature_names,
        "rng": rng, "seed": seed,
        "response_type": response_type,
        "n_repeats": cfg_dict["permutation"]["n_repeats"],
        "risk_cfg": risk_cfg,
        "cfg": {**cfg_dict, "risk": risk_cfg},
        "kcfg": cfg_dict["knockoffs"],
        "grid_resolution": cfg_dict["pdp"]["grid_resolution"],
        "ig_cfg": cfg_dict["integrated_gradients"],
        "gcm_cfg": cfg_dict["gcm"],
    }

    fn = METHODS[method]
    result = fn(**{p: ctx[p] for p in inspect.signature(fn).parameters})
    _save_method(results_dir, stem, result)
    return f"done {stem}"


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):

    # read configuration and setup output directories
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    data_dir = _script_dir / "data"
    results_dir = _script_dir / "results"
    results_dir.mkdir(exist_ok=True)

    methods = [
        name for name in METHODS if cfg_dict["methods"].get(name, False)
    ]
    for name in DATASETS:
        if name not in cfg_dict["datasets"]:
            log.info(f"Skipping {name}: no entry in config datasets")

    tasks = [
        (name, rt, n, seed, method)
        for seed in cfg_dict["seeds"]
        for name in DATASETS
        if name in cfg_dict["datasets"]
        for rt in dataset_response_types(cfg_dict, name)
        for n in cfg_dict["sample_sizes"]
        for method in methods
    ]
    log.info(f"{len(tasks)} tasks over {len(methods)} methods")

    # parallelize across tasks
    n_jobs = cfg_dict.get("parallel", {}).get("n_jobs", -1)
    with parallel_backend("loky", inner_max_num_threads=1):
        Parallel(n_jobs=n_jobs, batch_size=1, verbose=0)(
            delayed(_run_task)(
                cfg_dict, name, rt, n, seed, method, data_dir, results_dir
            )
            for name, rt, n, seed, method in tasks
        )


if __name__ == "__main__":
    main()
