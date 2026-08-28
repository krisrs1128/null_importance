"""Feature importance methods on the TCGA BRCA multi-omics classifier.

Explain the random forest model from
``case_studies/src/attribute_classifier.py``, using settings defined in
``config.yaml``. After you've saved the data to data/X.parquet and the fitted
model to results/final_model.pkl:

    python case_studies/tcga_brca/sweep.py

For plots, see vignettes.py and visualize_importance.R.
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

# setup logging and imports from local scripts
log = logging.getLogger(__name__)
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir.parent / "src"))
sys.path.insert(0, str(_script_dir.parents[1] / "src"))
from importance import METHODS
from results import load_model


def _save_method(results_dir, name, series):
    pd.DataFrame({"feature": series.index, "importance": series.values}).to_csv(
        results_dir / f"{name}.csv", index=False
    )


def _task_rng(seed, dataset, response_type, n, method):
    """Random numbers for each task.

    This is more robust in the case that we resume a sweep and only run a subset
    of tasks the second time around.
    """
    label = f"{dataset}|{response_type}|{n}|{method}".encode()
    digest = hashlib.blake2b(label, digest_size=8).digest()
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int.from_bytes(digest, "big")])
    )


def _run_task(cfg_dict, bundle, X_df, y, seed, method, results_dir):
    """Save one method on one seed."""
    warnings.simplefilter("ignore")

    dataset = cfg_dict["dataset"]
    response_type = cfg_dict["response_type"]
    n = len(y)
    stem = f"{dataset}_{n}_{response_type}_{seed}_{method}"
    if (results_dir / f"{stem}.csv").exists():
        return f"skipped {stem}"

    # What features should be explained?
    feature_names = bundle["feature_names"]
    X_df = X_df[feature_names]
    risk_cfg = {
        **cfg_dict["risk"],
        "model_class": cfg_dict["risk"]["default_class"],
    }
    rng = _task_rng(seed, dataset, response_type, n, method)

    ctx = {
        "model": bundle["model"], "X": X_df.to_numpy(dtype=float), "y": y,
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

    X_df = pd.read_parquet(data_dir / "X.parquet")
    y = pd.read_parquet(data_dir / "y.parquet").iloc[:, 0].values
    bundle = load_model(results_dir)

    methods = [
        name for name in METHODS if cfg_dict["methods"].get(name, False)
    ]
    tasks = [
        (seed, method)
        for seed in cfg_dict["seeds"]
        for method in methods
    ]
    log.info(
        f"{len(tasks)} tasks over {len(methods)} methods, "
        f"{len(y)} samples x {len(bundle['feature_names'])} features"
    )

    # parallelize across tasks
    n_jobs = cfg_dict.get("parallel", {}).get("n_jobs", -1)
    with parallel_backend("loky", inner_max_num_threads=1):
        Parallel(n_jobs=n_jobs, batch_size=1, verbose=0)(
            delayed(_run_task)(
                cfg_dict, bundle, X_df, y, seed, method, results_dir
            )
            for seed, method in tasks
        )


if __name__ == "__main__":
    main()
