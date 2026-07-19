"""Feature importance methods on the Sweep 1 synthetic datasets.

Explains each dataset's data-generating function (see SCORES in model.py),
treating its output as predictions, rather than training a secondary model to
use in the explanation.

Run from the repo root, after generate.py creates data/{dataset}_{n}_{response_type}_{seed}.csv:
    python case_studies/sweep_tabular/sweep.py
"""

import inspect
import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from datasets import DATASETS
from model import SCORES, FunctionClassifier, FunctionRegressor

# setup logging and repo-level imports
log = logging.getLogger(__name__)
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir.parent / "src"))
sys.path.insert(0, str(_script_dir.parents[1] / "src"))
from importance import METHODS


def _save_method(results_dir, name, series):
    pd.DataFrame({"feature": series.index, "importance": series.values}).to_csv(
        results_dir / f"{name}.csv", index=False
    )


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):

    # read configuration and setup output directories
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    data_dir = _script_dir / "data"
    results_dir = _script_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # one independent rng per seed; loop over datasets/response types/sample sizes within each
    for seed in cfg.seeds:
        rng = np.random.default_rng(seed)
        for name in DATASETS:
            base_cfg = {**cfg_dict["dimensions"], **cfg_dict["datasets"][name]}
            for rt in cfg.response_types:
                dataset_cfg = {**base_cfg, "response_type": rt}
                for n in cfg.sample_sizes:

                    # read current data of interest
                    df = pd.read_csv(data_dir / f"{name}_{n}_{rt}_{seed}.csv")
                    X_df = df.drop(columns=["y"])
                    y = df["y"].values
                    X = X_df.values
                    feature_names = list(X_df.columns)

                    # create the mimic model and read explanation hyperparameters
                    ModelCls = FunctionClassifier if rt == "classification" else FunctionRegressor
                    model = ModelCls(SCORES[name], dataset_cfg).fit(X_df)
                    ctx = {
                        "model": model, "X": X, "y": y,
                        "X_df": X_df,
                        "feature_names": feature_names,
                        "rng": rng, "seed": seed,
                        "response_type": rt,
                        "n_repeats": cfg.permutation.n_repeats,
                        "mcfg": cfg_dict["minshap"],
                        "kshap_cfg": cfg_dict["kernelshap"],
                        "cfg": cfg_dict,
                        "kcfg": cfg_dict["knockoffs"],
                        "grid_resolution": cfg.pdp.grid_resolution,
                        "ig_cfg": cfg_dict["integrated_gradients"],
                        "gcm_cfg": cfg_dict["gcm"],
                    }

                    # run and save the explanations
                    for method, fn in METHODS.items():
                        if not cfg.methods[method]:
                            continue
                        log.info(f"Computing {name}_{n}_{rt}_{seed}_{method}...")
                        sig = inspect.signature(fn)
                        result = fn(**{p: ctx[p] for p in sig.parameters})
                        _save_method(results_dir, f"{name}_{n}_{rt}_{seed}_{method}", result)


if __name__ == "__main__":
    main()
