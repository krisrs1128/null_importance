"""Feature importance methods on the TCGA BRCA dataset."""

import inspect
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.inspection import partial_dependence as pd_func

log = logging.getLogger(__name__)

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir.parent / "src"))
sys.path.insert(0, str(_script_dir.parents[1] / "src"))

from results import load_model
from importance import METHODS





def _resolve_path(relative, base=_script_dir):
    return (base / relative).resolve()


def _load_method(results_dir, name):
    path = results_dir / f"{name}.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df.set_index("feature")["importance"]
    return None


def _save_method(results_dir, name, series):
    pd.DataFrame({"feature": series.index, "importance": series.values}).to_csv(
        results_dir / f"{name}.csv", index=False
    )


def _save_vignette_data(results_dir, X_df, y, model, feature_names, matrix, cfg):
    mat = matrix.drop(columns=["feature"]).values
    z = (mat - mat.mean(axis=0)) / (mat.std(axis=0) + 1e-12)
    row_var = z.var(axis=1)
    n_feat = cfg.vignettes.n_features
    top_idx = np.argsort(row_var)[::-1][:n_feat]
    top_features = [matrix["feature"].iloc[i] for i in top_idx]

    feat_df = X_df[top_features].copy()
    feat_df.insert(0, "y", y)
    feat_df.to_csv(results_dir / "vignette_features.csv", index=False)

    pdp_rows = []
    X_arr = X_df.values
    for feat in top_features:
        j = list(feature_names).index(feat)
        result = pd_func(
            model, X_arr, features=[j],
            grid_resolution=cfg.pdp.grid_resolution, kind="average",
        )
        grid = result["grid_values"][0]
        avg = result["average"][0]
        for g, a in zip(grid, avg):
            pdp_rows.append({"feature": feat, "grid_value": g, "pdp_value": a})
    pd.DataFrame(pdp_rows).to_csv(results_dir / "vignette_pdp.csv", index=False)
    log.info(f"Saved vignette data for {len(top_features)} features")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    X_df = pd.read_parquet(_resolve_path(cfg.data.parquet_dir) / "X.parquet")
    y = pd.read_parquet(_resolve_path(cfg.data.parquet_dir) / "y.parquet").iloc[:, 0].values
    bundle = load_model(_resolve_path(cfg.data.model_dir))
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    X = X_df[feature_names].values

    rng = np.random.default_rng(cfg.seed)
    results_dir = _script_dir / "results"
    results_dir.mkdir(exist_ok=True)

    log.info(f"Starting feature importance computation with seed {cfg.seed}")
    log.info(f"Results will be saved to {results_dir}")

    ctx = {
        "model": model, "X": X, "y": y,
        "X_df": X_df[feature_names],
        "feature_names": feature_names,
        "rng": rng, "seed": cfg.seed,
        "n_repeats": cfg.permutation.n_repeats,
        "mcfg": cfg_dict["minshap"],
        "kshap_cfg": cfg_dict["kernelshap"],
        "cfg": cfg_dict,
        "kcfg": cfg_dict["knockoffs"],
        "grid_resolution": cfg.pdp.grid_resolution,
    }

    computed = {}
    for name, fn in METHODS.items():
        if not cfg.methods[name]:
            log.info(f"Skipping {name}")
            continue
        cached = _load_method(results_dir, name)
        if cached is not None:
            log.info(f"Loaded cached {name}")
            computed[name] = cached
        else:
            log.info(f"Computing {name}...")
            sig = inspect.signature(fn)
            result = fn(**{p: ctx[p] for p in sig.parameters})
            _save_method(results_dir, name, result)
            computed[name] = result
            log.info(f"  done")

    matrix = pd.DataFrame({"feature": feature_names})
    for name, series in computed.items():
        matrix[name] = series.reindex(feature_names).values
    matrix.to_csv(results_dir / "importance_matrix.csv", index=False)
    log.info(f"Saved importance_matrix.csv: {matrix.shape[0]} features x {len(computed)} methods")

    _save_vignette_data(results_dir, X_df[feature_names], y, model, feature_names, matrix, cfg)


if __name__ == "__main__":
    main()
