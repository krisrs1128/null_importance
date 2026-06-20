"""Load + preprocess Drosophila enhancer activity data.

The source dataset is downloaded from a zenodo link.  This loader mimics the
approach in tcga_brca/data.py but for a single data modality.
"""

from __future__ import annotations

import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr


def download_if_needed(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"Downloading enhancer data → {dest}")
    urllib.request.urlretrieve(url, dest)


def _read_with_pyreadr(path: Path, x_object: str, y_object: str) -> tuple[pd.DataFrame, pd.Series]:
    objs = pyreadr.read_r(str(path))
    if x_object not in objs or y_object not in objs:
        keys = ", ".join(objs.keys())
        raise KeyError(f"Could not find '{x_object}'/'{y_object}' in RData keys: {keys}")

    X = objs[x_object].copy()
    y_df = objs[y_object]
    y = pd.Series(np.asarray(y_df).reshape(-1), index=X.index, name="y")
    return X, y


def _read_with_rscript(path: Path, x_object: str, y_object: str) -> tuple[pd.DataFrame, pd.Series]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        x_csv = tmp / "X.csv"
        y_csv = tmp / "y.csv"

        path_r = str(path).replace("'", "\\'")
        x_csv_r = str(x_csv).replace("'", "\\'")
        y_csv_r = str(y_csv).replace("'", "\\'")
        r_code = f"""
        load('{path_r}')
        if (!exists('{x_object}') || !exists('{y_object}')) {{
          stop('Missing expected objects in RData')
        }}
        X <- get('{x_object}')
        Y <- get('{y_object}')
        write.csv(X, '{x_csv_r}', row.names = TRUE)
        write.csv(data.frame(y = as.integer(Y)), '{y_csv_r}', row.names = FALSE)
        """
        subprocess.run(["Rscript", "-e", r_code], check=True)

        X = pd.read_csv(x_csv, index_col=0)
        y = pd.read_csv(y_csv)["y"]
        y.index = X.index
        return X, y


def load_raw(path: Path, x_object: str, y_object: str) -> tuple[pd.DataFrame, pd.Series]:
    try:
        return _read_with_pyreadr(path, x_object, y_object)
    except Exception as e:
        print(f"pyreadr load failed ({type(e).__name__}: {e}); trying Rscript fallback.")
        return _read_with_rscript(path, x_object, y_object)


def preprocess(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    # Keep only finite numeric predictors; mean-impute remaining missing entries.
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop all-NaN or constant columns.
    keep = (X.notna().any(axis=0)) & (X.nunique(dropna=True) > 1)
    X = X.loc[:, keep]

    col_means = X.mean(axis=0, skipna=True)
    X = X.fillna(col_means)

    # Match tcga style: integer-coded binary outcome.
    y = pd.Series(np.asarray(y).reshape(-1), index=X.index, name="y")
    y = y.astype(float).round().astype(int)

    valid = y.isin([0, 1])
    X = X.loc[valid]
    y = y.loc[valid]

    return X, y


def load_data(config):
    """Return (X, y, y_labels, omic_slices).

    X: DataFrame (samples × features)
    y: Series of integer labels in {0,1}
    y_labels: mapping of codes to class names
    omic_slices: single slice for compatibility with tcga interface
    """
    ds = config["dataset"]
    base = Path(__file__).parent
    raw_path = base / "data" / "raw" / ds["raw_file"]

    download_if_needed(ds["url"], raw_path)
    X, y = load_raw(raw_path, ds["x_object"], ds["y_object"])
    X, y = preprocess(X, y)

    y_labels = config["outcome"]["classes"]
    omic_slices = {"regulatory": slice(0, X.shape[1])}
    return X, y, y_labels, omic_slices
