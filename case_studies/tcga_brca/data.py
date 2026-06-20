"""Preprocess TCGA Multi-Omics

This is designed to reproduce the TCGA random forest experiments from

https://doi.org/10.1093/bib/bbae331
https://github.com/bioaster/benchmark-integrative-methods

It's automatically called in pipeline.py, so no need to run this on its own.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder


def load_cached_tsv(cache_dir, dataset_id):
    path = cache_dir / (dataset_id.replace("/", "_") + ".tsv")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `Rscript download_tcga.R` first."
        )
    return pd.read_csv(path, sep="\t", index_col=0)


def top_k_variable(df, k):
    cols = df.var(axis=0).nlargest(min(k, df.shape[1])).index
    return df[cols]


def drop_nan_columns(df):
    """Remove columns where all values are NaN."""
    return df.dropna(axis=1, how='all')


def adjust_covariates(X_df, cov_df):
    """Regress covariates out of each feature; return residuals (DataFrame)."""
    cat_cols = cov_df.select_dtypes(exclude="number").columns.tolist()
    num_cols = cov_df.select_dtypes("number").columns.tolist()

    parts = []
    if num_cols:
        arr = cov_df[num_cols].values.astype(float)
        col_means = np.nanmean(arr, axis=0)
        arr = np.where(np.isnan(arr), col_means, arr)
        parts.append(arr)
    if cat_cols:
        cat_arr = cov_df[cat_cols].fillna("MISSING").astype(str)
        enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        parts.append(enc.fit_transform(cat_arr))

    C = np.hstack(parts) if parts else np.ones((len(X_df), 1))

    X_arr = X_df.values.astype(float)
    col_means = np.nanmean(X_arr, axis=0)
    X_arr = np.where(np.isnan(X_arr), col_means, X_arr)

    lr = LinearRegression(fit_intercept=True).fit(C, X_arr)
    residuals = X_arr - lr.predict(C)
    return pd.DataFrame(residuals, index=X_df.index, columns=X_df.columns)


def load_data(config):
    """Return (X, y, y_labels) after full preprocessing."""
    cache_dir = Path(__file__).parent / "data" / "raw"
    ds = config["datasets"]

    # Genomic data: rows = features, cols = samples → transpose
    mrna = load_cached_tsv(cache_dir, ds["mrna"]).T
    mirna = load_cached_tsv(cache_dir, ds["mirna"]).T
    protein = load_cached_tsv(cache_dir, ds["protein"]).T
    clinical = load_cached_tsv(cache_dir, ds["clinical"])

    # Some omic files ship with a trailing sample-type suffix (e.g. "-01"). We
    # remove it so that all the omics have common sample-ID formats
    for df in (mirna, protein):
        df.index = df.index.str.replace(r"-01$", "", regex=True)

    # Filter clinical to the two histological subtypes
    htype = config["outcome"]["field"]
    classes = config["outcome"]["classes"]
    clinical = clinical[clinical[htype].isin(classes)].copy()

    # Intersect samples present in all four matrices
    shared = sorted(
        set(mrna.index) & set(mirna.index) & set(protein.index) & set(clinical.index)
    )
    mrna = mrna.loc[shared]
    mirna = mirna.loc[shared]
    protein = protein.loc[shared]
    clinical = clinical.loc[shared]

    # Integer-coded outcome
    cat = pd.Categorical(clinical[htype], categories=classes)
    y = pd.Series(cat.codes.astype(int), index=shared, name="y")
    y_labels = list(cat.categories)

    # Covariates
    cov_cols = [c for c in config.get("covariates", []) if c in clinical.columns]
    cov_df = clinical[cov_cols] if cov_cols else pd.DataFrame(index=shared)

    # Retain top-K variable features for mRNA; miRNA and protein kept in full
    mrna = top_k_variable(mrna, config.get("top_features", 1000))

    # Drop columns that are entirely NaN
    mrna = drop_nan_columns(mrna)
    mirna = drop_nan_columns(mirna)
    protein = drop_nan_columns(protein)

    # Covariate adjustment
    if cov_cols:
        mrna = adjust_covariates(mrna, cov_df)
        mirna = adjust_covariates(mirna, cov_df)
        protein = adjust_covariates(protein, cov_df)

    # Prefix columns so concatenation is unambiguous
    mrna.columns = [f"mrna__{c}" for c in mrna.columns]
    mirna.columns = [f"mirna__{c}" for c in mirna.columns]
    protein.columns = [f"protein__{c}" for c in protein.columns]

    X = pd.concat([mrna, mirna, protein], axis=1)

    return X, y, y_labels