## TCGA BRCA Multi-Omics

### Design

This case study applies the same explanation methods as
`case_studies/sweep_tabular` to a real classifier, where no ground truth null
importance is available. We reproduce the `RF_concat` pipeline of [Novoloaca et
al. (2024)](https://doi.org/10.1093/bib/bbae331): mRNA, miRNA, and RPPA protein
measurements from TCGA BRCA are intersected on their common primary tumors,
each block is residualized against age, gender, and race, and a random forest
predicts the histological subtype (infiltrating ductal vs. lobular carcinoma).
Sample sizes are small relative to the number of measurements — 461 tumors, of
which 99 are lobular.

Each omic block is screened to its top-$K$ most variable features before the
model is fit, with $K$ set per block in `config.yaml`,

```
screening:
  mrna: 150
  mirna: 100
  protein: 50
```

Screening happens in `data.py`, before training, so that the fitted model and
the explained feature set are identical — `permutation`, `pdp_variance`, and
`integrated_gradients` all evaluate the model at perturbed designs and would be
undefined otherwise. The total dimension $d$ also sets the cost of the
risk-based methods: `minshap` and `sage` fit `risk.n_orderings` $\times\ d$
models each, and `gcm` retrains two forests per feature, so raising the
screening thresholds is the main way to make a run expensive.

Unlike the synthetic sweep, the data and the model here are fixed. The `seeds`
in `config.yaml` re-run every method with fresh internal randomness — the
permutation shuffles, the sampled coalitions in `minshap`/`sage`, the knockoff
draws — so the seed axis measures each method's own sampling variability rather
than variability across datasets. The model is trained once, under `seeds[0]`.

The synthetic sweep substitutes a fitted CART for `mdi` and `treeshap`, because
its "model" is the known data generating function. That substitution does not
happen here: the pickled random forest is the model, and every method explains
it.

### Workflow

To ensure all necessary packages are available, you can use the
`ni_case_studies` environment,

```
conda create -f environment.yaml # if not already installed
conda activate ni_case_studies
```

The raw matrices are downloaded from the GDC with `TCGAbiolinks` into
`data/raw`, and are skipped if already present. Everything else reads
`config.yaml`. From this directory,

```
Rscript download_tcga.R
python ../src/attribute_classifier.py .
python sweep.py
python vignettes.py
python ../src/attribute_explainer.py . --quick
Rscript visualize_importance.R
Rscript visualize.R
```

`attribute_classifier.py` preprocesses, screens, and trains, caching the design
matrix into `data/X.parquet` and `data/y.parquet` and the tuned forest into
`results/final_model.pkl`. It reuses the parquet files if they exist, so change
`screening` only after deleting them.

`sweep.py` skips any `(seed, method)` whose CSV is already present, so delete
`results/` if you want to replace results. Individual methods can be switched
off in the `methods` block, and any hyperparameter can be overridden on the
command line, e.g.

```
python sweep.py 'seeds=[895404]' risk.n_orderings=5 parallel.n_jobs=2
```

### Outputs

Global importances go into the `results` subdirectory, one CSV per method and
seed, named `{dataset}_{n}_{response_type}_{seed}_{method}.csv` to match the
convention in `case_studies/sweep_tabular`. Each file has the form,

```
feature,importance
mrna__SCGB2A2,0.0002356388109131567
mrna__SCGB1D2,0.0007643352311559636
mirna__hsa-mir-190b,0.0
```

Feature names carry an `mrna__`, `mirna__`, or `protein__` prefix recording
which block they came from. Some methods give local importances, and we have
aggregated them by their mean absolute value across samples.

`vignettes.py` writes `vignette_features.csv` (the response and the raw values
of the features whose importance profiles disagree most across methods) and
`vignette_pdp.csv` (`feature,grid_value,pdp_value`), which back the per-feature
panels. `attribute_explainer.py` writes the local attributions used by
`visualize.R`: `shap_attributions.csv`, `minshap_attributions.csv`, and
`patient_meta.csv`.

Hydra records the resolved configuration and log of each run under
`outputs/{date}/{time}/`.

### Figure summaries

`visualize_importance.R` compares methods against each other, averaging each
method's profile over seeds.

- **`fig_method_pca.pdf`:** PCA of the methods, treating each method as a point
  in feature space. Methods targeting the same notion of null importance should
  land near each other.
- **`fig_method_corr.pdf`:** Spearman rank correlation between methods, with
  the methods ordered by hierarchical clustering on $1 - \rho$.
- **`fig_vignettes.pdf`:** For the features where the methods disagree most, the
  importance each method assigns beside the fitted partial dependence curve.

`visualize.R` compares SHAP with marginal minSHAP at the level of individual
patients.

- **`fig_credit_splitting.pdf`:** Mean $|$SHAP$|$ against mean $|$minSHAP$|$ per
  feature, colored by omic. Points below the diagonal are features whose credit
  SHAP splits among correlated copies.
- **`fig_patient_panels.pdf`:** Top attributions under each method for two
  confidently and two ambiguously classified lobular tumors.
- **`fig_subgroups.pdf`:** Heatmap of the highest-variance SHAP attributions,
  with patients and features clustered, beside each patient's predicted
  probability.
