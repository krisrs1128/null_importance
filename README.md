# axiomatic_interpretability

## Sweep 1: Synthetic Null-Importance Examples

We consider five DGPs in our synthetic data case study.  Each generates binary
responses Y ~ Bernoulli(σ(logit)) with signal/null dimensions as
hyperparameters.

**DGPs**

| DGP | Equation | Null Types |
|-----|----------|------------|
| `linear_additive` | logit(Y) = β·∑ x_j (j ∉ Z) | Z = null indices |
| `xor` | logit(Y) = -γ·∏ x_j, x_j ∈ {-1,+1} | Marginal null; functional non-null |
| `product_interaction` | logit(Y) = γ·∑ x_{2k-1}·x_{2k} | Marginal null (pairs functional nonnull) |
| `dependent_features` | logit(Y) = γ·∑ z_j, x_{2j-1}=z_j, x_{2j}=z_j+ε | Conditional null (given anchor) |
| `confounding` | logit(Y) = γ·∑ z_j, x_j = z_j + ε (z unobserved) | Causal null |

Null-pad columns: iid N(0,1), independent of Y.

**Generate** all samples [n ∈ {50, 500, 5000}]:

```bash
python case_studies/sweep_tabular/generate.py
```

Outputs: `case_studies/sweep_tabular/data/{dataset}_{n}.csv`.
Config: `case_studies/sweep_tabular/config.yaml` (seed, sample sizes, DGP parameters).
