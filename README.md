# axiomatic_interpretability

This repository contains a small compositional attribution library plus a
simulation workflow for testing attribution methods on generated data.


## Simulation data

This repository includes synthetic generators for the Section 8 simulation
experiments:

- `linear_additive`: Gaussian features with
  `y = 4*X0 + 4*X1 + 3*X2 + 2*X3 + epsilon` by default. Features `0..3`
  are relevant additive features; all later columns are pure noise.
- `xor`: binary XOR signal with `y_mean = 1{X0 != X1}`. Features `0` and
  `1` are the only relevant interaction features; all later columns are
  Bernoulli noise. XOR is generated as classification data only.
- `product_interaction`: Gaussian features with
  `y = gamma * X0 * X1 + epsilon` by default. Features `0` and `1` are the
  relevant interaction features; optional main effects can be enabled.

The default main sweep varies:

- `n in [200, 500, 1000, 5000]`
- `p in [10, 20, 50, 100]`

Generate a small smoke-test sweep:

```bash
python scripts/simulation/generate_data.py --minimal
```

Generate the full sweep:

```bash
python scripts/simulation/generate_data.py
```

Outputs are written under
`data/simulations/{dataset_type}/`, with `data/simulations/sweep_config.yaml`
recording the seed, grid, and loop order. Each dataset is saved as a CSV file
with a same-stem `.metadata.json` sidecar that records the simulation contract.
Existing output files are skipped by default; pass `--overwrite` to regenerate
them in place.

## Attribution Benchmark

The benchmark loads generated simulation datasets, rebuilds the oracle
data-generating function from each metadata sidecar, applies configured
`axiom_interp` methods, and writes attribution outputs.

Run attribution on the minimal generated data:

```bash
python scripts/simulation/run_attribution.py --config configs/simulation_attribution/minimal.yaml --overwrite
python scripts/simulation/summarize_attribution.py
```

Run the full workflow:

```bash
python scripts/simulation/generate_data.py --overwrite
python scripts/simulation/run_attribution.py --config configs/simulation_attribution/full.yaml --overwrite
python scripts/simulation/summarize_attribution.py
```

Result files under `results/simulation_attribution/`:

- `raw_scores/`: one feature-score row per dataset, method, input row, and feature.
- `metrics/`: one metric row per dataset, method, and input row.
- `summaries/`: aggregate CSVs produced from metric files.

Inspect the results in:

```bash
notebooks/simulation_attribution_results.ipynb
```
