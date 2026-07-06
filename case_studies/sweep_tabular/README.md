
## Synthetic Data Simulation

### Design

This case study defines synthetic datasets with different types of ground truth
null importance (functional, statistical, and causal) and applies representative
explainability techniques.  For synthetic data, we have variants of the data
generation from the minSHAP paper. These include,

- Linear model:

$$\mu(x) = \beta \sum_{j \in \mathcal{S}} x_j$$

- XOR/parity function:
$$\mu(x) = -\gamma \prod_{j \in \mathcal{S}} \text{sign}(x_j), \quad x_j \sim U[-1,1]$$

- Product:
$$\mu(x) = \gamma \sum_k x_{2k-1} x_{2k}$$

- Dependent features:
$$\mu(x) = \gamma \sum_{j \in \mathcal{S}} z_j, \quad \text{with } x_{2j-1} = z_j,\ x_{2j} = z_j + \epsilon$$

- Confounding:
$$\mu(x) = \gamma \sum_{j \in \mathcal{S}} z_j, \quad \text{with } x_j = z_j + \epsilon \ \ (z_j \text{ unobserved})$$

The null features $j \notin S$ are simulated from a random normal. Each data
generation function takes a random seed to ensure reproducibility. The $\gamma$
and $\beta$ parameters are signal strengths that can be set through
configuration file, which can also be used to modify sample sizes, the
dimensionality, and the number of signal features. Each data function has an
option for either regression or classification responses,

$$
y_i \sim \mathcal{N}\left(\mu\left(x_i\right), \sigma_{y}^{2}\right)
$$

$$
y_i \sim \text{Bernoulli}(\text{logit}^{-1}\left(\mu\left(x_i\right)\right))
$$

For explanation, we consider marginal correlation (pearson for regression,
biserial for classification), permutation importance, integrated gradients,
knockoffs (from the `knockpy` package) KernelSHAP, minSHAP, and PDP (variance of
the fitted profile). We aren't using MDI, TreeSHAP, or LOCO because our
implementations assume a tree model and for this synthetic data experiment we
treat the simulated mean response as the prediction.

### Workflow

To ensure all necessary packages are available, you can use the
`ni_case_studies` environment,

```
conda create -f environment.yaml # if not already installed
conda activate ni_case_studies
```

The synthetic data and explanation hyperparameters are defined in `config.yaml`.
By default we consider sample sizes $n \in \{50, 500, 5000\}$. To generate these
example data and the associated model explanations, run,

```
python generate.py
python sweep.py
```

The synthetic data are saved into separate CSVs in a `data` subdirectory of this
case study directory, with names like `data/{dataset}_{n}.csv`.  For example,
the first few rows of `linear_additive_50_regression.csv` look like,

```
x1,x2,x3,x4,noise_1,noise_2,noise_3,noise_4,noise_5,noise_6,y
1.6523808947562233,-1.0470781521721315,0.29296724733092233,-2.2104541916888585,0.353187643516266,1.3627899372449055,-1.4854824410667322,-0.3928019819052394,-0.557238341745487,-1.015890997694673,-4.995103563493986
-0.0958642111469407,-1.204958835321448,-0.5902583650233595,-1.1195487729200495,0.719636538626733,0.6043257776819094,0.5701461471083106,-1.027265809618265,1.078535227290113,-0.34898788895523347,-11.249742846237266
1.3307191981411821,-0.9569030409209924,-0.9024896952647593,0.23646669405323661,-3.3065107807835687,0.01830653561961367,-0.6889060379356782,-2.0523294112041737,-0.48028330868734825,1.5303530933069227,-0.6673742894512019
```

To visualize these simulated datasets, you can run,

```
quarto preview sanity_checks.qmd
```

which will generate a notebook with summary visualizations, e.g.,

![](https://github.com/user-attachments/assets/d7d0e608-f7d9-4dfe-b3e9-26c38f0b47c2)


A metadata file giving the git commit number, seed, and configuration paths is
saved in `data/run_metadata.yaml`. The global explanation variable importances
are saved into the `results` subdirectory. These files have the form,

```
feature,importance
x1,0.9243996708843824
x2,0.9216080730692336
x3,0.9219090852764292
x4,0.9189861560658712
noise_1,0.0
noise_2,0.0
noise_3,0.0
```

Note that some methods give local importances, and we have aggregated using
strategies explained in `model.py`.  Rerunning either the generation or
explanation scripts will first check whether the outputs are present and will
only rerun those that are not present.