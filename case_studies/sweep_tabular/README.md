
## Synthetic Data Simulation

### Design



### Workflow

To ensure all necessary packages are available, you can use the
`ni_case_studies` environment,

```
conda create -f environment.yaml # if not already installed
conda activate ni_case_studies
```

The synthetic data and explanation hyperparameters are defined in `config.yaml`.
To generate these example data and the associated model explanations, run,

```
python generate.py
python sweep.py
```

The synthetic data are saved into separate CSVs in a `data` subdirectory of this
case study directory. For example, the first few rows of `linear_additive_50_regression.csv` look like,

```
x1,x2,x3,x4,noise_1,noise_2,noise_3,noise_4,noise_5,noise_6,y
1.6523808947562233,-1.0470781521721315,0.29296724733092233,-2.2104541916888585,0.353187643516266,1.3627899372449055,-1.4854824410667322,-0.3928019819052394,-0.557238341745487,-1.015890997694673,-4.995103563493986
-0.0958642111469407,-1.204958835321448,-0.5902583650233595,-1.1195487729200495,0.719636538626733,0.6043257776819094,0.5701461471083106,-1.027265809618265,1.078535227290113,-0.34898788895523347,-11.249742846237266
1.3307191981411821,-0.9569030409209924,-0.9024896952647593,0.23646669405323661,-3.3065107807835687,0.01830653561961367,-0.6889060379356782,-2.0523294112041737,-0.48028330868734825,1.5303530933069227,-0.6673742894512019
```

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

Rerunning either the generation or explanation scripts will first check whether
the outputs are present and will only rerun those that are not present.