# MNIST ResNet Case Study

## Design

This case study uses an established pretrained ResNet classifier for MNIST as
the fixed prediction model for later local null-importance experiments. The
download stage does not train a model. It pulls the pinned Hugging Face
checkpoint configured in `config.yaml`, downloads the MNIST test set, and saves
the artifacts needed by later prediction, sample-selection, and importance
scripts.

The current checkpoint source is
[`fxmarty/resnet-tiny-mnist`](https://huggingface.co/fxmarty/resnet-tiny-mnist),
pinned by revision in `config.yaml`. The model source reports an evaluation
accuracy of about 0.985 on MNIST.

## Workflow

To ensure all necessary packages are available, create and activate the
case-study environment,

```bash
cd case_studies/mnist_resnet
conda env create -f environment.yaml
conda activate ni_case_studies
```

To download the pretrained model and MNIST test data, use,

```bash
python download.py
```

The main configuration is `config.yaml`. It controls the random seed, local data
and results paths, Hugging Face cache path, model repo, pinned model revision,
and download overwrite behavior. Hydra overrides can be used for one-off runs,
for example,

```bash
python download.py download.overwrite=true
```

To select examples for the local null-importance analysis, use,

```bash
python select_samples.py
```

This writes `results/sample_meta.csv`, with 100 MNIST test examples: for each
digit from 0-9, 5 correctly classified examples and 5 misclassified examples, selected
by the downloaded model's predicted probability. The first few rows have the form,

```csv
sample_index,true_label,predicted_label,predicted_probability,correct
7607,0,0,0.9998399019241333,True
2385,0,0,0.9998200535774231,True
7703,0,0,0.9998107552528381,True
7699,0,0,0.9998075366020203,True
7727,0,0,0.9997738003730774,True
```

To compute pixel-level SHAP and minSHAP importances for those examples, use,

```bash
python importance.py
```

The importance settings live in `importance.yaml`. The default
`n_orderings` and `n_background` values are intentionally small because each
MNIST image has 784 pixel features (with current config taking 30 minutes to finish).

The local explanation files are saved in `results/importance/`. The SHAP and
minSHAP files have one row per selected test example and one column per pixel
feature, `pixel_0` through `pixel_783`. For readability, the examples below
show the metadata columns, the first few pixel columns, and the last pixel
column.

`shap_attributions.csv` has the form,

```csv
sample_index,true_label,predicted_label,predicted_probability,correct,target_label,pixel_0,pixel_1,pixel_2,pixel_783
7607,0,0,0.9998399019241332,True,0,0.0038487184792757,-0.0089938039891421,0.0213100811699405,-0.0058836719428654
2385,0,0,0.9998200535774232,True,0,-0.0258322871290147,0.016603519860655,0.0647321720607578,-0.0252399119315668
7703,0,0,0.999810755252838,True,0,-0.0133678028243593,-0.0542421123245731,0.0023948723217472,-0.0097468154272064
```

`minshap_attributions.csv` has the same columns,

```csv
sample_index,true_label,predicted_label,predicted_probability,correct,target_label,pixel_0,pixel_1,pixel_2,pixel_783
7607,0,0,0.9998399019241332,True,0,-0.0096338709117844,-0.0721206665039062,-0.0090842805802822,-0.0162796499207615
2385,0,0,0.9998200535774232,True,0,-0.1522493362426757,-0.0047911708243191,-0.0116430539637804,-0.1110722194425761
7703,0,0,0.999810755252838,True,0,-0.1862225532531738,-0.2454226016998291,-0.0215538293123245,-0.047044270671904
```
