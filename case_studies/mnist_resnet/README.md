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

To use the case-study environment,

```bash
cd case_studies/mnist_resnet
conda env create -f environment.yaml
conda activate ni_case_studies
```

The main entry point is `run.py`. To run the full workflow, use,

```bash
python run.py
```

`run.py` uses `importance.yaml` for the seed, paths, model batch size, and
importance settings. It also reuses `config.yaml` for the pinned model source.

The run does the following:

- Downloads or reuses the pretrained ResNet checkpoint and MNIST data.
- Scores the MNIST test set and writes the selected-sample metadata.
- Samples flattened MNIST training images as the background/reference set.
- Computes the enabled local importance methods and writes the results to
  `results/`.

To spend more compute on SHAP/minSHAP Monte Carlo orderings and use a larger
background set, use Hydra overrides,

```bash
python run.py explain.n_orderings=20 explain.n_background=50
```

The selection stage writes `results/sample_meta.csv`, with 100 MNIST test
examples: for each digit from 0-9, 5 correctly classified examples and 5
misclassified examples, selected by the downloaded model's predicted
probability. The first few rows have the form,

```csv
sample_index,true_label,predicted_label,predicted_probability,correct
7607,0,0,0.9998399019241333,True
2385,0,0,0.9998200535774231,True
7703,0,0,0.9998107552528381,True
7699,0,0,0.9998075366020203,True
7727,0,0,0.9997738003730774,True
```

The local explanation files are saved as `results/{method}_attributions.csv`.
Each attribution file has one row per selected test example and one column per
pixel feature, `pixel_0` through `pixel_783`; row order matches
`results/sample_meta.csv`.

The default `n_orderings` and `n_background` values are moderate because each
MNIST image has 784 pixel features. Increase `n_orderings` first when the
runtime budget allows, and keep `n_background >= local_ttest.n_neighbors` for
the 20-nearest-neighbor t-statistic.

For readability, the example `shap_attributions.csv` shows only the first few pixel columns and
the last pixel column.

```csv
pixel_0,pixel_1,pixel_2,pixel_783
-0.0449809268116951,0.019151636958122255,0.07923728227615356,-0.060081002116203305
-0.022464396059513093,-0.023283451795578003,0.033800172805786136,-0.008888739347457885
0.041246681660413745,-0.018755125999450683,0.03381760716438294,-0.00964224487543106
```


### Importance Methods

Each method produces one local score for each of the 784 pixels in a selected
MNIST test image.

- **SHAP:** Treats pixels as features. Missing pixels are replaced using
  background training images, and the score estimates each pixel's marginal
  contribution to the ResNet probability for the target class.

- **minSHAP:** Uses the same marginal-contribution tensor as SHAP, but changes
  the aggregation rule to the minSHAP rule.

- **Integrated gradients:** Uses a zero image as the baseline, follows the path
  from that zero image to the selected image, and attributes the target-class
  probability change to individual pixels.

- **Local t-statistic:** Finds the nearest background images in pixel space,
  splits them by whether the ResNet predicts the same class as the selected
  image, and computes a per-pixel t-statistic between those two local groups.

