# MNIST ResNet Case Study

## Design

This case study uses an established pretrained ResNet classifier for MNIST as the fixed prediction model for later local null-importance experiments. The download stage does not train a model. It pulls the pinned Hugging Face checkpoint configured in `config.yaml`, downloads the MNIST test set, and saves the artifacts needed by later prediction, sample-selection, and importance scripts.

The current checkpoint source is [`fxmarty/resnet-tiny-mnist`](https://huggingface.co/fxmarty/resnet-tiny-mnist), pinned by revision in `config.yaml`. The model source reports an evaluation accuracy of about 0.985 on MNIST.

## Workflow

To ensure all necessary packages are available, create and activate the case-study environment,

```bash
conda env create -f environment.yaml
conda activate ni_mnist_resnet
```

To download the pretrained model and MNIST test data, use,

```bash
cd case_studies/mnist_resnet
python download.py
```

The main configuration is `config.yaml`. It controls the random seed, local data and results paths, Hugging Face cache path, model repo, pinned model revision, and download overwrite behavior. Hydra overrides can be used for one-off runs, for example,

```bash
python download.py download.overwrite=true
```

