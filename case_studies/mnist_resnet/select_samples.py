"""Select MNIST predictions for local null-importance analysis.

Run from this case-study directory, after download.py:
    python select_samples.py
"""

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from transformers import AutoConfig, AutoModelForImageClassification

log = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent

N_CLASSES = 10
N_PER_GROUP = 5
BATCH_SIZE = 256
SAMPLE_COLUMNS = [
    "sample_index",
    "true_label",
    "predicted_label",
    "predicted_probability",
    "correct",
]


def case_path(path: str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def load_model(cfg: DictConfig) -> AutoModelForImageClassification:
    """Reconstruct the downloaded model without training or network access."""
    model_dir = case_path(cfg.paths.model_source_dir)
    model_path = case_path(cfg.paths.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}. Run download.py first.")

    model = AutoModelForImageClassification.from_config(
        AutoConfig.from_pretrained(model_dir, local_files_only=True)
    )
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def mnist_loader(cfg: DictConfig) -> DataLoader:
    """Load MNIST test data with the checkpoint's own preprocessing stats."""
    preprocessor_path = case_path(cfg.paths.model_source_dir) / "preprocessor_config.json"
    with preprocessor_path.open() as handle:
        preprocessor = json.load(handle)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                preprocessor.get("image_mean", [0.45]),
                preprocessor.get("image_std", [0.22]),
            ),
        ]
    )
    # download=False keeps this stage tied to the explicit download.py output.
    dataset = MNIST(
        root=str(case_path(cfg.paths.data_dir)),
        train=cfg.data.train,
        download=False,
        transform=transform,
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def predict_test_set(cfg: DictConfig) -> pd.DataFrame:
    """Score every test example; sample_index is canonical MNIST test order."""
    model = load_model(cfg)
    rows = []
    offset = 0

    with torch.no_grad():
        for images, labels in mnist_loader(cfg):
            probs = torch.softmax(model(pixel_values=images).logits, dim=1)
            pred_prob, pred_label = probs.max(dim=1)
            n_batch = len(labels)
            rows.append(
                pd.DataFrame(
                    {
                        "sample_index": np.arange(offset, offset + n_batch),
                        "true_label": labels.numpy().astype(int),
                        "predicted_label": pred_label.numpy().astype(int),
                        # Confidence of the argmax class, not the true label.
                        "predicted_probability": pred_prob.numpy().astype(float),
                    }
                )
            )
            offset += n_batch

    predictions = pd.concat(rows, ignore_index=True)
    predictions["correct"] = predictions["true_label"] == predictions["predicted_label"]
    return predictions


def select_samples(predictions: pd.DataFrame) -> pd.DataFrame:
    """Pick 5 correct and 5 misclassified examples per true digit."""
    selected = []
    for label in range(N_CLASSES):
        for correct in (True, False):
            candidates = predictions[
                (predictions["true_label"] == label) & (predictions["correct"] == correct)
            ]
            if len(candidates) < N_PER_GROUP:
                raise ValueError(
                    f"label={label}, correct={correct}: "
                    f"need {N_PER_GROUP}, found {len(candidates)}"
                )

            # Stable tie-break by sample_index makes reruns identical.
            selected.append(
                candidates.sort_values(
                    ["predicted_probability", "sample_index"],
                    ascending=[False, True],
                ).head(N_PER_GROUP)
            )

    return pd.concat(selected, ignore_index=True)[SAMPLE_COLUMNS]


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    sample_meta = select_samples(predict_test_set(cfg))
    output_path = case_path(cfg.paths.results_dir) / "sample_meta.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_meta.to_csv(output_path, index=False)

    counts = sample_meta.groupby(["true_label", "correct"]).size().unstack(fill_value=0)
    log.info("Saved %s selected samples to %s", len(sample_meta), output_path)
    log.info("Selection counts by true label and correct flag:\n%s", counts)


if __name__ == "__main__":
    main()
