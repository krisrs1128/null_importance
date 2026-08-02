"""Use the pretrained MNIST ResNet as f(flat pixel batch) -> probabilities."""

import json
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import MNIST
from transformers import AutoConfig, AutoModelForImageClassification

CASE_DIR = Path(__file__).resolve().parent
IMAGE_SHAPE = (1, 28, 28)
N_PIXELS = 28 * 28


def case_path(path: str | Path, base_dir: Path = CASE_DIR) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (base_dir / path).resolve()


def flatten_images(images) -> np.ndarray:
    """Convert MNIST image arrays to row-major flat pixels with shape (n, 784)."""
    arr = np.asarray(images, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim == 2 and arr.shape == IMAGE_SHAPE[1:]:
        arr = arr[None, :, :]

    flat = arr.reshape(arr.shape[0], -1)
    if flat.shape[1] != N_PIXELS:
        raise ValueError(f"Expected {N_PIXELS} pixels, got shape {arr.shape}")
    return flat


def unflatten_pixels(flat_pixels) -> np.ndarray:
    """Convert flat pixel rows back to 1 x 28 x 28 image tensors."""
    return flatten_images(flat_pixels).reshape(-1, *IMAGE_SHAPE)


def pixel_feature_names() -> list[str]:
    return [f"pixel_{j}" for j in range(N_PIXELS)]


def load_mnist_flat(data_dir: Path, train: bool) -> tuple[np.ndarray, np.ndarray]:
    """Load downloaded MNIST as raw pixel intensities in [0, 1]."""
    dataset = MNIST(root=str(data_dir), train=train, download=False)
    images = dataset.data.numpy().astype(np.float32) / 255.0
    labels = dataset.targets.numpy().astype(int)
    return flatten_images(images), labels


class MNISTResNetFlat:
    """Callable probability model for flattened MNIST pixels."""

    def __init__(self, model, image_mean, image_std, batch_size: int = 256):
        self.model = model.eval()
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(1, 1, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(1, 1, 1, 1)
        self.batch_size = batch_size

    @classmethod
    def from_downloaded(
        cls,
        model_source_dir: Path,
        model_path: Path,
        batch_size: int = 256,
    ) -> "MNISTResNetFlat":
        """Rebuild the downloaded checkpoint without training or network access."""
        with (model_source_dir / "preprocessor_config.json").open() as handle:
            preprocessor = json.load(handle)

        model = AutoModelForImageClassification.from_config(
            AutoConfig.from_pretrained(model_source_dir, local_files_only=True)
        )
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

        return cls(
            model=model,
            image_mean=preprocessor.get("image_mean", [0.45]),
            image_std=preprocessor.get("image_std", [0.22]),
            batch_size=batch_size,
        )

    def _preprocess(self, flat_batch) -> torch.Tensor:
        images = torch.tensor(unflatten_pixels(flat_batch), dtype=torch.float32)
        return (images - self.image_mean) / self.image_std

    def predict_proba(self, flat_batch) -> np.ndarray:
        """Return class probabilities for raw flat pixel vectors."""
        images = self._preprocess(flat_batch)

        probs = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = images[start : start + self.batch_size]
                logits = self.model(pixel_values=batch).logits
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(probs)

    def final_embeddings(self, flat_batch) -> np.ndarray:
        """Get resnet pre-classification embeddings"""
        images = self._preprocess(flat_batch)

        embeddings = []
        with torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = images[start : start + self.batch_size]
                pooled = self.model.resnet(pixel_values=batch).pooler_output
                embeddings.append(pooled.flatten(start_dim=1).cpu().numpy())
        return np.vstack(embeddings)

    def class_probability(self, label: int):
        """Scalar f(batch) -> P(label) for axiom_interp explainers."""
        label = int(label)

        def f(batch) -> np.ndarray:
            return self.predict_proba(batch)[:, label]

        return f
