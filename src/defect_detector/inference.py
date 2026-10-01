"""Safe, read-only model loading and image prediction."""

from __future__ import annotations

import io
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from .data import CLASS_NAMES, build_transform
from .exceptions import InvalidImageError, ModelArtifactError
from .model import build_model
from .training import resolve_device

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    """Public prediction contract."""

    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str


class Predictor:
    """Loads a checkpoint once and performs deterministic CPU/GPU inference."""

    def __init__(self, checkpoint_path: Path, device_name: str = "auto") -> None:
        if not checkpoint_path.is_file():
            raise ModelArtifactError(f"Checkpoint does not exist: {checkpoint_path}")
        self.checkpoint_path = checkpoint_path.resolve()
        self.device = resolve_device(device_name)
        try:
            self.checkpoint: dict[str, Any] = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=False
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ModelArtifactError(f"Unable to load checkpoint: {checkpoint_path}") from exc
        class_names = tuple(self.checkpoint.get("class_names", []))
        if class_names != CLASS_NAMES:
            raise ModelArtifactError(f"Checkpoint classes must be {CLASS_NAMES}, got {class_names}")
        model_name = str(self.checkpoint.get("model", ""))
        image_size = int(self.checkpoint.get("image_size", 224))
        if not model_name or image_size <= 0:
            raise ModelArtifactError("Checkpoint is missing model metadata")
        self.model = build_model(
            model_name, num_classes=len(CLASS_NAMES), pretrained=False, freeze_backbone=False
        ).to(self.device)
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.eval()
        self.transform = build_transform(image_size, training=False)
        self.threshold = float(self.checkpoint.get("threshold", 0.5))
        self.model_version = str(self.checkpoint.get("model_version", self.checkpoint_path.stem))
        LOGGER.info(
            "Loaded model=%s version=%s device=%s", model_name, self.model_version, self.device
        )

    def predict_bytes(self, content: bytes) -> Prediction:
        """Decode one image and return a JSON-serializable prediction."""
        try:
            with Image.open(io.BytesIO(content)) as image:
                image.load()
                image_rgb = image.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise InvalidImageError("Uploaded content is not a valid image") from exc
        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities_tensor = torch.softmax(self.model(tensor), dim=1)[0]
        defective_probability = float(probabilities_tensor[1].item())
        predicted_index = 1 if defective_probability >= self.threshold else 0
        probabilities = {
            CLASS_NAMES[index]: round(float(probabilities_tensor[index].item()), 6)
            for index in range(len(CLASS_NAMES))
        }
        return Prediction(
            predicted_class=CLASS_NAMES[predicted_index],
            confidence=round(probabilities[CLASS_NAMES[predicted_index]], 6),
            probabilities=probabilities,
            model_version=self.model_version,
        )

    def predict_path(self, image_path: Path) -> Prediction:
        return self.predict_bytes(image_path.read_bytes())


def prediction_as_dict(prediction: Prediction) -> dict[str, Any]:
    """Convert the dataclass to the API/CLI response shape."""
    return asdict(prediction)
