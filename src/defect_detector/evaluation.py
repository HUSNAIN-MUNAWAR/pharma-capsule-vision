"""Checkpoint evaluation on the held-out test split."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import CLASS_NAMES, ProductImageDataset, Sample, build_transform, class_label_for_path
from .exceptions import DatasetError
from .metrics import compute_metrics, error_rows, write_evaluation_artifacts
from .model import build_model
from .training import resolve_device

LOGGER = logging.getLogger(__name__)


def _test_loader(samples: list[Sample], image_size: int, batch_size: int) -> DataLoader:
    dataset = ProductImageDataset(samples, transform=build_transform(image_size, training=False))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate_checkpoint(
    data_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    batch_size: int = 16,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Evaluate the exact test paths captured during training."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    test_paths = [Path(path) for path in checkpoint.get("test_paths", [])]
    if not test_paths:
        raise ValueError("Checkpoint does not contain a held-out test split")
    samples = []
    data_root = data_dir.resolve()
    for path in test_paths:
        resolved = path if path.is_absolute() else (data_root / path)
        if not resolved.is_file():
            raise FileNotFoundError(f"Test image from checkpoint is missing: {resolved}")
        try:
            label = class_label_for_path(resolved, data_root)
        except (DatasetError, ValueError) as exc:
            raise ValueError(f"Cannot infer class from test path: {resolved}") from exc
        samples.append(Sample(path=resolved, label=label))

    device = resolve_device(device_name)
    model_name = str(checkpoint["model"])
    image_size = int(checkpoint.get("image_size", 224))
    threshold = float(checkpoint.get("threshold", 0.5))
    model = build_model(
        model_name, num_classes=len(CLASS_NAMES), pretrained=False, freeze_backbone=False
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loader = _test_loader(samples, image_size=image_size, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    paths: list[str] = []
    total_loss = 0.0
    with torch.no_grad():
        for images, batch_labels, batch_paths in loader:
            logits = model(images.to(device))
            total_loss += float(criterion(logits, batch_labels.to(device)).item()) * len(
                batch_labels
            )
            defective_probabilities = torch.softmax(logits, dim=1)[:, 1]
            batch_predictions = (defective_probabilities >= threshold).long()
            labels.extend(batch_labels.tolist())
            predictions.extend(batch_predictions.cpu().tolist())
            probabilities.extend(defective_probabilities.cpu().tolist())
            paths.extend(batch_paths)
    metrics = compute_metrics(labels, predictions, probabilities)
    metrics["test_loss"] = total_loss / max(len(samples), 1)
    metrics["model_version"] = checkpoint.get("model_version", checkpoint_path.stem)
    metrics["threshold"] = threshold
    errors = error_rows(paths, labels, predictions, probabilities)
    write_evaluation_artifacts(metrics, errors, output_dir)
    LOGGER.info(
        "test_f1_defective=%.4f test_recall_defective=%.4f errors=%d",
        metrics["f1_defective"],
        metrics["recall_defective"],
        sum(row["error_type"] in {"false_positive", "false_negative"} for row in errors),
    )
    return metrics
