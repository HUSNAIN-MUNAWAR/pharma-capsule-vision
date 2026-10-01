"""Reproducible training and checkpoint lifecycle."""

from __future__ import annotations

import csv
import json
import logging
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .data import (
    CLASS_NAMES,
    ProductImageDataset,
    Sample,
    build_transform,
    class_weights,
    split_samples,
    write_manifests,
)
from .metrics import compute_metrics
from .model import build_model

LOGGER = logging.getLogger(__name__)
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class TrainingConfig:
    """All values that affect a training run, saved into the checkpoint."""

    model: str = "resnet18"
    pretrained: bool = False
    freeze_backbone: bool = True
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.2
    scheduler_patience: int = 3
    minimum_learning_rate: float = 1e-6
    patience: int = 4
    seed: int = 42
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    threshold: float = 0.5
    num_workers: int = 0


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable comparisons."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(value: str = "auto") -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def _loader(
    samples: Sequence[Sample], config: TrainingConfig, training: bool
) -> DataLoader[tuple[Tensor, int, str]]:
    dataset = ProductImageDataset(
        samples, transform=build_transform(config.image_size, training=training)
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=training,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, int, str]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
    threshold: float = 0.5,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    paths: list[str] = []

    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for images, batch_labels, batch_paths in loader:
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(images)
            loss = criterion(logits, batch_labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            probabilities_tensor = torch.softmax(logits, dim=1)[:, 1]
            batch_predictions = (probabilities_tensor >= threshold).long()
            total_loss += float(loss.item()) * len(batch_labels)
            labels.extend(batch_labels.detach().cpu().tolist())
            predictions.extend(batch_predictions.detach().cpu().tolist())
            probabilities.extend(probabilities_tensor.detach().cpu().tolist())
            paths.extend(batch_paths)

    average_loss = total_loss / max(len(loader.dataset), 1)
    metrics = compute_metrics(labels, predictions, probabilities)
    errors = []
    for path, label, prediction, probability in zip(
        paths, labels, predictions, probabilities, strict=False
    ):
        if label != prediction:
            errors.append(
                {
                    "path": path,
                    "actual_class": CLASS_NAMES[label],
                    "predicted_class": CLASS_NAMES[prediction],
                    "defective_probability": float(probability),
                    "error_type": "false_negative" if label == 1 else "false_positive",
                }
            )
    return average_loss, metrics, errors


def train_model(
    samples: Sequence[Sample],
    output_dir: Path,
    config: TrainingConfig,
    device_name: str = "auto",
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Train, select by validation F1, and save an immutable model checkpoint."""
    started = time.perf_counter()
    set_seed(config.seed)
    device = resolve_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = split_samples(
        samples,
        seed=config.seed,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )
    data_root = data_dir.resolve() if data_dir is not None else Path(samples[0].path).parents[1]
    write_manifests(splits, data_root, output_dir / "manifests")

    model = build_model(
        config.model,
        num_classes=len(CLASS_NAMES),
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(splits["train"]).to(device))
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.minimum_learning_rate,
    )
    train_loader = _loader(splits["train"], config, training=True)
    validation_loader = _loader(splits["validation"], config, training=False)
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss, train_metrics, _ = _run_epoch(
            model, train_loader, criterion, device, optimizer, config.threshold
        )
        validation_loss, validation_metrics, validation_errors = _run_epoch(
            model, validation_loader, criterion, device, threshold=config.threshold
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_f1_defective": train_metrics["f1_defective"],
            "validation_f1_defective": validation_metrics["f1_defective"],
            "validation_error_count": len(validation_errors),
        }
        current_f1 = float(validation_metrics["f1_defective"])
        scheduler.step(current_f1)
        row["learning_rate"] = optimizer.param_groups[0]["lr"]
        history.append(row)
        LOGGER.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_f1=%.4f",
            epoch,
            train_loss,
            validation_loss,
            validation_metrics["f1_defective"],
        )
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model": config.model,
                    "class_names": list(CLASS_NAMES),
                    "image_size": config.image_size,
                    "threshold": config.threshold,
                    "model_version": f"{config.model}-seed{config.seed}-epoch{epoch}",
                    "config": asdict(config),
                    "train_paths": [
                        sample.path.resolve().relative_to(data_root).as_posix()
                        for sample in splits["train"]
                    ],
                    "validation_paths": [
                        sample.path.resolve().relative_to(data_root).as_posix()
                        for sample in splits["validation"]
                    ],
                    "test_paths": [
                        sample.path.resolve().relative_to(data_root).as_posix()
                        for sample in splits["test"]
                    ],
                },
                output_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                LOGGER.info(
                    "Early stopping after %d epochs without validation improvement", config.patience
                )
                break

    training_seconds = time.perf_counter() - started
    summary = {
        "best_epoch": best_epoch,
        "best_validation_f1_defective": best_f1,
        "epochs_completed": len(history),
        "max_epochs": config.epochs,
        "early_stopped": len(history) < config.epochs,
        "training_seconds": round(training_seconds, 3),
        "history": history,
    }
    (output_dir / "training_history.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(history[0]) if history else ["epoch"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "device": str(device),
                "data_dir": str(data_root),
                "seed": config.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_training_curves(history, output_dir / "training_curves.png")
    LOGGER.info("Saved best checkpoint to %s", output_dir / "best.pt")
    return summary


def _write_training_curves(history: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Save loss and validation-F1 plots as evidence for each training run."""
    if not history:
        return
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in history], label="train")
    axes[0].plot(epochs, [float(row["validation_loss"]) for row in history], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(
        epochs,
        [float(row["train_f1_defective"]) for row in history],
        label="train",
    )
    axes[1].plot(
        epochs,
        [float(row["validation_f1_defective"]) for row in history],
        label="validation",
    )
    axes[1].set_title("Defective F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
