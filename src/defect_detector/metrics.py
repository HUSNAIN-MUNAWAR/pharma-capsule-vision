"""Evaluation metrics and false-positive/false-negative analysis."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import CLASS_NAMES


def compute_metrics(
    labels: Sequence[int], predictions: Sequence[int], probabilities: Sequence[float]
) -> dict[str, Any]:
    """Compute binary metrics with explicit positive-class semantics (`defective`)."""
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    total = max(len(labels), 1)
    metrics: dict[str, Any] = {
        "accuracy": float(np.trace(matrix) / total),
        "precision_defective": float(precision[1]),
        "recall_defective": float(recall[1]),
        "f1_defective": float(f1[1]),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "support": {CLASS_NAMES[index]: int(support[index]) for index in range(2)},
        "confusion_matrix": matrix.tolist(),
        "positive_class": "defective",
        "mean_defective_probability": float(np.mean(probabilities)) if probabilities else 0.0,
    }
    return metrics


def error_rows(
    paths: Sequence[str],
    labels: Sequence[int],
    predictions: Sequence[int],
    probabilities: Sequence[float],
) -> list[dict[str, Any]]:
    """Return per-image records for false-positive and false-negative review."""
    rows: list[dict[str, Any]] = []
    for path, label, prediction, probability in zip(
        paths, labels, predictions, probabilities, strict=False
    ):
        if label == prediction:
            error_type = "true_positive" if label == 1 else "true_negative"
        else:
            error_type = "false_negative" if label == 1 else "false_positive"
        rows.append(
            {
                "path": path,
                "actual_class": CLASS_NAMES[label],
                "predicted_class": CLASS_NAMES[prediction],
                "defective_probability": round(float(probability), 6),
                "error_type": error_type,
            }
        )
    return rows


def write_evaluation_artifacts(
    metrics: dict[str, Any], errors: Sequence[dict[str, Any]], output_dir: Path
) -> None:
    """Write machine-readable metrics and human-reviewable plots/tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(errors[0]) if errors else ["error_type"])
        writer.writeheader()
        writer.writerows(errors)
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        matrix_writer = csv.writer(stream)
        matrix_writer.writerow(["actual\\predicted", *CLASS_NAMES])
        for name, row in zip(CLASS_NAMES, metrics["confusion_matrix"], strict=False):
            matrix_writer.writerow([name, *row])

    matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set_title("Confusion matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_xticks(range(2), CLASS_NAMES)
    axis.set_yticks(range(2), CLASS_NAMES)
    for row_index in range(2):
        for column_index in range(2):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
            )
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)
