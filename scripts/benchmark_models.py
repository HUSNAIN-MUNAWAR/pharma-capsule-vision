"""Train and compare multiple model families on the same real dataset split."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from defect_detector.data import discover_samples
from defect_detector.evaluation import evaluate_checkpoint
from defect_detector.logging_config import configure_logging
from defect_detector.model import build_model
from defect_detector.training import TrainingConfig, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark defect-classification models")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model-benchmark"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["small_cnn", "resnet18", "mobilenet_v3_small", "efficientnet_b0"],
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Train the pretrained backbone as well as the classifier head",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(args.output_dir / "benchmark.log")
    samples = discover_samples(args.data_dir)
    rows: list[dict[str, object]] = []
    for model_name in args.models:
        model_output = args.output_dir / model_name
        config = TrainingConfig(
            model=model_name,
            pretrained=model_name != "small_cnn",
            freeze_backbone=not args.fine_tune,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            patience=args.patience,
            seed=args.seed,
        )
        started = time.perf_counter()
        training_summary = train_model(
            samples,
            model_output,
            config,
            device_name=args.device,
            data_dir=args.data_dir,
        )
        training_seconds = time.perf_counter() - started
        metrics = evaluate_checkpoint(
            args.data_dir,
            model_output / "best.pt",
            model_output / "evaluation",
            batch_size=args.batch_size,
            device_name=args.device,
        )
        benchmark_model = build_model(model_name, pretrained=False, freeze_backbone=False)
        row = {
            "model": model_name,
            "pretrained": config.pretrained,
            "freeze_backbone": config.freeze_backbone,
            "parameters": sum(parameter.numel() for parameter in benchmark_model.parameters()),
            "training_seconds": round(training_seconds, 2),
            "best_epoch": training_summary["best_epoch"],
            "epochs_completed": training_summary["epochs_completed"],
            "early_stopped": training_summary["early_stopped"],
            "best_validation_f1_defective": training_summary["best_validation_f1_defective"],
            "accuracy": metrics["accuracy"],
            "precision_defective": metrics["precision_defective"],
            "recall_defective": metrics["recall_defective"],
            "f1_defective": metrics["f1_defective"],
            "f1_macro": metrics["f1_macro"],
            "checkpoint": str((model_output / "best.pt").resolve()),
        }
        rows.append(row)

    ranked = sorted(
        rows,
        key=lambda row: float(row["best_validation_f1_defective"]),
        reverse=True,
    )
    result = {
        "selection_metric": "best_validation_f1_defective",
        "test_metrics_are_report_only": True,
        "ranking": ranked,
        "all_runs": rows,
    }
    (args.output_dir / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output_dir / "benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ranked[0]) if ranked else ["model"])
        writer.writeheader()
        writer.writerows(ranked)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
