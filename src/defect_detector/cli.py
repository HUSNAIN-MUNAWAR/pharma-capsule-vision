"""Command-line entry points for inspection, splitting, training, and evaluation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .data import discover_samples, inspect_dataset, json_safe_stats, split_samples, write_manifests
from .evaluation import evaluate_checkpoint
from .logging_config import configure_logging
from .training import TrainingConfig, train_model

LOGGER = logging.getLogger(__name__)


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visual defect detection workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    _common_parser(inspect_parser)
    inspect_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/inspection"))

    split_parser = subparsers.add_parser("split")
    _common_parser(split_parser)
    split_parser.add_argument("--output-dir", type=Path, default=Path("data/processed/manifests"))
    split_parser.add_argument("--seed", type=int, default=42)

    train_parser = subparsers.add_parser("train")
    _common_parser(train_parser)
    train_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/run"))
    train_parser.add_argument(
        "--model",
        choices=[
            "small_cnn",
            "resnet18",
            "mobilenet_v3_small",
            "efficientnet_b0",
            "convnext_tiny",
        ],
        default="resnet18",
    )
    train_parser.add_argument("--pretrained", action="store_true")
    train_parser.add_argument("--unfreeze-backbone", action="store_true")
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--patience", type=int, default=4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default="auto")

    evaluate_parser = subparsers.add_parser("evaluate")
    _common_parser(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    evaluate_parser.add_argument("--batch-size", type=int, default=16)
    evaluate_parser.add_argument("--device", default="auto")
    return parser


def inspect_command(args: argparse.Namespace) -> None:
    stats = inspect_dataset(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dataset_report.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe_stats(args.data_dir), indent=2))


def split_command(args: argparse.Namespace) -> None:
    samples = discover_samples(args.data_dir)
    splits = split_samples(samples, seed=args.seed)
    write_manifests(splits, args.data_dir, args.output_dir)
    print(json.dumps({key: len(value) for key, value in splits.items()}, indent=2))


def train_command(args: argparse.Namespace | None = None) -> None:
    configure_logging()
    if args is None:
        args = _build_parser().parse_args()
    if args.command != "train":
        raise ValueError("train_command must be called with the train subcommand")
    samples = discover_samples(args.data_dir)
    config = TrainingConfig(
        model=args.model,
        pretrained=args.pretrained,
        freeze_backbone=not args.unfreeze_backbone,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
    )
    summary = train_model(
        samples, args.output_dir, config, device_name=args.device, data_dir=args.data_dir
    )
    print(json.dumps(summary, indent=2))


def evaluate_command(args: argparse.Namespace) -> None:
    metrics = evaluate_checkpoint(
        args.data_dir,
        args.checkpoint,
        args.output_dir,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(json.dumps(metrics, indent=2))


def main() -> None:
    configure_logging()
    args = _build_parser().parse_args()
    if args.command == "inspect":
        inspect_command(args)
    elif args.command == "split":
        split_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)


if __name__ == "__main__":
    main()
