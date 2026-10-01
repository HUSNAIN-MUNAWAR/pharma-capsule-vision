"""Dataset discovery, validation, splitting, and PyTorch input transforms."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from .exceptions import DatasetError, InvalidImageError

LOGGER = logging.getLogger(__name__)
CLASS_NAMES = ("normal", "defective")
CLASS_DIRECTORY_ALIASES = {
    "normal": ("normal", "Normal", "good", "Good"),
    "defective": ("defective", "Defective", "anomaly", "Anomaly", "defect", "Defect"),
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    """A validated image path and its binary class label."""

    path: Path
    label: int

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.label]


def _iter_images(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _find_class_directory(data_dir: Path, class_name: str) -> Path:
    aliases = {alias.casefold() for alias in CLASS_DIRECTORY_ALIASES[class_name]}
    matches = [
        entry for entry in data_dir.iterdir() if entry.is_dir() and entry.name.casefold() in aliases
    ]
    if not matches:
        alias_names = ", ".join(CLASS_DIRECTORY_ALIASES[class_name])
        raise DatasetError(
            f"Missing required {class_name} directory; accepted names: {alias_names}"
        )
    if len(matches) > 1:
        raise DatasetError(f"Ambiguous directories for {class_name}: {matches}")
    return matches[0]


def class_label_for_path(path: Path, data_dir: Path) -> int:
    """Map an image under a supported class directory to the canonical binary label."""
    relative_parts = path.resolve().relative_to(data_dir.resolve()).parts
    if not relative_parts:
        raise DatasetError(f"Image is outside dataset directory: {path}")
    class_directory = relative_parts[0]
    for label, class_name in enumerate(CLASS_NAMES):
        aliases = {alias.casefold() for alias in CLASS_DIRECTORY_ALIASES[class_name]}
        if class_directory.casefold() in aliases:
            return label
    raise DatasetError(f"Unknown class directory in path: {class_directory}")


def discover_samples(data_dir: Path) -> list[Sample]:
    """Discover images under `normal/` and `defective/` and validate the contract."""
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise DatasetError(f"Dataset directory does not exist: {data_dir}")

    samples: list[Sample] = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = _find_class_directory(data_dir, class_name)
        class_paths = list(_iter_images(class_dir))
        samples.extend(Sample(path=path, label=label) for path in class_paths)
        LOGGER.info("Discovered %d %s images", len(class_paths), class_name)

    counts = Counter(sample.label for sample in samples)
    if not samples or any(counts[label] == 0 for label in range(len(CLASS_NAMES))):
        raise DatasetError("Dataset must contain at least one image in each class")
    return samples


def _image_metadata(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
    except (OSError, UnidentifiedImageError) as exc:
        raise InvalidImageError(f"Unable to decode image: {path}") from exc

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"width": width, "height": height, "mode": mode, "sha256": digest}


def inspect_dataset(data_dir: Path) -> dict[str, Any]:
    """Return auditable data quality and class-balance statistics."""
    samples = discover_samples(data_dir)
    records: list[dict[str, Any]] = []
    corrupt: list[str] = []
    hashes: dict[str, list[str]] = {}
    widths: list[int] = []
    heights: list[int] = []

    for sample in samples:
        try:
            metadata = _image_metadata(sample.path)
        except InvalidImageError:
            corrupt.append(str(sample.path))
            continue
        relative_path = sample.path.relative_to(Path(data_dir).resolve()).as_posix()
        record = {
            "path": relative_path,
            "label": sample.label,
            "class_name": sample.class_name,
            **metadata,
        }
        records.append(record)
        hashes.setdefault(metadata["sha256"], []).append(relative_path)
        widths.append(metadata["width"])
        heights.append(metadata["height"])

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    counts = Counter(record["class_name"] for record in records)
    largest = max(counts.values(), default=0)
    smallest = min(counts.values(), default=0)
    stats = {
        "data_dir": str(Path(data_dir).resolve()),
        "total_images": len(samples),
        "valid_images": len(records),
        "corrupt_images": corrupt,
        "class_counts": dict(counts),
        "imbalance_ratio_largest_to_smallest": round(largest / smallest, 4) if smallest else None,
        "duplicate_groups": duplicate_groups,
        "image_dimensions": {
            "width_min": min(widths) if widths else None,
            "width_max": max(widths) if widths else None,
            "height_min": min(heights) if heights else None,
            "height_max": max(heights) if heights else None,
        },
        "records": records,
    }
    return stats


def split_samples(
    samples: Sequence[Sample],
    seed: int = 42,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, list[Sample]]:
    """Create deterministic, stratified splits with explicit small-data guardrails."""
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise DatasetError("Validation and test fractions must be between 0 and 1")
    if validation_fraction + test_fraction >= 1:
        raise DatasetError("Validation and test fractions must leave training data")

    labels = [sample.label for sample in samples]
    counts = Counter(labels)
    if len(counts) != len(CLASS_NAMES) or min(counts.values()) < 3:
        raise DatasetError(
            "Each class needs at least 3 images for stratified train/validation/test splits"
        )

    test_count = max(len(CLASS_NAMES), round(len(samples) * test_fraction))
    train_val, test = train_test_split(
        list(samples), test_size=test_count, random_state=seed, stratify=labels
    )
    validation_count = max(len(CLASS_NAMES), round(len(samples) * validation_fraction))
    validation_count = max(len(CLASS_NAMES), round(validation_count / (1 - test_fraction)))
    train, validation = train_test_split(
        train_val,
        test_size=validation_count,
        random_state=seed,
        stratify=[sample.label for sample in train_val],
    )
    result = {
        "train": sorted(train, key=lambda item: str(item.path)),
        "validation": sorted(validation, key=lambda item: str(item.path)),
        "test": sorted(test, key=lambda item: str(item.path)),
    }
    for split_name, split in result.items():
        split_counts = Counter(sample.class_name for sample in split)
        LOGGER.info("%s split: %d images (%s)", split_name, len(split), dict(split_counts))
    return result


def write_manifests(splits: dict[str, list[Sample]], data_dir: Path, output_dir: Path) -> None:
    """Write human-readable CSV and JSON split manifests."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_dir).resolve()
    manifest_json: dict[str, list[dict[str, Any]]] = {}
    for split_name, split_samples_list in splits.items():
        rows: list[dict[str, Any]] = []
        for sample in split_samples_list:
            row = {
                "path": sample.path.resolve().relative_to(data_root).as_posix(),
                "label": sample.label,
                "class_name": sample.class_name,
            }
            rows.append(row)
        manifest_json[split_name] = rows
        with (output_dir / f"{split_name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "label", "class_name"])
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "splits.json").write_text(json.dumps(manifest_json, indent=2), encoding="utf-8")


def load_manifest_samples(manifest: Path, data_dir: Path) -> list[Sample]:
    """Load one CSV manifest while ensuring paths stay within the dataset root."""
    data_root = data_dir.resolve()
    samples: list[Sample] = []
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            path = (data_root / row["path"]).resolve()
            if data_root not in path.parents:
                raise DatasetError(f"Manifest path escapes dataset directory: {row['path']}")
            samples.append(Sample(path=path, label=int(row["label"])))
    return samples


class ProductImageDataset(Dataset[tuple[Tensor, int, str]]):
    """PyTorch dataset that returns image tensor, label, and source path."""

    def __init__(self, samples: Sequence[Sample], transform: Any) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str]:
        sample = self.samples[index]
        try:
            with Image.open(sample.path) as image:
                image_rgb = image.convert("RGB")
                tensor = self.transform(image_rgb)
        except (OSError, UnidentifiedImageError) as exc:
            raise InvalidImageError(f"Unable to load image: {sample.path}") from exc
        return tensor, sample.label, str(sample.path)


def build_transform(image_size: int, training: bool) -> Any:
    """Build train/eval transforms with augmentation isolated to the train split."""
    operations: list[Any] = [transforms.Resize((image_size, image_size))]
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            ]
        )
    operations.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(operations)


def class_weights(samples: Sequence[Sample]) -> Tensor:
    """Compute inverse-frequency weights for weighted cross-entropy."""
    counts = Counter(sample.label for sample in samples)
    total = float(len(samples))
    weights = [
        total / (len(CLASS_NAMES) * counts.get(label, 1)) for label in range(len(CLASS_NAMES))
    ]
    return torch.tensor(weights, dtype=torch.float32)


def json_safe_stats(data_dir: Path) -> dict[str, Any]:
    """Return inspect data without embedding every record, for CLI summaries."""
    stats = inspect_dataset(data_dir)
    reduced = {key: value for key, value in stats.items() if key != "records"}
    reduced["sample_records"] = stats["records"][:10]
    return reduced
