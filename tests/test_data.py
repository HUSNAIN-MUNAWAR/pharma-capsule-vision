from pathlib import Path

import pytest

from defect_detector.data import discover_samples, inspect_dataset, split_samples, write_manifests
from defect_detector.exceptions import DatasetError


def test_inspect_and_stratified_split(tiny_dataset: Path, tmp_path: Path) -> None:
    report = inspect_dataset(tiny_dataset)
    assert report["total_images"] == 24
    assert report["corrupt_images"] == []
    assert report["class_counts"] == {"normal": 12, "defective": 12}

    splits = split_samples(discover_samples(tiny_dataset), seed=7)
    assert set(splits) == {"train", "validation", "test"}
    assert all({sample.label for sample in split} == {0, 1} for split in splits.values())
    write_manifests(splits, tiny_dataset, tmp_path / "manifests")
    assert (tmp_path / "manifests" / "splits.json").is_file()


def test_split_rejects_too_small_classes(tiny_dataset: Path) -> None:
    samples = discover_samples(tiny_dataset)
    with pytest.raises(DatasetError, match="at least 3"):
        split_samples(samples[:5])
