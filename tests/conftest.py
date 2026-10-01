from __future__ import annotations

import random
from pathlib import Path

import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    randomizer = random.Random(11)
    root = tmp_path / "dataset"
    for class_name, color in (("normal", (80, 120, 180)), ("defective", (190, 50, 50))):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for index in range(12):
            image = Image.new("RGB", (40, 40), color)
            draw = ImageDraw.Draw(image)
            draw.rectangle((5, 5, 34, 34), outline=(230, 230, 230), width=2)
            draw.point((randomizer.randrange(40), randomizer.randrange(40)), fill=(0, 0, 0))
            image.save(class_dir / f"{class_name}-{index}.png")
    return root
