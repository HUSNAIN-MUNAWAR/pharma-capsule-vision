"""Generate a dataset-matched capsule-belt animation with auditable labels.

Every rendered object is a real image from the labelled capsule dataset. The
source class is retained as ground truth, while the trained model classifies
the same object before the green/red overlay is drawn. This makes the video a
controlled visual demonstration whose labels and image data are aligned.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from defect_detector.inference import Prediction, Predictor

CANVAS_SIZE = (1280, 720)
LANE_Y = (185, 350, 515)
BELT_X1 = 70
BELT_X2 = 1210
OBJECT_SIZE = (62, 62)


@dataclass(frozen=True)
class CapsuleAsset:
    """One real labelled capsule image and its model decision."""

    track_id: int
    image_path: Path
    ground_truth: str
    prediction: Prediction
    sprite: np.ndarray
    alpha: np.ndarray
    lane: int
    phase: float
    y_jitter: int
    angle: float


def _discover_assets(data_dir: Path, seed: int, count: int) -> list[tuple[Path, str]]:
    """Select a deterministic balanced set from Normal and Anomaly folders."""
    class_dirs = {"normal": data_dir / "Normal", "defective": data_dir / "Anomaly"}
    discovered: list[tuple[Path, str]] = []
    for class_name, class_dir in class_dirs.items():
        paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        )
        if not paths:
            raise FileNotFoundError(f"No images found for {class_name} in {class_dir}")
        discovered.extend((path, class_name) for path in paths)

    rng = random.Random(seed)
    rng.shuffle(discovered)
    per_class = max(1, count // 2)
    selected: list[tuple[Path, str]] = []
    for class_name in ("normal", "defective"):
        class_items = [item for item in discovered if item[1] == class_name]
        selected.extend(class_items[:per_class])
    rng.shuffle(selected)
    return selected


def _make_sprite(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Create a capsule sprite and soft foreground alpha from a source image."""
    image = Image.open(image_path).convert("RGB").resize(OBJECT_SIZE, Image.Resampling.LANCZOS)
    rgb = np.asarray(image, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    background = float(np.median(border))
    foreground = (np.abs(gray.astype(np.float32) - background) > 7).astype(np.uint8) * 255
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    foreground = cv2.GaussianBlur(foreground, (5, 5), 0)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), foreground


def _rotate_sprite(
    sprite: np.ndarray, alpha: np.ndarray, angle: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a sprite and alpha mask without changing canvas size."""
    height, width = sprite.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_sprite = cv2.warpAffine(sprite, matrix, (width, height), borderValue=(218, 218, 218))
    rotated_alpha = cv2.warpAffine(alpha, matrix, (width, height), borderValue=0)
    return rotated_sprite, rotated_alpha


def _paste_sprite(
    canvas: np.ndarray, sprite: np.ndarray, alpha: np.ndarray, x: int, y: int
) -> None:
    """Alpha-composite a capsule sprite onto the belt canvas."""
    height, width = sprite.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(canvas.shape[1], x + width)
    y2 = min(canvas.shape[0], y + height)
    if x1 >= x2 or y1 >= y2:
        return
    source_x1 = x1 - x
    source_y1 = y1 - y
    source_x2 = source_x1 + (x2 - x1)
    source_y2 = source_y1 + (y2 - y1)
    mask = alpha[source_y1:source_y2, source_x1:source_x2].astype(np.float32) / 255.0
    mask = mask[..., None]
    source = sprite[source_y1:source_y2, source_x1:source_x2].astype(np.float32)
    target = canvas[y1:y2, x1:x2].astype(np.float32)
    canvas[y1:y2, x1:x2] = (source * mask + target * (1.0 - mask)).astype(np.uint8)


def _belt_background(frame_index: int) -> np.ndarray:
    """Render a stable industrial belt background with moving lane texture."""
    canvas = np.full((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), (18, 29, 40), dtype=np.uint8)
    for lane_index, lane_y in enumerate(LANE_Y):
        y1, y2 = lane_y - 54, lane_y + 54
        cv2.rectangle(canvas, (BELT_X1, y1), (BELT_X2, y2), (208, 218, 221), -1)
        cv2.line(canvas, (BELT_X1, y1), (BELT_X2, y1), (245, 249, 248), 3)
        cv2.line(canvas, (BELT_X1, y2), (BELT_X2, y2), (95, 112, 120), 3)
        offset = (frame_index * 6 + lane_index * 50) % 80
        for x in range(BELT_X1 - 80 + offset, BELT_X2, 80):
            cv2.line(canvas, (x, y1 + 10), (x + 34, y2 - 10), (183, 195, 199), 2)
    cv2.rectangle(canvas, (0, 0), (CANVAS_SIZE[0], 84), (8, 18, 29), -1)
    cv2.putText(
        canvas,
        "PHARMA CAPSULE VISION  /  DATASET-MATCHED LINE SIMULATION",
        (34, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (231, 242, 246),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "REAL NORMAL / ANOMALY CAPSULE CROPS  |  GREEN = NORMAL  |  RED = DEFECTIVE",
        (36, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (123, 210, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _draw_asset(
    canvas: np.ndarray, asset: CapsuleAsset, frame_index: int, fps: int
) -> dict[str, Any] | None:
    """Draw one moving capsule and return its auditable frame row."""
    belt_width = BELT_X2 - BELT_X1
    x = int(BELT_X1 + ((asset.phase + frame_index * 4.5) % (belt_width + 140)) - 100)
    y = LANE_Y[asset.lane] + asset.y_jitter - OBJECT_SIZE[1] // 2
    sprite, alpha = _rotate_sprite(asset.sprite, asset.alpha, asset.angle)
    _paste_sprite(canvas, sprite, alpha, x, y)
    if x + OBJECT_SIZE[0] < 0 or x > CANVAS_SIZE[0]:
        return None

    predicted = asset.prediction.predicted_class
    color = (75, 205, 95) if predicted == "normal" else (50, 65, 230)
    x1, y1 = max(0, x), max(0, y)
    x2 = min(CANVAS_SIZE[0] - 1, x + OBJECT_SIZE[0])
    y2 = min(CANVAS_SIZE[1] - 1, y + OBJECT_SIZE[1])
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    label = f"{predicted[:3].upper()} {asset.prediction.confidence:.2f}"
    label_y = max(100, y1 - 5)
    cv2.putText(canvas, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return {
        "frame_index": frame_index,
        "timestamp_seconds": round(frame_index / fps, 3),
        "track_id": asset.track_id,
        "source_image": str(asset.image_path),
        "ground_truth_class": asset.ground_truth,
        "predicted_class": predicted,
        "confidence": asset.prediction.confidence,
        "probability_normal": asset.prediction.probabilities["normal"],
        "probability_defective": asset.prediction.probabilities["defective"],
        "is_correct": predicted == asset.ground_truth,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def generate_animation(
    data_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    frames: int = 180,
    fps: int = 24,
    seed: int = 42,
    asset_count: int = 36,
    device: str = "auto",
) -> dict[str, Any]:
    """Create the dataset-matched video and its per-object evidence."""
    if frames < 1 or fps < 1 or asset_count < 2:
        raise ValueError("frames/fps must be positive and asset_count must be at least two")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _discover_assets(data_dir, seed, asset_count)
    predictor = Predictor(checkpoint_path, device_name=device)
    rng = random.Random(seed)
    assets: list[CapsuleAsset] = []
    for track_id, (image_path, ground_truth) in enumerate(selected, start=1):
        prediction = predictor.predict_path(image_path)
        sprite, alpha = _make_sprite(image_path)
        assets.append(
            CapsuleAsset(
                track_id=track_id,
                image_path=image_path,
                ground_truth=ground_truth,
                prediction=prediction,
                sprite=sprite,
                alpha=alpha,
                lane=(track_id - 1) % len(LANE_Y),
                phase=rng.uniform(0, BELT_X2 - BELT_X1 + 140),
                y_jitter=rng.randint(-17, 17),
                angle=rng.uniform(-18, 18),
            )
        )

    output_path = output_dir / "annotated_full.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        CANVAS_SIZE,
    )
    if not writer.isOpened():
        raise RuntimeError("Unable to create dataset-matched annotated MP4")

    rows: list[dict[str, Any]] = []
    try:
        for frame_index in range(frames):
            canvas = _belt_background(frame_index)
            frame_rows: list[dict[str, Any]] = []
            for asset in assets:
                row = _draw_asset(canvas, asset, frame_index, fps)
                if row is not None:
                    frame_rows.append(row)
            visible_counts = Counter(row["predicted_class"] for row in frame_rows)
            correct = sum(1 for row in frame_rows if row["is_correct"])
            footer = (
                f"FRAME {frame_index + 1:03d}/{frames}  |  OBJECTS {len(frame_rows):02d}  |  "
                f"NORMAL {visible_counts['normal']:02d}  |  "
                f"DEFECTIVE {visible_counts['defective']:02d}  |  "
                f"MATCH {correct}/{len(frame_rows)}"
            )
            cv2.putText(
                canvas,
                footer,
                (34, 690),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (235, 242, 246),
                1,
                cv2.LINE_AA,
            )
            if frame_index == 0:
                cv2.imwrite(str(output_dir / "annotated_frame.png"), canvas)
            writer.write(canvas)
            rows.extend(frame_rows)
    finally:
        writer.release()

    fieldnames = [
        "frame_index",
        "timestamp_seconds",
        "track_id",
        "source_image",
        "ground_truth_class",
        "predicted_class",
        "confidence",
        "probability_normal",
        "probability_defective",
        "is_correct",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    with (output_dir / "frame_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer_csv = csv.DictWriter(stream, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    prediction_counts = Counter(row["predicted_class"] for row in rows)
    ground_truth_counts = Counter(row["ground_truth_class"] for row in rows)
    correct = sum(1 for row in rows if row["is_correct"])
    summary = {
        "video_type": "dataset_matched_animation",
        "source_dataset": str(data_dir.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "frames": frames,
        "fps": fps,
        "duration_seconds": round(frames / fps, 3),
        "unique_capsules_rendered": len(assets),
        "object_decisions": len(rows),
        "ground_truth_available": True,
        "ground_truth_class_counts": dict(ground_truth_counts),
        "predicted_class_counts": dict(prediction_counts),
        "correct_object_decisions": correct,
        "object_accuracy": round(correct / len(rows), 6) if rows else 0.0,
        "annotation_semantics": {
            "normal": "green",
            "defective": "red",
            "box_color_source": "predicted_class",
        },
        "annotated_video": str(output_path.resolve()),
        "interpretation": (
            "Every visible object originates from the labelled capsule dataset; metrics are "
            "animation-set checks, not factory-video accuracy."
        ),
    }
    (output_dir / "video_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a dataset-matched capsule belt animation"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--asset-count", type=int, default=36)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    summary = generate_animation(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        frames=args.frames,
        fps=args.fps,
        seed=args.seed,
        asset_count=args.asset_count,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
