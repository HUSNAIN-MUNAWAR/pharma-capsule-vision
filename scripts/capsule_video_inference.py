"""Annotate each moving capsule in a close-up belt video.

The pipeline is intentionally two-stage:

1. detect capsule proposals from the green cap on the light conveyor;
2. classify every crop with the trained Normal/Anomaly model.

The video is an unlabeled integration asset. Green/red overlays are model
decisions and must not be interpreted as ground-truth accuracy measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from defect_detector.inference import Predictor
from defect_detector.video import CapsuleBox, detect_capsules

SOURCE_VIDEO_URL = "https://www.youtube.com/watch?v=hmEHlahGjD8"
SOURCE_VIDEO_TITLE = "Conveyor with Gelatin Capsules | Stock Footage - Videohive"


def _label_color(predicted_class: str) -> tuple[int, int, int]:
    """Return a BGR overlay color for a model decision."""
    return (70, 200, 85) if predicted_class == "normal" else (45, 65, 225)


def _crop(frame: Any, box: CapsuleBox) -> Any:
    """Return a bounded crop for one capsule proposal."""
    return frame[box.y1 : box.y2, box.x1 : box.x2]


def _annotate_frame(
    frame: Any,
    boxes: list[CapsuleBox],
    predictor: Predictor,
) -> tuple[Any, list[dict[str, Any]]]:
    """Classify and draw all capsule proposals in one frame."""
    annotated = frame.copy()
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for object_index, box in enumerate(boxes, start=1):
        encoded, buffer = cv2.imencode(".jpg", _crop(frame, box))
        if not encoded:
            continue
        prediction = predictor.predict_bytes(buffer.tobytes())
        counts[prediction.predicted_class] += 1
        color = _label_color(prediction.predicted_class)
        cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), color, 2, cv2.LINE_AA)
        label = (
            f"{object_index:02d} {prediction.predicted_class[:3].upper()} "
            f"{prediction.confidence:.2f}"
        )
        label_y = max(14, box.y1 - 4)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
        )
        cv2.rectangle(
            annotated,
            (box.x1, label_y - text_height - baseline - 3),
            (box.x1 + text_width + 4, label_y + 2),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (box.x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rows.append(
            {
                "object_index": object_index,
                "x1": box.x1,
                "y1": box.y1,
                "x2": box.x2,
                "y2": box.y2,
                "predicted_class": prediction.predicted_class,
                "confidence": prediction.confidence,
                "probability_normal": prediction.probabilities["normal"],
                "probability_defective": prediction.probabilities["defective"],
                "model_version": prediction.model_version,
            }
        )

    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (annotated.shape[1], 42), (18, 27, 38), -1)
    annotated = cv2.addWeighted(overlay, 0.88, annotated, 0.12, 0)
    hud = (
        f"OBJECT-LEVEL AI  |  CAPSULES {len(rows):02d}  |  "
        f"NORMAL {counts['normal']:02d}  |  DEFECTIVE {counts['defective']:02d}"
    )
    cv2.putText(
        annotated,
        hud,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 242, 246),
        1,
        cv2.LINE_AA,
    )
    return annotated, rows


def run_capsule_video_inference(
    video_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    sample_every: int = 1,
    max_frames: int = 0,
    device: str = "auto",
    min_area: int = 15,
    max_area: int = 800,
    output_scale: int = 2,
) -> dict[str, Any]:
    """Run object-level inference over the complete capsule-belt clip."""
    if sample_every < 1 or max_frames < 0 or output_scale < 1:
        raise ValueError(
            "sample_every/output_scale must be positive; max_frames cannot be negative"
        )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Video has invalid dimensions: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / "annotated_full.mp4"
    output_size = (width * output_scale, height * output_scale)
    writer = cv2.VideoWriter(
        str(annotated_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(fps / sample_every, 1.0),
        output_size,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Unable to create annotated MP4; check the local video codec")

    predictor = Predictor(checkpoint_path, device_name=device)
    rows: list[dict[str, Any]] = []
    frame_index = 0
    annotated_frames = 0
    total_predictions: Counter[str] = Counter()
    try:
        while max_frames == 0 or frame_index < max_frames:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % sample_every == 0:
                boxes = detect_capsules(frame, min_area=min_area, max_area=max_area)
                annotated, frame_rows = _annotate_frame(frame, boxes, predictor)
                for row in frame_rows:
                    row["frame_index"] = frame_index
                    row["timestamp_seconds"] = round(frame_index / fps, 3)
                    rows.append(row)
                    total_predictions[row["predicted_class"]] += 1
                if annotated_frames == 0:
                    cv2.imwrite(str(output_dir / "annotated_frame.png"), annotated)
                writer.write(cv2.resize(annotated, output_size, interpolation=cv2.INTER_LINEAR))
                annotated_frames += 1
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    fieldnames = [
        "frame_index",
        "timestamp_seconds",
        "object_index",
        "x1",
        "y1",
        "x2",
        "y2",
        "predicted_class",
        "confidence",
        "probability_normal",
        "probability_defective",
        "model_version",
    ]
    with (output_dir / "frame_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer_csv = csv.DictWriter(stream, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    summary = {
        "video_path": str(video_path.resolve()),
        "source_video_url": SOURCE_VIDEO_URL,
        "source_video_title": SOURCE_VIDEO_TITLE,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "inference_type": "object_level_capsule_classification",
        "proposal_detector": "green-cap HSV connected components",
        "source_fps": fps,
        "source_dimensions": {"width": width, "height": height},
        "source_frames_read": frame_index,
        "sample_every": sample_every,
        "annotated_frames": annotated_frames,
        "objects_annotated_total": len(rows),
        "predicted_object_class_counts": dict(total_predictions),
        "annotated_video": str(annotated_path.resolve()),
        "ground_truth_available": False,
        "interpretation": (
            "Object overlays are model decisions on an unlabeled integration video, "
            "not accuracy evidence."
        ),
    }
    (output_dir / "video_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run object-level inference on a capsule belt MP4")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means the complete video")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-area", type=int, default=15)
    parser.add_argument("--max-area", type=int, default=800)
    parser.add_argument("--output-scale", type=int, default=2)
    args = parser.parse_args()
    summary = run_capsule_video_inference(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        sample_every=args.sample_every,
        max_frames=args.max_frames,
        device=args.device,
        min_area=args.min_area,
        max_area=args.max_area,
        output_scale=args.output_scale,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
