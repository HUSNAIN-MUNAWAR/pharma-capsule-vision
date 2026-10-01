"""Run frame-level inference over a recorded pharmaceutical manufacturing video.

The video is a process-context test asset, not a labelled benchmark. The output
therefore measures pipeline execution and frame-level prediction distribution,
not classification accuracy.
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


def run_video_inference(
    video_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    sample_every: int = 30,
    max_frames: int = 120,
    device: str = "auto",
) -> dict[str, Any]:
    """Sample frames, run the API-equivalent predictor, and write an annotated MP4."""
    if sample_every < 1 or max_frames < 1:
        raise ValueError("sample_every and max_frames must be positive")
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
    annotated_path = output_dir / "annotated_sampled.mp4"
    writer = cv2.VideoWriter(
        str(annotated_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(fps / sample_every, 1.0),
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Unable to create annotated MP4; check the local video codec")

    predictor = Predictor(checkpoint_path, device_name=device)
    rows: list[dict[str, Any]] = []
    frame_index = 0
    try:
        while len(rows) < max_frames:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % sample_every == 0:
                encoded, buffer = cv2.imencode(".jpg", frame)
                if not encoded:
                    raise RuntimeError(f"Unable to encode frame {frame_index}")
                prediction = predictor.predict_bytes(buffer.tobytes())
                row = {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_index / fps, 3),
                    "predicted_class": prediction.predicted_class,
                    "confidence": prediction.confidence,
                    **{
                        f"probability_{name}": value
                        for name, value in prediction.probabilities.items()
                    },
                }
                rows.append(row)
                label = f"{prediction.predicted_class} ({prediction.confidence:.3f})"
                cv2.putText(
                    frame,
                    label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255) if prediction.predicted_class == "defective" else (0, 160, 0),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    predictions = Counter(row["predicted_class"] for row in rows)
    summary = {
        "video_path": str(video_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "source_fps": fps,
        "source_dimensions": {"width": width, "height": height},
        "source_frames_read": frame_index,
        "sample_every": sample_every,
        "sampled_frames": len(rows),
        "predicted_class_counts": dict(predictions),
        "annotated_video": str(annotated_path.resolve()),
    }
    with (output_dir / "frame_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = (
            list(rows[0])
            if rows
            else ["frame_index", "timestamp_seconds", "predicted_class", "confidence"]
        )
        writer_csv = csv.DictWriter(stream, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    (output_dir / "video_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frame inference on an MP4")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-every", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    summary = run_video_inference(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        sample_every=args.sample_every,
        max_frames=args.max_frames,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
