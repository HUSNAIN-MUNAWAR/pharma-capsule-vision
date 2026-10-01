"""Predict one local image with a saved checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from defect_detector.inference import Predictor, prediction_as_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    predictor = Predictor(args.checkpoint, device_name=args.device)
    print(json.dumps(prediction_as_dict(predictor.predict_path(args.image)), indent=2))


if __name__ == "__main__":
    main()
