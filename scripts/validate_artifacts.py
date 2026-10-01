"""Fail fast when expected evaluation artifacts are missing or malformed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    args = parser.parse_args()
    required = ["metrics.json", "errors.csv", "confusion_matrix.csv", "confusion_matrix.png"]
    missing = [name for name in required if not (args.evaluation_dir / name).is_file()]
    if missing:
        raise SystemExit(f"Missing evaluation artifacts: {missing}")
    metrics = json.loads((args.evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
    for key in ("precision_defective", "recall_defective", "f1_defective", "confusion_matrix"):
        if key not in metrics:
            raise SystemExit(f"metrics.json is missing {key}")
    print(f"Validated evaluation artifacts in {args.evaluation_dir}")


if __name__ == "__main__":
    main()
