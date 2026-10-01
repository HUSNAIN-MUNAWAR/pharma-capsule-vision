# Testing and release checks

`tests/` contains unit and API contract tests. This folder documents the release gate expected before a model is promoted:

```powershell
ruff check src scripts tests testing
mypy src
pytest --cov=src/defect_detector --cov-report=term-missing
python scripts/validate_artifacts.py --evaluation-dir artifacts/demo/evaluation
```

For a real model, add a fixed test fixture for camera resolution, one malformed image, one oversized request, and representative false-positive/false-negative examples. Keep customer images out of source control.

