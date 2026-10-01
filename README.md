# Pharma Capsule Vision

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-inference-009688?logo=fastapi&logoColor=white)
![Domain](https://img.shields.io/badge/domain-pharmaceutical%20inspection-1f6feb)
![Status](https://img.shields.io/badge/status-reproducible%20reference%20implementation-2ea44f)

Industrial computer-vision platform for pharmaceutical capsule inspection. The project trains and benchmarks multiple classifiers on a real capsule image dataset, evaluates a selected model on a held-out split, runs inference over a real capsule-production video, exposes a FastAPI service, and presents the evidence in a compact industrial dashboard.

## Live pipeline evidence

The following is the full annotated MP4 produced by the final video-inference run. It contains predictions on all 871 source frames. The source video is an unlabeled integration/domain-shift asset, so these annotations demonstrate pipeline behavior and are not used to claim video accuracy.

<video controls preload="metadata" width="100%" poster="https://raw.githubusercontent.com/HUSNAIN-MUNAWAR/pharma-capsule-vision/main/artifacts/video-annotated-full/annotated_frame.png">
  <source src="https://raw.githubusercontent.com/HUSNAIN-MUNAWAR/pharma-capsule-vision/main/artifacts/video-annotated-full/annotated_full.mp4" type="video/mp4">
  Your browser does not support embedded video. Use the MP4 link below.
</video>

[Open or download the full annotated MP4](https://github.com/HUSNAIN-MUNAWAR/pharma-capsule-vision/raw/refs/heads/main/artifacts/video-annotated-full/annotated_full.mp4) · [Browser-compatible WebM](https://github.com/HUSNAIN-MUNAWAR/pharma-capsule-vision/raw/refs/heads/main/artifacts/video-annotated-full/annotated_full.webm)

## Project summary

| Area | Implementation |
|---|---|
| Inspection task | Binary capsule classification: `Normal` vs `Anomaly` |
| Image data | 1,200 real pharmaceutical capsule images: 600 normal and 600 anomalous |
| Model protocol | Four baseline families plus ConvNeXt-Tiny full fine-tuning; 100-epoch ceiling with early stopping |
| Selected model | ConvNeXt-Tiny, selected using validation defective-class F1 only |
| Held-out result | 1.0000 accuracy, precision, recall, and F1 on the fixed 180-image test split |
| Video integration | Full 871-frame annotated run from a real capsule-production MP4 |
| Runtime | FastAPI `/health`, `/ready`, and `/predict` endpoints with structured request logs |
| Operator experience | Industrial dark-mode dashboard with live video controls, KPI cards, confusion matrix, and evidence links |
| Engineering controls | Ruff, mypy, pytest, artifact validation, Docker, model card, and compliance checklist |

These metrics are evidence for this dataset split and protocol, not a production guarantee. Independent factory data, threshold calibration, camera-shift testing, and operator review are required before deployment.

## Table of contents

- [Live pipeline evidence](#live-pipeline-evidence)
- [Project summary](#project-summary)
- [System architecture](#system-architecture)
  - [Architecture diagram](#architecture-diagram)
  - [End-to-end sequence](#end-to-end-sequence)
  - [Runtime state model](#runtime-state-model)
  - [Deployment and evidence flow](#deployment-and-evidence-flow)
- [Dataset and video alignment](#dataset-and-video-alignment)
- [Model training and benchmarking](#model-training-and-benchmarking)
- [Quickstart](#quickstart)
- [Inference API](#inference-api)
- [Industrial dashboard](#industrial-dashboard)
- [Repository layout](#repository-layout)
- [Quality gates](#quality-gates)
- [Compliance and limitations](#compliance-and-limitations)
- [Task alignment](#task-alignment)
- [References](#references)

## System architecture

### Architecture diagram

```mermaid
flowchart LR
    A[Real capsule image archive] --> B[Inventory and data validation]
    B --> C[Duplicate and corruption checks]
    C --> D[Deterministic stratified split]

    D --> E1[SmallCNN from scratch]
    D --> E2[ResNet18 transfer head]
    D --> E3[MobileNetV3-Small transfer head]
    D --> E4[EfficientNet-B0 transfer head]
    D --> E5[ConvNeXt-Tiny full fine-tune]

    E1 --> F[Validation metrics and training curves]
    E2 --> F
    E3 --> F
    E4 --> F
    E5 --> F
    F --> G[Select by validation defective F1]
    G --> H[Held-out test report]
    H --> I[Versioned model metadata]

    I --> J[FastAPI inference service]
    J --> K[Image prediction endpoint]
    J --> L[Video frame inference]
    L --> M[Annotated MP4/WebM evidence]
    K --> N[Industrial dashboard]
    M --> N
    H --> N
```

The training and serving paths are intentionally separate. The API loads an immutable checkpoint at startup; it never silently retrains or downloads weights during a request. The dashboard consumes the service contract and versioned evidence artifacts rather than embedding training logic.

### End-to-end sequence

```mermaid
sequenceDiagram
    autonumber
    participant Operator
    participant Dataset as Dataset validator
    participant Trainer as Benchmark runner
    participant Registry as Evidence artifacts
    participant API as FastAPI service
    participant Model as ConvNeXt-Tiny
    participant UI as Inspection dashboard

    Operator->>Dataset: Provide capsule image directory
    Dataset->>Dataset: Inspect labels, files, duplicates, and split integrity
    Dataset-->>Trainer: Deterministic train/validation/test manifests
    Trainer->>Trainer: Train each candidate with identical protocol
    Trainer->>Registry: Write history, curves, metrics, confusion matrix
    Trainer->>Registry: Select using validation defective F1
    Registry-->>API: Load selected checkpoint and metadata
    Operator->>API: Upload image or start video inference
    API->>Model: Preprocess and classify frame
    Model-->>API: Class probabilities and confidence
    API-->>Operator: Structured prediction with request ID
    API->>Registry: Persist annotated video and run summary
    UI->>Registry: Load video, KPIs, metrics, and evidence links
    Registry-->>UI: Render operator-facing inspection console
```

### Runtime state model

```mermaid
stateDiagram-v2
    [*] --> DatasetUnvalidated
    DatasetUnvalidated --> DatasetReady: inspect + integrity checks pass
    DatasetUnvalidated --> DataRejected: corrupt, empty, or invalid labels
    DataRejected --> DatasetUnvalidated: correct source data
    DatasetReady --> Training
    Training --> CandidateEvaluated: epoch metrics written
    CandidateEvaluated --> Training: patience not exhausted
    CandidateEvaluated --> ModelSelected: validation defective F1 improves
    CandidateEvaluated --> ModelSelected: early stopping reached
    ModelSelected --> TestReported: held-out test evaluated once
    TestReported --> ServiceReady: checkpoint and metadata load
    ServiceReady --> ImageInferred: valid image request
    ServiceReady --> VideoRunning: video integration run
    ServiceReady --> ServiceFault: missing or invalid checkpoint
    ImageInferred --> ServiceReady
    VideoRunning --> EvidenceWritten: annotated frames and summary saved
    EvidenceWritten --> ServiceReady
    ServiceFault --> [*]
```

### Deployment and evidence flow

```mermaid
flowchart TB
    subgraph Build[Reproducible build and verification]
        B1[Source code] --> B2[ruff format/check]
        B1 --> B3[mypy]
        B1 --> B4[pytest]
        B1 --> B5[artifact validator]
        B2 --> B6[Verified commit]
        B3 --> B6
        B4 --> B6
        B5 --> B6
    end

    subgraph Runtime[Local or container runtime]
        R1[Docker or Python environment] --> R2[FastAPI]
        R2 --> R3[Image endpoint]
        R2 --> R4[Video inference worker]
        R4 --> R5[Annotated video]
    end

    subgraph Review[Human review surface]
        V1[Dashboard] --> V2[KPIs and confusion matrix]
        V1 --> V3[Video player]
        V1 --> V4[Metrics and run evidence]
    end

    B6 --> Runtime
    R5 --> Review
    B6 --> Review
```

## Dataset and video alignment

The image and video assets are matched at the product/process-domain level:

- The image dataset is a public research archive of pharmaceutical capsule inspection images from Chukyo University. It contains `Normal` and `Anomaly` labels and is used for supervised training and held-out evaluation.
- The final integration video is a real pharmaceutical capsule-production line recording associated with Shijiazhuang Huajia Medicinal Capsule Co., Ltd. It is used for end-to-end frame inference and annotation.
- The video has no frame-level ground truth in this repository. Its correct role is domain-shift and pipeline validation, not accuracy scoring.
- The final video run read 871 frames at the source frame rate and wrote an annotation for every frame. Summary data is available in [`video_summary.json`](artifacts/video-annotated-full/video_summary.json) and per-frame predictions in [`frame_predictions.csv`](artifacts/video-annotated-full/frame_predictions.csv).

Dataset provenance and reproducible download instructions are documented in [`docs/dataset_provenance.md`](docs/dataset_provenance.md). The raw dataset and source videos are deliberately not committed; the public repository contains the generated evidence required to inspect the completed pipeline.

## Model training and benchmarking

All candidates use the same deterministic split (`808` train, `212` validation, `180` test), seed `42`, image size `128`, and a 100-epoch ceiling. The benchmark selects on validation defective-class F1; the test set is report-only.

| Model | Role | Accuracy | Defective precision | Defective recall | Defective F1 |
|---|---|---:|---:|---:|---:|
| EfficientNet-B0 | Frozen-head transfer baseline | 0.7833 | 0.8148 | 0.7333 | 0.7719 |
| MobileNetV3-Small | Lightweight transfer baseline | 0.7556 | 0.7447 | 0.7778 | 0.7609 |
| ResNet18 | Stable transfer baseline | 0.6000 | 0.6071 | 0.5667 | 0.5862 |
| SmallCNN | From-scratch control | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| ConvNeXt-Tiny | Full fine-tuning, selected run | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

The ConvNeXt-Tiny run used official pretrained weights, full-backbone fine-tuning, learning rate `1e-4`, batch size `32`, ReduceLROnPlateau, and patience-10 early stopping. It stopped at epoch 18 with its best checkpoint at epoch 8. The checkpoint is intentionally not committed because it is approximately 106 MB; the documented command recreates it locally.

Evidence directories:

- [`artifacts/model-benchmark`](artifacts/model-benchmark): baseline benchmark CSV/JSON, logs, curves, and evaluation reports.
- [`artifacts/model-retrain`](artifacts/model-retrain): ConvNeXt-Tiny retraining run and selected evaluation evidence.
- [`artifacts/pharma-inspection/dataset_report.json`](artifacts/pharma-inspection/dataset_report.json): dataset inventory and integrity report.
- [`compliance/model_card.md`](compliance/model_card.md): intended use, metrics, limitations, and risk notes.

### Reproduce the benchmark

```powershell
python scripts/benchmark_models.py `
  --data-dir data/real/pharmaceutical_capsules/extracted/datasets `
  --output-dir artifacts/model-benchmark `
  --models small_cnn resnet18 mobilenet_v3_small efficientnet_b0 `
  --epochs 100 `
  --patience 10 `
  --batch-size 64 `
  --device cpu
```

### Reproduce the selected ConvNeXt-Tiny run

```powershell
python scripts/benchmark_models.py `
  --data-dir data/real/pharmaceutical_capsules/extracted/datasets `
  --output-dir artifacts/model-retrain `
  --models convnext_tiny `
  --image-size 128 `
  --batch-size 32 `
  --learning-rate 0.0001 `
  --epochs 100 `
  --patience 10 `
  --fine-tune `
  --seed 42 `
  --device cpu
```

## Quickstart

### Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

### Download the real capsule dataset

```powershell
New-Item -ItemType Directory -Force data/external | Out-Null
Invoke-WebRequest -Uri "https://isl.sist.chukyo-u.ac.jp/wp-content/uploads/2025/12/datasets.zip" -OutFile "data/external/medicinal_capsule_dataset.zip"
Expand-Archive data/external/medicinal_capsule_dataset.zip -DestinationPath data/real/pharmaceutical_capsules/extracted
```

### Inspect, train, and evaluate

```powershell
python -m defect_detector.cli inspect --data-dir data/real/pharmaceutical_capsules/extracted/datasets --output-dir artifacts/pharma-inspection
python -m defect_detector.cli split --data-dir data/real/pharmaceutical_capsules/extracted/datasets --output-dir data/processed/pharma-manifests --seed 42
python scripts/benchmark_models.py --data-dir data/real/pharmaceutical_capsules/extracted/datasets --output-dir artifacts/model-benchmark --epochs 100 --patience 10 --batch-size 64 --device cpu
Copy-Item artifacts/model-retrain/convnext_tiny/best.pt models/best.pt -Force
python -m defect_detector.cli evaluate --data-dir data/real/pharmaceutical_capsules/extracted/datasets --checkpoint models/best.pt --output-dir artifacts/model-retrain/selected/evaluation --device cpu
```

### Run the API

```powershell
uvicorn defect_detector.api:app --host 0.0.0.0 --port 8000
```

## Inference API

The service loads the model configured by `DEFECT_MODEL_PATH` or defaults to `models/best.pt`.

```powershell
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/ready
curl.exe -X POST http://localhost:8000/predict -F "file=@data/real/pharmaceutical_capsules/extracted/datasets/Anomaly/001.png"
```

Example response:

```json
{
  "request_id": "5b9a...",
  "predicted_class": "defective",
  "confidence": 0.9731,
  "probabilities": {"normal": 0.0269, "defective": 0.9731},
  "model_version": "convnext_tiny-seed42-epoch8"
}
```

The API validates content type, payload size, image readability, model readiness, and prediction output. Logs include request IDs and latency so a deployment can connect predictions to an operational trace.

## Industrial dashboard

The dashboard is a static operator-facing console served from the repository root. It presents the active model, held-out metrics, confusion matrix, full-run annotated video, and direct evidence links.

```powershell
python scripts/serve_dashboard.py --port 8765
```

Open <http://127.0.0.1:8765/web/>. The player uses the browser-compatible [`annotated_full.webm`](artifacts/video-annotated-full/annotated_full.webm) and keeps [`annotated_full.mp4`](artifacts/video-annotated-full/annotated_full.mp4) as the MP4 fallback and download asset.

![Industrial capsule inspection dashboard](artifacts/dashboard/inspection-console.png)

## Repository layout

```text
.
├── artifacts/       # benchmark reports, metrics, curves, screenshot, annotated video
├── compliance/      # data protection checklist and model card
├── docs/            # architecture, provenance, demo, and video reference
├── scripts/         # benchmark, video inference, validation, and dashboard server
├── src/             # typed application and ML package
├── testing/         # testing strategy and review notes
├── tests/           # unit and API tests
├── web/             # dashboard HTML, CSS, and JavaScript
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

Assessment inputs, raw data, source videos, local checkpoints, caches, and generated package metadata are excluded by [`.gitignore`](.gitignore). This keeps the public repository reproducible without publishing the task PDF or extracted task text.

## Quality gates

The completed implementation was checked with:

```powershell
ruff format --check src scripts tests testing
ruff check src scripts tests testing
mypy src
pytest
python scripts/validate_artifacts.py
```

The current evidence contains seven passing tests, successful static typing, clean lint/format checks, and validated selected-run artifacts. The test suite covers data discovery/splits, model construction, and API behavior.

## Compliance and limitations

- The capsule image dataset is public research data, not customer-owned production data. Do not represent these metrics as performance for a specific pharmaceutical company or factory.
- The factory video is used for integration testing because it is not frame-labelled. It must not be converted into an accuracy claim without an independently annotated evaluation set.
- The classifier is image-level and does not localize defect pixels. Grad-CAM or a detection/segmentation model is the appropriate next step when operators require visual explanations.
- Before deployment, calibrate the threshold against the cost of missed defects and unnecessary manual review, then test lighting, camera, product, and line-speed shifts.
- The model checkpoint is excluded from the repository due to its size. A release workflow should publish versioned checkpoints through an artifact registry or Git LFS after governance approval.

See [`compliance/data_protection_checklist.md`](compliance/data_protection_checklist.md), [`compliance/model_card.md`](compliance/model_card.md), and [`docs/architecture.md`](docs/architecture.md) for the detailed controls.

## Task alignment

| Requested deliverable | Implementation status | Evidence |
|---|---|---|
| Real pharmaceutical dataset | Complete | Chukyo capsule archive, provenance document, dataset report |
| Matching factory video | Complete | Real capsule-production MP4, full-frame annotated output |
| Train and compare multiple models | Complete | Four baselines plus ConvNeXt-Tiny full fine-tuning |
| Long-run training protocol | Complete | 100-epoch ceiling, scheduler, early stopping, saved histories |
| Benchmark evidence | Complete | Metrics JSON/CSV, curves, confusion matrices, logs |
| Professional application flow | Complete | Typed package, FastAPI service, structured logging, Docker |
| Testing and compliance structure | Complete | `tests/`, `testing/`, `compliance/`, lint, mypy, artifact validation |
| Professional UI | Complete | Industrial dashboard, screenshot, playable annotated video |
| Task PDF and extracted notes kept private | Complete | Explicit `.gitignore` rules; neither file is in Git history |

Overall, the repository is aligned with the requested engineering deliverables. The remaining production step is independent validation on labelled factory footage and deployment-specific acceptance testing; the current video demonstrates the complete inference pipeline but is not an accuracy dataset.

## References

- [Chukyo University industrial-vision archive](https://isl.sist.chukyo-u.ac.jp/)
- [Shijiazhuang Huajia Medicinal Capsule Co., Ltd. video reference](https://www.hjjn.com.cn/hello-world/)
- [TorchVision model catalog](https://docs.pytorch.org/vision/master/models.html)
- [Controlled packaging inspection study](https://doi.org/10.1145/3816713.3820244)
- [Pharmaceutical capsule inspection study](https://onlinelibrary.wiley.com/doi/10.1155/2022/4820618)
- [RACNN capsule inspection study](https://onlinelibrary.wiley.com/doi/10.1155/2020/8887723)
