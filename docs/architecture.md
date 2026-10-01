# Architecture

```mermaid
flowchart LR
    A[Product image] --> B[FastAPI validation]
    B --> C[Decode and resize]
    C --> D[Immutable PyTorch checkpoint]
    D --> E[Softmax probabilities]
    E --> F[Threshold policy]
    F --> G[Class + confidence + request ID]

    H[Class folders] --> I[Dataset inventory]
    I --> J[Stratified manifests]
    J --> K[Augmented training loader]
    K --> L[Baseline benchmark + architecture search]
    L --> M[Weighted loss + scheduler + early stopping]
    M --> N[Candidate checkpoints]
    N --> O[Validation-only model selection]
    O --> P[Final held-out test report]
    P --> D
    J --> Q[Evaluation]
    Q --> R[Metrics + FP/FN analysis]
    S[Labelled capsule image archive] --> T[Balanced sprite selection]
    T --> V[Dataset-matched belt renderer]
    V --> U[ConvNeXt-Tiny per-capsule inference]
    U --> W[Green/red annotations + ground-truth audit]
```

The selected retraining path is ConvNeXt-Tiny with official pretrained weights and full
backbone fine-tuning. The baseline models use frozen pretrained backbones and trained classifier
heads; the test set is report-only and is not used for model selection. The published video
demonstration uses the selected ConvNeXt-Tiny checkpoint on real labelled capsule images rendered
into a deterministic moving-belt scene. Each decision is retained with its source ground truth.

## Operational boundaries

- Training and evaluation are offline jobs; they write versioned artifacts and do not modify raw images.
- Inference is a read-only process over a checkpoint and does not download data or model weights at request time.
- The model decision is not a safety certification. Low-confidence predictions should be routed to manual review.
- The image dataset has Normal/Anomaly labels; the generated capsule-belt MP4 retains those labels for an animation-set audit. It is not real factory footage and is not a substitute for labelled target-line validation.

