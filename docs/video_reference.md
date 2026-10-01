# Dataset-matched capsule-belt animation

The published video is a controlled, dataset-matched animation generated from the same labelled pharmaceutical capsule archive used for training. This makes the object-level demo auditable: every visible capsule has a source image, a `Normal` or `Anomaly` ground-truth label, and a recorded model decision.

This replaces an unlabeled stock or factory clip for the primary evidence run. It is intentionally not presented as real factory footage. A separately annotated factory recording remains a required deployment-validation asset.

## Source data and rendering contract

- Source directory: `data/real/pharmaceutical_capsules/extracted/datasets`
- Classes: `Normal` and `Anomaly`, mapped to `normal` and `defective`
- Renderer seed: `42`
- Unique source capsules: `36` balanced assets
- Layout: three moving belt lanes on a 1280 x 720 canvas
- Model: the selected `ConvNeXt-Tiny` checkpoint at `models/best.pt`
- Annotation colors: predicted `normal` is green; predicted `defective` is red

The renderer places real source images into a deterministic capsule-belt scene, classifies each source crop with the checkpoint, and writes the prediction and source label together. The animation is therefore a pipeline and alignment test rather than a new independent model test.

## Reproduce the full run

```powershell
python scripts/generate_capsule_belt_animation.py `
  --data-dir data/real/pharmaceutical_capsules/extracted/datasets `
  --checkpoint models/best.pt `
  --output-dir artifacts/video-annotated-full `
  --frames 180 `
  --fps 24 `
  --seed 42 `
  --asset-count 36 `
  --device cpu
```

The command writes:

- `annotated_full.mp4`: full 180-frame annotated animation;
- `annotated_full.webm`: browser-compatible dashboard copy;
- `annotated_frame.png`: poster frame;
- `frame_predictions.csv`: one row per capsule decision per frame, including source path, ground truth, prediction, confidence, and box coordinates;
- `video_summary.json`: reproducibility metadata and ground-truth audit counts.

## Final evidence run

The checked-in run is 7.5 seconds at 24 FPS:

- `180` frames;
- `36` unique source capsules;
- `6,480` rendered object decisions;
- balanced source ground truth: `3,240` normal and `3,240` defective;
- `6,300 / 6,480` decisions match the source label (`97.22%` animation-set audit).

The `97.22%` figure is not a replacement for independent factory-video accuracy. It measures the selected model while the same labelled capsule images are repeatedly rendered into a belt scene. Production acceptance still requires labelled footage from the target camera, product, lighting, and line speed.
