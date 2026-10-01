# Real pharmaceutical dataset provenance

## Selected dataset

The project uses the **Medicinal capsule image dataset** published by Chukyo University’s Advanced Sensing and Machine Intelligence archive:

- Source page: <https://isl.sist.chukyo-u.ac.jp/archives/capsule_en/>
- Direct archive retrieved for this workspace: `https://isl.sist.chukyo-u.ac.jp/wp-content/uploads/2025/12/datasets.zip`
- Retrieval date: 2026-10-01 (Asia/Karachi)
- Archive size: 10,294,710 bytes
- Classes: `Normal` (600 images) and `Anomaly` (600 images)
- Image format: PNG, grayscale release images, 128 x 128 pixels
- Original capture: industrial camera, 1920 x 1080 pixels, 3 color channels; the release page states that images were center-cropped, converted to grayscale, and resized.
- Intended use stated by the source: evaluating visual inspection algorithms, including supervised normal/anomaly classification and unsupervised anomaly detection.

The repository’s loader accepts the source class names without modifying the downloaded images. It maps `Normal` to `normal` and `Anomaly` to the positive `defective` class for the assessment API.

## Reproducible download

```powershell
New-Item -ItemType Directory -Force data/external | Out-Null
Invoke-WebRequest `
  -Uri "https://isl.sist.chukyo-u.ac.jp/wp-content/uploads/2025/12/datasets.zip" `
  -OutFile "data/external/medicinal_capsule_dataset.zip"
Expand-Archive data/external/medicinal_capsule_dataset.zip `
  -DestinationPath data/real/pharmaceutical_capsules/extracted
```

The source page requests citation of its archive URL/reference for publications. No separate permissive software license is stated on the page, so this repository treats the data as research-use material and does not redistribute it through Git. Confirm the source’s current terms before commercial deployment.

## Why it matches the assessment

It is real industrial-camera pharmaceutical capsule imagery with normal and anomalous classes, rather than generated demo images. The assessment’s binary `normal`/`defective` interface is preserved by treating the source `Anomaly` class as `defective`. The dataset is still a benchmark and not proof of a particular company’s production performance; camera, line speed, product SKU, lighting, and defect taxonomy must be validated with the target manufacturer.

