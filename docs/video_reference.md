# Pharmaceutical inspection video reference and pipeline test asset

## Downloaded MP4 used for final pipeline testing

The repository downloads the following direct MP4 from **Shijiazhuang Huajia Medicinal Capsule Co., Ltd.**:

- Factory video page: <https://www.hjjn.com.cn/hello-world/>
- Automatic capsule production line: <https://www.hjjn.com.cn/wp-content/uploads/2022/09/Automatic-capsule-production-line.mp4>
- Capsule printing process: <https://www.hjjn.com.cn/wp-content/uploads/2022/09/Printing-on-capsules.mp4>

The first file is approximately 1:27 and is used by `scripts/video_inference.py`. It is a real recorded capsule-production-line video, but it is not frame-labelled as Normal/Anomaly. The video run is therefore a pipeline/integration test (decode -> sample -> encode -> model -> annotated MP4), not an accuracy test. Accuracy remains measured on the labelled capsule image dataset.

```powershell
python scripts/video_inference.py `
  --video data/external/automatic-capsule-production-line.mp4 `
  --checkpoint models/best.pt `
  --output-dir artifacts/video-test `
  --sample-every 30 `
  --max-frames 120 `
  --device cpu
```

Outputs are `frame_predictions.csv`, `video_summary.json`, and `annotated_sampled.mp4`.

## Additional inspection-machine reference

The real-world inspection workflow is also represented by the official **Netra VS6 Tablet Inspection** page from Accura Pharmaquip:

- Company/product page: <https://www.netra-accura.com/video.html>
- Embedded video 1: <https://www.youtube.com/watch?v=fsGr3qTU8lQ>
- Embedded video 2: <https://www.youtube.com/watch?v=QDMC3J7E8cM>

The page describes high-speed pharmaceutical tablet inspection, defect sorting, and reporting, and identifies Accura Pharmaquip as a manufacturer of pharmaceutical/food inspection systems. The videos are used only as an industry-context reference for the assessment presentation; they are not scraped into the repository or used as training data.

