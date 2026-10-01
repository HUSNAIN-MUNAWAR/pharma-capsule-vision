# Two-to-three minute demo outline

1. Show the dataset contract and run `inspect` to explain counts, dimensions, corrupt files, duplicates, and class balance.
2. Run the baseline benchmark and the ConvNeXt-Tiny full-fine-tuning run on the real capsule dataset. Point out augmentation, class-weighted loss, checkpointing, early stopping, validation-only model selection, and why ConvNeXt-Tiny was selected by validation defective F1.
3. Open the evaluation outputs and explain precision/recall/F1, the confusion matrix, and why false negatives are operationally expensive.
4. Start FastAPI, call `/health`, upload a normal and defective image to `/predict`, and show the predicted class, calibrated-to-be-selected confidence, request ID, and logs.
5. Run `scripts/video_inference.py` on the downloaded capsule-production MP4 and show `video_summary.json` plus the annotated sampled MP4. Explain that this validates video decoding, frame sampling, preprocessing, inference, and output encoding; it is not an accuracy score because the video is not frame-labelled.
6. Close with limitations: the public capsule dataset is research data, production threshold/camera slices require validation, and model monitoring is required.

