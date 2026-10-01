# Two-to-three minute demo outline

1. Show the dataset contract and run `inspect` to explain counts, dimensions, corrupt files, duplicates, and class balance.
2. Run the baseline benchmark and the ConvNeXt-Tiny full-fine-tuning run on the real capsule dataset. Point out augmentation, class-weighted loss, checkpointing, early stopping, validation-only model selection, and why ConvNeXt-Tiny was selected by validation defective F1.
3. Open the evaluation outputs and explain precision/recall/F1, the confusion matrix, and why false negatives are operationally expensive.
4. Start FastAPI, call `/health`, upload a normal and defective image to `/predict`, and show the predicted class, calibrated-to-be-selected confidence, request ID, and logs.
5. Run `scripts/generate_capsule_belt_animation.py` and show the one-minute `video_summary.json`, the per-object CSV, and the looping annotated MP4. Explain that every moving capsule comes from the labelled Normal/Anomaly archive, the selected ConvNeXt-Tiny checkpoint classifies each object, and green-normal/red-defective boxes are model predictions. Use the CSV to compare each prediction with source ground truth.
6. Close with limitations: the generated animation is an auditable integration test, not recorded factory footage; production threshold/camera slices require labelled target-line validation and model monitoring.

