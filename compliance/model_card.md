# Model card: visual defect classifier

## Intended use

Assist a manufacturing inspection operator by flagging image-level `normal` and `defective` products for a fixed camera/product setup.

## Out-of-scope use

Safety certification, autonomous disposal, products/cameras/lighting not represented in validation data, or diagnosis of defects outside the trained binary labels.

## Data and labels

The validated benchmark uses the public Chukyo University medicinal-capsule image archive with `Normal` and `Anomaly` folders. The working dataset contains 600 images per class; no synthetic images are used. Dataset reports capture class counts, dimensions, corrupt files, duplicate hashes, and deterministic split manifests. The public benchmark must not be presented as a specific customer's production performance.

## Metrics and decision policy

The positive class is `defective`. Recall is operationally important because a false negative allows a defective product through. Precision matters because false positives consume manual review capacity. The threshold is a policy parameter and must be tuned on validation data with the client’s cost matrix.

## Known risks

- Camera, lighting, product orientation, and background shift can invalidate results.
- Duplicate or near-duplicate images can inflate metrics if split across partitions; exact duplicate hashes are reported and should be removed or grouped before training.
- A confidence score is model probability, not a guarantee of correctness or calibrated risk.
- The model does not explain defect location.

## Human oversight

Low-confidence predictions, out-of-distribution images, and all deployment incidents should route to a human review queue. Never use the service as the sole control for a safety-critical decision.

