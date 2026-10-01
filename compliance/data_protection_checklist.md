# Data protection and release checklist

- [ ] Confirm the customer owns/authorizes every image and any derived artifact.
- [ ] Confirm rights to download, retain, annotate, and redistribute any recorded factory video or extracted frames.
- [ ] Remove faces, badges, serial numbers, and unrelated personal data where possible.
- [ ] Keep raw images outside Git and restrict model/artifact storage permissions.
- [ ] Record dataset version, label source, camera setup, and collection date.
- [ ] Check exact duplicates and leakage across train/validation/test.
- [ ] Review false positives and false negatives with a domain expert.
- [ ] Set a documented threshold and manual-review policy.
- [ ] Scan dependencies and container images before deployment.
- [ ] Log request IDs and outcomes without logging image bytes or sensitive metadata.
- [ ] Define retention, deletion, rollback, and incident-response procedures.

