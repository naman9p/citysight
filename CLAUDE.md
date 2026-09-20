# CitySight — Claude Instructions

## Current Scope

Phase 1 (ANPR Engine) is complete: Steps 1–16, video → detection → tracking →
quality → rectification → OCR → fusion → normalization → confidence →
canonical observation → SQLite persistence → HTTP API → dashboard → watchlist
alerts.

Phase 2 is development-complete through Step 33: city camera topology, deterministic
recorded multi-camera replay, exact-plate trajectories, optional whole-vehicle
enrichment, explainable bounded cross-camera candidate hypotheses, explicit
ground-truth candidate evaluation, the demo/evaluation runbook, and the frozen
real-world two-camera baseline.

Step 34A/34B is the real-world ground-truth evaluation benchmark. Its format,
capture/evaluation infrastructure, frame preparation, and local annotation
tooling are complete. Exhaustive manual labels and final population-level
metrics remain incomplete, so no accuracy claim is authorized. Phase 3
development may proceed from Step 35, but final Phase 2-versus-Phase 3 benchmark
claims remain blocked until Step 34B is complete.

Do not implement later roadmap features unless explicitly asked.

## Phase 1 Pipeline (as built)

Video
→ YOLO License Plate Detection
→ ByteTrack Tracking
→ Quality Scoring (best-frame selection)
→ Perspective Rectification
→ PaddleOCR
→ Multi-frame Fusion
→ Indian Plate Normalization
→ Confidence Decision (accepted / review / abstained)
→ Canonical Observation Event
→ SQLite Persistence (+ filesystem evidence store)
→ Stdlib HTTP API + Dashboard
→ Watchlist Deduplicated Alerts

## Phase 2 Flow (as built)

Recorded Multi-Camera Scenario
→ Canonical Observations + Optional Transient Vehicle Fingerprints
→ Explainable Pairwise Cross-Camera Matching
→ Bounded Deterministic Candidate Collection
→ Read-Only Candidate Report
→ Explicit External Ground Truth
→ Candidate Recall@K + HitRate@K + MRR

Candidate hypotheses are not confirmed vehicle identities. Exact-plate
trajectories remain independent from fingerprint candidate hypotheses.

## Tech Stack

* Python 3.11+
* OpenCV
* Ultralytics YOLO + BYTETracker
* PaddleOCR
* NumPy
* Pydantic
* PyYAML
* SQLite (stdlib), stdlib http.server
* pytest

## Important Rules

* Keep code simple and readable.
* Prefer modular code over unnecessary abstraction.
* Do not use Redis.
* Do not use Kafka.
* Do not use Kubernetes.
* Do not use cloud services.
* Persistence is SQLite + a local filesystem evidence store. Their repository /
  evidence-store interfaces are intentionally swappable (Postgres/MinIO/S3
  later) — keep callers depending on the interfaces, not concrete backends.
* The API is the stdlib `http.server` (no FastAPI/framework). The dashboard is
  plain HTML/CSS/vanilla JS served by it. These were approved for the SIH demo.
* Do not modify unrelated files.
* Do not scan the entire repository unless required.
* Use config.yaml for important thresholds and paths.
* Support CPU and CUDA where possible.
* Do not fake model results.
* Do not claim accuracy without evaluation.
* Evaluation labels must come from external ground truth, never from matching
  plates, fingerprints, similarity, candidate decisions, or ranking output.
* Do not download model weights automatically; fail clearly if weights missing.
* Preserve original plate crops.
* Run relevant tests after implementation.
* Fix errors caused by your changes.
* Keep final responses concise.

## Output Contract

Canonical ANPR observations follow
`contracts/events/plate-observation.schema.json` and are validated against it.
They are persisted to SQLite (and served via the API) and are also consumed by
the implemented Phase 2 replay, trajectory, and candidate workflows.

Each observation contains:

* event_id
* camera_id
* track_id
* timestamp
* plate_raw
* plate_normalized
* confidence
* status
* detector_confidence
* ocr_confidence
* quality_score
* best_frame_number
* plate_image_path
* model_version

Observations are persisted to SQLite via the repository layer and exposed
through the HTTP API.

## Development Policy

Implement one step at a time.

Do not start later steps automatically.

At the end of each task report only:

1. Files changed
2. What was implemented
3. Command to test
4. Any blocker
