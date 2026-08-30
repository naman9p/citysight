# CitySight — Claude Instructions

## Current Scope

Phase 1 (ANPR Engine) is complete: Steps 1–16, video → detection → tracking →
quality → rectification → OCR → fusion → normalization → confidence →
canonical observation → SQLite persistence → HTTP API → dashboard → watchlist
alerts.

Do not implement Phase 2 or later features unless explicitly asked.

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
* Do not download model weights automatically; fail clearly if weights missing.
* Preserve original plate crops.
* Run relevant tests after implementation.
* Fix errors caused by your changes.
* Keep final responses concise.

## Output Contract

Canonical ANPR observations follow
`contracts/events/plate-observation.schema.json` and are validated against it.
They are persisted to SQLite (and served via the API) and will later be
consumed by Phase 2.

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
