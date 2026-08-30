# CitySight — Claude Instructions

## Current Scope

We are currently building only Phase 1: ANPR Engine.

Do not implement Phase 2 or later features unless explicitly asked.

## Phase 1 Pipeline

Video
→ License Plate Detection
→ Tracking
→ Best Frame Selection
→ OCR
→ Multi-frame Fusion
→ Confidence
→ JSONL Observation

## Tech Stack

* Python 3.11+
* OpenCV
* Ultralytics YOLO
* ByteTrack
* PaddleOCR
* NumPy
* Pydantic
* PyYAML
* pytest

## Important Rules

* Keep code simple and readable.
* Prefer modular code over unnecessary abstraction.
* Do not build frontend.
* Do not build FastAPI.
* Do not use PostgreSQL.
* Do not use Redis.
* Do not use Kafka.
* Do not use Kubernetes.
* Do not use cloud services.
* Do not modify unrelated files.
* Do not scan the entire repository unless required.
* Use config.yaml for important thresholds and paths.
* Support CPU and CUDA where possible.
* Do not fake model results.
* Do not claim accuracy without evaluation.
* Preserve original plate crops.
* Run relevant tests after implementation.
* Fix errors caused by your changes.
* Keep final responses concise.

## Output Contract

Final ANPR observations will later be consumed by Phase 2.

Each observation should contain approximately:

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

Observations should be written as JSON Lines.

## Development Policy

Implement one step at a time.

Do not start later steps automatically.

At the end of each task report only:

1. Files changed
2. What was implemented
3. Command to test
4. Any blocker
