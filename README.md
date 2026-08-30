# CitySight — Phase 1: ANPR Engine (SIH26127)

CitySight is a smart-city license-plate intelligence system. **Phase 1** is a
complete, self-contained Automatic Number Plate Recognition (ANPR) engine: it
ingests recorded traffic video, detects and tracks license plates, selects the
best frames, recognizes and fuses plate text across frames, scores confidence,
and persists canonical observations to SQLite. A read-only HTTP API and a
lightweight operator dashboard expose the data, and a plate watchlist raises
deduplicated alerts on accepted sightings.

> Scope: **only Phase 1** is implemented. Phase 2 is future work (see below).

## Architecture

Phase 1 is modular and dependency-injected end to end — every stage is a small
component with a clear interface, wired together by a single orchestrator
(`phase1_anpr/pipeline/anpr_pipeline.py`). This keeps the heavy models
(YOLO/PaddleOCR) swappable and lets the full pipeline run under test with fakes.

## Pipeline

```
Video
 → YOLO plate detection
 → ByteTrack tracking
 → quality scoring (best-frame selection)
 → perspective rectification
 → PaddleOCR
 → multi-frame fusion
 → Indian-plate normalization
 → confidence decision (accepted / review / abstained)
 → canonical observation event
 → SQLite persistence (+ filesystem evidence)
 → HTTP API
 → dashboard
 → watchlist alert
```

## Implemented features

- **Video ingestion** with configurable process-FPS sub-sampling.
- **YOLO plate detection** (Ultralytics) with CPU/CUDA support.
- **Real ByteTrack** tracking (Ultralytics BYTETracker) with per-track crop
  accumulation and stale/finalize handling.
- **Deterministic quality scoring** (sharpness/size/brightness/contrast) and
  top-K best-crop selection.
- **OpenCV perspective rectification** of plate crops.
- **PaddleOCR** recognition behind a small injectable backend interface.
- **Multi-frame fusion** of OCR results into ranked candidates.
- **Indian plate normalization/validation** (standard + BH-series, state codes;
  no ambiguous-character substitution).
- **Confidence scoring** with accepted / review / abstained decisions.
- **Canonical observation events** validated against a JSON Schema
  (`contracts/events/plate-observation.schema.json`).
- **SQLite persistence** with idempotent `event_id`, plus a swappable
  filesystem **evidence store** for plate crops (references stored in DB, not
  bytes).
- **Watchlist + deduplicated alerts**: accepted exact matches raise one alert
  per (watchlist entry, observation); review/abstained never auto-alert.
- **Stdlib HTTP API** (no framework) and a **vanilla-JS operator dashboard**
  with a live observation feed, exact plate search, watchlist management, and a
  recent-alerts panel.
- **End-to-end orchestrator + CLI demo** and a **server entry point**, both
  reading/writing the same SQLite database.

## Setup

Requires **Python 3.11+**.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### GitHub Codespaces

**Code → Codespaces → Create codespace on main**. The devcontainer provides
Python 3.11, Node.js, and the system libraries OpenCV/PaddleOCR need.
Dependencies are not installed automatically:

```bash
pip install -r requirements.txt
```

### YOLO weights

Trained license-plate YOLO weights are **not committed** (gitignored). Place
your weights at:

```
weights/plate_detector.pt
```

or point `detection.weights_path` in `phase1_anpr/config/config.yaml` at your
file. Weights are never downloaded automatically; if they are missing the demo
fails with a clear, actionable message.

## Running

All commands below use the venv Python. On Windows:

### Tests

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

### Video demo (end-to-end pipeline)

Processes a recorded video and persists observations/alerts to SQLite:

```bash
.\.venv\Scripts\python.exe -m phase1_anpr.demo --video inputs/videos/sample.mp4 --camera-id cam_01
```

### Dashboard server

Serves the API + dashboard against the same SQLite database the demo writes to:

```bash
.\.venv\Scripts\python.exe -m phase1_anpr.serve
```

Dashboard URL: **http://127.0.0.1:8000/dashboard** (Ctrl+C to stop).

### HTTP API endpoints

`GET /health`, `GET /observations`, `GET /observations/{event_id}`,
`GET /plates/{plate}/observations`, `GET /cameras/{camera_id}/observations`,
`GET /watchlist`, `POST /watchlist`, `DELETE /watchlist/{watchlist_id}`,
`GET /alerts`.

## Project structure

```
phase1_anpr/
  config/          config.yaml (thresholds + paths)
  video/           video reading / frame iteration
  detection/       YOLO license-plate detection
  tracking/        ByteTrack multi-object tracking
  quality/         quality scoring / best-frame selection
  preprocessing/   plate crop preparation for OCR
  rectification/   OpenCV perspective rectification
  ocr/             PaddleOCR text recognition
  pipeline/        track processing + end-to-end orchestrator
  normalization/   Indian plate normalization/validation
  confidence/      confidence scoring + decision
  observation/     canonical observation event builder
  persistence/     SQLite repositories + filesystem evidence store
  api/             stdlib HTTP API + dashboard
  models/          model versioning
  utils/           config loader, shared helpers
  demo.py          end-to-end video demo runner
  serve.py         API + dashboard server entry point
contracts/events/  plate-observation JSON Schema
inputs/videos/     input videos (not committed)
outputs/plates/    preserved plate crops + evidence
outputs/observations/ SQLite DB + observation output
weights/           YOLO weights (not committed)
tests/             pytest suite
run.py             config/video-iteration entry point
```

## Not committed

Local **YOLO weights** (`weights/`) and **videos** (`inputs/videos/`) are
intentionally gitignored — they are large and/or environment-specific. Add your
own locally as described above.

## Current limitations

- Confidence is a **heuristic**, not a calibrated probability; no accuracy
  metrics are claimed without a formal evaluation dataset.
- Normalization targets **Indian** plate formats.
- Single-camera, **recorded-video** processing (no live multi-camera streaming).
- Persistence uses **SQLite** and a **local filesystem** evidence store
  (interfaces are swappable for Postgres/MinIO/S3 later, not implemented).
- The API/dashboard are an unauthenticated **SIH demo** surface (no auth/RBAC).

## Future work (Phase 2 — not implemented)

Multi-camera trajectory correlation, traffic analytics, streaming ingestion,
notifications, authentication/RBAC, and a production datastore are **Phase 2**
and out of scope for this repository's current state.
