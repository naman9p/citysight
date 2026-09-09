# CitySight — ANPR and Cross-Camera Candidate Intelligence (SIH26127)

CitySight is a smart-city license-plate intelligence system. **Phase 1** is a
complete, self-contained Automatic Number Plate Recognition (ANPR) engine: it
ingests recorded traffic video, detects and tracks license plates, selects the
best frames, recognizes and fuses plate text across frames, scores confidence,
and persists canonical observations to SQLite. A read-only HTTP API and a
lightweight operator dashboard expose the data, and a plate watchlist raises
deduplicated alerts on accepted sightings.

**Phase 2 is implemented through the reproducible cross-camera candidate
evaluation workflow (Steps 17–31).** It adds city camera topology,
multi-camera recorded-video replay, exact-plate trajectories, optional
whole-vehicle enrichment, explainable bounded candidate hypotheses, and
external-ground-truth retrieval evaluation. Candidate hypotheses are not
confirmed vehicle identities.

## Architecture

Phase 1 is modular and dependency-injected end to end — every stage is a small
component with a clear interface, wired together by a single orchestrator
(`phase1_anpr/pipeline/anpr_pipeline.py`). This keeps the heavy models
(YOLO/PaddleOCR) swappable and lets the full pipeline run under test with fakes.

Phase 2 consumes completed replay observations in memory. Pairwise matching,
bounded collection, reporting, and evaluation remain separate layers; exact
plate trajectories are not modified by candidate hypotheses.

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
 → multi-camera replay + optional vehicle fingerprint enrichment
 → explainable bounded cross-camera candidate hypotheses
 → explicit-ground-truth Recall@K / HitRate@K / MRR evaluation
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
- **Directed city camera topology** and deterministic recorded multi-camera
  replay with timezone-aware source start times.
- **Exact-plate trajectory reconstruction** kept independent from fingerprint
  candidate hypotheses.
- **Optional whole-vehicle enrichment** with colour/class evidence and no
  automatic model download.
- **Bounded deterministic cross-camera candidate collection** with explainable
  fingerprint, topology, and travel evidence.
- **Read-only candidate reporting** plus YAML-backed evaluation using candidate
  Recall@K, HitRate@K, MRR, and truncation diagnostics.

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

### Phase 2 multi-camera replay

The example references local videos that are not committed. Supply the clips or
update their paths before running it:

```bash
.\.venv\Scripts\python.exe -m phase2_city.replay --scenario phase2_city/config/replay.example.yaml --city-config phase2_city/config/city.yaml --config phase1_anpr/config/config.yaml
```

Candidate reporting and externally labeled evaluation are documented in the
[Phase 2 demo and evaluation runbook](docs/phase2-demo-evaluation.md), including
all required policy arguments and the ground-truth protocol.

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
`GET /alerts`, `POST /v1/trajectories`, `GET /v1/cameras`,
`GET /v1/cameras/{id}`, `GET /v1/cameras/{id}/links`, and `GET /v1/links`.

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
phase2_city/        city topology, replay, trajectories, vehicle enrichment,
                    candidate matching/collection/reporting/evaluation
docs/               Phase 2 demo and evaluation runbook
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
- Processing is **recorded-video** based; there is no live multi-camera
  streaming ingestion.
- Persistence uses **SQLite** and a **local filesystem** evidence store
  (interfaces are swappable for Postgres/MinIO/S3 later, not implemented).
- The API/dashboard are an unauthenticated **SIH demo** surface (no auth/RBAC).
- Cross-camera matches are bounded candidate hypotheses, not global vehicle
  identities or inferred route proof.
- Candidate fingerprints/results remain transient in the completed replay
  result; no candidate persistence schema is implemented.
- Candidate-retrieval metrics require genuine external labels and do not
  represent universal ANPR/OCR accuracy.

## Beyond the current prototype

Live streaming ingestion, global identity resolution, learned vehicle ReID,
traffic analytics, notifications, authentication/RBAC, and a production
datastore remain out of scope.
