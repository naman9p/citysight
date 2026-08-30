# CitySight — Phase 1: ANPR Engine

Automatic Number Plate Recognition (ANPR) engine. Reads a video, detects and
tracks license plates, picks the best frame per plate, runs OCR, fuses results
across frames, and writes confidence-scored observations as JSON Lines.

> Scope: **only Phase 1** is being built right now. See `CLAUDE.md`.

## Pipeline

```
Video
 → License Plate Detection
 → Tracking
 → Best Frame Selection
 → OCR
 → Multi-frame Fusion
 → Confidence
 → JSONL Observation
```

## Project structure

```
phase1_anpr/
  config/          config.yaml (thresholds + paths)
  video/           video reading / frame iteration
  detection/       license plate detection (YOLO)
  tracking/        multi-object tracking (ByteTrack)
  quality/         best-frame selection / quality scoring
  preprocessing/   plate crop preparation for OCR
  ocr/             text recognition (PaddleOCR)
  normalization/   plate string normalization
  models/          model loading / versioning
  pipeline/        end-to-end orchestration
  utils/           shared helpers (config loader, etc.)
inputs/videos/     input videos (not tracked)
outputs/plates/    preserved plate crops
outputs/annotated/ annotated frames/video (debug)
outputs/observations/ observations.jsonl output
weights/           model weights (not tracked)
tests/             pytest tests
run.py             entry point
config.yaml        -> phase1_anpr/config/config.yaml
```

## Requirements

- Python 3.11+
- See `requirements.txt`

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## GitHub Codespaces setup

On GitHub: **Code → Codespaces → Create codespace on main**. The devcontainer
(`.devcontainer/devcontainer.json`) provides Python 3.11, Node.js (for the
Claude Code CLI), and the system libraries OpenCV/PaddleOCR need. Project
dependencies are not installed automatically — install them when needed:

```bash
pip install -r requirements.txt
```

## Usage

Step 1 only loads and validates configuration:

```bash
python run.py --config phase1_anpr/config/config.yaml
```

## Tests

```bash
pytest -q
```

## Status

Step 1 complete: project structure and configuration. Detection, tracking,
OCR, fusion, and output are not implemented yet.
