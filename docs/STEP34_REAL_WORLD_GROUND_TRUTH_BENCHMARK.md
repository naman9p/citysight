# Step 34 — Real-World Ground-Truth Accuracy Benchmark

## Status

Part A is implemented: strict annotation and prediction formats, fixed-baseline
capture, stage-correct metrics, machine-readable JSON output, Markdown output,
failure details, and focused tests.

Part B is intentionally pending manual labeling. **No real-world accuracy is
reported by this document.** The known Step 33 cross-camera example is not a
substitute for a labeled benchmark population.

## Existing infrastructure reused

Step 34 extends the existing `phase1_anpr.evaluation` package. It reuses its
one-to-one confidence-ordered IoU matcher, OCR exact/edit metrics, deterministic
character alignment, report CLI, and existing Phase 2 Recall@K, HitRate@K, and
MRR definitions. The production detector, OCR, tracker, fusion, normalization,
confidence policy, replay orchestration, and cross-camera matcher are not
changed.

The older Phase 1 manifest remains useful for extracted-image experiments and
controlled sweeps. Step 34 adds a video-aware adapter because the real-video
benchmark needs source/frame identity, empty reviewed frames, plate-passage
identity, manual replay-event linkage, and optional cross-camera identity.

## Fixed baseline

The capture command uses the checked-in Step 33 files without tuning:

- scenario: `phase2_city/config/replay.step33.yaml`;
- city topology: `phase2_city/config/city.step33.yaml`;
- Phase 1 config: `phase1_anpr/config/config.step33.yaml`;
- process FPS: 10;
- detector image size: 640;
- detector confidence threshold: 0.35;
- detector NMS IoU: 0.45;
- detector device: CUDA;
- PaddleOCR device: CPU;
- vehicle enrichment: disabled.

Capture uses an in-memory observation repository and no evidence store, so it
does not add rows or images to the normal Step 33 output stores. It hashes the
videos, weights, scenario, topology, and pipeline config in the prediction
artifact.

## Manual annotation format

The practical starting point is the preparation command below. It extracts
exactly the frames selected by the existing 10 FPS `VideoReader` and creates a
manifest in which every frame starts as `reviewed: false`:

```powershell
.\venv\Scripts\python.exe -m phase1_anpr.evaluation real-world-prepare `
  --scenario phase2_city\config\replay.step33.yaml `
  --city-config phase2_city\config\city.step33.yaml `
  --config phase1_anpr\config\config.step33.yaml `
  --output D:\citysight-step34\ground_truth.yaml `
  --frames-dir D:\citysight-step34\frames
```

The command refuses to overwrite an existing manifest. Alternatively, copy
`phase1_anpr/evaluation/step34_ground_truth.template.yaml` to a working
location and populate it manually. Neither route creates any labels.

Each source contains:

- `source_id`, `camera_id`, and `video_path`;
- `frames`: source-video frame numbers selected by the 10 FPS replay;
- `reviewed`: false until a human has checked the entire frame;
- `plates`: every visible plate box in a reviewed frame, including unreadable
  plates;
- `plate_instances`: one stable record per physical plate passage/track;
- exact visible `plate_text` and `text_evaluable`;
- optional manually known `vehicle_identity` across cameras;
- optional `cross_camera_cases` using the existing explicit
  `source_event_id` / `relevant_event_ids` convention; relevant IDs must be
  later sightings eligible for the forward candidate search;
- `observation_event_id` plus `observation_link_reviewed` after a human links
  the instance to the deterministic replay observation, or confirms that no
  observation exists;
- manually confirmed non-plate observation event IDs;
- `observations_reviewed`, set true only after every replay observation for the
  source is adjudicated.

Coordinates are full-resolution `[x1, y1, x2, y2]` pixels. Frame numbers are
zero-based source-video indices, matching `VideoReader`. A repeated plate uses
the same `instance_id` in all frames. A frame with no visible plate must still
be present as `reviewed: true` and `plates: []`; otherwise detector false
positives on negative frames cannot be measured. Unreviewed frames are excluded
and cannot contain plate labels.

`plate_text` preserves the human transcription. Exact-match calculations use
the project convention of uppercase alphanumeric canonicalization for both
ground truth and predictions; no grammar correction or guessed character is
applied. Set `text_evaluable: false` when an exact transcription cannot be made
reliably.

## Exact manual work required

For both `CAM01.mp4` and `CAM02.mp4`:

1. Review every frame selected by the Step 33 10 FPS reader. Mark the frame
   `reviewed: true`, including frames with no plates.
2. Draw a tight full-plate box around every visible plate and assign a stable
   per-passage `instance_id`.
3. For each instance, transcribe the complete visible plate only when readable;
   otherwise set `plate_text: null` and `text_evaluable: false`.
4. Add an optional `vehicle_identity` only when the same physical vehicle is
   manually known across cameras. Do not infer it from CitySight's output.
5. After baseline capture, inspect every replay observation and either link its
   `event_id` to the correct instance or list it under
   `non_plate_observation_event_ids`. Mark each instance's
   `observation_link_reviewed: true` even when its link is null, then set the
   source's `observations_reviewed: true` only when the entire observation list
   is adjudicated.
6. If cross-camera evaluation is justified, add explicit forward
   `cross_camera_cases`. Do not automatically treat every member of an identity
   group as relevant to every other member; CitySight retrieval is directional
   and searches later observations only.

Bounding-box and text labels must be created without viewing model predictions.
Event linkage/adjudication necessarily uses the captured event list, but it
must not change the already completed boxes or transcriptions.

## Capture predictions

After frames, boxes, and transcriptions are complete, run the unchanged Step 33
baseline:

```powershell
.\venv\Scripts\python.exe -m phase1_anpr.evaluation real-world-capture `
  --ground-truth D:\citysight-step34\ground_truth.yaml `
  --scenario phase2_city\config\replay.step33.yaml `
  --city-config phase2_city\config\city.step33.yaml `
  --config phase1_anpr\config\config.step33.yaml `
  --output outputs\annotated\step34_predictions.json
```

For cross-camera capture, all three fixed policy inputs are mandatory. Choose
and record them before examining retrieval results:

```powershell
  --candidate-window-seconds 120 `
  --candidate-max-results 100 `
  --candidate-maximum-speed-kph 60
```

The first two values above are workflow examples, not tuned Step 34 defaults.
The speed value matches the Step 33 physical-feasibility run. If there are not
at least two independently labeled source cases with rankings, cross-camera
percentages remain unavailable.

## Calculate the benchmark

After event linkage/adjudication is complete, rerun capture if the ground-truth
file gained cross-camera source events, then evaluate:

```powershell
.\venv\Scripts\python.exe -m phase1_anpr.evaluation real-world `
  --ground-truth D:\citysight-step34\ground_truth.yaml `
  --predictions outputs\annotated\step34_predictions.json `
  --primary-iou 0.5 `
  --diagnostic-iou 0.3 `
  --cross-camera-k 1,3,5 `
  --minimum-cross-camera-cases 2 `
  --output-dir outputs\annotated\step34_evaluation
```

Outputs:

- `step34_real_world_benchmark.json`: complete machine-readable metrics,
  outcomes, failures, input hashes, configuration/model provenance, and run
  metadata;
- `step34_real_world_benchmark.md`: concise human-readable results.

Generated predictions and reports remain under ignored `outputs/` paths. Raw
videos, databases, evidence images, and bulky generated artifacts must not be
committed.

## Metric definitions

### Plate detection

At the primary IoU threshold 0.5, predictions are processed in descending
confidence order and matched one-to-one to GT boxes.

- precision = TP / (TP + FP);
- recall = TP / (TP + FN);
- F1 = 2 × precision × recall / (precision + recall).

IoU 0.3 is retained only as a localization diagnostic. Duplicate predictions
against one GT box become false positives.

### OCR

OCR exact-match accuracy is:

```text
canonical exact OCR matches
-----------------------------------------------
detector-matched, text-evaluable, linked plates
```

Detector misses are reported separately and never counted as OCR errors.
Character accuracy and deterministic substitution confusions are secondary
diagnostics.

### End-to-end recognition

End-to-end exact recognition is:

```text
detected GT instances with canonical-exact text
------------------------------------------------
all text-evaluable GT plate instances
```

The numerator and denominator are always printed. The metric is unavailable
until every evaluable instance's observation linkage has been manually
reviewed.

### Confidence/status behavior

The report separately gives accepted, review, abstained, and no-observation
counts; accepted-only precision; accepted coverage; incorrect accepted
observations; and correct observations sent to review/abstained. Acceptance
rate is never presented as OCR or end-to-end accuracy.

### False positives

Unmatched detector predictions and manually confirmed non-plate observations
are listed with their actual output text. No text such as `PATEL` or vehicle
slogans is hardcoded as a special case. Unadjudicated observations remain
explicitly visible and prevent a claim of complete observation review.

### Cross-camera retrieval

When sufficient manual identities and captured rankings exist, the report uses
the established CitySight definitions:

- Recall@K: relevant events retrieved in the first K divided by all relevant
  events;
- HitRate@K: source cases with at least one relevant event in the first K
  divided by source cases;
- MRR: mean reciprocal rank of the first relevant event.

It also reports positive identity decisions and topology/travel rejection
counts. A single positive example never produces aggregate cross-camera
percentages.
