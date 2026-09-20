# CitySight — Current System Architecture

> **Status:** Living architecture document
> **Project:** CitySight — SIH26127
> **Purpose:** This file is the architectural source of truth for the implementation that actually exists in the repository.
> **Maintenance rule:** If a roadmap step changes modules, data flow, interfaces, persistence, APIs, ML components, evaluation, or deployment assumptions, update this file in the **same logical commit** as that step.

---

## 1. How to use this document

This document describes the architecture CitySight **currently uses**, not a hypothetical enterprise architecture.

Rules:

1. The Git repository is the final source of truth.
2. Do not add planned components here as if they already exist.
3. Clearly mark future work as **Planned / Not Implemented**.
4. Every architectural change must update:
   - the relevant component section;
   - the high-level data-flow diagram if needed;
   - the roadmap/status table;
   - the architecture decision/change log at the end.
5. Do not silently replace existing architectural choices such as SQLite or the stdlib HTTP API.
6. Prefer the smallest architecture change that satisfies the current roadmap step.
7. Preserve backward compatibility with the Phase 1 ANPR pipeline unless a reviewed change explicitly requires otherwise.
8. Raw external datasets and videos stay outside Git unless explicitly approved.

---

## 2. Project objective

CitySight is an explainable multi-camera ANPR and vehicle-correlation system.

It combines:

- YOLO-based number-plate detection;
- BYTETrack temporal tracking;
- track-level image-quality selection;
- OpenCV plate rectification;
- PaddleOCR recognition;
- multi-frame OCR fusion;
- Indian number-plate normalization;
- confidence-based accepted / review / abstained decisions;
- SQLite persistence;
- local filesystem evidence storage;
- a lightweight stdlib HTTP API;
- a static HTML/CSS/JavaScript dashboard;
- watchlist matching and alert deduplication;
- camera topology;
- recorded-video event timestamps;
- multi-camera replay;
- trajectory reconstruction;
- whole-vehicle detection;
- vehicle attributes;
- deterministic vehicle fingerprints;
- physical travel-feasibility reasoning;
- bounded cross-camera candidate retrieval;
- candidate reporting;
- externally supplied ground truth;
- Recall@K, HitRate@K and MRR evaluation.

The current architecture favors **correctness, explainability, determinism, testability, and SIH demonstrability** over unnecessary infrastructure complexity.

---

## 3. Current architecture at a glance

```mermaid
flowchart TD
    A[Recorded Video / Camera] --> B[Video Reader]
    B --> C[YOLO Plate Detector]
    C --> D[BYTETrack Plate Tracking]
    D --> E[Track-level Candidate Frames]
    E --> F[Quality Scoring / Top-K Crops]
    F --> G[Plate Rectification]
    G --> H[PaddleOCR]
    H --> I[Multi-frame OCR Fusion]
    I --> J[Indian Plate Normalization]
    J --> K[Confidence / Decision Engine]
    K --> L[PlateObservation]

    L --> M[(SQLite Observation Repository)]
    L --> N[Local Evidence Store]
    L --> O[Watchlist / Alert Logic]
    M --> P[stdlib HTTP API]
    P --> Q[Static Dashboard]

    L --> R[Multi-camera Replay / City Layer]
    R --> S[Camera Graph + Recorded Event Time]
    S --> T[Trajectory Reconstruction]

    E --> U[Optional Whole-Vehicle Enrichment]
    U --> V[Vehicle Detection]
    V --> W[Plate-to-Vehicle Association]
    W --> X[Vehicle Attribute Extraction]
    X --> Y[VehicleFingerprint]

    L --> Z[Cross-camera Candidate Input]
    Y --> Z
    S --> Z
    Z --> AA[Pairwise Candidate Matcher]
    AA --> AB[Bounded Candidate Collector]
    AB --> AC[Deterministic Ranked Candidates]
    AC --> AD[Candidate Report]

    AC --> AE[Candidate Evaluation]
    AF[External Ground Truth] --> AE
    AE --> AG[Recall@K / HitRate@K / MRR / Truncation]
```

---

## 4. Repository-level architectural layers

CitySight is a modular Python application rather than a distributed microservice platform.

### 4.1 Phase 1 — Core ANPR layer

Primary responsibility:

> Turn a video stream into reliable structured number-plate observations.

Conceptual flow:

```text
video
  ↓
plate detection
  ↓
plate tracking
  ↓
best-frame selection
  ↓
rectification
  ↓
OCR
  ↓
multi-frame fusion
  ↓
normalization
  ↓
confidence decision
  ↓
PlateObservation
  ↓
persistence / watchlist / API / dashboard
```

### 4.2 Phase 2 — Multi-camera intelligence layer

Primary responsibility:

> Correlate observations across cameras without making unjustified identity claims.

Conceptual flow:

```text
multiple PlateObservations
        +
camera topology
        +
recorded event time
        +
optional vehicle fingerprints
        ↓
pairwise compatibility / physical feasibility
        ↓
bounded candidate collection
        ↓
deterministic ranking
        ↓
human-readable report
        ↓
external-ground-truth evaluation
```

---

## 5. Core Phase 1 architecture

### 5.1 Video ingestion

The video reader wraps OpenCV video decoding.

Important metadata includes:

- frame index;
- image/frame;
- FPS;
- timing information.

Recorded-event time is distinct from processing time.

For recorded video, a configured video start time or per-frame timestamp can be used to derive the true observation time.

---

### 5.2 Number-plate detection

Technology:

- Ultralytics YOLO;
- dedicated license-plate detector;
- plate detector class `0` represents `license_plate`.

Responsibilities:

- locate plate bounding boxes;
- expose detector confidence;
- produce plate detections for tracking.

Important architectural rule:

> Plate-detector class IDs must never be interpreted as vehicle classes.

Whole-vehicle detection is a separate Phase 2 capability.

---

### 5.3 Plate tracking

Technology:

- BYTETrack through the Ultralytics integration.

Purpose:

- associate repeated plate detections across consecutive frames;
- produce stable per-source track IDs;
- avoid performing independent OCR decisions on every frame.

A track represents one local temporal object hypothesis, **not a globally unique vehicle identity**.

---

### 5.4 Quality scoring

Instead of OCRing every detected frame, CitySight ranks plate crops.

Signals include implementation-defined measures such as:

- sharpness;
- crop/plate size;
- brightness;
- contrast.

Current architecture keeps a small top-K set of useful crops for OCR.

This reduces:

- duplicate OCR work;
- noisy recognition;
- poor-frame dependence.

---

### 5.5 Plate rectification

Technology:

- OpenCV perspective transformation.

Purpose:

- improve angled/skewed plate crops before OCR.

Failure policy:

- rectification is opportunistic;
- if a reliable transform cannot be produced, retain/use the original crop;
- preprocessing failure must not unnecessarily destroy usable evidence.

---

### 5.6 OCR

Technology:

- PaddleOCR 3.7;
- recognition-focused use on already detected plate crops.

Architectural rules:

- reuse the OCR predictor;
- do not instantiate PaddleOCR for every crop;
- tests should not require heavyweight OCR initialization when a fake can be injected.

---

### 5.7 Multi-frame OCR fusion

Multiple high-quality crops from the same track provide temporal redundancy.

Conceptual input:

```text
crop 1 → OCR text + confidence
crop 2 → OCR text + confidence
crop 3 → OCR text + confidence
```

The fusion layer combines evidence rather than blindly trusting one frame.

Exact coefficients and formulas must always be read from the implementation and must not be invented in documentation.

---

### 5.8 Indian plate normalization

Responsibilities include:

- cleanup;
- uppercase canonicalization;
- removal of unwanted characters;
- validation of supported Indian registration formats;
- BH-series support;
- state-code handling where implemented.

Safety rule:

> Do not blindly convert `O ↔ 0` or `I ↔ 1`.

CitySight prefers retained uncertainty over creating a plausible but false plate.

---

### 5.9 Confidence and decision engine

A successful OCR result does not automatically become a trusted observation.

Current decision states include:

- `accepted`;
- `review`;
- `abstained`.

Signals can include:

- OCR confidence;
- detector confidence;
- crop quality;
- multi-frame agreement;
- normalization/format validity.

Architectural principle:

> The system is allowed to say that evidence is insufficient.

---

### 5.10 Observation domain model

The central Phase 1 domain output is a structured `PlateObservation`.

Conceptually it includes information such as:

- event ID;
- camera ID;
- event timestamp;
- normalized plate result;
- confidence;
- decision/status;
- track metadata;
- evidence references.

The exact schema in the repository is authoritative.

The JSON event contract lives under the repository contracts area.

---

## 6. Persistence architecture

### 6.1 Structured data

Current implementation:

- SQLite;
- Python `sqlite3`.

Repository abstractions isolate SQL/storage concerns from domain logic.

### 6.2 Evidence data

Current implementation:

- local filesystem evidence store.

Conceptually:

```text
PlateObservation
   ├── structured metadata → SQLite
   └── evidence images     → local filesystem
```

### 6.3 Storage non-goals

Do not replace the current SIH storage architecture with:

- PostgreSQL;
- MongoDB;
- Kafka;
- ClickHouse;
- distributed object/event infrastructure;
- cloud databases;

unless a future reviewed requirement justifies the migration.

---

## 7. API architecture

Current implementation:

- Python standard-library `http.server`.

FastAPI is intentionally not part of the current architecture.

Existing API areas include observation retrieval and later read-only Phase 2 surfaces such as trajectory/topology where wired.

API principles:

- minimal;
- backward compatible;
- tested;
- read-only where mutation is unnecessary;
- return explicit errors rather than silently guessing.

The exact repository routes and response schemas are authoritative.

---

## 8. Dashboard architecture

Technology:

- static HTML;
- CSS;
- JavaScript.

Purpose:

- demonstrate CitySight results;
- provide a lightweight operator/demo surface;
- avoid spending SIH scope on a large frontend framework.

The dashboard consumes the existing HTTP API.

---

## 9. Watchlist architecture

Accepted normalized plates can be checked against a persisted watchlist.

Architecture:

```text
PlateObservation
     ↓
normalized accepted plate
     ↓
watchlist lookup
     ↓
match?
     ↓
alert with deduplication
```

Alert deduplication avoids repeated alerts from the same vehicle/track visibility window.

---

## 10. Phase 2 city and topology architecture

### 10.1 Camera domain model

Important concepts:

- `Camera`;
- `CameraLink`;
- `CityCameraGraph`;
- city repository/config loader.

A directed camera link can include:

- source camera;
- destination camera;
- distance in metres;
- road name;
- descriptive travel direction.

A link represents known connectivity/context. It does **not** prove that a vehicle actually travelled along that link.

---

### 10.2 Recorded-video timestamps

Processing time and event time are separate concepts.

Where supported:

```text
event_time = video_start_time + frame_position / fps
```

or a supplied per-frame timestamp is used.

Cross-camera temporal reasoning must use meaningful event time.

Absolute timestamps must never be fabricated when the dataset only supports relative synchronized time.

---

## 11. Multi-camera replay architecture

Recorded videos can simulate independent cameras.

A replay scenario associates:

- camera ID;
- source ID;
- video/input;
- timing metadata.

Source-safe event identity incorporates sufficient source context so identical local `track_id` values from different videos/cameras do not collide.

Conceptually:

```text
camera_id + source_id + track_id
            ↓
        event identity
```

Exact event-ID construction remains repository-defined.

---

## 12. Trajectory reconstruction

Trajectory reconstruction is read-only.

Conceptual behavior:

1. retrieve relevant observations;
2. apply allowed status filtering;
3. order by timestamp;
4. collapse unnecessary duplicates;
5. enrich with camera information;
6. build transitions;
7. calculate elapsed travel time;
8. attach direct-link metadata when available.

Important distinction:

> Trajectory reconstruction from plate observations is not equivalent to global vehicle re-identification.

---

## 13. Whole-vehicle enrichment

Whole-vehicle enrichment is optional and must not break the core ANPR path.

Conceptual flow:

```text
tracked plate
    ↓
select suitable frame
    ↓
whole-vehicle detector
    ↓
associate plate box with vehicle box
    ↓
safe vehicle crop
    ↓
vehicle attributes
    ↓
VehicleFingerprint
```

Heavy inference should be bounded and performed only when needed.

---

## 14. Whole-vehicle detection

Technology:

- Ultralytics detector using relevant COCO vehicle classes.

Relevant coarse classes include:

- car;
- motorcycle;
- bus;
- truck.

Model-loading rule:

- lazy-load when possible;
- do not make optional enrichment mandatory for normal ANPR execution;
- tests should use injected fakes rather than model downloads.

---

## 15. Plate-to-vehicle association

Purpose:

> Associate a detected plate bounding box with the correct whole-vehicle bounding box in the same frame.

The association is deterministic.

Safety requirements:

- clip coordinates to image boundaries;
- reject invalid crops;
- handle ambiguous spatial containment deterministically.

---

## 16. Vehicle attributes

Current practical attributes:

- coarse vehicle class;
- vehicle colour.

Colour is derived from the vehicle image using deterministic image processing according to the repository implementation.

Do not invent undocumented HSV thresholds or make/model classifications.

---

## 17. Vehicle fingerprint architecture

`VehicleFingerprint` is an explainable representation of available identity evidence.

Current core signals:

- normalized plate;
- vehicle colour;
- vehicle class.

The current deterministic foundation assigns stronger importance to plate evidence than coarse colour/class evidence.

Important semantics:

```text
missing evidence ≠ contradictory evidence
```

Example:

```text
white vs unknown  → missing evidence
white vs black    → contradictory evidence
```

A fingerprint is evidence for correlation. It is **not a guaranteed identity**.

---

## 18. Pairwise cross-camera candidate matching

The pairwise matcher evaluates one source observation against one possible later observation.

Inputs can include:

- `PlateObservation`;
- optional `VehicleFingerprint`;
- source/candidate camera IDs;
- event timestamps;
- directed camera topology.

It evaluates evidence such as:

- fingerprint compatibility;
- plate compatibility;
- temporal ordering;
- camera relationship;
- physical travel feasibility.

The matcher is:

- deterministic;
- bounded;
- explainable;
- read-only.

It produces a candidate judgement/result, not a declaration of global identity.

---

## 19. Physical travel-feasibility architecture

When a directed camera link and meaningful timestamps are available, CitySight can evaluate whether travel is physically plausible.

Conceptually:

```text
distance between cameras
        +
elapsed event time
        ↓
required movement / speed implication
        ↓
feasible / impossible / insufficient evidence
```

Exact policies, speed limits, decision labels and formulas must be read from the implementation.

A missing topology link must not automatically be interpreted as proof of impossible travel unless the implementation/policy explicitly defines that behavior.

---

## 20. Bounded candidate collection

Step 28 extends pairwise matching into candidate retrieval.

Given one source event:

```text
source
  ↓
all replay observations
  ↓
structural eligibility filter
  ↓
later timestamp only
  ↓
different camera
  ↓
configured inclusive time window
  ↓
bounded candidate selection
  ↓
pairwise matcher
  ↓
deterministic ranking
```

Key properties:

- read-only;
- bounded;
- deterministic;
- stable tie-breaking;
- explicit truncation;
- expensive matching is performed only after structural filtering.

The candidate cap prevents uncontrolled all-pairs growth.

---

## 21. Candidate report architecture

Step 29 provides a human-readable surface over candidate collection.

The report presents information such as:

- source event;
- candidate hypotheses;
- ranking;
- decision;
- evidence/reasons;
- truncation information.

Terminology rule:

> Reports must use language such as “candidate”, “plausible”, “strong/weak evidence”, etc., rather than “confirmed same vehicle” unless an independent truth source warrants that claim.

---

## 22. Evaluation architecture

### 22.1 Ground-truth principle

Ground truth must be **independent of CitySight predictions**.

Never create evaluation labels from:

- equal OCR/plate output;
- fingerprint similarity;
- candidate score;
- candidate rank;
- matcher decision;
- any prediction produced by the system being evaluated.

### 22.2 Candidate evaluation

Step 30 evaluates ranked candidate retrieval.

Current metrics include:

- Candidate Recall@K;
- HitRate@K;
- Mean Reciprocal Rank (MRR);
- truncation statistics.

Conceptually:

```text
predicted ranked candidates
              +
independent relevant-event labels
              ↓
          evaluator
              ↓
Recall@K / HitRate@K / MRR / truncation
```

### 22.3 Ground-truth dataset and runner

Step 31 adds a strict file-backed ground-truth workflow and evaluation runner.

Correct modules:

```text
phase2_city/candidate_ground_truth.py
phase2_city/candidate_evaluation_runner.py
```

Do not use obsolete/incorrect names such as:

```text
evaluation_dataset.py
evaluation_runner.py
```

The loader/runner architecture exists so evaluation is:

- explicit;
- reproducible;
- validation-driven;
- deterministic.

---

## 23. Step 32 — demonstration and evaluation runbook

Step 32 documents how to execute the Phase 2 replay/candidate/evaluation workflow reproducibly.

Current documentation includes:

```text
docs/phase2-demo-evaluation.md
```

This is an operational/runbook layer, not a new matching algorithm.

---

## 24. Step 33 — external evaluation architecture

### Current status

**Phase 2 is active. Step 33 is the intended final Phase 2 step.**

Step 33 goal:

> Produce the first credible externally annotated cross-camera candidate-retrieval baseline using the already-built Steps 27–32 architecture.

No new ReID/matching algorithm should be added merely to obtain a better baseline.

### External dataset compatibility decision

RoundaboutHD was inspected and validated as a genuine synchronized
multi-camera vehicle dataset. It is **not** the selected primary end-to-end
Step 33 evaluation dataset because it is incompatible with CitySight's current
ANPR observation-generation boundary.

Local dataset location during development:

```text
N:\RoundaboutHD\RoundaboutHD
```

This path is **external to the Git repository** and must never be hardcoded into production code.

Expected external structure includes:

```text
RoundaboutHD/
├── imagesc001/
├── imagesc002/
├── imagesc003/
├── imagesc004/
├── RoundaboutHD_Reid_subset/
├── layout.jpg
├── Multi_CAM_Ground_Turth.txt
├── README.md
└── Vehicle_statistic.xlsx
```

The misspelled annotation filename above is the actual filename in the
downloaded dataset and must not be silently renamed in provenance records.

### Validated timing semantics

The official documentation describes MCVT frame IDs as starting at zero, while
the downloaded `Multi_CAM_Ground_Turth.txt` annotations use frame numbers up to
9000 for videos containing 9000 OpenCV frames indexed from 0 through 8999.
Temporary bounding-box overlays were visually checked near the beginning,
middle and end of all four camera videos. The confirmed mapping is:

```text
video_frame_index = annotation_frame_num - 1
relative_seconds = (annotation_frame_num - 1) / 15.0
```

No absolute recording timestamp is known. A synthetic timezone-aware UTC epoch
may be used only as a deterministic encoding of synchronized relative dataset
time; it must never be described as the real recording date or time.

### Ambiguous ground-truth identities

Although the four videos are synchronized and their fields of view are
non-overlapping, three global identities have overlapping annotation intervals
across different cameras:

```text
553
644
677
```

These IDs would require explicit exclusion as ambiguous/anomalous labels from
any evaluation using this dataset. Their timestamps must not be shifted and
their identities must not be split, repaired or reinterpreted to manufacture a
valid transition.

### CitySight compatibility preflight

An unchanged CitySight ANPR preflight processed 150 frames from each of the
four cameras (600 frames total). It produced:

```text
plate detections: 0
plate tracks:     0
observations:     0
```

With no replay observations, independently verified RoundaboutHD identity to
CitySight event bindings cannot be created and the existing candidate evaluator
cannot produce a credible end-to-end retrieval baseline.

Therefore RoundaboutHD is rejected as the **primary end-to-end Step 33
evaluation dataset**. This is a dataset/pipeline compatibility limitation, not
a claim that RoundaboutHD is defective. CitySight's detector, thresholds, OCR,
normalizer, matcher, candidate collector and evaluator will not be modified
merely to make this benchmark produce observations. An oracle or component
benchmark must not be substituted for the failed end-to-end evaluation.

Architecture rule:

```text
External dataset
      ↓
dataset semantics / ground-truth adapter
      ↓
generic CitySight evaluation structures
      ↓
existing candidate collector + evaluator
      ↓
reproducible baseline metrics
```

Raw videos/images must remain outside Git.

RoundaboutHD may remain available externally for possible future vehicle-only,
ReID or component research, but none of those uses constitutes the primary
Step 33 baseline. Step 33 remains in progress and now requires a replacement
external dataset compatible with the existing Indian ANPR
observation-generation pipeline.

### Step 33 validation requirements

Before relying on any replacement Step 33 dataset:

- verify annotation schema;
- verify camera identity semantics;
- verify global/cross-camera vehicle identity semantics;
- verify frame/timing synchronization;
- use relative time only when justified;
- do not invent absolute timestamps;
- do not invent physical camera distances or road directions;
- keep dataset-specific parsing separated from the generic evaluator.

Dataset compatibility must be established with a bounded ANPR preflight before
building a full adapter or inspecting final retrieval metrics.

---

## 25. Data architecture

### Main logical data objects

```text
Frame
  ↓
PlateDetection
  ↓
Tracked plate / track ID
  ↓
Quality-selected crop
  ↓
OCR evidence
  ↓
PlateObservation
```

Optional vehicle branch:

```text
Frame
  ↓
VehicleDetection
  ↓
Plate ↔ Vehicle Association
  ↓
VehicleAttributes
  ↓
VehicleFingerprint
```

Cross-camera branch:

```text
PlateObservation + optional VehicleFingerprint
                    +
              CameraGraph
                    +
                Event Time
                    ↓
          Candidate Match Result
                    ↓
          Ranked Candidate Result
```

Evaluation branch:

```text
Ranked Candidate Result
          +
Independent Ground Truth
          ↓
Evaluation Case / Summary
```

---

## 26. Identity semantics

CitySight deliberately distinguishes these concepts:

| Concept | Meaning |
|---|---|
| Detection | Object found in one frame |
| Track | Same local detection sequence across frames |
| Track ID | Local tracker identifier |
| Observation | Structured ANPR event derived from a track |
| Sighting | Observation considered in trajectory context |
| Trajectory | Ordered set of sightings |
| VehicleFingerprint | Partial identifying/appearance evidence |
| Candidate match | Potential relationship between two observations |
| Ground-truth identity | Externally established same-vehicle relationship |
| Global vehicle identity | Stronger persistent identity claim; not currently inferred by CitySight |

Never treat `track_id` as a global identity.

Never treat a high candidate score as external truth.

---

## 27. Determinism

For non-neural decision logic, same inputs/configuration should produce the same outputs.

Determinism is especially required for:

- candidate eligibility;
- candidate ranking;
- tie-breaking;
- trajectory ordering;
- evaluation metrics;
- serialization/output where possible;
- tests.

This makes the system reproducible and judge/debug-friendly.

---

## 28. Explainability

Candidate results should expose reasons rather than only a score.

Preferred:

```text
candidate score/result
+ plate evidence compatible
+ colour compatible
+ class compatible
+ timing feasible
- one field missing
```

Avoid unexplained:

```text
same vehicle = 0.81
```

Explainability is a core CitySight architectural property.

---

## 29. Failure and uncertainty policy

CitySight fails safely.

Examples:

- rectification failure → keep/use original crop;
- missing vehicle enrichment → core ANPR can continue;
- missing fingerprint field → represent missing evidence;
- insufficient OCR evidence → review/abstain;
- unknown topology → represent unknown rather than inventing a route;
- unavailable timestamp → do not fabricate synchronization;
- uncertain ground truth → exclude/adjudicate rather than guess.

---

## 30. Performance architecture

Avoid obvious unnecessary work.

Preferred:

- detect then track;
- OCR only top-quality crops;
- reuse loaded ML models;
- enrich a track only when needed;
- structurally filter before expensive candidate scoring;
- cap candidate search;
- keep unit tests independent of real model inference and large datasets.

Avoid:

- OCR on every frame;
- YOLO/Paddle model initialization for every item;
- all-pairs cross-camera comparison without bounds;
- full 8 GB dataset requirements in normal unit tests.

---

## 31. Testing architecture

Testing is part of the architecture, not an afterthought.

Roadmap-step workflow:

```text
verify repository
      ↓
design one step
      ↓
implement narrowly
      ↓
focused tests
      ↓
regression tests
      ↓
full pytest suite
      ↓
git diff --check
      ↓
review
      ↓
retest
      ↓
explicit staging
      ↓
commit
      ↓
push
      ↓
verify clean main == origin/main
      ↓
STOP
```

Test principles:

- use deterministic synthetic fixtures;
- inject fakes for heavyweight ML;
- no network downloads in normal unit tests;
- external dataset availability must not be required for the ordinary full suite;
- regression protection is mandatory.

### Phase 1 accuracy evaluation

The optional `phase1_anpr.evaluation` package provides a strict external-label
boundary for ANPR experiments:

```text
versioned external manifest
        ↓
integrity / duplicate / sequence-leakage audit
        ↓
detector operating-point sweep + false-negative analysis
        +
OCR preprocessing ablation + track-fusion comparison
        ↓
linked end-to-end recognition when plate_id/track_id ground truth exists
        ↓
stage-separated JSON + Markdown metrics with model/manifest hashes
```

Detector metrics remain separate from crop OCR, fused track recognition, and
accepted-set accuracy/coverage. The evaluator writes generated artifacts only
under ignored output paths by default and never changes production config.
PaddleOCR evaluation requires an explicit existing local model directory; model
weights are not downloaded automatically.

Step 34 adds a video-aware adapter in
`phase1_anpr/evaluation/real_world.py` without replacing that framework. Its
strict YAML ground truth represents reviewed source frames, full-resolution
plate boxes, physical plate-passage instances, exact-text evaluability,
optional manually known cross-camera identity, and manually adjudicated replay
event links. Unreviewed frames are excluded rather than silently treated as
negative frames.

Step 34 Part B adds a local annotation boundary in
`phase1_anpr/evaluation/annotation.py` with a plain
`phase1_anpr/evaluation/annotation_ui.html` browser client. The loopback-only
stdlib HTTP server exposes extracted manifest frames and manual label fields;
it does not load predictions or production inference components. Saves are
revision checked, strictly validated, backed up by default, and atomically
replace the existing YAML while preserving unrelated labels and replay-event
adjudication fields.

```text
manual Step 34 video labels
        ↑
prediction-blind local annotation UI
        +
fixed Step 33 config/scenario/topology
        ↓
real-world-capture (unchanged detector + replay, isolated in-memory store)
        ↓
hashed prediction/provenance JSON
        ↓
real-world evaluator
        ↓
detection @ IoU 0.5 (+ 0.3 diagnostic)
OCR exact match on detector-matched evaluable instances
end-to-end exact recognition
confidence precision / coverage / deferrals
false-positive adjudication
optional Recall@K / HitRate@K / MRR when labels are sufficient
```

The preparation, local annotation, capture, and evaluator CLIs are persistent
evaluation interfaces. Preparation extracts exactly the configured replay
frames but marks all of them unreviewed; it does not invent negative labels.
Annotation edits only human labels and intentionally excludes prediction and
replay-event surfaces. These interfaces do not sweep or tune production
settings. Cross-camera percentages require at least two manually labeled source
cases by default, so the single confirmed Step 33 match cannot be presented as
an aggregate accuracy result.

---

## 32. Git architecture/workflow rules

One roadmap step should correspond to one controlled logical commit.

Do not blindly use:

```text
git add .
git add -A
```

Instead stage only intended files.

Before commit:

```text
git status
git diff
git diff --check
```

After push:

- verify `main`;
- verify `HEAD == origin/main`;
- verify working tree is clean.

---

## 33. Privacy and dataset boundaries

ANPR can process identifying information.

Current prototype principles:

- keep raw traffic videos outside Git;
- keep RoundaboutHD raw media outside Git;
- do not commit extracted plate crops unnecessarily;
- prefer opaque event IDs in evaluation labels;
- commit only sanitized evaluation/config/result artifacts when licensing and authorization allow;
- do not derive ground truth from predictions;
- minimize unnecessary identifying data exposure.

---

## 34. Current technology stack

| Area | Current choice |
|---|---|
| Language | Python 3.11 |
| Computer vision | OpenCV |
| Plate detection | Ultralytics YOLO |
| Tracking | BYTETrack |
| OCR | PaddleOCR 3.7 |
| Numerical/image processing | NumPy |
| Persistence | SQLite via `sqlite3` |
| Evidence storage | Local filesystem |
| API | Python stdlib `http.server` |
| Dashboard | HTML / CSS / JavaScript |
| Testing | pytest |
| Tracking assignment dependency | lap (explicit Ultralytics BYTETrack dependency) |
| CLI | Python modules / argparse-style command interfaces |
| Configuration | Repository YAML/config loaders |
| Development OS | Windows / PowerShell |
| External evaluation dataset | Replacement required; RoundaboutHD rejected as primary end-to-end dataset |

---

## 35. Explicitly NOT part of the current architecture

Unless a future reviewed step changes this document, CitySight does **not** currently depend on:

- FastAPI;
- Flask;
- Django;
- PostgreSQL;
- PostGIS;
- MongoDB;
- Redis;
- Kafka;
- Redpanda;
- ClickHouse;
- Kubernetes;
- a microservice mesh;
- cloud inference;
- vector databases;
- learned vehicle-ReID embeddings;
- automatic make/model recognition;
- global probabilistic vehicle identity assignment.

Do not describe any of the above as implemented.

---

## 36. Architectural boundaries that must remain clear

### Plate detection vs vehicle detection

```text
plate YOLO class → license plate
vehicle detector class → car / bus / truck / motorcycle
```

They are different detectors/responsibilities.

### Processing time vs recording time

```text
processing time ≠ event/recording time
```

Cross-camera reasoning must use meaningful event time.

### Missing vs contradictory evidence

```text
unknown colour ≠ different colour
```

### Candidate vs identity

```text
high-ranking candidate ≠ confirmed same vehicle
```

### Camera connectivity vs actual route

```text
known graph link ≠ proof the vehicle used that road
```

---

## 37. Current roadmap checkpoint

> Verify repository state before relying on this table; Git is authoritative.

| Phase | Step | Capability | Status |
|---|---:|---|---|
| Phase 1 | 1–16 | Core single-camera ANPR, persistence, API, dashboard, watchlist, demo | Complete |
| Phase 2 | 17 | Camera registry/topology | Complete |
| Phase 2 | 18 | Recorded-video timestamps | Complete |
| Phase 2 | 19 | Multi-camera replay | Complete |
| Phase 2 | 20 | Trajectory reconstruction | Complete |
| Phase 2 | 21 | Trajectory API | Complete |
| Phase 2 | 22 | Topology API | Complete |
| Phase 2 | 23 | Vehicle fingerprint foundation | Complete |
| Phase 2 | 24 | Whole-vehicle detection + plate association | Complete |
| Phase 2 | 25 | Vehicle attribute extraction | Complete |
| Phase 2 | 26 | Optional per-track vehicle enrichment | Complete |
| Phase 2 | 27 | Explainable pairwise cross-camera candidate matching | Complete |
| Phase 2 | 28 | Bounded replay candidate collection | Complete |
| Phase 2 | 29 | Read-only candidate report | Complete |
| Phase 2 | 30 | Candidate retrieval evaluation | Complete |
| Phase 2 | 31 | Ground-truth loader + evaluation runner | Complete |
| Phase 2 | 32 | Demo/evaluation runbook | Complete |
| Phase 2 | 33 | Real-world two-camera replay validation | Complete |
| Evaluation | 34A | Real-world annotation format + fixed-baseline evaluator | Complete |
| Evaluation | 34B | Manually labeled Step 33 accuracy results | **Pending manual labels** |
| Phase 3 | — | To be decided after Step 34 measurement | Not started |

Last known verified pre-Step-34 checkpoint:

```text
branch: main
commit: c80da80 (Step 33 documentation commit; abbreviated)
working tree: clean
main == origin/main
full suite: 806 passed
```

Always verify this against Git before the next implementation step.

---

## 38. Current Step 34 architecture objective

The immediate system objective is **manual ground-truth measurement**, not
adding more intelligence.

Required sequence:

```text
review every Step 33 frame selected at 10 FPS
        ↓
label all visible plate boxes and exact-text evaluability
        ↓
assign stable physical plate-passage identities
        ↓
capture the unchanged Step 33 baseline with provenance
        ↓
manually link/adjudicate replay observations
        ↓
calculate stage-separated detector / OCR / end-to-end metrics
        ↓
analyze confidence coverage and false positives
        ↓
evaluate cross-camera retrieval only if identity labels are sufficient
        ↓
document reproducible protocol and results
```

Do not add ReID embeddings before measuring the deterministic baseline.

RoundaboutHD has completed this compatibility gate and was rejected as the
primary end-to-end dataset after producing zero plate detections, tracks and
observations across 600 preflight frames. Do not build a RoundaboutHD adapter
or change CitySight inference policies merely to bypass that result.

---

## 39. Potential future architecture — NOT IMPLEMENTED

These are possible future directions only after evaluation justifies them:

### Learned vehicle ReID

```text
vehicle crop
   ↓
feature encoder
   ↓
embedding
   ↓
appearance similarity
```

The RoundaboutHD ReID subset may help evaluate this later.

### Make/model recognition

Would require an appropriate model/dataset and independent evaluation.

### Improved visual dashboard

Could display:

- camera locations;
- trajectories;
- candidate reasoning;
- evaluation summaries.

### Production-scale storage/infrastructure

Only if actual scale requirements justify migration away from the current lightweight SIH architecture.

None of these should be presented as current implementation.

---

## 40. Architecture update protocol for every future step

Every future coding-agent prompt should include:

> **Architecture documentation rule:**
> Read `docs/CURRENT_ARCHITECTURE.md` before designing this step.
> If this step changes architecture, data flow, modules, interfaces, dependencies, persistence, API surface, ML model usage, evaluation flow, deployment assumptions, or roadmap status, update `docs/CURRENT_ARCHITECTURE.md` in the same commit.
> Do not record planned functionality as implemented until tests pass and the step is accepted.
> After implementation, verify that the document matches the actual repository.

After each completed step update at minimum:

1. **Roadmap checkpoint**
2. **Relevant component section**
3. **Architecture/data-flow diagram**, if flow changed
4. **Technology stack**, if dependencies changed
5. **Current limitations**
6. **Architecture Change Log**

---

## 41. Architecture Decision Principles

When making a new architectural decision, use this priority:

```text
correctness
    ↓
testability
    ↓
explainability
    ↓
reproducibility
    ↓
backward compatibility
    ↓
SIH demonstrability
    ↓
performance
    ↓
additional complexity only when justified
```

The project must not become complex merely to appear advanced.

---

## 42. Architecture Change Log

Keep this concise. Record architecture-level changes, not every code edit.

| Date | Step | Architecture change | Status |
|---|---:|---|---|
| 2026-08 | 1–16 | Established modular single-camera ANPR architecture with YOLO, BYTETrack, quality selection, PaddleOCR, SQLite, stdlib API, dashboard and watchlist | Implemented |
| 2026-08/09 | 17–22 | Added multi-camera topology, recorded event time, replay, trajectory and read-only Phase 2 API surfaces | Implemented |
| 2026-09 | 23–26 | Added optional whole-vehicle evidence and deterministic vehicle fingerprints | Implemented |
| 2026-09 | 27–29 | Added explainable pairwise matching, bounded candidate collection and reporting | Implemented |
| 2026-09 | 30–31 | Added independent-ground-truth candidate evaluation and reproducible runner | Implemented |
| 2026-09 | 32 | Added Phase 2 demo/evaluation runbook | Implemented |
| 2026-09 | 33 | Validated RoundaboutHD timing/identity semantics, then rejected it as the primary end-to-end dataset after an unchanged 600-frame CitySight preflight produced zero plate detections, tracks and observations; replacement dataset required | In progress |
| 2026-09 | 34 Part A/B helper | Added the strict real-video benchmark adapter, frame preparation, and a loopback-only prediction-blind manual annotation UI with revision-checked atomic YAML persistence; exhaustive human labeling remains pending | Implemented; labels pending |
| 2026-09 | Phase 1 accuracy audit | Added external-label manifests, dataset leakage/integrity audit, detector configuration sweeps, OCR preprocessing and track-fusion ablations, linked end-to-end recognition, stage-correct metrics, and experiment provenance without changing production inference | Implemented; labeled dataset required for measurements |

---

## 43. One-sentence architecture summary

**CitySight is a lightweight, modular, explainable multi-camera ANPR and vehicle-correlation system that converts video into confidence-aware plate observations, enriches them with optional vehicle evidence and camera/time context, ranks physically plausible cross-camera candidates deterministically, and evaluates retrieval against independent ground truth without pretending uncertain correlations are confirmed identities.**

---

## 44. Source-of-truth rule

If this document ever conflicts with actual code:

1. stop;
2. inspect the repository;
3. determine whether the code or documentation is stale;
4. correct the stale side explicitly;
5. add an Architecture Change Log entry when the change is architectural.

**Do not silently make the documentation fit an assumption.**

## Step 33 — Real-World Two-Camera Validation

CitySight has been validated on two self-recorded 1080p road videos using
CAM_01 -> CAM_02 replay.

Current validated runtime baseline:

- Replay processing rate: 10 FPS
- Detector image size: 640
- Detector device: CUDA
- OCR device: CPU
- Directed camera link distance: approximately 50 m
- Vehicle enrichment: disabled

The baseline replay produced 38 observations:
10 accepted, 18 review, and 10 abstained.

A genuine cross-camera sighting of GJ04EJ5933 was associated from CAM_01
to CAM_02 as `possible_strong`. The elapsed time was 9.726797 seconds over
approximately 50 m, which is physically feasible under the configured
60 km/h maximum-speed policy.

A controlled detector image-size experiment at 960 produced more total
observations (49) but fewer accepted observations (6), more abstentions
(19), and no increase in normalized plate availability. Therefore
image_size=640 remains the Step 33 baseline.

Detailed experiment results are documented in:
`docs/STEP33_REAL_WORLD_EVALUATION.md`.

## Step 34 — Real-World Ground-Truth Benchmark

Step 34 Part A adds the strict, label-driven workflow documented in
`docs/STEP34_REAL_WORLD_GROUND_TRUTH_BENCHMARK.md` and the fillable
`phase1_anpr/evaluation/step34_ground_truth.template.yaml`. Part B adds the
local prediction-blind annotation helper for the prepared YAML and extracted
frames.

Exhaustive independent labeling remains pending. The architecture must not
claim detector, OCR, end-to-end, confidence, or cross-camera percentages until
those manual labels exist and the evaluator has calculated the stated
numerators and denominators.
