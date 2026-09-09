# Phase 2 Demo and External Ground-Truth Evaluation

This runbook covers CitySight's implemented Phase 2 recorded-video workflow
through Step 31. It demonstrates explainable cross-camera **candidate
hypotheses** and evaluates their retrieval order against independently supplied
ground truth. A candidate is not a confirmed vehicle identity.

## Prerequisites

- Python 3.11+ with `requirements.txt` installed in `.venv`.
- Local license-plate detector weights configured by
  `detection.weights_path` in `phase1_anpr/config/config.yaml`.
- Local whole-vehicle detector weights when `vehicle_enrichment.enabled` is
  `true`. They are not loaded when enrichment is disabled.
- Recorded clips placed under `inputs/videos/`, or another local path used by
  the scenario file.
- A city topology such as `phase2_city/config/city.yaml`.
- A replay scenario such as `phase2_city/config/replay.example.yaml`.
- For evaluation, an externally labeled YAML file following
  `phase2_city/config/candidate_ground_truth.example.yaml`.

The repository does not include videos or model weights and never downloads
weights automatically. The example replay refers to `cam01.mp4` and
`cam02.mp4`; supply those clips or update the example paths before running it.

## Implemented architecture

The relevant Phase 2 steps are:

- Steps 17–22: city cameras and directed links, recorded multi-camera replay,
  exact-plate trajectory reconstruction, and topology/trajectory API surfaces.
- Steps 23–26: vehicle fingerprints, plate-to-vehicle association, vehicle
  colour/class extraction, and optional replay-time enrichment.
- Step 27: explainable pairwise cross-camera candidate comparison using
  fingerprint evidence, directed topology, and travel feasibility.
- Step 28: bounded, deterministic collection and ranking of later
  observations from other cameras.
- Step 29: a read-only candidate-hypothesis report attached to scenario replay.
- Step 30: deterministic candidate Recall@K, HitRate@K, MRR, and per-case
  diagnostics using explicit labels.
- Step 31: strict YAML label loading and a reproducible evaluation runner.

The end-to-end data flow is:

```text
recorded multi-camera sources
  → replay
  → PlateObservation + optional transient VehicleFingerprint
  → pairwise cross-camera matching
  → bounded candidate collection and deterministic ranking
  → candidate report
  → comparison with external ground truth
  → Recall@K / HitRate@K / MRR
```

Fingerprint candidate hypotheses and exact-plate trajectories are separate
layers. Candidate results are not inserted into trajectories and do not create
global vehicle identities.

## Replay a scenario

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe -m phase2_city.replay `
  --scenario phase2_city/config/replay.example.yaml `
  --city-config phase2_city/config/city.yaml `
  --config phase1_anpr/config/config.yaml
```

The scenario requires a non-empty `scenario_id` and at least one source. Each
source requires a unique `source_id`, a camera present in the city topology, an
existing `video_path`, and a timezone-aware `start_time`. Relative video paths
are resolved from the scenario YAML's directory. Replay performs fail-fast
scenario validation before processing any source.

The command prints per-source and aggregate observation/status counts. It uses
the same configured SQLite observation/watchlist repositories and filesystem
evidence store as the existing demo. A source's explicit `source_id` is folded
into the deterministic UUID5 event-ID input with its camera and produced track
ID. This prevents track-number collisions between clips. Track output can
change if source data, model behavior, or processing configuration changes, so
retain those inputs when reproducing event IDs.

## Inspect candidate hypotheses

First replace every `<...>` placeholder below with an explicit value:

```powershell
.\.venv\Scripts\python.exe -m phase2_city.replay `
  --scenario phase2_city/config/replay.example.yaml `
  --city-config phase2_city/config/city.yaml `
  --config phase1_anpr/config/config.yaml `
  --candidate-source-event-id "<source-event-id>" `
  --candidate-window-seconds <positive-seconds> `
  --candidate-max-results <positive-integer> `
  --candidate-maximum-speed-kph <positive-kilometres-per-hour>
```

Candidate mode is opt-in. Once any candidate-report option is used, all four
candidate inputs shown above are required:

- `--candidate-source-event-id` selects exactly one replay observation.
- `--candidate-window-seconds` is the inclusive forward-time search window.
- `--candidate-max-results` caps Step 27 evaluations before matching.
- `--candidate-maximum-speed-kph` supplies Step 27's physical-feasibility
  ceiling.

The report preserves Step 28's deterministic order and normally shows only
`possible_strong` and `possible_weak` hypotheses. These labels mean
**plausible candidate**, not “same vehicle confirmed.”

Add `--include-candidate-diagnostics` to show every bounded evaluation,
including `insufficient_evidence`, `identity_conflict`,
`physically_impossible`, and `ineligible`. Reasons, topology status, travel
status, elapsed time, similarity, and evidence coverage come directly from the
Step 27 result.

When more observations satisfy the structural search rules than the configured
cap permits, the report explicitly says evaluation was truncated. It reports
the full structurally eligible count and the smaller evaluated count; it does
not imply that every eligible observation was matched.

## Obtain event IDs for annotation

No separate observation-catalog system exists. Use the supported persisted
observation surfaces:

1. Run the scenario once to populate the configured SQLite observation store.
2. Start the local server:

   ```powershell
   .\.venv\Scripts\python.exe -m phase1_anpr.serve
   ```

3. Inspect `http://127.0.0.1:8000/dashboard` or `GET /observations`. Canonical
   observations include `event_id`, `camera_id`, timestamp, status, track ID,
   and available plate fields.
4. Map those observations back to the underlying source footage using the
   scenario's camera, source, and start-time context.

The API/dashboard helps locate observations; it does not provide candidate
ground truth. The candidate report can be rerun with a selected deterministic
event ID while the same replay inputs and configuration are retained.

## External ground-truth protocol

Ground truth must come from manual verification of the underlying footage or
another trusted, independent source. Annotators should determine that two
events show the same physical vehicle without using the system output being
evaluated as the answer key.

Never derive a relevant-event label from:

- matching or normalized number plates;
- vehicle colour/class or fingerprint similarity;
- candidate score, similarity, evidence coverage, decision, or rank;
- `possible_strong` or `possible_weak` output;
- any other prediction made by CitySight.

Using those predictions to create labels is evaluation leakage: the system
would help manufacture the truth used to score itself.

The Step 31 YAML schema is intentionally minimal:

```yaml
scenario_id: demo-city-01

cases:
  - source_event_id: "SOURCE_EVENT_UUID_FROM_REPLAY"
    relevant_event_ids:
      - "LATER_RELEVANT_EVENT_UUID_FROM_REPLAY"
```

- `scenario_id` identifies the replay scenario and must match it exactly.
- `cases` is a non-empty list.
- Each `source_event_id` identifies one labeled query observation.
- `relevant_event_ids` is a non-empty list of externally verified later
  sightings of the same physical vehicle.
- IDs must be non-empty strings. Relevant IDs must be unique and cannot include
  their source ID.
- Ground-truth source cases must be unique.
- Every source and relevant event must exist exactly once in the replay result.
- Unknown dataset/case fields are rejected.

The schema does not independently infer or validate physical identity. Step 28
only retrieves later, different-camera observations, so ground-truth cases
should label that same retrieval problem.

## Fix policies before evaluation

The evaluation runner requires one explicit policy configuration:

- `--k`: one or more unique positive integer ranks; Step 30 stores them in
  ascending order.
- `--candidate-window-seconds`: finite positive Step 28 search window.
- `--candidate-max-results`: positive integer Step 28 evaluation cap.
- `--candidate-maximum-speed-kph`: finite positive Step 27 speed ceiling.

There are no CLI defaults for these four evaluation choices. Choose and record
them before examining evaluation results. Do not try several values and report
only the best outcome as though it were an untouched evaluation. Other Step 27
matcher thresholds are not exposed by this runner; record the CitySight commit
alongside results so the exact implementation defaults remain identifiable.

## Run an evaluation

Replace every `<...>` placeholder with the fixed input or policy value:

```powershell
.\.venv\Scripts\python.exe -m phase2_city.candidate_evaluation_runner `
  --scenario phase2_city/config/replay.example.yaml `
  --ground-truth phase2_city/config/candidate_ground_truth.example.yaml `
  --k <K1> <K2> <K3> `
  --candidate-window-seconds <positive-seconds> `
  --candidate-max-results <positive-integer> `
  --candidate-maximum-speed-kph <positive-kilometres-per-hour> `
  --city-config phase2_city/config/city.yaml `
  --config phase1_anpr/config/config.yaml
```

The example ground-truth file contains placeholders and is not an evaluation
dataset. Replace them with independently labeled event IDs from the matching
scenario before running the command.

The default output is a compact text summary. Add `--include-cases` for source
event, relevant IDs, returned candidate IDs, first relevant rank, truncation,
structurally eligible count, and evaluated count for each case. Add `--json` to
print the exact deterministic `CandidateEvaluationSummary.to_dict()` shape.
JSON already contains the per-case results. The runner does not automatically
write a report file.

The reported metrics are:

- **Candidate Recall@K:** total relevant events retrieved in the first K usable
  candidate hypotheses divided by all labeled relevant events (micro recall).
- **Candidate HitRate@K:** fraction of source cases with at least one relevant
  event among the first K candidate hypotheses.
- **Candidate MRR:** mean reciprocal rank of the first relevant candidate; a
  case with no retrieved relevant candidate contributes zero.
- **Truncated cases:** count and rate of cases where Step 28 had more
  structurally eligible observations than its evaluation cap.

The evaluator measures `possible_strong` and `possible_weak` candidate
hypotheses only. A labeled event present solely among rejected diagnostic
results is a retrieval miss. Fewer than K returned candidates are evaluated as
the system actually returned them; no synthetic entries are added.

Zero Recall@K, zero HitRate@K, or zero MRR is a valid successful evaluation,
not an execution error. Missing or duplicate event IDs, scenario mismatch,
invalid policy values, malformed labels, and replay failures are errors and
return a non-zero status.

## Reproducibility checklist

Retain the following with any reported result:

- CitySight commit hash;
- scenario ID and scenario YAML revision;
- identity/version of every source clip;
- city topology and Phase 1 configuration revisions;
- ground-truth filename and revision;
- exact command, including K values, candidate window, cap, and speed ceiling;
- raw JSON output when appropriate.

For a fixed completed `ScenarioResult`, labels, and policies, Steps 28 and 30
produce deterministic logical ordering and metrics. CPU/GPU, model-version,
library-version, or runtime differences may affect upstream detection and
tracking, so those inputs also belong in an evaluation record.

## Interpret results carefully

- Zero hits are valid measurements and must not be hidden.
- Truncation is measured behavior, not an evaluator failure.
- A short time window or small candidate cap can reduce candidate recall.
- MRR rewards placing the first relevant candidate earlier.
- Recall@K and HitRate@K measure candidate retrieval for the labeled cases.
  They are not universal vehicle-identification or ANPR/OCR accuracy.
- Candidate decisions are explainable hypotheses, not identity proof or route
  proof.

Do not publish a performance or accuracy claim without a genuine external
ground-truth dataset and its evaluation protocol.

## Privacy and security

Traffic footage, number plates, crops, databases, and label relationships can
contain sensitive information. Videos, model weights, generated evidence, and
local SQLite databases are gitignored and should remain private unless there is
explicit authority to share them. Publish only sanitized demonstration data,
and redact or replace real registration numbers in public screenshots and
presentations where appropriate.

The stdlib HTTP API/dashboard has no authentication or RBAC. Its default server
host is `127.0.0.1`; keep it local for the prototype and do not expose it to an
untrusted network.

## Current limitations

- Candidate hypotheses do not create global vehicle identities.
- There is no neural vehicle-ReID embedding or vector search system.
- Candidate evidence and fingerprints used by Steps 27–31 remain in the
  completed replay result in memory; they are not stored as candidate records.
- Exact-plate trajectory reconstruction is independent and does not consume
  candidate hypotheses.
- Direct camera links support travel-feasibility evidence; they do not prove a
  vehicle's route.
- The workflow processes recorded video rather than live multi-camera streams.
- No candidate-retrieval quality claim is valid until external labels are
  supplied and the fixed evaluation protocol is reported.
