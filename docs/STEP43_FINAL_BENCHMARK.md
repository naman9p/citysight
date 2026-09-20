# Step 43 Final Benchmark and Ablation Runbook

## Status

Step 43A prepares the read-only evaluation tooling. Step 43B remains pending
new real-world videos, prediction-blind manual labels, and separately captured
frozen Phase 2 and Phase 3 results. This document contains no benchmark result
or accuracy claim.

The incomplete local Step 34 labels are historical work in progress and are
not an input to the final benchmark. Step 43B must begin with a new benchmark
ID and newly supplied videos.

## Non-negotiable comparison rule

Every Phase 2 versus Phase 3 comparison uses one ground-truth file, its exact
SHA-256, and the same manually verified cross-camera source cases. A paired
metric records both numerators, denominators and case-ID sets. An absolute
delta is emitted only when the metric has the same meaning, denominators are
equal and case-ID sets are identical.

The evaluator refuses a final-claim report when any manifest frame is
unreviewed, any plate/observation link is unreviewed, source observations have
not been adjudicated, or fewer than two manually verified cross-camera cases
exist. It does not calculate partial accuracy in that state.

## Frozen and current systems

The independently reportable Phase 2 baseline is fixed at repository reference
`a86baa8`. Its Step 33 config, city topology/config and replay scenario are
supplied as hashed artifacts. The existing Phase 2 prediction capture retains
its historical matcher and collector behavior; in particular, Step 43 does not
rewrite it to use Phase 3 trusted-plate contradiction semantics.

The Phase 3 implementation reference is an explicit manifest input. The
manifest also records appearance model/version and weights/preprocessing hashes
when appearance is available, or an explicit unavailable reason otherwise. It
records retrieval, physical, hybrid and hypothesis policies. Git state is not
silently inferred.

## Inputs

Step 43 consumes three kinds of input:

1. A Step 34-schema, prediction-blind ground-truth file for the new videos.
2. A Step 34-schema prediction capture produced by the frozen Phase 2 path.
3. A Step 43 Phase 3 result bundle containing the actual bounded Step 39
   candidate population, Step 37 diagnostics, and optional Step 41 hypothesis
   paths for each manually verified source case.

The Phase 3 bundle is prediction output, not ground truth. It must have this
shape:

```json
{
  "schema_version": 1,
  "benchmark_id": "the-new-benchmark-id",
  "ground_truth_sha256": "<64 lowercase hexadecimal characters>",
  "ordering_semantics": "chronological_retrieval_order",
  "cases": [
    {
      "source_event_id": "source-event-id",
      "candidates": [
        {
          "event_id": "candidate-event-id",
          "hybrid_status": "eligible_with_evidence",
          "physical_gate_status": "feasible",
          "trusted_plate_relation": "both_unavailable",
          "appearance_status": "available",
          "appearance_available": true,
          "appearance_cosine_similarity": 0.0
        }
      ],
      "hypotheses": {
        "count": 0,
        "branch_count": 0,
        "coverage_event_ids": [],
        "paths": [],
        "truncated": false
      }
    }
  ]
}
```

The cosine value above illustrates the field type only; it is not a threshold,
probability, decision rule or reported result. Candidate array position is the
captured chronological retrieval position. It is never called a relevance
rank.

## Run manifest

The YAML or JSON run manifest has schema version 1:

```yaml
schema_version: 1
benchmark_id: the-new-benchmark-id
created_at: 2026-01-01T00:00:00+00:00
evaluation_at: 2026-01-02T00:00:00+00:00
ground_truth:
  path: path/to/new_ground_truth.yaml
  sha256: <sha256>
sources:
  - source_id: source-id
    camera_id: camera-id
    video:
      path: path/to/new_video.mp4
      sha256: <sha256>
    metadata:
      device: optional device label
      resolution: optional resolution
      nominal_fps: optional FPS
      codec: optional codec
      notes: optional capture notes
phase2:
  baseline_reference: a86baa8
  matcher_identifier: citysight-phase2-frozen-step33
  predictions:
    path: path/to/frozen_phase2_predictions.json
    sha256: <sha256>
  artifacts:
    config:
      path: phase1_anpr/config/config.step33.yaml
      sha256: <sha256>
    city_config:
      path: phase2_city/config/city.step33.yaml
      sha256: <sha256>
    baseline_replay_config:
      path: phase2_city/config/replay.step33.yaml
      sha256: <sha256>
    benchmark_scenario:
      path: path/to/the-new-frozen-baseline-scenario.yaml
      sha256: <sha256>
  retrieval_policy:
    window_seconds: <explicit value>
    max_results: <explicit value>
  physical_policy:
    maximum_speed_kph: <explicit value>
phase3:
  implementation_reference: <explicit commit or immutable build ID>
  results:
    path: path/to/phase3_results.json
    sha256: <sha256>
  appearance:
    available: true
    model_id: <model ID>
    model_version: <model version>
    weights_sha256: <sha256>
    preprocessing_sha256: <sha256>
  retrieval_policy:
    window_seconds: <explicit value>
    max_results: <explicit value>
    ordering: chronological
  physical_policy:
    maximum_speed_kph: <explicit value>
  hybrid_policy:
    implementation_id: <Step 37 implementation ID>
    version: 1
  hypothesis_policy:
    implementation_id: citysight-phase3-step41-v1
    version: 1
    max_hops: <explicit value>
```

If no appearance model was available, replace the appearance block with:

```yaml
appearance:
  available: false
  unavailable_reason: <explicit reason>
```

Paths may be absolute or relative to the manifest. Every referenced file must
exist and match its declared SHA-256. Source IDs, camera IDs and resolved video
paths must agree with the ground truth. Phase 2 predictions and Phase 3 results
must use the same benchmark ID. The Phase 3 bundle must repeat the exact
ground-truth hash.

## Future Step 43B workflow

1. Register the new videos in a new scenario and record their hashes and
   optional capture metadata.
2. Use the Step 34 preparation and local annotation workflow to create manual
   labels without inspecting predictions.
3. Finish all frame review, observation adjudication, stable instance linking
   and manual cross-camera cases.
4. Capture frozen Phase 2 predictions with the frozen Step 33 runtime/config
   and preserve the prediction provenance.
5. Capture actual Steps 37/39/41 evidence into the Phase 3 result bundle. Do not
   transform chronological retrieval into a relevance rank.
6. Calculate and enter all file hashes and explicit policy/provenance fields in
   the run manifest.
7. Run:

```powershell
python -m phase1_anpr.evaluation step43 `
  --manifest path\to\step43_run.yaml `
  --output-dir outputs\step43_final
```

Use the repository virtual-environment Python when `python` is not on `PATH`.

## Outputs

The output directory contains:

- `step43_final_benchmark.json`: authoritative readiness, provenance, shared
  Phase 1 metrics, frozen Phase 2 cross-camera metrics, paired-comparison
  records and Phase 3 diagnostics.
- `step43_final_benchmark.md`: deterministic summary that keeps denominators
  and unavailable reasons visible.
- `step43_ablation.json` and `step43_ablation.md`: supported diagnostic
  ablations and explicitly unsupported variants.
- `step43_failures.json`: detector FN/FP, OCR errors, missing observations,
  confidence deferrals, per-case gates/evidence/hypothesis ambiguity, and
  Phase 2/Phase 3 retrieved-set disagreements.

JSON is authoritative. Files are written by same-directory temporary file plus
atomic replacement. JSON serialization rejects NaN and Infinity.

## Metrics and interpretation

The existing Step 34 implementation remains authoritative for combined and
per-camera detector precision/recall/F1 at IoU 0.5, diagnostic IoU 0.3, OCR
exact/character metrics, end-to-end exact recognition, confidence/status
metrics, false positives, and frozen Phase 2 Recall@K, HitRate@K and MRR.
Unreadable visible plates remain detector ground truth and stay out of text
denominators when `text_evaluable: false`. Reviewed negative frames remain in
the detector population.

Per-camera groups are generated from the manifest camera IDs; the code does not
assume `CAM01` or `CAM02`. Combined metrics use the same input objects and the
same Step 34 evaluator. Phase 3 diagnostics are also grouped by each source
case's camera ID as well as reported in aggregate; cross-camera outcomes remain
case-level and are not misrepresented as single-camera identity accuracy.

Phase 3 reports eligible candidate, physical rejection, trusted-plate
contradiction, appearance available/incompatible, eligible-with/without-
evidence, hypothesis, branch and hypothesis-coverage diagnostics. They are
diagnostics unless independently verified ground truth gives a value an
accuracy interpretation. Capture metadata is descriptive and does not support
causal claims.

## Supported and unsupported ablations

Supported reporting covers the frozen Phase 2 matcher, Phase 3 bounded
structural retrieval, missing/disabled appearance, compatible appearance
evidence, appearance coverage, physical hard-gate effects, trusted-plate
contradiction effects and hypothesis diagnostics.

The following are intentionally unsupported:

- an appearance-only winner or same-vehicle classifier;
- a cosine threshold or conversion to identity probability;
- an aggregate hybrid numeric score;
- sorting candidates by cosine;
- Recall@K, HitRate@K or MRR for the current Phase 3 chronological order;
- statistical-significance claims.

Exact Phase 2 trajectories and inferred Phase 3 hypotheses remain separate in
both machine and human reports. A hypothesis is never relabeled as an exact
trajectory or a persistent global vehicle identity.
