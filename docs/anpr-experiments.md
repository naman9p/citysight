# CitySight ANPR Experiment Ledger

## Baseline status — 2026-09-18

The figures supplied in the improvement brief are recorded below but could not
be reproduced from this repository: no Phase 1 detector annotations, OCR
transcriptions, track-level ground truth, or original evaluation outputs are
present. They remain **reported / unverified**, not silently replaced.

| Experiment | Detector recall | Detector precision | Detector F1 | OCR exact | Track fusion exact | Runtime | Accepted? |
|---|---:|---:|---:|---:|---:|---:|---|
| Reported baseline, IoU >= 0.5 (~445 frames, ~654 plates) | 74.16% | 91.68% | 81.99% (derived) | 74.00% (296/~400) | unavailable | unavailable | Baseline only; source artifacts missing |
| Reported baseline, IoU >= 0.3 | 78.13% | 96.60% | 86.39% (derived) | — | — | — | Diagnostic only |
| Local video preflight, production detector config, CPU, 40 evenly sampled frames | not measurable | not measurable | not measurable | not measured | not measured | 1.376 FPS; 13 detections | Not an accuracy experiment; no labels |
| Local PaddleOCR smoke test, 5 unlabeled evidence crops, CPU | — | — | — | not measurable | not measurable | 727.17 ms/crop after one warm pass | Not an accuracy experiment; no labels |

The local preflight used `weights/plate_detector.pt`, confidence 0.35, NMS IoU
0.45, image size 640, and CPU. It sampled 20 frames from each local video:
12 detections from `traffic1.mp4` and 1 from `traffic2.mp4`. Detection counts do
not imply correctness.

The existing local `PP-OCRv6_medium_rec` model produced stable outputs through
the production recognition-only adapter. A three-pass timed smoke run over the
five unlabeled crops averaged 727.17 ms/crop on CPU. The sample is far too small
and unlabeled to support an accuracy or general throughput claim.

## Audit findings

- Production detector: custom Ultralytics YOLO, class 0 `license_plate`,
  confidence 0.35, NMS IoU 0.45, image size 640. Ultralytics letterboxing
  preserves aspect ratio.
- Detector weights exist locally and identify their training data as
  `data/yolo/data.yaml`, but that dataset and its split metadata are absent.
- Quality selection keeps top 3 crops after minimum 60x20 filtering.
- Production performs OpenCV contour-based rectification, then PaddleOCR
  recognition-only inference.
- The configured `preprocessing` section is currently not wired into the
  production pipeline; it must not be enabled without an ablation result.
- Fusion sums OCR/quality/detector evidence for identical strings only; K is
  effectively 3. No labeled track set exists to compare K=1/3/5 or alternate
  fusion safely.
- Normalization validates standard and BH-series Indian formats and deliberately
  performs no ambiguous-character substitution.
- Confidence is heuristic and uncalibrated; accepted/review/abstained accuracy
  and coverage are unavailable without track labels.
- A clean environment exposed an undeclared BYTETrack runtime dependency,
  `lap`; it is now explicit in `requirements.txt`.

## Changes accepted in this iteration

| Change | Accuracy effect | Runtime effect | Decision |
|---|---:|---:|---|
| Add strict evaluation manifests, integrity/leakage audit, detector sweeps, OCR ablation, fusion benchmark, linked end-to-end recognition, confidence/coverage metrics, and failure reports | Not measurable without labels | Evaluation-only | Accepted: makes future measurements reproducible; no production behavior change |
| Declare `lap>=0.5.12` | None | None | Accepted: fresh-install regression suite otherwise cannot collect |

## Changes explicitly not integrated

| Candidate | Reason rejected/deferred |
|---|---|
| Lower detector confidence | No labels to measure recall/precision tradeoff |
| Larger detector image size | No labels; local CPU runtime is already 1.376 FPS at 640 in a small preflight |
| Enable configured CLAHE/upscaling | Current config is unused and PaddleOCR preprocessing can regress accuracy; requires ablation |
| Contextual O/0, I/1, B/8 substitutions | No confusion evidence; hardcoding would risk false corrections |
| Change fusion strategy or K | No labeled multi-frame tracks |
| Fine-tune detector | Training dataset, provenance, sequence-level split, and integrity evidence are absent |

## Next acceptance gate

Provide the original labeled detector frames and OCR crops (or a new held-out,
independently labeled set) in the manifest format documented in
`docs/anpr-evaluation.md`. Run the audit first. The highest-impact next
experiment is the detector confidence/image-size/NMS sweep because reported
recall at IoU 0.5 is the current bottleneck.
