# CitySight Phase 1 ANPR Evaluation

This is the reproducible protocol for detector, OCR, and track-level fusion
experiments. It keeps stage metrics separate and never derives labels from
CitySight predictions.

## Required labeled data

Copy `phase1_anpr/evaluation/manifest.example.yaml` outside the raw-data tree
and replace its placeholders. Paths may be absolute or relative to the
manifest. Every detector frame needs independently annotated plate boxes in
`x1, y1, x2, y2` pixels. Every OCR crop needs independently transcribed plate
text. Crops from the same tracked plate should share a `track_id` and ground
truth.

For end-to-end metrics, assign the same stable identity to each repeated
detector box using `plate_id` and to its OCR crops using `track_id`. These IDs
must describe independently labeled physical plate tracks, not CitySight's
predicted tracker IDs.

Use a stable dataset version or content hash in `version`. Set `sequence_id` to
the source video or capture sequence and assign splits at sequence level. Do
not split adjacent frames from one video across train, validation, and test.

Optional failure tags are manual adjudications and take precedence over image
heuristics. Recommended detector tags include `very_small_plate`,
`distant_vehicle`, `motion_blur`, `low_lighting`, `glare`, `oblique_angle`,
`partial_plate`, `plate_occlusion`, `dirty_plate`, `front_plate`, `rear_plate`,
`unusual_aspect_ratio`, and `annotation_issue`. Recommended OCR tags include
`detector_crop_incomplete`, `perspective_distortion`, `blur`, `low_resolution`,
`glare`, `dark_plate`, `double_line_plate`, `character_confusion`,
`punctuation_noise`, `ocr_confidence_issue`, `invalid_indian_plate_pattern`, and
`wrong_plate_region`.

## 1. Dataset integrity gate

```powershell
.\.venv\Scripts\python.exe -m phase1_anpr.evaluation audit `
  --manifest D:\datasets\citysight-eval\manifest.yaml `
  --output-dir outputs\annotated\evaluation
```

The audit checks missing/corrupt images, exact duplicates, duplicates across
splits, sequence-level split leakage, boxes outside image bounds, split counts,
small-object prevalence, and coarse lighting distribution. A final evaluation
must not use `suitable_for_evaluation: false` without adjudicating the reported
issues.

## 2. Detector sweep and failure analysis

```powershell
$env:YOLO_CONFIG_DIR = (Resolve-Path outputs\annotated).Path
.\.venv\Scripts\python.exe -m phase1_anpr.evaluation detector `
  --manifest D:\datasets\citysight-eval\manifest.yaml `
  --weights weights\plate_detector.pt `
  --device cuda `
  --confidence-thresholds 0.10,0.15,0.20,0.25,0.30,0.35,0.40 `
  --nms-iou-thresholds 0.45,0.60 `
  --image-sizes 640,960 `
  --match-ious 0.3,0.5 `
  --primary-match-iou 0.5 `
  --output-dir outputs\annotated\evaluation
```

Ultralytics already uses aspect-ratio-preserving letterbox resizing. The runner
loads the detector once, performs inference at a low confidence floor for each
image-size/NMS combination, and filters those predictions for the confidence
sweep. It reports TP, FP, FN, precision, recall, F1, and throughput for every
operating point. The recommendation is selected by F1, then recall, precision,
and runtime at the primary match IoU; it does not edit production config.

False negatives use manual labels when present. Otherwise bounded heuristics
identify very small plates, confidence-threshold misses, localization below the
match IoU, dark crops, blur, unusual aspect ratios, or manual-review cases.
Annotated examples are capped by `--max-failure-images` and written under the
ignored output directory.

## 3. OCR preprocessing and fusion ablation

PaddleOCR model weights must already exist locally. The evaluator will not
download them automatically.

```powershell
.\.venv\Scripts\python.exe -m phase1_anpr.evaluation ocr `
  --manifest D:\datasets\citysight-eval\manifest.yaml `
  --config phase1_anpr\config\config.yaml `
  --ocr-model-dir D:\models\PP-OCRv6_medium_rec `
  --variants original,rectified,rectified_upscale,rectified_clahe,rectified_upscale_sharpen `
  --k-values 1,3,5 `
  --output-dir outputs\annotated\evaluation
```

The controlled crop variants are original, current OpenCV rectification,
rectification plus small-crop upscaling, rectification plus CLAHE, and
rectification/upscaling plus mild sharpening. Exact plate accuracy, micro
character accuracy, edit errors, and time per crop are reported separately.

For labeled tracks, the same OCR outputs are evaluated with best-frame,
majority, confidence-weighted, quality-times-confidence-weighted, and
character-position-weighted fusion at K=1, 3, and 5. The report also includes
accepted-set accuracy, accepted coverage, review rate, abstention rate, and an
aligned character-confusion summary. Contextual grammar correction is not
performed by this benchmark, so it cannot hide OCR errors.

## 4. Linked end-to-end recognition

After detector and OCR runs from the same manifest:

```powershell
.\.venv\Scripts\python.exe -m phase1_anpr.evaluation end-to-end `
  --detector-report outputs\annotated\evaluation\detector_evaluation.json `
  --ocr-report outputs\annotated\evaluation\ocr_evaluation.json `
  --output-dir outputs\annotated\evaluation
```

The command verifies that both reports have the same manifest hash. A
ground-truth track is counted end-to-end correct only when at least one linked
ground-truth detector box was matched at the selected operating point and the
selected fused OCR text exactly matches its transcription. It reports detector
track coverage, track OCR exact accuracy, end-to-end exact accuracy,
accepted-set accuracy and coverage, review rate, and abstention rate. If OCR
crops are created from predicted CitySight tracks, this also exercises the
real crop/track evidence path; it does not otherwise manufacture a tracking
accuracy claim from static images.

## Output and acceptance rule

Each command writes machine-readable JSON and a concise Markdown report. JSON
includes the manifest hash, detector/model hash, Python/OpenCV environment, and
timestamp. Generated reports and failure images stay under ignored output
paths unless a small sanitized report is intentionally reviewed and committed.

A production change is eligible only when it improves the practical held-out
operating point, preserves strong precision and coverage, has acceptable
runtime, and passes the full regression suite. Detector precision is never
reported as ANPR accuracy; crop OCR, fused track accuracy, and end-to-end
recognition remain distinct metrics.
