"""CLI and reusable runners for CitySight Phase 1 accuracy experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

from phase1_anpr.confidence.confidence_scorer import ConfidenceScorer
from phase1_anpr.evaluation.dataset import (
    EvaluationDatasetError,
    OCRCrop,
    audit_manifest,
    load_evaluation_manifest,
)
from phase1_anpr.evaluation.metrics import (
    aligned_substitutions,
    evaluate_detections,
    evaluate_ocr,
    intersection_over_union,
)
from phase1_anpr.normalization.plate_normalizer import PlateNormalizer
from phase1_anpr.ocr.plate_ocr import OCRResult, PlateOCR
from phase1_anpr.pipeline.track_processor import FusedCandidate, TrackResult
from phase1_anpr.quality.quality_scorer import ScoredCrop
from phase1_anpr.rectification.rectifier import PlateRectifier
from phase1_anpr.utils.config import load_config


def _parse_numbers(value: str, cast=float) -> tuple:
    try:
        values = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid comma-separated values: {value}") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    return values


def _write_reports(output_dir: Path, name: str, report: dict, markdown: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / f"{name}.md").write_text(markdown, encoding="utf-8")


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(_sha256_file(file_path).encode("ascii"))
    return digest.hexdigest()


def _provenance(manifest_path) -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "opencv": cv2.__version__,
    }


def _detector_markdown(report: dict) -> str:
    lines = [
        f"# Detector evaluation — {report['dataset']} ({report['dataset_version']})",
        "",
        f"Images: {report['images']}  ",
        f"Ground-truth plates: {report['ground_truth_plates']}  ",
        "",
        "| Image size | NMS IoU | Confidence | Match IoU | TP | FP | FN | Precision | Recall | F1 | FPS |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["experiments"]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['image_size']} | {row['nms_iou']} | {row['confidence_threshold']} "
            f"| {metrics['iou_threshold']} | {metrics['true_positives']} "
            f"| {metrics['false_positives']} | {metrics['false_negatives']} "
            f"| {metrics['precision']:.4f} | {metrics['recall']:.4f} "
            f"| {metrics['f1']:.4f} | {row['fps']:.2f} |"
        )
    selected = report["selected_operating_point"]
    lines.extend([
        "",
        "## Selected operating point",
        "",
        "Selected by highest F1, then recall, then precision at the primary match IoU. "
        "This is an evaluation recommendation; it does not change production configuration.",
        "",
        f"`imgsz={selected['image_size']}, nms_iou={selected['nms_iou']}, "
        f"confidence={selected['confidence_threshold']}`",
        "",
        "## False-negative categories",
        "",
    ])
    for category, count in sorted(report["false_negative_categories"].items()):
        lines.append(f"- {category}: {count}")
    return "\n".join(lines) + "\n"


def _classify_false_negative(image, labeled_box, raw_predictions, selected_predictions,
                             match_iou: float, selected_confidence: float) -> str:
    if labeled_box.tags:
        return labeled_box.tags[0]
    x1, y1, x2, y2 = (int(round(value)) for value in labeled_box.bbox)
    height, width = image.shape[:2]
    plate_width, plate_height = x2 - x1, y2 - y1
    area_ratio = max(0, plate_width) * max(0, plate_height) / max(1, width * height)
    if area_ratio < 0.0015 or plate_height < 16:
        return "very_small_plate"

    best_raw = max(
        ((intersection_over_union(labeled_box.bbox, box), confidence)
         for box, confidence in raw_predictions),
        default=(0.0, 0.0),
    )
    if best_raw[0] >= match_iou and best_raw[1] < selected_confidence:
        return "detector_confidence_too_low"
    best_selected_iou = max(
        (intersection_over_union(labeled_box.bbox, box) for box, _ in selected_predictions),
        default=0.0,
    )
    if 0.1 <= best_selected_iou < match_iou:
        return "localization_below_iou"

    crop = image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    if crop.size:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        if float(gray.mean()) < 55.0:
            return "low_lighting"
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 45.0:
            return "motion_blur"
    aspect_ratio = plate_width / max(1, plate_height)
    if aspect_ratio < 1.5 or aspect_ratio > 7.0:
        return "unusual_aspect_ratio"
    return "unclassified_manual_review_required"


def run_detector(args) -> dict:
    # Lazy import keeps dataset auditing independent of Ultralytics settings and
    # heavyweight model imports.
    from phase1_anpr.detection.detector import PlateDetector

    manifest = load_evaluation_manifest(args.manifest)
    if not manifest.detection_images:
        raise EvaluationDatasetError("manifest contains no detection images")
    audit = audit_manifest(manifest)
    if audit["corrupt_or_unreadable"]:
        raise EvaluationDatasetError("detection dataset contains unreadable images")
    if audit["sequence_split_leakage"] and not args.allow_sequence_leakage:
        raise EvaluationDatasetError(
            "sequence leakage detected across splits; fix the manifest or pass "
            "--allow-sequence-leakage only for a non-final diagnostic run"
        )

    confidences = tuple(sorted(set(args.confidence_thresholds)))
    nms_ious = tuple(sorted(set(args.nms_iou_thresholds)))
    image_sizes = tuple(sorted(set(args.image_sizes)))
    match_ious = tuple(sorted(set(args.match_ious)))
    inference_floor = min(0.01, min(confidences))
    detector = PlateDetector(
        weights_path=args.weights,
        device=args.device,
        confidence_threshold=inference_floor,
        iou_threshold=nms_ious[0],
        plate_class_id=args.plate_class_id,
        image_size=image_sizes[0],
    )

    truth = {
        item.image_id: [box.bbox for box in item.boxes]
        for item in manifest.detection_images
    }
    experiments: list[dict] = []
    raw_by_config: dict[tuple[int, float], dict] = {}
    for image_size in image_sizes:
        for nms_iou in nms_ious:
            detector.image_size = image_size
            detector.iou_threshold = nms_iou
            detector.confidence_threshold = inference_floor
            predictions: dict[str, list] = {}
            started = time.perf_counter()
            for frame_number, item in enumerate(manifest.detection_images):
                image = cv2.imread(str(item.path))
                detections = detector.detect(image, frame_number)
                predictions[item.image_id] = [
                    (tuple(float(value) for value in detection.bbox), detection.confidence)
                    for detection in detections
                ]
            elapsed = time.perf_counter() - started
            raw_by_config[(image_size, nms_iou)] = predictions
            for confidence in confidences:
                filtered = {
                    image_id: [(box, score) for box, score in rows if score >= confidence]
                    for image_id, rows in predictions.items()
                }
                for match_iou in match_ious:
                    evaluation = evaluate_detections(truth, filtered, match_iou)
                    experiments.append({
                        "image_size": image_size,
                        "nms_iou": nms_iou,
                        "confidence_threshold": confidence,
                        "metrics": evaluation.metrics.to_dict(),
                        "runtime_seconds": elapsed,
                        "fps": len(manifest.detection_images) / elapsed if elapsed else 0.0,
                    })

    primary_iou = args.primary_match_iou
    primary = [row for row in experiments if row["metrics"]["iou_threshold"] == primary_iou]
    if not primary:
        raise EvaluationDatasetError("primary match IoU must be included in --match-ious")
    selected = max(primary, key=lambda row: (
        row["metrics"]["f1"], row["metrics"]["recall"],
        row["metrics"]["precision"], -row["runtime_seconds"],
    ))
    selected_raw = raw_by_config[(selected["image_size"], selected["nms_iou"])]
    selected_predictions = {
        image_id: [(box, score) for box, score in rows
                   if score >= selected["confidence_threshold"]]
        for image_id, rows in selected_raw.items()
    }
    selected_evaluation = evaluate_detections(truth, selected_predictions, primary_iou)
    match_by_ground_truth = {
        (match.image_id, match.ground_truth_index): match
        for match in selected_evaluation.matches
    }
    ground_truth_outcomes = []
    for item in manifest.detection_images:
        for ground_truth_index, labeled_box in enumerate(item.boxes):
            match = match_by_ground_truth.get((item.image_id, ground_truth_index))
            ground_truth_outcomes.append({
                "image_id": item.image_id,
                "ground_truth_index": ground_truth_index,
                "plate_id": labeled_box.plate_id,
                "bbox": list(labeled_box.bbox),
                "detected": match is not None,
                "matched_iou": match.iou if match is not None else None,
            })
    categories = Counter()
    failure_rows: list[dict] = []
    item_by_id = {item.image_id: item for item in manifest.detection_images}
    examples_dir = Path(args.output_dir) / "detector_failures"
    saved = 0
    for image_id, indices in selected_evaluation.unmatched_ground_truth.items():
        item = item_by_id[image_id]
        image = cv2.imread(str(item.path))
        for ground_truth_index in indices:
            labeled_box = item.boxes[ground_truth_index]
            category = _classify_false_negative(
                image, labeled_box, selected_raw.get(image_id, ()),
                selected_predictions.get(image_id, ()), primary_iou,
                selected["confidence_threshold"],
            )
            categories[category] += 1
            failure_rows.append({
                "image_id": image_id,
                "ground_truth_index": ground_truth_index,
                "bbox": list(labeled_box.bbox),
                "category": category,
                "manual_tags": list(labeled_box.tags),
            })
            if saved < args.max_failure_images:
                examples_dir.mkdir(parents=True, exist_ok=True)
                visual = image.copy()
                x1, y1, x2, y2 = (int(value) for value in labeled_box.bbox)
                cv2.rectangle(visual, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(visual, category, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imwrite(str(examples_dir / f"{image_id}_{ground_truth_index}.jpg"), visual)
                saved += 1

    report = {
        "dataset": manifest.name,
        "dataset_version": manifest.version,
        "weights": str(Path(args.weights).resolve()),
        "weights_sha256": _sha256_file(args.weights),
        "provenance": _provenance(args.manifest),
        "images": len(manifest.detection_images),
        "ground_truth_plates": sum(len(item.boxes) for item in manifest.detection_images),
        "audit": audit,
        "selection_rule": "max_f1_then_recall_then_precision_then_runtime",
        "experiments": experiments,
        "selected_operating_point": selected,
        "selected_ground_truth_outcomes": ground_truth_outcomes,
        "false_negative_categories": dict(sorted(categories.items())),
        "false_negatives": failure_rows,
    }
    _write_reports(Path(args.output_dir), "detector_evaluation", report, _detector_markdown(report))
    return report


def _clean_plate_text(text: str) -> str:
    return "".join(character for character in str(text).upper() if character.isalnum())


def _upscale(image, target_height: int = 48):
    if image.shape[0] >= target_height:
        return image
    scale = target_height / image.shape[0]
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _sharpen(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
    return cv2.addWeighted(image, 1.35, blurred, -0.35, 0)


def _ocr_variants(image, rectifier):
    rectified = rectifier.rectify(image).rectified_crop
    upscaled = _upscale(rectified)
    return {
        "original": image,
        "rectified": rectified,
        "rectified_upscale": upscaled,
        "rectified_clahe": _clahe(rectified),
        "rectified_upscale_sharpen": _sharpen(upscaled),
    }


def _weighted_choice(rows, score_fn):
    scores = defaultdict(float)
    supports = Counter()
    for row in rows:
        if row.text:
            scores[row.text] += score_fn(row)
            supports[row.text] += 1
    if not scores:
        return ""
    return min(scores, key=lambda text: (-scores[text], -supports[text], text))


def _character_vote(rows):
    non_empty = [row for row in rows if row.text]
    if not non_empty:
        return ""
    length_scores = defaultdict(float)
    for row in non_empty:
        length_scores[len(row.text)] += row.ocr_confidence * max(row.quality_score, 0.01)
    length = min(length_scores, key=lambda value: (-length_scores[value], value))
    eligible = [row for row in non_empty if len(row.text) == length]
    output = []
    for position in range(length):
        votes = defaultdict(float)
        for row in eligible:
            votes[row.text[position]] += row.ocr_confidence * max(row.quality_score, 0.01)
        output.append(min(votes, key=lambda char: (-votes[char], char)))
    return "".join(output)


def _fuse(rows, strategy: str) -> str:
    if not rows:
        return ""
    if strategy == "best_frame":
        return rows[0].text
    if strategy == "majority":
        return _weighted_choice(rows, lambda row: 1.0)
    if strategy == "confidence_weighted":
        return _weighted_choice(rows, lambda row: row.ocr_confidence)
    if strategy == "quality_confidence_weighted":
        return _weighted_choice(
            rows,
            lambda row: row.ocr_confidence * max(row.quality_score, 0.01),
        )
    if strategy == "character_position_weighted":
        return _character_vote(rows)
    raise ValueError(f"unknown fusion strategy: {strategy}")


def _score_fused_text(fused, evidence, normalizer, confidence_scorer):
    track_result = TrackResult(
        track_id=0,
        best_text=fused or None,
        candidates=[FusedCandidate(fused, 0.0, len(evidence))] if fused else [],
        evidence=evidence,
    )
    confidence = confidence_scorer.score(track_result, normalizer.normalize(fused))
    # Character-position voting can synthesize a string that no individual OCR
    # crop produced. Preserve that useful candidate, but never auto-accept it
    # without direct supporting evidence.
    decision = confidence.decision
    if fused and not any(row.text == fused for row in evidence):
        decision = "review"
    return confidence.confidence_score, decision


def _ocr_markdown(report: dict) -> str:
    lines = [
        f"# OCR and fusion evaluation — {report['dataset']} ({report['dataset_version']})",
        "",
        "## Single-crop preprocessing ablation",
        "",
        "| Variant | Samples | Exact | Character accuracy | Avg ms/crop |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["preprocessing_experiments"]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['variant']} | {metrics['samples']} | {metrics['exact_accuracy']:.4f} "
            f"| {metrics['character_accuracy']:.4f} | {row['average_milliseconds']:.2f} |"
        )
    lines.extend([
        "",
        "## Track-level fusion",
        "",
        "| Variant | Strategy | K | Tracks | Exact | Character accuracy | Accepted accuracy | Coverage | Review | Abstain |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["fusion_experiments"]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['variant']} | {row['strategy']} | {row['k']} | {metrics['samples']} "
            f"| {metrics['exact_accuracy']:.4f} | {metrics['character_accuracy']:.4f} "
            f"| {row['accepted_accuracy']:.4f} | {row['coverage']:.4f} "
            f"| {row['review_rate']:.4f} | {row['abstention_rate']:.4f} |"
        )
    lines.extend(["", "## Character confusions", ""])
    for pair, count in report["character_confusions"].items():
        lines.append(f"- {pair}: {count}")
    return "\n".join(lines) + "\n"


def run_ocr(args) -> dict:
    # PaddleOCR may initialize native runtimes, so import it only for an actual
    # OCR run (not for manifest auditing or CLI help).
    from phase1_anpr.ocr.plate_ocr import PaddleOCRBackend

    manifest = load_evaluation_manifest(args.manifest)
    if not manifest.ocr_crops:
        raise EvaluationDatasetError("manifest contains no OCR crops")
    audit = audit_manifest(manifest)
    if audit["corrupt_or_unreadable"]:
        raise EvaluationDatasetError("OCR dataset contains unreadable images")
    config = load_config(args.config)
    normalizer = PlateNormalizer.from_config(config)
    confidence_scorer = ConfidenceScorer.from_config(config)
    rectifier = PlateRectifier.from_config(config) if hasattr(PlateRectifier, "from_config") else PlateRectifier()
    model_dir = Path(args.ocr_model_dir).resolve()
    if not model_dir.is_dir():
        raise EvaluationDatasetError(
            f"local OCR model directory does not exist: {model_dir}. "
            "Evaluation never downloads model weights automatically."
        )
    ocr = PlateOCR(PaddleOCRBackend(
        model_name=args.ocr_model_name,
        model_dir=str(model_dir),
    ))
    requested_variants = tuple(args.variants)
    predictions: dict[str, dict[str, OCRResult]] = {variant: {} for variant in requested_variants}
    elapsed_by_variant = Counter()

    for crop in manifest.ocr_crops:
        image = cv2.imread(str(crop.path))
        variants = _ocr_variants(image, rectifier)
        for variant in requested_variants:
            if variant not in variants:
                raise EvaluationDatasetError(f"unknown OCR preprocessing variant: {variant}")
            scored = ScoredCrop(
                crop=variants[variant], quality_score=crop.quality_score,
                detector_confidence=crop.detector_confidence,
                frame_number=crop.frame_number,
            )
            started = time.perf_counter()
            result = ocr.read(scored)
            elapsed_by_variant[variant] += time.perf_counter() - started
            result.text = _clean_plate_text(result.text)
            predictions[variant][crop.crop_id] = result

    preprocessing_rows = []
    for variant in requested_variants:
        metrics = evaluate_ocr(
            (crop.expected_text, predictions[variant][crop.crop_id].text)
            for crop in manifest.ocr_crops
        )
        preprocessing_rows.append({
            "variant": variant,
            "metrics": metrics.to_dict(),
            "runtime_seconds": elapsed_by_variant[variant],
            "average_milliseconds": elapsed_by_variant[variant] * 1000.0 / len(manifest.ocr_crops),
        })

    tracks: dict[str, list[OCRCrop]] = defaultdict(list)
    for crop in manifest.ocr_crops:
        if crop.track_id is not None:
            tracks[crop.track_id].append(crop)
    for track_id, crops in tracks.items():
        if len({crop.expected_text for crop in crops}) != 1:
            raise EvaluationDatasetError(f"track {track_id} contains conflicting ground-truth plates")
        crops.sort(key=lambda crop: (-crop.quality_score, crop.frame_number, crop.crop_id))

    strategies = (
        "best_frame", "majority", "confidence_weighted",
        "quality_confidence_weighted", "character_position_weighted",
    )
    fusion_rows = []
    chosen_predictions: list[tuple[str, str]] = []
    selected_track_outcomes: list[dict] = []
    for variant in requested_variants:
        for strategy in strategies:
            for k in args.k_values:
                pairs = []
                accepted_correct = accepted = review = abstained = 0
                for track_id in sorted(tracks):
                    crop_rows = tracks[track_id][:k]
                    evidence = [predictions[variant][crop.crop_id] for crop in crop_rows]
                    fused = _fuse(evidence, strategy)
                    expected = crop_rows[0].expected_text
                    pairs.append((expected, fused))
                    _, decision = _score_fused_text(
                        fused, evidence, normalizer, confidence_scorer
                    )
                    if decision == "accepted":
                        accepted += 1
                        accepted_correct += expected == fused
                    elif decision == "review":
                        review += 1
                    else:
                        abstained += 1
                metrics = evaluate_ocr(pairs)
                count = len(pairs)
                fusion_rows.append({
                    "variant": variant,
                    "strategy": strategy,
                    "k": k,
                    "metrics": metrics.to_dict(),
                    "accepted_accuracy": accepted_correct / accepted if accepted else 0.0,
                    "coverage": accepted / count if count else 0.0,
                    "review_rate": review / count if count else 0.0,
                    "abstention_rate": abstained / count if count else 0.0,
                })

    best_preprocessing = max(
        preprocessing_rows,
        key=lambda row: (row["metrics"]["exact_accuracy"], row["metrics"]["character_accuracy"],
                         -row["runtime_seconds"]),
    )
    if fusion_rows:
        selected_fusion = max(
            fusion_rows,
            key=lambda row: (row["metrics"]["exact_accuracy"], row["metrics"]["character_accuracy"],
                             row["coverage"], -row["k"]),
        )
        for track_id in sorted(tracks):
            crops = tracks[track_id][:selected_fusion["k"]]
            evidence = [
                predictions[selected_fusion["variant"]][crop.crop_id]
                for crop in crops
            ]
            fused = _fuse(
                evidence,
                selected_fusion["strategy"],
            )
            expected = crops[0].expected_text
            chosen_predictions.append((expected, fused))
            confidence_score, decision = _score_fused_text(
                fused, evidence, normalizer, confidence_scorer
            )
            selected_track_outcomes.append({
                "track_id": track_id,
                "expected_text": expected,
                "predicted_text": fused,
                "correct": expected == fused,
                "decision": decision,
                "confidence": confidence_score,
            })
    else:
        selected_fusion = None
        chosen_predictions = [
            (crop.expected_text, predictions[best_preprocessing["variant"]][crop.crop_id].text)
            for crop in manifest.ocr_crops
        ]

    confusion_counts = Counter(
        f"{expected_char}->{predicted_char}"
        for expected, predicted in chosen_predictions
        for expected_char, predicted_char in aligned_substitutions(expected, predicted)
    )
    failure_categories = Counter()
    best_variant = best_preprocessing["variant"]
    for crop in manifest.ocr_crops:
        predicted = predictions[best_variant][crop.crop_id].text
        if predicted == crop.expected_text:
            continue
        if crop.tags:
            failure_categories[crop.tags[0]] += 1
            continue
        image = cv2.imread(str(crop.path), cv2.IMREAD_GRAYSCALE)
        if image.shape[0] < 20 or image.shape[1] < 60:
            failure_categories["low_resolution"] += 1
        elif float(image.mean()) < 55.0:
            failure_categories["dark_plate"] += 1
        elif float(cv2.Laplacian(image, cv2.CV_64F).var()) < 45.0:
            failure_categories["blur"] += 1
        elif aligned_substitutions(crop.expected_text, predicted):
            failure_categories["character_confusion"] += 1
        else:
            failure_categories["unclassified_manual_review_required"] += 1

    report = {
        "dataset": manifest.name,
        "dataset_version": manifest.version,
        "ocr_engine": "PaddleOCR TextRecognition",
        "ocr_model_name": args.ocr_model_name,
        "ocr_model_dir": str(model_dir),
        "ocr_model_sha256": _sha256_tree(model_dir),
        "provenance": _provenance(args.manifest),
        "ocr_crops": len(manifest.ocr_crops),
        "tracks": len(tracks),
        "audit": audit,
        "preprocessing_experiments": preprocessing_rows,
        "selected_preprocessing": best_preprocessing,
        "fusion_experiments": fusion_rows,
        "selected_fusion": selected_fusion,
        "selected_track_outcomes": selected_track_outcomes,
        "character_confusions": dict(confusion_counts.most_common()),
        "failure_categories": dict(sorted(failure_categories.items())),
        "selection_policy": "exact_accuracy_then_character_accuracy_then_runtime_or_coverage",
    }
    _write_reports(Path(args.output_dir), "ocr_evaluation", report, _ocr_markdown(report))
    return report


def run_audit(args) -> dict:
    manifest = load_evaluation_manifest(args.manifest)
    report = audit_manifest(manifest)
    report["provenance"] = _provenance(args.manifest)
    markdown = "\n".join([
        f"# Dataset audit — {report['dataset']} ({report['version']})",
        "",
        f"- Detection images: {report['detection_images']}",
        f"- Ground-truth plates: {report['ground_truth_plates']}",
        f"- OCR crops: {report['ocr_crops']}",
        f"- Corrupt/unreadable: {len(report['corrupt_or_unreadable'])}",
        f"- Exact duplicate groups: {len(report['exact_duplicate_groups'])}",
        f"- Sequence split leakage groups: {len(report['sequence_split_leakage'])}",
        f"- Suitable for evaluation: {report['suitable_for_evaluation']}",
        "",
    ])
    _write_reports(Path(args.output_dir), "dataset_audit", report, markdown)
    return report


def _end_to_end_markdown(report: dict) -> str:
    metrics = report["metrics"]
    return "\n".join([
        f"# End-to-end ANPR evaluation — {report['dataset']}",
        "",
        f"- Ground-truth tracks: {metrics['ground_truth_tracks']}",
        f"- Detector track coverage: {metrics['detector_track_coverage']:.4f}",
        f"- Track OCR exact accuracy: {metrics['track_ocr_exact_accuracy']:.4f}",
        f"- End-to-end correct tracks: {metrics['end_to_end_correct_tracks']}",
        f"- End-to-end exact accuracy: {metrics['end_to_end_exact_accuracy']:.4f}",
        f"- Accepted-set accuracy: {metrics['accepted_set_accuracy']:.4f}",
        f"- Accepted coverage: {metrics['accepted_coverage']:.4f}",
        f"- Review rate: {metrics['review_rate']:.4f}",
        f"- Abstention rate: {metrics['abstention_rate']:.4f}",
        "",
        "A track is end-to-end correct only when at least one linked detector "
        "ground-truth box was detected and the selected fused OCR text exactly "
        "matches its independent transcription.",
        "",
    ])


def run_end_to_end(args) -> dict:
    detector_path = Path(args.detector_report).resolve()
    ocr_path = Path(args.ocr_report).resolve()
    if not detector_path.is_file() or not ocr_path.is_file():
        raise FileNotFoundError("detector and OCR JSON reports must both exist")
    detector_report = json.loads(detector_path.read_text(encoding="utf-8"))
    ocr_report = json.loads(ocr_path.read_text(encoding="utf-8"))
    detector_hash = detector_report.get("provenance", {}).get("manifest_sha256")
    ocr_hash = ocr_report.get("provenance", {}).get("manifest_sha256")
    if not detector_hash or detector_hash != ocr_hash:
        raise EvaluationDatasetError(
            "detector and OCR reports must come from the same manifest hash"
        )

    detector_outcomes = detector_report.get("selected_ground_truth_outcomes", [])
    ocr_outcomes = ocr_report.get("selected_track_outcomes", [])
    if not detector_outcomes or not ocr_outcomes:
        raise EvaluationDatasetError(
            "reports do not contain linked detector/OCR outcomes; rerun both evaluators"
        )
    if any(not row.get("plate_id") for row in detector_outcomes):
        raise EvaluationDatasetError(
            "every detector ground-truth box needs plate_id for end-to-end evaluation"
        )
    ocr_by_track = {row["track_id"]: row for row in ocr_outcomes}
    if len(ocr_by_track) != len(ocr_outcomes):
        raise EvaluationDatasetError("OCR report contains duplicate track ids")

    detected_by_track = defaultdict(bool)
    for row in detector_outcomes:
        detected_by_track[row["plate_id"]] |= bool(row["detected"])
    ground_truth_tracks = sorted(detected_by_track)
    missing_ocr = [track_id for track_id in ground_truth_tracks if track_id not in ocr_by_track]
    extra_ocr = sorted(set(ocr_by_track) - set(ground_truth_tracks))
    if missing_ocr or extra_ocr:
        raise EvaluationDatasetError(
            f"detector/OCR track linkage mismatch; missing OCR={missing_ocr}, extra OCR={extra_ocr}"
        )

    rows = []
    for track_id in ground_truth_tracks:
        ocr_row = ocr_by_track[track_id]
        detected = detected_by_track[track_id]
        correct = detected and bool(ocr_row["correct"])
        rows.append({
            "track_id": track_id,
            "detected": detected,
            "expected_text": ocr_row["expected_text"],
            "predicted_text": ocr_row["predicted_text"],
            "ocr_correct": bool(ocr_row["correct"]),
            "end_to_end_correct": correct,
            "decision": ocr_row["decision"] if detected else "abstained",
        })

    count = len(rows)
    detected_count = sum(row["detected"] for row in rows)
    ocr_correct = sum(row["ocr_correct"] for row in rows)
    end_to_end_correct = sum(row["end_to_end_correct"] for row in rows)
    accepted_rows = [row for row in rows if row["decision"] == "accepted"]
    review = sum(row["decision"] == "review" for row in rows)
    abstained = sum(row["decision"] == "abstained" for row in rows)
    metrics = {
        "ground_truth_tracks": count,
        "detected_tracks": detected_count,
        "detector_track_coverage": detected_count / count if count else 0.0,
        "track_ocr_exact_accuracy": ocr_correct / count if count else 0.0,
        "end_to_end_correct_tracks": end_to_end_correct,
        "end_to_end_exact_accuracy": end_to_end_correct / count if count else 0.0,
        "accepted_tracks": len(accepted_rows),
        "accepted_set_accuracy": (
            sum(row["end_to_end_correct"] for row in accepted_rows) / len(accepted_rows)
            if accepted_rows else 0.0
        ),
        "accepted_coverage": len(accepted_rows) / count if count else 0.0,
        "review_rate": review / count if count else 0.0,
        "abstention_rate": abstained / count if count else 0.0,
    }
    report = {
        "dataset": detector_report.get("dataset", "unknown"),
        "dataset_version": detector_report.get("dataset_version", "unknown"),
        "manifest_sha256": detector_hash,
        "detector_report_sha256": _sha256_file(detector_path),
        "ocr_report_sha256": _sha256_file(ocr_path),
        "metrics": metrics,
        "track_outcomes": rows,
    }
    _write_reports(
        Path(args.output_dir), "end_to_end_evaluation", report,
        _end_to_end_markdown(report),
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CitySight label-driven ANPR evaluator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="validate manifest integrity and split leakage")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--output-dir", default="outputs/annotated/evaluation")
    audit.set_defaults(handler=run_audit)

    detector = subparsers.add_parser("detector", help="run detector threshold/configuration sweep")
    detector.add_argument("--manifest", required=True)
    detector.add_argument("--weights", default="weights/plate_detector.pt")
    detector.add_argument("--device", default="cpu")
    detector.add_argument("--plate-class-id", type=int, default=0)
    detector.add_argument("--confidence-thresholds", type=lambda value: _parse_numbers(value, float),
                          default=(0.15, 0.25, 0.35, 0.45))
    detector.add_argument("--nms-iou-thresholds", type=lambda value: _parse_numbers(value, float),
                          default=(0.45, 0.60))
    detector.add_argument("--image-sizes", type=lambda value: _parse_numbers(value, int),
                          default=(640, 960))
    detector.add_argument("--match-ious", type=lambda value: _parse_numbers(value, float),
                          default=(0.3, 0.5))
    detector.add_argument("--primary-match-iou", type=float, default=0.5)
    detector.add_argument("--max-failure-images", type=int, default=30)
    detector.add_argument("--allow-sequence-leakage", action="store_true")
    detector.add_argument("--output-dir", default="outputs/annotated/evaluation")
    detector.set_defaults(handler=run_detector)

    ocr = subparsers.add_parser("ocr", help="run OCR preprocessing and track-fusion ablations")
    ocr.add_argument("--manifest", required=True)
    ocr.add_argument("--config", default="phase1_anpr/config/config.yaml")
    ocr.add_argument("--ocr-model-dir", required=True,
                     help="existing local PaddleOCR recognition model directory")
    ocr.add_argument("--ocr-model-name", default="PP-OCRv6_medium_rec")
    ocr.add_argument("--variants", type=lambda value: _parse_numbers(value, str),
                     default=("original", "rectified", "rectified_upscale",
                              "rectified_clahe", "rectified_upscale_sharpen"))
    ocr.add_argument("--k-values", type=lambda value: _parse_numbers(value, int),
                     default=(1, 3, 5))
    ocr.add_argument("--output-dir", default="outputs/annotated/evaluation")
    ocr.set_defaults(handler=run_ocr)

    end_to_end = subparsers.add_parser(
        "end-to-end", help="combine linked detector and OCR reports"
    )
    end_to_end.add_argument("--detector-report", required=True)
    end_to_end.add_argument("--ocr-report", required=True)
    end_to_end.add_argument("--output-dir", default="outputs/annotated/evaluation")
    end_to_end.set_defaults(handler=run_end_to_end)

    real_world = subparsers.add_parser(
        "real-world",
        help="evaluate fixed real-video predictions against manual Step 34 labels",
    )
    real_world.add_argument("--ground-truth", required=True)
    real_world.add_argument("--predictions", required=True)
    real_world.add_argument("--primary-iou", type=float, default=0.5)
    real_world.add_argument("--diagnostic-iou", type=float, default=0.3)
    real_world.add_argument(
        "--cross-camera-k", type=lambda value: _parse_numbers(value, int),
        default=(1, 3, 5),
    )
    real_world.add_argument("--minimum-cross-camera-cases", type=int, default=2)
    real_world.add_argument(
        "--output-dir", default="outputs/annotated/step34_evaluation")

    def _run_real_world(args):
        from phase1_anpr.evaluation.real_world import run_real_world_evaluation

        return run_real_world_evaluation(
            args.ground_truth,
            args.predictions,
            args.output_dir,
            primary_iou=args.primary_iou,
            diagnostic_iou=args.diagnostic_iou,
            cross_camera_k=tuple(args.cross_camera_k),
            minimum_cross_camera_cases=args.minimum_cross_camera_cases,
        )

    real_world.set_defaults(handler=_run_real_world)

    capture = subparsers.add_parser(
        "real-world-capture",
        help="capture fixed Step 34 detector/replay predictions on reviewed frames",
    )
    capture.add_argument("--ground-truth", required=True)
    capture.add_argument("--scenario", required=True)
    capture.add_argument("--city-config", required=True)
    capture.add_argument("--config", required=True)
    capture.add_argument(
        "--output", default="outputs/annotated/step34_predictions.json")
    capture.add_argument("--candidate-window-seconds", type=float)
    capture.add_argument("--candidate-max-results", type=int)
    capture.add_argument("--candidate-maximum-speed-kph", type=float)

    def _capture_real_world(args):
        from phase1_anpr.evaluation.real_world import capture_real_world_predictions

        return capture_real_world_predictions(
            args.ground_truth,
            args.scenario,
            args.city_config,
            args.config,
            args.output,
            candidate_window_seconds=args.candidate_window_seconds,
            candidate_max_results=args.candidate_max_results,
            candidate_maximum_speed_kph=args.candidate_maximum_speed_kph,
        )

    capture.set_defaults(handler=_capture_real_world)

    prepare = subparsers.add_parser(
        "real-world-prepare",
        help="extract processed Step 34 frames and create an unreviewed label file",
    )
    prepare.add_argument("--scenario", required=True)
    prepare.add_argument("--city-config", required=True)
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--frames-dir")
    prepare.add_argument(
        "--benchmark-id", default="step34-step33-real-videos-v1")

    def _prepare_real_world(args):
        from phase1_anpr.evaluation.real_world import (
            prepare_real_world_annotation_workspace,
        )

        return prepare_real_world_annotation_workspace(
            args.scenario,
            args.city_config,
            args.config,
            args.output,
            benchmark_id=args.benchmark_id,
            frames_dir=args.frames_dir,
        )

    prepare.set_defaults(handler=_prepare_real_world)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = args.handler(args)
    except (EvaluationDatasetError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
