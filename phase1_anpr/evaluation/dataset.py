"""Strict manifests and integrity checks for label-driven ANPR evaluation."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import yaml

from phase1_anpr.evaluation.metrics import BBox


class EvaluationDatasetError(ValueError):
    """Raised when an evaluation manifest is unsafe or internally invalid."""


@dataclass(frozen=True)
class LabeledBox:
    bbox: BBox
    plate_id: Optional[str] = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class DetectionImage:
    image_id: str
    path: Path
    sequence_id: str
    split: str
    boxes: tuple[LabeledBox, ...]


@dataclass(frozen=True)
class OCRCrop:
    crop_id: str
    path: Path
    expected_text: str
    track_id: Optional[str]
    frame_number: int
    quality_score: float
    detector_confidence: float
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationManifest:
    name: str
    version: str
    detection_images: tuple[DetectionImage, ...]
    ocr_crops: tuple[OCRCrop, ...]


def _mapping(value, context: str) -> dict:
    if not isinstance(value, dict):
        raise EvaluationDatasetError(f"{context} must be a mapping")
    return value


def _list(value, context: str) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise EvaluationDatasetError(f"{context} must be a list")
    return value


def _text(value, context: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, (str, int)):
        raise EvaluationDatasetError(f"{context} must be a string")
    result = str(value).strip()
    if not result and not allow_empty:
        raise EvaluationDatasetError(f"{context} must not be empty")
    return result


def _tags(value, context: str) -> tuple[str, ...]:
    return tuple(_text(tag, f"{context} tag") for tag in _list(value, context))


def _resolve_file(base: Path, value, context: str) -> Path:
    path = Path(_text(value, context))
    path = path if path.is_absolute() else base / path
    path = path.resolve()
    if not path.is_file():
        raise EvaluationDatasetError(f"{context} does not exist: {path}")
    return path


def _bbox(value, context: str) -> BBox:
    if not isinstance(value, list) or len(value) != 4:
        raise EvaluationDatasetError(f"{context} must be [x1, y1, x2, y2]")
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise EvaluationDatasetError(f"{context} contains a non-numeric value") from exc
    if min(x1, y1) < 0 or x2 <= x1 or y2 <= y1:
        raise EvaluationDatasetError(f"{context} has invalid coordinates: {value!r}")
    return x1, y1, x2, y2


def load_evaluation_manifest(path) -> EvaluationManifest:
    """Load a combined detector/OCR manifest with paths relative to its file."""
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"evaluation manifest not found: {manifest_path}")
    try:
        root = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise EvaluationDatasetError(f"invalid evaluation YAML: {exc}") from exc
    root = _mapping(root, "evaluation manifest")
    unknown = set(root) - {"name", "version", "detection", "ocr"}
    if unknown:
        raise EvaluationDatasetError(f"unknown manifest field(s): {', '.join(sorted(unknown))}")

    base = manifest_path.parent
    detection = _mapping(root.get("detection", {}), "detection")
    ocr = _mapping(root.get("ocr", {}), "ocr")
    if set(detection) - {"images"}:
        raise EvaluationDatasetError("detection contains unknown fields")
    if set(ocr) - {"crops"}:
        raise EvaluationDatasetError("ocr contains unknown fields")

    detection_images: list[DetectionImage] = []
    seen_image_ids: set[str] = set()
    for index, raw in enumerate(_list(detection.get("images"), "detection.images")):
        item = _mapping(raw, f"detection.images[{index}]")
        unknown = set(item) - {"id", "path", "sequence_id", "split", "boxes"}
        if unknown:
            raise EvaluationDatasetError(f"detection.images[{index}] has unknown fields: {sorted(unknown)}")
        image_id = _text(item.get("id"), f"detection.images[{index}].id")
        if image_id in seen_image_ids:
            raise EvaluationDatasetError(f"duplicate detection image id: {image_id}")
        seen_image_ids.add(image_id)
        boxes: list[LabeledBox] = []
        for box_index, raw_box in enumerate(_list(item.get("boxes"), f"{image_id}.boxes")):
            box = _mapping(raw_box, f"{image_id}.boxes[{box_index}]")
            if set(box) - {"bbox", "plate_id", "tags"}:
                raise EvaluationDatasetError(f"{image_id}.boxes[{box_index}] has unknown fields")
            plate_id = box.get("plate_id")
            boxes.append(LabeledBox(
                bbox=_bbox(box.get("bbox"), f"{image_id}.boxes[{box_index}].bbox"),
                plate_id=_text(plate_id, f"{image_id}.boxes[{box_index}].plate_id")
                if plate_id is not None else None,
                tags=_tags(box.get("tags"), f"{image_id}.boxes[{box_index}].tags"),
            ))
        detection_images.append(DetectionImage(
            image_id=image_id,
            path=_resolve_file(base, item.get("path"), f"{image_id}.path"),
            sequence_id=_text(item.get("sequence_id", image_id), f"{image_id}.sequence_id"),
            split=_text(item.get("split", "test"), f"{image_id}.split").lower(),
            boxes=tuple(boxes),
        ))

    ocr_crops: list[OCRCrop] = []
    seen_crop_ids: set[str] = set()
    for index, raw in enumerate(_list(ocr.get("crops"), "ocr.crops")):
        item = _mapping(raw, f"ocr.crops[{index}]")
        unknown = set(item) - {
            "id", "path", "text", "track_id", "frame_number",
            "quality_score", "detector_confidence", "tags",
        }
        if unknown:
            raise EvaluationDatasetError(f"ocr.crops[{index}] has unknown fields: {sorted(unknown)}")
        crop_id = _text(item.get("id"), f"ocr.crops[{index}].id")
        if crop_id in seen_crop_ids:
            raise EvaluationDatasetError(f"duplicate OCR crop id: {crop_id}")
        seen_crop_ids.add(crop_id)
        expected = re.sub(r"[^A-Z0-9]", "", _text(item.get("text"), f"{crop_id}.text").upper())
        if not expected:
            raise EvaluationDatasetError(f"{crop_id}.text has no alphanumeric ground truth")
        track_value = item.get("track_id")
        ocr_crops.append(OCRCrop(
            crop_id=crop_id,
            path=_resolve_file(base, item.get("path"), f"{crop_id}.path"),
            expected_text=expected,
            track_id=_text(track_value, f"{crop_id}.track_id") if track_value is not None else None,
            frame_number=int(item.get("frame_number", index)),
            quality_score=float(item.get("quality_score", 0.0)),
            detector_confidence=float(item.get("detector_confidence", 0.0)),
            tags=_tags(item.get("tags"), f"{crop_id}.tags"),
        ))

    return EvaluationManifest(
        name=_text(root.get("name", manifest_path.stem), "name"),
        version=_text(root.get("version", "unversioned"), "version"),
        detection_images=tuple(detection_images),
        ocr_crops=tuple(ocr_crops),
    )


def audit_manifest(manifest: EvaluationManifest) -> dict:
    """Return corruption, duplicate, and sequence-leakage diagnostics."""
    corrupt: list[str] = []
    duplicate_hashes: dict[str, list[str]] = {}
    dimensions: dict[str, tuple[int, int]] = {}
    image_statistics: dict[str, tuple[int, int, float]] = {}
    for item_id, path in [
        *((item.image_id, item.path) for item in manifest.detection_images),
        *((item.crop_id, item.path) for item in manifest.ocr_crops),
    ]:
        image = cv2.imread(str(path))
        if image is None or image.size == 0:
            corrupt.append(item_id)
            continue
        dimensions[item_id] = (int(image.shape[1]), int(image.shape[0]))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        image_statistics[item_id] = (
            int(image.shape[1]), int(image.shape[0]), float(gray.mean())
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        duplicate_hashes.setdefault(digest, []).append(item_id)

    duplicates = [ids for ids in duplicate_hashes.values() if len(ids) > 1]
    sequence_splits: dict[str, set[str]] = {}
    for item in manifest.detection_images:
        sequence_splits.setdefault(item.sequence_id, set()).add(item.split)
    leakage = {
        sequence_id: sorted(splits)
        for sequence_id, splits in sequence_splits.items()
        if len(splits) > 1
    }
    detection_by_id = {item.image_id: item for item in manifest.detection_images}
    boxes_outside_images: list[dict] = []
    small_objects = 0
    very_small_objects = 0
    brightness_buckets = Counter()
    split_counts = Counter(item.split for item in manifest.detection_images)
    for image_id, item in detection_by_id.items():
        stats = image_statistics.get(image_id)
        if not stats:
            continue
        width, height, brightness = stats
        if brightness < 55.0:
            brightness_buckets["dark"] += 1
        elif brightness > 200.0:
            brightness_buckets["bright"] += 1
        else:
            brightness_buckets["normal"] += 1
        for box_index, box in enumerate(item.boxes):
            x1, y1, x2, y2 = box.bbox
            if x2 > width or y2 > height:
                boxes_outside_images.append({
                    "image_id": image_id,
                    "box_index": box_index,
                    "bbox": list(box.bbox),
                    "image_size": [width, height],
                })
            area_ratio = (x2 - x1) * (y2 - y1) / max(1, width * height)
            if area_ratio < 0.005:
                small_objects += 1
            if area_ratio < 0.0015 or (y2 - y1) < 16:
                very_small_objects += 1

    duplicate_across_splits: list[dict] = []
    split_by_id = {item.image_id: item.split for item in manifest.detection_images}
    for ids in duplicates:
        splits = sorted({split_by_id[item_id] for item_id in ids if item_id in split_by_id})
        if len(splits) > 1:
            duplicate_across_splits.append({"ids": sorted(ids), "splits": splits})
    plate_count = sum(len(item.boxes) for item in manifest.detection_images)
    return {
        "dataset": manifest.name,
        "version": manifest.version,
        "detection_images": len(manifest.detection_images),
        "ground_truth_plates": plate_count,
        "ocr_crops": len(manifest.ocr_crops),
        "split_counts": dict(sorted(split_counts.items())),
        "sequences": len(sequence_splits),
        "corrupt_or_unreadable": sorted(corrupt),
        "exact_duplicate_groups": sorted(duplicates),
        "exact_duplicates_across_splits": duplicate_across_splits,
        "sequence_split_leakage": leakage,
        "boxes_outside_images": boxes_outside_images,
        "small_object_plates": small_objects,
        "very_small_object_plates": very_small_objects,
        "small_object_fraction": small_objects / plate_count if plate_count else 0.0,
        "very_small_object_fraction": very_small_objects / plate_count if plate_count else 0.0,
        "brightness_buckets": dict(sorted(brightness_buckets.items())),
        "dimensions": dimensions,
        "suitable_for_evaluation": not (
            corrupt or leakage or boxes_outside_images or duplicate_across_splits
        ),
    }
