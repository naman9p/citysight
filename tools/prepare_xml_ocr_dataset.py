"""Prepare XML-labelled Indian vehicle images for CitySight OCR evaluation.

This utility targets Pascal-VOC-style XML annotations where each <object>:
- uses <name> for the ground-truth plate text;
- contains a <bndbox> with xmin/ymin/xmax/ymax.

It extracts plate crops without modifying the source dataset, assigns deterministic
label-grouped train/val/test splits to reduce leakage, and writes CitySight
evaluation manifests plus an audit CSV/JSON report.

Important:
- OCR ground truth is the annotation's plate text, canonicalized to A-Z/0-9.
- Indian-format validation is reported, not used to silently rewrite labels.
- Multiple objects per image are supported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import yaml

from phase1_anpr.normalization.plate_normalizer import PlateNormalizer


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _canonical_label(text: str) -> str:
    return "".join(ch for ch in (text or "").upper() if ch.isalnum())


def _split_for_label(
    label: str,
    *,
    seed: str,
    train_ratio: float,
    val_ratio: float,
) -> str:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def _find_matching_image(xml_path: Path, annotation_root: ET.Element) -> Path | None:
    filename = (annotation_root.findtext("filename") or "").strip()
    if filename:
        direct = xml_path.parent / filename
        if direct.is_file():
            return direct

    stem = xml_path.stem
    # Some annotations are named image.jpg.xml while the image is image.jpg.jpeg.
    candidates = sorted(
        path
        for path in xml_path.parent.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and (path.stem == stem or path.name.startswith(stem + "."))
    )
    if candidates:
        return candidates[0]

    # Last fallback: strip one embedded extension from the XML stem.
    lowered = stem.lower()
    for embedded in IMAGE_SUFFIXES:
        if lowered.endswith(embedded):
            base = stem[: -len(embedded)]
            candidates = sorted(
                path
                for path in xml_path.parent.iterdir()
                if path.is_file()
                and path.suffix.lower() in IMAGE_SUFFIXES
                and (path.stem == base or path.name.startswith(base + "."))
            )
            if candidates:
                return candidates[0]
    return None


def _parse_box(node: ET.Element) -> tuple[int, int, int, int] | None:
    box = node.find("bndbox")
    if box is None:
        return None
    try:
        xmin = int(round(float(box.findtext("xmin", ""))))
        ymin = int(round(float(box.findtext("ymin", ""))))
        xmax = int(round(float(box.findtext("xmax", ""))))
        ymax = int(round(float(box.findtext("ymax", ""))))
    except ValueError:
        return None
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def _manifest(name: str, version: str, rows: list[dict]) -> dict:
    return {
        "name": name,
        "version": version,
        "detection": {"images": []},
        "ocr": {"crops": rows},
    }


def _write_manifest(path: Path, payload: dict) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract OCR crops + CitySight manifests from plate-text XML annotations."
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/ocr_xml_dataset",
        help="Output directory for crops, manifests, and audit reports.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument(
        "--seed",
        default="citysight-indian-vehicle-ocr-v1",
        help="Stable split seed. Identical canonical plate labels stay in one split.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="Fractional bbox padding on each side, e.g. 0.05 for 5%%.",
    )
    parser.add_argument(
        "--valid-indian-only",
        action="store_true",
        help="Exclude labels that fail CitySight's current standard/BH plate validator.",
    )
    args = parser.parse_args()

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(value < 0.0 for value in ratios):
        parser.error("split ratios must be non-negative")
    if abs(sum(ratios) - 1.0) > 1e-9:
        parser.error("train/val/test ratios must sum to 1.0")
    if args.padding < 0.0 or args.padding > 0.5:
        parser.error("--padding must be in [0.0, 0.5]")

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    crop_root = output_dir / "crops"
    if not dataset_dir.is_dir():
        parser.error(f"dataset directory does not exist: {dataset_dir}")

    xml_paths = sorted(dataset_dir.rglob("*.xml"))
    if not xml_paths:
        parser.error(f"no XML annotations found under: {dataset_dir}")

    normalizer = PlateNormalizer()
    counters = Counter()
    audit_rows: list[dict] = []
    manifest_rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    labels_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    version_hasher = hashlib.sha256()

    for xml_index, xml_path in enumerate(xml_paths):
        counters["xml_files"] += 1
        try:
            root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError) as exc:
            counters["xml_parse_errors"] += 1
            audit_rows.append({
                "xml_path": str(xml_path),
                "image_path": "",
                "object_index": "",
                "raw_label": "",
                "label": "",
                "valid_indian": "",
                "format_type": "",
                "state_code": "",
                "bbox": "",
                "split": "",
                "crop_path": "",
                "status": f"xml_parse_error:{type(exc).__name__}",
            })
            continue

        image_path = _find_matching_image(xml_path, root)
        if image_path is None:
            counters["xml_without_image"] += 1
            audit_rows.append({
                "xml_path": str(xml_path),
                "image_path": "",
                "object_index": "",
                "raw_label": "",
                "label": "",
                "valid_indian": "",
                "format_type": "",
                "state_code": "",
                "bbox": "",
                "split": "",
                "crop_path": "",
                "status": "missing_image",
            })
            continue

        image = cv2.imread(str(image_path))
        if image is None or image.size == 0:
            counters["unreadable_images"] += 1
            continue
        height, width = image.shape[:2]
        objects = root.findall("object")
        if not objects:
            counters["xml_without_objects"] += 1

        for object_index, obj in enumerate(objects):
            counters["objects"] += 1
            raw_label = (obj.findtext("name") or "").strip()
            label = _canonical_label(raw_label)
            bbox = _parse_box(obj)
            if not label:
                counters["empty_labels"] += 1
                status = "empty_label"
                valid_indian = False
                fmt = ""
                state_code = ""
                split = ""
                crop_path = ""
            elif bbox is None:
                counters["invalid_boxes"] += 1
                status = "invalid_bbox"
                normalized = normalizer.normalize(label)
                valid_indian = bool(normalized.is_valid)
                fmt = normalized.format_type or ""
                state_code = normalized.state_code or ""
                split = ""
                crop_path = ""
            else:
                normalized = normalizer.normalize(label)
                valid_indian = bool(normalized.is_valid)
                fmt = normalized.format_type or ""
                state_code = normalized.state_code or ""
                counters["valid_indian_labels"] += int(valid_indian)
                counters["nonmatching_current_indian_format"] += int(not valid_indian)
                if fmt:
                    counters[f"format_{fmt}"] += 1

                if args.valid_indian_only and not valid_indian:
                    counters["excluded_invalid_indian"] += 1
                    status = "excluded_invalid_indian"
                    split = ""
                    crop_path = ""
                else:
                    xmin, ymin, xmax, ymax = bbox
                    box_w, box_h = xmax - xmin, ymax - ymin
                    pad_x = int(round(box_w * args.padding))
                    pad_y = int(round(box_h * args.padding))
                    x1 = max(0, xmin - pad_x)
                    y1 = max(0, ymin - pad_y)
                    x2 = min(width, xmax + pad_x)
                    y2 = min(height, ymax + pad_y)
                    if x2 <= x1 or y2 <= y1:
                        counters["clipped_empty_boxes"] += 1
                        status = "clipped_empty_bbox"
                        split = ""
                        crop_path = ""
                    else:
                        crop = image[y1:y2, x1:x2]
                        split = _split_for_label(
                            label,
                            seed=args.seed,
                            train_ratio=args.train_ratio,
                            val_ratio=args.val_ratio,
                        )
                        labels_by_split[split].add(label)
                        relative_xml = xml_path.relative_to(dataset_dir).as_posix()
                        stable_id = hashlib.sha1(
                            f"{relative_xml}|{object_index}|{label}".encode("utf-8")
                        ).hexdigest()[:16]
                        crop_dir = crop_root / split
                        crop_dir.mkdir(parents=True, exist_ok=True)
                        crop_file = crop_dir / f"{stable_id}.png"
                        if not cv2.imwrite(str(crop_file), crop):
                            counters["crop_write_errors"] += 1
                            status = "crop_write_error"
                            crop_path = ""
                            split = ""
                        else:
                            counters[f"included_{split}"] += 1
                            counters["included_total"] += 1
                            crop_path = str(crop_file.resolve())
                            status = "included"
                            tag = fmt if valid_indian and fmt else "annotation_text_nonstandard"
                            manifest_rows[split].append({
                                "id": stable_id,
                                "path": crop_path,
                                "text": label,
                                # Standalone still images are not temporal repeats. A
                                # unique track prevents artificial fusion across unrelated
                                # photos sharing the same plate text.
                                "track_id": stable_id,
                                "frame_number": 0,
                                "quality_score": 0.0,
                                "detector_confidence": 0.0,
                                "tags": [tag],
                            })
                            version_hasher.update(
                                f"{relative_xml}|{object_index}|{label}|{bbox}\n".encode("utf-8")
                            )

            audit_rows.append({
                "xml_path": str(xml_path),
                "image_path": str(image_path),
                "object_index": object_index,
                "raw_label": raw_label,
                "label": label,
                "valid_indian": valid_indian,
                "format_type": fmt,
                "state_code": state_code,
                "bbox": "" if bbox is None else ",".join(map(str, bbox)),
                "split": split,
                "crop_path": crop_path,
                "status": status,
            })

    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = labels_by_split[first] & labels_by_split[second]
        if overlap:
            raise RuntimeError(
                f"label leakage detected between {first} and {second}: {sorted(overlap)[:10]}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    version = "indian-vehicle-xml-" + version_hasher.hexdigest()[:16]

    for split in ("train", "val", "test"):
        _write_manifest(
            output_dir / f"{split}.yaml",
            _manifest(
                f"indian-vehicle-xml-{split}",
                version,
                manifest_rows[split],
            ),
        )

    with (output_dir / "annotation_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "xml_path",
                "image_path",
                "object_index",
                "raw_label",
                "label",
                "valid_indian",
                "format_type",
                "state_code",
                "bbox",
                "split",
                "crop_path",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(audit_rows)

    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "version": version,
        "padding": args.padding,
        "valid_indian_only": bool(args.valid_indian_only),
        "counts": dict(sorted(counters.items())),
        "unique_labels": {
            split: len(labels_by_split[split])
            for split in ("train", "val", "test")
        },
        "manifests": {
            split: str((output_dir / f"{split}.yaml").resolve())
            for split in ("train", "val", "test")
        },
        "audit_csv": str((output_dir / "annotation_audit.csv").resolve()),
        "notes": [
            "Ground truth comes from XML <object><name> and is canonicalized only to uppercase alphanumeric text.",
            "Bounding boxes come from XML <bndbox>; source images are never modified.",
            "Labels that do not match CitySight's current standard/BH regex remain included by default and are tagged for audit.",
            "Identical canonical plate labels are deterministically kept in a single split.",
            "quality_score and detector_confidence are zero because these crops come from external annotations, not CitySight detections.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    if counters["included_total"] == 0:
        print("\nNo OCR crops were prepared. Inspect annotation_audit.csv.")
        return 2
    print("\nPrepared OCR dataset successfully.")
    print(f"Test manifest: {output_dir / 'test.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
