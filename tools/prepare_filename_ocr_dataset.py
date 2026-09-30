"""Prepare filename-labeled OCR data for CitySight evaluation.

The Umar/Kaggle "Indian License Plate Images" dataset documents the image
filename stem as the OCR label. This utility:
- scans a directory of plate crops;
- canonicalizes filename stems to A-Z/0-9 labels;
- validates them with CitySight's Indian/BH PlateNormalizer;
- keeps the Indian-format subset by default;
- groups identical labels into the same deterministic split to avoid leakage;
- writes separate train/validation/test evaluation manifests.

It never modifies the source dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from phase1_anpr.normalization.plate_normalizer import PlateNormalizer


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _canonical_label(stem: str) -> str:
    return "".join(ch for ch in stem.upper() if ch.isalnum())


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
        description=(
            "Prepare filename-labelled plate crops for CitySight OCR evaluation. "
            "Identical labels are kept in the same split."
        )
    )
    parser.add_argument("--dataset-dir", required=True, help="Directory containing plate crop images.")
    parser.add_argument(
        "--output-dir",
        default="outputs/ocr_filename_dataset",
        help="Directory for manifests and reports.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument(
        "--seed",
        default="citysight-filename-ocr-v1",
        help="Stable split seed; changing it changes the label-level split.",
    )
    parser.add_argument(
        "--include-non-indian",
        action="store_true",
        help=(
            "Also include filename labels that do not validate as supported Indian/BH "
            "plates. Not recommended for the SIH Indian-plate benchmark."
        ),
    )
    args = parser.parse_args()

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(value < 0.0 for value in ratios):
        parser.error("split ratios must be non-negative")
    if abs(sum(ratios) - 1.0) > 1e-9:
        parser.error("train/val/test ratios must sum to 1.0")

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        parser.error(f"dataset directory does not exist: {dataset_dir}")

    images = sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        parser.error(f"no supported images found under: {dataset_dir}")

    normalizer = PlateNormalizer()
    rows_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    inspection_rows: list[dict] = []
    counters = Counter()
    labels_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

    for index, path in enumerate(images):
        label = _canonical_label(path.stem)
        normalized = normalizer.normalize(label)
        valid_indian = bool(normalized.is_valid)
        format_type = normalized.format_type or ""
        state_code = normalized.state_code or ""

        if not label:
            counters["empty_label"] += 1
            split = ""
            included = False
        else:
            split = _split_for_label(
                label,
                seed=args.seed,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
            )
            included = valid_indian or args.include_non_indian

        counters["total_images"] += 1
        counters["valid_indian"] += int(valid_indian)
        counters["invalid_indian"] += int(not valid_indian)
        if format_type:
            counters[f"format_{format_type}"] += 1
        if included:
            counters[f"included_{split}"] += 1
            labels_by_split[split].add(label)
            crop_id = f"{split}_{index:06d}_{hashlib.sha1(str(path).encode('utf-8')).hexdigest()[:10]}"
            # This is a standalone crop dataset, not a temporal track dataset.
            # A unique track_id prevents unrelated images of the same textual plate
            # from being treated as multi-frame evidence by the fusion benchmark.
            manifest_row = {
                "id": crop_id,
                "path": str(path),
                "text": label,
                "track_id": crop_id,
                "frame_number": 0,
                "quality_score": 0.0,
                "detector_confidence": 0.0,
                "tags": [format_type] if format_type else ["non_indian_format"],
            }
            rows_by_split[split].append(manifest_row)

        inspection_rows.append(
            {
                "path": str(path),
                "filename": path.name,
                "label": label,
                "valid_indian": valid_indian,
                "format_type": format_type,
                "state_code": state_code,
                "split": split,
                "included": included,
            }
        )

    # Leakage guard: one textual identity must never appear in multiple splits.
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = labels_by_split[first] & labels_by_split[second]
        if overlap:
            raise RuntimeError(
                f"split leakage detected between {first} and {second}: "
                f"{sorted(overlap)[:10]}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    version_digest = hashlib.sha256(
        "\n".join(f"{row['path']}|{row['label']}" for row in inspection_rows).encode("utf-8")
    ).hexdigest()[:16]
    version = f"umar-final-licence-{version_digest}"

    for split in ("train", "val", "test"):
        _write_manifest(
            output_dir / f"{split}.yaml",
            _manifest(
                f"umar-final-licence-{split}",
                version,
                rows_by_split[split],
            ),
        )

    with (output_dir / "filename_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "filename",
                "label",
                "valid_indian",
                "format_type",
                "state_code",
                "split",
                "included",
            ],
        )
        writer.writeheader()
        writer.writerows(inspection_rows)

    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "version": version,
        "include_non_indian": bool(args.include_non_indian),
        "counts": dict(sorted(counters.items())),
        "unique_labels": {
            split: len(labels_by_split[split])
            for split in ("train", "val", "test")
        },
        "manifests": {
            split: str(output_dir / f"{split}.yaml")
            for split in ("train", "val", "test")
        },
        "note": (
            "Filename stems are treated as ground truth according to the source dataset's "
            "published description. For the SIH Indian-plate benchmark, only labels accepted "
            "by CitySight's standard/BH normalizer are included unless --include-non-indian "
            "is explicitly supplied. quality_score and detector_confidence are zero because "
            "this is a crop-only OCR dataset; do not interpret accepted/review/abstain metrics "
            "from these manifests as production confidence calibration."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    if not any(rows_by_split.values()):
        print(
            "\nNo usable OCR crops were included. Inspect filename_audit.csv before "
            "attempting an OCR benchmark."
        )
        return 2

    print("\nPrepared manifests successfully.")
    print(f"Test manifest: {output_dir / 'test.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
