"""Tests for Step 3 plate detection that do not need real YOLO weights."""

import numpy as np
import pytest

from phase1_anpr.detection.detector import (
    Detection,
    DetectorError,
    PlateDetector,
    _resolve_device,
)


class _FakeBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = [np.array(xyxy, dtype=float)]
        self.conf = [conf]
        self.cls = [cls]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


def _bare_detector(**kw):
    """Build a PlateDetector without running __init__ (no weights needed)."""
    det = PlateDetector.__new__(PlateDetector)
    det.plate_class_id = kw.get("plate_class_id", 0)
    det.save_crops = kw.get("save_crops", False)
    det.crops_dir = kw.get("crops_dir")
    return det


def test_missing_weights_raises(tmp_path):
    with pytest.raises(DetectorError):
        PlateDetector(tmp_path / "nope.pt")


def test_resolve_device_explicit():
    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("cuda") == "cuda"
    assert _resolve_device("gpu") == "cuda"


def test_parse_results_filters_class_and_reads_fields():
    det = _bare_detector(plate_class_id=0)
    results = [_FakeResult([
        _FakeBox([10, 20, 110, 70], 0.9, 0),   # plate -> kept
        _FakeBox([0, 0, 50, 50], 0.8, 1),       # other class -> dropped
    ])]
    parsed = det._parse_results(results, frame_number=6)
    assert len(parsed) == 1
    d = parsed[0]
    assert d.bbox == (10, 20, 110, 70)
    assert d.confidence == pytest.approx(0.9)
    assert d.class_id == 0
    assert d.frame_number == 6


def test_crop_clamps_to_frame_bounds():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    d = Detection(bbox=(-10, -5, 1000, 1000), confidence=0.5, class_id=0, frame_number=0)
    crop = PlateDetector.crop(frame, d)
    assert crop.shape == (48, 64, 3)


def test_save_crop_disabled_returns_none():
    det = _bare_detector(save_crops=False)
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    assert det.save_crop(crop, 0, 0) is None


def test_save_crop_writes_file(tmp_path):
    det = _bare_detector(save_crops=True, crops_dir=tmp_path)
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    path = det.save_crop(crop, 3, 1)
    assert path is not None and path.exists()
