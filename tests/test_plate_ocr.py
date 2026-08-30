"""Tests for Step 6 OCR using a mock backend (no model download / internet)."""

import numpy as np

from phase1_anpr.ocr.plate_ocr import OCRResult, PlateOCR
from phase1_anpr.quality.quality_scorer import ScoredCrop


class FakeBackend:
    """Returns a scripted list of (text, confidence) segments."""

    def __init__(self, segments):
        self.segments = segments
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        return list(self.segments)


def make_scored(frame=1, quality=0.7, det=0.9, shape=(20, 60, 3)):
    crop = np.zeros(shape, dtype=np.uint8) if shape else None
    return ScoredCrop(crop=crop, quality_score=quality,
                      detector_confidence=det, frame_number=frame)


def test_reads_and_cleans_text():
    ocr = PlateOCR(FakeBackend([("  mh12 ab ", 0.88)]))
    result = ocr.read(make_scored(frame=3))
    assert isinstance(result, OCRResult)
    assert result.text == "MH12 AB"
    assert result.ocr_confidence == 0.88
    assert result.quality_score == 0.7
    assert result.detector_confidence == 0.9
    assert result.frame_number == 3


def test_multiple_segments_join_and_take_min_confidence():
    ocr = PlateOCR(FakeBackend([("mh", 0.9), ("12ab", 0.6)]))
    result = ocr.read(make_scored())
    assert result.text == "MH12AB"
    assert result.ocr_confidence == 0.6


def test_no_text_result_is_safe():
    ocr = PlateOCR(FakeBackend([]))
    result = ocr.read(make_scored(frame=5))
    assert result.text == "" and result.ocr_confidence == 0.0
    assert result.frame_number == 5


def test_blank_segments_ignored():
    ocr = PlateOCR(FakeBackend([("   ", 0.99)]))
    assert ocr.read(make_scored()).text == ""


def test_empty_crop_skips_backend():
    backend = FakeBackend([("SHOULDNOTRUN", 0.9)])
    ocr = PlateOCR(backend)
    result = ocr.read(make_scored(shape=None))
    assert result.text == "" and backend.calls == 0


def test_read_many():
    ocr = PlateOCR(FakeBackend([("abc", 0.7)]))
    results = ocr.read_many([make_scored(frame=1), make_scored(frame=2)])
    assert [r.frame_number for r in results] == [1, 2]
    assert all(r.text == "ABC" for r in results)


# --- PaddleOCRBackend tests (no real model download / inference) ---

class FakeTextRecognition:
    """Stand-in for paddleocr.TextRecognition; records init/predict calls."""

    instances = 0

    def __init__(self, *args, **kwargs):
        FakeTextRecognition.instances += 1
        self.predict_calls = 0
        self.scripted = []  # list of outputs to return, one per predict call

    def predict(self, image):
        out = self.scripted[self.predict_calls] if self.predict_calls < len(self.scripted) else []
        self.predict_calls += 1
        return out


def _make_paddle_backend(monkeypatch):
    import paddleocr
    from phase1_anpr.ocr.plate_ocr import PaddleOCRBackend
    FakeTextRecognition.instances = 0
    monkeypatch.setattr(paddleocr, "TextRecognition", FakeTextRecognition)
    return PaddleOCRBackend()


def test_paddle_backend_parses_rec_text_and_score(monkeypatch):
    backend = _make_paddle_backend(monkeypatch)
    backend._rec.scripted = [[{"rec_text": "MH12AB", "rec_score": 0.93}]]
    assert backend.recognize(object()) == [("MH12AB", 0.93)]


def test_paddle_backend_handles_empty_output(monkeypatch):
    backend = _make_paddle_backend(monkeypatch)
    backend._rec.scripted = [[]]
    assert backend.recognize(object()) == []


def test_paddle_backend_handles_none_output(monkeypatch):
    backend = _make_paddle_backend(monkeypatch)
    backend._rec.scripted = []  # predict returns [] via fallback
    assert backend.recognize(object()) == []


def test_paddle_backend_initializes_model_once(monkeypatch):
    backend = _make_paddle_backend(monkeypatch)
    backend._rec.scripted = [
        [{"rec_text": "AAA", "rec_score": 0.8}],
        [{"rec_text": "BBB", "rec_score": 0.7}],
    ]
    r1 = backend.recognize(object())
    r2 = backend.recognize(object())
    assert r1 == [("AAA", 0.8)] and r2 == [("BBB", 0.7)]
    assert FakeTextRecognition.instances == 1      # constructed once
    assert backend._rec.predict_calls == 2          # reused per crop


def test_paddle_backend_end_to_end_with_plate_ocr(monkeypatch):
    backend = _make_paddle_backend(monkeypatch)
    backend._rec.scripted = [[{"rec_text": " mh 12 ab ", "rec_score": 0.6}]]
    ocr = PlateOCR(backend)
    result = ocr.read(make_scored(frame=7, quality=0.5, det=0.8))
    assert result.text == "MH 12 AB"
    assert result.ocr_confidence == 0.6
    assert result.frame_number == 7
