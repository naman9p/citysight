"""Tests for Step 8 track-level rectify + OCR + fusion (mocked rectifier/OCR)."""

import numpy as np

from phase1_anpr.ocr.plate_ocr import OCRResult
from phase1_anpr.pipeline.track_processor import TrackProcessor
from phase1_anpr.quality.quality_scorer import ScoredCrop
from phase1_anpr.rectification.rectifier import RectificationResult


class FakeRectifier:
    """Records inputs; echoes crop back as a successful rectification."""

    def __init__(self, fallback=False):
        self.fallback = fallback
        self.seen = []

    def rectify(self, crop):
        self.seen.append(crop)
        return RectificationResult(
            rectified_crop=crop,
            success=not self.fallback,
            corners=None,
            fallback_used=self.fallback,
        )


class ScriptedOCR:
    """Returns OCRResults from a per-frame script keyed by frame_number."""

    def __init__(self, script):
        self.script = script  # frame_number -> (text, ocr_conf)
        self.seen_crops = []

    def read(self, scored_crop):
        self.seen_crops.append(scored_crop.crop)
        text, conf = self.script.get(scored_crop.frame_number, ("", 0.0))
        return OCRResult(
            text=text,
            ocr_confidence=conf,
            quality_score=scored_crop.quality_score,
            detector_confidence=scored_crop.detector_confidence,
            frame_number=scored_crop.frame_number,
        )


def crop(tag):
    c = np.zeros((20, 60, 3), dtype=np.uint8)
    c[0, 0, 0] = tag  # make crops distinguishable
    return c


def scored(frame, quality=0.7, det=0.9, tag=0):
    return ScoredCrop(crop=crop(tag), quality_score=quality,
                      detector_confidence=det, frame_number=frame)


def test_same_text_across_frames_wins():
    ocr = ScriptedOCR({0: ("MH12AB", 0.7), 1: ("MH12AB", 0.6), 2: ("XX99", 0.95)})
    tp = TrackProcessor(FakeRectifier(), ocr)
    result = tp.process(track_id=5, scored_crops=[scored(0), scored(1), scored(2)])
    assert result.track_id == 5
    assert result.best_text == "MH12AB"          # 2 supports beat 1 strong
    top = result.candidates[0]
    assert top.support == 2 and top.text == "MH12AB"


def test_stronger_repeated_candidate_beats_weaker():
    ocr = ScriptedOCR({0: ("AAA", 0.9), 1: ("AAA", 0.9), 2: ("BBB", 0.5)})
    tp = TrackProcessor(FakeRectifier(), ocr)
    result = tp.process(1, [scored(0), scored(1), scored(2)])
    assert [c.text for c in result.candidates] == ["AAA", "BBB"]
    assert result.candidates[0].score > result.candidates[1].score


def test_all_empty_returns_no_plate():
    ocr = ScriptedOCR({})  # everything -> ("", 0.0)
    tp = TrackProcessor(FakeRectifier(), ocr)
    result = tp.process(7, [scored(0), scored(1)])
    assert result.best_text is None
    assert result.candidates == []
    assert len(result.evidence) == 2  # OCR still attempted per crop


def test_rectifier_fallback_still_reaches_ocr():
    rect = FakeRectifier(fallback=True)
    ocr = ScriptedOCR({0: ("PLATE1", 0.8)})
    tp = TrackProcessor(rect, ocr)
    result = tp.process(2, [scored(0)])
    assert len(rect.seen) == 1 and len(ocr.seen_crops) == 1
    assert result.best_text == "PLATE1"


def test_metadata_preserved_in_evidence():
    ocr = ScriptedOCR({3: ("ZZ", 0.55)})
    tp = TrackProcessor(FakeRectifier(), ocr)
    result = tp.process(9, [scored(3, quality=0.42, det=0.88)])
    ev = result.evidence[0]
    assert ev.frame_number == 3
    assert ev.quality_score == 0.42
    assert ev.detector_confidence == 0.88
    assert ev.ocr_confidence == 0.55


def test_deterministic_tie_breaks_by_text():
    # Two candidates with identical evidence -> alphabetical order, stable.
    ocr = ScriptedOCR({0: ("BBB", 0.5), 1: ("AAA", 0.5)})
    tp = TrackProcessor(FakeRectifier(), ocr)
    r1 = tp.process(1, [scored(0, quality=0.5, det=0.5), scored(1, quality=0.5, det=0.5)])
    r2 = tp.process(1, [scored(1, quality=0.5, det=0.5), scored(0, quality=0.5, det=0.5)])
    assert [c.text for c in r1.candidates] == ["AAA", "BBB"]
    assert [c.text for c in r1.candidates] == [c.text for c in r2.candidates]


def test_only_top_k_crops_processed():
    ocr = ScriptedOCR({0: ("A", 0.9), 1: ("A", 0.9), 2: ("A", 0.9), 3: ("A", 0.9)})
    tp = TrackProcessor(FakeRectifier(), ocr, top_k=3)
    result = tp.process(1, [scored(0), scored(1), scored(2), scored(3)])
    assert len(result.evidence) == 3
