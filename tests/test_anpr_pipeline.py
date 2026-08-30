"""Step 16 end-to-end integration tests. Fully mocked components: no real
YOLO/Paddle weights, no video files, no network."""

import numpy as np
import pytest

from phase1_anpr.detection.detector import Detection
from phase1_anpr.ocr.plate_ocr import OCRResult
from phase1_anpr.confidence.confidence_scorer import ConfidenceScorer
from phase1_anpr.normalization.plate_normalizer import PlateNormalizer
from phase1_anpr.observation.observation_builder import ObservationBuilder
from phase1_anpr.pipeline.anpr_pipeline import (
    ANPRPipeline,
    build_pipeline_from_config,
)
from phase1_anpr.pipeline.track_processor import TrackProcessor
from phase1_anpr.quality.quality_scorer import QualityScorer, ScoredCrop
from phase1_anpr.tracking.tracker import PlateTracker
from phase1_anpr.persistence import (
    SQLiteObservationRepository,
    SQLiteWatchlistRepository,
)


# --- fakes --------------------------------------------------------------------

class FakeDetector:
    """Emits one detection per frame for a fixed bbox; crops are dummy arrays."""

    def __init__(self, per_frame=1, conf=0.9):
        self.per_frame = per_frame
        self.conf = conf

    def detect(self, frame, frame_number):
        return [Detection(bbox=(10, 10, 90, 40), confidence=self.conf,
                          class_id=0, frame_number=frame_number)
                for _ in range(self.per_frame)]

    @staticmethod
    def crop(frame, detection):
        return np.full((30, 80, 3), 127, dtype=np.uint8)


class NoDetections(FakeDetector):
    def detect(self, frame, frame_number):
        return []


class FakeRectifier:
    def rectify(self, crop):
        from types import SimpleNamespace
        return SimpleNamespace(rectified_crop=crop)


class FakeOCR:
    """Returns a fixed plate string + confidence for every crop."""

    def __init__(self, text="MH12AB1234", conf=0.95):
        self.text = text
        self.conf = conf

    def read(self, scored_crop):
        return OCRResult(text=self.text, ocr_confidence=self.conf,
                         quality_score=scored_crop.quality_score,
                         detector_confidence=scored_crop.detector_confidence,
                         frame_number=scored_crop.frame_number)


class FakeEvidenceStore:
    def __init__(self):
        self.saved = []

    def store(self, event_id, source_path, ext="jpg"):
        ref = f"{event_id}.jpg"
        self.saved.append(ref)
        return ref


class FrameSource:
    """Minimal video-reader stand-in usable as a context manager."""

    def __init__(self, frames):
        self._frames = frames

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def frames(self):
        from phase1_anpr.video.video_reader import ProcessedFrame
        for i in range(self._frames):
            yield ProcessedFrame(frame=np.zeros((100, 100, 3), np.uint8),
                                 frame_number=i, video_timestamp=float(i),
                                 camera_id="cam_01")


def make_pipeline(detector=None, ocr=None, obs_repo=None, wl_repo=None,
                  evidence=None, accept_threshold=0.6):
    obs_repo = obs_repo or SQLiteObservationRepository(":memory:")
    return ANPRPipeline(
        detector=detector or FakeDetector(),
        tracker=PlateTracker(max_age=2),
        quality_scorer=QualityScorer(),
        track_processor=TrackProcessor(FakeRectifier(), ocr or FakeOCR()),
        normalizer=PlateNormalizer(),
        confidence_scorer=ConfidenceScorer(accept_threshold=accept_threshold),
        observation_builder=ObservationBuilder(),
        observation_repo=obs_repo,
        evidence_store=evidence,
        watchlist_repo=wl_repo,
        camera_id="cam_01",
        timestamp_fn=lambda track: "2026-08-31T10:00:00+05:30",
    ), obs_repo


# --- tests --------------------------------------------------------------------

def test_happy_path_end_to_end():
    evidence = FakeEvidenceStore()
    pipe, repo = make_pipeline(evidence=evidence)
    results = pipe.run(FrameSource(4))
    assert len(results) == 1
    obs = results[0].observation
    assert obs.status == "accepted"
    assert obs.plate_normalized == "MH12AB1234"
    assert repo.get(obs.event_id)["plate_normalized"] == "MH12AB1234"
    assert evidence.saved  # evidence stored for accepted plate


def test_multiple_frames_one_observation():
    pipe, repo = make_pipeline()
    results = pipe.run(FrameSource(6))
    assert len(results) == 1
    assert repo.count() == 1


def test_accepted_watchlist_match_one_alert():
    wl = SQLiteWatchlistRepository(":memory:")
    wl.add("MH12AB1234", label="stolen")
    pipe, _ = make_pipeline(wl_repo=wl)
    results = pipe.run(FrameSource(4))
    assert sum(len(r.alerts) for r in results) == 1
    assert len(wl.list_alerts()) == 1


def test_review_persisted_but_no_alert():
    wl = SQLiteWatchlistRepository(":memory:")
    wl.add("MH12AB1234")
    # High accept threshold forces review instead of accepted.
    pipe, repo = make_pipeline(wl_repo=wl, accept_threshold=0.99)
    results = pipe.run(FrameSource(4))
    assert results[0].observation.status == "review"
    assert repo.count() == 1
    assert wl.list_alerts() == []


def test_abstained_persisted_without_identity():
    # Empty OCR text -> abstained.
    pipe, repo = make_pipeline(ocr=FakeOCR(text="", conf=0.0))
    results = pipe.run(FrameSource(4))
    obs = results[0].observation
    assert obs.status == "abstained"
    assert obs.plate_normalized is None and obs.plate_raw is None
    assert obs.plate_image_path is None
    assert repo.get(obs.event_id)["plate_normalized"] is None


def test_duplicate_retry_no_duplicates():
    wl = SQLiteWatchlistRepository(":memory:")
    wl.add("MH12AB1234")
    obs_repo = SQLiteObservationRepository(":memory:")
    # First run.
    p1, _ = make_pipeline(obs_repo=obs_repo, wl_repo=wl)
    p1.run(FrameSource(4))
    # Reprocess the same video with a fresh pipeline/tracker (same track ids).
    p2, _ = make_pipeline(obs_repo=obs_repo, wl_repo=wl)
    results2 = p2.run(FrameSource(4))
    assert all(r.inserted is False for r in results2)  # already persisted
    assert obs_repo.count() == 1
    assert len(wl.list_alerts()) == 1


def test_no_detections_no_observations():
    pipe, repo = make_pipeline(detector=NoDetections())
    results = pipe.run(FrameSource(5))
    assert results == []
    assert repo.count() == 0


def test_finalize_flushes_remaining_tracks():
    # Short video (track never goes stale) still flushes on finalize().
    pipe, repo = make_pipeline()
    pipe.process_frame(np.zeros((100, 100, 3), np.uint8), 0)
    pipe.process_frame(np.zeros((100, 100, 3), np.uint8), 1)
    assert repo.count() == 0  # nothing finalized yet
    results = pipe.finalize()
    assert len(results) == 1
    assert repo.count() == 1


def test_missing_yolo_weights_clear_error():
    from phase1_anpr.detection.detector import DetectorError
    config = {
        "detection": {"weights_path": "weights/does_not_exist.pt", "device": "cpu"},
        "video": {"camera_id": "cam_01"},
        "output": {},
    }
    with pytest.raises(DetectorError) as exc:
        build_pipeline_from_config(config, observation_repo=SQLiteObservationRepository(":memory:"))
    assert "weights" in str(exc.value).lower()
