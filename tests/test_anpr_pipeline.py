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
    parse_video_start_time,
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


# --- Step 18: recorded-video event time --------------------------------------

from datetime import datetime, timezone
from types import SimpleNamespace

from phase1_anpr.pipeline.anpr_pipeline import _validate_video_timestamp


def _track(track_id=1, last_seen=5):
    return SimpleNamespace(track_id=track_id, last_seen=last_seen)


def _pipeline(video_start_time=None, timestamp_fn=None):
    pipe, _ = make_pipeline()
    pipe.timestamp_fn = timestamp_fn
    pipe.video_start_time = parse_video_start_time(video_start_time)
    return pipe


# start-time parsing
def test_parse_start_time_offset():
    dt = parse_video_start_time("2026-08-31T10:00:00+05:30")
    assert dt == datetime(2026, 8, 31, 4, 30, 0, tzinfo=timezone.utc)
    assert dt.tzinfo == timezone.utc


def test_parse_start_time_utc_z():
    dt = parse_video_start_time("2026-08-31T12:00:00Z")
    assert dt == datetime(2026, 8, 31, 12, 0, 0, tzinfo=timezone.utc)


def test_parse_start_time_none_and_empty():
    assert parse_video_start_time(None) is None
    assert parse_video_start_time("") is None


def test_parse_start_time_naive_rejected():
    with pytest.raises(ValueError, match="timezone"):
        parse_video_start_time("2026-08-31T10:00:00")


def test_parse_start_time_malformed_rejected():
    with pytest.raises(ValueError, match="ISO-8601"):
        parse_video_start_time("not-a-timestamp")


# event-time resolution
def test_event_time_best_crop_offset():
    pipe = _pipeline(video_start_time="2026-08-31T10:00:00+05:30")
    pipe._frame_timestamps = {2: 15.0}
    scored = [ScoredCrop(crop=np.zeros((1, 1), np.uint8), quality_score=0.9,
                         detector_confidence=0.9, frame_number=2)]
    ts = pipe._resolve_observation_timestamp(_track(), scored)
    assert ts == "2026-08-31T04:30:15+00:00"


def test_event_time_fractional_offset_utc():
    pipe = _pipeline(video_start_time="2026-08-31T12:00:00Z")
    pipe._frame_timestamps = {7: 90.25}
    scored = [ScoredCrop(crop=np.zeros((1, 1), np.uint8), quality_score=0.9,
                         detector_confidence=0.9, frame_number=7)]
    ts = pipe._resolve_observation_timestamp(_track(), scored)
    assert ts == "2026-08-31T12:01:30.250000+00:00"
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


# best-frame semantics: best quality crop is NOT the last frame
def test_event_time_uses_best_quality_frame_not_last():
    scorer = QualityScorer()
    low = np.full((30, 80, 3), 127, dtype=np.uint8)   # flat -> low quality
    rng = np.random.RandomState(0)
    high = rng.randint(0, 256, size=(40, 120, 3), dtype=np.uint8)  # sharp/large
    # frame 2 (middle) holds the best crop; frame 5 is the last, low quality.
    candidates = [(0, low, 0.9), (2, high, 0.9), (5, low, 0.9)]
    scored = scorer.select_best(candidates)
    assert scored[0].frame_number == 2  # best is the middle frame, not the last

    pipe = _pipeline(video_start_time="2026-08-31T00:00:00Z")
    pipe._frame_timestamps = {0: 0.0, 2: 4.0, 5: 10.0}
    ts = pipe._resolve_observation_timestamp(_track(last_seen=5), scored)
    assert ts == "2026-08-31T00:00:04+00:00"  # frame 2's offset, not frame 5's


# run() propagation
def test_run_propagates_video_timestamp_into_event_time():
    pipe, _ = make_pipeline()
    pipe.timestamp_fn = None
    pipe.video_start_time = parse_video_start_time("2026-08-31T10:00:00+05:30")
    results = pipe.run(FrameSource(4))  # ProcessedFrame.video_timestamp = i
    assert len(results) == 1
    ts = results[0].observation.timestamp
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0
    # Event time is the video start plus the best crop's whole-second offset.
    assert parsed.year == 2026 and parsed.hour == 4 and parsed.minute == 30


# backward compatibility
def test_process_frame_two_arg_still_works():
    pipe, repo = make_pipeline()
    pipe.process_frame(np.zeros((100, 100, 3), np.uint8), 0)
    pipe.process_frame(np.zeros((100, 100, 3), np.uint8), 1)
    results = pipe.finalize()
    assert len(results) == 1


def test_no_start_time_uses_processing_fallback():
    pipe = _pipeline(video_start_time=None)  # timestamp_fn None, no start time
    before = datetime.now(timezone.utc)
    ts = pipe._resolve_observation_timestamp(_track(), scored=[])
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None
    assert parsed >= before.replace(microsecond=0)


def test_explicit_timestamp_fn_still_authoritative():
    pipe = _pipeline(timestamp_fn=lambda t: "2020-01-01T00:00:00+00:00")
    scored = [ScoredCrop(crop=np.zeros((1, 1), np.uint8), quality_score=0.9,
                         detector_confidence=0.9, frame_number=2)]
    pipe._frame_timestamps = {2: 15.0}
    assert pipe._resolve_observation_timestamp(_track(), scored) == \
        "2020-01-01T00:00:00+00:00"


def test_timestamp_fn_overrides_configured_start_time():
    pipe = _pipeline(video_start_time="2026-08-31T10:00:00+05:30",
                     timestamp_fn=lambda t: "1999-12-31T23:59:59+00:00")
    scored = [ScoredCrop(crop=np.zeros((1, 1), np.uint8), quality_score=0.9,
                         detector_confidence=0.9, frame_number=2)]
    pipe._frame_timestamps = {2: 15.0}
    assert pipe._resolve_observation_timestamp(_track(), scored) == \
        "1999-12-31T23:59:59+00:00"


# invalid supplied video position
def test_negative_video_timestamp_rejected():
    pipe, _ = make_pipeline()
    with pytest.raises(ValueError):
        pipe.process_frame(np.zeros((10, 10, 3), np.uint8), 0, video_timestamp=-1.0)


def test_nan_and_inf_video_timestamp_rejected():
    for bad in (float("nan"), float("inf"), "x", True):
        with pytest.raises(ValueError):
            _validate_video_timestamp(bad)


# no valid scored crop -> safe fallback to most recent observed frame
def test_no_scored_crop_uses_recent_frame_offset():
    pipe = _pipeline(video_start_time="2026-08-31T00:00:00Z")
    pipe._frame_timestamps = {5: 30.0}
    ts = pipe._resolve_observation_timestamp(_track(last_seen=5), scored=[])
    assert ts == "2026-08-31T00:00:30+00:00"


def test_no_scored_crop_no_mapped_offset_falls_back_to_processing_time():
    pipe = _pipeline(video_start_time="2026-08-31T00:00:00Z")
    pipe._frame_timestamps = {}  # nothing captured for this track
    before = datetime.now(timezone.utc)
    ts = pipe._resolve_observation_timestamp(_track(last_seen=99), scored=[])
    parsed = datetime.fromisoformat(ts)
    assert parsed >= before.replace(microsecond=0)  # did not crash / fabricate


# --- Step 19: optional source_id in the deterministic event id ---------------
# source_id is an ingestion-only discriminator: it changes ONLY how event_id is
# derived. It is not persisted as a DB column and not exposed via the API.

import uuid


def _pipe_with_source(source_id, obs_repo=None):
    obs_repo = obs_repo or SQLiteObservationRepository(":memory:")
    pipe = ANPRPipeline(
        detector=FakeDetector(),
        tracker=PlateTracker(max_age=2),
        quality_scorer=QualityScorer(),
        track_processor=TrackProcessor(FakeRectifier(), FakeOCR()),
        normalizer=PlateNormalizer(),
        confidence_scorer=ConfidenceScorer(accept_threshold=0.6),
        observation_builder=ObservationBuilder(),
        observation_repo=obs_repo,
        camera_id="cam_01",
        source_id=source_id,
        timestamp_fn=lambda track: "2026-08-31T10:00:00+05:30",
    )
    return pipe, obs_repo


def test_event_id_legacy_unchanged_when_source_none():
    pipe, _ = _pipe_with_source(None)
    expected = str(uuid.uuid5(uuid.NAMESPACE_URL, "citysight/obs/cam_01/7"))
    assert pipe._event_id_for(_track(track_id=7)) == expected


def test_event_id_includes_source_when_set():
    pipe, _ = _pipe_with_source("vidA")
    got = pipe._event_id_for(_track(track_id=7))
    assert got == str(
        uuid.uuid5(uuid.NAMESPACE_URL, "citysight/obs/cam_01/vidA/7"))
    # Distinct from the legacy id, so the two never collide.
    assert got != str(uuid.uuid5(uuid.NAMESPACE_URL, "citysight/obs/cam_01/7"))


def test_source_id_blank_normalized_to_none():
    pipe, _ = _pipe_with_source("   ")
    assert pipe.source_id is None
    assert pipe._event_id_for(_track(track_id=3)) == str(
        uuid.uuid5(uuid.NAMESPACE_URL, "citysight/obs/cam_01/3"))


def test_distinct_source_ids_avoid_collision_same_camera():
    # Two videos from the same camera reuse track ids from 1; distinct
    # source_ids keep their observations from colliding on event_id. Each
    # pipeline is built immediately before it runs, matching the replay flow
    # (and the fresh-per-source BYTETracker id counter).
    obs_repo = SQLiteObservationRepository(":memory:")
    p1, _ = _pipe_with_source("videoA", obs_repo=obs_repo)
    p1.run(FrameSource(4))
    p2, _ = _pipe_with_source("videoB", obs_repo=obs_repo)
    p2.run(FrameSource(4))
    assert obs_repo.count() == 2


def test_same_camera_no_source_still_collides():
    # Legacy behavior preserved: without source_id the identical track ids
    # collide to a single row (idempotent reprocess).
    obs_repo = SQLiteObservationRepository(":memory:")
    p1, _ = _pipe_with_source(None, obs_repo=obs_repo)
    p1.run(FrameSource(4))
    p2, _ = _pipe_with_source(None, obs_repo=obs_repo)
    p2.run(FrameSource(4))
    assert obs_repo.count() == 1
