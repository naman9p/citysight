"""End-to-end ANPR pipeline orchestrator for Phase 1 (Step 16).

Wires the already-built components into one flow without duplicating their logic.
Every collaborator is dependency-injected, so integration tests run with fakes
and need no real YOLO/Paddle weights.

Per-frame:  detect -> crop -> BYTETrack update (crops accumulate on tracks).
On finalize: for each finished track -> quality select -> rectify+OCR+fuse ->
normalize -> confidence -> build canonical observation -> persist evidence +
observation -> process accepted observation against the watchlist.

Idempotency / one-observation-per-track:
  event_id is deterministic per (camera_id, track_id) via uuid5. A per-run guard
  prevents processing the same track twice, and the deterministic id lets the
  repository (event_id PK) and watchlist (unique watchlist_id+event_id) reject
  duplicates when a whole video is reprocessed.
"""

import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class PipelineResult:
    """Outcome of finalizing one track."""

    observation: object          # PlateObservation
    inserted: bool               # False if the observation already existed
    alerts: list                 # alerts newly created for this observation


def _default_timestamp(track) -> str:
    return datetime.now(timezone.utc).isoformat()


class ANPRPipeline:
    """Orchestrates the completed ANPR components end to end."""

    def __init__(self, detector, tracker, quality_scorer, track_processor,
                 normalizer, confidence_scorer, observation_builder,
                 observation_repo, evidence_store=None, watchlist_repo=None,
                 camera_id="cam_01", model_version="phase1-anpr-0.1.0",
                 timestamp_fn=None):
        self.detector = detector
        self.tracker = tracker
        self.quality_scorer = quality_scorer
        self.track_processor = track_processor
        self.normalizer = normalizer
        self.confidence_scorer = confidence_scorer
        self.observation_builder = observation_builder
        self.observation_repo = observation_repo
        self.evidence_store = evidence_store
        self.watchlist_repo = watchlist_repo
        self.camera_id = camera_id
        self.model_version = model_version
        self.timestamp_fn = timestamp_fn or _default_timestamp
        # Guard: a track is turned into a canonical observation at most once.
        self._processed_track_ids = set()

    # --- per-frame ------------------------------------------------------------

    def process_frame(self, frame, frame_number):
        """Detect plates on one frame and feed them to the tracker."""
        detections = self.detector.detect(frame, frame_number)
        crops = [self.detector.crop(frame, d) for d in detections]
        self.tracker.update(detections, frame_number, crops)
        return detections

    # --- finalize -------------------------------------------------------------

    def _event_id_for(self, track) -> str:
        return str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"citysight/obs/{self.camera_id}/{track.track_id}"))

    def _store_evidence(self, event_id, crop) -> Optional[str]:
        """Persist the best crop and return a reference; safe on any failure."""
        if self.evidence_store is None or crop is None or getattr(crop, "size", 0) == 0:
            return None
        try:
            import cv2
            fd, tmp = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            try:
                if not cv2.imwrite(tmp, crop):
                    return None
                return self.evidence_store.store(event_id, tmp)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        except Exception:
            # Missing/invalid evidence must not break observation persistence.
            return None

    def _finalize_track(self, track) -> Optional[PipelineResult]:
        if track.track_id in self._processed_track_ids:
            return None
        self._processed_track_ids.add(track.track_id)

        scored = self.quality_scorer.select_best(track.candidates)
        track_result = self.track_processor.process(track.track_id, scored)
        normalization = self.normalizer.normalize(track_result.best_text or "")
        confidence = self.confidence_scorer.score(track_result, normalization)

        event_id = self._event_id_for(track)
        abstained = confidence.decision == "abstained"

        # No plate identity is asserted for abstained tracks -> no evidence link.
        plate_image_path = None
        if not abstained and scored:
            plate_image_path = self._store_evidence(event_id, scored[0].crop)

        observation = self.observation_builder.build(
            track_result, normalization, confidence,
            camera_id=self.camera_id,
            timestamp=self.timestamp_fn(track),
            model_version=self.model_version,
            plate_image_path=plate_image_path,
            event_id=event_id,
        )

        inserted = self.observation_repo.save(
            observation,
            format_type=normalization.format_type,
            state_code=normalization.state_code,
        )

        alerts = []
        if self.watchlist_repo is not None:
            # process_observation only alerts on accepted exact matches.
            alerts = self.watchlist_repo.process_observation(observation)

        return PipelineResult(observation=observation, inserted=inserted,
                              alerts=alerts)

    def finalize(self) -> List[PipelineResult]:
        """Flush every finished/remaining track into canonical observations."""
        results = []
        for track in self.tracker.finalize():
            result = self._finalize_track(track)
            if result is not None:
                results.append(result)
        return results

    def run(self, video_reader) -> List[PipelineResult]:
        """Process an entire video reader, then finalize all tracks."""
        with video_reader as reader:
            for pf in reader.frames():
                self.process_frame(pf.frame, pf.frame_number)
        return self.finalize()


def build_pipeline_from_config(config, observation_repo, evidence_store=None,
                               watchlist_repo=None):
    """Construct a pipeline from config using the real components.

    Raises DetectorError (clear, actionable) if the YOLO weights are missing —
    weights are never downloaded automatically.
    """
    from phase1_anpr.detection.detector import PlateDetector
    from phase1_anpr.tracking.tracker import PlateTracker
    from phase1_anpr.quality.quality_scorer import QualityScorer
    from phase1_anpr.rectification.rectifier import PlateRectifier
    from phase1_anpr.ocr.plate_ocr import PlateOCR
    from phase1_anpr.pipeline.track_processor import TrackProcessor
    from phase1_anpr.normalization.plate_normalizer import PlateNormalizer
    from phase1_anpr.confidence.confidence_scorer import ConfidenceScorer
    from phase1_anpr.observation.observation_builder import ObservationBuilder

    detector = PlateDetector.from_config(config)  # raises if weights missing
    tracking = config.get("tracking", {})
    tracker = PlateTracker(
        track_buffer=tracking.get("track_buffer", 30),
        match_thresh=tracking.get("match_threshold", 0.8),
    )
    quality_scorer = QualityScorer.from_config(config)
    rectifier = PlateRectifier.from_config(config) if hasattr(
        PlateRectifier, "from_config") else PlateRectifier()
    ocr = PlateOCR.from_config(config) if hasattr(PlateOCR, "from_config") \
        else PlateOCR(_require_ocr_backend())
    track_processor = TrackProcessor.from_config(rectifier, ocr, config)
    normalizer = PlateNormalizer.from_config(config)
    confidence_scorer = ConfidenceScorer.from_config(config)
    observation_builder = ObservationBuilder()

    return ANPRPipeline(
        detector=detector, tracker=tracker, quality_scorer=quality_scorer,
        track_processor=track_processor, normalizer=normalizer,
        confidence_scorer=confidence_scorer,
        observation_builder=observation_builder,
        observation_repo=observation_repo, evidence_store=evidence_store,
        watchlist_repo=watchlist_repo,
        camera_id=config.get("video", {}).get("camera_id", "cam_01"),
        model_version=config.get("models", {}).get("model_version",
                                                    "phase1-anpr-0.1.0"),
    )


def _require_ocr_backend():
    from phase1_anpr.ocr.plate_ocr import PaddleOCRBackend
    return PaddleOCRBackend()
