"""End-to-end ANPR pipeline orchestrator for Phase 1 (Step 16).

Wires the already-built components into one flow without duplicating their logic.
Every collaborator is dependency-injected, so integration tests run with fakes
and need no real YOLO/Paddle weights.

Per-frame:  detect -> crop -> BYTETrack update (crops accumulate on tracks).
On finalize: for each finished track -> quality select -> rectify+OCR+fuse ->
normalize -> confidence -> build canonical observation -> persist evidence +
observation -> process accepted observation against the watchlist.

Idempotency / one-observation-per-track:
  event_id is deterministic per (camera_id, source_id, track_id) via uuid5. A
  per-run guard prevents processing the same track twice, and the deterministic
  id lets the repository (event_id PK) and watchlist (unique
  watchlist_id+event_id) reject duplicates when a whole video is reprocessed.

  ``source_id`` is optional (Step 19). Track ids restart at 1 for every recorded
  video, so two different videos from the SAME camera would otherwise mint
  identical event ids and silently collide (INSERT OR IGNORE). Supplying a
  distinct ``source_id`` per video disambiguates them. When ``source_id`` is
  None/empty the legacy id derivation is preserved byte-for-byte, so existing
  event ids and stores are unaffected.
"""

import math
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional


@dataclass
class PipelineResult:
    """Outcome of finalizing one track."""

    observation: object          # PlateObservation
    inserted: bool               # False if the observation already existed
    alerts: list                 # alerts newly created for this observation


def _default_timestamp(track) -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_video_start_time(value):
    """Parse an optional configured video start time to a UTC ``datetime``.

    ``None``/empty means "not configured" (returns None). A supplied value must
    be a valid ISO-8601 datetime that includes timezone information; naive or
    malformed values raise ``ValueError`` (fail fast, before any inference).
    Accepts a trailing ``Z`` (UTC) as well as explicit offsets like ``+05:30``.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        # datetime.fromisoformat handles "Z" on 3.11+, but normalize explicitly
        # so behavior is identical regardless of patch version.
        if text[-1] in ("Z", "z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(
                f"video.start_time is not a valid ISO-8601 datetime: {value!r}"
            ) from exc
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(
            f"video.start_time must include timezone information (got naive "
            f"value {value!r})"
        )
    return dt.astimezone(timezone.utc)


def _validate_video_timestamp(video_timestamp):
    """Validate an explicitly supplied per-frame video timestamp (seconds)."""
    if isinstance(video_timestamp, bool) or not isinstance(
            video_timestamp, (int, float)):
        raise ValueError(
            f"video_timestamp must be numeric, got {video_timestamp!r}")
    value = float(video_timestamp)
    if not math.isfinite(value):
        raise ValueError(
            f"video_timestamp must be finite, got {video_timestamp!r}")
    if value < 0:
        raise ValueError(
            f"video_timestamp must be >= 0, got {video_timestamp!r}")
    return value


class ANPRPipeline:
    """Orchestrates the completed ANPR components end to end."""

    def __init__(self, detector, tracker, quality_scorer, track_processor,
                 normalizer, confidence_scorer, observation_builder,
                 observation_repo, evidence_store=None, watchlist_repo=None,
                 camera_id="cam_01", model_version="phase1-anpr-0.1.0",
                 timestamp_fn=None, video_start_time=None, source_id=None):
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
        # Optional recorded-video source discriminator (Step 19). Empty/blank is
        # normalized to None so it behaves identically to the legacy single-video
        # path (no extra event-id segment, unchanged ids).
        if isinstance(source_id, str):
            source_id = source_id.strip() or None
        self.source_id = source_id or None
        self.model_version = model_version
        # Kept as None (not the default fn) so we can honor the documented
        # priority: an explicit timestamp_fn always wins over video start time.
        self.timestamp_fn = timestamp_fn
        # Parsed/validated up front so a bad configured value fails immediately.
        self.video_start_time = parse_video_start_time(video_start_time)
        # Lightweight frame_number -> video_timestamp (seconds) map for processed
        # frames; avoids widening tracker/quality candidate tuples.
        self._frame_timestamps = {}
        # Guard: a track is turned into a canonical observation at most once.
        self._processed_track_ids = set()

    # --- per-frame ------------------------------------------------------------

    def process_frame(self, frame, frame_number, video_timestamp=None):
        """Detect plates on one frame and feed them to the tracker.

        ``video_timestamp`` (seconds since the start of the source video) is
        optional and backward-compatible: old two-arg callers still work. When
        supplied it is validated and remembered for event-time resolution.
        """
        if video_timestamp is not None:
            self._frame_timestamps[frame_number] = _validate_video_timestamp(
                video_timestamp)
        detections = self.detector.detect(frame, frame_number)
        crops = [self.detector.crop(frame, d) for d in detections]
        self.tracker.update(detections, frame_number, crops)
        return detections

    # --- finalize -------------------------------------------------------------

    def _event_id_for(self, track) -> str:
        # Legacy path (source_id is None) keeps the exact original name so ids
        # generated before Step 19 remain identical. When a source_id is set it
        # adds one segment, disambiguating identical track ids across videos.
        if self.source_id:
            name = (f"citysight/obs/{self.camera_id}/{self.source_id}"
                    f"/{track.track_id}")
        else:
            name = f"citysight/obs/{self.camera_id}/{track.track_id}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, name))

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

    def _resolve_observation_timestamp(self, track, scored) -> str:
        """Resolve the canonical observation timestamp (ISO-8601 string).

        Priority:
          1. an explicit caller-provided ``timestamp_fn`` (authoritative);
          2. recorded-video event time when ``video_start_time`` is configured
             and the sighting frame's video offset is known;
          3. the processing-time UTC fallback.

        The sighting frame is the best selected quality crop
        (``scored[0].frame_number``); if there is no valid scored crop, the
        track's most recent observed frame is used, and only if that offset was
        never captured do we fall back to processing time.
        """
        if self.timestamp_fn is not None:
            return self.timestamp_fn(track)

        if self.video_start_time is not None:
            offset = None
            if scored:
                offset = self._frame_timestamps.get(scored[0].frame_number)
            if offset is None:
                offset = self._frame_timestamps.get(track.last_seen)
            if offset is not None:
                event_time = self.video_start_time + timedelta(seconds=offset)
                return event_time.isoformat()

        return _default_timestamp(track)

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
            timestamp=self._resolve_observation_timestamp(track, scored),
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
                self.process_frame(
                    pf.frame, pf.frame_number,
                    video_timestamp=getattr(pf, "video_timestamp", None))
        return self.finalize()


def build_pipeline_from_config(config, observation_repo, evidence_store=None,
                               watchlist_repo=None, *, camera_id=None,
                               source_id=None, video_start_time=None):
    """Construct a pipeline from config using the real components.

    Raises DetectorError (clear, actionable) if the YOLO weights are missing —
    weights are never downloaded automatically.

    ``camera_id``, ``source_id`` and ``video_start_time`` are optional per-source
    overrides used by the multi-video replay runner (Step 19). When left as None
    the values from ``config['video']`` are used, so the single-video path is
    unchanged.
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
    # Validate the optional recorded-video start time up front, before any
    # (expensive) YOLO/OCR inference can begin. An explicit override wins over
    # the configured default.
    video_cfg = config.get("video", {})
    start_time_value = video_start_time if video_start_time is not None \
        else video_cfg.get("start_time")
    video_start_time = parse_video_start_time(start_time_value)
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
        camera_id=camera_id if camera_id is not None
        else video_cfg.get("camera_id", "cam_01"),
        source_id=source_id,
        model_version=config.get("models", {}).get("model_version",
                                                    "phase1-anpr-0.1.0"),
        video_start_time=video_start_time,
    )


def _require_ocr_backend():
    from phase1_anpr.ocr.plate_ocr import PaddleOCRBackend
    return PaddleOCRBackend()
