"""Canonical plate-observation event construction for Phase 1 (Step 11).

Maps existing pipeline outputs (TrackResult, NormalizationResult,
ConfidenceResult) plus explicit camera/time metadata into the canonical
`contracts/events/plate-observation.schema.json` event. Field names/types are
taken verbatim from that schema — this does not invent a competing format.

Metadata ANPR cannot produce (camera_id, timestamp, model_version, image path)
must be passed in; nothing is fabricated. For an `abstained` decision no
definite plate identity is asserted (plate fields and evidence are null).
"""

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jsonschema

SCHEMA_PATH = Path("contracts/events/plate-observation.schema.json")

# Fields ordered to match the schema for stable, readable serialization.
_FIELD_ORDER = (
    "event_id", "camera_id", "track_id", "timestamp", "plate_raw",
    "plate_normalized", "confidence", "status", "detector_confidence",
    "ocr_confidence", "quality_score", "best_frame_number",
    "plate_image_path", "model_version",
)


@dataclass
class PlateObservation:
    """A canonical observation; `event_id` is fixed once at construction."""

    event_id: str
    camera_id: str
    track_id: int
    timestamp: str
    plate_raw: Optional[str]
    plate_normalized: Optional[str]
    confidence: float
    status: str
    detector_confidence: Optional[float]
    ocr_confidence: Optional[float]
    quality_score: Optional[float]
    best_frame_number: Optional[int]
    plate_image_path: Optional[str]
    model_version: str

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in _FIELD_ORDER}

    def to_json(self) -> str:
        # sort_keys=False keeps schema order; content is deterministic either way.
        return json.dumps(self.to_dict(), separators=(",", ":"))


def _load_schema(schema_path=SCHEMA_PATH) -> dict:
    with Path(schema_path).open("r", encoding="utf-8") as f:
        return json.load(f)


class ObservationBuilder:
    """Build and validate canonical plate observations."""

    def __init__(self, schema_path=SCHEMA_PATH):
        self.schema = _load_schema(schema_path)
        self._validator = jsonschema.Draft7Validator(self.schema)

    def build(self, track_result, normalization, confidence_result,
              camera_id, timestamp, model_version, plate_image_path=None,
              event_id=None) -> PlateObservation:
        """Assemble a PlateObservation from pipeline outputs + explicit metadata.

        camera_id, timestamp and model_version are required inputs (not derivable
        from ANPR). `event_id` is generated once here if not supplied.
        """
        decision = confidence_result.decision
        abstained = decision == "abstained"

        # Pick the supporting evidence row that backs the best text (highest
        # quality among agreeing frames); None when abstained / no text.
        best_ev = None
        if not abstained and track_result.best_text:
            agreeing = [e for e in track_result.evidence
                        if e.text == track_result.best_text]
            if agreeing:
                best_ev = max(agreeing, key=lambda e: e.quality_score)

        if abstained:
            # Do not assert a plate identity for abstained observations.
            plate_raw = None
            plate_normalized = None
            detector_confidence = None
            ocr_confidence = None
            quality_score = None
            best_frame_number = None
        else:
            plate_raw = track_result.best_text
            plate_normalized = normalization.normalized_text or None
            detector_confidence = best_ev.detector_confidence if best_ev else None
            ocr_confidence = best_ev.ocr_confidence if best_ev else None
            quality_score = best_ev.quality_score if best_ev else None
            best_frame_number = best_ev.frame_number if best_ev else None

        return PlateObservation(
            event_id=event_id or str(uuid.uuid4()),
            camera_id=camera_id,
            track_id=int(track_result.track_id),
            timestamp=timestamp,
            plate_raw=plate_raw,
            plate_normalized=plate_normalized,
            confidence=float(confidence_result.confidence_score),
            status=decision,
            detector_confidence=detector_confidence,
            ocr_confidence=ocr_confidence,
            quality_score=quality_score,
            best_frame_number=best_frame_number,
            plate_image_path=plate_image_path,
            model_version=model_version,
        )

    def validate(self, observation) -> None:
        """Raise jsonschema.ValidationError if the observation is off-contract."""
        payload = observation.to_dict() if isinstance(observation, PlateObservation) \
            else observation
        self._validator.validate(payload)

    def is_valid(self, observation) -> bool:
        payload = observation.to_dict() if isinstance(observation, PlateObservation) \
            else observation
        return self._validator.is_valid(payload)
