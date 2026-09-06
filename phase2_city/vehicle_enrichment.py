"""Optional track-level vehicle enrichment integration (Step 26).

This coordinator joins the existing Step 24 vehicle detection/association
foundation to the Step 25 attribute extractor without changing plate tracking.
Each qualifying plate track receives exactly one enrichment attempt.  Vehicle
detections are computed once for a frame and reused for every qualifying track
in that frame; only small per-track result state survives the call.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Optional

from phase1_anpr.observation.observation_builder import PlateObservation
from phase2_city.fingerprint import VehicleFingerprint
from phase2_city.plate_vehicle_association import associate_plate_to_vehicle
from phase2_city.vehicle_attributes import (
    VehicleAttributes,
    extract_vehicle_attributes,
    fingerprint_from_observation_and_attributes,
)
from phase2_city.vehicle_detection import crop_vehicle


_LOGGER = logging.getLogger(__name__)


class TrackVehicleEnricher:
    """Run optional whole-vehicle enrichment once per qualifying plate track."""

    def __init__(
            self,
            vehicle_detector,
            *,
            association_fn: Callable = associate_plate_to_vehicle,
            crop_fn: Callable = crop_vehicle,
            attribute_extractor: Callable = extract_vehicle_attributes,
            fingerprint_adapter: Callable =
            fingerprint_from_observation_and_attributes,
    ):
        self.vehicle_detector = vehicle_detector
        self.association_fn = association_fn
        self.crop_fn = crop_fn
        self.attribute_extractor = attribute_extractor
        self.fingerprint_adapter = fingerprint_adapter

        # Presence in attempted_track_ids means the one allowed attempt was
        # consumed.  A present mapping value of None means clean abstention;
        # an absent mapping after an attempt means a terminal exception.
        self.attempted_track_ids: set[int] = set()
        self.attributes_by_track_id: dict[int, Optional[VehicleAttributes]] = {}

    def enrich_frame(self, frame, frame_number: int,
                     plate_tracks: Iterable) -> None:
        """Enrich newly qualifying tracks using one detector call for the frame.

        The caller supplies only tracks whose current plate crop passed the
        existing quality-validity check.  Attempts are marked before inference
        so detector failures cannot create retries on later frames.
        """
        pending = {}
        for track in plate_tracks:
            track_id = int(track.track_id)
            if track_id in self.attempted_track_ids:
                continue
            if int(track.frame_number) != int(frame_number):
                continue
            pending.setdefault(track_id, track)

        if not pending:
            return

        ordered_track_ids = sorted(pending)
        self.attempted_track_ids.update(ordered_track_ids)

        try:
            vehicle_detections = tuple(
                self.vehicle_detector.detect(frame, frame_number))
        except Exception as exc:
            _LOGGER.warning(
                "Optional vehicle detection failed for frame %s: %s",
                frame_number,
                exc,
            )
            return

        for track_id in ordered_track_ids:
            track = pending[track_id]
            try:
                association = self.association_fn(track, vehicle_detections)
                if association.outcome != "associated":
                    # None is a real clean-abstention result, not an empty
                    # VehicleAttributes sentinel.
                    self.attributes_by_track_id[track_id] = None
                    continue
            except Exception as exc:
                _LOGGER.warning(
                    "Optional vehicle association failed for track %s: %s",
                    track_id,
                    exc,
                )
                continue

            try:
                vehicle_crop = self.crop_fn(
                    frame, association.vehicle_detection)
            except Exception as exc:
                _LOGGER.warning(
                    "Optional vehicle crop failed for track %s: %s",
                    track_id,
                    exc,
                )
                continue

            try:
                attributes = self.attribute_extractor(
                    vehicle_crop, association)
                if not isinstance(attributes, VehicleAttributes):
                    raise TypeError(
                        "attribute extractor must return VehicleAttributes")
            except Exception as exc:
                _LOGGER.warning(
                    "Optional vehicle attribute extraction failed for track "
                    "%s: %s",
                    track_id,
                    exc,
                )
                continue

            self.attributes_by_track_id[track_id] = attributes

    def finalize_track(
            self,
            track_id: int,
            observation: PlateObservation,
    ) -> tuple[Optional[VehicleAttributes], Optional[VehicleFingerprint]]:
        """Consume transient state and create the final in-memory fingerprint."""
        track_id = int(track_id)
        if track_id not in self.attempted_track_ids:
            return None, None

        self.attempted_track_ids.discard(track_id)
        if track_id not in self.attributes_by_track_id:
            # The attempt ended in an exception.  There is no clean enrichment
            # result from which Step 26 should manufacture a fingerprint.
            return None, None

        attributes = self.attributes_by_track_id.pop(track_id)
        try:
            if attributes is None:
                fingerprint = VehicleFingerprint.from_plate_observation(
                    observation)
            else:
                fingerprint = self.fingerprint_adapter(
                    observation, attributes)
        except Exception as exc:
            _LOGGER.warning(
                "Optional vehicle fingerprint adaptation failed for track "
                "%s: %s",
                track_id,
                exc,
            )
            return attributes, None

        return attributes, fingerprint


__all__ = ["TrackVehicleEnricher"]
