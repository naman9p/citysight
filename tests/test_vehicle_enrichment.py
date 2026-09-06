"""Step 26 track-level vehicle enrichment tests (no real ML models)."""

from types import SimpleNamespace

import numpy as np

from phase1_anpr.observation.observation_builder import PlateObservation
from phase2_city.fingerprint import VehicleFingerprint
from phase2_city.vehicle_attributes import VehicleAttributes
from phase2_city.vehicle_detection import (
    UltralyticsVehicleDetector,
    VehicleDetection,
)
from phase2_city.vehicle_enrichment import TrackVehicleEnricher


def _track(track_id=1, bbox=(20, 80, 80, 100), frame_number=0):
    return SimpleNamespace(
        track_id=track_id,
        bbox=bbox,
        frame_number=frame_number,
    )


def _vehicle(bbox=(0, 0, 120, 120), frame_number=0,
             vehicle_class="car", confidence=0.9):
    class_id = {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}[
        vehicle_class]
    return VehicleDetection(
        bbox=bbox,
        confidence=confidence,
        class_id=class_id,
        vehicle_class=vehicle_class,
        frame_number=frame_number,
    )


def _observation(status="accepted", plate="GJ01AB1234", track_id=1):
    return PlateObservation(
        event_id=f"event-{track_id}",
        camera_id="CAM_01",
        track_id=track_id,
        timestamp="2026-08-31T10:00:00+00:00",
        plate_raw=plate,
        plate_normalized=plate,
        confidence=0.9,
        status=status,
        detector_confidence=0.9,
        ocr_confidence=0.9,
        quality_score=0.8,
        best_frame_number=0,
        plate_image_path=None,
        model_version="phase1-anpr-0.1.0",
    )


class RecordingVehicleDetector:
    def __init__(self, detections=(), error=None):
        self.detections = list(detections)
        self.error = error
        self.calls = []

    def detect(self, frame, frame_number):
        self.calls.append((frame, frame_number))
        if self.error is not None:
            raise self.error
        return list(self.detections)


def test_two_tracks_share_one_frame_inference_and_never_retry():
    frame = np.zeros((140, 320, 3), dtype=np.uint8)
    tracks = [
        _track(1),
        _track(2, bbox=(180, 80, 240, 100)),
    ]
    detector = RecordingVehicleDetector([
        _vehicle(),
        _vehicle(bbox=(150, 0, 300, 120), vehicle_class="truck"),
    ])
    enricher = TrackVehicleEnricher(detector)

    enricher.enrich_frame(frame, 0, tracks)
    for track in tracks:
        track.frame_number = 1
    enricher.enrich_frame(frame, 1, tracks)

    assert len(detector.calls) == 1
    assert enricher.attributes_by_track_id[1].vehicle_class == "car"
    assert enricher.attributes_by_track_id[2].vehicle_class == "truck"


def test_zero_new_tracks_causes_zero_detector_calls():
    detector = RecordingVehicleDetector([_vehicle()])
    enricher = TrackVehicleEnricher(detector)

    enricher.enrich_frame(np.zeros((10, 10, 3), dtype=np.uint8), 0, [])

    assert detector.calls == []


def test_clean_unmatched_is_none_result_and_builds_plate_only_fingerprint():
    detector = RecordingVehicleDetector([])
    enricher = TrackVehicleEnricher(detector)
    track = _track()

    enricher.enrich_frame(np.zeros((120, 120, 3), dtype=np.uint8), 0, [track])

    assert track.track_id in enricher.attempted_track_ids
    assert track.track_id in enricher.attributes_by_track_id
    assert enricher.attributes_by_track_id[track.track_id] is None

    attributes, fingerprint = enricher.finalize_track(
        track.track_id, _observation())
    assert attributes is None
    assert fingerprint == VehicleFingerprint(normalized_plate="GJ01AB1234")
    assert track.track_id not in enricher.attempted_track_ids
    assert track.track_id not in enricher.attributes_by_track_id


def test_colour_abstention_preserves_associated_vehicle_class():
    # Black frame is deliberately underexposed for Step 25 colour extraction.
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    enricher = TrackVehicleEnricher(
        RecordingVehicleDetector([_vehicle()]))

    enricher.enrich_frame(frame, 0, [_track()])
    attributes, fingerprint = enricher.finalize_track(1, _observation())

    assert attributes == VehicleAttributes(
        vehicle_class="car", class_confidence=0.9)
    assert fingerprint == VehicleFingerprint(
        normalized_plate="GJ01AB1234", vehicle_class="car")


def test_none_vehicle_crop_preserves_class_evidence():
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    enricher = TrackVehicleEnricher(
        RecordingVehicleDetector([_vehicle()]),
        crop_fn=lambda frame, detection: None,
    )

    enricher.enrich_frame(frame, 0, [_track()])
    attributes, _ = enricher.finalize_track(1, _observation())

    assert attributes == VehicleAttributes(
        vehicle_class="car", class_confidence=0.9)


def test_detector_exception_is_terminal_and_never_retried():
    detector = RecordingVehicleDetector(error=RuntimeError("detector failed"))
    enricher = TrackVehicleEnricher(detector)
    track = _track()
    frame = np.zeros((120, 120, 3), dtype=np.uint8)

    enricher.enrich_frame(frame, 0, [track])
    track.frame_number = 1
    enricher.enrich_frame(frame, 1, [track])

    assert len(detector.calls) == 1
    assert track.track_id in enricher.attempted_track_ids
    assert track.track_id not in enricher.attributes_by_track_id
    assert enricher.finalize_track(track.track_id, _observation()) == (None, None)
    assert track.track_id not in enricher.attempted_track_ids


def test_missing_optional_local_weights_fail_safely_without_model_loading(
        tmp_path):
    detector = UltralyticsVehicleDetector(
        weights_path=tmp_path / "missing.pt",
        device="cpu",
        model_factory=lambda path: (_ for _ in ()).throw(
            AssertionError("model factory must not be called")),
    )
    enricher = TrackVehicleEnricher(detector)

    enricher.enrich_frame(
        np.zeros((120, 120, 3), dtype=np.uint8), 0, [_track()])

    assert enricher.attempted_track_ids == {1}
    assert enricher.attributes_by_track_id == {}
    assert enricher.finalize_track(1, _observation()) == (None, None)


def test_association_failure_for_one_track_does_not_block_another():
    frame = np.zeros((140, 320, 3), dtype=np.uint8)
    good_track = _track(2, bbox=(180, 80, 240, 100))

    def association(track, detections):
        if track.track_id == 1:
            raise RuntimeError("bad plate geometry")
        from phase2_city.plate_vehicle_association import (
            associate_plate_to_vehicle,
        )
        return associate_plate_to_vehicle(track, detections)

    enricher = TrackVehicleEnricher(
        RecordingVehicleDetector([
            _vehicle(),
            _vehicle(bbox=(150, 0, 300, 120), vehicle_class="bus"),
        ]),
        association_fn=association,
    )

    enricher.enrich_frame(frame, 0, [_track(1), good_track])

    assert 1 in enricher.attempted_track_ids
    assert 1 not in enricher.attributes_by_track_id
    assert enricher.attributes_by_track_id[2].vehicle_class == "bus"


def test_crop_failure_for_one_track_does_not_block_another():
    frame = np.zeros((140, 320, 3), dtype=np.uint8)

    def crop(frame, detection):
        if detection.vehicle_class == "car":
            raise RuntimeError("crop failed")
        from phase2_city.vehicle_detection import crop_vehicle
        return crop_vehicle(frame, detection)

    enricher = TrackVehicleEnricher(
        RecordingVehicleDetector([
            _vehicle(),
            _vehicle(bbox=(150, 0, 300, 120), vehicle_class="bus"),
        ]),
        crop_fn=crop,
    )

    enricher.enrich_frame(
        frame, 0, [_track(1), _track(2, bbox=(180, 80, 240, 100))])

    assert 1 not in enricher.attributes_by_track_id
    assert enricher.attributes_by_track_id[2].vehicle_class == "bus"


def test_extractor_failure_for_one_track_does_not_block_another():
    frame = np.zeros((140, 320, 3), dtype=np.uint8)

    def extract(crop, association):
        if association.vehicle_detection.vehicle_class == "car":
            raise RuntimeError("extraction failed")
        return VehicleAttributes(
            vehicle_class=association.vehicle_detection.vehicle_class,
            class_confidence=association.vehicle_detection.confidence,
        )

    enricher = TrackVehicleEnricher(
        RecordingVehicleDetector([
            _vehicle(),
            _vehicle(bbox=(150, 0, 300, 120), vehicle_class="truck"),
        ]),
        attribute_extractor=extract,
    )

    enricher.enrich_frame(
        frame, 0, [_track(1), _track(2, bbox=(180, 80, 240, 100))])

    assert 1 not in enricher.attributes_by_track_id
    assert enricher.attributes_by_track_id[2].vehicle_class == "truck"


def test_fingerprint_failure_preserves_attributes_and_cleans_state():
    attributes = VehicleAttributes(
        vehicle_colour="blue",
        colour_confidence=0.8,
        vehicle_class="car",
        class_confidence=0.9,
    )
    enricher = TrackVehicleEnricher(
        RecordingVehicleDetector([_vehicle()]),
        attribute_extractor=lambda crop, association: attributes,
        fingerprint_adapter=lambda observation, value: (_ for _ in ()).throw(
            RuntimeError("fingerprint failed")),
    )
    enricher.enrich_frame(
        np.zeros((120, 120, 3), dtype=np.uint8), 0, [_track()])

    result_attributes, fingerprint = enricher.finalize_track(1, _observation())

    assert result_attributes is attributes
    assert fingerprint is None
    assert enricher.attempted_track_ids == set()
    assert enricher.attributes_by_track_id == {}


def test_identical_inputs_produce_deterministic_results():
    def run_once():
        enricher = TrackVehicleEnricher(
            RecordingVehicleDetector([_vehicle(vehicle_class="motorcycle")]))
        enricher.enrich_frame(
            np.zeros((120, 120, 3), dtype=np.uint8), 0, [_track()])
        return enricher.finalize_track(1, _observation())

    assert run_once() == run_once()
