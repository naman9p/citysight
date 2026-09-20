"""Focused tests for the prediction-blind Step 34 annotation helper."""

import copy

import cv2
import numpy as np
import pytest
import yaml

from phase1_anpr.evaluation.annotation import (
    AnnotationConflictError,
    AnnotationError,
    AnnotationStore,
    validate_bbox,
)
from phase1_anpr.evaluation.real_world import load_real_world_ground_truth


def _frame(number, *, reviewed=False, plates=()):
    return {
        "frame_number": number,
        "timestamp_seconds": number / 30.0,
        "reviewed": reviewed,
        "plates": list(plates),
    }


def _instance(instance_id="plate-1", *, text="GJ04EJ5933",
              event_id=None):
    return {
        "instance_id": instance_id,
        "plate_text": text,
        "text_evaluable": text is not None,
        "vehicle_identity": None,
        "observation_event_id": event_id,
        "observation_link_reviewed": event_id is not None,
        "notes": None,
    }


def _plate(instance_id="plate-1", bbox=(10, 10, 50, 30)):
    return {"instance_id": instance_id, "bbox": list(bbox), "notes": None}


def _editable_instance(instance_id="plate-1", *, text="GJ04EJ5933",
                       evaluable=True, vehicle_identity=None, notes=None):
    return {
        "instance_id": instance_id,
        "plate_text": text,
        "text_evaluable": evaluable,
        "vehicle_identity": vehicle_identity,
        "notes": notes,
    }


def _workspace(tmp_path, *, frames=None, instances=(), source_overrides=None,
               root_overrides=None, backup=True):
    frames = list(frames or [_frame(0), _frame(3)])
    source = {
        "source_id": "cam01",
        "camera_id": "CAM_01",
        "video_path": "cam01.mp4",
        "observations_reviewed": False,
        "non_plate_observation_event_ids": [],
        "plate_instances": list(instances),
        "frames": frames,
    }
    source.update(source_overrides or {})
    document = {
        "schema_version": 1,
        "benchmark_id": "step34-test",
        "sources": [source],
        "cross_camera_cases": [],
    }
    document.update(root_overrides or {})
    manifest = tmp_path / "ground_truth.yaml"
    manifest.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    frames_dir = tmp_path / "frames" / "cam01"
    frames_dir.mkdir(parents=True)
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    for frame in frames:
        assert cv2.imwrite(
            str(frames_dir / f"frame_{frame['frame_number']:08d}.jpg"), image)
    return manifest, AnnotationStore(
        manifest, tmp_path / "frames", create_backup=backup)


def _payload(store, frame_number, *, reviewed=True, plates=(), instances=(),
             confirm_reviewed_update=False):
    payload = {
        "revision": store.bootstrap()["revision"],
        "source_id": "cam01",
        "frame_number": frame_number,
        "reviewed": reviewed,
        "plates": list(plates),
        "instances": list(instances),
    }
    if confirm_reviewed_update:
        payload["confirm_reviewed_update"] = True
    return payload


@pytest.mark.parametrize("bbox", [
    [0, 0, 10],
    [0, 0, 0, 10],
    [0, -1, 10, 10],
    [0, 0, float("inf"), 10],
    [0, 0, 101, 10],
    [0, 0, 10, 61],
])
def test_bbox_validation_rejects_invalid_or_out_of_image_boxes(bbox):
    with pytest.raises(AnnotationError, match="bbox"):
        validate_bbox(bbox, width=100, height=60)


def test_bbox_validation_retains_full_resolution_coordinates():
    assert validate_bbox([10, 11, 90, 55], width=100, height=60) == [10, 11, 90, 55]


def test_instance_creation_and_reuse_keeps_one_physical_instance(tmp_path):
    manifest, store = _workspace(tmp_path)
    created = store.save_frame(_payload(
        store,
        0,
        plates=[_plate("passage-1")],
        instances=[_editable_instance("passage-1")],
    ))
    store.save_frame({
        **_payload(
            store,
            3,
            plates=[_plate("passage-1", (12, 11, 52, 31))],
            instances=[_editable_instance("passage-1")],
        ),
        "revision": created["revision"],
    })

    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    source = document["sources"][0]
    assert [item["instance_id"] for item in source["plate_instances"]] == ["passage-1"]
    assert [frame["plates"][0]["instance_id"] for frame in source["frames"]] == [
        "passage-1", "passage-1"]


def test_deleting_the_last_box_removes_only_the_now_unused_instance(tmp_path):
    instance = _instance(event_id=None)
    frame = _frame(0, reviewed=True, plates=[_plate()])
    manifest, store = _workspace(
        tmp_path, frames=[frame, _frame(3)], instances=[instance])

    store.save_frame(_payload(
        store, 0, reviewed=True, confirm_reviewed_update=True))

    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    source = document["sources"][0]
    assert source["frames"][0]["plates"] == []
    assert source["plate_instances"] == []


def test_changing_reviewed_frame_requires_explicit_confirmation(tmp_path):
    instance = _instance(event_id=None)
    frame = _frame(0, reviewed=True, plates=[_plate()])
    manifest, store = _workspace(
        tmp_path, frames=[frame, _frame(3)], instances=[instance])
    original = manifest.read_bytes()

    with pytest.raises(AnnotationError, match="explicit confirmation"):
        store.save_frame(_payload(store, 0, reviewed=True))

    assert manifest.read_bytes() == original


def test_prediction_blind_helper_will_not_delete_replay_link_fields(tmp_path):
    instance = _instance(event_id="event-1")
    frame = _frame(0, reviewed=True, plates=[_plate()])
    manifest, store = _workspace(
        tmp_path, frames=[frame, _frame(3)], instances=[instance])
    original = manifest.read_bytes()

    with pytest.raises(AnnotationError, match="replay-linked"):
        store.save_frame(_payload(
            store, 0, reviewed=True, confirm_reviewed_update=True))

    assert manifest.read_bytes() == original


def test_reviewed_negative_frame_is_persisted(tmp_path):
    manifest, store = _workspace(tmp_path)

    result = store.save_frame(_payload(store, 0, reviewed=True))

    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert document["sources"][0]["frames"][0]["reviewed"] is True
    assert document["sources"][0]["frames"][0]["plates"] == []
    assert result["progress"]["reviewed"] == 1
    assert result["progress"]["remaining"] == 1


def test_persistence_is_atomic_and_keeps_previous_bytes_as_backup(tmp_path):
    manifest, store = _workspace(tmp_path)
    original = manifest.read_bytes()

    store.save_frame(_payload(store, 0, reviewed=True))

    assert manifest.with_suffix(".yaml.bak").read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
    loaded = load_real_world_ground_truth(manifest)
    assert loaded.sources[0].frames[0].reviewed is True


def test_stale_revision_cannot_overwrite_a_completed_annotation(tmp_path):
    manifest, store = _workspace(tmp_path)
    stale = store.bootstrap()["revision"]
    store.save_frame(_payload(store, 0, reviewed=True))
    completed = manifest.read_bytes()

    with pytest.raises(AnnotationConflictError, match="changed"):
        store.save_frame({
            **_payload(store, 0, reviewed=False),
            "revision": stale,
        })

    assert manifest.read_bytes() == completed


def test_invalid_instance_reference_is_rejected_without_writing(tmp_path):
    manifest, store = _workspace(tmp_path)
    original = manifest.read_bytes()

    with pytest.raises(AnnotationError, match="unknown instance_id"):
        store.save_frame(_payload(
            store, 0, plates=[_plate("missing")], instances=[]))

    assert manifest.read_bytes() == original


def test_unreadable_plate_is_stored_as_null_and_not_text_evaluable(tmp_path):
    manifest, store = _workspace(tmp_path)

    store.save_frame(_payload(
        store,
        0,
        plates=[_plate("unreadable-1")],
        instances=[_editable_instance(
            "unreadable-1", text=None, evaluable=False)],
    ))

    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    instance = document["sources"][0]["plate_instances"][0]
    assert instance["plate_text"] is None
    assert instance["text_evaluable"] is False


def test_save_preserves_existing_manifest_content_and_event_fields(tmp_path):
    existing_instance = _instance(event_id="event-1")
    existing_frame = _frame(0, reviewed=True, plates=[_plate()])
    frames = [existing_frame, _frame(3)]
    overrides = {
        "observations_reviewed": True,
        "non_plate_observation_event_ids": ["nonplate-1"],
    }
    manifest, store = _workspace(
        tmp_path, frames=frames, instances=[existing_instance],
        source_overrides=overrides)
    before = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    preserved_frame = copy.deepcopy(before["sources"][0]["frames"][0])
    preserved_instance = copy.deepcopy(before["sources"][0]["plate_instances"][0])

    store.save_frame(_payload(store, 3, reviewed=True))

    after = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    source = after["sources"][0]
    assert after["schema_version"] == before["schema_version"]
    assert after["benchmark_id"] == before["benchmark_id"]
    assert after["cross_camera_cases"] == before["cross_camera_cases"]
    assert source["camera_id"] == "CAM_01"
    assert source["video_path"] == "cam01.mp4"
    assert source["observations_reviewed"] is True
    assert source["non_plate_observation_event_ids"] == ["nonplate-1"]
    assert source["frames"][0] == preserved_frame
    assert source["plate_instances"][0] == preserved_instance
