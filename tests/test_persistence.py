"""Tests for Step 12 persistence layer (repository + evidence store).

No running Postgres/MinIO/Docker: sqlite is in-memory/tmp and evidence uses a
temp dir.
"""

import pytest

from phase1_anpr.observation.observation_builder import PlateObservation
from phase1_anpr.persistence import (
    EvidenceError,
    LocalFilesystemEvidenceStore,
    SQLiteObservationRepository,
)


def make_obs(event_id="e-1", status="accepted", plate="MH12AB1234",
             image_path=None):
    abstained = status == "abstained"
    return PlateObservation(
        event_id=event_id,
        camera_id="cam_01",
        track_id=7,
        timestamp="2026-08-31T10:00:00+05:30",
        plate_raw=None if abstained else plate,
        plate_normalized=None if abstained else plate,
        confidence=0.2 if abstained else 0.9,
        status=status,
        detector_confidence=None if abstained else 0.8,
        ocr_confidence=None if abstained else 0.85,
        quality_score=None if abstained else 0.7,
        best_frame_number=None if abstained else 42,
        plate_image_path=image_path,
        model_version="phase1-anpr-0.1.0",
    )


@pytest.fixture
def repo():
    r = SQLiteObservationRepository(":memory:")
    yield r
    r.close()


# --- repository ---------------------------------------------------------------

def test_save_and_get_roundtrip(repo):
    inserted = repo.save(make_obs(), format_type="standard", state_code="MH")
    assert inserted is True
    row = repo.get("e-1")
    assert row["camera_id"] == "cam_01"
    assert row["plate_normalized"] == "MH12AB1234"
    assert row["format_type"] == "standard"
    assert row["state_code"] == "MH"
    assert row["confidence"] == pytest.approx(0.9)


def test_idempotent_event_id(repo):
    assert repo.save(make_obs(event_id="dup")) is True
    # Retrying the same event does not duplicate or overwrite.
    assert repo.save(make_obs(event_id="dup", plate="XXXXXX")) is False
    assert repo.count() == 1
    assert repo.get("dup")["plate_normalized"] == "MH12AB1234"


def test_abstained_asserts_no_identity(repo):
    repo.save(make_obs(event_id="abs", status="abstained"))
    row = repo.get("abs")
    assert row["status"] == "abstained"
    assert row["plate_raw"] is None
    assert row["plate_normalized"] is None
    assert row["evidence_ref"] is None


def test_get_missing_returns_none(repo):
    assert repo.get("nope") is None


def test_evidence_ref_defaults_to_image_path(repo):
    repo.save(make_obs(event_id="withimg", image_path="e-9.jpg"))
    assert repo.get("withimg")["evidence_ref"] == "e-9.jpg"


# --- evidence store -----------------------------------------------------------

def test_store_evidence_deterministic_key(tmp_path):
    src = tmp_path / "crop.jpg"
    src.write_bytes(b"\xff\xd8fake")
    store = LocalFilesystemEvidenceStore(tmp_path / "evi")
    ref1 = store.store("event-abc", src)
    ref2 = store.store("event-abc", src)
    assert ref1 == ref2 == "event-abc.jpg"
    assert store.exists(ref1)
    assert store.resolve(ref1).read_bytes() == b"\xff\xd8fake"


def test_store_missing_source_raises(tmp_path):
    store = LocalFilesystemEvidenceStore(tmp_path / "evi")
    with pytest.raises(EvidenceError):
        store.store("event-x", tmp_path / "does-not-exist.jpg")
    with pytest.raises(EvidenceError):
        store.store("event-x", None)


def test_exists_false_for_missing_or_empty(tmp_path):
    store = LocalFilesystemEvidenceStore(tmp_path / "evi")
    assert store.exists(None) is False
    assert store.exists("ghost.jpg") is False
