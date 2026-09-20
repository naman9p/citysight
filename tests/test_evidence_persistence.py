"""Focused tests for Step 38 observation-scoped Phase 3 persistence."""

import math
import sqlite3
import struct
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from phase1_anpr.observation.observation_builder import PlateObservation
from phase1_anpr.persistence import SQLiteObservationRepository
from phase2_city import FingerprintObservation, VehicleFingerprint
from phase3_city import (
    EMBEDDING_SERIALIZATION_FORMAT,
    EMBEDDING_SERIALIZATION_VERSION,
    EMBEDDING_VECTOR_TOLERANCE,
    SQLiteVehicleEvidenceRepository,
    VEHICLE_EVIDENCE_SCHEMA_VERSION,
    VehicleAppearanceEmbedding,
    VehicleEvidenceConflictError,
    VehicleEvidenceCorruptionError,
    VehicleEvidencePersistenceError,
    VehicleEvidenceRecord,
    compare_appearance_embeddings,
)


_TIMESTAMP = datetime(2026, 9, 20, 10, 30, tzinfo=timezone.utc)
_WEIGHTS_A = "a" * 64
_WEIGHTS_B = "b" * 64
_PREPROCESSING = "c" * 64


@pytest.fixture
def repo():
    repository = SQLiteVehicleEvidenceRepository()
    try:
        yield repository
    finally:
        repository.close()


def _embedding(
        vector=(1.0, 2.0, 3.0),
        *,
        model_id="test-reid",
        model_version="1",
        weights_sha256=_WEIGHTS_A,
        preprocessing_sha256=_PREPROCESSING,
):
    return VehicleAppearanceEmbedding(
        vector=vector,
        dimension=len(vector),
        model_id=model_id,
        model_version=model_version,
        weights_sha256=weights_sha256,
        preprocessing_sha256=preprocessing_sha256,
    )


def _record(
        *,
        event_id="event-1",
        camera_id="CAM_A",
        timestamp=_TIMESTAMP,
        status="accepted",
        plate="GJ01AB1234",
        colour="white",
        vehicle_class="car",
        appearance=None,
):
    return VehicleEvidenceRecord(
        event_id=event_id,
        camera_id=camera_id,
        timestamp=timestamp,
        observation_status=status,
        fingerprint=VehicleFingerprint(
            normalized_plate=plate,
            vehicle_colour=colour,
            vehicle_class=vehicle_class,
        ),
        appearance_embedding=appearance,
    )


def _row(repo, event_id="event-1"):
    return repo._conn.execute(
        "SELECT * FROM phase3_vehicle_evidence WHERE event_id = ?",
        (event_id,),
    ).fetchone()


def test_schema_initializes_on_empty_sqlite_database(tmp_path):
    db_path = tmp_path / "phase3.db"
    repository = SQLiteVehicleEvidenceRepository(db_path)
    try:
        table = repository._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("phase3_vehicle_evidence",),
        ).fetchone()
        columns = {
            row[1]
            for row in repository._conn.execute(
                "PRAGMA table_info(phase3_vehicle_evidence)").fetchall()
        }
    finally:
        repository.close()

    assert table is not None
    assert {
        "event_id",
        "camera_id",
        "observation_status",
        "normalized_plate",
        "vehicle_colour",
        "vehicle_class",
        "record_schema_version",
        "embedding_serialization_version",
        "embedding_serialization_format",
        "appearance_present",
        "embedding_vector",
        "embedding_dimension",
        "embedding_model_id",
        "embedding_model_version",
        "embedding_weights_sha256",
        "embedding_preprocessing_sha256",
    }.issubset(columns)


def test_schema_initialization_is_idempotent(tmp_path):
    db_path = tmp_path / "phase3.db"
    first = SQLiteVehicleEvidenceRepository(db_path)
    first._migrate()
    first.close()

    second = SQLiteVehicleEvidenceRepository(db_path)
    try:
        second._migrate()
        tables = second._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type = 'table' AND name = ?",
            ("phase3_vehicle_evidence",),
        ).fetchone()[0]
    finally:
        second.close()

    assert tables == 1


def test_save_vehicle_evidence_uses_event_primary_key(repo):
    record = _record(appearance=None)

    assert repo.save(record) is True
    assert repo.exists(record.event_id) is True
    assert repo.exists("missing") is False
    assert repo._conn.execute(
        "SELECT COUNT(*) FROM phase3_vehicle_evidence"
    ).fetchone()[0] == 1


def test_load_vehicle_evidence_reconstructs_validated_record(repo):
    original = _record(appearance=None)
    repo.save(original)

    loaded = repo.get(original.event_id)

    assert loaded == original
    assert loaded.to_fingerprint_observation() == FingerprintObservation(
        event_id=original.event_id,
        camera_id=original.camera_id,
        timestamp=original.timestamp,
        observation_status=original.observation_status,
        fingerprint=original.fingerprint,
    )
    assert repo.get("missing") is None


def test_save_appearance_embedding_uses_explicit_binary_format(repo):
    embedding = _embedding((3.0, 4.0))
    repo.save(_record(appearance=embedding))

    row = _row(repo)

    assert row["appearance_present"] == 1
    assert row["embedding_serialization_format"] == (
        EMBEDDING_SERIALIZATION_FORMAT)
    assert isinstance(row["embedding_vector"], bytes)
    assert len(row["embedding_vector"]) == embedding.dimension * 8
    assert struct.unpack("<2d", row["embedding_vector"]) == embedding.vector


def test_load_appearance_embedding_returns_step35_value_object(repo):
    embedding = _embedding((3.0, 4.0))
    repo.save(_record(appearance=embedding))

    loaded = repo.get_appearance("event-1")

    assert isinstance(loaded, VehicleAppearanceEmbedding)
    assert loaded.vector == pytest.approx(embedding.vector, abs=1e-15)


def test_embedding_round_trip_preserves_vector_and_all_provenance(repo):
    embedding = _embedding(
        (0.123456789012345, -0.987654321098765, 0.456789012345678),
        model_id="model / unusual",
        model_version="2026.09+local",
    )
    repo.save(_record(appearance=embedding))

    loaded = repo.get_appearance("event-1")

    assert loaded.dimension == embedding.dimension
    assert loaded.model_id == embedding.model_id
    assert loaded.model_version == embedding.model_version
    assert loaded.weights_sha256 == embedding.weights_sha256
    assert loaded.preprocessing_sha256 == embedding.preprocessing_sha256
    assert loaded.vector == pytest.approx(
        embedding.vector, rel=0.0, abs=EMBEDDING_VECTOR_TOLERANCE)


def test_schema_and_serialization_versions_are_preserved(repo):
    original = _record(appearance=_embedding())
    repo.save(original)

    loaded = repo.get(original.event_id)
    row = _row(repo)

    assert loaded.schema_version == VEHICLE_EVIDENCE_SCHEMA_VERSION
    assert loaded.embedding_serialization_version == (
        EMBEDDING_SERIALIZATION_VERSION)
    assert row["record_schema_version"] == VEHICLE_EVIDENCE_SCHEMA_VERSION
    assert row["embedding_serialization_version"] == (
        EMBEDDING_SERIALIZATION_VERSION)


def test_missing_appearance_is_explicitly_absent_without_placeholder(repo):
    repo.save(_record(appearance=None))

    loaded = repo.get("event-1")
    row = _row(repo)

    assert loaded.appearance_embedding is None
    assert repo.get_appearance("event-1") is None
    assert row["appearance_present"] == 0
    assert row["embedding_vector"] is None
    assert row["embedding_dimension"] is None
    assert row["embedding_model_id"] is None


@pytest.mark.parametrize("invalid_vector", [
    (),
    (0.0, 0.0),
    (1.0, math.nan),
    (1.0, math.inf),
])
def test_invalid_embedding_is_rejected_before_persistence(repo, invalid_vector):
    with pytest.raises(ValueError):
        _embedding(invalid_vector)

    assert repo._conn.execute(
        "SELECT COUNT(*) FROM phase3_vehicle_evidence"
    ).fetchone()[0] == 0


@pytest.mark.parametrize(("column", "value"), [
    ("embedding_vector", sqlite3.Binary(struct.pack("<3d", math.nan, 0.0, 1.0))),
    ("embedding_weights_sha256", "corrupted"),
])
def test_corrupted_persisted_embedding_fails_clearly(repo, column, value):
    repo.save(_record(appearance=_embedding()))
    repo._conn.execute(
        f"UPDATE phase3_vehicle_evidence SET {column} = ? WHERE event_id = ?",
        (value, "event-1"),
    )
    repo._conn.commit()

    with pytest.raises(VehicleEvidenceCorruptionError):
        repo.get("event-1")


def test_absent_embedding_marker_with_payload_fails_clearly(repo):
    repo.save(_record(appearance=None))
    repo._conn.execute("PRAGMA ignore_check_constraints = ON")
    repo._conn.execute(
        "UPDATE phase3_vehicle_evidence SET embedding_vector = ? "
        "WHERE event_id = ?",
        (sqlite3.Binary(struct.pack("<d", 1.0)), "event-1"),
    )
    repo._conn.commit()

    with pytest.raises(VehicleEvidenceCorruptionError):
        repo.get("event-1")


def test_duplicate_save_is_idempotent_and_missing_does_not_erase(repo):
    embedding = _embedding()
    original = _record(appearance=embedding)

    assert repo.save(original) is True
    assert repo.save(original) is False
    assert repo.save(_record(appearance=None)) is False
    loaded = repo.get_appearance("event-1")
    assert loaded is not None
    assert loaded.to_dict() == embedding.to_dict()


def test_monotonic_upsert_adds_missing_fields_and_appearance(repo):
    sparse = _record(
        plate=None,
        colour=None,
        vehicle_class=None,
        appearance=None,
    )
    enriched = _record(appearance=_embedding())

    assert repo.save(sparse) is True
    assert repo.save(enriched) is True

    loaded = repo.get("event-1")
    assert loaded.fingerprint == enriched.fingerprint
    assert loaded.appearance_embedding is not None


@pytest.mark.parametrize("conflicting", [
    _record(camera_id="CAM_B"),
    _record(colour="black"),
    _record(vehicle_class="truck"),
])
def test_conflicting_retry_is_rejected_without_changing_original(
        repo, conflicting):
    original = _record()
    repo.save(original)

    with pytest.raises(VehicleEvidenceConflictError):
        repo.save(conflicting)

    assert repo.get("event-1") == original


def test_event_cannot_silently_receive_incompatible_embedding_provenance(repo):
    original = _record(appearance=_embedding(weights_sha256=_WEIGHTS_A))
    incompatible = _record(appearance=_embedding(weights_sha256=_WEIGHTS_B))
    repo.save(original)

    with pytest.raises(
            VehicleEvidenceConflictError,
            match="incompatible embedding provenance"):
        repo.save(incompatible)

    assert repo.get("event-1").appearance_embedding.weights_sha256 == (
        _WEIGHTS_A)


def test_event_cannot_silently_receive_different_embedding_vector(repo):
    original = _record(appearance=_embedding((1.0, 0.0, 0.0)))
    different = _record(appearance=_embedding((0.0, 1.0, 0.0)))
    repo.save(original)

    with pytest.raises(
            VehicleEvidenceConflictError,
            match="different embedding vector"):
        repo.save(different)


def test_transaction_rolls_back_failed_update(repo):
    original = _record(appearance=None)
    repo.save(original)
    repo._conn.executescript("""
        CREATE TRIGGER fail_phase3_update
        BEFORE UPDATE ON phase3_vehicle_evidence
        BEGIN
            SELECT RAISE(ABORT, 'forced test failure');
        END;
    """)

    with pytest.raises(VehicleEvidencePersistenceError, match="forced test"):
        repo.save(_record(appearance=_embedding()))

    assert repo.get("event-1") == original


def test_parameterized_sql_handles_unusual_identifiers_and_text(repo):
    event_id = "event'; DROP TABLE phase3_vehicle_evidence; --"
    record = _record(
        event_id=event_id,
        camera_id="CAM '?; --",
        colour="blue ' quoted",
        appearance=_embedding(model_id="model '?; --"),
    )

    assert repo.save(record) is True
    assert repo.get(event_id) == record
    assert repo._conn.execute(
        "SELECT COUNT(*) FROM phase3_vehicle_evidence"
    ).fetchone()[0] == 1


def test_loaded_record_is_immutable(repo):
    repo.save(_record(appearance=_embedding()))
    loaded = repo.get("event-1")

    with pytest.raises(FrozenInstanceError):
        loaded.camera_id = "CAM_B"
    with pytest.raises(FrozenInstanceError):
        loaded.fingerprint.vehicle_colour = "black"
    with pytest.raises(FrozenInstanceError):
        loaded.appearance_embedding.dimension = 4


def test_save_does_not_mutate_source_record(repo):
    original = _record(appearance=_embedding())
    before = original.to_dict()

    repo.save(original)

    assert original.to_dict() == before


def test_step36_comparison_is_preserved_after_round_trip(repo):
    source_embedding = _embedding((1.0, 2.0, 3.0))
    candidate_embedding = _embedding((3.0, 2.0, 1.0))
    expected = compare_appearance_embeddings(
        source_embedding, candidate_embedding)
    repo.save(_record(event_id="source", appearance=source_embedding))
    repo.save(_record(event_id="candidate", appearance=candidate_embedding))

    actual = compare_appearance_embeddings(
        repo.get_appearance("source"),
        repo.get_appearance("candidate"),
    )

    assert actual.status == expected.status
    assert actual.cosine_similarity == pytest.approx(
        expected.cosine_similarity, rel=0.0, abs=1e-12)
    assert actual.euclidean_distance == pytest.approx(
        expected.euclidean_distance, rel=0.0, abs=1e-12)
    assert actual.source_provenance == expected.source_provenance


def test_additive_schema_remains_compatible_with_phase1_database(tmp_path):
    db_path = tmp_path / "shared.db"
    phase1 = SQLiteObservationRepository(db_path)
    observation = PlateObservation(
        event_id="phase1-event",
        camera_id="CAM_A",
        track_id=1,
        timestamp=_TIMESTAMP.isoformat(),
        plate_raw="GJ01AB1234",
        plate_normalized="GJ01AB1234",
        confidence=0.9,
        status="accepted",
        detector_confidence=0.9,
        ocr_confidence=0.9,
        quality_score=0.8,
        best_frame_number=5,
        plate_image_path=None,
        model_version="phase1-test",
    )
    phase1.save(observation)
    phase1.close()

    phase3 = SQLiteVehicleEvidenceRepository(db_path)
    phase3.save(_record(event_id="phase1-event", appearance=_embedding()))
    phase3.close()

    reopened = SQLiteObservationRepository(db_path)
    try:
        assert reopened.get("phase1-event")["plate_normalized"] == (
            "GJ01AB1234")
        assert reopened.count() == 1
    finally:
        reopened.close()
