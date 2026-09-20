"""Focused tests for Step 40 immutable query-history persistence."""

import json
import math
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from phase2_city import (
    Camera,
    CityCameraGraph,
    CrossCameraMatchPolicy,
    FingerprintObservation,
    VehicleFingerprint,
)
from phase3_city import (
    CandidateQueryHistoryRecord,
    CandidateQueryResultRecord,
    HistoricalCandidateRetrievalPolicy,
    HistoricalCandidateRetriever,
    HybridResultSnapshot,
    Phase3HybridCandidateMatcher,
    QUERY_HISTORY_IMPLEMENTATION_ID,
    QUERY_HISTORY_SCHEMA_VERSION,
    QueryHistoryConflictError,
    QueryHistoryPersistenceError,
    QueryHistoryValidationError,
    RETRIEVAL_ORDER_KIND,
    SQLiteCandidateQueryHistoryRepository,
    SQLiteVehicleEvidenceRepository,
    VehicleAppearanceEmbedding,
    VehicleEvidenceRecord,
    build_candidate_query_history,
)


T0 = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)
EXECUTED = datetime(2026, 9, 20, 11, 0, tzinfo=timezone.utc)
PHYSICAL_POLICY = CrossCameraMatchPolicy(
    maximum_speed_kph=60.0,
    travel_time_tolerance_seconds=1.5,
    minimum_evidence_coverage=0.25,
    strong_similarity_threshold=0.7,
)
RETRIEVAL_POLICY = HistoricalCandidateRetrievalPolicy(
    max_time_delta_seconds=300.0,
    max_results=10,
)


@pytest.fixture
def repo():
    repository = SQLiteCandidateQueryHistoryRepository()
    try:
        yield repository
    finally:
        repository.close()


def _result(position, event_id, *, hybrid=None, seconds=None):
    elapsed = float(position + 1 if seconds is None else seconds)
    return CandidateQueryResultRecord(
        position=position,
        candidate_event_id=event_id,
        topology_status="no_direct_link",
        travel_status="unknown",
        physical_status="unknown",
        elapsed_seconds=elapsed,
        minimum_travel_seconds=None,
        hybrid_snapshot=hybrid,
    )


def _query(
        *results,
        query_id="query-1",
        source_event_id="source",
        retrieval_policy=RETRIEVAL_POLICY,
        physical_policy=PHYSICAL_POLICY,
        rows_considered=None,
        physically_impossible_excluded=0,
        truncated=False,
):
    count = len(results)
    return CandidateQueryHistoryRecord(
        query_id=query_id,
        source_event_id=source_event_id,
        executed_at=EXECUTED,
        retrieval_policy=retrieval_policy,
        physical_policy=physical_policy,
        rows_considered=(count if rows_considered is None
                         else rows_considered),
        physically_impossible_excluded=physically_impossible_excluded,
        returned_count=count,
        truncated=truncated,
        results=results,
    )


def _camera(camera_id):
    return Camera(
        camera_id=camera_id,
        name=f"Camera {camera_id}",
        latitude=28.6,
        longitude=77.2,
        road_name="Demo Road",
        heading_deg=90.0,
    )


def _graph():
    return CityCameraGraph(
        [_camera(name) for name in ("CAM_A", "CAM_B", "CAM_C", "CAM_D")],
        [],
    )


def _fingerprint(plate="GJ01AB1234"):
    return VehicleFingerprint(
        normalized_plate=plate,
        vehicle_colour="white",
        vehicle_class="car",
    )


def _observation(event_id, camera_id, seconds, *, plate="GJ01AB1234"):
    return FingerprintObservation(
        event_id=event_id,
        camera_id=camera_id,
        timestamp=T0 + timedelta(seconds=seconds),
        observation_status="accepted",
        fingerprint=_fingerprint(plate),
    )


def _embedding(vector):
    return VehicleAppearanceEmbedding(
        vector=vector,
        dimension=len(vector),
        model_id="test-reid",
        model_version="1",
        weights_sha256="a" * 64,
        preprocessing_sha256="b" * 64,
    )


def _hybrid_result(
        candidate_event_id="candidate",
        *,
        source_plate="GJ01AB1234",
        candidate_plate="GJ01AB1234",
        source_appearance=None,
        candidate_appearance=None,
):
    matcher = Phase3HybridCandidateMatcher(_graph(), PHYSICAL_POLICY)
    return matcher.compare(
        _observation("source", "CAM_A", 0, plate=source_plate),
        _observation(
            candidate_event_id,
            "CAM_B",
            10,
            plate=candidate_plate,
        ),
        source_appearance=source_appearance,
        candidate_appearance=candidate_appearance,
    )


def _snapshot(**kwargs):
    return HybridResultSnapshot.from_result(_hybrid_result(**kwargs))


def _evidence(event_id, camera_id, seconds, appearance=None):
    return VehicleEvidenceRecord(
        event_id=event_id,
        camera_id=camera_id,
        timestamp=T0 + timedelta(seconds=seconds),
        observation_status="accepted",
        fingerprint=_fingerprint(),
        appearance_embedding=appearance,
    )


def _retrieval_result(*, limit=2):
    evidence_repo = SQLiteVehicleEvidenceRepository()
    try:
        evidence_repo.save(_evidence("source", "CAM_A", 0))
        evidence_repo.save(_evidence("z", "CAM_B", 10))
        evidence_repo.save(_evidence("a", "CAM_C", 10))
        evidence_repo.save(_evidence("later", "CAM_D", 20))
        retriever = HistoricalCandidateRetriever(
            repository=evidence_repo,
            city_graph=_graph(),
            physical_policy=PHYSICAL_POLICY,
            retrieval_policy=HistoricalCandidateRetrievalPolicy(300, limit),
        )
        return retriever.retrieve("source")
    finally:
        evidence_repo.close()


def test_schema_initializes_on_empty_database(repo):
    tables = {
        row[0]
        for row in repo._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }

    assert "phase3_candidate_queries" in tables
    assert "phase3_candidate_query_results" in tables
    assert repo._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_schema_initialization_is_idempotent(tmp_path):
    path = tmp_path / "history.db"
    first = SQLiteCandidateQueryHistoryRepository(path)
    first._migrate()
    first.close()
    second = SQLiteCandidateQueryHistoryRepository(path)
    try:
        second._migrate()
        tables = second._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' "
            "AND name IN (?, ?)",
            ("phase3_candidate_queries", "phase3_candidate_query_results"),
        ).fetchone()[0]
    finally:
        second.close()

    assert tables == 2


@pytest.mark.parametrize("results", [
    (),
    (_result(0, "candidate-1"),),
    (_result(0, "candidate-1"), _result(1, "candidate-2")),
])
def test_save_and_load_zero_one_or_multiple_candidates(repo, results):
    record = _query(*results)

    assert repo.save_query(record) is True
    assert repo.get_query(record.query_id) == record
    assert repo.get_query_results(record.query_id) == results
    assert repo.exists(record.query_id) is True


def test_candidate_order_is_preserved_exactly(repo):
    record = _query(
        _result(0, "z-event"),
        _result(1, "a-event"),
        _result(2, "middle-event"),
    )
    repo.save_query(record)

    assert [item.candidate_event_id
            for item in repo.get_query_results("query-1")] == [
                "z-event", "a-event", "middle-event"
            ]


@pytest.mark.parametrize("results", [
    (_result(1, "candidate"),),
    (_result(0, "one"), _result(2, "two")),
])
def test_positions_must_be_zero_based_and_contiguous(results):
    with pytest.raises(QueryHistoryValidationError, match="contiguous"):
        _query(*results)


def test_duplicate_candidate_within_query_is_rejected():
    with pytest.raises(QueryHistoryValidationError, match="unique"):
        _query(_result(0, "duplicate"), _result(1, "duplicate"))


def test_identical_retry_is_idempotent(repo):
    record = _query(_result(0, "candidate"))

    assert repo.save_query(record) is True
    assert repo.save_query(record) is False
    assert repo._conn.execute(
        "SELECT COUNT(*) FROM phase3_candidate_queries"
    ).fetchone()[0] == 1


@pytest.mark.parametrize("conflicting", [
    _query(_result(0, "candidate"), source_event_id="other-source"),
    _query(
        _result(0, "candidate"),
        retrieval_policy=HistoricalCandidateRetrievalPolicy(600, 10),
    ),
    _query(_result(0, "other")),
    _query(_result(0, "candidate", seconds=2)),
])
def test_conflicting_query_id_cannot_overwrite_history(repo, conflicting):
    original = _query(_result(0, "candidate"))
    repo.save_query(original)

    with pytest.raises(QueryHistoryConflictError):
        repo.save_query(conflicting)

    assert repo.get_query("query-1") == original


def test_conflicting_candidate_order_is_rejected(repo):
    original = _query(_result(0, "first"), _result(1, "second"))
    reordered = _query(_result(0, "second"), _result(1, "first"))
    repo.save_query(original)

    with pytest.raises(QueryHistoryConflictError):
        repo.save_query(reordered)


def test_conflicting_hybrid_evidence_is_rejected(repo):
    original = _query(_result(0, "candidate", hybrid=_snapshot()))
    changed = _query(_result(
        0,
        "candidate",
        hybrid=_snapshot(candidate_plate="DL01ZZ9999"),
    ))
    repo.save_query(original)

    with pytest.raises(QueryHistoryConflictError):
        repo.save_query(changed)


def test_failed_candidate_insert_rolls_back_whole_query(repo):
    repo._conn.executescript("""
        CREATE TRIGGER fail_second_query_result
        BEFORE INSERT ON phase3_candidate_query_results
        WHEN NEW.position = 1
        BEGIN
            SELECT RAISE(ABORT, 'forced result failure');
        END;
    """)
    record = _query(_result(0, "first"), _result(1, "second"))

    with pytest.raises(QueryHistoryPersistenceError, match="forced result"):
        repo.save_query(record)

    assert repo.exists("query-1") is False
    assert repo._conn.execute(
        "SELECT COUNT(*) FROM phase3_candidate_query_results"
    ).fetchone()[0] == 0


def test_unusual_query_and_event_ids_are_parameterized(repo):
    query_id = "query'; DROP TABLE phase3_candidate_queries; --"
    source_id = "source'? --"
    candidate_id = "candidate'; SELECT 1; --"
    record = _query(
        _result(0, candidate_id),
        query_id=query_id,
        source_event_id=source_id,
    )

    repo.save_query(record)

    assert repo.get_query(query_id) == record
    assert repo.exists(query_id)


def test_retrieval_policy_and_diagnostics_round_trip(repo):
    record = _query(
        _result(0, "candidate"),
        retrieval_policy=HistoricalCandidateRetrievalPolicy(123.5, 3),
        rows_considered=7,
        physically_impossible_excluded=2,
        truncated=True,
    )
    repo.save_query(record)

    loaded = repo.get_query("query-1")

    assert loaded.retrieval_policy == record.retrieval_policy
    assert loaded.physical_policy == PHYSICAL_POLICY
    assert loaded.rows_considered == 7
    assert loaded.physically_impossible_excluded == 2
    assert loaded.returned_count == 1
    assert loaded.truncated is True


def test_optional_hybrid_snapshot_absent(repo):
    record = _query(_result(0, "candidate"))
    repo.save_query(record)

    loaded = repo.get_query_results("query-1")[0]

    assert loaded.hybrid_snapshot is None


def test_optional_hybrid_snapshot_present_and_structured(repo):
    snapshot = _snapshot()
    repo.save_query(_query(_result(0, "candidate", hybrid=snapshot)))

    loaded = repo.get_query_results("query-1")[0].hybrid_snapshot

    assert loaded == snapshot
    assert loaded.status == "eligible_with_evidence"
    assert loaded.physical_gate_status == "unknown"
    assert loaded.trusted_plate_relation == "agreement"
    assert loaded.attribute_outcome == "consistent"


def test_appearance_unavailable_state_is_preserved(repo):
    snapshot = _snapshot()
    repo.save_query(_query(_result(0, "candidate", hybrid=snapshot)))

    loaded = repo.get_query_results("query-1")[0].hybrid_snapshot

    assert loaded.appearance_status == "both_unavailable"
    assert loaded.appearance_cosine_similarity is None
    assert loaded.appearance_provenance_compatible is None


def test_appearance_similarity_and_provenance_are_preserved_as_evidence(repo):
    snapshot = _snapshot(
        source_appearance=_embedding((1.0, 2.0, 3.0)),
        candidate_appearance=_embedding((3.0, 2.0, 1.0)),
    )
    repo.save_query(_query(_result(0, "candidate", hybrid=snapshot)))

    loaded = repo.get_query_results("query-1")[0].hybrid_snapshot
    payload = loaded.to_dict()

    assert loaded.appearance_status == "available"
    assert loaded.appearance_cosine_similarity == pytest.approx(5 / 7)
    assert loaded.appearance_provenance_compatible is True
    assert payload["appearance_comparison"]["source_provenance"] == (
        payload["appearance_comparison"]["candidate_provenance"])
    assert "probability" not in loaded.canonical_json


def test_schema_and_implementation_provenance_round_trip(repo):
    record = _query()
    repo.save_query(record)
    loaded = repo.get_query("query-1")

    assert loaded.schema_version == QUERY_HISTORY_SCHEMA_VERSION
    assert loaded.implementation_id == QUERY_HISTORY_IMPLEMENTATION_ID
    assert loaded.retrieval_order_kind == RETRIEVAL_ORDER_KIND
    assert loaded.retrieval_policy_version == 1
    assert loaded.physical_policy_version == 1


def test_loaded_dtos_are_immutable(repo):
    snapshot = _snapshot()
    repo.save_query(_query(_result(0, "candidate", hybrid=snapshot)))
    loaded = repo.get_query("query-1")

    with pytest.raises(FrozenInstanceError):
        loaded.query_id = "changed"
    with pytest.raises(FrozenInstanceError):
        loaded.results[0].position = 2
    with pytest.raises(FrozenInstanceError):
        loaded.results[0].hybrid_snapshot.canonical_json = "{}"


def test_source_input_objects_are_not_mutated(repo):
    snapshot = _snapshot()
    result = _result(0, "candidate", hybrid=snapshot)
    record = _query(result)
    before = record.to_dict()

    repo.save_query(record)

    assert record.to_dict() == before
    assert result.hybrid_snapshot == snapshot


def test_repository_does_not_call_hybrid_or_appearance_logic(
        repo, monkeypatch):
    import phase3_city.appearance_comparison as appearance_module
    import phase3_city.hybrid_matching as hybrid_module

    def unexpected(*args, **kwargs):
        raise AssertionError("matching/comparison must not run")

    monkeypatch.setattr(
        appearance_module, "compare_appearance_embeddings", unexpected)
    monkeypatch.setattr(
        hybrid_module.Phase3HybridCandidateMatcher, "compare", unexpected)

    record = _query(_result(0, "candidate"))
    repo.save_query(record)

    assert repo.get_query("query-1") == record


def test_builder_preserves_exact_step39_order_and_diagnostics():
    retrieval = _retrieval_result(limit=2)

    record = build_candidate_query_history(
        query_id="query-from-step39",
        executed_at=EXECUTED,
        retrieval_result=retrieval,
        physical_policy=PHYSICAL_POLICY,
    )

    assert [item.candidate_event_id for item in record.results] == ["a", "z"]
    assert [item.position for item in record.results] == [0, 1]
    assert record.returned_count == retrieval.count
    assert record.rows_considered == retrieval.rows_considered
    assert record.truncated == retrieval.truncated is True
    assert record.retrieval_order_kind == RETRIEVAL_ORDER_KIND


def test_builder_accepts_only_supplied_candidate_hybrid_results():
    retrieval = _retrieval_result(limit=2)
    hybrid = _hybrid_result(candidate_event_id="a")

    record = build_candidate_query_history(
        query_id="query-with-hybrid",
        executed_at=EXECUTED,
        retrieval_result=retrieval,
        physical_policy=PHYSICAL_POLICY,
        hybrid_results={"a": hybrid},
    )

    assert record.results[0].candidate_event_id == "a"
    assert record.results[0].hybrid_snapshot is not None
    assert record.results[1].hybrid_snapshot is None


def test_builder_rejects_hybrid_result_outside_retrieval_set():
    retrieval = _retrieval_result(limit=1)

    with pytest.raises(QueryHistoryValidationError, match="outside"):
        build_candidate_query_history(
            query_id="query",
            executed_at=EXECUTED,
            retrieval_result=retrieval,
            physical_policy=PHYSICAL_POLICY,
            hybrid_results={"not-returned": _hybrid_result("not-returned")},
        )


def test_shared_step38_database_remains_compatible(tmp_path):
    path = tmp_path / "shared.db"
    evidence_repo = SQLiteVehicleEvidenceRepository(path)
    evidence = _evidence("source", "CAM_A", 0, _embedding((1, 2, 3)))
    evidence_repo.save(evidence)
    evidence_repo.close()

    history_repo = SQLiteCandidateQueryHistoryRepository(path)
    history_repo.save_query(_query())
    history_repo.close()

    reopened = SQLiteVehicleEvidenceRepository(path)
    try:
        assert reopened.get("source") == evidence
    finally:
        reopened.close()


def test_missing_query_is_distinct_from_zero_result_query(repo):
    assert repo.get_query("missing") is None
    assert repo.get_query_results("missing") is None
    repo.save_query(_query())
    assert repo.get_query_results("query-1") == ()


@pytest.mark.parametrize("value", [None, 1, "", "   "])
def test_query_id_validation(value, repo):
    with pytest.raises(QueryHistoryValidationError, match="query_id"):
        repo.get_query(value)


def test_nonfinite_candidate_evidence_is_rejected():
    with pytest.raises(QueryHistoryValidationError, match="elapsed_seconds"):
        _result(0, "candidate", seconds=math.nan)


def test_canonical_hybrid_snapshot_rejects_noncanonical_json():
    snapshot = _snapshot()

    with pytest.raises(QueryHistoryValidationError, match="canonical form"):
        HybridResultSnapshot(
            canonical_json=json.dumps(snapshot.to_dict(), sort_keys=True)
        )
