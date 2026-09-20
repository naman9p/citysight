"""Focused tests for Step 39 bounded historical candidate retrieval."""

import math
import random
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from phase2_city import (
    Camera,
    CameraLink,
    CityCameraGraph,
    CrossCameraMatchPolicy,
    VehicleFingerprint,
)
from phase3_city import (
    HistoricalCandidateRetrievalError,
    HistoricalCandidateRetrievalPolicy,
    HistoricalCandidateRetrievalResult,
    HistoricalCandidateRetriever,
    SQLiteVehicleEvidenceRepository,
    VehicleAppearanceEmbedding,
    VehicleEvidenceRecord,
)


T0 = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
def repo():
    repository = SQLiteVehicleEvidenceRepository()
    try:
        yield repository
    finally:
        repository.close()


def _camera(camera_id):
    return Camera(
        camera_id=camera_id,
        name=f"Camera {camera_id}",
        latitude=28.6,
        longitude=77.2,
        road_name="Demo Road",
        heading_deg=90.0,
    )


def _link(source="CAM_A", candidate="CAM_B", distance_m=100.0):
    return CameraLink(
        from_camera_id=source,
        to_camera_id=candidate,
        distance_m=distance_m,
        road_name="Demo Road",
        travel_direction="eastbound",
    )


def _graph(*links, camera_ids=("CAM_A", "CAM_B", "CAM_C", "CAM_D")):
    return CityCameraGraph(
        [_camera(camera_id) for camera_id in camera_ids],
        links,
    )


def _embedding():
    return VehicleAppearanceEmbedding(
        vector=(1.0, 2.0, 3.0),
        dimension=3,
        model_id="test-reid",
        model_version="1",
        weights_sha256="a" * 64,
        preprocessing_sha256="b" * 64,
    )


def _record(
        event_id,
        camera_id,
        seconds,
        *,
        plate="GJ01AB1234",
        appearance=None,
):
    return VehicleEvidenceRecord(
        event_id=event_id,
        camera_id=camera_id,
        timestamp=None if seconds is None else T0 + timedelta(seconds=seconds),
        observation_status="accepted",
        fingerprint=VehicleFingerprint(
            normalized_plate=plate,
            vehicle_colour="white",
            vehicle_class="car",
        ),
        appearance_embedding=appearance,
    )


def _save(repo, *records):
    for record in records:
        repo.save(record)


def _retriever(
        repo,
        *,
        graph=None,
        window=30.0,
        limit=10,
        speed=36.0,
        tolerance=0.0,
):
    return HistoricalCandidateRetriever(
        repository=repo,
        city_graph=graph if graph is not None else _graph(),
        physical_policy=CrossCameraMatchPolicy(
            maximum_speed_kph=speed,
            travel_time_tolerance_seconds=tolerance,
        ),
        retrieval_policy=HistoricalCandidateRetrievalPolicy(
            max_time_delta_seconds=window,
            max_results=limit,
        ),
    )


def _ids(result):
    return [candidate.evidence.event_id for candidate in result.candidates]


def test_strict_future_only_excludes_source_and_earlier_event(repo):
    _save(
        repo,
        _record("earlier", "CAM_B", -1),
        _record("source", "CAM_A", 0),
        _record("later", "CAM_B", 1),
    )

    result = _retriever(repo).retrieve("source")

    assert _ids(result) == ["later"]
    assert "source" not in _ids(result)
    assert "earlier" not in _ids(result)


def test_exact_upper_boundary_is_included(repo):
    _save(
        repo,
        _record("source", "CAM_A", 0),
        _record("boundary", "CAM_B", 30),
    )

    assert _ids(_retriever(repo, window=30).retrieve("source")) == [
        "boundary"
    ]


def test_just_outside_upper_boundary_is_excluded(repo):
    _save(
        repo,
        _record("source", "CAM_A", 0),
        _record("outside", "CAM_B", 30.000001),
    )

    assert _retriever(repo, window=30).retrieve("source").count == 0


def test_same_camera_is_excluded_and_different_camera_retained(repo):
    _save(
        repo,
        _record("source", "CAM_A", 0),
        _record("same", "CAM_A", 1),
        _record("cross", "CAM_B", 1),
    )

    assert _ids(_retriever(repo).retrieve("source")) == ["cross"]


def test_missing_direct_topology_remains_eligible_and_unknown(repo):
    _save(
        repo,
        _record("source", "CAM_A", 0, plate="GJ01AB1234"),
        _record("candidate", "CAM_B", 1, plate="DL01ZZ9999"),
    )

    result = _retriever(repo, graph=_graph()).retrieve("source")

    assert _ids(result) == ["candidate"]
    assert result.candidates[0].topology_status == "no_direct_link"
    assert result.candidates[0].travel_status == "unknown"
    assert result.candidates[0].physical_status == "unknown"


def test_unknown_camera_topology_remains_eligible(repo):
    _save(
        repo,
        _record("source", "CAM_A", 0),
        _record("candidate", "UNREGISTERED", 1),
    )

    candidate = _retriever(repo, graph=_graph()).retrieve(
        "source").candidates[0]

    assert candidate.topology_status == "unknown_camera"
    assert candidate.physical_status == "unknown"


def test_physically_impossible_direct_link_candidate_is_excluded(repo):
    _save(
        repo,
        _record("source", "CAM_A", 0),
        _record("too-fast", "CAM_B", 1),
    )

    result = _retriever(
        repo,
        graph=_graph(_link(distance_m=100.0)),
        speed=36.0,
    ).retrieve("source")

    assert result.count == 0
    assert result.physically_impossible_excluded == 1


def test_physically_feasible_direct_link_candidate_is_retained(repo):
    _save(
        repo,
        _record("source", "CAM_A", 0),
        _record("feasible", "CAM_B", 10),
    )

    candidate = _retriever(
        repo,
        graph=_graph(_link(distance_m=100.0)),
        speed=36.0,
    ).retrieve("source").candidates[0]

    assert candidate.evidence.event_id == "feasible"
    assert candidate.travel_status == "feasible"
    assert candidate.minimum_travel_seconds == pytest.approx(10.0)


def test_negative_and_same_time_candidates_never_reach_physical_results(repo):
    _save(
        repo,
        _record("earlier", "CAM_B", -1),
        _record("same-time", "CAM_B", 0),
        _record("source", "CAM_A", 0),
    )

    result = _retriever(repo).retrieve("source")

    assert result.count == 0
    assert result.rows_considered == 0


def test_candidates_order_by_timestamp_then_event_id(repo):
    records = [
        _record("z-later", "CAM_B", 2),
        _record("z-tie", "CAM_C", 1),
        _record("a-tie", "CAM_D", 1),
        _record("source", "CAM_A", 0),
    ]
    random.Random(39).shuffle(records)
    _save(repo, *records)

    assert _ids(_retriever(repo).retrieve("source")) == [
        "a-tie", "z-tie", "z-later"
    ]


def test_max_results_is_enforced_after_physical_filtering(repo):
    _save(
        repo,
        _record("source", "CAM_A", 0),
        _record("impossible-1", "CAM_B", 1),
        _record("impossible-2", "CAM_B", 2),
        _record("valid-later", "CAM_B", 10),
    )

    result = _retriever(
        repo,
        graph=_graph(_link(distance_m=100.0)),
        limit=1,
        speed=36.0,
    ).retrieve("source")

    assert _ids(result) == ["valid-later"]
    assert result.physically_impossible_excluded == 2
    assert result.truncated is False


def test_deterministic_truncation_retains_first_eligible_candidates(repo):
    _save(
        repo,
        _record("source", "CAM_A", 0),
        _record("c", "CAM_B", 1),
        _record("a", "CAM_C", 1),
        _record("b", "CAM_D", 1),
    )

    result = _retriever(repo, limit=2).retrieve("source")

    assert _ids(result) == ["a", "b"]
    assert result.count == 2
    assert result.truncated is True


def test_max_results_one_boundary(repo):
    _save(
        repo,
        _record("source", "CAM_A", 0),
        _record("first", "CAM_B", 1),
        _record("second", "CAM_C", 2),
    )

    result = _retriever(repo, limit=1).retrieve("source")

    assert _ids(result) == ["first"]
    assert result.truncated is True


@pytest.mark.parametrize("value", [0, -1, True, 1.0, "1", None])
def test_invalid_max_results_is_rejected(value):
    with pytest.raises(HistoricalCandidateRetrievalError,
                       match="max_results"):
        HistoricalCandidateRetrievalPolicy(30, value)


@pytest.mark.parametrize("value", [
    0,
    -1,
    math.nan,
    math.inf,
    -math.inf,
    True,
    "30",
    None,
])
def test_invalid_time_window_is_rejected(value):
    with pytest.raises(HistoricalCandidateRetrievalError,
                       match="max_time_delta_seconds"):
        HistoricalCandidateRetrievalPolicy(value, 1)


def test_empty_history_returns_empty_result(repo):
    repo.save(_record("source", "CAM_A", 0))

    result = _retriever(repo).retrieve("source")

    assert result.count == 0
    assert result.candidates == ()
    assert result.truncated is False


def test_retrieval_does_not_invoke_appearance_or_hybrid_matching(
        repo, monkeypatch):
    import phase3_city.appearance_comparison as appearance_module
    import phase3_city.hybrid_matching as hybrid_module

    def unexpected(*args, **kwargs):
        raise AssertionError("appearance/hybrid matching must not run")

    monkeypatch.setattr(
        appearance_module, "compare_appearance_embeddings", unexpected)
    monkeypatch.setattr(
        hybrid_module.Phase3HybridCandidateMatcher, "compare", unexpected)
    _save(
        repo,
        _record("source", "CAM_A", 0, appearance=_embedding()),
        _record("candidate", "CAM_B", 1, appearance=_embedding()),
    )

    result = _retriever(repo).retrieve("source")

    assert _ids(result) == ["candidate"]
    assert result.candidates[0].evidence.appearance_embedding is not None


def test_source_and_candidate_records_are_not_mutated(repo):
    source = _record("source", "CAM_A", 0, appearance=_embedding())
    candidate = _record("candidate", "CAM_B", 1, appearance=_embedding())
    source_before = source.to_dict()
    candidate_before = candidate.to_dict()
    _save(repo, source, candidate)

    _retriever(repo).retrieve("source")

    assert source.to_dict() == source_before
    assert candidate.to_dict() == candidate_before


def test_policy_result_and_candidate_are_immutable(repo):
    policy = HistoricalCandidateRetrievalPolicy(30, 1)
    _save(
        repo,
        _record("source", "CAM_A", 0),
        _record("candidate", "CAM_B", 1),
    )
    retriever = HistoricalCandidateRetriever(
        repo,
        _graph(),
        CrossCameraMatchPolicy(maximum_speed_kph=36),
        policy,
    )
    result = retriever.retrieve("source")

    assert isinstance(result, HistoricalCandidateRetrievalResult)
    with pytest.raises(FrozenInstanceError):
        policy.max_results = 2
    with pytest.raises(FrozenInstanceError):
        result.truncated = True
    with pytest.raises(FrozenInstanceError):
        result.candidates[0].physical_status = "impossible"


def test_unusual_event_and_camera_ids_are_parameterized_safely(repo):
    source_id = "source'; DROP TABLE phase3_vehicle_evidence; --"
    camera_id = "CAM_A' OR 1=1; --"
    _save(
        repo,
        _record(source_id, camera_id, 0),
        _record("candidate'? --", "CAM_B", 1),
    )

    result = _retriever(repo).retrieve(source_id)

    assert _ids(result) == ["candidate'? --"]
    assert repo.exists(source_id)


def test_schema_index_initialization_is_idempotent(tmp_path):
    db_path = tmp_path / "phase3.db"
    first = SQLiteVehicleEvidenceRepository(db_path)
    first._migrate()
    first.close()
    second = SQLiteVehicleEvidenceRepository(db_path)
    try:
        second._migrate()
        indexes = second._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND name = ?",
            ("idx_phase3_vehicle_evidence_timestamp_event",),
        ).fetchall()
    finally:
        second.close()

    assert len(indexes) == 1


def test_step38_save_and_load_remain_unchanged(repo):
    record = _record("event", "CAM_A", 0, appearance=_embedding())

    assert repo.save(record) is True
    assert repo.save(record) is False
    assert repo.get("event") == record
    assert repo.get_appearance("event") == record.appearance_embedding


def test_invalid_source_requests_fail_clearly(repo):
    for event_id in (None, 1, "", "   "):
        with pytest.raises(HistoricalCandidateRetrievalError,
                           match="source_event_id"):
            _retriever(repo).retrieve(event_id)
    with pytest.raises(HistoricalCandidateRetrievalError, match="not found"):
        _retriever(repo).retrieve("missing")


def test_source_without_timestamp_is_rejected(repo):
    repo.save(_record("source", "CAM_A", None))

    with pytest.raises(HistoricalCandidateRetrievalError,
                       match="no usable timestamp"):
        _retriever(repo).retrieve("source")


def test_result_serialization_is_deterministic(repo):
    _save(
        repo,
        _record("source", "CAM_A", 0),
        _record("candidate", "CAM_B", 1),
    )

    result = _retriever(repo).retrieve("source")
    serialized = result.to_dict()

    assert serialized == result.to_dict()
    assert serialized["source_event_id"] == "source"
    assert serialized["count"] == 1
    assert serialized["candidates"][0]["evidence"]["event_id"] == (
        "candidate")
