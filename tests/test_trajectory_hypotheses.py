"""Focused tests for Step 41 inferred trajectory hypotheses."""

import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from phase2_city import (
    Camera,
    CameraLink,
    CityCameraGraph,
    CrossCameraMatchPolicy,
    FingerprintObservation,
    VehicleFingerprint,
)
from phase3_city import (
    CandidateQueryHistoryRecord,
    CandidateQueryResultRecord,
    HistoricalCandidateRetrievalPolicy,
    HybridResultSnapshot,
    LocalTorchScriptAppearanceEncoder,
    Phase3HybridCandidateMatcher,
    TrajectoryHypothesis,
    TrajectoryHypothesisBuilder,
    TrajectoryHypothesisDataError,
    TrajectoryHypothesisPolicy,
    VehicleAppearanceEmbedding,
)


T0 = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)
EXECUTED = datetime(2026, 9, 20, 11, 0, tzinfo=timezone.utc)
PHYSICAL_POLICY = CrossCameraMatchPolicy(maximum_speed_kph=36.0)
RETRIEVAL_POLICY = HistoricalCandidateRetrievalPolicy(300.0, 10)


def _camera(camera_id):
    return Camera(
        camera_id=camera_id,
        name=f"Camera {camera_id}",
        latitude=28.6,
        longitude=77.2,
        road_name="Demo Road",
        heading_deg=90.0,
    )


def _graph(*links):
    return CityCameraGraph(
        [_camera(name) for name in ("CAM_A", "CAM_B", "CAM_C", "CAM_D")],
        links,
    )


def _link(distance_m=100.0):
    return CameraLink(
        from_camera_id="CAM_A",
        to_camera_id="CAM_B",
        distance_m=distance_m,
        road_name="Demo Road",
        travel_direction="eastbound",
    )


def _fingerprint(plate="GJ01AB1234", colour="white", vehicle_class="car"):
    return VehicleFingerprint(
        normalized_plate=plate,
        vehicle_colour=colour,
        vehicle_class=vehicle_class,
    )


def _observation(
        event_id,
        camera_id,
        seconds,
        *,
        status="accepted",
        plate="GJ01AB1234",
        colour="white",
        vehicle_class="car",
):
    return FingerprintObservation(
        event_id=event_id,
        camera_id=camera_id,
        timestamp=T0 + timedelta(seconds=seconds),
        observation_status=status,
        fingerprint=_fingerprint(plate, colour, vehicle_class),
    )


def _embedding(vector=(1.0, 2.0, 3.0)):
    return VehicleAppearanceEmbedding(
        vector=vector,
        dimension=len(vector),
        model_id="test-reid",
        model_version="1",
        weights_sha256="a" * 64,
        preprocessing_sha256="b" * 64,
    )


def _snapshot(
        source_id="source",
        candidate_id="candidate",
        *,
        source_camera="CAM_A",
        candidate_camera="CAM_B",
        source_seconds=0,
        candidate_seconds=10,
        source_status="accepted",
        candidate_status="accepted",
        source_plate="GJ01AB1234",
        candidate_plate="GJ01AB1234",
        source_colour="white",
        candidate_colour="white",
        source_class="car",
        candidate_class="car",
        source_appearance=None,
        candidate_appearance=None,
        graph=None,
):
    matcher = Phase3HybridCandidateMatcher(
        graph if graph is not None else _graph(),
        PHYSICAL_POLICY,
    )
    result = matcher.compare(
        _observation(
            source_id,
            source_camera,
            source_seconds,
            status=source_status,
            plate=source_plate,
            colour=source_colour,
            vehicle_class=source_class,
        ),
        _observation(
            candidate_id,
            candidate_camera,
            candidate_seconds,
            status=candidate_status,
            plate=candidate_plate,
            colour=candidate_colour,
            vehicle_class=candidate_class,
        ),
        source_appearance=source_appearance,
        candidate_appearance=candidate_appearance,
    )
    return HybridResultSnapshot.from_result(result)


def _canonical_snapshot(payload):
    return HybridResultSnapshot(canonical_json=json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ))


def _result(position, candidate_id, snapshot=None, *, elapsed=None):
    physical = None if snapshot is None else snapshot.to_dict()[
        "physical_feasibility"]
    elapsed_seconds = (
        float(position + 1) if elapsed is None else float(elapsed)
    )
    return CandidateQueryResultRecord(
        position=position,
        candidate_event_id=candidate_id,
        topology_status=(
            "no_direct_link" if physical is None
            else physical["topology_status"]),
        travel_status=(
            "unknown" if physical is None else physical["travel_status"]),
        physical_status=(
            "feasible"
            if physical is not None and physical["gate_status"] == "feasible"
            else "unknown"),
        elapsed_seconds=elapsed_seconds,
        minimum_travel_seconds=(
            None if physical is None
            else physical["minimum_travel_seconds"]),
        hybrid_snapshot=snapshot,
    )


def _history(query_id, source_id, *results):
    return CandidateQueryHistoryRecord(
        query_id=query_id,
        source_event_id=source_id,
        executed_at=EXECUTED,
        retrieval_policy=RETRIEVAL_POLICY,
        physical_policy=PHYSICAL_POLICY,
        rows_considered=len(results),
        physically_impossible_excluded=0,
        returned_count=len(results),
        truncated=False,
        results=results,
    )


def _builder(*, hops=5, hypotheses=10, branching=10):
    return TrajectoryHypothesisBuilder(TrajectoryHypothesisPolicy(
        max_hops=hops,
        max_hypotheses=hypotheses,
        max_branching_per_event=branching,
    ))


def _single_history(snapshot=None, query_id="query-1"):
    snapshot = snapshot or _snapshot()
    return _history(
        query_id,
        snapshot.source_event_id,
        _result(0, snapshot.candidate_event_id, snapshot, elapsed=10),
    )


def test_single_eligible_edge_creates_one_hop_inferred_hypothesis():
    result = _builder().build("source", [_single_history()])

    assert result.count == 1
    hypothesis = result.hypotheses[0]
    assert isinstance(hypothesis, TrajectoryHypothesis)
    assert hypothesis.inferred is True
    assert hypothesis.event_ids == ("source", "candidate")
    assert len(hypothesis.edges) == 1


def test_physically_impossible_edge_is_excluded():
    snapshot = _snapshot(
        candidate_seconds=1,
        graph=_graph(_link(distance_m=100.0)),
    )

    result = _builder().build("source", [_single_history(snapshot)])

    assert snapshot.status == "physically_impossible"
    assert result.count == 0
    assert result.hard_gate_excluded == 1


def test_trusted_plate_contradiction_is_excluded():
    snapshot = _snapshot(candidate_plate="DL01ZZ9999")

    result = _builder().build("source", [_single_history(snapshot)])

    assert snapshot.status == "trusted_plate_contradiction"
    assert result.count == 0
    assert result.hard_gate_excluded == 1


def test_high_appearance_cannot_rescue_physical_impossibility():
    appearance = _embedding()
    snapshot = _snapshot(
        candidate_seconds=1,
        graph=_graph(_link(distance_m=100.0)),
        source_appearance=appearance,
        candidate_appearance=appearance,
    )

    result = _builder().build("source", [_single_history(snapshot)])

    assert snapshot.appearance_cosine_similarity == pytest.approx(1.0)
    assert result.count == 0
    assert result.hard_gate_excluded == 1


def test_high_appearance_cannot_rescue_plate_contradiction():
    appearance = _embedding()
    snapshot = _snapshot(
        candidate_plate="DL01ZZ9999",
        source_appearance=appearance,
        candidate_appearance=appearance,
    )

    result = _builder().build("source", [_single_history(snapshot)])

    assert snapshot.appearance_cosine_similarity == pytest.approx(1.0)
    assert result.count == 0
    assert result.hard_gate_excluded == 1


def test_missing_appearance_is_neutral_when_other_evidence_is_eligible():
    snapshot = _snapshot()

    result = _builder().build("source", [_single_history(snapshot)])

    assert snapshot.appearance_status == "both_unavailable"
    assert result.count == 1
    assert result.hypotheses[0].edges[0].appearance_status == (
        "both_unavailable")


def test_missing_topology_remains_eligible_unknown():
    snapshot = _snapshot(graph=_graph())

    result = _builder().build("source", [_single_history(snapshot)])
    edge = result.hypotheses[0].edges[0]

    assert edge.physical_gate_status == "unknown"
    assert edge.topology_status == "no_direct_link"


def test_same_time_edge_is_rejected():
    snapshot = _snapshot(candidate_seconds=0)
    history = _history(
        "query",
        "source",
        _result(0, "candidate", snapshot, elapsed=1),
    )

    result = _builder().build("source", [history])

    assert result.count == 0
    assert result.temporal_excluded == 1


def test_backwards_time_edge_is_rejected():
    snapshot = _snapshot(candidate_seconds=-1)

    result = _builder().build("source", [_single_history(snapshot)])

    assert result.count == 0
    assert result.hard_gate_excluded == 1


def test_repeated_event_edge_is_rejected():
    snapshot = _snapshot(source_id="same", candidate_id="same")
    history = _history(
        "query",
        "same",
        _result(0, "same", snapshot, elapsed=10),
    )

    result = _builder().build("same", [history])

    assert result.count == 0
    assert result.repeated_event_excluded == 1


def test_cycle_is_prevented_by_consistent_forward_event_metadata():
    forward = _single_history(_snapshot(
        source_id="a", candidate_id="b", source_seconds=0,
        candidate_seconds=10))
    backward = _single_history(_snapshot(
        source_id="b", candidate_id="a", source_seconds=10,
        candidate_seconds=20), query_id="query-2")

    with pytest.raises(TrajectoryHypothesisDataError,
                       match="inconsistent metadata"):
        _builder().build("a", [forward, backward])


def test_multi_hop_hypothesis_preserves_event_and_camera_order():
    first = _single_history(_snapshot(
        source_id="root", candidate_id="middle",
        source_camera="CAM_A", candidate_camera="CAM_B",
        source_seconds=0, candidate_seconds=10))
    second = _single_history(_snapshot(
        source_id="middle", candidate_id="end",
        source_camera="CAM_B", candidate_camera="CAM_C",
        source_seconds=10, candidate_seconds=20), query_id="query-2")

    hypothesis = _builder().build("root", [second, first]).hypotheses[0]

    assert hypothesis.event_ids == ("root", "middle", "end")
    assert hypothesis.camera_ids == ("CAM_A", "CAM_B", "CAM_C")
    assert hypothesis.timestamps == (
        T0, T0 + timedelta(seconds=10), T0 + timedelta(seconds=20))


def test_max_hops_stops_path_expansion():
    histories = [
        _single_history(_snapshot(
            source_id="a", candidate_id="b", source_seconds=0,
            candidate_seconds=10, source_camera="CAM_A",
            candidate_camera="CAM_B"), "q1"),
        _single_history(_snapshot(
            source_id="b", candidate_id="c", source_seconds=10,
            candidate_seconds=20, source_camera="CAM_B",
            candidate_camera="CAM_C"), "q2"),
        _single_history(_snapshot(
            source_id="c", candidate_id="d", source_seconds=20,
            candidate_seconds=30, source_camera="CAM_C",
            candidate_camera="CAM_D"), "q3"),
    ]

    hypothesis = _builder(hops=2).build("a", histories).hypotheses[0]

    assert hypothesis.event_ids == ("a", "b", "c")
    assert len(hypothesis.edges) == 2


def test_max_hypotheses_is_enforced_in_deterministic_order():
    snapshots = [
        _snapshot(candidate_id=name, candidate_seconds=10,
                  candidate_camera=camera)
        for name, camera in (("c", "CAM_D"), ("a", "CAM_B"),
                             ("b", "CAM_C"))
    ]
    history = _history(
        "query",
        "source",
        *(_result(i, item.candidate_event_id, item, elapsed=10)
          for i, item in enumerate(snapshots)),
    )

    result = _builder(hypotheses=2).build("source", [history])

    assert [item.event_ids for item in result.hypotheses] == [
        ("source", "a"), ("source", "b")]
    assert result.hypotheses_truncated is True


def test_max_branching_per_event_is_enforced():
    snapshots = [
        _snapshot(candidate_id=name, candidate_seconds=10,
                  candidate_camera=camera)
        for name, camera in (("c", "CAM_D"), ("a", "CAM_B"),
                             ("b", "CAM_C"))
    ]
    history = _history(
        "query",
        "source",
        *(_result(i, item.candidate_event_id, item, elapsed=10)
          for i, item in enumerate(snapshots)),
    )

    result = _builder(branching=2).build("source", [history])

    assert [item.event_ids for item in result.hypotheses] == [
        ("source", "a"), ("source", "b")]
    assert result.branching_edges_omitted == 1


def test_multiple_continuations_are_branches_not_silent_choice():
    history = _history(
        "query",
        "source",
        _result(0, "later", _snapshot(
            candidate_id="later", candidate_seconds=20), elapsed=20),
        _result(1, "earlier", _snapshot(
            candidate_id="earlier", candidate_seconds=10), elapsed=10),
    )

    result = _builder().build("source", [history])

    assert [item.event_ids for item in result.hypotheses] == [
        ("source", "earlier"), ("source", "later")]


def test_same_candidate_from_multiple_histories_merges_references():
    snapshot = _snapshot()
    first = _single_history(snapshot, "query-b")
    second = _single_history(snapshot, "query-a")

    result = _builder().build("source", [first, second])
    edge = result.hypotheses[0].edges[0]

    assert result.count == 1
    assert result.duplicate_edge_records_merged == 1
    assert [item.query_id for item in edge.history_references] == [
        "query-a", "query-b"]


def test_any_duplicate_hard_contradiction_vetoes_edge():
    eligible = _single_history(_snapshot(), "eligible-query")
    contradiction = _single_history(
        _snapshot(candidate_plate="DL01ZZ9999"),
        "contradiction-query",
    )

    result = _builder().build("source", [eligible, contradiction])

    assert result.count == 0
    assert result.hard_gate_excluded == 1


def test_absent_hybrid_snapshot_is_conservatively_insufficient():
    history = _history(
        "query",
        "source",
        _result(0, "candidate", None, elapsed=10),
    )

    result = _builder().build("source", [history])

    assert result.count == 0
    assert result.insufficient_evidence_excluded == 1


def test_eligible_without_evidence_is_conservatively_excluded():
    snapshot = _snapshot(
        source_status="review",
        candidate_status="review",
        source_plate=None,
        candidate_plate=None,
        source_colour=None,
        candidate_colour=None,
        source_class=None,
        candidate_class=None,
    )

    result = _builder().build("source", [_single_history(snapshot)])

    assert snapshot.status == "eligible_without_evidence"
    assert result.count == 0
    assert result.insufficient_evidence_excluded == 1


def test_edge_evidence_and_query_references_are_preserved():
    snapshot = _snapshot(
        source_appearance=_embedding((1, 2, 3)),
        candidate_appearance=_embedding((3, 2, 1)),
    )

    edge = _builder().build(
        "source", [_single_history(snapshot)]).hypotheses[0].edges[0]

    assert edge.trusted_plate_relation == "agreement"
    assert edge.attribute_outcome == "consistent"
    assert edge.appearance_status == "available"
    assert edge.appearance_cosine_similarity == pytest.approx(5 / 7)
    assert edge.history_references[0].query_id == "query-1"
    assert edge.history_references[0].retrieval_position == 0


def test_policy_and_hypothesis_models_are_immutable():
    policy = TrajectoryHypothesisPolicy(2, 2, 2)
    hypothesis = TrajectoryHypothesisBuilder(policy).build(
        "source", [_single_history()]).hypotheses[0]

    with pytest.raises(FrozenInstanceError):
        policy.max_hops = 3
    with pytest.raises(FrozenInstanceError):
        hypothesis.inferred = False
    with pytest.raises(FrozenInstanceError):
        hypothesis.edges[0].source.camera_id = "changed"


@pytest.mark.parametrize("field,value", [
    ("max_hops", 0),
    ("max_hops", -1),
    ("max_hypotheses", 0),
    ("max_branching_per_event", 0),
    ("max_branching_per_event", 1.5),
    ("max_hops", True),
])
def test_invalid_policy_bounds_are_rejected(field, value):
    values = {
        "max_hops": 2,
        "max_hypotheses": 2,
        "max_branching_per_event": 2,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        TrajectoryHypothesisPolicy(**values)


def test_query_history_inputs_are_not_mutated():
    history = _single_history()
    before = history.to_dict()

    _builder().build("source", [history])

    assert history.to_dict() == before


def test_builder_does_not_rerun_matcher_comparison_or_encoder(
        monkeypatch):
    history = _single_history()

    def unexpected(*args, **kwargs):
        raise AssertionError("inference must consume snapshots only")

    import phase3_city.appearance_comparison as appearance_module
    import phase3_city.hybrid_matching as hybrid_module

    monkeypatch.setattr(
        appearance_module, "compare_appearance_embeddings", unexpected)
    monkeypatch.setattr(
        hybrid_module.Phase3HybridCandidateMatcher, "compare", unexpected)
    monkeypatch.setattr(LocalTorchScriptAppearanceEncoder, "encode", unexpected)

    assert _builder().build("source", [history]).count == 1


def test_builder_does_not_retrieve_or_write_persistence(monkeypatch):
    history = _single_history()

    def unexpected(*args, **kwargs):
        raise AssertionError("hypothesis construction must be read-only")

    from phase3_city.evidence_persistence import (
        SQLiteVehicleEvidenceRepository,
    )
    from phase3_city.historical_candidate_retrieval import (
        HistoricalCandidateRetriever,
    )
    from phase3_city.query_history import (
        SQLiteCandidateQueryHistoryRepository,
    )

    monkeypatch.setattr(HistoricalCandidateRetriever, "retrieve", unexpected)
    monkeypatch.setattr(
        SQLiteCandidateQueryHistoryRepository, "save_query", unexpected)
    monkeypatch.setattr(SQLiteVehicleEvidenceRepository, "save", unexpected)

    assert _builder().build("source", [history]).count == 1


def test_no_identity_or_probability_fields_exist():
    payload = _builder().build(
        "source", [_single_history()]).hypotheses[0].to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert "global_vehicle_id" not in serialized
    assert "canonical_vehicle_id" not in serialized
    assert "identity_probability" not in serialized


def test_hypothesis_id_is_deterministic_for_event_path():
    history = _single_history()

    first = _builder().build("source", [history]).hypotheses[0]
    second = _builder().build("source", [history]).hypotheses[0]

    assert first.hypothesis_id == second.hypothesis_id
    assert first.hypothesis_id.startswith("trajectory-hypothesis-v1:")


def test_missing_candidate_metadata_fails_clearly():
    snapshot = _snapshot()
    payload = snapshot.to_dict()
    del payload["candidate"]["camera_id"]
    malformed = _canonical_snapshot(payload)
    history = _single_history(malformed)

    with pytest.raises(TrajectoryHypothesisDataError,
                       match="missing camera_id"):
        _builder().build("source", [history])


def test_duplicate_positions_in_mutated_external_history_are_rejected():
    first = _result(0, "one", _snapshot(candidate_id="one"), elapsed=10)
    second = _result(1, "two", _snapshot(candidate_id="two"), elapsed=10)
    history = _history("query", "source", first, second)
    object.__setattr__(second, "position", 0)

    with pytest.raises(TrajectoryHypothesisDataError, match="contiguous"):
        _builder().build("source", [history])


def test_inconsistent_candidate_reference_is_rejected():
    result = _result(0, "candidate", _snapshot(), elapsed=10)
    history = _history("query", "source", result)
    object.__setattr__(result, "candidate_event_id", "different")

    with pytest.raises(TrajectoryHypothesisDataError,
                       match="does not match"):
        _builder().build("source", [history])


def test_empty_or_unreachable_history_returns_no_hypothesis():
    result = _builder().build("unrelated", [_single_history()])

    assert result.count == 0
    assert result.hypotheses == ()


def test_exact_phase2_trajectory_type_is_never_returned():
    from phase2_city.trajectory import Trajectory

    hypothesis = _builder().build(
        "source", [_single_history()]).hypotheses[0]

    assert not isinstance(hypothesis, Trajectory)
    assert type(hypothesis).__name__ == "TrajectoryHypothesis"
