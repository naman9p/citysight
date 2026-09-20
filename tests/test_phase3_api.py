"""Focused Step 42 read-only Phase 3 API/dashboard tests."""

import json
import threading
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from phase1_anpr.api import create_server
from phase1_anpr.observation.observation_builder import PlateObservation
from phase1_anpr.persistence import SQLiteObservationRepository
from phase1_anpr.serve import build_phase3_read_api
from phase2_city import (
    Camera,
    CityCameraGraph,
    CrossCameraMatchPolicy,
    FingerprintObservation,
    VehicleFingerprint,
)
from phase2_city.trajectory import TrajectoryReconstructor
from phase3_city import (
    CandidateQueryHistoryRecord,
    CandidateQueryResultRecord,
    HistoricalCandidateRetrievalPolicy,
    HybridResultSnapshot,
    Phase3HybridCandidateMatcher,
    Phase3ReadApi,
    QueryHistoryCorruptionError,
    SQLiteCandidateQueryHistoryRepository,
    SQLiteVehicleEvidenceRepository,
    VehicleAppearanceEmbedding,
    VehicleEvidenceCorruptionError,
    VehicleEvidencePersistenceError,
    VehicleEvidenceRecord,
)


T0 = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)
PHYSICAL_POLICY = CrossCameraMatchPolicy(maximum_speed_kph=60.0)
RETRIEVAL_POLICY = HistoricalCandidateRetrievalPolicy(300.0, 10)
PLATE = "GJ01AB1234"


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
        [_camera(name) for name in
         ("CAM_A", "CAM_B", "CAM_C", "CAM_D")],
        [],
    )


def _fingerprint(plate=PLATE):
    return VehicleFingerprint(
        normalized_plate=plate,
        vehicle_colour="white",
        vehicle_class="car",
    )


def _fingerprint_observation(event_id, camera_id, seconds, *, plate=PLATE):
    return FingerprintObservation(
        event_id=event_id,
        camera_id=camera_id,
        timestamp=T0 + timedelta(seconds=seconds),
        observation_status="accepted",
        fingerprint=_fingerprint(plate),
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
        source_id,
        candidate_id,
        source_camera,
        candidate_camera,
        source_seconds,
        candidate_seconds,
        *,
        candidate_plate=PLATE,
        appearance=False,
):
    source_embedding = _embedding() if appearance else None
    candidate_embedding = _embedding() if appearance else None
    result = Phase3HybridCandidateMatcher(_graph(), PHYSICAL_POLICY).compare(
        _fingerprint_observation(
            source_id, source_camera, source_seconds),
        _fingerprint_observation(
            candidate_id,
            candidate_camera,
            candidate_seconds,
            plate=candidate_plate,
        ),
        source_appearance=source_embedding,
        candidate_appearance=candidate_embedding,
    )
    return HybridResultSnapshot.from_result(result)


def _result(position, snapshot, elapsed_seconds):
    physical = snapshot.to_dict()["physical_feasibility"]
    return CandidateQueryResultRecord(
        position=position,
        candidate_event_id=snapshot.candidate_event_id,
        topology_status=physical["topology_status"],
        travel_status=physical["travel_status"],
        physical_status=(
            "feasible" if physical["gate_status"] == "feasible"
            else "unknown"),
        elapsed_seconds=elapsed_seconds,
        minimum_travel_seconds=physical["minimum_travel_seconds"],
        hybrid_snapshot=snapshot,
    )


def _query(query_id, source_id, *results):
    return CandidateQueryHistoryRecord(
        query_id=query_id,
        source_event_id=source_id,
        executed_at=T0 + timedelta(hours=1),
        retrieval_policy=RETRIEVAL_POLICY,
        physical_policy=PHYSICAL_POLICY,
        rows_considered=len(results),
        physically_impossible_excluded=0,
        returned_count=len(results),
        truncated=False,
        results=results,
    )


def _plate_observation(event_id, camera_id, seconds):
    timestamp = (T0 + timedelta(seconds=seconds)).isoformat()
    return PlateObservation(
        event_id=event_id,
        camera_id=camera_id,
        track_id=1,
        timestamp=timestamp,
        plate_raw=PLATE,
        plate_normalized=PLATE,
        confidence=0.9,
        status="accepted",
        detector_confidence=0.8,
        ocr_confidence=0.85,
        quality_score=0.7,
        best_frame_number=5,
        plate_image_path=None,
        model_version="test",
    )


@pytest.fixture
def phase3_client():
    evidence = SQLiteVehicleEvidenceRepository(":memory:")
    evidence.save(VehicleEvidenceRecord.from_observation(
        _fingerprint_observation("root", "CAM_A", 0),
        _embedding(),
    ))
    evidence.save(VehicleEvidenceRecord.from_observation(
        _fingerprint_observation("no-appearance", "CAM_C", 20),
    ))

    root_to_b = _snapshot(
        "root", "b", "CAM_A", "CAM_B", 0, 10, appearance=True)
    root_to_c = _snapshot(
        "root", "c", "CAM_A", "CAM_C", 0, 20)
    blocked = _snapshot(
        "root", "blocked", "CAM_A", "CAM_D", 0, 30,
        candidate_plate="DL01ZZ9999", appearance=True)
    b_to_d = _snapshot(
        "b", "d", "CAM_B", "CAM_D", 10, 25)

    queries = SQLiteCandidateQueryHistoryRepository(":memory:")
    # Supplied retrieval order intentionally differs from hypothesis time order.
    queries.save_query(_query(
        "q-root",
        "root",
        _result(0, root_to_c, 20),
        _result(1, root_to_b, 10),
        _result(2, blocked, 30),
    ))
    queries.save_query(_query(
        "q-b", "b", _result(0, b_to_d, 15)))
    queries.save_query(_query(
        "q-absent",
        "root",
        CandidateQueryResultRecord(
            position=0,
            candidate_event_id="snapshot-absent",
            topology_status="no_direct_link",
            travel_status="unknown",
            physical_status="unknown",
            elapsed_seconds=40,
            minimum_travel_seconds=None,
            hybrid_snapshot=None,
        ),
    ))

    observations = SQLiteObservationRepository(":memory:")
    observations.save(_plate_observation("exact-a", "CAM_A", 0))
    observations.save(_plate_observation("exact-b", "CAM_B", 10))
    graph = _graph()
    exact = TrajectoryReconstructor(observations, graph)
    api = Phase3ReadApi(
        evidence_repository=evidence,
        query_repository=queries,
    )
    server = create_server(
        observations,
        port=0,
        trajectory_reconstructor=exact,
        city_graph=graph,
        phase3_api=api,
    )
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with httpx.Client(base_url=f"http://{host}:{port}") as client:
        yield client, evidence, queries
    server.shutdown()
    server.server_close()
    observations.close()
    api.close()


def _client_for_api(api):
    observations = SQLiteObservationRepository(":memory:")
    server = create_server(observations, port=0, phase3_api=api)
    host, port = server.server_address
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = httpx.Client(base_url=f"http://{host}:{port}")
    return client, server, observations


def test_phase3_disabled_does_not_break_existing_server():
    observations = SQLiteObservationRepository(":memory:")
    server = create_server(observations, port=0)
    host, port = server.server_address
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with httpx.Client(base_url=f"http://{host}:{port}") as client:
            assert client.get("/health").json() == {"status": "ok"}
            response = client.get("/v1/phase3/evidence/root")
            assert response.status_code == 503
            assert response.json() == {
                "error": "Phase 3 API is not configured"}
    finally:
        server.shutdown()
        server.server_close()
        observations.close()


def test_evidence_endpoint_exposes_metadata_without_raw_vector(phase3_client):
    client, _, _ = phase3_client
    response = client.get("/v1/phase3/evidence/root")

    assert response.status_code == 200
    body = response.json()
    assert body["event_id"] == "root"
    assert body["camera_id"] == "CAM_A"
    assert body["fingerprint"]["normalized_plate"] == PLATE
    assert body["appearance"] == {
        "present": True,
        "dimension": 3,
        "model_id": "test-reid",
        "model_version": "1",
        "weights_sha256": "a" * 64,
        "preprocessing_sha256": "b" * 64,
    }
    assert "vector" not in json.dumps(body)


def test_missing_appearance_is_explicit(phase3_client):
    client, _, _ = phase3_client
    appearance = client.get(
        "/v1/phase3/evidence/no-appearance").json()["appearance"]

    assert appearance["present"] is False
    assert all(value is None for key, value in appearance.items()
               if key != "present")


def test_unknown_evidence_event_is_404(phase3_client):
    client, _, _ = phase3_client
    response = client.get("/v1/phase3/evidence/unknown")

    assert response.status_code == 404
    assert response.json()["error"] == "Phase 3 evidence event not found"


def test_query_header_exposes_policy_and_diagnostics(phase3_client):
    client, _, _ = phase3_client
    body = client.get("/v1/phase3/queries/q-root").json()

    assert body["query_id"] == "q-root"
    assert body["source_event_id"] == "root"
    assert body["retrieval_policy"]["max_results"] == 10
    assert body["diagnostics"]["returned_count"] == 3
    assert body["position_semantics"] == (
        "retrieval_position_not_relevance_rank")
    assert "results" not in body


def test_query_results_preserve_exact_stored_retrieval_order(phase3_client):
    client, _, _ = phase3_client
    body = client.get("/v1/phase3/queries/q-root/results").json()

    assert [item["candidate_event_id"] for item in body["results"]] == [
        "c", "b", "blocked"]
    assert [item["retrieval_position"] for item in body["results"]] == [
        0, 1, 2]
    assert body["appearance_similarity_semantics"] == "evidence_only"


def test_query_result_serializes_hybrid_evidence_components(phase3_client):
    client, _, _ = phase3_client
    results = client.get("/v1/phase3/queries/q-root/results").json()[
        "results"]

    available = results[1]["hybrid_snapshot"]
    hard_gate = results[2]["hybrid_snapshot"]
    assert available["appearance_comparison"]["status"] == "available"
    assert available["appearance_comparison"]["cosine_similarity"] == 1.0
    assert available["physical_feasibility"]["topology_status"] == (
        "no_direct_link")
    assert available["physical_feasibility"]["gate_status"] == "unknown"
    assert hard_gate["status"] == "trusted_plate_contradiction"
    assert hard_gate["trusted_plate_evidence"]["relation"] == (
        "contradiction")
    assert hard_gate["hard_gate_reason"] is not None


def test_absent_hybrid_snapshot_serializes_as_null(phase3_client):
    client, _, _ = phase3_client
    result = client.get(
        "/v1/phase3/queries/q-absent/results").json()["results"][0]

    assert result["hybrid_snapshot"] is None


def test_unknown_query_is_404_for_header_and_results(phase3_client):
    client, _, _ = phase3_client
    assert client.get("/v1/phase3/queries/nope").status_code == 404
    assert client.get("/v1/phase3/queries/nope/results").status_code == 404


def test_hypothesis_endpoint_is_explicitly_inferred_and_bounded(phase3_client):
    client, _, _ = phase3_client
    response = client.get(
        "/v1/phase3/hypotheses/root",
        params=[
            ("query_id", "q-root"),
            ("query_id", "q-b"),
            ("max_hops", "4"),
            ("max_hypotheses", "5"),
            ("max_branching_per_event", "3"),
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["trajectory_kind"] == "inferred_hypothesis"
    assert body["inferred"] is True
    assert body["exact_trajectory"] is False
    assert body["result"]["policy"] == {
        "max_hops": 4,
        "max_hypotheses": 5,
        "max_branching_per_event": 3,
        "version": 1,
    }
    assert [item["event_ids"] for item in body["result"]["hypotheses"]] == [
        ["root", "b", "d"], ["root", "c"]]
    assert body["result"]["hard_gate_excluded"] == 1


def test_hypothesis_edge_keeps_query_references_and_separate_evidence(
        phase3_client):
    client, _, _ = phase3_client
    body = client.get(
        "/v1/phase3/hypotheses/root",
        params=[("query_id", "q-root"), ("query_id", "q-b")],
    ).json()
    first_edge = body["result"]["hypotheses"][0]["edges"][0]

    assert first_edge["history_references"][0]["query_id"] == "q-root"
    assert first_edge["trusted_plate_relation"] == "agreement"
    assert first_edge["attribute_outcome"] == "consistent"
    assert first_edge["appearance_status"] == "available"
    assert first_edge["physical_gate_status"] == "unknown"


@pytest.mark.parametrize("params", [
    [],
    [("query_id", "q-root"), ("query_id", "q-root")],
    [("query_id", "q-root"), ("max_hops", "0")],
    [("query_id", "q-root"), ("max_hypotheses", "101")],
    [("query_id", "q-root"), ("unknown", "1")],
])
def test_invalid_hypothesis_query_sets_and_policy_are_400(
        phase3_client, params):
    client, _, _ = phase3_client
    response = client.get(
        "/v1/phase3/hypotheses/root", params=params)
    assert response.status_code == 400
    assert set(response.json()) == {"error"}


def test_unknown_hypothesis_query_id_is_404(phase3_client):
    client, _, _ = phase3_client
    response = client.get(
        "/v1/phase3/hypotheses/root", params={"query_id": "unknown"})
    assert response.status_code == 404


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_phase3_endpoints_reject_mutation_methods(phase3_client, method):
    client, _, _ = phase3_client
    response = client.request(method, "/v1/phase3/evidence/root", json={})

    assert response.status_code == 405
    assert response.json() == {"error": "Phase 3 endpoints are read-only"}


@pytest.mark.parametrize("path", [
    "/v1/phase3/evidence/%2E%2E",
    "/v1/phase3/evidence/..%2F..%2Fsecret",
    "/v1/phase3/queries/%5Cwindows",
    "/v1/phase3/hypotheses/%25bad?query_id=q-root",
])
def test_malformed_identifiers_and_traversal_are_rejected(
        phase3_client, path):
    client, _, _ = phase3_client
    response = client.get(path)
    assert response.status_code in {400, 404}
    assert "traceback" not in response.text.lower()


def test_phase3_json_is_deterministic_and_finite(phase3_client):
    client, _, _ = phase3_client
    first = client.get("/v1/phase3/queries/q-root/results").content
    second = client.get("/v1/phase3/queries/q-root/results").content

    assert first == second
    assert b"NaN" not in first and b"Infinity" not in first


def test_repository_corruption_maps_to_controlled_error_without_path():
    class CorruptEvidence:
        def get(self, event_id):
            raise VehicleEvidenceCorruptionError(
                r"broken C:\private\phase3.db row")

    api = Phase3ReadApi(evidence_repository=CorruptEvidence())
    client, server, observations = _client_for_api(api)
    try:
        response = client.get("/v1/phase3/evidence/root")
        assert response.status_code == 500
        assert response.json() == {
            "error": "Phase 3 evidence data is unavailable"}
        assert "private" not in response.text.lower()
        assert "traceback" not in response.text.lower()
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        observations.close()


def test_query_corruption_maps_to_controlled_error():
    class CorruptQueries:
        def get_query(self, query_id):
            raise QueryHistoryCorruptionError("internal row detail")

    api = Phase3ReadApi(query_repository=CorruptQueries())
    client, server, observations = _client_for_api(api)
    try:
        response = client.get("/v1/phase3/queries/query")
        assert response.status_code == 500
        assert response.json() == {
            "error": "Phase 3 query-history data is unavailable"}
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        observations.close()


def test_phase3_gets_do_not_call_repository_write_methods(
        phase3_client, monkeypatch):
    client, evidence, queries = phase3_client

    def unexpected(*args, **kwargs):
        raise AssertionError("read API must never write")

    monkeypatch.setattr(evidence, "save", unexpected)
    monkeypatch.setattr(queries, "save_query", unexpected)

    assert client.get("/v1/phase3/evidence/root").status_code == 200
    assert client.get("/v1/phase3/queries/q-root").status_code == 200
    assert client.get(
        "/v1/phase3/hypotheses/root",
        params={"query_id": "q-root"},
    ).status_code == 200


def test_hypothesis_api_does_not_rerun_retrieval_matching_or_models(
        phase3_client, monkeypatch):
    client, _, _ = phase3_client

    def unexpected(*args, **kwargs):
        raise AssertionError("Step 42 must consume stored evidence only")

    import phase3_city.appearance_comparison as appearance_module
    import phase3_city.appearance_embedding as embedding_module
    import phase3_city.historical_candidate_retrieval as retrieval_module
    import phase3_city.hybrid_matching as hybrid_module

    monkeypatch.setattr(
        appearance_module, "compare_appearance_embeddings", unexpected)
    monkeypatch.setattr(
        embedding_module.LocalTorchScriptAppearanceEncoder,
        "encode",
        unexpected,
    )
    monkeypatch.setattr(
        retrieval_module.HistoricalCandidateRetriever,
        "retrieve",
        unexpected,
    )
    monkeypatch.setattr(
        hybrid_module.Phase3HybridCandidateMatcher,
        "compare",
        unexpected,
    )

    response = client.get(
        "/v1/phase3/hypotheses/root",
        params=[("query_id", "q-root"), ("query_id", "q-b")],
    )
    assert response.status_code == 200


def test_exact_trajectory_response_remains_separate_and_unchanged(
        phase3_client):
    client, _, _ = phase3_client
    exact = client.post("/v1/trajectories", json={"plate": PLATE})

    assert exact.status_code == 200
    body = exact.json()
    assert body["camera_sequence"] == ["CAM_A", "CAM_B"]
    assert "hypotheses" not in body
    assert "inferred" not in body
    assert "trajectory_kind" not in body


def test_dashboard_has_distinct_phase3_read_only_panel(phase3_client):
    client, _, _ = phase3_client
    body = client.get("/dashboard").text

    assert "Phase 3 — Read-only evidence inspection" in body
    assert "Exact trajectory" in body
    assert "Inferred trajectory hypothesis" in body
    assert "Retrieval position" in body or "retrieval position" in body
    assert "/v1/phase3/evidence/" in body
    assert "/v1/phase3/queries/" in body
    assert "/v1/phase3/hypotheses/" in body
    assert "Same vehicle probability" not in body
    assert "AI certainty" not in body
    assert "innerHTML" not in body


def test_build_phase3_api_disabled_or_absent_returns_none():
    assert build_phase3_read_api({}) is None
    assert build_phase3_read_api({"phase3_api": {"enabled": False}}) is None


def test_configured_databases_open_read_only_without_modification(tmp_path):
    evidence_path = tmp_path / "evidence.db"
    query_path = tmp_path / "queries.db"
    evidence_writer = SQLiteVehicleEvidenceRepository(evidence_path)
    evidence_writer.save(VehicleEvidenceRecord.from_observation(
        _fingerprint_observation("root", "CAM_A", 0)))
    evidence_writer.close()
    query_writer = SQLiteCandidateQueryHistoryRepository(query_path)
    query_writer.save_query(_query("empty-query", "root"))
    query_writer.close()
    before = (evidence_path.stat().st_mtime_ns, query_path.stat().st_mtime_ns)

    api = build_phase3_read_api({
        "phase3_api": {
            "enabled": True,
            "evidence_db_path": str(evidence_path),
            "query_history_db_path": str(query_path),
            "hypothesis_policy": {
                "max_hops": 4,
                "max_hypotheses": 6,
                "max_branching_per_event": 3,
            },
        },
    })
    try:
        assert api.evidence_repository.get("root") is not None
        assert api.query_repository.get_query("empty-query") is not None
        with pytest.raises(VehicleEvidencePersistenceError, match="read-only"):
            api.evidence_repository.save(
                VehicleEvidenceRecord.from_observation(
                    _fingerprint_observation("new", "CAM_B", 1)))
        assert api.default_policy.max_hops == 4
    finally:
        api.close()

    after = (evidence_path.stat().st_mtime_ns, query_path.stat().st_mtime_ns)
    assert after == before


def test_configured_missing_database_fails_without_creating_file(tmp_path):
    missing = tmp_path / "missing.db"

    with pytest.raises(ValueError, match="unavailable"):
        build_phase3_read_api({
            "phase3_api": {
                "enabled": True,
                "evidence_db_path": str(missing),
                "query_history_db_path": None,
            },
        })

    assert not missing.exists()


@pytest.mark.parametrize("settings", [
    {"enabled": "yes"},
    {"enabled": True, "evidence_db_path": None,
     "query_history_db_path": None},
    {"enabled": True, "evidence_db_path": "",
     "query_history_db_path": None},
])
def test_invalid_phase3_configuration_fails_clearly(settings):
    with pytest.raises(ValueError):
        build_phase3_read_api({"phase3_api": settings})
