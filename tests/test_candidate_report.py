"""Step 29: read-only candidate report formatting and replay CLI surface."""

from datetime import datetime, timedelta, timezone

import pytest

import phase2_city.replay as replay_module
from phase1_anpr.observation.observation_builder import PlateObservation
from phase1_anpr.pipeline.anpr_pipeline import PipelineResult
from phase1_anpr.replay import ReplaySource, SourceReplayResult
from phase2_city.candidate_collection import (
    CandidateCollectionPolicy,
    CandidateCollectionResult,
    ReplayCandidateCollector,
)
from phase2_city.candidate_report import format_candidate_report
from phase2_city.cross_camera_matching import (
    CrossCameraCandidateMatcher,
    CrossCameraCandidateResult,
    CrossCameraMatchPolicy,
    FingerprintObservation,
)
from phase2_city.fingerprint import FingerprintComparison, VehicleFingerprint
from phase2_city.graph import CityCameraGraph
from phase2_city.models import Camera, CameraLink
from phase2_city.scenario import ScenarioResult


T0 = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)


def _camera(camera_id):
    return Camera(
        camera_id=camera_id,
        name=f"Camera {camera_id}",
        latitude=28.6,
        longitude=77.2,
        road_name="Demo Road",
        heading_deg=90.0,
    )


def _link(target, distance_m=1000.0):
    return CameraLink(
        from_camera_id="CAM_A",
        to_camera_id=target,
        distance_m=distance_m,
        road_name="Demo Road",
        travel_direction="eastbound",
    )


def _graph():
    return CityCameraGraph(
        [_camera(name) for name in ("CAM_A", "CAM_B", "CAM_C", "CAM_D")],
        [_link("CAM_B"), _link("CAM_C"), _link("CAM_D")],
    )


def _observation(
        event_id,
        camera_id,
        seconds,
        *,
        plate="GJ01AB1234",
        status="accepted",
):
    abstained = status == "abstained"
    return PlateObservation(
        event_id=event_id,
        camera_id=camera_id,
        track_id=1,
        timestamp=(T0 + timedelta(seconds=seconds)).isoformat(),
        plate_raw=None if abstained else plate,
        plate_normalized=None if abstained else plate,
        confidence=0.9,
        status=status,
        detector_confidence=None if abstained else 0.9,
        ocr_confidence=None if abstained else 0.9,
        quality_score=None if abstained else 0.8,
        best_frame_number=None if abstained else 5,
        plate_image_path=None,
        model_version="phase1-anpr-0.1.0",
    )


def _pipeline(observation, fingerprint=None):
    return PipelineResult(
        observation=observation,
        inserted=True,
        alerts=[],
        vehicle_fingerprint=fingerprint,
    )


def _source_result(source_id, camera_id, results):
    return SourceReplayResult(
        source=ReplaySource(
            video_path=f"{source_id}.mp4",
            camera_id=camera_id,
            source_id=source_id,
        ),
        results=list(results),
    )


def _scenario(*candidates, source_fingerprint=None):
    source = _pipeline(
        _observation("source", "CAM_A", 0), source_fingerprint)
    return ScenarioResult(
        scenario_id="scenario-29",
        source_results=[
            _source_result("source-video", "CAM_A", [source]),
            _source_result("candidate-video", "CAM_B", candidates),
        ],
    )


def _collect(scenario, *, limit=10, window=300):
    matcher = CrossCameraCandidateMatcher(
        _graph(), CrossCameraMatchPolicy(maximum_speed_kph=36))
    return ReplayCandidateCollector(
        matcher, CandidateCollectionPolicy(window, limit)
    ).collect(scenario, "source")


def _patch_main_dependencies(monkeypatch, scenario_result):
    graph = _graph()
    replay_calls = []
    monkeypatch.setattr(replay_module, "load_config", lambda path: {})
    monkeypatch.setattr(
        replay_module,
        "load_city_config",
        lambda path: (graph.list_cameras(), graph.list_links()),
    )
    monkeypatch.setattr(
        replay_module, "load_scenario", lambda path, graph: object())
    monkeypatch.setattr(
        replay_module, "_build_stores", lambda config: (None, None, None))

    def fake_run_scenario(*args, **kwargs):
        replay_calls.append((args, kwargs))
        return scenario_result

    monkeypatch.setattr(replay_module, "run_scenario", fake_run_scenario)
    return replay_calls


def _candidate_args(source_event_id="source"):
    return [
        "--scenario", "scenario.yaml",
        "--candidate-source-event-id", source_event_id,
        "--candidate-window-seconds", "300",
        "--candidate-max-results", "10",
        "--candidate-maximum-speed-kph", "36",
    ]


def test_successful_report_shows_source_candidate_evidence_and_reasons():
    result = _collect(_scenario(_pipeline(
        _observation("candidate", "CAM_B", 100))))
    report = format_candidate_report(result)

    assert "Cross-Camera Candidate Report" in report
    assert "Scenario: scenario-29" in report
    assert "Event ID: source" in report
    assert "Camera: CAM_A" in report
    assert "Plate: GJ01AB1234" in report
    assert "Candidate hypotheses: 1" in report
    assert "Candidate hypothesis 1 (possible candidate match)" in report
    assert "Decision: possible_strong" in report
    assert "Event ID: candidate" in report
    assert "Camera: CAM_B" in report
    assert "Elapsed seconds: 100.000000" in report
    assert "Overall similarity: 0.800000" in report
    assert "Evidence coverage:" in report
    assert "Topology: direct_link" in report
    assert "Travel: feasible" in report
    assert "Reasons:\n- fingerprint:" in report
    assert "same vehicle confirmed" not in report.lower()


def test_report_preserves_step28_candidate_order_without_sorting():
    source_fingerprint = VehicleFingerprint(
        normalized_plate="GJ01AB1234",
        vehicle_colour="red",
        vehicle_class="car",
    )
    strong = _pipeline(
        _observation("strong", "CAM_B", 120), source_fingerprint)
    weak = _pipeline(
        _observation("weak", "CAM_C", 100),
        VehicleFingerprint(
            normalized_plate="GJ01AB1234",
            vehicle_colour="blue",
            vehicle_class="car",
        ),
    )
    result = _collect(_scenario(
        weak, strong, source_fingerprint=source_fingerprint))
    assert [item.candidate.event_id for item in result.candidates] == [
        "strong", "weak"
    ]

    report = format_candidate_report(result)
    assert report.index("Event ID: strong") < report.index("Event ID: weak")
    assert "Decision: possible_strong" in report
    assert "Decision: possible_weak" in report


def test_zero_structural_candidates_is_a_successful_clear_report():
    result = _collect(_scenario())
    report = format_candidate_report(result)
    assert "Structurally eligible: 0" in report
    assert "Evaluated: 0" in report
    assert "Candidate hypotheses: 0" in report
    assert "No structurally eligible later cross-camera observations." in report


def test_evaluated_but_rejected_candidates_are_clear_and_diagnostic():
    result = _collect(_scenario(_pipeline(_observation(
        "conflict", "CAM_B", 100, plate="MH12CD5678"))))
    normal = format_candidate_report(result)
    assert "Evaluated: 1" in normal
    assert "Candidate hypotheses: 0" in normal
    assert "No plausible strong/weak candidate hypotheses were found." in normal
    assert "identity_conflict" not in normal

    diagnostic = format_candidate_report(result, include_diagnostics=True)
    assert "Ranked evaluations (diagnostics)" in diagnostic
    assert "Decision: identity_conflict" in diagnostic
    assert "identity_conflict:" in diagnostic


def test_diagnostics_preserve_all_rejected_decisions_and_reasons():
    source = FingerprintObservation(
        event_id="source",
        camera_id="CAM_A",
        timestamp=T0,
        observation_status="accepted",
        fingerprint=VehicleFingerprint(),
    )
    evaluations = tuple(
        _evaluation(source, decision, index)
        for index, decision in enumerate((
            "insufficient_evidence",
            "identity_conflict",
            "physically_impossible",
            "ineligible",
        ), start=1)
    )
    result = CandidateCollectionResult(
        scenario_id="diagnostics",
        source=source,
        total_observations_considered=5,
        structurally_eligible_count=4,
        truncated=False,
        ranked_results=evaluations,
        candidates=(),
    )
    report = format_candidate_report(result, include_diagnostics=True)
    for decision in (
            "insufficient_evidence", "identity_conflict",
            "physically_impossible", "ineligible"):
        assert f"Decision: {decision}" in report
        assert f"reason: {decision}" in report


def _evaluation(source, decision, index):
    candidate = FingerprintObservation(
        event_id=f"event-{index}",
        camera_id="CAM_B",
        timestamp=T0 + timedelta(seconds=index),
        observation_status="accepted",
        fingerprint=VehicleFingerprint(),
    )
    comparison = FingerprintComparison(
        overall_similarity=None,
        evidence_coverage=0.0,
        matched_weight=0.0,
        contradicted_weight=0.0,
        net_evidence=0.0,
        outcome="insufficient_evidence",
        field_comparisons=(),
    )
    return CrossCameraCandidateResult(
        source=source,
        candidate=candidate,
        fingerprint_comparison=comparison,
        topology_status="no_direct_link",
        link=None,
        elapsed_seconds=float(index),
        minimum_travel_seconds=None,
        travel_status="unknown",
        decision=decision,
        reasons=(f"reason: {decision}",),
    )


def test_truncation_is_prominent_and_reports_eligible_vs_evaluated():
    result = _collect(_scenario(
        _pipeline(_observation("one", "CAM_B", 100)),
        _pipeline(_observation("two", "CAM_C", 110)),
        _pipeline(_observation("three", "CAM_D", 120)),
    ), limit=1)
    report = format_candidate_report(result)
    assert "Structurally eligible: 3" in report
    assert "Evaluated: 1" in report
    assert "Truncated: Yes" in report
    assert "Candidate evaluation was truncated by max_candidates" in report
    assert "3 observations were structurally eligible; 1 was evaluated" in report


def test_report_output_is_deterministic_for_equivalent_input_permutations():
    candidates = [
        _pipeline(_observation("late", "CAM_C", 120)),
        _pipeline(_observation("early", "CAM_B", 100)),
    ]
    first = format_candidate_report(_collect(_scenario(*candidates)))
    second = format_candidate_report(_collect(_scenario(*reversed(candidates))))
    assert first == second


def test_formatter_rejects_non_collection_result():
    with pytest.raises(TypeError, match="CandidateCollectionResult"):
        format_candidate_report(object())


def test_replay_cli_without_candidate_options_keeps_existing_output(
        monkeypatch, capsys):
    scenario = _scenario(_pipeline(_observation("candidate", "CAM_B", 100)))
    replay_calls = _patch_main_dependencies(monkeypatch, scenario)

    assert replay_module.main(["--scenario", "scenario.yaml"]) == 0
    assert len(replay_calls) == 1
    assert capsys.readouterr().out == (
        "scenario_id   : scenario-29\n"
        "[source-video] camera=CAM_A observations=1 accepted=1 review=0 "
        "abstained=0 alerts=0\n"
        "[candidate-video] camera=CAM_B observations=1 accepted=1 review=0 "
        "abstained=0 alerts=0\n"
        "sources       : 2\n"
        "observations  : 2\n"
        "  accepted    : 2\n"
        "  review      : 0\n"
        "  abstained   : 0\n"
        "alerts        : 0\n"
    )


def test_replay_cli_candidate_mode_runs_real_in_memory_integration(
        monkeypatch, capsys):
    scenario = _scenario(_pipeline(_observation("candidate", "CAM_B", 100)))
    replay_calls = _patch_main_dependencies(monkeypatch, scenario)

    assert replay_module.main(_candidate_args()) == 0
    assert len(replay_calls) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Cross-Camera Candidate Report" in captured.out
    assert "Scenario: scenario-29" in captured.out
    assert "Event ID: source" in captured.out
    assert "Decision: possible_strong" in captured.out
    assert "Event ID: candidate" in captured.out
    assert "fingerprint: outcome=consistent" in captured.out


def test_replay_cli_diagnostic_switch_shows_rejected_evaluation(
        monkeypatch, capsys):
    scenario = _scenario(_pipeline(_observation(
        "conflict", "CAM_B", 100, plate="MH12CD5678")))
    _patch_main_dependencies(monkeypatch, scenario)

    args = [*_candidate_args(), "--include-candidate-diagnostics"]
    assert replay_module.main(args) == 0
    output = capsys.readouterr().out
    assert "Ranked evaluations (diagnostics)" in output
    assert "Decision: identity_conflict" in output


def test_replay_cli_invalid_source_is_a_clear_failure(monkeypatch, capsys):
    scenario = _scenario(_pipeline(_observation("candidate", "CAM_B", 100)))
    _patch_main_dependencies(monkeypatch, scenario)

    assert replay_module.main(_candidate_args("missing")) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: source event 'missing' was not found" in captured.err


def test_replay_cli_empty_source_is_a_clear_failure(monkeypatch, capsys):
    scenario = _scenario(_pipeline(_observation("candidate", "CAM_B", 100)))
    _patch_main_dependencies(monkeypatch, scenario)

    assert replay_module.main(_candidate_args("")) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "source_event_id must be a non-empty string" in captured.err


def test_replay_cli_duplicate_replay_event_ids_are_a_clear_failure(
        monkeypatch, capsys):
    scenario = _scenario(_pipeline(_observation("source", "CAM_B", 100)))
    _patch_main_dependencies(monkeypatch, scenario)

    assert replay_module.main(_candidate_args()) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "duplicate observation event_id: 'source'" in captured.err


@pytest.mark.parametrize(("option", "value", "message"), [
    ("--candidate-window-seconds", "0", "max_time_delta_seconds"),
    ("--candidate-window-seconds", "-1", "max_time_delta_seconds"),
    ("--candidate-window-seconds", "nan", "max_time_delta_seconds"),
    ("--candidate-window-seconds", "inf", "max_time_delta_seconds"),
    ("--candidate-max-results", "0", "max_candidates"),
    ("--candidate-max-results", "-1", "max_candidates"),
    ("--candidate-maximum-speed-kph", "0", "maximum_speed_kph"),
    ("--candidate-maximum-speed-kph", "-1", "maximum_speed_kph"),
    ("--candidate-maximum-speed-kph", "nan", "maximum_speed_kph"),
    ("--candidate-maximum-speed-kph", "inf", "maximum_speed_kph"),
])
def test_replay_cli_invalid_policy_is_rejected_by_existing_policy(
        option, value, message, capsys):
    args = _candidate_args()
    args[args.index(option) + 1] = value
    with pytest.raises(SystemExit) as exc:
        replay_module.main(args)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(("option", "value", "message"), [
    ("--candidate-window-seconds", "not-a-number", "invalid float value"),
    ("--candidate-max-results", "1.5", "invalid int value"),
    ("--candidate-maximum-speed-kph", "not-a-number", "invalid float value"),
])
def test_replay_cli_nonnumeric_policy_is_rejected_by_argparse(
        option, value, message, capsys):
    args = _candidate_args()
    args[args.index(option) + 1] = value
    with pytest.raises(SystemExit) as exc:
        replay_module.main(args)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("args", [
    ["--candidate-source-event-id", "source"],
    ["--candidate-window-seconds", "300"],
    ["--candidate-max-results", "10"],
    ["--candidate-maximum-speed-kph", "36"],
    ["--include-candidate-diagnostics"],
])
def test_replay_cli_requires_complete_candidate_option_group(args, capsys):
    with pytest.raises(SystemExit) as exc:
        replay_module.main(["--scenario", "scenario.yaml", *args])
    assert exc.value.code == 2
    assert "candidate reporting requires" in capsys.readouterr().err


def test_report_composition_calls_step28_collector_once(monkeypatch):
    calls = []
    sentinel = object()

    class FakeCollector:
        def __init__(self, matcher, policy):
            calls.append(("init", matcher, policy))

        def collect(self, scenario_result, source_event_id):
            calls.append(("collect", scenario_result, source_event_id))
            return sentinel

    fake_matcher = object()
    monkeypatch.setattr(
        replay_module, "CrossCameraCandidateMatcher",
        lambda graph, policy: fake_matcher)
    monkeypatch.setattr(replay_module, "ReplayCandidateCollector", FakeCollector)
    scenario = object()
    collection_policy = CandidateCollectionPolicy(300, 10)
    match_policy = CrossCameraMatchPolicy(36)

    result = replay_module._collect_candidate_report(
        scenario, object(), "source", collection_policy, match_policy)
    assert result is sentinel
    assert calls == [
        ("init", fake_matcher, collection_policy),
        ("collect", scenario, "source"),
    ]
