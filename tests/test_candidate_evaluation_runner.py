"""Step 31: file-backed labels and reproducible candidate evaluation."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest
import yaml

from phase1_anpr.observation.observation_builder import PlateObservation
from phase1_anpr.pipeline.anpr_pipeline import PipelineResult
from phase1_anpr.replay import ReplaySource, SourceReplayResult
from phase2_city.candidate_collection import CandidateCollectionPolicy
from phase2_city.candidate_evaluation import (
    CandidateEvaluationCaseResult,
    CandidateEvaluationError,
    CandidateEvaluationPolicy,
    CandidateEvaluationSummary,
)
from phase2_city.candidate_evaluation_runner import (
    _build_policies,
    format_candidate_evaluation_summary,
    main,
)
from phase2_city.candidate_ground_truth import (
    CandidateGroundTruthDataset,
    CandidateGroundTruthLoadError,
    load_candidate_ground_truth,
)
from phase2_city.cross_camera_matching import CrossCameraMatchPolicy
from phase2_city.models import Camera, CameraLink
from phase2_city.scenario import Scenario, ScenarioResult


T0 = datetime(2026, 9, 9, 10, 0, tzinfo=timezone.utc)


def _write_labels(tmp_path, data, name="labels.yaml"):
    path = tmp_path / name
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _labels(*cases, scenario_id="evaluation-scenario"):
    return {
        "scenario_id": scenario_id,
        "cases": [
            {
                "source_event_id": source,
                "relevant_event_ids": list(relevant),
            }
            for source, relevant in cases
        ],
    }


def _case_result(
        source="source-a",
        relevant=("truth-a",),
        candidates=("truth-a", "other"),
        *,
        first_rank=1,
        truncated=True,
):
    return CandidateEvaluationCaseResult(
        source_event_id=source,
        relevant_event_ids=tuple(relevant),
        candidate_event_ids=tuple(candidates),
        first_relevant_rank=first_rank,
        truncated=truncated,
        structurally_eligible_count=4,
        evaluated_count=2,
        retrieved_relevant_at_k=((1, int(first_rank == 1)), (3, 1)),
    )


def _summary(*, zero=False):
    case = _case_result(
        candidates=() if zero else ("truth-a", "other"),
        first_rank=None if zero else 1,
        truncated=False if zero else True,
    )
    return CandidateEvaluationSummary(
        scenario_id="evaluation-scenario",
        policy=CandidateEvaluationPolicy((1, 3)),
        total_cases=1,
        total_relevant_events=1,
        recall_at_k=((1, 0.0 if zero else 1.0),
                     (3, 0.0 if zero else 1.0)),
        hit_rate_at_k=((1, 0.0 if zero else 1.0),
                       (3, 0.0 if zero else 1.0)),
        mean_reciprocal_rank=0.0 if zero else 1.0,
        truncated_case_count=0 if zero else 1,
        truncated_case_rate=0.0 if zero else 1.0,
        case_results=(case,),
    )


def _camera(camera_id, longitude):
    return Camera(
        camera_id=camera_id,
        name=camera_id,
        latitude=23.0,
        longitude=longitude,
        road_name="Test Road",
        heading_deg=0.0,
    )


def _observation(event_id, camera_id, seconds, plate):
    return PlateObservation(
        event_id=event_id,
        camera_id=camera_id,
        track_id=1,
        timestamp=(T0 + timedelta(seconds=seconds)).isoformat(),
        plate_raw=plate,
        plate_normalized=plate,
        confidence=0.9,
        status="accepted",
        detector_confidence=0.9,
        ocr_confidence=0.9,
        quality_score=0.8,
        best_frame_number=1,
        plate_image_path=None,
        model_version="phase1-anpr-0.1.0",
    )


def _scenario_result(*observations):
    source_results = []
    for index, observation in enumerate(observations):
        source_results.append(SourceReplayResult(
            source=ReplaySource(
                video_path=f"source-{index}.mp4",
                camera_id=observation.camera_id,
                source_id=f"source-{index}",
            ),
            results=[PipelineResult(
                observation=observation,
                inserted=True,
                alerts=[],
            )],
        ))
    return ScenarioResult(
        scenario_id="evaluation-scenario",
        source_results=source_results,
    )


def _argv(label_path, **overrides):
    values = {
        "scenario": "scenario.yaml",
        "ground_truth": str(label_path),
        "k": ["1", "3"],
        "candidate_window_seconds": "300",
        "candidate_max_results": "10",
        "candidate_maximum_speed_kph": "80",
    }
    values.update(overrides)
    args = []
    mapping = (
        ("scenario", "--scenario"),
        ("ground_truth", "--ground-truth"),
        ("k", "--k"),
        ("candidate_window_seconds", "--candidate-window-seconds"),
        ("candidate_max_results", "--candidate-max-results"),
        ("candidate_maximum_speed_kph", "--candidate-maximum-speed-kph"),
    )
    for key, option in mapping:
        value = values[key]
        if value is None:
            continue
        args.append(option)
        if isinstance(value, list):
            args.extend(value)
        else:
            args.append(value)
    return args


def _patch_runtime(monkeypatch, scenario_result, *, scenario_id=None):
    import phase2_city.candidate_evaluation_runner as runner

    cameras = [_camera("CAM_A", 72.0), _camera("CAM_B", 72.1),
               _camera("CAM_C", 72.2)]
    links = [
        CameraLink("CAM_A", "CAM_B", 1000.0, "Test Road"),
        CameraLink("CAM_A", "CAM_C", 500.0, "Test Road"),
    ]
    scenario = Scenario(
        scenario_id=scenario_id or scenario_result.scenario_id,
        sources=[],
    )
    calls = {"replay": 0}

    monkeypatch.setattr(runner, "load_config", lambda path: {})
    monkeypatch.setattr(
        runner, "load_city_config", lambda path: (cameras, links))
    monkeypatch.setattr(runner, "load_scenario", lambda path, graph: scenario)
    monkeypatch.setattr(
        runner, "_build_stores", lambda config: (object(), object(), object()))

    def fake_run_scenario(*args, **kwargs):
        calls["replay"] += 1
        return scenario_result

    monkeypatch.setattr(runner, "run_scenario", fake_run_scenario)
    return calls


def test_loader_accepts_one_case_and_constructs_step_30_model(tmp_path):
    path = _write_labels(tmp_path, _labels(("source-a", ("truth-a",))))

    dataset = load_candidate_ground_truth(path)

    assert dataset.scenario_id == "evaluation-scenario"
    assert dataset.cases[0].source_event_id == "source-a"
    assert dataset.cases[0].relevant_event_ids == ("truth-a",)


def test_loader_canonicalizes_multiple_case_and_relevant_id_order(tmp_path):
    path = _write_labels(tmp_path, _labels(
        ("source-z", ("truth-z", "truth-a")),
        ("source-a", ("truth-c", "truth-b")),
    ))

    dataset = load_candidate_ground_truth(path)

    assert tuple(case.source_event_id for case in dataset.cases) == (
        "source-a", "source-z")
    assert dataset.cases[0].relevant_event_ids == ("truth-b", "truth-c")


def test_dataset_is_frozen_and_serializes_deterministically(tmp_path):
    first = load_candidate_ground_truth(_write_labels(
        tmp_path,
        _labels(("source-z", ("truth-z",)),
                ("source-a", ("truth-a",))),
        "first.yaml",
    ))
    second = load_candidate_ground_truth(_write_labels(
        tmp_path,
        _labels(("source-a", ("truth-a",)),
                ("source-z", ("truth-z",))),
        "second.yaml",
    ))

    assert first.to_dict() == second.to_dict()
    with pytest.raises(FrozenInstanceError):
        first.scenario_id = "changed"


@pytest.mark.parametrize("content", [
    "cases: [\n",
    "- not-a-mapping\n",
])
def test_loader_rejects_malformed_yaml_or_wrong_root(tmp_path, content):
    path = _write_labels(tmp_path, content)
    with pytest.raises(CandidateGroundTruthLoadError):
        load_candidate_ground_truth(path)


@pytest.mark.parametrize("data, message", [
    ({"scenario_id": "x"}, "define cases"),
    ({"scenario_id": "x", "cases": {}}, "must be a list"),
    ({"scenario_id": "x", "cases": []}, "must not be empty"),
    ({"scenario_id": "x", "cases": ["bad"]}, "must be a mapping"),
    ({"scenario_id": "x", "cases": [{}]}, "source_event_id"),
    ({"scenario_id": "x", "cases": [{"source_event_id": "a"}]},
     "relevant_event_ids"),
    ({"scenario_id": "x", "cases": [{
        "source_event_id": "a", "relevant_event_ids": "b"}]},
     "must be a list"),
])
def test_loader_rejects_invalid_outer_or_case_schema(tmp_path, data, message):
    path = _write_labels(tmp_path, data)
    with pytest.raises(CandidateGroundTruthLoadError, match=message):
        load_candidate_ground_truth(path)


@pytest.mark.parametrize("data", [
    {"scenario_id": "x", "cases": [{
        "source_event_id": "a", "relevant_event_ids": ["b"]}],
     "predicted_match": "b"},
    {"scenario_id": "x", "cases": [{
        "source_event_id": "a", "relevant_event_ids": ["b"],
        "similarity": 1.0}]},
])
def test_loader_rejects_unknown_prediction_or_score_fields(tmp_path, data):
    path = _write_labels(tmp_path, data)
    with pytest.raises(CandidateGroundTruthLoadError, match="unknown"):
        load_candidate_ground_truth(path)


@pytest.mark.parametrize("source, relevant", [
    ("", ("truth",)),
    ("   ", ("truth",)),
    ("source", ("",)),
    ("source", ("truth", "truth")),
    ("source", ("source",)),
])
def test_loader_preserves_step_30_case_validation(tmp_path, source, relevant):
    path = _write_labels(tmp_path, _labels((source, relevant)))
    with pytest.raises(CandidateEvaluationError):
        load_candidate_ground_truth(path)


def test_loader_missing_file_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="ground-truth file not found"):
        load_candidate_ground_truth(tmp_path / "missing.yaml")


def test_policy_builder_uses_real_step_27_28_and_30_policies():
    args = SimpleNamespace(
        candidate_maximum_speed_kph=80.0,
        candidate_window_seconds=300.0,
        candidate_max_results=10,
        k_values=[5, 1, 3],
    )
    parser = pytest.importorskip("argparse").ArgumentParser()

    match, collection, evaluation = _build_policies(parser, args)

    assert match == CrossCameraMatchPolicy(maximum_speed_kph=80.0)
    assert collection == CandidateCollectionPolicy(
        max_time_delta_seconds=300.0, max_candidates=10)
    assert evaluation.k_values == (1, 3, 5)


@pytest.mark.parametrize("field, value", [
    ("candidate_window_seconds", 0.0),
    ("candidate_window_seconds", -1.0),
    ("candidate_window_seconds", float("nan")),
    ("candidate_window_seconds", float("inf")),
    ("candidate_window_seconds", True),
    ("candidate_max_results", 0),
    ("candidate_max_results", -1),
    ("candidate_max_results", True),
    ("candidate_maximum_speed_kph", 0.0),
    ("candidate_maximum_speed_kph", -1.0),
    ("candidate_maximum_speed_kph", float("nan")),
    ("candidate_maximum_speed_kph", float("inf")),
    ("candidate_maximum_speed_kph", True),
    ("k_values", [0]),
    ("k_values", [-1]),
    ("k_values", [True]),
])
def test_policy_builder_rejects_invalid_domain_values(field, value):
    values = dict(
        candidate_maximum_speed_kph=80.0,
        candidate_window_seconds=300.0,
        candidate_max_results=10,
        k_values=[1, 3],
    )
    values[field] = value
    parser = pytest.importorskip("argparse").ArgumentParser()
    with pytest.raises(SystemExit) as exc_info:
        _build_policies(parser, SimpleNamespace(**values))
    assert exc_info.value.code == 2


@pytest.mark.parametrize("missing", [
    "scenario",
    "ground_truth",
    "k",
    "candidate_window_seconds",
    "candidate_max_results",
    "candidate_maximum_speed_kph",
])
def test_cli_requires_every_evaluation_input(tmp_path, missing):
    path = _write_labels(tmp_path, _labels(("source", ("truth",))))
    with pytest.raises(SystemExit) as exc_info:
        main(_argv(path, **{missing: None}))
    assert exc_info.value.code == 2


@pytest.mark.parametrize("field, value", [
    ("k", ["not-an-int"]),
    ("candidate_window_seconds", "not-a-number"),
    ("candidate_max_results", "1.5"),
    ("candidate_maximum_speed_kph", "not-a-number"),
])
def test_cli_rejects_nonnumeric_policy_text(tmp_path, field, value):
    path = _write_labels(tmp_path, _labels(("source", ("truth",))))
    with pytest.raises(SystemExit) as exc_info:
        main(_argv(path, **{field: value}))
    assert exc_info.value.code == 2


def test_human_output_uses_step_30_metrics_and_surfaces_truncation():
    output = format_candidate_evaluation_summary(_summary())

    assert "Scenario: evaluation-scenario" in output
    assert "Cases: 1" in output
    assert "Relevant sightings: 1" in output
    assert "Candidate Recall@1: 1.000" in output
    assert "Candidate HitRate@3: 1.000" in output
    assert "Candidate MRR: 1.000" in output
    assert "Truncated cases: 1 / 1 (1.000)" in output
    assert "accuracy" not in output.lower()


def test_per_case_diagnostics_preserve_step_30_order_and_fields():
    output = format_candidate_evaluation_summary(
        _summary(), include_cases=True)

    assert "Source event: source-a" in output
    assert "Relevant events: truth-a" in output
    assert "Candidate events: truth-a, other" in output
    assert "First relevant rank: 1" in output
    assert "Structurally eligible: 4" in output
    assert "Evaluated: 2" in output


def test_default_human_output_stays_compact():
    output = format_candidate_evaluation_summary(_summary())
    assert "Per-case diagnostics" not in output
    assert "Source event:" not in output


def test_zero_hit_summary_is_successful_and_finite_in_human_output():
    output = format_candidate_evaluation_summary(_summary(zero=True))
    assert "Candidate Recall@1: 0.000" in output
    assert "Candidate HitRate@3: 0.000" in output
    assert "Candidate MRR: 0.000" in output
    assert "Truncated cases: 0 / 1 (0.000)" in output


def test_scenario_id_mismatch_fails_before_replay(tmp_path, monkeypatch, capsys):
    path = _write_labels(
        tmp_path, _labels(("source", ("truth",)), scenario_id="labels-id"))
    result = _scenario_result(
        _observation("source", "CAM_A", 0, "GJ01AB1234"),
        _observation("truth", "CAM_B", 100, "GJ01AB1234"),
    )
    calls = _patch_runtime(
        monkeypatch, result, scenario_id="different-scenario")

    assert main(_argv(path)) == 1
    assert calls["replay"] == 0
    assert "does not match replay scenario_id" in capsys.readouterr().err


def test_missing_ground_truth_file_returns_clear_failure(tmp_path, capsys):
    missing = tmp_path / "missing.yaml"
    assert main(_argv(missing)) == 1
    assert "ground-truth file not found" in capsys.readouterr().err


def test_runner_executes_replay_and_evaluator_once_for_all_cases(
        tmp_path, monkeypatch, capsys):
    import phase2_city.candidate_evaluation_runner as runner

    path = _write_labels(tmp_path, _labels(
        ("source-a", ("truth-a",)),
        ("source-b", ("truth-b",)),
    ))
    result = _scenario_result(
        _observation("source-a", "CAM_A", 0, "GJ01AA0001"),
        _observation("truth-a", "CAM_B", 100, "GJ01AA0001"),
        _observation("source-b", "CAM_A", 200, "GJ01BB0002"),
        _observation("truth-b", "CAM_B", 300, "GJ01BB0002"),
    )
    calls = _patch_runtime(monkeypatch, result)
    evaluator_calls = []

    class FakeEvaluator:
        def evaluate(self, scenario_result, cases):
            evaluator_calls.append((scenario_result, cases))
            return _summary()

    monkeypatch.setattr(
        runner, "_create_evaluator", lambda *args: FakeEvaluator())

    assert main(_argv(path)) == 0
    assert calls["replay"] == 1
    assert len(evaluator_calls) == 1
    assert evaluator_calls[0][0] is result
    assert tuple(case.source_event_id for case in evaluator_calls[0][1]) == (
        "source-a", "source-b")
    assert "Candidate Recall@1" in capsys.readouterr().out


def test_json_output_is_exact_step_30_serialization(
        tmp_path, monkeypatch, capsys):
    import phase2_city.candidate_evaluation_runner as runner

    path = _write_labels(tmp_path, _labels(("source", ("truth",))))
    result = _scenario_result(
        _observation("source", "CAM_A", 0, "GJ01AB1234"),
        _observation("truth", "CAM_B", 100, "GJ01AB1234"),
    )
    _patch_runtime(monkeypatch, result)
    summary = _summary()

    class FakeEvaluator:
        def evaluate(self, scenario_result, cases):
            return summary

    monkeypatch.setattr(
        runner, "_create_evaluator", lambda *args: FakeEvaluator())

    assert main(_argv(path) + ["--json"]) == 0
    assert json.loads(capsys.readouterr().out) == summary.to_dict()


def test_duplicate_source_cases_fail_through_step_30_not_as_misses(
        tmp_path, monkeypatch, capsys):
    path = _write_labels(tmp_path, _labels(
        ("source", ("truth",)),
        ("source", ("truth",)),
    ))
    result = _scenario_result(
        _observation("source", "CAM_A", 0, "GJ01AB1234"),
        _observation("truth", "CAM_B", 100, "GJ01AB1234"),
    )
    calls = _patch_runtime(monkeypatch, result)

    assert main(_argv(path)) == 1
    assert calls["replay"] == 1
    assert "duplicate ground-truth source" in capsys.readouterr().err


def test_missing_label_reference_fails_through_step_30_not_as_a_miss(
        tmp_path, monkeypatch, capsys):
    path = _write_labels(tmp_path, _labels(("source", ("missing",))))
    result = _scenario_result(
        _observation("source", "CAM_A", 0, "GJ01AB1234"))
    _patch_runtime(monkeypatch, result)

    assert main(_argv(path)) == 1
    assert "relevant event does not exist" in capsys.readouterr().err


def test_real_step_27_28_30_seam_preserves_explicit_labels_and_candidate_order(
        tmp_path, monkeypatch, capsys):
    path = _write_labels(tmp_path, _labels(("source", ("truth",))))
    # The earlier same-plate observation is deliberately NOT labeled relevant.
    # It remains a candidate, but Step 30 must not manufacture truth from plate.
    result = _scenario_result(
        _observation("source", "CAM_A", 0, "GJ01AB1234"),
        _observation("same-plate-unlabeled", "CAM_C", 50, "GJ01AB1234"),
        _observation("truth", "CAM_B", 100, "GJ01AB1234"),
    )
    calls = _patch_runtime(monkeypatch, result)

    assert main(_argv(path, k=["1", "2"]) + ["--include-cases"]) == 0
    output = capsys.readouterr().out
    assert calls["replay"] == 1
    assert "Candidate Recall@1: 0.000" in output
    assert "Candidate Recall@2: 1.000" in output
    assert "Candidate HitRate@1: 0.000" in output
    assert "Candidate HitRate@2: 1.000" in output
    assert "Candidate MRR: 0.500" in output
    assert "Candidate events: same-plate-unlabeled, truth" in output
    assert "Relevant events: truth" in output


def test_real_runner_zero_hit_is_success(tmp_path, monkeypatch, capsys):
    path = _write_labels(tmp_path, _labels(("source", ("truth",))))
    result = _scenario_result(
        _observation("source", "CAM_A", 0, "GJ01AB1234"),
        _observation("truth", "CAM_B", 100, "MH02ZZ9999"),
    )
    _patch_runtime(monkeypatch, result)

    assert main(_argv(path)) == 0
    output = capsys.readouterr().out
    assert "Candidate Recall@1: 0.000" in output
    assert "Candidate HitRate@3: 0.000" in output
    assert "Candidate MRR: 0.000" in output
