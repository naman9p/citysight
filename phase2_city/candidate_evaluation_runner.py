"""Run one labeled candidate evaluation over one completed replay scenario."""

from __future__ import annotations

import argparse
import json
import sys

from phase1_anpr.detection.detector import DetectorError
from phase1_anpr.utils.config import DEFAULT_CONFIG_PATH, load_config
from phase1_anpr.video.video_reader import VideoReaderError
from phase2_city.candidate_collection import (
    CandidateCollectionError,
    CandidateCollectionPolicy,
    ReplayCandidateCollector,
)
from phase2_city.candidate_evaluation import (
    CandidateEvaluationError,
    CandidateEvaluationPolicy,
    CandidateEvaluationSummary,
    ReplayCandidateEvaluator,
)
from phase2_city.candidate_ground_truth import (
    CandidateGroundTruthLoadError,
    load_candidate_ground_truth,
)
from phase2_city.config.loader import DEFAULT_CITY_CONFIG_PATH, load_city_config
from phase2_city.cross_camera_matching import (
    CrossCameraCandidateMatcher,
    CrossCameraMatchPolicy,
    CrossCameraMatchValidationError,
)
from phase2_city.graph import CityCameraGraph
from phase2_city.replay import _build_stores, run_scenario
from phase2_city.scenario import ScenarioError, load_scenario


def _build_policies(parser, args):
    """Construct one explicit Step 27, Step 28, and Step 30 policy set."""
    try:
        match_policy = CrossCameraMatchPolicy(
            maximum_speed_kph=args.candidate_maximum_speed_kph)
        collection_policy = CandidateCollectionPolicy(
            max_time_delta_seconds=args.candidate_window_seconds,
            max_candidates=args.candidate_max_results,
        )
        evaluation_policy = CandidateEvaluationPolicy(tuple(args.k_values))
    except (CrossCameraMatchValidationError,
            CandidateCollectionError,
            CandidateEvaluationError) as exc:
        parser.error(str(exc))
    return match_policy, collection_policy, evaluation_policy


def format_candidate_evaluation_summary(
        summary: CandidateEvaluationSummary,
        *,
        include_cases: bool = False,
) -> str:
    """Format Step 30 fields without recalculating any evaluation metric."""
    if not isinstance(summary, CandidateEvaluationSummary):
        raise TypeError("summary must be a CandidateEvaluationSummary")

    lines = [
        "Cross-Camera Candidate Evaluation",
        "=================================",
        f"Scenario: {summary.scenario_id}",
        f"Cases: {summary.total_cases}",
        f"Relevant sightings: {summary.total_relevant_events}",
        "",
    ]
    lines.extend(
        f"Candidate Recall@{k}: {value:.3f}"
        for k, value in summary.recall_at_k
    )
    lines.append("")
    lines.extend(
        f"Candidate HitRate@{k}: {value:.3f}"
        for k, value in summary.hit_rate_at_k
    )
    lines.extend([
        "",
        f"Candidate MRR: {summary.mean_reciprocal_rank:.3f}",
        "Truncated cases: "
        f"{summary.truncated_case_count} / {summary.total_cases} "
        f"({summary.truncated_case_rate:.3f})",
    ])

    if include_cases:
        lines.extend(["", "Per-case diagnostics", "--------------------"])
        for index, case in enumerate(summary.case_results, start=1):
            if index > 1:
                lines.append("")
            lines.extend([
                f"Case {index}",
                f"Source event: {case.source_event_id}",
                "Relevant events: " + ", ".join(case.relevant_event_ids),
                "Candidate events: " + (
                    ", ".join(case.candidate_event_ids)
                    if case.candidate_event_ids else "none"),
                "First relevant rank: " + (
                    str(case.first_relevant_rank)
                    if case.first_relevant_rank is not None else "not found"),
                f"Truncated: {'Yes' if case.truncated else 'No'}",
                "Structurally eligible: "
                f"{case.structurally_eligible_count}",
                f"Evaluated: {case.evaluated_count}",
            ])
    return "\n".join(lines)


def _create_evaluator(graph, match_policy, collection_policy,
                      evaluation_policy):
    matcher = CrossCameraCandidateMatcher(graph, match_policy)
    collector = ReplayCandidateCollector(matcher, collection_policy)
    return ReplayCandidateEvaluator(collector, evaluation_policy)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate CitySight cross-camera candidate retrieval")
    parser.add_argument("--scenario", required=True,
                        help="Path to the replay scenario YAML")
    parser.add_argument("--ground-truth", required=True,
                        help="Path to explicit candidate ground-truth YAML")
    parser.add_argument("--k", dest="k_values", required=True, type=int,
                        nargs="+", help="Candidate ranks to evaluate")
    parser.add_argument("--candidate-window-seconds", required=True,
                        type=float, help="Inclusive Step 28 search window")
    parser.add_argument("--candidate-max-results", required=True, type=int,
                        help="Maximum Step 27 evaluations per source case")
    parser.add_argument("--candidate-maximum-speed-kph", required=True,
                        type=float, help="Step 27 speed ceiling in km/h")
    parser.add_argument("--city-config", default=str(DEFAULT_CITY_CONFIG_PATH),
                        help="Step 17 city topology config")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="Phase 1 replay and persistence config")
    parser.add_argument("--include-cases", action="store_true",
                        help="Include per-case diagnostics in text output")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="Print the exact Step 30 summary as JSON")
    args = parser.parse_args(argv)
    match_policy, collection_policy, evaluation_policy = \
        _build_policies(parser, args)

    try:
        dataset = load_candidate_ground_truth(args.ground_truth)
        config = load_config(args.config)
        cameras, links = load_city_config(args.city_config)
        graph = CityCameraGraph(cameras, links)
        scenario = load_scenario(args.scenario, graph)
    except (CandidateGroundTruthLoadError,
            CandidateEvaluationError,
            FileNotFoundError,
            ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if dataset.scenario_id != scenario.scenario_id:
        print(
            "error: candidate ground-truth scenario_id "
            f"{dataset.scenario_id!r} does not match replay scenario_id "
            f"{scenario.scenario_id!r}",
            file=sys.stderr,
        )
        return 1

    observation_repo, watchlist_repo, evidence_store = _build_stores(config)
    try:
        scenario_result = run_scenario(
            scenario,
            config,
            observation_repo=observation_repo,
            evidence_store=evidence_store,
            watchlist_repo=watchlist_repo,
        )
    except DetectorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("Add trained license-plate YOLO weights and set "
              "detection.weights_path in config.yaml, then re-run.",
              file=sys.stderr)
        return 2
    except VideoReaderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        evaluator = _create_evaluator(
            graph,
            match_policy,
            collection_policy,
            evaluation_policy,
        )
        summary = evaluator.evaluate(scenario_result, dataset.cases)
    except (CandidateEvaluationError,
            CandidateCollectionError,
            CrossCameraMatchValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(summary.to_dict(), indent=2))
    else:
        print(format_candidate_evaluation_summary(
            summary, include_cases=args.include_cases))
    return 0


if __name__ == "__main__":
    sys.exit(main())
