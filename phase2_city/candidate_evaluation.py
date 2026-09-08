"""Deterministic evaluation of Step 28 candidate retrieval (Step 30).

Ground-truth relationships are supplied explicitly by an evaluator.  This
module never derives labels from plates or fingerprints, changes candidate
ranking, tunes policies, or persists results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from phase1_anpr.observation.observation_builder import PlateObservation
from phase1_anpr.pipeline.anpr_pipeline import PipelineResult
from phase1_anpr.replay import SourceReplayResult
from phase2_city.candidate_collection import (
    CandidateCollectionResult,
    ReplayCandidateCollector,
)
from phase2_city.scenario import ScenarioResult


class CandidateEvaluationError(ValueError):
    """Raised when an evaluation policy, label, or scenario is invalid."""


def _validate_event_id(value, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CandidateEvaluationError(
            f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class CandidateGroundTruthCase:
    """Explicit relevant sightings for one labeled source observation."""

    source_event_id: str
    relevant_event_ids: Tuple[str, ...]

    def __post_init__(self):
        source_event_id = _validate_event_id(
            self.source_event_id, "source_event_id")
        relevant_event_ids = self.relevant_event_ids
        if not isinstance(relevant_event_ids, tuple):
            raise CandidateEvaluationError(
                "relevant_event_ids must be a tuple")
        if not relevant_event_ids:
            raise CandidateEvaluationError(
                "relevant_event_ids must contain at least one event ID")

        checked = []
        seen = set()
        for index, event_id in enumerate(relevant_event_ids):
            event_id = _validate_event_id(
                event_id, f"relevant_event_ids[{index}]")
            if event_id in seen:
                raise CandidateEvaluationError(
                    f"duplicate relevant event ID: {event_id!r}")
            seen.add(event_id)
            checked.append(event_id)
        if source_event_id in seen:
            raise CandidateEvaluationError(
                "source_event_id cannot appear in relevant_event_ids")

        object.__setattr__(self, "relevant_event_ids", tuple(sorted(checked)))

    def to_dict(self) -> dict:
        return {
            "source_event_id": self.source_event_id,
            "relevant_event_ids": list(self.relevant_event_ids),
        }


@dataclass(frozen=True)
class CandidateEvaluationPolicy:
    """The explicit, deterministic ranks at which retrieval is measured."""

    k_values: Tuple[int, ...]

    def __post_init__(self):
        values = self.k_values
        if not isinstance(values, tuple):
            raise CandidateEvaluationError("k_values must be a tuple")
        if not values:
            raise CandidateEvaluationError(
                "k_values must contain at least one K")

        checked = []
        seen = set()
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, int):
                raise CandidateEvaluationError(
                    f"k_values[{index}] must be an integer >= 1")
            if value < 1:
                raise CandidateEvaluationError(
                    f"k_values[{index}] must be >= 1")
            if value in seen:
                raise CandidateEvaluationError(
                    f"duplicate K value: {value}")
            seen.add(value)
            checked.append(value)

        object.__setattr__(self, "k_values", tuple(sorted(checked)))

    def to_dict(self) -> dict:
        return {"k_values": list(self.k_values)}


@dataclass(frozen=True)
class CandidateEvaluationCaseResult:
    """Compact retrieval diagnostics for one labeled source case."""

    source_event_id: str
    relevant_event_ids: Tuple[str, ...]
    candidate_event_ids: Tuple[str, ...]
    first_relevant_rank: Optional[int]
    truncated: bool
    structurally_eligible_count: int
    evaluated_count: int
    retrieved_relevant_at_k: Tuple[Tuple[int, int], ...]

    def to_dict(self) -> dict:
        return {
            "source_event_id": self.source_event_id,
            "relevant_event_ids": list(self.relevant_event_ids),
            "candidate_event_ids": list(self.candidate_event_ids),
            "first_relevant_rank": self.first_relevant_rank,
            "truncated": self.truncated,
            "structurally_eligible_count": self.structurally_eligible_count,
            "evaluated_count": self.evaluated_count,
            "retrieved_relevant_at_k": {
                str(k): count
                for k, count in self.retrieved_relevant_at_k
            },
        }


@dataclass(frozen=True)
class CandidateEvaluationSummary:
    """Aggregate candidate Recall@K, HitRate@K, MRR, and diagnostics."""

    scenario_id: str
    policy: CandidateEvaluationPolicy
    total_cases: int
    total_relevant_events: int
    recall_at_k: Tuple[Tuple[int, float], ...]
    hit_rate_at_k: Tuple[Tuple[int, float], ...]
    mean_reciprocal_rank: float
    truncated_case_count: int
    truncated_case_rate: float
    case_results: Tuple[CandidateEvaluationCaseResult, ...]

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "policy": self.policy.to_dict(),
            "total_cases": self.total_cases,
            "total_relevant_events": self.total_relevant_events,
            "recall_at_k": {
                str(k): value for k, value in self.recall_at_k
            },
            "hit_rate_at_k": {
                str(k): value for k, value in self.hit_rate_at_k
            },
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "truncated_case_count": self.truncated_case_count,
            "truncated_case_rate": self.truncated_case_rate,
            "case_results": [result.to_dict()
                             for result in self.case_results],
        }


def _scenario_event_ids(scenario_result: ScenarioResult) -> frozenset[str]:
    """Build a validation-only event index without collecting candidates."""
    if not isinstance(scenario_result, ScenarioResult):
        raise CandidateEvaluationError(
            "scenario_result must be a ScenarioResult")

    event_ids = set()
    for source_index, source_result in enumerate(
            scenario_result.source_results):
        if not isinstance(source_result, SourceReplayResult):
            raise CandidateEvaluationError(
                f"source_results[{source_index}] must be a SourceReplayResult")
        for result_index, pipeline_result in enumerate(source_result.results):
            if not isinstance(pipeline_result, PipelineResult):
                raise CandidateEvaluationError(
                    f"source_results[{source_index}].results[{result_index}] "
                    "must be a PipelineResult")
            observation = pipeline_result.observation
            if not isinstance(observation, PlateObservation):
                raise CandidateEvaluationError(
                    f"source_results[{source_index}].results[{result_index}]"
                    ".observation must be a PlateObservation")
            event_id = _validate_event_id(
                observation.event_id,
                f"source_results[{source_index}].results[{result_index}]"
                ".observation.event_id",
            )
            if event_id in event_ids:
                raise CandidateEvaluationError(
                    f"duplicate scenario event ID: {event_id!r}")
            event_ids.add(event_id)
    return frozenset(event_ids)


class ReplayCandidateEvaluator:
    """Measure explicitly labeled cases using an existing Step 28 collector."""

    def __init__(
            self,
            collector: ReplayCandidateCollector,
            policy: CandidateEvaluationPolicy,
    ):
        if not callable(getattr(collector, "collect", None)):
            raise CandidateEvaluationError(
                "collector must provide a callable collect method")
        if not isinstance(policy, CandidateEvaluationPolicy):
            raise CandidateEvaluationError(
                "policy must be a CandidateEvaluationPolicy")
        self.collector = collector
        self.policy = policy

    def evaluate(
            self,
            scenario_result: ScenarioResult,
            ground_truth_cases: Tuple[CandidateGroundTruthCase, ...],
    ) -> CandidateEvaluationSummary:
        """Evaluate explicit labels without changing Step 28 candidate order."""
        event_ids = _scenario_event_ids(scenario_result)
        if not isinstance(ground_truth_cases, tuple):
            raise CandidateEvaluationError(
                "ground_truth_cases must be a tuple")
        if not ground_truth_cases:
            raise CandidateEvaluationError(
                "ground_truth_cases must contain at least one case")

        case_by_source = {}
        for index, case in enumerate(ground_truth_cases):
            if not isinstance(case, CandidateGroundTruthCase):
                raise CandidateEvaluationError(
                    f"ground_truth_cases[{index}] must be a "
                    "CandidateGroundTruthCase")
            if case.source_event_id in case_by_source:
                raise CandidateEvaluationError(
                    "duplicate ground-truth source event ID: "
                    f"{case.source_event_id!r}")
            case_by_source[case.source_event_id] = case

        ordered_cases = tuple(
            case_by_source[source_event_id]
            for source_event_id in sorted(case_by_source)
        )
        for case in ordered_cases:
            if case.source_event_id not in event_ids:
                raise CandidateEvaluationError(
                    "ground-truth source event does not exist in scenario: "
                    f"{case.source_event_id!r}")
            for relevant_event_id in case.relevant_event_ids:
                if relevant_event_id not in event_ids:
                    raise CandidateEvaluationError(
                        "ground-truth relevant event does not exist in "
                        f"scenario: {relevant_event_id!r}")

        total_relevant_events = sum(
            len(case.relevant_event_ids) for case in ordered_cases)
        retrieved_totals = {k: 0 for k in self.policy.k_values}
        hit_totals = {k: 0 for k in self.policy.k_values}
        reciprocal_rank_total = 0.0
        truncated_case_count = 0
        case_results = []

        for case in ordered_cases:
            collection_result = self.collector.collect(
                scenario_result,
                source_event_id=case.source_event_id,
            )
            if not isinstance(collection_result, CandidateCollectionResult):
                raise CandidateEvaluationError(
                    "collector.collect must return a "
                    "CandidateCollectionResult")

            candidate_event_ids = tuple(
                result.candidate.event_id
                for result in collection_result.candidates
            )
            relevant = frozenset(case.relevant_event_ids)
            first_relevant_rank = next((
                rank for rank, event_id in enumerate(
                    candidate_event_ids, start=1)
                if event_id in relevant
            ), None)
            if first_relevant_rank is not None:
                reciprocal_rank_total += 1.0 / first_relevant_rank

            retrieved_at_k = []
            for k in self.policy.k_values:
                retrieved = len(relevant.intersection(candidate_event_ids[:k]))
                retrieved_totals[k] += retrieved
                if retrieved > 0:
                    hit_totals[k] += 1
                retrieved_at_k.append((k, retrieved))

            if collection_result.truncated:
                truncated_case_count += 1
            case_results.append(CandidateEvaluationCaseResult(
                source_event_id=case.source_event_id,
                relevant_event_ids=case.relevant_event_ids,
                candidate_event_ids=candidate_event_ids,
                first_relevant_rank=first_relevant_rank,
                truncated=collection_result.truncated,
                structurally_eligible_count=(
                    collection_result.structurally_eligible_count),
                evaluated_count=len(collection_result.ranked_results),
                retrieved_relevant_at_k=tuple(retrieved_at_k),
            ))

        total_cases = len(ordered_cases)
        recall_at_k = tuple(
            (k, retrieved_totals[k] / total_relevant_events)
            for k in self.policy.k_values
        )
        hit_rate_at_k = tuple(
            (k, hit_totals[k] / total_cases)
            for k in self.policy.k_values
        )
        return CandidateEvaluationSummary(
            scenario_id=scenario_result.scenario_id,
            policy=self.policy,
            total_cases=total_cases,
            total_relevant_events=total_relevant_events,
            recall_at_k=recall_at_k,
            hit_rate_at_k=hit_rate_at_k,
            mean_reciprocal_rank=reciprocal_rank_total / total_cases,
            truncated_case_count=truncated_case_count,
            truncated_case_rate=truncated_case_count / total_cases,
            case_results=tuple(case_results),
        )


__all__ = [
    "CandidateEvaluationError",
    "CandidateGroundTruthCase",
    "CandidateEvaluationPolicy",
    "CandidateEvaluationCaseResult",
    "CandidateEvaluationSummary",
    "ReplayCandidateEvaluator",
]
