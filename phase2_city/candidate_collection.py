"""Bounded replay candidate collection and deterministic ranking (Step 28).

This module is an opt-in, in-memory consumer of a completed ``ScenarioResult``.
It performs only structural forward-window filtering and delegates every
identity, topology, and travel decision to the Step 27 matcher.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from numbers import Real
from typing import Tuple

from phase1_anpr.observation.observation_builder import PlateObservation
from phase1_anpr.pipeline.anpr_pipeline import PipelineResult
from phase1_anpr.replay import SourceReplayResult
from phase2_city.cross_camera_matching import (
    CrossCameraCandidateMatcher,
    CrossCameraCandidateResult,
    CrossCameraMatchValidationError,
    FingerprintObservation,
)
from phase2_city.scenario import ScenarioResult


class CandidateCollectionError(ValueError):
    """Raised when a Step 28 request or replay result is malformed."""


@dataclass(frozen=True)
class CandidateCollectionPolicy:
    """Required structural search-window and evaluation bounds."""

    max_time_delta_seconds: float
    max_candidates: int

    def __post_init__(self):
        value = self.max_time_delta_seconds
        if isinstance(value, bool) or not isinstance(value, Real):
            raise CandidateCollectionError(
                "max_time_delta_seconds must be a finite number")
        try:
            seconds = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise CandidateCollectionError(
                "max_time_delta_seconds must be a finite number") from exc
        if not math.isfinite(seconds):
            raise CandidateCollectionError(
                "max_time_delta_seconds must be finite")
        if seconds <= 0.0:
            raise CandidateCollectionError(
                "max_time_delta_seconds must be > 0")

        limit = self.max_candidates
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise CandidateCollectionError(
                "max_candidates must be an integer >= 1")
        if limit < 1:
            raise CandidateCollectionError(
                "max_candidates must be >= 1")

        object.__setattr__(self, "max_time_delta_seconds", seconds)

    def to_dict(self) -> dict:
        return {
            "max_time_delta_seconds": self.max_time_delta_seconds,
            "max_candidates": self.max_candidates,
        }


@dataclass(frozen=True)
class CandidateCollectionResult:
    """Bounded Step 27 evaluations for one replay observation.

    ``total_observations_considered`` is the number of ``PipelineResult``
    observations across every replay source, including the selected source and
    observations later excluded by structural filtering.
    """

    scenario_id: str
    source: FingerprintObservation
    total_observations_considered: int
    structurally_eligible_count: int
    truncated: bool
    ranked_results: Tuple[CrossCameraCandidateResult, ...]
    candidates: Tuple[CrossCameraCandidateResult, ...]

    def to_dict(self) -> dict:
        """Return a stable shape while reusing Step 27 serializers."""
        return {
            "scenario_id": self.scenario_id,
            "source": self.source.to_dict(),
            "total_observations_considered":
                self.total_observations_considered,
            "structurally_eligible_count": self.structurally_eligible_count,
            "truncated": self.truncated,
            "ranked_results": [result.to_dict()
                               for result in self.ranked_results],
            "candidates": [result.to_dict() for result in self.candidates],
        }


_DECISION_PRIORITY = {
    "possible_strong": 0,
    "possible_weak": 1,
    "insufficient_evidence": 2,
    "identity_conflict": 3,
    "physically_impossible": 4,
    "ineligible": 5,
}
_CANDIDATE_DECISIONS = frozenset({"possible_strong", "possible_weak"})


def _structural_order(observation: FingerprintObservation) -> tuple:
    return (
        observation.timestamp,
        observation.camera_id,
        observation.event_id,
    )


def _descending_optional_number(value) -> tuple:
    return (value is None, 0.0 if value is None else -value)


def _ascending_optional_number(value) -> tuple:
    return (value is None, 0.0 if value is None else value)


def _ascending_optional_timestamp(value) -> tuple:
    return (value is None, "" if value is None else value)


def _ranking_key(result: CrossCameraCandidateResult) -> tuple:
    comparison = result.fingerprint_comparison
    return (
        _DECISION_PRIORITY[result.decision],
        *_descending_optional_number(comparison.overall_similarity),
        -comparison.evidence_coverage,
        *_ascending_optional_number(result.elapsed_seconds),
        *_ascending_optional_timestamp(result.candidate.timestamp),
        result.candidate.camera_id,
        result.candidate.event_id,
    )


class ReplayCandidateCollector:
    """Collect a bounded candidate pool and evaluate it through Step 27."""

    def __init__(
            self,
            matcher: CrossCameraCandidateMatcher,
            policy: CandidateCollectionPolicy,
    ):
        if not callable(getattr(matcher, "compare", None)):
            raise CandidateCollectionError(
                "matcher must provide a callable compare method")
        if not isinstance(policy, CandidateCollectionPolicy):
            raise CandidateCollectionError(
                "policy must be a CandidateCollectionPolicy")
        self.matcher = matcher
        self.policy = policy

    def collect(
            self,
            scenario_result: ScenarioResult,
            source_event_id: str,
    ) -> CandidateCollectionResult:
        """Return bounded, deterministically ranked candidate hypotheses."""
        if not isinstance(scenario_result, ScenarioResult):
            raise CandidateCollectionError(
                "scenario_result must be a ScenarioResult")
        if not isinstance(source_event_id, str) or not source_event_id.strip():
            raise CandidateCollectionError(
                "source_event_id must be a non-empty string")

        pipeline_results = self._flatten(scenario_result)
        adapted = tuple(self._adapt(result) for result in pipeline_results)
        matches = [observation for observation in adapted
                   if observation.event_id == source_event_id]
        if not matches:
            raise CandidateCollectionError(
                f"source event {source_event_id!r} was not found")
        if len(matches) != 1:  # defensive; _flatten rejects all duplicates
            raise CandidateCollectionError(
                f"source event {source_event_id!r} is not unique")

        source = matches[0]
        if source.timestamp is None:
            raise CandidateCollectionError(
                f"source event {source.event_id!r} has no usable timestamp")

        eligible = []
        for observation in adapted:
            if observation.event_id == source.event_id:
                continue
            if observation.timestamp is None:
                continue
            elapsed = (observation.timestamp
                       - source.timestamp).total_seconds()
            if elapsed <= 0.0:
                continue
            if elapsed > self.policy.max_time_delta_seconds:
                continue
            if observation.camera_id == source.camera_id:
                continue
            eligible.append(observation)

        structurally_eligible_count = len(eligible)
        retained = heapq.nsmallest(
            self.policy.max_candidates,
            eligible,
            key=_structural_order,
        )
        evaluated = tuple(
            self.matcher.compare(source, candidate)
            for candidate in retained
        )
        ranked = tuple(sorted(evaluated, key=_ranking_key))
        candidates = tuple(
            result for result in ranked
            if result.decision in _CANDIDATE_DECISIONS
        )

        return CandidateCollectionResult(
            scenario_id=scenario_result.scenario_id,
            source=source,
            total_observations_considered=len(adapted),
            structurally_eligible_count=structurally_eligible_count,
            truncated=(
                structurally_eligible_count > self.policy.max_candidates
            ),
            ranked_results=ranked,
            candidates=candidates,
        )

    @staticmethod
    def _flatten(scenario_result: ScenarioResult) -> list[PipelineResult]:
        pipeline_results = []
        seen_event_ids = set()
        for source_index, source_result in enumerate(
                scenario_result.source_results):
            if not isinstance(source_result, SourceReplayResult):
                raise CandidateCollectionError(
                    f"source_results[{source_index}] must be a "
                    "SourceReplayResult")
            for result_index, pipeline_result in enumerate(
                    source_result.results):
                if not isinstance(pipeline_result, PipelineResult):
                    raise CandidateCollectionError(
                        f"source_results[{source_index}].results"
                        f"[{result_index}] must be a PipelineResult")
                observation = pipeline_result.observation
                if not isinstance(observation, PlateObservation):
                    raise CandidateCollectionError(
                        f"source_results[{source_index}].results"
                        f"[{result_index}].observation must be a "
                        "PlateObservation")
                event_id = observation.event_id
                if not isinstance(event_id, str) or not event_id.strip():
                    raise CandidateCollectionError(
                        f"observation event_id at source_results"
                        f"[{source_index}].results[{result_index}] must be a "
                        "non-empty string")
                if event_id in seen_event_ids:
                    raise CandidateCollectionError(
                        f"duplicate observation event_id: {event_id!r}")
                seen_event_ids.add(event_id)
                pipeline_results.append(pipeline_result)
        return pipeline_results

    @staticmethod
    def _adapt(pipeline_result: PipelineResult) -> FingerprintObservation:
        observation = pipeline_result.observation
        try:
            return FingerprintObservation.from_plate_observation(
                observation,
                pipeline_result.vehicle_fingerprint,
            )
        except CrossCameraMatchValidationError as exc:
            raise CandidateCollectionError(
                f"observation {observation.event_id!r} is invalid: {exc}"
            ) from exc
