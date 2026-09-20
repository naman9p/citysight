"""Bounded persistence-backed historical candidate retrieval (Step 39).

This module performs chronological metadata retrieval and authoritative
physical filtering before any appearance comparison or hybrid matching.  It
does not rank by identity evidence, persist queries, infer trajectories, or
create global vehicle identities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from numbers import Real
from typing import Optional, Tuple

from phase2_city.cross_camera_matching import (
    CrossCameraCandidateMatcher,
    CrossCameraCandidateResult,
    CrossCameraMatchPolicy,
    CrossCameraMatchValidationError,
)
from phase2_city.graph import CityCameraGraph
from phase2_city.models import CameraLink
from phase3_city.evidence_persistence import (
    VehicleEvidenceQueryCursor,
    VehicleEvidenceRecord,
    VehicleEvidenceRepository,
    VehicleEvidenceSummary,
)


_SQL_SCAN_PAGE_SIZE = 100


class HistoricalCandidateRetrievalError(ValueError):
    """Raised when a Step 39 request or dependency is invalid."""


@dataclass(frozen=True)
class HistoricalCandidateRetrievalPolicy:
    """Mandatory forward time-window and final-result bounds."""

    max_time_delta_seconds: float
    max_results: int

    def __post_init__(self):
        value = self.max_time_delta_seconds
        if isinstance(value, bool) or not isinstance(value, Real):
            raise HistoricalCandidateRetrievalError(
                "max_time_delta_seconds must be a finite number")
        try:
            seconds = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise HistoricalCandidateRetrievalError(
                "max_time_delta_seconds must be a finite number") from exc
        if not math.isfinite(seconds):
            raise HistoricalCandidateRetrievalError(
                "max_time_delta_seconds must be finite")
        if seconds <= 0.0:
            raise HistoricalCandidateRetrievalError(
                "max_time_delta_seconds must be > 0")
        try:
            timedelta(seconds=seconds)
        except OverflowError as exc:
            raise HistoricalCandidateRetrievalError(
                "max_time_delta_seconds is outside datetime range") from exc

        limit = self.max_results
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise HistoricalCandidateRetrievalError(
                "max_results must be an integer >= 1")
        if limit < 1:
            raise HistoricalCandidateRetrievalError(
                "max_results must be >= 1")

        object.__setattr__(self, "max_time_delta_seconds", seconds)

    def to_dict(self) -> dict:
        return {
            "max_time_delta_seconds": self.max_time_delta_seconds,
            "max_results": self.max_results,
        }


@dataclass(frozen=True)
class HistoricalCandidate:
    """One retained event plus its structural/physical diagnostics."""

    evidence: VehicleEvidenceRecord
    topology_status: str
    link: Optional[CameraLink]
    elapsed_seconds: float
    minimum_travel_seconds: Optional[float]
    travel_status: str
    physical_status: str

    def to_dict(self) -> dict:
        link = None
        if self.link is not None:
            link = {
                "from_camera_id": self.link.from_camera_id,
                "to_camera_id": self.link.to_camera_id,
                "distance_m": self.link.distance_m,
                "road_name": self.link.road_name,
                "travel_direction": self.link.travel_direction,
            }
        return {
            "evidence": self.evidence.to_dict(),
            "topology_status": self.topology_status,
            "link": link,
            "elapsed_seconds": self.elapsed_seconds,
            "minimum_travel_seconds": self.minimum_travel_seconds,
            "travel_status": self.travel_status,
            "physical_status": self.physical_status,
        }


@dataclass(frozen=True)
class HistoricalCandidateRetrievalResult:
    """Immutable, chronologically ordered result for one source event."""

    source_event_id: str
    source: VehicleEvidenceRecord
    policy: HistoricalCandidateRetrievalPolicy
    rows_considered: int
    physically_impossible_excluded: int
    truncated: bool
    candidates: Tuple[HistoricalCandidate, ...]

    @property
    def count(self) -> int:
        return len(self.candidates)

    def to_dict(self) -> dict:
        return {
            "source_event_id": self.source_event_id,
            "source": self.source.to_dict(),
            "policy": self.policy.to_dict(),
            "rows_considered": self.rows_considered,
            "physically_impossible_excluded":
                self.physically_impossible_excluded,
            "truncated": self.truncated,
            "count": self.count,
            "candidates": [candidate.to_dict()
                           for candidate in self.candidates],
        }


class HistoricalCandidateRetriever:
    """Retrieve a structurally bounded cross-camera historical candidate set."""

    def __init__(
            self,
            repository: VehicleEvidenceRepository,
            city_graph: CityCameraGraph,
            physical_policy: CrossCameraMatchPolicy,
            retrieval_policy: HistoricalCandidateRetrievalPolicy,
    ):
        if not isinstance(repository, VehicleEvidenceRepository):
            raise HistoricalCandidateRetrievalError(
                "repository must be a VehicleEvidenceRepository")
        if not isinstance(city_graph, CityCameraGraph):
            raise HistoricalCandidateRetrievalError(
                "city_graph must be a CityCameraGraph")
        if not isinstance(physical_policy, CrossCameraMatchPolicy):
            raise HistoricalCandidateRetrievalError(
                "physical_policy must be a CrossCameraMatchPolicy")
        if not isinstance(
                retrieval_policy, HistoricalCandidateRetrievalPolicy):
            raise HistoricalCandidateRetrievalError(
                "retrieval_policy must be a "
                "HistoricalCandidateRetrievalPolicy")
        self.repository = repository
        self.city_graph = city_graph
        self.physical_policy = physical_policy
        self.retrieval_policy = retrieval_policy
        self._physical_matcher = CrossCameraCandidateMatcher(
            city_graph,
            physical_policy,
        )

    def retrieve(
            self,
            source_event_id: str,
    ) -> HistoricalCandidateRetrievalResult:
        """Return physically eligible evidence in chronological order."""
        if (not isinstance(source_event_id, str)
                or not source_event_id.strip()):
            raise HistoricalCandidateRetrievalError(
                "source_event_id must be a non-empty string")
        source = self.repository.get(source_event_id)
        if source is None:
            raise HistoricalCandidateRetrievalError(
                f"source event {source_event_id!r} was not found")
        if source.timestamp is None:
            raise HistoricalCandidateRetrievalError(
                f"source event {source_event_id!r} has no usable timestamp")

        try:
            end = source.timestamp + timedelta(
                seconds=self.retrieval_policy.max_time_delta_seconds)
        except OverflowError as exc:
            raise HistoricalCandidateRetrievalError(
                "retrieval window exceeds the supported datetime range"
            ) from exc
        source_observation = source.to_fingerprint_observation()
        retained = []
        rows_considered = 0
        physically_impossible_excluded = 0
        truncated = False
        cursor = None
        page_limit = min(
            _SQL_SCAN_PAGE_SIZE,
            self.retrieval_policy.max_results + 1,
        )

        while True:
            page = self.repository.fetch_forward_candidate_summaries(
                start_exclusive=source.timestamp,
                end_inclusive=end,
                exclude_camera_id=source.camera_id,
                limit=page_limit,
                after=cursor,
            )
            if not page:
                break
            for summary in page:
                rows_considered += 1
                physical = self._evaluate_physical(
                    source_observation,
                    summary,
                )
                if physical.decision in {
                        "ineligible", "physically_impossible"}:
                    physically_impossible_excluded += 1
                    continue
                if len(retained) >= self.retrieval_policy.max_results:
                    truncated = True
                    break
                retained.append((summary, physical))
            if truncated or len(page) < page_limit:
                break
            last = page[-1]
            cursor = VehicleEvidenceQueryCursor(
                timestamp=last.timestamp,
                event_id=last.event_id,
            )

        candidates = tuple(
            self._load_candidate(summary, physical)
            for summary, physical in retained
        )
        return HistoricalCandidateRetrievalResult(
            source_event_id=source.event_id,
            source=source,
            policy=self.retrieval_policy,
            rows_considered=rows_considered,
            physically_impossible_excluded=physically_impossible_excluded,
            truncated=truncated,
            candidates=candidates,
        )

    def _evaluate_physical(
            self,
            source_observation,
            summary: VehicleEvidenceSummary,
    ) -> CrossCameraCandidateResult:
        try:
            return self._physical_matcher.compare(
                source_observation,
                summary.to_fingerprint_observation(),
            )
        except CrossCameraMatchValidationError as exc:
            raise HistoricalCandidateRetrievalError(
                f"physical evaluation failed for event "
                f"{summary.event_id!r}: {exc}"
            ) from exc

    def _load_candidate(
            self,
            summary: VehicleEvidenceSummary,
            physical: CrossCameraCandidateResult,
    ) -> HistoricalCandidate:
        evidence = self.repository.get(summary.event_id)
        if evidence is None:  # pragma: no cover - no delete API/concurrency
            raise HistoricalCandidateRetrievalError(
                f"candidate event {summary.event_id!r} disappeared during "
                "retrieval")
        physical_status = (
            "feasible" if physical.travel_status == "feasible"
            else "unknown"
        )
        return HistoricalCandidate(
            evidence=evidence,
            topology_status=physical.topology_status,
            link=physical.link,
            elapsed_seconds=physical.elapsed_seconds,
            minimum_travel_seconds=physical.minimum_travel_seconds,
            travel_status=physical.travel_status,
            physical_status=physical_status,
        )


__all__ = [
    "HistoricalCandidate",
    "HistoricalCandidateRetrievalError",
    "HistoricalCandidateRetrievalPolicy",
    "HistoricalCandidateRetrievalResult",
    "HistoricalCandidateRetriever",
]
