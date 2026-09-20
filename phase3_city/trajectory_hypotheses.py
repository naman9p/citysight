"""Bounded, explainable inferred trajectory hypotheses (Step 41).

These in-memory hypotheses consume explicitly supplied Step 40 history.  They
are not Phase 2 exact plate trajectories, persisted identities, probabilities,
or an excuse to rerun retrieval, matching, appearance comparison, or models.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

from phase3_city.query_history import (
    CandidateQueryHistoryRecord,
    CandidateQueryResultRecord,
    HybridResultSnapshot,
)


TRAJECTORY_HYPOTHESIS_SCHEMA_VERSION = 1
TRAJECTORY_HYPOTHESIS_POLICY_VERSION = 1
TRAJECTORY_HYPOTHESIS_IMPLEMENTATION_ID = "citysight-phase3-step41-v1"

_ELIGIBLE_STATUS = "eligible_with_evidence"
_HARD_GATE_STATUSES = frozenset({
    "ineligible",
    "physically_impossible",
    "trusted_plate_contradiction",
})
_HARD_PHYSICAL_GATES = frozenset({"ineligible", "impossible"})


class TrajectoryHypothesisError(ValueError):
    """Base error for Step 41 hypothesis construction."""


class TrajectoryHypothesisDataError(TrajectoryHypothesisError):
    """Raised when selected query history is malformed or inconsistent."""


def _required_text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrajectoryHypothesisError(f"{name} must be a non-empty string")
    return value


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TrajectoryHypothesisError(f"{name} must be an integer >= 1")
    return value


def _utc_timestamp(value, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TrajectoryHypothesisDataError(
            f"{name} must be a timezone-aware datetime")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise TrajectoryHypothesisDataError(
            f"{name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _parse_timestamp(value, name: str) -> datetime:
    if not isinstance(value, str):
        raise TrajectoryHypothesisDataError(
            f"{name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise TrajectoryHypothesisDataError(
            f"{name} is not a valid ISO-8601 timestamp") from exc
    return _utc_timestamp(parsed, name)


@dataclass(frozen=True)
class TrajectoryHypothesisPolicy:
    """Mandatory hard bounds for deterministic path expansion."""

    max_hops: int
    max_hypotheses: int
    max_branching_per_event: int
    version: int = TRAJECTORY_HYPOTHESIS_POLICY_VERSION

    def __post_init__(self):
        _positive_integer(self.max_hops, "max_hops")
        _positive_integer(self.max_hypotheses, "max_hypotheses")
        _positive_integer(
            self.max_branching_per_event,
            "max_branching_per_event",
        )
        if (isinstance(self.version, bool)
                or self.version != TRAJECTORY_HYPOTHESIS_POLICY_VERSION):
            raise TrajectoryHypothesisError(
                "unsupported trajectory hypothesis policy version")

    def to_dict(self) -> dict:
        return {
            "max_hops": self.max_hops,
            "max_hypotheses": self.max_hypotheses,
            "max_branching_per_event": self.max_branching_per_event,
            "version": self.version,
        }


@dataclass(frozen=True)
class TrajectoryHypothesisEvent:
    """Observation metadata copied from a supplied Step 37 snapshot."""

    event_id: str
    camera_id: str
    timestamp: datetime
    observation_status: str

    def __post_init__(self):
        _required_text(self.event_id, "event_id")
        _required_text(self.camera_id, "camera_id")
        _required_text(self.observation_status, "observation_status")
        object.__setattr__(
            self,
            "timestamp",
            _utc_timestamp(self.timestamp, "event timestamp"),
        )

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp.isoformat(),
            "observation_status": self.observation_status,
        }


@dataclass(frozen=True)
class TrajectoryHistoryReference:
    """One immutable Step 40 row supporting or qualifying an edge."""

    query_id: str
    retrieval_position: int
    topology_status: str
    travel_status: str
    physical_status: str
    elapsed_seconds: float
    minimum_travel_seconds: Optional[float]
    hybrid_snapshot: Optional[HybridResultSnapshot]

    def __post_init__(self):
        _required_text(self.query_id, "query_id")
        if (isinstance(self.retrieval_position, bool)
                or not isinstance(self.retrieval_position, int)
                or self.retrieval_position < 0):
            raise TrajectoryHypothesisError(
                "retrieval_position must be an integer >= 0")
        _required_text(self.topology_status, "topology_status")
        _required_text(self.travel_status, "travel_status")
        _required_text(self.physical_status, "physical_status")
        if not isinstance(self.elapsed_seconds, (int, float)):
            raise TrajectoryHypothesisError(
                "elapsed_seconds must be numeric")
        if self.hybrid_snapshot is not None and not isinstance(
                self.hybrid_snapshot, HybridResultSnapshot):
            raise TrajectoryHypothesisError(
                "hybrid_snapshot must be HybridResultSnapshot or None")

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "retrieval_position": self.retrieval_position,
            "topology_status": self.topology_status,
            "travel_status": self.travel_status,
            "physical_status": self.physical_status,
            "elapsed_seconds": self.elapsed_seconds,
            "minimum_travel_seconds": self.minimum_travel_seconds,
            "hybrid_snapshot": (
                None if self.hybrid_snapshot is None
                else self.hybrid_snapshot.to_dict()
            ),
        }


@dataclass(frozen=True)
class TrajectoryHypothesisEdge:
    """One inferred, hard-gate-safe forward edge between observations."""

    source: TrajectoryHypothesisEvent
    candidate: TrajectoryHypothesisEvent
    primary_snapshot: HybridResultSnapshot
    history_references: Tuple[TrajectoryHistoryReference, ...]
    inferred: bool = True

    def __post_init__(self):
        if not isinstance(self.source, TrajectoryHypothesisEvent) \
                or not isinstance(self.candidate, TrajectoryHypothesisEvent):
            raise TrajectoryHypothesisError(
                "edge endpoints must be TrajectoryHypothesisEvent values")
        if self.source.event_id == self.candidate.event_id:
            raise TrajectoryHypothesisError(
                "an inferred edge cannot repeat an event")
        if self.candidate.timestamp <= self.source.timestamp:
            raise TrajectoryHypothesisError(
                "an inferred edge must move strictly forward in time")
        if not isinstance(self.primary_snapshot, HybridResultSnapshot):
            raise TrajectoryHypothesisError(
                "primary_snapshot must be a HybridResultSnapshot")
        references = tuple(self.history_references)
        if not references or any(
                not isinstance(item, TrajectoryHistoryReference)
                for item in references):
            raise TrajectoryHypothesisError(
                "history_references must contain at least one reference")
        if not isinstance(self.inferred, bool) or not self.inferred:
            raise TrajectoryHypothesisError(
                "trajectory hypothesis edges must be marked inferred")
        object.__setattr__(self, "history_references", references)

    @property
    def topology_status(self) -> str:
        return self.primary_snapshot.to_dict()[
            "physical_feasibility"]["topology_status"]

    @property
    def travel_status(self) -> str:
        return self.primary_snapshot.to_dict()[
            "physical_feasibility"]["travel_status"]

    @property
    def physical_gate_status(self) -> str:
        return self.primary_snapshot.physical_gate_status

    @property
    def trusted_plate_relation(self) -> str:
        return self.primary_snapshot.trusted_plate_relation

    @property
    def attribute_outcome(self) -> str:
        return self.primary_snapshot.attribute_outcome

    @property
    def appearance_status(self) -> str:
        return self.primary_snapshot.appearance_status

    @property
    def appearance_cosine_similarity(self) -> Optional[float]:
        return self.primary_snapshot.appearance_cosine_similarity

    def to_dict(self) -> dict:
        return {
            "inferred": self.inferred,
            "source": self.source.to_dict(),
            "candidate": self.candidate.to_dict(),
            "hybrid_status": self.primary_snapshot.status,
            "physical_gate_status": self.physical_gate_status,
            "topology_status": self.topology_status,
            "travel_status": self.travel_status,
            "trusted_plate_relation": self.trusted_plate_relation,
            "attribute_outcome": self.attribute_outcome,
            "appearance_status": self.appearance_status,
            "appearance_cosine_similarity":
                self.appearance_cosine_similarity,
            "history_references": [item.to_dict()
                                   for item in self.history_references],
        }


@dataclass(frozen=True)
class TrajectoryHypothesis:
    """One inferred event path; never an exact Phase 2 trajectory."""

    hypothesis_id: str
    root_event_id: str
    events: Tuple[TrajectoryHypothesisEvent, ...]
    edges: Tuple[TrajectoryHypothesisEdge, ...]
    inferred: bool = True
    schema_version: int = TRAJECTORY_HYPOTHESIS_SCHEMA_VERSION
    implementation_id: str = TRAJECTORY_HYPOTHESIS_IMPLEMENTATION_ID

    def __post_init__(self):
        _required_text(self.hypothesis_id, "hypothesis_id")
        _required_text(self.root_event_id, "root_event_id")
        events = tuple(self.events)
        edges = tuple(self.edges)
        if len(events) < 2 or len(edges) != len(events) - 1:
            raise TrajectoryHypothesisError(
                "a hypothesis requires linked events and at least one edge")
        if events[0].event_id != self.root_event_id:
            raise TrajectoryHypothesisError(
                "the root event must be the first hypothesis event")
        event_ids = tuple(item.event_id for item in events)
        if len(set(event_ids)) != len(event_ids):
            raise TrajectoryHypothesisError(
                "a hypothesis cannot repeat an event")
        for index, edge in enumerate(edges):
            if edge.source != events[index] \
                    or edge.candidate != events[index + 1]:
                raise TrajectoryHypothesisError(
                    "hypothesis edges must connect consecutive events")
        if not isinstance(self.inferred, bool) or not self.inferred:
            raise TrajectoryHypothesisError(
                "trajectory hypotheses must be marked inferred")
        if (isinstance(self.schema_version, bool)
                or self.schema_version != TRAJECTORY_HYPOTHESIS_SCHEMA_VERSION):
            raise TrajectoryHypothesisError(
                "unsupported trajectory hypothesis schema version")
        if self.implementation_id != TRAJECTORY_HYPOTHESIS_IMPLEMENTATION_ID:
            raise TrajectoryHypothesisError(
                "unsupported trajectory hypothesis implementation")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "edges", edges)

    @property
    def event_ids(self) -> tuple:
        return tuple(item.event_id for item in self.events)

    @property
    def camera_ids(self) -> tuple:
        return tuple(item.camera_id for item in self.events)

    @property
    def timestamps(self) -> tuple:
        return tuple(item.timestamp for item in self.events)

    @property
    def query_ids(self) -> tuple:
        return tuple(dict.fromkeys(
            reference.query_id
            for edge in self.edges
            for reference in edge.history_references
        ))

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "inferred": self.inferred,
            "root_event_id": self.root_event_id,
            "event_ids": list(self.event_ids),
            "camera_ids": list(self.camera_ids),
            "timestamps": [item.isoformat() for item in self.timestamps],
            "query_ids": list(self.query_ids),
            "schema_version": self.schema_version,
            "implementation_id": self.implementation_id,
            "events": [item.to_dict() for item in self.events],
            "edges": [item.to_dict() for item in self.edges],
        }


@dataclass(frozen=True)
class TrajectoryHypothesisResult:
    """Bounded deterministic hypotheses and construction diagnostics."""

    root_event_id: str
    policy: TrajectoryHypothesisPolicy
    selected_query_ids: Tuple[str, ...]
    histories_considered: int
    candidate_records_considered: int
    eligible_edge_count: int
    hard_gate_excluded: int
    insufficient_evidence_excluded: int
    temporal_excluded: int
    repeated_event_excluded: int
    duplicate_edge_records_merged: int
    branching_edges_omitted: int
    hypotheses_truncated: bool
    hypotheses: Tuple[TrajectoryHypothesis, ...]

    @property
    def count(self) -> int:
        return len(self.hypotheses)

    def to_dict(self) -> dict:
        return {
            "root_event_id": self.root_event_id,
            "policy": self.policy.to_dict(),
            "selected_query_ids": list(self.selected_query_ids),
            "histories_considered": self.histories_considered,
            "candidate_records_considered":
                self.candidate_records_considered,
            "eligible_edge_count": self.eligible_edge_count,
            "hard_gate_excluded": self.hard_gate_excluded,
            "insufficient_evidence_excluded":
                self.insufficient_evidence_excluded,
            "temporal_excluded": self.temporal_excluded,
            "repeated_event_excluded": self.repeated_event_excluded,
            "duplicate_edge_records_merged":
                self.duplicate_edge_records_merged,
            "branching_edges_omitted": self.branching_edges_omitted,
            "hypotheses_truncated": self.hypotheses_truncated,
            "count": self.count,
            "hypotheses": [item.to_dict() for item in self.hypotheses],
        }


class TrajectoryHypothesisBuilder:
    """Build bounded paths from explicit immutable Step 40 snapshots."""

    def __init__(self, policy: TrajectoryHypothesisPolicy):
        if not isinstance(policy, TrajectoryHypothesisPolicy):
            raise TrajectoryHypothesisError(
                "policy must be a TrajectoryHypothesisPolicy")
        self.policy = policy

    def build(
            self,
            root_event_id: str,
            query_records,
    ) -> TrajectoryHypothesisResult:
        _required_text(root_event_id, "root_event_id")
        histories = self._canonical_histories(query_records)
        groups = {}
        candidate_records_considered = 0
        for history in histories:
            for result in history.results:
                candidate_records_considered += 1
                key = (history.source_event_id, result.candidate_event_id)
                groups.setdefault(key, []).append((history, result))

        event_metadata = {}
        edges = []
        hard_gate_excluded = 0
        insufficient_excluded = 0
        temporal_excluded = 0
        repeated_excluded = 0
        duplicate_merged = 0
        for key in sorted(groups):
            entries = sorted(
                groups[key],
                key=lambda item: (item[0].query_id, item[1].position),
            )
            duplicate_merged += max(0, len(entries) - 1)
            outcome = self._build_edge(
                key,
                entries,
                event_metadata,
            )
            if isinstance(outcome, TrajectoryHypothesisEdge):
                edges.append(outcome)
            elif outcome == "hard_gate":
                hard_gate_excluded += 1
            elif outcome == "temporal":
                temporal_excluded += 1
            elif outcome == "repeated":
                repeated_excluded += 1
            else:
                insufficient_excluded += 1

        adjacency = {}
        for edge in edges:
            adjacency.setdefault(edge.source.event_id, []).append(edge)
        branching_omitted = 0
        for source_event_id, choices in adjacency.items():
            ordered = sorted(choices, key=self._edge_order)
            branching_omitted += max(
                0, len(ordered) - self.policy.max_branching_per_event)
            adjacency[source_event_id] = ordered[
                :self.policy.max_branching_per_event]

        paths = []
        seen_paths = set()
        self._expand(
            root_event_id,
            adjacency,
            event_metadata,
            (),
            (),
            frozenset({root_event_id}),
            paths,
            seen_paths,
        )
        truncated = len(paths) > self.policy.max_hypotheses
        paths = paths[:self.policy.max_hypotheses]
        hypotheses = tuple(
            self._to_hypothesis(root_event_id, events, path_edges)
            for events, path_edges in paths
        )
        return TrajectoryHypothesisResult(
            root_event_id=root_event_id,
            policy=self.policy,
            selected_query_ids=tuple(item.query_id for item in histories),
            histories_considered=len(histories),
            candidate_records_considered=candidate_records_considered,
            eligible_edge_count=len(edges),
            hard_gate_excluded=hard_gate_excluded,
            insufficient_evidence_excluded=insufficient_excluded,
            temporal_excluded=temporal_excluded,
            repeated_event_excluded=repeated_excluded,
            duplicate_edge_records_merged=duplicate_merged,
            branching_edges_omitted=branching_omitted,
            hypotheses_truncated=truncated,
            hypotheses=hypotheses,
        )

    @staticmethod
    def _canonical_histories(query_records) -> tuple:
        if isinstance(query_records, (str, bytes)):
            raise TrajectoryHypothesisError(
                "query_records must be an iterable of history records")
        try:
            supplied = tuple(query_records)
        except TypeError as exc:
            raise TrajectoryHypothesisError(
                "query_records must be an iterable of history records"
            ) from exc
        by_id = {}
        for record in supplied:
            validated = TrajectoryHypothesisBuilder._validate_history(record)
            current = by_id.get(validated.query_id)
            if current is not None and current != validated:
                raise TrajectoryHypothesisDataError(
                    f"conflicting history for query_id {validated.query_id!r}")
            by_id[validated.query_id] = validated
        return tuple(by_id[key] for key in sorted(by_id))

    @staticmethod
    def _validate_history(record) -> CandidateQueryHistoryRecord:
        if not isinstance(record, CandidateQueryHistoryRecord):
            raise TrajectoryHypothesisDataError(
                "query history input must be CandidateQueryHistoryRecord")
        try:
            validated_results = tuple(
                CandidateQueryResultRecord(
                    position=result.position,
                    candidate_event_id=result.candidate_event_id,
                    topology_status=result.topology_status,
                    travel_status=result.travel_status,
                    physical_status=result.physical_status,
                    elapsed_seconds=result.elapsed_seconds,
                    minimum_travel_seconds=result.minimum_travel_seconds,
                    hybrid_snapshot=result.hybrid_snapshot,
                )
                for result in record.results
            )
            return CandidateQueryHistoryRecord(
                query_id=record.query_id,
                source_event_id=record.source_event_id,
                executed_at=record.executed_at,
                retrieval_policy=record.retrieval_policy,
                physical_policy=record.physical_policy,
                rows_considered=record.rows_considered,
                physically_impossible_excluded=(
                    record.physically_impossible_excluded),
                returned_count=record.returned_count,
                truncated=record.truncated,
                results=validated_results,
                retrieval_order_kind=record.retrieval_order_kind,
                schema_version=record.schema_version,
                retrieval_policy_version=record.retrieval_policy_version,
                physical_policy_version=record.physical_policy_version,
                implementation_id=record.implementation_id,
            )
        except ValueError as exc:
            raise TrajectoryHypothesisDataError(
                f"query history {record.query_id!r} is invalid: {exc}"
            ) from exc

    def _build_edge(self, key, entries, event_metadata):
        source_event_id, candidate_event_id = key
        references = tuple(
            self._reference(history.query_id, result)
            for history, result in entries
        )
        snapshots = tuple(
            reference.hybrid_snapshot
            for reference in references
            if reference.hybrid_snapshot is not None
        )
        if source_event_id == candidate_event_id:
            return "repeated"
        if not snapshots:
            return "insufficient"
        if any(self._is_hard_gate(snapshot) for snapshot in snapshots):
            return "hard_gate"

        events = tuple(
            self._snapshot_events(snapshot)
            for snapshot in snapshots
        )
        source = events[0][0]
        candidate = events[0][1]
        for other_source, other_candidate in events[1:]:
            if other_source != source or other_candidate != candidate:
                raise TrajectoryHypothesisDataError(
                    f"inconsistent metadata for edge "
                    f"{source_event_id!r}->{candidate_event_id!r}")
        self._merge_event(event_metadata, source)
        self._merge_event(event_metadata, candidate)
        if candidate.timestamp <= source.timestamp:
            return "temporal"
        for _, result in entries:
            elapsed = (candidate.timestamp - source.timestamp).total_seconds()
            if abs(elapsed - result.elapsed_seconds) > 1e-9:
                raise TrajectoryHypothesisDataError(
                    f"elapsed time conflicts for edge "
                    f"{source_event_id!r}->{candidate_event_id!r}")
        eligible = tuple(
            snapshot for snapshot in snapshots
            if snapshot.status == _ELIGIBLE_STATUS
        )
        if not eligible:
            return "insufficient"
        primary = min(
            eligible,
            key=lambda snapshot: (
                self._reference_key_for_snapshot(snapshot, references),
                snapshot.canonical_json,
            ),
        )
        return TrajectoryHypothesisEdge(
            source=source,
            candidate=candidate,
            primary_snapshot=primary,
            history_references=references,
        )

    @staticmethod
    def _reference(query_id, result: CandidateQueryResultRecord):
        return TrajectoryHistoryReference(
            query_id=query_id,
            retrieval_position=result.position,
            topology_status=result.topology_status,
            travel_status=result.travel_status,
            physical_status=result.physical_status,
            elapsed_seconds=result.elapsed_seconds,
            minimum_travel_seconds=result.minimum_travel_seconds,
            hybrid_snapshot=result.hybrid_snapshot,
        )

    @staticmethod
    def _reference_key_for_snapshot(snapshot, references):
        return min(
            (item.query_id, item.retrieval_position)
            for item in references
            if item.hybrid_snapshot == snapshot
        )

    @staticmethod
    def _is_hard_gate(snapshot: HybridResultSnapshot) -> bool:
        return (
            snapshot.status in _HARD_GATE_STATUSES
            or snapshot.physical_gate_status in _HARD_PHYSICAL_GATES
            or snapshot.trusted_plate_relation == "contradiction"
        )

    @staticmethod
    def _snapshot_events(snapshot: HybridResultSnapshot):
        payload = snapshot.to_dict()
        return (
            TrajectoryHypothesisBuilder._event_from_payload(
                payload["source"], "source"),
            TrajectoryHypothesisBuilder._event_from_payload(
                payload["candidate"], "candidate"),
        )

    @staticmethod
    def _event_from_payload(payload, role):
        if not isinstance(payload, dict):
            raise TrajectoryHypothesisDataError(
                f"hybrid {role} metadata must be an object")
        try:
            return TrajectoryHypothesisEvent(
                event_id=payload["event_id"],
                camera_id=payload["camera_id"],
                timestamp=_parse_timestamp(
                    payload["timestamp"],
                    f"hybrid {role} timestamp",
                ),
                observation_status=payload["observation_status"],
            )
        except KeyError as exc:
            raise TrajectoryHypothesisDataError(
                f"hybrid {role} metadata is missing {exc.args[0]}") from exc

    @staticmethod
    def _merge_event(event_metadata, event):
        current = event_metadata.get(event.event_id)
        if current is not None and current != event:
            raise TrajectoryHypothesisDataError(
                f"event {event.event_id!r} has inconsistent metadata")
        event_metadata[event.event_id] = event

    @staticmethod
    def _edge_order(edge):
        first_reference = edge.history_references[0]
        return (
            edge.candidate.timestamp,
            edge.candidate.event_id,
            first_reference.query_id,
            first_reference.retrieval_position,
        )

    def _expand(
            self,
            current_event_id,
            adjacency,
            event_metadata,
            path_events,
            path_edges,
            visited,
            paths,
            seen_paths,
    ):
        if len(paths) > self.policy.max_hypotheses:
            return
        if not path_events:
            root = event_metadata.get(current_event_id)
            if root is None:
                return
            path_events = (root,)
        if len(path_edges) >= self.policy.max_hops:
            self._append_path(path_events, path_edges, paths, seen_paths)
            return
        continuations = tuple(
            edge for edge in adjacency.get(current_event_id, ())
            if edge.candidate.event_id not in visited
        )
        if not continuations:
            if path_edges:
                self._append_path(path_events, path_edges, paths, seen_paths)
            return
        for edge in continuations:
            self._expand(
                edge.candidate.event_id,
                adjacency,
                event_metadata,
                path_events + (edge.candidate,),
                path_edges + (edge,),
                visited | {edge.candidate.event_id},
                paths,
                seen_paths,
            )
            if len(paths) > self.policy.max_hypotheses:
                return

    @staticmethod
    def _append_path(events, edges, paths, seen_paths):
        event_ids = tuple(item.event_id for item in events)
        if event_ids in seen_paths:
            return
        seen_paths.add(event_ids)
        paths.append((events, edges))

    @staticmethod
    def _to_hypothesis(root_event_id, events, edges):
        event_ids = tuple(item.event_id for item in events)
        digest = hashlib.sha256(
            "\0".join(event_ids).encode("utf-8")
        ).hexdigest()
        return TrajectoryHypothesis(
            hypothesis_id=f"trajectory-hypothesis-v1:{digest}",
            root_event_id=root_event_id,
            events=events,
            edges=edges,
        )


__all__ = [
    "TRAJECTORY_HYPOTHESIS_IMPLEMENTATION_ID",
    "TRAJECTORY_HYPOTHESIS_POLICY_VERSION",
    "TRAJECTORY_HYPOTHESIS_SCHEMA_VERSION",
    "TrajectoryHistoryReference",
    "TrajectoryHypothesis",
    "TrajectoryHypothesisBuilder",
    "TrajectoryHypothesisDataError",
    "TrajectoryHypothesisEdge",
    "TrajectoryHypothesisError",
    "TrajectoryHypothesisEvent",
    "TrajectoryHypothesisPolicy",
    "TrajectoryHypothesisResult",
]
