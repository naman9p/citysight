"""Read-only stdlib HTTP integration service for Phase 3 (Step 42).

The service exposes existing Step 38/40 records and constructs Step 41
hypotheses only from explicitly selected query histories.  It has no mutation,
retrieval, matching, ranking, model, ground-truth, or accuracy behavior.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

from phase3_city.evidence_persistence import (
    SQLiteVehicleEvidenceRepository,
    VehicleEvidenceCorruptionError,
    VehicleEvidencePersistenceError,
    VehicleEvidenceValidationError,
)
from phase3_city.query_history import (
    QueryHistoryCorruptionError,
    QueryHistoryPersistenceError,
    QueryHistoryValidationError,
    SQLiteCandidateQueryHistoryRepository,
)
from phase3_city.trajectory_hypotheses import (
    TrajectoryHypothesisBuilder,
    TrajectoryHypothesisDataError,
    TrajectoryHypothesisError,
    TrajectoryHypothesisPolicy,
)


PHASE3_API_SCHEMA_VERSION = 1
PHASE3_API_IMPLEMENTATION_ID = "citysight-phase3-step42-v1"

DEFAULT_MAX_HOPS = 8
DEFAULT_MAX_HYPOTHESES = 25
DEFAULT_MAX_BRANCHING_PER_EVENT = 10
MAX_MAX_HOPS = 32
MAX_MAX_HYPOTHESES = 100
MAX_MAX_BRANCHING_PER_EVENT = 25
MAX_QUERY_IDS = 50

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}")
_PHASE3_PREFIX = "/v1/phase3/"
_HYPOTHESIS_PARAMETERS = frozenset({
    "query_id",
    "max_hops",
    "max_hypotheses",
    "max_branching_per_event",
})


class Phase3ApiError(Exception):
    """Controlled, client-safe error returned by the HTTP adapter."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _identifier(value: str, name: str) -> str:
    decoded = unquote(value)
    if _IDENTIFIER.fullmatch(decoded) is None:
        raise Phase3ApiError(
            400,
            f"{name} must be 1-128 safe identifier characters",
        )
    return decoded


def _require_no_parameters(qs: Mapping) -> None:
    if qs:
        raise Phase3ApiError(400, "this endpoint does not accept query parameters")


def _bounded_integer(qs, name, default, maximum):
    values = qs.get(name)
    if values is None:
        return default
    if len(values) != 1:
        raise Phase3ApiError(400, f"{name} must be supplied once")
    try:
        value = int(values[0])
    except (TypeError, ValueError) as exc:
        raise Phase3ApiError(400, f"{name} must be an integer") from exc
    if value < 1 or value > maximum:
        raise Phase3ApiError(
            400, f"{name} must be between 1 and {maximum}")
    return value


def _checked_payload(payload):
    """Reject non-finite or non-JSON Phase 3 data before the handler writes."""
    try:
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise Phase3ApiError(
            500, "Phase 3 data could not be serialized safely") from exc
    return payload


class Phase3ReadApi:
    """Narrow read service used by the existing stdlib HTTP router."""

    def __init__(
            self,
            *,
            evidence_repository=None,
            query_repository=None,
            default_policy: Optional[TrajectoryHypothesisPolicy] = None,
    ):
        self.evidence_repository = evidence_repository
        self.query_repository = query_repository
        self.default_policy = default_policy or TrajectoryHypothesisPolicy(
            max_hops=DEFAULT_MAX_HOPS,
            max_hypotheses=DEFAULT_MAX_HYPOTHESES,
            max_branching_per_event=DEFAULT_MAX_BRANCHING_PER_EVENT,
        )
        if not isinstance(self.default_policy, TrajectoryHypothesisPolicy):
            raise TypeError("default_policy must be TrajectoryHypothesisPolicy")
        if self.default_policy.max_hops > MAX_MAX_HOPS \
                or self.default_policy.max_hypotheses > MAX_MAX_HYPOTHESES \
                or self.default_policy.max_branching_per_event \
                > MAX_MAX_BRANCHING_PER_EVENT:
            raise ValueError("default hypothesis policy exceeds API bounds")

    def dispatch(self, path: str, qs: Mapping):
        evidence_match = re.fullmatch(
            rf"{_PHASE3_PREFIX}evidence/([^/]+)", path)
        if evidence_match:
            _require_no_parameters(qs)
            return _checked_payload(self._get_evidence(
                _identifier(evidence_match.group(1), "event_id")))

        results_match = re.fullmatch(
            rf"{_PHASE3_PREFIX}queries/([^/]+)/results", path)
        if results_match:
            _require_no_parameters(qs)
            return _checked_payload(self._get_query_results(
                _identifier(results_match.group(1), "query_id")))

        query_match = re.fullmatch(
            rf"{_PHASE3_PREFIX}queries/([^/]+)", path)
        if query_match:
            _require_no_parameters(qs)
            return _checked_payload(self._get_query_header(
                _identifier(query_match.group(1), "query_id")))

        hypothesis_match = re.fullmatch(
            rf"{_PHASE3_PREFIX}hypotheses/([^/]+)", path)
        if hypothesis_match:
            return _checked_payload(self._get_hypotheses(
                _identifier(hypothesis_match.group(1), "root_event_id"),
                qs,
            ))

        raise Phase3ApiError(404, "Phase 3 endpoint not found")

    def _require_evidence(self):
        if self.evidence_repository is None:
            raise Phase3ApiError(
                503, "Phase 3 evidence repository is not configured")
        return self.evidence_repository

    def _require_queries(self):
        if self.query_repository is None:
            raise Phase3ApiError(
                503, "Phase 3 query-history repository is not configured")
        return self.query_repository

    def _get_evidence(self, event_id):
        try:
            record = self._require_evidence().get(event_id)
        except VehicleEvidenceValidationError as exc:
            raise Phase3ApiError(400, "invalid evidence event ID") from exc
        except (VehicleEvidenceCorruptionError,
                VehicleEvidencePersistenceError) as exc:
            raise Phase3ApiError(
                500, "Phase 3 evidence data is unavailable") from exc
        if record is None:
            raise Phase3ApiError(404, "Phase 3 evidence event not found")
        embedding = record.appearance_embedding
        appearance = {
            "present": embedding is not None,
            "dimension": None if embedding is None else embedding.dimension,
            "model_id": None if embedding is None else embedding.model_id,
            "model_version": (
                None if embedding is None else embedding.model_version),
            "weights_sha256": (
                None if embedding is None else embedding.weights_sha256),
            "preprocessing_sha256": (
                None if embedding is None
                else embedding.preprocessing_sha256),
        }
        return {
            "resource_type": "phase3_vehicle_evidence",
            "event_id": record.event_id,
            "camera_id": record.camera_id,
            "timestamp": (
                None if record.timestamp is None
                else record.timestamp.isoformat()),
            "observation_status": record.observation_status,
            "fingerprint": record.fingerprint.to_dict(),
            "appearance": appearance,
            "schema_version": record.schema_version,
            "embedding_serialization_version": (
                record.embedding_serialization_version),
            "api_schema_version": PHASE3_API_SCHEMA_VERSION,
            "api_implementation_id": PHASE3_API_IMPLEMENTATION_ID,
        }

    def _load_query(self, query_id):
        try:
            record = self._require_queries().get_query(query_id)
        except QueryHistoryValidationError as exc:
            raise Phase3ApiError(400, "invalid query ID") from exc
        except (QueryHistoryCorruptionError,
                QueryHistoryPersistenceError) as exc:
            raise Phase3ApiError(
                500, "Phase 3 query-history data is unavailable") from exc
        if record is None:
            raise Phase3ApiError(404, "Phase 3 query history not found")
        return record

    def _get_query_header(self, query_id):
        record = self._load_query(query_id)
        return {
            "resource_type": "phase3_query_history",
            "query_id": record.query_id,
            "source_event_id": record.source_event_id,
            "executed_at": record.executed_at.isoformat(),
            "retrieval_policy": record.retrieval_policy.to_dict(),
            "physical_policy": record.physical_policy.to_dict(),
            "retrieval_order_kind": record.retrieval_order_kind,
            "position_semantics": "retrieval_position_not_relevance_rank",
            "diagnostics": {
                "rows_considered": record.rows_considered,
                "physically_impossible_excluded": (
                    record.physically_impossible_excluded),
                "returned_count": record.returned_count,
                "truncated": record.truncated,
            },
            "versions": {
                "query_schema": record.schema_version,
                "retrieval_policy": record.retrieval_policy_version,
                "physical_policy": record.physical_policy_version,
                "implementation_id": record.implementation_id,
                "api_schema": PHASE3_API_SCHEMA_VERSION,
                "api_implementation_id": PHASE3_API_IMPLEMENTATION_ID,
            },
        }

    def _get_query_results(self, query_id):
        record = self._load_query(query_id)
        results = []
        for item in record.results:
            results.append({
                "retrieval_position": item.position,
                "candidate_event_id": item.candidate_event_id,
                "topology_status": item.topology_status,
                "travel_status": item.travel_status,
                "physical_status": item.physical_status,
                "elapsed_seconds": item.elapsed_seconds,
                "minimum_travel_seconds": item.minimum_travel_seconds,
                "hybrid_snapshot": (
                    None if item.hybrid_snapshot is None
                    else item.hybrid_snapshot.to_dict()),
            })
        return {
            "resource_type": "phase3_query_results",
            "query_id": record.query_id,
            "source_event_id": record.source_event_id,
            "retrieval_order_kind": record.retrieval_order_kind,
            "position_semantics": "retrieval_position_not_relevance_rank",
            "appearance_similarity_semantics": "evidence_only",
            "count": len(results),
            "results": results,
            "api_schema_version": PHASE3_API_SCHEMA_VERSION,
            "api_implementation_id": PHASE3_API_IMPLEMENTATION_ID,
        }

    def _get_hypotheses(self, root_event_id, qs):
        unknown = set(qs) - _HYPOTHESIS_PARAMETERS
        if unknown:
            raise Phase3ApiError(400, "unsupported hypothesis query parameter")
        supplied_ids = qs.get("query_id", [])
        if not supplied_ids:
            raise Phase3ApiError(
                400, "at least one query_id is required")
        if len(supplied_ids) > MAX_QUERY_IDS:
            raise Phase3ApiError(
                400, f"at most {MAX_QUERY_IDS} query_id values are allowed")
        query_ids = tuple(_identifier(value, "query_id")
                          for value in supplied_ids)
        if len(set(query_ids)) != len(query_ids):
            raise Phase3ApiError(400, "query_id values must be unique")

        policy = TrajectoryHypothesisPolicy(
            max_hops=_bounded_integer(
                qs, "max_hops", self.default_policy.max_hops, MAX_MAX_HOPS),
            max_hypotheses=_bounded_integer(
                qs,
                "max_hypotheses",
                self.default_policy.max_hypotheses,
                MAX_MAX_HYPOTHESES,
            ),
            max_branching_per_event=_bounded_integer(
                qs,
                "max_branching_per_event",
                self.default_policy.max_branching_per_event,
                MAX_MAX_BRANCHING_PER_EVENT,
            ),
        )
        records = tuple(self._load_query(query_id) for query_id in query_ids)
        try:
            result = TrajectoryHypothesisBuilder(policy).build(
                root_event_id, records)
        except TrajectoryHypothesisDataError as exc:
            raise Phase3ApiError(
                400,
                "selected query histories are inconsistent for hypothesis "
                "construction",
            ) from exc
        except TrajectoryHypothesisError as exc:
            raise Phase3ApiError(400, "invalid hypothesis request") from exc
        return {
            "resource_type": "phase3_inferred_trajectory_hypotheses",
            "trajectory_kind": "inferred_hypothesis",
            "inferred": True,
            "exact_trajectory": False,
            "selected_query_ids": list(query_ids),
            "result": result.to_dict(),
            "api_schema_version": PHASE3_API_SCHEMA_VERSION,
            "api_implementation_id": PHASE3_API_IMPLEMENTATION_ID,
        }

    def close(self):
        """Close repositories owned by a configured server, once each."""
        closed = set()
        for repository in (self.evidence_repository, self.query_repository):
            if repository is None or id(repository) in closed:
                continue
            close = getattr(repository, "close", None)
            if close is not None:
                close()
            closed.add(id(repository))


def open_phase3_read_api(
        *,
        evidence_db_path=None,
        query_history_db_path=None,
        default_policy=None,
):
    """Open configured Phase 3 SQLite stores in enforced read-only mode."""
    evidence = None
    queries = None
    try:
        if evidence_db_path is not None:
            if not Path(evidence_db_path).is_file():
                raise ValueError("Phase 3 evidence database is unavailable")
            evidence = SQLiteVehicleEvidenceRepository(
                evidence_db_path, read_only=True)
        if query_history_db_path is not None:
            if not Path(query_history_db_path).is_file():
                raise ValueError("Phase 3 query-history database is unavailable")
            queries = SQLiteCandidateQueryHistoryRepository(
                query_history_db_path, read_only=True)
        return Phase3ReadApi(
            evidence_repository=evidence,
            query_repository=queries,
            default_policy=default_policy,
        )
    except Exception:
        for repository in (evidence, queries):
            if repository is not None:
                repository.close()
        raise


__all__ = [
    "DEFAULT_MAX_BRANCHING_PER_EVENT",
    "DEFAULT_MAX_HOPS",
    "DEFAULT_MAX_HYPOTHESES",
    "MAX_MAX_BRANCHING_PER_EVENT",
    "MAX_MAX_HOPS",
    "MAX_MAX_HYPOTHESES",
    "MAX_QUERY_IDS",
    "PHASE3_API_IMPLEMENTATION_ID",
    "PHASE3_API_SCHEMA_VERSION",
    "Phase3ApiError",
    "Phase3ReadApi",
    "open_phase3_read_api",
]
