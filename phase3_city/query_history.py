"""Immutable SQLite audit history for supplied Phase 3 query results (Step 40).

The persisted position is Step 39 chronological retrieval order, not a
relevance rank.  Optional Step 37 hybrid results must already have been
computed by the caller; this module never runs matching or comparison logic.
"""

from __future__ import annotations

import json
import math
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from phase2_city.cross_camera_matching import CrossCameraMatchPolicy
from phase3_city.appearance_comparison import (
    AppearanceComparisonStatus,
    AppearanceEmbeddingProvenance,
)
from phase3_city.historical_candidate_retrieval import (
    HistoricalCandidateRetrievalPolicy,
    HistoricalCandidateRetrievalResult,
)
from phase3_city.hybrid_matching import (
    HybridCandidateMatchResult,
    HybridCandidateStatus,
    PhysicalGateStatus,
    TrustedPlateRelation,
)


QUERY_HISTORY_SCHEMA_VERSION = 1
RETRIEVAL_POLICY_VERSION = 1
PHYSICAL_POLICY_VERSION = 1
HYBRID_SNAPSHOT_SCHEMA_VERSION = 1
QUERY_HISTORY_IMPLEMENTATION_ID = "citysight-phase3-step40-v1"
RETRIEVAL_ORDER_KIND = "chronological_timestamp_event_id"


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS phase3_candidate_queries (
    query_id                          TEXT PRIMARY KEY,
    source_event_id                   TEXT NOT NULL,
    executed_at                       TEXT NOT NULL,
    max_time_delta_seconds            REAL NOT NULL
        CHECK (max_time_delta_seconds > 0),
    max_results                       INTEGER NOT NULL
        CHECK (max_results >= 1),
    maximum_speed_kph                 REAL NOT NULL
        CHECK (maximum_speed_kph > 0),
    travel_time_tolerance_seconds     REAL NOT NULL
        CHECK (travel_time_tolerance_seconds >= 0),
    minimum_evidence_coverage         REAL NOT NULL
        CHECK (minimum_evidence_coverage >= 0
               AND minimum_evidence_coverage <= 1),
    strong_similarity_threshold       REAL NOT NULL
        CHECK (strong_similarity_threshold >= 0
               AND strong_similarity_threshold <= 1),
    rows_considered                   INTEGER NOT NULL
        CHECK (rows_considered >= 0),
    physically_impossible_excluded    INTEGER NOT NULL
        CHECK (physically_impossible_excluded >= 0),
    returned_count                    INTEGER NOT NULL
        CHECK (returned_count >= 0 AND returned_count <= max_results),
    truncated                         INTEGER NOT NULL
        CHECK (truncated IN (0, 1)),
    retrieval_order_kind              TEXT NOT NULL
        CHECK (retrieval_order_kind = '{RETRIEVAL_ORDER_KIND}'),
    query_schema_version              INTEGER NOT NULL
        CHECK (query_schema_version = {QUERY_HISTORY_SCHEMA_VERSION}),
    retrieval_policy_version          INTEGER NOT NULL
        CHECK (retrieval_policy_version = {RETRIEVAL_POLICY_VERSION}),
    physical_policy_version           INTEGER NOT NULL
        CHECK (physical_policy_version = {PHYSICAL_POLICY_VERSION}),
    implementation_id                 TEXT NOT NULL
        CHECK (implementation_id = '{QUERY_HISTORY_IMPLEMENTATION_ID}')
);

CREATE TABLE IF NOT EXISTS phase3_candidate_query_results (
    query_id                          TEXT NOT NULL,
    position                          INTEGER NOT NULL CHECK (position >= 0),
    candidate_event_id                TEXT NOT NULL,
    topology_status                   TEXT NOT NULL,
    travel_status                     TEXT NOT NULL,
    physical_status                   TEXT NOT NULL
        CHECK (physical_status IN ('feasible', 'unknown')),
    elapsed_seconds                   REAL NOT NULL CHECK (elapsed_seconds > 0),
    minimum_travel_seconds            REAL,
    hybrid_present                    INTEGER NOT NULL
        CHECK (hybrid_present IN (0, 1)),
    hybrid_snapshot_schema_version    INTEGER,
    hybrid_status                     TEXT,
    hard_gate_reason                  TEXT,
    physical_gate_status              TEXT,
    trusted_plate_relation            TEXT,
    attribute_outcome                 TEXT,
    attribute_evidence_coverage       REAL,
    attribute_net_evidence            REAL,
    appearance_status                 TEXT,
    appearance_cosine_similarity      REAL,
    appearance_provenance_compatible  INTEGER,
    hybrid_snapshot_json              TEXT,
    PRIMARY KEY (query_id, position),
    UNIQUE (query_id, candidate_event_id),
    FOREIGN KEY (query_id) REFERENCES phase3_candidate_queries(query_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    CHECK (minimum_travel_seconds IS NULL
           OR minimum_travel_seconds > 0),
    CHECK (appearance_cosine_similarity IS NULL
           OR (appearance_cosine_similarity >= -1
               AND appearance_cosine_similarity <= 1)),
    CHECK (appearance_provenance_compatible IS NULL
           OR appearance_provenance_compatible IN (0, 1)),
    CHECK (
        (hybrid_present = 0
         AND hybrid_snapshot_schema_version IS NULL
         AND hybrid_status IS NULL
         AND hard_gate_reason IS NULL
         AND physical_gate_status IS NULL
         AND trusted_plate_relation IS NULL
         AND attribute_outcome IS NULL
         AND attribute_evidence_coverage IS NULL
         AND attribute_net_evidence IS NULL
         AND appearance_status IS NULL
         AND appearance_cosine_similarity IS NULL
         AND appearance_provenance_compatible IS NULL
         AND hybrid_snapshot_json IS NULL)
        OR
        (hybrid_present = 1
         AND hybrid_snapshot_schema_version =
             {HYBRID_SNAPSHOT_SCHEMA_VERSION}
         AND hybrid_status IS NOT NULL
         AND physical_gate_status IS NOT NULL
         AND trusted_plate_relation IS NOT NULL
         AND attribute_outcome IS NOT NULL
         AND attribute_evidence_coverage IS NOT NULL
         AND attribute_net_evidence IS NOT NULL
         AND appearance_status IS NOT NULL
         AND hybrid_snapshot_json IS NOT NULL)
    )
);
"""


class QueryHistoryValidationError(ValueError):
    """Raised when an audit-history snapshot is invalid."""


class QueryHistoryPersistenceError(RuntimeError):
    """Raised when SQLite cannot complete a history operation safely."""


class QueryHistoryConflictError(QueryHistoryPersistenceError):
    """Raised when a query ID is reused for a different immutable snapshot."""


class QueryHistoryCorruptionError(QueryHistoryPersistenceError):
    """Raised when persisted history fails reconstruction or validation."""


def _required_text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QueryHistoryValidationError(f"{name} must be a non-empty string")
    return value


def _version(value, expected: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) \
            or value != expected:
        raise QueryHistoryValidationError(
            f"{name} must be integer version {expected}")
    return value


def _nonnegative_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QueryHistoryValidationError(
            f"{name} must be an integer >= 0")
    return value


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QueryHistoryValidationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise QueryHistoryValidationError(f"{name} must be finite")
    return result


def _utc_datetime(value, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise QueryHistoryValidationError(
            f"{name} must be a timezone-aware datetime")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise QueryHistoryValidationError(
            f"{name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _canonical_json(value) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise QueryHistoryValidationError(
            f"hybrid snapshot is not JSON-safe: {exc}") from exc


_HYBRID_KEYS = frozenset({
    "source",
    "candidate",
    "status",
    "physical_feasibility",
    "trusted_plate_evidence",
    "fingerprint_comparison",
    "attribute_evidence",
    "appearance_comparison",
    "components",
    "hard_gate_reason",
    "reasons",
})


def _validate_provenance(value, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise QueryHistoryValidationError(f"{name} must be an object or null")
    try:
        AppearanceEmbeddingProvenance(
            dimension=value["dimension"],
            model_id=value["model_id"],
            model_version=value["model_version"],
            weights_sha256=value["weights_sha256"],
            preprocessing_sha256=value["preprocessing_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise QueryHistoryValidationError(
            f"{name} is invalid: {exc}") from exc


def _validate_hybrid_payload(value) -> dict:
    if not isinstance(value, dict) or frozenset(value) != _HYBRID_KEYS:
        raise QueryHistoryValidationError(
            "hybrid snapshot has an unsupported schema")
    for name in ("source", "candidate"):
        item = value[name]
        if not isinstance(item, dict):
            raise QueryHistoryValidationError(
                f"hybrid {name} must be an object")
        _required_text(item.get("event_id"), f"hybrid {name} event_id")

    allowed_statuses = {item.value for item in HybridCandidateStatus}
    if value["status"] not in allowed_statuses:
        raise QueryHistoryValidationError("hybrid status is invalid")
    hard_gate = value["hard_gate_reason"]
    if hard_gate is not None:
        _required_text(hard_gate, "hybrid hard_gate_reason")

    physical = value["physical_feasibility"]
    if not isinstance(physical, dict):
        raise QueryHistoryValidationError(
            "hybrid physical_feasibility must be an object")
    if physical.get("gate_status") not in {
            item.value for item in PhysicalGateStatus}:
        raise QueryHistoryValidationError(
            "hybrid physical gate status is invalid")
    _required_text(physical.get("topology_status"),
                   "hybrid topology_status")
    _required_text(physical.get("travel_status"), "hybrid travel_status")

    trusted = value["trusted_plate_evidence"]
    if not isinstance(trusted, dict) or trusted.get("relation") not in {
            item.value for item in TrustedPlateRelation}:
        raise QueryHistoryValidationError(
            "hybrid trusted plate relation is invalid")

    attributes = value["attribute_evidence"]
    if not isinstance(attributes, dict):
        raise QueryHistoryValidationError(
            "hybrid attribute_evidence must be an object")
    _required_text(attributes.get("outcome"), "hybrid attribute outcome")
    _finite_number(attributes.get("evidence_coverage"),
                   "hybrid attribute evidence_coverage")
    _finite_number(attributes.get("net_evidence"),
                   "hybrid attribute net_evidence")

    appearance = value["appearance_comparison"]
    if not isinstance(appearance, dict) or appearance.get("status") not in {
            item.value for item in AppearanceComparisonStatus}:
        raise QueryHistoryValidationError(
            "hybrid appearance status is invalid")
    cosine = appearance.get("cosine_similarity")
    if cosine is not None:
        cosine = _finite_number(cosine, "hybrid appearance cosine_similarity")
        if not -1.0 <= cosine <= 1.0:
            raise QueryHistoryValidationError(
                "hybrid appearance cosine_similarity must be within [-1, 1]")
    compatible = appearance.get("provenance_compatible")
    if compatible is not None and not isinstance(compatible, bool):
        raise QueryHistoryValidationError(
            "hybrid appearance provenance_compatible must be bool or null")
    _validate_provenance(
        appearance.get("source_provenance"),
        "hybrid source_provenance",
    )
    _validate_provenance(
        appearance.get("candidate_provenance"),
        "hybrid candidate_provenance",
    )
    if not isinstance(value["components"], list) \
            or not isinstance(value["reasons"], list) \
            or not isinstance(value["fingerprint_comparison"], dict):
        raise QueryHistoryValidationError(
            "hybrid component, reason, or fingerprint snapshot is invalid")
    return value


@dataclass(frozen=True)
class HybridResultSnapshot:
    """Canonical JSON snapshot of one caller-supplied Step 37 result."""

    canonical_json: str
    schema_version: int = HYBRID_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self):
        _version(
            self.schema_version,
            HYBRID_SNAPSHOT_SCHEMA_VERSION,
            "hybrid snapshot schema_version",
        )
        if not isinstance(self.canonical_json, str):
            raise QueryHistoryValidationError(
                "hybrid canonical_json must be a string")
        try:
            payload = json.loads(self.canonical_json)
        except (TypeError, ValueError) as exc:
            raise QueryHistoryValidationError(
                f"hybrid canonical_json is invalid: {exc}") from exc
        _validate_hybrid_payload(payload)
        if _canonical_json(payload) != self.canonical_json:
            raise QueryHistoryValidationError(
                "hybrid canonical_json is not in canonical form")

    @classmethod
    def from_result(
            cls,
            result: HybridCandidateMatchResult,
    ) -> "HybridResultSnapshot":
        if not isinstance(result, HybridCandidateMatchResult):
            raise QueryHistoryValidationError(
                "hybrid result must be a HybridCandidateMatchResult")
        return cls(canonical_json=_canonical_json(result.to_dict()))

    def to_dict(self) -> dict:
        return json.loads(self.canonical_json)

    @property
    def source_event_id(self) -> str:
        return self.to_dict()["source"]["event_id"]

    @property
    def candidate_event_id(self) -> str:
        return self.to_dict()["candidate"]["event_id"]

    @property
    def status(self) -> str:
        return self.to_dict()["status"]

    @property
    def hard_gate_reason(self) -> Optional[str]:
        return self.to_dict()["hard_gate_reason"]

    @property
    def physical_gate_status(self) -> str:
        return self.to_dict()["physical_feasibility"]["gate_status"]

    @property
    def trusted_plate_relation(self) -> str:
        return self.to_dict()["trusted_plate_evidence"]["relation"]

    @property
    def attribute_outcome(self) -> str:
        return self.to_dict()["attribute_evidence"]["outcome"]

    @property
    def attribute_evidence_coverage(self) -> float:
        return self.to_dict()["attribute_evidence"]["evidence_coverage"]

    @property
    def attribute_net_evidence(self) -> float:
        return self.to_dict()["attribute_evidence"]["net_evidence"]

    @property
    def appearance_status(self) -> str:
        return self.to_dict()["appearance_comparison"]["status"]

    @property
    def appearance_cosine_similarity(self) -> Optional[float]:
        return self.to_dict()["appearance_comparison"]["cosine_similarity"]

    @property
    def appearance_provenance_compatible(self) -> Optional[bool]:
        return self.to_dict()[
            "appearance_comparison"]["provenance_compatible"]


@dataclass(frozen=True)
class CandidateQueryResultRecord:
    """One zero-based supplied retrieval position and evidence snapshot."""

    position: int
    candidate_event_id: str
    topology_status: str
    travel_status: str
    physical_status: str
    elapsed_seconds: float
    minimum_travel_seconds: Optional[float]
    hybrid_snapshot: Optional[HybridResultSnapshot] = None

    def __post_init__(self):
        _nonnegative_integer(self.position, "position")
        _required_text(self.candidate_event_id, "candidate_event_id")
        _required_text(self.topology_status, "topology_status")
        _required_text(self.travel_status, "travel_status")
        if self.physical_status not in {"feasible", "unknown"}:
            raise QueryHistoryValidationError(
                "physical_status must be feasible or unknown")
        elapsed = _finite_number(self.elapsed_seconds, "elapsed_seconds")
        if elapsed <= 0.0:
            raise QueryHistoryValidationError("elapsed_seconds must be > 0")
        minimum = self.minimum_travel_seconds
        if minimum is not None:
            minimum = _finite_number(minimum, "minimum_travel_seconds")
            if minimum <= 0.0:
                raise QueryHistoryValidationError(
                    "minimum_travel_seconds must be > 0 when present")
        snapshot = self.hybrid_snapshot
        if snapshot is not None:
            if not isinstance(snapshot, HybridResultSnapshot):
                raise QueryHistoryValidationError(
                    "hybrid_snapshot must be HybridResultSnapshot or None")
            if snapshot.candidate_event_id != self.candidate_event_id:
                raise QueryHistoryValidationError(
                    "hybrid candidate event does not match result record")
            physical = snapshot.to_dict()["physical_feasibility"]
            if physical["topology_status"] != self.topology_status \
                    or physical["travel_status"] != self.travel_status:
                raise QueryHistoryValidationError(
                    "hybrid physical metadata conflicts with retrieval result")
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(self, "minimum_travel_seconds", minimum)

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "candidate_event_id": self.candidate_event_id,
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
class CandidateQueryHistoryRecord:
    """Complete immutable snapshot of one supplied Phase 3 query."""

    query_id: str
    source_event_id: str
    executed_at: datetime
    retrieval_policy: HistoricalCandidateRetrievalPolicy
    physical_policy: CrossCameraMatchPolicy
    rows_considered: int
    physically_impossible_excluded: int
    returned_count: int
    truncated: bool
    results: Tuple[CandidateQueryResultRecord, ...]
    retrieval_order_kind: str = RETRIEVAL_ORDER_KIND
    schema_version: int = QUERY_HISTORY_SCHEMA_VERSION
    retrieval_policy_version: int = RETRIEVAL_POLICY_VERSION
    physical_policy_version: int = PHYSICAL_POLICY_VERSION
    implementation_id: str = QUERY_HISTORY_IMPLEMENTATION_ID

    def __post_init__(self):
        _required_text(self.query_id, "query_id")
        _required_text(self.source_event_id, "source_event_id")
        object.__setattr__(
            self,
            "executed_at",
            _utc_datetime(self.executed_at, "executed_at"),
        )
        if not isinstance(
                self.retrieval_policy, HistoricalCandidateRetrievalPolicy):
            raise QueryHistoryValidationError(
                "retrieval_policy must be HistoricalCandidateRetrievalPolicy")
        if not isinstance(self.physical_policy, CrossCameraMatchPolicy):
            raise QueryHistoryValidationError(
                "physical_policy must be CrossCameraMatchPolicy")
        rows = _nonnegative_integer(self.rows_considered, "rows_considered")
        excluded = _nonnegative_integer(
            self.physically_impossible_excluded,
            "physically_impossible_excluded",
        )
        returned = _nonnegative_integer(self.returned_count, "returned_count")
        if not isinstance(self.truncated, bool):
            raise QueryHistoryValidationError("truncated must be bool")
        try:
            results = tuple(self.results)
        except TypeError as exc:
            raise QueryHistoryValidationError(
                "results must be an iterable") from exc
        if any(not isinstance(item, CandidateQueryResultRecord)
               for item in results):
            raise QueryHistoryValidationError(
                "results must contain CandidateQueryResultRecord values")
        if tuple(item.position for item in results) != tuple(range(len(results))):
            raise QueryHistoryValidationError(
                "result positions must be zero-based and contiguous")
        event_ids = tuple(item.candidate_event_id for item in results)
        if len(set(event_ids)) != len(event_ids):
            raise QueryHistoryValidationError(
                "candidate_event_id must be unique within a query")
        if returned != len(results):
            raise QueryHistoryValidationError(
                "returned_count must equal the number of results")
        if returned > self.retrieval_policy.max_results:
            raise QueryHistoryValidationError(
                "returned_count cannot exceed policy max_results")
        if excluded > rows or returned > rows:
            raise QueryHistoryValidationError(
                "retrieval diagnostics are internally inconsistent")
        for result in results:
            snapshot = result.hybrid_snapshot
            if snapshot is not None \
                    and snapshot.source_event_id != self.source_event_id:
                raise QueryHistoryValidationError(
                    "hybrid source event does not match query source")
        if self.retrieval_order_kind != RETRIEVAL_ORDER_KIND:
            raise QueryHistoryValidationError(
                "unsupported retrieval_order_kind")
        _version(self.schema_version, QUERY_HISTORY_SCHEMA_VERSION,
                 "schema_version")
        _version(self.retrieval_policy_version, RETRIEVAL_POLICY_VERSION,
                 "retrieval_policy_version")
        _version(self.physical_policy_version, PHYSICAL_POLICY_VERSION,
                 "physical_policy_version")
        if self.implementation_id != QUERY_HISTORY_IMPLEMENTATION_ID:
            raise QueryHistoryValidationError("unsupported implementation_id")
        object.__setattr__(self, "rows_considered", rows)
        object.__setattr__(self, "physically_impossible_excluded", excluded)
        object.__setattr__(self, "returned_count", returned)
        object.__setattr__(self, "results", results)

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "source_event_id": self.source_event_id,
            "executed_at": self.executed_at.isoformat(),
            "retrieval_policy": self.retrieval_policy.to_dict(),
            "physical_policy": self.physical_policy.to_dict(),
            "rows_considered": self.rows_considered,
            "physically_impossible_excluded":
                self.physically_impossible_excluded,
            "returned_count": self.returned_count,
            "truncated": self.truncated,
            "retrieval_order_kind": self.retrieval_order_kind,
            "schema_version": self.schema_version,
            "retrieval_policy_version": self.retrieval_policy_version,
            "physical_policy_version": self.physical_policy_version,
            "implementation_id": self.implementation_id,
            "results": [result.to_dict() for result in self.results],
        }


def build_candidate_query_history(
        *,
        query_id: str,
        executed_at: datetime,
        retrieval_result: HistoricalCandidateRetrievalResult,
        physical_policy: CrossCameraMatchPolicy,
        hybrid_results: Optional[Mapping] = None,
) -> CandidateQueryHistoryRecord:
    """Snapshot supplied Step 39 order and optional precomputed Step 37 output."""
    if not isinstance(retrieval_result, HistoricalCandidateRetrievalResult):
        raise QueryHistoryValidationError(
            "retrieval_result must be HistoricalCandidateRetrievalResult")
    if not isinstance(physical_policy, CrossCameraMatchPolicy):
        raise QueryHistoryValidationError(
            "physical_policy must be CrossCameraMatchPolicy")
    if hybrid_results is None:
        supplied = {}
    elif isinstance(hybrid_results, Mapping):
        supplied = dict(hybrid_results)
    else:
        raise QueryHistoryValidationError(
            "hybrid_results must be a mapping or None")

    candidate_ids = {
        candidate.evidence.event_id
        for candidate in retrieval_result.candidates
    }
    extras = set(supplied) - candidate_ids
    if extras:
        raise QueryHistoryValidationError(
            "hybrid_results contains events outside the retrieval result")
    results = []
    for position, candidate in enumerate(retrieval_result.candidates):
        event_id = candidate.evidence.event_id
        hybrid = supplied.get(event_id)
        snapshot = (
            None if hybrid is None
            else HybridResultSnapshot.from_result(hybrid)
        )
        results.append(CandidateQueryResultRecord(
            position=position,
            candidate_event_id=event_id,
            topology_status=candidate.topology_status,
            travel_status=candidate.travel_status,
            physical_status=candidate.physical_status,
            elapsed_seconds=candidate.elapsed_seconds,
            minimum_travel_seconds=candidate.minimum_travel_seconds,
            hybrid_snapshot=snapshot,
        ))
    return CandidateQueryHistoryRecord(
        query_id=query_id,
        source_event_id=retrieval_result.source_event_id,
        executed_at=executed_at,
        retrieval_policy=retrieval_result.policy,
        physical_policy=physical_policy,
        rows_considered=retrieval_result.rows_considered,
        physically_impossible_excluded=(
            retrieval_result.physically_impossible_excluded),
        returned_count=retrieval_result.count,
        truncated=retrieval_result.truncated,
        results=tuple(results),
    )


class CandidateQueryHistoryRepository(ABC):
    """Small immutable audit-history repository port."""

    @abstractmethod
    def save_query(self, record: CandidateQueryHistoryRecord) -> bool:
        """Atomically save one snapshot; return false for an exact retry."""

    @abstractmethod
    def get_query(
            self,
            query_id: str,
    ) -> Optional[CandidateQueryHistoryRecord]:
        """Return one complete query snapshot or None."""

    @abstractmethod
    def get_query_results(
            self,
            query_id: str,
    ) -> Optional[Tuple[CandidateQueryResultRecord, ...]]:
        """Return ordered results, an empty tuple, or None for no query."""

    @abstractmethod
    def exists(self, query_id: str) -> bool:
        """Return whether one exact query ID exists."""


class SQLiteCandidateQueryHistoryRepository(CandidateQueryHistoryRepository):
    """Additive stdlib-sqlite immutable query-history repository."""

    def __init__(self, db_path=":memory:"):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def _migrate(self) -> None:
        try:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            raise QueryHistoryPersistenceError(
                f"failed to initialize query-history schema: {exc}"
            ) from exc

    def save_query(self, record: CandidateQueryHistoryRecord) -> bool:
        if not isinstance(record, CandidateQueryHistoryRecord):
            raise QueryHistoryValidationError(
                "record must be a CandidateQueryHistoryRecord")
        query_values = self._query_values(record)
        result_values = tuple(
            self._result_values(record.query_id, result)
            for result in record.results
        )
        try:
            with self._conn:
                existing = self._select_query(record.query_id)
                if existing is not None:
                    current = self._load_query(existing)
                    if current == record:
                        return False
                    raise QueryHistoryConflictError(
                        f"query_id {record.query_id!r} already stores a "
                        "different immutable snapshot")
                self._insert_query(query_values)
                for values in result_values:
                    self._insert_result(values)
                return True
        except (QueryHistoryConflictError, QueryHistoryCorruptionError):
            raise
        except sqlite3.Error as exc:
            raise QueryHistoryPersistenceError(
                f"failed to save query history for {record.query_id!r}: "
                f"{exc}"
            ) from exc

    def get_query(
            self,
            query_id: str,
    ) -> Optional[CandidateQueryHistoryRecord]:
        _required_text(query_id, "query_id")
        try:
            row = self._select_query(query_id)
        except sqlite3.Error as exc:
            raise QueryHistoryPersistenceError(
                f"failed to load query history for {query_id!r}: {exc}"
            ) from exc
        return None if row is None else self._load_query(row)

    def get_query_results(
            self,
            query_id: str,
    ) -> Optional[Tuple[CandidateQueryResultRecord, ...]]:
        record = self.get_query(query_id)
        return None if record is None else record.results

    def exists(self, query_id: str) -> bool:
        _required_text(query_id, "query_id")
        try:
            row = self._conn.execute(
                "SELECT 1 FROM phase3_candidate_queries WHERE query_id = ?",
                (query_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise QueryHistoryPersistenceError(
                f"failed to check query history for {query_id!r}: {exc}"
            ) from exc
        return row is not None

    def close(self) -> None:
        self._conn.close()

    def _select_query(self, query_id: str):
        return self._conn.execute(
            "SELECT * FROM phase3_candidate_queries WHERE query_id = ?",
            (query_id,),
        ).fetchone()

    def _load_query(self, row) -> CandidateQueryHistoryRecord:
        try:
            truncated = row["truncated"]
            if truncated not in (0, 1):
                raise QueryHistoryValidationError(
                    "stored truncated flag must be 0 or 1")
            results = tuple(
                self._row_to_result(result_row)
                for result_row in self._conn.execute(
                    "SELECT * FROM phase3_candidate_query_results "
                    "WHERE query_id = ? ORDER BY position ASC",
                    (row["query_id"],),
                ).fetchall()
            )
            return CandidateQueryHistoryRecord(
                query_id=row["query_id"],
                source_event_id=row["source_event_id"],
                executed_at=datetime.fromisoformat(row["executed_at"]),
                retrieval_policy=HistoricalCandidateRetrievalPolicy(
                    max_time_delta_seconds=row["max_time_delta_seconds"],
                    max_results=row["max_results"],
                ),
                physical_policy=CrossCameraMatchPolicy(
                    maximum_speed_kph=row["maximum_speed_kph"],
                    travel_time_tolerance_seconds=row[
                        "travel_time_tolerance_seconds"],
                    minimum_evidence_coverage=row[
                        "minimum_evidence_coverage"],
                    strong_similarity_threshold=row[
                        "strong_similarity_threshold"],
                ),
                rows_considered=row["rows_considered"],
                physically_impossible_excluded=row[
                    "physically_impossible_excluded"],
                returned_count=row["returned_count"],
                truncated=bool(truncated),
                results=results,
                retrieval_order_kind=row["retrieval_order_kind"],
                schema_version=row["query_schema_version"],
                retrieval_policy_version=row["retrieval_policy_version"],
                physical_policy_version=row["physical_policy_version"],
                implementation_id=row["implementation_id"],
            )
        except QueryHistoryCorruptionError:
            raise
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise QueryHistoryCorruptionError(
                f"stored query history is invalid: {exc}") from exc

    @staticmethod
    def _row_to_result(row) -> CandidateQueryResultRecord:
        try:
            present = row["hybrid_present"]
            snapshot = None
            if present == 1:
                snapshot = HybridResultSnapshot(
                    canonical_json=row["hybrid_snapshot_json"],
                    schema_version=row["hybrid_snapshot_schema_version"],
                )
                SQLiteCandidateQueryHistoryRepository._validate_snapshot_row(
                    row, snapshot)
            elif present == 0:
                nullable_fields = (
                    "hybrid_snapshot_schema_version",
                    "hybrid_status",
                    "hard_gate_reason",
                    "physical_gate_status",
                    "trusted_plate_relation",
                    "attribute_outcome",
                    "attribute_evidence_coverage",
                    "attribute_net_evidence",
                    "appearance_status",
                    "appearance_cosine_similarity",
                    "appearance_provenance_compatible",
                    "hybrid_snapshot_json",
                )
                if any(row[name] is not None for name in nullable_fields):
                    raise QueryHistoryValidationError(
                        "hybrid is marked absent but snapshot data is present")
            else:
                raise QueryHistoryValidationError(
                    "hybrid_present must be 0 or 1")
            return CandidateQueryResultRecord(
                position=row["position"],
                candidate_event_id=row["candidate_event_id"],
                topology_status=row["topology_status"],
                travel_status=row["travel_status"],
                physical_status=row["physical_status"],
                elapsed_seconds=row["elapsed_seconds"],
                minimum_travel_seconds=row["minimum_travel_seconds"],
                hybrid_snapshot=snapshot,
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise QueryHistoryCorruptionError(
                f"stored candidate query result is invalid: {exc}"
            ) from exc

    @staticmethod
    def _validate_snapshot_row(row, snapshot: HybridResultSnapshot) -> None:
        expected = {
            "hybrid_status": snapshot.status,
            "hard_gate_reason": snapshot.hard_gate_reason,
            "physical_gate_status": snapshot.physical_gate_status,
            "trusted_plate_relation": snapshot.trusted_plate_relation,
            "attribute_outcome": snapshot.attribute_outcome,
            "attribute_evidence_coverage":
                snapshot.attribute_evidence_coverage,
            "attribute_net_evidence": snapshot.attribute_net_evidence,
            "appearance_status": snapshot.appearance_status,
            "appearance_cosine_similarity":
                snapshot.appearance_cosine_similarity,
            "appearance_provenance_compatible": (
                None
                if snapshot.appearance_provenance_compatible is None
                else int(snapshot.appearance_provenance_compatible)
            ),
        }
        if any(row[name] != value for name, value in expected.items()):
            raise QueryHistoryCorruptionError(
                "normalized hybrid fields conflict with canonical snapshot")

    @staticmethod
    def _query_values(record: CandidateQueryHistoryRecord) -> tuple:
        physical = record.physical_policy
        return (
            record.query_id,
            record.source_event_id,
            record.executed_at.isoformat(),
            record.retrieval_policy.max_time_delta_seconds,
            record.retrieval_policy.max_results,
            physical.maximum_speed_kph,
            physical.travel_time_tolerance_seconds,
            physical.minimum_evidence_coverage,
            physical.strong_similarity_threshold,
            record.rows_considered,
            record.physically_impossible_excluded,
            record.returned_count,
            int(record.truncated),
            record.retrieval_order_kind,
            record.schema_version,
            record.retrieval_policy_version,
            record.physical_policy_version,
            record.implementation_id,
        )

    @staticmethod
    def _result_values(
            query_id: str,
            result: CandidateQueryResultRecord,
    ) -> tuple:
        snapshot = result.hybrid_snapshot
        if snapshot is None:
            hybrid_values = (0,) + (None,) * 12
        else:
            hybrid_values = (
                1,
                snapshot.schema_version,
                snapshot.status,
                snapshot.hard_gate_reason,
                snapshot.physical_gate_status,
                snapshot.trusted_plate_relation,
                snapshot.attribute_outcome,
                snapshot.attribute_evidence_coverage,
                snapshot.attribute_net_evidence,
                snapshot.appearance_status,
                snapshot.appearance_cosine_similarity,
                (
                    None
                    if snapshot.appearance_provenance_compatible is None
                    else int(snapshot.appearance_provenance_compatible)
                ),
                snapshot.canonical_json,
            )
        return (
            query_id,
            result.position,
            result.candidate_event_id,
            result.topology_status,
            result.travel_status,
            result.physical_status,
            result.elapsed_seconds,
            result.minimum_travel_seconds,
            *hybrid_values,
        )

    def _insert_query(self, values: tuple) -> None:
        self._conn.execute(
            """
            INSERT INTO phase3_candidate_queries (
                query_id, source_event_id, executed_at,
                max_time_delta_seconds, max_results,
                maximum_speed_kph, travel_time_tolerance_seconds,
                minimum_evidence_coverage, strong_similarity_threshold,
                rows_considered, physically_impossible_excluded,
                returned_count, truncated, retrieval_order_kind,
                query_schema_version, retrieval_policy_version,
                physical_policy_version, implementation_id
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            values,
        )

    def _insert_result(self, values: tuple) -> None:
        self._conn.execute(
            """
            INSERT INTO phase3_candidate_query_results (
                query_id, position, candidate_event_id,
                topology_status, travel_status, physical_status,
                elapsed_seconds, minimum_travel_seconds, hybrid_present,
                hybrid_snapshot_schema_version, hybrid_status,
                hard_gate_reason, physical_gate_status,
                trusted_plate_relation, attribute_outcome,
                attribute_evidence_coverage, attribute_net_evidence,
                appearance_status, appearance_cosine_similarity,
                appearance_provenance_compatible, hybrid_snapshot_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            values,
        )


__all__ = [
    "CandidateQueryHistoryRecord",
    "CandidateQueryHistoryRepository",
    "CandidateQueryResultRecord",
    "HYBRID_SNAPSHOT_SCHEMA_VERSION",
    "HybridResultSnapshot",
    "PHYSICAL_POLICY_VERSION",
    "QUERY_HISTORY_IMPLEMENTATION_ID",
    "QUERY_HISTORY_SCHEMA_VERSION",
    "QueryHistoryConflictError",
    "QueryHistoryCorruptionError",
    "QueryHistoryPersistenceError",
    "QueryHistoryValidationError",
    "RETRIEVAL_ORDER_KIND",
    "RETRIEVAL_POLICY_VERSION",
    "SQLiteCandidateQueryHistoryRepository",
    "build_candidate_query_history",
]
