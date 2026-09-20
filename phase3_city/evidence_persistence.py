"""Observation-scoped SQLite vehicle/appearance evidence storage (Step 38).

This module is deliberately storage-only.  It provides primary-key lookup by
an existing event ID, but no historical search, candidate retrieval, ranking,
similarity query, identity table, or production-pipeline integration.
"""

from __future__ import annotations

import math
import sqlite3
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from phase2_city.cross_camera_matching import FingerprintObservation
from phase2_city.fingerprint import FingerprintValidationError, VehicleFingerprint
from phase3_city.appearance_embedding import (
    AppearanceEmbeddingValidationError,
    VehicleAppearanceEmbedding,
)


VEHICLE_EVIDENCE_SCHEMA_VERSION = 1
EMBEDDING_SERIALIZATION_VERSION = 1
EMBEDDING_SERIALIZATION_FORMAT = "ieee754-float64-le"
EMBEDDING_VECTOR_TOLERANCE = 1e-15


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS phase3_vehicle_evidence (
    event_id                         TEXT PRIMARY KEY,
    camera_id                        TEXT NOT NULL,
    timestamp                        TEXT,
    observation_status               TEXT NOT NULL
        CHECK (observation_status IN ('accepted', 'review', 'abstained')),
    normalized_plate                 TEXT,
    vehicle_colour                   TEXT,
    vehicle_class                    TEXT,
    record_schema_version            INTEGER NOT NULL
        CHECK (record_schema_version = {VEHICLE_EVIDENCE_SCHEMA_VERSION}),
    embedding_serialization_version  INTEGER NOT NULL
        CHECK (embedding_serialization_version =
               {EMBEDDING_SERIALIZATION_VERSION}),
    embedding_serialization_format   TEXT NOT NULL
        CHECK (embedding_serialization_format =
               '{EMBEDDING_SERIALIZATION_FORMAT}'),
    appearance_present               INTEGER NOT NULL
        CHECK (appearance_present IN (0, 1)),
    embedding_vector                 BLOB,
    embedding_dimension              INTEGER,
    embedding_model_id               TEXT,
    embedding_model_version          TEXT,
    embedding_weights_sha256         TEXT,
    embedding_preprocessing_sha256   TEXT,
    CHECK (
        (appearance_present = 0
         AND embedding_vector IS NULL
         AND embedding_dimension IS NULL
         AND embedding_model_id IS NULL
         AND embedding_model_version IS NULL
         AND embedding_weights_sha256 IS NULL
         AND embedding_preprocessing_sha256 IS NULL)
        OR
        (appearance_present = 1
         AND embedding_vector IS NOT NULL
         AND embedding_dimension > 0
         AND length(embedding_vector) = embedding_dimension * 8
         AND embedding_model_id IS NOT NULL
         AND embedding_model_version IS NOT NULL
         AND embedding_weights_sha256 IS NOT NULL
         AND embedding_preprocessing_sha256 IS NOT NULL)
    )
);
"""


class VehicleEvidenceValidationError(ValueError):
    """Raised when a record does not satisfy the Step 38 domain contract."""


class VehicleEvidencePersistenceError(RuntimeError):
    """Raised when SQLite cannot complete an evidence operation safely."""


class VehicleEvidenceConflictError(VehicleEvidencePersistenceError):
    """Raised when an event retry contradicts already-persisted evidence."""


class VehicleEvidenceCorruptionError(VehicleEvidencePersistenceError):
    """Raised when a stored row cannot be reconstructed and validated."""


def _supported_version(value, expected: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VehicleEvidenceValidationError(
            f"{name} must be integer version {expected}")
    if value != expected:
        raise VehicleEvidenceValidationError(
            f"unsupported {name}: {value}; expected {expected}")
    return value


@dataclass(frozen=True)
class VehicleEvidenceRecord:
    """Immutable evidence persisted for one canonical observation event."""

    event_id: str
    camera_id: str
    timestamp: Optional[datetime]
    observation_status: str
    fingerprint: VehicleFingerprint
    appearance_embedding: Optional[VehicleAppearanceEmbedding] = None
    schema_version: int = VEHICLE_EVIDENCE_SCHEMA_VERSION
    embedding_serialization_version: int = EMBEDDING_SERIALIZATION_VERSION

    def __post_init__(self):
        if not isinstance(self.fingerprint, VehicleFingerprint):
            raise VehicleEvidenceValidationError(
                "fingerprint must be a VehicleFingerprint")
        if (self.appearance_embedding is not None
                and not isinstance(
                    self.appearance_embedding, VehicleAppearanceEmbedding)):
            raise VehicleEvidenceValidationError(
                "appearance_embedding must be VehicleAppearanceEmbedding "
                "or None")
        _supported_version(
            self.schema_version,
            VEHICLE_EVIDENCE_SCHEMA_VERSION,
            "schema_version",
        )
        _supported_version(
            self.embedding_serialization_version,
            EMBEDDING_SERIALIZATION_VERSION,
            "embedding_serialization_version",
        )
        try:
            observation = FingerprintObservation(
                event_id=self.event_id,
                camera_id=self.camera_id,
                timestamp=self.timestamp,
                observation_status=self.observation_status,
                fingerprint=self.fingerprint,
            )
        except (TypeError, ValueError) as exc:
            raise VehicleEvidenceValidationError(str(exc)) from exc
        object.__setattr__(self, "event_id", observation.event_id)
        object.__setattr__(self, "camera_id", observation.camera_id)
        object.__setattr__(self, "timestamp", observation.timestamp)
        object.__setattr__(self, "observation_status",
                           observation.observation_status)

    @classmethod
    def from_observation(
            cls,
            observation: FingerprintObservation,
            appearance_embedding: Optional[VehicleAppearanceEmbedding] = None,
    ) -> "VehicleEvidenceRecord":
        if not isinstance(observation, FingerprintObservation):
            raise VehicleEvidenceValidationError(
                "observation must be a FingerprintObservation")
        return cls(
            event_id=observation.event_id,
            camera_id=observation.camera_id,
            timestamp=observation.timestamp,
            observation_status=observation.observation_status,
            fingerprint=observation.fingerprint,
            appearance_embedding=appearance_embedding,
        )

    def to_fingerprint_observation(self) -> FingerprintObservation:
        return FingerprintObservation(
            event_id=self.event_id,
            camera_id=self.camera_id,
            timestamp=self.timestamp,
            observation_status=self.observation_status,
            fingerprint=self.fingerprint,
        )

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "timestamp": (
                None if self.timestamp is None
                else self.timestamp.isoformat()
            ),
            "observation_status": self.observation_status,
            "fingerprint": self.fingerprint.to_dict(),
            "appearance_embedding": (
                None if self.appearance_embedding is None
                else self.appearance_embedding.to_dict()
            ),
            "schema_version": self.schema_version,
            "embedding_serialization_version":
                self.embedding_serialization_version,
            "embedding_serialization_format":
                EMBEDDING_SERIALIZATION_FORMAT,
        }


class VehicleEvidenceRepository(ABC):
    """Small event-keyed repository port for Phase 3 evidence."""

    @abstractmethod
    def save(self, record: VehicleEvidenceRecord) -> bool:
        """Persist or monotonically enrich one event.

        Returns ``True`` when stored state changed and ``False`` for an
        idempotent retry or an incoming record containing less evidence.
        """

    @abstractmethod
    def get(self, event_id: str) -> Optional[VehicleEvidenceRecord]:
        """Return one event record, or ``None`` when it is absent."""

    @abstractmethod
    def get_appearance(
            self,
            event_id: str,
    ) -> Optional[VehicleAppearanceEmbedding]:
        """Return one event's embedding, or ``None`` when absent/unavailable."""

    @abstractmethod
    def exists(self, event_id: str) -> bool:
        """Return whether one exact event key exists."""


def _encode_embedding(embedding: VehicleAppearanceEmbedding) -> bytes:
    try:
        return struct.pack(
            f"<{embedding.dimension}d",
            *embedding.vector,
        )
    except (OverflowError, struct.error) as exc:
        raise VehicleEvidenceValidationError(
            f"embedding cannot be serialized as float64-le: {exc}") from exc


def _decode_embedding(row) -> Optional[VehicleAppearanceEmbedding]:
    present = row["appearance_present"]
    if present == 0:
        nullable_fields = (
            "embedding_vector",
            "embedding_dimension",
            "embedding_model_id",
            "embedding_model_version",
            "embedding_weights_sha256",
            "embedding_preprocessing_sha256",
        )
        if any(row[field] is not None for field in nullable_fields):
            raise VehicleEvidenceCorruptionError(
                "appearance is marked absent but embedding data is present")
        return None
    if present != 1:
        raise VehicleEvidenceCorruptionError(
            "appearance_present must be 0 or 1")
    dimension = row["embedding_dimension"]
    if isinstance(dimension, bool) or not isinstance(dimension, int) \
            or dimension <= 0:
        raise VehicleEvidenceCorruptionError(
            "stored embedding_dimension must be a positive integer")
    blob = row["embedding_vector"]
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise VehicleEvidenceCorruptionError(
            "stored embedding_vector must be a BLOB")
    raw = bytes(blob)
    expected_bytes = dimension * 8
    if len(raw) != expected_bytes:
        raise VehicleEvidenceCorruptionError(
            "stored embedding byte length does not match dimension")
    try:
        vector = struct.unpack(f"<{dimension}d", raw)
        return VehicleAppearanceEmbedding(
            vector=vector,
            dimension=dimension,
            model_id=row["embedding_model_id"],
            model_version=row["embedding_model_version"],
            weights_sha256=row["embedding_weights_sha256"],
            preprocessing_sha256=row[
                "embedding_preprocessing_sha256"],
        )
    except (struct.error, AppearanceEmbeddingValidationError) as exc:
        raise VehicleEvidenceCorruptionError(
            f"stored appearance embedding is invalid: {exc}") from exc


def _embedding_provenance(embedding: VehicleAppearanceEmbedding) -> tuple:
    return (
        embedding.dimension,
        embedding.model_id,
        embedding.model_version,
        embedding.weights_sha256,
        embedding.preprocessing_sha256,
    )


def _embeddings_equivalent(
        current: VehicleAppearanceEmbedding,
        incoming: VehicleAppearanceEmbedding,
) -> bool:
    if _embedding_provenance(current) != _embedding_provenance(incoming):
        return False
    return all(
        math.isclose(
            left,
            right,
            rel_tol=0.0,
            abs_tol=EMBEDDING_VECTOR_TOLERANCE,
        )
        for left, right in zip(current.vector, incoming.vector)
    )


def _merge_optional_text(
        current: Optional[str],
        incoming: Optional[str],
        field_name: str,
) -> tuple[Optional[str], bool]:
    if current is not None and incoming is not None and current != incoming:
        raise VehicleEvidenceConflictError(
            f"event evidence conflicts on {field_name}")
    if current is None and incoming is not None:
        return incoming, True
    return current, False


def _merge_records(
        current: VehicleEvidenceRecord,
        incoming: VehicleEvidenceRecord,
) -> VehicleEvidenceRecord:
    for field_name in (
            "event_id", "camera_id", "timestamp", "observation_status"):
        if getattr(current, field_name) != getattr(incoming, field_name):
            raise VehicleEvidenceConflictError(
                f"event evidence conflicts on {field_name}")

    plate, plate_changed = _merge_optional_text(
        current.fingerprint.normalized_plate,
        incoming.fingerprint.normalized_plate,
        "normalized_plate",
    )
    colour, colour_changed = _merge_optional_text(
        current.fingerprint.vehicle_colour,
        incoming.fingerprint.vehicle_colour,
        "vehicle_colour",
    )
    vehicle_class, class_changed = _merge_optional_text(
        current.fingerprint.vehicle_class,
        incoming.fingerprint.vehicle_class,
        "vehicle_class",
    )

    appearance = current.appearance_embedding
    appearance_changed = False
    if appearance is None and incoming.appearance_embedding is not None:
        appearance = incoming.appearance_embedding
        appearance_changed = True
    elif appearance is not None and incoming.appearance_embedding is not None:
        if (_embedding_provenance(appearance)
                != _embedding_provenance(incoming.appearance_embedding)):
            raise VehicleEvidenceConflictError(
                "event cannot receive incompatible embedding provenance")
        if not _embeddings_equivalent(
                appearance, incoming.appearance_embedding):
            raise VehicleEvidenceConflictError(
                "event cannot receive a different embedding vector")

    if not any((plate_changed, colour_changed, class_changed,
                appearance_changed)):
        return current
    try:
        fingerprint = VehicleFingerprint(
            normalized_plate=plate,
            vehicle_colour=colour,
            vehicle_class=vehicle_class,
        )
    except FingerprintValidationError as exc:  # pragma: no cover - defensive
        raise VehicleEvidenceConflictError(str(exc)) from exc
    return VehicleEvidenceRecord(
        event_id=current.event_id,
        camera_id=current.camera_id,
        timestamp=current.timestamp,
        observation_status=current.observation_status,
        fingerprint=fingerprint,
        appearance_embedding=appearance,
    )


class SQLiteVehicleEvidenceRepository(VehicleEvidenceRepository):
    """Additive stdlib-sqlite repository keyed by canonical ``event_id``."""

    def __init__(self, db_path=":memory:"):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        try:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            raise VehicleEvidencePersistenceError(
                f"failed to initialize Phase 3 evidence schema: {exc}"
            ) from exc

    def save(self, record: VehicleEvidenceRecord) -> bool:
        if not isinstance(record, VehicleEvidenceRecord):
            raise VehicleEvidenceValidationError(
                "record must be a VehicleEvidenceRecord")
        # Serialize before entering the transaction so a validation failure can
        # never leave a partially changed row.
        self._row_values(record)
        try:
            with self._conn:
                row = self._select(record.event_id)
                if row is None:
                    self._insert(record)
                    return True
                current = self._row_to_record(row)
                merged = _merge_records(current, record)
                if merged is current:
                    return False
                self._update(merged)
                return True
        except (VehicleEvidenceConflictError,
                VehicleEvidenceCorruptionError):
            raise
        except sqlite3.Error as exc:
            raise VehicleEvidencePersistenceError(
                f"failed to save vehicle evidence for "
                f"{record.event_id!r}: {exc}") from exc

    def get(self, event_id: str) -> Optional[VehicleEvidenceRecord]:
        if not isinstance(event_id, str) or not event_id.strip():
            raise VehicleEvidenceValidationError(
                "event_id must be a non-empty string")
        try:
            row = self._select(event_id)
        except sqlite3.Error as exc:
            raise VehicleEvidencePersistenceError(
                f"failed to load vehicle evidence for {event_id!r}: {exc}"
            ) from exc
        return None if row is None else self._row_to_record(row)

    def get_appearance(
            self,
            event_id: str,
    ) -> Optional[VehicleAppearanceEmbedding]:
        record = self.get(event_id)
        return None if record is None else record.appearance_embedding

    def exists(self, event_id: str) -> bool:
        if not isinstance(event_id, str) or not event_id.strip():
            raise VehicleEvidenceValidationError(
                "event_id must be a non-empty string")
        try:
            row = self._conn.execute(
                "SELECT 1 FROM phase3_vehicle_evidence WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise VehicleEvidencePersistenceError(
                f"failed to check vehicle evidence for {event_id!r}: {exc}"
            ) from exc
        return row is not None

    def close(self) -> None:
        self._conn.close()

    def _select(self, event_id: str):
        return self._conn.execute(
            "SELECT * FROM phase3_vehicle_evidence WHERE event_id = ?",
            (event_id,),
        ).fetchone()

    def _insert(self, record: VehicleEvidenceRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO phase3_vehicle_evidence (
                event_id, camera_id, timestamp, observation_status,
                normalized_plate, vehicle_colour, vehicle_class,
                record_schema_version, embedding_serialization_version,
                embedding_serialization_format, appearance_present,
                embedding_vector, embedding_dimension, embedding_model_id,
                embedding_model_version, embedding_weights_sha256,
                embedding_preprocessing_sha256
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            self._row_values(record),
        )

    def _update(self, record: VehicleEvidenceRecord) -> None:
        values = self._row_values(record)
        self._conn.execute(
            """
            UPDATE phase3_vehicle_evidence SET
                camera_id = ?, timestamp = ?, observation_status = ?,
                normalized_plate = ?, vehicle_colour = ?, vehicle_class = ?,
                record_schema_version = ?,
                embedding_serialization_version = ?,
                embedding_serialization_format = ?, appearance_present = ?,
                embedding_vector = ?, embedding_dimension = ?,
                embedding_model_id = ?, embedding_model_version = ?,
                embedding_weights_sha256 = ?,
                embedding_preprocessing_sha256 = ?
            WHERE event_id = ?
            """,
            values[1:] + (values[0],),
        )

    @staticmethod
    def _row_values(record: VehicleEvidenceRecord) -> tuple:
        embedding = record.appearance_embedding
        if embedding is None:
            appearance_values = (0, None, None, None, None, None, None)
        else:
            appearance_values = (
                1,
                sqlite3.Binary(_encode_embedding(embedding)),
                embedding.dimension,
                embedding.model_id,
                embedding.model_version,
                embedding.weights_sha256,
                embedding.preprocessing_sha256,
            )
        return (
            record.event_id,
            record.camera_id,
            None if record.timestamp is None else record.timestamp.isoformat(),
            record.observation_status,
            record.fingerprint.normalized_plate,
            record.fingerprint.vehicle_colour,
            record.fingerprint.vehicle_class,
            record.schema_version,
            record.embedding_serialization_version,
            EMBEDDING_SERIALIZATION_FORMAT,
            *appearance_values,
        )

    @staticmethod
    def _row_to_record(row) -> VehicleEvidenceRecord:
        try:
            _supported_version(
                row["record_schema_version"],
                VEHICLE_EVIDENCE_SCHEMA_VERSION,
                "record_schema_version",
            )
            _supported_version(
                row["embedding_serialization_version"],
                EMBEDDING_SERIALIZATION_VERSION,
                "embedding_serialization_version",
            )
            if (row["embedding_serialization_format"]
                    != EMBEDDING_SERIALIZATION_FORMAT):
                raise VehicleEvidenceValidationError(
                    "unsupported embedding_serialization_format")
            timestamp = (
                None if row["timestamp"] is None
                else datetime.fromisoformat(row["timestamp"])
            )
            fingerprint = VehicleFingerprint(
                normalized_plate=row["normalized_plate"],
                vehicle_colour=row["vehicle_colour"],
                vehicle_class=row["vehicle_class"],
            )
            embedding = _decode_embedding(row)
            return VehicleEvidenceRecord(
                event_id=row["event_id"],
                camera_id=row["camera_id"],
                timestamp=timestamp,
                observation_status=row["observation_status"],
                fingerprint=fingerprint,
                appearance_embedding=embedding,
                schema_version=row["record_schema_version"],
                embedding_serialization_version=row[
                    "embedding_serialization_version"],
            )
        except VehicleEvidenceCorruptionError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise VehicleEvidenceCorruptionError(
                f"stored vehicle evidence is invalid: {exc}") from exc


__all__ = [
    "EMBEDDING_SERIALIZATION_FORMAT",
    "EMBEDDING_SERIALIZATION_VERSION",
    "EMBEDDING_VECTOR_TOLERANCE",
    "SQLiteVehicleEvidenceRepository",
    "VEHICLE_EVIDENCE_SCHEMA_VERSION",
    "VehicleEvidenceConflictError",
    "VehicleEvidenceCorruptionError",
    "VehicleEvidencePersistenceError",
    "VehicleEvidenceRecord",
    "VehicleEvidenceRepository",
    "VehicleEvidenceValidationError",
]
