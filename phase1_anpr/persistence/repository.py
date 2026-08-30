"""Repository layer for canonical plate observations (Step 12).

`ObservationRepository` is the port that later API/search code depends on. A
stdlib `sqlite3` implementation is provided as the default (no new dependencies,
no running service needed for tests). The SQL is deliberately vanilla so a
Postgres implementation can be dropped in behind the same interface for the SIH
demo without changing callers.

Idempotency: `event_id` is the primary key; re-saving the same event is a no-op
(`INSERT OR IGNORE`), so retries never create duplicates.

Evidence: only a reference/path is stored, never image bytes. Abstained
observations are persisted without asserting a plate identity (plate/evidence
fields stay NULL, mirroring the canonical event).
"""

import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from phase1_anpr.observation.observation_builder import PlateObservation


# Columns persisted for later search/API. Kept flat and explicit; names track
# the canonical observation contract plus the format/state metadata that the
# builder does not carry on the event itself.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS plate_observations (
    event_id            TEXT PRIMARY KEY,
    camera_id           TEXT NOT NULL,
    track_id            INTEGER NOT NULL,
    timestamp           TEXT NOT NULL,
    plate_raw           TEXT,
    plate_normalized    TEXT,
    confidence          REAL NOT NULL,
    status              TEXT NOT NULL,
    detector_confidence REAL,
    ocr_confidence      REAL,
    quality_score       REAL,
    best_frame_number   INTEGER,
    format_type         TEXT,
    state_code          TEXT,
    evidence_ref        TEXT,
    model_version       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obs_camera_ts ON plate_observations (camera_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_obs_plate ON plate_observations (plate_normalized);
"""


class ObservationRepository(ABC):
    """Port for persisting and reading back canonical plate observations."""

    @abstractmethod
    def save(self, observation: PlateObservation, *, format_type: Optional[str] = None,
             state_code: Optional[str] = None, evidence_ref: Optional[str] = None) -> bool:
        """Persist an observation idempotently.

        Returns True if a new row was inserted, False if `event_id` already
        existed (idempotent no-op).
        """

    @abstractmethod
    def get(self, event_id: str) -> Optional[dict]:
        """Return the stored row as a dict, or None if not found."""

    @abstractmethod
    def list_recent(self, limit: int = 50) -> list:
        """Return the most recent observations, newest first."""

    @abstractmethod
    def list_by_plate(self, plate_normalized: str, limit: int = 50) -> list:
        """Return observations for an exact normalized plate, newest first."""

    @abstractmethod
    def list_by_camera(self, camera_id: str, limit: int = 50) -> list:
        """Return observations for a camera, newest first."""


class SQLiteObservationRepository(ObservationRepository):
    """sqlite3-backed repository. `db_path=":memory:"` for tests/fakes."""

    def __init__(self, db_path=":memory:"):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False keeps it usable from a simple pipeline; sqlite3
        # still serializes writes. Row factory gives dict-like reads.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        """Create schema if absent (safe to call repeatedly)."""
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save(self, observation, *, format_type=None, state_code=None,
             evidence_ref=None) -> bool:
        d = observation.to_dict()
        # Default the evidence reference to the observation's own image path so
        # callers that already populated it don't have to repeat themselves.
        ref = evidence_ref if evidence_ref is not None else d.get("plate_image_path")
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO plate_observations (
                event_id, camera_id, track_id, timestamp, plate_raw,
                plate_normalized, confidence, status, detector_confidence,
                ocr_confidence, quality_score, best_frame_number, format_type,
                state_code, evidence_ref, model_version
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                d["event_id"], d["camera_id"], d["track_id"], d["timestamp"],
                d["plate_raw"], d["plate_normalized"], d["confidence"],
                d["status"], d["detector_confidence"], d["ocr_confidence"],
                d["quality_score"], d["best_frame_number"], format_type,
                state_code, ref, d["model_version"],
            ),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get(self, event_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM plate_observations WHERE event_id = ?", (event_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    # Newest-first ordering uses timestamp then event_id as a stable tiebreak.
    def list_recent(self, limit: int = 50) -> list:
        rows = self._conn.execute(
            "SELECT * FROM plate_observations "
            "ORDER BY timestamp DESC, event_id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_by_plate(self, plate_normalized: str, limit: int = 50) -> list:
        rows = self._conn.execute(
            "SELECT * FROM plate_observations WHERE plate_normalized = ? "
            "ORDER BY timestamp DESC, event_id DESC LIMIT ?",
            (plate_normalized, int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_by_camera(self, camera_id: str, limit: int = 50) -> list:
        rows = self._conn.execute(
            "SELECT * FROM plate_observations WHERE camera_id = ? "
            "ORDER BY timestamp DESC, event_id DESC LIMIT ?",
            (camera_id, int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM plate_observations"
        ).fetchone()[0]

    def close(self) -> None:
        self._conn.close()
