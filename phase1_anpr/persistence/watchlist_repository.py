"""Watchlist + deduplicated alert persistence (Step 15).

Self-contained sqlite3 repository, same approach as
`SQLiteObservationRepository`. Kept separate from observation storage so the
watchlist feature is isolated; alert generation consumes an already-built
observation dict and needs no cross-table join.

Idempotency: an alert's identity is (watchlist_id, event_id) — reprocessing the
same accepted observation against the same entry never duplicates. Only
`accepted` observations with an exact normalized-plate match against an *enabled*
entry generate alerts.
"""

import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from phase1_anpr.normalization.plate_normalizer import PlateNormalizer

_SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist (
    watchlist_id     TEXT PRIMARY KEY,
    normalized_plate TEXT NOT NULL,
    label            TEXT,
    enabled          INTEGER NOT NULL DEFAULT 1,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_watchlist_plate ON watchlist (normalized_plate);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id         TEXT PRIMARY KEY,
    watchlist_id     TEXT NOT NULL,
    event_id         TEXT NOT NULL,
    normalized_plate TEXT NOT NULL,
    camera_id        TEXT NOT NULL,
    timestamp        TEXT NOT NULL,
    confidence       REAL NOT NULL,
    status           TEXT NOT NULL,
    UNIQUE (watchlist_id, event_id)
);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts (timestamp);
"""

_WL_FIELDS = ("watchlist_id", "normalized_plate", "label", "enabled", "created_at")
_ALERT_FIELDS = ("alert_id", "watchlist_id", "event_id", "normalized_plate",
                 "camera_id", "timestamp", "confidence", "status")


class WatchlistError(Exception):
    """Invalid watchlist input (e.g. empty/invalid plate)."""


def _deterministic_alert_id(watchlist_id: str, event_id: str) -> str:
    # Deterministic so the same (entry, observation) always maps to one id.
    return str(uuid.uuid5(uuid.NAMESPACE_URL,
                          f"citysight/alert/{watchlist_id}/{event_id}"))


class SQLiteWatchlistRepository:
    """sqlite3-backed watchlist + alerts. `db_path=":memory:"` for tests."""

    def __init__(self, db_path=":memory:", normalizer=None, now=None):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._normalizer = normalizer or PlateNormalizer()
        # Injectable clock keeps created_at deterministic in tests.
        self._now = now or (lambda: __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat())
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # --- watchlist ------------------------------------------------------------

    def add(self, plate: str, label: Optional[str] = None,
            watchlist_id: Optional[str] = None) -> dict:
        """Add an enabled entry. Raises WatchlistError on empty/invalid plate."""
        if plate is None or not str(plate).strip():
            raise WatchlistError("plate is required")
        normalized = self._normalizer.normalize_text(str(plate))
        if not normalized:
            raise WatchlistError("plate is empty after normalization")
        wid = watchlist_id or str(uuid.uuid4())
        created = self._now()
        self._conn.execute(
            "INSERT INTO watchlist (watchlist_id, normalized_plate, label, "
            "enabled, created_at) VALUES (?,?,?,1,?)",
            (wid, normalized, label, created),
        )
        self._conn.commit()
        return {"watchlist_id": wid, "normalized_plate": normalized,
                "label": label, "enabled": 1, "created_at": created}

    def list(self) -> list:
        rows = self._conn.execute(
            "SELECT * FROM watchlist ORDER BY created_at DESC, watchlist_id DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get(self, watchlist_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM watchlist WHERE watchlist_id = ?", (watchlist_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def disable(self, watchlist_id: str) -> bool:
        """Disable an entry (soft-delete). Returns True if a row changed."""
        cur = self._conn.execute(
            "UPDATE watchlist SET enabled = 0 WHERE watchlist_id = ? AND enabled = 1",
            (watchlist_id,),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete(self, watchlist_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM watchlist WHERE watchlist_id = ?", (watchlist_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def enabled_for_plate(self, normalized_plate: str) -> list:
        rows = self._conn.execute(
            "SELECT * FROM watchlist WHERE normalized_plate = ? AND enabled = 1",
            (normalized_plate,),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- alerts ---------------------------------------------------------------

    def list_alerts(self, limit: int = 50) -> list:
        rows = self._conn.execute(
            "SELECT * FROM alerts ORDER BY timestamp DESC, alert_id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]

    def _insert_alert(self, entry: dict, obs: dict) -> Optional[dict]:
        alert_id = _deterministic_alert_id(entry["watchlist_id"], obs["event_id"])
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO alerts (alert_id, watchlist_id, event_id, "
            "normalized_plate, camera_id, timestamp, confidence, status) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (alert_id, entry["watchlist_id"], obs["event_id"],
             obs["plate_normalized"], obs["camera_id"], obs["timestamp"],
             obs["confidence"], obs["status"]),
        )
        self._conn.commit()
        if cur.rowcount == 0:
            return None  # duplicate — idempotent no-op
        return {"alert_id": alert_id, "watchlist_id": entry["watchlist_id"],
                "event_id": obs["event_id"],
                "normalized_plate": obs["plate_normalized"],
                "camera_id": obs["camera_id"], "timestamp": obs["timestamp"],
                "confidence": obs["confidence"], "status": obs["status"]}

    def process_observation(self, observation) -> list:
        """Create alerts for an accepted, exact-matching observation.

        Accepts a PlateObservation or a dict. Returns the list of newly created
        alerts (empty on non-accepted, no match, or duplicate).
        """
        obs = observation.to_dict() if hasattr(observation, "to_dict") else dict(observation)
        # Only ACCEPTED observations generate exact-match alerts.
        if obs.get("status") != "accepted":
            return []
        plate = obs.get("plate_normalized")
        if not plate:
            return []
        created = []
        for entry in self.enabled_for_plate(plate):
            alert = self._insert_alert(entry, obs)
            if alert is not None:
                created.append(alert)
        return created

    def close(self) -> None:
        self._conn.close()
