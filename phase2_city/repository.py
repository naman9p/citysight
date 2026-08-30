"""Phase 2 SQLite city repository: cameras + directed links (Step 17).

Mirrors the Phase 1 repository style: an abstract port plus a stdlib ``sqlite3``
implementation, vanilla parameterized SQL, idempotent schema, ``":memory:"`` for
tests. Phase 2 tables (``cameras``, ``camera_links``) are separate from Phase 1
tables, so the same ``db_path`` may be shared safely.

Referential integrity is enforced in application code (a link requires both
cameras to already exist) and, additionally, via SQLite foreign keys. A camera
still referenced by any link cannot be deleted — accidental deletion is
prevented rather than silently cascaded.
"""

import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from phase2_city.models import Camera, CameraLink


class CameraNotFoundError(LookupError):
    """Raised when an operation references a camera that does not exist."""


class InvalidCameraLinkError(ValueError):
    """Raised when a link cannot be stored (missing endpoint camera, etc.)."""


class CameraInUseError(RuntimeError):
    """Raised when deleting a camera still referenced by one or more links."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    camera_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    latitude    REAL NOT NULL,
    longitude   REAL NOT NULL,
    road_name   TEXT NOT NULL,
    heading_deg REAL NOT NULL,
    zone        TEXT,
    enabled     INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS camera_links (
    from_camera_id   TEXT NOT NULL,
    to_camera_id     TEXT NOT NULL,
    distance_m       REAL NOT NULL,
    road_name        TEXT NOT NULL,
    travel_direction TEXT,
    PRIMARY KEY (from_camera_id, to_camera_id),
    FOREIGN KEY (from_camera_id) REFERENCES cameras (camera_id),
    FOREIGN KEY (to_camera_id) REFERENCES cameras (camera_id)
);
CREATE INDEX IF NOT EXISTS idx_links_from ON camera_links (from_camera_id);
CREATE INDEX IF NOT EXISTS idx_links_to ON camera_links (to_camera_id);
"""


class CityRepository(ABC):
    """Port for persisting the city camera registry and directed link graph."""

    @abstractmethod
    def save_camera(self, camera: Camera) -> None:
        """Insert or update (upsert) a camera by ``camera_id``."""

    @abstractmethod
    def get_camera(self, camera_id: str) -> Optional[Camera]:
        """Return the camera, or None if not found."""

    @abstractmethod
    def list_cameras(self) -> list:
        """Return all cameras, ordered by ``camera_id``."""

    @abstractmethod
    def delete_camera(self, camera_id: str) -> bool:
        """Delete an unreferenced camera. Returns True if a row was removed."""

    @abstractmethod
    def save_link(self, link: CameraLink) -> None:
        """Insert or update (upsert) a directed link. Both cameras must exist."""

    @abstractmethod
    def get_link(self, from_camera_id: str, to_camera_id: str) -> Optional[CameraLink]:
        """Return the directed link, or None if not found."""

    @abstractmethod
    def list_links(self) -> list:
        """Return all directed links."""

    @abstractmethod
    def delete_link(self, from_camera_id: str, to_camera_id: str) -> bool:
        """Delete a directed link. Returns True if a row was removed."""


class SQLiteCityRepository(CityRepository):
    """sqlite3-backed city repository. ``db_path=":memory:"`` for tests."""

    def __init__(self, db_path=":memory:"):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Enforce declared foreign keys (off by default in sqlite3).
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def _migrate(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # --- cameras ---------------------------------------------------------
    def save_camera(self, camera: Camera) -> None:
        """Upsert a camera keyed by ``camera_id``."""
        self._conn.execute(
            """
            INSERT INTO cameras (
                camera_id, name, latitude, longitude, road_name, heading_deg,
                zone, enabled
            ) VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(camera_id) DO UPDATE SET
                name=excluded.name, latitude=excluded.latitude,
                longitude=excluded.longitude, road_name=excluded.road_name,
                heading_deg=excluded.heading_deg, zone=excluded.zone,
                enabled=excluded.enabled
            """,
            (
                camera.camera_id, camera.name, camera.latitude, camera.longitude,
                camera.road_name, camera.heading_deg, camera.zone,
                1 if camera.enabled else 0,
            ),
        )
        self._conn.commit()

    def get_camera(self, camera_id: str) -> Optional[Camera]:
        row = self._conn.execute(
            "SELECT * FROM cameras WHERE camera_id = ?", (camera_id,)
        ).fetchone()
        return self._row_to_camera(row) if row is not None else None

    def list_cameras(self) -> list:
        rows = self._conn.execute(
            "SELECT * FROM cameras ORDER BY camera_id"
        ).fetchall()
        return [self._row_to_camera(r) for r in rows]

    def delete_camera(self, camera_id: str) -> bool:
        referencing = self._conn.execute(
            "SELECT COUNT(*) FROM camera_links "
            "WHERE from_camera_id = ? OR to_camera_id = ?",
            (camera_id, camera_id),
        ).fetchone()[0]
        if referencing:
            raise CameraInUseError(
                f"camera '{camera_id}' is referenced by {referencing} link(s); "
                "delete those links first"
            )
        cur = self._conn.execute(
            "DELETE FROM cameras WHERE camera_id = ?", (camera_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    # --- links -----------------------------------------------------------
    def save_link(self, link: CameraLink) -> None:
        """Upsert a directed link; both endpoint cameras must already exist."""
        for cid in (link.from_camera_id, link.to_camera_id):
            if self.get_camera(cid) is None:
                raise CameraNotFoundError(
                    f"cannot store link {link.key}: camera '{cid}' does not exist"
                )
        try:
            self._conn.execute(
                """
                INSERT INTO camera_links (
                    from_camera_id, to_camera_id, distance_m, road_name,
                    travel_direction
                ) VALUES (?,?,?,?,?)
                ON CONFLICT(from_camera_id, to_camera_id) DO UPDATE SET
                    distance_m=excluded.distance_m, road_name=excluded.road_name,
                    travel_direction=excluded.travel_direction
                """,
                (
                    link.from_camera_id, link.to_camera_id, link.distance_m,
                    link.road_name, link.travel_direction,
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError as exc:  # pragma: no cover - defensive
            raise InvalidCameraLinkError(str(exc)) from exc

    def get_link(self, from_camera_id: str, to_camera_id: str) -> Optional[CameraLink]:
        row = self._conn.execute(
            "SELECT * FROM camera_links "
            "WHERE from_camera_id = ? AND to_camera_id = ?",
            (from_camera_id, to_camera_id),
        ).fetchone()
        return self._row_to_link(row) if row is not None else None

    def list_links(self) -> list:
        rows = self._conn.execute(
            "SELECT * FROM camera_links ORDER BY from_camera_id, to_camera_id"
        ).fetchall()
        return [self._row_to_link(r) for r in rows]

    def delete_link(self, from_camera_id: str, to_camera_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM camera_links "
            "WHERE from_camera_id = ? AND to_camera_id = ?",
            (from_camera_id, to_camera_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        self._conn.close()

    # --- row mapping -----------------------------------------------------
    @staticmethod
    def _row_to_camera(row) -> Camera:
        return Camera(
            camera_id=row["camera_id"], name=row["name"],
            latitude=row["latitude"], longitude=row["longitude"],
            road_name=row["road_name"], heading_deg=row["heading_deg"],
            zone=row["zone"], enabled=bool(row["enabled"]),
        )

    @staticmethod
    def _row_to_link(row) -> CameraLink:
        return CameraLink(
            from_camera_id=row["from_camera_id"], to_camera_id=row["to_camera_id"],
            distance_m=row["distance_m"], road_name=row["road_name"],
            travel_direction=row["travel_direction"],
        )
