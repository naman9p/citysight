"""Persistence layer for canonical plate observations and evidence (Step 12)."""

from phase1_anpr.persistence.evidence_store import (
    EvidenceStore,
    LocalFilesystemEvidenceStore,
    EvidenceError,
)
from phase1_anpr.persistence.repository import (
    ObservationRepository,
    SQLiteObservationRepository,
)
from phase1_anpr.persistence.watchlist_repository import (
    SQLiteWatchlistRepository,
    WatchlistError,
)

__all__ = [
    "EvidenceStore",
    "LocalFilesystemEvidenceStore",
    "EvidenceError",
    "ObservationRepository",
    "SQLiteObservationRepository",
    "SQLiteWatchlistRepository",
    "WatchlistError",
]
