"""Evidence storage abstraction for plate crops (Step 12).

Callers depend only on the `EvidenceStore` interface; the local filesystem
implementation here can later be swapped for MinIO/S3 without changing callers.
Only references/keys are meant to be persisted in the DB — never image bytes.

Keys are deterministic per observation (derived from `event_id`), so retrying
the same observation overwrites its own evidence instead of creating duplicates,
and the key is collision-safe as long as `event_id` is unique.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import re


class EvidenceError(Exception):
    """Raised when evidence cannot be stored (missing/invalid input)."""


# event_id is a UUID string in practice; be defensive and keep keys filesystem-
# and object-store-safe regardless of what is passed in.
_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def evidence_key(event_id: str, ext: str = "jpg") -> str:
    """Deterministic, collision-safe key for an observation's evidence.

    Same `event_id` always maps to the same key. `event_id` must be non-empty.
    """
    if not event_id:
        raise EvidenceError("event_id is required to build an evidence key")
    safe_id = _SAFE.sub("_", str(event_id))
    ext = ext.lstrip(".").lower() or "jpg"
    return f"{safe_id}.{ext}"


class EvidenceStore(ABC):
    """Store plate-crop evidence and return an opaque reference string.

    The reference is what gets persisted in the DB. Implementations decide what
    it means (a filesystem path, an S3 key, etc.).
    """

    @abstractmethod
    def store(self, event_id: str, source_path, ext: str = "jpg") -> str:
        """Store the evidence for `event_id` from `source_path`; return its reference.

        Raises EvidenceError if the source is missing or invalid.
        """

    @abstractmethod
    def exists(self, reference: str) -> bool:
        """Return True if the referenced evidence is present."""


class LocalFilesystemEvidenceStore(EvidenceStore):
    """Copies plate crops under a base directory keyed by observation.

    The stored reference is the path relative to `base_dir`, so the DB stays
    portable if the base directory moves.
    """

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)

    def store(self, event_id: str, source_path, ext: str = "jpg") -> str:
        if source_path is None:
            raise EvidenceError("source_path is required")
        src = Path(source_path)
        if not src.exists() or not src.is_file():
            raise EvidenceError(f"evidence source not found: {src}")

        # Preserve the source extension when present; fall back to `ext`.
        suffix = src.suffix.lstrip(".").lower() or ext
        key = evidence_key(event_id, suffix)
        dest = self.base_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Copy bytes so the original crop is preserved and unmodified.
        dest.write_bytes(src.read_bytes())
        return key

    def exists(self, reference: Optional[str]) -> bool:
        if not reference:
            return False
        return (self.base_dir / reference).is_file()

    def resolve(self, reference: str) -> Path:
        """Absolute path for a stored reference (convenience for readers)."""
        return self.base_dir / reference
