"""Deterministic, explainable appearance-evidence comparison (Step 36).

This module compares two Step 35 embeddings only when they belong to the same
learned representation space.  It produces appearance evidence, never an
identity probability, threshold decision, candidate rank, or match label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Optional

from phase3_city.appearance_embedding import VehicleAppearanceEmbedding


_SHA256_LENGTH = 64
_COSINE_DRIFT_TOLERANCE = 1e-12
_INCOMPATIBLE_PROVENANCE_REASON = "incompatible_provenance"
_PROVENANCE_FIELDS = (
    "dimension",
    "model_id",
    "model_version",
    "weights_sha256",
    "preprocessing_sha256",
)


class AppearanceComparisonValidationError(ValueError):
    """Raised when a comparison input or result violates its contract."""


class AppearanceComparisonStatus(str, Enum):
    """Availability state; unavailable states carry no numeric evidence."""

    AVAILABLE = "available"
    SOURCE_UNAVAILABLE = "source_unavailable"
    CANDIDATE_UNAVAILABLE = "candidate_unavailable"
    BOTH_UNAVAILABLE = "both_unavailable"
    INCOMPATIBLE_PROVENANCE = "incompatible_provenance"


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise AppearanceComparisonValidationError(
            f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise AppearanceComparisonValidationError(
            f"{name} must be a positive integer")
    return result


def _required_text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AppearanceComparisonValidationError(
            f"{name} must be a non-empty string")
    return value.strip()


def _sha256_text(value, name: str) -> str:
    text = _required_text(value, name).lower()
    if len(text) != _SHA256_LENGTH or any(
            character not in "0123456789abcdef" for character in text):
        raise AppearanceComparisonValidationError(
            f"{name} must be a 64-character hexadecimal SHA-256")
    return text


def _optional_finite_number(value, name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AppearanceComparisonValidationError(
            f"{name} must be a finite number or None")
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise AppearanceComparisonValidationError(
            f"{name} must be a finite number or None") from exc
    if not math.isfinite(numeric):
        raise AppearanceComparisonValidationError(
            f"{name} must be finite")
    return numeric


@dataclass(frozen=True)
class AppearanceEmbeddingProvenance:
    """Immutable representation-space identity without embedding values."""

    dimension: int
    model_id: str
    model_version: str
    weights_sha256: str
    preprocessing_sha256: str

    def __post_init__(self):
        object.__setattr__(self, "dimension", _positive_integer(
            self.dimension, "dimension"))
        object.__setattr__(self, "model_id", _required_text(
            self.model_id, "model_id"))
        object.__setattr__(self, "model_version", _required_text(
            self.model_version, "model_version"))
        object.__setattr__(self, "weights_sha256", _sha256_text(
            self.weights_sha256, "weights_sha256"))
        object.__setattr__(self, "preprocessing_sha256", _sha256_text(
            self.preprocessing_sha256, "preprocessing_sha256"))

    @classmethod
    def from_embedding(
            cls,
            embedding: VehicleAppearanceEmbedding,
    ) -> "AppearanceEmbeddingProvenance":
        if not isinstance(embedding, VehicleAppearanceEmbedding):
            raise AppearanceComparisonValidationError(
                "embedding must be a VehicleAppearanceEmbedding")
        return cls(
            dimension=embedding.dimension,
            model_id=embedding.model_id,
            model_version=embedding.model_version,
            weights_sha256=embedding.weights_sha256,
            preprocessing_sha256=embedding.preprocessing_sha256,
        )

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "weights_sha256": self.weights_sha256,
            "preprocessing_sha256": self.preprocessing_sha256,
        }


@dataclass(frozen=True)
class AppearanceComparisonResult:
    """Immutable appearance-only evidence with explicit neutral absence."""

    status: AppearanceComparisonStatus
    cosine_similarity: Optional[float]
    euclidean_distance: Optional[float]
    embedding_dimension: Optional[int]
    provenance_compatible: Optional[bool]
    reason: Optional[str]
    incompatible_fields: tuple
    source_provenance: Optional[AppearanceEmbeddingProvenance]
    candidate_provenance: Optional[AppearanceEmbeddingProvenance]

    def __post_init__(self):
        if not isinstance(self.status, AppearanceComparisonStatus):
            raise AppearanceComparisonValidationError(
                "status must be an AppearanceComparisonStatus")
        cosine = _optional_finite_number(
            self.cosine_similarity, "cosine_similarity")
        distance = _optional_finite_number(
            self.euclidean_distance, "euclidean_distance")
        if self.embedding_dimension is None:
            dimension = None
        else:
            dimension = _positive_integer(
                self.embedding_dimension, "embedding_dimension")
        if self.provenance_compatible is not None and not isinstance(
                self.provenance_compatible, bool):
            raise AppearanceComparisonValidationError(
                "provenance_compatible must be true, false, or None")
        if self.reason is not None:
            reason = _required_text(self.reason, "reason")
        else:
            reason = None
        try:
            incompatible_fields = tuple(self.incompatible_fields)
        except TypeError as exc:
            raise AppearanceComparisonValidationError(
                "incompatible_fields must be an iterable of field names"
            ) from exc
        if any(not isinstance(field, str) or not field
               for field in incompatible_fields):
            raise AppearanceComparisonValidationError(
                "incompatible_fields must contain non-empty field names")
        for name, provenance in (
                ("source_provenance", self.source_provenance),
                ("candidate_provenance", self.candidate_provenance)):
            if provenance is not None and not isinstance(
                    provenance, AppearanceEmbeddingProvenance):
                raise AppearanceComparisonValidationError(
                    f"{name} must be AppearanceEmbeddingProvenance or None")

        if self.status is AppearanceComparisonStatus.AVAILABLE:
            self._validate_available(
                cosine, distance, dimension, reason, incompatible_fields)
        elif self.status is AppearanceComparisonStatus.INCOMPATIBLE_PROVENANCE:
            self._validate_incompatible(
                cosine, distance, dimension, reason, incompatible_fields)
        else:
            self._validate_missing(
                cosine, distance, dimension, reason, incompatible_fields)

        object.__setattr__(self, "cosine_similarity", cosine)
        object.__setattr__(self, "euclidean_distance", distance)
        object.__setattr__(self, "embedding_dimension", dimension)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "incompatible_fields", incompatible_fields)

    def _validate_available(
            self, cosine, distance, dimension, reason, incompatible_fields):
        if cosine is None or not -1.0 <= cosine <= 1.0:
            raise AppearanceComparisonValidationError(
                "available cosine_similarity must be within [-1, 1]")
        if distance is None or not 0.0 <= distance <= 2.0:
            raise AppearanceComparisonValidationError(
                "available euclidean_distance must be within [0, 2]")
        if dimension is None:
            raise AppearanceComparisonValidationError(
                "available comparison requires embedding_dimension")
        if self.provenance_compatible is not True:
            raise AppearanceComparisonValidationError(
                "available comparison requires compatible provenance")
        if reason is not None or incompatible_fields:
            raise AppearanceComparisonValidationError(
                "available comparison cannot have an unavailable reason")
        if self.source_provenance is None \
                or self.candidate_provenance is None:
            raise AppearanceComparisonValidationError(
                "available comparison requires both provenance records")
        if (self.source_provenance != self.candidate_provenance
                or self.source_provenance.dimension != dimension):
            raise AppearanceComparisonValidationError(
                "available comparison provenance must match exactly")
        expected_squared_distance = 2.0 - (2.0 * cosine)
        if not math.isclose(
                distance * distance,
                expected_squared_distance,
                rel_tol=1e-12,
                abs_tol=1e-12):
            raise AppearanceComparisonValidationError(
                "distance must satisfy the normalized-vector cosine relation")

    def _validate_incompatible(
            self, cosine, distance, dimension, reason, incompatible_fields):
        if cosine is not None or distance is not None or dimension is not None:
            raise AppearanceComparisonValidationError(
                "incompatible provenance cannot carry numeric evidence")
        if self.provenance_compatible is not False:
            raise AppearanceComparisonValidationError(
                "incompatible provenance requires provenance_compatible=false")
        if reason != _INCOMPATIBLE_PROVENANCE_REASON:
            raise AppearanceComparisonValidationError(
                "incompatible provenance requires its explicit reason")
        if not incompatible_fields:
            raise AppearanceComparisonValidationError(
                "incompatible provenance requires mismatch diagnostics")
        if self.source_provenance is None \
                or self.candidate_provenance is None:
            raise AppearanceComparisonValidationError(
                "incompatible comparison requires both provenance records")
        expected_fields = _incompatible_fields(
            self.source_provenance, self.candidate_provenance)
        if incompatible_fields != expected_fields:
            raise AppearanceComparisonValidationError(
                "incompatible_fields must exactly describe provenance "
                "differences in canonical order")

    def _validate_missing(
            self, cosine, distance, dimension, reason, incompatible_fields):
        if cosine is not None or distance is not None or dimension is not None:
            raise AppearanceComparisonValidationError(
                "missing evidence cannot carry numeric evidence")
        if self.provenance_compatible is not None:
            raise AppearanceComparisonValidationError(
                "missing evidence has unknown provenance compatibility")
        if reason != self.status.value:
            raise AppearanceComparisonValidationError(
                "missing evidence reason must match its status")
        if incompatible_fields:
            raise AppearanceComparisonValidationError(
                "missing evidence cannot carry incompatibility diagnostics")
        expected_presence = {
            AppearanceComparisonStatus.SOURCE_UNAVAILABLE: (False, True),
            AppearanceComparisonStatus.CANDIDATE_UNAVAILABLE: (True, False),
            AppearanceComparisonStatus.BOTH_UNAVAILABLE: (False, False),
        }
        source_present, candidate_present = expected_presence[self.status]
        if ((self.source_provenance is not None) != source_present
                or (self.candidate_provenance is not None)
                != candidate_present):
            raise AppearanceComparisonValidationError(
                "missing-evidence provenance presence is inconsistent")

    @property
    def available(self) -> bool:
        return self.status is AppearanceComparisonStatus.AVAILABLE

    def to_dict(self) -> dict:
        """Return a stable serialization without identity semantics."""
        return {
            "status": self.status.value,
            "available": self.available,
            "cosine_similarity": self.cosine_similarity,
            "euclidean_distance": self.euclidean_distance,
            "embedding_dimension": self.embedding_dimension,
            "provenance_compatible": self.provenance_compatible,
            "reason": self.reason,
            "incompatible_fields": list(self.incompatible_fields),
            "source_provenance": (
                None if self.source_provenance is None
                else self.source_provenance.to_dict()
            ),
            "candidate_provenance": (
                None if self.candidate_provenance is None
                else self.candidate_provenance.to_dict()
            ),
        }


def _result_for_missing(
        source: Optional[VehicleAppearanceEmbedding],
        candidate: Optional[VehicleAppearanceEmbedding],
) -> AppearanceComparisonResult:
    source_provenance = (
        None if source is None
        else AppearanceEmbeddingProvenance.from_embedding(source)
    )
    candidate_provenance = (
        None if candidate is None
        else AppearanceEmbeddingProvenance.from_embedding(candidate)
    )
    if source is None and candidate is None:
        status = AppearanceComparisonStatus.BOTH_UNAVAILABLE
    elif source is None:
        status = AppearanceComparisonStatus.SOURCE_UNAVAILABLE
    else:
        status = AppearanceComparisonStatus.CANDIDATE_UNAVAILABLE
    return AppearanceComparisonResult(
        status=status,
        cosine_similarity=None,
        euclidean_distance=None,
        embedding_dimension=None,
        provenance_compatible=None,
        reason=status.value,
        incompatible_fields=(),
        source_provenance=source_provenance,
        candidate_provenance=candidate_provenance,
    )


def _incompatible_fields(
        source: AppearanceEmbeddingProvenance,
        candidate: AppearanceEmbeddingProvenance,
) -> tuple:
    return tuple(
        field for field in _PROVENANCE_FIELDS
        if getattr(source, field) != getattr(candidate, field)
    )


def _bounded_cosine(value: float) -> float:
    if not math.isfinite(value):
        raise AppearanceComparisonValidationError(
            "cosine similarity calculation must be finite")
    if value > 1.0:
        if value <= 1.0 + _COSINE_DRIFT_TOLERANCE:
            return 1.0
        raise AppearanceComparisonValidationError(
            "cosine similarity exceeded 1 beyond floating-point tolerance")
    if value < -1.0:
        if value >= -1.0 - _COSINE_DRIFT_TOLERANCE:
            return -1.0
        raise AppearanceComparisonValidationError(
            "cosine similarity fell below -1 beyond floating-point tolerance")
    return value


def compare_appearance_embeddings(
        source: Optional[VehicleAppearanceEmbedding],
        candidate: Optional[VehicleAppearanceEmbedding],
) -> AppearanceComparisonResult:
    """Compare compatible normalized embeddings without making a decision.

    Step 35 guarantees unit L2 norm, so cosine similarity is the dot product:
    ``cos(source, candidate) = source.vector · candidate.vector``.  The
    Euclidean diagnostic follows ``distance² = 2 - 2 * cosine``.
    """
    if source is not None and not isinstance(
            source, VehicleAppearanceEmbedding):
        raise AppearanceComparisonValidationError(
            "source must be VehicleAppearanceEmbedding or None")
    if candidate is not None and not isinstance(
            candidate, VehicleAppearanceEmbedding):
        raise AppearanceComparisonValidationError(
            "candidate must be VehicleAppearanceEmbedding or None")
    if source is None or candidate is None:
        return _result_for_missing(source, candidate)

    source_provenance = AppearanceEmbeddingProvenance.from_embedding(source)
    candidate_provenance = AppearanceEmbeddingProvenance.from_embedding(
        candidate)
    incompatible_fields = _incompatible_fields(
        source_provenance, candidate_provenance)
    if incompatible_fields:
        return AppearanceComparisonResult(
            status=AppearanceComparisonStatus.INCOMPATIBLE_PROVENANCE,
            cosine_similarity=None,
            euclidean_distance=None,
            embedding_dimension=None,
            provenance_compatible=False,
            reason=_INCOMPATIBLE_PROVENANCE_REASON,
            incompatible_fields=incompatible_fields,
            source_provenance=source_provenance,
            candidate_provenance=candidate_provenance,
        )

    cosine = _bounded_cosine(math.fsum(
        left * right for left, right
        in zip(source.vector, candidate.vector)
    ))
    squared_distance = max(0.0, 2.0 - (2.0 * cosine))
    distance = math.sqrt(squared_distance)
    return AppearanceComparisonResult(
        status=AppearanceComparisonStatus.AVAILABLE,
        cosine_similarity=cosine,
        euclidean_distance=distance,
        embedding_dimension=source.dimension,
        provenance_compatible=True,
        reason=None,
        incompatible_fields=(),
        source_provenance=source_provenance,
        candidate_provenance=candidate_provenance,
    )


__all__ = [
    "AppearanceComparisonResult",
    "AppearanceComparisonStatus",
    "AppearanceComparisonValidationError",
    "AppearanceEmbeddingProvenance",
    "compare_appearance_embeddings",
]
