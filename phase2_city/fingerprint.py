"""Deterministic vehicle fingerprint comparison foundation (Step 23).

This module deliberately contains only value objects and pure comparison logic.
It does not extract vehicle attributes, perform I/O, persist fingerprints, or
decide that two observations belong to the same vehicle. Scores are heuristic
evidence summaries, not calibrated probabilities.
"""

import math
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from phase1_anpr.normalization.plate_normalizer import PlateNormalizer
from phase1_anpr.observation.observation_builder import PlateObservation


DEFAULT_FINGERPRINT_WEIGHTS = MappingProxyType({
    "normalized_plate": 0.60,
    "vehicle_colour": 0.25,
    "vehicle_class": 0.15,
})

_FINGERPRINT_FIELDS = tuple(DEFAULT_FINGERPRINT_WEIGHTS)
_PLATE_NORMALIZER = PlateNormalizer()


class FingerprintValidationError(ValueError):
    """Raised when fingerprint data or scorer weights are invalid."""


def _canonicalize_plate(value) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise FingerprintValidationError(
            "normalized_plate must be a string or None")
    return _PLATE_NORMALIZER.normalize_text(value) or None


def _canonicalize_label(value, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise FingerprintValidationError(
            f"{field_name} must be a string or None")
    normalized = " ".join(value.split()).lower()
    return normalized or None


@dataclass(frozen=True)
class VehicleFingerprint:
    """Optional canonical identity evidence for one observed vehicle."""

    normalized_plate: Optional[str] = None
    vehicle_colour: Optional[str] = None
    vehicle_class: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(
            self, "normalized_plate", _canonicalize_plate(self.normalized_plate))
        object.__setattr__(
            self, "vehicle_colour",
            _canonicalize_label(self.vehicle_colour, "vehicle_colour"))
        object.__setattr__(
            self, "vehicle_class",
            _canonicalize_label(self.vehicle_class, "vehicle_class"))

    @classmethod
    def from_plate_observation(cls, observation: PlateObservation):
        """Build the evidence currently available from a Phase 1 observation.

        Only an accepted normalized plate is sufficiently reliable to become
        fingerprint identity evidence. Phase 1 does not currently produce
        vehicle colour or class, so those fields remain unset.
        """
        if not isinstance(observation, PlateObservation):
            raise FingerprintValidationError(
                "observation must be a PlateObservation")
        plate = (observation.plate_normalized
                 if observation.status == "accepted" else None)
        return cls(normalized_plate=plate)

    def to_dict(self) -> dict:
        return {
            "normalized_plate": self.normalized_plate,
            "vehicle_colour": self.vehicle_colour,
            "vehicle_class": self.vehicle_class,
        }


@dataclass(frozen=True)
class FingerprintFieldComparison:
    """Comparison details for one fingerprint attribute."""

    field: str
    left_value: Optional[str]
    right_value: Optional[str]
    state: str                 # match | mismatch | missing
    weight: float
    vote: int                  # +1 | -1 | 0
    contribution: float        # normalized weight * vote

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "left_value": self.left_value,
            "right_value": self.right_value,
            "state": self.state,
            "weight": self.weight,
            "vote": self.vote,
            "contribution": self.contribution,
        }


@dataclass(frozen=True)
class FingerprintComparison:
    """Structured, auditable result of comparing two fingerprints."""

    overall_similarity: Optional[float]
    evidence_coverage: float
    matched_weight: float
    contradicted_weight: float
    net_evidence: float
    outcome: str
    field_comparisons: Tuple[FingerprintFieldComparison, ...]

    def to_dict(self) -> dict:
        return {
            "overall_similarity": self.overall_similarity,
            "evidence_coverage": self.evidence_coverage,
            "matched_weight": self.matched_weight,
            "contradicted_weight": self.contradicted_weight,
            "net_evidence": self.net_evidence,
            "outcome": self.outcome,
            "field_comparisons": [
                comparison.to_dict() for comparison in self.field_comparisons
            ],
        }


class VehicleFingerprintScorer:
    """Compare fingerprints with deterministic signed evidence weights."""

    def __init__(self, weights: Optional[Mapping[str, float]] = None):
        configured = dict(DEFAULT_FINGERPRINT_WEIGHTS)
        if weights is not None:
            if not isinstance(weights, Mapping):
                raise FingerprintValidationError("weights must be a mapping")
            unknown = set(weights) - set(_FINGERPRINT_FIELDS)
            if unknown:
                names = ", ".join(sorted(str(name) for name in unknown))
                raise FingerprintValidationError(f"unknown fingerprint weights: {names}")
            configured.update(weights)

        checked = {}
        for name in _FINGERPRINT_FIELDS:
            value = configured[name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise FingerprintValidationError(
                    f"weight for {name} must be a finite number")
            try:
                numeric = float(value)
            except (OverflowError, TypeError, ValueError) as exc:
                raise FingerprintValidationError(
                    f"weight for {name} must be a finite number") from exc
            if not math.isfinite(numeric):
                raise FingerprintValidationError(
                    f"weight for {name} must be finite")
            if numeric < 0:
                raise FingerprintValidationError(
                    f"weight for {name} must be >= 0")
            checked[name] = numeric

        max_weight = max(checked.values())
        if max_weight == 0:
            raise FingerprintValidationError(
                "at least one fingerprint weight must be > 0")
        scaled = {
            name: value / max_weight for name, value in checked.items()
        }
        scaled_total = sum(scaled.values())
        self.weights = MappingProxyType(
            {name: value / scaled_total for name, value in scaled.items()})

    def compare(self, left: VehicleFingerprint,
                right: VehicleFingerprint) -> FingerprintComparison:
        if not isinstance(left, VehicleFingerprint):
            raise FingerprintValidationError("left must be a VehicleFingerprint")
        if not isinstance(right, VehicleFingerprint):
            raise FingerprintValidationError("right must be a VehicleFingerprint")

        details = []
        matched_weight = 0.0
        contradicted_weight = 0.0

        for field_name in _FINGERPRINT_FIELDS:
            left_value = getattr(left, field_name)
            right_value = getattr(right, field_name)
            weight = self.weights[field_name]

            if left_value is None or right_value is None:
                state, vote = "missing", 0
            elif left_value == right_value:
                state, vote = "match", 1
                matched_weight += weight
            else:
                state, vote = "mismatch", -1
                contradicted_weight += weight

            details.append(FingerprintFieldComparison(
                field=field_name,
                left_value=left_value,
                right_value=right_value,
                state=state,
                weight=weight,
                vote=vote,
                contribution=weight * vote,
            ))

        evidence_coverage = sum(
            comparison.weight for comparison in details
            if comparison.vote != 0
        )
        net_evidence = sum(
            comparison.contribution for comparison in details
        )

        if evidence_coverage == 0:
            outcome = "insufficient_evidence"
            overall_similarity = None
        else:
            overall_similarity = round((net_evidence + 1.0) / 2.0, 6)
            if matched_weight > 0 and contradicted_weight == 0:
                outcome = "consistent"
            elif contradicted_weight > 0 and matched_weight == 0:
                outcome = "contradictory"
            else:
                outcome = "mixed"

        return FingerprintComparison(
            overall_similarity=overall_similarity,
            evidence_coverage=evidence_coverage,
            matched_weight=matched_weight,
            contradicted_weight=contradicted_weight,
            net_evidence=net_evidence,
            outcome=outcome,
            field_comparisons=tuple(details),
        )


__all__ = [
    "DEFAULT_FINGERPRINT_WEIGHTS",
    "FingerprintValidationError",
    "VehicleFingerprint",
    "FingerprintFieldComparison",
    "FingerprintComparison",
    "VehicleFingerprintScorer",
]
