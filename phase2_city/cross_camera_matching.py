"""Explainable directional cross-camera candidate matching (Step 27).

This module combines the existing Step 23 fingerprint comparison with the
direct Step 17 camera topology and an explicit travel-speed policy.  It is a
pure pairwise primitive: no repository access, candidate search, persistence,
trajectory modification, or identity assertion occurs here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from typing import Literal, Optional, Tuple

from phase1_anpr.observation.observation_builder import (
    PlateObservation,
)
from phase1_anpr.pipeline.anpr_pipeline import parse_video_start_time
from phase2_city.fingerprint import (
    FingerprintComparison,
    FingerprintValidationError,
    VehicleFingerprint,
    VehicleFingerprintScorer,
)
from phase2_city.graph import CityCameraGraph
from phase2_city.models import CameraLink


_CANONICAL_OBSERVATION_STATUSES = frozenset({
    "accepted",
    "review",
    "abstained",
})

TopologyStatus = Literal[
    "direct_link",
    "reverse_only",
    "no_direct_link",
    "unknown_camera",
    "same_camera",
]
TravelStatus = Literal[
    "feasible",
    "too_fast",
    "unknown",
    "invalid_order",
    "not_applicable",
]
CandidateDecision = Literal[
    "ineligible",
    "physically_impossible",
    "insufficient_evidence",
    "identity_conflict",
    "possible_weak",
    "possible_strong",
]


class CrossCameraMatchValidationError(ValueError):
    """Raised when a Step 27 observation or policy is malformed."""


def _validate_text(value, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CrossCameraMatchValidationError(
            f"{field_name} must be a non-empty string")
    return value


def _validate_number(value, field_name: str, *, minimum: float,
                     minimum_inclusive: bool = True,
                     maximum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CrossCameraMatchValidationError(
            f"{field_name} must be a finite number")
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise CrossCameraMatchValidationError(
            f"{field_name} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise CrossCameraMatchValidationError(
            f"{field_name} must be finite")
    below_minimum = numeric < minimum if minimum_inclusive \
        else numeric <= minimum
    if below_minimum:
        operator = ">=" if minimum_inclusive else ">"
        raise CrossCameraMatchValidationError(
            f"{field_name} must be {operator} {minimum}")
    if maximum is not None and numeric > maximum:
        raise CrossCameraMatchValidationError(
            f"{field_name} must be <= {maximum}")
    return numeric


@dataclass(frozen=True)
class FingerprintObservation:
    """One observation's metadata and available vehicle identity evidence.

    ``timestamp=None`` explicitly represents unavailable time evidence.  A
    present timestamp is always normalized to timezone-aware UTC.
    """

    event_id: str
    camera_id: str
    timestamp: Optional[datetime]
    observation_status: str
    fingerprint: VehicleFingerprint

    def __post_init__(self):
        _validate_text(self.event_id, "event_id")
        _validate_text(self.camera_id, "camera_id")
        if (not isinstance(self.observation_status, str)
                or self.observation_status
                not in _CANONICAL_OBSERVATION_STATUSES):
            allowed = ", ".join(sorted(_CANONICAL_OBSERVATION_STATUSES))
            raise CrossCameraMatchValidationError(
                f"observation_status must be one of: {allowed}")
        if not isinstance(self.fingerprint, VehicleFingerprint):
            raise CrossCameraMatchValidationError(
                "fingerprint must be a VehicleFingerprint")

        moment = self.timestamp
        if moment is not None:
            if not isinstance(moment, datetime):
                raise CrossCameraMatchValidationError(
                    "timestamp must be a datetime or None")
            if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
                raise CrossCameraMatchValidationError(
                    "timestamp must include timezone information")
            object.__setattr__(self, "timestamp",
                               moment.astimezone(timezone.utc))

    @classmethod
    def from_plate_observation(
            cls,
            observation: PlateObservation,
            fingerprint: Optional[VehicleFingerprint] = None,
    ) -> "FingerprintObservation":
        """Adapt canonical observation metadata plus optional Step 26 evidence.

        When enrichment did not supply a fingerprint, only the plate evidence
        admitted by the existing Step 23 adapter is used.  This preserves
        accepted plate evidence without claiming that enrichment succeeded.
        """
        if not isinstance(observation, PlateObservation):
            raise CrossCameraMatchValidationError(
                "observation must be a PlateObservation")
        if fingerprint is not None and not isinstance(
                fingerprint, VehicleFingerprint):
            raise CrossCameraMatchValidationError(
                "fingerprint must be a VehicleFingerprint or None")

        try:
            moment = parse_video_start_time(observation.timestamp)
        except ValueError as exc:
            raise CrossCameraMatchValidationError(
                f"observation timestamp is invalid: {exc}") from exc

        available_fingerprint = fingerprint
        if available_fingerprint is None:
            try:
                available_fingerprint = \
                    VehicleFingerprint.from_plate_observation(observation)
            except FingerprintValidationError as exc:
                raise CrossCameraMatchValidationError(
                    f"observation fingerprint evidence is invalid: {exc}"
                ) from exc

        return cls(
            event_id=observation.event_id,
            camera_id=observation.camera_id,
            timestamp=moment,
            observation_status=observation.status,
            fingerprint=available_fingerprint,
        )

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "timestamp": (
                self.timestamp.isoformat()
                if self.timestamp is not None else None
            ),
            "observation_status": self.observation_status,
            "fingerprint": self.fingerprint.to_dict(),
        }


@dataclass(frozen=True)
class CrossCameraMatchPolicy:
    """Explicit heuristic thresholds for one directional comparison."""

    maximum_speed_kph: float
    travel_time_tolerance_seconds: float = 0.0
    minimum_evidence_coverage: float = 0.25
    strong_similarity_threshold: float = 0.70

    def __post_init__(self):
        maximum_speed_kph = _validate_number(
            self.maximum_speed_kph,
            "maximum_speed_kph",
            minimum=0.0,
            minimum_inclusive=False,
        )
        maximum_speed_mps = maximum_speed_kph / 3.6
        if not math.isfinite(maximum_speed_mps) or maximum_speed_mps <= 0.0:
            raise CrossCameraMatchValidationError(
                "maximum_speed_kph must convert to a finite value > 0 m/s")
        object.__setattr__(self, "maximum_speed_kph", maximum_speed_kph)
        object.__setattr__(self, "travel_time_tolerance_seconds",
                           _validate_number(
                               self.travel_time_tolerance_seconds,
                               "travel_time_tolerance_seconds",
                               minimum=0.0,
                           ))
        object.__setattr__(self, "minimum_evidence_coverage",
                           _validate_number(
                               self.minimum_evidence_coverage,
                               "minimum_evidence_coverage",
                               minimum=0.0,
                               maximum=1.0,
                           ))
        object.__setattr__(self, "strong_similarity_threshold",
                           _validate_number(
                               self.strong_similarity_threshold,
                               "strong_similarity_threshold",
                               minimum=0.0,
                               maximum=1.0,
                           ))

    def to_dict(self) -> dict:
        return {
            "maximum_speed_kph": self.maximum_speed_kph,
            "travel_time_tolerance_seconds":
                self.travel_time_tolerance_seconds,
            "minimum_evidence_coverage": self.minimum_evidence_coverage,
            "strong_similarity_threshold": self.strong_similarity_threshold,
        }


@dataclass(frozen=True)
class CrossCameraCandidateResult:
    """Auditable result for one directional source-to-candidate comparison."""

    source: FingerprintObservation
    candidate: FingerprintObservation
    fingerprint_comparison: FingerprintComparison
    topology_status: TopologyStatus
    link: Optional[CameraLink]
    elapsed_seconds: Optional[float]
    minimum_travel_seconds: Optional[float]
    travel_status: TravelStatus
    decision: CandidateDecision
    reasons: Tuple[str, ...]

    def to_dict(self) -> dict:
        link = None
        if self.link is not None:
            link = {
                "from_camera_id": self.link.from_camera_id,
                "to_camera_id": self.link.to_camera_id,
                "distance_m": self.link.distance_m,
                "road_name": self.link.road_name,
                "travel_direction": self.link.travel_direction,
            }
        return {
            "source": self.source.to_dict(),
            "candidate": self.candidate.to_dict(),
            "fingerprint_comparison": self.fingerprint_comparison.to_dict(),
            "topology_status": self.topology_status,
            "link": link,
            "elapsed_seconds": self.elapsed_seconds,
            "minimum_travel_seconds": self.minimum_travel_seconds,
            "travel_status": self.travel_status,
            "decision": self.decision,
            "reasons": list(self.reasons),
        }


class CrossCameraCandidateMatcher:
    """Compare two caller-supplied observations without searching or I/O."""

    def __init__(
            self,
            city_graph: CityCameraGraph,
            policy: CrossCameraMatchPolicy,
            fingerprint_scorer: Optional[VehicleFingerprintScorer] = None,
    ):
        if not isinstance(city_graph, CityCameraGraph):
            raise CrossCameraMatchValidationError(
                "city_graph must be a CityCameraGraph")
        if not isinstance(policy, CrossCameraMatchPolicy):
            raise CrossCameraMatchValidationError(
                "policy must be a CrossCameraMatchPolicy")
        if fingerprint_scorer is not None and not isinstance(
                fingerprint_scorer, VehicleFingerprintScorer):
            raise CrossCameraMatchValidationError(
                "fingerprint_scorer must be a VehicleFingerprintScorer or None")
        self.city_graph = city_graph
        self.policy = policy
        self.fingerprint_scorer = (
            fingerprint_scorer or VehicleFingerprintScorer())

    def compare(
            self,
            source: FingerprintObservation,
            candidate: FingerprintObservation,
    ) -> CrossCameraCandidateResult:
        """Return a directional, explainable candidate classification."""
        if not isinstance(source, FingerprintObservation):
            raise CrossCameraMatchValidationError(
                "source must be a FingerprintObservation")
        if not isinstance(candidate, FingerprintObservation):
            raise CrossCameraMatchValidationError(
                "candidate must be a FingerprintObservation")

        # Step 23 remains the sole fingerprint comparison implementation.  It
        # is intentionally evaluated even when a later physical gate wins.
        fingerprint_comparison = self.fingerprint_scorer.compare(
            source.fingerprint, candidate.fingerprint)
        reasons = [self._fingerprint_reason(fingerprint_comparison)]

        topology_status, link = self._classify_topology(source, candidate)
        reasons.append(self._topology_reason(
            source, candidate, topology_status, link))

        elapsed_seconds = self._elapsed_seconds(source, candidate)
        minimum_travel_seconds, travel_status = self._classify_travel(
            topology_status, link, elapsed_seconds)
        reasons.append(self._travel_reason(
            elapsed_seconds, minimum_travel_seconds, travel_status))

        decision, decision_reason = self._decide(
            source,
            candidate,
            fingerprint_comparison,
            topology_status,
            elapsed_seconds,
            travel_status,
        )
        reasons.append(decision_reason)

        return CrossCameraCandidateResult(
            source=source,
            candidate=candidate,
            fingerprint_comparison=fingerprint_comparison,
            topology_status=topology_status,
            link=link,
            elapsed_seconds=elapsed_seconds,
            minimum_travel_seconds=minimum_travel_seconds,
            travel_status=travel_status,
            decision=decision,
            reasons=tuple(reasons),
        )

    def _classify_topology(
            self,
            source: FingerprintObservation,
            candidate: FingerprintObservation,
    ) -> tuple[TopologyStatus, Optional[CameraLink]]:
        if source.camera_id == candidate.camera_id:
            return "same_camera", None
        if self.city_graph.get_camera(source.camera_id) is None \
                or self.city_graph.get_camera(candidate.camera_id) is None:
            return "unknown_camera", None

        forward_link = self.city_graph.get_link(
            source.camera_id, candidate.camera_id)
        if forward_link is not None:
            _validate_number(
                forward_link.distance_m,
                "forward link distance_m",
                minimum=0.0,
                minimum_inclusive=False,
            )
            return "direct_link", forward_link
        if self.city_graph.get_link(
                candidate.camera_id, source.camera_id) is not None:
            return "reverse_only", None
        return "no_direct_link", None

    @staticmethod
    def _elapsed_seconds(source, candidate) -> Optional[float]:
        if source.timestamp is None or candidate.timestamp is None:
            return None
        return (candidate.timestamp - source.timestamp).total_seconds()

    def _classify_travel(
            self,
            topology_status: TopologyStatus,
            link: Optional[CameraLink],
            elapsed_seconds: Optional[float],
    ) -> tuple[Optional[float], TravelStatus]:
        if topology_status == "same_camera":
            return None, "not_applicable"
        if elapsed_seconds is None:
            return None, "unknown"
        if elapsed_seconds < 0:
            return None, "invalid_order"
        if topology_status != "direct_link" or link is None:
            return None, "unknown"

        maximum_speed_mps = self.policy.maximum_speed_kph / 3.6
        minimum_travel_seconds = link.distance_m / maximum_speed_mps
        if (elapsed_seconds + self.policy.travel_time_tolerance_seconds
                < minimum_travel_seconds):
            return minimum_travel_seconds, "too_fast"
        return minimum_travel_seconds, "feasible"

    def _decide(
            self,
            source,
            candidate,
            comparison,
            topology_status,
            elapsed_seconds,
            travel_status,
    ) -> tuple[CandidateDecision, str]:
        if source.event_id == candidate.event_id:
            return "ineligible", "ineligible: source and candidate are the same event"
        if topology_status == "same_camera":
            return "ineligible", "ineligible: Step 27 compares different cameras"
        if elapsed_seconds is not None and elapsed_seconds < 0:
            return (
                "physically_impossible",
                "physically_impossible: candidate timestamp precedes source",
            )
        if travel_status == "too_fast":
            return (
                "physically_impossible",
                "physically_impossible: direct-link travel exceeds the "
                "configured speed ceiling",
            )

        if (comparison.overall_similarity is None
                or comparison.evidence_coverage
                < self.policy.minimum_evidence_coverage):
            return (
                "insufficient_evidence",
                "insufficient_evidence: comparable fingerprint coverage is "
                "below policy",
            )
        if comparison.outcome == "contradictory":
            return (
                "identity_conflict",
                "identity_conflict: all comparable fingerprint evidence "
                "contradicts",
            )
        if comparison.outcome == "mixed":
            return (
                "possible_weak",
                "possible_weak: fingerprint evidence is mixed",
            )
        if (comparison.outcome == "consistent"
                and comparison.overall_similarity
                >= self.policy.strong_similarity_threshold):
            return (
                "possible_strong",
                "possible_strong: consistent fingerprint evidence meets "
                "the strong-similarity policy",
            )
        return (
            "possible_weak",
            "possible_weak: consistent fingerprint evidence is below the "
            "strong-similarity policy",
        )

    @staticmethod
    def _fingerprint_reason(comparison: FingerprintComparison) -> str:
        similarity = ("none" if comparison.overall_similarity is None
                      else f"{comparison.overall_similarity:.6f}")
        return (f"fingerprint: outcome={comparison.outcome} "
                f"similarity={similarity} "
                f"coverage={comparison.evidence_coverage:.6f}")

    @staticmethod
    def _topology_reason(source, candidate, status, link) -> str:
        direction = f"{source.camera_id}->{candidate.camera_id}"
        if status == "direct_link":
            suffix = (f" distance_m={float(link.distance_m):.6f}"
                      f" road={link.road_name}")
            if link.travel_direction is not None:
                suffix += f" direction={link.travel_direction}"
            return f"topology: direct_link {direction}{suffix}"
        if status == "reverse_only":
            return (f"topology: reverse_only {direction}; reverse distance "
                    "is not used")
        if status == "no_direct_link":
            return f"topology: no_direct_link {direction}; route is unknown"
        if status == "unknown_camera":
            return f"topology: unknown_camera {direction}"
        return f"topology: same_camera {source.camera_id}"

    def _travel_reason(self, elapsed, minimum, status) -> str:
        if status == "feasible":
            return (f"travel: feasible elapsed={elapsed:.6f}s "
                    f"minimum={minimum:.6f}s "
                    f"maximum_speed_kph={self.policy.maximum_speed_kph:.6f} "
                    f"tolerance={self.policy.travel_time_tolerance_seconds:.6f}s")
        if status == "too_fast":
            return (f"travel: too_fast elapsed={elapsed:.6f}s "
                    f"minimum={minimum:.6f}s "
                    f"maximum_speed_kph={self.policy.maximum_speed_kph:.6f} "
                    f"tolerance={self.policy.travel_time_tolerance_seconds:.6f}s")
        if status == "invalid_order":
            return f"travel: invalid_order elapsed={elapsed:.6f}s"
        if status == "not_applicable":
            return "travel: not_applicable for a same-camera comparison"
        if elapsed is None:
            return "travel: unknown because a timestamp is unavailable"
        return (f"travel: unknown for topology without a forward direct link; "
                f"elapsed={elapsed:.6f}s")


__all__ = [
    "CrossCameraMatchValidationError",
    "FingerprintObservation",
    "CrossCameraMatchPolicy",
    "CrossCameraCandidateResult",
    "CrossCameraCandidateMatcher",
]
