"""Parallel deterministic Phase 3 hybrid candidate matching (Step 37).

The historical Phase 2 matcher remains the sole implementation of topology,
travel feasibility, and fingerprint field comparison.  This module reuses
those public results, adds a stricter trusted-plate gate, and attaches optional
Step 36 appearance evidence without changing Phase 2 decisions or collection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from phase2_city.cross_camera_matching import (
    CrossCameraCandidateMatcher,
    CrossCameraCandidateResult,
    CrossCameraMatchPolicy,
    FingerprintObservation,
)
from phase2_city.fingerprint import (
    FingerprintComparison,
    FingerprintFieldComparison,
    VehicleFingerprintScorer,
)
from phase2_city.graph import CityCameraGraph
from phase2_city.models import CameraLink
from phase3_city.appearance_comparison import (
    AppearanceComparisonResult,
    compare_appearance_embeddings,
)
from phase3_city.appearance_embedding import VehicleAppearanceEmbedding


_ATTRIBUTE_FIELDS = frozenset({"vehicle_colour", "vehicle_class"})


class HybridCandidateStatus(str, Enum):
    """Gate/eligibility outcome, not a same-vehicle classification."""

    INELIGIBLE = "ineligible"
    PHYSICALLY_IMPOSSIBLE = "physically_impossible"
    TRUSTED_PLATE_CONTRADICTION = "trusted_plate_contradiction"
    ELIGIBLE_WITH_EVIDENCE = "eligible_with_evidence"
    ELIGIBLE_WITHOUT_EVIDENCE = "eligible_without_evidence"


class PhysicalGateStatus(str, Enum):
    """Phase 2 physical result as consumed by the Phase 3 hard gate."""

    INELIGIBLE = "ineligible"
    IMPOSSIBLE = "impossible"
    FEASIBLE = "feasible"
    UNKNOWN = "unknown"


class TrustedPlateRelation(str, Enum):
    """Relation between accepted normalized plate evidence only."""

    AGREEMENT = "agreement"
    CONTRADICTION = "contradiction"
    SOURCE_UNAVAILABLE = "source_unavailable"
    CANDIDATE_UNAVAILABLE = "candidate_unavailable"
    BOTH_UNAVAILABLE = "both_unavailable"


@dataclass(frozen=True)
class PhysicalFeasibilityEvidence:
    """Immutable projection of the existing Phase 2 physical evaluation."""

    gate_status: PhysicalGateStatus
    topology_status: str
    link: Optional[CameraLink]
    elapsed_seconds: Optional[float]
    minimum_travel_seconds: Optional[float]
    travel_status: str
    reason: str

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
            "gate_status": self.gate_status.value,
            "topology_status": self.topology_status,
            "link": link,
            "elapsed_seconds": self.elapsed_seconds,
            "minimum_travel_seconds": self.minimum_travel_seconds,
            "travel_status": self.travel_status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TrustedPlateEvidence:
    """Accepted normalized plate evidence and its authoritative relation."""

    relation: TrustedPlateRelation
    source_trusted: bool
    candidate_trusted: bool
    source_plate: Optional[str]
    candidate_plate: Optional[str]

    def to_dict(self) -> dict:
        return {
            "relation": self.relation.value,
            "source_trusted": self.source_trusted,
            "candidate_trusted": self.candidate_trusted,
            "source_plate": self.source_plate,
            "candidate_plate": self.candidate_plate,
        }


@dataclass(frozen=True)
class AttributeEvidence:
    """Colour/class projection of the unchanged fingerprint comparison."""

    outcome: str
    evidence_coverage: float
    matched_weight: float
    contradicted_weight: float
    net_evidence: float
    field_comparisons: Tuple[FingerprintFieldComparison, ...]

    @property
    def available(self) -> bool:
        return self.evidence_coverage > 0.0

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "available": self.available,
            "evidence_coverage": self.evidence_coverage,
            "matched_weight": self.matched_weight,
            "contradicted_weight": self.contradicted_weight,
            "net_evidence": self.net_evidence,
            "field_comparisons": [
                comparison.to_dict()
                for comparison in self.field_comparisons
            ],
        }


@dataclass(frozen=True)
class HybridEvidenceComponent:
    """One auditable component; values are not summed or probabilities."""

    component: str
    state: str
    available: bool
    evidence_value: Optional[float]
    hard_gate: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "state": self.state,
            "available": self.available,
            "evidence_value": self.evidence_value,
            "hard_gate": self.hard_gate,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HybridCandidateMatchResult:
    """Explainable Phase 3 result with strict hard-gate authority."""

    source: FingerprintObservation
    candidate: FingerprintObservation
    status: HybridCandidateStatus
    physical_feasibility: PhysicalFeasibilityEvidence
    trusted_plate_evidence: TrustedPlateEvidence
    fingerprint_comparison: FingerprintComparison
    attribute_evidence: AttributeEvidence
    appearance_comparison: AppearanceComparisonResult
    components: Tuple[HybridEvidenceComponent, ...]
    hard_gate_reason: Optional[str]
    reasons: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "source": self.source.to_dict(),
            "candidate": self.candidate.to_dict(),
            "status": self.status.value,
            "physical_feasibility": self.physical_feasibility.to_dict(),
            "trusted_plate_evidence": self.trusted_plate_evidence.to_dict(),
            "fingerprint_comparison": self.fingerprint_comparison.to_dict(),
            "attribute_evidence": self.attribute_evidence.to_dict(),
            "appearance_comparison": self.appearance_comparison.to_dict(),
            "components": [component.to_dict()
                           for component in self.components],
            "hard_gate_reason": self.hard_gate_reason,
            "reasons": list(self.reasons),
        }


class Phase3HybridCandidateMatcher:
    """Pairwise opt-in matcher parallel to the frozen Phase 2 matcher."""

    def __init__(
            self,
            city_graph: CityCameraGraph,
            physical_policy: CrossCameraMatchPolicy,
            fingerprint_scorer: Optional[VehicleFingerprintScorer] = None,
    ):
        # Construction through the public Phase 2 matcher deliberately reuses
        # its dependency validation and exact physical/fingerprint semantics.
        self._phase2_matcher = CrossCameraCandidateMatcher(
            city_graph,
            physical_policy,
            fingerprint_scorer=fingerprint_scorer,
        )

    @property
    def physical_policy(self) -> CrossCameraMatchPolicy:
        return self._phase2_matcher.policy

    def compare(
            self,
            source: FingerprintObservation,
            candidate: FingerprintObservation,
            source_appearance: Optional[VehicleAppearanceEmbedding] = None,
            candidate_appearance: Optional[VehicleAppearanceEmbedding] = None,
    ) -> HybridCandidateMatchResult:
        """Return a hybrid evidence result without search, I/O, or inference."""
        phase2_result = self._phase2_matcher.compare(source, candidate)
        physical = self._physical_evidence(phase2_result)
        trusted_plate = self._trusted_plate_evidence(source, candidate)
        attributes = self._attribute_evidence(
            phase2_result.fingerprint_comparison)
        appearance = compare_appearance_embeddings(
            source_appearance, candidate_appearance)
        components = self._components(
            trusted_plate, attributes, appearance)
        status, hard_gate_reason = self._status(
            physical, trusted_plate, components)
        reasons = self._reasons(
            status, physical, trusted_plate, attributes, appearance,
            hard_gate_reason)

        return HybridCandidateMatchResult(
            source=source,
            candidate=candidate,
            status=status,
            physical_feasibility=physical,
            trusted_plate_evidence=trusted_plate,
            fingerprint_comparison=phase2_result.fingerprint_comparison,
            attribute_evidence=attributes,
            appearance_comparison=appearance,
            components=components,
            hard_gate_reason=hard_gate_reason,
            reasons=reasons,
        )

    @staticmethod
    def _physical_evidence(
            phase2_result: CrossCameraCandidateResult,
    ) -> PhysicalFeasibilityEvidence:
        if phase2_result.decision == "ineligible":
            gate_status = PhysicalGateStatus.INELIGIBLE
            reason = phase2_result.reasons[-1]
        elif phase2_result.decision == "physically_impossible":
            gate_status = PhysicalGateStatus.IMPOSSIBLE
            reason = phase2_result.reasons[-1]
        elif phase2_result.travel_status == "feasible":
            gate_status = PhysicalGateStatus.FEASIBLE
            reason = phase2_result.reasons[2]
        else:
            gate_status = PhysicalGateStatus.UNKNOWN
            reason = phase2_result.reasons[2]
        return PhysicalFeasibilityEvidence(
            gate_status=gate_status,
            topology_status=phase2_result.topology_status,
            link=phase2_result.link,
            elapsed_seconds=phase2_result.elapsed_seconds,
            minimum_travel_seconds=phase2_result.minimum_travel_seconds,
            travel_status=phase2_result.travel_status,
            reason=reason,
        )

    @staticmethod
    def _trusted_plate(observation: FingerprintObservation) -> Optional[str]:
        if observation.observation_status != "accepted":
            return None
        return observation.fingerprint.normalized_plate

    @classmethod
    def _trusted_plate_evidence(
            cls,
            source: FingerprintObservation,
            candidate: FingerprintObservation,
    ) -> TrustedPlateEvidence:
        source_plate = cls._trusted_plate(source)
        candidate_plate = cls._trusted_plate(candidate)
        if source_plate is None and candidate_plate is None:
            relation = TrustedPlateRelation.BOTH_UNAVAILABLE
        elif source_plate is None:
            relation = TrustedPlateRelation.SOURCE_UNAVAILABLE
        elif candidate_plate is None:
            relation = TrustedPlateRelation.CANDIDATE_UNAVAILABLE
        elif source_plate == candidate_plate:
            relation = TrustedPlateRelation.AGREEMENT
        else:
            relation = TrustedPlateRelation.CONTRADICTION
        return TrustedPlateEvidence(
            relation=relation,
            source_trusted=source_plate is not None,
            candidate_trusted=candidate_plate is not None,
            source_plate=source_plate,
            candidate_plate=candidate_plate,
        )

    @staticmethod
    def _attribute_evidence(
            fingerprint: FingerprintComparison,
    ) -> AttributeEvidence:
        details = tuple(
            comparison for comparison in fingerprint.field_comparisons
            if comparison.field in _ATTRIBUTE_FIELDS
        )
        comparable = tuple(
            comparison for comparison in details
            if comparison.vote != 0
        )
        evidence_coverage = math.fsum(
            comparison.weight for comparison in comparable)
        matched_weight = math.fsum(
            comparison.weight for comparison in comparable
            if comparison.vote > 0)
        contradicted_weight = math.fsum(
            comparison.weight for comparison in comparable
            if comparison.vote < 0)
        net_evidence = math.fsum(
            comparison.contribution for comparison in details)
        if not comparable or evidence_coverage == 0.0:
            outcome = "insufficient_evidence"
        elif matched_weight > 0.0 and contradicted_weight == 0.0:
            outcome = "consistent"
        elif contradicted_weight > 0.0 and matched_weight == 0.0:
            outcome = "contradictory"
        else:
            outcome = "mixed"
        return AttributeEvidence(
            outcome=outcome,
            evidence_coverage=evidence_coverage,
            matched_weight=matched_weight,
            contradicted_weight=contradicted_weight,
            net_evidence=net_evidence,
            field_comparisons=details,
        )

    @staticmethod
    def _components(
            plate: TrustedPlateEvidence,
            attributes: AttributeEvidence,
            appearance: AppearanceComparisonResult,
    ) -> Tuple[HybridEvidenceComponent, ...]:
        if plate.relation is TrustedPlateRelation.AGREEMENT:
            plate_value = 1.0
            plate_available = True
        elif plate.relation is TrustedPlateRelation.CONTRADICTION:
            plate_value = -1.0
            plate_available = True
        else:
            plate_value = None
            plate_available = False
        return (
            HybridEvidenceComponent(
                component="trusted_plate",
                state=plate.relation.value,
                available=plate_available,
                evidence_value=plate_value,
                hard_gate=(
                    plate.relation is TrustedPlateRelation.CONTRADICTION),
                reason=(
                    "accepted normalized plates agree"
                    if plate.relation is TrustedPlateRelation.AGREEMENT
                    else "accepted normalized plates contradict"
                    if plate.relation is TrustedPlateRelation.CONTRADICTION
                    else "trusted plate evidence is incomplete"
                ),
            ),
            HybridEvidenceComponent(
                component="vehicle_attributes",
                state=attributes.outcome,
                available=attributes.available,
                evidence_value=(
                    attributes.net_evidence
                    if attributes.available else None),
                hard_gate=False,
                reason=(
                    "colour/class field evidence uses unchanged Phase 2 "
                    "weights and missing-evidence semantics"),
            ),
            HybridEvidenceComponent(
                component="appearance",
                state=appearance.status.value,
                available=appearance.available,
                evidence_value=appearance.cosine_similarity,
                hard_gate=False,
                reason=(
                    "compatible cosine appearance evidence"
                    if appearance.available
                    else "appearance evidence is neutral because comparison "
                    "is unavailable"),
            ),
        )

    @staticmethod
    def _status(
            physical: PhysicalFeasibilityEvidence,
            plate: TrustedPlateEvidence,
            components: Tuple[HybridEvidenceComponent, ...],
    ) -> tuple[HybridCandidateStatus, Optional[str]]:
        if physical.gate_status is PhysicalGateStatus.INELIGIBLE:
            return HybridCandidateStatus.INELIGIBLE, physical.reason
        if physical.gate_status is PhysicalGateStatus.IMPOSSIBLE:
            return (
                HybridCandidateStatus.PHYSICALLY_IMPOSSIBLE,
                physical.reason,
            )
        if plate.relation is TrustedPlateRelation.CONTRADICTION:
            return (
                HybridCandidateStatus.TRUSTED_PLATE_CONTRADICTION,
                "trusted_plate_contradiction: accepted normalized plates "
                "differ",
            )
        if any(component.available for component in components):
            return HybridCandidateStatus.ELIGIBLE_WITH_EVIDENCE, None
        return HybridCandidateStatus.ELIGIBLE_WITHOUT_EVIDENCE, None

    @staticmethod
    def _reasons(
            status: HybridCandidateStatus,
            physical: PhysicalFeasibilityEvidence,
            plate: TrustedPlateEvidence,
            attributes: AttributeEvidence,
            appearance: AppearanceComparisonResult,
            hard_gate_reason: Optional[str],
    ) -> Tuple[str, ...]:
        similarity = (
            "none" if appearance.cosine_similarity is None
            else f"{appearance.cosine_similarity:.12f}"
        )
        reasons = [
            (f"physical: gate={physical.gate_status.value} "
             f"topology={physical.topology_status} "
             f"travel={physical.travel_status}"),
            (f"trusted_plate: relation={plate.relation.value} "
             f"source_trusted={str(plate.source_trusted).lower()} "
             f"candidate_trusted={str(plate.candidate_trusted).lower()}"),
            (f"attributes: outcome={attributes.outcome} "
             f"coverage={attributes.evidence_coverage:.12f} "
             f"net={attributes.net_evidence:.12f}"),
            (f"appearance: status={appearance.status.value} "
             f"cosine={similarity}"),
        ]
        if hard_gate_reason is not None:
            reasons.append(f"hard_gate: {hard_gate_reason}")
        reasons.append(f"hybrid_status: {status.value}")
        return tuple(reasons)


__all__ = [
    "AttributeEvidence",
    "HybridCandidateMatchResult",
    "HybridCandidateStatus",
    "HybridEvidenceComponent",
    "Phase3HybridCandidateMatcher",
    "PhysicalFeasibilityEvidence",
    "PhysicalGateStatus",
    "TrustedPlateEvidence",
    "TrustedPlateRelation",
]
