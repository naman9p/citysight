"""Phase 3 appearance-evidence foundations for CitySight."""

from phase3_city.appearance_comparison import (
    AppearanceComparisonResult,
    AppearanceComparisonStatus,
    AppearanceComparisonValidationError,
    AppearanceEmbeddingProvenance,
    compare_appearance_embeddings,
)
from phase3_city.appearance_embedding import (
    AppearanceEmbeddingValidationError,
    AppearanceEncoderError,
    AppearanceEvidenceUnavailableError,
    AppearancePreprocessingConfig,
    LocalTorchScriptAppearanceEncoder,
    VehicleAppearanceEmbedding,
    VehicleAppearanceEncoder,
    build_vehicle_appearance_encoder,
    preprocess_vehicle_crop,
)


_HYBRID_EXPORTS = frozenset({
    "AttributeEvidence",
    "HybridCandidateMatchResult",
    "HybridCandidateStatus",
    "HybridEvidenceComponent",
    "Phase3HybridCandidateMatcher",
    "PhysicalFeasibilityEvidence",
    "PhysicalGateStatus",
    "TrustedPlateEvidence",
    "TrustedPlateRelation",
})


def __getattr__(name):
    """Lazy-load Step 37 without coupling Step 35/36 imports to Phase 2."""
    if name not in _HYBRID_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from phase3_city import hybrid_matching
    value = getattr(hybrid_matching, name)
    globals()[name] = value
    return value

__all__ = [
    "AppearanceComparisonResult",
    "AppearanceComparisonStatus",
    "AppearanceComparisonValidationError",
    "AppearanceEmbeddingProvenance",
    "AppearanceEmbeddingValidationError",
    "AppearanceEncoderError",
    "AppearanceEvidenceUnavailableError",
    "AppearancePreprocessingConfig",
    "AttributeEvidence",
    "HybridCandidateMatchResult",
    "HybridCandidateStatus",
    "HybridEvidenceComponent",
    "LocalTorchScriptAppearanceEncoder",
    "Phase3HybridCandidateMatcher",
    "PhysicalFeasibilityEvidence",
    "PhysicalGateStatus",
    "TrustedPlateEvidence",
    "TrustedPlateRelation",
    "VehicleAppearanceEmbedding",
    "VehicleAppearanceEncoder",
    "build_vehicle_appearance_encoder",
    "compare_appearance_embeddings",
    "preprocess_vehicle_crop",
]
