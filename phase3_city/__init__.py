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

_PERSISTENCE_EXPORTS = frozenset({
    "EMBEDDING_SERIALIZATION_FORMAT",
    "EMBEDDING_SERIALIZATION_VERSION",
    "EMBEDDING_VECTOR_TOLERANCE",
    "SQLiteVehicleEvidenceRepository",
    "VEHICLE_EVIDENCE_SCHEMA_VERSION",
    "VehicleEvidenceConflictError",
    "VehicleEvidenceCorruptionError",
    "VehicleEvidencePersistenceError",
    "VehicleEvidenceRecord",
    "VehicleEvidenceRepository",
    "VehicleEvidenceValidationError",
})


def __getattr__(name):
    """Lazy-load Phase 2-backed Step 37/38 modules on demand."""
    if name in _HYBRID_EXPORTS:
        from phase3_city import hybrid_matching as module
    elif name in _PERSISTENCE_EXPORTS:
        from phase3_city import evidence_persistence as module
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
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
    "EMBEDDING_SERIALIZATION_FORMAT",
    "EMBEDDING_SERIALIZATION_VERSION",
    "EMBEDDING_VECTOR_TOLERANCE",
    "HybridCandidateMatchResult",
    "HybridCandidateStatus",
    "HybridEvidenceComponent",
    "LocalTorchScriptAppearanceEncoder",
    "Phase3HybridCandidateMatcher",
    "PhysicalFeasibilityEvidence",
    "PhysicalGateStatus",
    "SQLiteVehicleEvidenceRepository",
    "TrustedPlateEvidence",
    "TrustedPlateRelation",
    "VEHICLE_EVIDENCE_SCHEMA_VERSION",
    "VehicleEvidenceConflictError",
    "VehicleEvidenceCorruptionError",
    "VehicleEvidencePersistenceError",
    "VehicleEvidenceRecord",
    "VehicleEvidenceRepository",
    "VehicleEvidenceValidationError",
    "VehicleAppearanceEmbedding",
    "VehicleAppearanceEncoder",
    "build_vehicle_appearance_encoder",
    "compare_appearance_embeddings",
    "preprocess_vehicle_crop",
]
