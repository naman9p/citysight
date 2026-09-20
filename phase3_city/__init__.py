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
    "VehicleEvidenceQueryCursor",
    "VehicleEvidenceSummary",
    "VehicleEvidenceValidationError",
})

_RETRIEVAL_EXPORTS = frozenset({
    "HistoricalCandidate",
    "HistoricalCandidateRetrievalError",
    "HistoricalCandidateRetrievalPolicy",
    "HistoricalCandidateRetrievalResult",
    "HistoricalCandidateRetriever",
})

_QUERY_HISTORY_EXPORTS = frozenset({
    "CandidateQueryHistoryRecord",
    "CandidateQueryHistoryRepository",
    "CandidateQueryResultRecord",
    "HYBRID_SNAPSHOT_SCHEMA_VERSION",
    "HybridResultSnapshot",
    "PHYSICAL_POLICY_VERSION",
    "QUERY_HISTORY_IMPLEMENTATION_ID",
    "QUERY_HISTORY_SCHEMA_VERSION",
    "QueryHistoryConflictError",
    "QueryHistoryCorruptionError",
    "QueryHistoryPersistenceError",
    "QueryHistoryValidationError",
    "RETRIEVAL_ORDER_KIND",
    "RETRIEVAL_POLICY_VERSION",
    "SQLiteCandidateQueryHistoryRepository",
    "build_candidate_query_history",
})

_TRAJECTORY_HYPOTHESIS_EXPORTS = frozenset({
    "TRAJECTORY_HYPOTHESIS_IMPLEMENTATION_ID",
    "TRAJECTORY_HYPOTHESIS_POLICY_VERSION",
    "TRAJECTORY_HYPOTHESIS_SCHEMA_VERSION",
    "TrajectoryHistoryReference",
    "TrajectoryHypothesis",
    "TrajectoryHypothesisBuilder",
    "TrajectoryHypothesisDataError",
    "TrajectoryHypothesisEdge",
    "TrajectoryHypothesisError",
    "TrajectoryHypothesisEvent",
    "TrajectoryHypothesisPolicy",
    "TrajectoryHypothesisResult",
})

_API_EXPORTS = frozenset({
    "MAX_QUERY_IDS",
    "PHASE3_API_IMPLEMENTATION_ID",
    "PHASE3_API_SCHEMA_VERSION",
    "Phase3ApiError",
    "Phase3ReadApi",
    "open_phase3_read_api",
})


def __getattr__(name):
    """Lazy-load Phase 2-backed Step 37–42 modules on demand."""
    if name in _HYBRID_EXPORTS:
        from phase3_city import hybrid_matching as module
    elif name in _PERSISTENCE_EXPORTS:
        from phase3_city import evidence_persistence as module
    elif name in _RETRIEVAL_EXPORTS:
        from phase3_city import historical_candidate_retrieval as module
    elif name in _QUERY_HISTORY_EXPORTS:
        from phase3_city import query_history as module
    elif name in _TRAJECTORY_HYPOTHESIS_EXPORTS:
        from phase3_city import trajectory_hypotheses as module
    elif name in _API_EXPORTS:
        from phase3_city import api as module
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
    "CandidateQueryHistoryRecord",
    "CandidateQueryHistoryRepository",
    "CandidateQueryResultRecord",
    "EMBEDDING_SERIALIZATION_FORMAT",
    "EMBEDDING_SERIALIZATION_VERSION",
    "EMBEDDING_VECTOR_TOLERANCE",
    "HybridCandidateMatchResult",
    "HybridCandidateStatus",
    "HybridEvidenceComponent",
    "HistoricalCandidate",
    "HistoricalCandidateRetrievalError",
    "HistoricalCandidateRetrievalPolicy",
    "HistoricalCandidateRetrievalResult",
    "HistoricalCandidateRetriever",
    "HYBRID_SNAPSHOT_SCHEMA_VERSION",
    "HybridResultSnapshot",
    "LocalTorchScriptAppearanceEncoder",
    "MAX_QUERY_IDS",
    "PHASE3_API_IMPLEMENTATION_ID",
    "PHASE3_API_SCHEMA_VERSION",
    "Phase3ApiError",
    "Phase3HybridCandidateMatcher",
    "Phase3ReadApi",
    "PhysicalFeasibilityEvidence",
    "PhysicalGateStatus",
    "PHYSICAL_POLICY_VERSION",
    "QUERY_HISTORY_IMPLEMENTATION_ID",
    "QUERY_HISTORY_SCHEMA_VERSION",
    "QueryHistoryConflictError",
    "QueryHistoryCorruptionError",
    "QueryHistoryPersistenceError",
    "QueryHistoryValidationError",
    "RETRIEVAL_ORDER_KIND",
    "RETRIEVAL_POLICY_VERSION",
    "SQLiteVehicleEvidenceRepository",
    "SQLiteCandidateQueryHistoryRepository",
    "TrustedPlateEvidence",
    "TrustedPlateRelation",
    "TRAJECTORY_HYPOTHESIS_IMPLEMENTATION_ID",
    "TRAJECTORY_HYPOTHESIS_POLICY_VERSION",
    "TRAJECTORY_HYPOTHESIS_SCHEMA_VERSION",
    "TrajectoryHistoryReference",
    "TrajectoryHypothesis",
    "TrajectoryHypothesisBuilder",
    "TrajectoryHypothesisDataError",
    "TrajectoryHypothesisEdge",
    "TrajectoryHypothesisError",
    "TrajectoryHypothesisEvent",
    "TrajectoryHypothesisPolicy",
    "TrajectoryHypothesisResult",
    "VEHICLE_EVIDENCE_SCHEMA_VERSION",
    "VehicleEvidenceConflictError",
    "VehicleEvidenceCorruptionError",
    "VehicleEvidencePersistenceError",
    "VehicleEvidenceRecord",
    "VehicleEvidenceRepository",
    "VehicleEvidenceQueryCursor",
    "VehicleEvidenceSummary",
    "VehicleEvidenceValidationError",
    "VehicleAppearanceEmbedding",
    "VehicleAppearanceEncoder",
    "build_vehicle_appearance_encoder",
    "build_candidate_query_history",
    "compare_appearance_embeddings",
    "open_phase3_read_api",
    "preprocess_vehicle_crop",
]
