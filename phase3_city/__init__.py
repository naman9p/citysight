"""Phase 3 appearance-evidence foundations for CitySight."""

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

__all__ = [
    "AppearanceEmbeddingValidationError",
    "AppearanceEncoderError",
    "AppearanceEvidenceUnavailableError",
    "AppearancePreprocessingConfig",
    "LocalTorchScriptAppearanceEncoder",
    "VehicleAppearanceEmbedding",
    "VehicleAppearanceEncoder",
    "build_vehicle_appearance_encoder",
    "preprocess_vehicle_crop",
]
