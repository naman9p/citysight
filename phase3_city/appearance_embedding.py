"""Model-neutral learned vehicle appearance embedding foundation (Step 35).

This module turns one already-associated full-vehicle BGR crop into immutable
appearance evidence.  It deliberately provides no similarity function,
threshold, candidate decision, persistence, search, or identity semantics.
PyTorch is imported only when the optional local TorchScript adapter first
loads a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


_SHA256_LENGTH = 64
_PREPROCESSING_SCHEMA = "citysight-appearance-preprocessing-v1"


class AppearanceEmbeddingValidationError(ValueError):
    """Raised when appearance evidence or preprocessing input is malformed."""


class AppearanceEncoderError(RuntimeError):
    """Raised when the optional learned encoder cannot be configured or run."""


class AppearanceEvidenceUnavailableError(AppearanceEncoderError):
    """Raised when an explicit operational reason leaves no appearance evidence."""

    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(message)


def _required_text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AppearanceEmbeddingValidationError(
            f"{name} must be a non-empty string")
    return value.strip()


def _sha256_text(value, name: str) -> str:
    text = _required_text(value, name).lower()
    if len(text) != _SHA256_LENGTH or any(
            character not in "0123456789abcdef" for character in text):
        raise AppearanceEmbeddingValidationError(
            f"{name} must be a 64-character hexadecimal SHA-256")
    return text


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise AppearanceEmbeddingValidationError(
            f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise AppearanceEmbeddingValidationError(
            f"{name} must be a positive integer")
    return result


def _three_numbers(value, name: str, *, positive: bool = False) -> tuple:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) \
            or len(value) != 3:
        raise AppearanceEmbeddingValidationError(
            f"{name} must contain exactly three finite numbers")
    result = []
    for part in value:
        if isinstance(part, bool) or not isinstance(part, Real):
            raise AppearanceEmbeddingValidationError(
                f"{name} must contain exactly three finite numbers")
        numeric = float(part)
        if not math.isfinite(numeric) or (positive and numeric <= 0.0):
            qualifier = " positive" if positive else ""
            raise AppearanceEmbeddingValidationError(
                f"{name} must contain exactly three finite{qualifier} numbers")
        result.append(numeric)
    return tuple(result)


def _normalized_vector(value) -> tuple:
    if isinstance(value, (str, bytes)):
        raise AppearanceEmbeddingValidationError(
            "vector must be a non-empty sequence of finite numbers")
    try:
        raw = tuple(value)
    except TypeError as exc:
        raise AppearanceEmbeddingValidationError(
            "vector must be a non-empty sequence of finite numbers") from exc
    if not raw:
        raise AppearanceEmbeddingValidationError("vector must not be empty")

    numbers = []
    for part in raw:
        if isinstance(part, bool) or not isinstance(part, Real):
            raise AppearanceEmbeddingValidationError(
                "vector must contain only finite numbers")
        numeric = float(part)
        if not math.isfinite(numeric):
            raise AppearanceEmbeddingValidationError(
                "vector must contain only finite numbers")
        numbers.append(numeric)

    scale = max(abs(number) for number in numbers)
    if scale == 0.0:
        raise AppearanceEmbeddingValidationError(
            "vector must have a non-zero L2 norm")
    scaled_norm = math.sqrt(math.fsum(
        (number / scale) ** 2 for number in numbers))
    if not math.isfinite(scaled_norm) or scaled_norm == 0.0:
        raise AppearanceEmbeddingValidationError(
            "vector must have a finite non-zero L2 norm")
    return tuple((number / scale) / scaled_norm for number in numbers)


@dataclass(frozen=True)
class VehicleAppearanceEmbedding:
    """Immutable learned appearance evidence without identity semantics."""

    vector: tuple
    dimension: int
    model_id: str
    model_version: str
    weights_sha256: str
    preprocessing_sha256: str

    def __post_init__(self):
        dimension = _positive_integer(self.dimension, "dimension")
        vector = _normalized_vector(self.vector)
        if len(vector) != dimension:
            raise AppearanceEmbeddingValidationError(
                "dimension must exactly match vector length")
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "model_id", _required_text(
            self.model_id, "model_id"))
        object.__setattr__(self, "model_version", _required_text(
            self.model_version, "model_version"))
        object.__setattr__(self, "weights_sha256", _sha256_text(
            self.weights_sha256, "weights_sha256"))
        object.__setattr__(self, "preprocessing_sha256", _sha256_text(
            self.preprocessing_sha256, "preprocessing_sha256"))

    def to_dict(self) -> dict:
        """Return a stable, explicit representation of the evidence."""
        return {
            "vector": list(self.vector),
            "dimension": self.dimension,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "weights_sha256": self.weights_sha256,
            "preprocessing_sha256": self.preprocessing_sha256,
        }


@dataclass(frozen=True)
class AppearancePreprocessingConfig:
    """Explicit deterministic preprocessing contract for one ReID model."""

    input_width: int
    input_height: int
    mean: tuple
    std: tuple

    def __post_init__(self):
        object.__setattr__(self, "input_width", _positive_integer(
            self.input_width, "input_width"))
        object.__setattr__(self, "input_height", _positive_integer(
            self.input_height, "input_height"))
        object.__setattr__(self, "mean", _three_numbers(self.mean, "mean"))
        object.__setattr__(self, "std", _three_numbers(
            self.std, "std", positive=True))

    def to_dict(self) -> dict:
        return {
            "schema": _PREPROCESSING_SCHEMA,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "source_channel_order": "BGR",
            "model_channel_order": "RGB",
            "resize_interpolation": "INTER_LINEAR",
            "numeric_dtype": "float32",
            "value_scale": "1/255",
            "mean": list(self.mean),
            "std": list(self.std),
            "tensor_order": "CHW",
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


def preprocess_vehicle_crop(
        vehicle_crop,
        config: AppearancePreprocessingConfig,
) -> np.ndarray:
    """Convert an associated uint8 BGR crop to deterministic float32 CHW."""
    if not isinstance(config, AppearancePreprocessingConfig):
        raise AppearanceEmbeddingValidationError(
            "config must be an AppearancePreprocessingConfig")
    if vehicle_crop is None:
        raise AppearanceEvidenceUnavailableError(
            "no_vehicle_crop", "no associated full-vehicle crop is available")
    if not isinstance(vehicle_crop, np.ndarray) \
            or vehicle_crop.dtype != np.uint8 \
            or vehicle_crop.ndim != 3 \
            or vehicle_crop.shape[2] != 3 \
            or vehicle_crop.size == 0:
        raise AppearanceEmbeddingValidationError(
            "vehicle_crop must be a non-empty uint8 HxWx3 BGR image")

    # cvtColor and resize both allocate outputs; the caller's crop is never
    # mutated or returned through an aliased view.
    rgb = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb,
        (config.input_width, config.input_height),
        interpolation=cv2.INTER_LINEAR,
    )
    numeric = resized.astype(np.float32) / np.float32(255.0)
    mean = np.asarray(config.mean, dtype=np.float32).reshape((1, 1, 3))
    std = np.asarray(config.std, dtype=np.float32).reshape((1, 1, 3))
    normalized = (numeric - mean) / std
    return np.ascontiguousarray(normalized.transpose((2, 0, 1)))


class VehicleAppearanceEncoder(ABC):
    """Model-neutral port for encoding an associated full-vehicle crop."""

    @abstractmethod
    def encode(self, vehicle_crop) -> VehicleAppearanceEmbedding:
        """Return learned appearance evidence or raise a domain error."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LocalTorchScriptAppearanceEncoder(VehicleAppearanceEncoder):
    """Lazy local TorchScript adapter; never discovers or downloads weights."""

    def __init__(
            self,
            model_path,
            *,
            model_id: str,
            model_version: str,
            device: str,
            preprocessing: AppearancePreprocessingConfig,
            model_loader: Optional[Callable] = None,
            torch_module=None,
    ):
        try:
            self.model_path = Path(model_path)
        except TypeError as exc:
            raise AppearanceEmbeddingValidationError(
                "model_path must be path-like") from exc
        self.model_id = _required_text(model_id, "model_id")
        self.model_version = _required_text(model_version, "model_version")
        requested_device = _required_text(device, "device").lower()
        if requested_device == "gpu":
            requested_device = "cuda"
        if requested_device not in {"cpu", "cuda", "auto"}:
            raise AppearanceEmbeddingValidationError(
                "device must be one of: cpu, cuda, gpu, auto")
        if not isinstance(preprocessing, AppearancePreprocessingConfig):
            raise AppearanceEmbeddingValidationError(
                "preprocessing must be an AppearancePreprocessingConfig")
        if model_loader is not None and not callable(model_loader):
            raise AppearanceEmbeddingValidationError(
                "model_loader must be callable or None")

        self.requested_device = requested_device
        self.preprocessing = preprocessing
        self._model_loader = model_loader
        self._torch = torch_module
        self._model = None
        self._resolved_device = None
        self._weights_sha256 = None
        self._load_lock = threading.Lock()

    def _torch_runtime(self):
        if self._torch is None:
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise AppearanceEvidenceUnavailableError(
                    "runtime_unavailable",
                    "PyTorch is unavailable; the local TorchScript ReID "
                    "adapter requires the existing Ultralytics PyTorch runtime",
                ) from exc
            self._torch = torch
        return self._torch

    def _device(self, torch_runtime) -> str:
        if self._resolved_device is not None:
            return self._resolved_device
        if self.requested_device == "auto":
            resolved = (
                "cuda" if torch_runtime.cuda.is_available() else "cpu")
        elif self.requested_device == "cuda":
            if not torch_runtime.cuda.is_available():
                raise AppearanceEvidenceUnavailableError(
                    "device_unavailable",
                    "CUDA was requested for appearance encoding but is not "
                    "available",
                )
            resolved = "cuda"
        else:
            resolved = "cpu"
        self._resolved_device = resolved
        return resolved

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            if not self.model_path.is_file():
                raise AppearanceEvidenceUnavailableError(
                    "missing_weights",
                    f"ReID weights not found: {self.model_path}. Provide an "
                    "approved local TorchScript model; automatic download is "
                    "disabled.",
                )
            torch_runtime = self._torch_runtime()
            device = self._device(torch_runtime)
            loader = self._model_loader
            if loader is None:
                loader = lambda path, target: torch_runtime.jit.load(
                    path, map_location=target)
            try:
                model = loader(str(self.model_path), device)
                model = model.to(device)
                model.eval()
            except Exception as exc:
                raise AppearanceEvidenceUnavailableError(
                    "model_load_failed",
                    f"failed to load local TorchScript ReID model: {exc}",
                ) from exc
            self._weights_sha256 = _file_sha256(self.model_path)
            self._model = model
            return self._model

    @staticmethod
    def _one_dimensional_output(output) -> tuple:
        try:
            array = output.detach().cpu().numpy()
        except (AttributeError, TypeError, RuntimeError) as exc:
            raise AppearanceEmbeddingValidationError(
                "TorchScript model output must be one tensor") from exc
        array = np.asarray(array)
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        elif array.ndim != 1:
            raise AppearanceEmbeddingValidationError(
                "model output must have shape [D] or [1, D]")
        if array.size == 0:
            raise AppearanceEmbeddingValidationError(
                "model output embedding must not be empty")
        # Preserve value types here so the value-object validator, rather than
        # an implicit float coercion, rejects booleans, strings, complex
        # values, NaN, and infinity with the same domain-specific error.
        return tuple(array.tolist())

    def encode(self, vehicle_crop) -> VehicleAppearanceEmbedding:
        processed = preprocess_vehicle_crop(vehicle_crop, self.preprocessing)
        model = self._get_model()
        torch_runtime = self._torch_runtime()
        device = self._device(torch_runtime)
        tensor = torch_runtime.from_numpy(processed[np.newaxis, ...]).to(device)
        try:
            with torch_runtime.inference_mode():
                output = model(tensor)
        except Exception as exc:
            raise AppearanceEvidenceUnavailableError(
                "inference_failed",
                f"local ReID model inference failed: {exc}",
            ) from exc
        vector = self._one_dimensional_output(output)
        return VehicleAppearanceEmbedding(
            vector=vector,
            dimension=len(vector),
            model_id=self.model_id,
            model_version=self.model_version,
            weights_sha256=self._weights_sha256,
            preprocessing_sha256=self.preprocessing.sha256,
        )


def build_vehicle_appearance_encoder(
        config,
        *,
        model_loader: Optional[Callable] = None,
        torch_module=None,
) -> Optional[LocalTorchScriptAppearanceEncoder]:
    """Build the optional encoder from an ``appearance_embedding`` section.

    A missing section or ``enabled: false`` returns ``None`` before inspecting
    any model path or importing PyTorch.
    """
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise AppearanceEmbeddingValidationError("config must be a mapping")
    section = config.get("appearance_embedding")
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise AppearanceEmbeddingValidationError(
            "appearance_embedding must be a mapping")
    enabled = section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise AppearanceEmbeddingValidationError(
            "appearance_embedding.enabled must be true or false")
    if not enabled:
        return None

    required = (
        "model_path", "model_id", "model_version", "device",
        "input_width", "input_height", "mean", "std",
    )
    missing = [name for name in required if name not in section]
    if missing:
        raise AppearanceEmbeddingValidationError(
            "enabled appearance_embedding is missing fields: "
            + ", ".join(missing))
    preprocessing = AppearancePreprocessingConfig(
        input_width=section["input_width"],
        input_height=section["input_height"],
        mean=section["mean"],
        std=section["std"],
    )
    return LocalTorchScriptAppearanceEncoder(
        section["model_path"],
        model_id=section["model_id"],
        model_version=section["model_version"],
        device=section["device"],
        preprocessing=preprocessing,
        model_loader=model_loader,
        torch_module=torch_module,
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
