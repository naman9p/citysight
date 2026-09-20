"""Focused tests for Step 35 learned appearance-embedding foundations."""

import hashlib
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from phase3_city.appearance_embedding import (
    AppearanceEmbeddingValidationError,
    AppearanceEvidenceUnavailableError,
    AppearancePreprocessingConfig,
    LocalTorchScriptAppearanceEncoder,
    VehicleAppearanceEmbedding,
    VehicleAppearanceEncoder,
    build_vehicle_appearance_encoder,
    preprocess_vehicle_crop,
)


_WEIGHTS_HASH = "a" * 64
_PREPROCESSING_HASH = "b" * 64


def _embedding(vector=(3.0, 4.0), dimension=2):
    return VehicleAppearanceEmbedding(
        vector=vector,
        dimension=dimension,
        model_id="test-reid",
        model_version="1",
        weights_sha256=_WEIGHTS_HASH,
        preprocessing_sha256=_PREPROCESSING_HASH,
    )


def _preprocessing(**overrides):
    values = {
        "input_width": 2,
        "input_height": 2,
        "mean": (0.0, 0.0, 0.0),
        "std": (1.0, 1.0, 1.0),
    }
    values.update(overrides)
    return AppearancePreprocessingConfig(**values)


class _FakeTensor:
    def __init__(self, array, device_log=None):
        self.array = np.asarray(array)
        self.device_log = device_log

    def to(self, device):
        if self.device_log is not None:
            self.device_log.append(device)
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array.copy()


class _FakeCuda:
    def __init__(self, available=False):
        self.available = available

    def is_available(self):
        return self.available


class _FakeTorch:
    def __init__(self, cuda_available=False):
        self.cuda = _FakeCuda(cuda_available)
        self.input_devices = []
        self.input_arrays = []

    def from_numpy(self, array):
        self.input_arrays.append(np.asarray(array).copy())
        return _FakeTensor(array, self.input_devices)

    @staticmethod
    def inference_mode():
        return nullcontext()


class _FakeModel:
    def __init__(self, output=(3.0, 4.0), error=None):
        self.output = output
        self.error = error
        self.to_calls = []
        self.eval_calls = 0
        self.inference_calls = 0

    def to(self, device):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def __call__(self, tensor):
        self.inference_calls += 1
        if self.error is not None:
            raise self.error
        output = self.output(tensor) if callable(self.output) else self.output
        return _FakeTensor(output)


def _encoder(tmp_path, *, model=None, torch_runtime=None, device="cpu"):
    weights = tmp_path / "reid.ts"
    weights.write_bytes(b"local test weights")
    model = model or _FakeModel()
    torch_runtime = torch_runtime or _FakeTorch()
    loader_calls = []

    def loader(path, target_device):
        loader_calls.append((path, target_device))
        return model

    encoder = LocalTorchScriptAppearanceEncoder(
        weights,
        model_id="test-reid",
        model_version="1",
        device=device,
        preprocessing=_preprocessing(),
        model_loader=loader,
        torch_module=torch_runtime,
    )
    return encoder, weights, model, torch_runtime, loader_calls


def test_embedding_is_immutable_and_canonical():
    embedding = _embedding(vector=[3, 4])

    assert embedding.vector == (0.6, 0.8)
    assert embedding.to_dict() == {
        "vector": [0.6, 0.8],
        "dimension": 2,
        "model_id": "test-reid",
        "model_version": "1",
        "weights_sha256": _WEIGHTS_HASH,
        "preprocessing_sha256": _PREPROCESSING_HASH,
    }
    with pytest.raises(FrozenInstanceError):
        embedding.dimension = 3
    with pytest.raises(TypeError):
        embedding.vector[0] = 1.0


@pytest.mark.parametrize("vector", [(), []])
def test_empty_embedding_is_rejected(vector):
    with pytest.raises(AppearanceEmbeddingValidationError, match="empty"):
        _embedding(vector=vector, dimension=1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_embedding_is_rejected(value):
    with pytest.raises(AppearanceEmbeddingValidationError, match="finite"):
        _embedding(vector=(1.0, value))


def test_zero_norm_embedding_is_rejected():
    with pytest.raises(AppearanceEmbeddingValidationError, match="non-zero"):
        _embedding(vector=(0.0, 0.0))


def test_embedding_dimension_must_match_vector():
    with pytest.raises(AppearanceEmbeddingValidationError, match="exactly"):
        _embedding(vector=(1.0, 2.0), dimension=3)


def test_l2_normalization_is_deterministic_and_handles_large_values():
    first = _embedding(vector=(3e300, 4e300))
    second = _embedding(vector=(3e300, 4e300))

    assert first == second
    assert first.vector == pytest.approx((0.6, 0.8))
    assert np.linalg.norm(first.vector) == pytest.approx(1.0)


def test_preprocessing_converts_bgr_to_rgb_with_explicit_chw_order():
    crop = np.array([[[10, 20, 30]]], dtype=np.uint8)

    result = preprocess_vehicle_crop(
        crop,
        _preprocessing(input_width=1, input_height=1),
    )

    assert result.shape == (3, 1, 1)
    assert result.dtype == np.float32
    assert result[:, 0, 0] == pytest.approx(
        np.array([30, 20, 10], dtype=np.float32) / 255.0)


def test_resize_and_normalization_are_deterministic():
    crop = np.arange(3 * 4 * 3, dtype=np.uint8).reshape((3, 4, 3))
    config = _preprocessing(
        input_width=3,
        input_height=5,
        mean=(0.1, 0.2, 0.3),
        std=(0.5, 0.25, 0.125),
    )

    first = preprocess_vehicle_crop(crop, config)
    second = preprocess_vehicle_crop(crop, config)

    assert first.shape == (3, 5, 3)
    assert first.flags.c_contiguous
    assert np.array_equal(first, second)
    assert config.sha256 == _preprocessing(
        input_width=3,
        input_height=5,
        mean=(0.1, 0.2, 0.3),
        std=(0.5, 0.25, 0.125),
    ).sha256


def test_preprocessing_does_not_mutate_source_image():
    crop = np.arange(4 * 5 * 3, dtype=np.uint8).reshape((4, 5, 3))
    original = crop.copy()

    result = preprocess_vehicle_crop(crop, _preprocessing())
    result[:] = 0

    assert np.array_equal(crop, original)


@pytest.mark.parametrize("crop", [
    np.empty((0, 2, 3), dtype=np.uint8),
    np.empty((2, 0, 3), dtype=np.uint8),
    np.zeros((2, 2), dtype=np.uint8),
    np.zeros((2, 2, 4), dtype=np.uint8),
    np.zeros((2, 2, 3), dtype=np.float32),
    "not an image",
])
def test_invalid_or_empty_crop_is_rejected(crop):
    with pytest.raises(AppearanceEmbeddingValidationError, match="uint8"):
        preprocess_vehicle_crop(crop, _preprocessing())


def test_no_crop_is_explicitly_unavailable_and_never_loads_model(tmp_path):
    encoder, _, model, _, loader_calls = _encoder(tmp_path)

    with pytest.raises(AppearanceEvidenceUnavailableError) as exc_info:
        encoder.encode(None)

    assert exc_info.value.reason == "no_vehicle_crop"
    assert loader_calls == []
    assert model.inference_calls == 0


def test_encoder_port_is_model_neutral():
    assert issubclass(LocalTorchScriptAppearanceEncoder, VehicleAppearanceEncoder)
    assert VehicleAppearanceEncoder.__abstractmethods__ == {"encode"}


def test_model_loading_is_lazy_and_model_is_reused(tmp_path):
    encoder, _, model, _, loader_calls = _encoder(tmp_path)
    crop = np.zeros((2, 2, 3), dtype=np.uint8)

    assert loader_calls == []
    encoder.encode(crop)
    encoder.encode(crop)

    assert len(loader_calls) == 1
    assert model.to_calls == ["cpu"]
    assert model.eval_calls == 1
    assert model.inference_calls == 2


def test_module_import_does_not_import_torch_or_load_a_model():
    command = (
        "import sys; "
        "assert 'torch' not in sys.modules; "
        "import phase3_city.appearance_embedding; "
        "assert 'torch' not in sys.modules"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_disabled_configuration_performs_no_loading_or_inference():
    loader_calls = []

    encoder = build_vehicle_appearance_encoder(
        {"appearance_embedding": {"enabled": False}},
        model_loader=lambda *args: loader_calls.append(args),
        torch_module=object(),
    )

    assert encoder is None
    assert loader_calls == []


def test_missing_local_weights_fail_clearly_before_runtime_or_loader(tmp_path):
    loader_calls = []
    encoder = LocalTorchScriptAppearanceEncoder(
        tmp_path / "missing.ts",
        model_id="test-reid",
        model_version="1",
        device="cpu",
        preprocessing=_preprocessing(),
        model_loader=lambda *args: loader_calls.append(args),
        torch_module=object(),
    )

    with pytest.raises(AppearanceEvidenceUnavailableError) as exc_info:
        encoder.encode(np.zeros((2, 2, 3), dtype=np.uint8))

    assert exc_info.value.reason == "missing_weights"
    assert "automatic download is disabled" in str(exc_info.value)
    assert loader_calls == []


def test_cpu_device_path_is_explicit_for_model_and_input(tmp_path):
    encoder, _, model, torch_runtime, loader_calls = _encoder(
        tmp_path, device="cpu")

    encoder.encode(np.zeros((2, 2, 3), dtype=np.uint8))

    assert loader_calls[0][1] == "cpu"
    assert model.to_calls == ["cpu"]
    assert torch_runtime.input_devices == ["cpu"]


def test_repeated_inference_is_deterministic(tmp_path):
    encoder, _, _, _, _ = _encoder(tmp_path)
    crop = np.arange(12, dtype=np.uint8).reshape((2, 2, 3))

    first = encoder.encode(crop)
    second = encoder.encode(crop)

    assert first == second
    assert first.vector == pytest.approx((0.6, 0.8))


def test_injected_loader_and_model_receive_deterministic_preprocessed_input(
        tmp_path):
    def output(tensor):
        flattened = tensor.array.reshape(-1)
        return flattened[:3]

    model = _FakeModel(output=output)
    encoder, weights, _, torch_runtime, loader_calls = _encoder(
        tmp_path, model=model)
    crop = np.array([[[0, 127, 255]]], dtype=np.uint8)

    embedding = encoder.encode(crop)

    assert loader_calls == [(str(weights), "cpu")]
    assert torch_runtime.input_arrays[0].shape == (1, 3, 2, 2)
    assert embedding.dimension == 3


@pytest.mark.parametrize("shape", [(), (2, 3), (1, 2, 3), (0,)])
def test_ambiguous_or_empty_model_output_shapes_are_rejected(tmp_path, shape):
    model = _FakeModel(output=np.zeros(shape, dtype=np.float32))
    encoder, _, _, _, _ = _encoder(tmp_path, model=model)

    with pytest.raises(AppearanceEmbeddingValidationError, match="shape|empty"):
        encoder.encode(np.zeros((2, 2, 3), dtype=np.uint8))


@pytest.mark.parametrize("output", [
    np.array([3.0, 4.0], dtype=np.float32),
    np.array([[3.0, 4.0]], dtype=np.float32),
])
def test_only_vector_or_single_item_batch_output_is_accepted(tmp_path, output):
    encoder, _, _, _, _ = _encoder(
        tmp_path, model=_FakeModel(output=output))

    embedding = encoder.encode(np.zeros((2, 2, 3), dtype=np.uint8))

    assert embedding.vector == pytest.approx((0.6, 0.8))


@pytest.mark.parametrize("output", [
    np.array([1.0, np.nan]),
    np.array([1.0, np.inf]),
    np.array([0.0, 0.0]),
])
def test_unusable_model_output_values_are_rejected(tmp_path, output):
    encoder, _, _, _, _ = _encoder(
        tmp_path, model=_FakeModel(output=output))

    with pytest.raises(AppearanceEmbeddingValidationError):
        encoder.encode(np.zeros((2, 2, 3), dtype=np.uint8))


def test_inference_failure_has_explicit_neutral_unavailable_reason(tmp_path):
    encoder, _, _, _, _ = _encoder(
        tmp_path, model=_FakeModel(error=RuntimeError("test failure")))

    with pytest.raises(AppearanceEvidenceUnavailableError) as exc_info:
        encoder.encode(np.zeros((2, 2, 3), dtype=np.uint8))

    assert exc_info.value.reason == "inference_failed"


def test_weights_sha256_is_recorded_from_exact_local_artifact(tmp_path):
    encoder, weights, _, _, _ = _encoder(tmp_path)

    embedding = encoder.encode(np.zeros((2, 2, 3), dtype=np.uint8))

    assert embedding.weights_sha256 == hashlib.sha256(
        weights.read_bytes()).hexdigest()
    assert embedding.preprocessing_sha256 == encoder.preprocessing.sha256


def test_enabled_factory_builds_lazy_local_encoder(tmp_path):
    weights = tmp_path / "reid.ts"
    weights.write_bytes(b"test weights")
    loader_calls = []
    config = {
        "appearance_embedding": {
            "enabled": True,
            "model_path": str(weights),
            "model_id": "test-reid",
            "model_version": "1",
            "device": "cpu",
            "input_width": 2,
            "input_height": 2,
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        }
    }

    encoder = build_vehicle_appearance_encoder(
        config,
        model_loader=lambda *args: loader_calls.append(args),
        torch_module=_FakeTorch(),
    )

    assert isinstance(encoder, LocalTorchScriptAppearanceEncoder)
    assert loader_calls == []


def test_enabled_factory_requires_explicit_complete_configuration():
    with pytest.raises(AppearanceEmbeddingValidationError, match="missing"):
        build_vehicle_appearance_encoder({
            "appearance_embedding": {"enabled": True},
        })
