"""Focused tests for Step 36 explainable appearance-evidence comparison."""

import inspect
import math
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields

import pytest

from phase3_city import (
    AppearanceComparisonResult,
    AppearanceComparisonStatus,
    AppearanceComparisonValidationError,
    AppearanceEmbeddingProvenance,
    VehicleAppearanceEmbedding,
    compare_appearance_embeddings,
)


_WEIGHTS_A = "a" * 64
_WEIGHTS_B = "b" * 64
_PREPROCESSING_A = "c" * 64
_PREPROCESSING_B = "d" * 64


def _embedding(
        vector=(1.0, 0.0),
        *,
        model_id="test-reid",
        model_version="1",
        weights_sha256=_WEIGHTS_A,
        preprocessing_sha256=_PREPROCESSING_A,
):
    return VehicleAppearanceEmbedding(
        vector=vector,
        dimension=len(vector),
        model_id=model_id,
        model_version=model_version,
        weights_sha256=weights_sha256,
        preprocessing_sha256=preprocessing_sha256,
    )


def test_identical_embeddings_produce_maximum_cosine_similarity():
    source = _embedding((3.0, 4.0))

    result = compare_appearance_embeddings(source, source)

    assert result.status is AppearanceComparisonStatus.AVAILABLE
    assert result.available is True
    assert result.cosine_similarity == pytest.approx(1.0)
    assert result.euclidean_distance == pytest.approx(0.0)


def test_orthogonal_embeddings_produce_zero_cosine_similarity():
    result = compare_appearance_embeddings(
        _embedding((1.0, 0.0)),
        _embedding((0.0, 1.0)),
    )

    assert result.cosine_similarity == pytest.approx(0.0)
    assert result.euclidean_distance == pytest.approx(math.sqrt(2.0))


def test_opposite_embeddings_produce_minimum_cosine_similarity():
    result = compare_appearance_embeddings(
        _embedding((1.0, 0.0)),
        _embedding((-1.0, 0.0)),
    )

    assert result.cosine_similarity == pytest.approx(-1.0)
    assert result.euclidean_distance == pytest.approx(2.0)


def test_repeated_comparison_and_serialization_are_deterministic():
    source = _embedding((1.0, 2.0, 3.0))
    candidate = _embedding((3.0, 2.0, 1.0))

    first = compare_appearance_embeddings(source, candidate)
    second = compare_appearance_embeddings(source, candidate)

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert list(first.to_dict()) == [
        "status",
        "available",
        "cosine_similarity",
        "euclidean_distance",
        "embedding_dimension",
        "provenance_compatible",
        "reason",
        "incompatible_fields",
        "source_provenance",
        "candidate_provenance",
    ]


@pytest.mark.parametrize(("source_vector", "candidate_vector"), [
    ((1.0, 0.0), (1.0, 0.0)),
    ((1.0, 0.0), (0.0, 1.0)),
    ((1.0, 0.0), (-1.0, 0.0)),
    ((1.0, 2.0, 3.0), (-3.0, 2.0, 1.0)),
])
def test_available_results_are_finite_and_within_documented_ranges(
        source_vector, candidate_vector):
    result = compare_appearance_embeddings(
        _embedding(source_vector),
        _embedding(candidate_vector),
    )

    assert math.isfinite(result.cosine_similarity)
    assert -1.0 <= result.cosine_similarity <= 1.0
    assert math.isfinite(result.euclidean_distance)
    assert 0.0 <= result.euclidean_distance <= 2.0


def test_cosine_equals_dot_product_of_stored_normalized_vectors():
    source = _embedding((2.0, -1.0, 4.0))
    candidate = _embedding((-3.0, 5.0, 2.0))

    result = compare_appearance_embeddings(source, candidate)
    expected = math.fsum(
        left * right for left, right
        in zip(source.vector, candidate.vector)
    )

    assert result.cosine_similarity == expected


def test_euclidean_diagnostic_satisfies_normalized_vector_relation():
    result = compare_appearance_embeddings(
        _embedding((1.0, 2.0, 3.0)),
        _embedding((3.0, -2.0, 1.0)),
    )

    assert result.euclidean_distance ** 2 == pytest.approx(
        2.0 - (2.0 * result.cosine_similarity))


def test_source_embedding_unavailable_is_explicit_and_neutral():
    result = compare_appearance_embeddings(None, _embedding())

    assert result.status is AppearanceComparisonStatus.SOURCE_UNAVAILABLE
    assert result.reason == "source_unavailable"
    assert result.source_provenance is None
    assert result.candidate_provenance is not None
    assert result.provenance_compatible is None
    _assert_no_numeric_evidence(result)


def test_candidate_embedding_unavailable_is_explicit_and_neutral():
    result = compare_appearance_embeddings(_embedding(), None)

    assert result.status is AppearanceComparisonStatus.CANDIDATE_UNAVAILABLE
    assert result.reason == "candidate_unavailable"
    assert result.source_provenance is not None
    assert result.candidate_provenance is None
    assert result.provenance_compatible is None
    _assert_no_numeric_evidence(result)


def test_both_embeddings_unavailable_are_explicit_and_neutral():
    result = compare_appearance_embeddings(None, None)

    assert result.status is AppearanceComparisonStatus.BOTH_UNAVAILABLE
    assert result.reason == "both_unavailable"
    assert result.source_provenance is None
    assert result.candidate_provenance is None
    assert result.provenance_compatible is None
    _assert_no_numeric_evidence(result)


def _assert_no_numeric_evidence(result):
    assert result.available is False
    assert result.cosine_similarity is None
    assert result.euclidean_distance is None
    assert result.embedding_dimension is None


def test_missing_evidence_is_not_converted_to_zero_or_negative_evidence():
    for result in (
        compare_appearance_embeddings(None, _embedding()),
        compare_appearance_embeddings(_embedding(), None),
        compare_appearance_embeddings(None, None),
    ):
        _assert_no_numeric_evidence(result)
        assert result.incompatible_fields == ()
        assert "reject" not in result.to_dict()
        assert "contradiction" not in result.to_dict()


@pytest.mark.parametrize(("source", "candidate", "field"), [
    (_embedding((1.0, 0.0)), _embedding((1.0, 0.0, 0.0)), "dimension"),
    (_embedding(), _embedding(model_id="other-reid"), "model_id"),
    (_embedding(), _embedding(model_version="2"), "model_version"),
    (_embedding(), _embedding(weights_sha256=_WEIGHTS_B), "weights_sha256"),
    (
        _embedding(),
        _embedding(preprocessing_sha256=_PREPROCESSING_B),
        "preprocessing_sha256",
    ),
])
def test_each_provenance_mismatch_is_explicitly_non_comparable(
        source, candidate, field):
    result = compare_appearance_embeddings(source, candidate)

    assert result.status is (
        AppearanceComparisonStatus.INCOMPATIBLE_PROVENANCE)
    assert result.reason == "incompatible_provenance"
    assert result.provenance_compatible is False
    assert result.incompatible_fields == (field,)
    assert result.source_provenance is not None
    assert result.candidate_provenance is not None
    _assert_no_numeric_evidence(result)


def test_multiple_incompatibilities_use_canonical_field_order():
    result = compare_appearance_embeddings(
        _embedding((1.0, 0.0), model_id="source", model_version="1"),
        _embedding(
            (1.0, 0.0, 0.0),
            model_id="candidate",
            model_version="2",
            weights_sha256=_WEIGHTS_B,
            preprocessing_sha256=_PREPROCESSING_B,
        ),
    )

    assert result.incompatible_fields == (
        "dimension",
        "model_id",
        "model_version",
        "weights_sha256",
        "preprocessing_sha256",
    )


def test_matching_provenance_exposes_compatible_snapshot_and_succeeds():
    source = _embedding((1.0, 2.0))
    candidate = _embedding((2.0, 1.0))

    result = compare_appearance_embeddings(source, candidate)

    expected_provenance = AppearanceEmbeddingProvenance.from_embedding(source)
    assert result.provenance_compatible is True
    assert result.embedding_dimension == 2
    assert result.source_provenance == expected_provenance
    assert result.candidate_provenance == expected_provenance
    assert result.incompatible_fields == ()
    assert result.reason is None


def test_comparison_result_and_nested_provenance_are_immutable():
    result = compare_appearance_embeddings(_embedding(), _embedding())

    with pytest.raises(FrozenInstanceError):
        result.cosine_similarity = 0.0
    with pytest.raises(FrozenInstanceError):
        result.source_provenance.model_id = "changed"
    with pytest.raises(TypeError):
        result.incompatible_fields[0] = "dimension"


def test_comparison_does_not_mutate_input_embeddings():
    source = _embedding((3.0, 4.0))
    candidate = _embedding((4.0, 3.0))
    source_before = source.to_dict()
    candidate_before = candidate.to_dict()

    compare_appearance_embeddings(source, candidate)

    assert source.to_dict() == source_before
    assert candidate.to_dict() == candidate_before


def test_result_has_no_identity_probability_or_decision_semantics():
    result_field_names = {field.name for field in fields(
        AppearanceComparisonResult)}
    serialized_keys = set(compare_appearance_embeddings(
        _embedding(), _embedding()).to_dict())
    forbidden = {
        "same_vehicle",
        "identity",
        "identity_probability",
        "probability",
        "decision",
        "match",
        "threshold",
    }

    assert result_field_names.isdisjoint(forbidden)
    assert serialized_keys.isdisjoint(forbidden)


def test_public_comparison_has_no_threshold_or_policy_parameter():
    signature = inspect.signature(compare_appearance_embeddings)

    assert tuple(signature.parameters) == ("source", "candidate")
    assert all(
        "threshold" not in field.name
        for field in fields(AppearanceComparisonResult)
    )


@pytest.mark.parametrize(("source", "candidate", "name"), [
    (object(), _embedding(), "source"),
    (_embedding(), object(), "candidate"),
])
def test_invalid_non_missing_inputs_fail_clearly(source, candidate, name):
    with pytest.raises(AppearanceComparisonValidationError, match=name):
        compare_appearance_embeddings(source, candidate)


def test_step35_embedding_normalization_and_serialization_are_unchanged():
    embedding = _embedding((3.0, 4.0))

    assert embedding.vector == (0.6, 0.8)
    assert embedding.dimension == 2
    assert embedding.to_dict() == {
        "vector": [0.6, 0.8],
        "dimension": 2,
        "model_id": "test-reid",
        "model_version": "1",
        "weights_sha256": _WEIGHTS_A,
        "preprocessing_sha256": _PREPROCESSING_A,
    }


def test_step36_import_does_not_import_phase2_matcher_or_collector():
    command = (
        "import sys; "
        "import phase3_city.appearance_comparison; "
        "assert 'phase2_city.cross_camera_matching' not in sys.modules; "
        "assert 'phase2_city.candidate_collection' not in sys.modules"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
