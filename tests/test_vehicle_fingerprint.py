"""Step 23: deterministic Vehicle Fingerprint foundation."""

from dataclasses import FrozenInstanceError

import pytest

from phase1_anpr.observation.observation_builder import PlateObservation
from phase2_city import (
    FingerprintComparison,
    FingerprintValidationError,
    VehicleFingerprint,
    VehicleFingerprintScorer,
)


def fingerprint(plate=None, colour=None, vehicle_class=None):
    return VehicleFingerprint(
        normalized_plate=plate,
        vehicle_colour=colour,
        vehicle_class=vehicle_class,
    )


def observation(status="accepted", plate="GJ01AB1234"):
    return PlateObservation(
        event_id="event-1",
        camera_id="CAM_01",
        track_id=1,
        timestamp="2026-08-31T10:00:00+00:00",
        plate_raw=plate,
        plate_normalized=plate,
        confidence=0.9,
        status=status,
        detector_confidence=0.9,
        ocr_confidence=0.9,
        quality_score=0.8,
        best_frame_number=10,
        plate_image_path=None,
        model_version="phase1-anpr-0.1.0",
    )


def field(result, name):
    return next(item for item in result.field_comparisons if item.field == name)


def test_empty_fingerprint_represents_missing_evidence():
    value = VehicleFingerprint()
    assert value.normalized_plate is None
    assert value.vehicle_colour is None
    assert value.vehicle_class is None
    assert value.to_dict() == {
        "normalized_plate": None,
        "vehicle_colour": None,
        "vehicle_class": None,
    }


def test_fingerprint_canonicalizes_all_fields():
    value = fingerprint(" gj-01 ab 1234 ", "  Pearl   WHITE ", "  Sport  Utility ")
    assert value == fingerprint("GJ01AB1234", "pearl white", "sport utility")


def test_blank_and_whitespace_values_become_missing():
    value = fingerprint("---", "  \t ", "\n")
    assert value == VehicleFingerprint()


@pytest.mark.parametrize("kwargs", [
    {"normalized_plate": 1234},
    {"vehicle_colour": ["white"]},
    {"vehicle_class": True},
])
def test_invalid_fingerprint_field_types_rejected(kwargs):
    with pytest.raises(FingerprintValidationError):
        VehicleFingerprint(**kwargs)


def test_vehicle_fingerprint_is_immutable():
    value = fingerprint("GJ01AB1234", "white", "suv")
    with pytest.raises(FrozenInstanceError):
        value.vehicle_colour = "black"


def test_full_match():
    left = fingerprint("GJ01AB1234", "white", "suv")
    result = VehicleFingerprintScorer().compare(left, left)
    assert isinstance(result, FingerprintComparison)
    assert result.overall_similarity == 1.0
    assert result.evidence_coverage == 1.0
    assert result.matched_weight == 1.0
    assert result.contradicted_weight == 0.0
    assert result.net_evidence == 1.0
    assert result.outcome == "consistent"


def test_full_contradiction():
    left = fingerprint("GJ01AB1234", "white", "suv")
    right = fingerprint("MH12CD5678", "black", "sedan")
    result = VehicleFingerprintScorer().compare(left, right)
    assert result.overall_similarity == 0.0
    assert result.evidence_coverage == 1.0
    assert result.matched_weight == 0.0
    assert result.contradicted_weight == 1.0
    assert result.net_evidence == -1.0
    assert result.outcome == "contradictory"


def test_plate_only_match_is_not_renormalized_to_one():
    result = VehicleFingerprintScorer().compare(
        fingerprint("GJ01AB1234"), fingerprint("GJ01AB1234"))
    assert result.overall_similarity == 0.8
    assert result.evidence_coverage == 0.6
    assert result.matched_weight == 0.6
    assert result.net_evidence == 0.6


def test_colour_and_class_only_match_is_not_renormalized_to_one():
    result = VehicleFingerprintScorer().compare(
        fingerprint(colour="white", vehicle_class="suv"),
        fingerprint(colour="white", vehicle_class="suv"),
    )
    assert result.overall_similarity == 0.7
    assert result.evidence_coverage == 0.4
    assert result.matched_weight == 0.4


def test_missing_plate_with_matching_colour_and_class_exact_score():
    left = fingerprint("GJ01AB1234", "white", "suv")
    right = fingerprint(None, "white", "suv")
    result = VehicleFingerprintScorer().compare(left, right)
    assert result.overall_similarity == 0.7
    assert result.evidence_coverage == 0.4
    assert result.matched_weight == 0.4
    assert result.contradicted_weight == 0.0
    assert result.net_evidence == 0.4
    assert result.outcome == "consistent"
    assert field(result, "normalized_plate").state == "missing"
    assert field(result, "normalized_plate").contribution == 0.0


def test_mismatching_plate_with_matching_colour_and_class_exact_score():
    left = fingerprint("GJ01AB1234", "white", "suv")
    right = fingerprint("MH12CD5678", "white", "suv")
    result = VehicleFingerprintScorer().compare(left, right)
    assert result.overall_similarity == 0.4
    assert result.evidence_coverage == 1.0
    assert result.matched_weight == 0.4
    assert result.contradicted_weight == 0.6
    assert result.net_evidence == pytest.approx(-0.2)
    assert result.outcome == "mixed"
    assert field(result, "normalized_plate").state == "mismatch"
    assert field(result, "normalized_plate").contribution == -0.6


def test_missing_means_either_side_lacks_the_field():
    result = VehicleFingerprintScorer().compare(
        fingerprint("GJ01AB1234", "white"),
        fingerprint("GJ01AB1234", vehicle_class="suv"),
    )
    assert field(result, "vehicle_colour").state == "missing"
    assert field(result, "vehicle_class").state == "missing"


def test_zero_comparable_evidence_is_insufficient():
    result = VehicleFingerprintScorer().compare(
        fingerprint("GJ01AB1234"), fingerprint(colour="white"))
    assert result.overall_similarity is None
    assert result.evidence_coverage == 0.0
    assert result.matched_weight == 0.0
    assert result.contradicted_weight == 0.0
    assert result.net_evidence == 0.0
    assert result.outcome == "insufficient_evidence"


def test_all_four_outcomes_follow_evidence_only():
    scorer = VehicleFingerprintScorer()
    insufficient = scorer.compare(fingerprint("GJ01AB1234"), fingerprint(colour="red"))
    consistent = scorer.compare(fingerprint(colour="red"), fingerprint(colour="red"))
    contradictory = scorer.compare(fingerprint(colour="red"), fingerprint(colour="blue"))
    mixed = scorer.compare(
        fingerprint(colour="red", vehicle_class="suv"),
        fingerprint(colour="blue", vehicle_class="suv"),
    )
    assert insufficient.outcome == "insufficient_evidence"
    assert consistent.outcome == "consistent"
    assert contradictory.outcome == "contradictory"
    assert mixed.outcome == "mixed"


def test_scoring_is_symmetric():
    left = fingerprint("GJ01AB1234", "white", "suv")
    right = fingerprint(None, "black", "suv")
    scorer = VehicleFingerprintScorer()
    forward = scorer.compare(left, right)
    reverse = scorer.compare(right, left)
    assert forward.overall_similarity == reverse.overall_similarity
    assert forward.evidence_coverage == reverse.evidence_coverage
    assert forward.matched_weight == reverse.matched_weight
    assert forward.contradicted_weight == reverse.contradicted_weight
    assert forward.net_evidence == reverse.net_evidence
    assert forward.outcome == reverse.outcome
    assert [item.state for item in forward.field_comparisons] == [
        item.state for item in reverse.field_comparisons
    ]


def test_scoring_is_deterministic():
    left = fingerprint("GJ01AB1234", "white", "suv")
    right = fingerprint(None, "white", "sedan")
    scorer = VehicleFingerprintScorer()
    assert scorer.compare(left, right) == scorer.compare(left, right)


def test_comparison_to_dict_is_stable_and_ordered():
    result = VehicleFingerprintScorer().compare(
        fingerprint("GJ01AB1234"), fingerprint("GJ01AB1234"))
    assert result.to_dict() == {
        "overall_similarity": 0.8,
        "evidence_coverage": 0.6,
        "matched_weight": 0.6,
        "contradicted_weight": 0.0,
        "net_evidence": 0.6,
        "outcome": "consistent",
        "field_comparisons": [
            {
                "field": "normalized_plate",
                "left_value": "GJ01AB1234",
                "right_value": "GJ01AB1234",
                "state": "match",
                "weight": 0.6,
                "vote": 1,
                "contribution": 0.6,
            },
            {
                "field": "vehicle_colour",
                "left_value": None,
                "right_value": None,
                "state": "missing",
                "weight": 0.25,
                "vote": 0,
                "contribution": 0.0,
            },
            {
                "field": "vehicle_class",
                "left_value": None,
                "right_value": None,
                "state": "missing",
                "weight": 0.15,
                "vote": 0,
                "contribution": 0.0,
            },
        ],
    }


def test_comparison_details_are_tuple_backed_and_immutable():
    result = VehicleFingerprintScorer().compare(VehicleFingerprint(), VehicleFingerprint())
    assert isinstance(result.field_comparisons, tuple)
    with pytest.raises(FrozenInstanceError):
        result.outcome = "consistent"


def test_custom_weights_are_normalized():
    scorer = VehicleFingerprintScorer({
        "normalized_plate": 2,
        "vehicle_colour": 1,
        "vehicle_class": 1,
    })
    assert dict(scorer.weights) == {
        "normalized_plate": 0.5,
        "vehicle_colour": 0.25,
        "vehicle_class": 0.25,
    }
    result = scorer.compare(fingerprint("GJ01AB1234"), fingerprint("GJ01AB1234"))
    assert result.overall_similarity == 0.75
    assert result.evidence_coverage == 0.5


def test_large_finite_weights_are_normalized_without_overflow():
    scorer = VehicleFingerprintScorer({
        "normalized_plate": 1e308,
        "vehicle_colour": 1e308,
        "vehicle_class": 1e308,
    })
    assert sum(scorer.weights.values()) == pytest.approx(1.0)
    assert all(
        weight == pytest.approx(1 / 3)
        for weight in scorer.weights.values()
    )


def test_uneven_large_finite_weights_preserve_relative_proportions():
    scorer = VehicleFingerprintScorer({
        "normalized_plate": 1e308,
        "vehicle_colour": 5e307,
        "vehicle_class": 1e307,
    })
    assert sum(scorer.weights.values()) == pytest.approx(1.0)
    assert dict(scorer.weights) == pytest.approx({
        "normalized_plate": 0.625,
        "vehicle_colour": 0.3125,
        "vehicle_class": 0.0625,
    })


def test_extremely_large_integer_weight_raises_validation_error():
    with pytest.raises(FingerprintValidationError):
        VehicleFingerprintScorer({"normalized_plate": 10**400})


def test_net_evidence_matches_sum_of_stored_contributions():
    scorer = VehicleFingerprintScorer({
        "normalized_plate": 1,
        "vehicle_colour": 2,
        "vehicle_class": 4,
    })
    result = scorer.compare(
        fingerprint("GJ01AB1234", "white", "suv"),
        fingerprint("GJ01AB1234", "white", "sedan"),
    )
    contribution_sum = sum(
        comparison.contribution for comparison in result.field_comparisons
    )
    assert result.net_evidence == pytest.approx(contribution_sum)


def test_negative_weight_rejected():
    with pytest.raises(FingerprintValidationError):
        VehicleFingerprintScorer({"vehicle_colour": -0.1})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_weight_rejected(value):
    with pytest.raises(FingerprintValidationError):
        VehicleFingerprintScorer({"normalized_plate": value})


def test_non_numeric_weight_rejected():
    with pytest.raises(FingerprintValidationError):
        VehicleFingerprintScorer({"normalized_plate": "0.6"})
    with pytest.raises(FingerprintValidationError):
        VehicleFingerprintScorer({"normalized_plate": True})


def test_all_zero_weights_rejected():
    with pytest.raises(FingerprintValidationError):
        VehicleFingerprintScorer({
            "normalized_plate": 0,
            "vehicle_colour": 0,
            "vehicle_class": 0,
        })


def test_ocr_confusable_characters_are_not_substituted():
    observed = fingerprint("gj-o1 ib 8oo0")
    substituted = fingerprint("GJ011888000")
    assert observed.normalized_plate == "GJO1IB8OO0"
    result = VehicleFingerprintScorer().compare(observed, substituted)
    assert field(result, "normalized_plate").state == "mismatch"


def test_accepted_plate_observation_adapter_includes_plate_only():
    result = VehicleFingerprint.from_plate_observation(
        observation(status="accepted", plate=" gj-01 ab 1234 "))
    assert result == fingerprint("GJ01AB1234")
    assert result.vehicle_colour is None and result.vehicle_class is None


def test_review_plate_observation_adapter_omits_plate():
    result = VehicleFingerprint.from_plate_observation(
        observation(status="review", plate="GJ01AB1234"))
    assert result == VehicleFingerprint()


def test_abstained_plate_observation_adapter_omits_plate():
    result = VehicleFingerprint.from_plate_observation(
        observation(status="abstained", plate=None))
    assert result == VehicleFingerprint()


def test_accepted_observation_with_blank_plate_has_no_evidence():
    result = VehicleFingerprint.from_plate_observation(
        observation(status="accepted", plate="  "))
    assert result == VehicleFingerprint()


def test_adapter_rejects_non_observation():
    with pytest.raises(FingerprintValidationError):
        VehicleFingerprint.from_plate_observation({"status": "accepted"})
