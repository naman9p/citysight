"""Tests for Step 9 Indian plate normalization + validation."""

from phase1_anpr.normalization.plate_normalizer import (
    NormalizationResult,
    PlateNormalizer,
)


def norm():
    return PlateNormalizer()


def test_standard_plate_valid_with_state():
    r = norm().normalize("MH12AB1234")
    assert isinstance(r, NormalizationResult)
    assert r.normalized_text == "MH12AB1234"
    assert r.is_valid and r.format_type == "standard"
    assert r.state_code == "MH"


def test_standard_single_digit_rto():
    r = norm().normalize("DL1C0001")
    assert r.is_valid and r.format_type == "standard" and r.state_code == "DL"


def test_bh_series_valid():
    r = norm().normalize("22BH1234AB")
    assert r.is_valid and r.format_type == "bh_series"
    assert r.state_code is None  # national series, no state


def test_separators_and_case_are_normalized():
    r = norm().normalize(" mh-12 ab 1234 ")
    assert r.normalized_text == "MH12AB1234"
    assert r.is_valid and r.state_code == "MH"


def test_unknown_state_code_not_confidently_valid():
    r = norm().normalize("ZZ12AB1234")  # matches shape, ZZ is not a real RTO
    assert r.normalized_text == "ZZ12AB1234"
    assert not r.is_valid
    assert r.format_type is None and r.state_code is None


def test_invalid_format_text():
    r = norm().normalize("HELLO")
    assert not r.is_valid and r.format_type is None and r.state_code is None


def test_empty_text_is_safe():
    for value in ("", None):
        r = norm().normalize(value)
        assert r.normalized_text == "" and not r.is_valid
        assert r.format_type is None and r.state_code is None


def test_ambiguous_characters_are_not_substituted():
    # O/0, I/1, B/8 must be preserved exactly as OCR produced them.
    n = norm()
    assert n.normalize_text("MHO1IB8OO0") == "MHO1IB8OO0"
    # A plate with letter O where a digit belongs stays invalid (no O->0 fix).
    r = n.normalize("MH12ABO234")
    assert r.normalized_text == "MH12ABO234"
    assert not r.is_valid


def test_from_config_uses_config_values():
    config = {"normalization": {
        "allowed_chars": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "patterns": {"standard": r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$",
                     "bh_series": r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$"},
        "valid_state_codes": ["MH", "DL"],
    }}
    n = PlateNormalizer.from_config(config)
    assert n.normalize("MH12AB1234").is_valid
    assert not n.normalize("KA12AB1234").is_valid  # KA not in restricted list
