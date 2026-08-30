"""Tests for Step 5 quality scoring (synthetic crops, no weights/video)."""

import numpy as np
import pytest

from phase1_anpr.quality.quality_scorer import QualityScorer, ScoredCrop


def noisy_crop(h=80, w=160, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def flat_crop(value=128, h=80, w=160):
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_invalid_crops_rejected():
    sc = QualityScorer(min_plate_width=10, min_plate_height=10)
    assert not sc.is_valid(None)
    assert not sc.is_valid(np.zeros((0, 0, 3), dtype=np.uint8))
    assert not sc.is_valid(np.zeros((5, 5, 3), dtype=np.uint8))  # too tiny
    assert sc.score(None) == 0.0


def test_score_in_unit_range():
    sc = QualityScorer()
    for crop in (noisy_crop(), flat_crop(), flat_crop(0), flat_crop(255)):
        s = sc.score(crop)
        assert 0.0 <= s <= 1.0


def test_score_is_deterministic():
    sc = QualityScorer()
    crop = noisy_crop(seed=42)
    assert sc.score(crop) == sc.score(crop.copy())


def test_sharp_beats_flat():
    sc = QualityScorer()
    assert sc.score(noisy_crop(seed=1)) > sc.score(flat_crop(128))


def test_larger_scores_higher_size_factor():
    sc = QualityScorer()
    small = flat_crop(128, h=20, w=40)
    large = flat_crop(128, h=100, w=200)
    assert sc.score(large) > sc.score(small)


def test_select_best_returns_top_k_sorted():
    sc = QualityScorer(top_k=3)
    candidates = [
        (0, flat_crop(128), 0.5),      # low sharpness
        (1, noisy_crop(seed=1), 0.9),  # high
        (2, noisy_crop(seed=2), 0.8),  # high
        (3, flat_crop(0), 0.4),        # low
        (4, None, 0.99),               # invalid -> skipped
    ]
    best = sc.select_best(candidates)
    assert len(best) == 3
    assert all(isinstance(b, ScoredCrop) for b in best)
    scores = [b.quality_score for b in best]
    assert scores == sorted(scores, reverse=True)
    # provenance preserved
    assert best[0].detector_confidence in (0.9, 0.8)
    assert 4 not in [b.frame_number for b in best]


def test_from_config_reads_weights():
    config = {"quality": {"sharpness_weight": 0.4, "size_weight": 0.3,
                          "brightness_weight": 0.15, "contrast_weight": 0.15,
                          "target_area": 8000, "top_k": 2}}
    sc = QualityScorer.from_config(config)
    assert sc.top_k == 2
    assert sc.weights["sharpness_weight"] == 0.4
