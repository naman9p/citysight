"""Tests for Step 7 perspective rectification (synthetic, weight-free)."""

import numpy as np
import pytest

from phase1_anpr.rectification.rectifier import (
    PlateRectifier,
    RectificationResult,
    order_corners,
)


def skewed_plate(size=(200, 300)):
    """Black background with a bright, skewed quadrilateral 'plate'."""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    quad = np.array([[60, 40], [250, 20], [270, 150], [40, 170]], dtype=np.int32)
    cv_fill(img, quad)
    return img, quad


def cv_fill(img, quad):
    import cv2
    cv2.fillPoly(img, [quad], (255, 255, 255))
    # darker inner region so borders are strong, unambiguous edges
    inner = (quad * 0.85 + np.array([30, 30]) * 0.15).astype(np.int32)
    cv2.fillPoly(img, [inner], (60, 60, 60))


def test_order_corners_consistent():
    scrambled = np.array([[270, 150], [60, 40], [40, 170], [250, 20]], dtype=np.float32)
    ordered = order_corners(scrambled)
    # tl has smallest sum, br the largest
    assert tuple(ordered[0]) == (60, 40)
    assert tuple(ordered[2]) == (270, 150)
    assert tuple(ordered[1]) == (250, 20)   # top-right
    assert tuple(ordered[3]) == (40, 170)   # bottom-left


def test_valid_skewed_plate_rectifies():
    img, _ = skewed_plate()
    result = PlateRectifier().rectify(img)
    assert isinstance(result, RectificationResult)
    assert result.success and not result.fallback_used
    assert result.corners is not None and result.corners.shape == (4, 2)


def test_output_dimensions_reasonable():
    img, _ = skewed_plate()
    result = PlateRectifier().rectify(img)
    h, w = result.rectified_crop.shape[:2]
    assert w >= 16 and h >= 16
    # A plate is wider than tall; the warped result should reflect that.
    assert w > h


def test_fallback_when_no_quad():
    noise = np.random.default_rng(0).integers(0, 60, (120, 160, 3), dtype=np.uint8)
    result = PlateRectifier(min_area_ratio=0.5).rectify(noise)
    assert result.fallback_used and not result.success
    assert result.corners is None
    assert result.rectified_crop is noise  # original returned untouched


@pytest.mark.parametrize("crop", [
    None,
    np.zeros((0, 0, 3), dtype=np.uint8),
    np.zeros((8, 8, 3), dtype=np.uint8),   # tiny
])
def test_invalid_crops_fallback(crop):
    result = PlateRectifier().rectify(crop)
    assert result.fallback_used and not result.success
    assert result.rectified_crop is crop
