"""Perspective rectification of plate crops for Phase 1 (Step 7).

Deterministic, OpenCV-only. Tries to find a 4-corner plate quadrilateral via
grayscale → blur → edges → contours, orders the corners consistently, and warps
to a fronto-parallel view. If no reliable quad is found (or the crop is
invalid/tiny), it safely returns the original crop as a fallback.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# A crop smaller than this in either dimension is not worth rectifying.
MIN_SIDE = 16


@dataclass
class RectificationResult:
    """Outcome of a rectification attempt."""

    rectified_crop: np.ndarray
    success: bool
    corners: np.ndarray  # ordered (tl, tr, br, bl) float32, or None on fallback
    fallback_used: bool


def order_corners(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    ordered[0] = pts[np.argmin(s)]      # top-left: smallest x+y
    ordered[2] = pts[np.argmax(s)]      # bottom-right: largest x+y
    ordered[1] = pts[np.argmin(diff)]   # top-right: smallest y-x
    ordered[3] = pts[np.argmax(diff)]   # bottom-left: largest y-x
    return ordered


class PlateRectifier:
    """Rectify a skewed plate crop to a fronto-parallel view (OpenCV only)."""

    def __init__(self, min_area_ratio=0.2, approx_epsilon_ratio=0.02):
        # Quad must cover at least this fraction of the crop to be trusted.
        self.min_area_ratio = min_area_ratio
        self.approx_epsilon_ratio = approx_epsilon_ratio

    @staticmethod
    def _is_valid(crop):
        if crop is None or not hasattr(crop, "size") or crop.size == 0:
            return False
        if crop.ndim not in (2, 3):
            return False
        h, w = crop.shape[:2]
        return w >= MIN_SIDE and h >= MIN_SIDE

    def _find_quad(self, crop):
        """Return an ordered 4-corner array, or None if none is reliable."""
        gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        # Close small gaps so the plate border forms a single contour.
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        crop_area = crop.shape[0] * crop.shape[1]
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if area < self.min_area_ratio * crop_area:
                break  # remaining contours are only smaller
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, self.approx_epsilon_ratio * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return order_corners(approx)
        return None

    def rectify(self, crop):
        """Rectify a plate crop; always returns a RectificationResult."""
        if not self._is_valid(crop):
            return RectificationResult(
                rectified_crop=crop, success=False, corners=None, fallback_used=True
            )

        quad = self._find_quad(crop)
        if quad is None:
            return RectificationResult(
                rectified_crop=crop, success=False, corners=None, fallback_used=True
            )

        tl, tr, br, bl = quad
        # Output size from the detected quad, preserving its aspect sensibly.
        width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
        height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
        if width < MIN_SIDE or height < MIN_SIDE:
            return RectificationResult(
                rectified_crop=crop, success=False, corners=None, fallback_used=True
            )

        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(quad, dst)
        warped = cv2.warpPerspective(crop, matrix, (width, height))
        return RectificationResult(
            rectified_crop=warped, success=True, corners=quad, fallback_used=False
        )
