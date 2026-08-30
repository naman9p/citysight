"""YOLO license-plate detection for Phase 1 (Step 3).

Wraps an Ultralytics YOLO model trained specifically for license plates.
Weights come from config.yaml — the standard COCO model is not used here.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class DetectorError(Exception):
    """Raised for missing weights or model load failures."""


@dataclass
class Detection:
    """A single license-plate detection on one frame."""

    bbox: tuple  # (x1, y1, x2, y2) in original-frame pixels
    confidence: float
    class_id: int
    frame_number: int


def _resolve_device(device):
    """Return the device to run on, auto-selecting CUDA when available."""
    requested = (device or "auto").lower()
    if requested in ("cuda", "gpu"):
        return "cuda"
    if requested == "cpu":
        return "cpu"
    # auto
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class PlateDetector:
    """Detect license plates in frames using a custom-trained YOLO model."""

    def __init__(self, weights_path, device="auto", confidence_threshold=0.35,
                 iou_threshold=0.45, plate_class_id=0, image_size=640,
                 crops_dir=None, save_crops=False):
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise DetectorError(
                f"Detector weights not found: {self.weights_path}. "
                "Set detection.weights_path in config.yaml to your trained "
                "license-plate model."
            )

        self.device = _resolve_device(device)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.plate_class_id = plate_class_id
        self.image_size = image_size
        self.save_crops = save_crops
        self.crops_dir = Path(crops_dir) if crops_dir else None
        if self.save_crops and self.crops_dir:
            self.crops_dir.mkdir(parents=True, exist_ok=True)

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - import guard
            raise DetectorError(
                "ultralytics is not installed; run pip install -r requirements.txt"
            ) from exc

        try:
            self.model = YOLO(str(self.weights_path))
        except Exception as exc:
            raise DetectorError(f"Failed to load YOLO model: {exc}") from exc

    @classmethod
    def from_config(cls, config):
        """Build a detector from the loaded config dict."""
        det = config["detection"]
        out = config.get("output", {})
        return cls(
            weights_path=det["weights_path"],
            device=det.get("device", "auto"),
            confidence_threshold=det.get("confidence_threshold", 0.35),
            iou_threshold=det.get("iou_threshold", 0.45),
            plate_class_id=det.get("plate_class_id", 0),
            image_size=det.get("image_size", 640),
            crops_dir=out.get("plates_dir"),
            save_crops=out.get("save_plate_crops", False),
        )

    def detect(self, frame, frame_number):
        """Run detection on one frame and return a list of Detection objects."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results, frame_number)

    def _parse_results(self, results, frame_number):
        detections = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id != self.plate_class_id:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                detections.append(
                    Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=float(box.conf[0]),
                        class_id=class_id,
                        frame_number=frame_number,
                    )
                )
        return detections

    @staticmethod
    def crop(frame, detection):
        """Crop a plate from the original-resolution frame (clamped to bounds)."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = detection.bbox
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        return frame[y1:y2, x1:x2].copy()

    def save_crop(self, crop, frame_number, index):
        """Optionally persist a plate crop for debugging; returns the path or None."""
        if not (self.save_crops and self.crops_dir):
            return None
        if crop is None or crop.size == 0:
            return None
        path = self.crops_dir / f"frame{frame_number:06d}_plate{index:02d}.png"
        cv2.imwrite(str(path), crop)
        return path
