"""Video reading and frame sub-sampling for Phase 1 (Step 2).

Reads a video with OpenCV and yields only the frames we actually want to
process, based on a target processing FPS.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class VideoReaderError(Exception):
    """Raised when a video cannot be found or opened."""


@dataclass
class ProcessedFrame:
    """A single frame selected for processing."""

    frame: np.ndarray
    frame_number: int  # index in the source video, starting at 0
    video_timestamp: float  # seconds since the start of the video
    camera_id: str


class VideoReader:
    """Iterate over a video, yielding roughly `process_fps` frames per second.

    Example: a 30 FPS video with process_fps=5 gives a stride of 6, so every
    sixth frame is processed.
    """

    def __init__(self, video_path, camera_id, process_fps):
        self.video_path = Path(video_path)
        self.camera_id = camera_id

        if not self.video_path.exists():
            raise VideoReaderError(f"Video path does not exist: {self.video_path}")
        if not self.video_path.is_file():
            raise VideoReaderError(f"Video path is not a file: {self.video_path}")

        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            self.capture.release()
            raise VideoReaderError(f"Could not open video: {self.video_path}")

        # Some containers report 0 or NaN FPS; fall back to a sane default.
        reported_fps = self.capture.get(cv2.CAP_PROP_FPS)
        if reported_fps is None or reported_fps <= 0 or not np.isfinite(reported_fps):
            reported_fps = 30.0
        self.source_fps = float(reported_fps)

        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        self.process_fps = self._clean_process_fps(process_fps)
        self.frame_stride = self._compute_stride(self.source_fps, self.process_fps)

    def _clean_process_fps(self, process_fps):
        """Fall back to the source FPS when process_fps is missing or invalid."""
        try:
            value = float(process_fps)
        except (TypeError, ValueError):
            return self.source_fps
        if value <= 0 or not np.isfinite(value):
            return self.source_fps
        # Asking for more FPS than the source has just means "every frame".
        return min(value, self.source_fps)

    @staticmethod
    def _compute_stride(source_fps, process_fps):
        """Number of source frames to advance between processed frames."""
        return max(1, int(round(source_fps / process_fps)))

    def frames(self):
        """Yield ProcessedFrame objects for the selected frames."""
        frame_number = 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break

            if frame_number % self.frame_stride == 0:
                yield ProcessedFrame(
                    frame=frame,
                    frame_number=frame_number,
                    video_timestamp=frame_number / self.source_fps,
                    camera_id=self.camera_id,
                )

            frame_number += 1

    def release(self):
        """Release the underlying OpenCV capture."""
        if self.capture is not None:
            self.capture.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False
