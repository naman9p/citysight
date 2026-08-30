"""Camera-local plate tracking for Phase 1 (Step 4).

Uses Ultralytics' real BYTETracker for detection association, without loading a
YOLO model. Our Detection objects are adapted into the minimal "results-like"
shape BYTETracker.update() expects (see the installed ultralytics 8.4.x API:
`.conf`, `.xywh`, `.cls`, length, and boolean indexing). On top of that we keep
a small per-camera metadata layer: track_id, first_seen, last_seen, capped
candidate crops, detector_confidence, frame_number, plus stale/finalize logic.
"""

from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from ultralytics.trackers.byte_tracker import BYTETracker


class _DetectionResults:
    """Minimal Results-like adapter over our Detection list for BYTETracker.

    BYTETracker reads `.conf`, `.xywh` (center x, y, w, h), `.cls`, `len()`, and
    boolean-mask indexing. Each row keeps its original detection index so we can
    map tracks back to the crop that produced them.
    """

    def __init__(self, conf, xywh, cls):
        self.conf = np.asarray(conf, dtype=np.float32)
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.cls = np.asarray(cls, dtype=np.float32)

    @classmethod
    def from_detections(cls, detections):
        conf, xywh, classes = [], [], []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            conf.append(d.confidence)
            xywh.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1])
            classes.append(d.class_id)
        return cls(conf, xywh, classes)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return _DetectionResults(self.conf[mask], self.xywh[mask], self.cls[mask])


def _default_args(track_buffer, match_thresh):
    """BYTETracker argument namespace matching ultralytics' bytetrack.yaml."""
    return SimpleNamespace(
        tracker_type="bytetrack",
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        fuse_score=True,
    )


@dataclass
class Track:
    """State for one tracked plate within a camera session."""

    track_id: int
    first_seen: int          # frame number first observed
    last_seen: int           # frame number last observed
    bbox: tuple              # most recent bbox (x1, y1, x2, y2)
    detector_confidence: float
    frame_number: int        # most recent frame number
    candidates: deque = field(default_factory=deque)  # recent (frame, crop, conf)


class PlateTracker:
    """Associate per-frame detections into camera-local tracks via BYTETracker."""

    def __init__(self, track_buffer=30, match_thresh=0.8, max_age=30,
                 max_candidates=10):
        self.max_age = max_age          # frames a track may go unseen before staling
        self.max_candidates = max_candidates
        self._bt = BYTETracker(_default_args(track_buffer, match_thresh))
        self.active = {}                # track_id -> Track
        self.finished = []              # finalized tracks

    def update(self, detections, frame_number, crops=None):
        """Feed detections to BYTETracker; returns the tracks touched this frame."""
        crops = crops or [None] * len(detections)
        results = _DetectionResults.from_detections(detections)
        tracked = self._bt.update(results)  # (M, 8): xyxy, id, score, cls, idx

        touched = []
        for row in tracked:
            x1, y1, x2, y2 = (int(v) for v in row[:4])
            track_id = int(row[4])
            score = float(row[5])
            det_index = int(row[7])
            crop = crops[det_index] if 0 <= det_index < len(crops) else None
            touched.append(
                self._record(track_id, (x1, y1, x2, y2), score, frame_number, crop)
            )

        self._retire_stale(frame_number)
        return touched

    def _record(self, track_id, bbox, score, frame_number, crop):
        track = self.active.get(track_id)
        if track is None:
            track = Track(
                track_id=track_id,
                first_seen=frame_number,
                last_seen=frame_number,
                bbox=bbox,
                detector_confidence=score,
                frame_number=frame_number,
                candidates=deque(maxlen=self.max_candidates),
            )
            self.active[track_id] = track
        else:
            track.last_seen = frame_number
            track.frame_number = frame_number
            track.bbox = bbox
            track.detector_confidence = score
        if crop is not None:
            track.candidates.append((frame_number, crop, score))
        return track

    def _retire_stale(self, frame_number):
        stale = [
            tid for tid, t in self.active.items()
            if frame_number - t.last_seen > self.max_age
        ]
        for tid in stale:
            self.finished.append(self.active.pop(tid))

    def finalize(self):
        """Flush all remaining active tracks and return every finished track."""
        self.finished.extend(self.active.values())
        self.active.clear()
        return self.finished
