"""Tests for Step 4 camera-local tracking using real Ultralytics BYTETracker.

All synthetic — no YOLO weights are loaded.
"""

from phase1_anpr.detection.detector import Detection
from phase1_anpr.tracking.tracker import PlateTracker


def det(bbox, conf=0.9, frame=0):
    return Detection(bbox=bbox, confidence=conf, class_id=0, frame_number=frame)


def only_track(tracker):
    assert len(tracker.active) == 1
    return next(iter(tracker.active.values()))


def test_consecutive_frames_keep_one_track_id():
    tr = PlateTracker()
    ids = set()
    for f in range(4):
        touched = tr.update([det((10, 10, 60, 50), frame=f)], f)
        ids.update(t.track_id for t in touched)
    track = only_track(tr)
    assert ids == {track.track_id}
    assert track.first_seen == 0 and track.last_seen == 3


def test_two_plates_get_separate_ids():
    tr = PlateTracker()
    seen = set()
    for f in range(3):
        dets = [det((10, 10, 60, 50), frame=f), det((300, 300, 360, 350), frame=f)]
        for t in tr.update(dets, f):
            seen.add(t.track_id)
    assert len(tr.active) == 2
    assert len(seen) == 2


def test_empty_frame_is_safe():
    tr = PlateTracker()
    assert tr.update([], 0) == []
    # A detection after an empty frame still tracks fine.
    tr.update([det((10, 10, 60, 50), frame=1)], 1)
    tr.update([det((10, 10, 60, 50), frame=2)], 2)
    assert len(tr.active) == 1


def test_stale_track_is_finalized():
    tr = PlateTracker(max_age=5)
    for f in range(3):
        tr.update([det((10, 10, 60, 50), frame=f)], f)
    # Long gap with a far-away plate: the first track goes stale.
    tr.update([det((400, 400, 460, 450), frame=20)], 20)
    assert any(t.first_seen == 0 for t in tr.finished)


def test_finalize_flushes_active():
    tr = PlateTracker()
    for f in range(2):
        tr.update([det((10, 10, 60, 50), frame=f)], f)
    finished = tr.finalize()
    assert len(finished) >= 1 and not tr.active


def test_candidate_crops_are_capped():
    tr = PlateTracker(max_candidates=3)
    import numpy as np
    crop = np.zeros((4, 4, 3), dtype=np.uint8)
    for f in range(6):
        tr.update([det((10, 10, 60, 50), frame=f)], f, crops=[crop])
    track = only_track(tr)
    assert len(track.candidates) == 3


def test_independent_tracker_instances():
    tr1 = PlateTracker()
    for f in range(2):
        tr1.update([det((10, 10, 60, 50), frame=f)], f)
    tr2 = PlateTracker()
    assert tr2.active == {} and tr2.finished == []
    assert tr1 is not tr2
