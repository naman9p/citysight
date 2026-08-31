"""Step 19 tests: multi-camera recorded-video replay orchestration + config.

Uses fake pipeline/reader factories so no YOLO/Paddle weights or video files
are needed — mirrors the dependency-injected design of `phase1_anpr.replay`.
"""

from types import SimpleNamespace

import pytest

from phase1_anpr.replay import (
    ReplaySource,
    SourceReplayResult,
    load_replay_sources,
    replay,
)


# --- fakes --------------------------------------------------------------------

class _FakePipeline:
    def __init__(self, results):
        self._results = results

    def run(self, reader):
        return self._results


def _result(status="accepted", alerts=0):
    return SimpleNamespace(
        observation=SimpleNamespace(status=status),
        alerts=[object()] * alerts,
    )


# --- load_replay_sources ------------------------------------------------------

def test_load_sources_legacy_single_when_no_sources():
    cfg = {"video": {"input_path": "a.mp4", "camera_id": "cam_01",
                     "start_time": "2026-08-31T00:00:00Z"}}
    sources = load_replay_sources(cfg)
    assert len(sources) == 1
    assert sources[0].source_id is None  # legacy id derivation preserved
    assert sources[0].video_path == "a.mp4"
    assert sources[0].camera_id == "cam_01"
    assert sources[0].start_time == "2026-08-31T00:00:00Z"


def test_load_sources_from_list_defaults_source_id_to_stem():
    cfg = {"video": {"camera_id": "camX"},
           "sources": [{"video": "inputs/cam_01_morning.mp4"}]}
    sources = load_replay_sources(cfg)
    assert len(sources) == 1
    assert sources[0].source_id == "cam_01_morning"  # from file stem
    assert sources[0].camera_id == "camX"  # falls back to video.camera_id


def test_load_sources_explicit_fields():
    cfg = {"sources": [
        {"video": "a.mp4", "camera_id": "c1", "source_id": "s1",
         "start_time": "2026-08-31T00:00:00Z"}]}
    s = load_replay_sources(cfg)[0]
    assert (s.camera_id, s.source_id) == ("c1", "s1")
    assert s.start_time == "2026-08-31T00:00:00Z"


def test_load_sources_duplicate_camera_source_rejected():
    cfg = {"sources": [
        {"video": "a.mp4", "camera_id": "c1", "source_id": "s1"},
        {"video": "b.mp4", "camera_id": "c1", "source_id": "s1"}]}
    with pytest.raises(ValueError, match="unique"):
        load_replay_sources(cfg)


def test_load_sources_missing_video_rejected():
    with pytest.raises(ValueError, match="video"):
        load_replay_sources({"sources": [{"camera_id": "c1"}]})


def test_load_sources_not_a_list_rejected():
    with pytest.raises(ValueError, match="list"):
        load_replay_sources({"sources": {"video": "a.mp4"}})


def test_load_sources_entry_not_mapping_rejected():
    with pytest.raises(ValueError, match="mapping"):
        load_replay_sources({"sources": ["a.mp4"]})


# --- replay orchestration -----------------------------------------------------

def test_replay_runs_each_source_in_order():
    sources = [ReplaySource("a.mp4", "c1", "s1"),
               ReplaySource("b.mp4", "c1", "s2")]
    built, readers = [], []

    def pf(src):
        built.append(src)
        return _FakePipeline([_result("accepted", alerts=1)])

    def rf(src):
        readers.append(src)
        return object()

    out = replay(sources, pf, rf)
    assert [s.source_id for s in built] == ["s1", "s2"]  # fresh pipeline each
    assert len(readers) == 2
    assert len(out) == 2
    assert out[0].source.source_id == "s1"
    assert out[0].counts["observations"] == 1
    assert out[0].counts["accepted"] == 1
    assert out[0].counts["alerts"] == 1


def test_source_replay_result_counts_tally():
    res = [_result("accepted", 1), _result("review"), _result("abstained"),
           _result("accepted")]
    srr = SourceReplayResult(source=ReplaySource("a.mp4", "c1", "s1"),
                             results=res)
    assert srr.counts == {"observations": 4, "accepted": 2, "review": 1,
                          "abstained": 1, "alerts": 1}
