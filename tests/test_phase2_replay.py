"""Step 19 completion: Phase 2 multi-camera replay scenario validation + summary.

Uses temp scenario YAML + dummy video files and a fake replay function, so no
YOLO/PaddleOCR weights or real video decoding are needed.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from phase2_city.graph import CityCameraGraph
from phase2_city.models import Camera
from phase2_city.scenario import ScenarioError, load_scenario
from phase2_city.replay import run_scenario


# --- fixtures / helpers -------------------------------------------------------

def _graph():
    cams = [
        Camera(camera_id="CAM_01", name="Junction 1", latitude=28.61,
               longitude=77.22, road_name="Ring Road", heading_deg=90.0),
        Camera(camera_id="CAM_02", name="Junction 2", latitude=28.62,
               longitude=77.24, road_name="Ring Road", heading_deg=90.0),
    ]
    return CityCameraGraph(cams, [])


def _valid_data():
    return {
        "scenario_id": "demo-city-01",
        "sources": [
            {"source_id": "s1", "camera_id": "CAM_01",
             "video_path": "videos/cam01.mp4",
             "start_time": "2026-08-31T10:00:00+05:30"},
            {"source_id": "s2", "camera_id": "CAM_02",
             "video_path": "videos/cam02.mp4",
             "start_time": "2026-08-31T04:30:00Z"},
        ],
    }


def _write_scenario(tmp_path, data, videos=("cam01.mp4", "cam02.mp4")):
    vids = tmp_path / "videos"
    vids.mkdir(exist_ok=True)
    for name in videos:
        (vids / name).write_bytes(b"\x00")  # existence-only stub
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- valid loads --------------------------------------------------------------

def test_valid_scenario_loads(tmp_path):
    sc = load_scenario(_write_scenario(tmp_path, _valid_data()), _graph())
    assert sc.scenario_id == "demo-city-01"
    assert [s.source_id for s in sc.sources] == ["s1", "s2"]
    assert sc.sources[0].camera_id == "CAM_01"
    assert Path(sc.sources[0].video_path).is_absolute()
    assert Path(sc.sources[0].video_path).exists()


def test_valid_offset_start_time_accepted(tmp_path):
    d = _valid_data()
    d["sources"] = [d["sources"][0]]  # "+05:30"
    sc = load_scenario(_write_scenario(tmp_path, d), _graph())
    assert sc.sources[0].start_time.utcoffset().total_seconds() == 0  # stored UTC


def test_valid_utc_z_start_time_accepted(tmp_path):
    d = _valid_data()
    d["sources"] = [d["sources"][1]]  # "Z", CAM_02
    sc = load_scenario(_write_scenario(tmp_path, d), _graph())
    assert sc.sources[0].camera_id == "CAM_02"


# --- structure ----------------------------------------------------------------

def test_scenario_id_missing_rejected(tmp_path):
    d = _valid_data(); del d["scenario_id"]
    with pytest.raises(ScenarioError, match="scenario_id"):
        load_scenario(_write_scenario(tmp_path, d), _graph())


def test_zero_sources_rejected(tmp_path):
    d = _valid_data(); d["sources"] = []
    with pytest.raises(ScenarioError, match="at least one source"):
        load_scenario(_write_scenario(tmp_path, d), _graph())


# --- source_id ----------------------------------------------------------------

def test_source_id_missing_rejected(tmp_path):
    d = _valid_data(); del d["sources"][0]["source_id"]
    with pytest.raises(ScenarioError, match="source_id"):
        load_scenario(_write_scenario(tmp_path, d), _graph())


def test_source_id_empty_rejected(tmp_path):
    d = _valid_data(); d["sources"][0]["source_id"] = ""
    with pytest.raises(ScenarioError, match="source_id"):
        load_scenario(_write_scenario(tmp_path, d), _graph())


def test_source_id_whitespace_rejected(tmp_path):
    d = _valid_data(); d["sources"][0]["source_id"] = "   "
    with pytest.raises(ScenarioError, match="source_id"):
        load_scenario(_write_scenario(tmp_path, d), _graph())


def test_duplicate_source_id_rejected(tmp_path):
    d = _valid_data(); d["sources"][1]["source_id"] = "s1"
    with pytest.raises(ScenarioError, match="duplicate source_id"):
        load_scenario(_write_scenario(tmp_path, d), _graph())


# --- camera_id ----------------------------------------------------------------

def test_camera_id_missing_rejected(tmp_path):
    d = _valid_data(); del d["sources"][0]["camera_id"]
    with pytest.raises(ScenarioError, match="camera_id"):
        load_scenario(_write_scenario(tmp_path, d), _graph())


def test_unknown_camera_rejected(tmp_path):
    d = _valid_data(); d["sources"][0]["camera_id"] = "CAM_99"
    with pytest.raises(ScenarioError, match="CAM_99"):
        load_scenario(_write_scenario(tmp_path, d), _graph())


# --- video_path ---------------------------------------------------------------

def test_video_path_missing_rejected(tmp_path):
    d = _valid_data(); del d["sources"][0]["video_path"]
    with pytest.raises(ScenarioError, match="video_path"):
        load_scenario(_write_scenario(tmp_path, d), _graph())


def test_missing_video_file_rejected(tmp_path):
    d = _valid_data(); d["sources"][0]["video_path"] = "videos/nope.mp4"
    with pytest.raises(ScenarioError, match="video file not found"):
        load_scenario(_write_scenario(tmp_path, d), _graph())


def test_relative_video_path_resolved_to_scenario_dir(tmp_path):
    d = _valid_data(); d["sources"] = [d["sources"][0]]
    sc = load_scenario(_write_scenario(tmp_path, d), _graph())
    assert Path(sc.sources[0].video_path) == (tmp_path / "videos" / "cam01.mp4").resolve()


# --- start_time ---------------------------------------------------------------

def test_start_time_missing_rejected(tmp_path):
    d = _valid_data(); del d["sources"][0]["start_time"]
    with pytest.raises(ScenarioError, match="start_time"):
        load_scenario(_write_scenario(tmp_path, d), _graph())


def test_start_time_null_rejected(tmp_path):
    d = _valid_data(); d["sources"][0]["start_time"] = None
    with pytest.raises(ScenarioError, match="start_time"):
        load_scenario(_write_scenario(tmp_path, d), _graph())


def test_start_time_naive_rejected(tmp_path):
    d = _valid_data(); d["sources"][0]["start_time"] = "2026-08-31T10:00:00"
    with pytest.raises(ScenarioError, match="timezone"):
        load_scenario(_write_scenario(tmp_path, d), _graph())


def test_start_time_malformed_rejected(tmp_path):
    d = _valid_data(); d["sources"][0]["start_time"] = "not-a-timestamp"
    with pytest.raises(ScenarioError, match="ISO-8601"):
        load_scenario(_write_scenario(tmp_path, d), _graph())


# --- fail-fast ordering + aggregation ----------------------------------------

def test_invalid_third_source_prevents_processing(tmp_path):
    calls = []

    def spy_replay(*a, **k):
        calls.append(1)
        return []

    d = _valid_data()
    d["sources"].append({"source_id": "s3", "camera_id": "CAM_99",
                         "video_path": "videos/cam01.mp4",
                         "start_time": "2026-08-31T10:00:00+05:30"})
    path = _write_scenario(tmp_path, d)
    with pytest.raises(ScenarioError):
        scenario = load_scenario(path, _graph())  # raises before run_scenario
        run_scenario(scenario, config={}, observation_repo=None,
                     replay_fn=spy_replay)
    assert calls == []  # source 1 was never processed


def test_source_order_preserved(tmp_path):
    d = _valid_data()
    d["sources"].append({"source_id": "s3", "camera_id": "CAM_01",
                         "video_path": "videos/cam01.mp4",
                         "start_time": "2026-08-31T11:00:00+05:30"})
    sc = load_scenario(_write_scenario(tmp_path, d), _graph())
    assert [s.source_id for s in sc.sources] == ["s1", "s2", "s3"]


def test_scenario_aggregate_summary_correct(tmp_path):
    scenario = load_scenario(_write_scenario(tmp_path, _valid_data()), _graph())

    fake_results = [
        SimpleNamespace(source=SimpleNamespace(source_id="s1", camera_id="CAM_01"),
                        counts={"observations": 3, "accepted": 2, "review": 1,
                                "abstained": 0, "alerts": 1}),
        SimpleNamespace(source=SimpleNamespace(source_id="s2", camera_id="CAM_02"),
                        counts={"observations": 2, "accepted": 1, "review": 0,
                                "abstained": 1, "alerts": 0}),
    ]
    captured = {}

    def spy_replay(config, *, observation_repo, evidence_store,
                   watchlist_repo, sources):
        captured["sources"] = sources
        return fake_results

    result = run_scenario(scenario, config={}, observation_repo=None,
                          replay_fn=spy_replay)
    assert result.scenario_id == "demo-city-01"
    assert result.source_results is fake_results
    assert result.totals == {"observations": 5, "accepted": 3, "review": 1,
                             "abstained": 1, "alerts": 1}
    # scenario sources are mapped to phase1 ReplaySource in order
    assert [s.source_id for s in captured["sources"]] == ["s1", "s2"]
    assert [s.camera_id for s in captured["sources"]] == ["CAM_01", "CAM_02"]
