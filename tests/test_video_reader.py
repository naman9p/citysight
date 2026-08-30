"""Tests for the Step 2 video reader."""

import cv2
import numpy as np
import pytest

from phase1_anpr.video.video_reader import VideoReader, VideoReaderError


def make_video(path, fps, num_frames, size=(64, 48)):
    """Write a small synthetic video and return its path."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened(), "OpenCV could not open a video writer"
    for i in range(num_frames):
        frame = np.full((size[1], size[0], 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def video_30fps(tmp_path):
    return make_video(tmp_path / "sample.mp4", fps=30, num_frames=60)


def test_missing_path_raises(tmp_path):
    with pytest.raises(VideoReaderError):
        VideoReader(tmp_path / "does_not_exist.mp4", "cam_01", 5)


def test_directory_path_raises(tmp_path):
    with pytest.raises(VideoReaderError):
        VideoReader(tmp_path, "cam_01", 5)


def test_unopenable_file_raises(tmp_path):
    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(VideoReaderError):
        VideoReader(bad, "cam_01", 5)


def test_stride_for_30fps_at_5fps(video_30fps):
    with VideoReader(video_30fps, "cam_01", 5) as reader:
        assert reader.frame_stride == 6
        numbers = [f.frame_number for f in reader.frames()]
    # 60 frames, every 6th -> 0, 6, 12, ... 54
    assert numbers == list(range(0, 60, 6))


def test_process_fps_above_source_processes_every_frame(video_30fps):
    with VideoReader(video_30fps, "cam_01", 120) as reader:
        assert reader.frame_stride == 1
        count = sum(1 for _ in reader.frames())
    assert count == 60


def test_invalid_process_fps_falls_back_to_source(video_30fps):
    for bad_value in (0, -5, None, "abc"):
        with VideoReader(video_30fps, "cam_01", bad_value) as reader:
            assert reader.frame_stride == 1


def test_frame_fields(video_30fps):
    with VideoReader(video_30fps, "cam_A", 5) as reader:
        frames = list(reader.frames())

    first, second = frames[0], frames[1]
    assert first.camera_id == "cam_A"
    assert first.frame_number == 0
    assert first.video_timestamp == pytest.approx(0.0)
    assert second.frame_number == 6
    assert second.video_timestamp == pytest.approx(6 / 30.0, abs=1e-6)
    assert first.frame.shape == (48, 64, 3)
