"""CitySight Phase 1 — ANPR Engine entry point.

Step 2: read a video and iterate the frames selected for processing.
Detection, tracking, OCR and fusion are wired in later steps.
"""

import argparse
import sys

from phase1_anpr.utils.config import DEFAULT_CONFIG_PATH, load_config
from phase1_anpr.video.video_reader import VideoReader, VideoReaderError


def main():
    parser = argparse.ArgumentParser(description="CitySight Phase 1 ANPR Engine")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config.yaml",
    )
    parser.add_argument("--video", help="Override video.input_path from config")
    parser.add_argument("--camera-id", help="Override video.camera_id from config")
    args = parser.parse_args()

    config = load_config(args.config)
    video_config = config["video"]

    video_path = args.video or video_config["input_path"]
    camera_id = args.camera_id or video_config["camera_id"]
    process_fps = video_config.get("process_fps")

    try:
        reader = VideoReader(video_path, camera_id, process_fps)
    except VideoReaderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    processed = 0
    last_timestamp = 0.0
    with reader:
        for processed_frame in reader.frames():
            processed += 1
            last_timestamp = processed_frame.video_timestamp

    print(f"camera_id      : {camera_id}")
    print(f"source_fps     : {reader.source_fps:.2f}")
    print(f"process_fps    : {reader.process_fps:.2f}")
    print(f"frame_stride   : {reader.frame_stride}")
    print(f"frames_processed: {processed}")
    print(f"last_timestamp : {last_timestamp:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
