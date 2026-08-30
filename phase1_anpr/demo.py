"""CitySight Phase 1 — end-to-end ANPR demo runner (Step 16).

Runs the full pipeline over a recorded video using config.yaml, persisting
observations/evidence into the same SQLite + filesystem stores the API/dashboard
read from. Requires real YOLO plate weights; it fails with a clear, actionable
message (and does not download anything) when they are missing.
"""

import argparse
import sys

from phase1_anpr.utils.config import DEFAULT_CONFIG_PATH, load_config
from phase1_anpr.video.video_reader import VideoReader, VideoReaderError
from phase1_anpr.persistence import (
    SQLiteObservationRepository,
    SQLiteWatchlistRepository,
    LocalFilesystemEvidenceStore,
)
from phase1_anpr.detection.detector import DetectorError
from phase1_anpr.pipeline.anpr_pipeline import build_pipeline_from_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="CitySight Phase 1 ANPR demo")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--video", help="Override video.input_path")
    parser.add_argument("--camera-id", help="Override video.camera_id")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    video_cfg = config["video"]
    if args.camera_id:
        video_cfg["camera_id"] = args.camera_id
    video_path = args.video or video_cfg["input_path"]

    persistence = config.get("persistence", {})
    obs_repo = SQLiteObservationRepository(
        persistence.get("db_path", "outputs/observations/observations.db"))
    wl_repo = SQLiteWatchlistRepository(
        persistence.get("watchlist_db_path",
                        persistence.get("db_path",
                                        "outputs/observations/observations.db")))
    evidence = LocalFilesystemEvidenceStore(
        persistence.get("evidence_dir", "outputs/plates/evidence"))

    try:
        pipeline = build_pipeline_from_config(
            config, observation_repo=obs_repo, evidence_store=evidence,
            watchlist_repo=wl_repo)
    except DetectorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("Add trained license-plate YOLO weights and set "
              "detection.weights_path in config.yaml, then re-run.",
              file=sys.stderr)
        return 2

    try:
        reader = VideoReader(video_path, video_cfg["camera_id"],
                             video_cfg.get("process_fps"))
    except VideoReaderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    results = pipeline.run(reader)
    accepted = sum(1 for r in results if r.observation.status == "accepted")
    review = sum(1 for r in results if r.observation.status == "review")
    abstained = sum(1 for r in results if r.observation.status == "abstained")
    alerts = sum(len(r.alerts) for r in results)

    print(f"camera_id     : {video_cfg['camera_id']}")
    print(f"observations  : {len(results)}")
    print(f"  accepted    : {accepted}")
    print(f"  review      : {review}")
    print(f"  abstained   : {abstained}")
    print(f"alerts        : {alerts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
