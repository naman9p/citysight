"""CitySight Phase 1 — end-to-end ANPR demo runner (Step 16, replay in Step 19).

Runs the full pipeline over recorded video using config.yaml, persisting
observations/evidence into the same SQLite + filesystem stores the API/dashboard
read from. Requires real YOLO plate weights; it fails with a clear, actionable
message (and does not download anything) when they are missing.

Supports multi-camera recorded-video replay: if config.yaml defines a `sources`
list, every source is replayed in order into the shared stores. Otherwise the
single `video` section is used (legacy behavior). ``--video`` forces a single
source; ``--source-id`` disambiguates it from other videos of the same camera.
"""

import argparse
import sys
from dataclasses import replace

from phase1_anpr.utils.config import DEFAULT_CONFIG_PATH, load_config
from phase1_anpr.video.video_reader import VideoReaderError
from phase1_anpr.persistence import (
    SQLiteObservationRepository,
    SQLiteWatchlistRepository,
    LocalFilesystemEvidenceStore,
)
from phase1_anpr.detection.detector import DetectorError
from phase1_anpr.replay import ReplaySource, load_replay_sources, run_replay


def _resolve_sources(config, args):
    """Build the source list, honoring --video / --source-id / --camera-id."""
    video_cfg = config.get("video", {})
    camera_id = args.camera_id or video_cfg.get("camera_id", "cam_01")
    if args.video:
        # An explicit video overrides any configured `sources`.
        return [ReplaySource(
            video_path=args.video, camera_id=camera_id,
            source_id=args.source_id, start_time=video_cfg.get("start_time"))]

    sources = load_replay_sources(config)
    if args.camera_id:
        sources = [replace(s, camera_id=camera_id) for s in sources]
    if args.source_id and len(sources) == 1:
        sources = [replace(sources[0], source_id=args.source_id)]
    return sources


def main(argv=None):
    parser = argparse.ArgumentParser(description="CitySight Phase 1 ANPR demo")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--video", help="Override video.input_path (single source)")
    parser.add_argument("--camera-id", help="Override video.camera_id")
    parser.add_argument("--source-id",
                        help="Source discriminator for a single video; keeps "
                             "track ids from different videos from colliding")
    args = parser.parse_args(argv)

    config = load_config(args.config)

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
        sources = _resolve_sources(config, args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        source_results = run_replay(
            config, observation_repo=obs_repo, evidence_store=evidence,
            watchlist_repo=wl_repo, sources=sources)
    except DetectorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("Add trained license-plate YOLO weights and set "
              "detection.weights_path in config.yaml, then re-run.",
              file=sys.stderr)
        return 2
    except VideoReaderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    totals = {"observations": 0, "accepted": 0, "review": 0,
              "abstained": 0, "alerts": 0}
    for sr in source_results:
        counts = sr.counts
        for key in totals:
            totals[key] += counts[key]
        label = sr.source.source_id or sr.source.camera_id
        print(f"[{label}] camera={sr.source.camera_id} "
              f"observations={counts['observations']} "
              f"accepted={counts['accepted']} review={counts['review']} "
              f"abstained={counts['abstained']} alerts={counts['alerts']}")

    print(f"sources       : {len(source_results)}")
    print(f"observations  : {totals['observations']}")
    print(f"  accepted    : {totals['accepted']}")
    print(f"  review      : {totals['review']}")
    print(f"  abstained   : {totals['abstained']}")
    print(f"alerts        : {totals['alerts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
