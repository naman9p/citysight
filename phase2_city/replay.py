"""Phase 2 multi-camera replay CLI + orchestration (Step 19).

Loads a validated replay scenario (see ``phase2_city.scenario``), then delegates
to the existing ``phase1_anpr.replay`` orchestration — one fresh pipeline per
source, shared observation/evidence/watchlist stores. No detection/OCR/tracking
logic is duplicated here.

CLI::

    python -m phase2_city.replay \\
        --scenario phase2_city/config/replay.example.yaml \\
        --city-config phase2_city/config/city.yaml \\
        --config phase1_anpr/config/config.yaml

Persistence/evidence locations come from the Phase 1 ``config.yaml`` so the
scenario writes into the same stores the API/dashboard read from.
"""

import argparse
import sys

from phase1_anpr.utils.config import DEFAULT_CONFIG_PATH, load_config
from phase1_anpr.detection.detector import DetectorError
from phase1_anpr.video.video_reader import VideoReaderError
from phase1_anpr.persistence import (
    SQLiteObservationRepository,
    SQLiteWatchlistRepository,
    LocalFilesystemEvidenceStore,
)
from phase2_city.config.loader import DEFAULT_CITY_CONFIG_PATH, load_city_config
from phase2_city.graph import CityCameraGraph
from phase2_city.scenario import ScenarioError, ScenarioResult, load_scenario


def run_scenario(scenario, config, observation_repo, evidence_store=None,
                 watchlist_repo=None, *, replay_fn=None) -> ScenarioResult:
    """Replay every scenario source into the shared stores, in order.

    Maps each validated ``ScenarioSource`` to a phase1 ``ReplaySource`` and
    delegates to ``phase1_anpr.replay.run_replay`` (real components). Tests may
    inject ``replay_fn`` to avoid YOLO/PaddleOCR. Returns a ``ScenarioResult``.
    """
    from phase1_anpr.replay import ReplaySource, run_replay

    sources = [ReplaySource(video_path=s.video_path, camera_id=s.camera_id,
                            source_id=s.source_id, start_time=s.start_time)
               for s in scenario.sources]
    replay_fn = replay_fn or run_replay
    source_results = replay_fn(
        config, observation_repo=observation_repo,
        evidence_store=evidence_store, watchlist_repo=watchlist_repo,
        sources=sources)
    return ScenarioResult(scenario_id=scenario.scenario_id,
                          source_results=source_results)


def _build_stores(config):
    """Build the shared SQLite/evidence stores from the Phase 1 config."""
    persistence = config.get("persistence", {})
    db_path = persistence.get("db_path", "outputs/observations/observations.db")
    obs_repo = SQLiteObservationRepository(db_path)
    wl_repo = SQLiteWatchlistRepository(
        persistence.get("watchlist_db_path", db_path))
    evidence = LocalFilesystemEvidenceStore(
        persistence.get("evidence_dir", "outputs/plates/evidence"))
    return obs_repo, wl_repo, evidence


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="CitySight Phase 2 multi-camera recorded-video replay")
    parser.add_argument("--scenario", required=True,
                        help="Path to the replay scenario YAML")
    parser.add_argument("--city-config", default=str(DEFAULT_CITY_CONFIG_PATH),
                        help="Step 17 city topology config")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="Phase 1 pipeline config (persistence/thresholds)")
    args = parser.parse_args(argv)

    config = load_config(args.config)

    try:
        cameras, links = load_city_config(args.city_config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    graph = CityCameraGraph(cameras, links)

    # Full fail-fast preflight: nothing is processed unless every source is valid.
    try:
        scenario = load_scenario(args.scenario, graph)
    except (ScenarioError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    obs_repo, wl_repo, evidence = _build_stores(config)

    try:
        result = run_scenario(scenario, config, observation_repo=obs_repo,
                              evidence_store=evidence, watchlist_repo=wl_repo)
    except DetectorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("Add trained license-plate YOLO weights and set "
              "detection.weights_path in config.yaml, then re-run.",
              file=sys.stderr)
        return 2
    except VideoReaderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"scenario_id   : {result.scenario_id}")
    for sr in result.source_results:
        counts = sr.counts
        print(f"[{sr.source.source_id}] camera={sr.source.camera_id} "
              f"observations={counts['observations']} "
              f"accepted={counts['accepted']} review={counts['review']} "
              f"abstained={counts['abstained']} alerts={counts['alerts']}")

    totals = result.totals
    print(f"sources       : {len(result.source_results)}")
    print(f"observations  : {totals['observations']}")
    print(f"  accepted    : {totals['accepted']}")
    print(f"  review      : {totals['review']}")
    print(f"  abstained   : {totals['abstained']}")
    print(f"alerts        : {totals['alerts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
