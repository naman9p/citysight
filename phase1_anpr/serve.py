"""CitySight — start the read/watchlist API + /dashboard (final demo).

Serves the existing stdlib API against the SAME SQLite database used by the demo
pipeline (phase1_anpr.demo), so the dashboard shows real persisted observations,
watchlist entries, and alerts. No new dependencies, no feature changes.

When the Phase 2 city topology config exists, the trajectory reconstruction
endpoint (POST /v1/trajectories) and camera topology endpoints (GET /v1/cameras,
GET /v1/cameras/{id}, GET /v1/cameras/{id}/links, GET /v1/links) are enabled
automatically. If the file is absent, the server starts without Phase 2 support.
If it is present but malformed, startup fails with a clear error.

Step 42 optionally opens configured Phase 3 evidence/query SQLite files in
SQLite read-only mode. Omitted or disabled configuration leaves existing
Phase 1/2 startup unchanged.
"""

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

from phase1_anpr.api import create_server
from phase1_anpr.utils.config import DEFAULT_CONFIG_PATH, load_config
from phase1_anpr.persistence import (
    SQLiteObservationRepository,
    SQLiteWatchlistRepository,
)
from phase2_city.config.loader import DEFAULT_CITY_CONFIG_PATH, load_city_config
from phase2_city.graph import CityCameraGraph
from phase2_city.trajectory import TrajectoryReconstructor


def build_city_graph(city_config_path):
    """Build a ``CityCameraGraph`` if the city topology config exists.

    Returns ``None`` when the file does not exist (graceful degradation).
    **Raises** if the file exists but is malformed or invalid (deployment error).
    """
    path = Path(city_config_path)
    if not path.exists():
        return None
    cameras, links = load_city_config(city_config_path)
    return CityCameraGraph(cameras, links)


def build_trajectory_reconstructor(obs_repo, city_config_path, config,
                                   *, city_graph=None):
    """Build a ``TrajectoryReconstructor`` if the city topology config exists.

    Returns ``None`` when the file does not exist (graceful degradation).
    **Raises** if the file exists but is malformed or invalid (deployment error).

    When ``city_graph`` is provided, it is reused instead of building a new one
    from the config file. This avoids double-construction when the caller already
    built the graph (e.g. serve.py main).
    """
    if city_graph is not None:
        return TrajectoryReconstructor.from_config(obs_repo, city_graph, config)
    path = Path(city_config_path)
    if not path.exists():
        return None
    cameras, links = load_city_config(city_config_path)
    graph = CityCameraGraph(cameras, links)
    return TrajectoryReconstructor.from_config(obs_repo, graph, config)


def build_phase3_read_api(config):
    """Build the opt-in Step 42 service from the ordinary YAML config."""
    settings = config.get("phase3_api")
    if settings is None:
        return None
    if not isinstance(settings, Mapping):
        raise ValueError("phase3_api must be a mapping")
    enabled = settings.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("phase3_api.enabled must be bool")
    if not enabled:
        return None

    from phase3_city.api import open_phase3_read_api
    from phase3_city.trajectory_hypotheses import TrajectoryHypothesisPolicy

    evidence_path = settings.get("evidence_db_path")
    query_path = settings.get("query_history_db_path")
    for name, value in (
            ("evidence_db_path", evidence_path),
            ("query_history_db_path", query_path)):
        if value is not None and (
                not isinstance(value, str) or not value.strip()):
            raise ValueError(f"phase3_api.{name} must be a path or null")
    if evidence_path is None and query_path is None:
        raise ValueError(
            "enabled phase3_api requires at least one database path")

    raw_policy = settings.get("hypothesis_policy")
    policy = None
    if raw_policy is not None:
        if not isinstance(raw_policy, Mapping):
            raise ValueError("phase3_api.hypothesis_policy must be a mapping")
        expected = {
            "max_hops", "max_hypotheses", "max_branching_per_event"}
        if set(raw_policy) != expected:
            raise ValueError(
                "phase3_api.hypothesis_policy requires max_hops, "
                "max_hypotheses and max_branching_per_event")
        policy = TrajectoryHypothesisPolicy(
            max_hops=raw_policy["max_hops"],
            max_hypotheses=raw_policy["max_hypotheses"],
            max_branching_per_event=raw_policy[
                "max_branching_per_event"],
        )

    return open_phase3_read_api(
        evidence_db_path=evidence_path,
        query_history_db_path=query_path,
        default_policy=policy,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="CitySight API + dashboard")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--city-config", default=str(DEFAULT_CITY_CONFIG_PATH),
                        help="Phase 2 city topology config (default: %(default)s)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    persistence = config.get("persistence", {})
    db_path = persistence.get("db_path", "outputs/observations/observations.db")
    obs_repo = SQLiteObservationRepository(db_path)
    wl_repo = SQLiteWatchlistRepository(
        persistence.get("watchlist_db_path", db_path))

    # Build the city camera graph once; share it with trajectory + topology API.
    graph = build_city_graph(args.city_config)
    reconstructor = build_trajectory_reconstructor(
        obs_repo, args.city_config, config, city_graph=graph)
    if graph is not None:
        print(f"Phase 2 topology + trajectory enabled "
              f"(city config: {args.city_config})")
    else:
        print(f"City config not found ({args.city_config}); "
              "Phase 2 topology + trajectory disabled")

    phase3_api = build_phase3_read_api(config)
    if phase3_api is None:
        print("Phase 3 read API disabled")
    else:
        print("Phase 3 read API enabled (configured SQLite files, read-only)")

    server = create_server(obs_repo, host=args.host, port=args.port,
                           watchlist_repo=wl_repo,
                           trajectory_reconstructor=reconstructor,
                           city_graph=graph,
                           phase3_api=phase3_api)
    host, port = server.server_address
    print(f"CitySight dashboard: http://{host}:{port}/dashboard")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down...")
    finally:
        server.shutdown()
        server.server_close()
        obs_repo.close()
        wl_repo.close()
        if phase3_api is not None:
            phase3_api.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
