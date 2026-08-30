"""CitySight Phase 1 — start the read/watchlist API + /dashboard (final demo).

Serves the existing stdlib API against the SAME SQLite database used by the demo
pipeline (phase1_anpr.demo), so the dashboard shows real persisted observations,
watchlist entries, and alerts. No new dependencies, no feature changes.
"""

import argparse
import sys

from phase1_anpr.api import create_server
from phase1_anpr.utils.config import DEFAULT_CONFIG_PATH, load_config
from phase1_anpr.persistence import (
    SQLiteObservationRepository,
    SQLiteWatchlistRepository,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="CitySight Phase 1 API + dashboard")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    persistence = load_config(args.config).get("persistence", {})
    db_path = persistence.get("db_path", "outputs/observations/observations.db")
    obs_repo = SQLiteObservationRepository(db_path)
    wl_repo = SQLiteWatchlistRepository(
        persistence.get("watchlist_db_path", db_path))

    server = create_server(obs_repo, host=args.host, port=args.port,
                           watchlist_repo=wl_repo)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
