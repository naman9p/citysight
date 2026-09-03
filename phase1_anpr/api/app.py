"""Read-only HTTP API for cameras, observations, and exact plate search (Step 13).

Per CLAUDE.md this uses only the Python standard library (`http.server`) — no
FastAPI/new dependencies. Handlers delegate to `ObservationRepository`; no SQL is
issued from the routing layer. Exact plate search normalizes the query with the
existing `PlateNormalizer`. Abstained rows are already stored without a plate
identity, so they are returned as-is (plate fields NULL).

`create_server(repository, ...)` builds a `ThreadingHTTPServer` bound to a given
host/port (use port 0 in tests for an ephemeral port), so tests can drive it over
real HTTP with httpx and an in-memory SQLite repository.
"""

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs, unquote

from phase1_anpr.normalization.plate_normalizer import PlateNormalizer
from phase1_anpr.api.dashboard import DASHBOARD_HTML
from phase1_anpr.persistence.watchlist_repository import WatchlistError
from phase2_city.trajectory import TrajectoryQueryError, TrajectoryDataError

DEFAULT_LIMIT = 50
MAX_LIMIT = 500

# JSON-safe columns returned to clients (drops nothing sensitive; explicit so the
# response shape is stable regardless of internal schema additions).
_FIELDS = (
    "event_id", "camera_id", "track_id", "timestamp", "plate_raw",
    "plate_normalized", "confidence", "status", "detector_confidence",
    "ocr_confidence", "quality_score", "best_frame_number", "format_type",
    "state_code", "evidence_ref", "model_version",
)


class ApiError(Exception):
    """Carries an HTTP status for a client-facing error."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _to_response(row: dict) -> dict:
    return {k: row.get(k) for k in _FIELDS}


def _parse_limit(qs: dict) -> int:
    raw = qs.get("limit", [None])[0]
    if raw is None:
        return DEFAULT_LIMIT
    try:
        limit = int(raw)
    except (TypeError, ValueError):
        raise ApiError(400, "limit must be an integer")
    if limit < 1 or limit > MAX_LIMIT:
        raise ApiError(400, f"limit must be between 1 and {MAX_LIMIT}")
    return limit


class _Router:
    """Resolves a path/query into a JSON-serializable result or raises ApiError."""

    def __init__(self, repository, normalizer=None, watchlist_repo=None,
                 trajectory_reconstructor=None):
        self.repo = repository
        self.normalizer = normalizer or PlateNormalizer()
        self.watchlist = watchlist_repo
        self.trajectory = trajectory_reconstructor

    def _require_watchlist(self):
        if self.watchlist is None:
            raise ApiError(404, "watchlist not enabled")
        return self.watchlist

    def _require_trajectory(self):
        if self.trajectory is None:
            raise ApiError(404, "trajectory not enabled")
        return self.trajectory

    def dispatch(self, path: str, qs: dict):
        if path == "/health":
            return {"status": "ok"}

        if path == "/observations":
            return [_to_response(r) for r in self.repo.list_recent(_parse_limit(qs))]

        if path == "/watchlist":
            return self._require_watchlist().list()

        if path == "/alerts":
            return self._require_watchlist().list_alerts(_parse_limit(qs))

        m = re.fullmatch(r"/observations/([^/]+)", path)
        if m:
            event_id = unquote(m.group(1))
            row = self.repo.get(event_id)
            if row is None:
                raise ApiError(404, f"observation not found: {event_id}")
            return _to_response(row)

        m = re.fullmatch(r"/plates/([^/]+)/observations", path)
        if m:
            normalized = self.normalizer.normalize_text(unquote(m.group(1)))
            if not normalized:
                raise ApiError(400, "plate query is empty after normalization")
            rows = self.repo.list_by_plate(normalized, _parse_limit(qs))
            return [_to_response(r) for r in rows]

        m = re.fullmatch(r"/cameras/([^/]+)/observations", path)
        if m:
            camera_id = unquote(m.group(1))
            rows = self.repo.list_by_camera(camera_id, _parse_limit(qs))
            return [_to_response(r) for r in rows]

        raise ApiError(404, "not found")

    def dispatch_post(self, path: str, body: dict):
        if path == "/watchlist":
            wl = self._require_watchlist()
            if not isinstance(body, dict):
                raise ApiError(400, "JSON object body required")
            try:
                return (201, wl.add(body.get("plate"), body.get("label")))
            except WatchlistError as e:
                raise ApiError(400, str(e))

        if path == "/v1/trajectories":
            reconstructor = self._require_trajectory()
            if not isinstance(body, dict):
                raise ApiError(400, "JSON object body required")
            plate = body.get("plate")
            if plate is None:
                raise ApiError(400, "plate is required")
            if not isinstance(plate, str):
                raise ApiError(400, "plate must be a string")
            try:
                trajectory = reconstructor.reconstruct(
                    plate,
                    start=body.get("start"),
                    end=body.get("end"),
                )
            except TrajectoryQueryError as e:
                raise ApiError(400, str(e))
            except TrajectoryDataError as e:
                raise ApiError(500, str(e))
            return (200, trajectory.to_dict())

        raise ApiError(404, "not found")

    def dispatch_delete(self, path: str):
        m = re.fullmatch(r"/watchlist/([^/]+)", path)
        if m:
            wl = self._require_watchlist()
            wid = unquote(m.group(1))
            # Disable (soft-delete) so alert history stays meaningful.
            if not wl.disable(wid) and wl.get(wid) is None:
                raise ApiError(404, f"watchlist entry not found: {wid}")
            return {"watchlist_id": wid, "enabled": 0}
        raise ApiError(404, "not found")


def make_handler(repository, normalizer=None, watchlist_repo=None,
                 trajectory_reconstructor=None):
    router = _Router(repository, normalizer, watchlist_repo,
                     trajectory_reconstructor=trajectory_reconstructor)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html):
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                raise ApiError(400, "invalid JSON body")

        def do_GET(self):
            parts = urlsplit(self.path)
            if parts.path == "/dashboard":
                self._send_html(DASHBOARD_HTML)
                return
            try:
                result = router.dispatch(parts.path, parse_qs(parts.query))
            except ApiError as e:
                self._send(e.status, {"error": e.message})
            else:
                self._send(200, result)

        def do_POST(self):
            parts = urlsplit(self.path)
            try:
                status, result = router.dispatch_post(parts.path, self._read_json())
            except ApiError as e:
                self._send(e.status, {"error": e.message})
            else:
                self._send(status, result)

        def do_DELETE(self):
            parts = urlsplit(self.path)
            try:
                result = router.dispatch_delete(parts.path)
            except ApiError as e:
                self._send(e.status, {"error": e.message})
            else:
                self._send(200, result)

        def log_message(self, *args):  # silence stderr during tests
            pass

    return Handler


def create_server(repository, host="127.0.0.1", port=0, normalizer=None,
                  watchlist_repo=None, trajectory_reconstructor=None):
    """Build a ThreadingHTTPServer. port=0 binds an ephemeral port."""
    return ThreadingHTTPServer(
        (host, port), make_handler(repository, normalizer, watchlist_repo,
                                  trajectory_reconstructor=trajectory_reconstructor))
