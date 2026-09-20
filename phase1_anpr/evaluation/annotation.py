"""Local, prediction-blind annotation helper for the Step 34 manifest.

The helper intentionally exposes only manual frame and plate-instance fields.
Replay observations, candidate rankings, and production ANPR components are not
loaded by this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Mapping, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

import yaml

from phase1_anpr.evaluation.real_world import (
    RealWorldEvaluationError,
    load_real_world_ground_truth,
)


class AnnotationError(ValueError):
    """Raised when an annotation request is invalid."""


class AnnotationConflictError(AnnotationError):
    """Raised when the manifest changed after the browser loaded it."""


INSTANCE_EDITABLE_FIELDS = {
    "instance_id", "plate_text", "text_evaluable", "vehicle_identity", "notes",
}


def validate_bbox(value, *, width: Optional[int] = None,
                  height: Optional[int] = None) -> list[float]:
    """Return a validated full-resolution ``[x1, y1, x2, y2]`` box."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise AnnotationError("bbox must be [x1, y1, x2, y2]")
    if any(isinstance(part, bool) or not isinstance(part, (int, float))
           for part in value):
        raise AnnotationError("bbox coordinates must be numeric")
    result = [float(part) for part in value]
    x1, y1, x2, y2 = result
    if (not all(math.isfinite(part) for part in result)
            or x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1):
        raise AnnotationError("bbox has invalid coordinates")
    if width is not None and x2 > width:
        raise AnnotationError(f"bbox exceeds image width {width}")
    if height is not None and y2 > height:
        raise AnnotationError(f"bbox exceeds image height {height}")
    return [int(part) if part.is_integer() else part for part in result]


def _optional_text(value, field: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AnnotationError(f"{field} must be text or null")
    value = value.strip()
    return value or None


def _required_text(value, field: str) -> str:
    result = _optional_text(value, field)
    if result is None:
        raise AnnotationError(f"{field} must be a non-empty string")
    return result


def _revision(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_yaml(path: Path) -> tuple[bytes, dict]:
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise AnnotationError(f"annotation manifest not found: {path}") from exc
    try:
        document = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise AnnotationError(f"invalid annotation manifest: {exc}") from exc
    if not isinstance(document, dict):
        raise AnnotationError("annotation manifest must be a mapping")
    return data, document


def _atomic_bytes(path: Path, data: bytes) -> None:
    """Write bytes through a same-directory temp file and atomic replacement."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validated_instance(raw: Mapping, existing: Optional[Mapping] = None) -> dict:
    if not isinstance(raw, Mapping):
        raise AnnotationError("instance must be an object")
    extra = set(raw) - INSTANCE_EDITABLE_FIELDS
    if extra:
        raise AnnotationError(
            "instance contains non-editable field(s): " + ", ".join(sorted(extra)))
    instance_id = _required_text(raw.get("instance_id"), "instance_id")
    text_evaluable = raw.get("text_evaluable")
    if not isinstance(text_evaluable, bool):
        raise AnnotationError("text_evaluable must be true or false")
    plate_text = _optional_text(raw.get("plate_text"), "plate_text")
    if text_evaluable and plate_text is None:
        raise AnnotationError(
            "plate_text is required when text_evaluable is true")
    result = dict(existing or {})
    result.update({
        "instance_id": instance_id,
        "plate_text": plate_text,
        "text_evaluable": text_evaluable,
        "vehicle_identity": _optional_text(
            raw.get("vehicle_identity"), "vehicle_identity"),
        "notes": _optional_text(raw.get("notes"), "notes"),
    })
    if existing is None:
        result["observation_event_id"] = None
        result["observation_link_reviewed"] = False
    return result


class AnnotationStore:
    """Thread-safe manifest reader/editor with revision-checked atomic saves."""

    def __init__(self, manifest_path, frames_dir, *, create_backup: bool = True):
        self.manifest_path = Path(manifest_path).resolve()
        self.frames_dir = Path(frames_dir).resolve()
        self.create_backup = create_backup
        self._lock = threading.RLock()
        load_real_world_ground_truth(self.manifest_path)
        if not self.frames_dir.is_dir():
            raise AnnotationError(f"frames directory not found: {self.frames_dir}")

    def _snapshot(self) -> tuple[bytes, dict, str]:
        data, document = _read_yaml(self.manifest_path)
        return data, document, _revision(data)

    @staticmethod
    def _source(document: Mapping, source_id: str) -> dict:
        for source in document.get("sources", []):
            if source.get("source_id") == source_id:
                return source
        raise AnnotationError(f"unknown source_id: {source_id}")

    @staticmethod
    def _frame(source: Mapping, frame_number: int) -> dict:
        for frame in source.get("frames", []):
            if frame.get("frame_number") == frame_number:
                return frame
        raise AnnotationError(
            f"unknown frame_number {frame_number} for {source.get('source_id')}")

    @staticmethod
    def _progress(document: Mapping) -> dict:
        sources = document.get("sources", [])
        frames = [frame for source in sources for frame in source.get("frames", [])]
        reviewed = sum(frame.get("reviewed") is True for frame in frames)
        plates = sum(len(frame.get("plates", [])) for frame in frames)
        instances = sum(
            len(source.get("plate_instances", [])) for source in sources)
        return {
            "reviewed": reviewed,
            "total": len(frames),
            "remaining": len(frames) - reviewed,
            "plates_labeled": plates,
            "plate_instances": instances,
        }

    def bootstrap(self) -> dict:
        with self._lock:
            _, document, revision = self._snapshot()
            sources = []
            for source in document.get("sources", []):
                source_frames = source.get("frames", [])
                sources.append({
                    "source_id": source["source_id"],
                    "camera_id": source["camera_id"],
                    "frames": [{
                        "frame_number": frame["frame_number"],
                        "timestamp_seconds": frame.get("timestamp_seconds"),
                        "reviewed": frame.get("reviewed", False),
                        "plate_count": len(frame.get("plates", [])),
                    } for frame in source_frames],
                    "progress": self._progress({"sources": [source]}),
                })
            return {
                "schema_version": document.get("schema_version"),
                "benchmark_id": document.get("benchmark_id"),
                "revision": revision,
                "sources": sources,
                "progress": self._progress(document),
            }

    def frame(self, source_id: str, frame_number: int) -> dict:
        with self._lock:
            _, document, revision = self._snapshot()
            source = self._source(document, source_id)
            frame = self._frame(source, frame_number)
            return {
                "revision": revision,
                "source_id": source_id,
                "camera_id": source["camera_id"],
                "frame": copy.deepcopy(frame),
                "instances": copy.deepcopy(source.get("plate_instances", [])),
                "image_url": (
                    f"/frames/{quote(source_id, safe='')}/"
                    f"frame_{frame_number:08d}.jpg"),
            }

    def image_path(self, source_id: str, frame_number: int) -> Path:
        with self._lock:
            _, document, _ = self._snapshot()
            source = self._source(document, source_id)
            self._frame(source, frame_number)
            source_root = (self.frames_dir / source_id).resolve()
            try:
                source_root.relative_to(self.frames_dir)
            except ValueError as exc:
                raise AnnotationError("invalid source frame path") from exc
            image = (source_root / f"frame_{frame_number:08d}.jpg").resolve()
            try:
                image.relative_to(source_root)
            except ValueError as exc:
                raise AnnotationError("invalid frame path") from exc
            if not image.is_file():
                raise AnnotationError(f"annotation image not found: {image}")
            return image

    def _image_size(self, source_id: str, frame_number: int) -> tuple[int, int]:
        import cv2

        image = cv2.imread(str(self.image_path(source_id, frame_number)))
        if image is None:
            raise AnnotationError("annotation image is unreadable")
        height, width = image.shape[:2]
        return width, height

    def save_frame(self, payload: Mapping) -> dict:
        """Apply one explicit frame update and preserve all unrelated labels."""
        if not isinstance(payload, Mapping):
            raise AnnotationError("request body must be an object")
        allowed = {
            "revision", "source_id", "frame_number", "reviewed", "plates",
            "instances", "confirm_reviewed_update",
        }
        extra = set(payload) - allowed
        if extra:
            raise AnnotationError(
                "request contains unknown field(s): " + ", ".join(sorted(extra)))
        expected_revision = _required_text(payload.get("revision"), "revision")
        source_id = _required_text(payload.get("source_id"), "source_id")
        frame_number = payload.get("frame_number")
        if (isinstance(frame_number, bool) or not isinstance(frame_number, int)
                or frame_number < 0):
            raise AnnotationError("frame_number must be an integer >= 0")
        reviewed = payload.get("reviewed")
        if not isinstance(reviewed, bool):
            raise AnnotationError("reviewed must be true or false")
        raw_plates = payload.get("plates")
        raw_instances = payload.get("instances", [])
        if not isinstance(raw_plates, list) or not isinstance(raw_instances, list):
            raise AnnotationError("plates and instances must be lists")
        if raw_plates and not reviewed:
            raise AnnotationError("a frame with plates must be reviewed")

        with self._lock:
            original_data, document, current_revision = self._snapshot()
            if current_revision != expected_revision:
                raise AnnotationConflictError(
                    "manifest changed since this frame was loaded; reload before saving")
            candidate = copy.deepcopy(document)
            source = self._source(candidate, source_id)
            frame = self._frame(source, frame_number)
            was_reviewed = frame.get("reviewed") is True
            width, height = self._image_size(source_id, frame_number)

            existing_instances = {
                item["instance_id"]: item
                for item in source.get("plate_instances", [])
            }
            updates: dict[str, dict] = {}
            for raw_instance in raw_instances:
                instance_id = _required_text(
                    raw_instance.get("instance_id")
                    if isinstance(raw_instance, Mapping) else None,
                    "instance_id",
                )
                if instance_id in updates:
                    raise AnnotationError(
                        f"duplicate instance update: {instance_id}")
                updates[instance_id] = _validated_instance(
                    raw_instance, existing_instances.get(instance_id))

            plates = []
            used_ids: set[str] = set()
            for index, raw_plate in enumerate(raw_plates):
                if not isinstance(raw_plate, Mapping):
                    raise AnnotationError(f"plates[{index}] must be an object")
                extra_plate = set(raw_plate) - {"instance_id", "bbox", "notes"}
                if extra_plate:
                    raise AnnotationError(
                        f"plates[{index}] has unknown field(s): "
                        + ", ".join(sorted(extra_plate)))
                instance_id = _required_text(
                    raw_plate.get("instance_id"),
                    f"plates[{index}].instance_id",
                )
                if instance_id in used_ids:
                    raise AnnotationError(
                        f"instance_id occurs more than once in frame: {instance_id}")
                if instance_id not in existing_instances and instance_id not in updates:
                    raise AnnotationError(
                        f"plate references unknown instance_id: {instance_id}")
                used_ids.add(instance_id)
                plates.append({
                    "instance_id": instance_id,
                    "bbox": validate_bbox(
                        raw_plate.get("bbox"), width=width, height=height),
                    "notes": _optional_text(
                        raw_plate.get("notes"), f"plates[{index}].notes"),
                })

            previous_ids = {
                plate.get("instance_id") for plate in frame.get("plates", [])
            }
            frame["reviewed"] = reviewed
            frame["plates"] = plates

            merged_instances = []
            for instance in source.get("plate_instances", []):
                instance_id = instance["instance_id"]
                merged_instances.append(updates.get(instance_id, instance))
            for instance_id, instance in updates.items():
                if instance_id not in existing_instances:
                    merged_instances.append(instance)

            remaining_references = {
                plate["instance_id"]
                for candidate_frame in source.get("frames", [])
                for plate in candidate_frame.get("plates", [])
            }
            removable = previous_ids - remaining_references
            protected = [
                instance_id for instance_id in removable
                if (existing_instances[instance_id].get("observation_event_id")
                    or existing_instances[instance_id].get(
                        "observation_link_reviewed") is True)
            ]
            if protected:
                raise AnnotationError(
                    "cannot remove the final box for replay-linked instance(s) "
                    "in the prediction-blind annotation helper: "
                    + ", ".join(sorted(protected)))
            source["plate_instances"] = [
                instance for instance in merged_instances
                if instance["instance_id"] not in removable
            ]

            if (was_reviewed and candidate != document
                    and payload.get("confirm_reviewed_update") is not True):
                raise AnnotationError(
                    "changing a reviewed frame requires explicit confirmation")

            serialized = yaml.safe_dump(
                candidate, sort_keys=False, allow_unicode=True).encode("utf-8")
            descriptor, temporary_name = tempfile.mkstemp(
                dir=self.manifest_path.parent,
                prefix=f".{self.manifest_path.name}.validated.", suffix=".tmp")
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                temporary.write_bytes(serialized)
                load_real_world_ground_truth(temporary)
            except (OSError, RealWorldEvaluationError) as exc:
                temporary.unlink(missing_ok=True)
                if isinstance(exc, RealWorldEvaluationError):
                    raise AnnotationError(str(exc)) from exc
                raise
            temporary.unlink(missing_ok=True)

            if self.create_backup:
                _atomic_bytes(
                    self.manifest_path.with_suffix(
                        self.manifest_path.suffix + ".bak"),
                    original_data,
                )
            _atomic_bytes(self.manifest_path, serialized)
            new_revision = _revision(serialized)
            return {
                "revision": new_revision,
                "frame": copy.deepcopy(frame),
                "progress": self._progress(candidate),
                "source_progress": self._progress({"sources": [source]}),
            }


def _json_bytes(value) -> bytes:
    return (json.dumps(value, separators=(",", ":"), ensure_ascii=False)
            + "\n").encode("utf-8")


def create_annotation_server(store: AnnotationStore, *, host="127.0.0.1",
                             port: int = 8765) -> ThreadingHTTPServer:
    """Create a loopback-only stdlib server. ``port=0`` selects a free port."""
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise AnnotationError("annotation server must bind to a loopback host")
    ui_path = Path(__file__).with_name("annotation_ui.html")
    ui_bytes = ui_path.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "CitySightAnnotation/1.0"

        def log_message(self, format, *args):
            print(f"annotation: {self.address_string()} - {format % args}")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; img-src 'self'; "
                             "style-src 'self' 'unsafe-inline'; "
                             "script-src 'self' 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value) -> None:
            self._send(status, _json_bytes(value), "application/json; charset=utf-8")

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message})

        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path in {"/", "/index.html"}:
                    self._send(HTTPStatus.OK, ui_bytes, "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/bootstrap":
                    self._json(HTTPStatus.OK, store.bootstrap())
                    return
                if parsed.path == "/api/frame":
                    query = parse_qs(parsed.query)
                    source_id = query.get("source_id", [None])[0]
                    frame_value = query.get("frame_number", [None])[0]
                    if source_id is None or frame_value is None:
                        raise AnnotationError(
                            "source_id and frame_number are required")
                    try:
                        frame_number = int(frame_value)
                    except ValueError as exc:
                        raise AnnotationError(
                            "frame_number must be an integer") from exc
                    self._json(
                        HTTPStatus.OK, store.frame(source_id, frame_number))
                    return
                if parsed.path.startswith("/frames/"):
                    parts = parsed.path.split("/")
                    if len(parts) != 4 or not parts[3].startswith("frame_"):
                        raise AnnotationError("invalid frame image path")
                    frame_number = int(parts[3][6:-4])
                    image = store.image_path(unquote(parts[2]), frame_number)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(image.stat().st_size))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    with image.open("rb") as source:
                        shutil.copyfileobj(source, self.wfile)
                    return
                self._error(HTTPStatus.NOT_FOUND, "not found")
            except (AnnotationError, ValueError) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))

        def do_POST(self):
            if urlparse(self.path).path != "/api/save-frame":
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1024 * 1024:
                    raise AnnotationError("request body size is invalid")
                try:
                    payload = json.loads(self.rfile.read(length))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise AnnotationError("request body must be valid JSON") from exc
                self._json(HTTPStatus.OK, store.save_frame(payload))
            except AnnotationConflictError as exc:
                self._error(HTTPStatus.CONFLICT, str(exc))
            except AnnotationError as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
            except OSError as exc:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    return ThreadingHTTPServer((host, port), Handler)


def run_annotation_server(manifest_path, frames_dir, *, port: int = 8765,
                          create_backup: bool = True,
                          open_browser: bool = False) -> dict:
    """Run the local annotation UI until interrupted with Ctrl+C."""
    store = AnnotationStore(
        manifest_path, frames_dir, create_backup=create_backup)
    server = create_annotation_server(store, port=port)
    actual_port = server.server_address[1]
    url = f"http://127.0.0.1:{actual_port}/"
    print(f"CitySight Step 34 annotation UI: {url}")
    print("Press Ctrl+C to stop. Predictions are not loaded by this server.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return {
        "ground_truth_path": str(Path(manifest_path).resolve()),
        "frames_dir": str(Path(frames_dir).resolve()),
        "stopped": True,
    }


__all__ = [
    "AnnotationConflictError",
    "AnnotationError",
    "AnnotationStore",
    "create_annotation_server",
    "run_annotation_server",
    "validate_bbox",
]
