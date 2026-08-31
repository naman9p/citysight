"""Multi-camera recorded-video replay for Phase 1 (Step 19).

Replays one or more recorded videos in order into the SAME SQLite + evidence
stores. Each video is a "source" with its own ``camera_id`` and optional
``source_id``; the pipeline folds ``source_id`` into the deterministic event id
so identical track ids from different videos never collide (see
``ANPRPipeline._event_id_for``).

Design mirrors the rest of Phase 1: the orchestrator is dependency-injected
(``pipeline_factory`` / ``reader_factory``), so it is fully testable with fakes
and no YOLO/Paddle weights. ``run_replay`` wires the real components.

Backward compatibility: when ``config`` has no ``sources`` list, a single source
is derived from the existing ``video`` section with ``source_id=None`` — i.e.
the legacy single-video demo, byte-for-byte unchanged event ids.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional


@dataclass(frozen=True)
class ReplaySource:
    """One recorded video to replay, plus its identity metadata."""

    video_path: str
    camera_id: str
    source_id: Optional[str] = None
    start_time: Optional[str] = None


@dataclass
class SourceReplayResult:
    """Per-source outcome: the source and the observations it produced."""

    source: ReplaySource
    results: list  # list[PipelineResult]

    @property
    def counts(self) -> dict:
        """Status tally (accepted / review / abstained / alerts) for this source."""
        tally = {"observations": len(self.results), "accepted": 0,
                 "review": 0, "abstained": 0, "alerts": 0}
        for r in self.results:
            status = r.observation.status
            if status in tally:
                tally[status] += 1
            tally["alerts"] += len(r.alerts)
        return tally


def load_replay_sources(config) -> List[ReplaySource]:
    """Resolve the list of sources to replay from ``config``.

    With a non-empty ``config['sources']`` list, each entry becomes a
    ``ReplaySource`` (``video`` path required; ``camera_id`` and ``start_time``
    fall back to the ``video`` section; ``source_id`` defaults to the video file
    stem). Without it, a single legacy source is returned with ``source_id=None``.

    Raises ``ValueError`` on a malformed entry or on a duplicate
    ``(camera_id, source_id)`` pair — duplicates would defeat the whole point of
    Step 19 by re-colliding event ids, so we fail fast.
    """
    video_cfg = config.get("video", {}) or {}
    raw_sources = config.get("sources")

    if not raw_sources:
        return [ReplaySource(
            video_path=video_cfg.get("input_path"),
            camera_id=video_cfg.get("camera_id", "cam_01"),
            source_id=None,
            start_time=video_cfg.get("start_time"),
        )]

    if not isinstance(raw_sources, list):
        raise ValueError("config 'sources' must be a list of mappings")

    sources: List[ReplaySource] = []
    seen = set()
    for i, entry in enumerate(raw_sources):
        if not isinstance(entry, dict):
            raise ValueError(f"sources[{i}] must be a mapping")
        video_path = entry.get("video") or entry.get("input_path")
        if not video_path:
            raise ValueError(f"sources[{i}] is missing a 'video' path")
        camera_id = entry.get("camera_id") or video_cfg.get("camera_id", "cam_01")
        source_id = entry.get("source_id") or Path(str(video_path)).stem
        start_time = entry.get("start_time", video_cfg.get("start_time"))

        key = (camera_id, source_id)
        if key in seen:
            raise ValueError(
                f"sources[{i}] duplicates (camera_id={camera_id!r}, "
                f"source_id={source_id!r}); source ids must be unique per camera")
        seen.add(key)
        sources.append(ReplaySource(str(video_path), camera_id, source_id,
                                    start_time))
    return sources


def replay(sources: List[ReplaySource],
           pipeline_factory: Callable[[ReplaySource], object],
           reader_factory: Callable[[ReplaySource], object]
           ) -> List[SourceReplayResult]:
    """Replay each source through a fresh pipeline; return per-source results.

    A fresh pipeline (hence a fresh tracker and per-run guard) is built per
    source, so tracking state never leaks between videos. Stores passed to the
    factory are expected to be shared, giving one combined observation set.
    """
    out: List[SourceReplayResult] = []
    for source in sources:
        pipeline = pipeline_factory(source)
        reader = reader_factory(source)
        results = pipeline.run(reader)
        out.append(SourceReplayResult(source=source, results=results))
    return out


def run_replay(config, observation_repo, evidence_store=None,
               watchlist_repo=None, sources=None) -> List[SourceReplayResult]:
    """Replay recorded videos using the real components and shared stores.

    Raises DetectorError if YOLO weights are missing and VideoReaderError if a
    source video cannot be opened (both surfaced from the per-source factories).
    """
    from phase1_anpr.pipeline.anpr_pipeline import build_pipeline_from_config
    from phase1_anpr.video.video_reader import VideoReader

    sources = sources if sources is not None else load_replay_sources(config)
    process_fps = (config.get("video", {}) or {}).get("process_fps")

    def pipeline_factory(src: ReplaySource):
        return build_pipeline_from_config(
            config, observation_repo=observation_repo,
            evidence_store=evidence_store, watchlist_repo=watchlist_repo,
            camera_id=src.camera_id, source_id=src.source_id,
            video_start_time=src.start_time)

    def reader_factory(src: ReplaySource):
        return VideoReader(src.video_path, src.camera_id, process_fps)

    return replay(sources, pipeline_factory, reader_factory)
