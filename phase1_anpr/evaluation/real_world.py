"""Step 34 real-video, independently labeled ANPR benchmark.

The evaluator is deliberately inference-independent: it compares a strict
manual ground-truth file with a captured prediction file.  It never derives
labels from predictions and never changes production policies.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

import yaml

from phase1_anpr.evaluation.metrics import (
    BBox,
    aligned_substitutions,
    evaluate_detections,
    evaluate_ocr,
)


class RealWorldEvaluationError(ValueError):
    """Raised for invalid or inconsistent Step 34 evaluation inputs."""


@dataclass(frozen=True)
class PlateInstance:
    instance_id: str
    plate_text: Optional[str]
    text_evaluable: bool
    vehicle_identity: Optional[str]
    observation_event_id: Optional[str]
    observation_link_reviewed: bool
    notes: Optional[str]


@dataclass(frozen=True)
class FramePlate:
    instance_id: str
    bbox: BBox
    notes: Optional[str]


@dataclass(frozen=True)
class ReviewedFrame:
    frame_number: int
    timestamp_seconds: Optional[float]
    reviewed: bool
    plates: tuple[FramePlate, ...]


@dataclass(frozen=True)
class GroundTruthSource:
    source_id: str
    camera_id: str
    video_path: str
    observations_reviewed: bool
    non_plate_observation_event_ids: tuple[str, ...]
    instances: tuple[PlateInstance, ...]
    frames: tuple[ReviewedFrame, ...]


@dataclass(frozen=True)
class RealWorldGroundTruth:
    benchmark_id: str
    schema_version: int
    sources: tuple[GroundTruthSource, ...]
    cross_camera_cases: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class DetectionPrediction:
    frame_number: int
    bbox: BBox
    confidence: float


@dataclass(frozen=True)
class ObservationPrediction:
    event_id: str
    best_frame_number: Optional[int]
    status: str
    ocr_text: Optional[str]
    plate_raw: Optional[str]
    plate_normalized: Optional[str]
    confidence: float


@dataclass(frozen=True)
class CandidateRanking:
    source_event_id: str
    candidate_event_ids: tuple[str, ...]
    topology_rejections: int
    travel_rejections: int
    decisions: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PredictionSource:
    source_id: str
    camera_id: str
    detections: tuple[DetectionPrediction, ...]
    observations: tuple[ObservationPrediction, ...]


@dataclass(frozen=True)
class RealWorldPredictions:
    benchmark_id: str
    schema_version: int
    provenance: Mapping[str, object]
    sources: tuple[PredictionSource, ...]
    candidate_rankings: tuple[CandidateRanking, ...]


def _mapping(value, context: str) -> dict:
    if not isinstance(value, dict):
        raise RealWorldEvaluationError(f"{context} must be a mapping")
    return value


def _list(value, context: str) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RealWorldEvaluationError(f"{context} must be a list")
    return value


def _unknown(item: Mapping, allowed: set[str], context: str) -> None:
    extra = set(item) - allowed
    if extra:
        raise RealWorldEvaluationError(
            f"{context} has unknown field(s): {', '.join(sorted(extra))}")


def _text(value, context: str, *, optional: bool = False) -> Optional[str]:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RealWorldEvaluationError(f"{context} must be a non-empty string")
    return value.strip()


def _boolean(value, context: str) -> bool:
    if not isinstance(value, bool):
        raise RealWorldEvaluationError(f"{context} must be true or false")
    return value


def _integer(value, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RealWorldEvaluationError(
            f"{context} must be an integer >= {minimum}")
    return value


def _number(value, context: str, *, minimum: float = 0.0,
            maximum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RealWorldEvaluationError(f"{context} must be numeric")
    result = float(value)
    if (not math.isfinite(result) or result < minimum
            or (maximum is not None and result > maximum)):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise RealWorldEvaluationError(
            f"{context} must be >= {minimum}{suffix}")
    return result


def _bbox(value, context: str) -> BBox:
    if not isinstance(value, list) or len(value) != 4:
        raise RealWorldEvaluationError(f"{context} must be [x1, y1, x2, y2]")
    try:
        x1, y1, x2, y2 = (float(part) for part in value)
    except (TypeError, ValueError) as exc:
        raise RealWorldEvaluationError(f"{context} must be numeric") from exc
    if (not all(math.isfinite(part) for part in (x1, y1, x2, y2))
            or min(x1, y1) < 0 or x2 <= x1 or y2 <= y1):
        raise RealWorldEvaluationError(f"{context} has invalid coordinates")
    return x1, y1, x2, y2


def _read_document(path) -> tuple[Path, dict]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Step 34 input not found: {source}")
    try:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RealWorldEvaluationError(f"invalid YAML/JSON: {exc}") from exc
    return source, _mapping(data, "Step 34 document")


def load_real_world_ground_truth(path) -> RealWorldGroundTruth:
    """Load strict, manually authored frame/plate labels."""
    manifest_path, root = _read_document(path)
    _unknown(root, {
        "schema_version", "benchmark_id", "sources", "cross_camera_cases",
    }, "ground truth")
    version = _integer(root.get("schema_version"), "schema_version", minimum=1)
    if version != 1:
        raise RealWorldEvaluationError("unsupported ground-truth schema_version")
    benchmark_id = _text(root.get("benchmark_id"), "benchmark_id")
    sources: list[GroundTruthSource] = []
    seen_sources: set[str] = set()
    seen_events: set[str] = set()
    plate_event_ids: set[str] = set()
    plate_event_sources: dict[str, str] = {}
    source_cameras: dict[str, str] = {}

    for source_index, raw_source in enumerate(_list(root.get("sources"), "sources")):
        context = f"sources[{source_index}]"
        item = _mapping(raw_source, context)
        _unknown(item, {
            "source_id", "camera_id", "video_path", "observations_reviewed",
            "non_plate_observation_event_ids", "plate_instances", "frames",
        }, context)
        source_id = _text(item.get("source_id"), f"{context}.source_id")
        if source_id in seen_sources:
            raise RealWorldEvaluationError(f"duplicate source_id: {source_id}")
        seen_sources.add(source_id)
        camera_id = _text(item.get("camera_id"), f"{context}.camera_id")
        source_cameras[source_id] = camera_id
        instances: list[PlateInstance] = []
        instance_ids: set[str] = set()
        for instance_index, raw_instance in enumerate(
                _list(item.get("plate_instances"), f"{context}.plate_instances")):
            icontext = f"{context}.plate_instances[{instance_index}]"
            row = _mapping(raw_instance, icontext)
            _unknown(row, {
                "instance_id", "plate_text", "text_evaluable", "vehicle_identity",
                "observation_event_id", "observation_link_reviewed", "notes",
            }, icontext)
            instance_id = _text(row.get("instance_id"), f"{icontext}.instance_id")
            if instance_id in instance_ids:
                raise RealWorldEvaluationError(
                    f"duplicate instance_id in {source_id}: {instance_id}")
            instance_ids.add(instance_id)
            text_evaluable = _boolean(
                row.get("text_evaluable"), f"{icontext}.text_evaluable")
            plate_text = _text(
                row.get("plate_text"), f"{icontext}.plate_text", optional=True)
            if text_evaluable and not _canonical_plate(plate_text):
                raise RealWorldEvaluationError(
                    f"{icontext}.plate_text is required when text_evaluable is true")
            event_id = _text(
                row.get("observation_event_id"),
                f"{icontext}.observation_event_id", optional=True)
            if event_id:
                if event_id in seen_events:
                    raise RealWorldEvaluationError(
                        f"duplicate observation_event_id: {event_id}")
                seen_events.add(event_id)
                plate_event_ids.add(event_id)
                plate_event_sources[event_id] = source_id
            instances.append(PlateInstance(
                instance_id=instance_id,
                plate_text=plate_text,
                text_evaluable=text_evaluable,
                vehicle_identity=_text(
                    row.get("vehicle_identity"), f"{icontext}.vehicle_identity",
                    optional=True),
                observation_event_id=event_id,
                observation_link_reviewed=_boolean(
                    row.get("observation_link_reviewed", False),
                    f"{icontext}.observation_link_reviewed"),
                notes=_text(row.get("notes"), f"{icontext}.notes", optional=True),
            ))

        frames: list[ReviewedFrame] = []
        seen_frames: set[int] = set()
        referenced_instances: set[str] = set()
        for frame_index, raw_frame in enumerate(_list(item.get("frames"), f"{context}.frames")):
            fcontext = f"{context}.frames[{frame_index}]"
            row = _mapping(raw_frame, fcontext)
            _unknown(row, {"frame_number", "timestamp_seconds", "reviewed", "plates"}, fcontext)
            frame_number = _integer(row.get("frame_number"), f"{fcontext}.frame_number")
            if frame_number in seen_frames:
                raise RealWorldEvaluationError(
                    f"duplicate reviewed frame in {source_id}: {frame_number}")
            seen_frames.add(frame_number)
            timestamp = row.get("timestamp_seconds")
            if timestamp is not None:
                timestamp = _number(timestamp, f"{fcontext}.timestamp_seconds")
            reviewed = _boolean(
                row.get("reviewed", False), f"{fcontext}.reviewed")
            plates: list[FramePlate] = []
            frame_instance_ids: set[str] = set()
            for plate_index, raw_plate in enumerate(_list(row.get("plates"), f"{fcontext}.plates")):
                pcontext = f"{fcontext}.plates[{plate_index}]"
                plate = _mapping(raw_plate, pcontext)
                _unknown(plate, {"instance_id", "bbox", "notes"}, pcontext)
                instance_id = _text(plate.get("instance_id"), f"{pcontext}.instance_id")
                if instance_id not in instance_ids:
                    raise RealWorldEvaluationError(
                        f"{pcontext} references unknown instance_id: {instance_id}")
                if instance_id in frame_instance_ids:
                    raise RealWorldEvaluationError(
                        f"duplicate instance_id in one frame: {instance_id}")
                frame_instance_ids.add(instance_id)
                referenced_instances.add(instance_id)
                plates.append(FramePlate(
                    instance_id=instance_id,
                    bbox=_bbox(plate.get("bbox"), f"{pcontext}.bbox"),
                    notes=_text(plate.get("notes"), f"{pcontext}.notes", optional=True),
                ))
            if plates and not reviewed:
                raise RealWorldEvaluationError(
                    f"{fcontext} cannot contain plates until reviewed is true")
            frames.append(ReviewedFrame(
                frame_number,
                timestamp,
                reviewed,
                tuple(plates),
            ))

        unreferenced = instance_ids - referenced_instances
        if unreferenced:
            raise RealWorldEvaluationError(
                f"{source_id} has plate_instances absent from reviewed frames: "
                + ", ".join(sorted(unreferenced)))
        non_plate_ids = tuple(sorted(
            _text(value, f"{context}.non_plate_observation_event_ids")
            for value in _list(item.get("non_plate_observation_event_ids"),
                               f"{context}.non_plate_observation_event_ids")))
        if len(non_plate_ids) != len(set(non_plate_ids)):
            raise RealWorldEvaluationError(
                f"{source_id} has duplicate non-plate observation event ids")
        overlap = set(non_plate_ids) & seen_events
        if overlap:
            raise RealWorldEvaluationError(
                "observation event cannot label both a plate and non-plate: "
                + ", ".join(sorted(overlap)))
        seen_events.update(non_plate_ids)
        video_value = _text(item.get("video_path"), f"{context}.video_path")
        video_path = Path(video_value)
        if not video_path.is_absolute():
            video_path = manifest_path.parent / video_path
        sources.append(GroundTruthSource(
            source_id=source_id,
            camera_id=camera_id,
            video_path=str(video_path.resolve()),
            observations_reviewed=_boolean(
                item.get("observations_reviewed", False),
                f"{context}.observations_reviewed"),
            non_plate_observation_event_ids=non_plate_ids,
            instances=tuple(sorted(instances, key=lambda value: value.instance_id)),
            frames=tuple(sorted(frames, key=lambda value: value.frame_number)),
        ))
    cross_camera_cases = []
    seen_case_sources: set[str] = set()
    for case_index, raw_case in enumerate(_list(
            root.get("cross_camera_cases"), "cross_camera_cases")):
        context = f"cross_camera_cases[{case_index}]"
        case = _mapping(raw_case, context)
        _unknown(case, {"source_event_id", "relevant_event_ids"}, context)
        source_event_id = _text(
            case.get("source_event_id"), f"{context}.source_event_id")
        if source_event_id in seen_case_sources:
            raise RealWorldEvaluationError(
                f"duplicate cross-camera source event: {source_event_id}")
        seen_case_sources.add(source_event_id)
        relevant = tuple(sorted(
            _text(value, f"{context}.relevant_event_ids")
            for value in _list(
                case.get("relevant_event_ids"), f"{context}.relevant_event_ids")))
        if not relevant:
            raise RealWorldEvaluationError(
                f"{context}.relevant_event_ids must not be empty")
        if len(relevant) != len(set(relevant)):
            raise RealWorldEvaluationError(
                f"{context}.relevant_event_ids contains duplicates")
        if source_event_id in relevant:
            raise RealWorldEvaluationError(
                f"{context}.source_event_id cannot be relevant to itself")
        missing = ({source_event_id} | set(relevant)) - plate_event_ids
        if missing:
            raise RealWorldEvaluationError(
                f"{context} references event IDs not linked to plate instances: "
                + ", ".join(sorted(missing)))
        same_camera = sorted(
            event_id for event_id in relevant
            if source_cameras[plate_event_sources[event_id]]
            == source_cameras[plate_event_sources[source_event_id]]
        )
        if same_camera:
            raise RealWorldEvaluationError(
                f"{context} relevant events must come from another camera: "
                + ", ".join(same_camera))
        cross_camera_cases.append((source_event_id, relevant))
    return RealWorldGroundTruth(
        benchmark_id=benchmark_id,
        schema_version=version,
        sources=tuple(sorted(sources, key=lambda value: value.source_id)),
        cross_camera_cases=tuple(sorted(cross_camera_cases)),
    )


def load_real_world_predictions(path) -> RealWorldPredictions:
    """Load deterministic detector/replay output captured at the fixed baseline."""
    _, root = _read_document(path)
    _unknown(root, {
        "schema_version", "benchmark_id", "provenance", "sources",
        "candidate_rankings",
    }, "predictions")
    version = _integer(root.get("schema_version"), "schema_version", minimum=1)
    if version != 1:
        raise RealWorldEvaluationError("unsupported prediction schema_version")
    sources: list[PredictionSource] = []
    seen_sources: set[str] = set()
    seen_events: set[str] = set()
    for source_index, raw_source in enumerate(_list(root.get("sources"), "sources")):
        context = f"sources[{source_index}]"
        item = _mapping(raw_source, context)
        _unknown(item, {"source_id", "camera_id", "detections", "observations"}, context)
        source_id = _text(item.get("source_id"), f"{context}.source_id")
        if source_id in seen_sources:
            raise RealWorldEvaluationError(f"duplicate prediction source_id: {source_id}")
        seen_sources.add(source_id)
        detections: list[DetectionPrediction] = []
        for index, raw_detection in enumerate(_list(item.get("detections"), f"{context}.detections")):
            dcontext = f"{context}.detections[{index}]"
            row = _mapping(raw_detection, dcontext)
            _unknown(row, {"frame_number", "bbox", "confidence"}, dcontext)
            detections.append(DetectionPrediction(
                frame_number=_integer(row.get("frame_number"), f"{dcontext}.frame_number"),
                bbox=_bbox(row.get("bbox"), f"{dcontext}.bbox"),
                confidence=_number(row.get("confidence"), f"{dcontext}.confidence", maximum=1.0),
            ))
        observations: list[ObservationPrediction] = []
        for index, raw_observation in enumerate(_list(item.get("observations"), f"{context}.observations")):
            ocontext = f"{context}.observations[{index}]"
            row = _mapping(raw_observation, ocontext)
            _unknown(row, {
                "event_id", "best_frame_number", "status", "ocr_text", "plate_raw",
                "plate_normalized", "confidence",
            }, ocontext)
            event_id = _text(row.get("event_id"), f"{ocontext}.event_id")
            if event_id in seen_events:
                raise RealWorldEvaluationError(f"duplicate prediction event_id: {event_id}")
            seen_events.add(event_id)
            status = _text(row.get("status"), f"{ocontext}.status")
            if status not in {"accepted", "review", "abstained"}:
                raise RealWorldEvaluationError(f"{ocontext}.status is invalid")
            frame_number = row.get("best_frame_number")
            if frame_number is not None:
                frame_number = _integer(frame_number, f"{ocontext}.best_frame_number")
            observations.append(ObservationPrediction(
                event_id=event_id,
                best_frame_number=frame_number,
                status=status,
                ocr_text=_text(row.get("ocr_text"), f"{ocontext}.ocr_text", optional=True),
                plate_raw=_text(row.get("plate_raw"), f"{ocontext}.plate_raw", optional=True),
                plate_normalized=_text(
                    row.get("plate_normalized"), f"{ocontext}.plate_normalized", optional=True),
                confidence=_number(row.get("confidence"), f"{ocontext}.confidence", maximum=1.0),
            ))
        sources.append(PredictionSource(
            source_id=source_id,
            camera_id=_text(item.get("camera_id"), f"{context}.camera_id"),
            detections=tuple(detections),
            observations=tuple(sorted(observations, key=lambda value: value.event_id)),
        ))

    rankings: list[CandidateRanking] = []
    ranking_sources: set[str] = set()
    for index, raw_ranking in enumerate(_list(root.get("candidate_rankings"), "candidate_rankings")):
        context = f"candidate_rankings[{index}]"
        row = _mapping(raw_ranking, context)
        _unknown(row, {
            "source_event_id", "candidate_event_ids", "topology_rejections",
            "travel_rejections", "decisions",
        }, context)
        source_event_id = _text(row.get("source_event_id"), f"{context}.source_event_id")
        if source_event_id in ranking_sources:
            raise RealWorldEvaluationError(
                f"duplicate candidate ranking source: {source_event_id}")
        ranking_sources.add(source_event_id)
        candidate_ids = tuple(
            _text(value, f"{context}.candidate_event_ids")
            for value in _list(row.get("candidate_event_ids"), f"{context}.candidate_event_ids"))
        if len(candidate_ids) != len(set(candidate_ids)):
            raise RealWorldEvaluationError(f"{context} has duplicate candidate ids")
        decisions: list[tuple[str, str]] = []
        for decision_index, raw_decision in enumerate(_list(row.get("decisions"), f"{context}.decisions")):
            dcontext = f"{context}.decisions[{decision_index}]"
            decision = _mapping(raw_decision, dcontext)
            _unknown(decision, {"candidate_event_id", "decision"}, dcontext)
            decisions.append((
                _text(decision.get("candidate_event_id"), f"{dcontext}.candidate_event_id"),
                _text(decision.get("decision"), f"{dcontext}.decision"),
            ))
        rankings.append(CandidateRanking(
            source_event_id=source_event_id,
            candidate_event_ids=candidate_ids,
            topology_rejections=_integer(
                row.get("topology_rejections", 0), f"{context}.topology_rejections"),
            travel_rejections=_integer(
                row.get("travel_rejections", 0), f"{context}.travel_rejections"),
            decisions=tuple(decisions),
        ))
    provenance = _mapping(root.get("provenance", {}), "provenance")
    return RealWorldPredictions(
        benchmark_id=_text(root.get("benchmark_id"), "benchmark_id"),
        schema_version=version,
        provenance=provenance,
        sources=tuple(sorted(sources, key=lambda value: value.source_id)),
        candidate_rankings=tuple(sorted(rankings, key=lambda value: value.source_event_id)),
    )


def _canonical_plate(value: Optional[str]) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _file_provenance(path) -> dict:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def _frame_key(source_id: str, frame_number: int) -> str:
    return f"{source_id}:{frame_number}"


def _cross_camera_metrics(
        ground_truth: RealWorldGroundTruth,
        predictions: RealWorldPredictions,
        k_values: tuple[int, ...],
        minimum_cases: int,
) -> dict:
    event_identity: dict[str, str] = {}
    for source in ground_truth.sources:
        for instance in source.instances:
            if instance.vehicle_identity and instance.observation_event_id:
                event_identity[instance.observation_event_id] = instance.vehicle_identity
    cases = []
    ranking_by_source = {
        row.source_event_id: row for row in predictions.candidate_rankings
    }
    for source_event_id, relevant_event_ids in ground_truth.cross_camera_cases:
        ranking = ranking_by_source.get(source_event_id)
        if ranking is None:
            continue
        identity = event_identity.get(source_event_id)
        cases.append((identity, source_event_id, set(relevant_event_ids), ranking))
    if len(cases) < minimum_cases:
        return {
            "available": False,
            "reason": (
                f"requires at least {minimum_cases} independently labeled source cases; "
                f"found {len(cases)}"
            ),
            "labeled_cases": len(cases),
            "metrics": None,
        }
    total_relevant = sum(len(case[2]) for case in cases)
    recall = {}
    hit_rate = {}
    reciprocal_rank = 0.0
    case_rows = []
    for identity, source_event_id, relevant, ranking in cases:
        first_rank = next((
            index for index, event_id in enumerate(ranking.candidate_event_ids, start=1)
            if event_id in relevant
        ), None)
        if first_rank:
            reciprocal_rank += 1.0 / first_rank
        retrieved = {}
        for k in k_values:
            retrieved[str(k)] = len(relevant.intersection(ranking.candidate_event_ids[:k]))
        case_rows.append({
            "vehicle_identity": identity,
            "source_event_id": source_event_id,
            "relevant_event_ids": sorted(relevant),
            "candidate_event_ids": list(ranking.candidate_event_ids),
            "first_relevant_rank": first_rank,
            "retrieved_relevant_at_k": retrieved,
        })
    for k in k_values:
        recalled = sum(row["retrieved_relevant_at_k"][str(k)] for row in case_rows)
        hits = sum(row["retrieved_relevant_at_k"][str(k)] > 0 for row in case_rows)
        recall[str(k)] = recalled / total_relevant if total_relevant else 0.0
        hit_rate[str(k)] = hits / len(cases)
    correct_decisions = incorrect_decisions = 0
    relevant_by_source = {
        source_event_id: relevant
        for _, source_event_id, relevant, _ in cases
    }
    for _, source_event_id, _, ranking in cases:
        relevant = relevant_by_source[source_event_id]
        for candidate_event_id, decision in ranking.decisions:
            if decision not in {"possible_weak", "possible_strong"}:
                continue
            if candidate_event_id in relevant:
                correct_decisions += 1
            elif (candidate_event_id in event_identity
                  and source_event_id in event_identity):
                incorrect_decisions += 1
    return {
        "available": True,
        "reason": None,
        "labeled_cases": len(cases),
        "metrics": {
            "total_relevant_events": total_relevant,
            "recall_at_k": recall,
            "hit_rate_at_k": hit_rate,
            "mean_reciprocal_rank": reciprocal_rank / len(cases),
            "correct_positive_identity_decisions": correct_decisions,
            "incorrect_positive_identity_decisions": incorrect_decisions,
            "topology_rejections": sum(case[3].topology_rejections for case in cases),
            "travel_rejections": sum(case[3].travel_rejections for case in cases),
            "cases": case_rows,
        },
    }


def evaluate_real_world(
        ground_truth: RealWorldGroundTruth,
        predictions: RealWorldPredictions,
        *,
        primary_iou: float = 0.5,
        diagnostic_iou: float = 0.3,
        cross_camera_k: tuple[int, ...] = (1, 3, 5),
        minimum_cross_camera_cases: int = 2,
) -> dict:
    """Calculate stage-separated metrics without modifying model behavior."""
    if (isinstance(minimum_cross_camera_cases, bool)
            or not isinstance(minimum_cross_camera_cases, int)
            or minimum_cross_camera_cases < 2):
        raise RealWorldEvaluationError(
            "minimum_cross_camera_cases must be an integer >= 2")
    if (not cross_camera_k or any(
            isinstance(k, bool) or not isinstance(k, int) or k < 1
            for k in cross_camera_k)):
        raise RealWorldEvaluationError(
            "cross_camera_k must contain positive integers")
    if ground_truth.benchmark_id != predictions.benchmark_id:
        raise RealWorldEvaluationError(
            "ground truth and predictions use different benchmark_id values")
    gt_sources = {source.source_id: source for source in ground_truth.sources}
    pred_sources = {source.source_id: source for source in predictions.sources}
    if set(gt_sources) != set(pred_sources):
        raise RealWorldEvaluationError(
            "ground-truth and prediction source_id sets must match exactly")
    for source_id in gt_sources:
        if gt_sources[source_id].camera_id != pred_sources[source_id].camera_id:
            raise RealWorldEvaluationError(
                f"camera_id mismatch for source {source_id}")

    truth: dict[str, list[BBox]] = {}
    predicted: dict[str, list[tuple[BBox, float]]] = {}
    frame_plates: dict[str, tuple[FramePlate, ...]] = {}
    reviewed_keys: set[str] = set()
    ignored_prediction_count = 0
    for source_id, source in gt_sources.items():
        for frame in source.frames:
            if not frame.reviewed:
                continue
            key = _frame_key(source_id, frame.frame_number)
            reviewed_keys.add(key)
            truth[key] = [plate.bbox for plate in frame.plates]
            frame_plates[key] = frame.plates
        for detection in pred_sources[source_id].detections:
            key = _frame_key(source_id, detection.frame_number)
            if key not in reviewed_keys:
                ignored_prediction_count += 1
                continue
            predicted.setdefault(key, []).append((detection.bbox, detection.confidence))

    primary = evaluate_detections(truth, predicted, primary_iou)
    diagnostic = evaluate_detections(truth, predicted, diagnostic_iou)
    matched_instances: set[tuple[str, str]] = set()
    for match in primary.matches:
        source_id = match.image_id.split(":", 1)[0]
        matched_instances.add((
            source_id,
            frame_plates[match.image_id][match.ground_truth_index].instance_id,
        ))

    unmatched_prediction_rows = []
    for key, indices in sorted(primary.unmatched_predictions.items()):
        source_id, frame_text = key.rsplit(":", 1)
        rows = predicted.get(key, ())
        for index in indices:
            bbox, confidence = rows[index]
            unmatched_prediction_rows.append({
                "source_id": source_id,
                "frame_number": int(frame_text),
                "bbox": list(bbox),
                "confidence": confidence,
            })
    unmatched_ground_truth_rows = []
    for key, indices in sorted(primary.unmatched_ground_truth.items()):
        source_id, frame_text = key.rsplit(":", 1)
        rows = frame_plates.get(key, ())
        for index in indices:
            plate = rows[index]
            unmatched_ground_truth_rows.append({
                "source_id": source_id,
                "frame_number": int(frame_text),
                "instance_id": plate.instance_id,
                "bbox": list(plate.bbox),
            })

    observation_by_id = {
        observation.event_id: observation
        for source in predictions.sources
        for observation in source.observations
    }
    observation_source_by_id = {
        observation.event_id: source.source_id
        for source in predictions.sources
        for observation in source.observations
    }
    linked_ids = {
        instance.observation_event_id
        for source in ground_truth.sources
        for instance in source.instances
        if instance.observation_event_id
    }
    labeled_nonplates = {
        event_id
        for source in ground_truth.sources
        for event_id in source.non_plate_observation_event_ids
    }
    unknown_labeled_events = sorted((linked_ids | labeled_nonplates) - set(observation_by_id))
    if unknown_labeled_events:
        raise RealWorldEvaluationError(
            "ground truth references prediction event IDs that do not exist: "
            + ", ".join(unknown_labeled_events))
    for source in ground_truth.sources:
        expected_source = source.source_id
        labeled_ids = set(source.non_plate_observation_event_ids) | {
            instance.observation_event_id
            for instance in source.instances
            if instance.observation_event_id
        }
        wrong_source = sorted(
            event_id for event_id in labeled_ids
            if observation_source_by_id[event_id] != expected_source
        )
        if wrong_source:
            raise RealWorldEvaluationError(
                f"{expected_source} labels observations from another source: "
                + ", ".join(wrong_source))

    ocr_pairs = []
    ocr_rows = []
    excluded_detector_misses = excluded_unlinked = 0
    end_to_end_rows = []
    linkage_incomplete = []
    for source in ground_truth.sources:
        for instance in source.instances:
            if not instance.text_evaluable:
                continue
            expected = _canonical_plate(instance.plate_text)
            detected = (source.source_id, instance.instance_id) in matched_instances
            observation = observation_by_id.get(instance.observation_event_id or "")
            if not instance.observation_link_reviewed:
                linkage_incomplete.append(f"{source.source_id}:{instance.instance_id}")
            predicted_text = None
            if observation is not None:
                predicted_text = observation.ocr_text
                if predicted_text is None:
                    predicted_text = observation.plate_raw
            canonical_prediction = _canonical_plate(predicted_text)
            if not detected:
                excluded_detector_misses += 1
            elif observation is None:
                excluded_unlinked += 1
            else:
                ocr_pairs.append((expected, canonical_prediction))
                ocr_rows.append({
                    "source_id": source.source_id,
                    "instance_id": instance.instance_id,
                    "event_id": observation.event_id,
                    "expected_text": expected,
                    "predicted_text": canonical_prediction,
                    "correct": expected == canonical_prediction,
                })
            correct = bool(detected and observation is not None
                           and expected == canonical_prediction)
            end_to_end_rows.append({
                "source_id": source.source_id,
                "instance_id": instance.instance_id,
                "event_id": observation.event_id if observation else None,
                "detected": detected,
                "expected_text": expected,
                "predicted_text": canonical_prediction if observation else None,
                "correct": correct,
                "status": observation.status if observation else "no_observation",
            })

    ocr_metrics = evaluate_ocr(ocr_pairs)
    confusions = Counter(
        f"{expected}->{predicted}"
        for expected_text, predicted_text in ocr_pairs
        for expected, predicted in aligned_substitutions(expected_text, predicted_text)
    )
    e2e_available = bool(end_to_end_rows) and not linkage_incomplete
    e2e_correct = sum(row["correct"] for row in end_to_end_rows)
    status_counts = Counter(row["status"] for row in end_to_end_rows)
    accepted_rows = [row for row in end_to_end_rows if row["status"] == "accepted"]
    correct_deferred = [
        row for row in end_to_end_rows
        if row["correct"] and row["status"] in {"review", "abstained"}
    ]
    confirmed_nonplates = []
    for event_id in sorted(labeled_nonplates):
        observation = observation_by_id[event_id]
        confirmed_nonplates.append({
            "event_id": event_id,
            "status": observation.status,
            "plate_raw": observation.plate_raw,
            "plate_normalized": observation.plate_normalized,
            "confidence": observation.confidence,
        })
    unadjudicated = sorted(set(observation_by_id) - linked_ids - labeled_nonplates)
    fully_reviewed = all(source.observations_reviewed for source in ground_truth.sources)
    if fully_reviewed and unadjudicated:
        raise RealWorldEvaluationError(
            "observations_reviewed is true but prediction events remain unadjudicated: "
            + ", ".join(unadjudicated))

    return {
        "benchmark_id": ground_truth.benchmark_id,
        "metric_definitions": {
            "detection_precision": "TP / (TP + FP)",
            "detection_recall": "TP / (TP + FN)",
            "detection_f1": "2 * precision * recall / (precision + recall)",
            "ocr_exact_match": (
                "canonical exact OCR matches / detector-matched, text-evaluable, "
                "manually linked plate instances"
            ),
            "end_to_end_exact": (
                "detected and canonical-exact recognized GT plate instances / "
                "all text-evaluable GT plate instances"
            ),
            "accepted_precision": (
                "correct accepted GT plate instances / accepted GT plate instances"
            ),
            "accepted_coverage": (
                "accepted GT plate instances / all text-evaluable GT plate instances"
            ),
        },
        "detection": {
            "available": bool(truth) and any(truth.values()),
            "reason": (
                None if truth and any(truth.values())
                else "no reviewed ground-truth plate boxes"
            ),
            "reviewed_frames": len(reviewed_keys),
            "ground_truth_plate_boxes": sum(len(value) for value in truth.values()),
            "primary": primary.metrics.to_dict(),
            "diagnostic": diagnostic.metrics.to_dict(),
            "false_negative_boxes": unmatched_ground_truth_rows,
            "predictions_outside_reviewed_frames_ignored": ignored_prediction_count,
        },
        "ocr": {
            "available": bool(ocr_pairs),
            "correct_count": ocr_metrics.exact_matches,
            "evaluable_count": ocr_metrics.samples,
            "exact_match_accuracy": (
                ocr_metrics.exact_accuracy if ocr_metrics.samples else None),
            "character_accuracy": (
                ocr_metrics.character_accuracy if ocr_metrics.samples else None),
            "excluded_detector_misses": excluded_detector_misses,
            "excluded_unlinked_observations": excluded_unlinked,
            "character_confusions": dict(confusions.most_common()),
            "outcomes": ocr_rows,
        },
        "end_to_end": {
            "available": e2e_available,
            "reason": None if e2e_available else (
                "no text-evaluable ground truth" if not end_to_end_rows else
                "manual observation linkage is incomplete"
            ),
            "correct_count": e2e_correct if e2e_available else None,
            "evaluable_count": len(end_to_end_rows) if e2e_available else None,
            "exact_recognition": (
                _safe_ratio(e2e_correct, len(end_to_end_rows)) if e2e_available else None),
            "incomplete_links": linkage_incomplete,
            "outcomes": end_to_end_rows,
        },
        "confidence_status": {
            "available": e2e_available,
            "accepted": status_counts["accepted"],
            "review": status_counts["review"],
            "abstained": status_counts["abstained"],
            "no_observation": status_counts["no_observation"],
            "accepted_correct": sum(row["correct"] for row in accepted_rows),
            "accepted_only_precision": (
                _safe_ratio(sum(row["correct"] for row in accepted_rows),
                            len(accepted_rows)) if e2e_available else None),
            "coverage": (
                _safe_ratio(len(accepted_rows), len(end_to_end_rows))
                if e2e_available else None),
            "incorrect_accepted_observations": [
                row for row in accepted_rows if not row["correct"]
            ],
            "correct_sent_to_review_or_abstained": correct_deferred,
        },
        "false_positive_analysis": {
            "unmatched_detector_predictions": unmatched_prediction_rows,
            "confirmed_non_plate_observations": confirmed_nonplates,
            "unadjudicated_observation_event_ids": unadjudicated,
            "observation_adjudication_complete": fully_reviewed and not unadjudicated,
        },
        "cross_camera": _cross_camera_metrics(
            ground_truth, predictions, tuple(sorted(set(cross_camera_k))),
            minimum_cross_camera_cases),
        "provenance": dict(predictions.provenance),
    }


def _format_ratio(value: Optional[float]) -> str:
    return "not available" if value is None else f"{value:.4f}"


def real_world_markdown(report: dict) -> str:
    """Render a concise, stage-separated human report from calculated data."""
    detection = report["detection"]
    primary = detection["primary"]
    diagnostic = detection["diagnostic"]
    ocr = report["ocr"]
    end_to_end = report["end_to_end"]
    confidence = report["confidence_status"]
    false_positive = report["false_positive_analysis"]
    cross_camera = report["cross_camera"]
    lines = [
        f"# Step 34 benchmark result — {report['benchmark_id']}",
        "",
        "No metric in this report is inferred from predictions; all denominators "
        "come from the supplied manual ground truth.",
        "",
        "## Plate detection",
        "",
        f"- Reviewed frames: {detection['reviewed_frames']}",
        f"- GT plate boxes: {detection['ground_truth_plate_boxes']}",
    ]
    if detection["available"]:
        lines.extend([
            f"- Primary IoU: {primary['iou_threshold']}",
            "- TP / FP / FN: "
            f"{primary['true_positives']} / {primary['false_positives']} / "
            f"{primary['false_negatives']}",
            "- Precision / recall / F1: "
            f"{primary['precision']:.4f} / {primary['recall']:.4f} / "
            f"{primary['f1']:.4f}",
            f"- Diagnostic IoU {diagnostic['iou_threshold']} F1: "
            f"{diagnostic['f1']:.4f}",
        ])
    else:
        lines.append(f"- Not available: {detection['reason']}")
    lines.extend([
        "",
        "## OCR (detector-matched evaluable plates only)",
        "",
        f"- Exact matches: {ocr['correct_count']} / {ocr['evaluable_count']}",
        f"- Exact-match accuracy: {_format_ratio(ocr['exact_match_accuracy'])}",
        f"- Detector misses excluded from OCR: {ocr['excluded_detector_misses']}",
        "",
        "## End-to-end recognition",
        "",
    ])
    if end_to_end["available"]:
        lines.extend([
            f"- Exact correct: {end_to_end['correct_count']} / {end_to_end['evaluable_count']}",
            f"- Exact recognition: {end_to_end['exact_recognition']:.4f}",
        ])
    else:
        lines.append(f"- Not available: {end_to_end['reason']}")
    lines.extend([
        "",
        "## Confidence/status behavior",
        "",
        "- Accepted / review / abstained: "
        f"{confidence['accepted']} / {confidence['review']} / "
        f"{confidence['abstained']}",
        f"- Accepted-only precision: {_format_ratio(confidence['accepted_only_precision'])}",
        f"- Coverage: {_format_ratio(confidence['coverage'])}",
        f"- Incorrect accepted: {len(confidence['incorrect_accepted_observations'])}",
        "",
        "## False positives",
        "",
        f"- Unmatched detector predictions: {len(false_positive['unmatched_detector_predictions'])}",
        f"- Manually confirmed non-plate observations: {len(false_positive['confirmed_non_plate_observations'])}",
        f"- Unadjudicated observations: {len(false_positive['unadjudicated_observation_event_ids'])}",
        "",
        "## Failure breakdown",
        "",
        f"- Detector false-negative boxes: {len(detection['false_negative_boxes'])}",
        f"- OCR character substitutions: {sum(ocr['character_confusions'].values())}",
        "- OCR confusion pairs: " + (
            ", ".join(
                f"{pair} ({count})"
                for pair, count in ocr["character_confusions"].items()
            ) or "none"
        ),
        "- Confirmed non-plate outputs: " + (
            ", ".join(
                f"{row['event_id']}={row['plate_raw']!r}"
                for row in false_positive["confirmed_non_plate_observations"]
            ) or "none"
        ),
        "- Incorrect accepted event IDs: " + (
            ", ".join(
                str(row["event_id"])
                for row in confidence["incorrect_accepted_observations"]
            ) or "none"
        ),
        "",
        "## Metric definitions",
        "",
        "- Detection precision = TP / (TP + FP).",
        "- Detection recall = TP / (TP + FN).",
        "- Detection F1 = 2 × precision × recall / (precision + recall).",
        "- OCR exact match = canonical exact matches / detector-matched, "
        "text-evaluable, linked GT plate instances.",
        "- End-to-end exact recognition = detected canonical-exact GT plate "
        "instances / all text-evaluable GT plate instances.",
        "- Accepted-only precision = correct accepted / all accepted GT "
        "plate instances; coverage = accepted / all text-evaluable GT plate "
        "instances.",
        "",
        "## Cross-camera retrieval",
        "",
    ])
    if cross_camera["available"]:
        metrics = cross_camera["metrics"]
        lines.append(f"- Labeled source cases: {cross_camera['labeled_cases']}")
        for k, value in metrics["recall_at_k"].items():
            lines.append(f"- Recall@{k}: {value:.4f}")
        for k, value in metrics["hit_rate_at_k"].items():
            lines.append(f"- HitRate@{k}: {value:.4f}")
        lines.append(f"- MRR: {metrics['mean_reciprocal_rank']:.4f}")
    else:
        lines.append(f"- Not reported: {cross_camera['reason']}")
    return "\n".join(lines) + "\n"


def run_real_world_evaluation(
        ground_truth_path,
        predictions_path,
        output_dir,
        *,
        primary_iou: float = 0.5,
        diagnostic_iou: float = 0.3,
        cross_camera_k: tuple[int, ...] = (1, 3, 5),
        minimum_cross_camera_cases: int = 2,
) -> dict:
    """Load inputs, evaluate, add reproducibility metadata, and write reports."""
    ground_truth = load_real_world_ground_truth(ground_truth_path)
    predictions = load_real_world_predictions(predictions_path)
    report = evaluate_real_world(
        ground_truth,
        predictions,
        primary_iou=primary_iou,
        diagnostic_iou=diagnostic_iou,
        cross_camera_k=cross_camera_k,
        minimum_cross_camera_cases=minimum_cross_camera_cases,
    )
    report["run_metadata"] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ground_truth_path": str(Path(ground_truth_path).resolve()),
        "ground_truth_sha256": _sha256(ground_truth_path),
        "predictions_path": str(Path(predictions_path).resolve()),
        "predictions_sha256": _sha256(predictions_path),
        "primary_iou": primary_iou,
        "diagnostic_iou": diagnostic_iou,
        "cross_camera_k": list(cross_camera_k),
        "minimum_cross_camera_cases": minimum_cross_camera_cases,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "step34_real_world_benchmark.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "step34_real_world_benchmark.md").write_text(
        real_world_markdown(report), encoding="utf-8")
    return report


def capture_real_world_predictions(
        ground_truth_path,
        scenario_path,
        city_config_path,
        pipeline_config_path,
        output_path,
        *,
        candidate_window_seconds: Optional[float] = None,
        candidate_max_results: Optional[int] = None,
        candidate_maximum_speed_kph: Optional[float] = None,
) -> dict:
    """Run the unchanged configured detector/replay and capture evaluation data.

    Detection is run only for explicitly reviewed frames.  The production
    scenario replay still runs normally, but uses an in-memory repository and
    no evidence store so capture cannot contaminate the normal Step 33 outputs.
    """
    from phase1_anpr.detection.detector import PlateDetector
    from phase1_anpr.persistence import SQLiteObservationRepository
    from phase1_anpr.utils.config import load_config
    from phase1_anpr.video.video_reader import VideoReader
    from phase2_city.candidate_collection import (
        CandidateCollectionPolicy,
        ReplayCandidateCollector,
    )
    from phase2_city.config.loader import load_city_config
    from phase2_city.cross_camera_matching import (
        CrossCameraCandidateMatcher,
        CrossCameraMatchPolicy,
    )
    from phase2_city.graph import CityCameraGraph
    from phase2_city.replay import run_scenario
    from phase2_city.scenario import load_scenario

    ground_truth = load_real_world_ground_truth(ground_truth_path)
    config = load_config(pipeline_config_path)
    cameras, links = load_city_config(city_config_path)
    graph = CityCameraGraph(cameras, links)
    scenario = load_scenario(scenario_path, graph)
    gt_by_source = {source.source_id: source for source in ground_truth.sources}
    scenario_by_source = {source.source_id: source for source in scenario.sources}
    if set(gt_by_source) != set(scenario_by_source):
        raise RealWorldEvaluationError(
            "ground-truth and replay scenario source_id sets must match exactly")
    for source_id, gt_source in gt_by_source.items():
        scenario_source = scenario_by_source[source_id]
        if scenario_source.camera_id != gt_source.camera_id:
            raise RealWorldEvaluationError(
                f"camera_id mismatch between labels and scenario for {source_id}")
        if Path(scenario_source.video_path).resolve() != Path(gt_source.video_path).resolve():
            raise RealWorldEvaluationError(
                f"video_path mismatch between labels and scenario for {source_id}")

    detection_config = config["detection"]
    detector = PlateDetector(
        weights_path=detection_config["weights_path"],
        device=detection_config.get("device", "auto"),
        confidence_threshold=detection_config.get("confidence_threshold", 0.35),
        iou_threshold=detection_config.get("iou_threshold", 0.45),
        plate_class_id=detection_config.get("plate_class_id", 0),
        image_size=detection_config.get("image_size", 640),
        save_crops=False,
    )
    process_fps = (config.get("video", {}) or {}).get("process_fps")
    detection_rows: dict[str, list[dict]] = {}
    for scenario_source in scenario.sources:
        reviewed = {
            frame.frame_number
            for frame in gt_by_source[scenario_source.source_id].frames
            if frame.reviewed
        }
        found: set[int] = set()
        rows = []
        with VideoReader(
                scenario_source.video_path,
                scenario_source.camera_id,
                process_fps) as reader:
            for processed_frame in reader.frames():
                if processed_frame.frame_number not in reviewed:
                    continue
                found.add(processed_frame.frame_number)
                for detection in detector.detect(
                        processed_frame.frame, processed_frame.frame_number):
                    rows.append({
                        "frame_number": detection.frame_number,
                        "bbox": list(detection.bbox),
                        "confidence": detection.confidence,
                    })
        missing = sorted(reviewed - found)
        if missing:
            raise RealWorldEvaluationError(
                f"{scenario_source.source_id} labels frames that are not selected "
                f"at process_fps={process_fps}: {missing}")
        detection_rows[scenario_source.source_id] = sorted(
            rows, key=lambda row: (row["frame_number"], row["bbox"], -row["confidence"]))

    repository = SQLiteObservationRepository(":memory:")
    scenario_result = run_scenario(
        scenario,
        config,
        observation_repo=repository,
        evidence_store=None,
        watchlist_repo=None,
    )
    result_by_source = {
        result.source.source_id: result for result in scenario_result.source_results
    }
    output_sources = []
    for source_id in sorted(gt_by_source):
        observations = []
        for pipeline_result in result_by_source[source_id].results:
            observation = pipeline_result.observation
            observations.append({
                "event_id": observation.event_id,
                "best_frame_number": observation.best_frame_number,
                "status": observation.status,
                "ocr_text": observation.plate_raw,
                "plate_raw": observation.plate_raw,
                "plate_normalized": observation.plate_normalized,
                "confidence": observation.confidence,
            })
        output_sources.append({
            "source_id": source_id,
            "camera_id": gt_by_source[source_id].camera_id,
            "detections": detection_rows[source_id],
            "observations": sorted(observations, key=lambda row: row["event_id"]),
        })

    candidate_values = (
        candidate_window_seconds,
        candidate_max_results,
        candidate_maximum_speed_kph,
    )
    if any(value is not None for value in candidate_values) and not all(
            value is not None for value in candidate_values):
        raise RealWorldEvaluationError(
            "candidate capture requires window, max-results, and maximum-speed together")
    candidate_rankings = []
    candidate_policy = None
    if all(value is not None for value in candidate_values):
        match_policy = CrossCameraMatchPolicy(
            maximum_speed_kph=candidate_maximum_speed_kph)
        collection_policy = CandidateCollectionPolicy(
            max_time_delta_seconds=candidate_window_seconds,
            max_candidates=candidate_max_results)
        collector = ReplayCandidateCollector(
            CrossCameraCandidateMatcher(graph, match_policy), collection_policy)
        source_events = [
            source_event_id
            for source_event_id, _ in ground_truth.cross_camera_cases
        ]
        for source_event_id in source_events:
            result = collector.collect(scenario_result, source_event_id)
            candidate_rankings.append({
                "source_event_id": source_event_id,
                "candidate_event_ids": [
                    candidate.candidate.event_id for candidate in result.candidates
                ],
                "topology_rejections": sum(
                    candidate.topology_status != "direct_link"
                    for candidate in result.ranked_results),
                "travel_rejections": sum(
                    candidate.travel_status == "too_fast"
                    for candidate in result.ranked_results),
                "decisions": [{
                    "candidate_event_id": candidate.candidate.event_id,
                    "decision": candidate.decision,
                } for candidate in result.ranked_results],
            })
        candidate_policy = {
            "window_seconds": candidate_window_seconds,
            "max_results": candidate_max_results,
            "maximum_speed_kph": candidate_maximum_speed_kph,
        }

    weights_path = Path(detection_config["weights_path"])
    videos = {
        source.source_id: _file_provenance(source.video_path)
        for source in scenario.sources
    }
    payload = {
        "schema_version": 1,
        "benchmark_id": ground_truth.benchmark_id,
        "provenance": {
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "scenario_id": scenario.scenario_id,
            "scenario": _file_provenance(scenario_path),
            "city_config": _file_provenance(city_config_path),
            "pipeline_config": _file_provenance(pipeline_config_path),
            "detector_weights": _file_provenance(weights_path),
            "videos": videos,
            "fixed_settings": {
                "process_fps": process_fps,
                "detector": dict(detection_config),
                "ocr": dict(config.get("ocr", {})),
                "vehicle_enrichment": dict(config.get("vehicle_enrichment", {})),
                "candidate_policy": candidate_policy,
            },
            "model_version": (config.get("models", {}) or {}).get("model_version"),
        },
        "sources": output_sources,
        "candidate_rankings": candidate_rankings,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    repository.close()
    return payload


def prepare_real_world_annotation_workspace(
        scenario_path,
        city_config_path,
        pipeline_config_path,
        output_path,
        *,
        benchmark_id: str = "step34-step33-real-videos-v1",
        frames_dir=None,
) -> dict:
    """Create an unlabeled manifest and JPEGs for every processed replay frame.

    Generated frames start with ``reviewed: false`` and therefore contribute no
    ground truth until a human explicitly reviews them. Existing manifests are
    never overwritten.
    """
    import cv2

    from phase1_anpr.utils.config import load_config
    from phase1_anpr.video.video_reader import VideoReader
    from phase2_city.config.loader import load_city_config
    from phase2_city.graph import CityCameraGraph
    from phase2_city.scenario import load_scenario

    output = Path(output_path).resolve()
    if output.exists():
        raise RealWorldEvaluationError(
            f"refusing to overwrite existing annotation manifest: {output}")
    frame_root = (
        Path(frames_dir).resolve() if frames_dir is not None
        else output.parent / "frames"
    )
    config = load_config(pipeline_config_path)
    cameras, links = load_city_config(city_config_path)
    scenario = load_scenario(
        scenario_path, CityCameraGraph(cameras, links))
    process_fps = (config.get("video", {}) or {}).get("process_fps")
    sources = []
    for source in scenario.sources:
        source_frame_dir = frame_root / source.source_id
        source_frame_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        with VideoReader(
                source.video_path, source.camera_id, process_fps) as reader:
            for processed_frame in reader.frames():
                image_path = source_frame_dir / (
                    f"frame_{processed_frame.frame_number:08d}.jpg")
                if not cv2.imwrite(str(image_path), processed_frame.frame):
                    raise RealWorldEvaluationError(
                        f"failed to write annotation frame: {image_path}")
                frames.append({
                    "frame_number": processed_frame.frame_number,
                    "timestamp_seconds": processed_frame.video_timestamp,
                    "reviewed": False,
                    "plates": [],
                })
        sources.append({
            "source_id": source.source_id,
            "camera_id": source.camera_id,
            "video_path": source.video_path,
            "observations_reviewed": False,
            "non_plate_observation_event_ids": [],
            "plate_instances": [],
            "frames": frames,
        })
    payload = {
        "schema_version": 1,
        "benchmark_id": _text(benchmark_id, "benchmark_id"),
        "sources": sources,
        "cross_camera_cases": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return {
        "ground_truth_path": str(output),
        "frames_dir": str(frame_root),
        "source_frame_counts": {
            source["source_id"]: len(source["frames"]) for source in sources
        },
        "process_fps": process_fps,
    }


__all__ = [
    "RealWorldEvaluationError",
    "RealWorldGroundTruth",
    "RealWorldPredictions",
    "capture_real_world_predictions",
    "evaluate_real_world",
    "load_real_world_ground_truth",
    "load_real_world_predictions",
    "prepare_real_world_annotation_workspace",
    "real_world_markdown",
    "run_real_world_evaluation",
]
