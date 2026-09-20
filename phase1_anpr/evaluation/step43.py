"""Read-only Step 43 paired benchmark and ablation reporting.

The module consumes prediction-blind Step 34 ground truth, a frozen Phase 2
prediction capture, and an explicitly captured Phase 3 evidence bundle.  It
does not run or change any production matcher, retrieval, model, persistence,
or hypothesis code.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence

import yaml

from phase1_anpr.evaluation.real_world import (
    RealWorldGroundTruth,
    RealWorldPredictions,
    evaluate_real_world,
    load_real_world_ground_truth,
    load_real_world_predictions,
)


STEP43_SCHEMA_VERSION = 1
STEP43_PHASE3_RESULTS_SCHEMA_VERSION = 1
STEP43_IMPLEMENTATION_ID = "citysight-step43a-v1"
FROZEN_PHASE2_REFERENCE = "a86baa8"
PHASE3_UNORDERED_REASON = (
    "Phase 3 Step 39 supplies bounded chronological retrieval order, not an "
    "approved relevance ranking; Recall@K, HitRate@K, and MRR are unavailable"
)


class Step43EvaluationError(ValueError):
    """Raised when Step 43 inputs are malformed or inconsistent."""


def _mapping(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise Step43EvaluationError(f"{name} must be a mapping")
    return value


def _sequence(value, name: str) -> list:
    if not isinstance(value, list):
        raise Step43EvaluationError(f"{name} must be a list")
    return value


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Step43EvaluationError(f"{name} must be a non-empty string")
    return value


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Step43EvaluationError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Step43EvaluationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Step43EvaluationError(f"{name} must be finite")
    return result


def _timestamp(value, name: str) -> str:
    value = _text(value, name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Step43EvaluationError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Step43EvaluationError(f"{name} must include a timezone")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(value, name: str) -> str:
    value = _text(value, name).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise Step43EvaluationError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _resolve(base: Path, value, name: str) -> Path:
    raw = Path(_text(value, name))
    result = raw if raw.is_absolute() else base / raw
    return result.resolve()


def _file_reference(value, name: str, base: Path) -> dict:
    row = _mapping(value, name)
    unknown = set(row) - {"path", "sha256"}
    if unknown:
        raise Step43EvaluationError(f"{name} has unknown fields: {sorted(unknown)}")
    path = _resolve(base, row.get("path"), f"{name}.path")
    expected = _hash(row.get("sha256"), f"{name}.sha256")
    if not path.is_file():
        raise Step43EvaluationError(f"{name}.path does not exist: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise Step43EvaluationError(
            f"{name} SHA-256 mismatch: expected {expected}, found {actual}")
    return {"path": str(path), "sha256": actual, "size_bytes": path.stat().st_size}


def _json_value(value, name: str):
    """Validate provenance/policy data as finite deterministic JSON."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Step43EvaluationError(f"{name} must not contain NaN or Infinity")
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{name}[{index}]")
                for index, item in enumerate(value)]
    if isinstance(value, dict):
        result = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise Step43EvaluationError(f"{name} keys must be non-empty strings")
            result[key] = _json_value(value[key], f"{name}.{key}")
        return result
    raise Step43EvaluationError(f"{name} contains a non-JSON value")


def _named_policy(value, name: str) -> dict:
    row = _mapping(value, name)
    if not row:
        raise Step43EvaluationError(f"{name} must not be empty")
    return _json_value(row, name)


def load_step43_manifest(path) -> dict:
    """Load and hash-validate a Step 43 run manifest."""
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise Step43EvaluationError(f"manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    root = _mapping(raw, "manifest")
    allowed = {
        "schema_version", "benchmark_id", "created_at", "evaluation_at", "ground_truth",
        "sources", "phase2", "phase3", "notes",
    }
    unknown = set(root) - allowed
    if unknown:
        raise Step43EvaluationError(f"manifest has unknown fields: {sorted(unknown)}")
    if _integer(root.get("schema_version"), "schema_version", 1) != STEP43_SCHEMA_VERSION:
        raise Step43EvaluationError("unsupported Step 43 manifest schema_version")
    benchmark_id = _text(root.get("benchmark_id"), "benchmark_id")
    created_at = _timestamp(root.get("created_at"), "created_at")
    evaluation_at = _timestamp(root.get("evaluation_at"), "evaluation_at")
    base = manifest_path.parent
    ground_truth = _file_reference(root.get("ground_truth"), "ground_truth", base)

    sources = []
    seen_sources = set()
    for index, value in enumerate(_sequence(root.get("sources"), "sources")):
        name = f"sources[{index}]"
        row = _mapping(value, name)
        unknown = set(row) - {"source_id", "camera_id", "video", "metadata"}
        if unknown:
            raise Step43EvaluationError(f"{name} has unknown fields: {sorted(unknown)}")
        source_id = _text(row.get("source_id"), f"{name}.source_id")
        camera_id = _text(row.get("camera_id"), f"{name}.camera_id")
        if source_id in seen_sources:
            raise Step43EvaluationError(f"duplicate source_id: {source_id}")
        seen_sources.add(source_id)
        sources.append({
            "source_id": source_id,
            "camera_id": camera_id,
            "video": _file_reference(row.get("video"), f"{name}.video", base),
            "metadata": _json_value(row.get("metadata", {}), f"{name}.metadata"),
        })
    if not sources:
        raise Step43EvaluationError("sources must contain at least one source")

    phase2_raw = _mapping(root.get("phase2"), "phase2")
    required_phase2 = {
        "baseline_reference", "matcher_identifier", "predictions", "artifacts",
        "retrieval_policy", "physical_policy",
    }
    if set(phase2_raw) != required_phase2:
        raise Step43EvaluationError(
            f"phase2 fields must be exactly {sorted(required_phase2)}")
    baseline = _text(phase2_raw["baseline_reference"], "phase2.baseline_reference")
    if baseline != FROZEN_PHASE2_REFERENCE:
        raise Step43EvaluationError(
            f"phase2.baseline_reference must be {FROZEN_PHASE2_REFERENCE}")
    artifacts_raw = _mapping(phase2_raw["artifacts"], "phase2.artifacts")
    required_artifacts = {
        "config", "city_config", "baseline_replay_config", "benchmark_scenario"}
    if set(artifacts_raw) != required_artifacts:
        raise Step43EvaluationError(
            f"phase2.artifacts must be exactly {sorted(required_artifacts)}")
    phase2 = {
        "baseline_reference": baseline,
        "matcher_identifier": _text(
            phase2_raw["matcher_identifier"], "phase2.matcher_identifier"),
        "predictions": _file_reference(
            phase2_raw["predictions"], "phase2.predictions", base),
        "artifacts": {
            key: _file_reference(artifacts_raw[key], f"phase2.artifacts.{key}", base)
            for key in sorted(artifacts_raw)
        },
        "retrieval_policy": _named_policy(
            phase2_raw["retrieval_policy"], "phase2.retrieval_policy"),
        "physical_policy": _named_policy(
            phase2_raw["physical_policy"], "phase2.physical_policy"),
    }

    phase3_raw = _mapping(root.get("phase3"), "phase3")
    required_phase3 = {
        "implementation_reference", "results", "appearance", "retrieval_policy",
        "physical_policy", "hybrid_policy", "hypothesis_policy",
    }
    if set(phase3_raw) != required_phase3:
        raise Step43EvaluationError(
            f"phase3 fields must be exactly {sorted(required_phase3)}")
    appearance = _mapping(phase3_raw["appearance"], "phase3.appearance")
    available = appearance.get("available")
    if not isinstance(available, bool):
        raise Step43EvaluationError("phase3.appearance.available must be boolean")
    if available:
        expected = {"available", "model_id", "model_version", "weights_sha256",
                    "preprocessing_sha256"}
        if set(appearance) != expected:
            raise Step43EvaluationError(
                f"available phase3.appearance fields must be exactly {sorted(expected)}")
        appearance_result = {
            "available": True,
            "model_id": _text(appearance["model_id"], "phase3.appearance.model_id"),
            "model_version": _text(
                appearance["model_version"], "phase3.appearance.model_version"),
            "weights_sha256": _hash(
                appearance["weights_sha256"], "phase3.appearance.weights_sha256"),
            "preprocessing_sha256": _hash(
                appearance["preprocessing_sha256"],
                "phase3.appearance.preprocessing_sha256"),
        }
    else:
        if set(appearance) != {"available", "unavailable_reason"}:
            raise Step43EvaluationError(
                "unavailable phase3.appearance requires unavailable_reason only")
        appearance_result = {
            "available": False,
            "unavailable_reason": _text(
                appearance["unavailable_reason"],
                "phase3.appearance.unavailable_reason"),
        }
    phase3 = {
        "implementation_reference": _text(
            phase3_raw["implementation_reference"],
            "phase3.implementation_reference"),
        "results": _file_reference(phase3_raw["results"], "phase3.results", base),
        "appearance": appearance_result,
        "retrieval_policy": _named_policy(
            phase3_raw["retrieval_policy"], "phase3.retrieval_policy"),
        "physical_policy": _named_policy(
            phase3_raw["physical_policy"], "phase3.physical_policy"),
        "hybrid_policy": _named_policy(
            phase3_raw["hybrid_policy"], "phase3.hybrid_policy"),
        "hypothesis_policy": _named_policy(
            phase3_raw["hypothesis_policy"], "phase3.hypothesis_policy"),
    }
    return {
        "schema_version": STEP43_SCHEMA_VERSION,
        "implementation_id": STEP43_IMPLEMENTATION_ID,
        "benchmark_id": benchmark_id,
        "created_at": created_at,
        "evaluation_at": evaluation_at,
        "manifest": {
            "path": str(manifest_path), "sha256": _sha256(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        },
        "ground_truth": ground_truth,
        "sources": sources,
        "phase2": phase2,
        "phase3": phase3,
        "notes": None if root.get("notes") is None else _text(root["notes"], "notes"),
    }


_HYBRID_STATUSES = {
    "ineligible", "physically_impossible", "trusted_plate_contradiction",
    "eligible_with_evidence", "eligible_without_evidence",
}
_PHYSICAL_STATUSES = {"ineligible", "impossible", "feasible", "unknown"}
_TRUSTED_PLATE_RELATIONS = {
    "agreement", "contradiction", "source_unavailable", "candidate_unavailable",
    "both_unavailable",
}
_APPEARANCE_STATUSES = {
    "available", "source_unavailable", "candidate_unavailable", "both_unavailable",
    "incompatible_provenance",
}


def load_phase3_results(path, *, benchmark_id: str, ground_truth_sha256: str) -> dict:
    """Load a prediction-only Phase 3 evidence capture."""
    result_path = Path(path).resolve()
    with result_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    root = _mapping(raw, "phase3 results")
    required = {"schema_version", "benchmark_id", "ground_truth_sha256",
                "ordering_semantics", "cases"}
    if set(root) != required:
        raise Step43EvaluationError(
            f"phase3 results fields must be exactly {sorted(required)}")
    if _integer(root["schema_version"], "phase3 results schema_version", 1) \
            != STEP43_PHASE3_RESULTS_SCHEMA_VERSION:
        raise Step43EvaluationError("unsupported Phase 3 results schema_version")
    if _text(root["benchmark_id"], "phase3 results benchmark_id") != benchmark_id:
        raise Step43EvaluationError("Phase 3 results use a different benchmark_id")
    if _hash(root["ground_truth_sha256"], "phase3 results ground_truth_sha256") \
            != ground_truth_sha256:
        raise Step43EvaluationError("Phase 3 results use a different ground-truth hash")
    ordering = _text(root["ordering_semantics"], "ordering_semantics")
    if ordering != "chronological_retrieval_order":
        raise Step43EvaluationError(
            "current Phase 3 results must use chronological_retrieval_order")
    cases = []
    seen_cases = set()
    for case_index, case_value in enumerate(_sequence(root["cases"], "cases")):
        name = f"cases[{case_index}]"
        case = _mapping(case_value, name)
        required_case = {"source_event_id", "candidates", "hypotheses"}
        if set(case) != required_case:
            raise Step43EvaluationError(
                f"{name} fields must be exactly {sorted(required_case)}")
        source_event_id = _text(case["source_event_id"], f"{name}.source_event_id")
        if source_event_id in seen_cases:
            raise Step43EvaluationError(f"duplicate Phase 3 case: {source_event_id}")
        seen_cases.add(source_event_id)
        candidates = []
        seen_candidates = set()
        for candidate_index, candidate_value in enumerate(
                _sequence(case["candidates"], f"{name}.candidates")):
            cname = f"{name}.candidates[{candidate_index}]"
            candidate = _mapping(candidate_value, cname)
            fields = {"event_id", "hybrid_status", "physical_gate_status",
                      "trusted_plate_relation", "appearance_status",
                      "appearance_available", "appearance_cosine_similarity"}
            if set(candidate) != fields:
                raise Step43EvaluationError(
                    f"{cname} fields must be exactly {sorted(fields)}")
            event_id = _text(candidate["event_id"], f"{cname}.event_id")
            if event_id == source_event_id:
                raise Step43EvaluationError(f"{cname} cannot reference its source event")
            if event_id in seen_candidates:
                raise Step43EvaluationError(
                    f"duplicate candidate event in {source_event_id}: {event_id}")
            seen_candidates.add(event_id)
            hybrid_status = _text(candidate["hybrid_status"], f"{cname}.hybrid_status")
            if hybrid_status not in _HYBRID_STATUSES:
                raise Step43EvaluationError(f"{cname}.hybrid_status is unsupported")
            physical = _text(
                candidate["physical_gate_status"], f"{cname}.physical_gate_status")
            if physical not in _PHYSICAL_STATUSES:
                raise Step43EvaluationError(f"{cname}.physical_gate_status is unsupported")
            plate_relation = _text(
                candidate["trusted_plate_relation"],
                f"{cname}.trusted_plate_relation")
            if plate_relation not in _TRUSTED_PLATE_RELATIONS:
                raise Step43EvaluationError(
                    f"{cname}.trusted_plate_relation is unsupported")
            appearance_status = _text(
                candidate["appearance_status"], f"{cname}.appearance_status")
            if appearance_status not in _APPEARANCE_STATUSES:
                raise Step43EvaluationError(f"{cname}.appearance_status is unsupported")
            appearance_available = candidate["appearance_available"]
            if not isinstance(appearance_available, bool):
                raise Step43EvaluationError(f"{cname}.appearance_available must be boolean")
            similarity = candidate["appearance_cosine_similarity"]
            if similarity is not None:
                similarity = _number(similarity, f"{cname}.appearance_cosine_similarity")
                if not appearance_available:
                    raise Step43EvaluationError(
                        f"{cname} cannot have cosine similarity when appearance is unavailable")
            if appearance_available != (appearance_status == "available"):
                raise Step43EvaluationError(
                    f"{cname} appearance status and availability disagree")
            if appearance_available and similarity is None:
                raise Step43EvaluationError(
                    f"{cname} available appearance requires cosine similarity")
            if physical == "ineligible" and hybrid_status != "ineligible":
                raise Step43EvaluationError(
                    f"{cname} does not preserve the ineligible physical hard gate")
            if physical == "impossible" and hybrid_status != "physically_impossible":
                raise Step43EvaluationError(
                    f"{cname} does not preserve the impossible physical hard gate")
            if (physical not in {"ineligible", "impossible"}
                    and plate_relation == "contradiction"
                    and hybrid_status != "trusted_plate_contradiction"):
                raise Step43EvaluationError(
                    f"{cname} does not preserve trusted-plate contradiction authority")
            candidates.append({
                "event_id": event_id,
                "retrieval_position": candidate_index,
                "hybrid_status": hybrid_status,
                "physical_gate_status": physical,
                "trusted_plate_relation": plate_relation,
                "appearance_status": appearance_status,
                "appearance_available": appearance_available,
                "appearance_cosine_similarity": similarity,
            })
        hypotheses = _mapping(case["hypotheses"], f"{name}.hypotheses")
        hfields = {"count", "branch_count", "coverage_event_ids", "paths",
                   "truncated"}
        if set(hypotheses) != hfields:
            raise Step43EvaluationError(
                f"{name}.hypotheses fields must be exactly {sorted(hfields)}")
        paths = []
        for path_index, path_value in enumerate(
                _sequence(hypotheses["paths"], f"{name}.hypotheses.paths")):
            event_path = tuple(
                _text(event_id, f"{name}.hypotheses.paths[{path_index}]")
                for event_id in _sequence(
                    path_value, f"{name}.hypotheses.paths[{path_index}]"))
            if len(event_path) < 2 or event_path[0] != source_event_id \
                    or len(set(event_path)) != len(event_path):
                raise Step43EvaluationError(
                    f"{name}.hypotheses.paths[{path_index}] is invalid")
            paths.append(list(event_path))
        count = _integer(hypotheses["count"], f"{name}.hypotheses.count")
        if count != len(paths):
            raise Step43EvaluationError(f"{name}.hypotheses.count does not match paths")
        coverage_ids = []
        seen_coverage = set()
        for index, event_id in enumerate(_sequence(
                hypotheses["coverage_event_ids"],
                f"{name}.hypotheses.coverage_event_ids")):
            event_id = _text(event_id, f"{name}.hypotheses.coverage_event_ids[{index}]")
            if event_id in seen_coverage:
                raise Step43EvaluationError(f"duplicate hypothesis coverage event: {event_id}")
            seen_coverage.add(event_id)
            coverage_ids.append(event_id)
        if not isinstance(hypotheses["truncated"], bool):
            raise Step43EvaluationError(f"{name}.hypotheses.truncated must be boolean")
        cases.append({
            "source_event_id": source_event_id,
            "candidates": candidates,
            "hypotheses": {
                "count": count,
                "branch_count": _integer(
                    hypotheses["branch_count"], f"{name}.hypotheses.branch_count"),
                "coverage_event_ids": sorted(coverage_ids),
                "paths": paths,
                "truncated": hypotheses["truncated"],
            },
        })
    return {
        "schema_version": STEP43_PHASE3_RESULTS_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "ground_truth_sha256": ground_truth_sha256,
        "ordering_semantics": ordering,
        "cases": sorted(cases, key=lambda item: item["source_event_id"]),
    }


def _ground_truth_readiness(ground_truth: RealWorldGroundTruth) -> dict:
    unreviewed_frames = []
    incomplete_links = []
    unreviewed_observations = []
    for source in ground_truth.sources:
        unreviewed_frames.extend(
            f"{source.source_id}:{frame.frame_number}"
            for frame in source.frames if not frame.reviewed)
        incomplete_links.extend(
            f"{source.source_id}:{instance.instance_id}"
            for instance in source.instances if not instance.observation_link_reviewed)
        if not source.observations_reviewed:
            unreviewed_observations.append(source.source_id)
    reasons = []
    if unreviewed_frames:
        reasons.append("not all manifest frames are manually reviewed")
    if incomplete_links:
        reasons.append("not all plate-instance observation links are reviewed")
    if unreviewed_observations:
        reasons.append("not all prediction observations are adjudicated")
    if len(ground_truth.cross_camera_cases) < 2:
        reasons.append("fewer than two manually verified cross-camera cases are present")
    return {
        "complete": not reasons,
        "reasons": reasons,
        "unreviewed_frame_count": len(unreviewed_frames),
        "incomplete_observation_link_count": len(incomplete_links),
        "sources_with_unreviewed_observations": sorted(unreviewed_observations),
        "cross_camera_case_count": len(ground_truth.cross_camera_cases),
    }


def _validate_sources(manifest: dict, ground_truth: RealWorldGroundTruth) -> None:
    expected = {row["source_id"]: row["camera_id"] for row in manifest["sources"]}
    actual = {source.source_id: source.camera_id for source in ground_truth.sources}
    if expected != actual:
        raise Step43EvaluationError(
            "manifest and ground truth source_id/camera_id mappings differ")
    manifest_sources = {row["source_id"]: row for row in manifest["sources"]}
    ground_truth_dir = Path(manifest["ground_truth"]["path"]).parent
    for source in ground_truth.sources:
        raw = Path(source.video_path)
        video_path = (raw if raw.is_absolute() else ground_truth_dir / raw).resolve()
        if video_path != Path(manifest_sources[source.source_id]["video"]["path"]):
            raise Step43EvaluationError(
                f"manifest and ground truth video path differ for {source.source_id}")


def _validate_case_population(
        ground_truth: RealWorldGroundTruth,
        predictions: RealWorldPredictions,
        phase3: dict,
) -> tuple[str, ...]:
    gt_cases = tuple(sorted(source for source, _ in ground_truth.cross_camera_cases))
    if len(set(gt_cases)) != len(gt_cases):
        raise Step43EvaluationError("ground truth has duplicate cross-camera source cases")
    phase2_cases = tuple(sorted(row.source_event_id for row in predictions.candidate_rankings))
    phase3_cases = tuple(sorted(row["source_event_id"] for row in phase3["cases"]))
    if gt_cases != phase2_cases or gt_cases != phase3_cases:
        raise Step43EvaluationError(
            "ground truth, Phase 2, and Phase 3 case populations must match exactly")
    known_events = {
        observation.event_id for source in predictions.sources
        for observation in source.observations
    }
    for case in phase3["cases"]:
        references = {row["event_id"] for row in case["candidates"]}
        references.update(case["hypotheses"]["coverage_event_ids"])
        references.update(
            event_id for path in case["hypotheses"]["paths"] for event_id in path)
        unknown = sorted(references - known_events)
        if unknown:
            raise Step43EvaluationError(
                f"Phase 3 case {case['source_event_id']} references unknown events: {unknown}")
    return gt_cases


def evaluate_step34_by_camera(
        ground_truth: RealWorldGroundTruth,
        predictions: RealWorldPredictions,
        **kwargs,
) -> dict:
    """Reuse the Step 34 evaluator for combined and arbitrary camera groups."""
    combined = evaluate_real_world(ground_truth, predictions, **kwargs)
    cameras = {}
    for camera_id in sorted({source.camera_id for source in ground_truth.sources}):
        source_ids = {
            source.source_id for source in ground_truth.sources
            if source.camera_id == camera_id
        }
        gt = replace(
            ground_truth,
            sources=tuple(source for source in ground_truth.sources
                          if source.source_id in source_ids),
            cross_camera_cases=(),
        )
        pred = replace(
            predictions,
            sources=tuple(source for source in predictions.sources
                          if source.source_id in source_ids),
            candidate_rankings=(),
        )
        cameras[camera_id] = evaluate_real_world(gt, pred, **kwargs)
    return {"combined": combined, "by_camera": cameras}


def paired_metric(
        *, name: str, phase2_numerator: Optional[float],
        phase2_denominator: Optional[int], phase3_numerator: Optional[float],
        phase3_denominator: Optional[int], phase2_case_ids: Sequence[str],
        phase3_case_ids: Sequence[str], unavailable_reason: Optional[str] = None,
        excluded_case_ids: Sequence[str] = (),
) -> dict:
    """Build a comparison record without hiding denominator/case differences."""
    p2_cases = tuple(sorted(set(phase2_case_ids)))
    p3_cases = tuple(sorted(set(phase3_case_ids)))
    case_sets_equal = p2_cases == p3_cases
    denominators_equal = (
        phase2_denominator is not None
        and phase3_denominator is not None
        and phase2_denominator == phase3_denominator
    )
    available = (
        unavailable_reason is None and case_sets_equal and denominators_equal
        and phase2_numerator is not None and phase3_numerator is not None
        and phase2_denominator is not None and phase2_denominator > 0
    )
    reason = unavailable_reason
    if reason is None and not case_sets_equal:
        reason = "evaluated case ID sets differ"
    if reason is None and not denominators_equal:
        reason = "metric denominators differ or are unavailable"
    if reason is None and not available:
        reason = "metric numerator or denominator is unavailable"
    p2_value = (
        phase2_numerator / phase2_denominator
        if phase2_numerator is not None and phase2_denominator else None)
    p3_value = (
        phase3_numerator / phase3_denominator
        if phase3_numerator is not None and phase3_denominator else None)
    return {
        "metric": _text(name, "metric name"),
        "available": available,
        "unavailable_reason": None if available else reason,
        "case_sets_equal": case_sets_equal,
        "denominators_equal": denominators_equal,
        "evaluated_case_ids": {
            "phase2": list(p2_cases), "phase3": list(p3_cases)},
        "excluded_case_ids": sorted(set(excluded_case_ids)),
        "phase2": {"numerator": phase2_numerator,
                   "denominator": phase2_denominator, "value": p2_value},
        "phase3": {"numerator": phase3_numerator,
                   "denominator": phase3_denominator, "value": p3_value},
        "absolute_delta": p3_value - p2_value if available else None,
    }


def _phase3_diagnostics(ground_truth: RealWorldGroundTruth, phase3: dict) -> tuple[dict, list]:
    relevant_by_source = {
        source: set(relevant) for source, relevant in ground_truth.cross_camera_cases}
    camera_by_event = {
        instance.observation_event_id: source.camera_id
        for source in ground_truth.sources for instance in source.instances
        if instance.observation_event_id
    }
    counts = {
        "candidate_count": 0,
        "physical_rejection_count": 0,
        "trusted_plate_contradiction_count": 0,
        "appearance_available_count": 0,
        "appearance_incompatible_count": 0,
        "eligible_with_evidence_count": 0,
        "eligible_without_evidence_count": 0,
        "hypothesis_count": 0,
        "branch_count": 0,
        "hypothesis_covered_relevant_events": 0,
        "relevant_event_count": sum(len(value) for value in relevant_by_source.values()),
    }
    rows = []
    for case in phase3["cases"]:
        candidates = case["candidates"]
        hypotheses = case["hypotheses"]
        counts["candidate_count"] += len(candidates)
        counts["physical_rejection_count"] += sum(
            row["physical_gate_status"] in {"ineligible", "impossible"}
            for row in candidates)
        counts["trusted_plate_contradiction_count"] += sum(
            row["trusted_plate_relation"] == "contradiction" for row in candidates)
        counts["appearance_available_count"] += sum(
            row["appearance_available"] for row in candidates)
        counts["appearance_incompatible_count"] += sum(
            "incompatible" in row["appearance_status"] for row in candidates)
        counts["eligible_with_evidence_count"] += sum(
            row["hybrid_status"] == "eligible_with_evidence" for row in candidates)
        counts["eligible_without_evidence_count"] += sum(
            row["hybrid_status"] == "eligible_without_evidence" for row in candidates)
        counts["hypothesis_count"] += hypotheses["count"]
        counts["branch_count"] += hypotheses["branch_count"]
        relevant = relevant_by_source[case["source_event_id"]]
        retrieved = {row["event_id"] for row in candidates}
        coverage = set(hypotheses["coverage_event_ids"])
        covered_relevant = sorted(relevant & coverage)
        counts["hypothesis_covered_relevant_events"] += len(covered_relevant)
        rows.append({
            "source_event_id": case["source_event_id"],
            "source_camera_id": camera_by_event[case["source_event_id"]],
            "relevant_event_ids": sorted(relevant),
            "phase3_candidate_event_ids_chronological": [
                row["event_id"] for row in candidates],
            "phase3_structural_relevant_event_ids": sorted(relevant & retrieved),
            "physical_rejected_event_ids": [
                row["event_id"] for row in candidates
                if row["physical_gate_status"] in {"ineligible", "impossible"}],
            "trusted_plate_contradiction_event_ids": [
                row["event_id"] for row in candidates
                if row["trusted_plate_relation"] == "contradiction"],
            "missing_appearance_event_ids": [
                row["event_id"] for row in candidates
                if not row["appearance_available"]],
            "incompatible_appearance_event_ids": [
                row["event_id"] for row in candidates
                if "incompatible" in row["appearance_status"]],
            "eligible_with_evidence_event_ids": [
                row["event_id"] for row in candidates
                if row["hybrid_status"] == "eligible_with_evidence"],
            "eligible_without_evidence_event_ids": [
                row["event_id"] for row in candidates
                if row["hybrid_status"] == "eligible_without_evidence"],
            "hypothesis_count": hypotheses["count"],
            "branch_count": hypotheses["branch_count"],
            "hypothesis_branch_ambiguous": hypotheses["branch_count"] > 0,
            "hypothesis_covered_relevant_event_ids": covered_relevant,
            "hypotheses_truncated": hypotheses["truncated"],
        })
    denominator = counts["candidate_count"]
    counts["appearance_availability_rate"] = (
        counts["appearance_available_count"] / denominator if denominator else None)
    counts["hypothesis_coverage"] = (
        counts["hypothesis_covered_relevant_events"]
        / counts["relevant_event_count"] if counts["relevant_event_count"] else None)
    counts["interpretation"] = (
        "diagnostics only; no value is an identity probability or same-vehicle decision")
    return counts, rows


def _phase3_by_camera(rows: list) -> dict:
    result = {}
    for camera_id in sorted({row["source_camera_id"] for row in rows}):
        selected = [row for row in rows if row["source_camera_id"] == camera_id]
        candidate_count = sum(
            len(row["phase3_candidate_event_ids_chronological"]) for row in selected)
        appearance_available = sum(
            len(row["phase3_candidate_event_ids_chronological"])
            - len(row["missing_appearance_event_ids"]) for row in selected)
        relevant_count = sum(len(row["relevant_event_ids"]) for row in selected)
        covered = sum(
            len(row["hypothesis_covered_relevant_event_ids"]) for row in selected)
        result[camera_id] = {
            "case_count": len(selected),
            "candidate_count": candidate_count,
            "physical_rejection_count": sum(
                len(row["physical_rejected_event_ids"]) for row in selected),
            "trusted_plate_contradiction_count": sum(
                len(row["trusted_plate_contradiction_event_ids"])
                for row in selected),
            "appearance_available_count": appearance_available,
            "appearance_availability_rate": (
                appearance_available / candidate_count if candidate_count else None),
            "appearance_incompatible_count": sum(
                len(row["incompatible_appearance_event_ids"]) for row in selected),
            "eligible_with_evidence_count": sum(
                len(row["eligible_with_evidence_event_ids"]) for row in selected),
            "eligible_without_evidence_count": sum(
                len(row["eligible_without_evidence_event_ids"]) for row in selected),
            "hypothesis_count": sum(row["hypothesis_count"] for row in selected),
            "branch_count": sum(row["branch_count"] for row in selected),
            "hypothesis_coverage": covered / relevant_count if relevant_count else None,
            "interpretation": "diagnostics grouped by source camera; not identity accuracy",
        }
    return result


def _failures(step34: dict, phase3_rows: list) -> dict:
    combined = step34["combined"]
    phase2_cases = {
        row["source_event_id"]: row
        for row in ((combined["cross_camera"].get("metrics") or {}).get("cases") or [])
    }
    disagreements = []
    for row in phase3_rows:
        p2 = phase2_cases.get(row["source_event_id"])
        p2_relevant = set() if p2 is None else set(p2["candidate_event_ids"]) \
            & set(row["relevant_event_ids"])
        p3_relevant = set(row["phase3_structural_relevant_event_ids"])
        if p2_relevant != p3_relevant:
            disagreements.append({
                "source_event_id": row["source_event_id"],
                "phase2_retrieved_relevant_event_ids": sorted(p2_relevant),
                "phase3_structural_retrieved_relevant_event_ids": sorted(p3_relevant),
                "note": "set disagreement only; Phase 3 chronological order is not a relevance rank",
            })
    return {
        "benchmark_id": combined["benchmark_id"],
        "detector_false_negatives": combined["detection"]["false_negative_boxes"],
        "detector_false_positives": combined["false_positive_analysis"][
            "unmatched_detector_predictions"],
        "ocr_wrong_text": [row for row in combined["ocr"]["outcomes"]
                           if not row["correct"]],
        "no_observation": [row for row in combined["end_to_end"]["outcomes"]
                           if row["status"] == "no_observation"],
        "confidence_review_or_abstained": [
            row for row in combined["end_to_end"]["outcomes"]
            if row["status"] in {"review", "abstained"}],
        "confidence_incorrect_accepted": combined["confidence_status"][
            "incorrect_accepted_observations"],
        "confidence_correct_deferred": combined["confidence_status"][
            "correct_sent_to_review_or_abstained"],
        "phase3_cases": phase3_rows,
        "phase2_phase3_disagreements": disagreements,
    }


def _safe_json(data: dict) -> str:
    try:
        return json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise Step43EvaluationError("report is not finite deterministic JSON") from exc


def step43_markdown(report: dict) -> str:
    """Return a deterministic human summary with denominators and caveats."""
    lines = [
        "# CitySight Step 43 Final Benchmark",
        "",
        f"- Benchmark: `{report['benchmark_id']}`",
        f"- Final claim ready: `{str(report['final_claim_ready']).lower()}`",
        f"- Frozen Phase 2 reference: `{report['provenance']['phase2']['baseline_reference']}`",
        f"- Phase 3 reference: `{report['provenance']['phase3']['implementation_reference']}`",
        "",
    ]
    if not report["final_claim_ready"]:
        lines.extend(["## Unavailable", ""] + [
            f"- {reason}" for reason in report["readiness"]["reasons"]])
        return "\n".join(lines) + "\n"
    combined = report["phase1_shared_metrics"]["combined"]
    lines.extend([
        "## Shared Phase 1 metrics", "",
        "These inputs and denominators are shared by both comparison paths.", "",
        f"- Detector TP/FP/FN: {combined['detection']['primary']['true_positives']} / "
        f"{combined['detection']['primary']['false_positives']} / "
        f"{combined['detection']['primary']['false_negatives']}",
        f"- OCR exact: {combined['ocr']['correct_count']} / "
        f"{combined['ocr']['evaluable_count']}",
        f"- End-to-end exact: {combined['end_to_end']['correct_count']} / "
        f"{combined['end_to_end']['evaluable_count']}",
        "", "## Cross-camera comparison", "",
    ])
    for metric in report["paired_cross_camera_metrics"]:
        if metric["available"]:
            lines.append(
                f"- {metric['metric']}: Phase 2 {metric['phase2']['numerator']} / "
                f"{metric['phase2']['denominator']}; Phase 3 "
                f"{metric['phase3']['numerator']} / {metric['phase3']['denominator']}; "
                f"delta {metric['absolute_delta']:.4f}")
        else:
            lines.append(f"- {metric['metric']}: unavailable — {metric['unavailable_reason']}")
    lines.extend([
        "", "## Phase 3 diagnostics", "",
        "Diagnostics are evidence coverage and gate counts, not identity probabilities.", "",
    ])
    diagnostics = report["phase3_diagnostics"]
    for key in (
        "candidate_count", "physical_rejection_count",
        "trusted_plate_contradiction_count", "appearance_available_count",
        "appearance_incompatible_count", "eligible_with_evidence_count",
        "eligible_without_evidence_count", "hypothesis_count", "branch_count",
    ):
        lines.append(f"- {key}: {diagnostics[key]}")
    return "\n".join(lines) + "\n"


def ablation_markdown(report: dict) -> str:
    lines = ["# CitySight Step 43 Ablation Diagnostics", ""]
    for row in report["ablations"]:
        lines.append(f"- {row['name']}: {row['status']} — {row['interpretation']}")
    return "\n".join(lines) + "\n"


def _ablation_report(benchmark_id: str, diagnostics: dict) -> dict:
    return {
        "schema_version": STEP43_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "ablations": [
            {"name": "frozen_phase2_matcher", "status": "supported",
             "interpretation": "reported with unchanged historical candidate ranking semantics"},
            {"name": "phase3_structural_retrieval", "status": "supported_diagnostic",
             "interpretation": "bounded candidate population; chronological order is not relevance rank"},
            {"name": "phase3_appearance_unavailable", "status": "supported_diagnostic",
             "interpretation": "missing appearance remains neutral evidence"},
            {"name": "phase3_compatible_appearance", "status": "supported_diagnostic",
             "interpretation": "cosine evidence is retained without a threshold or identity decision"},
            {"name": "appearance_coverage", "status": "supported_diagnostic",
             "interpretation": f"{diagnostics['appearance_available_count']} / {diagnostics['candidate_count']} candidates"},
            {"name": "physical_hard_gates", "status": "supported_diagnostic",
             "interpretation": f"{diagnostics['physical_rejection_count']} candidates rejected"},
            {"name": "trusted_plate_contradictions", "status": "supported_diagnostic",
             "interpretation": f"{diagnostics['trusted_plate_contradiction_count']} authoritative contradictions"},
            {"name": "appearance_only_winner", "status": "unsupported",
             "interpretation": "no approved appearance threshold or appearance relevance rank exists"},
            {"name": "numeric_hybrid_score", "status": "unsupported",
             "interpretation": "Step 37 intentionally defines no aggregate score"},
        ],
    }


def evaluate_step43(manifest: dict) -> tuple[dict, dict, dict]:
    """Evaluate already captured inputs without invoking production systems."""
    ground_truth = load_real_world_ground_truth(manifest["ground_truth"]["path"])
    predictions = load_real_world_predictions(manifest["phase2"]["predictions"]["path"])
    if ground_truth.benchmark_id != manifest["benchmark_id"] \
            or predictions.benchmark_id != manifest["benchmark_id"]:
        raise Step43EvaluationError("manifest, ground truth, and predictions benchmark_id differ")
    _validate_sources(manifest, ground_truth)
    phase3 = load_phase3_results(
        manifest["phase3"]["results"]["path"],
        benchmark_id=manifest["benchmark_id"],
        ground_truth_sha256=manifest["ground_truth"]["sha256"],
    )
    if not manifest["phase3"]["appearance"]["available"] and any(
            candidate["appearance_available"]
            for case in phase3["cases"] for candidate in case["candidates"]):
        raise Step43EvaluationError(
            "Phase 3 results contain appearance evidence but manifest provenance marks it unavailable")
    case_ids = _validate_case_population(ground_truth, predictions, phase3)
    readiness = _ground_truth_readiness(ground_truth)
    provenance = {"manifest": manifest["manifest"], "ground_truth": manifest["ground_truth"],
                  "sources": manifest["sources"], "phase2": manifest["phase2"],
                  "phase3": manifest["phase3"]}
    if not readiness["complete"]:
        report = {
            "schema_version": STEP43_SCHEMA_VERSION,
            "implementation_id": STEP43_IMPLEMENTATION_ID,
            "benchmark_id": manifest["benchmark_id"],
            "created_at": manifest["created_at"],
            "evaluation_at": manifest["evaluation_at"],
            "final_claim_ready": False,
            "readiness": readiness,
            "provenance": provenance,
            "phase1_shared_metrics": None,
            "phase2_cross_camera": None,
            "phase3_diagnostics": None,
            "phase3_diagnostics_by_camera": None,
            "paired_cross_camera_metrics": [],
        }
        return report, _ablation_report(manifest["benchmark_id"], {
            "appearance_available_count": 0, "candidate_count": 0,
            "physical_rejection_count": 0, "trusted_plate_contradiction_count": 0,
        }), {"benchmark_id": manifest["benchmark_id"], "unavailable": readiness["reasons"]}
    step34 = evaluate_step34_by_camera(ground_truth, predictions)
    diagnostics, phase3_rows = _phase3_diagnostics(ground_truth, phase3)
    p2_cross = step34["combined"]["cross_camera"]
    paired = []
    p2_metrics = p2_cross.get("metrics") if p2_cross["available"] else None
    if p2_metrics:
        total_relevant = p2_metrics["total_relevant_events"]
        for k, value in sorted(
                p2_metrics["recall_at_k"].items(), key=lambda item: int(item[0])):
            paired.append(paired_metric(
                name=f"recall_at_{k}",
                phase2_numerator=value * total_relevant,
                phase2_denominator=total_relevant,
                phase3_numerator=None, phase3_denominator=None,
                phase2_case_ids=case_ids, phase3_case_ids=case_ids,
                unavailable_reason=PHASE3_UNORDERED_REASON))
        for k, value in sorted(
                p2_metrics["hit_rate_at_k"].items(), key=lambda item: int(item[0])):
            paired.append(paired_metric(
                name=f"hit_rate_at_{k}",
                phase2_numerator=value * len(case_ids),
                phase2_denominator=len(case_ids),
                phase3_numerator=None, phase3_denominator=None,
                phase2_case_ids=case_ids, phase3_case_ids=case_ids,
                unavailable_reason=PHASE3_UNORDERED_REASON))
        paired.append(paired_metric(
            name="mean_reciprocal_rank",
            phase2_numerator=p2_metrics["mean_reciprocal_rank"] * len(case_ids),
            phase2_denominator=len(case_ids),
            phase3_numerator=None, phase3_denominator=None,
            phase2_case_ids=case_ids, phase3_case_ids=case_ids,
            unavailable_reason=PHASE3_UNORDERED_REASON))
    else:
        paired.append(paired_metric(
            name="phase2_cross_camera_metrics",
            phase2_numerator=None, phase2_denominator=None,
            phase3_numerator=None, phase3_denominator=None,
            phase2_case_ids=case_ids, phase3_case_ids=case_ids,
            unavailable_reason=p2_cross["reason"]))
    report = {
        "schema_version": STEP43_SCHEMA_VERSION,
        "implementation_id": STEP43_IMPLEMENTATION_ID,
        "benchmark_id": manifest["benchmark_id"],
        "created_at": manifest["created_at"],
        "evaluation_at": manifest["evaluation_at"],
        "final_claim_ready": True,
        "readiness": readiness,
        "provenance": provenance,
        "phase1_shared_metrics": step34,
        "phase2_cross_camera": p2_cross,
        "phase3_diagnostics": diagnostics,
        "phase3_diagnostics_by_camera": _phase3_by_camera(phase3_rows),
        "paired_cross_camera_metrics": paired,
        "comparison_rules": {
            "same_ground_truth_hash_required": True,
            "same_case_population_required": True,
            "absolute_delta_requires_equal_denominators_and_case_ids": True,
            "phase3_ordering_semantics": phase3["ordering_semantics"],
            "no_statistical_significance_claimed": True,
        },
    }
    return report, _ablation_report(manifest["benchmark_id"], diagnostics), \
        _failures(step34, phase3_rows)


def run_step43_benchmark(manifest_path, output_dir) -> dict:
    """Validate inputs and atomically write deterministic Step 43 artifacts."""
    manifest = load_step43_manifest(manifest_path)
    report, ablations, failures = evaluate_step43(manifest)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "step43_final_benchmark.json": _safe_json(report),
        "step43_final_benchmark.md": step43_markdown(report),
        "step43_ablation.json": _safe_json(ablations),
        "step43_ablation.md": ablation_markdown(ablations),
        "step43_failures.json": _safe_json(failures),
    }
    for name, content in artifacts.items():
        path = destination / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)
    return deepcopy(report)


__all__ = [
    "FROZEN_PHASE2_REFERENCE",
    "PHASE3_UNORDERED_REASON",
    "STEP43_IMPLEMENTATION_ID",
    "STEP43_PHASE3_RESULTS_SCHEMA_VERSION",
    "STEP43_SCHEMA_VERSION",
    "Step43EvaluationError",
    "ablation_markdown",
    "evaluate_step34_by_camera",
    "evaluate_step43",
    "load_phase3_results",
    "load_step43_manifest",
    "paired_metric",
    "run_step43_benchmark",
    "step43_markdown",
]
