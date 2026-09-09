"""YAML loading for explicit Step 30 candidate-evaluation labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import yaml

from phase2_city.candidate_evaluation import CandidateGroundTruthCase


class CandidateGroundTruthLoadError(ValueError):
    """Raised when a candidate ground-truth file has an invalid schema."""


@dataclass(frozen=True)
class CandidateGroundTruthDataset:
    """One scenario's deterministic, explicitly supplied evaluation cases."""

    scenario_id: str
    cases: Tuple[CandidateGroundTruthCase, ...]

    def __post_init__(self):
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise CandidateGroundTruthLoadError(
                "ground-truth scenario_id must be a non-empty string")
        if not isinstance(self.cases, tuple) or not self.cases:
            raise CandidateGroundTruthLoadError(
                "ground-truth cases must be a non-empty tuple")
        if not all(isinstance(case, CandidateGroundTruthCase)
                   for case in self.cases):
            raise CandidateGroundTruthLoadError(
                "ground-truth cases must contain CandidateGroundTruthCase "
                "objects")
        object.__setattr__(
            self,
            "cases",
            tuple(sorted(self.cases, key=lambda case: case.source_event_id)),
        )

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "cases": [case.to_dict() for case in self.cases],
        }


def load_candidate_ground_truth(path) -> CandidateGroundTruthDataset:
    """Load a strict minimal YAML dataset and construct Step 30 cases.

    File/schema problems use ``CandidateGroundTruthLoadError``. Detailed case
    validation remains owned by ``CandidateGroundTruthCase`` and its
    ``CandidateEvaluationError``.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Candidate ground-truth file not found: {dataset_path}")
    try:
        with dataset_path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise CandidateGroundTruthLoadError(
            f"invalid candidate ground-truth YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise CandidateGroundTruthLoadError(
            "candidate ground truth must be a mapping with scenario_id and "
            "cases")
    allowed_root_keys = {"scenario_id", "cases"}
    unknown_root_keys = set(data) - allowed_root_keys
    if unknown_root_keys:
        names = ", ".join(sorted(str(name) for name in unknown_root_keys))
        raise CandidateGroundTruthLoadError(
            f"unknown candidate ground-truth field(s): {names}")

    scenario_id = data.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        raise CandidateGroundTruthLoadError(
            "ground-truth scenario_id must be a non-empty string")
    if "cases" not in data:
        raise CandidateGroundTruthLoadError(
            "candidate ground truth must define cases")
    raw_cases = data["cases"]
    if not isinstance(raw_cases, list):
        raise CandidateGroundTruthLoadError(
            "candidate ground-truth cases must be a list")
    if not raw_cases:
        raise CandidateGroundTruthLoadError(
            "candidate ground-truth cases must not be empty")

    cases = []
    allowed_case_keys = {"source_event_id", "relevant_event_ids"}
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise CandidateGroundTruthLoadError(
                f"cases[{index}] must be a mapping")
        unknown_case_keys = set(raw_case) - allowed_case_keys
        if unknown_case_keys:
            names = ", ".join(
                sorted(str(name) for name in unknown_case_keys))
            raise CandidateGroundTruthLoadError(
                f"cases[{index}] has unknown field(s): {names}")
        if "source_event_id" not in raw_case:
            raise CandidateGroundTruthLoadError(
                f"cases[{index}] must define source_event_id")
        if "relevant_event_ids" not in raw_case:
            raise CandidateGroundTruthLoadError(
                f"cases[{index}] must define relevant_event_ids")
        relevant_event_ids = raw_case["relevant_event_ids"]
        if not isinstance(relevant_event_ids, list):
            raise CandidateGroundTruthLoadError(
                f"cases[{index}].relevant_event_ids must be a list")
        cases.append(CandidateGroundTruthCase(
            source_event_id=raw_case["source_event_id"],
            relevant_event_ids=tuple(relevant_event_ids),
        ))

    return CandidateGroundTruthDataset(
        scenario_id=scenario_id,
        cases=tuple(cases),
    )


__all__ = [
    "CandidateGroundTruthLoadError",
    "CandidateGroundTruthDataset",
    "load_candidate_ground_truth",
]
