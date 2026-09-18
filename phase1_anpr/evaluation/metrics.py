"""Stage-correct metrics used by the Phase 1 ANPR evaluation runners.

The functions in this module are deliberately model-independent.  Ground truth
is supplied by the caller and is never inferred from CitySight predictions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence


BBox = tuple[float, float, float, float]
Prediction = tuple[BBox, float]


@dataclass(frozen=True)
class DetectionMetrics:
    iou_threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DetectionMatch:
    image_id: str
    prediction_index: int
    ground_truth_index: int
    iou: float


@dataclass(frozen=True)
class DetectionEvaluation:
    metrics: DetectionMetrics
    matches: tuple[DetectionMatch, ...]
    unmatched_predictions: Mapping[str, tuple[int, ...]]
    unmatched_ground_truth: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class OCRMetrics:
    samples: int
    exact_matches: int
    exact_accuracy: float
    character_accuracy: float
    character_errors: int
    ground_truth_characters: int

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_box(box: Sequence[float]) -> BBox:
    if len(box) != 4:
        raise ValueError(f"bounding box must contain four values, got {box!r}")
    x1, y1, x2, y2 = (float(value) for value in box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bounding box has non-positive area: {box!r}")
    return x1, y1, x2, y2


def intersection_over_union(first: Sequence[float], second: Sequence[float]) -> float:
    """Return axis-aligned IoU for two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = _validate_box(first)
    bx1, by1, bx2, by2 = _validate_box(second)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection == 0.0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union else 0.0


def evaluate_detections(
    ground_truth: Mapping[str, Sequence[Sequence[float]]],
    predictions: Mapping[str, Sequence[Prediction]],
    iou_threshold: float = 0.5,
) -> DetectionEvaluation:
    """Greedily match predictions by confidence and calculate P/R/F1.

    Each ground-truth box can match at most one prediction.  Predictions are
    processed highest-confidence first, which is the conventional detector
    operating-point evaluation and makes duplicate detections false positives.
    """
    if not 0.0 < float(iou_threshold) <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")

    image_ids = sorted(set(ground_truth) | set(predictions))
    matches: list[DetectionMatch] = []
    unmatched_predictions: dict[str, tuple[int, ...]] = {}
    unmatched_ground_truth: dict[str, tuple[int, ...]] = {}

    for image_id in image_ids:
        truth = [_validate_box(box) for box in ground_truth.get(image_id, ())]
        predicted = [(_validate_box(box), float(conf)) for box, conf in predictions.get(image_id, ())]
        order = sorted(range(len(predicted)), key=lambda index: (-predicted[index][1], index))
        available_truth = set(range(len(truth)))
        missed_predictions: list[int] = []

        for prediction_index in order:
            box, _ = predicted[prediction_index]
            candidates = [
                (intersection_over_union(box, truth[truth_index]), truth_index)
                for truth_index in available_truth
            ]
            best_iou, best_truth = max(candidates, default=(0.0, -1))
            if best_iou >= iou_threshold:
                available_truth.remove(best_truth)
                matches.append(DetectionMatch(
                    image_id=image_id,
                    prediction_index=prediction_index,
                    ground_truth_index=best_truth,
                    iou=best_iou,
                ))
            else:
                missed_predictions.append(prediction_index)

        if missed_predictions:
            unmatched_predictions[image_id] = tuple(sorted(missed_predictions))
        if available_truth:
            unmatched_ground_truth[image_id] = tuple(sorted(available_truth))

    true_positives = len(matches)
    false_positives = sum(len(indices) for indices in unmatched_predictions.values())
    false_negatives = sum(len(indices) for indices in unmatched_ground_truth.values())
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    return DetectionEvaluation(
        metrics=DetectionMetrics(
            iou_threshold=float(iou_threshold),
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1=f1,
        ),
        matches=tuple(matches),
        unmatched_predictions=unmatched_predictions,
        unmatched_ground_truth=unmatched_ground_truth,
    )


def levenshtein_distance(expected: str, predicted: str) -> int:
    """Return character edit distance using O(min(n, m)) memory."""
    expected, predicted = str(expected), str(predicted)
    if len(expected) < len(predicted):
        expected, predicted = predicted, expected
    previous = list(range(len(predicted) + 1))
    for row, expected_char in enumerate(expected, start=1):
        current = [row]
        for column, predicted_char in enumerate(predicted, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (expected_char != predicted_char),
            ))
        previous = current
    return previous[-1]


def character_accuracy(expected: str, predicted: str) -> float:
    """Return one minus normalized edit distance, clamped to ``[0, 1]``."""
    expected = str(expected)
    if not expected:
        return 1.0 if not str(predicted) else 0.0
    return max(0.0, 1.0 - levenshtein_distance(expected, str(predicted)) / len(expected))


def evaluate_ocr(pairs: Iterable[tuple[str, str]]) -> OCRMetrics:
    """Calculate exact and micro character accuracy for labeled OCR pairs."""
    rows = [(str(expected), str(predicted)) for expected, predicted in pairs]
    exact = sum(expected == predicted for expected, predicted in rows)
    errors = sum(levenshtein_distance(expected, predicted) for expected, predicted in rows)
    characters = sum(len(expected) for expected, _ in rows)
    return OCRMetrics(
        samples=len(rows),
        exact_matches=exact,
        exact_accuracy=exact / len(rows) if rows else 0.0,
        character_accuracy=max(0.0, 1.0 - errors / characters) if characters else 0.0,
        character_errors=errors,
        ground_truth_characters=characters,
    )


def aligned_substitutions(expected: str, predicted: str) -> tuple[tuple[str, str], ...]:
    """Return substitution pairs from one deterministic optimal edit alignment.

    Insertions and deletions are excluded because they are not character
    confusions.  Ties prefer a diagonal alignment, then deletion, then
    insertion, which keeps the output deterministic.
    """
    expected, predicted = str(expected), str(predicted)
    rows, columns = len(expected) + 1, len(predicted) + 1
    table = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        table[row][0] = row
    for column in range(columns):
        table[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            table[row][column] = min(
                table[row - 1][column] + 1,
                table[row][column - 1] + 1,
                table[row - 1][column - 1]
                + (expected[row - 1] != predicted[column - 1]),
            )

    substitutions: list[tuple[str, str]] = []
    row, column = len(expected), len(predicted)
    while row or column:
        if row and column:
            cost = expected[row - 1] != predicted[column - 1]
            if table[row][column] == table[row - 1][column - 1] + cost:
                if cost:
                    substitutions.append((expected[row - 1], predicted[column - 1]))
                row -= 1
                column -= 1
                continue
        if row and table[row][column] == table[row - 1][column] + 1:
            row -= 1
        else:
            column -= 1
    substitutions.reverse()
    return tuple(substitutions)
