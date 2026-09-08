"""Human-readable read-only reporting for Step 28 candidate hypotheses.

Formatting consumes an already collected ``CandidateCollectionResult``.  It
does not search, score, reorder, persist, or infer vehicle identity.
"""

from phase2_city.candidate_collection import CandidateCollectionResult


def _display(value) -> str:
    return "not available" if value is None else str(value)


def _display_number(value) -> str:
    return "not available" if value is None else f"{value:.6f}"


def _observation_lines(observation) -> list[str]:
    fingerprint = observation.fingerprint
    timestamp = (observation.timestamp.isoformat()
                 if observation.timestamp is not None else None)
    return [
        f"Event ID: {observation.event_id}",
        f"Camera: {observation.camera_id}",
        f"Timestamp: {_display(timestamp)}",
        f"Plate: {_display(fingerprint.normalized_plate)}",
        f"Vehicle colour: {_display(fingerprint.vehicle_colour)}",
        f"Vehicle class: {_display(fingerprint.vehicle_class)}",
    ]


def _evaluation_lines(index, result, *, diagnostic: bool) -> list[str]:
    heading = (f"Evaluation {index}" if diagnostic
               else f"Candidate hypothesis {index} (possible candidate match)")
    comparison = result.fingerprint_comparison
    lines = [
        heading,
        "-" * len(heading),
        f"Decision: {result.decision}",
        *_observation_lines(result.candidate),
        f"Elapsed seconds: {_display_number(result.elapsed_seconds)}",
        "Overall similarity: "
        f"{_display_number(comparison.overall_similarity)}",
        f"Evidence coverage: {_display_number(comparison.evidence_coverage)}",
        f"Topology: {result.topology_status}",
        f"Travel: {result.travel_status}",
        "Minimum travel seconds: "
        f"{_display_number(result.minimum_travel_seconds)}",
        "Reasons:",
    ]
    lines.extend(f"- {reason}" for reason in result.reasons)
    return lines


def format_candidate_report(
        result: CandidateCollectionResult,
        *,
        include_diagnostics: bool = False,
) -> str:
    """Format Step 28 output without changing its ordering or semantics."""
    if not isinstance(result, CandidateCollectionResult):
        raise TypeError("result must be a CandidateCollectionResult")

    lines = [
        "Cross-Camera Candidate Report",
        "=============================",
        f"Scenario: {result.scenario_id}",
        "",
        "Source observation",
        "------------------",
        *_observation_lines(result.source),
        "",
        "Collection",
        "----------",
        f"Total observations considered: {result.total_observations_considered}",
        f"Structurally eligible: {result.structurally_eligible_count}",
        f"Evaluated: {len(result.ranked_results)}",
        f"Candidate hypotheses: {len(result.candidates)}",
        f"Truncated: {'Yes' if result.truncated else 'No'}",
    ]

    if result.truncated:
        evaluated_count = len(result.ranked_results)
        evaluated_verb = "was" if evaluated_count == 1 else "were"
        lines.extend([
            "Candidate evaluation was truncated by max_candidates: "
            f"{result.structurally_eligible_count} observations were "
            f"structurally eligible; {evaluated_count} {evaluated_verb} "
            "evaluated.",
        ])

    lines.append("")
    if include_diagnostics:
        heading = "Ranked evaluations (diagnostics)"
        evaluations = result.ranked_results
    else:
        heading = "Plausible candidate hypotheses"
        evaluations = result.candidates
    lines.extend([heading, "-" * len(heading)])

    if not evaluations:
        if result.structurally_eligible_count == 0:
            lines.append(
                "No structurally eligible later cross-camera observations.")
        else:
            lines.append(
                "No plausible strong/weak candidate hypotheses were found.")
        return "\n".join(lines)

    for index, evaluation in enumerate(evaluations, start=1):
        if index > 1:
            lines.append("")
        lines.extend(_evaluation_lines(
            index, evaluation, diagnostic=include_diagnostics))
    return "\n".join(lines)


__all__ = ["format_candidate_report"]
