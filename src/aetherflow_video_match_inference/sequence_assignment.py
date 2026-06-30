"""Sequence-level assignment for source-window candidates."""

from __future__ import annotations

from itertools import count
from typing import Any


def assign_ranked_reference_sequence(rows: list[dict[str, Any]], *, top_n: int = 6, beam_width: int = 256) -> dict[str, Any]:
    assignment_rows = [row for row in rows if not row.get("microcut")]
    if not assignment_rows:
        return {"selectedPairs": [], "skippedSourceSegments": [], "globalScore": 0.0}

    serial = count()
    beams = [{"score": 0.0, "path": [], "windows": []}]
    for row in assignment_rows:
        candidates = row.get("rankedCandidates", [])[:top_n]
        if not candidates:
            continue
        next_beams = []
        for beam in beams:
            for candidate in candidates:
                pair = candidate_pair(row, candidate)
                window = candidate_window(pair)
                transition = transition_score(beam["windows"], window)
                next_beams.append(
                    {
                        "score": float(beam["score"]) + float(pair["score"]) + transition,
                        "path": [*beam["path"], pair],
                        "windows": [*beam["windows"], window],
                        "serial": next(serial),
                    }
                )
        beams = sorted(next_beams, key=lambda beam: (-float(beam["score"]), int(beam["serial"])))[:beam_width]
    best = beams[0] if beams else {"score": 0.0, "path": [], "windows": []}
    selected_pairs = best["path"]
    used_segments = {pair["candidateSourceSegmentId"] for pair in selected_pairs}
    seen_segments = {
        str(candidate.get("candidateSourceSegmentId"))
        for row in assignment_rows
        for candidate in row.get("rankedCandidates", [])[:top_n]
        if candidate.get("candidateSourceSegmentId")
    }
    return {
        "assignmentMethod": "beam_search_overlap_aware",
        "selectedPairs": selected_pairs,
        "skippedSourceSegments": sorted(seen_segments - used_segments),
        "globalScore": round(float(best["score"]) / max(1, len(selected_pairs)), 6),
    }


def candidate_pair(row: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "referenceSegmentId": row["referenceSegmentId"],
        "candidateSourceSegmentId": str(candidate["candidateSourceSegmentId"]),
        "candidateSourceStartFrame": int(candidate["candidateSourceStartFrame"]),
        "candidateSourceEndFrame": int(candidate["candidateSourceEndFrame"]),
        "score": float(candidate["finalScore"]),
    }


def candidate_window(pair: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(pair["candidateSourceSegmentId"]),
        int(pair["candidateSourceStartFrame"]),
        int(pair["candidateSourceEndFrame"]),
    )


def transition_score(previous_windows: list[tuple[str, int, int]], current_window: tuple[str, int, int]) -> float:
    current_segment, current_start, current_end = current_window
    score = 0.0
    if previous_windows:
        previous_segment, previous_start, _previous_end = previous_windows[-1]
        if current_segment == previous_segment:
            score -= 0.015
    for previous_segment, previous_start, previous_end in previous_windows:
        overlap = max(0, min(previous_end, current_end) - max(previous_start, current_start))
        if overlap <= 0:
            continue
        score -= min(0.6, overlap / 100.0)
        if previous_segment == current_segment:
            score -= 0.2
    return score
