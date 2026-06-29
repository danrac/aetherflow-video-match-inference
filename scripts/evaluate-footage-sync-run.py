#!/usr/bin/env python3
"""Evaluate source-window inference outputs against a Footage Sync fixture run."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from aetherflow_video_match_inference.cli import load_source_window_match_request
from aetherflow_video_match_inference.engine import candidate_source_segment_id, match_source_windows
from aetherflow_video_match_inference.sequence_assignment import assign_ranked_reference_sequence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--fixture-comparison")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    comparison_path = Path(args.fixture_comparison) if args.fixture_comparison else run_dir / "footage_sync_fixture_comparison.json"
    expected_rows = load_expected_rows(comparison_path)
    started = time.time()
    rows = []
    selected_pairs = []
    skipped_references = []

    for request_path in sorted((run_dir / "_source_window_match").glob("reference_*/source_window_request.json")):
        request = load_source_window_match_request(request_path)
        request_doc = json.loads(request_path.read_text(encoding="utf-8"))
        reference_doc = json.loads(Path(request_doc["reference_feature_manifest_path"]).read_text(encoding="utf-8"))
        reference_window = reference_doc.get("source_window") if isinstance(reference_doc.get("source_window"), dict) else {}
        reference_start = int(reference_window.get("source_in", 0) or 0)
        reference_end = int(reference_window.get("source_out", reference_start + 1) or reference_start + 1)
        microcut = reference_end - reference_start <= 2
        result = match_source_windows(request)
        ranked = result.get("diagnostics", {}).get("rankedCandidates", [])
        expected = expected_for_reference(expected_rows, reference_start)
        expected_segment = expected_segment_for_request(request_doc, expected["sourceStartFrame"]) if expected else None
        selected = ranked[0] if ranked else None
        top_segments = [row["candidateSourceSegmentId"] for row in ranked[:3]]
        selected_segment = selected["candidateSourceSegmentId"] if selected else None
        selected_source_start = selected["candidateSourceStartFrame"] if selected else None
        source_error = abs(selected_source_start - expected["sourceStartFrame"]) if selected and expected else None
        reference_error = abs(reference_start - expected["referenceStartFrame"]) if expected else None
        row = {
            "referenceSegmentId": str(reference_window.get("source_clip_id") or request_path.parent.name),
            "referenceStartFrame": reference_start,
            "referenceEndFrame": reference_end,
            "microcut": microcut,
            "expectedSourceStartFrame": expected["sourceStartFrame"] if expected else None,
            "expectedReferenceStartFrame": expected["referenceStartFrame"] if expected else None,
            "expectedCandidateSourceSegmentId": expected_segment,
            "selectedCandidateSourceSegmentId": selected_segment,
            "selectedSourceStartFrame": selected_source_start,
            "top1ShotIdentity": bool(expected_segment and selected_segment == expected_segment),
            "top3ShotIdentity": bool(expected_segment and expected_segment in top_segments),
            "sourceStartFrameAbsError": source_error,
            "referenceStartFrameAbsError": reference_error,
            "rankedCandidates": ranked,
        }
        rows.append(row)
        if selected is None:
            skipped_references.append(row["referenceSegmentId"])
        else:
            selected_pairs.append(
                {
                    "referenceSegmentId": row["referenceSegmentId"],
                    "candidateSourceSegmentId": selected_segment,
                    "candidateSourceStartFrame": selected_source_start,
                    "score": selected["finalScore"],
                }
            )

    normal_rows = [row for row in rows if not row["microcut"] and row["expectedCandidateSourceSegmentId"]]
    assignment = assign_ranked_reference_sequence(rows)
    report = {
        "schemaVersion": "1.0.0",
        "kind": "aetherflowVideoMatchFixtureEvaluation",
        "runDir": str(run_dir),
        "fixtureComparisonPath": str(comparison_path),
        "metrics": metrics(normal_rows, rows, time.time() - started, assignment),
        "microcutHandling": [
            {
                "referenceSegmentId": row["referenceSegmentId"],
                "referenceStartFrame": row["referenceStartFrame"],
                "selectedCandidateSourceSegmentId": row["selectedCandidateSourceSegmentId"],
                "policy": "reduced_weight_diagnostics_only",
            }
            for row in rows
            if row["microcut"]
        ],
        "globalAssignmentDiagnostics": {
            "assignmentMethod": assignment["assignmentMethod"],
            "skippedReferenceSegments": skipped_references,
            "skippedSourceSegments": assignment["skippedSourceSegments"],
            "selectedPairs": assignment["selectedPairs"],
            "localTop1Pairs": selected_pairs,
            "globalScore": assignment["globalScore"],
            "confidenceCalibrationNotes": "Overlap-aware beam assignment is intended for batch stringout matching when per-reference top candidates are available.",
        },
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


def load_expected_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = []
    for item in data.get("comparisons", []):
        exp = item.get("expected") if isinstance(item, dict) else None
        if not isinstance(exp, dict):
            continue
        expected.append({
            "sourceStartFrame": int(exp["sourceStartFrame"]),
            "referenceStartFrame": int(exp["referenceStartFrame"]),
        })
    return sorted(expected, key=lambda row: row["referenceStartFrame"])


def expected_for_reference(expected_rows: list[dict], reference_start: int) -> dict | None:
    if not expected_rows:
        return None
    exact = [row for row in expected_rows if row["referenceStartFrame"] == reference_start]
    if exact:
        return exact[0]
    nearby = [row for row in expected_rows if abs(row["referenceStartFrame"] - reference_start) <= 2]
    if nearby:
        return sorted(nearby, key=lambda row: (abs(row["referenceStartFrame"] - reference_start), row["referenceStartFrame"]))[0]
    return None


def expected_segment_for_request(request_doc: dict, expected_source_start: int) -> str | None:
    for candidate in request_doc.get("candidates", []):
        source_in = int(candidate.get("source_in", 0))
        source_out = int(candidate.get("source_out", source_in + 1))
        if source_in <= expected_source_start < source_out:
            return candidate_source_segment_id(str(candidate["candidate_id"]))
    return None


def metrics(normal_rows: list[dict], all_rows: list[dict], runtime_duration: float, assignment: dict | None = None) -> dict:
    source_errors = [row["sourceStartFrameAbsError"] for row in normal_rows if row["sourceStartFrameAbsError"] is not None]
    reference_errors = [row["referenceStartFrameAbsError"] for row in normal_rows if row["referenceStartFrameAbsError"] is not None]
    assignment_rows = sequence_assignment_rows(normal_rows, assignment or {})
    assignment_source_errors = [row["sourceStartFrameAbsError"] for row in assignment_rows if row["sourceStartFrameAbsError"] is not None]
    return {
        "referenceSegmentCount": len(all_rows),
        "normalEvaluatedSegmentCount": len(normal_rows),
        "microcutCount": len([row for row in all_rows if row["microcut"]]),
        "shotIdentityAccuracy": average([1.0 if row["top1ShotIdentity"] else 0.0 for row in normal_rows]),
        "top1CandidateAccuracy": average([1.0 if row["top1ShotIdentity"] else 0.0 for row in normal_rows]),
        "top3CandidateAccuracy": average([1.0 if row["top3ShotIdentity"] else 0.0 for row in normal_rows]),
        "sourceStartFrameMAE": average(source_errors),
        "referenceStartFrameMAE": average(reference_errors),
        "sourceStartFrameMaxError": max(source_errors) if source_errors else None,
        "referenceStartFrameMaxError": max(reference_errors) if reference_errors else None,
        "sequenceShotIdentityAccuracy": average([1.0 if row["top1ShotIdentity"] else 0.0 for row in assignment_rows]),
        "sequenceSourceStartFrameMAE": average(assignment_source_errors),
        "sequenceSourceStartFrameMaxError": max(assignment_source_errors) if assignment_source_errors else None,
        "confidenceCalibration": "diagnostic_scores_included",
        "runtimeDurationSec": round(runtime_duration, 3),
    }


def sequence_assignment_rows(normal_rows: list[dict], assignment: dict) -> list[dict]:
    pair_by_reference = {pair["referenceSegmentId"]: pair for pair in assignment.get("selectedPairs", [])}
    rows = []
    for row in normal_rows:
        pair = pair_by_reference.get(row["referenceSegmentId"])
        if pair is None:
            continue
        selected_segment = pair["candidateSourceSegmentId"]
        selected_source_start = int(pair["candidateSourceStartFrame"])
        expected_segment = row["expectedCandidateSourceSegmentId"]
        expected_source_start = row["expectedSourceStartFrame"]
        rows.append(
            {
                **row,
                "selectedCandidateSourceSegmentId": selected_segment,
                "selectedSourceStartFrame": selected_source_start,
                "top1ShotIdentity": bool(expected_segment and selected_segment == expected_segment),
                "sourceStartFrameAbsError": abs(selected_source_start - expected_source_start) if expected_source_start is not None else None,
            }
        )
    return rows


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


if __name__ == "__main__":
    raise SystemExit(main())
