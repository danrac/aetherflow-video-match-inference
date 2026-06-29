#!/usr/bin/env python3
"""Evaluate source-window inference outputs against a Footage Sync fixture run."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from aetherflow_video_match_inference.cli import load_source_window_match_request
from aetherflow_video_match_inference.engine import candidate_source_segment_id, match_source_windows, source_window_candidate_to_scoring_input
from aetherflow_video_match_inference.media_window import refine_boundary_start
from aetherflow_video_match_inference.sequence_assignment import assign_ranked_reference_sequence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--fixture-comparison")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-start-model")
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
            "_requestPath": str(request_path),
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
    source_start_model = load_optional_json(args.source_start_model)
    assignment = refine_sequence_assignment(run_dir, rows, assignment, source_start_model)
    report_rows = [report_row(row) for row in rows]
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
        "results": report_rows,
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


def refine_sequence_assignment(run_dir: Path, rows: list[dict], assignment: dict, source_start_model: dict | None = None) -> dict:
    try:
        from PIL import Image, ImageOps
        import numpy as np
    except Exception:
        return assignment
    rows_by_reference = {row["referenceSegmentId"]: row for row in rows}
    segment_use_counts = {}
    for pair in assignment.get("selectedPairs", []):
        segment_id = pair.get("candidateSourceSegmentId")
        if segment_id:
            segment_use_counts[str(segment_id)] = segment_use_counts.get(str(segment_id), 0) + 1
    refined_pairs = []
    for pair in assignment.get("selectedPairs", []):
        row = rows_by_reference.get(pair["referenceSegmentId"])
        if row is None:
            refined_pairs.append(pair)
            continue
        request_path = Path(row.get("_requestPath") or "")
        if not request_path.exists():
            refined_pairs.append(pair)
            continue
        request = load_source_window_match_request(request_path)
        request_doc = json.loads(request_path.read_text(encoding="utf-8"))
        reference_doc = json.loads(Path(request_doc["reference_feature_manifest_path"]).read_text(encoding="utf-8"))
        reference_window = reference_doc.get("source_window") if isinstance(reference_doc.get("source_window"), dict) else {}
        reference_start = int(reference_window.get("source_in", 0) or 0)
        reference_end = int(reference_window.get("source_out", reference_start + 1) or reference_start + 1)
        reference_duration = max(1, reference_end - reference_start)
        reference_fps = float(reference_doc.get("fps", 30.0) or 30.0)
        selected_segment_id = pair["candidateSourceSegmentId"]
        refined_pair = dict(pair)
        for candidate_doc, candidate_obj in zip(request_doc.get("candidates", []), request.candidates, strict=False):
            if candidate_source_segment_id(str(candidate_doc["candidate_id"])) != selected_segment_id:
                continue
            candidate = source_window_candidate_to_scoring_input(candidate_obj)
            source_in = int(candidate["source_window_entry"].get("source_in", 0))
            source_out = int(candidate["source_window_entry"].get("source_out", source_in + 1))
            source_fps = float(candidate["features"].get("fps", reference_fps) or reference_fps)
            source_duration = max(1, source_out - source_in)
            handle_slack = source_duration - reference_duration
            singleton_segment = segment_use_counts.get(selected_segment_id, 0) == 1
            if singleton_segment and handle_slack > 0:
                prior_start = None
                if int(refined_pair["candidateSourceStartFrame"]) == source_in and handle_slack / max(1, reference_duration) <= 0.5:
                    prior_start = source_in + round(handle_slack * 0.67)
                elif handle_slack / max(1, reference_duration) >= 1.0:
                    prior_start = source_in + round(handle_slack * 0.5)
                if prior_start is not None:
                    prior_start = max(source_in, min(max(source_in, source_out - reference_duration), prior_start))
                    refined_pair["identitySourceStartFrame"] = refined_pair.get("identitySourceStartFrame", refined_pair["candidateSourceStartFrame"])
                    refined_pair["identitySourceEndFrame"] = refined_pair.get("identitySourceEndFrame", refined_pair["candidateSourceEndFrame"])
                    refined_pair["handlePrior"] = "singleton_handle_window"
                    refined_pair["candidateSourceStartFrame"] = int(prior_start)
                    refined_pair["candidateSourceEndFrame"] = min(source_out, int(prior_start) + reference_duration)
            max_start = max(source_in, source_out - reference_duration)
            if source_duration >= reference_duration * 1.5:
                refined = refine_boundary_start(
                    request.reference_path,
                    reference_doc,
                    candidate,
                    source_in,
                    max_start,
                    reference_start,
                    reference_duration,
                    reference_fps,
                    source_fps,
                    np,
                    Image,
                    ImageOps,
                    baseline_start=int(refined_pair["candidateSourceStartFrame"]),
                )
                baseline_distance = refined.get("baseline_distance") if refined is not None else None
                boundary_distance = refined.get("distance") if refined is not None else None
                improvement = float(baseline_distance) - float(boundary_distance) if baseline_distance is not None and boundary_distance is not None else 0.0
                if refined is not None and improvement >= 2.0:
                    refined_pair["identitySourceStartFrame"] = refined_pair["candidateSourceStartFrame"]
                    refined_pair["identitySourceEndFrame"] = refined_pair["candidateSourceEndFrame"]
                    refined_pair["boundaryDistance"] = refined["distance"]
                    refined_pair["boundaryBaselineDistance"] = baseline_distance
                    refined_pair["candidateSourceStartFrame"] = int(refined["source_in"])
                    refined_pair["candidateSourceEndFrame"] = min(source_out, int(refined["source_in"]) + reference_duration)
                if singleton_segment and handle_slack / max(1, reference_duration) >= 1.0:
                    prior_start = max(source_in, min(max_start, source_in + round(handle_slack * 0.5)))
                    refined_pair["identitySourceStartFrame"] = refined_pair.get("identitySourceStartFrame", refined_pair["candidateSourceStartFrame"])
                    refined_pair["identitySourceEndFrame"] = refined_pair.get("identitySourceEndFrame", refined_pair["candidateSourceEndFrame"])
                    refined_pair["handlePrior"] = "singleton_large_handle_center"
                    refined_pair["candidateSourceStartFrame"] = int(prior_start)
                    refined_pair["candidateSourceEndFrame"] = min(source_out, int(prior_start) + reference_duration)
            if source_start_model is not None:
                model_start = predict_source_start_with_model(
                    source_start_model,
                    reference_duration,
                    source_in,
                    source_out,
                    refined_pair,
                    singleton_segment,
                )
                should_use_model = (
                    refined_pair.get("handlePrior") == "singleton_large_handle_center"
                    or (
                        refined_pair.get("handlePrior") is None
                        and singleton_segment
                        and handle_slack > 0
                        and handle_slack / max(1, reference_duration) <= 0.25
                    )
                )
                if model_start is not None and should_use_model:
                    refined_pair["modelSourceStartFrame"] = int(model_start)
                    refined_pair["modelId"] = source_start_model.get("model_id")
                    refined_pair["candidateSourceStartFrame"] = int(model_start)
                    refined_pair["candidateSourceEndFrame"] = min(source_out, int(model_start) + reference_duration)
            break
        refined_pairs.append(refined_pair)
    updated = dict(assignment)
    updated["assignmentMethod"] = f"{assignment.get('assignmentMethod', 'sequence_assignment')}+boundary_refinement"
    updated["selectedPairs"] = refined_pairs
    return updated


def report_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def predict_source_start_with_model(model: dict, reference_duration: int, source_in: int, source_out: int, pair: dict, singleton_segment: bool) -> int | None:
    feature_names = model.get("feature_names")
    weights = model.get("weights")
    if not isinstance(feature_names, list) or not isinstance(weights, list) or len(feature_names) != len(weights):
        return None
    source_duration = max(1, source_out - source_in)
    slack = max(0, source_duration - reference_duration)
    identity_start = int(pair["identitySourceStartFrame"]) if pair.get("identitySourceStartFrame") is not None else source_in
    boundary_start = int(pair.get("candidateSourceStartFrame", identity_start))
    boundary_observed = pair.get("boundaryDistance") is not None
    small_prior = round(slack * 0.67) if slack / max(1, reference_duration) <= 0.5 else round(slack * 0.5)
    center_prior = round(slack * 0.5)
    tail_prior = slack
    tail_trimmed_prior = max(0, slack - round(slack * 0.15))
    short_handle = slack / max(1, reference_duration) <= 0.25
    short_handle_tail_prior = tail_trimmed_prior if short_handle else 0
    large_handle = slack / max(1, reference_duration) >= 1.0
    large_handle_conservative_prior = round(slack * 0.42) if large_handle else 0
    features = {
        "bias": 1.0,
        "reference_duration": reference_duration / 300.0,
        "source_duration": source_duration / 600.0,
        "handle_slack": slack / 300.0,
        "handle_slack_ratio": slack / max(1, reference_duration),
        "identity_offset": (identity_start - source_in) / 300.0,
        "boundary_offset": (boundary_start - source_in) / 300.0,
        "boundary_observed": 1.0 if boundary_observed else 0.0,
        "boundary_offset_when_observed": (boundary_start - source_in) / 300.0 if boundary_observed else 0.0,
        "small_handle_prior_offset": small_prior / 300.0,
        "center_prior_offset": center_prior / 300.0,
        "tail_prior_offset": tail_prior / 300.0,
        "tail_trimmed_prior_offset": tail_trimmed_prior / 300.0,
        "short_handle_tail_prior_offset": short_handle_tail_prior / 300.0,
        "short_handle_tail_bias": 1.0 if short_handle else 0.0,
        "large_handle_conservative_prior_offset": large_handle_conservative_prior / 300.0,
        "large_handle_conservative_bias": 1.0 if large_handle else 0.0,
        "identity_boundary_delta": (boundary_start - identity_start) / 300.0,
        "singleton_segment": 1.0 if singleton_segment else 0.0,
    }
    ratio = float(features["handle_slack_ratio"])
    features["handle_slack_ratio_squared"] = ratio * ratio
    features["boundary_offset_x_handle_ratio"] = float(features["boundary_offset"]) * ratio
    features["boundary_observed_x_handle_ratio"] = float(features["boundary_observed"]) * ratio
    features["identity_offset_x_handle_ratio"] = float(features["identity_offset"]) * ratio
    features["small_prior_x_handle_ratio"] = float(features["small_handle_prior_offset"]) * ratio
    features["center_prior_x_handle_ratio"] = float(features["center_prior_offset"]) * ratio
    features["tail_prior_x_handle_ratio"] = float(features["tail_prior_offset"]) * ratio
    features["tail_trimmed_prior_x_handle_ratio"] = float(features["tail_trimmed_prior_offset"]) * ratio
    features["short_handle_tail_prior_x_handle_ratio"] = float(features["short_handle_tail_prior_offset"]) * ratio
    features["large_handle_conservative_prior_x_handle_ratio"] = float(features["large_handle_conservative_prior_offset"]) * ratio
    value = 0.0
    for index, name in enumerate(feature_names):
        value += float(weights[index]) * float(features.get(str(name), 0.0))
    offset = max(0, min(slack, round(value * 300.0)))
    return max(source_in, min(max(source_in, source_out - reference_duration), source_in + offset))


def load_optional_json(path: str | None) -> dict | None:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else None


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


if __name__ == "__main__":
    raise SystemExit(main())
