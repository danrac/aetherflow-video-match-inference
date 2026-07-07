"""Request-shape audits for source-window benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def audit_source_window_run(run_dir: str | Path, *, fixture_manifest_path: str | Path | None = None) -> dict[str, Any]:
    run_path = Path(run_dir)
    request_rows = load_request_rows(run_path)
    canonical_rows = load_canonical_reference_rows(fixture_manifest_path) if fixture_manifest_path else []
    selected_pairs = load_selected_pairs(run_path)
    selected_sequence_match_count = selected_match_count(run_path, selected_pairs)
    short_rows = [row for row in request_rows if int(row["durationFrames"]) < 30]
    oversegmentation_ratio = round(len(request_rows) / max(1, len(canonical_rows)), 6) if canonical_rows else None
    status = "ok"
    findings = []
    if canonical_rows and len(request_rows) != len(canonical_rows):
        status = "segment_shape_mismatch"
        findings.append(
            f"Request count {len(request_rows)} does not match canonical reference count {len(canonical_rows)}."
        )
    if short_rows:
        status = "segment_shape_mismatch" if status == "ok" else status
        findings.append(f"{len(short_rows)} reference requests are shorter than 30 frames.")
    if selected_sequence_match_count is not None and selected_sequence_match_count > max(1, len(canonical_rows) if canonical_rows else len(request_rows)):
        status = "segment_shape_mismatch" if status == "ok" else status
        findings.append(f"Sequence assignment selected {selected_sequence_match_count} matches.")

    return {
        "schemaVersion": "1.0.0",
        "kind": "aetherflowVideoMatcherRequestShapeAudit",
        "status": status,
        "runDir": str(run_path),
        "fixtureManifestPath": str(fixture_manifest_path) if fixture_manifest_path else None,
        "summary": {
            "canonicalReferenceSegmentCount": len(canonical_rows) if canonical_rows else None,
            "cepReferenceRequestCount": len(request_rows),
            "shortReferenceRequestCount": len(short_rows),
            "selectedSequenceMatchCount": selected_sequence_match_count,
            "oversegmentationRatio": oversegmentation_ratio,
            "allRequestsHavePlacementModelPath": all(row["placementModelPathPresent"] for row in request_rows) if request_rows else False,
            "candidateCounts": sorted({int(row["candidateCount"]) for row in request_rows if row.get("candidateCount") is not None}),
        },
        "findings": findings,
        "recommendation": recommendation(status, canonical_rows, request_rows),
        "canonicalReferenceSegments": canonical_rows,
        "cepReferenceRequests": attach_parent_canonical_rows(request_rows, canonical_rows),
        "selectedPairs": selected_pairs,
    }


def load_request_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    source_window_dir = run_dir / "_source_window_match"
    for request_path in sorted(source_window_dir.glob("reference_*/source_window_request.json")):
        request = load_json_object(request_path)
        reference_features_path = Path(str(request["reference_feature_manifest_path"]))
        reference_features = load_json_object(reference_features_path)
        reference_window = reference_features.get("source_window") if isinstance(reference_features.get("source_window"), dict) else {}
        start_frame = int(reference_window.get("source_in", 0) or 0)
        end_frame = int(reference_window.get("source_out", start_frame + 1) or start_frame + 1)
        rows.append(
            {
                "referenceSegmentId": str(reference_window.get("source_clip_id") or request_path.parent.name),
                "referenceStartFrame": start_frame,
                "referenceEndFrame": end_frame,
                "durationFrames": max(0, end_frame - start_frame),
                "candidateCount": len(request.get("candidates", [])),
                "placementModelPathPresent": bool(request.get("placement_model_path")),
                "requestPath": str(request_path),
                "referenceFeatureManifestPath": str(reference_features_path),
            }
        )
    return rows if rows else load_report_match_reference_rows(run_dir)


def load_report_match_reference_rows(run_dir: Path) -> list[dict[str, Any]]:
    report_path = run_dir / "footage_sync_match_report.json"
    if not report_path.exists():
        return []
    report = load_json_object(report_path)
    rows = []
    for index, match in enumerate(report.get("matches", []) if isinstance(report.get("matches"), list) else [], start=1):
        if not isinstance(match, dict):
            continue
        selected = match.get("selectedCandidate") if isinstance(match.get("selectedCandidate"), dict) else {}
        reference_range = selected.get("selectedReferenceRange") if isinstance(selected.get("selectedReferenceRange"), dict) else {}
        start_frame = int(reference_range.get("startFrame", match.get("reference_in", 0)) or 0)
        end_frame = int(reference_range.get("endFrame", match.get("reference_out", start_frame + 1)) or start_frame + 1)
        rows.append(
            {
                "referenceSegmentId": str(selected.get("referenceSegmentId") or match.get("referenceSegmentId") or f"selected_match_{index:03d}"),
                "referenceStartFrame": start_frame,
                "referenceEndFrame": end_frame,
                "durationFrames": max(0, end_frame - start_frame),
                "candidateCount": None,
                "placementModelPathPresent": bool(selected.get("placementCandidatePolicy")),
                "requestPath": None,
                "referenceFeatureManifestPath": None,
                "source": "footage_sync_match_report.selectedCandidate",
            }
        )
    return rows


def load_canonical_reference_rows(fixture_manifest_path: str | Path | None) -> list[dict[str, Any]]:
    if not fixture_manifest_path:
        return []
    manifest_path = Path(fixture_manifest_path)
    manifest = load_json_object(manifest_path)
    fps = float(manifest.get("output_fps", 30.0) or 30.0)
    cursor = 0
    rows = []
    for index, segment in enumerate(manifest.get("reference_segments", []), start=1):
        if not isinstance(segment, dict):
            continue
        duration = max(1, round(float(segment.get("duration_seconds", 0.0) or 0.0) * fps))
        source_path = str(segment.get("source_path", ""))
        rows.append(
            {
                "canonicalReferenceSegmentId": f"canonical_ref_{index:03d}",
                "referenceStartFrame": cursor,
                "referenceEndFrame": cursor + duration,
                "durationFrames": duration,
                "sourcePath": source_path,
                "sourceName": Path(source_path).name if source_path else "",
            }
        )
        cursor += duration
    return rows


def attach_parent_canonical_rows(request_rows: list[dict[str, Any]], canonical_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not canonical_rows:
        return request_rows
    rows = []
    for row in request_rows:
        start_frame = int(row["referenceStartFrame"])
        end_frame = int(row["referenceEndFrame"])
        parents = []
        for canonical in canonical_rows:
            overlap = max(
                0,
                min(end_frame, int(canonical["referenceEndFrame"])) - max(start_frame, int(canonical["referenceStartFrame"])),
            )
            if overlap > 0:
                parents.append(
                    {
                        "canonicalReferenceSegmentId": canonical["canonicalReferenceSegmentId"],
                        "overlapFrames": overlap,
                    }
                )
        rows.append({**row, "parentCanonicalSegments": parents})
    return rows


def load_selected_pairs(run_dir: Path) -> list[dict[str, Any]]:
    plan_path = run_dir / "footage_sync_match_plan.json"
    if not plan_path.exists():
        return []
    plan = load_json_object(plan_path)
    matcher = plan.get("matcher") if isinstance(plan.get("matcher"), dict) else {}
    source_window = matcher.get("sourceWindow") if isinstance(matcher.get("sourceWindow"), dict) else {}
    assignment = source_window.get("sequenceAssignment") if isinstance(source_window.get("sequenceAssignment"), dict) else {}
    pairs = assignment.get("selectedPairs")
    return pairs if isinstance(pairs, list) else []


def selected_match_count(run_dir: Path, selected_pairs: list[dict[str, Any]]) -> int | None:
    if selected_pairs:
        return len(selected_pairs)
    plan_path = run_dir / "footage_sync_match_plan.json"
    if plan_path.exists():
        plan = load_json_object(plan_path)
        matcher = plan.get("matcher") if isinstance(plan.get("matcher"), dict) else {}
        source_window = matcher.get("sourceWindow") if isinstance(matcher.get("sourceWindow"), dict) else {}
        value = source_window.get("sequenceSelectedCount")
        if value is not None:
            return int(value)
    report_path = run_dir / "footage_sync_match_report.json"
    if report_path.exists():
        report = load_json_object(report_path)
        matches = report.get("matches")
        if isinstance(matches, list):
            return len(matches)
    audit_path = run_dir / "footage_sync_candidate_audit.json"
    if audit_path.exists():
        audit = load_json_object(audit_path)
        summary = audit.get("summary") if isinstance(audit.get("summary"), dict) else {}
        value = summary.get("selectedMatchCount")
        if value is not None:
            return int(value)
    return None


def recommendation(status: str, canonical_rows: list[dict[str, Any]], request_rows: list[dict[str, Any]]) -> str:
    if status == "ok":
        return "Request shape matches the available benchmark reference structure."
    if canonical_rows:
        return (
            "Use canonical fixture segment windows for model benchmark smokes, or merge/filter live scene-detected "
            "reference fragments before source-window requests and sequence assignment."
        )
    return "Inspect and filter short/noisy live scene-detected reference fragments before sequence assignment."


def load_json_object(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return document
