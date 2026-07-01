"""Inference engine boundary."""

from dataclasses import dataclass
import json
from pathlib import Path

from .features import color_distance, confidence_from_distance, load_feature_manifest
from .media_window import media_window_rescore, refine_boundary_start
from .placement import load_placement_model, placement_candidates_for_match, placement_model_summary
from .reranker import load_reranker_model, rank_candidates, reranker_model_summary


@dataclass(frozen=True)
class MatchRequest:
    reference_path: str
    source_paths: tuple[str, ...]
    model_manifest_path: str
    reference_feature_manifest_path: str | None = None
    source_feature_manifest_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceWindowCandidate:
    candidate_id: str
    candidate_group_id: str
    source_path: str
    source_clip_id: str
    feature_manifest_path: str
    source_in: int
    source_out: int
    role: str = "source"
    timeline_track: int = 0


@dataclass(frozen=True)
class SourceWindowMatchRequest:
    reference_path: str
    model_manifest_path: str
    reference_feature_manifest_path: str
    candidates: tuple[SourceWindowCandidate, ...]
    transforms: tuple[dict, ...] = ()
    reranker_model_path: str | None = None
    placement_model_path: str | None = None


def describe_request(request: MatchRequest) -> dict[str, object]:
    return {
        "reference_path": request.reference_path,
        "source_count": len(request.source_paths),
        "model_manifest_path": request.model_manifest_path,
        "reference_feature_manifest_path": request.reference_feature_manifest_path,
        "source_feature_manifest_count": len(request.source_feature_manifest_paths),
    }


def describe_source_window_request(request: SourceWindowMatchRequest) -> dict[str, object]:
    return {
        "reference_path": request.reference_path,
        "model_manifest_path": request.model_manifest_path,
        "reference_feature_manifest_path": request.reference_feature_manifest_path,
        "candidate_count": len(request.candidates),
        "candidate_group_count": len({candidate.candidate_group_id for candidate in request.candidates}),
        "transform_types": sorted({str(transform.get("type")) for transform in request.transforms if transform.get("type")}),
        "reranker_model_path": request.reranker_model_path,
        "placement_model_path": request.placement_model_path,
    }


def load_model_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Expected model manifest object at {path}")
    return document


def match_source_windows(request: SourceWindowMatchRequest) -> dict:
    """Rank explicit grouped source-window candidates and reconstruct timeline edits."""

    model_manifest = load_model_manifest(request.model_manifest_path)
    reference_features = load_feature_manifest(request.reference_feature_manifest_path)
    reranker_model = load_reranker_model(request.reranker_model_path) if request.reranker_model_path else None
    placement_model = load_placement_model(request.placement_model_path) if request.placement_model_path else None
    candidates = [source_window_candidate_to_scoring_input(candidate) for candidate in request.candidates]
    ranked = rank_candidates(reference_features, candidates, list(request.transforms), reranker_model)
    ranked = rescore_ranked_source_windows_with_media(request, reference_features, candidates, ranked)
    if not ranked:
        return source_window_result(request, model_manifest, reference_features, reranker_model, placement_model, [], [])

    selected_group_id = str(ranked[0]["candidate_id"])
    selected_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.get("candidate_group_id") or candidate["candidate_id"]) == selected_group_id
    ]
    selected_candidates.sort(key=lambda candidate: (candidate["source_window_entry"].get("role", ""), int(candidate["source_window_entry"].get("source_in", 0)), candidate["candidate_id"]))
    selected_matches = source_window_matches_from_candidates(request, reference_features, selected_candidates, ranked[0], placement_model)
    return source_window_result(request, model_manifest, reference_features, reranker_model, placement_model, ranked, selected_matches)


def rescore_ranked_source_windows_with_media(request: SourceWindowMatchRequest, reference_features: dict, candidates: list[dict], ranked: list[dict]) -> list[dict]:
    if not ranked:
        return ranked
    candidate_by_group = {str(candidate.get("candidate_group_id") or candidate["candidate_id"]): candidate for candidate in candidates}
    rescored = []
    media_rescore_limit = 6
    for rank_index, item in enumerate(ranked):
        candidate = candidate_by_group.get(str(item.get("candidate_id", "")))
        if candidate is None or rank_index >= media_rescore_limit:
            updated = dict(item)
            if rank_index >= media_rescore_limit:
                updated["media_window"] = {"distance": None, "skipped": True, "reason": "outside_media_rescore_top_n"}
            rescored.append(updated)
            continue
        media = media_window_rescore(request.reference_path, reference_features, candidate)
        if media is None or media.get("distance") is None:
            updated = dict(item)
            updated["media_window"] = media
            rescored.append(updated)
            continue
        updated = dict(item)
        media_distance = float(media["distance"])
        feature_distance = float(item.get("distance", float("inf")))
        updated["feature_distance"] = round(feature_distance, 6) if feature_distance != float("inf") else float("inf")
        updated["media_window"] = media
        updated["distance"] = round((feature_distance * 0.10) + (media_distance * 0.90), 6)
        updated["raw_distance"] = updated["distance"]
        updated["window_candidates"] = annotate_media_window_candidates(item.get("window_candidates", []), media)
        rescored.append(updated)
    return sorted(rescored, key=lambda row: (float(row["distance"]), row["candidate_id"]))


def annotate_media_window_candidates(window_candidates: list[dict], media: dict) -> list[dict]:
    if not window_candidates:
        return window_candidates
    annotated = []
    for index, candidate in enumerate(window_candidates):
        updated = dict(candidate)
        if index == 0:
            updated["media_distance"] = media.get("distance")
            updated["source_in"] = int(media.get("source_in", updated.get("source_in", 0)))
            updated["source_out"] = int(media.get("source_out", updated.get("source_out", updated.get("source_in", 0) + 1)))
        annotated.append(updated)
    return annotated


def source_window_candidate_to_scoring_input(candidate: SourceWindowCandidate) -> dict:
    feature_document = load_feature_manifest(candidate.feature_manifest_path)
    entry = {
        "source_clip_id": candidate.source_clip_id,
        "source_in": candidate.source_in,
        "source_out": candidate.source_out,
        "role": candidate.role,
        "source_path": candidate.source_path,
    }
    feature_document["source_window_entry"] = entry
    if "source_window" not in feature_document:
        feature_document["source_window"] = {"source_in": candidate.source_in, "source_out": candidate.source_out}
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_group_id": candidate.candidate_group_id,
        "clip_id": candidate.source_clip_id,
        "source_path": candidate.source_path,
        "source_window_entry": entry,
        "features": feature_document,
        "timeline_track": candidate.timeline_track,
    }


def source_window_matches_from_candidates(request: SourceWindowMatchRequest, reference_features: dict, candidates: list[dict], top_ranked: dict, placement_model: dict | None = None) -> list[dict]:
    reference_duration = reference_window_duration(reference_features)
    source_window_lengths = [source_window_length(candidate["features"]) for candidate in candidates]
    total_window_length = sum(source_window_lengths)
    reference_cursor = 0
    matches = []
    for index, candidate in enumerate(candidates):
        feature_document = candidate["features"]
        source_in, source_out = source_range(feature_document, reference_duration)
        source_duration = max(1, source_out - source_in)
        if total_window_length > 0 and reference_duration > 0:
            if index == len(candidates) - 1:
                reference_out = reference_duration
            else:
                reference_out = min(reference_duration, reference_cursor + max(1, round(reference_duration * source_window_lengths[index] / total_window_length)))
            reference_in = reference_cursor
            reference_cursor = reference_out
        else:
            reference_in = 0
            reference_out = max(1, min(reference_duration or source_duration, source_duration))
        scored_window = next((item for item in top_ranked.get("window_candidates", []) if item.get("candidate_id") == candidate["candidate_id"]), None)
        window_distance = float(scored_window["distance"]) if scored_window and scored_window.get("distance") != float("inf") else float(top_ranked["distance"])
        if scored_window:
            source_in = int(scored_window.get("source_in", source_in))
            source_out = int(scored_window.get("source_out", source_out))
            source_in, source_out = refine_selected_source_window(
                request.reference_path,
                reference_features,
                candidate,
                source_in,
                source_out,
                reference_duration,
            )
        placement = None
        if placement_model is not None:
            placement = placement_candidates_for_match(
                reference_path=request.reference_path,
                source_path=candidate["source_path"],
                reference_start_frame=reference_feature_start(reference_features) + reference_in,
                reference_duration=max(1, reference_out - reference_in),
                source_start_frame=source_in,
                source_duration=max(1, source_out - source_in),
                fps=float(reference_features.get("fps", 30.0) or 30.0),
                model=placement_model,
            )
        reconstruction_parameters = {
            "candidate_id": candidate["candidate_id"],
            "candidate_group_id": candidate.get("candidate_group_id"),
            "source_clip_id": candidate["clip_id"],
            "role": candidate["source_window_entry"].get("role"),
            "group_distance": top_ranked["distance"],
            "raw_group_distance": top_ranked.get("raw_distance"),
            "window_distance": window_distance,
            "transforms": list(request.transforms),
        }
        if placement is not None:
            reconstruction_parameters["placement"] = placement
        matches.append(
            {
                "source_path": candidate["source_path"],
                "reference_in": reference_in,
                "reference_out": reference_out,
                "source_in": source_in,
                "source_out": source_out,
                **placement_match_fields(placement),
                "timeline_track": int(candidate.get("timeline_track", 0)),
                "confidence": confidence_from_distance(max(0.0, window_distance)),
                "reconstruction": {
                    "operation": "source_window_reranker_match",
                    "parameters": reconstruction_parameters,
                },
            }
        )
    return matches


def source_window_result(request: SourceWindowMatchRequest, model_manifest: dict, reference_features: dict, reranker_model: dict | None, placement_model: dict | None, ranked: list[dict], matches: list[dict]) -> dict:
    diagnostics = source_window_diagnostics(request, reference_features, ranked, matches)
    return {
        "schema_version": "0.1.0",
        "model_id": model_manifest["model_id"],
        "model_version": model_manifest["model_version"],
        "reference": {
            "path": request.reference_path,
            "fps": float(reference_features.get("fps", 24.0) or 24.0),
            "duration_frames": max(1, reference_window_duration(reference_features)),
        },
        "ranking": {
            "candidate_group_count": len({candidate.candidate_group_id for candidate in request.candidates}),
            "top_candidates": ranked[:10],
            "reranker_model": reranker_model_summary(reranker_model),
            "placement_model": placement_model_summary(placement_model),
        },
        "diagnostics": diagnostics,
        "matches": matches,
    }


def source_window_diagnostics(request: SourceWindowMatchRequest, reference_features: dict, ranked: list[dict], matches: list[dict] | None = None) -> dict:
    reference_window = reference_features.get("source_window") if isinstance(reference_features.get("source_window"), dict) else {}
    reference_start = int(reference_window.get("source_in", 0) or 0)
    reference_end = int(reference_window.get("source_out", reference_start + max(1, int(reference_features.get("duration_frames", 0) or 1))) or reference_start + 1)
    reference_segment_id = str(reference_window.get("source_clip_id") or Path(request.reference_feature_manifest_path).stem)
    selected_id = str(ranked[0]["candidate_id"]) if ranked else ""
    selected_match_by_candidate_id = {}
    for match in matches or []:
        params = match.get("reconstruction", {}).get("parameters", {}) if isinstance(match.get("reconstruction"), dict) else {}
        candidate_id_for_match = str(params.get("candidate_id", ""))
        if candidate_id_for_match:
            selected_match_by_candidate_id[candidate_id_for_match] = match
    candidate_rows = []
    for item in ranked:
        candidate_id = str(item.get("candidate_id", ""))
        window = item.get("window_candidates", [{}])[0] if item.get("window_candidates") else {}
        source_start = int(window.get("source_in", item.get("source_in", 0)) or 0)
        source_end = int(window.get("source_out", item.get("source_out", source_start + 1)) or source_start + 1)
        selected_match = selected_match_by_candidate_id.get(candidate_id)
        if selected_match is not None:
            source_start = int(selected_match.get("source_in", source_start))
            source_end = int(selected_match.get("source_out", source_end))
        raw_distance = float(item.get("raw_distance", item.get("distance", float("inf"))))
        final_distance = float(item.get("distance", float("inf")))
        candidate_rows.append(
            {
                "referenceSegmentId": reference_segment_id,
                "candidateSourceSegmentId": candidate_source_segment_id(candidate_id),
                "candidateSourceStartFrame": source_start,
                "candidateSourceEndFrame": source_end,
                "referenceStartFrame": reference_start,
                "referenceEndFrame": reference_end,
                "visualScore": confidence_from_distance(raw_distance if raw_distance != float("inf") else None),
                "temporalOrderScore": None,
                "modelScore": confidence_from_distance(final_distance if final_distance != float("inf") else None),
                "finalScore": confidence_from_distance(final_distance if final_distance != float("inf") else None),
                "selected": candidate_id == selected_id,
                "rejectionReason": "" if candidate_id == selected_id else "lower_ranked_candidate",
            }
        )
    return {
        "rankedCandidates": candidate_rows,
        "globalAssignment": {
            "assignmentMethod": "per_reference_ranking",
            "skippedReferenceSegments": [],
            "skippedSourceSegments": [],
            "selectedPairs": [
                {
                    "referenceSegmentId": reference_segment_id,
                    "candidateSourceSegmentId": candidate_rows[0]["candidateSourceSegmentId"],
                    "candidateSourceStartFrame": candidate_rows[0]["candidateSourceStartFrame"],
                    "score": candidate_rows[0]["finalScore"],
                }
            ] if candidate_rows else [],
            "globalScore": candidate_rows[0]["finalScore"] if candidate_rows else 0.0,
            "confidenceCalibrationNotes": "Single-reference source-window ranking; batch monotonic assignment is available in fixture evaluation.",
        },
    }


def candidate_source_segment_id(candidate_id: str) -> str:
    if "_refcut_" in candidate_id:
        return candidate_id.split("_refcut_", 1)[0]
    return candidate_id


def reference_window_duration(reference_features: dict) -> int:
    reference_window = reference_features.get("source_window")
    if isinstance(reference_window, dict):
        source_in = int(reference_window.get("source_in", 0) or 0)
        source_out = int(reference_window.get("source_out", source_in + 1) or source_in + 1)
        if source_out > source_in:
            return source_out - source_in
    return max(1, int(reference_features.get("duration_frames", 0) or 1))


def reference_feature_start(reference_features: dict) -> int:
    reference_window = reference_features.get("source_window")
    if isinstance(reference_window, dict):
        return int(reference_window.get("source_in", 0) or 0)
    return 0


def placement_match_fields(placement: dict | None) -> dict:
    if placement is None:
        return {}
    return {
        "referencePlacementFrame": placement.get("referencePlacementFrame"),
        "sourcePlacementFrame": placement.get("sourcePlacementFrame"),
        "referencePlacementTime": placement.get("referencePlacementTime"),
        "sourcePlacementTime": placement.get("sourcePlacementTime"),
        "placementFrameConfidence": placement.get("placementFrameConfidence"),
        "cropXHint": placement.get("cropXHint"),
        "scalePrior": placement.get("scalePrior"),
        "placementSampleCandidates": placement.get("placementSampleCandidates", []),
        "placementSampleCandidateCount": placement.get("placementSampleCandidateCount"),
        "placementCandidatePolicy": placement.get("placementCandidatePolicy"),
    }


def refine_selected_source_window(
    reference_path: str,
    reference_features: dict,
    candidate: dict,
    source_in: int,
    source_out: int,
    reference_duration: int,
) -> tuple[int, int]:
    full_source_in, full_source_out = source_range(candidate["features"], reference_duration)
    max_start = max(full_source_in, full_source_out - reference_duration)
    if reference_duration <= 2 or max_start <= full_source_in:
        return source_in, source_out

    refined_start = max(full_source_in, min(max_start, source_in))
    handle_slack = max(0, full_source_out - full_source_in - reference_duration)
    if refined_start == full_source_in and handle_slack > 0 and handle_slack / max(1, reference_duration) <= 0.5:
        refined_start = max(full_source_in, min(max_start, full_source_in + round(handle_slack * 0.67)))

    try:
        from PIL import Image, ImageOps
        import numpy as np
    except Exception:
        return refined_start, min(full_source_out, refined_start + reference_duration)

    source_duration = max(1, full_source_out - full_source_in)
    if source_duration >= reference_duration * 1.5:
        reference_window = reference_features.get("source_window") if isinstance(reference_features.get("source_window"), dict) else {}
        reference_start = int(reference_window.get("source_in", 0) or 0)
        reference_fps = float(reference_features.get("fps", 30.0) or 30.0)
        source_fps = float(candidate["features"].get("fps", reference_fps) or reference_fps)
        boundary = refine_boundary_start(
            reference_path,
            reference_features,
            candidate,
            full_source_in,
            max_start,
            reference_start,
            reference_duration,
            reference_fps,
            source_fps,
            np,
            Image,
            ImageOps,
            baseline_start=refined_start,
        )
        baseline_distance = boundary.get("baseline_distance") if boundary is not None else None
        boundary_distance = boundary.get("distance") if boundary is not None else None
        improvement = float(baseline_distance) - float(boundary_distance) if baseline_distance is not None and boundary_distance is not None else 0.0
        if boundary is not None and improvement >= 2.0:
            refined_start = int(boundary["source_in"])

    return refined_start, min(full_source_out, refined_start + reference_duration)


def match(request: MatchRequest) -> dict:
    """Return a deterministic contract-level match result.

    This is a placeholder runtime until real ONNX feature matching lands.
    """

    model_manifest = load_model_manifest(request.model_manifest_path)
    if request.reference_feature_manifest_path and request.source_feature_manifest_paths:
        return match_from_feature_manifests(request, model_manifest)

    matches = []
    for index, source_path in enumerate(request.source_paths):
        matches.append(
            {
                "source_path": source_path,
                "reference_in": index * 120,
                "reference_out": (index + 1) * 120,
                "source_in": 0,
                "source_out": 120,
                "timeline_track": 0,
                "confidence": 0.5,
                "reconstruction": {
                    "operation": "placeholder_match",
                    "parameters": {
                        "runtime": "contract-smoke",
                    },
                },
            }
        )

    return {
        "schema_version": "0.1.0",
        "model_id": model_manifest["model_id"],
        "model_version": model_manifest["model_version"],
        "reference": {
            "path": request.reference_path,
            "fps": 24.0,
            "duration_frames": max(120, len(matches) * 120),
        },
        "matches": matches,
    }


def match_from_feature_manifests(request: MatchRequest, model_manifest: dict) -> dict:
    reference_features = load_feature_manifest(request.reference_feature_manifest_path or "")
    source_features = [load_feature_manifest(path) for path in request.source_feature_manifest_paths]
    matches = []
    reference_duration = int(reference_features.get("duration_frames", 0) or 0)
    reference_fps = float(reference_features.get("fps", 24.0) or 24.0)
    source_window_lengths = [source_window_length(feature_document) for feature_document in source_features]
    total_window_length = sum(source_window_lengths)
    reference_cursor = 0

    for index, source_path in enumerate(request.source_paths):
        feature_document = source_features[index] if index < len(source_features) else {}
        distance = color_distance(reference_features, feature_document)
        source_in, source_out = source_range(feature_document, reference_duration)
        source_duration = max(1, source_out - source_in)
        if total_window_length > 0 and reference_duration > 0:
            if index == len(request.source_paths) - 1:
                reference_out = reference_duration
            else:
                reference_out = min(reference_duration, reference_cursor + max(1, round(reference_duration * source_window_lengths[index] / total_window_length)))
            reference_in = reference_cursor
            reference_cursor = reference_out
        else:
            reference_in = 0
            reference_out = max(1, min(reference_duration or source_duration, source_duration))
        matches.append(
            {
                "source_path": source_path,
                "reference_in": reference_in,
                "reference_out": reference_out,
                "source_in": source_in,
                "source_out": source_out,
                "timeline_track": 0,
                "confidence": confidence_from_distance(distance),
                "reconstruction": {
                    "operation": "feature_manifest_match",
                    "parameters": {
                        "feature_version": feature_document.get("feature_version", "unknown"),
                        "distance": distance,
                        "source_window": feature_document.get("source_window"),
                    },
                },
            }
        )

    return {
        "schema_version": "0.1.0",
        "model_id": model_manifest["model_id"],
        "model_version": model_manifest["model_version"],
        "reference": {
            "path": request.reference_path,
            "fps": reference_fps,
            "duration_frames": max(1, reference_duration),
        },
        "matches": matches,
    }


def source_window_length(feature_document: dict) -> int:
    source_window = feature_document.get("source_window")
    if not isinstance(source_window, dict):
        return 0
    return max(0, int(source_window.get("source_out", 0)) - int(source_window.get("source_in", 0)))


def source_range(feature_document: dict, fallback_duration: int) -> tuple[int, int]:
    source_window = feature_document.get("source_window")
    if isinstance(source_window, dict):
        source_in = int(source_window.get("source_in", 0))
        source_out = int(source_window.get("source_out", source_in + 1))
        if source_out > source_in:
            return source_in, source_out
    duration = int(feature_document.get("duration_frames", 0) or fallback_duration or 1)
    return 0, max(1, duration)
