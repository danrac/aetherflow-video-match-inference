"""Inference engine boundary."""

from dataclasses import dataclass
import json
from pathlib import Path

from .features import color_distance, confidence_from_distance, load_feature_manifest
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
    candidates = [source_window_candidate_to_scoring_input(candidate) for candidate in request.candidates]
    ranked = rank_candidates(reference_features, candidates, list(request.transforms), reranker_model)
    if not ranked:
        return source_window_result(request, model_manifest, reference_features, reranker_model, [], [])

    selected_group_id = str(ranked[0]["candidate_id"])
    selected_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.get("candidate_group_id") or candidate["candidate_id"]) == selected_group_id
    ]
    selected_candidates.sort(key=lambda candidate: (candidate["source_window_entry"].get("role", ""), int(candidate["source_window_entry"].get("source_in", 0)), candidate["candidate_id"]))
    selected_matches = source_window_matches_from_candidates(request, reference_features, selected_candidates, ranked[0])
    return source_window_result(request, model_manifest, reference_features, reranker_model, ranked, selected_matches)


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


def source_window_matches_from_candidates(request: SourceWindowMatchRequest, reference_features: dict, candidates: list[dict], top_ranked: dict) -> list[dict]:
    reference_duration = int(reference_features.get("duration_frames", 0) or 0)
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
        matches.append(
            {
                "source_path": candidate["source_path"],
                "reference_in": reference_in,
                "reference_out": reference_out,
                "source_in": source_in,
                "source_out": source_out,
                "timeline_track": int(candidate.get("timeline_track", 0)),
                "confidence": confidence_from_distance(max(0.0, window_distance)),
                "reconstruction": {
                    "operation": "source_window_reranker_match",
                    "parameters": {
                        "candidate_id": candidate["candidate_id"],
                        "candidate_group_id": candidate.get("candidate_group_id"),
                        "source_clip_id": candidate["clip_id"],
                        "role": candidate["source_window_entry"].get("role"),
                        "group_distance": top_ranked["distance"],
                        "raw_group_distance": top_ranked.get("raw_distance"),
                        "window_distance": window_distance,
                        "transforms": list(request.transforms),
                    },
                },
            }
        )
    return matches


def source_window_result(request: SourceWindowMatchRequest, model_manifest: dict, reference_features: dict, reranker_model: dict | None, ranked: list[dict], matches: list[dict]) -> dict:
    return {
        "schema_version": "0.1.0",
        "model_id": model_manifest["model_id"],
        "model_version": model_manifest["model_version"],
        "reference": {
            "path": request.reference_path,
            "fps": float(reference_features.get("fps", 24.0) or 24.0),
            "duration_frames": max(1, int(reference_features.get("duration_frames", 0) or 0)),
        },
        "ranking": {
            "candidate_group_count": len({candidate.candidate_group_id for candidate in request.candidates}),
            "top_candidates": ranked[:10],
            "reranker_model": reranker_model_summary(reranker_model),
        },
        "matches": matches,
    }


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
