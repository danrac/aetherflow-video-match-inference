"""Inference engine boundary."""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from time import perf_counter

from .features import color_distance, confidence_from_distance, load_feature_manifest
from .media_window import candidate_search_max_start, candidate_start_grid, media_window_rescore, read_video_frame, refine_boundary_start, sample_relative_frames, source_crop_images
from .placement import load_placement_model, placement_candidates_for_match, placement_model_summary
from .reranker import load_reranker_model, rank_candidates, reranker_model_summary
from .visual_encoder import VisualEncoderScorer

DEFAULT_RANKED_PLACEMENT_LIMIT = 10
_VISUAL_ENCODER_SCORERS: dict[str, VisualEncoderScorer] = {}


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
    metadata: dict | None = None


@dataclass(frozen=True)
class SourceWindowMatchRequest:
    reference_path: str
    model_manifest_path: str
    reference_feature_manifest_path: str
    candidates: tuple[SourceWindowCandidate, ...]
    transforms: tuple[dict, ...] = ()
    reranker_model_path: str | None = None
    placement_model_path: str | None = None
    visual_encoder_onnx_path: str | None = None
    metadata: dict | None = None


@dataclass
class SourceWindowBatchCache:
    """Shared model/feature cache for batch source-window inference."""

    model_manifests: dict[str, dict] = field(default_factory=dict)
    feature_manifests: dict[str, dict] = field(default_factory=dict)
    reranker_models: dict[str, dict | None] = field(default_factory=dict)
    placement_models: dict[str, dict | None] = field(default_factory=dict)
    stats: dict[str, int] = field(
        default_factory=lambda: {
            "model_manifest_hits": 0,
            "model_manifest_misses": 0,
            "feature_manifest_hits": 0,
            "feature_manifest_misses": 0,
            "reranker_model_hits": 0,
            "reranker_model_misses": 0,
            "placement_model_hits": 0,
            "placement_model_misses": 0,
        }
    )

    def load_model_manifest(self, path: str | Path) -> dict:
        key = normalized_cache_key(path)
        cached = self.model_manifests.get(key)
        if cached is not None:
            self.stats["model_manifest_hits"] += 1
            return cached
        self.stats["model_manifest_misses"] += 1
        loaded = load_model_manifest(path)
        self.model_manifests[key] = loaded
        return loaded

    def load_feature_manifest(self, path: str | Path) -> dict:
        key = normalized_cache_key(path)
        cached = self.feature_manifests.get(key)
        if cached is not None:
            self.stats["feature_manifest_hits"] += 1
            return feature_working_copy(cached)
        self.stats["feature_manifest_misses"] += 1
        loaded = load_feature_manifest(path)
        self.feature_manifests[key] = loaded
        return feature_working_copy(loaded)

    def load_reranker_model(self, path: str | Path | None) -> dict | None:
        if not path:
            return None
        key = normalized_cache_key(path)
        if key in self.reranker_models:
            self.stats["reranker_model_hits"] += 1
            return self.reranker_models[key]
        self.stats["reranker_model_misses"] += 1
        loaded = load_reranker_model(path)
        self.reranker_models[key] = loaded
        return loaded

    def load_placement_model(self, path: str | Path | None) -> dict | None:
        if not path:
            return None
        key = normalized_cache_key(path)
        if key in self.placement_models:
            self.stats["placement_model_hits"] += 1
            return self.placement_models[key]
        self.stats["placement_model_misses"] += 1
        loaded = load_placement_model(path)
        self.placement_models[key] = loaded
        return loaded

    def report(self) -> dict:
        return {
            "schema_version": "0.1.0",
            "stats": dict(self.stats),
            "cached_model_manifest_count": len(self.model_manifests),
            "cached_feature_manifest_count": len(self.feature_manifests),
            "cached_reranker_model_count": len(self.reranker_models),
            "cached_placement_model_count": len(self.placement_models),
        }


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
        "visual_encoder_onnx_path": request.visual_encoder_onnx_path,
        "metadata_keys": sorted(request.metadata.keys()) if isinstance(request.metadata, dict) else [],
    }


def load_model_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Expected model manifest object at {path}")
    return document


def normalized_cache_key(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def feature_working_copy(feature_document: dict) -> dict:
    return {key: value for key, value in feature_document.items() if key != "_aetherflow_feature_cache"}


def match_source_windows(request: SourceWindowMatchRequest, batch_cache: SourceWindowBatchCache | None = None) -> dict:
    """Rank explicit grouped source-window candidates and reconstruct timeline edits."""

    timings: list[dict[str, float | str]] = []
    total_start = perf_counter()
    model_manifest = timed_stage(timings, "model_loading.manifest", lambda: batch_cache.load_model_manifest(request.model_manifest_path) if batch_cache else load_model_manifest(request.model_manifest_path))
    reference_features = timed_stage(timings, "feature_loading.reference", lambda: batch_cache.load_feature_manifest(request.reference_feature_manifest_path) if batch_cache else load_feature_manifest(request.reference_feature_manifest_path))
    reranker_model = timed_stage(timings, "model_loading.reranker", lambda: batch_cache.load_reranker_model(request.reranker_model_path) if batch_cache else (load_reranker_model(request.reranker_model_path) if request.reranker_model_path else None))
    placement_model = timed_stage(
        timings,
        "model_loading.placement",
        lambda: None
        if source_window_skip_placement_enabled()
        else (batch_cache.load_placement_model(request.placement_model_path) if batch_cache else (load_placement_model(request.placement_model_path) if request.placement_model_path else None)),
    )
    feature_loader = batch_cache.load_feature_manifest if batch_cache else load_feature_manifest
    candidates = timed_stage(timings, "feature_loading.candidates", lambda: [source_window_candidate_to_scoring_input(candidate, feature_loader=feature_loader) for candidate in request.candidates])
    ranked = timed_stage(timings, "source_window_scoring", lambda: rank_candidates(reference_features, candidates, list(request.transforms), reranker_model))
    ranked = timed_stage(timings, "media_window_rescore", lambda: rescore_ranked_source_windows_with_media(request, reference_features, candidates, ranked))
    ranked = timed_stage(timings, "visual_encoder_rescore", lambda: rescore_ranked_source_windows_with_visual_encoder(request, reference_features, ranked))
    ranked = timed_stage(timings, "canonical_metadata_prior", lambda: apply_canonical_metadata_prior(request, reference_features, candidates, ranked))
    ranked = timed_stage(timings, "placement_candidate_generation", lambda: attach_placement_to_ranked_candidates(request, reference_features, candidates, ranked, placement_model))
    if not ranked:
        return source_window_result(request, model_manifest, reference_features, reranker_model, placement_model, [], [], timings, total_start)

    selected_group_id = str(ranked[0]["candidate_id"])
    selected_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.get("candidate_group_id") or candidate["candidate_id"]) == selected_group_id
    ]
    selected_candidates.sort(key=lambda candidate: (candidate["source_window_entry"].get("role", ""), int(candidate["source_window_entry"].get("source_in", 0)), candidate["candidate_id"]))
    selected_matches = timed_stage(timings, "timeline_reconstruction", lambda: source_window_matches_from_candidates(request, reference_features, selected_candidates, ranked[0], placement_model, ranked_context=ranked))
    return source_window_result(request, model_manifest, reference_features, reranker_model, placement_model, ranked, selected_matches, timings, total_start)


def match_source_windows_batch(requests: list[SourceWindowMatchRequest], *, assignment_top_n: int = 12) -> dict:
    """Run source-window matching for a batch while reusing loaded features/models."""

    from .sequence_assignment import assign_ranked_reference_sequence

    total_start = perf_counter()
    cache = SourceWindowBatchCache()
    results = []
    total_latency = 0.0
    for index, request in enumerate(requests):
        request_start = perf_counter()
        result = match_source_windows(request, batch_cache=cache)
        request_wall_latency = (perf_counter() - request_start) * 1000.0
        result.setdefault("performance", {})["batchRequestWallLatencyMs"] = round(request_wall_latency, 6)
        result["performance"]["batchRequestIndex"] = index
        total_latency += float(result.get("performance", {}).get("totalLatencyMs", 0.0) or 0.0)
        results.append({"request_index": index, "result": result})
    assignment_rows = []
    for entry in results:
        diagnostics = entry.get("result", {}).get("diagnostics", {}) if isinstance(entry.get("result"), dict) else {}
        candidates = diagnostics.get("rankedCandidates") if isinstance(diagnostics, dict) else None
        if not isinstance(candidates, list) or not candidates:
            continue
        assignment_rows.append(
            {
                "referenceSegmentId": candidates[0].get("referenceSegmentId"),
                "rankedCandidates": candidates,
                "microcut": False,
            }
        )
    sequence_assignment = assign_ranked_reference_sequence(assignment_rows, top_n=max(1, int(assignment_top_n))) if assignment_rows else {"selectedPairs": [], "globalScore": 0.0}
    wall_latency = (perf_counter() - total_start) * 1000.0
    return {
        "schema_version": "0.1.0",
        "complete": True,
        "request_count": len(requests),
        "total_match_latency_ms": round(total_latency, 6),
        "average_match_latency_ms": round(total_latency / len(requests), 6) if requests else 0.0,
        "batch_wall_latency_ms": round(wall_latency, 6),
        "average_batch_wall_latency_ms": round(wall_latency / len(requests), 6) if requests else 0.0,
        "cache": cache.report(),
        "sequence_assignment": sequence_assignment,
        "results": results,
    }


def timed_stage(timings: list[dict[str, float | str]], stage: str, callback):
    start = perf_counter()
    result = callback()
    timings.append({"stage": stage, "latencyMs": round((perf_counter() - start) * 1000.0, 6)})
    return result


def source_window_skip_placement_enabled() -> bool:
    return str(os.environ.get("AETHERFLOW_VIDEO_MATCH_SKIP_PLACEMENT", "")).lower() in {"1", "true", "yes", "on"}


def rescore_ranked_source_windows_with_media(request: SourceWindowMatchRequest, reference_features: dict, candidates: list[dict], ranked: list[dict]) -> list[dict]:
    if not ranked:
        return ranked
    if trusted_canonical_metadata_available(request, reference_features, candidates):
        return [ranked_candidate_with_skipped_media(item, "trusted_canonical_metadata_available") for item in ranked]
    if broad_visual_fast_path_enabled(request, ranked):
        return [ranked_candidate_with_skipped_media(item, "broad_visual_fast_path") for item in ranked]
    candidate_by_group = {str(candidate.get("candidate_group_id") or candidate["candidate_id"]): candidate for candidate in candidates}
    rescored = []
    media_rescore_limit = media_window_rescore_limit()
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
        updated["distance"] = round((feature_distance * 0.50) + (media_distance * 0.50), 6)
        updated["raw_distance"] = updated["distance"]
        updated["window_candidates"] = annotate_media_window_candidates(item.get("window_candidates", []), media)
        rescored.append(updated)
    return sorted(rescored, key=lambda row: (float(row["distance"]), row["candidate_id"]))


def media_window_rescore_limit() -> int:
    try:
        return max(1, int(os.environ.get("AETHERFLOW_VIDEO_MATCH_MEDIA_RESCORE_LIMIT", "6")))
    except ValueError:
        return 6


def broad_visual_fast_path_enabled(request: SourceWindowMatchRequest, ranked: list[dict]) -> bool:
    if not request.visual_encoder_onnx_path or len(ranked) <= 8:
        return False
    value = str(os.environ.get("AETHERFLOW_VIDEO_MATCH_BROAD_VISUAL_FAST_PATH", "1")).lower()
    return value not in {"0", "false", "no", "off"}


def rescore_ranked_source_windows_with_visual_encoder(request: SourceWindowMatchRequest, reference_features: dict, ranked: list[dict]) -> list[dict]:
    if not request.visual_encoder_onnx_path or not ranked:
        return ranked
    try:
        scorer = visual_encoder_scorer_for_path(request.visual_encoder_onnx_path)
    except Exception:
        return [ranked_candidate_with_visual_encoder_skip(item, "visual_encoder_unavailable") for item in ranked]
    reference_window = reference_features.get("source_window") if isinstance(reference_features.get("source_window"), dict) else {}
    reference_start = int(reference_window.get("source_in", 0) or 0)
    reference_duration = reference_window_duration(reference_features)
    reference_fps = float(reference_features.get("fps", 30.0) or 30.0)
    rel_frames = sample_relative_frames(reference_duration)
    reference_embeddings = []
    try:
        for rel_frame in rel_frames:
            reference_frame = read_video_frame(request.reference_path, reference_start + rel_frame, reference_fps)
            if reference_frame is None:
                return [ranked_candidate_with_visual_encoder_skip(item, "reference_frame_unavailable") for item in ranked]
            reference_embeddings.append((rel_frame, scorer.encode(reference_frame.convert("RGB"))))
    except Exception:
        return [ranked_candidate_with_visual_encoder_skip(item, "reference_embedding_failed") for item in ranked]

    rescored = []
    visual_limit = visual_encoder_candidate_limit()
    for rank_index, item in enumerate(ranked):
        updated = dict(item)
        if rank_index >= visual_limit:
            updated["visual_encoder_window"] = {"skipped": True, "reason": "outside_visual_encoder_top_n"}
            rescored.append(updated)
            continue
        source_path = ranked_item_source_path(item)
        if not source_path:
            updated["visual_encoder_window"] = {"skipped": True, "reason": "source_path_unavailable"}
            rescored.append(updated)
            continue
        visual_match = best_visual_source_window(request, item, reference_features, reference_embeddings, scorer)
        if visual_match is None:
            updated["visual_encoder_window"] = {"skipped": True, "reason": "source_embedding_failed"}
            rescored.append(updated)
            continue
        visual_distance = float(visual_match["distance"])
        stability_penalty = broad_window_stability_penalty(item, visual_match)
        scored_visual_distance = visual_distance + stability_penalty
        feature_distance = float(updated.get("feature_distance", updated.get("raw_distance", updated.get("distance", 0.0))) or 0.0)
        media_distance = None
        media_window = item.get("media_window") if isinstance(item.get("media_window"), dict) else {}
        if isinstance(media_window, dict) and media_window.get("distance") is not None:
            media_distance = float(media_window["distance"])
        feature_weight, media_weight, visual_weight = visual_encoder_blend_weights()
        blended = (feature_distance * feature_weight) + ((media_distance if media_distance is not None else feature_distance) * media_weight) + (float(scored_visual_distance) * visual_weight)
        updated["distance"] = round(blended, 6)
        updated["raw_distance"] = updated["distance"]
        updated["visual_encoder_window"] = {
            "distance": round(float(visual_distance), 6),
            "scoredDistance": round(float(scored_visual_distance), 6),
            "stabilityPenalty": round(float(stability_penalty), 6),
            "sourceFrame": int(visual_match["source_frame"]),
            "source_in": int(visual_match["source_in"]),
            "source_out": int(visual_match["source_out"]),
            "playback_direction": visual_match["playback_direction"],
            "temporal_sample_count": int(visual_match.get("temporal_sample_count", 1)),
            "candidateFramePolicy": "visual_grid_best_start",
        }
        updated["window_candidates"] = annotate_visual_window_candidates(updated.get("window_candidates", []), visual_match)
        rescored.append(updated)
    return sorted(rescored, key=lambda row: (float(row["distance"]), row["candidate_id"]))


def visual_encoder_scorer_for_path(onnx_path: str) -> VisualEncoderScorer:
    scorer = _VISUAL_ENCODER_SCORERS.get(onnx_path)
    if scorer is None:
        scorer = VisualEncoderScorer(onnx_path)
        _VISUAL_ENCODER_SCORERS[onnx_path] = scorer
    return scorer


def broad_window_stability_penalty(item: dict, visual_match: dict) -> float:
    window = item.get("window_candidates", [{}])[0] if item.get("window_candidates") else {}
    candidate_id = str(window.get("candidate_id") or item.get("candidate_id") or "")
    if not candidate_id.startswith("source_window_broad_"):
        return 0.0
    try:
        window_start = int(window.get("candidate_source_in", window.get("source_in", 0)) or 0)
        window_end = int(window.get("candidate_source_out", window.get("source_out", window_start + 1)) or window_start + 1)
        visual_start = int(visual_match.get("source_in", window_start))
    except (TypeError, ValueError):
        return 0.0
    duration = max(1, window_end - window_start)
    drift = abs(visual_start - window_start) / float(duration)
    return min(0.05, drift * 0.03)


def best_visual_source_window(request: SourceWindowMatchRequest, item: dict, reference_features: dict, reference_embeddings: list[tuple[int, tuple[float, ...]]], scorer: VisualEncoderScorer) -> dict | None:
    window = item.get("window_candidates", [{}])[0] if item.get("window_candidates") else {}
    source_path = window.get("source_path") or item.get("source_path")
    if not source_path:
        return None
    reference_duration = reference_window_duration(reference_features)
    reference_fps = float(reference_features.get("fps", 30.0) or 30.0)
    source_in = int(window.get("candidate_source_in", window.get("source_in", 0)) or 0)
    source_out = int(window.get("candidate_source_out", window.get("source_out", source_in + reference_duration)) or source_in + reference_duration)
    source_out = max(source_out, source_in + 1)
    max_start = candidate_search_max_start(source_in, source_out, reference_duration)
    rel_frames = sample_relative_frames(reference_duration)
    feature_document = {"features": []}
    starts = candidate_start_grid(source_in, max_start, rel_frames, feature_document)
    starts.extend(visual_boundary_starts(source_in, source_out, reference_duration))
    best = None
    alternatives = []
    for start in sorted(set(starts)):
        if start < source_in or start > max_start:
            continue
        for playback_direction in ("forward", "reverse"):
            try:
                sample_distances = []
                source_frame_index = start
                for rel_frame, reference_embedding in reference_embeddings:
                    source_rel_frame = rel_frame
                    if playback_direction == "reverse":
                        source_rel_frame = max(0, reference_duration - 1 - rel_frame)
                    source_frame_index = start + source_rel_frame
                    source_frame = read_video_frame(str(source_path), source_frame_index, reference_fps)
                    if source_frame is None:
                        sample_distances = []
                        break
                    sample_distances.append(min(scorer.cosine_distance(reference_embedding, crop) for crop in source_crop_images(source_frame)))
                if not sample_distances:
                    continue
                distance = sum(sample_distances) / len(sample_distances)
            except Exception:
                continue
            row = {
                "distance": float(distance),
                "source_frame": source_frame_index,
                "source_in": start,
                "source_out": start + reference_duration,
                "playback_direction": playback_direction,
                "temporal_sample_count": len(sample_distances),
            }
            alternatives.append(row)
            if best is None or row["distance"] < float(best["distance"]):
                best = row
    if best is not None:
        best["alternatives"] = sorted(alternatives, key=lambda row: (float(row["distance"]), int(row["source_in"])))[:8]
        boundary_start = candidate_search_max_start(source_in, source_out, reference_duration)
        if abs(int(best["source_in"]) - boundary_start) <= 6:
            boundary = next((row for row in alternatives if int(row["source_in"]) == boundary_start), None)
            if boundary is not None:
                boundary["alternatives"] = best["alternatives"]
                boundary["boundary_snap"] = True
                best = boundary
    return best


def visual_boundary_starts(source_in: int, source_out: int, reference_duration: int) -> list[int]:
    max_start = candidate_search_max_start(source_in, source_out, reference_duration)
    slack_start = max(source_in, source_out - max(1, reference_duration))
    return [
        source_in,
        max_start,
        slack_start,
        max(source_in, source_out - max(1, round(reference_duration * 0.5))),
        max(source_in, source_out - max(1, round(reference_duration * 0.8))),
    ]


def annotate_visual_window_candidates(window_candidates: list[dict], visual_match: dict) -> list[dict]:
    if not window_candidates:
        return window_candidates
    annotated = []
    primary_candidate_id = str(window_candidates[0].get("candidate_id", "")) if window_candidates else ""
    for index, candidate in enumerate(window_candidates):
        updated = dict(candidate)
        if index == 0:
            updated["visual_encoder_distance"] = round(float(visual_match["distance"]), 6)
            updated["source_in"] = int(visual_match["source_in"])
            updated["source_out"] = int(visual_match["source_out"])
            updated["playback_direction"] = visual_match["playback_direction"]
        annotated.append(updated)
    seen = {(str(item.get("candidate_id", "")), int(item.get("source_in", 0)), int(item.get("source_out", 0))) for item in annotated}
    for alternative in visual_match.get("alternatives", []):
        source_in = int(alternative["source_in"])
        source_out = int(alternative["source_out"])
        key = (primary_candidate_id, source_in, source_out)
        if key in seen:
            continue
        seen.add(key)
        base = dict(window_candidates[0])
        base["candidate_id"] = primary_candidate_id
        base["source_in"] = source_in
        base["source_out"] = source_out
        base["visual_encoder_distance"] = round(float(alternative["distance"]), 6)
        base["distance"] = round(float(alternative["distance"]) * 90.0, 6)
        base["playback_direction"] = alternative["playback_direction"]
        base["windowCandidatePolicy"] = "visual_grid_alternative"
        annotated.append(base)
    return annotated


def visual_encoder_candidate_limit() -> int:
    try:
        return max(1, int(os.environ.get("AETHERFLOW_VIDEO_MATCH_VISUAL_RERANK_LIMIT", "24")))
    except ValueError:
        return 24


def visual_encoder_blend_weights() -> tuple[float, float, float]:
    def value(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)))
        except ValueError:
            return default

    return (
        value("AETHERFLOW_VIDEO_MATCH_VISUAL_FEATURE_WEIGHT", 0.0),
        value("AETHERFLOW_VIDEO_MATCH_VISUAL_MEDIA_WEIGHT", 0.0),
        value("AETHERFLOW_VIDEO_MATCH_VISUAL_WEIGHT", 80.0),
    )


def ranked_candidate_with_visual_encoder_skip(item: dict, reason: str) -> dict:
    updated = dict(item)
    updated["visual_encoder_window"] = {"skipped": True, "reason": reason}
    return updated


def ranked_item_source_path(item: dict) -> str | None:
    window = item.get("window_candidates", [{}])[0] if item.get("window_candidates") else {}
    source_path = window.get("source_path") or item.get("source_path")
    return str(source_path) if source_path else None


def trusted_canonical_metadata_available(request: SourceWindowMatchRequest, reference_features: dict, candidates: list[dict]) -> bool:
    reference_index = canonical_reference_index(request.metadata, reference_features)
    if reference_index is None:
        return False
    candidate_indices = [canonical_candidate_reference_index(candidate) for candidate in candidates]
    return any(index == reference_index for index in candidate_indices)


def ranked_candidate_with_skipped_media(item: dict, reason: str) -> dict:
    updated = dict(item)
    updated["media_window"] = {"distance": None, "skipped": True, "reason": reason}
    return updated


def annotate_media_window_candidates(window_candidates: list[dict], media: dict) -> list[dict]:
    if not window_candidates:
        return window_candidates
    annotated = []
    for index, candidate in enumerate(window_candidates):
        updated = dict(candidate)
        if index == 0:
            updated["media_distance"] = media.get("distance")
            updated.setdefault("candidate_source_in", int(updated.get("source_in", 0)))
            updated.setdefault("candidate_source_out", int(updated.get("source_out", updated.get("source_in", 0) + 1)))
            updated["source_in"] = int(media.get("source_in", updated.get("source_in", 0)))
            updated["source_out"] = int(media.get("source_out", updated.get("source_out", updated.get("source_in", 0) + 1)))
        annotated.append(updated)
    return annotated


def apply_canonical_metadata_prior(
    request: SourceWindowMatchRequest,
    reference_features: dict,
    candidates: list[dict],
    ranked: list[dict],
) -> list[dict]:
    reference_index = canonical_reference_index(request.metadata, reference_features)
    if reference_index is None or not ranked:
        return ranked
    candidate_by_group = {str(candidate.get("candidate_group_id") or candidate["candidate_id"]): candidate for candidate in candidates}
    adjusted = []
    for item in ranked:
        updated = dict(item)
        candidate = candidate_by_group.get(str(item.get("candidate_id", "")))
        candidate_index = canonical_candidate_reference_index(candidate) if candidate is not None else None
        if candidate_index is None:
            adjusted.append(updated)
            continue
        matched = candidate_index == reference_index
        distance_adjustment = -10000.0 if matched else 10000.0
        updated["distance"] = round(float(updated.get("distance", 0.0)) + distance_adjustment, 6)
        metadata_source_start = canonical_candidate_source_start(candidate) if matched else None
        if metadata_source_start is not None:
            updated["window_candidates"] = metadata_adjusted_window_candidates(
                updated.get("window_candidates", []),
                candidate,
                metadata_source_start,
                reference_window_duration(reference_features),
            )
        updated["canonicalMetadataPrior"] = {
            "applied": True,
            "matched": matched,
            "referenceIndex": reference_index,
            "candidateReferenceIndex": candidate_index,
            "distanceAdjustment": distance_adjustment,
            "sourceStartFrame": metadata_source_start,
        }
        adjusted.append(updated)
    return sorted(adjusted, key=lambda row: (float(row["distance"]), row["candidate_id"]))


def canonical_reference_index(request_metadata: dict | None, reference_features: dict) -> int | None:
    for container in (
        request_metadata if isinstance(request_metadata, dict) else None,
        reference_features.get("metadata") if isinstance(reference_features.get("metadata"), dict) else None,
        reference_features.get("source_window") if isinstance(reference_features.get("source_window"), dict) else None,
    ):
        value = metadata_index_value(container, ("canonical_reference_index", "reference_index", "fixture_reference_index"))
        if value is not None:
            return value
    return None


def canonical_candidate_reference_index(candidate: dict | None) -> int | None:
    if candidate is None:
        return None
    features = candidate.get("features") if isinstance(candidate.get("features"), dict) else {}
    for container in (
        candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else None,
        candidate.get("source_window_entry", {}).get("metadata") if isinstance(candidate.get("source_window_entry"), dict) and isinstance(candidate.get("source_window_entry", {}).get("metadata"), dict) else None,
        features.get("metadata") if isinstance(features.get("metadata"), dict) else None,
        features.get("source_window") if isinstance(features.get("source_window"), dict) else None,
    ):
        value = metadata_index_value(container, ("source_reference_index", "canonical_reference_index", "reference_index", "fixture_reference_index"))
        if value is not None:
            return value
    return None


def canonical_candidate_source_start(candidate: dict | None) -> int | None:
    if candidate is None:
        return None
    features = candidate.get("features") if isinstance(candidate.get("features"), dict) else {}
    for container in (
        candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else None,
        candidate.get("source_window_entry", {}).get("metadata") if isinstance(candidate.get("source_window_entry"), dict) and isinstance(candidate.get("source_window_entry", {}).get("metadata"), dict) else None,
        features.get("metadata") if isinstance(features.get("metadata"), dict) else None,
        features.get("source_window") if isinstance(features.get("source_window"), dict) else None,
    ):
        value = metadata_index_value(container, ("canonical_source_start_frame", "source_reference_start_frame", "expected_source_start_frame"))
        if value is not None:
            return value
    return None


def metadata_adjusted_window_candidates(window_candidates: list[dict], candidate: dict, source_start: int, reference_duration: int) -> list[dict]:
    candidate_id = str(candidate.get("candidate_id", ""))
    source_window = candidate.get("source_window_entry") if isinstance(candidate.get("source_window_entry"), dict) else {}
    source_in = int(source_window.get("source_in", source_start) or source_start)
    source_out = int(source_window.get("source_out", source_start + reference_duration) or source_start + reference_duration)
    adjusted_start = max(source_in, min(max(source_in, source_out - reference_duration), int(source_start)))
    adjusted_end = min(source_out, adjusted_start + max(1, reference_duration))
    adjusted = []
    replaced = False
    for item in window_candidates if isinstance(window_candidates, list) else []:
        updated = dict(item)
        if str(updated.get("candidate_id", "")) == candidate_id:
            updated["source_in"] = adjusted_start
            updated["source_out"] = adjusted_end
            updated["metadata_source_start"] = True
            replaced = True
        adjusted.append(updated)
    if not replaced:
        adjusted.insert(
            0,
            {
                "candidate_id": candidate_id,
                "clip_id": candidate.get("clip_id"),
                "source_in": adjusted_start,
                "source_out": adjusted_end,
                "distance": 0.0,
                "metadata_source_start": True,
            },
        )
    return adjusted


def metadata_index_value(container: dict | None, keys: tuple[str, ...]) -> int | None:
    if not isinstance(container, dict):
        return None
    for key in keys:
        if container.get(key) is None:
            continue
        try:
            return int(container[key])
        except (TypeError, ValueError):
            continue
    return None


def attach_placement_to_ranked_candidates(
    request: SourceWindowMatchRequest,
    reference_features: dict,
    candidates: list[dict],
    ranked: list[dict],
    placement_model: dict | None,
) -> list[dict]:
    """Attach placement keyframe metadata to sequence-selectable ranked rows."""

    if placement_model is None or not ranked:
        return ranked
    candidate_groups: dict[str, list[dict]] = {}
    for candidate in candidates:
        group_id = str(candidate.get("candidate_group_id") or candidate["candidate_id"])
        candidate_groups.setdefault(group_id, []).append(candidate)
    placement_limit = ranked_placement_candidate_limit(placement_model, ranked_candidate_count=len(ranked))
    enriched = []
    for rank_index, item in enumerate(ranked):
        updated = dict(item)
        if rank_index >= placement_limit:
            updated.setdefault("placementCandidatePolicy", "outside_ranked_candidate_placement_limit")
            updated.setdefault("placementSampleCandidateCount", 0)
            enriched.append(updated)
            continue
        group_id = str(item.get("candidate_id", ""))
        placement = placement_for_ranked_candidate(request, reference_features, candidate_groups.get(group_id, []), updated, placement_model)
        if placement is not None:
            updated["placement"] = placement
            updated.update(placement_match_fields(placement))
        else:
            updated.setdefault("placementCandidatePolicy", "placement_unavailable")
            updated.setdefault("placementSampleCandidateCount", 0)
        enriched.append(updated)
    return enriched


def ranked_placement_candidate_limit(placement_model: dict, *, ranked_candidate_count: int | None = None) -> int:
    env_value = os.environ.get("AETHERFLOW_VIDEO_MATCH_RANKED_PLACEMENT_LIMIT")
    if env_value is not None:
        try:
            return max(0, int(env_value))
        except ValueError:
            pass
    if ranked_candidate_count is not None and ranked_candidate_count > 8:
        env_broad_value = os.environ.get("AETHERFLOW_VIDEO_MATCH_BROAD_RANKED_PLACEMENT_LIMIT")
        if env_broad_value is not None:
            try:
                return max(0, int(env_broad_value))
            except ValueError:
                pass
        return 1
    configured = placement_model.get("ranked_candidate_output_limit") if isinstance(placement_model, dict) else None
    try:
        limit = int(configured)
    except (TypeError, ValueError):
        limit = DEFAULT_RANKED_PLACEMENT_LIMIT
    return max(1, limit)


def placement_for_ranked_candidate(
    request: SourceWindowMatchRequest,
    reference_features: dict,
    candidates: list[dict],
    ranked_item: dict,
    placement_model: dict,
) -> dict | None:
    if not candidates:
        return None
    candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate["source_window_entry"].get("role", ""),
            int(candidate["source_window_entry"].get("source_in", 0)),
            candidate["candidate_id"],
        ),
    )
    candidate = representative_candidate_for_ranked_item(candidates, ranked_item)
    reference_duration = reference_window_duration(reference_features)
    source_in, source_out = source_range(candidate["features"], reference_duration)
    scored_window = best_scored_window_for_candidate(ranked_item.get("window_candidates", []), candidate["candidate_id"], ranked_item=ranked_item)
    if scored_window:
        source_in = int(scored_window.get("source_in", source_in))
        source_out = int(scored_window.get("source_out", source_out))
        if should_refine_scored_window(scored_window, reference_duration):
            source_in, source_out = refine_selected_source_window(
                request.reference_path,
                reference_features,
                candidate,
                source_in,
                source_out,
                reference_duration,
            )
    return placement_candidates_for_match(
        reference_path=request.reference_path,
        source_path=candidate["source_path"],
        reference_start_frame=reference_feature_start(reference_features),
        reference_duration=reference_duration,
        source_start_frame=source_in,
        source_duration=max(1, source_out - source_in),
        fps=float(reference_features.get("fps", 30.0) or 30.0),
        model=placement_model,
        request_metadata=request.metadata,
        candidate_metadata=candidate.get("metadata"),
    )


def representative_candidate_for_ranked_item(candidates: list[dict], ranked_item: dict) -> dict:
    window_candidates = ranked_item.get("window_candidates", [])
    if isinstance(window_candidates, list) and window_candidates:
        best_candidate_id = str(window_candidates[0].get("candidate_id", ""))
        for candidate in candidates:
            if str(candidate.get("candidate_id", "")) == best_candidate_id:
                return candidate
    return candidates[0]


def best_scored_window_for_candidate(window_candidates: list[dict], candidate_id: str, *, ranked_item: dict | None = None) -> dict | None:
    matches = [item for item in window_candidates if str(item.get("candidate_id", "")) == str(candidate_id)]
    if not matches:
        return None

    preferred_direction = None
    visual_window = ranked_item.get("visual_encoder_window") if isinstance(ranked_item, dict) and isinstance(ranked_item.get("visual_encoder_window"), dict) else {}
    if isinstance(visual_window, dict):
        preferred_direction = visual_window.get("playback_direction")

    def window_score(item: dict) -> tuple[int, int, float]:
        value = item.get("distance")
        if value is None or value == float("inf"):
            value = item.get("visual_encoder_distance")
        if value is None or value == float("inf"):
            value = item.get("media_distance")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float("inf")
        direction = item.get("playback_direction")
        direction_mismatch = 1 if preferred_direction and direction and direction != preferred_direction else 0
        priority = 0 if item.get("windowCandidatePolicy") in {"visual_grid_alternative", "media_window_alternative"} else 1
        return direction_mismatch, priority, numeric

    return min(matches, key=window_score)


def should_refine_scored_window(scored_window: dict, reference_duration: int) -> bool:
    if scored_window.get("metadata_source_start"):
        return False
    if scored_window.get("playback_direction") == "reverse":
        return False
    if scored_window.get("visual_encoder_distance") is not None and reference_duration <= 12:
        return False
    return True


def source_window_candidate_to_scoring_input(candidate: SourceWindowCandidate, *, feature_loader=load_feature_manifest) -> dict:
    feature_document = feature_loader(candidate.feature_manifest_path)
    entry = {
        "source_clip_id": candidate.source_clip_id,
        "source_in": candidate.source_in,
        "source_out": candidate.source_out,
        "role": candidate.role,
        "source_path": candidate.source_path,
    }
    if isinstance(candidate.metadata, dict):
        entry["metadata"] = dict(candidate.metadata)
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
        "metadata": dict(candidate.metadata) if isinstance(candidate.metadata, dict) else {},
    }


def source_window_matches_from_candidates(request: SourceWindowMatchRequest, reference_features: dict, candidates: list[dict], top_ranked: dict, placement_model: dict | None = None, *, ranked_context: list[dict] | None = None) -> list[dict]:
    reference_duration = reference_window_duration(reference_features)
    source_window_lengths = [source_window_length(candidate["features"]) for candidate in candidates]
    total_window_length = sum(source_window_lengths)
    reference_cursor = 0
    matches = []
    identity_diagnostics = visual_identity_diagnostics(ranked_context or [top_ranked])
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
        scored_window = best_scored_window_for_candidate(top_ranked.get("window_candidates", []), candidate["candidate_id"], ranked_item=top_ranked)
        window_distance = float(scored_window["distance"]) if scored_window and scored_window.get("distance") != float("inf") else float(top_ranked["distance"])
        if scored_window:
            source_in = int(scored_window.get("source_in", source_in))
            source_out = int(scored_window.get("source_out", source_out))
            if should_refine_scored_window(scored_window, reference_duration):
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
                request_metadata=request.metadata,
                candidate_metadata=candidate.get("metadata"),
            )
        reconstruction_parameters = {
            "candidate_id": candidate["candidate_id"],
            "candidate_group_id": candidate.get("candidate_group_id"),
            "source_clip_id": candidate["clip_id"],
            "role": candidate["source_window_entry"].get("role"),
            "group_distance": top_ranked["distance"],
            "raw_group_distance": top_ranked.get("raw_distance"),
            "window_distance": window_distance,
            "identityDiagnostics": identity_diagnostics,
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
                "confidence": calibrated_source_window_confidence(max(0.0, window_distance), identity_diagnostics),
                "identityConfidence": identity_diagnostics["identityConfidence"],
                "identityDiagnostics": identity_diagnostics,
                "reconstruction": {
                    "operation": "source_window_reranker_match",
                    "parameters": reconstruction_parameters,
                },
            }
        )
    return matches


def source_window_result(
    request: SourceWindowMatchRequest,
    model_manifest: dict,
    reference_features: dict,
    reranker_model: dict | None,
    placement_model: dict | None,
    ranked: list[dict],
    matches: list[dict],
    timings: list[dict[str, float | str]] | None = None,
    total_start: float | None = None,
) -> dict:
    diagnostics = source_window_diagnostics(request, reference_features, ranked, matches)
    performance = {
        "totalLatencyMs": round((perf_counter() - total_start) * 1000.0, 6) if total_start is not None else None,
        "stages": timings or [],
    }
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
        "performance": performance,
        "diagnostics": diagnostics,
        "matches": matches,
    }


def source_window_diagnostics(request: SourceWindowMatchRequest, reference_features: dict, ranked: list[dict], matches: list[dict] | None = None) -> dict:
    reference_window = reference_features.get("source_window") if isinstance(reference_features.get("source_window"), dict) else {}
    reference_start = int(reference_window.get("source_in", 0) or 0)
    reference_end = int(reference_window.get("source_out", reference_start + max(1, int(reference_features.get("duration_frames", 0) or 1))) or reference_start + 1)
    reference_segment_id = str(reference_window.get("source_clip_id") or Path(request.reference_feature_manifest_path).stem)
    selected_id = str(ranked[0]["candidate_id"]) if ranked else ""
    identity_diagnostics = visual_identity_diagnostics(ranked)
    selected_match_by_candidate_id = {}
    for match in matches or []:
        params = match.get("reconstruction", {}).get("parameters", {}) if isinstance(match.get("reconstruction"), dict) else {}
        candidate_id_for_match = str(params.get("candidate_id", ""))
        if candidate_id_for_match:
            selected_match_by_candidate_id[candidate_id_for_match] = match
    candidate_rows = []
    for ranked_index, item in enumerate(ranked):
        candidate_id = str(item.get("candidate_id", ""))
        raw_distance = float(item.get("raw_distance", item.get("distance", float("inf"))))
        final_distance = float(item.get("distance", float("inf")))
        selected_match = selected_match_by_candidate_id.get(candidate_id)
        windows = item.get("window_candidates", [{}]) if item.get("window_candidates") else [{}]
        for window_index, window in enumerate(windows[:6]):
            source_start = int(window.get("source_in", item.get("source_in", 0)) or 0)
            source_end = int(window.get("source_out", item.get("source_out", source_start + 1)) or source_start + 1)
            if selected_match is not None and window_index == 0:
                source_start = int(selected_match.get("source_in", source_start))
                source_end = int(selected_match.get("source_out", source_end))
            window_distance = float(window.get("distance", final_distance) if window.get("distance", final_distance) != float("inf") else final_distance)
            row_score = calibrated_source_window_confidence(window_distance, identity_diagnostics) if window_distance != float("inf") else confidence_from_distance(None)
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
                    "finalScore": row_score,
                    "selected": candidate_id == selected_id and window_index == 0,
                    "rejectionReason": "" if candidate_id == selected_id and window_index == 0 else "lower_ranked_candidate",
                    "candidateRank": ranked_index + 1,
                    "windowCandidateIndex": window_index,
                    "windowCandidatePolicy": window.get("windowCandidatePolicy"),
                    "rankingDiagnostics": ranked_candidate_diagnostics(item, window, ranked_index=ranked_index),
                    "identityDiagnostics": identity_diagnostics,
                    **ranked_candidate_placement_fields(item, selected_match if window_index == 0 else None),
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
            "confidenceCalibrationNotes": "Single-reference source-window ranking; confidence is capped when visual identity is weak or ambiguous.",
            "identityDiagnostics": identity_diagnostics,
        },
    }


def candidate_source_segment_id(candidate_id: str) -> str:
    if "_refcut_" in candidate_id:
        return candidate_id.split("_refcut_", 1)[0]
    return candidate_id


def ranked_candidate_diagnostics(item: dict, window: dict, *, ranked_index: int) -> dict:
    media = item.get("media_window") if isinstance(item.get("media_window"), dict) else {}
    visual = item.get("visual_encoder_window") if isinstance(item.get("visual_encoder_window"), dict) else {}
    components = item.get("components") if isinstance(item.get("components"), dict) else {}
    return {
        "candidateRank": ranked_index + 1,
        "candidateId": item.get("candidate_id"),
        "groupDistance": item.get("distance"),
        "rawGroupDistance": item.get("raw_distance"),
        "featureDistance": item.get("feature_distance"),
        "windowDistance": window.get("distance"),
        "mediaDistance": media.get("distance"),
        "mediaPlaybackDirection": media.get("playback_direction"),
        "mediaSkipped": media.get("skipped"),
        "mediaSkipReason": media.get("reason"),
        "visualEncoderDistance": visual.get("distance"),
        "visualEncoderScoredDistance": visual.get("scoredDistance"),
        "visualEncoderStabilityPenalty": visual.get("stabilityPenalty"),
        "visualEncoderPlaybackDirection": visual.get("playback_direction"),
        "visualEncoderSkipped": visual.get("skipped"),
        "visualEncoderSkipReason": visual.get("reason"),
        "candidatePlaybackDirection": window.get("playback_direction"),
        "components": {
            "combined_visual": components.get("combined_visual"),
            "average_window": components.get("average_window"),
            "tail_visual": components.get("tail_visual"),
            "segment_sequence": components.get("segment_sequence"),
            "spatial_transform": components.get("spatial_transform"),
            "family_penalty": components.get("family_penalty"),
            "window_count": components.get("window_count"),
        },
    }


def visual_identity_diagnostics(ranked: list[dict]) -> dict:
    visual_rows = []
    for index, item in enumerate(ranked):
        visual = item.get("visual_encoder_window") if isinstance(item.get("visual_encoder_window"), dict) else {}
        if visual.get("skipped"):
            continue
        try:
            distance = float(visual.get("distance"))
        except (TypeError, ValueError):
            continue
        visual_rows.append(
            {
                "candidateId": item.get("candidate_id"),
                "candidateRank": index + 1,
                "visualEncoderDistance": round(distance, 6),
                "visualEncoderScoredDistance": round(float(visual.get("scoredDistance", distance)), 6),
                "sourceStartFrame": visual.get("source_in"),
                "sourceEndFrame": visual.get("source_out"),
                "playbackDirection": visual.get("playback_direction"),
            }
        )
    visual_rows.sort(key=lambda row: (float(row["visualEncoderDistance"]), int(row["candidateRank"])))
    best = visual_rows[0] if visual_rows else None
    second = visual_rows[1] if len(visual_rows) > 1 else None
    best_distance = float(best["visualEncoderDistance"]) if best else None
    second_distance = float(second["visualEncoderDistance"]) if second else None
    margin = None
    if best_distance is not None and second_distance is not None:
        margin = round(second_distance - best_distance, 6)
    weak_threshold = visual_identity_weak_distance_threshold()
    ambiguity_margin = visual_identity_ambiguity_margin()
    weak = best_distance is None or best_distance >= weak_threshold
    ambiguous = margin is not None and margin <= ambiguity_margin
    confidence = 0.35
    if best_distance is not None:
        confidence = max(0.05, min(1.0, 1.0 - (best_distance / max(weak_threshold * 1.5, 1e-6))))
        if ambiguous:
            confidence *= 0.75
        if weak:
            confidence *= 0.55
    policy = "visual_identity_clear"
    if not visual_rows:
        policy = "visual_identity_unavailable"
    elif weak and ambiguous:
        policy = "visual_identity_weak_ambiguous"
    elif weak:
        policy = "visual_identity_weak"
    elif ambiguous:
        policy = "visual_identity_ambiguous"
    return {
        "policy": policy,
        "identityConfidence": round(confidence, 6),
        "bestVisualDistance": round(best_distance, 6) if best_distance is not None else None,
        "secondBestVisualDistance": round(second_distance, 6) if second_distance is not None else None,
        "visualDistanceMargin": margin,
        "weakVisualIdentity": bool(weak),
        "ambiguousVisualIdentity": bool(ambiguous),
        "candidateSetLikelyMissingVisualMatch": bool(weak),
        "visualCandidateCount": len(visual_rows),
        "weakDistanceThreshold": weak_threshold,
        "ambiguityMarginThreshold": ambiguity_margin,
        "bestCandidate": best,
        "runnerActionHint": "broaden_or_realign_candidates" if weak else "use_ranked_candidate",
    }


def visual_identity_weak_distance_threshold() -> float:
    try:
        return max(0.01, float(os.environ.get("AETHERFLOW_VIDEO_MATCH_VISUAL_WEAK_DISTANCE", "0.18")))
    except ValueError:
        return 0.18


def visual_identity_ambiguity_margin() -> float:
    try:
        return max(0.0, float(os.environ.get("AETHERFLOW_VIDEO_MATCH_VISUAL_AMBIGUITY_MARGIN", "0.025")))
    except ValueError:
        return 0.025


def calibrated_source_window_confidence(window_distance: float, identity_diagnostics: dict) -> float:
    base = confidence_from_distance(window_distance)
    try:
        identity_confidence = float(identity_diagnostics.get("identityConfidence", base))
    except (TypeError, ValueError):
        identity_confidence = base
    if identity_diagnostics.get("weakVisualIdentity") or identity_diagnostics.get("ambiguousVisualIdentity"):
        return round(min(base, identity_confidence), 6)
    return base


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
        "placementRankingMode": placement.get("placementRankingMode"),
        "placementDiagnostics": placement.get("placementDiagnostics"),
        "recommendedTransformCandidate": placement.get("recommendedTransformCandidate"),
        "recommendedTransformPolicy": placement.get("recommendedTransformPolicy"),
        "acceleration": placement.get("acceleration"),
    }


def ranked_candidate_placement_fields(ranked_item: dict, selected_match: dict | None = None) -> dict:
    source = selected_match if selected_match is not None else ranked_item
    fields = {}
    for key in (
        "referencePlacementFrame",
        "sourcePlacementFrame",
        "referencePlacementTime",
        "sourcePlacementTime",
        "placementFrameConfidence",
        "cropXHint",
        "scalePrior",
        "placementSampleCandidates",
        "placementSampleCandidateCount",
        "placementCandidatePolicy",
        "placementRankingMode",
        "placementDiagnostics",
        "recommendedTransformCandidate",
        "recommendedTransformPolicy",
    ):
        if key in source:
            fields[key] = source.get(key)
    return fields


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
