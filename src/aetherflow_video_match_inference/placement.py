"""Placement-keyframe candidate scoring for source-window matches."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .onnx_runtime import linear_onnx_score


FEATURE_NAMES = [
    "bias",
    "fraction",
    "direct_score",
    "best_crop_score",
    "best_crop_x",
    "reference_edge_density",
    "source_edge_density",
    "edge_density_delta",
    "source_start_error_abs",
    "near_boundary",
    "duration_ratio",
    "best_projection_score",
    "best_projection_scale",
    "best_projection_x",
    "best_projection_y",
    "feature_affine_inlier_ratio",
    "feature_affine_match_count",
    "crop_ambiguity_score",
    "geometry_stability_score",
]

DEFAULT_SAMPLE_FRACTIONS = [0.08, 0.16, 0.25, 0.38, 0.5, 0.62, 0.75, 0.88]
DEFAULT_CROP_X_FACTORS = [0.0, 0.15, 0.25, 0.5, 0.75, 0.85, 1.0]
COMPARE_SIZE = (160, 284)


def load_placement_model(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    model_path = Path(path)
    with model_path.open("r", encoding="utf-8") as handle:
        model = json.load(handle)
    if not isinstance(model, dict):
        raise ValueError(f"Expected placement model object at {path}")
    model["_model_path"] = str(model_path)
    return model


def placement_model_summary(model: dict[str, Any] | None) -> dict[str, Any] | None:
    if model is None:
        return None
    summary = {
        "model_id": model.get("model_id"),
        "model_version": model.get("model_version"),
        "model_type": model.get("model_type"),
    }
    runtime = model.get("_last_onnx_runtime") or model.get("onnx_runtime")
    if isinstance(runtime, dict):
        summary["onnx_runtime"] = runtime
    if model.get("_last_onnx_runtime_error"):
        summary["onnx_runtime_error"] = model.get("_last_onnx_runtime_error")
    return summary


def placement_sample_offsets(fractions: list[Any], duration: int) -> list[tuple[float, int]]:
    if duration <= 4:
        offsets = list(range(1, max(2, duration)))
        return [(round(offset / max(1, duration), 6), offset) for offset in offsets]

    seen: set[int] = set()
    samples: list[tuple[float, int]] = []
    min_offset = 2
    max_offset = max(min_offset, duration - 2)
    for raw_fraction in fractions:
        fraction = float(raw_fraction)
        offset = min(max(min_offset, round(duration * fraction)), max_offset)
        if offset in seen:
            continue
        seen.add(offset)
        samples.append((fraction, offset))
    if duration <= 12:
        for offset in range(min_offset, max_offset + 1):
            if offset not in seen:
                seen.add(offset)
                samples.append((round(offset / max(1, duration), 6), offset))
    return samples


def placement_candidates_for_match(
    *,
    reference_path: str,
    source_path: str,
    reference_start_frame: int,
    reference_duration: int,
    source_start_frame: int,
    source_duration: int,
    fps: float,
    model: dict[str, Any] | None,
    top_k: int | None = None,
    request_metadata: dict[str, Any] | None = None,
    candidate_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if reference_duration <= 2 or source_duration <= 2:
        return None
    recommended_transform = recommended_transform_candidate(
        reference_path=reference_path,
        source_path=source_path,
        request_metadata=request_metadata,
        candidate_metadata=candidate_metadata,
    )
    if recommended_transform is not None and use_recommended_transform_fast_path(model, request_metadata, candidate_metadata):
        return recommended_transform_fast_placement(
            recommended_transform=recommended_transform,
            reference_start_frame=reference_start_frame,
            reference_duration=reference_duration,
            source_start_frame=source_start_frame,
            source_duration=source_duration,
            fps=fps,
            model=model,
        )
    fractions = model.get("sample_fractions") if isinstance(model, dict) else None
    if not isinstance(fractions, list) or not fractions:
        fractions = DEFAULT_SAMPLE_FRACTIONS
    if top_k is None:
        configured_top_k = model.get("output_top_k") if isinstance(model, dict) else None
        top_k = int(configured_top_k) if configured_top_k else len(fractions)
    candidates = []
    duration = max(1, min(reference_duration, source_duration))
    sample_offsets = placement_sample_offsets(fractions, duration)
    for fraction, offset in sample_offsets:
        reference_frame = int(reference_start_frame + offset)
        source_frame = int(source_start_frame + offset)
        reference_image = read_video_frame(reference_path, reference_frame)
        source_image = read_video_frame(source_path, source_frame)
        if reference_image is None or source_image is None:
            continue
        features, hints = pair_features(reference_image, source_image, fraction, duration)
        confidence = predict_confidence(model, features) if model is not None else heuristic_confidence(features)
        ranking_score = placement_ranking_score(confidence, features, duration, model)
        diagnostics = placement_diagnostics(features, hints, duration)
        candidates.append(
            {
                "referencePlacementFrame": reference_frame,
                "sourcePlacementFrame": source_frame,
                "referencePlacementTime": round(reference_frame / fps, 6),
                "sourcePlacementTime": round(source_frame / fps, 6),
                "confidence": round(confidence, 6),
                "placementRankingScore": round(ranking_score, 6),
                "fraction": round(fraction, 6),
                "cropXHint": hints["cropXHint"],
                "scalePrior": hints["scalePrior"],
                **diagnostics,
                "scoreComponents": {key: round(float(value), 6) for key, value in features.items() if key != "bias"},
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-float(item.get("placementRankingScore", item["confidence"])), int(item["referencePlacementFrame"]), int(item["sourcePlacementFrame"])))
    selected = candidates[0]
    retained_candidates = candidates[: max(1, min(int(top_k), len(candidates)))]
    placement = {
        "referencePlacementFrame": selected["referencePlacementFrame"],
        "sourcePlacementFrame": selected["sourcePlacementFrame"],
        "referencePlacementTime": selected["referencePlacementTime"],
        "sourcePlacementTime": selected["sourcePlacementTime"],
        "placementFrameConfidence": selected["confidence"],
        "cropXHint": selected["cropXHint"],
        "scalePrior": selected["scalePrior"],
        "placementSampleCandidates": retained_candidates,
        "placementSampleCandidateCount": len(retained_candidates),
        "placementCandidatePolicy": "geometry_stable_ranked_fraction_samples",
        "placementRankingMode": "geometry_stability_v1",
        "placementDiagnostics": {
            "cropOnlyAmbiguous": selected.get("cropOnlyAmbiguous"),
            "cropOnlyAmbiguityScore": selected.get("cropOnlyAmbiguityScore"),
            "geometryStabilityScore": selected.get("geometryStabilityScore"),
            "transformTypeConfidence": selected.get("transformTypeConfidence"),
            "xPlacementConfidence": selected.get("xPlacementConfidence"),
            "yPlacementConfidence": selected.get("yPlacementConfidence"),
            "scalePriorConfidence": selected.get("scalePriorConfidence"),
        },
        "placementModel": placement_model_summary(model),
    }
    if recommended_transform is not None:
        placement["recommendedTransformCandidate"] = recommended_transform
        placement["recommendedTransformPolicy"] = recommended_transform["policy"]
    return placement


def use_recommended_transform_fast_path(
    model: dict[str, Any] | None,
    request_metadata: dict[str, Any] | None,
    candidate_metadata: dict[str, Any] | None,
) -> bool:
    if isinstance(model, dict) and model.get("disable_recommended_transform_fast_path"):
        return False
    if isinstance(model, dict) and model.get("recommended_transform_fast_path") is True:
        return True
    metadata = merged_transform_metadata(request_metadata, candidate_metadata)
    explicit = first_mapping(metadata, ("canonical_placement_transform", "placement_transform", "recommended_transform"))
    if explicit is not None and float(explicit.get("confidence", 1.0) or 1.0) >= 0.98:
        return True
    return first_mapping(metadata, ("canonical_editor_transform", "editor_transform", "transform")) is not None and bool(metadata.get("canonical_reference_index") is not None or metadata.get("source_reference_index") is not None)


def recommended_transform_fast_placement(
    *,
    recommended_transform: dict[str, Any],
    reference_start_frame: int,
    reference_duration: int,
    source_start_frame: int,
    source_duration: int,
    fps: float,
    model: dict[str, Any] | None,
) -> dict[str, Any]:
    duration = max(1, min(reference_duration, source_duration))
    fractions = model.get("sample_fractions") if isinstance(model, dict) else None
    if not isinstance(fractions, list) or not fractions:
        fractions = [0.5]
    sample_offsets = placement_sample_offsets(fractions, duration)[: max(1, min(4, len(fractions)))]
    candidates = []
    for fraction, offset in sample_offsets:
        reference_frame = int(reference_start_frame + offset)
        source_frame = int(source_start_frame + offset)
        candidates.append(
            {
                "referencePlacementFrame": reference_frame,
                "sourcePlacementFrame": source_frame,
                "referencePlacementTime": round(reference_frame / fps, 6),
                "sourcePlacementTime": round(source_frame / fps, 6),
                "confidence": 1.0,
                "placementRankingScore": 1.0,
                "fraction": round(float(fraction), 6),
                "cropXHint": None,
                "scalePrior": None,
                "estimatedCropDirection": "model_recommended_transform",
                "cropOnlyAmbiguous": False,
                "cropOnlyAmbiguityScore": 0.0,
                "geometryStabilityScore": 1.0,
                "transformTypeConfidence": {
                    "featureAffine": 0.0,
                    "projection": 0.0,
                    "cropOnly": 0.0,
                    "modelRecommendedTransform": 1.0,
                },
                "scalePriorConfidence": 1.0,
                "xPlacementConfidence": 1.0,
                "yPlacementConfidence": 1.0,
                "shortSegmentPlacement": bool(duration <= 12),
                "scoreComponents": {},
            }
        )
    selected = candidates[0]
    return {
        "referencePlacementFrame": selected["referencePlacementFrame"],
        "sourcePlacementFrame": selected["sourcePlacementFrame"],
        "referencePlacementTime": selected["referencePlacementTime"],
        "sourcePlacementTime": selected["sourcePlacementTime"],
        "placementFrameConfidence": 1.0,
        "cropXHint": selected["cropXHint"],
        "scalePrior": selected["scalePrior"],
        "placementSampleCandidates": candidates,
        "placementSampleCandidateCount": len(candidates),
        "placementCandidatePolicy": "model_recommended_transform_fast_path",
        "placementRankingMode": "recommended_transform_fast_path_v1",
        "placementDiagnostics": {
            "cropOnlyAmbiguous": False,
            "cropOnlyAmbiguityScore": 0.0,
            "geometryStabilityScore": 1.0,
            "transformTypeConfidence": selected["transformTypeConfidence"],
            "xPlacementConfidence": 1.0,
            "yPlacementConfidence": 1.0,
            "scalePriorConfidence": 1.0,
            "accelerated": True,
            "cpuOpenCvRegistrationSkipped": True,
        },
        "placementModel": placement_model_summary(model),
        "recommendedTransformCandidate": recommended_transform,
        "recommendedTransformPolicy": recommended_transform["policy"],
        "acceleration": {
            "stage": "placement_candidate_generation",
            "mode": "model_recommended_transform_fast_path",
            "cpuOpenCvRegistrationSkipped": True,
        },
    }


def recommended_transform_candidate(
    *,
    reference_path: str,
    source_path: str,
    request_metadata: dict[str, Any] | None,
    candidate_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    metadata = merged_transform_metadata(request_metadata, candidate_metadata)
    explicit = first_mapping(metadata, ("canonical_placement_transform", "placement_transform", "recommended_transform"))
    if explicit is not None:
        position = explicit.get("position")
        scale = explicit.get("scale")
        if valid_position(position) and scale is not None:
            return {
                "method": "metadata_recommended_transform",
                "policy": "canonical_metadata_transform",
                "position": [round(float(position[0]), 6), round(float(position[1]), 6)],
                "scale": round(float(scale), 6),
                "confidence": float(explicit.get("confidence", 1.0) or 1.0),
                "source": "metadata.placement_transform",
            }

    editor_transform = first_mapping(metadata, ("canonical_editor_transform", "editor_transform", "transform"))
    if editor_transform is None:
        return None
    target_width, target_height = target_size_from_metadata_or_video(metadata, reference_path)
    source_width, source_height = source_size_from_metadata_or_video(metadata, source_path)
    if target_width <= 0 or target_height <= 0 or source_height <= 0:
        return None
    scale_factor = float(editor_transform.get("scale_factor", 1.0) or 1.0)
    x_offset_percent = float(editor_transform.get("x_offset_percent", 0.0) or 0.0)
    y_offset_percent = float(editor_transform.get("y_offset_percent", 0.0) or 0.0)
    base_scale = float(metadata.get("base_scale", 0.0) or 0.0)
    if base_scale <= 0:
        base_scale = (target_height / source_height) * 100.0
    position_x = (target_width / 2.0) + ((x_offset_percent / 100.0) * target_width)
    position_y = (target_height / 2.0) + ((y_offset_percent / 100.0) * target_height)
    crop_basis = crop_basis_correction(metadata, target_width, target_height, source_width, scale_factor, base_scale)
    position_x += crop_basis["offset_x"]
    return {
        "method": "metadata_editor_transform",
        "policy": "canonical_metadata_transform",
        "position": [round(position_x, 6), round(position_y, 6)],
        "scale": round(base_scale * scale_factor, 6),
        "confidence": 1.0,
        "source": "metadata.editor_transform",
        "baseScale": round(base_scale, 6),
        "scaleFactor": round(scale_factor, 6),
        "offsetPercent": [round(x_offset_percent, 6), round(y_offset_percent, 6)],
        "targetSize": [int(round(target_width)), int(round(target_height))],
        "sourceSize": [int(round(source_width)), int(round(source_height))],
        "cropBasisCorrection": crop_basis,
    }


def merged_transform_metadata(request_metadata: dict[str, Any] | None, candidate_metadata: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(request_metadata, dict):
        merged.update(request_metadata)
    if isinstance(candidate_metadata, dict):
        merged.update(candidate_metadata)
    return merged


def first_mapping(metadata: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any] | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, dict):
            return value
    return None


def valid_position(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) >= 2 and value[0] is not None and value[1] is not None


def target_size_from_metadata_or_video(metadata: dict[str, Any], reference_path: str) -> tuple[float, float]:
    size = first_size(metadata, ("target_size", "output_size", "vertical_size", "comp_size", "reference_size"))
    if size is not None:
        return size
    return video_size(reference_path)


def source_size_from_metadata_or_video(metadata: dict[str, Any], source_path: str) -> tuple[float, float]:
    size = first_size(metadata, ("source_size", "full_frame_size"))
    if size is not None:
        return size
    width = metadata.get("source_width")
    height = metadata.get("source_height")
    if width is not None and height is not None:
        return float(width), float(height)
    return video_size(source_path)


def content_source_size_from_metadata(metadata: dict[str, Any]) -> tuple[float, float] | None:
    size = first_size(metadata, ("content_source_size", "original_source_size", "source_content_size"))
    if size is not None:
        return size
    width = metadata.get("content_source_width", metadata.get("original_source_width"))
    height = metadata.get("content_source_height", metadata.get("original_source_height"))
    if width is not None and height is not None:
        return float(width), float(height)
    if metadata.get("vertical_crop_x_factor") is not None and metadata.get("source_width") is not None and metadata.get("source_height") is not None:
        return float(metadata["source_width"]), float(metadata["source_height"])
    return None


def crop_basis_correction(metadata: dict[str, Any], target_width: float, target_height: float, layer_source_width: float, scale_factor: float, base_scale: float) -> dict[str, Any]:
    crop_factor = metadata.get("vertical_crop_x_factor", metadata.get("crop_x_factor"))
    content_size = content_source_size_from_metadata(metadata)
    if crop_factor is None or content_size is None:
        return {"applied": False, "offset_x": 0.0}
    content_width, content_height = content_size
    if content_width <= 0 or content_height <= 0 or layer_source_width <= 0:
        return {"applied": False, "offset_x": 0.0}
    target_aspect = target_width / max(1.0, target_height)
    content_aspect = content_width / max(1.0, content_height)
    if content_aspect <= target_aspect:
        return {"applied": False, "offset_x": 0.0, "reason": "content_not_wider_than_target"}
    crop_width = content_height * target_aspect
    crop_left = (content_width - crop_width) * float(crop_factor)
    crop_center_content = crop_left + (crop_width / 2.0)
    crop_center_layer = (crop_center_content / content_width) * layer_source_width
    ae_scale = (base_scale * scale_factor) / 100.0
    offset_x = ((layer_source_width / 2.0) - crop_center_layer) * ae_scale
    return {
        "applied": True,
        "offset_x": round(offset_x, 6),
        "verticalCropXFactor": round(float(crop_factor), 6),
        "contentSourceSize": [int(round(content_width)), int(round(content_height))],
        "cropCenterLayerX": round(crop_center_layer, 6),
        "reason": "full_layer_position_adjusted_to_match_vertical_crop_center",
    }


def first_size(metadata: dict[str, Any], keys: tuple[str, ...]) -> tuple[float, float] | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, dict) and value.get("width") is not None and value.get("height") is not None:
            return float(value["width"]), float(value["height"])
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return float(value[0]), float(value[1])
    return None


def video_size(path: str) -> tuple[float, float]:
    capture = cv2.VideoCapture(str(path))
    width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
    height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
    capture.release()
    return width, height


def pair_features(reference_image: Any, source_image: Any, fraction: float, duration: int) -> tuple[dict[str, float], dict[str, float]]:
    direct_score = frame_score(reference_image, resize_to_reference(source_image, reference_image))
    crop_x, best_crop_score, crop_scores = best_crop_score_for_pair(reference_image, source_image)
    projection = best_projection_score_for_pair(reference_image, source_image)
    affine = feature_affine_score_for_pair(reference_image, source_image)
    reference_edge = edge_density(reference_image)
    source_edge = edge_density(source_image)
    crop_ambiguity = crop_ambiguity_score(crop_scores)
    geometry_stability = geometry_stability_score(projection, affine, reference_edge, source_edge, crop_ambiguity)
    features = {
        "bias": 1.0,
        "fraction": float(fraction),
        "direct_score": direct_score,
        "best_crop_score": best_crop_score,
        "best_crop_x": float(crop_x),
        "reference_edge_density": reference_edge,
        "source_edge_density": source_edge,
        "edge_density_delta": abs(reference_edge - source_edge),
        "source_start_error_abs": 0.0,
        "near_boundary": 1.0 if fraction <= 0.12 or fraction >= 0.88 else 0.0,
        "duration_ratio": min(4.0, duration / 150.0),
        "best_projection_score": projection["score"],
        "best_projection_scale": projection["scale_factor"],
        "best_projection_x": projection["x_offset"],
        "best_projection_y": projection["y_offset"],
        "feature_affine_inlier_ratio": affine["inlier_ratio"],
        "feature_affine_match_count": min(80.0, float(affine["match_count"])) / 80.0,
        "crop_ambiguity_score": crop_ambiguity,
        "geometry_stability_score": geometry_stability,
    }
    return features, {"cropXHint": float(crop_x), "scalePrior": 1.0, "projectionHint": projection, "cropScores": crop_scores, "affineHint": affine}


def predict_confidence(model: dict[str, Any], features: dict[str, float]) -> float:
    weights = model.get("weights", [])
    if not isinstance(weights, list):
        return heuristic_confidence(features)
    feature_names = model.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        feature_names = FEATURE_NAMES
    means = model.get("feature_mean")
    stds = model.get("feature_std")
    values = placement_feature_values(features, feature_names, weights, means, stds)
    onnx_result = linear_onnx_score(model, values)
    if onnx_result is not None:
        model["_last_onnx_runtime"] = {
            "active": True,
            "onnx_path": onnx_result["onnx_path"],
            "requested_provider": onnx_result["requested_provider"],
            "session_providers": onnx_result["session_providers"],
        }
        value = float(onnx_result["score"])
    else:
        model["_last_onnx_runtime"] = {
            "active": False,
            "fallback": "json_linear_weights",
        }
        value = sum(float(weights[index]) * values[index] for index in range(min(len(weights), len(values))))
    confidence = sigmoid(value)
    confidence -= float(model.get("boundary_penalty", 0.0) or 0.0) * float(features.get("near_boundary", 0.0) or 0.0)
    return max(0.0, min(1.0, confidence))


def placement_feature_values(features: dict[str, float], feature_names: list[Any], weights: list[Any], means: Any, stds: Any) -> list[float]:
    values: list[float] = []
    for index, name in enumerate(feature_names[: len(weights)]):
        feature_value = float(features.get(str(name), 0.0))
        if isinstance(means, list) and isinstance(stds, list):
            if index == 0:
                feature_value = 1.0
            else:
                std = float(stds[index]) if index < len(stds) and float(stds[index]) != 0.0 else 1.0
                mean = float(means[index]) if index < len(means) else 0.0
                feature_value = (feature_value - mean) / std
        values.append(feature_value)
    return values


def heuristic_confidence(features: dict[str, float]) -> float:
    return max(0.0, min(1.0, (float(features["direct_score"]) * 0.35) + (float(features["best_crop_score"]) * 0.55) + (float(features["reference_edge_density"]) * 0.10)))


def placement_ranking_score(confidence: float, features: dict[str, float], duration: int, model: dict[str, Any] | None) -> float:
    geometry_weight = float(model.get("geometry_stability_weight", 0.22) if isinstance(model, dict) else 0.22)
    crop_penalty = float(model.get("crop_only_ambiguity_penalty", 0.18) if isinstance(model, dict) else 0.18)
    short_weight = float(model.get("short_segment_geometry_weight", 0.18) if isinstance(model, dict) else 0.18)
    geometry = float(features.get("geometry_stability_score", 0.0) or 0.0)
    crop_ambiguity = float(features.get("crop_ambiguity_score", 0.0) or 0.0)
    projection = float(features.get("best_projection_score", 0.0) or 0.0)
    direct = float(features.get("direct_score", 0.0) or 0.0)
    crop = float(features.get("best_crop_score", 0.0) or 0.0)
    crop_only = max(0.0, crop - max(projection, direct))
    score = float(confidence)
    score += geometry_weight * geometry
    score -= crop_penalty * max(crop_ambiguity, crop_only)
    if duration <= 12:
        score += short_weight * max(geometry, projection)
        score -= 0.08 * float(features.get("near_boundary", 0.0) or 0.0)
    return max(0.0, min(1.0, score))


def placement_diagnostics(features: dict[str, float], hints: dict[str, Any], duration: int) -> dict[str, Any]:
    projection = hints.get("projectionHint") if isinstance(hints.get("projectionHint"), dict) else {}
    affine = hints.get("affineHint") if isinstance(hints.get("affineHint"), dict) else {}
    crop_scores = hints.get("cropScores") if isinstance(hints.get("cropScores"), list) else []
    projection_score = float(features.get("best_projection_score", 0.0) or 0.0)
    crop_score = float(features.get("best_crop_score", 0.0) or 0.0)
    direct_score = float(features.get("direct_score", 0.0) or 0.0)
    geometry = float(features.get("geometry_stability_score", 0.0) or 0.0)
    crop_ambiguity = float(features.get("crop_ambiguity_score", 0.0) or 0.0)
    affine_ratio = float(features.get("feature_affine_inlier_ratio", 0.0) or 0.0)
    affine_match_norm = float(features.get("feature_affine_match_count", 0.0) or 0.0)
    crop_only_score = max(0.0, crop_score - max(projection_score, direct_score, affine_ratio))
    crop_x = float(hints.get("cropXHint", 0.5) or 0.0)
    if crop_x <= 0.25:
        crop_direction = "left"
    elif crop_x >= 0.75:
        crop_direction = "right"
    else:
        crop_direction = "center"
    affine_confidence = min(1.0, (affine_ratio * 0.7) + (affine_match_norm * 0.3))
    projection_confidence = min(1.0, projection_score * 1.5)
    crop_confidence = max(0.0, min(1.0, crop_score - (crop_ambiguity * 0.35) - (crop_only_score * 0.25)))
    stable_confidence = max(projection_confidence, affine_confidence, geometry)
    return {
        "estimatedCropDirection": crop_direction,
        "cropOnlyAmbiguous": bool(crop_ambiguity >= 0.65 or crop_only_score >= 0.12),
        "cropOnlyAmbiguityScore": round(max(crop_ambiguity, crop_only_score), 6),
        "geometryStabilityScore": round(geometry, 6),
        "transformTypeConfidence": {
            "featureAffine": round(affine_confidence, 6),
            "projection": round(projection_confidence, 6),
            "cropOnly": round(crop_confidence, 6),
        },
        "scalePriorConfidence": round(stable_confidence, 6),
        "xPlacementConfidence": round(stable_confidence * (1.0 - min(0.85, crop_ambiguity * 0.65)), 6),
        "yPlacementConfidence": round(max(projection_confidence, affine_confidence, direct_score), 6),
        "shortSegmentPlacement": bool(duration <= 12),
        "projectionHint": {
            "score": round(float(projection.get("score", 0.0) or 0.0), 6),
            "scaleFactor": round(float(projection.get("scale_factor", 1.0) or 1.0), 6),
            "xOffset": round(float(projection.get("x_offset", 0.0) or 0.0), 6),
            "yOffset": round(float(projection.get("y_offset", 0.0) or 0.0), 6),
        },
        "featureAffineHint": {
            "matchCount": int(affine.get("match_count", 0) or 0),
            "inlierCount": int(affine.get("inlier_count", 0) or 0),
            "inlierRatio": round(float(affine.get("inlier_ratio", 0.0) or 0.0), 6),
            "scaleFactor": round(float(affine.get("scale_factor", 1.0) or 1.0), 6),
            "xOffset": round(float(affine.get("x_offset", 0.0) or 0.0), 6),
            "yOffset": round(float(affine.get("y_offset", 0.0) or 0.0), 6),
        },
        "cropScoreSpread": [
            {"cropX": round(float(item["crop_x"]), 6), "score": round(float(item["score"]), 6)}
            for item in crop_scores
        ],
    }


def best_crop_score_for_pair(reference_frame: Any, source_frame: Any) -> tuple[float, float, list[dict[str, float]]]:
    best_x = 0.5
    best_score = -1.0
    scores: list[dict[str, float]] = []
    for crop_x in DEFAULT_CROP_X_FACTORS:
        score = frame_score(reference_frame, crop_9x16(source_frame, crop_x))
        scores.append({"crop_x": float(crop_x), "score": float(score)})
        if score > best_score:
            best_x = crop_x
            best_score = score
    return best_x, best_score, scores


def crop_ambiguity_score(crop_scores: list[dict[str, float]]) -> float:
    if len(crop_scores) < 2:
        return 0.0
    ranked = sorted((float(item["score"]) for item in crop_scores), reverse=True)
    best = ranked[0]
    second = ranked[1]
    if best <= 1e-6:
        return 1.0
    margin = max(0.0, best - second)
    return round(max(0.0, min(1.0, 1.0 - (margin / max(0.12, best)))), 6)


def geometry_stability_score(projection: dict[str, float], affine: dict[str, float], reference_edge: float, source_edge: float, crop_ambiguity: float) -> float:
    projection_score = float(projection.get("score", 0.0) or 0.0)
    affine_score = float(affine.get("inlier_ratio", 0.0) or 0.0)
    affine_match_norm = min(1.0, float(affine.get("match_count", 0.0) or 0.0) / 80.0)
    edge_score = max(0.0, min(1.0, (float(reference_edge) + float(source_edge)) * 18.0))
    score = (projection_score * 0.45) + (affine_score * 0.30) + (affine_match_norm * 0.15) + (edge_score * 0.10)
    score *= 1.0 - min(0.6, float(crop_ambiguity) * 0.35)
    return round(max(0.0, min(1.0, score)), 6)


def best_projection_score_for_pair(reference_frame: Any, source_frame: Any) -> dict[str, float]:
    reference = normalize_frame(reference_frame)
    source = normalize_frame(source_frame)
    ref_height, ref_width = reference.shape[:2]
    src_height, src_width = source.shape[:2]
    best = {"score": -1.0, "scale_factor": 1.0, "x_offset": 0.0, "y_offset": 0.0}
    for scale_factor in (0.65, 0.75, 0.9, 1.0, 1.12, 1.25, 1.4):
        scale = (ref_height / max(1, src_height)) * scale_factor
        scaled_width = max(1, round(src_width * scale))
        scaled_height = max(1, round(src_height * scale))
        scaled = cv2.resize(source, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
        for x_offset in (-0.22, -0.12, 0.0, 0.12, 0.22):
            for y_offset in (-0.22, -0.12, 0.0, 0.12, 0.22):
                canvas = np.zeros_like(reference)
                center_x = (ref_width / 2.0) + (x_offset * ref_width)
                center_y = (ref_height / 2.0) + (y_offset * ref_height)
                left = round(center_x - (scaled_width / 2.0))
                top = round(center_y - (scaled_height / 2.0))
                dst_left = max(0, left)
                dst_top = max(0, top)
                dst_right = min(ref_width, left + scaled_width)
                dst_bottom = min(ref_height, top + scaled_height)
                if dst_right <= dst_left or dst_bottom <= dst_top:
                    continue
                src_left = dst_left - left
                src_top = dst_top - top
                src_right = src_left + (dst_right - dst_left)
                src_bottom = src_top + (dst_bottom - dst_top)
                canvas[dst_top:dst_bottom, dst_left:dst_right] = scaled[src_top:src_bottom, src_left:src_right]
                score = normalized_frame_score(reference, canvas)
                if score > best["score"]:
                    best = {
                        "score": score,
                        "scale_factor": float(scale_factor),
                        "x_offset": float(x_offset),
                        "y_offset": float(y_offset),
                    }
    return best


def feature_affine_score_for_pair(reference_frame: Any, source_frame: Any) -> dict[str, float]:
    reference = cv2.resize(reference_frame, COMPARE_SIZE, interpolation=cv2.INTER_AREA)
    source = cv2.resize(source_frame, COMPARE_SIZE, interpolation=cv2.INTER_AREA)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    try:
        detector = cv2.ORB_create(nfeatures=180, fastThreshold=12)
        ref_keypoints, ref_descriptors = detector.detectAndCompute(reference_gray, None)
        src_keypoints, src_descriptors = detector.detectAndCompute(source_gray, None)
    except Exception:
        return empty_affine_hint()
    if ref_descriptors is None or src_descriptors is None or len(ref_keypoints) < 6 or len(src_keypoints) < 6:
        return empty_affine_hint()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = sorted(matcher.match(ref_descriptors, src_descriptors), key=lambda match: match.distance)
    except Exception:
        return empty_affine_hint()
    if len(matches) < 6:
        return empty_affine_hint(match_count=len(matches))
    retained = matches[: min(80, len(matches))]
    ref_points = np.float32([ref_keypoints[match.queryIdx].pt for match in retained]).reshape(-1, 1, 2)
    src_points = np.float32([src_keypoints[match.trainIdx].pt for match in retained]).reshape(-1, 1, 2)
    try:
        matrix, inliers = cv2.estimateAffinePartial2D(src_points, ref_points, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=1000)
    except Exception:
        return empty_affine_hint(match_count=len(retained))
    if matrix is None or inliers is None:
        return empty_affine_hint(match_count=len(retained))
    inlier_count = int(np.sum(inliers > 0))
    inlier_ratio = inlier_count / max(1, len(retained))
    scale_x = float(np.sqrt((matrix[0, 0] ** 2) + (matrix[0, 1] ** 2)))
    scale_y = float(np.sqrt((matrix[1, 0] ** 2) + (matrix[1, 1] ** 2)))
    scale_factor = (scale_x + scale_y) / 2.0
    return {
        "match_count": int(len(retained)),
        "inlier_count": inlier_count,
        "inlier_ratio": round(float(inlier_ratio), 6),
        "scale_factor": round(float(scale_factor), 6),
        "x_offset": round(float(matrix[0, 2]) / max(1, COMPARE_SIZE[0]), 6),
        "y_offset": round(float(matrix[1, 2]) / max(1, COMPARE_SIZE[1]), 6),
    }


def empty_affine_hint(match_count: int = 0) -> dict[str, float]:
    return {
        "match_count": int(match_count),
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "scale_factor": 1.0,
        "x_offset": 0.0,
        "y_offset": 0.0,
    }


def crop_9x16(frame: Any, crop_x: float) -> Any:
    height, width = frame.shape[:2]
    crop_width = min(width, max(1, round(height * 9.0 / 16.0)))
    left = max(0, min(width - crop_width, round((width - crop_width) * crop_x)))
    return frame[:, left : left + crop_width]


def frame_score(reference_frame: Any, candidate_frame: Any) -> float:
    reference = normalize_frame(reference_frame)
    candidate = normalize_frame(candidate_frame)
    return normalized_frame_score(reference, candidate)


def normalized_frame_score(reference: Any, candidate: Any) -> float:
    mse = float(np.mean((reference - candidate) ** 2))
    ref_flat = reference.reshape(-1)
    cand_flat = candidate.reshape(-1)
    if float(np.std(ref_flat)) <= 1e-6 or float(np.std(cand_flat)) <= 1e-6:
        corr_score = 0.0
    else:
        corr = float(np.corrcoef(ref_flat, cand_flat)[0, 1])
        corr_score = max(0.0, min(1.0, (corr + 1.0) / 2.0))
    return round(corr_score * max(0.0, 1.0 - min(1.0, mse * 8.0)), 6)


def edge_density(frame: Any) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return round(float(np.mean(edges > 0)), 6)


def normalize_frame(frame: Any) -> Any:
    return cv2.resize(frame, COMPARE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def resize_to_reference(frame: Any, reference_frame: Any) -> Any:
    height, width = reference_frame.shape[:2]
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=1024)
def read_video_frame(path: str, frame_index: int) -> Any | None:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-max(-50.0, min(50.0, value)))))
