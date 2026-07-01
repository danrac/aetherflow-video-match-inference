"""Placement-keyframe candidate scoring for source-window matches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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
]

DEFAULT_SAMPLE_FRACTIONS = [0.08, 0.16, 0.25, 0.38, 0.5, 0.62, 0.75, 0.88]
DEFAULT_CROP_X_FACTORS = [0.0, 0.15, 0.25, 0.5, 0.75, 0.85, 1.0]
COMPARE_SIZE = (160, 284)


def load_placement_model(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        model = json.load(handle)
    if not isinstance(model, dict):
        raise ValueError(f"Expected placement model object at {path}")
    return model


def placement_model_summary(model: dict[str, Any] | None) -> dict[str, Any] | None:
    if model is None:
        return None
    return {
        "model_id": model.get("model_id"),
        "model_version": model.get("model_version"),
        "model_type": model.get("model_type"),
    }


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
) -> dict[str, Any] | None:
    if reference_duration <= 2 or source_duration <= 2:
        return None
    fractions = model.get("sample_fractions") if isinstance(model, dict) else None
    if not isinstance(fractions, list) or not fractions:
        fractions = DEFAULT_SAMPLE_FRACTIONS
    if top_k is None:
        configured_top_k = model.get("output_top_k") if isinstance(model, dict) else None
        top_k = int(configured_top_k) if configured_top_k else len(fractions)
    candidates = []
    duration = max(1, min(reference_duration, source_duration))
    for fraction in fractions:
        fraction = float(fraction)
        offset = min(max(2, round(duration * fraction)), max(2, duration - 2))
        reference_frame = int(reference_start_frame + offset)
        source_frame = int(source_start_frame + offset)
        reference_image = read_video_frame(reference_path, reference_frame)
        source_image = read_video_frame(source_path, source_frame)
        if reference_image is None or source_image is None:
            continue
        features, hints = pair_features(reference_image, source_image, fraction, duration)
        confidence = predict_confidence(model, features) if model is not None else heuristic_confidence(features)
        candidates.append(
            {
                "referencePlacementFrame": reference_frame,
                "sourcePlacementFrame": source_frame,
                "referencePlacementTime": round(reference_frame / fps, 6),
                "sourcePlacementTime": round(source_frame / fps, 6),
                "confidence": round(confidence, 6),
                "fraction": round(fraction, 6),
                "cropXHint": hints["cropXHint"],
                "scalePrior": hints["scalePrior"],
                "scoreComponents": {key: round(float(value), 6) for key, value in features.items() if key != "bias"},
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-float(item["confidence"]), int(item["referencePlacementFrame"]), int(item["sourcePlacementFrame"])))
    selected = candidates[0]
    retained_candidates = candidates[: max(1, min(int(top_k), len(candidates)))]
    return {
        "referencePlacementFrame": selected["referencePlacementFrame"],
        "sourcePlacementFrame": selected["sourcePlacementFrame"],
        "referencePlacementTime": selected["referencePlacementTime"],
        "sourcePlacementTime": selected["sourcePlacementTime"],
        "placementFrameConfidence": selected["confidence"],
        "cropXHint": selected["cropXHint"],
        "scalePrior": selected["scalePrior"],
        "placementSampleCandidates": retained_candidates,
        "placementSampleCandidateCount": len(retained_candidates),
        "placementCandidatePolicy": "ranked_fraction_samples",
        "placementModel": placement_model_summary(model),
    }


def pair_features(reference_image: Any, source_image: Any, fraction: float, duration: int) -> tuple[dict[str, float], dict[str, float]]:
    direct_score = frame_score(reference_image, resize_to_reference(source_image, reference_image))
    crop_x, best_crop_score = best_crop_score_for_pair(reference_image, source_image)
    reference_edge = edge_density(reference_image)
    source_edge = edge_density(source_image)
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
    }
    return features, {"cropXHint": float(crop_x), "scalePrior": 1.0}


def predict_confidence(model: dict[str, Any], features: dict[str, float]) -> float:
    weights = model.get("weights", [])
    if not isinstance(weights, list):
        return heuristic_confidence(features)
    value = 0.0
    for index, name in enumerate(FEATURE_NAMES[: len(weights)]):
        value += float(weights[index]) * float(features.get(name, 0.0))
    return sigmoid(value)


def heuristic_confidence(features: dict[str, float]) -> float:
    return max(0.0, min(1.0, (float(features["direct_score"]) * 0.35) + (float(features["best_crop_score"]) * 0.55) + (float(features["reference_edge_density"]) * 0.10)))


def best_crop_score_for_pair(reference_frame: Any, source_frame: Any) -> tuple[float, float]:
    best_x = 0.5
    best_score = -1.0
    for crop_x in DEFAULT_CROP_X_FACTORS:
        score = frame_score(reference_frame, crop_9x16(source_frame, crop_x))
        if score > best_score:
            best_x = crop_x
            best_score = score
    return best_x, best_score


def crop_9x16(frame: Any, crop_x: float) -> Any:
    height, width = frame.shape[:2]
    crop_width = min(width, max(1, round(height * 9.0 / 16.0)))
    left = max(0, min(width - crop_width, round((width - crop_width) * crop_x)))
    return frame[:, left : left + crop_width]


def frame_score(reference_frame: Any, candidate_frame: Any) -> float:
    reference = normalize_frame(reference_frame)
    candidate = normalize_frame(candidate_frame)
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


def read_video_frame(path: str, frame_index: int) -> Any | None:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-max(-50.0, min(50.0, value)))))
