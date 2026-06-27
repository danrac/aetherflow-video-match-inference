"""Feature-manifest loading and lightweight matching helpers."""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Any


def load_feature_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Expected feature manifest object at {path}")
    return document


def color_distance(reference_features: dict[str, Any], source_features: dict[str, Any]) -> float | None:
    return visual_distance(reference_features, source_features)


def visual_distance(reference_features: dict[str, Any], source_features: dict[str, Any]) -> float | None:
    reference_vector = average_mean_rgb(reference_features)
    source_vector = average_mean_rgb(source_features)
    if reference_vector is None or source_vector is None:
        return None
    color_component = sqrt(sum((reference_vector[index] - source_vector[index]) ** 2 for index in range(3)))
    return round(
        color_component
        + scalar_delta(reference_features, source_features, "mean_absdiff_from_previous", weight=1.0)
        + scalar_delta(reference_features, source_features, "mean_luma", weight=0.25)
        + scalar_delta(reference_features, source_features, "edge_density", weight=50.0)
        + scalar_delta(reference_features, source_features, "scene_change_score", weight=1.0)
        + optical_flow_delta(reference_features, source_features),
        6,
    )


def average_mean_rgb(feature_document: dict[str, Any]) -> tuple[float, float, float] | None:
    frames = feature_document.get("features", [])
    vectors = [frame.get("mean_rgb") for frame in frames if isinstance(frame, dict) and frame.get("mean_rgb")]
    if not vectors:
        return None
    return tuple(sum(float(vector[index]) for vector in vectors) / len(vectors) for index in range(3))


def average_motion(feature_document: dict[str, Any]) -> float | None:
    return average_scalar(feature_document, "mean_absdiff_from_previous")


def average_scalar(feature_document: dict[str, Any], field: str) -> float | None:
    frames = feature_document.get("features", [])
    values = [
        float(frame[field])
        for frame in frames
        if isinstance(frame, dict) and frame.get(field) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def scalar_delta(reference_features: dict[str, Any], source_features: dict[str, Any], field: str, weight: float) -> float:
    reference_value = average_scalar(reference_features, field)
    source_value = average_scalar(source_features, field)
    if reference_value is None or source_value is None:
        return 0.0
    return abs(reference_value - source_value) * weight


def optical_flow_delta(reference_features: dict[str, Any], source_features: dict[str, Any]) -> float:
    return (
        scalar_delta_from_flow(reference_features, source_features, "mean_magnitude", weight=5.0)
        + scalar_delta_from_flow(reference_features, source_features, "mean_dx", weight=2.0)
        + scalar_delta_from_flow(reference_features, source_features, "mean_dy", weight=2.0)
    )


def scalar_delta_from_flow(reference_features: dict[str, Any], source_features: dict[str, Any], field: str, weight: float) -> float:
    reference_value = average_optical_flow_scalar(reference_features, field)
    source_value = average_optical_flow_scalar(source_features, field)
    if reference_value is None or source_value is None:
        return 0.0
    return abs(reference_value - source_value) * weight


def average_optical_flow_scalar(feature_document: dict[str, Any], field: str) -> float | None:
    frames = feature_document.get("features", [])
    values = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        flow = frame.get("optical_flow")
        if not isinstance(flow, dict) or flow.get(field) is None:
            continue
        values.append(float(flow[field]))
    if not values:
        return None
    return sum(values) / len(values)


def confidence_from_distance(distance: float | None) -> float:
    if distance is None:
        return 0.5
    normalized = min(distance / 441.67295593, 1.0)
    return round(1.0 - normalized, 6)
