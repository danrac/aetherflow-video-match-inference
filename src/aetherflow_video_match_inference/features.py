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
    reference_motion = average_motion(reference_features)
    source_motion = average_motion(source_features)
    if reference_motion is None or source_motion is None:
        return color_component
    return color_component + abs(reference_motion - source_motion)


def average_mean_rgb(feature_document: dict[str, Any]) -> tuple[float, float, float] | None:
    frames = feature_document.get("features", [])
    vectors = [frame.get("mean_rgb") for frame in frames if isinstance(frame, dict) and frame.get("mean_rgb")]
    if not vectors:
        return None
    return tuple(sum(float(vector[index]) for vector in vectors) / len(vectors) for index in range(3))


def average_motion(feature_document: dict[str, Any]) -> float | None:
    frames = feature_document.get("features", [])
    values = [
        float(frame["mean_absdiff_from_previous"])
        for frame in frames
        if isinstance(frame, dict) and frame.get("mean_absdiff_from_previous") is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def confidence_from_distance(distance: float | None) -> float:
    if distance is None:
        return 0.5
    normalized = min(distance / 441.67295593, 1.0)
    return round(1.0 - normalized, 6)
