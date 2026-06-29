"""Feature-manifest loading and lightweight matching helpers."""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Any

FEATURE_CACHE_KEY = "_aetherflow_feature_cache"


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
        + grid_delta(reference_features, source_features, weight=0.35)
        + histogram_delta(reference_features, source_features, "active_hue_histogram", weight=80.0)
        + temporal_signature_delta(reference_features, source_features, weight=120.0)
        + brightest_frame_delta(reference_features, source_features, weight=0.4)
        + scalar_delta(reference_features, source_features, "mean_absdiff_from_previous", weight=1.0)
        + scalar_delta(reference_features, source_features, "mean_luma", weight=0.25)
        + scalar_delta(reference_features, source_features, "edge_density", weight=50.0)
        + scalar_delta(reference_features, source_features, "scene_change_score", weight=1.0)
        + optical_flow_delta(reference_features, source_features)
        + motion_track_summary_delta(reference_features, source_features),
        6,
    )


def feature_cache(feature_document: dict[str, Any]) -> dict[str, Any]:
    cache = feature_document.get(FEATURE_CACHE_KEY)
    if not isinstance(cache, dict):
        cache = {}
        feature_document[FEATURE_CACHE_KEY] = cache
    return cache


def average_mean_rgb(feature_document: dict[str, Any]) -> tuple[float, float, float] | None:
    cache = feature_cache(feature_document)
    if "average_mean_rgb" in cache:
        return cache["average_mean_rgb"]
    frames = feature_document.get("features", [])
    vectors = [
        frame.get("active_mean_rgb") or frame.get("mean_rgb")
        for frame in frames
        if isinstance(frame, dict) and (frame.get("active_mean_rgb") or frame.get("mean_rgb"))
    ]
    if not vectors:
        cache["average_mean_rgb"] = None
        return None
    cache["average_mean_rgb"] = tuple(sum(float(vector[index]) for vector in vectors) / len(vectors) for index in range(3))
    return cache["average_mean_rgb"]


def grid_delta(reference_features: dict[str, Any], source_features: dict[str, Any], weight: float) -> float:
    reference_grid = average_grid_mean_rgb(reference_features)
    source_grid = average_grid_mean_rgb(source_features)
    if reference_grid is None or source_grid is None or len(reference_grid) != len(source_grid):
        return 0.0
    distances = [
        sqrt(sum((reference_cell[index] - source_cell[index]) ** 2 for index in range(3)))
        for reference_cell, source_cell in zip(reference_grid, source_grid, strict=False)
    ]
    return average(distances) * weight


def average_grid_mean_rgb(feature_document: dict[str, Any]) -> list[tuple[float, float, float]] | None:
    cache = feature_cache(feature_document)
    if "average_grid_mean_rgb" in cache:
        return cache["average_grid_mean_rgb"]
    frames = feature_document.get("features", [])
    grids = [frame.get("grid_mean_rgb_5x5") or frame.get("grid_mean_rgb") for frame in frames if isinstance(frame, dict) and (frame.get("grid_mean_rgb_5x5") or frame.get("grid_mean_rgb"))]
    if not grids:
        cache["average_grid_mean_rgb"] = None
        return None
    cell_count = len(grids[0])
    if any(len(grid) != cell_count for grid in grids):
        cache["average_grid_mean_rgb"] = None
        return None
    averaged = []
    for cell_index in range(cell_count):
        averaged.append(tuple(sum(float(grid[cell_index][channel]) for grid in grids) / len(grids) for channel in range(3)))
    cache["average_grid_mean_rgb"] = averaged
    return cache["average_grid_mean_rgb"]


def histogram_delta(reference_features: dict[str, Any], source_features: dict[str, Any], field: str, weight: float) -> float:
    reference_histogram = average_histogram(reference_features, field)
    source_histogram = average_histogram(source_features, field)
    if reference_histogram is None or source_histogram is None or len(reference_histogram) != len(source_histogram):
        return 0.0
    return sum(abs(reference_histogram[index] - source_histogram[index]) for index in range(len(reference_histogram))) * weight


def average_histogram(feature_document: dict[str, Any], field: str) -> list[float] | None:
    cache = feature_cache(feature_document)
    cache_key = f"average_histogram:{field}"
    if cache_key in cache:
        return cache[cache_key]
    frames = feature_document.get("features", [])
    histograms = [frame.get(field) for frame in frames if isinstance(frame, dict) and frame.get(field)]
    if not histograms:
        cache[cache_key] = None
        return None
    bin_count = len(histograms[0])
    if any(len(histogram) != bin_count for histogram in histograms):
        cache[cache_key] = None
        return None
    cache[cache_key] = [sum(float(histogram[index]) for histogram in histograms) / len(histograms) for index in range(bin_count)]
    return cache[cache_key]


def temporal_signature_delta(reference_features: dict[str, Any], source_features: dict[str, Any], weight: float) -> float:
    reference_signature = temporal_signature(reference_features)
    source_signature = temporal_signature(source_features)
    if reference_signature is None or source_signature is None:
        return 0.0
    return min(
        temporal_signature_distance(reference_signature, source_signature),
        temporal_signature_distance(reference_signature, list(reversed(source_signature))),
    ) * weight


def temporal_signature_distance(reference_signature: list[list[float]], source_signature: list[list[float]]) -> float:
    rows = min(len(reference_signature), len(source_signature))
    if rows == 0:
        return 0.0
    distances = []
    for row_index in range(rows):
        reference_row = reference_signature[row_index]
        source_row = source_signature[row_index]
        columns = min(len(reference_row), len(source_row))
        if columns == 0:
            continue
        distances.append(sum(abs(reference_row[column] - source_row[column]) for column in range(columns)) / columns)
    return average(distances)


def temporal_signature(feature_document: dict[str, Any]) -> list[list[float]] | None:
    cache = feature_cache(feature_document)
    if "temporal_signature" in cache:
        return cache["temporal_signature"]
    signature = feature_document.get("temporal_signature")
    if isinstance(signature, list) and signature and all(isinstance(row, list) for row in signature):
        cache["temporal_signature"] = [[float(value) for value in row] for row in signature]
        return cache["temporal_signature"]
    frames = [frame for frame in feature_document.get("features", []) if isinstance(frame, dict)]
    if not frames:
        cache["temporal_signature"] = None
        return None
    cache["temporal_signature"] = [temporal_signature_row(frame) for frame in frames]
    return cache["temporal_signature"]


def temporal_signature_row(frame: dict[str, Any]) -> list[float]:
    active_rgb = frame.get("active_mean_rgb") or frame.get("mean_rgb") or [0.0, 0.0, 0.0]
    flow = frame.get("optical_flow") if isinstance(frame.get("optical_flow"), dict) else {}
    normalized_magnitude = flow.get("mean_magnitude_normalized")
    if normalized_magnitude is None:
        normalized_magnitude = float(flow.get("mean_magnitude") or 0.0) / 255.0
    return [
        float(frame.get("mean_luma") or 0.0) / 255.0,
        float(frame.get("std_luma") or 0.0) / 128.0,
        float(frame.get("edge_density") or 0.0),
        float(frame.get("mean_absdiff_from_previous") or 0.0) / 255.0,
        float(normalized_magnitude or 0.0),
        float(active_rgb[0]) / 255.0,
        float(active_rgb[1]) / 255.0,
        float(active_rgb[2]) / 255.0,
    ]


def brightest_frame_delta(reference_features: dict[str, Any], source_features: dict[str, Any], weight: float) -> float:
    reference_vector = brightest_frame_mean_rgb(reference_features)
    source_vector = brightest_frame_mean_rgb(source_features)
    if reference_vector is None or source_vector is None:
        return 0.0
    return sqrt(sum((reference_vector[index] - source_vector[index]) ** 2 for index in range(3))) * weight


def brightest_frame_mean_rgb(feature_document: dict[str, Any]) -> tuple[float, float, float] | None:
    cache = feature_cache(feature_document)
    if "brightest_frame_mean_rgb" in cache:
        return cache["brightest_frame_mean_rgb"]
    frames = [frame for frame in feature_document.get("features", []) if isinstance(frame, dict) and (frame.get("active_mean_rgb") or frame.get("mean_rgb"))]
    if not frames:
        cache["brightest_frame_mean_rgb"] = None
        return None
    brightest = max(frames, key=lambda frame: float(frame.get("mean_luma") or 0.0))
    vector = brightest.get("active_mean_rgb") or brightest.get("mean_rgb")
    cache["brightest_frame_mean_rgb"] = tuple(float(value) for value in vector)
    return cache["brightest_frame_mean_rgb"]


def average_motion(feature_document: dict[str, Any]) -> float | None:
    return average_scalar(feature_document, "mean_absdiff_from_previous")


def average_scalar(feature_document: dict[str, Any], field: str) -> float | None:
    cache = feature_cache(feature_document)
    cache_key = f"average_scalar:{field}"
    if cache_key in cache:
        return cache[cache_key]
    frames = feature_document.get("features", [])
    values = [
        float(frame[field])
        for frame in frames
        if isinstance(frame, dict) and frame.get(field) is not None
    ]
    if not values:
        cache[cache_key] = None
        return None
    cache[cache_key] = sum(values) / len(values)
    return cache[cache_key]


def scalar_delta(reference_features: dict[str, Any], source_features: dict[str, Any], field: str, weight: float) -> float:
    reference_value = average_scalar(reference_features, field)
    source_value = average_scalar(source_features, field)
    if reference_value is None or source_value is None:
        return 0.0
    return abs(reference_value - source_value) * weight


def optical_flow_delta(reference_features: dict[str, Any], source_features: dict[str, Any]) -> float:
    if has_normalized_optical_flow(reference_features) and has_normalized_optical_flow(source_features):
        return (
            scalar_delta_from_flow(reference_features, source_features, "mean_magnitude_normalized", weight=120.0)
            + scalar_delta_from_flow(reference_features, source_features, "mean_dx_normalized", weight=40.0)
            + scalar_delta_from_flow(reference_features, source_features, "mean_dy_normalized", weight=40.0)
            + scalar_delta_from_flow(reference_features, source_features, "tracked_point_ratio", weight=20.0)
            + scalar_delta_from_flow(reference_features, source_features, "motion_consistency", weight=20.0)
        )
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
    if field in {"mean_dx", "mean_dy", "mean_dx_normalized", "mean_dy_normalized"}:
        return min(abs(reference_value - source_value), abs(reference_value + source_value)) * weight
    return abs(reference_value - source_value) * weight


def has_normalized_optical_flow(feature_document: dict[str, Any]) -> bool:
    return average_optical_flow_scalar(feature_document, "mean_magnitude_normalized") is not None


def average_optical_flow_scalar(feature_document: dict[str, Any], field: str) -> float | None:
    cache = feature_cache(feature_document)
    cache_key = f"average_optical_flow_scalar:{field}"
    if cache_key in cache:
        return cache[cache_key]
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
        cache[cache_key] = None
        return None
    cache[cache_key] = sum(values) / len(values)
    return cache[cache_key]


def motion_track_summary_delta(reference_features: dict[str, Any], source_features: dict[str, Any]) -> float:
    reference_summary = reference_features.get("motion_track_summary")
    source_summary = source_features.get("motion_track_summary")
    if not isinstance(reference_summary, dict) or not isinstance(source_summary, dict):
        return 0.0
    return (
        summary_scalar_delta(reference_summary, source_summary, "mean_magnitude_normalized", weight=120.0)
        + summary_scalar_delta(reference_summary, source_summary, "max_magnitude_normalized", weight=60.0)
        + summary_scalar_delta(reference_summary, source_summary, "mean_tracked_point_ratio", weight=20.0)
        + summary_scalar_delta(reference_summary, source_summary, "mean_motion_consistency", weight=20.0)
        + summary_axis_delta(reference_summary, source_summary, weight=10.0)
    )


def summary_scalar_delta(reference_summary: dict[str, Any], source_summary: dict[str, Any], field: str, weight: float) -> float:
    if reference_summary.get(field) is None or source_summary.get(field) is None:
        return 0.0
    return abs(float(reference_summary[field]) - float(source_summary[field])) * weight


def summary_axis_delta(reference_summary: dict[str, Any], source_summary: dict[str, Any], weight: float) -> float:
    reference_axis = reference_summary.get("dominant_axis")
    source_axis = source_summary.get("dominant_axis")
    if not reference_axis or not source_axis or reference_axis == "none" or source_axis == "none":
        return 0.0
    return 0.0 if reference_axis == source_axis else weight


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def confidence_from_distance(distance: float | None) -> float:
    if distance is None:
        return 0.5
    normalized = min(distance / 441.67295593, 1.0)
    return round(1.0 - normalized, 6)
