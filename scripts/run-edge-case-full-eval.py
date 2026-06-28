#!/usr/bin/env python3
"""Evaluate inference feature scoring across every sample in an edge-case fixture."""

from __future__ import annotations

from collections import defaultdict
import json
from math import sqrt
import os
from pathlib import Path
import sys

from aetherflow_video_match_inference.features import color_distance, load_feature_manifest, temporal_signature_row


DATA_ROOT = Path(os.environ.get("AETHERFLOW_VIDEO_MATCH_DATA_ROOT", "/Volumes/FrameFusion/AetherFlow_VideoMatcherData"))
DEFAULT_DATASET = DATA_ROOT / "datasets" / "video-match-edge-case-fixture" / "v0003" / "dataset_manifest.json"
DEFAULT_OUTPUT = DATA_ROOT / "inference" / "video-match-edge-case-full-eval" / "v0003-hist" / "report.json"


def main(argv: list[str]) -> int:
    dataset_manifest = Path(argv[1]) if len(argv) > 1 else DEFAULT_DATASET
    output_path = Path(argv[2]) if len(argv) > 2 else DEFAULT_OUTPUT
    root = dataset_manifest.parent
    dataset = load_json(dataset_manifest)
    candidate_features = load_source_window_candidates(root, dataset)
    results = []

    for sample in dataset.get("samples", []):
        reference_feature_manifest = sample.get("reference_feature_manifest")
        if not reference_feature_manifest:
            continue
        reference_features = load_feature_manifest(root / reference_feature_manifest)
        ground_truth = load_json(root / sample["ground_truth_path"])
        transform_types = transform_types_from_ground_truth(ground_truth)
        expected_clip_ids = source_clip_ids_from_ground_truth(ground_truth)
        ranked = rank_candidates(reference_features, candidate_features, transform_types)
        expected_ranks = [index + 1 for index, candidate in enumerate(ranked) if ranked_item_matches_expected(candidate, expected_clip_ids)]
        if not expected_ranks:
            continue
        best_rank = min(expected_ranks)
        results.append(
            {
                "sample_id": sample["sample_id"],
                "transform_types": sorted(transform_types),
                "expected_clip_ids": sorted(expected_clip_ids),
                "best_expected_rank": best_rank,
                "top_candidate_id": ranked[0]["candidate_id"],
                "top_clip_id": ranked[0]["clip_id"],
                "top_candidate_clip_ids": ranked[0].get("clip_ids", []),
                "top_candidate_window_count": ranked[0].get("window_count", 1),
                "top_distance": ranked[0]["distance"],
                "top1_correct": best_rank == 1,
                "top5_correct": best_rank <= 5,
                "top10_correct": best_rank <= 10,
            }
        )

    report = {
        "dataset_manifest": str(dataset_manifest),
        "candidate_count": len(candidate_features),
        "metrics": metrics(results),
        "breakdown_by_transform": breakdown_by_transform(results),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_path)
    return 0


def load_source_window_candidates(root: Path, dataset: dict) -> list[dict]:
    candidates = []
    for sample in dataset.get("samples", []):
        for index, entry in enumerate(sample.get("source_window_feature_manifests", [])):
            feature_document = load_feature_manifest(root / entry["path"])
            candidates.append(
                {
                    "candidate_id": f"{sample['sample_id']}:{index}:{entry['source_clip_id']}:{entry['source_in']}-{entry['source_out']}",
                    "candidate_group_id": str(sample["sample_id"]),
                    "clip_id": str(entry["source_clip_id"]),
                    "source_window_entry": entry,
                    "features": feature_document,
                }
            )
    return candidates


def rank_candidates(reference_features: dict, candidates: list[dict], reference_transform_types: set[str] | None = None) -> list[dict]:
    if any(candidate.get("candidate_group_id") for candidate in candidates):
        return rank_candidate_groups(reference_features, candidates, reference_transform_types or set())
    ranked = []
    for candidate in candidates:
        distance = color_distance(reference_features, candidate["features"])
        ranked.append(
            {
                "candidate_id": candidate["candidate_id"],
                "clip_id": candidate["clip_id"],
                "distance": distance if distance is not None else float("inf"),
            }
        )
    return sorted(ranked, key=lambda item: (float(item["distance"]), item["clip_id"], item["candidate_id"]))


def rank_candidate_groups(reference_features: dict, candidates: list[dict], reference_transform_types: set[str]) -> list[dict]:
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate.get("candidate_group_id") or candidate["candidate_id"])].append(candidate)

    ranked = []
    for group_id, group_candidates in grouped.items():
        scored_windows = []
        ordered_features = []
        for candidate in group_candidates:
            ordered_features.append(candidate["features"])
            distance = color_distance(reference_features, candidate["features"])
            scored_windows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "clip_id": candidate["clip_id"],
                    "distance": distance if distance is not None else float("inf"),
                }
            )
        combined_features = combine_feature_documents(ordered_features)
        combined_distance = color_distance(reference_features, combined_features) if combined_features is not None else None
        parallel_features = combine_parallel_feature_documents(group_candidates) if is_parallel_contributor_group(group_candidates) else None
        parallel_distance = color_distance(reference_features, parallel_features) if parallel_features is not None else None
        panel_layout_distance = (
            parallel_panel_layout_distance(reference_features, ordered_features)
            if "split_screen" in reference_transform_types and is_parallel_contributor_group(group_candidates)
            else None
        )
        finite_distances = [float(item["distance"]) for item in scored_windows if item["distance"] != float("inf")]
        average_window_distance = sum(finite_distances) / len(finite_distances) if finite_distances else None
        tail_distance = tail_visual_distance(reference_features, combined_features, start_fraction=0.7) if combined_features is not None else None
        group_distance = best_group_distance(combined_distance, average_window_distance, tail_distance, parallel_distance, panel_layout_distance)
        clip_ids = sorted({item["clip_id"] for item in scored_windows})
        ranked.append(
            {
                "candidate_id": group_id,
                "clip_id": clip_ids[0] if clip_ids else group_id,
                "clip_ids": clip_ids,
                "distance": round(group_distance, 6) if group_distance != float("inf") else float("inf"),
                "window_count": len(scored_windows),
                "window_candidates": sorted(scored_windows, key=lambda item: (float(item["distance"]), item["candidate_id"])),
            }
        )
    return sorted(ranked, key=lambda item: (float(item["distance"]), item["candidate_id"]))


def combine_feature_documents(feature_documents: list[dict]) -> dict | None:
    frames = []
    for document in feature_documents:
        frames.extend(frame for frame in document.get("features", []) if isinstance(frame, dict))
    if not frames:
        return None
    return {"features": frames, "temporal_signature": binned_temporal_signature(frames, bins=8)}


def is_parallel_contributor_group(candidates: list[dict]) -> bool:
    roles = [
        str(candidate.get("source_window_entry", {}).get("role", ""))
        for candidate in candidates
        if isinstance(candidate.get("source_window_entry"), dict)
    ]
    return bool(roles) and all(role.startswith("contributor-") for role in roles)


def combine_parallel_feature_documents(candidates: list[dict]) -> dict | None:
    feature_rows = [
        [frame for frame in candidate["features"].get("features", []) if isinstance(frame, dict)]
        for candidate in candidates
    ]
    feature_rows = [frames for frames in feature_rows if frames]
    if not feature_rows:
        return None
    frame_count = min(len(frames) for frames in feature_rows)
    if frame_count <= 0:
        return None
    frames = [average_frame_features([frames[index] for frames in feature_rows], panel_count=len(feature_rows)) for index in range(frame_count)]
    return {"features": frames, "temporal_signature": binned_temporal_signature(frames, bins=8)}


def average_frame_features(frames: list[dict], panel_count: int | None = None) -> dict:
    combined = {
        "frame_index": min(int(frame.get("frame_index") or 0) for frame in frames),
    }
    for field in ("mean_rgb", "std_rgb", "active_mean_rgb"):
        averaged_vector = average_vector([frame.get(field) for frame in frames])
        if averaged_vector is not None:
            combined[field] = averaged_vector
    averaged_grid = panel_layout_grid([frame.get("grid_mean_rgb") for frame in frames], panel_count or len(frames))
    if averaged_grid is None:
        averaged_grid = average_grid([frame.get("grid_mean_rgb") for frame in frames])
    if averaged_grid is not None:
        combined["grid_mean_rgb"] = averaged_grid
    averaged_histogram = average_vector([frame.get("active_hue_histogram") for frame in frames])
    if averaged_histogram is not None:
        combined["active_hue_histogram"] = averaged_histogram
    for field in ("active_pixel_ratio", "mean_luma", "std_luma", "edge_density", "mean_absdiff_from_previous", "scene_change_score"):
        combined[field] = average_optional_scalar([frame.get(field) for frame in frames])
    flows = [frame.get("optical_flow") for frame in frames if isinstance(frame.get("optical_flow"), dict)]
    combined["optical_flow"] = average_flow(flows) if flows else None
    return combined


def average_vector(vectors: list) -> list[float] | None:
    usable = [vector for vector in vectors if isinstance(vector, list) and vector]
    if not usable:
        return None
    length = len(usable[0])
    if any(len(vector) != length for vector in usable):
        return None
    return [round(sum(float(vector[index]) for vector in usable) / len(usable), 6) for index in range(length)]


def average_grid(grids: list) -> list[list[float]] | None:
    usable = [grid for grid in grids if isinstance(grid, list) and grid]
    if not usable:
        return None
    cell_count = len(usable[0])
    if any(len(grid) != cell_count for grid in usable):
        return None
    averaged = []
    for cell_index in range(cell_count):
        averaged_cell = average_vector([grid[cell_index] for grid in usable])
        if averaged_cell is None:
            return None
        averaged.append(averaged_cell)
    return averaged


def panel_layout_grid(grids: list, panel_count: int) -> list[list[float]] | None:
    usable = [grid for grid in grids if isinstance(grid, list) and len(grid) == 9]
    if len(usable) != panel_count or panel_count not in {2, 3}:
        return None
    combined = []
    for row in range(3):
        if panel_count == 3:
            for panel_index in range(3):
                combined.append([round(float(value), 6) for value in usable[panel_index][row * 3 + 1]])
        else:
            left = usable[0][row * 3 + 1]
            right = usable[1][row * 3 + 1]
            middle = [round((float(left[channel]) + float(right[channel])) / 2.0, 6) for channel in range(3)]
            combined.extend([
                [round(float(value), 6) for value in left],
                middle,
                [round(float(value), 6) for value in right],
            ])
    return combined


def parallel_panel_layout_distance(reference_features: dict, feature_documents: list[dict]) -> float | None:
    panel_count = len(feature_documents)
    if panel_count not in {2, 3}:
        return None
    reference_frames = [frame for frame in reference_features.get("features", []) if isinstance(frame, dict)]
    panel_frames = [
        [frame for frame in document.get("features", []) if isinstance(frame, dict)]
        for document in feature_documents
    ]
    if not reference_frames or any(not frames for frames in panel_frames):
        return None
    frame_count = min(len(reference_frames), *(len(frames) for frames in panel_frames))
    distances = []
    for frame_index in range(frame_count):
        reference_grid = reference_frames[frame_index].get("grid_mean_rgb")
        if not isinstance(reference_grid, list) or len(reference_grid) != 9:
            continue
        for panel_index, frames in enumerate(panel_frames):
            panel_grid = frames[frame_index].get("grid_mean_rgb")
            if not isinstance(panel_grid, list) or len(panel_grid) != 9:
                continue
            reference_column = panel_index if panel_count == 3 else (0 if panel_index == 0 else 2)
            for row in range(3):
                reference_cell = reference_grid[row * 3 + reference_column]
                panel_cell = panel_grid[row * 3 + 1]
                if not isinstance(reference_cell, list) or not isinstance(panel_cell, list):
                    continue
                distances.append(sqrt(sum((float(reference_cell[channel]) - float(panel_cell[channel])) ** 2 for channel in range(3))))
    if not distances:
        return None
    return average(distances)


def average_optional_scalar(values: list) -> float | None:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    return round(sum(usable) / len(usable), 6)


def average_flow(flows: list[dict]) -> dict:
    return {
        field: average_optional_scalar([flow.get(field) for flow in flows])
        for field in ("mean_magnitude", "mean_dx", "mean_dy")
    }


def binned_temporal_signature(frames: list[dict], bins: int) -> list[list[float]]:
    if bins <= 0 or not frames:
        return []
    if len(frames) == 1:
        return [temporal_signature_row(frames[0]) for _ in range(bins)]
    signature = []
    last_index = len(frames) - 1
    for bin_index in range(bins):
        signature.append(temporal_signature_row(frames[round(bin_index * last_index / (bins - 1))]))
    return signature


def tail_visual_distance(reference_features: dict, source_features: dict, start_fraction: float) -> float | None:
    reference_tail = tail_feature_document(reference_features, start_fraction)
    source_tail = tail_feature_document(source_features, start_fraction)
    if reference_tail is None or source_tail is None:
        return None
    return color_distance(reference_tail, source_tail)


def tail_feature_document(feature_document: dict, start_fraction: float) -> dict | None:
    frames = [frame for frame in feature_document.get("features", []) if isinstance(frame, dict)]
    if not frames:
        return None
    start_index = max(0, min(len(frames) - 1, round(len(frames) * start_fraction)))
    tail_frames = frames[start_index:]
    if not tail_frames:
        return None
    return {"features": tail_frames}


def best_group_distance(*distances: float | None) -> float:
    distances = [distance for distance in distances if distance is not None and distance != float("inf")]
    if not distances:
        return float("inf")
    return min(distances)


def ranked_item_matches_expected(item: dict, expected_clip_ids: set[str]) -> bool:
    clip_ids = set(item.get("clip_ids") or [])
    if not clip_ids and item.get("clip_id"):
        clip_ids.add(str(item["clip_id"]))
    return bool(clip_ids & expected_clip_ids)


def metrics(results: list[dict]) -> dict:
    return {
        "evaluated_sample_count": len(results),
        "top1_accuracy": average([1.0 if result["top1_correct"] else 0.0 for result in results]),
        "top5_accuracy": average([1.0 if result["top5_correct"] else 0.0 for result in results]),
        "top10_accuracy": average([1.0 if result["top10_correct"] else 0.0 for result in results]),
        "mean_best_expected_rank": average([float(result["best_expected_rank"]) for result in results]),
    }


def breakdown_by_transform(results: list[dict]) -> dict:
    grouped = defaultdict(list)
    for result in results:
        for transform_type in result.get("transform_types") or ["unknown"]:
            grouped[str(transform_type)].append(result)
    return {transform_type: metrics(items) for transform_type, items in sorted(grouped.items())}


def source_clip_ids_from_ground_truth(ground_truth: dict) -> set[str]:
    clip_ids = set()
    for segment in ground_truth.get("segments", []):
        if segment.get("source_clip_id"):
            clip_ids.add(str(segment["source_clip_id"]))
        for contributor in segment.get("source_contributors", []):
            if contributor.get("source_clip_id"):
                clip_ids.add(str(contributor["source_clip_id"]))
    return clip_ids


def transform_types_from_ground_truth(ground_truth: dict) -> set[str]:
    return {
        str(transform["type"])
        for segment in ground_truth.get("segments", [])
        for transform in segment.get("transforms", [])
        if transform.get("type")
    }


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return document


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
