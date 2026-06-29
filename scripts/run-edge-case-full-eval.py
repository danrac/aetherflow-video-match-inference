#!/usr/bin/env python3
"""Evaluate inference feature scoring across every sample in an edge-case fixture."""

from __future__ import annotations

from collections import defaultdict
import json
from math import isfinite, log1p, sqrt
import os
from pathlib import Path
import sys

from aetherflow_video_match_inference.features import color_distance, load_feature_manifest, temporal_signature_row


DATA_ROOT = Path(os.environ.get("AETHERFLOW_VIDEO_MATCH_DATA_ROOT", "/Volumes/FrameFusion/AetherFlow_VideoMatcherData"))
DEFAULT_DATASET = DATA_ROOT / "datasets" / "video-match-edge-case-fixture" / "v0003" / "dataset_manifest.json"
DEFAULT_OUTPUT = DATA_ROOT / "inference" / "video-match-edge-case-full-eval" / "v0003-hist" / "report.json"
FEATURE_NAMES = [
    "combined_visual",
    "average_window",
    "tail_visual",
    "parallel_visual",
    "panel_layout",
    "segment_sequence",
    "spatial_transform",
    "pip_overlay",
    "family_penalty",
    "window_count",
]
MISSING_DISTANCE = 1000.0


def main(argv: list[str]) -> int:
    dataset_manifest = Path(argv[1]) if len(argv) > 1 else DEFAULT_DATASET
    output_path = Path(argv[2]) if len(argv) > 2 else DEFAULT_OUTPUT
    reranker_model = load_reranker_model(argv[3]) if len(argv) > 3 else None
    component_cache_path = Path(argv[4]) if len(argv) > 4 else None
    if component_cache_path:
        report = evaluate_component_cache(dataset_manifest, component_cache_path, reranker_model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output_path)
        return 0
    root = dataset_manifest.parent
    dataset = load_json(dataset_manifest)
    candidate_features = load_source_window_candidates(root, dataset)
    candidate_groups = prepare_candidate_groups(candidate_features)
    results = []

    for sample in dataset.get("samples", []):
        reference_feature_manifest = sample.get("reference_feature_manifest")
        if not reference_feature_manifest:
            continue
        reference_features = load_feature_manifest(root / reference_feature_manifest)
        ground_truth = load_json(root / sample["ground_truth_path"])
        transforms = transforms_from_ground_truth(ground_truth)
        transform_types = transform_types_from_transforms(transforms)
        expected_clip_ids = source_clip_ids_from_ground_truth(ground_truth)
        ranked = rank_prepared_candidate_groups(reference_features, candidate_groups, transforms, reranker_model)
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
        "reranker_model": reranker_model_summary(reranker_model),
        "metrics": metrics(results),
        "breakdown_by_transform": breakdown_by_transform(results),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_path)
    return 0


def evaluate_component_cache(dataset_manifest: Path, component_cache_path: Path, reranker_model: dict | None) -> dict:
    cache = load_json(component_cache_path)
    if cache.get("feature_names") != FEATURE_NAMES:
        raise ValueError(f"component cache {component_cache_path} has incompatible feature_names")
    results = []
    for sample in cache.get("samples", []):
        expected_clip_ids = set(str(value) for value in sample.get("expected_clip_ids", []))
        transform_types = set(str(value) for value in sample.get("transform_types", []))
        ranked = []
        for candidate in sample.get("candidates", []):
            baseline_distance = float(candidate.get("baseline_distance", float("inf")))
            if reranker_model is not None and use_learned_reranker(transform_types, reranker_model):
                distance = linear_reranker_score(candidate.get("features", []), reranker_model)
            else:
                distance = baseline_distance
            ranked.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "clip_id": candidate.get("clip_id"),
                    "clip_ids": candidate.get("clip_ids", []),
                    "distance": round(distance, 6) if distance != float("inf") else float("inf"),
                    "window_count": candidate.get("window_count", 1),
                }
            )
        ranked = sorted(ranked, key=lambda item: (float(item["distance"]), str(item["candidate_id"])))
        expected_ranks = [index + 1 for index, candidate in enumerate(ranked) if ranked_item_matches_expected(candidate, expected_clip_ids)]
        if not expected_ranks:
            continue
        best_rank = min(expected_ranks)
        top_candidate = ranked[0]
        results.append(
            {
                "sample_id": sample.get("sample_id"),
                "transform_types": sorted(transform_types),
                "expected_clip_ids": sorted(expected_clip_ids),
                "best_expected_rank": best_rank,
                "top_candidate_id": top_candidate["candidate_id"],
                "top_clip_id": top_candidate["clip_id"],
                "top_candidate_clip_ids": top_candidate.get("clip_ids", []),
                "top_candidate_window_count": top_candidate.get("window_count", 1),
                "top_distance": top_candidate["distance"],
                "top1_correct": best_rank == 1,
                "top5_correct": best_rank <= 5,
                "top10_correct": best_rank <= 10,
            }
        )
    return {
        "dataset_manifest": str(dataset_manifest),
        "component_cache_path": str(component_cache_path),
        "candidate_count": int(cache.get("source_candidate_count", 0)),
        "reranker_model": reranker_model_summary(reranker_model),
        "metrics": metrics(results),
        "breakdown_by_transform": breakdown_by_transform(results),
        "results": results,
    }


def load_source_window_candidates(root: Path, dataset: dict) -> list[dict]:
    candidates = []
    for sample in dataset.get("samples", []):
        for index, entry in enumerate(sample.get("source_window_feature_manifests", [])):
            feature_document = load_feature_manifest(root / entry["path"])
            feature_document["source_window_entry"] = entry
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


def rank_candidates(reference_features: dict, candidates: list[dict], reference_transforms: list[dict] | None = None, reranker_model: dict | None = None) -> list[dict]:
    if any(candidate.get("candidate_group_id") for candidate in candidates):
        return rank_candidate_groups(reference_features, candidates, reference_transforms or [], reranker_model)
    ranked = []
    for candidate in candidates:
        distance = best_group_distance(
            color_distance(reference_features, candidate["features"]),
            spatial_transform_distance(reference_features, candidate["features"], reference_transforms or []),
        )
        ranked.append(
            {
                "candidate_id": candidate["candidate_id"],
                "clip_id": candidate["clip_id"],
                "distance": distance if distance is not None else float("inf"),
            }
        )
    return sorted(ranked, key=lambda item: (float(item["distance"]), item["clip_id"], item["candidate_id"]))


def rank_candidate_groups(reference_features: dict, candidates: list[dict], reference_transforms: list[dict], reranker_model: dict | None = None) -> list[dict]:
    return rank_prepared_candidate_groups(reference_features, prepare_candidate_groups(candidates), reference_transforms, reranker_model)


def prepare_candidate_groups(candidates: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate.get("candidate_group_id") or candidate["candidate_id"])].append(candidate)

    prepared_groups = []
    for group_id, group_candidates in grouped.items():
        ordered_features = [candidate["features"] for candidate in group_candidates]
        combined_features = combine_feature_documents(ordered_features)
        parallel_candidates = parallel_contributor_candidates(group_candidates)
        parallel_documents = [candidate["features"] for candidate in parallel_candidates]
        parallel_features = combine_parallel_feature_documents(parallel_candidates) if parallel_candidates else None
        prepared_groups.append(
            {
                "group_id": str(group_id),
                "group_candidates": group_candidates,
                "ordered_features": ordered_features,
                "combined_features": combined_features,
                "parallel_documents": parallel_documents,
                "parallel_features": parallel_features,
                "clip_ids": sorted({candidate["clip_id"] for candidate in group_candidates}),
            }
        )
    return prepared_groups


def rank_prepared_candidate_groups(reference_features: dict, prepared_groups: list[dict], reference_transforms: list[dict], reranker_model: dict | None = None) -> list[dict]:
    reference_transform_types = transform_types_from_transforms(reference_transforms)
    is_reverse_reference = "reverse" in reference_transform_types
    ranked = []
    for group in prepared_groups:
        group_id = str(group["group_id"])
        group_candidates = group["group_candidates"]
        ordered_features = group["ordered_features"]
        combined_features = group["combined_features"]
        parallel_documents = group["parallel_documents"]
        parallel_features = group["parallel_features"]
        scoring_ordered_features = reverse_feature_documents_for_playback(ordered_features) if is_reverse_reference else ordered_features
        scoring_combined_features = reverse_feature_document(combined_features) if is_reverse_reference and combined_features is not None else combined_features
        scoring_parallel_documents = [reverse_feature_document(document) for document in parallel_documents] if is_reverse_reference else parallel_documents
        scoring_parallel_features = reverse_feature_document(parallel_features) if is_reverse_reference and parallel_features is not None else parallel_features
        scored_windows = []
        for candidate in group_candidates:
            scoring_features = reverse_feature_document(candidate["features"]) if is_reverse_reference else candidate["features"]
            distance = color_distance(reference_features, scoring_features, allow_temporal_reverse=not is_reverse_reference)
            scored_windows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "clip_id": candidate["clip_id"],
                    "distance": distance if distance is not None else float("inf"),
                }
            )
        combined_distance = color_distance(reference_features, scoring_combined_features, allow_temporal_reverse=not is_reverse_reference) if scoring_combined_features is not None else None
        parallel_distance = color_distance(reference_features, scoring_parallel_features, allow_temporal_reverse=not is_reverse_reference) if scoring_parallel_features is not None else None
        panel_layout_distance = (
            parallel_panel_layout_distance(reference_features, scoring_parallel_documents)
            if "split_screen" in reference_transform_types and scoring_parallel_documents
            else None
        )
        split_panel_count_penalty = split_screen_panel_count_penalty(reference_transforms, parallel_documents)
        segment_sequence_distance = (
            simple_cut_segment_sequence_distance(reference_features, scoring_ordered_features)
            if "simple_cut" in reference_transform_types and is_segment_sequence_group(scoring_ordered_features)
            else None
        )
        pip_overlay_distance = (
            picture_in_picture_overlay_distance(reference_features, scoring_parallel_documents, reference_transforms)
            if "picture_in_picture" in reference_transform_types and scoring_parallel_documents
            else None
        )
        spatial_distance = spatial_transform_distance(reference_features, scoring_combined_features, reference_transforms) if scoring_combined_features is not None else None
        finite_distances = [float(item["distance"]) for item in scored_windows if item["distance"] != float("inf")]
        average_window_distance = sum(finite_distances) / len(finite_distances) if finite_distances else None
        tail_distance = tail_visual_distance(reference_features, scoring_combined_features, start_fraction=0.7, allow_temporal_reverse=not is_reverse_reference) if scoring_combined_features is not None else None
        components = {
            "combined_visual": combined_distance,
            "average_window": average_window_distance,
            "tail_visual": tail_distance,
            "parallel_visual": parallel_distance,
            "panel_layout": panel_layout_distance,
            "segment_sequence": segment_sequence_distance,
            "spatial_transform": spatial_distance,
            "pip_overlay": pip_overlay_distance,
            "family_penalty": candidate_family_penalty(ordered_features, reference_transform_types) + split_panel_count_penalty,
            "window_count": float(len(scored_windows)),
        }
        group_distance = best_group_distance(combined_distance, average_window_distance, tail_distance, parallel_distance, panel_layout_distance, segment_sequence_distance, spatial_distance, pip_overlay_distance)
        clip_ids = group["clip_ids"]
        baseline_distance = group_distance + float(components["family_penalty"]) if group_distance != float("inf") else float("inf")
        score_distance = reranker_distance(components, reference_transform_types, baseline_distance, reranker_model)
        ranked.append(
            {
                "candidate_id": group_id,
                "clip_id": clip_ids[0] if clip_ids else group_id,
                "clip_ids": clip_ids,
                "distance": round(score_distance, 6) if score_distance != float("inf") else float("inf"),
                "raw_distance": round(group_distance, 6) if group_distance != float("inf") else float("inf"),
                "window_count": len(scored_windows),
                "window_candidates": sorted(scored_windows, key=lambda item: (float(item["distance"]), item["candidate_id"])),
            }
        )
    return sorted(ranked, key=lambda item: (float(item["distance"]), item["candidate_id"]))


def reverse_feature_documents_for_playback(feature_documents: list[dict]) -> list[dict]:
    return [reverse_feature_document(document) for document in reversed(feature_documents)]


def reverse_feature_document(feature_document: dict | None) -> dict | None:
    if feature_document is None:
        return None
    frames = [reverse_frame_for_playback(frame) for frame in reversed(feature_document.get("features", [])) if isinstance(frame, dict)]
    reversed_document = {key: value for key, value in feature_document.items() if key not in {"features", "temporal_signature", "motion_track_summary", "_aetherflow_feature_cache"}}
    reversed_document["features"] = frames
    if frames:
        reversed_document["motion_track_summary"] = motion_track_summary_from_frames(frames)
        reversed_document["temporal_signature"] = binned_temporal_signature(frames, bins=8)
    return reversed_document


def reverse_frame_for_playback(frame: dict) -> dict:
    reversed_frame = dict(frame)
    flow = frame.get("optical_flow")
    if isinstance(flow, dict):
        reversed_flow = dict(flow)
        for field in ("mean_dx", "mean_dy", "mean_dx_normalized", "mean_dy_normalized"):
            if reversed_flow.get(field) is not None:
                reversed_flow[field] = -float(reversed_flow[field])
        reversed_frame["optical_flow"] = reversed_flow
    return reversed_frame


def combine_feature_documents(feature_documents: list[dict]) -> dict | None:
    frames = []
    for document in feature_documents:
        frames.extend(frame for frame in document.get("features", []) if isinstance(frame, dict))
    if not frames:
        return None
    return {"features": frames, "motion_track_summary": motion_track_summary_from_frames(frames), "temporal_signature": binned_temporal_signature(frames, bins=8)}


def is_parallel_contributor_group(feature_documents: list[dict]) -> bool:
    return bool(parallel_contributor_documents(feature_documents))


def parallel_contributor_candidates(candidates: list[dict]) -> list[dict]:
    contributors = []
    for candidate in candidates:
        entry = candidate.get("source_window_entry") if isinstance(candidate.get("source_window_entry"), dict) else {}
        role = str(entry.get("role", ""))
        if role.startswith("contributor-"):
            contributors.append((contributor_role_index(role), candidate))
    contributors.sort(key=lambda item: item[0])
    return [candidate for _, candidate in contributors] if len(contributors) >= 2 else []


def parallel_contributor_documents(feature_documents: list[dict]) -> list[dict]:
    contributors = []
    for document in feature_documents:
        entry = document.get("source_window_entry") if isinstance(document.get("source_window_entry"), dict) else {}
        role = str(entry.get("role", ""))
        if role.startswith("contributor-"):
            contributors.append((contributor_role_index(role), document))
    contributors.sort(key=lambda item: item[0])
    return [document for _, document in contributors] if len(contributors) >= 2 else []


def contributor_role_index(role: str) -> int:
    try:
        return int(role.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return 999


def is_segment_sequence_group(feature_documents: list[dict]) -> bool:
    roles = [
        str(document.get("source_window_entry", {}).get("role", ""))
        for document in feature_documents
        if isinstance(document.get("source_window_entry"), dict)
    ]
    return len(roles) >= 2 and all(role == "segment" for role in roles)


def candidate_family_penalty(feature_documents: list[dict], reference_transform_types: set[str]) -> float:
    if "simple_cut" in reference_transform_types and not is_segment_sequence_group(feature_documents):
        return 1000.0
    if "split_screen" in reference_transform_types and not parallel_contributor_documents(feature_documents):
        return 1000.0
    return 0.0


def split_screen_panel_count_penalty(transforms: list[dict], parallel_documents: list[dict]) -> float:
    expected_count = split_screen_panel_count(transforms)
    if expected_count is None or not parallel_documents:
        return 0.0
    return 0.0 if len(parallel_documents) == expected_count else 1000.0


def split_screen_panel_count(transforms: list[dict]) -> int | None:
    transform = next((transform for transform in transforms if transform.get("type") == "split_screen"), None)
    parameters = transform.get("parameters") if isinstance(transform, dict) and isinstance(transform.get("parameters"), dict) else {}
    try:
        panel_count = int(parameters.get("panels"))
    except (TypeError, ValueError):
        return None
    return panel_count if panel_count >= 2 else None


def simple_cut_segment_sequence_distance(reference_features: dict, feature_documents: list[dict]) -> float | None:
    reference_frames = [frame for frame in reference_features.get("features", []) if isinstance(frame, dict)]
    if not reference_frames or len(feature_documents) < 2:
        return None
    weights = []
    for document in feature_documents:
        entry = document.get("source_window_entry") if isinstance(document.get("source_window_entry"), dict) else {}
        source_in = entry.get("source_in")
        source_out = entry.get("source_out")
        try:
            weights.append(max(1, int(source_out) - int(source_in)))
        except (TypeError, ValueError):
            weights.append(max(1, len([frame for frame in document.get("features", []) if isinstance(frame, dict)])))
    reference_chunks = split_feature_frames_by_weights(reference_frames, weights)
    distances = []
    for chunk_frames, document in zip(reference_chunks, feature_documents, strict=False):
        candidate_frames = [frame for frame in document.get("features", []) if isinstance(frame, dict)]
        if not chunk_frames or not candidate_frames:
            continue
        candidate_document = {
            "features": candidate_frames,
            "motion_track_summary": motion_track_summary_from_frames(candidate_frames),
            "temporal_signature": binned_temporal_signature(candidate_frames, bins=8),
        }
        reference_document = {
            "features": chunk_frames,
            "motion_track_summary": motion_track_summary_from_frames(chunk_frames),
            "temporal_signature": binned_temporal_signature(chunk_frames, bins=8),
        }
        distance = color_distance(reference_document, candidate_document)
        if distance is not None:
            distances.append(distance)
    if not distances:
        return None
    return average(distances)


def split_feature_frames_by_weights(frames: list[dict], weights: list[int]) -> list[list[dict]]:
    total = sum(max(1, weight) for weight in weights)
    chunks = []
    start = 0
    for index, weight in enumerate(weights):
        if index == len(weights) - 1:
            end = len(frames)
        else:
            end = max(start + 1, round((sum(weights[: index + 1]) / total) * len(frames)))
        chunks.append(frames[start:end])
        start = end
    return chunks


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
    return {"features": frames, "motion_track_summary": motion_track_summary_from_frames(frames), "temporal_signature": binned_temporal_signature(frames, bins=8)}


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


def average_grid_mean_rgb(feature_document: dict) -> list[tuple[float, float, float]] | None:
    frames = feature_document.get("features", [])
    grid = average_grid([frame.get("grid_mean_rgb_5x5") or frame.get("grid_mean_rgb") for frame in frames if isinstance(frame, dict) and (frame.get("grid_mean_rgb_5x5") or frame.get("grid_mean_rgb"))])
    if grid is None:
        return None
    return [tuple(float(value) for value in cell) for cell in grid]


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
        reference_grid = frame_grid_mean_rgb(reference_frames[frame_index])
        grid_size = square_grid_size_from_raw(reference_grid)
        if reference_grid is None or grid_size is None:
            continue
        reference_cells = raw_grid_to_tuples(reference_grid)
        if len(reference_cells) != len(reference_grid):
            continue
        for panel_index, frames in enumerate(panel_frames):
            panel_grid = frame_grid_mean_rgb(frames[frame_index])
            if panel_grid is None or len(panel_grid) != len(reference_grid):
                continue
            panel_cells = raw_grid_to_tuples(panel_grid)
            if len(panel_cells) != len(panel_grid):
                continue
            if grid_size == 3:
                reference_column = panel_index if panel_count == 3 else (0 if panel_index == 0 else 2)
                for row in range(3):
                    reference_cell = reference_cells[row * 3 + reference_column]
                    panel_cell = panel_cells[row * 3 + 1]
                    distances.append(sqrt(sum((float(reference_cell[channel]) - float(panel_cell[channel])) ** 2 for channel in range(3))))
                continue
            panel_x0 = panel_index / panel_count
            panel_x1 = (panel_index + 1) / panel_count
            for row in range(grid_size):
                for column in range(grid_size):
                    cell_x0 = column / grid_size
                    cell_x1 = (column + 1) / grid_size
                    cell_y0 = row / grid_size
                    cell_y1 = (row + 1) / grid_size
                    overlap = rect_intersection((cell_x0, cell_y0, cell_x1, cell_y1), (panel_x0, 0.0, panel_x1, 1.0))
                    if overlap is None:
                        continue
                    reference_cell = reference_cells[row * grid_size + column]
                    panel_cell = sample_grid_region(
                        panel_cells,
                        grid_size,
                        (overlap[0] - panel_x0) / (panel_x1 - panel_x0),
                        overlap[1],
                        (overlap[2] - panel_x0) / (panel_x1 - panel_x0),
                        overlap[3],
                    )
                    distances.append(sqrt(sum((float(reference_cell[channel]) - float(panel_cell[channel])) ** 2 for channel in range(3))))
    if not distances:
        return None
    return average(distances) * 0.2


def frame_grid_mean_rgb(frame: dict) -> list | None:
    grid = frame.get("grid_mean_rgb_5x5") or frame.get("grid_mean_rgb")
    return grid if isinstance(grid, list) else None


def square_grid_size_from_raw(grid: list | None) -> int | None:
    if not isinstance(grid, list):
        return None
    size = round(sqrt(len(grid)))
    return size if size > 0 and size * size == len(grid) else None


def raw_grid_to_tuples(grid: list) -> list[tuple[float, float, float]]:
    return [tuple(float(value) for value in cell) for cell in grid if isinstance(cell, list) and len(cell) == 3]


def spatial_transform_distance(reference_features: dict, source_features: dict, transforms: list[dict]) -> float | None:
    distances = []
    for transform in transforms:
        transform_type = str(transform.get("type", ""))
        parameters = transform.get("parameters") if isinstance(transform.get("parameters"), dict) else {}
        if transform_type == "scale_position":
            distance = projected_grid_distance(reference_features, source_features, scale_position_geometry(parameters))
        elif transform_type == "letterbox":
            distance = projected_grid_distance(reference_features, source_features, aspect_fit_geometry(parameters, mode="letterbox"))
        elif transform_type == "crop":
            distance = projected_grid_distance(reference_features, source_features, aspect_fit_geometry(parameters, mode="crop"))
        else:
            distance = None
        if distance is not None:
            distances.append(distance)
    if not distances:
        return None
    return min(distances)


def picture_in_picture_overlay_distance(reference_features: dict, feature_documents: list[dict], transforms: list[dict]) -> float | None:
    if len(feature_documents) < 2:
        return None
    transform = next((transform for transform in transforms if transform.get("type") == "picture_in_picture"), None)
    if not isinstance(transform, dict):
        return None
    parameters = transform.get("parameters") if isinstance(transform.get("parameters"), dict) else {}
    geometry = picture_in_picture_geometry(parameters)
    if geometry is None:
        return None
    reference_grid = average_grid_mean_rgb(reference_features)
    base_grid = average_grid_mean_rgb(feature_documents[0])
    pip_grid = average_grid_mean_rgb(feature_documents[1])
    if reference_grid is None or base_grid is None or pip_grid is None or len(reference_grid) != len(base_grid) or len(reference_grid) != len(pip_grid):
        return None
    grid_size = square_grid_size(reference_grid)
    if grid_size is None:
        return None
    projected_grid = project_picture_in_picture_grid(base_grid, pip_grid, geometry, grid_size)
    distances = [
        sqrt(sum((reference_cell[index] - projected_cell[index]) ** 2 for index in range(3)))
        for reference_cell, projected_cell in zip(reference_grid, projected_grid, strict=False)
    ]
    return average(distances) * 0.2


def picture_in_picture_geometry(parameters: dict) -> dict | None:
    try:
        output_width = float(parameters["output_width"])
        output_height = float(parameters["output_height"])
        pip_width = float(parameters["width"])
        pip_height = float(parameters["height"])
        pip_x = float(parameters["x"])
        pip_y = float(parameters["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if min(output_width, output_height, pip_width, pip_height) <= 0:
        return None
    return {"output_width": output_width, "output_height": output_height, "x": pip_x, "y": pip_y, "width": pip_width, "height": pip_height}


def project_picture_in_picture_grid(base_grid: list[tuple[float, float, float]], pip_grid: list[tuple[float, float, float]], geometry: dict, grid_size: int) -> list[tuple[float, float, float]]:
    output_width = geometry["output_width"]
    output_height = geometry["output_height"]
    pip_rect = (geometry["x"], geometry["y"], geometry["x"] + geometry["width"], geometry["y"] + geometry["height"])
    projected = []
    for row in range(grid_size):
        cell_y0 = row * output_height / grid_size
        cell_y1 = (row + 1) * output_height / grid_size
        for column in range(grid_size):
            cell_x0 = column * output_width / grid_size
            cell_x1 = (column + 1) * output_width / grid_size
            base_color = sample_grid_region(base_grid, grid_size, column / grid_size, row / grid_size, (column + 1) / grid_size, (row + 1) / grid_size)
            overlap = rect_intersection((cell_x0, cell_y0, cell_x1, cell_y1), pip_rect)
            if overlap is None:
                projected.append(base_color)
            else:
                overlap_area = rect_area(overlap)
                cell_area = rect_area((cell_x0, cell_y0, cell_x1, cell_y1))
                pip_color = sample_projected_region(pip_grid, grid_size, geometry, overlap)
                pip_weight = overlap_area / cell_area if cell_area > 0 else 0.0
                projected.append(tuple(base_color[index] * (1.0 - pip_weight) + pip_color[index] * pip_weight for index in range(3)))
    return projected


def scale_position_geometry(parameters: dict) -> dict | None:
    try:
        output_width = float(parameters["output_width"])
        output_height = float(parameters["output_height"])
        scaled_width = float(parameters["width"])
        scaled_height = float(parameters["height"])
        x = float(parameters.get("x", 0.0))
        y = float(parameters.get("y", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    if min(output_width, output_height, scaled_width, scaled_height) <= 0:
        return None
    return {"output_width": output_width, "output_height": output_height, "x": x, "y": y, "width": scaled_width, "height": scaled_height}


def aspect_fit_geometry(parameters: dict, mode: str) -> dict | None:
    try:
        output_width = float(parameters["output_width"])
        output_height = float(parameters["output_height"])
    except (KeyError, TypeError, ValueError):
        return None
    source_ratio = parse_aspect_ratio(parameters.get("source_aspect_ratio") or parameters.get("aspect_ratio"))
    output_ratio = output_width / output_height if output_height else None
    if source_ratio is None or output_ratio is None or min(output_width, output_height) <= 0:
        return None
    if mode == "crop":
        if source_ratio > output_ratio:
            scaled_height = output_height
            scaled_width = output_height * source_ratio
        else:
            scaled_width = output_width
            scaled_height = output_width / source_ratio
    else:
        if source_ratio > output_ratio:
            scaled_width = output_width
            scaled_height = output_width / source_ratio
        else:
            scaled_height = output_height
            scaled_width = output_height * source_ratio
    return {
        "output_width": output_width,
        "output_height": output_height,
        "x": (output_width - scaled_width) / 2.0,
        "y": (output_height - scaled_height) / 2.0,
        "width": scaled_width,
        "height": scaled_height,
    }


def parse_aspect_ratio(value) -> float | None:
    if value is None:
        return None
    text = str(value)
    if ":" in text:
        numerator, denominator = text.split(":", 1)
        try:
            denominator_value = float(denominator)
            if denominator_value == 0:
                return None
            return float(numerator) / denominator_value
        except ValueError:
            return None
    try:
        ratio = float(text)
    except ValueError:
        return None
    return ratio if ratio > 0 else None


def projected_grid_distance(reference_features: dict, source_features: dict, geometry: dict | None) -> float | None:
    if geometry is None:
        return None
    reference_grid = average_grid_mean_rgb(reference_features)
    source_grid = average_grid_mean_rgb(source_features)
    if reference_grid is None or source_grid is None or len(reference_grid) != len(source_grid):
        return None
    grid_size = square_grid_size(source_grid)
    if grid_size is None:
        return None
    projected_grid = project_source_grid(source_grid, geometry, grid_size)
    distances = [
        sqrt(sum((reference_cell[index] - projected_cell[index]) ** 2 for index in range(3)))
        for reference_cell, projected_cell in zip(reference_grid, projected_grid, strict=False)
    ]
    return average(distances) * 0.2


def project_source_grid(source_grid: list[tuple[float, float, float]], geometry: dict, grid_size: int) -> list[tuple[float, float, float]]:
    output_width = geometry["output_width"]
    output_height = geometry["output_height"]
    source_rect = (geometry["x"], geometry["y"], geometry["x"] + geometry["width"], geometry["y"] + geometry["height"])
    projected = []
    for row in range(grid_size):
        cell_y0 = row * output_height / grid_size
        cell_y1 = (row + 1) * output_height / grid_size
        for column in range(grid_size):
            cell_x0 = column * output_width / grid_size
            cell_x1 = (column + 1) * output_width / grid_size
            overlap = rect_intersection((cell_x0, cell_y0, cell_x1, cell_y1), source_rect)
            if overlap is None:
                projected.append((0.0, 0.0, 0.0))
                continue
            overlap_area = rect_area(overlap)
            cell_area = rect_area((cell_x0, cell_y0, cell_x1, cell_y1))
            source_color = sample_projected_region(source_grid, grid_size, geometry, overlap)
            source_weight = overlap_area / cell_area if cell_area > 0 else 0.0
            projected.append(tuple(source_color[index] * source_weight for index in range(3)))
    return projected


def sample_projected_region(grid: list[tuple[float, float, float]], grid_size: int, geometry: dict, output_rect: tuple[float, float, float, float]) -> tuple[float, float, float]:
    x = geometry["x"]
    y = geometry["y"]
    width = geometry["width"]
    height = geometry["height"]
    return sample_grid_region(
        grid,
        grid_size,
        (output_rect[0] - x) / width,
        (output_rect[1] - y) / height,
        (output_rect[2] - x) / width,
        (output_rect[3] - y) / height,
    )


def sample_grid_region(grid: list[tuple[float, float, float]], grid_size: int, x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float]:
    x0 = max(0.0, min(1.0, x0))
    y0 = max(0.0, min(1.0, y0))
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    if x1 <= x0 or y1 <= y0:
        return (0.0, 0.0, 0.0)
    totals = [0.0, 0.0, 0.0]
    total_area = 0.0
    for row in range(grid_size):
        cell_y0 = row / grid_size
        cell_y1 = (row + 1) / grid_size
        for column in range(grid_size):
            cell_x0 = column / grid_size
            cell_x1 = (column + 1) / grid_size
            overlap = rect_intersection((x0, y0, x1, y1), (cell_x0, cell_y0, cell_x1, cell_y1))
            if overlap is None:
                continue
            area = rect_area(overlap)
            color = grid[row * grid_size + column]
            for index in range(3):
                totals[index] += color[index] * area
            total_area += area
    if total_area <= 0:
        return (0.0, 0.0, 0.0)
    return tuple(value / total_area for value in totals)


def rect_intersection(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def rect_area(rect: tuple[float, float, float, float]) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def square_grid_size(grid: list[tuple[float, float, float]]) -> int | None:
    size = round(sqrt(len(grid)))
    return size if size > 0 and size * size == len(grid) else None


def average_optional_scalar(values: list) -> float | None:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    return round(sum(usable) / len(usable), 6)


def average_flow(flows: list[dict]) -> dict:
    return {
        field: average_optional_scalar([flow.get(field) for flow in flows])
        for field in (
            "mean_magnitude",
            "mean_dx",
            "mean_dy",
            "mean_magnitude_normalized",
            "mean_dx_normalized",
            "mean_dy_normalized",
            "tracked_point_ratio",
            "motion_consistency",
        )
    }


def motion_track_summary_from_frames(frames: list[dict]) -> dict:
    flows = [frame.get("optical_flow") for frame in frames if isinstance(frame.get("optical_flow"), dict)]
    valid_flows = [flow for flow in flows if int(flow.get("tracked_points") or 0) > 0]
    if not flows:
        return {
            "tracked_frame_pairs": 0,
            "mean_magnitude_normalized": 0.0,
            "max_magnitude_normalized": 0.0,
            "mean_tracked_point_ratio": 0.0,
            "mean_motion_consistency": 0.0,
            "dominant_axis": "none",
        }
    magnitudes = [float(normalized_flow_value(flow, "mean_magnitude") or 0.0) for flow in flows]
    tracked_ratios = [float(flow.get("tracked_point_ratio") or 0.0) for flow in flows]
    consistencies = [float(flow.get("motion_consistency") or 0.0) for flow in valid_flows]
    mean_dx = sum(float(normalized_flow_value(flow, "mean_dx") or 0.0) for flow in valid_flows)
    mean_dy = sum(float(normalized_flow_value(flow, "mean_dy") or 0.0) for flow in valid_flows)
    return {
        "tracked_frame_pairs": len(valid_flows),
        "mean_magnitude_normalized": round(sum(magnitudes) / len(magnitudes), 6),
        "max_magnitude_normalized": round(max(magnitudes), 6),
        "mean_tracked_point_ratio": round(sum(tracked_ratios) / len(tracked_ratios), 6),
        "mean_motion_consistency": round(sum(consistencies) / len(consistencies), 6) if consistencies else 0.0,
        "dominant_axis": dominant_motion_axis(mean_dx, mean_dy),
    }


def normalized_flow_value(flow: dict, base_field: str) -> float | None:
    normalized_field = f"{base_field}_normalized"
    if flow.get(normalized_field) is not None:
        return float(flow[normalized_field])
    if base_field == "mean_magnitude" and flow.get("mean_magnitude") is not None:
        return float(flow["mean_magnitude"]) / 255.0
    if base_field == "mean_dx" and flow.get("mean_dx") is not None:
        return float(flow["mean_dx"]) / float(flow.get("frame_width") or 255.0)
    if base_field == "mean_dy" and flow.get("mean_dy") is not None:
        return float(flow["mean_dy"]) / float(flow.get("frame_height") or 255.0)
    return None


def dominant_motion_axis(mean_dx: float, mean_dy: float) -> str:
    if abs(mean_dx) < 1e-9 and abs(mean_dy) < 1e-9:
        return "none"
    return "horizontal" if abs(mean_dx) >= abs(mean_dy) else "vertical"


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


def tail_visual_distance(reference_features: dict, source_features: dict, start_fraction: float, *, allow_temporal_reverse: bool = True) -> float | None:
    reference_tail = tail_feature_document(reference_features, start_fraction)
    source_tail = tail_feature_document(source_features, start_fraction)
    if reference_tail is None or source_tail is None:
        return None
    return color_distance(reference_tail, source_tail, allow_temporal_reverse=allow_temporal_reverse)


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
    return transform_types_from_transforms(transforms_from_ground_truth(ground_truth))


def transforms_from_ground_truth(ground_truth: dict) -> list[dict]:
    return [
        transform
        for segment in ground_truth.get("segments", [])
        for transform in segment.get("transforms", [])
        if transform.get("type")
    ]


def transform_types_from_transforms(transforms: list[dict]) -> set[str]:
    return {str(transform["type"]) for transform in transforms if transform.get("type")}


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def load_reranker_model(path: str | Path) -> dict:
    model = load_json(path)
    weights = model.get("weights")
    if not isinstance(weights, list) or len(weights) != len(FEATURE_NAMES):
        raise ValueError(f"reranker model {path} does not contain {len(FEATURE_NAMES)} weights")
    model["weights"] = [float(value) for value in weights]
    model["bias"] = float(model.get("bias", 0.0))
    return model


def reranker_model_summary(model: dict | None) -> dict | None:
    if model is None:
        return None
    return {
        "model_id": model.get("model_id"),
        "model_version": model.get("model_version"),
        "routing": model.get("routing"),
    }


def reranker_distance(components: dict, transform_types: set[str], baseline_distance: float, model: dict | None) -> float:
    if model is None or not use_learned_reranker(transform_types, model):
        return baseline_distance
    features = normalize_components(components)
    return linear_reranker_score(features, model)


def linear_reranker_score(features: list[float], model: dict) -> float:
    return sum(float(model["weights"][index]) * features[index] for index in range(min(len(features), len(FEATURE_NAMES)))) + float(model.get("bias", 0.0))


def use_learned_reranker(transform_types: set[str], model: dict) -> bool:
    routing = model.get("routing") if isinstance(model.get("routing"), dict) else {}
    protected = set(routing.get("baseline_protected_transform_types") or ["crop", "letterbox", "picture_in_picture", "pillarbox", "scale_position"])
    learned = set(routing.get("learned_reranker_applies_to") or ["reverse", "simple_cut"])
    return bool(transform_types & learned) and not bool(transform_types & protected)


def normalize_components(components: dict) -> list[float]:
    features = []
    for name in FEATURE_NAMES:
        value = components.get(name)
        if value is None or not isfinite(float(value)):
            value = MISSING_DISTANCE
        if name == "family_penalty":
            features.append(min(float(value), 1000.0) / 1000.0)
        elif name == "window_count":
            features.append(min(float(value), 8.0) / 8.0)
        else:
            features.append(log1p(max(0.0, float(value))) / log1p(1000.0))
    return features


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return document


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
