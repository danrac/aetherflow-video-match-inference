#!/usr/bin/env python3
"""Evaluate inference feature scoring across every sample in an edge-case fixture."""

from __future__ import annotations

from collections import defaultdict
import json
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
        expected_clip_ids = source_clip_ids_from_ground_truth(ground_truth)
        ranked = rank_candidates(reference_features, candidate_features)
        expected_ranks = [index + 1 for index, candidate in enumerate(ranked) if ranked_item_matches_expected(candidate, expected_clip_ids)]
        if not expected_ranks:
            continue
        best_rank = min(expected_ranks)
        results.append(
            {
                "sample_id": sample["sample_id"],
                "transform_types": sorted(transform_types_from_ground_truth(ground_truth)),
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
                    "features": feature_document,
                }
            )
    return candidates


def rank_candidates(reference_features: dict, candidates: list[dict]) -> list[dict]:
    if any(candidate.get("candidate_group_id") for candidate in candidates):
        return rank_candidate_groups(reference_features, candidates)
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


def rank_candidate_groups(reference_features: dict, candidates: list[dict]) -> list[dict]:
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
        finite_distances = [float(item["distance"]) for item in scored_windows if item["distance"] != float("inf")]
        average_window_distance = sum(finite_distances) / len(finite_distances) if finite_distances else None
        group_distance = best_group_distance(combined_distance, average_window_distance)
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


def best_group_distance(combined_distance: float | None, average_window_distance: float | None) -> float:
    distances = [distance for distance in [combined_distance, average_window_distance] if distance is not None and distance != float("inf")]
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
