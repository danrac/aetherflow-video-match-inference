#!/usr/bin/env python3
"""Evaluate inference feature scoring across every sample in an edge-case fixture."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import sys

from aetherflow_video_match_inference.features import color_distance, load_feature_manifest


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
        expected_ranks = [index + 1 for index, candidate in enumerate(ranked) if candidate["clip_id"] in expected_clip_ids]
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
                    "clip_id": str(entry["source_clip_id"]),
                    "features": feature_document,
                }
            )
    return candidates


def rank_candidates(reference_features: dict, candidates: list[dict]) -> list[dict]:
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
