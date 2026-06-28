#!/usr/bin/env python3
"""Run inference smoke cases against the generated edge-case fixture."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from aetherflow_video_match_inference.adapters import to_host_payload
from aetherflow_video_match_inference.engine import MatchRequest, match


DATA_ROOT = Path(os.environ.get("AETHERFLOW_VIDEO_MATCH_DATA_ROOT", "/Volumes/FrameFusion/AetherFlow_VideoMatcherData"))
DEFAULT_DATASET = DATA_ROOT / "datasets" / "video-match-edge-case-fixture" / "v0002" / "dataset_manifest.json"
DEFAULT_MODEL = DATA_ROOT / "models" / "video-match-edge-case-baseline" / "v0001" / "model_manifest.json"
DEFAULT_OUTPUT = DATA_ROOT / "inference" / "video-match-edge-case-smoke" / "v0001"
CASE_TYPES = (
    "simple_cut",
    "crop",
    "rotation",
    "picture_in_picture",
    "partial_occlusion",
    "speed_change",
    "cross_dissolve",
)


def main(argv: list[str]) -> int:
    dataset_manifest = Path(argv[1]) if len(argv) > 1 else DEFAULT_DATASET
    model_manifest = Path(argv[2]) if len(argv) > 2 else DEFAULT_MODEL
    output_root = Path(argv[3]) if len(argv) > 3 else DEFAULT_OUTPUT

    dataset = load_json(dataset_manifest)
    root = dataset_manifest.parent
    clips_by_id = {clip["clip_id"]: clip for clip in dataset.get("clips", [])}
    selected_samples = select_samples(root, dataset, CASE_TYPES)

    summary = {
        "dataset_manifest": str(dataset_manifest),
        "model_manifest": str(model_manifest),
        "output_root": str(output_root),
        "cases": [],
    }
    for transform_type, sample, ground_truth in selected_samples:
        case_dir = output_root / transform_type / sample["sample_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        source_window_entries = sample.get("source_window_feature_manifests", [])
        if not source_window_entries:
            raise ValueError(f"{sample['sample_id']}: missing source_window_feature_manifests")
        request = MatchRequest(
            reference_path=str(root / sample["reference_path"]),
            source_paths=tuple(str(root / clips_by_id[entry["source_clip_id"]]["path"]) for entry in source_window_entries),
            model_manifest_path=str(model_manifest),
            reference_feature_manifest_path=str(root / sample["reference_feature_manifest"]),
            source_feature_manifest_paths=tuple(str(root / entry["path"]) for entry in source_window_entries),
        )
        result = match(request)
        result_path = case_dir / "match_result.json"
        payload_path = case_dir / "host_payload.json"
        write_json(result_path, result)
        write_json(payload_path, to_host_payload(result, "aetherflow"))

        expected_clip_ids = sorted(source_clip_ids_from_ground_truth(ground_truth))
        ranked_matches = sorted(
            zip(result["matches"], source_window_entries, strict=False),
            key=lambda item: float(item[0].get("confidence", 0.0)),
            reverse=True,
        )
        best_entry = ranked_matches[0][1] if ranked_matches else {}
        best_source_clip_id = best_entry.get("source_clip_id")
        summary["cases"].append(
            {
                "transform_type": transform_type,
                "sample_id": sample["sample_id"],
                "match_result": str(result_path),
                "host_payload": str(payload_path),
                "candidate_count": len(source_window_entries),
                "expected_source_clip_ids": expected_clip_ids,
                "best_source_clip_id": best_source_clip_id,
                "top_candidate_expected": best_source_clip_id in expected_clip_ids,
                "max_confidence": ranked_matches[0][0].get("confidence") if ranked_matches else None,
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    write_json(summary_path, summary)
    print(summary_path)
    return 0


def select_samples(root: Path, dataset: dict, case_types: tuple[str, ...]) -> list[tuple[str, dict, dict]]:
    selected = []
    for transform_type in case_types:
        for sample in dataset.get("samples", []):
            ground_truth = load_json(root / sample["ground_truth_path"])
            if transform_type in transform_types_from_ground_truth(ground_truth):
                selected.append((transform_type, sample, ground_truth))
                break
        else:
            raise ValueError(f"No sample found for transform type {transform_type}")
    return selected


def transform_types_from_ground_truth(ground_truth: dict) -> set[str]:
    return {
        str(transform["type"])
        for segment in ground_truth.get("segments", [])
        for transform in segment.get("transforms", [])
        if transform.get("type")
    }


def source_clip_ids_from_ground_truth(ground_truth: dict) -> set[str]:
    clip_ids = set()
    for segment in ground_truth.get("segments", []):
        if segment.get("source_clip_id"):
            clip_ids.add(str(segment["source_clip_id"]))
        for contributor in segment.get("source_contributors", []):
            if contributor.get("source_clip_id"):
                clip_ids.add(str(contributor["source_clip_id"]))
    return clip_ids


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return document


def write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
