#!/usr/bin/env python3
"""Build a source-window match request from a dataset manifest sample."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="build-source-window-match-request")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--model-manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--reranker-model")
    parser.add_argument("--max-candidate-groups", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = load_json(args.profile) if args.profile else {}
    dataset_manifest_path = Path(args.dataset_manifest)
    dataset_root = dataset_manifest_path.parent
    manifest = load_json(dataset_manifest_path)
    clips_by_id = {str(clip["clip_id"]): clip for clip in manifest.get("clips", []) if clip.get("clip_id")}
    target_sample = next((sample for sample in manifest.get("samples", []) if str(sample.get("sample_id")) == args.sample_id), None)
    if target_sample is None:
        raise ValueError(f"sample_id not found in dataset manifest: {args.sample_id}")

    model_manifest = expand_profile_path(args.model_manifest or profile.get("model_manifest"))
    if not model_manifest:
        raise ValueError("--model-manifest is required unless --profile provides model_manifest")
    reranker_model = expand_profile_path(args.reranker_model or profile.get("reranker_model"))

    candidates = []
    group_count = 0
    for sample in manifest.get("samples", []):
        if args.max_candidate_groups is not None and group_count >= args.max_candidate_groups:
            break
        group_candidates = candidates_from_sample(dataset_root, sample, clips_by_id)
        if not group_candidates:
            continue
        candidates.extend(group_candidates)
        group_count += 1

    document = {
        "schema_version": "0.1.0",
        "reference_path": resolve_dataset_path(dataset_root, target_sample["reference_path"]),
        "model_manifest_path": str(Path(model_manifest)),
        "reference_feature_manifest_path": resolve_dataset_path(dataset_root, target_sample["reference_feature_manifest"]),
        "transforms": transforms_from_ground_truth(dataset_root, target_sample),
        "candidates": candidates,
    }
    if reranker_model:
        document["reranker_model_path"] = str(Path(reranker_model))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_path)
    return 0


def candidates_from_sample(dataset_root: Path, sample: dict, clips_by_id: dict[str, dict]) -> list[dict]:
    group_id = str(sample.get("sample_id", ""))
    candidates = []
    for index, entry in enumerate(sample.get("source_window_feature_manifests", [])):
        clip_id = str(entry["source_clip_id"])
        clip = clips_by_id.get(clip_id, {})
        candidates.append(
            {
                "candidate_id": f"{group_id}:{index}:{clip_id}:{entry['source_in']}-{entry['source_out']}",
                "candidate_group_id": group_id,
                "source_path": str(clip.get("path") or clip.get("original_path") or clip_id),
                "source_clip_id": clip_id,
                "feature_manifest_path": resolve_dataset_path(dataset_root, entry["path"]),
                "source_in": int(entry["source_in"]),
                "source_out": int(entry["source_out"]),
                "role": str(entry.get("role", "source")),
                "timeline_track": int(entry.get("timeline_track", 0)),
            }
        )
    return candidates


def transforms_from_ground_truth(dataset_root: Path, sample: dict) -> list[dict]:
    ground_truth_path = sample.get("ground_truth_path")
    if not ground_truth_path:
        return []
    ground_truth = load_json(dataset_root / ground_truth_path)
    return [
        transform
        for segment in ground_truth.get("segments", [])
        for transform in segment.get("transforms", [])
        if transform.get("type")
    ]


def resolve_dataset_path(dataset_root: Path, path: str) -> str:
    candidate = Path(path)
    return str(candidate if candidate.is_absolute() else dataset_root / candidate)


def expand_profile_path(path: str | None) -> str | None:
    if path is None:
        return None
    return os.path.expandvars(path)


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return document


if __name__ == "__main__":
    raise SystemExit(main())
