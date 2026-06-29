#!/usr/bin/env python3
"""Run profile-driven source-window smokes across transform families."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


SMOKE_SCRIPT = Path(__file__).resolve().parent / "run-source-window-profile-smoke.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run-source-window-profile-smoke-grid")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--schema")
    parser.add_argument("--transform", action="append", default=[])
    parser.add_argument("--max-candidate-groups", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_manifest_path = Path(args.dataset_manifest)
    dataset_root = dataset_manifest_path.parent
    manifest = load_json(dataset_manifest_path)
    transforms = args.transform or ["reverse", "simple_cut", "scale_position", "picture_in_picture", "crop", "letterbox"]
    samples = select_samples_by_transform(dataset_root, manifest, transforms)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke = load_smoke_runner()

    results = []
    for transform_type, sample in samples.items():
        sample_output_dir = output_dir / transform_type
        smoke_args = [
            "--profile",
            args.profile,
            "--dataset-manifest",
            str(dataset_manifest_path),
            "--sample-id",
            sample["sample_id"],
            "--output-dir",
            str(sample_output_dir),
            "--comp-name",
            f"AetherFlow_{transform_type}_Smoke",
            "--sequence-name",
            f"AetherFlow {transform_type} Smoke",
        ]
        if args.schema:
            smoke_args.extend(["--schema", args.schema])
        if args.max_candidate_groups is not None:
            smoke_args.extend(["--max-candidate-groups", str(args.max_candidate_groups)])
        smoke.main(smoke_args)
        report = load_json(sample_output_dir / "smoke_report.json")
        report["transform_type"] = transform_type
        results.append(report)

    report = {
        "profile": str(args.profile),
        "dataset_manifest": str(dataset_manifest_path),
        "requested_transforms": transforms,
        "executed_transform_count": len(results),
        "results": results,
        "missing_transforms": [transform for transform in transforms if transform not in samples],
    }
    report_path = output_dir / "smoke_grid_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path)
    return 0


def select_samples_by_transform(dataset_root: Path, manifest: dict, transform_types: list[str]) -> dict[str, dict]:
    wanted = set(transform_types)
    selected = {}
    for sample in manifest.get("samples", []):
        sample_transforms = transforms_from_sample(dataset_root, sample)
        for transform_type in transform_types:
            if transform_type in selected:
                continue
            if transform_type in sample_transforms:
                selected[transform_type] = sample
        if set(selected) >= wanted:
            break
    return selected


def transforms_from_sample(dataset_root: Path, sample: dict) -> set[str]:
    ground_truth_path = sample.get("ground_truth_path")
    if not ground_truth_path:
        return set()
    ground_truth = load_json(dataset_root / ground_truth_path)
    return {
        str(transform["type"])
        for segment in ground_truth.get("segments", [])
        for transform in segment.get("transforms", [])
        if transform.get("type")
    }


def load_smoke_runner():
    spec = importlib.util.spec_from_file_location("aetherflow_source_window_profile_smoke", SMOKE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load source-window profile smoke script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return document


if __name__ == "__main__":
    raise SystemExit(main())
