#!/usr/bin/env python3
"""Run profile-driven source-window smokes over a selected sample batch."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path


SMOKE_SCRIPT = Path(__file__).resolve().parent / "run-source-window-profile-smoke.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run-source-window-profile-batch")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--schema")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--clip-id", action="append", default=[])
    parser.add_argument("--transform", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-candidate-groups", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_manifest_path = Path(args.dataset_manifest)
    dataset_root = dataset_manifest_path.parent
    manifest = load_json(dataset_manifest_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_samples = select_samples(
        dataset_root,
        manifest,
        sample_ids=set(args.sample_id),
        clip_ids=set(args.clip_id),
        transform_types=set(args.transform),
        limit=args.limit,
    )
    smoke = load_smoke_runner()
    results = []
    skipped = []
    for sample in selected_samples:
        sample_id = str(sample["sample_id"])
        sample_output_dir = output_dir / safe_slug(sample_id)
        report_path = sample_output_dir / "smoke_report.json"
        if args.skip_existing and report_path.exists():
            report = load_json(report_path)
            report["sample_id"] = sample_id
            report["skipped_existing"] = True
            results.append(enrich_report(dataset_root, sample, report))
            skipped.append({"sample_id": sample_id, "reason": "existing"})
            continue
        smoke_args = [
            "--profile",
            args.profile,
            "--dataset-manifest",
            str(dataset_manifest_path),
            "--sample-id",
            sample_id,
            "--output-dir",
            str(sample_output_dir),
            "--comp-name",
            f"AetherFlow_{safe_slug(sample_id)}",
            "--sequence-name",
            f"AetherFlow {sample_id}",
        ]
        if args.schema:
            smoke_args.extend(["--schema", args.schema])
        if args.max_candidate_groups is not None:
            smoke_args.extend(["--max-candidate-groups", str(args.max_candidate_groups)])
        smoke.main(smoke_args)
        results.append(enrich_report(dataset_root, sample, load_json(report_path)))

    report = {
        "profile": str(args.profile),
        "dataset_manifest": str(dataset_manifest_path),
        "selection": {
            "sample_ids": sorted(args.sample_id),
            "clip_ids": sorted(args.clip_id),
            "transform_types": sorted(args.transform),
            "limit": args.limit,
        },
        "selected_sample_count": len(selected_samples),
        "executed_sample_count": len(results) - len(skipped),
        "skipped": skipped,
        "top_candidate_expected_clip_match_count": sum(1 for result in results if result.get("top_candidate_matches_expected_clip")),
        "results": results,
    }
    report_path = output_dir / "batch_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path)
    return 0


def select_samples(
    dataset_root: Path,
    manifest: dict,
    *,
    sample_ids: set[str],
    clip_ids: set[str],
    transform_types: set[str],
    limit: int | None,
) -> list[dict]:
    selected = []
    for sample in manifest.get("samples", []):
        sample_id = str(sample.get("sample_id") or "")
        if sample_ids and sample_id not in sample_ids:
            continue
        expected_clip_ids = expected_source_clip_ids(dataset_root, sample)
        if clip_ids and not (expected_clip_ids & clip_ids):
            continue
        sample_transform_types = transforms_from_sample(dataset_root, sample)
        if transform_types and not (sample_transform_types & transform_types):
            continue
        selected.append(sample)
        if limit is not None and len(selected) >= max(0, int(limit)):
            break
    return selected


def enrich_report(dataset_root: Path, sample: dict, report: dict) -> dict:
    expected_clip_ids = expected_source_clip_ids(dataset_root, sample)
    top_clip_ids = {str(clip_id) for clip_id in report.get("top_candidate_clip_ids", [])}
    if report.get("top_candidate_clip_id"):
        top_clip_ids.add(str(report["top_candidate_clip_id"]))
    if not top_clip_ids and report.get("top_candidate_id"):
        top_clip_ids.add(str(report["top_candidate_id"]))
    return {
        **report,
        "sample_id": str(sample.get("sample_id") or report.get("sample_id") or ""),
        "expected_source_clip_ids": sorted(expected_clip_ids),
        "transform_types": sorted(transforms_from_sample(dataset_root, sample)),
        "top_candidate_matches_expected_clip": bool(expected_clip_ids & top_clip_ids),
    }


def expected_source_clip_ids(dataset_root: Path, sample: dict) -> set[str]:
    ground_truth_path = sample.get("ground_truth_path")
    if ground_truth_path:
        ground_truth = load_json(dataset_root / ground_truth_path)
        ids = {
            str(segment["source_clip_id"])
            for segment in ground_truth.get("segments", [])
            if segment.get("source_clip_id")
        }
        for segment in ground_truth.get("segments", []):
            for contributor in segment.get("contributors", []):
                if contributor.get("source_clip_id"):
                    ids.add(str(contributor["source_clip_id"]))
        if ids:
            return ids
    return {
        str(entry["source_clip_id"])
        for entry in sample.get("source_window_feature_manifests", [])
        if entry.get("source_clip_id")
    }


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


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip()).strip(".-")
    return slug[:120] or "sample"


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
