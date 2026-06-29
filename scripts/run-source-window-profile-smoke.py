#!/usr/bin/env python3
"""Run a profile-driven source-window match and host handoff smoke."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from aetherflow_video_match_inference.adapters import to_host_payload
from aetherflow_video_match_inference.cli import load_source_window_match_request, validate_source_window_request_document
from aetherflow_video_match_inference.engine import match_source_windows
from aetherflow_video_match_inference.interchange import export_after_effects_extendscript, export_cep_json, export_edl, export_premiere_json


REQUEST_BUILDER_SCRIPT = Path(__file__).resolve().parent / "build-source-window-match-request.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run-source-window-profile-smoke")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--schema")
    parser.add_argument("--max-candidate-groups", type=int)
    parser.add_argument("--host", default="aetherflow")
    parser.add_argument("--workflow-id", default="aetherflow-video-match")
    parser.add_argument("--sequence-name", default="AetherFlow Video Match")
    parser.add_argument("--comp-name", default="AetherFlow Video Match")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    request_path = output_dir / "request.json"
    match_result_path = output_dir / "match_result.json"
    host_payload_path = output_dir / "host_payload.json"
    cep_path = output_dir / "aetherflow_cep.json"
    premiere_path = output_dir / "premiere.json"
    edl_path = output_dir / "timeline.edl"
    ae_path = output_dir / "aetherflow_import.jsx"
    report_path = output_dir / "smoke_report.json"

    builder = load_request_builder()
    builder_args = [
        "--dataset-manifest",
        args.dataset_manifest,
        "--sample-id",
        args.sample_id,
        "--profile",
        args.profile,
        "--output",
        str(request_path),
    ]
    if args.max_candidate_groups is not None:
        builder_args.extend(["--max-candidate-groups", str(args.max_candidate_groups)])
    builder.main(builder_args)

    validate_source_window_request_document(request_path, args.schema)
    request = load_source_window_match_request(request_path, schema_path=args.schema)
    match_result = match_source_windows(request)
    match_result_path.write_text(json.dumps(match_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    host_payload = to_host_payload(match_result, args.host)
    host_payload_path.write_text(json.dumps(host_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    export_cep_json(host_payload, cep_path, args.workflow_id)
    export_premiere_json(host_payload, premiere_path, args.sequence_name)
    export_edl(host_payload, edl_path, "AETHERFLOW_VIDEO_MATCH")
    export_after_effects_extendscript(host_payload, ae_path, args.comp_name)

    top_candidate = match_result.get("ranking", {}).get("top_candidates", [{}])[0]
    report = {
        "profile": str(args.profile),
        "dataset_manifest": str(args.dataset_manifest),
        "sample_id": args.sample_id,
        "request": str(request_path),
        "match_result": str(match_result_path),
        "host_payload": str(host_payload_path),
        "cep_json": str(cep_path),
        "premiere_json": str(premiere_path),
        "edl": str(edl_path),
        "after_effects_extendscript": str(ae_path),
        "match_count": len(match_result.get("matches", [])),
        "top_candidate_id": top_candidate.get("candidate_id"),
        "top_candidate_clip_ids": top_candidate.get("clip_ids", []),
        "top_distance": top_candidate.get("distance"),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path)
    return 0


def load_request_builder():
    spec = importlib.util.spec_from_file_location("aetherflow_source_window_request_builder", REQUEST_BUILDER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load request builder script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
