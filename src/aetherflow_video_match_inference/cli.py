"""Command line interface for inference smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import to_host_payload
from .engine import MatchRequest, match
from .interchange import export_edit_json, export_edl
from .storage import inference_output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aetherflow-video-match-inference")
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("match")
    run.add_argument("--model-manifest", required=True)
    run.add_argument("--reference", required=True)
    run.add_argument("--source", action="append", required=True)
    run.add_argument("--reference-feature-manifest")
    run.add_argument("--source-feature-manifest", action="append", default=[])
    run.add_argument("--output")

    host = subcommands.add_parser("host-payload")
    host.add_argument("--match-result", required=True)
    host.add_argument("--host", required=True)
    host.add_argument("--output", required=True)

    edit_json = subcommands.add_parser("export-edit-json")
    edit_json.add_argument("--host-payload", required=True)
    edit_json.add_argument("--output", required=True)

    edl = subcommands.add_parser("export-edl")
    edl.add_argument("--host-payload", required=True)
    edl.add_argument("--output", required=True)
    edl.add_argument("--title", default="AETHERFLOW_VIDEO_MATCH")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "match":
        result = match(
            MatchRequest(
                reference_path=args.reference,
                source_paths=tuple(args.source),
                model_manifest_path=args.model_manifest,
                reference_feature_manifest_path=args.reference_feature_manifest,
                source_feature_manifest_paths=tuple(args.source_feature_manifest),
            )
        )
        output_path = inference_output_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output_path)
        return 0
    if args.command == "host-payload":
        with Path(args.match_result).open("r", encoding="utf-8") as handle:
            match_result = json.load(handle)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(to_host_payload(match_result, args.host), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output_path)
        return 0
    if args.command == "export-edit-json":
        with Path(args.host_payload).open("r", encoding="utf-8") as handle:
            host_payload = json.load(handle)
        print(export_edit_json(host_payload, args.output))
        return 0
    if args.command == "export-edl":
        with Path(args.host_payload).open("r", encoding="utf-8") as handle:
            host_payload = json.load(handle)
        print(export_edl(host_payload, args.output, args.title))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
