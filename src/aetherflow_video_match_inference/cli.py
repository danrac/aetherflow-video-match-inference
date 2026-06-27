"""Command line interface for inference smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import MatchRequest, match
from .storage import inference_output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aetherflow-video-match-inference")
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("match")
    run.add_argument("--model-manifest", required=True)
    run.add_argument("--reference", required=True)
    run.add_argument("--source", action="append", required=True)
    run.add_argument("--output")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "match":
        result = match(
            MatchRequest(
                reference_path=args.reference,
                source_paths=tuple(args.source),
                model_manifest_path=args.model_manifest,
            )
        )
        output_path = inference_output_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output_path)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
