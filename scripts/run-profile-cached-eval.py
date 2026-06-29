#!/usr/bin/env python3
"""Run cached reranker evaluation from an inference profile."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


FULL_EVAL_SCRIPT = Path(__file__).resolve().parent / "run-edge-case-full-eval.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run-profile-cached-eval")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--reranker-model")
    parser.add_argument("--component-cache")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = load_json(args.profile)
    dataset_manifest = expand_path(args.dataset_manifest or profile.get("dataset_manifest"))
    reranker_model = expand_path(args.reranker_model or profile.get("reranker_model"))
    component_cache = expand_path(args.component_cache or profile.get("reranker_component_cache"))
    if not dataset_manifest:
        raise ValueError("--dataset-manifest is required unless profile provides dataset_manifest")
    if not reranker_model:
        raise ValueError("--reranker-model is required unless profile provides reranker_model")
    if not component_cache:
        raise ValueError("--component-cache is required unless profile provides reranker_component_cache")

    full_eval = load_full_eval()
    return full_eval.main(["run-edge-case-full-eval.py", dataset_manifest, args.output, reranker_model, component_cache])


def expand_path(path: str | None) -> str | None:
    return os.path.expandvars(path) if path else None


def load_full_eval():
    spec = importlib.util.spec_from_file_location("aetherflow_profile_cached_eval", FULL_EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load full eval script")
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
