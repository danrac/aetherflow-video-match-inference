"""Command line interface for inference smoke runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .adapters import to_host_payload
from .interchange import export_after_effects_extendscript, export_cep_json, export_edit_json, export_edl, export_premiere_json
from .onnx_runtime import validate_onnx_model, validate_provider_route, validate_reranker_onnx_model
from .request_audit import audit_source_window_run
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

    source_windows = subcommands.add_parser("match-source-windows")
    source_windows.add_argument("--request", required=True)
    source_windows.add_argument("--output", required=True)
    source_windows.add_argument("--schema")

    source_windows_batch = subcommands.add_parser("match-source-windows-batch")
    source_windows_batch.add_argument("--request", action="append", default=[])
    source_windows_batch.add_argument("--request-dir")
    source_windows_batch.add_argument("--output", required=True)
    source_windows_batch.add_argument("--schema")
    source_windows_batch.add_argument("--skip-placement", action="store_true")
    source_windows_batch.add_argument("--assignment-top-n", type=int, default=12)

    visual_cache = subcommands.add_parser("precompute-visual-cache")
    visual_cache.add_argument("--request", action="append", default=[])
    visual_cache.add_argument("--request-dir")
    visual_cache.add_argument("--schema")
    visual_cache.add_argument("--output", required=True)
    visual_cache.add_argument("--limit", type=int, default=16)

    validate_source_window = subcommands.add_parser("validate-source-window-request")
    validate_source_window.add_argument("--request", required=True)
    validate_source_window.add_argument("--schema")

    audit_source_window = subcommands.add_parser("audit-source-window-run")
    audit_source_window.add_argument("--run-dir", required=True)
    audit_source_window.add_argument("--fixture-manifest")
    audit_source_window.add_argument("--output", required=True)

    validate_model = subcommands.add_parser("validate-model")
    validate_model.add_argument("--model-manifest", required=True)

    validate_reranker = subcommands.add_parser("validate-reranker-onnx")
    validate_reranker.add_argument("--model", required=True)
    validate_reranker.add_argument("--onnx", required=True)

    validate_provider = subcommands.add_parser("validate-provider")
    validate_provider.add_argument("--manifest", required=True)
    validate_provider.add_argument("--route", required=True)
    validate_provider.add_argument("--provider", required=True)
    validate_provider.add_argument("--smoke-input")
    validate_provider.add_argument("--tolerance", type=float, default=1e-4)

    host = subcommands.add_parser("host-payload")
    host.add_argument("--match-result", required=True)
    host.add_argument("--host", required=True)
    host.add_argument("--output", required=True)

    edit_json = subcommands.add_parser("export-edit-json")
    edit_json.add_argument("--host-payload", required=True)
    edit_json.add_argument("--output", required=True)

    cep_json = subcommands.add_parser("export-cep-json")
    cep_json.add_argument("--host-payload", required=True)
    cep_json.add_argument("--output", required=True)
    cep_json.add_argument("--workflow-id", default="aetherflow-video-match")

    premiere_json = subcommands.add_parser("export-premiere-json")
    premiere_json.add_argument("--host-payload", required=True)
    premiere_json.add_argument("--output", required=True)
    premiere_json.add_argument("--sequence-name", default="AetherFlow Video Match")

    edl = subcommands.add_parser("export-edl")
    edl.add_argument("--host-payload", required=True)
    edl.add_argument("--output", required=True)
    edl.add_argument("--title", default="AETHERFLOW_VIDEO_MATCH")

    ae = subcommands.add_parser("export-ae-extendscript")
    ae.add_argument("--host-payload", required=True)
    ae.add_argument("--output", required=True)
    ae.add_argument("--comp-name", default="AetherFlow Video Match")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "match":
        from .engine import MatchRequest, match

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
    if args.command == "match-source-windows":
        from .engine import match_source_windows

        request = load_source_window_match_request(args.request, schema_path=args.schema)
        result = match_source_windows(request)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output_path)
        return 0
    if args.command == "match-source-windows-batch":
        from .engine import match_source_windows_batch

        request_paths = source_window_batch_request_paths(args.request, args.request_dir)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        previous_skip_placement = os.environ.get("AETHERFLOW_VIDEO_MATCH_SKIP_PLACEMENT")
        if args.skip_placement:
            os.environ["AETHERFLOW_VIDEO_MATCH_SKIP_PLACEMENT"] = "1"
        try:
            requests = []
            for request_path in request_paths:
                requests.append(load_source_window_match_request(request_path, schema_path=args.schema))
                write_source_window_batch_output(
                    output_path,
                    request_count=len(requests),
                    total_latency=0.0,
                    results=[{"request_path": str(path), "result": None} for path in request_paths[: len(requests)]],
                    sequence_assignment=None,
                    complete=False,
                )
            batch_result = match_source_windows_batch(requests, assignment_top_n=max(1, int(args.assignment_top_n)))
        finally:
            if args.skip_placement:
                if previous_skip_placement is None:
                    os.environ.pop("AETHERFLOW_VIDEO_MATCH_SKIP_PLACEMENT", None)
                else:
                    os.environ["AETHERFLOW_VIDEO_MATCH_SKIP_PLACEMENT"] = previous_skip_placement
        results = []
        for request_path, entry in zip(request_paths, batch_result.get("results", []), strict=False):
            result_entry = dict(entry)
            result_entry["request_path"] = str(request_path)
            results.append(result_entry)
        write_source_window_batch_output(
            output_path,
            request_count=int(batch_result.get("request_count", len(results)) or len(results)),
            total_latency=float(batch_result.get("total_match_latency_ms", 0.0) or 0.0),
            results=results,
            sequence_assignment=batch_result.get("sequence_assignment"),
            complete=True,
            batch_wall_latency=float(batch_result.get("batch_wall_latency_ms", 0.0) or 0.0),
            cache=batch_result.get("cache") if isinstance(batch_result.get("cache"), dict) else None,
        )
        print(output_path)
        return 0
    if args.command == "validate-source-window-request":
        validate_source_window_request_document(args.request, args.schema)
        print("source-window request validation ok")
        return 0
    if args.command == "precompute-visual-cache":
        report = precompute_visual_cache_for_requests(args.request, args.request_dir, args.schema, limit=args.limit)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output_path)
        return 0
    if args.command == "audit-source-window-run":
        report = audit_source_window_run(args.run_dir, fixture_manifest_path=args.fixture_manifest)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output_path)
        return 0
    if args.command == "validate-model":
        with Path(args.model_manifest).open("r", encoding="utf-8") as handle:
            model_manifest = json.load(handle)
        print(json.dumps(validate_onnx_model(args.model_manifest, model_manifest), indent=2, sort_keys=True))
        return 0
    if args.command == "validate-reranker-onnx":
        print(json.dumps(validate_reranker_onnx_model(args.model, args.onnx), indent=2, sort_keys=True))
        return 0
    if args.command == "validate-provider":
        report = validate_provider_route(
            args.manifest,
            route_id=args.route,
            provider=args.provider,
            smoke_input_path=args.smoke_input,
            tolerance=args.tolerance,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("status") == "ok" else 2
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
    if args.command == "export-cep-json":
        with Path(args.host_payload).open("r", encoding="utf-8") as handle:
            host_payload = json.load(handle)
        print(export_cep_json(host_payload, args.output, args.workflow_id))
        return 0
    if args.command == "export-premiere-json":
        with Path(args.host_payload).open("r", encoding="utf-8") as handle:
            host_payload = json.load(handle)
        print(export_premiere_json(host_payload, args.output, args.sequence_name))
        return 0
    if args.command == "export-edl":
        with Path(args.host_payload).open("r", encoding="utf-8") as handle:
            host_payload = json.load(handle)
        print(export_edl(host_payload, args.output, args.title))
        return 0
    if args.command == "export-ae-extendscript":
        with Path(args.host_payload).open("r", encoding="utf-8") as handle:
            host_payload = json.load(handle)
        print(export_after_effects_extendscript(host_payload, args.output, args.comp_name))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


def write_source_window_batch_output(
    output_path: Path,
    *,
    request_count: int,
    total_latency: float,
    results: list[dict],
    sequence_assignment: dict | None,
    complete: bool,
    batch_wall_latency: float | None = None,
    cache: dict | None = None,
) -> None:
    document = {
        "schema_version": "0.1.0",
        "complete": complete,
        "request_count": request_count,
        "total_match_latency_ms": round(total_latency, 6),
        "average_match_latency_ms": round(total_latency / request_count, 6) if request_count else 0.0,
        "sequence_assignment": sequence_assignment,
        "results": results,
    }
    if batch_wall_latency is not None:
        document["batch_wall_latency_ms"] = round(batch_wall_latency, 6)
        document["average_batch_wall_latency_ms"] = round(batch_wall_latency / request_count, 6) if request_count else 0.0
    if cache is not None:
        document["cache"] = cache
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def load_source_window_match_request(path: str | Path, schema_path: str | Path | None = None) -> SourceWindowMatchRequest:
    from .engine import SourceWindowCandidate, SourceWindowMatchRequest

    document = load_json_object(path)
    validate_source_window_request_document(path, schema_path, document)
    if not isinstance(document, dict):
        raise ValueError(f"Expected source-window request object at {path}")
    candidates = []
    for index, candidate in enumerate(document.get("candidates", [])):
        if not isinstance(candidate, dict):
            raise ValueError(f"Expected candidate object at index {index}")
        candidates.append(
            SourceWindowCandidate(
                candidate_id=str(candidate["candidate_id"]),
                candidate_group_id=str(candidate["candidate_group_id"]),
                source_path=str(candidate["source_path"]),
                source_clip_id=str(candidate["source_clip_id"]),
                feature_manifest_path=str(candidate["feature_manifest_path"]),
                source_in=int(candidate["source_in"]),
                source_out=int(candidate["source_out"]),
                role=str(candidate.get("role", "source")),
                timeline_track=int(candidate.get("timeline_track", 0)),
                metadata=dict(candidate["metadata"]) if isinstance(candidate.get("metadata"), dict) else None,
            )
        )
    return SourceWindowMatchRequest(
        reference_path=str(document["reference_path"]),
        model_manifest_path=str(document["model_manifest_path"]),
        reference_feature_manifest_path=str(document["reference_feature_manifest_path"]),
        candidates=tuple(candidates),
        transforms=tuple(document.get("transforms", [])),
        reranker_model_path=str(document["reranker_model_path"]) if document.get("reranker_model_path") else None,
        placement_model_path=str(document["placement_model_path"]) if document.get("placement_model_path") else None,
        visual_encoder_onnx_path=str(document.get("visual_encoder_onnx_path") or document.get("visualEncoderOnnxPath")) if (document.get("visual_encoder_onnx_path") or document.get("visualEncoderOnnxPath")) else None,
        metadata=dict(document["metadata"]) if isinstance(document.get("metadata"), dict) else None,
    )


def precompute_visual_cache_for_requests(explicit_paths: list[str], request_dir: str | None, schema_path: str | Path | None, *, limit: int) -> dict:
    from .engine import reference_window_duration
    from .media_window import read_video_frame, source_crop_images
    from .visual_encoder import VisualEncoderScorer

    request_paths = source_window_batch_request_paths(explicit_paths, request_dir)
    encoded = 0
    skipped = 0
    rows = []
    scorer_by_path = {}
    for request_path in request_paths:
        request = load_source_window_match_request(request_path, schema_path=schema_path)
        if not request.visual_encoder_onnx_path:
            skipped += 1
            rows.append({"request_path": str(request_path), "status": "skipped", "reason": "visual_encoder_onnx_path_missing"})
            continue
        scorer = scorer_by_path.get(request.visual_encoder_onnx_path)
        if scorer is None:
            scorer = VisualEncoderScorer(request.visual_encoder_onnx_path)
            scorer_by_path[request.visual_encoder_onnx_path] = scorer
        from .features import load_feature_manifest

        reference_features = load_feature_manifest(request.reference_feature_manifest_path)
        reference_window = reference_features.get("source_window") if isinstance(reference_features.get("source_window"), dict) else {}
        reference_start = int(reference_window.get("source_in", 0) or 0)
        reference_duration = reference_window_duration(reference_features)
        fps = float(reference_features.get("fps", 30.0) or 30.0)
        reference_frame = read_video_frame(request.reference_path, reference_start + max(0, reference_duration // 2), fps)
        request_encoded = 0
        if reference_frame is not None:
            scorer.encode(reference_frame.convert("RGB"))
            encoded += 1
            request_encoded += 1
        for candidate in request.candidates[: max(1, int(limit))]:
            source_start = int(candidate.source_in)
            source_end = int(candidate.source_out)
            source_frame_index = source_start + max(0, (source_end - source_start) // 2)
            source_frame = read_video_frame(candidate.source_path, source_frame_index, fps)
            if source_frame is None:
                continue
            for crop in source_crop_images(source_frame):
                scorer.encode(crop)
                encoded += 1
                request_encoded += 1
        rows.append({"request_path": str(request_path), "status": "ok", "encoded_or_cached": request_encoded})
    return {
        "schema_version": "0.1.0",
        "request_count": len(request_paths),
        "encoded_or_cached": encoded,
        "skipped_request_count": skipped,
        "requests": rows,
    }


def source_window_batch_request_paths(explicit_paths: list[str], request_dir: str | None) -> list[Path]:
    paths = [Path(path) for path in explicit_paths]
    if request_dir:
        root = Path(request_dir)
        paths.extend(sorted(root.rglob("source_window_request.json")))
    seen: set[str] = set()
    unique = []
    for path in paths:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    if not unique:
        raise ValueError("match-source-windows-batch requires --request or --request-dir")
    return unique


def validate_source_window_request_document(path: str | Path, schema_path: str | Path | None = None, document: dict | None = None) -> None:
    resolved_schema = Path(schema_path) if schema_path else default_source_window_request_schema()
    if resolved_schema is None or not resolved_schema.exists():
        return
    request_document = document if document is not None else load_json_object(path)
    schema = load_json_object(resolved_schema)
    errors = validate_schema_subset(request_document, schema)
    if errors:
        raise ValueError("\n".join(errors))


def default_source_window_request_schema() -> Path | None:
    repo_contract_schema = Path(__file__).resolve().parents[3] / "contracts" / "schemas" / "source_window_match_request.schema.json"
    if repo_contract_schema.exists():
        return repo_contract_schema
    return None


def load_json_object(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return document


def validate_schema_subset(document, schema: dict, path: str = "$") -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type and not matches_schema_type(document, expected_type):
        return [f"{path}: expected {expected_type}, got {type(document).__name__}"]

    if expected_type == "object":
        if not isinstance(document, dict):
            return [f"{path}: expected object"]
        for key in schema.get("required", []):
            if key not in document:
                errors.append(f"{path}: missing required property {key}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in document:
                if key not in properties:
                    errors.append(f"{path}: additional property {key}")
        for key, value in document.items():
            if key in properties:
                errors.extend(validate_schema_subset(value, properties[key], f"{path}.{key}"))

    if expected_type == "array":
        if not isinstance(document, list):
            return [f"{path}: expected array"]
        min_items = schema.get("minItems")
        if min_items is not None and len(document) < int(min_items):
            errors.append(f"{path}: expected at least {min_items} items")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(document):
                errors.extend(validate_schema_subset(item, item_schema, f"{path}[{index}]"))

    if isinstance(document, (int, float)) and not isinstance(document, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        if minimum is not None and document < minimum:
            errors.append(f"{path}: expected >= {minimum}")
        if maximum is not None and document > maximum:
            errors.append(f"{path}: expected <= {maximum}")
        if exclusive_minimum is not None and document <= exclusive_minimum:
            errors.append(f"{path}: expected > {exclusive_minimum}")
    return errors


def matches_schema_type(value, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    return True


if __name__ == "__main__":
    raise SystemExit(main())
