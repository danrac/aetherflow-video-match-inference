"""Command line interface for inference smoke runs."""

from __future__ import annotations

import argparse
import json
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
    if args.command == "validate-source-window-request":
        validate_source_window_request_document(args.request, args.schema)
        print("source-window request validation ok")
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
        metadata=dict(document["metadata"]) if isinstance(document.get("metadata"), dict) else None,
    )


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
