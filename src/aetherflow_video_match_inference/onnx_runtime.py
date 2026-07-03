"""Optional ONNX Runtime loading boundary."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Any


_LINEAR_SESSION_CACHE: dict[tuple[str, tuple[str, ...]], Any] = {}
_VISUAL_ENCODER_COSINE_THRESHOLD = 0.995
_ACCELERATED_EVENT_RATIO_THRESHOLD = 0.50


def model_onnx_path(model_manifest_path: str | Path, model_manifest: dict[str, Any]) -> Path:
    onnx_path = Path(str(model_manifest["onnx_path"]))
    if onnx_path.is_absolute():
        return onnx_path
    return Path(model_manifest_path).parent / onnx_path


def model_runtime_onnx_path(model: dict[str, Any]) -> Path | None:
    """Resolve an ONNX path from a loaded JSON model object."""

    runtime = model.get("onnx_runtime") if isinstance(model.get("onnx_runtime"), dict) else {}
    path_value = (
        runtime.get("onnx_path")
        or runtime.get("model_path")
        or model.get("onnx_path")
        or model.get("onnxPath")
    )
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    model_path = model.get("_model_path")
    if model_path:
        return Path(str(model_path)).parent / path
    return path


def preferred_provider_sequence(model: dict[str, Any] | None = None) -> list[str]:
    """Return the provider preference requested by env/model metadata."""

    env_value = os.environ.get("AETHERFLOW_ONNX_PROVIDERS") or os.environ.get("AETHERFLOW_ONNX_PROVIDER")
    if env_value:
        providers = [item.strip() for item in env_value.split(",") if item.strip()]
    else:
        runtime = model.get("onnx_runtime") if isinstance(model, dict) and isinstance(model.get("onnx_runtime"), dict) else {}
        provider = (
            runtime.get("preferred_provider")
            or runtime.get("preferredProvider")
            or (model.get("preferred_provider") if isinstance(model, dict) else None)
            or (model.get("preferredProvider") if isinstance(model, dict) else None)
        )
        providers = [str(provider)] if provider else []
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
    return providers


def linear_onnx_score(model: dict[str, Any], feature_values: list[float]) -> dict[str, Any] | None:
    """Run a linear `features -> scores` ONNX model when available.

    Returns None when ONNX Runtime or the referenced model is unavailable so the
    caller can use the deterministic JSON scorer fallback.
    """

    if os.environ.get("AETHERFLOW_ONNX_DISABLE") in {"1", "true", "TRUE", "yes", "YES"}:
        return None
    onnx_path = model_runtime_onnx_path(model)
    if onnx_path is None or not onnx_path.exists():
        return None
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        return None

    available = set(ort.get_available_providers())
    requested = preferred_provider_sequence(model)
    provider_attempts = [provider for provider in requested if provider in available]
    if "CPUExecutionProvider" not in provider_attempts and "CPUExecutionProvider" in available:
        provider_attempts.append("CPUExecutionProvider")
    if not provider_attempts:
        return None

    last_error: Exception | None = None
    for provider in provider_attempts:
        cache_key = (str(onnx_path), (provider,))
        try:
            session = _LINEAR_SESSION_CACHE.get(cache_key)
            if session is None:
                session = ort.InferenceSession(str(onnx_path), providers=[provider])
                _LINEAR_SESSION_CACHE[cache_key] = session
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if not inputs or not outputs:
                return None
            row = np.asarray(feature_values, dtype=np.float32).reshape(1, -1)
            observed = session.run([outputs[0].name], {inputs[0].name: row})[0]
            return {
                "score": float(np.asarray(observed).reshape(-1)[0]),
                "onnx_path": str(onnx_path),
                "requested_provider": provider,
                "session_providers": list(session.get_providers()),
                "available_providers": sorted(available),
            }
        except Exception as exc:  # pragma: no cover - provider-specific failures need target runtimes.
            last_error = exc
            continue
    model["_last_onnx_runtime_error"] = str(last_error) if last_error else "no provider could load ONNX model"
    return None


def validate_provider_route(
    manifest_path: str | Path,
    route_id: str,
    provider: str,
    smoke_input_path: str | Path | None = None,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Validate that a routed artifact loads and runs on the requested provider."""

    try:
        import json
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        return {
            "status": "failed",
            "route": route_id,
            "requestedProvider": provider,
            "availableProviders": [],
            "providerAvailable": False,
            "actualProviders": [],
            "usedProvider": None,
            "cpuFallback": True,
            "latencyMs": None,
            "outputParity": {
                "maxDeltaVsCpu": None,
                "withinTolerance": False,
                "tolerance": float(tolerance),
            },
            "stages": [],
            "error": "onnxruntime and numpy are required to validate execution providers",
            "exception": str(exc),
        }

    manifest_file = Path(manifest_path)
    with manifest_file.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected manifest object at {manifest_file}")
    route = runtime_route(manifest, route_id)
    model_specs = route_model_specs(route)
    if not model_specs:
        raise ValueError(f"Route {route_id} does not declare ONNX scoring artifacts")
    effective_tolerance = float(route.get("providerValidationTolerance", tolerance))

    available = list(ort.get_available_providers())
    provider_available = provider in available
    stage_reports = []
    cpu_fallback = False
    total_start = perf_counter()
    for spec in model_specs:
        onnx_path = resolve_manifest_asset(manifest_file, spec["onnx"])
        try:
            if spec.get("kind") == "visual_encoder":
                provider_report = run_visual_encoder_smoke(onnx_path, provider, ["CPUExecutionProvider"], spec.get("provider_options"))
            else:
                rows = smoke_rows_for_model(onnx_path, smoke_input_path, spec["stage"])
                provider_report = run_onnx_smoke(onnx_path, rows, provider, ["CPUExecutionProvider"], effective_tolerance)
        except Exception as exc:
            provider_report = {
                "onnxPath": str(onnx_path),
                "requestedProvider": provider,
                "actualProviders": [],
                "usedProvider": None,
                "cpuFallback": True,
                "latencyMs": None,
                "cpuLatencyMs": None,
                "outputParity": {
                    "maxDeltaVsCpu": None,
                    "withinTolerance": False,
                    "tolerance": float(tolerance),
                },
                "error": str(exc),
            }
        provider_report["stage"] = spec["stage"]
        provider_report["artifact"] = spec["onnx"]
        stage_reports.append(provider_report)
        cpu_fallback = cpu_fallback or bool(provider_report["cpuFallback"])

    latency_ms = round((perf_counter() - total_start) * 1000.0, 6)
    used_provider = stage_reports[0]["usedProvider"] if stage_reports else None
    within_tolerance = all(bool(stage["outputParity"]["withinTolerance"]) for stage in stage_reports)
    status = "ok" if provider_available and not cpu_fallback and within_tolerance else "failed"
    cosine_values = [
        float(stage["outputParity"]["cosineSimilarityVsCpu"])
        for stage in stage_reports
        if stage.get("outputParity", {}).get("cosineSimilarityVsCpu") is not None
    ]
    return {
        "status": status,
        "route": route_id,
        "requestedProvider": provider,
        "availableProviders": available,
        "providerAvailable": provider_available,
        "actualProviders": stage_reports[0]["actualProviders"] if stage_reports else [],
        "usedProvider": used_provider,
        "cpuFallback": cpu_fallback,
        "latencyMs": latency_ms,
        "outputParity": {
            "maxDeltaVsCpu": max((float(stage["outputParity"]["maxDeltaVsCpu"]) for stage in stage_reports), default=None),
            "minCosineSimilarityVsCpu": min(cosine_values) if cosine_values else None,
            "withinTolerance": within_tolerance,
            "linearTolerance": effective_tolerance,
            "visualEncoderCosineThreshold": _VISUAL_ENCODER_COSINE_THRESHOLD if cosine_values else None,
        },
        "stages": stage_reports,
    }


def runtime_route(manifest: dict[str, Any], route_id: str) -> dict[str, Any]:
    routes = manifest.get("runtimeRoutes")
    if isinstance(routes, dict) and isinstance(routes.get(route_id), dict):
        return routes[route_id]
    for route in manifest.get("routes", []):
        if isinstance(route, dict) and str(route.get("route_id")) == route_id:
            return route
    raise ValueError(f"Unknown runtime route: {route_id}")


def route_model_specs(route: dict[str, Any]) -> list[dict[str, Any]]:
    specs = []
    for stage, key in (
        ("visualEncoder", "visualEncoderOnnx"),
        ("visualEncoder", "visual_encoder_onnx_asset"),
        ("reranker", "rerankerOnnx"),
        ("reranker", "reranker_onnx_asset"),
        ("placement", "placementOnnx"),
    ):
        value = route.get(key)
        if value:
            spec = {"stage": stage, "onnx": str(value)}
            if stage == "visualEncoder":
                spec["kind"] = "visual_encoder"
                if isinstance(route.get("visualEncoderProviderOptions"), dict):
                    spec["provider_options"] = dict(route["visualEncoderProviderOptions"])
            if spec not in specs:
                specs.append(spec)
    return specs


def resolve_manifest_asset(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def smoke_rows_for_model(onnx_path: Path, smoke_input_path: str | Path | None, stage: str) -> list[list[float]]:
    if smoke_input_path:
        import json

        with Path(smoke_input_path).open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        rows = rows_from_smoke_document(document, stage)
        if rows:
            return rows
    width = onnx_input_width(onnx_path)
    return [[float((index % 7) + 1) / 10.0 for index in range(width)]]


def rows_from_smoke_document(document: Any, stage: str) -> list[list[float]]:
    if isinstance(document, dict):
        if isinstance(document.get("featureRows"), dict):
            rows = document["featureRows"].get(stage)
            if isinstance(rows, list):
                return normalize_smoke_rows(rows)
        for key in (stage, "features", "rows"):
            if isinstance(document.get(key), list):
                return normalize_smoke_rows(document[key])
    if isinstance(document, list):
        return normalize_smoke_rows(document)
    return []


def normalize_smoke_rows(rows: list[Any]) -> list[list[float]]:
    if not rows:
        return []
    if all(isinstance(value, (int, float)) for value in rows):
        return [[float(value) for value in rows]]
    normalized = []
    for row in rows:
        if isinstance(row, list) and all(isinstance(value, (int, float)) for value in row):
            normalized.append([float(value) for value in row])
    return normalized


def onnx_input_width(onnx_path: Path) -> int:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required to inspect ONNX input width") from exc
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if not inputs or len(inputs[0].shape) < 2:
        raise ValueError(f"Cannot determine feature width for {onnx_path}")
    width = inputs[0].shape[1]
    if width in (None, "None"):
        raise ValueError(f"Dynamic feature width is not supported for generated provider smoke input: {onnx_path}")
    return int(width)


def run_onnx_smoke(onnx_path: Path, rows: list[list[float]], provider: str, cpu_providers: list[str], tolerance: float) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort

    inputs = np.asarray(rows, dtype=np.float32)
    provider_start = perf_counter()
    provider_session = ort.InferenceSession(str(onnx_path), providers=[provider])
    provider_inputs = provider_session.get_inputs()
    provider_outputs = provider_session.get_outputs()
    if not provider_inputs or not provider_outputs:
        raise ValueError(f"ONNX model has no usable IO contract: {onnx_path}")
    provider_result = provider_session.run([provider_outputs[0].name], {provider_inputs[0].name: inputs})[0]
    provider_latency_ms = round((perf_counter() - provider_start) * 1000.0, 6)

    cpu_start = perf_counter()
    cpu_session = ort.InferenceSession(str(onnx_path), providers=cpu_providers)
    cpu_inputs = cpu_session.get_inputs()
    cpu_outputs = cpu_session.get_outputs()
    cpu_result = cpu_session.run([cpu_outputs[0].name], {cpu_inputs[0].name: inputs})[0]
    cpu_latency_ms = round((perf_counter() - cpu_start) * 1000.0, 6)

    actual_providers = list(provider_session.get_providers())
    used_provider = actual_providers[0] if actual_providers else None
    max_delta = float(np.max(np.abs(np.asarray(provider_result, dtype=np.float32) - np.asarray(cpu_result, dtype=np.float32))))
    return {
        "onnxPath": str(onnx_path),
        "requestedProvider": provider,
        "actualProviders": actual_providers,
        "usedProvider": used_provider,
        "cpuFallback": used_provider != provider,
        "latencyMs": provider_latency_ms,
        "cpuLatencyMs": cpu_latency_ms,
        "outputParity": {
            "maxDeltaVsCpu": max_delta,
            "withinTolerance": bool(max_delta <= tolerance),
            "tolerance": float(tolerance),
        },
    }


def run_visual_encoder_smoke(onnx_path: Path, provider: str, cpu_providers: list[str], provider_options: dict[str, Any] | None = None) -> dict[str, Any]:
    import json

    import numpy as np
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.enable_profiling = True
    session_options.profile_file_prefix = str(Path(os.environ.get("TMPDIR", "/tmp")) / "aetherflow_visual_encoder_provider_profile")
    input_tensor = visual_encoder_smoke_tensor(onnx_path)

    provider_start = perf_counter()
    provider_route = (provider, provider_options) if provider_options else provider
    provider_session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=unique_provider_list([provider_route, *cpu_providers]))
    provider_inputs = provider_session.get_inputs()
    provider_outputs = provider_session.get_outputs()
    if not provider_inputs or not provider_outputs:
        raise ValueError(f"ONNX visual encoder has no usable IO contract: {onnx_path}")
    provider_result = provider_session.run([provider_outputs[0].name], {provider_inputs[0].name: input_tensor})[0]
    provider_latency_ms = round((perf_counter() - provider_start) * 1000.0, 6)
    profile_path = provider_session.end_profiling()
    provider_profile = provider_profile_summary(profile_path, provider)

    cpu_start = perf_counter()
    cpu_session = ort.InferenceSession(str(onnx_path), providers=cpu_providers)
    cpu_inputs = cpu_session.get_inputs()
    cpu_outputs = cpu_session.get_outputs()
    cpu_result = cpu_session.run([cpu_outputs[0].name], {cpu_inputs[0].name: input_tensor})[0]
    cpu_latency_ms = round((perf_counter() - cpu_start) * 1000.0, 6)

    actual_providers = list(provider_session.get_providers())
    used_provider = actual_providers[0] if actual_providers else None
    provider_array = np.asarray(provider_result, dtype=np.float32).reshape(-1)
    cpu_array = np.asarray(cpu_result, dtype=np.float32).reshape(-1)
    max_delta = float(np.max(np.abs(provider_array - cpu_array)))
    mean_delta = float(np.mean(np.abs(provider_array - cpu_array)))
    cosine_similarity = cosine(provider_array, cpu_array)
    provider_ratio = provider_profile["providerEventRatio"]
    cpu_heavy = used_provider != provider or provider_ratio < _ACCELERATED_EVENT_RATIO_THRESHOLD
    within_tolerance = cosine_similarity >= _VISUAL_ENCODER_COSINE_THRESHOLD and not cpu_heavy
    return {
        "onnxPath": str(onnx_path),
        "requestedProvider": provider,
        "actualProviders": actual_providers,
        "usedProvider": used_provider,
        "cpuFallback": used_provider != provider or cpu_heavy,
        "latencyMs": provider_latency_ms,
        "cpuLatencyMs": cpu_latency_ms,
        "providerProfile": provider_profile,
        "outputParity": {
            "maxDeltaVsCpu": max_delta,
            "meanDeltaVsCpu": mean_delta,
            "cosineSimilarityVsCpu": cosine_similarity,
            "cosineThreshold": _VISUAL_ENCODER_COSINE_THRESHOLD,
            "withinTolerance": bool(within_tolerance),
            "tolerance": _VISUAL_ENCODER_COSINE_THRESHOLD,
        },
    }


def visual_encoder_smoke_tensor(onnx_path: Path):
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if not inputs:
        raise ValueError(f"Cannot inspect visual encoder input shape: {onnx_path}")
    shape = list(inputs[0].shape)
    if len(shape) != 4:
        raise ValueError(f"Visual encoder smoke expects a 4D BCHW input, got {shape}: {onnx_path}")
    dims = [int(value) if isinstance(value, int) and value > 0 else fallback for value, fallback in zip(shape, [1, 3, 224, 224])]
    values = np.linspace(-1.0, 1.0, int(np.prod(dims)), dtype=np.float32)
    return values.reshape(dims)


def provider_profile_summary(profile_path: str, provider: str) -> dict[str, Any]:
    import json

    events = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    provider_counts: dict[str, int] = {}
    for event in events:
        event_provider = (event.get("args") or {}).get("provider")
        if event_provider:
            provider_counts[str(event_provider)] = provider_counts.get(str(event_provider), 0) + 1
    provider_events = provider_counts.get(provider, 0)
    cpu_events = provider_counts.get("CPUExecutionProvider", 0)
    total_provider_events = sum(provider_counts.values())
    provider_ratio = float(provider_events / total_provider_events) if total_provider_events else 0.0
    return {
        "profilePath": profile_path,
        "providerEventCounts": provider_counts,
        "providerEventRatio": provider_ratio,
        "acceleratedEventRatioThreshold": _ACCELERATED_EVENT_RATIO_THRESHOLD,
        "cpuHeavy": bool(provider != "CPUExecutionProvider" and provider_ratio < _ACCELERATED_EVENT_RATIO_THRESHOLD),
    }


def cosine(left: Any, right: Any) -> float:
    import numpy as np

    left_array = np.asarray(left, dtype=np.float32).reshape(-1)
    right_array = np.asarray(right, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator == 0.0:
        return 1.0 if float(np.linalg.norm(left_array - right_array)) == 0.0 else 0.0
    return float(np.dot(left_array, right_array) / denominator)


def unique_provider_list(providers: list[Any]) -> list[Any]:
    unique = []
    for provider in providers:
        if provider not in unique:
            unique.append(provider)
    return unique


def validate_onnx_model(model_manifest_path: str | Path, model_manifest: dict[str, Any]) -> dict[str, Any]:
    """Load a model with ONNX Runtime and return its IO contract."""

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required; run with an existing local runtime that already provides it") from exc

    path = model_onnx_path(model_manifest_path, model_manifest)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = [
        {
            "name": item.name,
            "shape": list(item.shape),
            "type": item.type,
        }
        for item in session.get_inputs()
    ]
    outputs = [
        {
            "name": item.name,
            "shape": list(item.shape),
            "type": item.type,
        }
        for item in session.get_outputs()
    ]
    architecture = model_manifest.get("model_architecture", {})
    expected_input = architecture.get("input_name")
    expected_output = architecture.get("output_name")
    if expected_input and (not inputs or inputs[0]["name"] != expected_input):
        raise ValueError(f"Unexpected ONNX input name: expected {expected_input}, got {inputs[0]['name'] if inputs else 'none'}")
    if expected_output and (not outputs or outputs[0]["name"] != expected_output):
        raise ValueError(f"Unexpected ONNX output name: expected {expected_output}, got {outputs[0]['name'] if outputs else 'none'}")
    return {
        "onnx_path": str(path),
        "providers": session.get_providers(),
        "inputs": inputs,
        "outputs": outputs,
    }


def validate_reranker_onnx_model(reranker_model_path: str | Path, onnx_path: str | Path) -> dict[str, Any]:
    """Load a standalone reranker ONNX artifact and compare it to JSON weights."""

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime and numpy are required; run with an existing local runtime that already provides them") from exc

    import json

    model_path = Path(reranker_model_path)
    with model_path.open("r", encoding="utf-8") as handle:
        model = json.load(handle)
    if not isinstance(model, dict):
        raise ValueError(f"Expected reranker model object at {model_path}")
    weights = model.get("weights")
    if not isinstance(weights, list) or not weights:
        raise ValueError(f"reranker model {model_path} does not contain weights")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = [
        {
            "name": item.name,
            "shape": list(item.shape),
            "type": item.type,
        }
        for item in session.get_inputs()
    ]
    outputs = [
        {
            "name": item.name,
            "shape": list(item.shape),
            "type": item.type,
        }
        for item in session.get_outputs()
    ]
    if not inputs or inputs[0]["name"] != "features":
        raise ValueError(f"Unexpected reranker ONNX input name: {inputs[0]['name'] if inputs else 'none'}")
    if not outputs or outputs[0]["name"] != "scores":
        raise ValueError(f"Unexpected reranker ONNX output name: {outputs[0]['name'] if outputs else 'none'}")
    input_width = inputs[0]["shape"][1] if len(inputs[0]["shape"]) > 1 else None
    if input_width not in (None, "None") and int(input_width) != len(weights):
        raise ValueError(f"Unexpected reranker ONNX input width: expected {len(weights)}, got {input_width}")

    feature_row = np.linspace(0.0, 0.9, len(weights), dtype=np.float32).reshape(1, -1)
    expected = feature_row @ np.asarray(weights, dtype=np.float32).reshape(-1, 1) + np.asarray([float(model.get("bias", 0.0))], dtype=np.float32)
    observed = session.run(None, {"features": feature_row})[0]
    max_abs_delta = float(np.max(np.abs(observed - expected)))
    if observed.shape != (1, 1):
        raise ValueError(f"Unexpected reranker ONNX output shape for smoke input: {observed.shape}")
    if max_abs_delta > 1e-5:
        raise ValueError(f"Reranker ONNX output differs from JSON weights by {max_abs_delta}")

    return {
        "reranker_model_path": str(model_path),
        "onnx_path": str(onnx_path),
        "providers": session.get_providers(),
        "inputs": inputs,
        "outputs": outputs,
        "feature_count": len(weights),
        "smoke_output": observed.tolist(),
        "expected_output": expected.tolist(),
        "max_abs_delta": max_abs_delta,
    }
