"""Optional ONNX Runtime loading boundary."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


_LINEAR_SESSION_CACHE: dict[tuple[str, tuple[str, ...]], Any] = {}


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
