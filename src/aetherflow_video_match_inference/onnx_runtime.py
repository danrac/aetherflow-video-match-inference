"""Optional ONNX Runtime loading boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def model_onnx_path(model_manifest_path: str | Path, model_manifest: dict[str, Any]) -> Path:
    onnx_path = Path(str(model_manifest["onnx_path"]))
    if onnx_path.is_absolute():
        return onnx_path
    return Path(model_manifest_path).parent / onnx_path


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
