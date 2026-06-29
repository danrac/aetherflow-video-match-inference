"""Feature-manifest matching tests."""

from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from aetherflow_video_match_inference.adapters import to_host_payload
from aetherflow_video_match_inference.engine import MatchRequest, match
from aetherflow_video_match_inference.features import visual_distance
from aetherflow_video_match_inference.interchange import export_after_effects_extendscript, export_cep_json, export_edit_json, export_edl, export_premiere_json, frames_to_timecode
from aetherflow_video_match_inference.onnx_runtime import validate_onnx_model

CONTRACTS_ROOT = Path(__file__).resolve().parents[2] / "contracts"
FULL_EVAL_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run-edge-case-full-eval.py"


class FeatureMatchingTests(unittest.TestCase):
    def test_feature_manifest_match_uses_feature_confidence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aetherflow-inference-features-") as temp_dir:
            root = Path(temp_dir)
            model_manifest = write_model_manifest(root)
            reference_features = write_feature_manifest(root / "reference.features.json", "reference", [100.0, 120.0, 140.0], motion=3.0)
            source_features = write_feature_manifest(root / "source.features.json", "source", [101.0, 121.0, 141.0], motion=3.5)

            result = match(
                MatchRequest(
                    reference_path="/tmp/reference.mp4",
                    source_paths=("/tmp/source.mp4",),
                    model_manifest_path=str(model_manifest),
                    reference_feature_manifest_path=str(reference_features),
                    source_feature_manifest_paths=(str(source_features),),
                )
            )

            self.assertEqual(result["matches"][0]["reconstruction"]["operation"], "feature_manifest_match")
            self.assertGreater(result["matches"][0]["confidence"], 0.9)
            self.assertGreater(result["matches"][0]["reconstruction"]["parameters"]["distance"], 1.7)
            self.assertEqual(result["reference"]["duration_frames"], 24)

    def test_feature_manifest_match_reconstructs_source_window_ranges(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aetherflow-inference-source-windows-") as temp_dir:
            root = Path(temp_dir)
            model_manifest = write_model_manifest(root)
            reference_features = write_feature_manifest(root / "reference.features.json", "reference", [100.0, 120.0, 140.0], motion=3.0)
            first_source_features = write_feature_manifest(
                root / "first-source.features.json",
                "first-source",
                [101.0, 121.0, 141.0],
                source_window={"source_in": 10, "source_out": 18},
            )
            second_source_features = write_feature_manifest(
                root / "second-source.features.json",
                "second-source",
                [102.0, 122.0, 142.0],
                source_window={"source_in": 40, "source_out": 56},
            )

            result = match(
                MatchRequest(
                    reference_path="/tmp/reference.mp4",
                    source_paths=("/tmp/first-source.mp4", "/tmp/second-source.mp4"),
                    model_manifest_path=str(model_manifest),
                    reference_feature_manifest_path=str(reference_features),
                    source_feature_manifest_paths=(str(first_source_features), str(second_source_features)),
                )
            )

            self.assertEqual(result["matches"][0]["reference_in"], 0)
            self.assertEqual(result["matches"][0]["reference_out"], 8)
            self.assertEqual(result["matches"][0]["source_in"], 10)
            self.assertEqual(result["matches"][0]["source_out"], 18)
            self.assertEqual(result["matches"][1]["reference_in"], 8)
            self.assertEqual(result["matches"][1]["reference_out"], 24)
            self.assertEqual(result["matches"][1]["source_in"], 40)
            self.assertEqual(result["matches"][1]["source_out"], 56)

    def test_match_falls_back_without_features(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aetherflow-inference-placeholder-") as temp_dir:
            model_manifest = write_model_manifest(Path(temp_dir))

            result = match(
                MatchRequest(
                    reference_path="/tmp/reference.mp4",
                    source_paths=("/tmp/source.mp4",),
                    model_manifest_path=str(model_manifest),
                )
            )

            self.assertEqual(result["matches"][0]["reconstruction"]["operation"], "placeholder_match")

    def test_validate_onnx_model_when_onnxruntime_is_available(self) -> None:
        if importlib.util.find_spec("onnxruntime") is None:
            self.skipTest("onnxruntime is not available in this Python runtime")
        with tempfile.TemporaryDirectory(prefix="aetherflow-inference-onnx-") as temp_dir:
            root = Path(temp_dir)
            (root / "model.onnx").write_bytes(identity_onnx_model_bytes())
            model_manifest = write_model_manifest(root)

            report = validate_onnx_model(model_manifest, json.loads(model_manifest.read_text(encoding="utf-8")))

            self.assertEqual(report["inputs"][0]["name"], "features")
            self.assertEqual(report["outputs"][0]["name"], "scores")

    def test_v3_feature_distance_uses_scene_and_flow_stats(self) -> None:
        reference = {
            "features": [
                {
                    "mean_rgb": [100.0, 120.0, 140.0],
                    "mean_luma": 118.0,
                    "edge_density": 0.10,
                    "scene_change_score": 2.0,
                    "mean_absdiff_from_previous": 2.0,
                    "optical_flow": {"mean_magnitude": 0.25, "mean_dx": 0.10, "mean_dy": 0.02},
                }
            ]
        }
        source_without_v3_delta = {
            "features": [
                {
                    "mean_rgb": [100.0, 120.0, 140.0],
                    "mean_luma": 118.0,
                    "edge_density": 0.10,
                    "scene_change_score": 2.0,
                    "mean_absdiff_from_previous": 2.0,
                    "optical_flow": {"mean_magnitude": 0.25, "mean_dx": 0.10, "mean_dy": 0.02},
                }
            ]
        }
        source_with_v3_delta = {
            "features": [
                {
                    "mean_rgb": [100.0, 120.0, 140.0],
                    "mean_luma": 130.0,
                    "edge_density": 0.40,
                    "scene_change_score": 7.0,
                    "mean_absdiff_from_previous": 2.0,
                    "optical_flow": {"mean_magnitude": 1.25, "mean_dx": 0.50, "mean_dy": -0.20},
                }
            ]
        }

        self.assertEqual(visual_distance(reference, source_without_v3_delta), 0.0)
        self.assertGreater(visual_distance(reference, source_with_v3_delta), 20.0)

    def test_feature_distance_accepts_reversed_temporal_signature_and_flow_direction(self) -> None:
        reference = {
            "features": [{"mean_rgb": [100.0, 120.0, 140.0]}],
            "temporal_signature": [
                [0.10, 0.02, 0.01, 0.00, 0.03, 0.20, 0.30, 0.40],
                [0.80, 0.04, 0.03, 0.02, 0.05, 0.70, 0.60, 0.50],
            ],
        }
        source_reversed = {
            "features": [{"mean_rgb": [100.0, 120.0, 140.0]}],
            "temporal_signature": list(reversed(reference["temporal_signature"])),
        }
        reference_with_flow = {"features": [{"mean_rgb": [1.0, 1.0, 1.0], "optical_flow": {"mean_magnitude": 2.0, "mean_dx": 0.5, "mean_dy": -0.25}}]}
        source_with_opposite_flow = {"features": [{"mean_rgb": [1.0, 1.0, 1.0], "optical_flow": {"mean_magnitude": 2.0, "mean_dx": -0.5, "mean_dy": 0.25}}]}

        self.assertEqual(visual_distance(reference, source_reversed), 0.0)
        self.assertEqual(visual_distance(reference_with_flow, source_with_opposite_flow), 0.0)
        self.assertGreater(visual_distance(reference, source_reversed, allow_temporal_reverse=False), 0.0)

    def test_full_eval_reranker_uses_temporal_hybrid_routing(self) -> None:
        module = load_full_eval_script()
        model = {
            "weights": [10.0 for _ in module.FEATURE_NAMES],
            "bias": 0.0,
            "routing": {
                "baseline_protected_transform_types": ["scale_position"],
                "learned_reranker_applies_to": ["reverse", "simple_cut"],
            },
        }
        components = {name: 1.0 for name in module.FEATURE_NAMES}
        components["family_penalty"] = 0.0
        components["window_count"] = 1.0

        simple_cut_score = module.reranker_distance(components, {"simple_cut"}, 123.0, model)
        scale_score = module.reranker_distance(components, {"scale_position"}, 123.0, model)
        split_screen_score = module.reranker_distance(components, {"split_screen"}, 123.0, model)

        self.assertNotEqual(simple_cut_score, 123.0)
        self.assertEqual(scale_score, 123.0)
        self.assertEqual(split_screen_score, 123.0)

    def test_host_payload_includes_timeline_edits(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aetherflow-inference-host-") as temp_dir:
            model_manifest = write_model_manifest(Path(temp_dir))
            result = match(
                MatchRequest(
                    reference_path="/tmp/reference.mp4",
                    source_paths=("/tmp/source.mp4",),
                    model_manifest_path=str(model_manifest),
                )
            )

            payload = to_host_payload(result, "aetherflow")

            self.assertEqual(payload["schema_version"], "0.1.0")
            self.assertEqual(payload["host"], "aetherflow")
            self.assertEqual(payload["timeline"]["edit_count"], 1)
            self.assertEqual(payload["timeline"]["edits"][0]["reference_in_seconds"], 0.0)
            self.assertEqual(payload["timeline"]["edits"][0]["reference_out_seconds"], 5.0)

    def test_host_payload_matches_contract_schema(self) -> None:
        schema_path = CONTRACTS_ROOT / "schemas" / "host_payload.schema.json"
        if not schema_path.exists():
            self.skipTest("contracts checkout is not available")
        with tempfile.TemporaryDirectory(prefix="aetherflow-inference-host-schema-") as temp_dir:
            model_manifest = write_model_manifest(Path(temp_dir))
            result = match(
                MatchRequest(
                    reference_path="/tmp/reference.mp4",
                    source_paths=("/tmp/source.mp4",),
                    model_manifest_path=str(model_manifest),
                )
            )
            payload = to_host_payload(result, "aetherflow")
            schema = json.loads(schema_path.read_text(encoding="utf-8"))

            errors = validate_schema_subset(payload, schema)

            self.assertEqual(errors, [])

    def test_interchange_exports_from_host_payload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aetherflow-inference-interchange-") as temp_dir:
            root = Path(temp_dir)
            model_manifest = write_model_manifest(root)
            result = match(
                MatchRequest(
                    reference_path="/tmp/reference.mp4",
                    source_paths=("/tmp/source.mp4",),
                    model_manifest_path=str(model_manifest),
                )
            )
            payload = to_host_payload(result, "aetherflow")

            json_path = export_edit_json(payload, root / "edits.json")
            cep_path = export_cep_json(payload, root / "aetherflow_cep.json")
            premiere_path = export_premiere_json(payload, root / "premiere.json")
            edl_path = export_edl(payload, root / "timeline.edl")
            jsx_path = export_after_effects_extendscript(payload, root / "aetherflow_import.jsx")

            self.assertTrue(json_path.exists())
            cep_json = json.loads(cep_path.read_text(encoding="utf-8"))
            self.assertEqual(cep_json["adapter"], "aetherflow-cep-json")
            self.assertEqual(cep_json["host"], "aetherflow-cep")
            self.assertEqual(cep_json["timeline"]["layers"][0]["layer_id"], "edit-0001")
            self.assertEqual(cep_json["timeline"]["layers"][0]["start_seconds"], 0.0)
            self.assertIn("place_layers_by_seconds", cep_json["instructions"])
            cep_schema_path = CONTRACTS_ROOT / "schemas" / "cep_handoff.schema.json"
            if cep_schema_path.exists():
                self.assertEqual(validate_schema_subset(cep_json, json.loads(cep_schema_path.read_text(encoding="utf-8"))), [])
            premiere_json = json.loads(premiere_path.read_text(encoding="utf-8"))
            self.assertEqual(premiere_json["adapter"], "premiere-pro-json")
            self.assertEqual(premiere_json["host"], "premiere-pro")
            self.assertEqual(premiere_json["clips"][0]["clip_id"], "edit-0001")
            self.assertEqual(premiere_json["clips"][0]["target_video_track"], 0)
            self.assertIn("insert_clips_by_frames", premiere_json["instructions"])
            premiere_schema_path = CONTRACTS_ROOT / "schemas" / "premiere_handoff.schema.json"
            if premiere_schema_path.exists():
                self.assertEqual(validate_schema_subset(premiere_json, json.loads(premiere_schema_path.read_text(encoding="utf-8"))), [])
            self.assertIn("TITLE: AETHERFLOW_VIDEO_MATCH", edl_path.read_text(encoding="utf-8"))
            self.assertIn("00:00:05:00", edl_path.read_text(encoding="utf-8"))
            jsx = jsx_path.read_text(encoding="utf-8")
            self.assertIn("app.beginUndoGroup", jsx)
            self.assertIn("project.items.addComp", jsx)
            self.assertIn("comp.layers.add", jsx)

    def test_frames_to_timecode_uses_rounded_fps(self) -> None:
        self.assertEqual(frames_to_timecode(120, 24.0), "00:00:05:00")


def write_model_manifest(root: Path) -> Path:
    path = root / "model_manifest.json"
    write_json(
        path,
        {
            "schema_version": "0.1.0",
            "model_id": "test-model",
            "model_version": "v0001",
            "created_at": "2026-06-27T00:00:00+00:00",
            "onnx_path": "model.onnx",
            "model_architecture": {
                "type": "identity_baseline",
                "input_name": "features",
                "output_name": "scores",
            },
            "trained_on_dataset": {"dataset_id": "test", "dataset_version": "v0001"},
            "metrics": {},
        },
    )
    return path


def write_feature_manifest(
    path: Path,
    clip_id: str,
    mean_rgb: list[float],
    motion: float | None = None,
    source_window: dict | None = None,
) -> Path:
    frame = {
        "frame_index": 0,
        "mean_rgb": mean_rgb,
        "std_rgb": [1.0, 1.0, 1.0],
    }
    if motion is not None:
        frame["mean_absdiff_from_previous"] = motion
    document = {
        "clip_id": clip_id,
        "checksum": "abc123",
        "feature_version": "opencv-visual-stats-v2",
        "duration_frames": 24,
        "fps": 24.0,
        "sampled_frame_count": 1,
        "features": [frame],
    }
    if source_window is not None:
        document["source_window"] = source_window
    write_json(path, document)
    return path


def write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_full_eval_script():
    spec = importlib.util.spec_from_file_location("aetherflow_full_eval_test_module", FULL_EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load full eval script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity_onnx_model_bytes() -> bytes:
    def varint(value: int) -> bytes:
        output = []
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                output.append(byte | 0x80)
            else:
                output.append(byte)
                break
        return bytes(output)

    def key(field_number: int, wire_type: int) -> bytes:
        return varint((field_number << 3) | wire_type)

    def int_field(field_number: int, value: int) -> bytes:
        return key(field_number, 0) + varint(value)

    def string_field(field_number: int, value: str) -> bytes:
        encoded = value.encode("utf-8")
        return key(field_number, 2) + varint(len(encoded)) + encoded

    def message_field(field_number: int, value: bytes) -> bytes:
        return key(field_number, 2) + varint(len(value)) + value

    def dimension(value: int | None = None) -> bytes:
        return b"" if value is None else int_field(1, value)

    def tensor_shape() -> bytes:
        return message_field(1, dimension(None)) + message_field(1, dimension(4))

    def tensor_type() -> bytes:
        return int_field(1, 1) + message_field(2, tensor_shape())

    def type_proto() -> bytes:
        return message_field(1, tensor_type())

    def value_info(name: str) -> bytes:
        return string_field(1, name) + message_field(2, type_proto())

    node = string_field(1, "features") + string_field(2, "scores") + string_field(3, "identity_baseline") + string_field(4, "Identity")
    graph = message_field(1, node) + string_field(2, "aetherflow_video_match_baseline") + message_field(11, value_info("features")) + message_field(12, value_info("scores"))
    opset = string_field(1, "") + int_field(2, 13)
    return int_field(1, 8) + string_field(2, "aetherflow-video-match-training") + message_field(7, graph) + message_field(8, opset)


def validate_schema_subset(document, schema: dict, path: str = "$") -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type and not matches_type(document, expected_type):
        return [f"{path}: expected {expected_type}"]

    if expected_type == "object":
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
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(document):
                errors.extend(validate_schema_subset(item, item_schema, f"{path}[{index}]"))

    if isinstance(document, (int, float)) and not isinstance(document, bool):
        if schema.get("minimum") is not None and document < schema["minimum"]:
            errors.append(f"{path}: expected >= {schema['minimum']}")
        if schema.get("maximum") is not None and document > schema["maximum"]:
            errors.append(f"{path}: expected <= {schema['maximum']}")
        if schema.get("exclusiveMinimum") is not None and document <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: expected > {schema['exclusiveMinimum']}")
    return errors


def matches_type(value, expected_type: str) -> bool:
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
    return True


if __name__ == "__main__":
    unittest.main()
