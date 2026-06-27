"""Feature-manifest matching tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aetherflow_video_match_inference.engine import MatchRequest, match


class FeatureMatchingTests(unittest.TestCase):
    def test_feature_manifest_match_uses_feature_confidence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aetherflow-inference-features-") as temp_dir:
            root = Path(temp_dir)
            model_manifest = write_model_manifest(root)
            reference_features = write_feature_manifest(root / "reference.features.json", "reference", [100.0, 120.0, 140.0])
            source_features = write_feature_manifest(root / "source.features.json", "source", [101.0, 121.0, 141.0])

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
            self.assertEqual(result["reference"]["duration_frames"], 24)

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
            "trained_on_dataset": {"dataset_id": "test", "dataset_version": "v0001"},
            "metrics": {},
        },
    )
    return path


def write_feature_manifest(path: Path, clip_id: str, mean_rgb: list[float]) -> Path:
    write_json(
        path,
        {
            "clip_id": clip_id,
            "checksum": "abc123",
            "feature_version": "opencv-color-stats-v1",
            "duration_frames": 24,
            "fps": 24.0,
            "sampled_frame_count": 1,
            "features": [
                {
                    "frame_index": 0,
                    "mean_rgb": mean_rgb,
                    "std_rgb": [1.0, 1.0, 1.0],
                }
            ],
        },
    )
    return path


def write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
