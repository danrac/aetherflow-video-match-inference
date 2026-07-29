"""Correctness tests for persistent visual-embedding reuse."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aetherflow_video_match_inference.visual_encoder import (
    EMBEDDING_CACHE_SCHEMA_ID,
    EMBEDDING_CACHE_SCHEMA_VERSION,
    embedding_cache_identity,
    embedding_cache_path,
    load_cached_embedding,
    lookup_cached_embedding,
    store_cached_embedding,
)


class FakeImage:
    def __init__(self, payload: bytes, size: tuple[int, int] = (2, 2)) -> None:
        self.payload = payload
        self.size = size

    def convert(self, mode: str):
        if mode != "RGB":
            raise ValueError(mode)
        return self

    def tobytes(self) -> bytes:
        return self.payload


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="aetherflow-embedding-cache-")
        self.root = Path(self.temporary.name)
        self.cache_root = self.root / "cache"
        self.model = self.root / "visual_encoder.onnx"
        self.model.write_bytes(b"model-v1")
        self.media = self.root / "source.mov"
        self.media.write_bytes(b"media-v1")
        self.image = FakeImage(b"\x01\x02\x03" * 4)
        self.context = {
            "media": {"path": str(self.media)},
            "frameSelection": {
                "frameIndex": 42,
                "frameRate": 30.0,
                "cropIndex": 0,
            },
            "runtime": {
                "requestedProviders": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
                "effectiveProviders": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
            },
            "inference": {
                "inputName": "pixel_values",
                "outputName": "image_embeds",
            },
        }
        self.environment = patch.dict(
            os.environ,
            {"AETHERFLOW_VIDEO_MATCH_VISUAL_CACHE_ROOT": str(self.cache_root)},
            clear=False,
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        self.temporary.cleanup()

    def test_identical_identity_reuses_embedding(self) -> None:
        embedding = (0.25, 0.5, 0.75)

        store_cached_embedding(str(self.model), self.image, embedding, self.context)
        lookup = lookup_cached_embedding(str(self.model), self.image, cache_context=self.context)

        self.assertTrue(lookup.hit)
        self.assertEqual(lookup.reason, "identity_match")
        self.assertEqual(lookup.embedding, embedding)
        self.assertEqual(load_cached_embedding(str(self.model), self.image, self.context), embedding)

    def test_cache_document_is_versioned_and_does_not_store_media_path_or_pixels(self) -> None:
        store_cached_embedding(str(self.model), self.image, (0.25, 0.5), self.context)

        cache_path = embedding_cache_path(str(self.model), self.image, self.context)
        document = json.loads(cache_path.read_text(encoding="utf-8"))
        serialized = cache_path.read_text(encoding="utf-8")

        self.assertEqual(document["schemaId"], EMBEDDING_CACHE_SCHEMA_ID)
        self.assertEqual(document["schemaVersion"], EMBEDDING_CACHE_SCHEMA_VERSION)
        self.assertNotIn(str(self.media), serialized)
        self.assertNotIn(self.image.payload.hex(), serialized)

    def test_media_modification_invalidates_embedding(self) -> None:
        store_cached_embedding(str(self.model), self.image, (0.25, 0.5), self.context)
        original = self.media.stat()
        self.media.write_bytes(b"media-v2-longer")
        os.utime(self.media, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000))

        lookup = lookup_cached_embedding(str(self.model), self.image, cache_context=self.context)

        self.assertFalse(lookup.hit)
        self.assertEqual(lookup.reason, "media_identity_changed")

    def test_frame_selection_change_does_not_reuse_embedding(self) -> None:
        store_cached_embedding(str(self.model), self.image, (0.25, 0.5), self.context)
        changed = {
            **self.context,
            "frameSelection": {
                **self.context["frameSelection"],
                "frameIndex": 43,
            },
        }

        lookup = lookup_cached_embedding(str(self.model), self.image, cache_context=changed)

        self.assertFalse(lookup.hit)
        self.assertEqual(lookup.reason, "not_cached")

    def test_preprocessing_change_invalidates_embedding(self) -> None:
        context = {**self.context, "preprocessing": {"samplingPolicy": "baseline"}}
        store_cached_embedding(str(self.model), self.image, (0.25, 0.5), context)
        changed = {**context, "preprocessing": {"samplingPolicy": "alternate"}}

        lookup = lookup_cached_embedding(str(self.model), self.image, cache_context=changed)

        self.assertFalse(lookup.hit)
        self.assertEqual(lookup.reason, "preprocessing_changed")

    def test_model_content_change_invalidates_embedding(self) -> None:
        store_cached_embedding(str(self.model), self.image, (0.25, 0.5), self.context)
        original = self.model.stat()
        self.model.write_bytes(b"model-v2")
        os.utime(self.model, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000))

        lookup = lookup_cached_embedding(str(self.model), self.image, cache_context=self.context)

        self.assertFalse(lookup.hit)
        self.assertEqual(lookup.reason, "model_identity_changed")

    def test_runtime_provider_change_invalidates_embedding(self) -> None:
        store_cached_embedding(str(self.model), self.image, (0.25, 0.5), self.context)
        changed = {
            **self.context,
            "runtime": {
                "requestedProviders": ["CPUExecutionProvider"],
                "effectiveProviders": ["CPUExecutionProvider"],
            },
        }

        lookup = lookup_cached_embedding(str(self.model), self.image, cache_context=changed)

        self.assertFalse(lookup.hit)
        self.assertEqual(lookup.reason, "runtime_configuration_changed")

    def test_inference_configuration_change_invalidates_embedding(self) -> None:
        store_cached_embedding(str(self.model), self.image, (0.25, 0.5), self.context)
        changed = {
            **self.context,
            "inference": {
                **self.context["inference"],
                "outputName": "alternate_embeds",
            },
        }

        lookup = lookup_cached_embedding(str(self.model), self.image, cache_context=changed)

        self.assertFalse(lookup.hit)
        self.assertEqual(lookup.reason, "inference_configuration_changed")

    def test_decoded_pixels_change_invalidates_embedding_even_when_media_stat_is_unchanged(self) -> None:
        store_cached_embedding(str(self.model), self.image, (0.25, 0.5), self.context)
        changed_image = FakeImage(b"\x09\x08\x07" * 4)

        lookup = lookup_cached_embedding(str(self.model), changed_image, cache_context=self.context)

        self.assertFalse(lookup.hit)
        self.assertEqual(lookup.reason, "decoded_pixels_changed")

    def test_corrupt_entry_fails_closed(self) -> None:
        path = embedding_cache_path(str(self.model), self.image, self.context)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{invalid", encoding="utf-8")

        lookup = lookup_cached_embedding(str(self.model), self.image, cache_context=self.context)

        self.assertFalse(lookup.hit)
        self.assertEqual(lookup.state, "invalid")
        self.assertEqual(lookup.reason, "corrupt_entry")

    def test_identity_covers_all_required_dimensions(self) -> None:
        identity = embedding_cache_identity(str(self.model), self.image, self.context)

        self.assertEqual(
            set(identity["componentDigests"]),
            {
                "media",
                "frameSelection",
                "decodedImage",
                "preprocessing",
                "model",
                "runtime",
                "inference",
            },
        )
        self.assertEqual(identity["media"]["sizeBytes"], self.media.stat().st_size)
        self.assertEqual(identity["frameSelection"]["frameIndex"], 42)
        self.assertEqual(identity["model"]["sha256"], self._sha256(self.model))

    @staticmethod
    def _sha256(path: Path) -> str:
        import hashlib

        return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
