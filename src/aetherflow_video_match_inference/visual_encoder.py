"""Optional visual-encoder frame scoring for source-window identity."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .storage import data_root


CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


class VisualEncoderUnavailable(RuntimeError):
    """Raised when the optional visual encoder cannot be loaded."""


class VisualEncoderScorer:
    def __init__(self, onnx_path: str | Path, providers: tuple[str, ...] | None = None) -> None:
        self.onnx_path = str(onnx_path)
        self.session = visual_encoder_session(self.onnx_path, providers or ("CoreMLExecutionProvider", "CPUExecutionProvider"))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def encode(self, image: Any) -> tuple[float, ...]:
        import numpy as np

        cached = load_cached_embedding(self.onnx_path, image)
        if cached is not None:
            return cached
        array = preprocess_clip_image(image, np)
        output = self.session.run([self.output_name], {self.input_name: array})[0]
        vector = output.reshape(-1).astype("float32")
        norm = float(np.linalg.norm(vector))
        if norm > 1e-12:
            vector = vector / norm
        embedding = tuple(float(value) for value in vector)
        store_cached_embedding(self.onnx_path, image, embedding)
        return embedding

    def cosine_distance(self, reference_embedding: tuple[float, ...], image: Any) -> float:
        source_embedding = self.encode(image)
        return cosine_distance(reference_embedding, source_embedding)


@lru_cache(maxsize=8)
def visual_encoder_session(onnx_path: str, providers: tuple[str, ...]):
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise VisualEncoderUnavailable("onnxruntime is not available") from exc
    path = Path(onnx_path)
    if not path.exists():
        raise VisualEncoderUnavailable(f"visual encoder does not exist: {onnx_path}")
    available = set(ort.get_available_providers())
    provider_list = [provider for provider in providers if provider in available]
    if "CPUExecutionProvider" not in provider_list:
        provider_list.append("CPUExecutionProvider")
    try:
        return ort.InferenceSession(str(path), providers=provider_list)
    except Exception as exc:  # pragma: no cover - provider-specific
        if provider_list != ["CPUExecutionProvider"] and "CPUExecutionProvider" in available:
            try:
                return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            except Exception:
                pass
        raise VisualEncoderUnavailable(str(exc)) from exc


def preprocess_clip_image(image: Any, np: Any):
    from PIL import Image, ImageOps

    resized = ImageOps.fit(image.convert("RGB"), (224, 224), method=Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.asarray(CLIP_IMAGE_MEAN, dtype=np.float32)
    std = np.asarray(CLIP_IMAGE_STD, dtype=np.float32)
    array = (array - mean) / std
    return np.transpose(array, (2, 0, 1))[None, :, :, :].astype("float32")


def cosine_distance(reference_embedding: tuple[float, ...], source_embedding: tuple[float, ...]) -> float:
    if len(reference_embedding) != len(source_embedding) or not reference_embedding:
        return 1.0
    dot = sum(float(a) * float(b) for a, b in zip(reference_embedding, source_embedding, strict=False))
    return 1.0 - max(-1.0, min(1.0, dot))


def visual_encoder_cache_root() -> Path:
    override = os.environ.get("AETHERFLOW_VIDEO_MATCH_VISUAL_CACHE_ROOT")
    if override:
        return Path(override).expanduser()
    return data_root() / "cache" / "visual_encoder"


def embedding_cache_path(onnx_path: str, image: Any) -> Path:
    model_hash = hashlib.sha256(str(Path(onnx_path).resolve()).encode("utf-8")).hexdigest()[:16]
    image_rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(str(image_rgb.size).encode("utf-8"))
    digest.update(image_rgb.tobytes())
    return visual_encoder_cache_root() / model_hash / f"{digest.hexdigest()}.json"


def load_cached_embedding(onnx_path: str, image: Any) -> tuple[float, ...] | None:
    path = embedding_cache_path(onnx_path, image)
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        values = document.get("embedding")
        if isinstance(values, list) and values:
            return tuple(float(value) for value in values)
    except Exception:
        return None
    return None


def store_cached_embedding(onnx_path: str, image: Any, embedding: tuple[float, ...]) -> None:
    path = embedding_cache_path(onnx_path, image)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"embedding": list(embedding)}, separators=(",", ":")), encoding="utf-8")
    except Exception:
        return
