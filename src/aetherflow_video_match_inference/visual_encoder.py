"""Optional visual-encoder frame scoring for source-window identity."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
import platform
from pathlib import Path
from time import perf_counter
from typing import Any

from .storage import data_root


CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
EMBEDDING_CACHE_SCHEMA_ID = "aetherflow.video-match-embedding-cache-entry"
EMBEDDING_CACHE_SCHEMA_VERSION = "1.0.0"
PREPROCESSING_CONTRACT_ID = "aetherflow.clip-image-preprocessing"
PREPROCESSING_CONTRACT_VERSION = "1.0.0"
INFERENCE_CONTRACT_ID = "aetherflow.visual-embedding-inference"
INFERENCE_CONTRACT_VERSION = "1.0.0"


class VisualEncoderUnavailable(RuntimeError):
    """Raised when the optional visual encoder cannot be loaded."""


@dataclass(frozen=True)
class EmbeddingCacheLookup:
    embedding: tuple[float, ...] | None
    state: str
    reason: str
    identity_digest: str
    duration_ms: float

    @property
    def hit(self) -> bool:
        return self.embedding is not None and self.state == "hit"


class VisualEncoderScorer:
    def __init__(
        self,
        onnx_path: str | Path,
        providers: tuple[str, ...] | None = None,
        cache_event_sink=None,
    ) -> None:
        self.onnx_path = str(onnx_path)
        self.requested_providers = providers or ("CoreMLExecutionProvider", "CPUExecutionProvider")
        self.session = visual_encoder_session(self.onnx_path, self.requested_providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.cache_event_sink = cache_event_sink if callable(cache_event_sink) else None

    def encode(self, image: Any, *, cache_context: dict[str, Any] | None = None) -> tuple[float, ...]:
        import numpy as np

        runtime_context = self._runtime_context(cache_context)
        lookup = lookup_cached_embedding(
            self.onnx_path,
            image,
            cache_context=runtime_context,
        )
        self._emit_cache_event({
            "event": "lookup",
            "state": lookup.state,
            "reason": lookup.reason,
            "durationMs": lookup.duration_ms,
            "identityDigest": lookup.identity_digest,
        })
        if lookup.hit:
            self._emit_cache_event({
                "event": "reuse",
                "state": "hit",
                "reason": "identity_match",
                "durationMs": lookup.duration_ms,
                "identityDigest": lookup.identity_digest,
            })
            return lookup.embedding
        started = perf_counter()
        embedding = generate_embedding(self.session, self.input_name, self.output_name, image, np)
        generation_ms = (perf_counter() - started) * 1000.0
        store_cached_embedding(
            self.onnx_path,
            image,
            embedding,
            cache_context=runtime_context,
        )
        self._emit_cache_event({
            "event": "generation",
            "state": "generated",
            "reason": lookup.reason,
            "durationMs": generation_ms,
            "identityDigest": lookup.identity_digest,
        })
        return embedding

    def cosine_distance(
        self,
        reference_embedding: tuple[float, ...],
        image: Any,
        *,
        cache_context: dict[str, Any] | None = None,
    ) -> float:
        source_embedding = self.encode(image, cache_context=cache_context)
        return cosine_distance(reference_embedding, source_embedding)

    def _runtime_context(self, cache_context: dict[str, Any] | None) -> dict[str, Any]:
        context = dict(cache_context or {})
        context["runtime"] = {
            **(context.get("runtime") if isinstance(context.get("runtime"), dict) else {}),
            "requestedProviders": list(self.requested_providers),
            "effectiveProviders": list(self.session.get_providers()),
        }
        context["inference"] = {
            **(context.get("inference") if isinstance(context.get("inference"), dict) else {}),
            "inputName": self.input_name,
            "outputName": self.output_name,
        }
        return context

    def _emit_cache_event(self, event: dict[str, Any]) -> None:
        if self.cache_event_sink is None:
            return
        try:
            self.cache_event_sink(dict(event))
        except Exception:
            pass


def generate_embedding(session: Any, input_name: str, output_name: str, image: Any, np: Any) -> tuple[float, ...]:
    array = preprocess_clip_image(image, np)
    output = session.run([output_name], {input_name: array})[0]
    vector = output.reshape(-1).astype("float32")
    norm = float(np.linalg.norm(vector))
    if norm > 1e-12:
        vector = vector / norm
    return tuple(float(value) for value in vector)


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


def embedding_cache_identity(
    onnx_path: str,
    image: Any,
    cache_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = cache_context if isinstance(cache_context, dict) else {}
    image_rgb = image.convert("RGB")
    image_digest = hashlib.sha256()
    image_digest.update(str(image_rgb.size).encode("utf-8"))
    image_digest.update(image_rgb.tobytes())
    model = model_artifact_identity(onnx_path, context.get("model"))
    media = media_cache_identity(context.get("media"))
    frame_selection = normalized_mapping(context.get("frameSelection"))
    preprocessing = {
        "contractId": PREPROCESSING_CONTRACT_ID,
        "contractVersion": PREPROCESSING_CONTRACT_VERSION,
        "colorMode": "RGB",
        "resize": [224, 224],
        "resizeMode": "fit",
        "resampling": "bicubic",
        "mean": list(CLIP_IMAGE_MEAN),
        "standardDeviation": list(CLIP_IMAGE_STD),
        **normalized_mapping(context.get("preprocessing")),
    }
    runtime = {
        "pythonImplementation": platform.python_implementation(),
        "pythonVersion": platform.python_version(),
        **normalized_mapping(context.get("runtime")),
    }
    try:
        import onnxruntime as ort

        runtime["onnxRuntimeVersion"] = str(ort.__version__)
    except Exception:
        runtime["onnxRuntimeVersion"] = "unavailable"
    inference = {
        "contractId": INFERENCE_CONTRACT_ID,
        "contractVersion": INFERENCE_CONTRACT_VERSION,
        "outputDtype": "float32",
        "normalization": "l2",
        **normalized_mapping(context.get("inference")),
    }
    identity = {
        "schemaId": EMBEDDING_CACHE_SCHEMA_ID,
        "schemaVersion": EMBEDDING_CACHE_SCHEMA_VERSION,
        "media": media,
        "frameSelection": frame_selection,
        "decodedImage": {
            "mode": "RGB",
            "size": list(image_rgb.size),
            "sha256": image_digest.hexdigest(),
        },
        "preprocessing": preprocessing,
        "model": model,
        "runtime": runtime,
        "inference": inference,
    }
    identity["componentDigests"] = {
        key: canonical_digest(identity[key])
        for key in ("media", "frameSelection", "decodedImage", "preprocessing", "model", "runtime", "inference")
    }
    identity["identityDigest"] = canonical_digest(identity)
    return identity


def normalized_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): normalize_identity_value(item)
        for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        if item is not None
    }


def normalize_identity_value(value: Any) -> Any:
    if isinstance(value, dict):
        return normalized_mapping(value)
    if isinstance(value, (list, tuple)):
        return [normalize_identity_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def media_cache_identity(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    path_value = source.get("path")
    identity = {
        key: normalize_identity_value(item)
        for key, item in source.items()
        if key != "path" and item is not None
    }
    if path_value:
        path = Path(str(path_value)).expanduser()
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        identity["locatorHash"] = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()
        try:
            stat = resolved.stat()
            identity["sizeBytes"] = int(stat.st_size)
            identity["modifiedTimeNs"] = int(stat.st_mtime_ns)
        except OSError:
            identity["availability"] = "missing"
    if not identity:
        identity["kind"] = "decoded-image-only"
    return normalized_mapping(identity)


def model_artifact_identity(onnx_path: str, supplied: Any = None) -> dict[str, Any]:
    path = Path(onnx_path).expanduser()
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    identity = {
        "artifactName": resolved.name,
        **normalized_mapping(supplied),
    }
    try:
        stat = resolved.stat()
        identity["sizeBytes"] = int(stat.st_size)
        identity["modifiedTimeNs"] = int(stat.st_mtime_ns)
        identity["sha256"] = model_artifact_sha256(str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    except OSError:
        identity["availability"] = "missing"
        identity["locatorHash"] = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()
    return normalized_mapping(identity)


@lru_cache(maxsize=16)
def model_artifact_sha256(path: str, _size: int, _modified_time_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedding_cache_path(
    onnx_path: str,
    image: Any,
    cache_context: dict[str, Any] | None = None,
) -> Path:
    identity = embedding_cache_identity(onnx_path, image, cache_context)
    return embedding_cache_path_from_identity(identity)


def embedding_cache_path_from_identity(identity: dict[str, Any]) -> Path:
    digest = str(identity["identityDigest"])
    return visual_encoder_cache_root() / "v1" / digest[:2] / f"{digest}.json"


def embedding_cache_subject_path(identity: dict[str, Any]) -> Path:
    media = identity.get("media") if isinstance(identity.get("media"), dict) else {}
    frame_selection = identity.get("frameSelection") if isinstance(identity.get("frameSelection"), dict) else {}
    subject = {
        "mediaLocatorHash": media.get("locatorHash"),
        "mediaKind": media.get("kind"),
        "frameSelection": frame_selection,
    }
    digest = canonical_digest(subject)
    return visual_encoder_cache_root() / "v1" / "subjects" / f"{digest}.json"


def lookup_cached_embedding(
    onnx_path: str,
    image: Any,
    *,
    cache_context: dict[str, Any] | None = None,
) -> EmbeddingCacheLookup:
    started = perf_counter()
    identity = embedding_cache_identity(onnx_path, image, cache_context)
    identity_digest = str(identity["identityDigest"])
    path = embedding_cache_path_from_identity(identity)
    reason = cache_miss_reason(identity)
    if not path.exists():
        return EmbeddingCacheLookup(None, "miss", reason, identity_digest, (perf_counter() - started) * 1000.0)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schemaId") != EMBEDDING_CACHE_SCHEMA_ID or document.get("schemaVersion") != EMBEDDING_CACHE_SCHEMA_VERSION:
            return EmbeddingCacheLookup(None, "invalid", "schema_mismatch", identity_digest, (perf_counter() - started) * 1000.0)
        if document.get("identity") != identity:
            return EmbeddingCacheLookup(None, "invalid", "identity_mismatch", identity_digest, (perf_counter() - started) * 1000.0)
        values = document.get("embedding")
        if isinstance(values, list) and values and all(math.isfinite(float(value)) for value in values):
            return EmbeddingCacheLookup(
                tuple(float(value) for value in values),
                "hit",
                "identity_match",
                identity_digest,
                (perf_counter() - started) * 1000.0,
            )
    except Exception:
        return EmbeddingCacheLookup(None, "invalid", "corrupt_entry", identity_digest, (perf_counter() - started) * 1000.0)
    return EmbeddingCacheLookup(None, "invalid", "invalid_embedding", identity_digest, (perf_counter() - started) * 1000.0)


def load_cached_embedding(
    onnx_path: str,
    image: Any,
    cache_context: dict[str, Any] | None = None,
) -> tuple[float, ...] | None:
    return lookup_cached_embedding(onnx_path, image, cache_context=cache_context).embedding


def store_cached_embedding(
    onnx_path: str,
    image: Any,
    embedding: tuple[float, ...],
    cache_context: dict[str, Any] | None = None,
) -> None:
    identity = embedding_cache_identity(onnx_path, image, cache_context)
    path = embedding_cache_path_from_identity(identity)
    subject_path = embedding_cache_subject_path(identity)
    try:
        atomic_json_write(path, {
            "schemaId": EMBEDDING_CACHE_SCHEMA_ID,
            "schemaVersion": EMBEDDING_CACHE_SCHEMA_VERSION,
            "identity": identity,
            "embedding": list(embedding),
        })
        atomic_json_write(subject_path, {
            "schemaId": EMBEDDING_CACHE_SCHEMA_ID + ".subject",
            "schemaVersion": EMBEDDING_CACHE_SCHEMA_VERSION,
            "identityDigest": identity["identityDigest"],
            "componentDigests": identity["componentDigests"],
        })
    except Exception:
        return


def cache_miss_reason(identity: dict[str, Any]) -> str:
    subject_path = embedding_cache_subject_path(identity)
    if not subject_path.exists():
        return "not_cached"
    try:
        previous = json.loads(subject_path.read_text(encoding="utf-8"))
        previous_components = previous.get("componentDigests")
        current_components = identity.get("componentDigests")
        if not isinstance(previous_components, dict) or not isinstance(current_components, dict):
            return "subject_metadata_invalid"
        reasons = {
            "media": "media_identity_changed",
            "frameSelection": "frame_selection_changed",
            "decodedImage": "decoded_pixels_changed",
            "preprocessing": "preprocessing_changed",
            "model": "model_identity_changed",
            "runtime": "runtime_configuration_changed",
            "inference": "inference_configuration_changed",
        }
        for component in ("media", "frameSelection", "decodedImage", "preprocessing", "model", "runtime", "inference"):
            if previous_components.get(component) != current_components.get(component):
                return reasons[component]
        return "entry_missing"
    except Exception:
        return "subject_metadata_invalid"


def atomic_json_write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    try:
        temporary.write_text(json.dumps(document, separators=(",", ":"), sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass
