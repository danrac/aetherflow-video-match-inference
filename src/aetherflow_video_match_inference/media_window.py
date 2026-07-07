"""Optional media-frame source-window rescoring.

This module is intentionally dependency-light. It uses the existing runtime
FFmpeg plus Pillow/NumPy when available and otherwise leaves feature-only
ranking untouched.
"""

from __future__ import annotations

from functools import lru_cache
import io
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


COMPARE_SIZE = (90, 160)
SOURCE_CROP_X_FACTORS = (0.15, 0.5, 0.85)
SOURCE_CROP_Y_FACTORS = (0.5,)
SOURCE_CROP_HEIGHT_FACTORS = (0.78, 1.0)
_CV2_CAPTURES: dict[str, Any] = {}


def media_window_rescore(reference_path: str, reference_features: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any] | None:
    try:
        from PIL import Image, ImageOps
        import numpy as np
    except Exception:
        return None

    reference_window = reference_features.get("source_window") if isinstance(reference_features.get("source_window"), dict) else {}
    reference_start = int(reference_window.get("source_in", 0) or 0)
    reference_end = int(reference_window.get("source_out", reference_start + 1) or reference_start + 1)
    reference_duration = max(1, reference_end - reference_start)
    if reference_duration <= 2:
        return {
            "distance": None,
            "source_in": int(candidate["source_window_entry"].get("source_in", 0)),
            "source_out": int(candidate["source_window_entry"].get("source_out", 1)),
            "microcut": True,
        }

    source_in = int(candidate["source_window_entry"].get("source_in", 0))
    source_out = int(candidate["source_window_entry"].get("source_out", source_in + 1))
    if source_out <= source_in:
        return None

    reference_fps = float(reference_features.get("fps", 30.0) or 30.0)
    source_fps = float(candidate["features"].get("fps", reference_fps) or reference_fps)
    source_duration = source_out - source_in
    max_start = candidate_search_max_start(source_in, source_out, reference_duration)
    identity_rel_frames = sample_relative_frames(reference_duration)
    candidate_starts = candidate_start_grid(source_in, max_start, identity_rel_frames, candidate["features"])
    if not candidate_starts:
        return None

    reference_arrays = []
    for rel_frame in identity_rel_frames:
        frame = read_video_frame(reference_path, reference_start + rel_frame, reference_fps)
        if frame is None:
            return None
        reference_image = frame.convert("RGB")
        reference_arrays.append(np.asarray(ImageOps.fit(reference_image, COMPARE_SIZE, method=Image.Resampling.BILINEAR), dtype=np.float32) / 255.0)

    best = None
    for start_frame in candidate_starts:
        for playback_direction in ("forward", "reverse"):
            distance = score_candidate_start(
                candidate["source_path"],
                start_frame,
                identity_rel_frames,
                source_fps,
                reference_arrays,
                np,
                ImageOps,
                reference_duration=reference_duration,
                playback_direction=playback_direction,
            )
            if distance is None:
                continue
            if best is None or distance < best["distance"]:
                best = {
                    "distance": distance,
                    "source_in": start_frame,
                    "source_out": start_frame + reference_duration,
                    "microcut": False,
                    "playback_direction": playback_direction,
                    "temporal_sample_count": len(identity_rel_frames),
                }

    return best


def sample_relative_frames(duration: int) -> list[int]:
    if duration <= 1:
        return [0]
    if duration <= 8:
        fractions = (0.5,)
    else:
        fractions = (0.2, 0.5, 0.8)
    return sorted({max(0, min(duration - 1, int(round((duration - 1) * fraction)))) for fraction in fractions})


def candidate_search_max_start(source_in: int, source_out: int, reference_duration: int) -> int:
    if source_out <= source_in:
        return source_in
    if not media_tail_starts_enabled():
        return max(source_in, source_out - reference_duration)
    source_duration = source_out - source_in
    if source_duration <= max(1, reference_duration):
        return max(source_in, source_out - 1)
    return max(source_in, source_out - reference_duration)


def media_tail_starts_enabled() -> bool:
    return str(os.environ.get("AETHERFLOW_VIDEO_MATCH_ALLOW_TAIL_STARTS", "")).lower() in {"1", "true", "yes", "on"}


def candidate_start_grid(source_in: int, max_start: int, rel_frames: list[int], feature_document: dict[str, Any]) -> list[int]:
    if max_start <= source_in:
        return [source_in]
    span = max_start - source_in
    coarse = []
    for value in (source_in, source_in + span // 2, max_start):
        if value not in coarse:
            coarse.append(value)
    sampled_frames = [
        int(frame.get("frame_index"))
        for frame in feature_document.get("features", [])
        if isinstance(frame, dict) and frame.get("frame_index") is not None
    ]
    sampled_starts = []
    for sampled_frame in sampled_frames:
        for rel_frame in rel_frames:
            start = sampled_frame - rel_frame
            if source_in <= start <= max_start:
                sampled_starts.append(start)
                coarse.append(start)
    sampled_starts = sorted(set(sampled_starts))
    for previous, current in zip(sampled_starts, sampled_starts[1:], strict=False):
        midpoint = previous + ((current - previous) // 2)
        coarse.append(midpoint)
    ordered: list[int] = []
    seen: set[int] = set()
    for deltas in ((0,), (-4, -2, 2, 4)):
        for center in coarse:
            for delta in deltas:
                value = center + delta
                if source_in <= value <= max_start and value not in seen:
                    seen.add(value)
                    ordered.append(value)
    return ordered[:media_start_grid_limit()]


def media_start_grid_limit() -> int:
    try:
        return max(1, int(os.environ.get("AETHERFLOW_VIDEO_MATCH_START_GRID_LIMIT", "16")))
    except ValueError:
        return 16


def refine_boundary_start(
    reference_path: str,
    reference_features: dict[str, Any],
    candidate: dict[str, Any],
    source_in: int,
    max_start: int,
    reference_start: int,
    reference_duration: int,
    reference_fps: float,
    source_fps: float,
    np: Any,
    Image: Any,
    ImageOps: Any,
    baseline_start: int | None = None,
) -> dict[str, Any] | None:
    frame = read_video_frame(reference_path, reference_start, reference_fps)
    if frame is None:
        return None
    reference_array = np.asarray(ImageOps.fit(frame.convert("RGB"), COMPARE_SIZE, method=Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    starts = boundary_start_grid(source_in, max_start, reference_start, reference_features, candidate["features"])
    baseline_distance = None
    if baseline_start is not None and source_in <= baseline_start <= max_start:
        baseline_distance = score_boundary_start(candidate["source_path"], baseline_start, source_fps, reference_array, np, Image, ImageOps)
    best = None
    for start_frame in starts:
        distance = score_boundary_start(candidate["source_path"], start_frame, source_fps, reference_array, np, Image, ImageOps)
        if distance is None:
            continue
        if best is None or distance < best["distance"]:
            best = {"distance": distance, "source_in": start_frame}
    if best is not None:
        best["baseline_distance"] = baseline_distance
    return best


def score_boundary_start(source_path: str, start_frame: int, fps: float, reference_array: Any, np: Any, Image: Any, ImageOps: Any) -> float | None:
    source_frame = read_video_frame(source_path, start_frame, fps)
    if source_frame is None:
        return None
    source_array = center_vertical_crop_array(source_frame, np, Image, ImageOps)
    keypoint_distance = structural_keypoint_distance(reference_array, source_array, np)
    if keypoint_distance is not None:
        return keypoint_distance
    return frame_distance(reference_array, source_array, np)


def center_vertical_crop_array(source_frame: Any, np: Any, Image: Any, ImageOps: Any) -> Any:
    source_rgb = source_frame.convert("RGB")
    width, height = source_rgb.size
    crop_width = min(width, max(1, int(round(height * 9.0 / 16.0))))
    left = int(round((width - crop_width) * 0.5))
    crop = source_rgb.crop((left, 0, left + crop_width, height))
    return np.asarray(ImageOps.fit(crop, COMPARE_SIZE, method=Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def boundary_start_grid(source_in: int, max_start: int, reference_start: int, reference_features: dict[str, Any], feature_document: dict[str, Any]) -> list[int]:
    starts: list[int] = []
    seen: set[int] = set()

    def add(value: int) -> None:
        if source_in <= value <= max_start and value not in seen:
            seen.add(value)
            starts.append(value)

    reference_rel_frames = [
        int(frame.get("frame_index")) - reference_start
        for frame in reference_features.get("features", [])
        if isinstance(frame, dict) and frame.get("frame_index") is not None
    ]
    if reference_rel_frames:
        reference_rel_frames = sorted(set([reference_rel_frames[0], reference_rel_frames[-1]]))
    sampled_frames = [
        int(frame.get("frame_index"))
        for frame in feature_document.get("features", [])
        if isinstance(frame, dict) and frame.get("frame_index") is not None
    ]
    for start in candidate_start_grid(source_in, max_start, [0], feature_document)[:8]:
        add(start)
    for sampled_frame in sampled_frames:
        for rel_frame in reference_rel_frames:
            center = sampled_frame - rel_frame
            for delta in (0, -5, 5):
                add(center + delta)
    return starts[:48]


def score_candidate_start(
    source_path: str,
    start_frame: int,
    rel_frames: list[int],
    fps: float,
    reference_arrays: list[Any],
    np: Any,
    ImageOps: Any,
    *,
    reference_duration: int | None = None,
    playback_direction: str = "forward",
) -> float | None:
    distances = []
    for index, rel_frame in enumerate(rel_frames):
        source_rel_frame = rel_frame
        if playback_direction == "reverse" and reference_duration is not None:
            source_rel_frame = max(0, int(reference_duration) - 1 - int(rel_frame))
        source_frame = read_video_frame(source_path, start_frame + source_rel_frame, fps)
        if source_frame is None:
            return None
        pixel_distance = best_crop_distance(reference_arrays[index], source_frame, np, ImageOps)
        distances.append(pixel_distance)
    return round(sum(distances) / len(distances), 6) if distances else None


def source_crop_images(source_frame: Any) -> list[Any]:
    source_rgb = source_frame.convert("RGB")
    width, height = source_rgb.size
    candidates = []
    seen_boxes = set()
    for height_factor in SOURCE_CROP_HEIGHT_FACTORS:
        crop_height = min(height, max(1, int(round(height * height_factor))))
        crop_width = min(width, max(1, int(round(crop_height * 9.0 / 16.0))))
        for x_factor in SOURCE_CROP_X_FACTORS:
            left = int(round((width - crop_width) * x_factor))
            for y_factor in SOURCE_CROP_Y_FACTORS:
                top = int(round((height - crop_height) * y_factor))
                box = (left, top, left + crop_width, top + crop_height)
                if box in seen_boxes:
                    continue
                seen_boxes.add(box)
                candidates.append(source_rgb.crop(box))
    return candidates


def best_crop_distance(reference_array: Any, source_frame: Any, np: Any, ImageOps: Any) -> float:
    candidates = []
    from PIL import Image

    if low_texture_reference(reference_array, np):
        source_array = center_vertical_crop_array(source_frame, np, Image, ImageOps)
        candidates.append(source_array)
    else:
        for crop in source_crop_images(source_frame):
            candidates.append(np.asarray(ImageOps.fit(crop, COMPARE_SIZE, method=Image.Resampling.BILINEAR), dtype=np.float32) / 255.0)
    return min(frame_distance(reference_array, candidate, np) for candidate in candidates)


def low_texture_reference(reference_array: Any, np: Any) -> bool:
    gray = np.mean(reference_array, axis=2)
    std = float(np.std(gray))
    dy, dx = np.gradient(gray)
    edge_mean = float(np.mean(np.sqrt((dx * dx) + (dy * dy))))
    return edge_mean < 0.025 or (std < 0.08 and edge_mean < 0.04)


def frame_distance(reference_array: Any, source_array: Any, np: Any) -> float:
    mse = float(np.mean((reference_array - source_array) ** 2))
    ref_flat = reference_array.reshape(-1)
    src_flat = source_array.reshape(-1)
    ref_std = float(np.std(ref_flat))
    src_std = float(np.std(src_flat))
    if ref_std <= 1e-6 or src_std <= 1e-6:
        corr_distance = 1.0
    else:
        corr = float(np.corrcoef(ref_flat, src_flat)[0, 1])
        corr_distance = 1.0 - max(-1.0, min(1.0, corr))
    ref_hist = np.histogram(reference_array, bins=16, range=(0.0, 1.0), density=True)[0]
    src_hist = np.histogram(source_array, bins=16, range=(0.0, 1.0), density=True)[0]
    hist_distance = float(np.mean(np.abs(ref_hist - src_hist))) / 16.0
    edge_distance = structural_edge_distance(reference_array, source_array, np)
    color_distance = (mse * 80.0) + (corr_distance * 25.0) + (hist_distance * 12.0) + (edge_distance * 20.0)
    if not media_keypoint_scoring_enabled():
        return color_distance
    keypoint_distance = structural_keypoint_distance(reference_array, source_array, np)
    if keypoint_distance is None:
        return color_distance
    return (color_distance * 0.25) + (keypoint_distance * 0.75)


def media_keypoint_scoring_enabled() -> bool:
    return str(os.environ.get("AETHERFLOW_VIDEO_MATCH_MEDIA_KEYPOINTS", "1")).lower() not in {"0", "false", "no", "off"}


def structural_edge_distance(reference_array: Any, source_array: Any, np: Any) -> float:
    reference_gray = np.mean(reference_array, axis=2)
    source_gray = np.mean(source_array, axis=2)
    ref_dy, ref_dx = np.gradient(reference_gray)
    src_dy, src_dx = np.gradient(source_gray)
    ref_edges = np.sqrt((ref_dx * ref_dx) + (ref_dy * ref_dy))
    src_edges = np.sqrt((src_dx * src_dx) + (src_dy * src_dy))
    edge_mse = float(np.mean((ref_edges - src_edges) ** 2))
    ref_flat = ref_edges.reshape(-1)
    src_flat = src_edges.reshape(-1)
    if float(np.std(ref_flat)) <= 1e-6 or float(np.std(src_flat)) <= 1e-6:
        edge_corr_distance = 1.0
    else:
        corr = float(np.corrcoef(ref_flat, src_flat)[0, 1])
        edge_corr_distance = 1.0 - max(-1.0, min(1.0, corr))
    return (edge_mse * 3.0) + edge_corr_distance


def structural_keypoint_distance(reference_array: Any, source_array: Any, np: Any) -> float | None:
    try:
        import cv2
    except Exception:
        return None
    reference_gray = (np.mean(reference_array, axis=2) * 255.0).clip(0, 255).astype("uint8")
    source_gray = (np.mean(source_array, axis=2) * 255.0).clip(0, 255).astype("uint8")
    try:
        extractor = cv2.ORB_create(nfeatures=300)
        reference_keypoints, reference_descriptors = extractor.detectAndCompute(reference_gray, None)
        source_keypoints, source_descriptors = extractor.detectAndCompute(source_gray, None)
    except Exception:
        return None
    if (
        reference_descriptors is None
        or source_descriptors is None
        or len(reference_keypoints) < 6
        or len(source_keypoints) < 6
    ):
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(reference_descriptors, source_descriptors), key=lambda match: match.distance)
    if not matches:
        return None
    strongest = matches[:24]
    return (sum(float(match.distance) for match in strongest) / len(strongest)) + (160.0 / max(1, len(strongest)))


@lru_cache(maxsize=512)
def read_video_frame(path: str, frame_index: int, fps: float):
    frame = read_video_frame_cv2(path, frame_index)
    if frame is not None:
        return frame
    ffmpeg = ffmpeg_executable()
    if ffmpeg is None or fps <= 0:
        return None
    try:
        from PIL import Image
    except Exception:
        return None
    timestamp = max(0.0, float(frame_index) / float(fps))
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.6f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        return None
    if not result.stdout:
        return None
    try:
        return Image.open(io.BytesIO(result.stdout)).convert("RGB")
    except Exception:
        return None


def read_video_frame_cv2(path: str, frame_index: int):
    try:
        import cv2
        from PIL import Image
    except Exception:
        return None
    capture = _CV2_CAPTURES.get(path)
    if capture is None:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            return None
        _CV2_CAPTURES[path] = capture
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def ffmpeg_executable() -> str | None:
    try:
        import imageio_ffmpeg

        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path:
            return str(path)
    except Exception:
        pass
    return shutil.which("ffmpeg")
