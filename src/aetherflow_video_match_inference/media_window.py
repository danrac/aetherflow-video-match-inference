"""Optional media-frame source-window rescoring.

This module is intentionally dependency-light. It uses the existing runtime
FFmpeg plus Pillow/NumPy when available and otherwise leaves feature-only
ranking untouched.
"""

from __future__ import annotations

from functools import lru_cache
import io
import shutil
import subprocess
from pathlib import Path
from typing import Any


COMPARE_SIZE = (90, 160)
SOURCE_CROP_X_FACTORS = (0.15, 0.5, 0.85)
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
    max_start = max(source_in, source_out - reference_duration)
    rel_frames = sample_relative_frames(reference_duration)
    candidate_starts = candidate_start_grid(source_in, max_start, rel_frames, candidate["features"])
    if not candidate_starts:
        return None

    reference_arrays = []
    for rel_frame in rel_frames:
        frame = read_video_frame(reference_path, reference_start + rel_frame, reference_fps)
        if frame is None:
            return None
        reference_arrays.append(np.asarray(ImageOps.fit(frame.convert("RGB"), COMPARE_SIZE, method=Image.Resampling.BILINEAR), dtype=np.float32) / 255.0)

    best = None
    for start_frame in candidate_starts:
        distance = score_candidate_start(
            candidate["source_path"],
            start_frame,
            rel_frames,
            source_fps,
            reference_arrays,
            np,
            ImageOps,
        )
        if distance is None:
            continue
        if best is None or distance < best["distance"]:
            best = {"distance": distance, "source_in": start_frame, "source_out": min(source_out, start_frame + reference_duration), "microcut": False}

    return best


def sample_relative_frames(duration: int) -> list[int]:
    if duration <= 1:
        return [0]
    fractions = (0.5,)
    return sorted({max(0, min(duration - 1, int(round((duration - 1) * fraction)))) for fraction in fractions})


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
    for sampled_frame in sampled_frames:
        for rel_frame in rel_frames:
            coarse.append(sampled_frame - rel_frame)
    refined = set()
    for center in coarse:
        for delta in (-2, 0, 2):
            value = center + delta
            if source_in <= value <= max_start:
                refined.add(value)
    midpoint = source_in + (max_start - source_in) // 2
    return sorted(refined, key=lambda value: (abs(value - midpoint), value))[:10]


def score_candidate_start(source_path: str, start_frame: int, rel_frames: list[int], fps: float, reference_arrays: list[Any], np: Any, ImageOps: Any) -> float | None:
    distances = []
    for index, rel_frame in enumerate(rel_frames):
        source_frame = read_video_frame(source_path, start_frame + rel_frame, fps)
        if source_frame is None:
            return None
        distances.append(best_crop_distance(reference_arrays[index], source_frame, np, ImageOps))
    return round(sum(distances) / len(distances), 6) if distances else None


def best_crop_distance(reference_array: Any, source_frame: Any, np: Any, ImageOps: Any) -> float:
    from PIL import Image

    source_rgb = source_frame.convert("RGB")
    width, height = source_rgb.size
    crop_width = min(width, int(round(height * 9.0 / 16.0)))
    candidates = []
    for factor in SOURCE_CROP_X_FACTORS:
        left = int(round((width - crop_width) * factor))
        crop = source_rgb.crop((left, 0, left + crop_width, height))
        candidates.append(np.asarray(ImageOps.fit(crop, COMPARE_SIZE, method=Image.Resampling.BILINEAR), dtype=np.float32) / 255.0)
    return min(frame_distance(reference_array, candidate, np) for candidate in candidates)


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
    return (mse * 120.0) + (corr_distance * 35.0) + (hist_distance * 25.0)


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
