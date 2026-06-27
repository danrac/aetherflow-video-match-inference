"""Storage root helpers for inference artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
import os
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/Volumes/FrameFusion/AetherFlow_VideoMatcherData")


def data_root() -> Path:
    override = os.environ.get("AETHERFLOW_VIDEO_MATCH_DATA_ROOT")
    if override:
        return Path(override).expanduser()
    return DEFAULT_DATA_ROOT


def inference_output_path(output: str | Path | None) -> Path:
    if output:
        return Path(output).expanduser()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return data_root() / "inference" / f"match_result_{timestamp}.json"
