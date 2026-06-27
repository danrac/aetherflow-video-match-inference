"""Timeline interchange exports for host integration."""

from __future__ import annotations

import json
from pathlib import Path


def export_edit_json(host_payload: dict, output_path: str | Path) -> Path:
    timeline = host_payload["timeline"]
    document = {
        "schema_version": host_payload.get("schema_version", "0.1.0"),
        "host": host_payload.get("host"),
        "reference_path": timeline.get("reference_path"),
        "fps": timeline.get("fps"),
        "edits": timeline.get("edits", []),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def export_edl(host_payload: dict, output_path: str | Path, title: str = "AETHERFLOW_VIDEO_MATCH") -> Path:
    timeline = host_payload["timeline"]
    fps = float(timeline.get("fps", 24.0) or 24.0)
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    for index, edit in enumerate(timeline.get("edits", []), start=1):
        reel = reel_name(edit.get("source_path", "SOURCE"))
        source_in = frames_to_timecode(int(edit["source_in_frames"]), fps)
        source_out = frames_to_timecode(int(edit["source_out_frames"]), fps)
        record_in = frames_to_timecode(int(edit["reference_in_frames"]), fps)
        record_out = frames_to_timecode(int(edit["reference_out_frames"]), fps)
        lines.append(f"{index:03d}  {reel:<8.8} V     C        {source_in} {source_out} {record_in} {record_out}")
        lines.append(f"* FROM CLIP NAME: {edit.get('source_path', '')}")
        lines.append(f"* AETHERFLOW CONFIDENCE: {float(edit.get('confidence', 0.0)):.6f}")
        lines.append("")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def frames_to_timecode(frame: int, fps: float) -> str:
    rounded_fps = max(1, round(fps))
    hours, remainder = divmod(frame, rounded_fps * 60 * 60)
    minutes, remainder = divmod(remainder, rounded_fps * 60)
    seconds, frames = divmod(remainder, rounded_fps)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"


def reel_name(source_path: str) -> str:
    stem = Path(source_path).stem or "SOURCE"
    cleaned = "".join(character for character in stem.upper() if character.isalnum())
    return (cleaned or "SOURCE")[:8]
