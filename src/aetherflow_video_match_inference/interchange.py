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


def export_cep_json(host_payload: dict, output_path: str | Path, workflow_id: str = "aetherflow-video-match") -> Path:
    """Export an AetherFlow CEP handoff JSON file.

    The CEP adapter keeps host-facing instructions explicit so the panel can
    import the file without understanding model internals.
    """

    timeline = host_payload["timeline"]
    document = {
        "schema_version": host_payload.get("schema_version", "0.1.0"),
        "adapter": "aetherflow-cep-json",
        "adapter_version": "0.1.0",
        "workflow_id": workflow_id,
        "host": "aetherflow-cep",
        "source_host_payload_host": host_payload.get("host"),
        "reference": {
            "path": timeline.get("reference_path"),
            "fps": timeline.get("fps"),
            "duration_frames": timeline.get("duration_frames"),
            "duration_seconds": timeline.get("duration_seconds"),
        },
        "timeline": {
            "fps": timeline.get("fps"),
            "duration_frames": timeline.get("duration_frames"),
            "duration_seconds": timeline.get("duration_seconds"),
            "edit_count": timeline.get("edit_count", 0),
            "layers": [cep_layer_from_edit(edit) for edit in timeline.get("edits", [])],
        },
        "instructions": [
            "import_source_media",
            "create_or_reuse_composition",
            "place_layers_by_seconds",
            "apply_source_trim",
            "preserve_confidence_metadata",
        ],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def cep_layer_from_edit(edit: dict) -> dict:
    return {
        "layer_id": edit["edit_id"],
        "source_path": edit["source_path"],
        "track": int(edit.get("track", 0)),
        "start_seconds": edit["reference_in_seconds"] - edit["source_in_seconds"],
        "in_point_seconds": edit["reference_in_seconds"],
        "out_point_seconds": edit["reference_out_seconds"],
        "source_in_seconds": edit["source_in_seconds"],
        "source_out_seconds": edit["source_out_seconds"],
        "reference_in_frames": edit["reference_in_frames"],
        "reference_out_frames": edit["reference_out_frames"],
        "source_in_frames": edit["source_in_frames"],
        "source_out_frames": edit["source_out_frames"],
        "confidence": edit.get("confidence", 0.0),
        "operation": edit.get("operation"),
        "parameters": edit.get("parameters", {}),
    }


def export_premiere_json(host_payload: dict, output_path: str | Path, sequence_name: str = "AetherFlow Video Match") -> Path:
    """Export a Premiere-oriented JSON handoff file."""

    timeline = host_payload["timeline"]
    document = {
        "schema_version": host_payload.get("schema_version", "0.1.0"),
        "adapter": "premiere-pro-json",
        "adapter_version": "0.1.0",
        "host": "premiere-pro",
        "sequence": {
            "name": sequence_name,
            "fps": timeline.get("fps"),
            "duration_frames": timeline.get("duration_frames"),
            "duration_seconds": timeline.get("duration_seconds"),
        },
        "clips": [premiere_clip_from_edit(edit) for edit in timeline.get("edits", [])],
        "instructions": [
            "import_source_media",
            "create_or_reuse_sequence",
            "insert_clips_by_frames",
            "apply_source_in_out",
            "preserve_match_metadata_as_markers",
        ],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def premiere_clip_from_edit(edit: dict) -> dict:
    return {
        "clip_id": edit["edit_id"],
        "source_path": edit["source_path"],
        "target_video_track": int(edit.get("track", 0)),
        "timeline_in_frames": edit["reference_in_frames"],
        "timeline_out_frames": edit["reference_out_frames"],
        "source_in_frames": edit["source_in_frames"],
        "source_out_frames": edit["source_out_frames"],
        "timeline_in_seconds": edit["reference_in_seconds"],
        "timeline_out_seconds": edit["reference_out_seconds"],
        "source_in_seconds": edit["source_in_seconds"],
        "source_out_seconds": edit["source_out_seconds"],
        "marker": {
            "name": "AetherFlow Video Match",
            "confidence": edit.get("confidence", 0.0),
            "operation": edit.get("operation"),
            "parameters": edit.get("parameters", {}),
        },
    }


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


def export_after_effects_extendscript(host_payload: dict, output_path: str | Path, comp_name: str = "AetherFlow Video Match") -> Path:
    timeline = host_payload["timeline"]
    payload_json = json.dumps(
        {
            "comp_name": comp_name,
            "fps": timeline.get("fps", 24.0),
            "duration_seconds": timeline.get("duration_seconds", 0.0),
            "edits": timeline.get("edits", []),
        },
        indent=2,
        sort_keys=True,
    )
    script = f"""// Generated by AetherFlow Video Match.
(function () {{
    app.beginUndoGroup("AetherFlow Video Match Import");
    try {{
        var payload = {payload_json};
        var project = app.project || app.newProject();
        var compWidth = 1920;
        var compHeight = 1080;
        var pixelAspect = 1.0;
        var duration = Math.max(payload.duration_seconds || 1.0, 1.0);
        var fps = payload.fps || 24.0;
        var comp = project.items.addComp(payload.comp_name, compWidth, compHeight, pixelAspect, duration, fps);

        for (var i = 0; i < payload.edits.length; i += 1) {{
            var edit = payload.edits[i];
            var file = new File(edit.source_path);
            if (!file.exists) {{
                $.writeln("AetherFlow missing source: " + edit.source_path);
                continue;
            }}
            var importOptions = new ImportOptions(file);
            var footage = project.importFile(importOptions);
            var layer = comp.layers.add(footage);
            layer.name = edit.edit_id + " " + file.name;
            layer.startTime = edit.reference_in_seconds - edit.source_in_seconds;
            layer.inPoint = edit.reference_in_seconds;
            layer.outPoint = edit.reference_out_seconds;
            layer.comment = "AetherFlow confidence=" + edit.confidence + " operation=" + edit.operation;
        }}
    }} finally {{
        app.endUndoGroup();
    }}
}}());
"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script, encoding="utf-8")
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
