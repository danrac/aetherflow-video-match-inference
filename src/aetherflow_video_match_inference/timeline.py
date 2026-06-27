"""Timeline reconstruction helpers."""


def sort_matches_by_reference_time(matches: list[dict]) -> list[dict]:
    return sorted(matches, key=lambda match: (match["reference_in"], match["reference_out"]))


def total_reference_span(matches: list[dict]) -> int:
    if not matches:
        return 0
    first = min(match["reference_in"] for match in matches)
    last = max(match["reference_out"] for match in matches)
    return last - first


def reconstruct_timeline(match_result: dict) -> dict:
    reference = match_result.get("reference", {})
    fps = float(reference.get("fps", 24.0) or 24.0)
    edits = [match_to_edit(match, fps, index) for index, match in enumerate(sort_matches_by_reference_time(match_result.get("matches", [])))]
    return {
        "reference_path": reference.get("path"),
        "fps": fps,
        "duration_frames": int(reference.get("duration_frames", 0) or 0),
        "duration_seconds": frames_to_seconds(int(reference.get("duration_frames", 0) or 0), fps),
        "edit_count": len(edits),
        "edits": edits,
    }


def match_to_edit(match: dict, fps: float, index: int) -> dict:
    return {
        "edit_id": f"edit-{index + 1:04d}",
        "source_path": match["source_path"],
        "track": int(match.get("timeline_track", 0)),
        "reference_in_frames": int(match["reference_in"]),
        "reference_out_frames": int(match["reference_out"]),
        "source_in_frames": int(match["source_in"]),
        "source_out_frames": int(match["source_out"]),
        "reference_in_seconds": frames_to_seconds(int(match["reference_in"]), fps),
        "reference_out_seconds": frames_to_seconds(int(match["reference_out"]), fps),
        "source_in_seconds": frames_to_seconds(int(match["source_in"]), fps),
        "source_out_seconds": frames_to_seconds(int(match["source_out"]), fps),
        "confidence": float(match["confidence"]),
        "operation": match.get("reconstruction", {}).get("operation"),
        "parameters": match.get("reconstruction", {}).get("parameters", {}),
    }


def frames_to_seconds(frame: int, fps: float) -> float:
    return round(frame / fps, 6)
