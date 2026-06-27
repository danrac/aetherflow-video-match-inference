"""Inference engine boundary."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MatchRequest:
    reference_path: str
    source_paths: tuple[str, ...]
    model_manifest_path: str


def describe_request(request: MatchRequest) -> dict[str, object]:
    return {
        "reference_path": request.reference_path,
        "source_count": len(request.source_paths),
        "model_manifest_path": request.model_manifest_path,
    }
