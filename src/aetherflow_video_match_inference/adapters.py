"""Host adapter boundaries."""

from .timeline import reconstruct_timeline


def to_host_payload(match_result: dict, host: str) -> dict:
    return {
        "host": host,
        "match_result": match_result,
        "timeline": reconstruct_timeline(match_result),
    }
