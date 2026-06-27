"""Host adapter boundaries."""

from . import HOST_PAYLOAD_SCHEMA_VERSION
from .timeline import reconstruct_timeline


def to_host_payload(match_result: dict, host: str) -> dict:
    return {
        "schema_version": HOST_PAYLOAD_SCHEMA_VERSION,
        "host": host,
        "match_result": match_result,
        "timeline": reconstruct_timeline(match_result),
    }
