"""Host adapter boundaries."""


def to_host_payload(match_result: dict, host: str) -> dict:
    return {
        "host": host,
        "match_result": match_result,
    }
