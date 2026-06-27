"""Timeline reconstruction helpers."""


def sort_matches_by_reference_time(matches: list[dict]) -> list[dict]:
    return sorted(matches, key=lambda match: (match["reference_in"], match["reference_out"]))


def total_reference_span(matches: list[dict]) -> int:
    if not matches:
        return 0
    first = min(match["reference_in"] for match in matches)
    last = max(match["reference_out"] for match in matches)
    return last - first
