"""Timeline reconstruction helpers."""


def sort_matches_by_reference_time(matches: list[dict]) -> list[dict]:
    return sorted(matches, key=lambda match: (match["reference_in"], match["reference_out"]))
