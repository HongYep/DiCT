"""Four-choice extraction from the 3D-RAD scoring adapter."""

import re

def extract_choice(text: str) -> str:
    match = re.search(
        r"^\s*(?:(?:the\s+)?answer\s*(?:is|:)?\s*)?(?:option\s*)?([A-D])(?:\b|[.):])",
        str(text),
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(r"\b([A-D])\b", str(text), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""
