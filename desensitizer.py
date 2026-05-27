"""Simple medical-data desensitization helpers."""
from __future__ import annotations

import re


def desensitize_text(text: str) -> str:
    """Mask common names, identity-like numeric identifiers, email addresses, and phones."""
    output = str(text)
    output = re.sub(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", "[NAME]", output)
    output = re.sub(r"\b\d{6}-?\d{2}-?\d{4}\b", "[ID]", output)
    output = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", output)
    output = re.sub(r"\b(?:\+?\d{1,3}[-\s]?)?(?:\d{2,4}[-\s]?){2,4}\d{2,4}\b", "[PHONE]", output)
    return output
