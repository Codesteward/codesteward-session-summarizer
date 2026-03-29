from __future__ import annotations

import hashlib


def compute_hash(text: str) -> str:
    """Compute a truncated SHA-256 hash of text.

    Returns 16 hex characters (64 bits) — sufficient for collision avoidance
    at the scale of prompt variants and session contexts.
    """
    return hashlib.sha256(text.encode()).hexdigest()[:16]
