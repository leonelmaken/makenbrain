"""Small diff helpers for dry-run responses."""
from __future__ import annotations

import difflib


def unified_diff(original: str, modified: str, *, fromfile: str = "before", tofile: str = "after") -> str:
    """Return a UTF-8 friendly unified diff."""
    return "\n".join(
        difflib.unified_diff(
            original.splitlines(),
            modified.splitlines(),
            fromfile=fromfile,
            tofile=tofile,
            lineterm="",
        )
    )
