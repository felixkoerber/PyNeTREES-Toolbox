"""Small shared argument helpers.

What used to live here were one-release compatibility shims for the W1
signature cleanup. They are gone: the port has no users to keep compatible
with, so every renamed parameter simply *is* its new name, and calling it by
the old one raises ``TypeError`` -- which is the loud failure an upgrading
caller needs (Design Decision #67).

What remains is the dimensionality validator, because ``dim`` is checked the
same way in a dozen places and the error message should read identically in
all of them.
"""

from __future__ import annotations

__all__ = ["resolve_dim"]


def resolve_dim(dim, default: int = 3) -> int:
    """Validate a dimensionality argument.

    ``dim`` is the integer 2 or 3 throughout the package (Design Decision
    #40); ``None`` means "use the default". Anything else raises rather than
    being silently treated as 3, since a typo there changes the geometry
    without changing the shape of the result.
    """
    if dim is None:
        return int(default)
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim!r}")
    return int(dim)
