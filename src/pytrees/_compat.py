"""One-release compatibility shims for the W1 signature cleanup.

The W1 pass (see ``REVIEW_PLAN.md``) renamed a handful of parameters to
follow two rules adopted in Design Decisions #40 and #41:

- **#40** -- the dimensionality argument is always ``dim: int`` in ``{2, 3}``,
  replacing the three spellings that had accumulated (``dim2: bool`` in
  ``cyl_tree``/``len_tree``/``chull_tree``, ``dim: int`` in ``eucl_tree``,
  ``dim: str`` -- ``"2d"``/``"3d"`` -- in ``vonMises_tree``/``bf_tree``).
- **#41** -- no negated boolean parameters, because a default of
  ``no_self=False`` makes the reader resolve a double negative to work out
  what actually happens. ``idpar_tree``'s ``no_self`` became ``root_self``
  and ``elimt_tree``'s ``no_root`` became ``at_root``, each with the default
  flipped so behaviour is unchanged.

Rather than repeat the same six-line warn-and-translate block in every
touched function, the translations live here. Each emits
:class:`DeprecationWarning` and keeps working for one release.

Note the asymmetry with the *return-value* change (Design Decision #42):
that one ships **without** a shim, because no shim can straddle "returns a
tuple" and "returns a Tree". It doesn't need one -- ``Tree`` defines neither
``__iter__`` nor ``__getitem__``, so a stale ``tree, order = sort_tree(t)``
raises ``TypeError: cannot unpack non-iterable Tree object`` immediately, at
the call site. A renamed *keyword*, by contrast, would raise a bare
"unexpected keyword argument" that says nothing about what to write instead
-- hence the shims here.
"""

from __future__ import annotations

import warnings

__all__ = ["resolve_dim", "resolve_flipped_bool"]


def resolve_dim(dim, dim2=None, *, legacy_name: str = "dim2", default: int = 3) -> int:
    """Normalise a dimensionality argument to the integer ``2`` or ``3``.

    Accepts the current form (``dim=2``/``dim=3``) and both retired ones:
    the boolean ``dim2=True`` and the string ``dim="2d"``/``"3d"``.

    Parameters
    ----------
    dim : int, str, or None
        Current-form value. ``None`` means "caller didn't pass one", and is
        why every ``dim`` parameter in the package defaults to ``None``
        rather than literally to ``3``: without a sentinel there is no way
        to tell an explicit ``dim=3`` from the default, so the contradiction
        ``len_tree(t, dim=3, dim2=True)`` would pass silently. The docstrings
        still document the effective default, which is what a reader needs.
        ``"2d"``/``"3d"`` strings are translated with a warning, since
        ``vonMises_tree``/``bf_tree`` used to take them under this name.
    dim2 : bool, optional
        Retired boolean form. ``None`` means "not passed".
    legacy_name : str
        Name to quote in the warning -- ``"dim2"`` for the boolean spelling.
    default : int, default 3
        Value to use when ``dim`` is ``None``.

    Returns
    -------
    int
        ``2`` or ``3``.

    Raises
    ------
    ValueError
        If ``dim`` is neither 2 nor 3 (after translation), or if both the
        current and retired spellings are passed at once -- silently
        preferring one would hide a real contradiction in the caller.
    """
    if dim2 is not None:
        if dim is not None:
            raise ValueError(
                f"pass either dim= or {legacy_name}=, not both "
                f"(got dim={dim!r}, {legacy_name}={dim2!r})"
            )
        warnings.warn(
            f"{legacy_name}= is deprecated and will be removed; "
            f"use dim={2 if dim2 else 3} instead (Design Decision #40)",
            DeprecationWarning,
            stacklevel=3,
        )
        return 2 if dim2 else 3

    if dim is None:
        return default

    if isinstance(dim, str):
        lowered = dim.lower()
        if lowered not in ("2d", "3d"):
            raise ValueError(f"dim must be 2 or 3, got {dim!r}")
        warnings.warn(
            f"dim={dim!r} is deprecated and will be removed; "
            f"use dim={lowered[0]} instead (Design Decision #40)",
            DeprecationWarning,
            stacklevel=3,
        )
        return int(lowered[0])

    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim!r}")
    return int(dim)


def resolve_flipped_bool(new_value, old_value, *, new_name: str, old_name: str) -> bool:
    """Resolve a boolean parameter that was renamed *and* had its sense flipped.

    ``old_value is None`` means the retired keyword wasn't passed, in which
    case ``new_value`` is returned unchanged. Otherwise the retired value is
    negated into the new sense and a :class:`DeprecationWarning` is raised.

    Passing both is an error rather than a silent preference: the two
    keywords express opposite senses of the same switch, so a caller that
    sets both has a genuine contradiction worth surfacing.
    """
    if old_value is None:
        return bool(new_value)
    warnings.warn(
        f"{old_name}= is deprecated and will be removed; "
        f"use {new_name}={not old_value} instead (Design Decision #41)",
        DeprecationWarning,
        stacklevel=3,
    )
    return not bool(old_value)
