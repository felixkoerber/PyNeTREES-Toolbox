"""What functions do when handed a tree with no nodes.

`delete_tree` can produce an empty tree (Design Decision #38), and a
population is allowed to contain one -- deleting it instead would renumber
every cell after it, silently. So an empty tree has to be a value you can
*use*, not just one you can create.

The rule, applied at each definition site with :func:`empty_safe` so it is
visible in the source rather than hidden in a base class:

**Per-node quantities** return an empty array of the right shape and dtype.
``len_tree`` of nothing is ``array([])``, not an error -- and crucially not
``array([0.0])``, which would put a phantom node into every downstream sum.

**Sums** return ``0.0``. Total length of no cable is zero.

**Means and fits** return ``nan``. Averaging nothing is *not* zero, and
returning zero there is exactly how a silently-wrong population mean
happens.

**Editing functions** return the empty tree unchanged.

**Genuinely unanswerable questions keep raising** -- ``tree.root``,
``MST_tree`` with no points, ``convexity_tree`` with fewer than two points.
There is no right answer to "where is the root of nothing", and inventing
one would hide a bug rather than handle a case.
"""

from __future__ import annotations

import functools

import numpy as np

__all__ = ["empty_safe", "is_empty"]


def is_empty(tree) -> bool:
    """Does this tree have no nodes?"""
    return getattr(tree, "n_nodes", None) == 0


def empty_safe(result="nodes", dtype=float):
    """Short-circuit a function when its tree is empty.

    Parameters
    ----------
    result : str or callable
        What to return. ``"nodes"`` gives ``(0,)``; ``"nodes3"`` gives
        ``(0, 3)``; ``"pairs"`` gives ``(0, 2)``; ``"tree"`` returns the
        tree itself; ``"zero"`` gives ``0.0``; ``"nan"`` gives ``nan``;
        ``"none"`` gives ``None``. A callable is passed the tree and its
        return value is used, for anything with a richer shape.
    dtype : type, default float
        Element type of the empty array, for the array forms.

    Notes
    -----
    Applied as a decorator so the behaviour is stated where the function is
    defined. The alternative -- one check inside each body -- puts the same
    three lines in fifty places and gets forgotten in the fifty-first.
    """
    shapes = {
        "nodes": lambda tree: np.empty(0, dtype=dtype),
        "nodes3": lambda tree: np.empty((0, 3), dtype=dtype),
        "pairs": lambda tree: np.empty((0, 2), dtype=dtype),
        "tree": lambda tree: tree,
        "zero": lambda tree: 0.0,
        "nan": lambda tree: float("nan"),
        "none": lambda tree: None,
    }
    make = result if callable(result) else shapes[result]

    def decorate(func):
        @functools.wraps(func)
        def wrapper(tree, *args, **kwargs):
            if is_empty(tree):
                return make(tree)
            return func(tree, *args, **kwargs)

        if wrapper.__doc__:
            wrapper.__doc__ = wrapper.__doc__.rstrip() + _NOTE[
                "tree" if result == "tree" else "value"
            ]
        return wrapper

    return decorate


_NOTE = {
    "value": """

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.
    """,
    "tree": """

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.
    """,
}
