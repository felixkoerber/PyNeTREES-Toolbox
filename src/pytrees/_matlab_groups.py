"""Loading a `.mtr` that holds *groups* of trees rather than a flat list.

`load_tree.m` documents a 2-level cell array -- ``{{treei1, treei2, ...},
{treej1, ...}, ...}`` -- as the layout `cgui_tree` writes for a population
split into named groups. :func:`~pytrees.load_mtr` deliberately flattens
that away, since almost every caller wants "the trees in this file". This
module keeps the grouping instead, which is what `stats_tree`'s
group-comparison API consumes.

Kept separate from ``io/mtr.py`` because it answers a different question:
``load_mtr`` answers "which trees are in here", this answers "how are they
grouped". Merging them would mean one function whose return type depends on
the file's nesting depth, which is exactly the kind of surprise this port
has avoided elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import Tree
from .io._matlab import looks_like_tree, read_matlab, to_plain
from .io.mtr import _flatten, _struct_to_tree


def load_grouped_mtr(
    path: str | Path, variable: str = "tree"
) -> dict[str, list[Tree]]:
    """Load a 2-level `.mtr` cell array as ``{group_name: [Tree, ...]}``.

    Parameters
    ----------
    path : str or Path
    variable : str, default "tree"
        Workspace variable holding the nested cell array.

    Returns
    -------
    dict[str, list[Tree]]
        Group names are taken from the trees themselves where they share a
        common alphabetic prefix (the archives name trees like ``HSE_1``,
        ``HSE_2``), and fall back to ``"group0"``, ``"group1"``, ... when
        they don't. Insertion order matches the file's.

    Raises
    ------
    ValueError
        If the file holds a flat list rather than groups -- use
        :func:`~pytrees.load_mtr` for those, rather than getting back a
        single spurious group.
    """
    path = Path(path)
    raw = to_plain(read_matlab(path)[variable])

    if looks_like_tree(raw):
        raise ValueError(
            f"{path}: holds a single tree, not groups -- use load_mtr instead"
        )

    groups: dict[str, list[Tree]] = {}
    for i, group in enumerate(raw):
        structs = [to_plain(s) for s in _flatten(group)]
        if not structs:
            continue
        trees = [
            _struct_to_tree(s, name=s.get("name") or f"{path.stem}_{i}_{j}")
            for j, s in enumerate(structs)
        ]
        name = _group_name(trees, i)
        while name in groups:  # never silently merge two distinct groups
            name = f"{name}_{i}"
        groups[name] = trees

    if len(groups) <= 1:
        raise ValueError(
            f"{path}: found {len(groups)} group(s); this looks like a flat "
            f"archive -- use load_mtr instead"
        )
    return groups


def _group_name(trees: list[Tree], index: int) -> str:
    """Name a group from the longest prefix its trees' names share.

    Trees in these archives are named ``<class><n>`` -- ``dhse1``,
    ``dhse2``, ... -- so the common prefix *is* the class name.

    The prefix must be the **longest common** one, not merely the leading
    alphabetic run: the dLPTCs groups are ``dvs2``, ``dvs3`` and ``dvs4``,
    which share the alphabetic prefix ``dvs`` and are distinguished only by
    the digit after it. Stripping digits collapsed three groups into one and
    silently lost 20 of the 55 trees.
    """
    names = [tree.name for tree in trees if tree.name]
    if not names:
        return f"group{index}"

    prefix = names[0]
    for name in names[1:]:
        while not name.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return f"group{index}"
    return prefix.rstrip("_-") or f"group{index}"


def group_arrays(groups: dict[str, list[Tree]]) -> tuple[list[list[Tree]], list[str]]:
    """Split a group dict into the ``(trees, names)`` pair `stats_tree` takes.

    Convenience for the common call, which otherwise needs two parallel
    comprehensions that must not drift apart::

        stats = stats_tree(*group_arrays(dLPTCs_trees()))
    """
    return list(groups.values()), list(groups.keys())


def _assert_rectangular(groups: dict[str, list[Tree]]) -> None:
    """Sanity check used by tests: every group non-empty and well-formed."""
    for name, trees in groups.items():
        assert trees, f"group {name!r} is empty"
        for tree in trees:
            assert isinstance(tree, Tree)
            assert np.isfinite(tree.X).all()
