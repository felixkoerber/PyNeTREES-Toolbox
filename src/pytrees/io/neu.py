"""NEURON ``.neu`` transfer format -- reader.

Ports the ``'.neu'`` branch of ``IO/load_tree.m``. A ``.neu`` file is
written from inside NEURON by the toolbox's own ``IO/neu_tree.hoc``, so it
is a one-way channel: NEURON out, TREES in. There is no MATLAB writer for
it and none here either.

The format is two blocks. First a topology table, one line per NEURON
section::

    axon[0] 0 soma[0] 0 13

meaning *this section, attached at its own 0 end, hangs off* ``soma[0]``
*at that section's 0 end, and carries 13 3D points*. Then a flat list of
those points, ``x y z d``, concatenated section by section in the same
order.

The interesting part of the port is the last column pair: NEURON lets a
section attach to either **end** of its parent, which SWC-style parent
indices cannot express directly. Attaching at the parent's ``1`` end means
"continue from the parent's last point"; attaching at its ``0`` end means
"branch from the parent's first point". Both cases are resolved into
ordinary parent indices here, exactly as MATLAB does.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse

from ..core import NO_PARENT, Tree

__all__ = ["load_neu"]

_SECTION_COUNT = re.compile(r"#\s*section lines:\s*(\d+)")
_POINT_COUNT = re.compile(r"#\s*3d points:\s*(\d+)")


def load_neu(path: str | Path, keep_sections: bool = False) -> Tree | list[Tree]:
    """Load a NEURON ``.neu`` file.

    Parameters
    ----------
    path : str or Path
    keep_sections : bool, default False
        Make every NEURON section its own region. By default the bracketed
        index is stripped, so ``axon[0]``, ``axon[1]``, ... collapse into a
        single ``axon[]`` region -- which is almost always what is wanted,
        since NEURON's section names are an implementation detail of the
        model, not an anatomical labelling. MATLAB spells this ``'-ks'``.

    Returns
    -------
    Tree or list[Tree]
        A list when the file holds several unconnected cells.

    Notes
    -----
    MATLAB skips the file header by reading exactly 16 whitespace-separated
    tokens (``textscan (neufid, '%s', 16)``) -- the token count of the
    three header lines its own writer happens to emit. Any other comment
    text silently shifts the parse. This reader locates the
    ``# section lines:`` and ``# 3d points:`` markers instead, so a file
    with a different preamble still loads.

    MATLAB additionally rejects any file where a section attaches at its
    *own* ``1`` end (``'sorry!! I assume that each new branch is attached
    at 0 end'``); the same restriction applies here, with the reason spelled
    out, because such a section would run backwards relative to its point
    list.
    """
    path = Path(path)
    text = path.read_text()

    names, own_end, parents, parent_end, counts = _read_sections(text, path)
    points = _read_points(text, path, counts.sum())

    if (own_end != 0).any():
        bad = names[own_end != 0][0]
        raise ValueError(
            f"{path}: section {bad!r} attaches to its parent at its own '1' "
            "end, so its 3D points run backwards relative to the connection; "
            "this reader (like MATLAB's) only handles sections attached at "
            "their '0' end"
        )

    parent_section = _resolve_parent_sections(names, parents)
    parent_node = _node_parents(counts, parent_section, parent_end)
    regions, rnames = _regions(names, counts, keep_sections)

    return _split_trees(points, parent_node, regions, rnames, path.stem)


def _read_sections(text: str, path: Path):
    """The topology table: name, own end, parent name, parent end, n points."""
    match = _SECTION_COUNT.search(text)
    if match is None:
        raise ValueError(f"{path}: no '# section lines:' marker -- not a .neu file?")
    n_sections = int(match.group(1))

    lines = _data_lines(text[match.end():], n_sections, path, "section")
    names, own_end, parents, parent_end, counts = [], [], [], [], []
    for line in lines:
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"{path}: expected 5 fields in section line, got {len(fields)}: {line!r}"
            )
        names.append(fields[0])
        own_end.append(int(fields[1]))
        parents.append(fields[2])
        parent_end.append(int(fields[3]))
        counts.append(int(fields[4]))
    return (np.array(names), np.array(own_end), np.array(parents),
            np.array(parent_end), np.array(counts))


def _read_points(text: str, path: Path, expected: int) -> np.ndarray:
    """The 3D point block: ``x y z d`` per node.

    The ``# 3d points:`` header is **not** trusted, because the NEURON-side
    writer gets it wrong: on all three fixtures shipped with the toolbox it
    reports ``n_sections * (points in one section)`` rather than the sum
    over sections -- 780 instead of 1214 for ``GC1.neu``, 875 instead of
    3220 for ``GC.neu``, 180 instead of 6800 for ``GCT.neu``. The section
    table's own counts are authoritative and self-consistent, so those are
    used and the header is only checked to warn. MATLAB sidesteps this by
    reading numbers to end-of-file and never looking at the count at all.
    """
    match = _POINT_COUNT.search(text)
    if match is None:
        raise ValueError(f"{path}: no '# 3d points:' marker")
    declared = int(match.group(1))
    if declared != expected:
        warnings.warn(
            f"{path.name}: header declares {declared} 3D points but the "
            f"section table accounts for {expected}; using the section "
            "table (the NEURON-side writer miscomputes this header)",
            stacklevel=3,
        )

    lines = _data_lines(text[match.end():], expected, path, "point")
    return np.array([[float(v) for v in line.split()] for line in lines])


def _data_lines(text: str, n: int, path: Path, what: str) -> list[str]:
    """The next ``n`` non-empty, non-comment lines."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        out.append(line)
        if len(out) == n:
            return out
    raise ValueError(f"{path}: expected {n} {what} lines, found {len(out)}")


def _resolve_parent_sections(names: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Index of each section's parent section, or -1 for a root."""
    index = {name: i for i, name in enumerate(names.tolist())}
    return np.array([index.get(p, NO_PARENT) for p in parents.tolist()])


def _node_parents(counts: np.ndarray, parent_section: np.ndarray,
                  parent_end: np.ndarray) -> np.ndarray:
    """Per-node parent indices for the concatenated point list.

    Within a section every point follows the one before it. Only the
    *first* point of each section needs a decision, and that is where the
    ``0``/``1`` parent end matters: end 1 continues from the parent's last
    point, end 0 branches from the parent's first.
    """
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ends = np.cumsum(counts) - 1

    parent_node = np.arange(-1, counts.sum() - 1)  # previous point
    attached = parent_section >= 0
    from_end = np.where(parent_end == 1, ends[parent_section], starts[parent_section])
    parent_node[starts[attached]] = from_end[attached]
    parent_node[starts[~attached]] = NO_PARENT
    return parent_node


def _regions(names: np.ndarray, counts: np.ndarray, keep_sections: bool):
    """Region index per node, plus the region names.

    Default collapses ``axon[0]``, ``axon[1]``, ... into ``axon[]``.

    Every bracketed index is blanked, not just the first. MATLAB blanks
    from the first ``[`` to the end of the name -- ``sa = [sa(1 : insa - 1)
    '[]']`` where ``insa = strfind (sa, '[')`` is a *vector* of all bracket
    positions and the colon operator silently takes its first element. For
    a plain ``axon[0]`` the two rules agree, but for NEURON's compound
    names they do not: the toolbox's own ``GCT.neu`` fixture has 90
    sections named ``GC7[0].adendGCL[3]``, ``GC7[0].soma[0]`` and so on,
    all of which MATLAB collapses into a **single** region called
    ``GC7[]``, discarding every anatomical label in the file. Here they
    become ``GC7[].adendGCL[]``, ``GC7[].soma[]``, ... -- which is what the
    stripping was evidently meant to do.
    """
    labels = names if keep_sections else np.array(
        [re.sub(r"\[[^\]]*\]", "[]", name) for name in names.tolist()]
    )
    rnames, section_region = np.unique(labels, return_inverse=True)
    return np.repeat(section_region, counts), [str(r) for r in rnames]


def _split_trees(points: np.ndarray, parent_node: np.ndarray,
                 regions: np.ndarray, rnames: list[str], stem: str):
    """One Tree per root, or a single Tree when there is only one."""
    roots = np.flatnonzero(parent_node == NO_PARENT)
    if len(roots) == 0:
        raise ValueError("no root node found -- every section claims a parent")

    children: list[list[int]] = [[] for _ in range(len(parent_node))]
    for node, parent in enumerate(parent_node.tolist()):
        if parent != NO_PARENT:
            children[parent].append(node)

    trees = []
    for count, root in enumerate(roots.tolist()):
        subset = _reachable(root, children)
        trees.append(_build(subset, points, parent_node, regions, rnames,
                            stem if len(roots) == 1 else f"{stem}_{count + 1}"))
    return trees[0] if len(trees) == 1 else trees


def _reachable(root: int, children: list[list[int]]) -> np.ndarray:
    seen, stack = [root], [root]
    while stack:
        node = stack.pop()
        for child in children[node]:
            seen.append(child)
            stack.append(child)
    return np.array(sorted(seen))


def _build(subset: np.ndarray, points: np.ndarray, parent_node: np.ndarray,
           regions: np.ndarray, rnames: list[str], name: str) -> Tree:
    n = len(subset)
    renumber = {old: new for new, old in enumerate(subset.tolist())}

    rows, cols = [], []
    for new, old in enumerate(subset.tolist()):
        parent = parent_node[old]
        if parent != NO_PARENT:
            rows.append(new)
            cols.append(renumber[parent])
    dA = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n, n)
    ).tocsr()

    # only the regions this subtree actually uses, renumbered from 0
    used, R = np.unique(regions[subset], return_inverse=True)
    return Tree(
        dA=dA,
        X=points[subset, 0].copy(),
        Y=points[subset, 1].copy(),
        Z=points[subset, 2].copy(),
        D=points[subset, 3].copy(),
        R=R,
        rnames=[rnames[i] for i in used.tolist()],
        name=name,
    )
