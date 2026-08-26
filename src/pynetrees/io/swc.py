"""SWC format I/O.

Mirrors the ``'.swc'`` branch of ``IO/load_tree.m`` (reading) and the core of
``IO/swc_tree.m`` (writing).

Design note (see PORT_STATUS.md, Design Decisions #8): this reader does
*not* call ``repair_tree`` -- that wasn't ported yet when this module was
written (Phase 1, before Phase 4). It faithfully builds whatever topology
the file encodes, including:

- unsorted / non-contiguous SWC node indices (handled via an index lookup,
  same idea as MATLAB's fallback branch), and
- multiple roots in one file (returned as a list of :class:`Tree`, one per
  connected component, matching MATLAB's behavior of returning a cell array
  in that case).

``save_swc`` writes full ``%.8f``-equivalent precision (matching MATLAB's
``swc_tree.m``) and gets parent indices from the real, general
:func:`~pynetrees.idpar_tree` (Phase 2) rather than the single-purpose
private helper this module used before that existed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

from ..core import NO_PARENT, Tree
from ..graphtheory import idpar_tree

SWC_COLUMNS = ("index", "type", "x", "y", "z", "radius", "parent")


def load_swc(path: str | Path) -> Tree | list[Tree]:
    """Load an SWC file into one :class:`Tree`, or a list if it has >1 root."""
    path = Path(path)
    swc = _read_swc_rows(path)

    file_index = swc[:, 0].astype(int)
    rtype = swc[:, 1].astype(int)
    X, Y, Z = swc[:, 2], swc[:, 3], swc[:, 4]
    D = swc[:, 5] * 2.0
    parent_file_index = swc[:, 6].astype(int)

    n = len(file_index)
    index_to_node = {idx: node for node, idx in enumerate(file_index)}
    parent_node = np.full(n, NO_PARENT, dtype=int)
    for node, pid in enumerate(parent_file_index):
        if pid != -1:
            parent_node[node] = index_to_node[pid]

    children: list[list[int]] = [[] for _ in range(n)]
    for node, p in enumerate(parent_node):
        if p != NO_PARENT:
            children[p].append(node)

    roots = np.flatnonzero(parent_node == NO_PARENT)
    if len(roots) == 0:
        raise ValueError(f"{path}: no root node found (no row with parent -1)")

    if len(roots) == 1:
        all_nodes = np.arange(n)
        return _tree_from_nodes(
            all_nodes, parent_node, X, Y, Z, D, rtype, name=path.stem
        )

    trees = []
    for i, root in enumerate(roots):
        subset = _connected_nodes(int(root), children)
        trees.append(
            _tree_from_nodes(
                subset, parent_node, X, Y, Z, D, rtype, name=f"{path.stem}_{i}"
            )
        )
    return trees


def save_swc(tree: Tree, path: str | Path) -> None:
    """Write a single-root :class:`Tree` to an SWC file.

    Node ``i`` is written at SWC index ``i + 1``; the root's parent column
    is ``-1``. Region names round-trip through the SWC integer ``type``
    column when they parse as integers (as produced by :func:`load_swc`),
    otherwise they're replaced by a 1-based region index.
    """
    path = Path(path)
    n = tree.n_nodes
    parent = idpar_tree(tree, root_self=False)

    try:
        type_values = [int(name) for name in tree.rnames]
    except ValueError:
        type_values = list(range(1, len(tree.rnames) + 1))
    rtype = np.asarray(type_values)[tree.R]

    lines = [
        f"# TREES toolbox (Python port) tree - {tree.name}",
        "# written by pynetrees.io.swc.save_swc",
        "#",
        "# inode R X Y Z D/2 idpar",
    ]
    for i in range(n):
        swc_parent = -1 if parent[i] == NO_PARENT else parent[i] + 1
        lines.append(
            f"{i + 1} {rtype[i]} {tree.X[i]:12.8f} {tree.Y[i]:12.8f} "
            f"{tree.Z[i]:12.8f} {tree.D[i] / 2:12.8f} {swc_parent}"
        )
    path.write_text("\n".join(lines) + "\n")


def _read_swc_rows(path: Path) -> np.ndarray:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(v) for v in line.split()])
    if not rows:
        raise ValueError(f"{path}: no data rows found")
    return np.asarray(rows, dtype=float)


def _connected_nodes(root: int, children: list[list[int]]) -> np.ndarray:
    seen = [root]
    stack = [root]
    while stack:
        node = stack.pop()
        for c in children[node]:
            seen.append(c)
            stack.append(c)
    return np.array(sorted(seen))


def _tree_from_nodes(subset, parent_node, X, Y, Z, D, rtype, name: str) -> Tree:
    n = len(subset)
    old_to_new = {old: new for new, old in enumerate(subset)}

    rows, cols = [], []
    for new, old in enumerate(subset):
        p = parent_node[old]
        if p != NO_PARENT:
            rows.append(new)
            cols.append(old_to_new[p])
    dA = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n, n)
    ).tocsr()

    uniq_types, R = np.unique(rtype[subset], return_inverse=True)
    rnames = [str(t) for t in uniq_types]

    return Tree(
        dA=dA,
        X=X[subset],
        Y=Y[subset],
        Z=Z[subset],
        D=D[subset],
        R=R,
        rnames=rnames,
        name=name,
    )
