"""pytrees' own portable Tree serialization: full-fidelity, single-tree.

Ports the *purpose* of ``IO/save_tree.m``/``IO/load_tree.m``'s ``.mtr``
branch (save/reload a Tree exactly, including things SWC can't represent --
sparse topology beyond a strict SWC parent chain, non-numeric region names,
the ``frustum`` flag) without literally porting ``.mtr``, which is just a
MATLAB ``.mat`` file (deferred, see PORT_STATUS.md Design Decision #9: no
real need for it yet, and no MATLAB compatibility to preserve even if there
were).

Deliberately **not** pickle-based, even though that would be the shortest
implementation: pickle executes arbitrary code on load, which is a bad
default for "open a tree file someone sent you". Instead this uses
``numpy.savez`` (a plain zip of named arrays) -- no code execution on load,
inspectable, and numpy already stores fixed-width string arrays (region
names) natively without needing ``allow_pickle``.

Only handles a single Tree per file. MATLAB's ``save_tree``/``load_tree``
also accept nested cell arrays of many trees; that's population-level
tooling out of scope here (see Phase 9's `list[Tree]` + `pandas` plan) --
save each tree to its own file, or build that batching in the caller.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

from ..core import Tree


def _with_npz_suffix(path: str | Path) -> Path:
    path = Path(path)
    return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")


def save_tree(tree: Tree, path: str | Path) -> None:
    """Save a Tree to pytrees' native ``.npz``-based format."""
    path = _with_npz_suffix(path)
    coo = tree.dA.tocoo()
    np.savez(
        path,
        dA_row=coo.row,
        dA_col=coo.col,
        dA_shape=np.array(coo.shape),
        X=tree.X,
        Y=tree.Y,
        Z=tree.Z,
        D=tree.D,
        R=tree.R,
        rnames=np.array(tree.rnames),
        name=np.array(tree.name),
        frustum=np.array(tree.frustum),
    )


def load_tree(path: str | Path) -> Tree:
    """Load a Tree previously written by :func:`save_tree`."""
    with np.load(_with_npz_suffix(path), allow_pickle=False) as data:
        shape = tuple(int(x) for x in data["dA_shape"])
        dA = sparse.coo_matrix(
            (np.ones(len(data["dA_row"])), (data["dA_row"], data["dA_col"])),
            shape=shape,
        ).tocsr()
        return Tree(
            dA=dA,
            X=data["X"], Y=data["Y"], Z=data["Z"], D=data["D"], R=data["R"],
            rnames=data["rnames"].tolist(),
            name=str(data["name"]),
            frustum=bool(data["frustum"]),
        )
