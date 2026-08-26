"""pynetrees' own portable Tree serialization: full-fidelity, single-tree.

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


def save_npz(tree: Tree, path: str | Path) -> Path:
    """Save a Tree to pynetrees' native ``.npz``-based format."""
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
    return path


def load_npz(path: str | Path) -> Tree:
    """Load a Tree previously written by :func:`save_npz`."""
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


# ---------------------------------------------------------------------------
# format dispatch
# ---------------------------------------------------------------------------

#: Extension -> loader. `.mat` is accepted alongside `.mtr` because a `.mtr`
#: *is* a `.mat` and users rename them freely.
_LOADERS: dict[str, str] = {
    ".npz": "npz",
    ".swc": "swc",
    ".neu": "neu",
    ".nmf": "nmf",
    ".mtr": "mtr",
    ".mat": "mtr",
    ".asc": "neurolucida",
}

_SAVERS: dict[str, str] = {
    ".npz": "npz",
    ".swc": "swc",
    ".nmf": "nmf",
    ".mtr": "mtr",
    ".mat": "mtr",
    ".hoc": "hoc",
    ".nrn": "nrn",
    ".xml": "neuroml",
}


def load_tree(path: str | Path, **kwargs) -> Tree | list[Tree]:
    """Load a tree from any format this port reads, chosen by extension.

    ==========  ============================================================
    ``.npz``    pynetrees' own lossless format (:func:`save_npz`)
    ``.mtr``    MATLAB tree archive -- also ``.mat``
    ``.swc``    the standard SWC text format
    ``.neu``    NEURON transfer format (see :func:`~pynetrees.io.load_neu`)
    ``.nmf``    the toolbox's HDF5 extended SWC
    ``.asc``    Neurolucida ASCII
    ==========  ============================================================

    Extra keyword arguments are passed to the format's own loader, e.g.
    ``load_tree("cell.neu", keep_sections=True)``. A path with no extension
    at all is taken as ``.npz``, which is what `save_tree` writes when given
    the same.

    Returns a single :class:`Tree`, or a list when the file holds several.

    Notes
    -----
    Two things MATLAB's `load_tree` does that this does not.

    It opens a **file dialog** when called with no argument. That is a
    reasonable default for a GUI-first toolbox and a bad one for a library:
    a function that blocks on a window cannot be called from a script, a
    notebook running headless, or a test. Use a file dialog in your own
    code if you want one; this function needs a path.

    It also applies :func:`~pynetrees.repair_tree` automatically to
    ``.swc``/``.neu``/``.nmf`` (its ``'-r'`` default), silently altering
    what was on disk. Loading and repairing are kept separate here so that
    "what does this file contain" has an answer -- call
    ``repair_tree`` yourself when you want it.
    """
    path = Path(path)
    suffix = path.suffix.lower() or ".npz"  # no extension means the native one
    if suffix not in _LOADERS:
        raise ValueError(
            f"{path.name}: unrecognised extension {suffix!r}; "
            f"this port reads {', '.join(sorted(_LOADERS))}"
        )
    return _loader(_LOADERS[suffix])(path, **kwargs)


def save_tree(tree: Tree | list[Tree], path: str | Path, **kwargs) -> Path:
    """Save a tree in the format named by ``path``'s extension.

    ==========  ============================================================
    ``.npz``    pynetrees' own lossless format -- the default and the only one
                that stores everything a :class:`Tree` holds
    ``.mtr``    MATLAB tree archive, for handing work back to MATLAB
    ``.swc``    standard SWC (loses region *names*, keeps their codes)
    ``.nmf``    HDF5 extended SWC
    ``.hoc``    NEURON cell file -- ``style="template"`` for a template
    ``.nrn``    NEURON, one section per graph segment
    ``.xml``    NeuroML
    ==========  ============================================================

    These last three are **export only**: nothing here reads them back, so
    a tree saved as ``.hoc`` cannot be reloaded with :func:`load_tree`.

    Only ``.mtr`` accepts a list of trees; the rest hold one tree per file.
    Returns the path actually written -- the extension is added if missing,
    so this is not always the path passed in.
    """
    path = Path(path)
    suffix = path.suffix.lower() or ".npz"  # no extension means the native one
    if suffix not in _SAVERS:
        raise ValueError(
            f"{path.name}: unrecognised extension {suffix!r}; "
            f"this port writes {', '.join(sorted(_SAVERS))}"
        )
    if isinstance(tree, (list, tuple)) and suffix not in (".mtr", ".mat"):
        raise ValueError(
            f"{suffix} holds a single tree; save each one separately, or use "
            ".mtr, which stores a list"
        )
    return _saver(_SAVERS[suffix])(tree, path, **kwargs)


def _loader(kind: str):
    """Import a format module lazily, so optional deps stay optional."""
    if kind == "npz":
        return load_npz
    if kind == "swc":
        from .swc import load_swc

        return load_swc
    if kind == "neu":
        from .neu import load_neu

        return load_neu
    if kind == "nmf":
        from .nmf import load_nmf

        return load_nmf
    if kind == "neurolucida":
        from .neurolucida import load_neurolucida

        return load_neurolucida
    from .mtr import load_mtr

    return load_mtr


def _saver(kind: str):
    if kind == "npz":
        return save_npz
    if kind == "swc":
        from .swc import save_swc

        return _returning_path(save_swc)
    if kind == "nmf":
        from .nmf import save_nmf

        return save_nmf
    if kind == "hoc":
        from .hoc import save_hoc

        return save_hoc
    if kind == "nrn":
        from .hoc import save_nrn

        return save_nrn
    if kind == "neuroml":
        from .neuroml import save_neuroml

        return save_neuroml
    from .mtr import save_mtr

    return save_mtr


def _returning_path(writer):
    """`save_swc` predates the "return where you wrote it" convention."""

    def wrapper(tree, path, **kwargs):
        writer(tree, path, **kwargs)
        return Path(path)

    return wrapper
