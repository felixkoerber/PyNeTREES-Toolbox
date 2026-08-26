"""``.nmf`` -- the toolbox's HDF5-backed extended SWC format.

Ports ``IO/nmf_tree.m`` (writer) and the ``'.nmf'`` branch of
``IO/load_tree.m`` (reader). The layout is a single HDF5 group ``/swc``
holding one dataset per SWC column::

    /swc/index          1 .. N
    /swc/parent_index   1-based parent, -1 at the root
    /swc/x, /y, /z      coordinates [um]
    /swc/r              *radius*, i.e. half the diameter
    /swc/type           region index per node

plus ``soma_type`` and ``info`` string attributes on the group.

Why bother, when plain SWC exists: SWC is a fixed seven-column text format
with no room for anything else, whereas MATLAB's reader copies **any**
extra dataset under ``/swc`` straight onto the tree struct. So ``.nmf`` is
the toolbox's answer to "SWC plus whatever else this analysis needed".
This port reads the seven standard columns and ignores extra datasets
rather than attaching them, because :class:`Tree` has a fixed set of
fields -- read such a file with ``h5py`` directly if you need them.

One deliberate addition: region *names* are written as a group attribute.
MATLAB's writer stores only ``tree.R``, the region **indices**, so a
``.nmf`` round trip through MATLAB renames ``axon``/``dendrite`` to
``1``/``2``. Storing the names in an attribute makes the round trip
lossless here while remaining invisible to MATLAB's reader, which iterates
datasets and ignores attributes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

from ..core import NO_PARENT, Tree

__all__ = ["load_nmf", "save_nmf"]

_GROUP = "/swc"
_REGION_NAMES = "region_names"


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency message
        raise ImportError(
            "reading and writing .nmf files needs h5py: "
            "`pip install pynetrees[nmf]`"
        ) from exc
    return h5py


def load_nmf(path: str | Path) -> Tree:
    """Load a ``.nmf`` file into a :class:`Tree`.

    Notes
    -----
    ``/swc/r`` is a radius; :attr:`Tree.D` is a diameter, so it is doubled
    on the way in and halved on the way out -- the same convention MATLAB
    uses, and the usual source of factor-of-two errors between SWC-family
    formats.
    """
    h5py = _h5py()
    path = Path(path)

    with h5py.File(path, "r") as handle:
        if _GROUP.strip("/") not in handle:
            raise ValueError(f"{path}: no '/swc' group -- not an .nmf file?")
        group = handle[_GROUP]
        columns = {key: np.asarray(group[key]).ravel() for key in group}
        names = group.attrs.get(_REGION_NAMES)

    for required in ("parent_index", "x", "y", "z"):
        if required not in columns:
            raise ValueError(f"{path}: missing required dataset /swc/{required}")

    idpar = columns["parent_index"].astype(int)
    n = len(idpar)
    # MATLAB writes 1-based parents with -1 at the root
    parent = np.where(idpar > 0, idpar - 1, NO_PARENT)
    parent[0] = NO_PARENT

    rows = np.flatnonzero(parent != NO_PARENT)
    dA = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, parent[rows])), shape=(n, n)
    ).tocsr()

    types = columns.get("type", np.ones(n))
    unique, R = np.unique(types, return_inverse=True)
    if names is not None:
        rnames = [str(x) for x in np.atleast_1d(names)]
        if len(rnames) != len(unique):
            raise ValueError(
                f"{path}: region_names has {len(rnames)} entries but "
                f"/swc/type uses {len(unique)} distinct values"
            )
    else:
        # MATLAB's fallback: name each region after its numeric code
        rnames = [str(int(u)) if float(u).is_integer() else str(u) for u in unique]

    diameter = columns["r"] * 2 if "r" in columns else np.ones(n)
    return Tree(
        dA=dA,
        X=columns["x"].astype(float),
        Y=columns["y"].astype(float),
        Z=columns["z"].astype(float),
        D=diameter.astype(float),
        R=R,
        rnames=rnames,
        name=path.stem,
    )


def save_nmf(tree: Tree, path: str | Path) -> Path:
    """Write a :class:`Tree` to a ``.nmf`` file.

    Returns the path written, so a caller that let the suffix be added
    knows where the file went.
    """
    h5py = _h5py()
    path = Path(path)
    if path.suffix != ".nmf":
        path = path.with_suffix(path.suffix + ".nmf")

    n = tree.n_nodes
    idpar = _parent_indices(tree)

    with h5py.File(path, "w") as handle:
        group = handle.create_group(_GROUP.strip("/"))
        group.attrs["soma_type"] = "Multiple cylinders"
        group.attrs["info"] = f"TREES toolbox tree - {tree.name}"
        group.attrs[_REGION_NAMES] = [str(r) for r in tree.rnames]

        # column vectors, matching what MATLAB's h5create([N 1]) produces
        for key, values in (
            ("index", np.arange(1, n + 1)),
            ("parent_index", idpar),
            ("x", tree.X),
            ("y", tree.Y),
            ("z", tree.Z),
            ("r", tree.D / 2),
            ("type", tree.R + 1),  # back to MATLAB's 1-based region indices
        ):
            group.create_dataset(key, data=np.asarray(values, dtype=float).reshape(n, 1))
    return path


def _parent_indices(tree: Tree) -> np.ndarray:
    """1-based parent index per node, ``-1`` at the root (MATLAB's layout)."""
    parent = np.full(tree.n_nodes, NO_PARENT, dtype=int)
    coo = tree.dA.tocoo()
    parent[coo.row] = coo.col
    return np.where(parent == NO_PARENT, -1, parent + 1)
