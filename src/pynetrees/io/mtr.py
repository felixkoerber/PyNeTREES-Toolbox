"""MATLAB `.mtr` tree archive import.

Ports the `.mtr` branch of `IO/load_tree.m`. `.mtr` is, per that function's
own comment, "just a matlab workspace" -- a `.mat` file holding one `tree`
struct, a cell array of tree structs, or a 2-level-nested cell array of them
(the `cgui_tree` layout).

Both MATLAB on-disk formats are handled: v5/v7 via `scipy.io`, and v7.3
(HDF5) via the optional `mat73` dependency. See :mod:`pynetrees.io._matlab`
for why that particular library, and why it matters (MATLAB's `save_tree`
writes v7.3 unconditionally).

Extra struct fields the MATLAB side sometimes carries (`col`, `NID`,
`jpoints`, `Rho_soma`, `Rho_AIS`, ...) are outside this port's `Tree` model
and are dropped; `Ri`/`Gm`/`Cm` *are* read, since Phase 8 gave `Tree` real
homes for them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core import Tree
from ._matlab import contains_tree, looks_like_tree, read_matlab, to_plain

__all__ = ["load_mtr", "save_mtr"]


def load_mtr(path: str | Path, variable: str | None = None) -> Tree | list[Tree]:
    """Load a MATLAB `.mtr`/`.mat` file into a Tree, or a list of Trees.

    Parameters
    ----------
    path : str or Path
    variable : str, optional
        Which workspace variable holds the tree(s). By default the loader
        takes the sole variable that looks like tree data, whatever it is
        called. Name it explicitly when a file holds several candidates.

    Returns
    -------
    Tree or list[Tree]
        A single Tree if the file holds exactly one, else a list -- the
        common case for `.mtr` archives, most of which bundle a whole
        population of reconstructions.

    Raises
    ------
    ValueError
        If no tree-shaped variable is found, if ``variable`` names one that
        isn't there, or if several candidates exist (the message lists them
        so you can pick).

    Notes
    -----
    **The name of the variable carries no weight.** Two earlier rules were
    both wrong: requiring it to be called exactly ``tree`` rejected any
    workspace saved by hand or by T2N, and *preferring* ``tree`` when
    several candidates existed silently loaded one population out of a file
    holding two, with nothing to say the rest had been dropped. What a
    variable is called is not evidence about which one you meant, so a file
    with one candidate loads it and a file with several refuses.
    """
    path = Path(path)
    data = read_matlab(path)

    raw = _select_variable(data, path, variable)
    structs = [to_plain(s) for s in _flatten(raw)]
    if not structs:
        raise ValueError(f"{path}: tree variable is empty")

    trees = [
        _struct_to_tree(s, name=s.get("name") or f"{path.stem}_{i}")
        for i, s in enumerate(structs)
    ]
    return trees[0] if len(trees) == 1 else trees


def _select_variable(data: dict, path: Path, variable: str | None):
    """Pick the workspace variable holding the tree data."""
    if variable is not None:
        if variable not in data:
            available = sorted(k for k in data if not k.startswith("__"))
            raise ValueError(
                f"{path}: no variable named {variable!r}; found {available}"
            )
        return data[variable]

    candidates = {
        k: v
        for k, v in data.items()
        if not k.startswith("__") and contains_tree(v)
    }
    if len(candidates) == 1:
        return next(iter(candidates.values()))
    if not candidates:
        available = sorted(k for k in data if not k.startswith("__"))
        raise ValueError(
            f"{path}: no tree data found. Variables in this file: {available}"
        )
    # Several candidates: refuse rather than pick. A name preference would
    # silently load one population out of a file holding two, and the
    # caller would have no sign that the rest was dropped.
    raise ValueError(
        f"{path}: several variables hold tree data ({sorted(candidates)}). "
        f"Pass variable='<name>' to choose one."
    )


def _flatten(raw):
    """Yield tree structs from ``raw``, whatever depth they are nested at.

    ``raw`` may be a single struct, a flat list of structs, or a 2-level
    nested list (MATLAB cell array of cell arrays, per `load_tree.m`'s
    docstring). `mat73` adds a further wrapping level of its own, so this
    recurses rather than indexing at a fixed depth.

    Structs arrive as either ``dict`` (usual) or ``mat_struct`` (scipy, at
    certain nesting depths) -- :func:`~pynetrees.io._matlab.to_plain` reconciles
    the two. Without that, `dLPTCs.mtr` failed to load at all.
    """
    raw = to_plain(raw)
    if looks_like_tree(raw):
        yield raw
    elif isinstance(raw, (list, tuple, np.ndarray)):
        for item in raw:
            yield from _flatten(item)
    elif isinstance(raw, dict):
        raise ValueError(
            f"struct is missing required tree fields (has: {sorted(raw)})"
        )
    else:
        raise ValueError(f"unexpected tree variable contents: {type(raw)}")


def _scalar_or_array(value):
    """Keep Ri/Gm/Cm as a float when uniform, else as a per-node array."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 0:
        return None
    return float(arr[0]) if arr.size == 1 else arr


def _struct_to_tree(struct: dict, name: str) -> Tree:
    """Build a :class:`~pynetrees.Tree` from one decoded MATLAB tree struct."""
    if "dA" not in struct or "X" not in struct:
        raise ValueError(
            f"tree struct is missing required fields (has: {sorted(struct)})"
        )

    n = len(np.ravel(struct["X"]))
    R_1based = np.asarray(struct.get("R", np.ones(n))).astype(int).ravel()
    rnames = struct.get("rnames", ["1"])
    if isinstance(rnames, str):
        rnames = [rnames]
    rnames = [str(r) for r in np.ravel(rnames)]

    return Tree(
        dA=struct["dA"],
        X=np.asarray(struct["X"], dtype=float).ravel(),
        Y=np.asarray(struct["Y"], dtype=float).ravel(),
        Z=np.asarray(struct["Z"], dtype=float).ravel(),
        D=np.asarray(struct.get("D", np.ones(n)), dtype=float).ravel(),
        R=R_1based - 1,  # MATLAB rnames indices are 1-based
        rnames=rnames,
        name=str(name),
        frustum=bool(struct.get("frustum", False)),
        Ri=_scalar_or_array(struct.get("Ri")),
        Gm=_scalar_or_array(struct.get("Gm")),
        Cm=_scalar_or_array(struct.get("Cm")),
    )


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def save_mtr(trees: Tree | list[Tree], path: str | Path) -> Path:
    """Write one tree, or a list of them, to a MATLAB-readable ``.mtr``.

    Ports ``IO/save_tree.m``, which is a one-liner: ``save (name, 'tree',
    '-v7.3')``. A ``.mtr`` is just a ``.mat`` workspace holding a variable
    called ``tree``, which is either a struct or a cell array of structs.

    Returns the path written.

    Notes
    -----
    **This writes v5, not v7.3.** MATLAB's `save_tree` forces ``-v7.3``
    (HDF5); this uses ``scipy.io.savemat``, which writes the older v5
    container. MATLAB's `load_tree` calls plain ``load``, which reads
    either, so nothing on the MATLAB side can tell the difference. The
    reason to prefer v5 is that writing a MATLAB *struct* into v7.3 by hand
    means reproducing undocumented ``MATLAB_class`` attributes and object
    references, i.e. reimplementing a format MATLAB never specified --
    whereas ``savemat`` is a maintained, tested writer for the format
    MATLAB has documented for decades. The only real v5 limit, 2 GB per
    variable, is orders of magnitude beyond any morphology.

    Region names round-trip as ``rnames`` and the electrotonic constants as
    ``Ri``/``Gm``/``Cm``; the ``R`` indices are converted back to MATLAB's
    1-based convention on the way out.
    """
    from scipy.io import savemat

    path = Path(path)
    if path.suffix != ".mtr":
        path = path.with_suffix(path.suffix + ".mtr")

    if isinstance(trees, Tree):
        payload = _tree_to_struct(trees)
    else:
        # a MATLAB cell array of structs is an object array to savemat
        payload = np.empty((1, len(trees)), dtype=object)
        for i, tree in enumerate(trees):
            payload[0, i] = _tree_to_struct(tree)

    savemat(str(path), {"tree": payload}, do_compression=True)
    return path


def _tree_to_struct(tree: Tree) -> dict:
    """One Tree as the field dict `savemat` turns into a MATLAB struct."""
    from scipy import sparse

    struct = {
        "dA": sparse.csc_matrix(tree.dA),
        "X": tree.X.reshape(-1, 1),
        "Y": tree.Y.reshape(-1, 1),
        "Z": tree.Z.reshape(-1, 1),
        "D": tree.D.reshape(-1, 1),
        "R": (np.asarray(tree.R, dtype=int) + 1).reshape(-1, 1),
        "rnames": np.array([str(r) for r in tree.rnames], dtype=object).reshape(1, -1),
        "name": str(tree.name),
    }
    if tree.frustum:
        struct["frustum"] = 1
    for field in ("Ri", "Gm", "Cm"):
        value = getattr(tree, field, None)
        if value is not None:
            struct[field] = value
    return struct
