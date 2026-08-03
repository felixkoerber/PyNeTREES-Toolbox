"""MATLAB `.mtr` tree archive import (MATLAB v5 / `.mat` format only).

Ports the `.mtr` branch of `IO/load_tree.m`. `.mtr` is, per that function's
own comment, "just a matlab workspace" -- a `.mat` file holding one `tree`
struct, a cell array of tree structs, or (rarely) a 2-level-nested cell
array of them. Deferred in Phase 1 (see PORT_STATUS.md Design Decision 9)
for lack of a concrete need; the bundled `Active GC Model/morphos/*.mtr`
granule-cell reconstructions are that concrete need, so this phase adds a
real (if scoped-down) loader rather than a one-off script.

**Scope**: MATLAB v5 `.mat` files only (`scipy.io.loadmat`). Some bundled
`.mtr` files (anything MATLAB saved with `-v7.3`, which is HDF5-based) need
`h5py` and aren't handled here -- `load_mtr` raises a clear error naming the
file and format rather than scipy's generic `NotImplementedError`. Extra
struct fields the MATLAB side sometimes carries (`Ri`, `Gm`, `Cm`, `col`,
`NID`, `jpoints`, `Rho_soma`, `Rho_AIS`, ...) are outside this port's `Tree`
model (Phase 8's electrotonics work is the natural home for `Ri`/`Gm`/`Cm`
specifically) and are silently dropped -- only `dA`/`X`/`Y`/`Z`/`D`/`R`/
`rnames`/`name`/`frustum` are read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core import Tree


def load_mtr(path: str | Path) -> Tree | list[Tree]:
    """Load a MATLAB v5 `.mtr` file into a Tree, or a list of Trees if it
    holds more than one (the common case for `.mtr` archives -- most
    bundle a whole population of reconstructions)."""
    from scipy.io import loadmat

    path = Path(path)
    try:
        # pass a str, not the Path object: scipy's mat reader gives a clean
        # FileNotFoundError for a missing str path but an opaque OSError
        # ("needs file name or open file-like object") for a missing Path
        data = loadmat(str(path), simplify_cells=True)
    except NotImplementedError as exc:
        raise ValueError(
            f"{path}: saved as MATLAB v7.3 (HDF5) -- load_mtr only reads v5 "
            ".mat files; re-save from MATLAB as v5, or load via h5py"
        ) from exc

    if "tree" not in data:
        raise ValueError(f"{path}: no 'tree' variable found in this .mtr file")

    structs = list(_flatten(data["tree"]))
    if not structs:
        raise ValueError(f"{path}: 'tree' variable is empty")

    trees = [_struct_to_tree(s, name=s.get("name") or f"{path.stem}_{i}") for i, s in enumerate(structs)]
    return trees[0] if len(trees) == 1 else trees


def _flatten(raw):
    """Yield tree structs from `raw`, which may be a single struct (dict),
    a flat list of structs, or a 2-level-nested list of structs (MATLAB
    cell array of cell arrays, per load_tree.m's docstring)."""
    if isinstance(raw, dict):
        yield raw
    elif isinstance(raw, (list, np.ndarray)):
        for item in raw:
            yield from _flatten(item)
    else:
        raise ValueError(f"unexpected 'tree' contents: {type(raw)}")


def _struct_to_tree(struct: dict, name: str) -> Tree:
    if "dA" not in struct or "X" not in struct:
        raise ValueError(f"tree struct is missing required fields (has: {list(struct)})")

    n = len(np.ravel(struct["X"]))
    R_1based = np.asarray(struct.get("R", np.ones(n))).astype(int).ravel()
    rnames = list(struct.get("rnames", ["1"]))
    if isinstance(rnames, str):
        rnames = [rnames]

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
    )
