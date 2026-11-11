"""Reading MATLAB workspaces: `.mat`, and the `.mtr` files TREES writes.

Two on-disk formats exist. v5/v7 files are read by `scipy.io`. v7.3 files are
HDF5, and are what a current MATLAB writes: `save_tree.m` uses `'-v7.3'`
unconditionally. They are read by :mod:`pynetrees.io._matlab_v73`, which keeps
every cell array at its real depth so that nested populations stay nested.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def is_v73(path: str | Path) -> bool:
    """Whether a `.mat`/`.mtr` file is v7.3 (HDF5-based).

    Read from the 128-byte text header MATLAB writes at the start of every
    `.mat` file ("MATLAB 5.0 MAT-file..." / "MATLAB 7.3 MAT-file...").

    Sniffing beats try-scipy-catch-`NotImplementedError`, which is what this
    did first: scipy raises `NotImplementedError` for *most* v7.3 files but
    ``ValueError: embedded null character`` for some, having got far enough
    into the HDF5 bytes to misparse them as a v5 structure. One of the
    bundled granule-cell files hits exactly that path, so the exception type
    is not a reliable signal. The header always is.
    """
    try:
        with open(path, "rb") as fh:
            header = fh.read(128)
    except OSError:
        return False
    return b"MATLAB 7.3" in header


def read_matlab(path: str | Path) -> dict:
    """Read a MATLAB `.mat`/`.mtr` file into a dict of plain Python objects.

    Dispatches on the file's actual format, read from its header rather than
    guessed from its extension.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    dict
        Variable name -> value, with structs as dicts and cell arrays as
        (possibly nested) lists.

    Raises
    ------
    ValueError
        If the file is v7.3 and `h5py` is not installed, naming the install
        command.
    """
    path = Path(path)

    if not is_v73(path):
        from scipy.io import loadmat

        # pass a str, not the Path: scipy's mat reader gives a clean
        # FileNotFoundError for a missing str path but an opaque OSError
        # ("needs file name or open file-like object") for a missing Path
        raw = loadmat(str(path), simplify_cells=False)
        return {key: _unpack_v5(value) for key, value in raw.items()
                if not key.startswith("__")}

    try:
        import h5py  # noqa: F401
    except ImportError:
        raise ValueError(
            f"{path} is a MATLAB v7.3 (HDF5) file, which scipy cannot read. "
            f"Install the optional dependency: pip install 'pynetrees[matlab]'"
        ) from None

    from ._matlab_v73 import read_v73

    return read_v73(path)


def _unpack_v5(value):
    """A v5 variable as plain values, keeping one-element cells as lists.

    scipy's ``simplify_cells`` drops any cell holding a single element, which
    would change the depth of a nested population.
    """
    if not isinstance(value, np.ndarray):
        return value
    if value.dtype.names:
        records = [{name: _unpack_v5(record[name]) for name in value.dtype.names}
                   for record in value.ravel(order="F")]
        return records[0] if len(records) == 1 else records
    if value.dtype == object:
        return [_unpack_v5(item) for item in value.ravel(order="F")]
    if value.dtype.kind in "US":
        return str(value.item()) if value.size == 1 else ""
    if value.size == 1:
        return value.ravel()[0].item()
    return np.squeeze(value)


def to_plain(obj):
    """Normalise one decoded MATLAB value into dicts/lists/arrays.

    `scipy.io.loadmat(simplify_cells=True)` *usually* returns structs as
    dicts, but for some nesting depths it hands back `mat_struct` objects
    instead -- which is why `load_mtr` could not read the bundled
    `dLPTCs.mtr` (55 reconstructions in 5 groups, and precisely the fixture
    `stats_tree`'s group-comparison API exists to consume). Normalising here
    means the tree-building code never has to know which representation it
    got (Design Decision #52).
    """
    if hasattr(obj, "_fieldnames"):  # scipy.io.matlab.mat_struct
        return {name: getattr(obj, name) for name in obj._fieldnames}
    return obj


def looks_like_tree(obj) -> bool:
    """Whether a decoded MATLAB value plausibly holds a TREES tree struct.

    Checks for the two fields every tree must have and that
    ``_struct_to_tree`` already requires: the adjacency matrix and at least
    one coordinate array.
    """
    obj = to_plain(obj)
    return isinstance(obj, dict) and "dA" in obj and "X" in obj


def contains_tree(obj, depth: int = 0) -> bool:
    """Whether ``obj`` is a tree struct, or a (nested) collection of them.

    Used to pick the right variable out of a `.mtr` that doesn't happen to
    call it ``tree``. Recursion is depth-limited: `load_tree.m` documents at
    most 2 levels of cell nesting, and an unbounded search over a large
    workspace would be slow for no benefit.
    """
    obj = to_plain(obj)
    if looks_like_tree(obj):
        return True
    if depth >= 3:
        return False
    if isinstance(obj, (list, tuple, np.ndarray)):
        return any(contains_tree(item, depth + 1) for item in obj)
    return False
