"""Reading MATLAB `.mat`/`.mtr` files of either vintage, as one call.

MATLAB has two on-disk formats and the split matters here more than it
usually does: **`save_tree.m` writes `'-v7.3'` unconditionally**, so every
`.mtr` produced by a current TREES install is HDF5-based and invisible to
`scipy.io.loadmat`, which reads only up to v7. That is not an edge case,
it is the default path, and it is the crux of MATLAB<->Python
interoperability for this toolbox.

Which library
-------------
Measured against the three real v7.3 files in `Active GC Model/morphos/`,
not chosen from documentation:

=================  =======================  ================================
library            flat ``{t1..t8}`` cell   nested ``{{t1..t15}}`` cell
=================  =======================  ================================
`scipy.io`         raises                   raises
`hdf5storage`      raises                   raises
`pymatreader`      works                    **returns raw h5py References**
`mat73`            works                    works
=================  =======================  ================================

`pymatreader` is the tempting choice -- it returns exactly the shape
`scipy.io.loadmat(simplify_cells=True)` does -- but it fails to dereference
the *nested* cell arrays that `load_tree.m` documents as the 2-level
`cgui_tree` layout, handing back bare `h5py.Reference` objects instead. Two
of the three bundled files use that layout. `mat73` handles both, so it is
what this module uses (Design Decision #47).

Worth knowing: **Octave cannot read these files either** (`load` warns
"can't read 'tree' (unknown datatype)" and yields nothing), so there is no
re-save-as-v5 escape hatch outside MATLAB itself.

`mat73` is an optional dependency -- `pip install pytrees[matlab]` -- and is
imported lazily, so the core package stays dependency-light and only pays
for HDF5 when a v7.3 file is actually opened.
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

    Dispatches on the file's actual format -- read from its header, not
    guessed from its extension -- using `scipy.io` for v5/v7 and `mat73` for
    v7.3/HDF5.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    dict
        Variable name -> value, with structs as dicts and cell arrays as
        (possibly nested) lists -- the shape
        `scipy.io.loadmat(simplify_cells=True)` produces. `mat73` adds an
        extra level of list nesting in places; callers should flatten
        recursively rather than index at a fixed depth (`mtr._flatten` does).

    Raises
    ------
    ValueError
        If the file is v7.3 and `mat73` is not installed -- naming the
        install command, since scipy's own error says nothing about how to
        proceed.
    """
    path = Path(path)

    if not is_v73(path):
        from scipy.io import loadmat

        # pass a str, not the Path: scipy's mat reader gives a clean
        # FileNotFoundError for a missing str path but an opaque OSError
        # ("needs file name or open file-like object") for a missing Path
        return loadmat(str(path), simplify_cells=True)

    try:
        import mat73
    except ImportError:
        raise ValueError(
            f"{path} is a MATLAB v7.3 (HDF5) file, which scipy cannot read. "
            f"Install the optional dependency: pip install 'pytrees[matlab]'"
        ) from None

    # mat73 logs a warning for any extension that isn't '.mat'; '.mtr' is a
    # perfectly ordinary MATLAB workspace, so the warning is pure noise here
    import logging

    mat73_log = logging.getLogger()
    previous = mat73_log.level
    mat73_log.setLevel(logging.ERROR)
    try:
        return mat73.loadmat(str(path))
    finally:
        mat73_log.setLevel(previous)


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
