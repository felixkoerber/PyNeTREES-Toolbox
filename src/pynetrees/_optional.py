"""Which optional dependencies are present, and what is unavailable without them.

Most of the package needs only numpy/scipy/pandas. Plotting, Blender export,
NEURON simulation, MATLAB v7.3 files and image stacks each need something
extra, and each is imported lazily at the point of use so that
``import pynetrees`` stays fast and cheap.

The cost of that laziness is that a missing dependency announces itself as an
``ImportError`` deep inside the first call that needed it -- often minutes
into a script, and named after the transitive dependency rather than the
feature. :func:`missing_dependencies` answers the question up front instead:
*what can this installation not do?*

The check is a **warning, not an error**, and is emitted once at import. Set
``PYNETREES_NO_DEPENDENCY_WARNING=1`` to silence it -- worth doing in CI for
an environment that is deliberately minimal.
"""

from __future__ import annotations

import importlib.util
import os
import warnings

__all__ = ["OPTIONAL_FEATURES", "missing_dependencies", "warn_about_missing"]

#: ``feature -> (extra, modules, what stops working)``. The extra is the
#: ``pip install pynetrees[...]`` name, so the warning can tell the reader
#: what to type rather than making them find it.
OPTIONAL_FEATURES: dict[str, tuple[str, tuple[str, ...], str]] = {
    "plotting": (
        "plot", ("pyvista", "matplotlib"),
        "pynetrees.plotting, hull_tree's 3D isosurface, the tutorials",
    ),
    "isosurfaces and stack skeletonisation": (
        "plot", ("skimage",),
        "hull_tree(dim=3), stacks.skeletonize_stack",
    ),
    "MATLAB v7.3 files": (
        "matlab", ("mat73",),
        "reading .mtr/.mat written by a current MATLAB (v7.3 is its default)",
    ),
    "HDF5 .nmf files": (
        "nmf", ("h5py",),
        "load_nmf, save_nmf",
    ),
    "image stacks": (
        "stacks", ("tifffile", "imageio"),
        "pynetrees.stacks",
    ),
    "NEURON simulation": (
        "neuron", ("neuron",),
        "build_neuron_model, GC_init, run_current_clamp, insert_mechanism",
    ),
    "Blender export": (
        "blender", ("bpy",),
        "pynetrees.blender (not imported by `import pynetrees` in any case)",
    ),
}


def _installed(module: str) -> bool:
    """Is this importable, without importing it?

    ``find_spec`` rather than ``import``: importing ``bpy`` alone costs
    hundreds of milliseconds and starts a Blender session, which is exactly
    the cost this module exists to avoid paying on every ``import
    pynetrees``.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # a package whose parent is itself missing, or a broken entry on the
        # path -- either way it is not usable
        return False


def missing_dependencies() -> dict[str, tuple[str, list[str], str]]:
    """What this installation cannot do, and what would fix it.

    Returns
    -------
    dict
        ``{feature: (extra, missing_modules, what_stops_working)}`` for every
        feature with at least one missing module. Empty on a full install.

    Examples
    --------
    >>> for feature, (extra, modules, effect) in missing_dependencies().items():
    ...     print(f"{feature}: pip install pynetrees[{extra}]  ({effect})")
    """
    out = {}
    for feature, (extra, modules, effect) in OPTIONAL_FEATURES.items():
        absent = [m for m in modules if not _installed(m)]
        if absent:
            out[feature] = (extra, absent, effect)
    return out


def warn_about_missing() -> None:
    """Emit one warning naming everything unavailable, or stay silent.

    One warning listing all of it, not one per feature: a minimal install is
    missing several at once, and a stack of near-identical warnings is read
    as noise and filtered out wholesale -- including the line that mattered.
    """
    if os.environ.get("PYNETREES_NO_DEPENDENCY_WARNING"):
        return

    missing = missing_dependencies()
    if not missing:
        return

    lines = [
        f"pynetrees: {len(missing)} optional feature(s) unavailable in this "
        f"environment."
    ]
    extras = sorted({extra for extra, _, _ in missing.values()})
    for feature, (extra, modules, effect) in missing.items():
        lines.append(
            f"  - {feature}: no {', '.join(modules)} "
            f"-> {effect} will not work (pip install 'pynetrees[{extra}]')"
        )
    lines.append(
        f"  Install everything at once with: pip install "
        f"'pynetrees[{','.join(extras)}]'"
    )
    lines.append(
        "  Silence this with PYNETREES_NO_DEPENDENCY_WARNING=1."
    )
    # UserWarning, not ImportWarning: Python's default filters *ignore*
    # ImportWarning, so the one category that reads as right for this
    # would make it invisible to exactly the people it is for.
    warnings.warn("\n".join(lines), UserWarning, stacklevel=3)
