"""Bundled sample morphologies, mirroring MATLAB's ``sample/mtr/*_tree.m``.

Four reconstructions ship with the toolbox, each a one-line loader in MATLAB
and here:

===================  =============  ======  ===================================
function             file           nodes   what it is
===================  =============  ======  ===================================
:func:`sample_tree`  sample.mtr        197  subtree of an HSN cell
:func:`sample2_tree` sample2.mtr        15  a minimal tree, good for doctests
:func:`hsn_tree`     hsn.mtr          1290  full HSN cell
:func:`hss_tree`     hss.mtr          2252  full HSS cell
===================  =============  ======  ===================================

plus :func:`dLPTCs_trees`, a 55-reconstruction population in 5 named groups.

A note on history, because it changed what ``sample_tree()`` returns
------------------------------------------------------------------
Until Design Decision #51, ``sample_tree()`` returned a **different cell**
from MATLAB's: the 2252-node ``25HSS.swc``. That was a Phase-1 stand-in --
``.mtr`` reading was still deferred (Design Decision #9) and
``sample/swc/`` holds exactly one file, so it became the fixture and kept
the name. ``.mtr`` support landed later (#32) and the sample was never
revisited.

The substitution cost more than a node count. ``25HSS.swc`` is the HSS cell
(same 2252 nodes, same 8100.26 um total length as ``hss.mtr``), but the SWC
export is **X-mirrored** relative to it, and SWC cannot carry region names,
so all three of ``axon``/``dend``/``soma`` collapsed into a single region
``'1'``. Region handling is exactly what ``dissect_tree``, ``stats_tree``
and the NEURON bridge exercise, so the default sample exercised none of it.

Nothing was lost in fixing this: that tree is now :func:`hss_tree`, in its
properly regioned ``.mtr`` form.
"""

from __future__ import annotations

from importlib import resources

from .core import Tree
from .io.mtr import load_mtr


def _load_bundled(filename: str, name: str) -> Tree | list[Tree]:
    """Load one of the bundled ``.mtr`` files from the package data dir."""
    with resources.as_file(resources.files("pytrees") / "data" / filename) as path:
        trees = load_mtr(path)
    if isinstance(trees, Tree):
        trees.name = name
    return trees


def sample_tree() -> Tree:
    """A subtree of a sample HSN cell (197 nodes) -- MATLAB's ``sample_tree``.

    The toolbox's default example morphology: small enough to plot, print
    and step through, but a real reconstruction with real branch structure
    and two regions (``dend``, ``soma``).

    Returns
    -------
    Tree
    """
    return _load_bundled("sample.mtr", "sample")


def sample2_tree() -> Tree:
    """A minimal 15-node sample tree -- MATLAB's ``sample2_tree``.

    Small enough that you can read its full node table at a glance, which
    makes it the right fixture for doctests and for reasoning about an
    algorithm by hand.

    Returns
    -------
    Tree
    """
    return _load_bundled("sample2.mtr", "sample2")


def hsn_tree() -> Tree:
    """A full HSN cell (1290 nodes) -- MATLAB's ``hsn_tree``.

    Returns
    -------
    Tree
    """
    return _load_bundled("hsn.mtr", "hsn")


def hss_tree() -> Tree:
    """A full HSS cell (2252 nodes) -- MATLAB's ``hss_tree``.

    This is the tree ``sample_tree()`` used to return, before Design
    Decision #51 restored MATLAB's meaning of that name -- but loaded from
    ``.mtr``, so unlike the old SWC version it carries its real
    ``axon``/``dend``/``soma`` regions and its original orientation.

    Returns
    -------
    Tree
    """
    return _load_bundled("hss.mtr", "hss")


def dLPTCs_trees() -> dict[str, list[Tree]]:
    """A population of 55 dipteran lobula-plate tangential cells, in 5 groups.

    The fixture MATLAB's ``stats_tree`` examples use, and what this port's
    group-comparison API was built for::

        groups = dLPTCs_trees()
        stats = stats_tree(list(groups.values()), group_names=list(groups))

    Returns
    -------
    dict[str, list[Tree]]
        Group name -> trees. Group names follow the cell classes in the
        original archive (``HSE``, ``HSN``, ...); the archive itself stores
        them positionally, so they are recovered from each tree's own
        ``name`` field where possible and numbered otherwise.

    Notes
    -----
    This file could not be read at all before Design Decision #52: at its
    nesting depth ``scipy.io.loadmat`` returns ``mat_struct`` objects even
    with ``simplify_cells=True``, which the ``.mtr`` flattener did not
    recognise.
    """
    from ._matlab_groups import load_grouped_mtr

    with resources.as_file(resources.files("pytrees") / "data" / "dLPTCs.mtr") as path:
        return load_grouped_mtr(path)
