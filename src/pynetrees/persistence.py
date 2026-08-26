"""Topological description of a morphology: barcodes and persistence images.

Ported from ``treestoolbox-master/graphtheory/``: ``barcode_tree``,
``persistenceimage_tree`` and ``realisations_tree``, all built on
:func:`pynetrees.BLO_tree`. Interpreted from Kanari et al. (2018), the
topological morphology descriptor.

The idea, briefly. :func:`pynetrees.BLO_tree` decomposes the arbor into
paths, longest first. Each path becomes one **bar** spanning the distances
from the root at which it starts and ends -- born where it branches off its
parent, dying at its tip. The resulting **barcode** is a description of the
cell's branching structure that does not depend on how the reconstruction
was sampled, rotated or embedded, which is what makes it useful for
comparing cells to each other.

A barcode is a variable-length list of bars, so it cannot be fed to a
clustering algorithm directly. :func:`persistenceimage_tree` renders it as
a fixed-size 2D density -- smear each bar into a Gaussian blob at
``(birth, death)`` and add them up -- which can be.

Where this port departs from MATLAB, and why, is recorded in each
function's Notes and in ``MATLAB_TOOLBOX_BUGS.md``.
"""

from __future__ import annotations

import numpy as np

from ._empty import empty_safe
from ._population import accepts_population
from .core import Tree
from .graphtheory import BLO_tree, Pvec_tree
from .metrics import eucl_tree, len_tree

__all__ = ["barcode_tree", "persistenceimage_tree", "realisations_tree"]


def _empty_barcode(tree):
    """No branches, so no bars -- but keep the two columns, so an empty
    cell still stacks with real ones."""
    return np.empty((0, 2))


#: Where MATLAB cuts its Gaussian off, in units of sigma. Its kernel is a
#: fixed 101x101 grid spanning [-1, 1] with sigma 0.35, i.e. 50 pixels at
#: sigma 17.5 -- so it truncates at 50/17.5 sigma and renormalises what is
#: left to sum 1. Kept exactly, so the port can be checked against MATLAB's
#: own output rather than merely resemble it.
_TRUNCATE = 50.0 / 17.5


def _smooth(image: np.ndarray, sigma: float) -> np.ndarray:
    """Blur with MATLAB's truncated, renormalised Gaussian.

    Applied as two 1D passes. The 2D kernel is the outer product of the 1D
    one, and normalising the square is the same as normalising each factor,
    so this is exact -- and ``O(n^2 r)`` rather than ``O(n^2 r^2)``, which
    is the difference between a moment and a minute on a large cell.

    Zero padding, not reflection: the image is a density on a bounded
    plane, and reflecting would pile mass from past the furthest tip back
    onto the tips.
    """
    from scipy import ndimage

    radius = int(round(_TRUNCATE * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    taps = np.exp(-(offsets ** 2) / (2.0 * sigma ** 2))
    taps /= taps.sum()

    for axis in (0, 1):
        image = ndimage.convolve1d(image, taps, axis=axis, mode="constant",
                                   cval=0.0)
    return image


def _values_for(tree: Tree, v, mode: str) -> np.ndarray:
    """The per-node quantity the barcode accumulates along each path."""
    if v is not None:
        return np.asarray(v, dtype=float)
    if mode == "length":
        return len_tree(tree)
    if mode == "euclidean":
        return eucl_tree(tree)
    if mode == "topological":
        return np.ones(tree.n_nodes)
    raise ValueError(
        f"mode must be 'length', 'euclidean' or 'topological', got {mode!r}"
    )


@accepts_population(paired="v")
@empty_safe(_empty_barcode)
def barcode_tree(tree: Tree, v: np.ndarray | None = None, *,
                 mode: str = "length", by: str = "nodes") -> np.ndarray:
    """Persistent-homology barcode: one bar per branch.

    Parameters
    ----------
    tree : Tree
    v : array_like, optional
        Per-node values to accumulate along each path. Overrides ``mode``.
    mode : {'length', 'euclidean', 'topological'}, keyword-only
        What to accumulate when ``v`` is not given: segment length in
        microns (MATLAB's ``'-l'``), Euclidean distance to the root
        (``'-E'``), or one per node, making the bars count nodes rather
        than measure distance (``'-t'``).
    by : {'nodes', 'length'}, keyword-only, default 'nodes'
        Passed to :func:`pynetrees.BLO_tree`, which decides which paths
        become branches. The default reproduces MATLAB; read that
        function's Notes before changing it, because the two disagree
        substantially.

    Returns
    -------
    np.ndarray
        ``(n_branches, 2)``, columns ``[birth, death]``: the distances from
        the root at which each branch starts and ends. Rows follow
        :func:`pynetrees.BLO_tree`'s order, so row 0 is the branch containing
        the root and its birth is 0.

    Notes
    -----
    Bars nest rather than overlap: a branch is born on its parent branch,
    so its birth always falls inside its parent's ``[birth, death]``. That
    is what :func:`realisations_tree` counts.
    """
    values = _values_for(tree, v, mode)
    order, _, cumulative = BLO_tree(tree, values, by=by)
    path_length = Pvec_tree(tree, values)

    n_branches = int(order.max())
    death = np.zeros(n_branches)
    reach = np.zeros(n_branches)
    np.maximum.at(death, order - 1, path_length)
    np.maximum.at(reach, order - 1, cumulative)
    return np.column_stack([death - reach, death])


def realisations_tree(tree: Tree | np.ndarray, v: np.ndarray | None = None, *,
                      mode: str = "length") -> int:
    """How many distinct trees share this tree's barcode.

    The barcode says which branches exist and where each begins and ends,
    but not *which* branch each one hangs off: a bar born at distance 40
    could have branched off any bar alive at 40. Multiplying those choices
    together counts the trees the barcode cannot tell apart -- a measure of
    how much shape information the description throws away.

    Parameters
    ----------
    tree : Tree or array_like
        A tree, or an ``(n_branches, 2)`` barcode from
        :func:`barcode_tree`, so a barcode computed once can be reused.
    v, mode
        As :func:`barcode_tree`, ignored when a barcode is passed.

    Returns
    -------
    int
        Exact, however large. For a real cell this number is astronomical
        -- a 1290-node cell runs to hundreds of digits -- which is the
        point being made. MATLAB computes it in double precision and
        returns ``Inf`` for anything past ~1e308; Python's integers do not
        overflow, so the value is usable (its logarithm, in practice).

    Notes
    -----
    Returns 0 if any non-root bar is born outside every other bar, which
    cannot happen for a barcode that came from a tree.
    """
    bars = (barcode_tree(tree, v, mode=mode) if isinstance(tree, Tree)
            else np.asarray(tree, dtype=float))
    if bars.ndim != 2 or bars.shape[1] != 2:
        raise ValueError(
            f"expected a tree or an (n_branches, 2) barcode, got array of "
            f"shape {bars.shape}"
        )

    # Sorted by birth, so walking from the last bar backwards asks the
    # question in the order the branches came into existence.
    bars = bars[np.lexsort((bars[:, 1], bars[:, 0]))]
    births, deaths = bars[:, 0], bars[:, 1]

    realisations = 1
    for index in range(len(bars) - 1, 0, -1):
        alive = (births <= births[index]) & (births[index] <= deaths)
        alive[index] = False  # a branch cannot hang off itself
        realisations *= int(alive.sum())
    return realisations


@accepts_population(paired="v")
def persistenceimage_tree(tree: Tree, v: np.ndarray | None = None, *,
                          mode: str = "length", sigma: float = 17.5,
                          size: int | None = None,
                          accumulate: bool = True) -> np.ndarray:
    """Render a barcode as a fixed-size 2D density -- the persistence image.

    Every cell gives an image on the same axes, so two cells can be
    compared, averaged or clustered pixel by pixel, which a barcode of
    unequal length does not allow.

    Parameters
    ----------
    tree : Tree
    v, mode
        As :func:`barcode_tree`.
    sigma : float, keyword-only, default 17.5
        Width of the Gaussian each bar is smeared into, in microns.
        MATLAB's fixed kernel works out to exactly this. Note it is
        **absolute**, not a fraction of the cell: a small cell is smoothed
        proportionally more than a large one.
    size : int, optional
        Side of the square image in pixels, one micron each. Defaults to
        ``round(1.25 * max(death))``, MATLAB's rule -- 25% of headroom past
        the furthest tip.
    accumulate : bool, keyword-only, default True
        Add up bars that land on the same pixel. ``False`` reproduces
        MATLAB, which marks occupied pixels with a 1 and so counts
        coincident branches once. See the Notes.

    Returns
    -------
    np.ndarray
        ``(size, size)``, indexed ``[birth, death]``. Only the **upper**
        triangle can be occupied, since a branch cannot die before it is
        born.

    Notes
    -----
    **This differs from MATLAB in two places, both deliberate.**

    MATLAB assigns ``M(...) = 1`` rather than accumulating, so branches
    whose births and deaths round to the same micron contribute once
    between them. Across the 55 cells in :func:`pynetrees.dLPTCs_trees` that
    silently drops a median of **1.5% of bars, at worst 4.0%** -- small,
    but it falls hardest on the densely branched cells, which are the ones
    a density is supposed to distinguish. The published method sums a
    kernel per bar, so this accumulates by default; pass
    ``accumulate=False`` to reproduce MATLAB's figures exactly.

    MATLAB also offsets the two axes inconsistently -- ``round(birth) + 1``
    against ``round(death)`` -- which shifts the image one pixel off the
    diagonal. Here both axes share an origin. Against a 17.5 um kernel one
    pixel is invisible, and it cancels between cells anyway, so this
    changes no clustering; it only matters if you read coordinates off the
    image.
    """
    bars = barcode_tree(tree, v, mode=mode)
    if len(bars) == 0:
        return np.zeros((0, 0))

    birth, death = bars[:, 0], bars[:, 1]
    extent = int(round(1.25 * death.max())) if size is None else int(size)
    if extent < 1:
        raise ValueError(
            f"the cell spans {death.max():.3g} um, too little to render on a "
            f"1 um grid; pass an explicit `size`"
        )

    rows = np.clip(np.round(birth).astype(int), 0, extent - 1)
    cols = np.clip(np.round(death).astype(int), 0, extent - 1)
    image = np.zeros((extent, extent))
    if accumulate:
        np.add.at(image, (rows, cols), 1.0)
    else:
        image[rows, cols] = 1.0

    return _smooth(image, sigma)
