"""Synthesising new trees from measured ones (B4).

Ports `construct/gscale_tree.m`, `construct/clone_tree.m`,
`construct/rpoints_tree.m`, `construct/in_c.m`,
`construct/dscam_tree.m`, `construct/spines_tree.m` and
`construct/PP_generator_tree.m`.

The pipeline, in the order it runs:

1. :func:`gscale_tree` measures a *group* of real cells -- region by
   region, how far each spans, where its mass sits, how many branch and
   termination points it carries, how it tapers, how much its cable
   wriggles -- and rescales them all to a common size, so the pooled point
   clouds are comparable.
2. :func:`rpoints_tree` draws new target points from that pooled density.
3. :func:`~pynetrees.MST_tree` wires them up.
4. :func:`clone_tree` runs 1-3 per region and reassembles the result, then
   restores taper, wriggle and soma.

The premise, stated in `gscale_tree`'s own docstring, is that the *density
of topological points* is roughly scale-invariant within a cell type. Whether
that holds is the user's call; this only implements it.

Two of MATLAB's helpers are deliberately not ported, because they exist only
to unpack MATLAB's packed ``contourc`` matrix -- a format this port never
produces. :func:`~pynetrees.hull_tree` returns 2D boundaries as a plain list
of polygons, so ``cpoints (c)`` is ``np.vstack(polygons)`` and
``cplotter (c)`` is a loop over ``ax.plot``. `in_c` does have real content
beyond unpacking -- outer ring versus holes -- and is ported as
:func:`in_hull`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .construct import (
    MST_NEW_REGION,
    MST_tree,
    jitter_tree,
    quaddiameter_tree,
    smooth_tree,
    soma_tree,
)
from ._population import accepts_population
from ._empty import empty_safe
from .core import Tree
from .density import DensityGrid, gdens_tree
from .edit import cat_tree, delete_tree, elim0_tree, resample_tree, root_tree
from .graphtheory import B_tree, C_tree, T_tree, sub_tree
from .metrics import direction_tree, len_tree, tran_tree

__all__ = [
    "RegionSpan",
    "Spanning",
    "gscale_tree",
    "rpoints_tree",
    "in_hull",
    "clone_tree",
    "dscam_tree",
    "SpineResult",
    "spines_tree",
    "PP_generator_tree",
]


# ---------------------------------------------------------------------------
# gscale_tree
# ---------------------------------------------------------------------------


@dataclass
class RegionSpan:
    """What one region looks like across a group of cells.

    Every per-tree array has one row per input tree, with ``NaN`` where the
    tree has no nodes in this region -- rather than silently shortening, so
    row ``i`` always refers to tree ``i``.
    """

    name: str
    extent: np.ndarray
    """``(n_trees, 3)`` bounding-box width per axis [um]."""
    centre: np.ndarray
    """``(n_trees, 3)`` centre of mass of the region's branch and
    termination points [um]. Continuation points are excluded: they are an
    artefact of how finely the cell was traced, and including them would
    weight densely-sampled stretches more heavily."""
    nodes: list[np.ndarray]
    """Per tree, the indices of this region's nodes."""
    n_points: np.ndarray
    """``(n_trees,)`` branch + termination points in this region."""
    points: list[np.ndarray]
    """Per tree, this region's branch and termination points rescaled to
    the group's mean extent -- the pooled cloud :func:`clone_tree` samples."""
    taper: np.ndarray
    """``(n_present, 2)`` per-tree ``(scale, offset)`` fit relating the
    measured diameters to what :func:`~pynetrees.quaddiameter_tree` predicts.
    Only rows for trees that *have* this region, so it is generally shorter
    than the arrays above -- MATLAB's layout, kept because the values are
    pooled rather than indexed by tree."""

    @property
    def mean_extent(self) -> np.ndarray:
        return _nanmean_or(self.extent, 1.0)

    @property
    def std_extent(self) -> np.ndarray:
        return _nanstd(self.extent)

    @property
    def mean_n(self) -> float:
        return float(np.nanmean(self.n_points))

    @property
    def std_n(self) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(np.nanstd(self.n_points, ddof=1))

    def pooled_points(self) -> np.ndarray:
        """Every tree's rescaled points, stacked."""
        present = [p for p in self.points if len(p)]
        return np.vstack(present) if present else np.empty((0, 3))


@dataclass
class Spanning:
    """Output of :func:`gscale_tree`: the group's measured envelope."""

    regions: list[RegionSpan]
    wriggle: np.ndarray
    """``(n_trees, 2)`` of ``(amplitude, wavelength)``. Amplitude is
    ``2 * (cable length / straight-line length - 1)``, i.e. how much longer
    the traced path is than the branch points alone would require;
    wavelength is fixed at 5 um, as in MATLAB."""
    scaled_trees: list[Tree] = field(default_factory=list)
    """The input trees rescaled region-wise to the group's mean extent."""

    def __getitem__(self, name: str) -> RegionSpan:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(
            f"no region {name!r}; this group has {[r.name for r in self.regions]}"
        )

    def __contains__(self, name: str) -> bool:
        return any(r.name == name for r in self.regions)

    @property
    def names(self) -> list[str]:
        return [r.name for r in self.regions]


def _nanmean_or(values: np.ndarray, fallback: float) -> np.ndarray:
    """Column means, with zeros replaced by ``fallback``.

    MATLAB substitutes 1 for a mean extent of 0, which happens on a planar
    group's z axis. Without it the scaling below divides by zero.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(values, axis=0)
    return np.where(mean == 0, fallback, mean)


def _nanstd(values: np.ndarray) -> np.ndarray:
    """Column standard deviations, MATLAB's ``std`` (``ddof=1``)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(values, axis=0, ddof=1)


def gscale_tree(trees: Tree | list[Tree]) -> Spanning:
    """Measure a group of cells region by region, and rescale them to a
    common size.

    Parameters
    ----------
    trees : Tree or list[Tree]

    Returns
    -------
    Spanning
        Per-region extents, centres of mass, point counts, taper fits and
        rescaled point clouds, plus the group's wriggle statistics and the
        rescaled trees themselves.

    Notes
    -----
    Each tree is translated to put its root at the origin first, so extents
    and centres are measured relative to the soma rather than to whatever
    coordinate frame the reconstruction happened to use.

    **The rescaled point clouds and the rescaled trees do not agree**, and
    that is MATLAB's behaviour, preserved deliberately. A region's *points*
    are scaled about that region's own centre of mass, so the centre stays
    put; the *trees* are scaled about the origin, i.e. the root. The point
    cloud is what `clone_tree` samples, and holding each region's centre
    fixed is what keeps the pooled cloud from smearing; the scaled trees are
    for display and comparison. Do not expect
    ``spanning['dendrite'].points[i]`` to be a subset of
    ``spanning.scaled_trees[i]``.

    MATLAB returns a ``spanning`` struct of fifteen parallel cell arrays
    indexed by region and then by tree. This returns a list of
    :class:`RegionSpan` objects, looked up by name
    (``spanning['dendrite']``), because the parallel-array layout makes
    every access a two-level index into containers that must be kept in
    step by hand.
    """
    if isinstance(trees, Tree):
        trees = [trees]
    trees = [tran_tree(t) for t in trees]

    names = sorted({name for tree in trees for name in tree.rnames})
    regions: list[RegionSpan] = []

    quads = [quaddiameter_tree(tree) for tree in trees]
    scaled = [_copy_tree(tree) for tree in trees]

    for name in names:
        extent = np.full((len(trees), 3), np.nan)
        centre = np.full((len(trees), 3), np.nan)
        counts = np.full(len(trees), np.nan)
        per_tree_nodes: list[np.ndarray] = []
        raw_points: list[np.ndarray] = []

        for index, tree in enumerate(trees):
            nodes = _region_nodes(tree, name)
            per_tree_nodes.append(nodes)
            if len(nodes) == 0:
                raw_points.append(np.empty((0, 3)))
                continue
            coords = np.column_stack([tree.X, tree.Y, tree.Z])
            extent[index] = coords[nodes].max(axis=0) - coords[nodes].min(axis=0)
            # branch and termination points only -- see `centre`'s docstring
            bt = nodes[C_tree(tree)[nodes] == 0]
            counts[index] = len(bt)
            if len(bt):
                centre[index] = coords[bt].mean(axis=0)
            raw_points.append(coords[bt])

        if np.all(np.isnan(counts)):
            continue  # region named by some tree's rnames but used by none

        region = RegionSpan(
            name=name, extent=extent, centre=centre, nodes=per_tree_nodes,
            n_points=counts, points=[], taper=np.empty((0, 2)),
        )
        mean_extent = region.mean_extent
        region.points = [
            _rescale_points(raw_points[i], centre[i], extent[i], mean_extent)
            for i in range(len(trees))
        ]
        region.taper = _region_taper(trees, quads, per_tree_nodes)
        _rescale_in_place(scaled, per_tree_nodes, extent, mean_extent)
        regions.append(region)

    return Spanning(regions=regions, wriggle=_wriggle(trees), scaled_trees=scaled)


def _region_nodes(tree: Tree, name: str) -> np.ndarray:
    """Indices of the nodes labelled ``name`` in ``tree``."""
    if name not in tree.rnames:
        return np.empty(0, dtype=int)
    return np.flatnonzero(np.asarray(tree.R) == tree.rnames.index(name))


def _rescale_points(points, centre, extent, mean_extent) -> np.ndarray:
    """Scale a region's points to the group's mean extent, centre fixed."""
    if len(points) == 0:
        return points
    out = points.copy()
    for axis in range(3):
        if extent[axis] != 0:
            out[:, axis] = (
                centre[axis]
                + mean_extent[axis] * (points[:, axis] - centre[axis]) / extent[axis]
            )
    return out


def _rescale_in_place(scaled, per_tree_nodes, extent, mean_extent) -> None:
    """Scale each tree's copy of this region about the root."""
    for index, nodes in enumerate(per_tree_nodes):
        if len(nodes) == 0:
            continue
        tree = scaled[index]
        for axis, values in enumerate((tree.X, tree.Y, tree.Z)):
            if extent[index, axis] != 0:
                values[nodes] *= mean_extent[axis] / extent[index, axis]


def _region_taper(trees, quads, per_tree_nodes) -> np.ndarray:
    """Per tree, how the measured diameters relate to the quadratic fit.

    :func:`~pynetrees.quaddiameter_tree` predicts a diameter profile from the
    topology alone; comparing its range against the real one gives the
    ``(scale, offset)`` pair needed to reproduce this cell type's taper on a
    synthetic tree.
    """
    rows = []
    for tree, quad, nodes in zip(trees, quads, per_tree_nodes):
        if len(nodes) == 0:
            continue
        lo, hi = tree.D[nodes].min(), tree.D[nodes].max()
        qlo, qhi = quad.D[nodes].min(), quad.D[nodes].max()
        scale = (hi - lo) / (qhi - qlo) if qhi != qlo else 0.0
        offset = lo / qlo if qlo != 0 else 0.0
        rows.append((0.5 * scale, 0.5 * offset))
    return np.array(rows) if rows else np.empty((0, 2))


def _wriggle(trees) -> np.ndarray:
    """How much longer each tree's cable is than its skeleton requires."""
    out = np.zeros((len(trees), 2))
    for index, tree in enumerate(trees):
        if tree.n_nodes == 0:
            continue  # an empty tree contributes no cable to wriggle
        total = len_tree(tree).sum()
        straight = len_tree(
            delete_tree(tree, np.flatnonzero(C_tree(tree)))
        ).sum()
        out[index] = (2 * (total / straight - 1) if straight > 0 else 0.0, 5.0)
    return out


def _copy_tree(tree: Tree) -> Tree:
    return Tree(
        dA=tree.dA.copy(), X=tree.X.copy(), Y=tree.Y.copy(), Z=tree.Z.copy(),
        D=tree.D.copy(), R=np.asarray(tree.R).copy(), rnames=list(tree.rnames),
        name=tree.name, frustum=tree.frustum,
        Ri=tree.Ri, Gm=tree.Gm, Cm=tree.Cm,
    )


# ---------------------------------------------------------------------------
# rpoints_tree
# ---------------------------------------------------------------------------


def rpoints_tree(density=None, n: int = 1000, *, x=None, y=None, z=None,
                 boundary=None, thr: float = 0.0, rng=None) -> np.ndarray:
    """Draw ``n`` random points from a density grid.

    Each point picks a voxel with probability proportional to its count,
    then lands uniformly inside that voxel -- so the result reproduces the
    density at grid resolution while staying continuous within it.

    Parameters
    ----------
    density : DensityGrid or ndarray, optional
        Usually a :func:`~pynetrees.gdens_tree` result. A bare array needs
        ``x``/``y``/``z`` to say where its voxels are. Omit it entirely to
        scatter points uniformly through the box given by ``x``/``y``
        (default ``[-500, 500]`` on both, as in MATLAB).
    n : int, default 1000
    x, y, z : array_like, optional
        Voxel-centre coordinates per axis. Ignored when ``density`` is a
        :class:`~pynetrees.density.DensityGrid`, which carries its own.
    boundary : list of (m, 2) arrays, optional
        Keep only points inside this 2D boundary -- a
        :func:`~pynetrees.hull_tree` polygon list. MATLAB takes its packed
        ``contourc`` matrix here instead; see this module's docstring.
    thr : float, default 0.0
        With ``boundary``, also drop points within ``thr`` [um] of it.
    rng : numpy Generator or int, optional

    Returns
    -------
    np.ndarray
        ``(n, 3)``, or fewer rows when ``boundary`` rejects some. **The
        count is not guaranteed** when filtering -- MATLAB has the same
        behaviour, and `clone_tree` compensates by asking for four times
        what it needs.

    Notes
    -----
    MATLAB picks each point with a Python-level loop over the cumulative
    density, calling ``ind2sub`` once per point and showing a waitbar every
    5000 -- which is what makes the waitbar worth having. The same thing is
    one ``searchsorted`` over the whole batch here, so `clone_tree`'s
    repeated calls stop dominating its runtime.
    """
    generator = np.random.default_rng(rng)

    if density is None:
        x = np.asarray([-500.0, 500.0] if x is None else x, dtype=float)
        y = np.asarray(x if y is None else y, dtype=float)
        if not np.isclose(np.ptp(x), np.ptp(y)):
            warnings.warn(
                "x and y spans differ, so points will not be isotropically "
                "distributed", stacklevel=2,
            )
        points = np.column_stack([
            generator.uniform(x.min(), x.max(), n),
            generator.uniform(y.min(), y.max(), n),
            np.zeros(n),
        ])
    else:
        points = _sample_density(density, n, x, y, z, generator)

    if boundary is not None:
        inside = in_hull(points[:, :2], boundary)
        if thr > 0:
            edge = np.vstack([np.asarray(p) for p in boundary])
            far = (
                np.linalg.norm(points[:, None, :2] - edge[None, :, :], axis=2).min(1)
                > thr
            )
            inside &= far
        points = points[inside]
    return points


def _sample_density(density, n, x, y, z, generator) -> np.ndarray:
    """Voxel-weighted sampling, vectorised over the whole batch."""
    if isinstance(density, DensityGrid):
        counts, axes = density.counts, (density.x, density.y, density.z)
    else:
        counts = np.asarray(density, dtype=float)
        if counts.ndim == 2:
            counts = counts[:, :, None]
        supplied = (x, y, z)
        axes = tuple(
            np.arange(counts.shape[i]) if supplied[i] is None
            else np.asarray(supplied[i], dtype=float)
            for i in range(3)
        )

    total = counts.sum()
    if total <= 0:
        raise ValueError("density grid is empty -- nothing to sample from")

    flat = np.cumsum(counts.ravel())
    picks = np.searchsorted(flat, generator.random(n) * total, side="right")
    voxel = np.unravel_index(np.minimum(picks, flat.size - 1), counts.shape)

    out = np.empty((n, 3))
    for axis in range(3):
        centres = axes[axis]
        width = centres[1] - centres[0] if len(centres) > 1 else 0.0
        out[:, axis] = centres[voxel[axis]] + (generator.random(n) - 0.5) * width
    return out


def in_hull(points, polygons) -> np.ndarray:
    """Which points lie inside a boundary made of several rings.

    The largest polygon is taken as the outer boundary and every other one
    as a hole, so a cell with a gap in its arbor is handled correctly.
    Ports `construct/in_c.m`, but takes
    :func:`~pynetrees.hull_tree`'s polygon list rather than MATLAB's packed
    ``contourc`` matrix.

    Parameters
    ----------
    points : (n, 2) array_like
    polygons : list of (m, 2) array_like

    Returns
    -------
    np.ndarray
        Boolean mask over ``points``.
    """
    from matplotlib.path import Path

    points = np.atleast_2d(np.asarray(points, dtype=float))[:, :2]
    rings = [np.asarray(p, dtype=float)[:, :2] for p in polygons]
    if not rings:
        return np.zeros(len(points), dtype=bool)

    outer = int(np.argmax([len(r) for r in rings]))
    inside = Path(rings[outer]).contains_points(points)
    for index, ring in enumerate(rings):
        if index != outer:
            inside &= ~Path(ring).contains_points(points)
    return inside


# ---------------------------------------------------------------------------
# clone_tree
# ---------------------------------------------------------------------------

#: Regions `clone_tree` treats specially. Everything else is grown as an
#: ordinary region, in the order it appears.
_SOMA = "soma"
_PRIMARY = "primary"
_IGNORED = ("spines", "axon")


def clone_tree(trees: Tree | list[Tree], n: int = 1, bf: float = 0.4, *,
               dim: int = 3, rng=None) -> list[Tree]:
    """Grow synthetic trees resembling a measured group.

    For each region in turn: pool the group's rescaled branch and
    termination points (:func:`gscale_tree`), pick a size for this clone
    from the group's spread, scatter fresh target points at that density
    (:func:`rpoints_tree`), and wire them with
    :func:`~pynetrees.MST_tree`. Then restore the group's taper, its wriggle,
    and its soma.

    Parameters
    ----------
    trees : Tree or list[Tree]
        The group to imitate. One tree works, but the variability that
        makes clones differ from each other comes from the spread *across*
        the group, so a single input yields near-identical clones.
    n : int, default 1
        How many clones. Each is an independent MST growth and is not fast.
    bf : float, default 0.4
        Balancing factor handed to :func:`~pynetrees.MST_tree`: 0 minimises
        total wire, 1 minimises path length to the root.
    dim : {2, 3}, default 3
        Grow flat clones (MATLAB's ``'-dim2'``).
    rng : numpy Generator or int, optional
        Seed. **Required for reproducibility** -- every size, count and
        taper parameter is drawn from a normal distribution.

    Returns
    -------
    list[Tree]

    Notes
    -----
    Regions are handled by *name*, and the names are load-bearing:
    ``"soma"`` becomes a small MST blob at the origin that everything else
    attaches to, ``"primary"`` is grown first and in two passes (far half
    then near half), ``"spines"`` and ``"axon"`` are skipped entirely, and
    anything else is grown in one pass onto whatever exists so far. That is
    MATLAB's scheme; a group whose regions are named otherwise will still
    clone, just without the special handling.

    **The two-stage point count is not a heuristic that can be dropped.**
    `MST_tree` connects some target points as continuation points rather
    than as branch or termination points, so asking for ``N`` targets
    yields fewer than ``N`` topological points. Both MATLAB and this grow a
    throwaway tree with ``N`` points, count how many survived as branch or
    termination points, and regrow with ``N * (N / survivors)`` -- capped at
    ``3.5 N``. It roughly doubles the growth cost and is why cloning is slow.

    MATLAB additionally runs an outlier pass that repeatedly bins the pooled
    cloud and deletes points sitting alone in a bin, halting once half the
    cloud is gone. It is not ported: it deletes from the *pooled* cloud, so
    what counts as an outlier depends on how many cells happen to be in the
    group, and the "stop at half" guard means a sparse group can lose half
    its points to it. Bin the cloud yourself before calling if you want
    that.
    """
    if isinstance(trees, Tree):
        trees = [trees]
    generator = np.random.default_rng(rng)

    spanning = gscale_tree(trees)
    return [_one_clone(trees, spanning, bf, dim, generator) for _ in range(n)]


def _draw(mean: float, std: float, generator, floor: bool = True) -> float:
    """A normal draw, refused if it lands more than one sigma low.

    MATLAB's recurring guard -- reset to the mean when the draw falls below
    ``mean - std``. It keeps a clone from coming out degenerate when the
    group's spread is wide, at the cost of a distribution that is not
    actually normal: the lower tail is folded onto the mean rather than
    truncated.
    """
    if not np.isfinite(std):
        std = 0.0
    value = generator.normal(mean, std)
    if floor and (value < mean - std or not np.isfinite(value)):
        return mean
    return value


def _mean_std(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0, 0.0
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _one_clone(trees, spanning, bf, dim, generator) -> Tree:
    soma, soma_diameter, soma_length = _clone_soma(trees, spanning, dim, generator)
    tree = soma
    primary = None

    if _PRIMARY in spanning:
        primary = _grow_region(tree, spanning[_PRIMARY], bf, generator,
                               split_by_distance=True)
        if primary is not None:
            primary = _label(primary, _PRIMARY)

    for region in spanning.regions:
        if region.name in (_SOMA, _PRIMARY) or region.name in _IGNORED:
            continue
        grown = _grow_region(tree, region, bf, generator, split_by_distance=False)
        if grown is not None:
            tree = _label(grown, region.name)
            tree = _apply_taper(tree, region, generator)

    if primary is not None:
        primary = _apply_taper(primary, spanning[_PRIMARY], generator)
        tree = cat_tree(tree, primary, soma.n_nodes - 1, 0)

    tree = _drop_unused_regions(tree)
    tree = resample_tree(elim0_tree(root_tree(tree)), 2.0, interp_diameter=True)
    tree = smooth_tree(tree)

    amplitude = _draw(*_mean_std(spanning.wriggle[:, 0]), generator)
    wavelength = _draw(*_mean_std(spanning.wriggle[:, 1]), generator)
    if amplitude > 0 and wavelength >= 1:
        tree = jitter_tree(tree, amplitude, int(round(wavelength)), rng=generator)
    if soma_diameter is not None and soma_diameter > 0:
        tree = soma_tree(tree, soma_diameter, soma_length)
    return tree


def _clone_soma(trees, spanning, dim, generator):
    """A small MST blob standing in for the soma, or a bare root node."""
    if _SOMA not in spanning:
        return _single_node_tree(), None, None

    region = spanning[_SOMA]
    diameters = [
        float(trees[i].D[region.nodes[i]].max())
        for i in range(len(trees)) if len(region.nodes[i])
    ]

    # a box around the origin, sized from the group's own soma extents
    half = np.array([
        _draw(*_mean_std(region.extent[:, axis] / 2), generator, floor=False)
        for axis in range(3)
    ])
    half = np.where(np.isfinite(half) & (half > 0), half, 0.0)
    if dim == 2:
        half[2] = 0.0
    if not half.any():
        return _single_node_tree(), (
            _draw(*_mean_std(diameters), generator) if diameters else None), None

    corners = np.array([[0.0, 0.0, 0.0], -half, half])
    tree = MST_tree(corners[:, 0], corners[:, 1], corners[:, 2],
                    start=0, bf=0.0, thr=10000.0, mplen=15000.0)
    tree = resample_tree(tree, 1.0)
    tree = jitter_tree(tree, 0.1, 4, rng=generator)
    tree = Tree(dA=tree.dA, X=tree.X, Y=tree.Y, Z=tree.Z, D=tree.D,
                R=np.zeros(tree.n_nodes, dtype=int), rnames=[_SOMA],
                name="clone")
    diameter = _draw(*_mean_std(diameters), generator) if diameters else None
    return tree, diameter, float(len_tree(tree).sum())


def _single_node_tree() -> Tree:
    from scipy import sparse

    return Tree(dA=sparse.csr_matrix((1, 1)), X=np.zeros(1), Y=np.zeros(1),
                Z=np.zeros(1), D=np.ones(1), R=np.zeros(1, dtype=int),
                rnames=[_SOMA], name="clone")


def _grow_region(tree, region, bf, generator, split_by_distance: bool):
    """Scatter this region's points at clone scale and wire them on."""
    points = _clone_cloud(region, generator)
    if len(points) < 4:
        return None

    if not split_by_distance:
        return _mst_onto(tree, points,
                         _draw(region.mean_n, region.std_n, generator),
                         bf, generator)

    # the primary dendrite is grown outside-in: MATLAB puts two thirds of
    # the points beyond half the maximum radius and the rest inside, so the
    # trunk is laid down before the fine branching near the soma
    radius = np.linalg.norm(points, axis=1)
    split = radius.max() / 2
    grown = tree
    for mask, share in ((radius > split, 2 / 3), (radius <= split, 1 / 3)):
        if mask.sum() < 4:
            continue
        candidate = _mst_onto(
            grown, points[mask],
            _draw(share * region.mean_n, share * region.std_n, generator),
            bf, generator,
        )
        if candidate is not None:
            grown = resample_tree(candidate, 5.0)
    return None if grown is tree else grown


def _clone_cloud(region, generator) -> np.ndarray:
    """This region's pooled points, rescaled to a size drawn for this clone."""
    points = region.pooled_points()
    if len(points) == 0:
        return points
    mean, std = region.mean_extent, region.std_extent
    factor = np.array([
        _draw(mean[axis], std[axis], generator, floor=False) / mean[axis]
        for axis in range(3)
    ])
    return points * np.where(np.isfinite(factor) & (factor > 0), factor, 1.0)


def _mst_onto(tree, points, target: float, bf, generator):
    """Two-pass MST growth aiming at ``target`` branch/termination points."""
    target = int(round(target))
    if target < 2:
        return None

    grid = gdens_tree(points, sr=max(_grid_spacing(points), 1.0))
    pool = rpoints_tree(grid, 4 * target, rng=generator)
    if len(pool) < target:
        return None

    trial = _mst(tree, pool[:target], bf)
    added = (B_tree(trial) | T_tree(trial))[tree.n_nodes:]
    survivors = int(added.sum())
    if survivors == 0:
        return None
    scaled = min(int(round(target * target / survivors)),
                 int(round(3.5 * target)), len(pool))
    return _mst(tree, pool[:scaled], bf)


def _grid_spacing(points: np.ndarray) -> float:
    """MATLAB's rule: the widest extent over 30."""
    return float(round(np.ptp(points, axis=0).max() / 30))


def _mst(tree, points, bf):
    return MST_tree(points[:, 0], points[:, 1], points[:, 2], start=tree,
                    bf=bf, thr=10000.0, mplen=150000.0,
                    avoid_multifurcations=True)


def _label(tree, name: str) -> Tree:
    """Rename the nodes a growth just added into a region called ``name``.

    `MST_tree` puts everything it adds to a seed tree into a region called
    ``"new"``; the seed's own nodes keep their regions. So the rename has a
    definite target rather than having to guess from node order.
    """
    if MST_NEW_REGION not in tree.rnames:
        return tree
    R = np.asarray(tree.R).copy()
    rnames = list(tree.rnames)
    grown = rnames.index(MST_NEW_REGION)
    if name in rnames:
        R[R == grown] = rnames.index(name)
    else:
        rnames[grown] = name
    return Tree(dA=tree.dA, X=tree.X, Y=tree.Y, Z=tree.Z, D=tree.D,
                R=R, rnames=rnames, name=tree.name)


def _apply_taper(tree, region, generator) -> Tree:
    """Give this region the group's diameter profile."""
    if len(region.taper) == 0 or region.name not in tree.rnames:
        return tree
    scale = _draw(*_mean_std(region.taper[:, 0]), generator)
    offset = _draw(*_mean_std(region.taper[:, 1]), generator)
    if scale <= 0 or offset <= 0:
        return tree
    quad = quaddiameter_tree(tree, scale, offset)
    mask = np.asarray(tree.R) == tree.rnames.index(region.name)
    D = tree.D.copy()
    D[mask] = quad.D[mask]
    return Tree(dA=tree.dA, X=tree.X, Y=tree.Y, Z=tree.Z, D=D,
                R=tree.R, rnames=tree.rnames, name=tree.name)


def _drop_unused_regions(tree: Tree) -> Tree:
    used, R = np.unique(np.asarray(tree.R), return_inverse=True)
    return Tree(dA=tree.dA, X=tree.X, Y=tree.Y, Z=tree.Z, D=tree.D, R=R,
                rnames=[tree.rnames[i] for i in used.tolist()],
                name=tree.name)


# ---------------------------------------------------------------------------
# dscam_tree
# ---------------------------------------------------------------------------


@accepts_population
@empty_safe("tree")
def dscam_tree(tree: Tree, iterations: int | None = None, *,
               move: float = 0.1, cluster: float = 2.0, rng=None) -> Tree:
    """Pull branches toward each other, as a DSCAM knockout does.

    DSCAM lets a neurite recognise its own siblings and avoid them; without
    it, branches that would normally repel each other clump together. This
    reproduces the effect crudely, as Bird, Deters & Cuntz (2021) do: pick a
    node at random, find the nearest node that is *not* one of its ancestors
    or descendants, and slide that node -- with its whole subtree -- ten
    percent of the way toward it. Repeat.

    Parameters
    ----------
    tree : Tree
    iterations : int, optional
        Default ``5 * n_nodes``, as in MATLAB.
    move : float, default 0.1
        Fraction of the gap closed per step.
    cluster : float, default 2.0
        Ignore candidate partners closer than this [um]. Without it the
        nearest non-relative is usually a node a micron away on a branch
        that is already touching, and nothing moves.
    rng : numpy Generator or int, optional

    Returns
    -------
    Tree
        Same topology and diameters; only coordinates change.

    Notes
    -----
    **Resample carefully afterwards** -- MATLAB's docstring says the same.
    The operation moves nodes without adding any, so segments that were
    evenly spaced no longer are.

    One divergence: MATLAB picks the partner with
    ``find (distance == min (distance (iVector)))``, which searches the
    *unmasked* distance vector for that minimum value -- so if an excluded
    node (an ancestor, or one inside the subtree) happens to sit at exactly
    the same distance, and comes first, it is chosen instead. Here the
    partner is taken from the masked set directly, which is what the line
    was evidently meant to do.
    """
    generator = np.random.default_rng(rng)
    tree = _copy_tree(tree)
    if iterations is None:
        iterations = 5 * tree.n_nodes

    from .graphtheory import ipar_tree

    coords = np.column_stack([tree.X, tree.Y, tree.Z])
    ancestors = ipar_tree(tree)

    for _ in range(int(iterations)):
        start = int(generator.integers(1, tree.n_nodes))
        eligible = np.ones(tree.n_nodes, dtype=bool)
        eligible[ancestors[start][ancestors[start] >= 0]] = False
        subtree = sub_tree(tree, start, with_tree=False).mask
        eligible[subtree] = False

        distance = np.linalg.norm(coords - coords[start], axis=1)
        eligible &= distance >= cluster
        if not eligible.any():
            continue

        partner = int(np.flatnonzero(eligible)[np.argmin(distance[eligible])])
        step = (coords[partner] - coords[start]) * move
        coords[start] += step
        coords[subtree] += step

    tree.X, tree.Y, tree.Z = coords[:, 0].copy(), coords[:, 1].copy(), coords[:, 2].copy()
    return tree


# ---------------------------------------------------------------------------
# spines_tree
# ---------------------------------------------------------------------------


class SpineResult(NamedTuple):
    """Output of :func:`spines_tree` with ``full_output=True``."""

    tree: Tree
    heads: np.ndarray
    """Node index of every spine head."""
    necks: np.ndarray
    """Node index of every spine neck."""


@accepts_population
@empty_safe("tree")
def spines_tree(tree: Tree, spines=100, neck_diameter: float = 0.5,
                head_diameter: float = 1.0, neck_length: float = 1.0,
                neck_length_std: float = 1.0, nodes=None, *,
                separate_regions: bool = False, rng=None,
                full_output: bool = False):
    """Attach spines to a tree.

    Each spine is two nodes: a **neck** of diameter ``neck_diameter``
    standing off the dendrite by a length drawn from
    ``N(neck_length, neck_length_std)``, and a **head** one
    ``head_diameter`` further out -- so the head is a cylinder as long as it
    is wide, which is what makes its surface area come out roughly right.

    The direction is perpendicular to the local dendrite, at a uniformly
    random angle around it, so spines fan out around the cable rather than
    all pointing the same way.

    Parameters
    ----------
    tree : Tree
    spines : int or array_like, default 100
        How many spines to add at randomly chosen nodes; **or** an integer
        array of node indices to spine; **or** an ``(n, 3)`` array of
        explicit neck coordinates.
    neck_diameter : float, default 0.5
    head_diameter : float, default 1.0
        Also the head's length -- see above.
    neck_length, neck_length_std : float, default 1.0
        Mean and spread of the neck length [um].
    nodes : array_like, optional
        Restrict random placement to these nodes (MATLAB's ``ipart``).
    separate_regions : bool, default False
        Label necks and heads as two regions (``spine_neck``,
        ``spine_head``) instead of one region called ``spines``. MATLAB's
        ``'-sr'``.
    rng : numpy Generator or int, optional
    full_output : bool, default False
        Return :class:`SpineResult` -- the tree plus the head and neck node
        indices -- instead of just the tree.

    Returns
    -------
    Tree or SpineResult

    Notes
    -----
    Three things MATLAB's version gets wrong here.

    **Its documented coordinate input cannot be reached.** ``XYZ`` is
    dispatched as ``numel (XYZ) == 1`` -> a count, ``elseif all (XYZ < N)``
    -> node indices, and nothing else. An ``(n, 3)`` matrix of coordinates
    therefore either falls into the *indices* branch (when the cell happens
    to sit near the origin, so every coordinate is below the node count) or
    matches neither, leaving ``indy`` undefined and raising. This port
    dispatches on **shape** -- an ``(n, 3)`` array is coordinates, a 1D
    integer array is indices -- so all three documented forms work.

    **Its ``'-sr'`` branch reads an undefined variable.** ``flag`` is only
    assigned when no ``spine_neck`` region exists; if one does but
    ``spine_head`` does not, ``iR (2) = ... + 1 + flag`` raises.

    **It returns only the last spine's indices.** ``indhead`` and
    ``indneck`` are overwritten each pass of the loop, so despite being
    documented as "node indices of spine heads"/"necks" they hold two
    numbers, not two arrays. Here they are the full arrays.

    A fourth, geometric: MATLAB draws the neck length from a normal
    distribution and then places the head at ``neck + dhead * dXYZ``
    regardless of sign. When the draw is negative -- 16% of the time at its
    own defaults of mean 1 and standard deviation 1 -- the neck goes one way
    and the head the other, so the head ends up between the neck and the
    dendrite or inside it. Here the *direction* is flipped rather than the
    length, which is distributionally identical (the direction is uniformly
    random to begin with) and keeps the head beyond the neck.
    """
    from .edit import insert_tree

    generator = np.random.default_rng(rng)
    candidates = (np.arange(tree.n_nodes) if nodes is None
                  else np.asarray(nodes, dtype=int))

    targets, positions = _spine_sites(tree, spines, candidates, generator)
    if len(targets) == 0:
        return SpineResult(tree, np.empty(0, int), np.empty(0, int)) \
            if full_output else tree

    direction = direction_tree(tree, normalize=True)
    outward = _perpendicular(direction[targets], generator)
    if positions is None:
        lengths = generator.normal(neck_length, neck_length_std, len(targets))
        # a normal draw can come out negative, which would put the neck on
        # one side of the cable and the head on the other -- the head would
        # sit between the neck and the dendrite, or pass through it. Flip
        # the direction instead: the direction is uniformly random anyway,
        # so this changes nothing distributionally and fixes the geometry.
        # MATLAB does not, and produces such spines at the rate the normal
        # distribution goes negative (16% at its defaults).
        outward = outward * np.sign(np.where(lengths == 0, 1.0, lengths))[:, None]
        lengths = np.abs(lengths)
        coords = np.column_stack([tree.X, tree.Y, tree.Z])[targets]
        positions = coords + lengths[:, None] * outward

    tree, neck_region, head_region = _spine_regions(tree, separate_regions)

    # every neck at once, then every head onto the necks -- two calls
    # rather than 2N, since `insert_tree` rebuilds the whole tree each time
    tree, necks = insert_tree(
        tree, positions[:, 0], positions[:, 1], positions[:, 2],
        np.full(len(targets), neck_diameter), targets,
        R=np.full(len(targets), neck_region), full_output=True,
    )
    necks = np.atleast_1d(necks)
    tips = positions + head_diameter * outward
    tree, heads = insert_tree(
        tree, tips[:, 0], tips[:, 1], tips[:, 2],
        np.full(len(targets), head_diameter), necks,
        R=np.full(len(targets), head_region), full_output=True,
    )
    heads = np.atleast_1d(heads)

    if full_output:
        return SpineResult(tree, heads, necks)
    return tree


def _spine_sites(tree, spines, candidates, generator):
    """Resolve the ``spines`` argument to (parent nodes, neck coordinates).

    Dispatch is on **shape**, not magnitude -- see this function's caller.
    """
    spines = np.asarray(spines)
    if spines.ndim == 0:
        picks = generator.integers(0, len(candidates), int(spines))
        return candidates[picks], None
    if spines.ndim == 2 and spines.shape[1] == 3:
        # explicit coordinates: attach each to the nearest candidate node
        coords = np.column_stack([tree.X, tree.Y, tree.Z])[candidates]
        nearest = np.argmin(
            np.linalg.norm(spines[:, None, :] - coords[None, :, :], axis=2),
            axis=1,
        )
        return candidates[nearest], spines.astype(float)
    if spines.ndim == 1:
        return np.asarray(spines, dtype=int), None
    raise ValueError(
        "spines must be a count, a 1D array of node indices, or an (n, 3) "
        f"array of coordinates; got shape {spines.shape}"
    )


def _perpendicular(direction: np.ndarray, generator) -> np.ndarray:
    """A unit vector perpendicular to each row, at a random angle about it.

    MATLAB builds this from an SVD of the direction plus a 4x4
    ``makehgtform`` rotation matrix. Two orthonormal vectors spanning the
    perpendicular plane, combined as ``cos(t) u + sin(t) v``, is the same
    thing and needs neither.
    """
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    axis = np.divide(direction, norm, out=np.zeros_like(direction),
                     where=norm > 0)
    axis[norm.ravel() == 0] = np.array([0.0, 0.0, 1.0])

    # any vector not parallel to the axis works as a seed for the cross
    seed = np.tile([1.0, 0.0, 0.0], (len(axis), 1))
    parallel = np.abs(axis[:, 0]) > 0.9
    seed[parallel] = [0.0, 1.0, 0.0]

    u = np.cross(axis, seed)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(axis, u)

    angle = generator.uniform(0, 2 * np.pi, len(axis))[:, None]
    return np.cos(angle) * u + np.sin(angle) * v


def _spine_regions(tree: Tree, separate: bool):
    """Ensure the spine regions exist, returning their indices."""
    rnames = list(tree.rnames)
    wanted = ("spine_neck", "spine_head") if separate else ("spines", "spines")
    indices = []
    for name in wanted:
        if name not in rnames:
            rnames.append(name)
        indices.append(rnames.index(name))
    relabelled = Tree(
        dA=tree.dA, X=tree.X, Y=tree.Y, Z=tree.Z, D=tree.D,
        R=np.asarray(tree.R), rnames=rnames, name=tree.name,
        frustum=tree.frustum, Ri=tree.Ri, Gm=tree.Gm, Cm=tree.Cm,
    )
    return relabelled, indices[0], indices[1]


# ---------------------------------------------------------------------------
# PP_generator_tree
# ---------------------------------------------------------------------------


def PP_generator_tree(n=100, R: float = 1.2, a: float = 0.1, *,
                      alpha: float = 0.5, n_mc: int = 20,
                      level: float = 0.05, epsilon: float = 0.0,
                      dim: int = 2, box: float = 100.0, tol: float = 0.01,
                      max_iter: int = 200, rng=None,
                      full_output: bool = False):
    """Scatter points with a prescribed degree of spatial order.

    Produces a cloud whose Clark-Evans ratio (:func:`~pynetrees.r_mc_tree`)
    matches ``R``: below 1 the points are clustered, 1 is Poisson, above 1
    they are spaced more regularly than chance. Useful as a synaptic or
    contact-point target for :func:`~pynetrees.MST_tree` when the arrangement
    of the targets is itself the thing under study.

    It works by repeatedly nudging every point along the line to its nearest
    neighbour -- toward it to cluster, away to disperse -- by a step that
    shrinks as the measured R approaches the target.

    Parameters
    ----------
    n : int or (n, dim) array_like, default 100
        How many points to place, or a starting cloud to rearrange.
    R : float, default 1.2
        Target Clark-Evans ratio.
    a : float, default 0.1
        Step size. The sign is chosen automatically; only the magnitude
        matters. Must be nonzero.
    alpha, n_mc, level : see :func:`~pynetrees.r_mc_tree`
        ``n_mc`` defaults to 20 rather than that function's 100, because R
        is remeasured on every iteration.
    epsilon : float, default 0.0
        Minimum separation between points [um] -- an exclusion zone
        standing in for the physical size of whatever the points represent.
        Moves that would violate it are refused.
    dim : {2, 3}, default 2
    box : float, default 100.0
        Points live in ``[-box, box]`` per axis and are clamped to it.
    tol : float, default 0.01
        Stop once ``|measured - R|`` falls below this.
    max_iter : int, default 200
        Give up after this many iterations and warn. MATLAB's loop has no
        bound at all and will spin forever on an unreachable target -- and
        many are unreachable, since R is capped by how tightly the exclusion
        zone and the box let points pack.
    rng : numpy Generator or int, optional
    full_output : bool, default False
        Also return the iteration count and the R measured at each step.

    Returns
    -------
    np.ndarray or tuple
        ``(n, dim)`` points, or ``(points, n_iterations, R_history)``.
    """
    from .stats import r_mc_tree

    if a == 0:
        raise ValueError("a must be nonzero -- it is the step size")
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    generator = np.random.default_rng(rng)

    points = (_initial_cloud(int(n), dim, box, epsilon, generator)
              if np.ndim(n) == 0
              else np.array(n, dtype=float)[:, :dim])

    measure = lambda cloud: r_mc_tree(  # noqa: E731 - one expression, used twice
        cloud, alpha=alpha, n_mc=n_mc, level=level, dim=dim, rng=generator
    ).R

    current = measure(points)
    history = [current]
    step = -abs(a) if current > R else abs(a)

    iterations = 0
    while abs(current - R) > tol and iterations < max_iter:
        gap = abs(R - current)
        points = _nudge(points, gap * step, box, epsilon)
        current = measure(points)
        history.append(current)
        iterations += 1

    if abs(current - R) > tol:
        warnings.warn(
            f"gave up after {max_iter} iterations at R = {current:.3f}, "
            f"target {R}; the target may be out of reach for {len(points)} "
            f"points in this box with epsilon = {epsilon}",
            stacklevel=2,
        )
    if full_output:
        return points, iterations, np.array(history)
    return points


def _initial_cloud(n, dim, box, epsilon, generator) -> np.ndarray:
    """A starting cloud, respecting the exclusion zone if there is one."""
    if epsilon <= 0:
        return generator.uniform(-box, box, (n, dim))

    points = np.zeros((1, dim))
    attempts = 0
    while len(points) < n:
        candidate = generator.uniform(-box, box, dim)
        if np.linalg.norm(points - candidate, axis=1).min() >= epsilon:
            points = np.vstack([points, candidate])
        attempts += 1
        if attempts > 1000 * n:
            raise ValueError(
                f"could not place {n} points at least {epsilon} apart in a "
                f"{2 * box} box -- the exclusion zone is too large"
            )
    return points


def _nudge(points: np.ndarray, step: float, box: float,
           epsilon: float) -> np.ndarray:
    """Move each point along the line to its nearest neighbour."""
    from scipy.spatial import cKDTree

    _, nearest = cKDTree(points).query(points, k=2)
    offset = (points - points[nearest[:, 1]]) * step
    moved = np.clip(points + offset, -box, box)

    if epsilon > 0:
        # refuse any move that would break the exclusion zone, point by
        # point, so one bad move does not discard the whole sweep
        for index in range(len(moved)):
            others = np.delete(moved, index, axis=0)
            if np.linalg.norm(others - moved[index], axis=1).min() < epsilon:
                moved[index] = points[index]
    return moved
