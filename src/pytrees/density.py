"""Density grids, space-filling hulls and territory volumes.

Ports `graphical/hull_tree.m`, `graphical/gdens_tree.m`,
`graphical/lego_tree.m`, `graphical/vhull_tree.m` and
`metrics/share_boundary_tree.m`. They are one module rather than five
scattered functions because they all rest on the same two operations --
**bin the tree into a voxel grid**, and **measure distance from arbitrary
points to the tree** -- and porting them separately would have meant
writing that twice more.

The distinction that matters throughout: `chull_tree` (in `plotting.py`)
computes the *convex* hull, a single enclosing shell. These compute
**space-filling** hulls: the surface at a fixed distance `thr` from the
arbor itself, which follows the arbor's real shape, concavities and all.
For anything about how a cell fills space -- territory, overlap between
cells, density gradients -- the convex hull is far too generous.

Distances are measured to the nearest **segment**, not the nearest node, so
the result does not depend on how finely the morphology happens to be
sampled. That is the same choice MATLAB makes, and it is why this cannot
simply be a `cKDTree` query over node coordinates.

`hull_tree`'s 3D isosurface needs `scikit-image` (marching cubes), an
optional dependency under the ``[plot]`` extra; the 2D contour needs only
matplotlib.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np

from ._compat import resolve_dim
from .core import Tree
from .graphtheory import T_tree
from .metrics import cyl_tree

__all__ = [
    "gdens_tree",
    "hull_tree",
    "lego_tree",
    "vhull_tree",
    "share_boundary_tree",
    "Boundary",
    "boundary_tree",
    "convexity_tree",
]


# ---------------------------------------------------------------------------
# shared machinery
# ---------------------------------------------------------------------------


def _grid_axis(values: np.ndarray, spec, margin: float) -> np.ndarray:
    """One axis of the sampling grid.

    ``spec`` is either an explicit array of coordinates, or a count of
    intervals -- matching MATLAB, where a scalar `bx` means "divide the
    padded extent into this many steps" rather than "put this many points".
    """
    spec = np.asarray(spec)
    if spec.size > 1:
        return spec.astype(float)
    lo = float(values.min()) - margin
    hi = float(values.max()) + margin
    return np.linspace(lo, hi, int(spec) + 1)


def _segment_distance(points: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                      chunk: int = 16384) -> np.ndarray:
    """Distance from each point to the nearest segment.

    Parameters
    ----------
    points : (P, D) array
    starts, ends : (N, D) arrays
        Segment endpoints. Zero-length segments are handled: they collapse
        to point-to-point distance rather than dividing by zero (the root's
        own self-segment is always one of these).
    chunk : int
        How many points to process at once. The full computation is
        ``P x N``, which for a 51^3 grid against a 2252-node cell is 3e8
        pairs -- fine in aggregate, but not as one allocation.

    Returns
    -------
    (P,) array of distances.

    Notes
    -----
    Point-to-*segment*, not point-to-node: sampling a morphology more
    finely must not change the shape of its hull. This is the projection
    formula, clamped to the segment, which is what MATLAB does inline.

    **Precision.** The squared distance is expanded rather than formed from
    an explicit closest point, which trades a little accuracy for roughly a
    3x speedup and a much smaller memory footprint. The expansion is the
    textbook-unstable one, so a distance that should be exactly zero comes
    back around ``|coords| * sqrt(eps)`` instead -- measured at **2e-6 um**
    on the bundled sample, and scaling with the coordinate range rather
    than growing without bound. Reconstruction precision is ~0.1 um and
    hull grids are spaced in whole microns, so this is around ten orders of
    magnitude below anything observable. Points are centred first, which
    halves it for free.
    """
    origin = starts.mean(axis=0)
    points = points - origin
    starts = starts - origin
    ends = ends - origin
    seg = ends - starts
    seg_len2 = np.einsum("ij,ij->i", seg, seg)
    # a zero-length segment has no direction to project onto; clamping its
    # parameter to 0 makes the formula degrade cleanly to point distance
    safe_len2 = np.where(seg_len2 > 0, seg_len2, 1.0)

    # Expanded rather than computed via an explicit closest point:
    #     |p - (s + u*seg)|^2 = |p - s|^2 - 2u (p-s).seg + u^2 |seg|^2
    # Every term on the right is a (P, N) matrix reachable by matmul, so the
    # (P, N, 3) intermediates disappear -- which is what made the first
    # version memory-bound. `p @ s.T` also lands in BLAS instead of a numpy
    # broadcast loop.
    start_sq = np.einsum("nd,nd->n", starts, starts)
    start_dot_seg = np.einsum("nd,nd->n", starts, seg)

    out = np.empty(len(points), dtype=float)
    for lo in range(0, len(points), chunk):
        block = points[lo : lo + chunk]
        block_sq = np.einsum("pd,pd->p", block, block)

        # squared distance to each segment's start point
        to_start_sq = (
            block_sq[:, None] - 2.0 * (block @ starts.T) + start_sq[None, :]
        )
        # projection of (p - s) onto the segment, clamped to it
        delta_dot_seg = (block @ seg.T) - start_dot_seg[None, :]
        u = np.clip(np.where(seg_len2 > 0, delta_dot_seg / safe_len2, 0.0), 0.0, 1.0)

        d2 = to_start_sq - 2.0 * u * delta_dot_seg + (u * u) * seg_len2[None, :]
        np.maximum(d2, 0.0, out=d2)  # rounding can push a zero slightly negative
        out[lo : lo + chunk] = np.sqrt(d2.min(axis=1))
    return out


def _tree_segments(tree: Tree, dim: int):
    """Segment endpoints as ``(starts, ends)`` arrays of shape ``(n, dim)``."""
    if dim == 2:
        X1, X2, Y1, Y2 = cyl_tree(tree, dim=2)
        return np.column_stack([X1, Y1]), np.column_stack([X2, Y2])
    X1, X2, Y1, Y2, Z1, Z2 = cyl_tree(tree, dim=3)
    return np.column_stack([X1, Y1, Z1]), np.column_stack([X2, Y2, Z2])


# ---------------------------------------------------------------------------
# gdens_tree / lego_tree
# ---------------------------------------------------------------------------


class DensityGrid(NamedTuple):
    """A tree binned into a regular voxel grid."""

    counts: np.ndarray
    """Node count per voxel, indexed ``[ix, iy, iz]``."""
    x: np.ndarray
    """Voxel-centre coordinates along x [um]."""
    y: np.ndarray
    z: np.ndarray

    @property
    def edges(self):
        """The bin edges, for handing straight to `pcolormesh`/`histogramdd`."""
        half = (self.x[1] - self.x[0]) / 2 if len(self.x) > 1 else 0.5
        return tuple(np.append(a - half, a[-1] + half) for a in (self.x, self.y, self.z))


def gdens_tree(tree: Tree, sr: float = 5.0, nodes=None) -> DensityGrid:
    """Bin a tree's nodes into a regular ``sr``-sized voxel grid.

    Parameters
    ----------
    tree : Tree
    sr : float, default 5.0
        Voxel edge length [um].
    nodes : array_like, optional
        Restrict to a subset of nodes (MATLAB's ``ipart``).

    Returns
    -------
    DensityGrid
        ``counts`` indexed ``[ix, iy, iz]``, plus the voxel-centre
        coordinates along each axis.

    Notes
    -----
    Counts **nodes**, not length, exactly as MATLAB does. That makes the
    result depend on how the morphology was sampled, so compare densities
    only between trees resampled the same way -- `resample_tree` first if
    they were not. (Length-weighted density would be the more robust
    measure, but it is not what this function is, and silently changing the
    quantity would make results incomparable with published MATLAB ones.)

    Indexing is ``[x, y, z]``. MATLAB's is ``[y, x, z]``, following its
    image convention; that transposition is deliberate here, since every
    other array in this port is ``[x, y, z]`` and mixing the two silently
    is exactly how axis bugs happen.
    """
    X, Y, Z = tree.X, tree.Y, tree.Z
    if nodes is not None:
        nodes = np.asarray(nodes, dtype=int)
        X, Y, Z = X[nodes], Y[nodes], Z[nodes]

    edges = [
        np.arange(a.min() - 2 * sr, a.max() + 3 * sr, sr) for a in (X, Y, Z)
    ]
    counts, _ = np.histogramdd(np.column_stack([X, Y, Z]), bins=edges)
    centres = [e[:-1] + sr / 2 for e in edges]
    return DensityGrid(counts, *centres)


def lego_tree(tree: Tree, sr: float = 5.0, nodes=None, ax=None,
              cmap: str = "viridis", threshold: float = 0.0):
    """Draw a tree's density grid as opaque voxels -- MATLAB's "lego" plot.

    Parameters
    ----------
    tree : Tree
    sr : float, default 5.0
        Voxel edge length [um].
    nodes : array_like, optional
    ax : matplotlib 3D Axes, optional
        Created if omitted.
    cmap : str
    threshold : float, default 0.0
        Only draw voxels holding more than this many nodes.

    Returns
    -------
    matplotlib.axes.Axes

    Notes
    -----
    A blunt instrument by design: it shows occupancy, not shape. For "where
    does this cell reach", :func:`hull_tree` is the better tool -- but a
    lego plot makes *density* differences legible in a way a smooth hull
    cannot, which is why the original has both.
    """
    import matplotlib.pyplot as plt

    grid = gdens_tree(tree, sr=sr, nodes=nodes)
    filled = grid.counts > threshold
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})

    if filled.any():
        normed = grid.counts / grid.counts.max()
        colors = plt.get_cmap(cmap)(normed)
        colors[..., 3] = np.where(filled, 0.9, 0.0)
        ax.voxels(filled, facecolors=colors, edgecolor=None)

    ax.set_xlabel("x [voxels]")
    ax.set_ylabel("y [voxels]")
    ax.set_zlabel("z [voxels]")
    ax.set_title(f"node density, {sr:g} um voxels")
    return ax


# ---------------------------------------------------------------------------
# hull_tree
# ---------------------------------------------------------------------------


class HullResult(NamedTuple):
    """A space-filling hull at a fixed distance from a tree."""

    vertices: np.ndarray
    """``(V, dim)`` contour vertices [um]."""
    faces: np.ndarray | None
    """``(F, 3)`` triangle indices in 3D; ``None`` in 2D, where the contour
    is a sequence of closed polylines instead."""
    polygons: list[np.ndarray] | None
    """2D only: the closed boundary polylines."""
    distances: np.ndarray | None
    """The sampled distance field, if ``return_distances``."""
    grid: tuple[np.ndarray, ...]
    """The sampling grid axes the field was evaluated on."""


def hull_tree(tree: Tree, thr: float = 25.0, bx=50, by=50, bz=50,
              dim: int | None = None, *, return_distances: bool = False,
              dim2=None) -> HullResult:
    """The surface lying ``thr`` um from the tree -- a space-filling hull.

    Samples distance-to-the-nearest-segment on a regular grid and extracts
    the ``thr`` isosurface. Unlike a convex hull this follows concavities,
    so it actually describes the volume a cell occupies rather than the
    volume it spans.

    Parameters
    ----------
    tree : Tree
    thr : float, default 25.0
        Distance [um] defining the surface. Smaller values track the arbor
        more tightly and need a finer grid to resolve.
    bx, by, bz : int or array_like, default 50
        Grid resolution per axis: an integer is a number of *intervals*
        across the padded extent (MATLAB's convention); an array is used as
        explicit coordinates.
    dim : {2, 3}, optional
        Default 3.
    return_distances : bool, default False
        Also return the sampled distance field (MATLAB's ``'-F'``).
    dim2 : bool, optional
        **Deprecated** boolean spelling of ``dim``.

    Returns
    -------
    HullResult

    Raises
    ------
    ImportError
        In 3D, if `scikit-image` is not installed (needed for marching
        cubes). Install the ``[plot]`` extra.

    Notes
    -----
    Cost is ``grid points x segments``, and the grid grows cubically -- the
    default 50 intervals per axis is 132651 points, which against a
    2252-node cell is 3e8 distance evaluations. Halving `thr` usually means
    doubling the resolution to resolve it, i.e. 8x the work; raise the
    resolution deliberately rather than by default.

    If the isosurface comes out empty, `thr` is likely smaller than the
    grid spacing, so no cell straddles the level -- the warning says so.
    """
    dim = resolve_dim(dim, dim2)
    starts, ends = _tree_segments(tree, dim)

    axes_values = [tree.X, tree.Y] + ([tree.Z] if dim == 3 else [])
    specs = [bx, by] + ([bz] if dim == 3 else [])
    axes = [
        _grid_axis(v, spec, margin=2 * thr)
        for v, spec in zip(axes_values, specs)
    ]

    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([m.ravel() for m in mesh])
    field = _segment_distance(points, starts, ends).reshape(mesh[0].shape)

    distances = field if return_distances else None

    if dim == 2:
        polygons = _contour_2d(axes, field, thr)
        vertices = np.vstack(polygons) if polygons else np.empty((0, 2))
        if not polygons:
            warnings.warn(
                f"hull_tree: no contour at thr={thr}; the grid spacing "
                f"({axes[0][1] - axes[0][0]:.2f} um) may be too coarse to "
                f"resolve it",
                stacklevel=2,
            )
        return HullResult(vertices, None, polygons, distances, tuple(axes))

    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError(
            "hull_tree needs scikit-image for the 3D isosurface; install it "
            "with `pip install pytrees[plot]` (2D works without it)"
        ) from None

    if field.min() > thr or field.max() < thr:
        warnings.warn(
            f"hull_tree: distance field spans {field.min():.2f}-"
            f"{field.max():.2f} um and never crosses thr={thr}, so the "
            f"isosurface is empty",
            stacklevel=2,
        )
        return HullResult(np.empty((0, 3)), np.empty((0, 3), dtype=int),
                          None, distances, tuple(axes))

    spacing = tuple(float(a[1] - a[0]) for a in axes)
    verts, faces, _normals, _values = marching_cubes(field, level=thr,
                                                     spacing=spacing)
    # marching_cubes works in grid coordinates; shift back to the tree's
    verts = verts + np.array([a[0] for a in axes])
    return HullResult(verts, faces, None, distances, tuple(axes))


def _contour_2d(axes, field, thr) -> list[np.ndarray]:
    """Closed boundary polylines of the 2D distance field at ``thr``."""
    import matplotlib.pyplot as plt

    figure = plt.figure()
    try:
        contour = plt.contour(axes[0], axes[1], field.T, levels=[thr])
        polygons = [np.asarray(p) for p in contour.allsegs[0]]
    finally:
        plt.close(figure)
    return polygons


# ---------------------------------------------------------------------------
# vhull_tree
# ---------------------------------------------------------------------------


class VoronoiResult(NamedTuple):
    """Per-node territory from a Voronoi subdivision."""

    volumes: np.ndarray
    """Territory volume [um^3] (or area [um^2] in 2D) per node. ``NaN``
    where the cell is unbounded and could not be clipped."""
    regions: list
    """The clipped polygon/polyhedron vertices per node."""


def vhull_tree(tree: Tree, nodes=None, boundary=None, thr: float = 25.0,
               dim: int | None = None, *, dim2=None) -> VoronoiResult:
    """Voronoi territory of every node, clipped to the tree's hull.

    Each node gets the region of space closer to it than to any other node,
    trimmed to the space-filling hull so the outermost nodes do not receive
    unbounded territory. The per-node volumes are what density statistics
    are built from.

    Parameters
    ----------
    tree : Tree
    nodes : array_like, optional
        Subset of nodes to tessellate.
    boundary : array_like, optional
        Explicit boundary points to clip against. Defaults to the vertices
        of :func:`hull_tree` at ``thr``.
    thr : float, default 25.0
        Hull distance used when ``boundary`` is not given.
    dim : {2, 3}, optional
        Default 3.
    dim2 : bool, optional
        **Deprecated** boolean spelling of ``dim``.

    Returns
    -------
    VoronoiResult

    Notes
    -----
    Unbounded Voronoi cells get ``NaN`` rather than a clipped guess.
    MATLAB's version silently drops them, which quietly biases any mean
    computed over the result -- the outermost nodes are exactly the ones
    with the largest territories.
    """
    dim = resolve_dim(dim, dim2)
    from scipy.spatial import ConvexHull, QhullError, Voronoi

    coords = np.column_stack(
        [tree.X, tree.Y] + ([tree.Z] if dim == 3 else [])
    )
    if nodes is not None:
        coords = coords[np.asarray(nodes, dtype=int)]

    if boundary is None:
        boundary = hull_tree(tree, thr=thr, dim=dim).vertices
    boundary = np.asarray(boundary, dtype=float)

    if len(coords) < dim + 2:
        raise ValueError(
            f"need at least {dim + 2} nodes to tessellate in {dim}D, "
            f"got {len(coords)}"
        )

    # Including the boundary points as Voronoi sites is what bounds the
    # outer cells: every real node then has neighbours on all sides.
    sites = np.vstack([coords, boundary]) if len(boundary) else coords
    voronoi = Voronoi(sites)

    volumes = np.full(len(coords), np.nan)
    regions: list = [None] * len(coords)
    for i in range(len(coords)):
        region_index = voronoi.point_region[i]
        vertex_ids = voronoi.regions[region_index]
        if not vertex_ids or -1 in vertex_ids:
            continue  # unbounded: no finite territory to report
        polygon = voronoi.vertices[vertex_ids]
        regions[i] = polygon
        try:
            volumes[i] = ConvexHull(polygon).volume
        except QhullError:
            pass  # degenerate cell, leave as NaN
    return VoronoiResult(volumes, regions)


# ---------------------------------------------------------------------------
# share_boundary_tree
# ---------------------------------------------------------------------------


def share_boundary_tree(tree1: Tree, tree2: Tree, thr: float = 25.0,
                        sr: float = 5.0) -> float:
    """Volume [um^3] shared by two trees' space-filling hulls.

    How much of the space one cell occupies is also occupied by the other --
    the quantity behind questions about territorial overlap and tiling.

    Parameters
    ----------
    tree1, tree2 : Tree
    thr : float, default 25.0
        Hull distance [um].
    sr : float, default 5.0
        Voxel edge [um] of the shared grid the two hulls are rasterised
        onto. The result is a voxel count times ``sr ** 3``, so accuracy is
        set by ``sr`` and cost by ``sr ** -3``.

    Returns
    -------
    float
        Shared volume [um^3]. Zero if the hulls do not meet.

    Notes
    -----
    Voxelised rather than computed as an exact mesh intersection, which is
    what MATLAB does too. Both hulls are sampled on **one common grid**
    spanning the union of their extents -- rasterising each on its own grid
    and comparing would compare different things.
    """
    all_x = np.concatenate([tree1.X, tree2.X])
    all_y = np.concatenate([tree1.Y, tree2.Y])
    all_z = np.concatenate([tree1.Z, tree2.Z])

    axes = [
        np.arange(a.min() - 2 * thr, a.max() + 2 * thr + sr, sr)
        for a in (all_x, all_y, all_z)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([m.ravel() for m in mesh])

    inside = []
    for tree in (tree1, tree2):
        starts, ends = _tree_segments(tree, 3)
        inside.append(_segment_distance(points, starts, ends) <= thr)

    shared = int(np.count_nonzero(inside[0] & inside[1]))
    return shared * sr**3


# ---------------------------------------------------------------------------
# boundary_tree / convexity_tree
# ---------------------------------------------------------------------------


class Boundary(NamedTuple):
    """A concave boundary (alpha shape) wrapped around a point cloud.

    Attributes
    ----------
    vertices : np.ndarray
        ``(v, dim)`` coordinates of the points lying on the surface.
    faces : np.ndarray
        ``(f, dim)`` triangles (3D) or edges (2D) making up the surface,
        indexing into ``vertices``.
    volume : float
        Enclosed volume [um^3] in 3D, enclosed area [um^2] in 2D. This is
        the sum over the filled simplices, so a boundary with holes or
        several disconnected lobes is measured correctly rather than as
        its outer envelope.
    points : np.ndarray
        ``(n, dim)`` the full point set that was wrapped.
    simplices : np.ndarray
        ``(s, dim + 1)`` the *filled* simplices -- tetrahedra in 3D,
        triangles in 2D -- indexing into ``points``. This is the interior,
        as opposed to ``faces`` which is only the shell. Needed to sample
        uniformly inside the boundary (see :func:`r_mc_tree`).
    polygon : np.ndarray or None
        2D only: ``(p, 2)`` surface vertices walked into boundary order, so
        consecutive rows are joined by an edge. ``None`` in 3D, where no
        such ordering exists. MATLAB's ``bound.xv``/``bound.yv``.
    """

    vertices: np.ndarray
    faces: np.ndarray
    volume: float
    points: np.ndarray
    simplices: np.ndarray
    polygon: np.ndarray | None


def boundary_tree(tree: Tree, shrink: float | None = None,
                  dim: int | None = None, nodes=None, *,
                  c: float | None = None, dim2=None) -> Boundary:
    """Concave boundary (alpha shape) around a tree's points.

    Parameters
    ----------
    tree : Tree
    shrink : float in [0, 1], default 0.5
        How tightly the boundary wraps. ``0`` gives the convex hull; ``1``
        gives the tightest shape that still envelops every point. Matches
        the sense of MATLAB's ``boundary`` shrink factor.
    dim : {2, 3}, optional
        Default 3.
    nodes : array_like, optional
        Subset of nodes to wrap. Defaults to all.
    c : float, optional
        Convexity, as returned by :func:`convexity_tree`. MATLAB's
        `boundary_tree` is parameterised this way and sets its shrink
        factor to ``1 - c``, so that a convex cell is wrapped loosely and a
        concave one tightly. Supplying ``c`` does exactly that. Mutually
        exclusive with ``shrink``.
    dim2 : bool, optional
        **Deprecated** boolean spelling of ``dim``.

    Returns
    -------
    Boundary

    Notes
    -----
    **Not verifiable against MATLAB here.** MATLAB's `boundary_tree` calls
    its built-in ``boundary()``, which Octave does not implement, so no
    differential check was possible on this machine. The algorithm below is
    the standard alpha-shape construction that ``boundary()`` documents
    itself as performing -- Delaunay triangulation with simplices discarded
    above a circumradius cutoff -- but the exact mapping from shrink factor
    to cutoff is undocumented on MATLAB's side, so **boundary vertices will
    not match theirs exactly**.

    Separately, MATLAB's `boundary_tree` cannot run at all unless ``c`` is
    passed: its default branch does ``pars = convexity_tree (...)``, which
    replaces the whole parsed-options struct with a bare scalar, so the very
    next line's ``pars.c`` errors out. Its own documented example,
    ``boundary_tree (sample_tree, '-dim3')``, is one of the calls that
    fails. See MATLAB_TOOLBOX_BUGS.md.

    For "how much space does this cell occupy", prefer :func:`hull_tree`:
    it wraps the *arbor* at a stated distance rather than wrapping its
    *points* by a unitless tightness knob, so the result means something
    physical.
    """
    if c is not None:
        if shrink is not None:
            raise ValueError(
                "pass either shrink or c, not both -- c sets shrink = 1 - c"
            )
        shrink = 1.0 - float(c)
    elif shrink is None:
        shrink = 0.5
    dim = resolve_dim(dim, dim2)
    if not 0.0 <= shrink <= 1.0:
        raise ValueError(f"shrink must lie in [0, 1], got {shrink}")

    coords = np.column_stack(
        [tree.X, tree.Y] + ([tree.Z] if dim == 3 else [])
    )
    if nodes is not None:
        coords = coords[np.asarray(nodes, dtype=int)]
    return _alpha_shape(coords, shrink)


def _alpha_shape(coords: np.ndarray, shrink: float) -> Boundary:
    """Alpha shape of a point cloud at a given tightness.

    One code path covers the whole family, including the convex hull: a
    Delaunay triangulation already fills exactly the convex hull, so
    keeping every simplex (``shrink = 0``) *is* the convex hull, and
    tightening only ever removes simplices from it.
    """
    from scipy.spatial import Delaunay

    triangulation = Delaunay(coords)
    radii = _circumradii(coords, triangulation.simplices)

    # MATLAB documents the two endpoints: shrink 0 is the convex hull, and
    # shrink 1 is "the tightest region that envelops the points" -- note
    # *envelops*, so even the tightest shape still contains every point.
    # So the family is interpolated between those two, rather than by
    # sliding a raw quantile: a quantile cutoff hits a shape that abandons
    # most of the tree long before shrink reaches 1 (at shrink=1 it kept a
    # 4-vertex sliver of a 197-node cell).
    tightest = _tightest_enveloping_cutoff(triangulation.simplices, radii,
                                           len(coords))

    # Interpolate by *rank* rather than by radius. The circumradii are
    # heavily skewed -- a handful of slivers spanning the arbor's concavities
    # are orders of magnitude larger than the rest -- so interpolating the
    # cutoff linearly in radius leaves the shape indistinguishable from the
    # convex hull until shrink is past 0.9, and then collapses it all at
    # once. Walking the sorted radii instead spreads the family evenly
    # across [0, 1] while still hitting both documented endpoints exactly.
    order = np.sort(radii)
    last = len(order) - 1
    first = int(np.searchsorted(order, tightest, side="left"))
    cutoff = order[int(round(last + shrink * (first - last)))]
    keep = triangulation.simplices[radii <= cutoff]

    faces = _boundary_faces(keep)
    used = np.unique(faces)
    remap = np.full(len(coords), -1, dtype=int)
    remap[used] = np.arange(len(used))
    vertices = coords[used]
    surface = remap[faces]
    polygon = _order_polygon(vertices, surface) if coords.shape[1] == 2 else None
    return Boundary(
        vertices=vertices,
        faces=surface,
        volume=float(_simplex_volumes(coords, keep).sum()),
        points=coords,
        simplices=keep,
        polygon=polygon,
    )


def _simplex_volumes(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Volume (3D) or area (2D) of each simplex."""
    from math import factorial

    verts = points[simplices]
    edges = verts[:, 1:, :] - verts[:, :1, :]
    return np.abs(np.linalg.det(edges)) / factorial(points.shape[1])


def _order_polygon(vertices: np.ndarray, edges: np.ndarray):
    """Walk boundary edges into a closed ring of vertex coordinates.

    A 2D alpha shape can enclose holes or fall into several lobes, so the
    edge set is not always one cycle. The longest cycle is returned -- the
    outer ring -- which is what a point-in-polygon test wants and what
    MATLAB's ``xv``/``yv`` are.
    """
    if len(edges) == 0:
        return np.empty((0, 2))

    neighbours: dict[int, list[int]] = {}
    for a, b in edges.tolist():
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)

    seen: set[int] = set()
    best: list[int] = []
    for start in neighbours:
        if start in seen:
            continue
        ring, node, previous = [start], start, None
        seen.add(start)
        while True:
            nxt = [k for k in neighbours[node] if k != previous and k not in seen]
            if not nxt:
                break
            previous, node = node, nxt[0]
            seen.add(node)
            ring.append(node)
        if len(ring) > len(best):
            best = ring
    return vertices[np.asarray(best, dtype=int)]


def _sample_in_simplices(points: np.ndarray, simplices: np.ndarray,
                         n: int, rng) -> np.ndarray:
    """Draw ``n`` uniformly distributed points from inside a simplex fill.

    Deliberately not MATLAB's approach. `r_mc_tree` samples the bounding
    box and rejects everything outside the boundary, one batch at a time,
    which for a thin arbor throws away the large majority of every batch
    and needs a point-in-polyhedron test on each. Picking a simplex with
    probability proportional to its volume and then sampling uniformly
    inside it is exact, allocation-free, and has no rejection loop at all.
    """
    volumes = _simplex_volumes(points, simplices)
    weights = volumes / volumes.sum()
    chosen = simplices[rng.choice(len(simplices), size=n, p=weights)]

    # uniform barycentric coordinates: sort d uniforms, take the gaps
    d = points.shape[1]
    cuts = np.sort(rng.random((n, d)), axis=1)
    bary = np.diff(np.column_stack([np.zeros(n), cuts, np.ones(n)]), axis=1)
    return np.einsum("nk,nkd->nd", bary, points[chosen])


def _tightest_enveloping_cutoff(simplices: np.ndarray, radii: np.ndarray,
                                n_points: int) -> float:
    """Smallest circumradius cutoff whose simplices still cover every point.

    Adding simplices smallest-first, this is the radius at which the last
    uncovered point gets picked up -- the tightest alpha shape that still
    envelops the whole set, which is what MATLAB's shrink factor of 1 is
    documented to produce.
    """
    covered = np.zeros(n_points, dtype=bool)
    cutoff = 0.0
    for index in np.argsort(radii):
        simplex = simplices[index]
        if not covered[simplex].all():
            covered[simplex] = True
            cutoff = float(radii[index])
            if covered.all():
                break
    return cutoff


def _circumradii(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Circumscribed-sphere radius of each Delaunay simplex.

    The size of a simplex's circumsphere is what "how concave" means for an
    alpha shape: a simplex spanning a big empty gap has a big circumsphere,
    and dropping it carves that gap out of the shape.
    """
    verts = points[simplices]
    base = verts[:, 0, :]
    edges = verts[:, 1:, :] - base[:, None, :]
    # solve |p - c|^2 equal for every vertex -> linear system in the centre
    rhs = 0.5 * np.einsum("nkd,nkd->nk", edges, edges)
    try:
        offsets = np.linalg.solve(edges, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        offsets = np.stack([
            np.linalg.lstsq(e, r, rcond=None)[0] for e, r in zip(edges, rhs)
        ])
    return np.linalg.norm(offsets, axis=1)


def _boundary_faces(simplices: np.ndarray) -> np.ndarray:
    """Faces belonging to exactly one simplex -- i.e. the outer surface."""
    from itertools import combinations

    n_vertices = simplices.shape[1]
    faces = np.vstack([
        simplices[:, list(combo)]
        for combo in combinations(range(n_vertices), n_vertices - 1)
    ])
    sorted_faces = np.sort(faces, axis=1)
    _unique, index, counts = np.unique(
        sorted_faces, axis=0, return_index=True, return_counts=True
    )
    return faces[index[counts == 1]]


def convexity_tree(tree: Tree, thr: float = 25.0, nodes=None,
                   samples: int = 24, max_pairs: int = 20000,
                   dim: int = 3, rng=None) -> float:
    """How convex a tree's occupied volume is, in ``[0, 1]``.

    Takes pairs of points on the tree and asks what fraction can "see" each
    other -- i.e. the straight line between them never leaves the volume the
    cell occupies. A convex shape scores 1. A cell that wraps around
    something, or splits into lobes with a gap between them, scores lower,
    because lines between its far parts pass through empty space.

    Parameters
    ----------
    tree : Tree
    thr : float, default 25.0
        Distance [um] defining the occupied volume, as in :func:`hull_tree`.
    nodes : array_like, optional
        Points to test between. Defaults to the termination points, which
        are the extremities and so the most informative.
    samples : int, default 24
        How many points to check along each connecting line. More is
        stricter about narrow gaps.
    dim : {2, 3}, default 3
        Measure in the plane rather than in space. MATLAB has separate 2D
        and 3D branches here; this is the same computation either way.
    max_pairs : int, default 20000
        Cap on the number of pairs tested; above it, a random subset is
        used. The pair count grows quadratically, so a 500-terminal cell
        would otherwise be 125000 lines.
    rng : numpy Generator or int, optional
        Seed for the subsampling, for reproducibility.

    Returns
    -------
    float
        Fraction of visible pairs, in ``[0, 1]``.

    Notes
    -----
    **This deliberately does not reproduce MATLAB's version**, whose 3D
    branch contradicts both its own documentation and its own 2D branch in
    two independent ways -- visible directly in the source, so this is a
    confirmed defect and not merely a suspicion, even though Octave's
    missing ``boundary`` meant it could not be executed here:

    1. It wraps the terminals with ``boundary (X, Y, Z, 0)``, and shrink 0
       is documented by MathWorks as the **convex hull** -- not the
       "tightest boundary" the docstring promises, and one against which
       every segment between interior points lies inside by definition.
       The 2D branch correctly asks for shrink 1.
    2. It then returns ``c = 1 - nnz (Inds) / (nS1 * nS2)`` where ``Inds``
       flags the pairs that *did not* cross the surface, i.e. the ones
       inside. The 2D branch returns ``nnz (Inds) / (nS1 * nS2)`` from the
       same flag. One of the two has the sign inverted, and it is the 3D
       one that disagrees with the documented meaning.

    Between them, the 3D result is close to the fraction of terminal pairs
    with an endpoint on the convex hull -- a measure of how many terminals
    happen to be extremal, not of convexity. See MATLAB_TOOLBOX_BUGS.md.

    This version tests against the **space-filling hull** instead, which is
    the standard definition of convexity for a shape and the only version
    that can distinguish a compact arbor from a lobed one.
    """
    generator = np.random.default_rng(rng)
    if nodes is None:
        nodes = np.flatnonzero(T_tree(tree))
    nodes = np.asarray(nodes, dtype=int)
    if len(nodes) < 2:
        raise ValueError(
            f"need at least 2 points to measure visibility, got {len(nodes)}"
        )

    coords = np.column_stack(
        [tree.X, tree.Y] + ([tree.Z] if dim == 3 else [])
    )[nodes]
    pairs = np.array(list(_pair_indices(len(coords))))
    if len(pairs) > max_pairs:
        pairs = pairs[generator.choice(len(pairs), max_pairs, replace=False)]

    # sample points along every connecting line at once
    fractions = np.linspace(0.0, 1.0, samples)[None, :, None]
    a = coords[pairs[:, 0]][:, None, :]
    b = coords[pairs[:, 1]][:, None, :]
    probes = (a + fractions * (b - a)).reshape(-1, dim)

    starts, ends = _tree_segments(tree, dim)
    inside = (_segment_distance(probes, starts, ends) <= thr)
    inside = inside.reshape(len(pairs), samples)
    return float(inside.all(axis=1).mean())


def _pair_indices(n: int):
    """All unordered index pairs of ``n`` items."""
    from itertools import combinations

    return combinations(range(n), 2)
