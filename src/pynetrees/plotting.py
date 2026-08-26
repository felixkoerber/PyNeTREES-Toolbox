"""Visualizing trees.

Ports (part of) treestoolbox-master/graphical/*.m. **Backend decision**
(left open through Phase 6, made here): `PyVista` (VTK) is the primary 3D
engine, `matplotlib` a lightweight 2D/quick-look fallback. Reasoning:

- MATLAB's `plot_tree.m` builds cylinder geometry by hand -- per segment, a
  singular-value decomposition to find two vectors orthogonal to that
  segment's direction (its own comment calls this loop "BOTTLENECK"), then
  assembles one giant vertex/face array for a single `patch` object. It's
  clever, but the SVD-per-segment cost scales linearly with segment count
  with a high constant factor, and MATLAB's own docstring warns line-mode
  ("-2l"/"-3l") is even slower.
- PyVista/VTK's `tube()` filter does the equivalent job (turn a polyline
  into a radius-varying tube mesh) as a single, GPU-friendly, compiled
  operation -- and it accepts a per-point radius array directly, giving
  the same tapered-frustum-per-segment look as MATLAB's `frustum` mode for
  free. Measured on the bundled 2252-node/2251-segment reconstruction:
  tube-mesh generation is ~0.17s, a full off-screen render ~0.2s more --
  and the result is a single mesh VTK can pan/zoom/rotate interactively at
  native framerates, not a static image. This is the "as efficient as
  possible for complex geometries" path the port needs.
- matplotlib's `mplot3d` has no equivalent tube primitive and is not
  built for large 3D scenes (no real GPU acceleration, `Line3DCollection`
  redraws the whole scene on every interaction). It stays useful for
  quick line-only previews, notebooks without a VTK-capable display, or
  publication-style flat figures -- so `plot_tree_mpl` exists alongside
  `plot_tree`, but is documented as the lighter, less capable option, not
  a peer.

Both `pyvista` and `matplotlib` are optional extras (see `pyproject.toml`)
-- imported lazily inside each function, never at module load time, so
`import pynetrees` never requires either.

**Deliberately not ported this phase** (see PORT_STATUS.md): `hull_tree`/
`vhull_tree` (density-grid-based isosurface hulls -- need 3D binning +
marching-cubes-style extraction, substantially more machinery than the
convex hull `chull_tree` provides); `gdens_tree`/`lego_tree` (density-grid
plots, same binning dependency); `plotsect_tree` (todo list: "has no
options to begin with"); `xplore_tree` (todo list: "not yet parsed",
effectively an interactive GUI tool -- closer to Phase 10 territory);
`pov_tree`/`x3d_tree` (already reclassified into this phase's scope back
in Phase 5, but external mesh-format export is a separate concern from
in-Python visualization -- revisit if a concrete need for Blender/POV-Ray
export shows up).
"""

from __future__ import annotations

import numpy as np

from ._population import (accepts_population, is_nested_population,
                          is_population)
from ._empty import empty_safe
from .core import Tree
from .graphtheory import (
    T_tree,
    _children_lists,
    _dfs_preorder,
    idpar_tree,
    ipar_tree,
)
from typing import NamedTuple

from ._compat import resolve_dim
from .metrics import tran_tree

# ---------------------------------------------------------------------------
# core 3D rendering (PyVista)
# ---------------------------------------------------------------------------


def _require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "plot_tree (and friends) needs pyvista: pip install pynetrees[plot]"
        ) from exc
    return pv


def _blank_plotter(show: bool):
    """An empty PyVista plotter, for a tree with nothing to draw."""
    pv = _require_pyvista()
    return pv.Plotter(off_screen=not show)



def _tree_line_mesh(tree: Tree, nodes=None, offset=(0.0, 0.0, 0.0)):
    """A PyVista PolyData of the tree's segments as line cells, with
    per-point 'radius' data -- the shared geometry both tube and line
    rendering build on."""
    pv = _require_pyvista()

    N = tree.n_nodes
    idpar = idpar_tree(tree)
    nodes = np.arange(N) if nodes is None else np.asarray(nodes)
    non_root = nodes[idpar[nodes] != nodes]

    points = np.column_stack([tree.X, tree.Y, tree.Z]) + np.asarray(offset)
    lines = np.empty((len(non_root), 3), dtype=np.int64)
    lines[:, 0] = 2
    lines[:, 1] = idpar[non_root]
    lines[:, 2] = non_root

    mesh = pv.PolyData()
    mesh.points = points
    mesh.lines = lines.ravel()
    mesh.point_data["radius"] = tree.D / 2.0
    return mesh


def _resolve_color(color, n_nodes: int):
    """Work out what MATLAB's polymorphic ``color`` argument means here.

    MATLAB's ``plot_tree`` overloads one argument three ways, and this
    reproduces that rather than splitting it into ``color=``/``scalars=``
    (two parameters where one is always ``None`` is the classic sign of a
    bad split, and the merged form is what a MATLAB user types anyway).

    Returns ``(flat_color, scalars, rgb_array)`` with exactly one non-None.

    The one genuine ambiguity is a **3-node tree**, where a length-3 vector
    could be an RGB triple or three per-node values. Resolved in favour of
    RGB, matching MATLAB; pass ``scalars=`` explicitly to override.
    """
    if color is None:
        return "black", None, None
    if isinstance(color, str):
        return color, None, None

    arr = np.asarray(color, dtype=float)
    if arr.ndim == 2 and arr.shape == (n_nodes, 3):
        return None, None, arr
    if arr.ndim == 1 and arr.size == 3 and (n_nodes != 3 or arr.max() <= 1.0):
        return tuple(arr), None, None  # an RGB triple
    if arr.ndim == 1 and arr.size == n_nodes:
        return None, arr, None
    if arr.ndim == 0:
        return None, np.full(n_nodes, float(arr)), None
    raise ValueError(
        f"color must be a colour name, an RGB triple, a length-{n_nodes} "
        f"vector of values to colour-map, or an ({n_nodes}, 3) RGB array; "
        f"got shape {arr.shape}"
    )


#: Cycled when a group of trees is drawn without an explicit colour, so
#: that neighbouring cells can be told apart. Matplotlib's ``tab10``, which
#: stays legible for the dozen-or-so cells anyone overlays at once; beyond
#: that the colours repeat, and telling the cells apart is the caller's
#: problem to solve with ``offset`` instead.
POPULATION_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)


def _all_numbers(value) -> bool:
    return all(isinstance(each, (int, float, np.integer, np.floating))
               for each in value)


def _per_tree_colors(color, count):
    """One colour for the whole group, or one per tree.

    ``("red", "green", "blue")`` for three trees is three colours;
    ``(1.0, 0.0, 0.0)`` for three trees is one RGB triple. Telling them
    apart by "are the elements numbers" is exact -- an RGB triple is always
    three numbers and a list of colour specs never is.
    """
    if color is None:
        return [POPULATION_COLORS[i % len(POPULATION_COLORS)]
                for i in range(count)]
    if isinstance(color, (list, tuple)) and len(color) == count \
            and not _all_numbers(color):
        return list(color)
    return [color] * count


def _per_tree_offsets(offset, count):
    """A single ``(dx, dy, dz)``, or one per tree.

    Unambiguous either way: a single offset is three *scalars*, a per-tree
    list is ``count`` *sequences*. An ``(count, 3)`` array counts as per
    tree, so ``plot_tree(trees, offset=spread_tree(trees).offsets)`` works.
    """
    array = np.asarray(offset, dtype=float)
    if array.ndim == 2 and array.shape[0] == count:
        return [tuple(row) for row in array]
    if array.shape == (3,):
        return [tuple(array)] * count
    raise ValueError(
        f"offset must be a single (dx, dy, dz) or one per tree -- an "
        f"({count}, 3) array for {count} trees; got shape {array.shape}"
    )


def _plot_population(trees, *, color, offset, scalars, plotter, show,
                     screenshot, **kwargs):
    """Draw a whole group into **one** plotter.

    The deliberate exception to this package's list-in/list-out rule (see
    :mod:`pynetrees._population`). Returning one plotter per tree would give
    a gallery of separate windows, when the reason to hand `plot_tree` a
    group is to see the cells together; ``spread_tree`` lays them out and
    this composes them.
    """
    if is_nested_population(trees):
        trees = [tree for group in trees for tree in group]
    if len(trees) == 0:
        raise ValueError("plot_tree: empty list of trees")

    if plotter is None:
        # made here rather than by the first tree, so that `off_screen`
        # follows the real `show` even when that first tree is empty
        plotter = _require_pyvista().Plotter(off_screen=not show)

    colors = _per_tree_colors(color, len(trees))
    offsets = _per_tree_offsets(offset, len(trees))
    scalar_list = (scalars if isinstance(scalars, (list, tuple))
                   and len(scalars) == len(trees) else [scalars] * len(trees))

    for index, tree in enumerate(trees):
        # `show`/`screenshot` are held back to the last tree, so the window
        # opens once with everything in it rather than once per cell
        last = index == len(trees) - 1
        plotter = plot_tree(
            tree, color=colors[index], offset=offsets[index],
            scalars=scalar_list[index], plotter=plotter,
            show=show and last, screenshot=screenshot if last else None,
            **kwargs)
    return plotter


def plot_tree(
    tree: Tree,
    color=None,
    offset=(0.0, 0.0, 0.0),
    nodes=None,
    res: int = 8,
    *,
    mode: str = "tube",
    cmap: str = "viridis",
    categories: bool = False,
    scalars=None,
    plotter=None,
    show: bool = False,
    screenshot: str | None = None,
    **mesh_kwargs,
):
    """Render a tree in 3D with PyVista.

    Parameters
    ----------
    tree : Tree or list of Tree
        A **list of trees** (or a list of lists) is drawn into one plotter,
        which is the deliberate exception to this package's list-in/list-out
        rule -- see :func:`_plot_population`. With no ``color`` given the
        cells cycle :data:`POPULATION_COLORS` so they can be told apart.
    color : optional
        Follows MATLAB's overloading:

        - a colour name (``"black"``) or RGB triple ``(r, g, b)`` -- one
          flat colour for the whole tree;
        - a length-``n_nodes`` vector -- per-node values mapped through
          ``cmap`` (branch order, region, path length, anything);
        - an ``(n_nodes, 3)`` array -- an explicit RGB colour per node.

        Defaults to black.
    offset : tuple, default (0, 0, 0)
        Translate the rendered geometry, for laying several trees out side
        by side without moving the trees themselves. MATLAB's ``DD``. For a
        group, an ``(n_trees, 3)`` array gives one offset each, so
        ``plot_tree(trees, offset=spread_tree(trees).offsets)`` is the whole
        gallery in one call.
    nodes : array_like, optional
        Render only these nodes' segments. MATLAB's ``ipart``.
    res : int, default 8
        Number of sides on each tube. MATLAB's ``res``.
    mode : {'tube', 'line'}, keyword-only, default 'tube'
        ``'tube'`` builds one diameter-tapered tube mesh for the whole tree
        -- realistic geometry, still a single fast mesh however many
        segments. ``'line'`` skips tubing for a faster, diameter-less
        preview.
    cmap : str, keyword-only
        Colormap used when ``color`` is a value vector.
    categories : bool, keyword-only, default False
        Treat mapped values as discrete categories (e.g. region indices)
        rather than a continuous scale.
    scalars : array_like, keyword-only, optional
        Explicit per-node values, overriding any interpretation of
        ``color``. Retained for callers written against the previous
        two-argument form, and as the escape hatch for the 3-node
        ambiguity noted below.
    plotter : pyvista.Plotter, keyword-only, optional
        Draw into an existing plotter, to overlay several trees (MATLAB's
        ``hold on``). One is created if omitted, with
        ``off_screen=not show`` so headless environments work.
    show, screenshot : keyword-only
        Display the window / write a PNG.

    Returns
    -------
    pyvista.Plotter

    Notes
    -----
    **Positional order matches MATLAB** (``intree, color, DD, ipart, res``)
    as of Design Decision #54, so translated code reads the same. Everything
    this port adds beyond MATLAB's five is keyword-only, which is what keeps
    the order matched: a future addition cannot wedge itself into a
    positional slot.

    ``scalars`` must be length ``n_nodes`` -- the *whole* tree, even when
    ``nodes`` renders a subset -- since the mesh keeps every node's
    coordinates regardless (``nodes`` only selects which line cells get
    built, which is the cost that matters on large trees).

    MATLAB's ``'-b'`` (flat "blatt" patches) and ``'-2q'``/``'-3q'``
    (quiver) render modes are not reproduced: ``'-b'`` exists to dodge the
    cost of real cylinders in MATLAB's renderer, which ``mode='tube'``
    simply does not have, and quiver plots of a 4000-segment tree are
    unreadable. ``'-2l'``/``'-3l'`` map onto ``mode='line'``.
    """
    if is_population(tree) or is_nested_population(tree):
        return _plot_population(
            tree, color=color, offset=offset, nodes=nodes, res=res, mode=mode,
            cmap=cmap, categories=categories, scalars=scalars,
            plotter=plotter, show=show, screenshot=screenshot, **mesh_kwargs)

    if mode not in ("tube", "line"):
        raise ValueError(f"mode must be 'tube' or 'line', got {mode!r}")

    pv = _require_pyvista()
    if tree.n_nodes == 0:
        # nothing to draw. Hand back a usable plotter so that a loop over a
        # population containing an empty tree still composes into one scene.
        return plotter if plotter is not None else _blank_plotter(show)

    flat_color, mapped, rgb = _resolve_color(color, tree.n_nodes)
    if scalars is not None:
        mapped, flat_color, rgb = np.asarray(scalars), None, None

    mesh = _tree_line_mesh(tree, nodes=nodes, offset=offset)
    if mapped is not None:
        mapped = np.asarray(mapped)
        if mapped.shape[0] != tree.n_nodes:
            raise ValueError(
                f"per-node values must be length tree.n_nodes "
                f"({tree.n_nodes}), not {mapped.shape[0]} -- they cover the "
                f"whole tree even when `nodes` renders a subset, see "
                f"plot_tree's docstring"
            )
        mesh.point_data["scalars"] = mapped
    elif rgb is not None:
        mesh.point_data["rgb"] = rgb

    render_mesh = (
        mesh.tube(scalars="radius", absolute=True, radius=1.0, n_sides=res)
        if mode == "tube"
        else mesh
    )

    if plotter is None:
        plotter = pv.Plotter(off_screen=not show)

    if mapped is not None:
        n_colors = len(np.unique(mapped)) if categories else 256
        plotter.add_mesh(render_mesh, scalars="scalars", cmap=cmap,
                         n_colors=max(n_colors, 2), **mesh_kwargs)
    elif rgb is not None:
        plotter.add_mesh(render_mesh, scalars="rgb", rgb=True, **mesh_kwargs)
    else:
        plotter.add_mesh(render_mesh, color=flat_color, **mesh_kwargs)

    if screenshot is not None:
        plotter.show(screenshot=screenshot) if show else plotter.screenshot(screenshot)
    elif show:
        plotter.show()
    return plotter

def vtext_tree(
    plotter,
    tree: Tree,
    values=None,
    nodes=None,
    color="red",
    font_size: int = 14,
    offset=(0.0, 0.0, 0.0),
):
    """Add text labels at node positions (default: node index) to an
    existing ``plotter``. Returns the same plotter."""
    N = tree.n_nodes
    nodes = np.arange(N) if nodes is None else np.asarray(nodes)
    values = np.arange(N) if values is None else np.asarray(values)
    points = np.column_stack([tree.X, tree.Y, tree.Z])[nodes] + np.asarray(offset)
    labels = [str(v) for v in values[nodes]]
    plotter.add_point_labels(
        points, labels, text_color=color, font_size=font_size,
        shape=None, always_visible=True,
    )
    return plotter


def pointer_tree(
    plotter,
    tree: Tree,
    nodes,
    style: str = "marker",
    color="red",
    size: float = 8.0,
    offset=(0.0, 0.0, 0.0),
):
    """Mark specific nodes on an existing ``plotter`` (electrodes,
    points of interest, ...). ``style`` is ``"marker"`` (a point, fast) or
    ``"sphere"`` (a small rendered sphere, clearer at a distance but
    heavier for many points). Returns the same plotter.

    MATLAB's tapering-electrode modes (`'-l'`/`'-v'`, built from a tiny
    synthetic frustum tree) aren't ported -- niche relative to marking a
    location, which `"sphere"`/`"marker"` already cover.
    """
    pv = _require_pyvista()
    nodes = np.atleast_1d(np.asarray(nodes))
    points = np.column_stack([tree.X, tree.Y, tree.Z])[nodes] + np.asarray(offset)

    if style == "sphere":
        for p in points:
            plotter.add_mesh(pv.Sphere(radius=size, center=p), color=color)
    elif style == "marker":
        plotter.add_points(points, color=color, point_size=size, render_points_as_spheres=True)
    else:
        raise ValueError(f"style must be 'marker' or 'sphere', got {style!r}")
    return plotter


@accepts_population
def chull_tree(tree: Tree, nodes=None, plotter=None, color="black",
               opacity: float = 0.2, dim: int | None = None):
    """Convex hull around ``nodes`` (default: all).

    Parameters
    ----------
    tree : Tree
    nodes : array_like, optional
        Subset of nodes to hull. Defaults to all of them.
    plotter : pyvista.Plotter or matplotlib.axes.Axes, optional
        If given, the hull is drawn onto it -- as a translucent surface
        for a PyVista plotter (3D), or as a closed polyline for a
        matplotlib Axes (2D). The object type selects which, so 2D and 3D
        do not need two different parameters.
    color, opacity
        Appearance of the drawn hull.
    dim : {2, 3}, optional
        Default 3. ``dim=2`` hulls the XY projection, measuring enclosed
        *area* rather than volume (Design Decision #40).

    Returns ``(points, scipy.spatial.ConvexHull | None)``. The hull is
    ``None`` whenever one cannot exist, which covers two cases:

    - **too few points** (fewer than 3 in 2D / 4 in 3D), and
    - **degenerate geometry** -- points that all lie on a plane (in 3D) or a
      line (in 2D) enclose no volume, so Qhull cannot build a simplex.

    That second case is not exotic: many reconstructions are traced in 2D
    with ``Z == 0``, and :func:`~pynetrees.flatten_tree` produces a planar tree
    by construction. Returning ``None`` keeps those callable rather than
    raising a raw ``QhullError`` from deep inside SciPy. For a planar tree
    you almost certainly want the 2D hull instead -- pass ``dim=2``,
    which measures the enclosed *area*.

    If ``plotter`` is given (3D only), the hull surface is added to it.
    """
    from scipy.spatial import ConvexHull, QhullError

    dim = resolve_dim(dim)
    nodes = np.arange(tree.n_nodes) if nodes is None else np.asarray(nodes)
    pts = (
        np.column_stack([tree.X[nodes], tree.Y[nodes]])
        if dim == 2
        else np.column_stack([tree.X[nodes], tree.Y[nodes], tree.Z[nodes]])
    )
    min_points = 3 if dim == 2 else 4
    if len(pts) < min_points:
        return pts, None

    try:
        hull = ConvexHull(pts)
    except QhullError:
        # flat/collinear point set -- no hull of this dimensionality exists
        return pts, None
    if plotter is not None:
        if dim == 2:
            # A 2D hull on a 3D PyVista scene is the wrong pairing, so the
            # 2D branch draws to matplotlib instead. Previously it computed
            # the hull and then silently drew nothing at all.
            loop = np.append(hull.vertices, hull.vertices[0])
            plotter.plot(pts[loop, 0], pts[loop, 1], color=color)
            plotter.fill(pts[loop, 0], pts[loop, 1], color=color, alpha=opacity)
        else:
            pv = _require_pyvista()
            faces = np.hstack([[3, *simplex] for simplex in hull.simplices])
            mesh = pv.PolyData(pts, faces)
            plotter.add_mesh(mesh, color=color, opacity=opacity)
    return pts, hull


# ---------------------------------------------------------------------------
# matplotlib fallback: quick, line-only, no extra dependency beyond the
# `plot` extra's other member
# ---------------------------------------------------------------------------


@accepts_population
@empty_safe("none")
def plot_mpl_tree(tree: Tree, ax=None, color="black", scalars=None, cmap: str = "viridis", linewidth: float = 1.0, nodes=None):
    """Quick line-only 3D render via matplotlib (no diameter, no GPU
    acceleration -- see this module's docstring for why `plot_tree`
    (PyVista) is the recommended path for anything but a fast preview).
    Fixes matplotlib's well-known default 3D aspect-ratio distortion via
    `set_box_aspect`, so anatomy isn't visually stretched. Returns the Axes.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    N = tree.n_nodes
    idpar = idpar_tree(tree)
    nodes = np.arange(N) if nodes is None else np.asarray(nodes)
    non_root = nodes[idpar[nodes] != nodes]

    starts = np.column_stack([tree.X[idpar[non_root]], tree.Y[idpar[non_root]], tree.Z[idpar[non_root]]])
    ends = np.column_stack([tree.X[non_root], tree.Y[non_root], tree.Z[non_root]])
    segments = np.stack([starts, ends], axis=1)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    lc = Line3DCollection(segments, linewidths=linewidth)
    if scalars is not None:
        lc.set_array(np.asarray(scalars)[non_root])
        lc.set_cmap(cmap)
    else:
        lc.set_color(color)
    ax.add_collection3d(lc)

    ax.set_xlim(tree.X.min(), tree.X.max())
    ax.set_ylim(tree.Y.min(), tree.Y.max())
    ax.set_zlim(tree.Z.min(), tree.Z.max())
    spans = [max(np.ptp(tree.X), 1e-9), max(np.ptp(tree.Y), 1e-9), max(np.ptp(tree.Z), 1e-9)]
    ax.set_box_aspect(spans)
    return ax


@accepts_population
@empty_safe("none")
def dA_tree(tree: Tree, ax=None):
    """Display a tree's adjacency matrix as a sparsity image (matplotlib
    `spy`) -- a quick structural/debugging view, not anatomy."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    ax.spy(tree.dA, markersize=1)
    ax.set_xlabel("parent")
    ax.set_ylabel("child")
    return ax


# ---------------------------------------------------------------------------
# dendrogram
# ---------------------------------------------------------------------------


@accepts_population
@empty_safe("nodes")
def xdend_tree(tree: Tree):
    """X-coordinate for each node useful for a dendrogram layout: each
    node's position is the midpoint of its leftmost and rightmost
    descendant terminal's rank (terminals ranked left-to-right in node
    order). Returns ``xdend`` (length ``n_nodes``).

    Reimplemented as an O(n_nodes) bottom-up tree accumulation (post-order:
    every node's [min, max] leaf-rank range is its children's ranges
    combined) instead of MATLAB's `ipar`-matrix sort-and-diff trick --
    same result, standard "assign each internal dendrogram node the
    average of its leaves' positions" algorithm, and avoids an O(n^2)
    blowup from testing "is this terminal a descendant of that node" for
    every (node, terminal) pair.
    """
    N = tree.n_nodes
    is_terminal = T_tree(tree)
    terminal_rank = np.full(N, -1)
    terminal_rank[np.flatnonzero(is_terminal)] = np.arange(int(is_terminal.sum()))

    children = _children_lists(tree.dA)
    order = _dfs_preorder(tree.dA)
    lo, hi = np.zeros(N), np.zeros(N)
    for node in reversed(order.tolist()):
        kids = children[node]
        if not kids:
            lo[node] = hi[node] = terminal_rank[node]
        else:
            lo[node] = min(lo[c] for c in kids)
            hi[node] = max(hi[c] for c in kids)
    return (lo + hi) / 2.0


@accepts_population
@empty_safe("none")
def dendrogram_tree(tree: Tree, yvec=None, ax=None, color="black", linewidth: float = 1.0):
    """A 2D dendrogram: each node at ``(xdend_tree(tree), yvec)`` (default
    ``yvec``: path length from the root), connected to its parent by an
    L-shaped (horizontal-then-vertical) line, the standard dendrogram
    convention. Rendered with matplotlib -- an abstract topological
    diagram, not spatial anatomy, so PyVista's 3D machinery isn't the
    right tool here.
    """
    import matplotlib.pyplot as plt

    from .graphtheory import Pvec_tree
    from .metrics import len_tree

    if yvec is None:
        yvec = Pvec_tree(tree, len_tree(tree))
    xdend = xdend_tree(tree)
    idpar = idpar_tree(tree)

    if ax is None:
        _, ax = plt.subplots()

    N = tree.n_nodes
    non_root = np.flatnonzero(idpar != np.arange(N))
    px, py = xdend[idpar[non_root]], yvec[idpar[non_root]]
    cx, cy = xdend[non_root], yvec[non_root]
    # L-shaped: horizontal at the parent's Y, then vertical to the child
    xs = np.stack([px, cx, cx], axis=1)
    ys = np.stack([py, py, cy], axis=1)
    for x, y in zip(xs, ys):
        ax.plot(x, y, color=color, linewidth=linewidth)

    ax.invert_yaxis()
    ax.set_xlabel("dendrogram position")
    ax.set_ylabel("path length from root [um]")
    return ax


# ---------------------------------------------------------------------------
# multi-tree layout
# ---------------------------------------------------------------------------


class SpreadResult(NamedTuple):
    """Result of :func:`spread_tree`: the laid-out trees and how far each
    one moved."""

    trees: list[Tree]
    """Translated copies, ready to plot."""
    offsets: list[tuple[float, float, float]]
    """The ``(dx, dy, dz)`` applied to each -- pass to ``plot_tree``'s
    ``offset=`` to move something else the same way, e.g. a second
    population that must line up with this one."""


def spread_tree(trees: list[Tree], dx: float = 50.0, dy: float = 50.0
                ) -> SpreadResult:
    """Lay trees out on a roughly square grid so they can be shown together
    without overlapping.

    Returns
    -------
    SpreadResult
        ``(trees, offsets)`` -- the translated copies *and* the offsets
        applied. MATLAB has two functions here, `spread_tree` returning the
        offsets and `spread_trees` returning the trees; they differ only in
        return type and one is a wrapper around the other, so this is one
        function returning both (REVIEW_PLAN P7).

    Notes
    -----
    Reimplemented as a straightforward greedy row-packing bin layout
    (accumulate widths along a row until a target row width is exceeded,
    then wrap) instead of MATLAB's `cumsum`/`mod` index arithmetic -- same
    "roughly square, no-overlap" goal, which is an aesthetic choice rather
    than a uniquely-determined one, and much easier to follow.
    """
    widths = np.array([t.X.max() - t.X.min() for t in trees])
    heights = np.array([t.Y.max() - t.Y.min() for t in trees])
    xmins = np.array([t.X.min() for t in trees])
    ymaxs = np.array([t.Y.max() for t in trees])
    zmins = np.array([t.Z.min() for t in trees])

    target_row_width = (widths + dx).sum() / np.sqrt(len(trees))

    offsets = []
    cursor_x, row_y, row_height = 0.0, 0.0, 0.0
    for i in range(len(trees)):
        w = widths[i] + dx
        if cursor_x > 0 and cursor_x + w > target_row_width:
            row_y -= row_height + dy
            cursor_x, row_height = 0.0, 0.0
        offsets.append((cursor_x - xmins[i], row_y - ymaxs[i], -zmins[i]))
        cursor_x += w
        row_height = max(row_height, heights[i])
    return SpreadResult(
        trees=[tran_tree(t, list(off)) for t, off in zip(trees, offsets)],
        offsets=offsets,
    )




# ---------------------------------------------------------------------------
# plotsect_tree / xplore_tree  (B5)
# ---------------------------------------------------------------------------


@accepts_population
def plotsect_tree(tree: Tree, sect, color="black", offset=(0.0, 0.0, 0.0),
                  ipar=None, ax=None, linewidth: float = 2.0, *,
                  full_output: bool = False):
    """Draw the path from one node down to another.

    Parameters
    ----------
    tree : Tree
    sect : (start, end)
        Two node indices. ``start`` must be an **ancestor** of ``end``:
        the path is read off the tree's own parent chain, not searched
        for, so it always runs away from the root.
    color : matplotlib color, default "black"
    offset : (dx, dy, dz), default (0, 0, 0)
        Shift the drawn path, for overlaying several trees.
    ipar : np.ndarray, optional
        A precomputed :func:`~pynetrees.ipar_tree`. Worth passing when
        drawing many sections of the same tree -- MATLAB's docstring calls
        computing it "the slow part of this function".
    ax : matplotlib 3D Axes, optional
    linewidth : float, default 2.0
    full_output : bool, default False
        Also return the node indices along the path.

    Returns
    -------
    Axes, or (Axes, indices)
    """
    import matplotlib.pyplot as plt

    start, end = int(sect[0]), int(sect[1])
    ipar = ipar_tree(tree) if ipar is None else ipar

    chain = ipar[end]
    hit = np.flatnonzero(chain == start)
    if len(hit) == 0:
        raise ValueError(
            f"node {start} is not an ancestor of node {end}, so there is no "
            "path down from it; plotsect_tree draws directed paths only"
        )
    indices = chain[: hit[0] + 1]

    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")
    ax.plot(tree.X[indices] + offset[0],
            tree.Y[indices] + offset[1],
            tree.Z[indices] + offset[2],
            color=color, linewidth=linewidth)
    return (ax, indices) if full_output else ax


@accepts_population
@empty_safe("none")
def xplore_tree(tree: Tree, mode: str = "nodes", color="black",
                offset=(0.0, 0.0, 0.0), fig=None):
    """Diagnostic views of a tree, for looking at one rather than
    presenting it.

    Parameters
    ----------
    tree : Tree
    mode : {'nodes', 'regions', 'projections'}, default 'nodes'
        ``'nodes'`` draws the arbor with every node's index written on it,
        which is what you want when a function has just told you something
        about node 412. ``'regions'`` colours by region and labels each at
        its centre of mass. ``'projections'`` shows the xy, yz and xz views
        stacked, so a cell's depth is visible without rotating anything.
        MATLAB spells these ``'-1'``, ``'-2'`` and ``'-3'``.
    color : matplotlib color, default "black"
    offset : (dx, dy, dz), default (0, 0, 0)
    fig : matplotlib Figure, optional

    Returns
    -------
    Figure

    Notes
    -----
    MATLAB's ``'-2'`` labels each region with ``tree.rnames{counter}``,
    indexing by the *loop* counter rather than by the region value
    ``uR (counter)`` it is labelling. On a tree whose regions are not
    ``1 : n`` -- which any tree that has had a region deleted is -- the
    labels are attached to the wrong regions. Fixed here.

    MATLAB's arrow overlay (``plot_tree (..., '-3q')``, a quiver per
    segment showing which way is away from the root) is not reproduced:
    matplotlib's 3D quiver draws one cone per arrow and is unusable past a
    few hundred segments. The node indices already carry the direction,
    since they increase away from the root in a sorted tree.
    """
    import matplotlib.pyplot as plt

    if mode not in ("nodes", "regions", "projections"):
        raise ValueError(
            f"mode must be 'nodes', 'regions' or 'projections', got {mode!r}"
        )
    shifted = tran_tree(tree, list(offset))
    fig = plt.figure() if fig is None else fig

    if mode == "projections":
        views = (("x", "y", (90, -90)), ("y", "z", (0, 0)), ("x", "z", (0, -90)))
        for index, (xlabel, ylabel, (elev, azim)) in enumerate(views):
            ax = fig.add_subplot(3, 1, index + 1, projection="3d")
            plot_mpl_tree(shifted, ax=ax, color=color, linewidth=1.0)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        return fig

    ax = fig.add_subplot(projection="3d")
    if mode == "nodes":
        plot_mpl_tree(shifted, ax=ax, color=color)
        for node in range(shifted.n_nodes):
            ax.text(shifted.X[node], shifted.Y[node], shifted.Z[node],
                    str(node), fontsize=7, color="red")
        return fig

    regions = np.asarray(shifted.R)
    plot_mpl_tree(shifted, ax=ax, scalars=regions, cmap="tab10")
    for value in np.unique(regions).tolist():
        members = regions == value
        # index rnames by the region's own value, not by loop position
        label = (shifted.rnames[value] if value < len(shifted.rnames)
                 else str(value))
        ax.text(shifted.X[members].mean(), shifted.Y[members].mean(),
                shifted.Z[members].mean(), label, color="red", fontsize=14)
    return fig
