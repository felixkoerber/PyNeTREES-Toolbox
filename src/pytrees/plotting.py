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
`import pytrees` never requires either.

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

from .core import Tree
from .graphtheory import (
    T_tree,
    _children_lists,
    _dfs_preorder,
    idpar_tree,
)
from .metrics import tran_tree

# ---------------------------------------------------------------------------
# core 3D rendering (PyVista)
# ---------------------------------------------------------------------------


def _require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "plot_tree (and friends) needs pyvista: pip install pytrees[plot]"
        ) from exc
    return pv


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


def plot_tree(
    tree: Tree,
    color="black",
    scalars=None,
    cmap: str = "viridis",
    mode: str = "tube",
    res: int = 8,
    nodes=None,
    offset=(0.0, 0.0, 0.0),
    plotter=None,
    show: bool = False,
    screenshot: str | None = None,
    **mesh_kwargs,
):
    """Render a tree in 3D with PyVista.

    ``mode="tube"`` (default) builds one diameter-tapered tube mesh for the
    whole tree -- realistic dendrite geometry, still a single fast mesh
    regardless of segment count. ``mode="line"`` skips tubing for a
    faster, diameter-less preview (or genuinely huge trees where even tube
    generation is unwanted).

    ``scalars``, if given, colors the tree by value (e.g. branch order,
    region, any per-node metric) via ``cmap`` instead of the flat
    ``color``. It must always be length ``tree.n_nodes`` -- the *whole*
    tree, not just ``nodes`` -- since the underlying mesh keeps every
    node's coordinates regardless of which are actually rendered (`nodes`
    only selects which line cells/tube segments get built, which is where
    the cost that matters for large trees actually is; holding every
    node's XYZ is negligible even for huge reconstructions). Index
    ``scalars`` yourself only if you deliberately want ``nodes``-subset
    values to double as a *different* per-node quantity than the full
    tree's.

    Pass an existing ``plotter`` to overlay several trees (mirrors
    MATLAB's `hold on`); one is created if omitted, using
    ``off_screen=not show`` for headless environments. Returns the
    ``pyvista.Plotter``.
    """
    if mode not in ("tube", "line"):
        raise ValueError(f"mode must be 'tube' or 'line', got {mode!r}")

    pv = _require_pyvista()
    mesh = _tree_line_mesh(tree, nodes=nodes, offset=offset)
    if scalars is not None:
        scalars = np.asarray(scalars)
        if scalars.shape[0] != tree.n_nodes:
            raise ValueError(
                f"scalars must be length tree.n_nodes ({tree.n_nodes}), not "
                f"{scalars.shape[0]} -- it covers the whole tree even when "
                "`nodes` selects a subset to render, see plot_tree's docstring"
            )
        mesh.point_data["scalars"] = scalars

    render_mesh = (
        mesh.tube(scalars="radius", absolute=True, radius=1.0, n_sides=res)
        if mode == "tube"
        else mesh
    )

    if plotter is None:
        plotter = pv.Plotter(off_screen=not show)

    if scalars is not None:
        plotter.add_mesh(render_mesh, scalars="scalars", cmap=cmap, **mesh_kwargs)
    else:
        plotter.add_mesh(render_mesh, color=color, **mesh_kwargs)

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


def chull_tree(tree: Tree, nodes=None, plotter=None, color="black", opacity: float = 0.2, dim2: bool = False):
    """Convex hull around ``nodes`` (default: all).

    Returns ``(points, scipy.spatial.ConvexHull | None)``. The hull is
    ``None`` whenever one cannot exist, which covers two cases:

    - **too few points** (fewer than 3 in 2D / 4 in 3D), and
    - **degenerate geometry** -- points that all lie on a plane (in 3D) or a
      line (in 2D) enclose no volume, so Qhull cannot build a simplex.

    That second case is not exotic: many reconstructions are traced in 2D
    with ``Z == 0``, and :func:`~pytrees.flatten_tree` produces a planar tree
    by construction. Returning ``None`` keeps those callable rather than
    raising a raw ``QhullError`` from deep inside SciPy. For a planar tree
    you almost certainly want the 2D hull instead -- pass ``dim2=True``,
    which measures the enclosed *area*.

    If ``plotter`` is given (3D only), the hull surface is added to it.
    """
    from scipy.spatial import ConvexHull, QhullError

    nodes = np.arange(tree.n_nodes) if nodes is None else np.asarray(nodes)
    pts = (
        np.column_stack([tree.X[nodes], tree.Y[nodes]])
        if dim2
        else np.column_stack([tree.X[nodes], tree.Y[nodes], tree.Z[nodes]])
    )
    min_points = 3 if dim2 else 4
    if len(pts) < min_points:
        return pts, None

    try:
        hull = ConvexHull(pts)
    except QhullError:
        # flat/collinear point set -- no hull of this dimensionality exists
        return pts, None
    if plotter is not None and not dim2:
        pv = _require_pyvista()
        faces = np.hstack([[3, *simplex] for simplex in hull.simplices])
        mesh = pv.PolyData(pts, faces)
        plotter.add_mesh(mesh, color=color, opacity=opacity)
    return pts, hull


# ---------------------------------------------------------------------------
# matplotlib fallback: quick, line-only, no extra dependency beyond the
# `plot` extra's other member
# ---------------------------------------------------------------------------


def plot_tree_mpl(tree: Tree, ax=None, color="black", scalars=None, cmap: str = "viridis", linewidth: float = 1.0, nodes=None):
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


def dA_tree_mpl(tree: Tree, ax=None):
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


def spread_tree(trees: list[Tree], dx: float = 50.0, dy: float = 50.0):
    """Offsets (one ``(dx, dy, dz)`` triple per tree) that lay ``trees``
    out on a roughly square grid by bounding-box, for combined display
    without overlap (e.g. ``plot_tree(t, offset=off)`` per tree).

    Reimplemented as a straightforward greedy row-packing bin layout
    (accumulate widths along a row until a target row width is exceeded,
    then wrap) instead of MATLAB's `cumsum`/`mod` index arithmetic -- same
    "roughly square, no-overlap" goal (an aesthetic layout choice, not a
    uniquely-determined one), much easier to follow. Apply the offsets
    with :func:`~pytrees.tran_tree` (or ``plot_tree``'s ``offset=``) to
    actually move each tree.
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
    return offsets


def spread_trees(trees: list[Tree], dx: float = 50.0, dy: float = 50.0) -> list[Tree]:
    """Like :func:`spread_tree`, but directly returns translated copies of
    ``trees`` (via :func:`~pytrees.tran_tree`) instead of raw offsets."""
    offsets = spread_tree(trees, dx=dx, dy=dy)
    return [tran_tree(t, list(off)) for t, off in zip(trees, offsets)]
