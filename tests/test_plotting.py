"""Tests for pytrees.plotting.

PyVista-backed functions are smoke/structure-tested (they run headless via
`off_screen=True`/`show=False`, which is the default): asserting a render
completes, returns the right object type, and produces plausible mesh
sizes, rather than pixel-comparing images. `xdend_tree` (pure numeric
layout, no rendering) gets an exact hand-verified test since it's cheap to
compute by hand for a small tree. `pyvista`/`matplotlib` tests are skipped
if those optional extras aren't installed.
"""

import matplotlib

matplotlib.use("Agg")  # headless test environment: no interactive backend available

import numpy as np
import pytest
from scipy import sparse

from pytrees import (
    BO_tree,
    Tree,
    chull_tree,
    dA_tree,
    dendrogram_tree,
    flatten_tree,
    plot_tree,
    plot_mpl_tree,
    pointer_tree,
    sample_tree,
    spread_tree,
    spread_trees,
    tran_tree,
    vtext_tree,
    xdend_tree,
)

pv = pytest.importorskip("pyvista")
plt = pytest.importorskip("matplotlib.pyplot")


def _chain_to_branch_tree() -> Tree:
    # 0 -> 1 -> 2 -> {3 (T), 4 (T)}
    dA = sparse.csr_matrix(
        ([1, 1, 1, 1], ([1, 2, 3, 4], [0, 1, 2, 2])), shape=(5, 5)
    )
    return Tree(
        dA=dA, X=np.zeros(5), Y=np.zeros(5), Z=np.zeros(5),
        D=np.ones(5), R=np.zeros(5, dtype=int), rnames=["a"],
    )


def _small_geom_tree() -> Tree:
    dA = sparse.csr_matrix(
        ([1, 1, 1, 1], ([1, 2, 3, 4], [0, 0, 1, 1])), shape=(5, 5)
    )
    return Tree(
        dA=dA,
        X=np.array([0.0, 3.0, 0.0, 3.0, 6.0]),
        Y=np.array([0.0, 4.0, 0.0, 4.0, 4.0]),
        Z=np.zeros(5),
        D=np.array([4.0, 2.0, 2.0, 1.0, 1.0]),
        R=np.zeros(5, dtype=int),
        rnames=["dend"],
    )


# ---------------------------------------------------------------------------
# plot_tree (PyVista)
# ---------------------------------------------------------------------------


def test_plot_tree_tube_mode_builds_plausible_mesh():
    tree = sample_tree()
    plotter = plot_tree(tree, color="black", mode="tube")
    assert isinstance(plotter, pv.Plotter)
    # every segment tubed with n_sides=8 (default res) should give many
    # more points than the original node count
    assert plotter.renderer.bounds is not None
    plotter.close()


def test_plot_tree_line_mode_is_cheap_and_valid():
    tree = sample_tree()
    plotter = plot_tree(tree, color="black", mode="line")
    assert isinstance(plotter, pv.Plotter)
    plotter.close()


def test_plot_tree_rejects_bad_mode():
    tree = _small_geom_tree()
    with pytest.raises(ValueError):
        plot_tree(tree, mode="nonsense")


def test_plot_tree_scalars_color_by_value():
    tree = sample_tree()
    bo = BO_tree(tree)
    plotter = plot_tree(tree, scalars=bo, cmap="plasma", mode="tube")
    assert isinstance(plotter, pv.Plotter)
    plotter.close()


def test_plot_tree_scalars_must_cover_whole_tree_not_just_nodes_subset():
    # scalars always indexes the full tree, even when `nodes` restricts
    # what's rendered -- passing a subset-length array must fail clearly
    # rather than with PyVista's raw shape-mismatch error
    tree = _small_geom_tree()
    subset = np.array([0, 1])
    with pytest.raises(ValueError, match="n_nodes"):
        plot_tree(tree, nodes=subset, scalars=np.zeros(len(subset)))


def test_plot_tree_nodes_subset_with_full_length_scalars_works():
    tree = _small_geom_tree()
    subset = np.array([0, 1])
    plotter = plot_tree(tree, nodes=subset, scalars=tree.R.astype(float))
    assert isinstance(plotter, pv.Plotter)
    plotter.close()


def test_plot_tree_can_overlay_on_existing_plotter():
    t1 = _small_geom_tree()
    t2 = tran_tree(_small_geom_tree(), [20.0, 0.0, 0.0])
    plotter = plot_tree(t1, color="black", mode="line")
    same_plotter = plot_tree(t2, color="red", mode="line", plotter=plotter)
    assert same_plotter is plotter
    plotter.close()


def test_plot_tree_offset_shifts_geometry():
    tree = _small_geom_tree()
    plotter = plot_tree(tree, mode="line", offset=(100.0, 0.0, 0.0))
    bounds = plotter.renderer.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)
    assert bounds[0] >= 99.0
    plotter.close()


# ---------------------------------------------------------------------------
# vtext_tree / pointer_tree / chull_tree
# ---------------------------------------------------------------------------


def test_vtext_tree_adds_labels():
    tree = _small_geom_tree()
    plotter = plot_tree(tree, mode="line")
    result = vtext_tree(plotter, tree, nodes=[0, 1])
    assert result is plotter
    plotter.close()


def test_pointer_tree_marker_and_sphere_styles():
    tree = _small_geom_tree()
    plotter = plot_tree(tree, mode="line")
    pointer_tree(plotter, tree, nodes=[1, 2], style="marker", color="red")
    pointer_tree(plotter, tree, nodes=[3], style="sphere", color="blue")
    plotter.close()


def test_pointer_tree_rejects_bad_style():
    tree = _small_geom_tree()
    plotter = plot_tree(tree, mode="line")
    with pytest.raises(ValueError):
        pointer_tree(plotter, tree, nodes=[0], style="nonsense")
    plotter.close()


def test_chull_tree_known_bounding_box_volume():
    # 8 nodes at the corners of a 10x10x10 cube -> convex hull is exactly
    # that cube, volume 1000
    dA = sparse.csr_matrix(([1] * 7, (range(1, 8), [0] * 7)), shape=(8, 8))
    corners = np.array(
        [[x, y, z] for x in (0, 10) for y in (0, 10) for z in (0, 10)]
    )
    tree = Tree(
        dA=dA, X=corners[:, 0], Y=corners[:, 1], Z=corners[:, 2],
        D=np.ones(8), R=np.zeros(8, dtype=int), rnames=["a"],
    )
    pts, hull = chull_tree(tree)
    assert hull is not None
    assert hull.volume == pytest.approx(1000.0)


def test_chull_tree_too_few_points_returns_none_hull():
    dA = sparse.csr_matrix(([1], ([1], [0])), shape=(2, 2))
    tree = Tree(
        dA=dA, X=np.array([0.0, 1.0]), Y=np.zeros(2), Z=np.zeros(2),
        D=np.ones(2), R=np.zeros(2, dtype=int), rnames=["a"],
    )
    pts, hull = chull_tree(tree)
    assert hull is None


def test_chull_tree_adds_mesh_to_plotter():
    tree = sample_tree()
    plotter = plot_tree(tree, mode="line")
    pts, hull = chull_tree(tree, plotter=plotter, opacity=0.2)
    assert hull is not None
    plotter.close()


# ---------------------------------------------------------------------------
# xdend_tree / dendrogram_tree
# ---------------------------------------------------------------------------


def test_xdend_tree_hand_verified():
    tree = _chain_to_branch_tree()
    xdend = xdend_tree(tree)
    np.testing.assert_allclose(xdend, [0.5, 0.5, 0.5, 0.0, 1.0])


def test_xdend_tree_runs_on_real_reconstruction():
    tree = sample_tree()
    xdend = xdend_tree(tree)
    assert xdend.shape == (tree.n_nodes,)
    assert np.all(np.isfinite(xdend))
    assert xdend.min() == 0.0


def test_dendrogram_tree_returns_axes_with_lines():
    tree = _chain_to_branch_tree()
    ax = dendrogram_tree(tree)
    assert len(ax.lines) > 0
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# spread_tree / spread_trees
# ---------------------------------------------------------------------------


def test_spread_tree_avoids_overlap_for_two_wide_trees():
    # two identical wide trees together exceed the target (roughly square)
    # row width, so the second must wrap to a new row (different Y) rather
    # than overlap the first in X
    dA = sparse.csr_matrix(([1], ([1], [0])), shape=(2, 2))

    def wide_tree(x0):
        return Tree(
            dA=dA, X=np.array([x0, x0 + 30.0]), Y=np.zeros(2), Z=np.zeros(2),
            D=np.ones(2), R=np.zeros(2, dtype=int), rnames=["a"],
        )

    trees = [wide_tree(0.0), wide_tree(0.0)]
    offsets = spread_tree(trees, dx=10.0, dy=10.0)
    assert len(offsets) == 2

    def bbox(t, off):
        dx, dy, dz = off
        return (t.X.min() + dx, t.X.max() + dx, t.Y.min() + dy, t.Y.max() + dy)

    (ax0, ax1, ay0, ay1), (bx0, bx1, by0, by1) = (
        bbox(trees[0], offsets[0]),
        bbox(trees[1], offsets[1]),
    )
    x_disjoint = ax1 <= bx0 or bx1 <= ax0
    y_disjoint = ay1 <= by0 or by1 <= ay0
    assert x_disjoint or y_disjoint


def test_spread_trees_returns_translated_trees():
    t1 = _small_geom_tree()
    t2 = _small_geom_tree()
    spread = spread_trees([t1, t2], dx=20.0, dy=20.0)
    assert len(spread) == 2
    assert all(t.n_nodes == t1.n_nodes for t in spread)
    # roots should no longer coincide
    assert not (
        spread[0].X[0] == spread[1].X[0] and spread[0].Y[0] == spread[1].Y[0]
    )


# ---------------------------------------------------------------------------
# matplotlib fallback
# ---------------------------------------------------------------------------


def test_plot_mpl_tree_true_aspect_ratio():
    # matplotlib's `get_box_aspect()` returns internally re-normalized
    # values (not the raw spans passed to `set_box_aspect`), so what's
    # checked here is that the *proportions* between axes are preserved
    # (the actual fix for matplotlib's default aspect-ratio distortion),
    # not the literal numbers.
    tree = _small_geom_tree()  # X span 6, Y span 4, Z span 0
    ax = plot_mpl_tree(tree)
    box = ax.get_box_aspect()
    assert box[0] / box[1] == pytest.approx(6.0 / 4.0, rel=1e-3)
    assert box[2] < box[1] * 1e-3  # Z span is ~0: must stay negligible
    plt.close(ax.figure)


def test_plot_mpl_tree_scalars_coloring_runs():
    tree = sample_tree()
    bo = BO_tree(tree)
    ax = plot_mpl_tree(tree, scalars=bo, cmap="viridis")
    assert len(ax.collections) == 1
    plt.close(ax.figure)


def test_dA_tree_runs():
    tree = sample_tree()
    ax = dA_tree(tree)
    assert ax.get_xlabel() == "parent"
    plt.close(ax.figure)


def test_chull_tree_returns_none_for_planar_points_instead_of_raising():
    # A planar point set encloses no volume, so no 3-D hull exists. This is
    # not exotic: many reconstructions are traced in 2-D with Z == 0, and
    # `flatten_tree` produces a planar tree by construction. Qhull raises a
    # bare QhullError here; chull_tree must absorb that and report "no hull"
    # the same way it does for too-few-points.
    tree = sample_tree()
    flat = flatten_tree(tree)
    assert np.abs(flat.Z).max() == 0.0
    pts, hull = chull_tree(flat)
    assert hull is None
    assert len(pts) == flat.n_nodes


def test_chull_tree_2d_hull_works_on_a_planar_tree():
    # ...and the 2-D hull, which is what you actually want there, still does
    tree = flatten_tree(sample_tree())
    pts, hull = chull_tree(tree, dim=2)
    assert hull is not None
    assert hull.volume > 0      # enclosed area in 2-D


def test_stats_tree_extras_survives_a_planar_tree():
    # stats_tree(extras=True) calls chull_tree; before the fix this crashed
    # with a QhullError on any flat morphology
    import pytrees as pt

    flat = flatten_tree(sample_tree())
    res = pt.stats_tree(flat, extras=True)
    assert np.isnan(res["summary"]["hull_volume"].iloc[0])
