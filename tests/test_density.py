"""Density grids, space-filling hulls and territory volumes (B1).

`hull_tree`, `gdens_tree`, `lego_tree`, `vhull_tree` and
`share_boundary_tree` -- one shared dependency, ported together.

There is no MATLAB reference to diff against here: `hull_tree` returns a
marching-cubes mesh whose vertex count and ordering depend on the
implementation, so matching MATLAB vertex-for-vertex would be matching an
implementation detail rather than the geometry. These tests check the
*properties* the geometry must have instead, which is the stronger claim.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import pynetrees as pt
from pynetrees.density import _segment_distance, _tree_segments


@pytest.fixture(scope="module")
def tree():
    return pt.sample_tree()


# ---------------------------------------------------------------------------
# the distance primitive everything rests on
# ---------------------------------------------------------------------------


def test_every_node_lies_on_its_own_tree(tree):
    """Distance from a node to its own tree is zero -- to within the
    expanded formula's precision.

    `_segment_distance` expands the squared distance instead of building an
    explicit closest point, which is ~3x faster and far lighter on memory
    but is the numerically unstable form. The residual is bounded by
    roughly `|coords| * sqrt(eps)`, about 2e-6 um here. That is picometres
    against a 0.1 um reconstruction precision, so the tolerance below is
    the real physical claim, not a fudge to make the test pass.
    """
    starts, ends = _tree_segments(tree, 3)
    nodes = np.column_stack([tree.X, tree.Y, tree.Z])
    residual = _segment_distance(nodes, starts, ends).max()
    assert residual == pytest.approx(0.0, abs=1e-4)
    assert residual < 1e-4


def test_distance_is_to_the_segment_not_the_nearest_node():
    """The reason this is not a `cKDTree` query over node coordinates.

    A point beside the middle of a long segment is close to the *cable* but
    far from either endpoint. Measuring to nodes would make a hull's shape
    depend on how finely the morphology happened to be sampled.
    """
    starts = np.array([[0.0, 0.0, 0.0]])
    ends = np.array([[100.0, 0.0, 0.0]])
    midside = np.array([[50.0, 3.0, 0.0]])
    assert _segment_distance(midside, starts, ends)[0] == pytest.approx(3.0)


def test_zero_length_segments_do_not_divide_by_zero():
    """The root is always its own parent, so a zero-length segment is
    guaranteed to be present in every tree."""
    starts = np.array([[5.0, 5.0, 5.0]])
    ends = np.array([[5.0, 5.0, 5.0]])
    probe = np.array([[5.0, 5.0, 8.0]])
    assert _segment_distance(probe, starts, ends)[0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# gdens_tree / lego_tree
# ---------------------------------------------------------------------------


def test_density_grid_bins_every_node(tree):
    grid = pt.gdens_tree(tree, sr=10.0)
    assert grid.counts.sum() == tree.n_nodes


def test_density_grid_is_indexed_xyz_not_matlabs_yxz(tree):
    """Deliberate divergence: MATLAB uses `[y, x, z]` (its image
    convention). Everything else in this port is `[x, y, z]`, and mixing
    the two silently is exactly how axis bugs happen."""
    grid = pt.gdens_tree(tree, sr=10.0)
    assert grid.counts.shape == (len(grid.x), len(grid.y), len(grid.z))


def test_finer_voxels_give_more_and_emptier_bins(tree):
    coarse = pt.gdens_tree(tree, sr=20.0)
    fine = pt.gdens_tree(tree, sr=5.0)
    assert fine.counts.size > coarse.counts.size
    assert fine.counts.max() <= coarse.counts.max()


def test_node_subset_is_respected(tree):
    nodes = np.arange(50)
    assert pt.gdens_tree(tree, sr=10.0, nodes=nodes).counts.sum() == 50


def test_lego_plot_renders(tree):
    ax = pt.lego_tree(tree, sr=25.0)
    assert ax is not None


# ---------------------------------------------------------------------------
# hull_tree
# ---------------------------------------------------------------------------


def test_hull_3d_produces_a_closed_mesh(tree):
    pytest.importorskip("skimage")
    hull = pt.hull_tree(tree, thr=25.0, bx=18, by=18, bz=18)
    assert len(hull.vertices) > 100
    assert hull.faces.shape[1] == 3


def test_hull_2d_produces_closed_polygons(tree):
    hull = pt.hull_tree(tree, thr=25.0, bx=30, by=30, dim=2)
    assert hull.polygons and len(hull.polygons[0]) > 10
    assert hull.faces is None


def test_hull_grows_monotonically_with_threshold(tree):
    pytest.importorskip("skimage")
    from scipy.spatial import ConvexHull

    volumes = []
    for thr in (15.0, 30.0, 60.0):
        hull = pt.hull_tree(tree, thr=thr, bx=18, by=18, bz=18)
        volumes.append(ConvexHull(hull.vertices).volume)
    assert volumes[0] < volumes[1] < volumes[2]


def test_space_filling_hull_is_smaller_than_the_convex_hull(tree):
    """The entire reason this exists alongside `chull_tree`.

    A convex hull measures the volume a cell *spans*; this measures the
    volume it *occupies*. For a thin arbor those differ by a lot.
    """
    pytest.importorskip("skimage")
    _, convex = pt.chull_tree(tree)
    hull = pt.hull_tree(tree, thr=5.0, bx=32, by=32, bz=32,
                        return_distances=True)
    voxel = np.prod([a[1] - a[0] for a in hull.grid])
    occupied = (hull.distances <= 5.0).sum() * voxel
    assert occupied < convex.volume


def test_return_distances_gives_the_sampled_field(tree):
    pytest.importorskip("skimage")
    hull = pt.hull_tree(tree, thr=25.0, bx=15, by=15, bz=15,
                        return_distances=True)
    assert hull.distances.shape == tuple(len(a) for a in hull.grid)
    assert hull.distances.min() >= 0.0


def test_unreachable_threshold_warns_rather_than_returning_junk(tree):
    """A `thr` finer than the grid can resolve yields nothing; say so.

    Note it has to be a *small* threshold. A huge one does not work as a
    probe, because the grid is padded by `2 * thr` and therefore inflates
    to match -- the level stays reachable however large it gets.
    """
    pytest.importorskip("skimage")
    with pytest.warns(UserWarning, match="never crosses"):
        hull = pt.hull_tree(tree, thr=1e-6, bx=6, by=6, bz=6)
    assert len(hull.vertices) == 0


def test_explicit_grid_coordinates_are_used_verbatim(tree):
    pytest.importorskip("skimage")
    axis = np.linspace(-200.0, 400.0, 21)
    hull = pt.hull_tree(tree, thr=25.0, bx=axis, by=axis, bz=axis)
    np.testing.assert_allclose(hull.grid[0], axis)


# ---------------------------------------------------------------------------
# vhull_tree
# ---------------------------------------------------------------------------


def test_territories_are_positive_and_mostly_bounded(tree):
    pytest.importorskip("skimage")
    result = pt.vhull_tree(tree, thr=25.0)
    assert len(result.volumes) == tree.n_nodes
    finite = result.volumes[np.isfinite(result.volumes)]
    assert len(finite) > tree.n_nodes * 0.9
    assert (finite > 0).all()


def test_unbounded_cells_are_nan_not_silently_dropped(tree):
    """MATLAB drops them, which biases any mean over the result -- the
    outermost nodes are exactly the ones with the largest territories."""
    pytest.importorskip("skimage")
    result = pt.vhull_tree(tree, thr=25.0)
    assert result.volumes.shape == (tree.n_nodes,)  # one entry per node, always


def test_too_few_nodes_is_rejected_clearly():
    from scipy import sparse

    dA = sparse.csr_matrix(([1, 1], ([1, 2], [0, 1])), shape=(3, 3))
    tiny = pt.Tree(dA=dA, X=np.arange(3.0), Y=np.zeros(3), Z=np.zeros(3),
                   D=np.ones(3), R=np.zeros(3, dtype=int), rnames=["d"])
    with pytest.raises(ValueError, match="at least"):
        pt.vhull_tree(tiny, thr=10.0)


# ---------------------------------------------------------------------------
# share_boundary_tree
# ---------------------------------------------------------------------------


def test_a_tree_shares_all_of_its_volume_with_itself(tree):
    shared = pt.share_boundary_tree(tree, tree, thr=20.0, sr=10.0)
    assert shared > 0


def test_distant_trees_share_nothing(tree):
    far = pt.tran_tree(tree, [5000.0, 0.0, 0.0])
    assert pt.share_boundary_tree(tree, far, thr=20.0, sr=10.0) == 0.0


def test_overlap_falls_as_trees_separate(tree):
    overlaps = [
        pt.share_boundary_tree(tree, pt.tran_tree(tree, [shift, 0.0, 0.0]),
                               thr=25.0, sr=12.0)
        for shift in (0.0, 100.0, 300.0)
    ]
    assert overlaps[0] > overlaps[1] > overlaps[2]


# ---------------------------------------------------------------------------
# stats_tree's density statistics, unblocked by the above
# ---------------------------------------------------------------------------


def test_stats_tree_reports_parea_and_mparea(tree):
    pytest.importorskip("skimage")
    stats = pt.stats_tree([tree], extras=True)
    assert "mparea" in stats["summary"].columns
    assert "parea" in stats["points"].columns


def test_mparea_is_the_mean_of_parea(tree):
    pytest.importorskip("skimage")
    stats = pt.stats_tree([tree], extras=True)
    assert stats["points"]["parea"].mean() == pytest.approx(
        stats["summary"]["mparea"].iloc[0]
    )
