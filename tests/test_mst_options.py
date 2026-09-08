"""`MST_tree`'s ported options: multi-tree growth, `dist`, cut ends, history.

Design Decision #58. The single-tree core was always here; these are the
parts of MATLAB's `MST_tree` that were deferred with it, and the multi-tree
mode in particular is what the published construction is normally used for.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import pynetrees as pt


@pytest.fixture
def cloud():
    """A reproducible point cloud with two well-separated lobes."""
    rng = np.random.default_rng(7)
    left = rng.normal(loc=(-60.0, 0.0, 0.0), scale=25.0, size=(80, 3))
    right = rng.normal(loc=(60.0, 0.0, 0.0), scale=25.0, size=(80, 3))
    pts = np.vstack([[-60.0, 0.0, 0.0], [60.0, 0.0, 0.0], left, right])
    return pts[:, 0], pts[:, 1], pts[:, 2]


# ---------------------------------------------------------------------------
# return contract
# ---------------------------------------------------------------------------


def test_single_start_returns_a_bare_tree(cloud):
    X, Y, Z = cloud
    result = pt.MST_tree(X, Y, Z, start=0, thr=60.0)
    assert isinstance(result, pt.Tree)


def test_full_output_carries_connected_indx_and_history(cloud):
    X, Y, Z = cloud
    result = pt.MST_tree(X, Y, Z, start=0, thr=60.0, full_output=True)
    assert isinstance(result.trees, pt.Tree)
    assert result.connected.shape == (len(X),)
    assert result.indx.shape == (len(X), 2)
    assert result.history is None  # not requested


# ---------------------------------------------------------------------------
# multi-tree competitive growth
# ---------------------------------------------------------------------------


def test_several_starts_grow_several_trees(cloud):
    X, Y, Z = cloud
    trees = pt.MST_tree(X, Y, Z, start=[0, 1], thr=60.0)
    assert isinstance(trees, list) and len(trees) == 2
    assert all(isinstance(t, pt.Tree) for t in trees)


def test_competing_trees_partition_the_cloud(cloud):
    """No point may end up in two trees: they bid, the cheapest bid wins."""
    X, Y, Z = cloud
    result = pt.MST_tree(X, Y, Z, start=[0, 1], thr=60.0, full_output=True)
    total = sum(t.n_nodes for t in result.trees)
    assert total == int(result.connected.sum())


def test_each_tree_claims_its_own_lobe(cloud):
    """Territory falls out of the growth rather than being assigned."""
    X, Y, Z = cloud
    trees = pt.MST_tree(X, Y, Z, start=[0, 1], thr=60.0)
    left, right = trees
    assert left.X.mean() < 0 < right.X.mean()


def test_indx_maps_input_points_back_to_tree_and_node(cloud):
    X, Y, Z = cloud
    result = pt.MST_tree(X, Y, Z, start=[0, 1], thr=60.0, full_output=True)
    trees, indx = result.trees, result.indx

    for point in np.flatnonzero(result.connected):
        tree_index, node_index = indx[point]
        assert trees[tree_index].X[node_index] == pytest.approx(X[point])
        assert trees[tree_index].Y[node_index] == pytest.approx(Y[point])

    unconnected = np.flatnonzero(~result.connected)
    assert (indx[unconnected] == -1).all()


def test_duplicate_start_points_are_rejected(cloud):
    X, Y, Z = cloud
    with pytest.raises(ValueError, match="distinct"):
        pt.MST_tree(X, Y, Z, start=[0, 0])


def test_out_of_range_start_is_rejected(cloud):
    X, Y, Z = cloud
    with pytest.raises(ValueError, match="start indices"):
        pt.MST_tree(X, Y, Z, start=[0, 10_000])


# ---------------------------------------------------------------------------
# dist: connection preferences
# ---------------------------------------------------------------------------


def test_dist_shape_is_validated(cloud):
    X, Y, Z = cloud
    with pytest.raises(ValueError, match="indexed over the input points"):
        pt.MST_tree(X, Y, Z, dist=sparse.csr_matrix((3, 3)))


def test_dist_preference_changes_who_attaches_where():
    """A strong preference pulls a point onto a different parent.

    `dist` values are in **distance units**: the penalty spans
    `0 .. max(dist)`, so a preference only competes with geometry if it is
    scaled like one. A `max(dist)` of 1.0 against 10 um spacings is
    correctly ignored -- which is what a first version of this test got
    wrong, not the implementation.
    """
    # collinear points; 3 would normally attach to its neighbour 2
    X = np.array([0.0, 10.0, 20.0, 30.0])
    Y = np.zeros(4)
    plain = pt.MST_tree(X, Y, start=0, thr=100.0, bf=0.0)
    assert plain.dA[3, 2] == 1  # baseline: chained

    dist = sparse.lil_matrix((4, 4))
    dist[0, 3] = 50.0  # a 50 um-worth preference for 3 attaching to 0
    preferred = pt.MST_tree(X, Y, start=0, thr=100.0, bf=0.0,
                            dist=dist.tocsr())
    assert not np.array_equal(plain.dA.toarray(), preferred.dA.toarray())


def test_cut_ends_requires_dist(cloud):
    X, Y, Z = cloud
    with pytest.raises(ValueError, match="needs dist"):
        pt.MST_tree(X, Y, Z, cut_ends=True)


def test_cut_ends_restricts_which_nodes_can_grow():
    """Only points with a positive `dist` row may sprout; growth stops there.

    `thr` has to be small enough that growth proceeds hop by hop -- with a
    reach that spans the whole cloud, everything attaches to the start
    directly and no intermediate node ever needs to sprout.
    """
    X = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    Y = np.zeros(5)
    dist = sparse.lil_matrix((5, 5))
    dist[1, 2] = 1.0  # node 1 is the only marked cut end
    grown = pt.MST_tree(X, Y, start=0, thr=15.0, dist=dist.tocsr(),
                        cut_ends=True)
    # 0 (start) sprouts 1, 1 (a cut end) sprouts 2, and 2 may not sprout
    assert grown.n_nodes == 3

    unrestricted = pt.MST_tree(X, Y, start=0, thr=15.0)
    assert unrestricted.n_nodes == 5


# ---------------------------------------------------------------------------
# growth history
# ---------------------------------------------------------------------------


def test_record_logs_every_attachment_in_order(cloud):
    X, Y, Z = cloud
    result = pt.MST_tree(X, Y, Z, start=0, thr=60.0, record=True,
                         full_output=True)
    history = result.history
    assert history.shape[1] == 3
    assert len(history) == result.trees.n_nodes - 1  # every node but the root

    # a point is only ever attached to something already in the tree
    present = {0}
    for _tree_index, point, parent in history.tolist():
        assert parent in present
        present.add(point)


def test_history_prefix_reconstructs_an_intermediate_state(cloud):
    """The log is stored instead of intermediate trees precisely because
    every intermediate state is one of its prefixes."""
    X, Y, Z = cloud
    result = pt.MST_tree(X, Y, Z, start=0, thr=60.0, record=True,
                         full_output=True)
    halfway = result.history[: len(result.history) // 2]
    nodes = {0} | {int(p) for _t, p, _par in halfway.tolist()}
    assert len(nodes) == len(halfway) + 1


# ---------------------------------------------------------------------------
# the original single-tree behaviour is unchanged
# ---------------------------------------------------------------------------


def test_balancing_factor_still_trades_wiring_for_path_length(cloud):
    X, Y, Z = cloud
    wire = pt.MST_tree(X, Y, Z, bf=0.0, thr=60.0)
    path = pt.MST_tree(X, Y, Z, bf=1.0, thr=60.0)
    assert wire.total_length <= path.total_length
    assert pt.Pvec_tree(path).mean() <= pt.Pvec_tree(wire).mean()


# ---------------------------------------------------------------------------
# multi-tree growth: trees take turns, as MATLAB's outer loop does
# ---------------------------------------------------------------------------


def test_competing_trees_take_turns_rather_than_racing():
    """MATLAB grows one connection per tree per sweep::

        while something_was_added
            for counter = 1 : n_trees      % strictly in index order
                take THIS tree's own cheapest available connection

    so equivalent seeds in a uniform cloud end up with equal territories.
    A single global heap instead pops the cheapest connection anywhere,
    letting a tree in a denser patch run ahead -- which on this cloud gave
    94/124/85 against MATLAB's 101/101/101.
    """
    rng = np.random.default_rng(20260908)
    n = 300
    X = np.concatenate([[20.0, 100.0, 180.0], rng.uniform(0, 200, n)])
    Y = np.concatenate([[50.0, 50.0, 50.0], rng.uniform(0, 100, n)])
    Z = np.zeros(n + 3)

    trees = pt.MST_tree(X, Y, Z, start=[0, 1, 2], bf=0.4, thr=40.0)

    sizes = sorted(t.n_nodes for t in trees)
    assert sum(sizes) == n + 3
    # verified against MATLAB R2026a's MST_tree on this exact cloud
    assert sizes == [101, 101, 101]


def test_a_point_claimed_in_a_sweep_is_gone_for_the_later_trees():
    """Within one sweep the trees are ordered, so tree 0 takes its pick
    before tree 1 sees the cloud -- no point may end up in two trees."""
    rng = np.random.default_rng(7)
    X = np.concatenate([[0.0, 30.0], rng.uniform(0, 30, 120)])
    Y = np.concatenate([[0.0, 0.0], rng.uniform(-15, 15, 120)])
    Z = np.zeros(122)

    trees = pt.MST_tree(X, Y, Z, start=[0, 1], bf=0.3, thr=25.0)
    placed = [
        (round(float(x), 6), round(float(y), 6))
        for t in trees for x, y in zip(t.X, t.Y)
    ]
    assert len(placed) == len(set(placed))


def test_single_tree_growth_is_unaffected_by_the_turn_taking():
    """One tree means one turn per sweep, i.e. exactly the old behaviour --
    and this case is bit-identical to MATLAB."""
    rng = np.random.default_rng(20260908)
    X = np.concatenate([[50.0], rng.uniform(0, 100, 150)])
    Y = np.concatenate([[50.0], rng.uniform(0, 100, 150)])
    Z = np.zeros(151)

    tree = pt.MST_tree(X, Y, Z, start=0, bf=0.5, thr=50.0)
    assert tree.n_nodes == 151
    # MATLAB R2026a on the same cloud: 976.1 um of cable
    assert float(pt.len_tree(tree).sum()) == pytest.approx(976.1, abs=0.1)
