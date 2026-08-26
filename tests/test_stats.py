"""Tests for pynetrees.stats: sholl_tree, vonMises_tree/bf_tree, peters_tree,
stats_tree.

`_geom_tree()` reuses the exact fixture from test_metrics.py (same
topology/coordinates, redefined here to keep this file self-contained):
root not at the origin, hand-computable segment lengths (5, 5, 3, 3) and
Euclidean distances -- see that file's docstring for the full layout. Every
node here is either a branch point (0, 1) or a termination point (2, 3, 4),
so `stats_tree`'s "topological points" set covers the whole tree, making
every summary statistic hand-computable.
"""

import numpy as np
import pytest
from scipy import sparse

from pynetrees import (
    Tree,
    bf_tree,
    peters_tree,
    rootangle_tree,
    sample_tree,
    sholl_tree,
    stats_tree,
    vonMises_tree,
)


def _geom_tree() -> Tree:
    dA = sparse.csr_matrix(
        ([1, 1, 1, 1], ([1, 2, 3, 4], [0, 0, 1, 1])), shape=(5, 5)
    )
    ox, oy, oz = 1.0, 2.0, 3.0
    X = np.array([0.0, 3.0, 0.0, 3.0, 6.0]) + ox
    Y = np.array([0.0, 4.0, 0.0, 4.0, 4.0]) + oy
    Z = np.array([0.0, 0.0, 5.0, 3.0, 0.0]) + oz
    D = np.array([4.0, 2.0, 2.0, 1.0, 1.0])
    return Tree(dA=dA, X=X, Y=Y, Z=Z, D=D, R=np.zeros(5, dtype=int), rnames=["dend"])


def _line_tree() -> Tree:
    # root at origin, single child 100um straight up the Z axis
    dA = sparse.csr_matrix(([1], ([1], [0])), shape=(2, 2))
    return Tree(
        dA=dA,
        X=np.zeros(2),
        Y=np.zeros(2),
        Z=np.array([0.0, 100.0]),
        D=np.array([2.0, 2.0]),
        R=np.zeros(2, dtype=int),
        rnames=["dend"],
    )


# ---------------------------------------------------------------------------
# sholl_tree
# ---------------------------------------------------------------------------


def test_sholl_tree_single_segment_hand_verified():
    tree = _line_tree()  # one 100um segment from the root
    dd = np.array([0.0, 50.0, 100.0, 150.0, 199.0, 201.0])
    result = sholl_tree(tree, dd, warn_double=False)
    # sphere radius = dd/2; a single straight segment of length 100 is
    # crossed exactly once as long as the radius doesn't exceed the
    # segment's length (100), and not at all once it does
    np.testing.assert_array_equal(result.s, [1, 1, 1, 1, 1, 0])
    np.testing.assert_array_equal(result.dd, dd)


def test_sholl_tree_zero_diameter_is_always_one_by_definition():
    tree = sample_tree()
    result = sholl_tree(tree, np.array([0.0, 50.0]), warn_double=False)
    assert result.s[0] == 1
    assert result.sd[0] == 0


def test_sholl_tree_single_only_subtracts_doubles():
    tree = sample_tree()
    full = sholl_tree(tree, 50.0, warn_double=False)
    single = sholl_tree(tree, 50.0, single_only=True, warn_double=False)
    np.testing.assert_allclose(single.s, full.s - full.sd)


# ---------------------------------------------------------------------------
# vonMises_tree / bf_tree
# ---------------------------------------------------------------------------


def test_vonMises_tree_accepts_tree_list_and_array_equivalently():
    tree = sample_tree()
    k_tree, _ = vonMises_tree(tree, dim=3)
    k_array, _ = vonMises_tree(rootangle_tree(tree), dim=3)
    assert k_tree == pytest.approx(k_array, rel=1e-9)


def test_vonMises_tree_pooling_same_tree_twice_matches_single():
    tree = sample_tree()
    k_single, _ = vonMises_tree(tree, dim=3)
    k_pooled, _ = vonMises_tree([tree, tree], dim=3)
    # pooling two copies of the identical distribution shouldn't change the fit
    assert k_pooled == pytest.approx(k_single, rel=1e-6)


def test_vonMises_tree_rejects_out_of_range_angles():
    with pytest.raises(ValueError):
        vonMises_tree(np.array([0.0, 4.0]))  # > pi


def test_bf_tree_stays_in_unit_range_on_real_data():
    tree = sample_tree()
    bf, k = bf_tree(tree, dim=3)
    assert 0.0 <= bf <= 1.0
    assert np.isfinite(k)


def test_bf_tree_clips_and_warns_when_out_of_range():
    # a deliberately pathological (tiny p1) fit-parameter choice blows up
    # (k/p1) regardless of the fitted k's sign/magnitude, driving bf past
    # 1 -- deterministically forces the out-of-range clip-and-warn branch
    angles = np.linspace(0.1, 3.0, 20)
    with pytest.warns(UserWarning, match="out of usual range"):
        bf, k = bf_tree(angles, dim=3, fit_constants=(1e-6, 1.0, 1.0))
    assert bf == 1.0


# ---------------------------------------------------------------------------
# peters_tree
# ---------------------------------------------------------------------------


def test_peters_tree_candidates_within_spinedis():
    dA = sparse.csr_matrix(
        ([1, 1], ([1, 2], [0, 1])), shape=(3, 3)
    )
    t1 = Tree(
        dA=dA, X=np.array([0.0, 1.0, 2.0]), Y=np.zeros(3), Z=np.zeros(3),
        D=np.ones(3), R=np.zeros(3, dtype=int), rnames=["a"],
    )
    t2 = Tree(
        dA=dA, X=np.array([0.5, 1.5, 2.5]), Y=np.zeros(3), Z=np.zeros(3),
        D=np.ones(3), R=np.zeros(3, dtype=int), rnames=["a"],
    )
    cand = peters_tree(t1, t2, spinedis=1.0, synapsedis=2.0, resample=False)
    assert cand.shape[1] == 3
    assert np.all(cand[:, 2] < 1.0)


def test_peters_tree_elimination_leaves_no_close_pair_behind():
    # property check on the greedy elimination itself: no two surviving
    # candidates can both be within synapsedis of each other in *both*
    # trees (that's precisely what the elimination step is supposed to
    # prevent) -- checked directly against the output, independent of the
    # exact candidate set peters_tree happens to produce
    dA = sparse.csr_matrix(
        ([1, 1, 1], ([1, 2, 3], [0, 1, 2])), shape=(4, 4)
    )
    t1 = Tree(
        dA=dA, X=np.array([0.0, 1.0, 2.0, 3.0]), Y=np.zeros(4), Z=np.zeros(4),
        D=np.ones(4), R=np.zeros(4, dtype=int), rnames=["a"],
    )
    t2 = Tree(
        dA=dA, X=np.array([0.3, 1.3, 2.3, 3.3]), Y=np.zeros(4), Z=np.zeros(4),
        D=np.ones(4), R=np.zeros(4, dtype=int), rnames=["a"],
    )
    synapsedis = 1.5
    cand = peters_tree(t1, t2, spinedis=1.0, synapsedis=synapsedis, resample=False)
    n1 = cand[:, 0].astype(int)
    n2 = cand[:, 1].astype(int)
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            d1 = abs(t1.X[n1[i]] - t1.X[n1[j]])
            d2 = abs(t2.X[n2[i]] - t2.X[n2[j]])
            assert not (d1 < synapsedis and d2 < synapsedis)


def test_peters_tree_no_candidates_returns_empty():
    dA = sparse.csr_matrix(([1], ([1], [0])), shape=(2, 2))
    t1 = Tree(dA=dA, X=np.zeros(2), Y=np.zeros(2), Z=np.zeros(2),
              D=np.ones(2), R=np.zeros(2, dtype=int), rnames=["a"])
    t2 = Tree(dA=dA, X=np.array([1000.0, 1001.0]), Y=np.zeros(2), Z=np.zeros(2),
              D=np.ones(2), R=np.zeros(2, dtype=int), rnames=["a"])
    cand = peters_tree(t1, t2, spinedis=1.0, synapsedis=1.0, resample=False)
    assert cand.shape == (0, 3)


# ---------------------------------------------------------------------------
# stats_tree
# ---------------------------------------------------------------------------


def test_stats_tree_summary_hand_verified():
    tree = _geom_tree()
    result = stats_tree(tree)
    summary = result["summary"]
    assert len(summary) == 1
    row = summary.iloc[0]

    assert row["len"] == pytest.approx(16.0)
    assert row["bpoints"] == 2
    assert row["max_plen"] == pytest.approx(8.0)
    assert row["mplen"] == pytest.approx(5.2)
    assert row["mblen"] == pytest.approx(4.0)
    assert row["mpeucl"] == pytest.approx(0.907564, rel=1e-5)
    assert row["chullx"] == pytest.approx(3.4)
    assert row["chully"] == pytest.approx(4.4)
    assert row["chullz"] == pytest.approx(4.6)
    assert row["wh"] == pytest.approx(1.5)
    assert row["wz"] == pytest.approx(1.2)

    # every node in this fixture is a branch or termination point
    assert len(result["points"]) == 5
    # 4 non-trivial dissected sections (root's own trivial one is filtered out)
    assert len(result["branches"]) == 4


def test_stats_tree_groups_and_multiple_trees():
    tree = _geom_tree()
    result = stats_tree([[tree, tree], [tree]], group_names=["a", "b"])
    summary = result["summary"]
    assert list(summary["group"]) == ["a", "a", "b"]
    assert list(summary["tree"]) == [0, 1, 0]


def test_stats_tree_group_names_length_mismatch_raises():
    tree = _geom_tree()
    with pytest.raises(ValueError):
        stats_tree([tree], group_names=["a", "b"])


def test_stats_tree_extras_adds_hull_and_sholl():
    pytest.importorskip("skimage")
    tree = sample_tree()
    result = stats_tree(tree, extras=True)
    assert "hull_volume" in result["summary"].columns
    assert "masym" in result["summary"].columns
    assert "sholl" in result
    assert result["sholl"]["radius"].nunique() == len(result["sholl"])
