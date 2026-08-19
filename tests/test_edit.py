"""Tests for pytrees.edit: structural editing, repair, resampling.

Also covers `abel_tree` and `rootangle_tree`, which MATLAB files under
"metrics" but which live in `edit.py` here since they need `delete_tree`/
`resample_tree` (see edit.py's module docstring for why).
"""

import numpy as np
import pytest
from scipy import sparse

from pytrees import (
    Tree,
    abel_tree,
    cat_tree,
    delete_tree,
    elim0_tree,
    elimt_tree,
    idpar_tree,
    insert_tree,
    insertp_tree,
    interpd_tree,
    len_tree,
    recon_tree,
    repair_tree,
    resample_tree,
    restrain_tree,
    root_tree,
    rootangle_tree,
    sample_tree,
    ver_tree,
)


def _branchy_tree() -> Tree:
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


def _chain_tree() -> Tree:
    dA = sparse.csr_matrix(([1, 1, 1], ([1, 2, 3], [0, 1, 2])), shape=(4, 4))
    return Tree(
        dA=dA,
        X=np.array([0.0, 10.0, 20.0, 30.0]),
        Y=np.zeros(4),
        Z=np.zeros(4),
        D=np.array([4.0, 3.0, 2.0, 1.0]),
        R=np.zeros(4, dtype=int),
        rnames=["a"],
    )


# ---------------------------------------------------------------------------
# delete_tree
# ---------------------------------------------------------------------------


def test_delete_tree_splices_children_to_grandparent():
    tree = _branchy_tree()
    result = delete_tree(tree, [1])
    assert isinstance(result, Tree)
    assert result.n_nodes == 4
    np.testing.assert_array_equal(idpar_tree(result), [0, 0, 0, 0])
    assert ver_tree(result, quiet=True) == []


def test_delete_tree_splits_forest_when_branching_root_deleted():
    tree = _branchy_tree()
    result = delete_tree(tree, [0])
    assert isinstance(result, list)
    assert sorted(t.n_nodes for t in result) == [1, 3]
    for t in result:
        assert ver_tree(t, quiet=True) == []


def test_delete_tree_boolean_mask_and_index_list_agree():
    tree = _branchy_tree()
    by_index = delete_tree(tree, [1])
    mask = np.array([False, True, False, False, False])
    by_mask = delete_tree(tree, mask)
    np.testing.assert_array_equal(by_index.dA.toarray(), by_mask.dA.toarray())


def test_delete_tree_rejects_deleting_everything():
    tree = _branchy_tree()
    with pytest.raises(ValueError):
        delete_tree(tree, np.ones(5, dtype=bool))


def test_delete_tree_trims_unused_regions_by_default():
    tree = _branchy_tree()
    tree.R = np.array([0, 0, 1, 1, 1])
    tree.rnames = ["soma", "dend"]
    result = delete_tree(tree, [1, 2, 3, 4])  # only region "soma" node survives
    assert result.rnames == ["soma"]
    assert result.R[0] == 0


# ---------------------------------------------------------------------------
# repair pipeline
# ---------------------------------------------------------------------------


def test_elim0_tree_removes_duplicate_zero_length_nodes():
    tree = _chain_tree()
    tree.X[2] = tree.X[1]  # node 2 now coincides with node 1: zero-length segment
    result = elim0_tree(tree)
    assert result.n_nodes == 3
    np.testing.assert_allclose(len_tree(result), len_tree(result))  # no crash
    assert not np.any(len_tree(result)[1:] == 0)


def test_elim0_tree_leaves_well_formed_tree_unchanged():
    tree = _chain_tree()
    result = elim0_tree(tree)
    assert result.n_nodes == tree.n_nodes


def test_elimt_tree_converts_trifurcation_to_two_bifurcations():
    dA = sparse.csr_matrix(([1, 1, 1], ([1, 2, 3], [0, 0, 0])), shape=(4, 4))
    tree = Tree(
        dA=dA, X=np.array([0.0, 1.0, 2.0, 3.0]), Y=np.array([0.0, 1.0, -1.0, 0.5]),
        Z=np.zeros(4), D=np.ones(4), R=np.zeros(4, dtype=int), rnames=["a"],
    )
    result = elimt_tree(tree)
    # Design Decision #42: elimt_tree returns just the Tree now. "Did it
    # change anything?" is recoverable from the result itself, which is why
    # the old boolean second output was dropped rather than kept.
    assert result is not tree
    assert result.n_nodes == 5  # one spacer node added
    children_count = np.asarray(result.dA.sum(axis=0)).ravel()
    assert children_count.max() == 2
    assert ver_tree(result, quiet=True) == []


def test_elimt_tree_no_op_when_already_binary():
    tree = _branchy_tree()
    result = elimt_tree(tree)
    assert result is tree  # unchanged input is returned as-is, not copied


def test_repair_tree_produces_strictly_binary_tree():
    dA = sparse.csr_matrix(([1, 1, 1], ([1, 2, 3], [0, 0, 0])), shape=(4, 4))
    tree = Tree(
        dA=dA, X=np.array([0.0, 1.0, 2.0, 3.0]), Y=np.array([0.0, 1.0, -1.0, 0.5]),
        Z=np.zeros(4), D=np.ones(4), R=np.zeros(4, dtype=int), rnames=["a"],
    )
    repaired = repair_tree(tree)
    children_count = np.asarray(repaired.dA.sum(axis=0)).ravel()
    assert children_count.max() <= 2
    assert ver_tree(repaired, quiet=True) == []


def test_repair_tree_on_real_reconstruction_is_idempotent():
    tree = repair_tree(sample_tree())
    twice = repair_tree(tree)
    assert tree.n_nodes == twice.n_nodes
    children_count = np.asarray(tree.dA.sum(axis=0)).ravel()
    assert children_count.max() <= 2
    assert ver_tree(tree, quiet=True) == []


# ---------------------------------------------------------------------------
# adding / moving nodes
# ---------------------------------------------------------------------------


def test_root_tree_prepends_node_and_shifts_indices():
    tree = _chain_tree()
    rooted = root_tree(tree)
    assert rooted.n_nodes == tree.n_nodes + 1
    np.testing.assert_array_equal(idpar_tree(rooted), [0, 0, 1, 2, 3])
    assert rooted.X[1] == pytest.approx(tree.X[0])  # old root's data preserved
    assert rooted.X[0] == pytest.approx(tree.X[0] - 0.0001)


def test_insert_tree_appends_leaves():
    tree = _chain_tree()
    result = insert_tree(tree, X=[100.0], Y=[0.0], Z=[0.0], D=[1.0], parent=[1])
    assert result.n_nodes == 5
    np.testing.assert_array_equal(idpar_tree(result)[4:], [1])
    assert result.X[4] == pytest.approx(100.0)


def test_insertp_tree_inserts_at_requested_path_lengths():
    tree = _chain_tree()
    new_tree, added = insertp_tree(tree, inode=3, plens=[5.0, 15.0, 25.0], full_output=True)
    assert new_tree.n_nodes == 7
    assert added.sum() == 3
    assert ver_tree(new_tree, quiet=True) == []
    # the 3 new nodes should sit exactly at X = 5, 15, 25 (chain runs along X)
    new_X = sorted(new_tree.X[added].tolist())
    np.testing.assert_allclose(new_X, [5.0, 15.0, 25.0])


def test_insertp_tree_skips_positions_that_already_exist():
    tree = _chain_tree()
    _, added = insertp_tree(tree, inode=3, plens=[10.0, 20.0], full_output=True)  # already nodes
    assert added.sum() == 0


def test_interpd_tree_linear_interpolation():
    tree = _chain_tree()
    tree.D = np.array([0.0, 10.0, 20.0, 30.0])
    result = interpd_tree(tree, 0, 3)
    np.testing.assert_allclose(result.D, [0.0, 10.0, 20.0, 30.0])


def test_interpd_tree_rejects_unrelated_nodes():
    dA = sparse.csr_matrix(([1, 1], ([1, 2], [0, 0])), shape=(3, 3))
    tree = Tree(
        dA=dA, X=np.zeros(3), Y=np.zeros(3), Z=np.zeros(3),
        D=np.ones(3), R=np.zeros(3, dtype=int), rnames=["a"],
    )
    with pytest.raises(ValueError):
        interpd_tree(tree, 1, 2)  # siblings, not on the same root path


def test_recon_tree_reconnects_and_shifts_subtree():
    tree = _branchy_tree()
    # reconnect node 2's subtree (just itself) onto node 3
    result = recon_tree(tree, ichilds=[2], ipars=[3])
    np.testing.assert_array_equal(idpar_tree(result)[2], 3)
    np.testing.assert_allclose([result.X[2], result.Y[2]], [result.X[3], result.Y[3]])


def test_restrain_tree_interpolates_terminal_back_to_maxpl():
    tree = _chain_tree()  # path lengths from root: 0, 10, 20, 30
    result = restrain_tree(tree, maxpl=15.0, interpolate=True)
    assert result.n_nodes == 3
    assert result.X[2] == pytest.approx(15.0)


def test_restrain_tree_deletes_without_interpolation():
    tree = _chain_tree()
    result = restrain_tree(tree, maxpl=15.0, interpolate=False)
    assert result.n_nodes == 2


def test_restrain_tree_no_op_when_under_limit():
    tree = _chain_tree()
    result = restrain_tree(tree, maxpl=1000.0)
    assert result.n_nodes == tree.n_nodes


def test_cat_tree_concatenates_two_trees():
    t1 = _chain_tree()
    t2 = _branchy_tree()  # t2's root is itself a branch point -> redirect_tree
    # warns about a trifurcation even though inode2 defaults to that same
    # root (a no-op reroot); this matches MATLAB's cat_tree/redirect_tree,
    # which perform the same unconditional check -- see cat_tree.m.
    with pytest.warns(UserWarning, match="trifurcation"):
        merged = cat_tree(t1, t2)
    assert merged.n_nodes == t1.n_nodes + t2.n_nodes
    assert ver_tree(merged, quiet=True) == []


def test_cat_tree_connects_to_closest_node_by_default():
    t1 = _chain_tree()
    t2 = _branchy_tree()
    t2.X = t2.X + 20.0  # place tree2's root near t1's node index 2 (X=20)
    with pytest.warns(UserWarning, match="trifurcation"):
        merged = cat_tree(t1, t2)
    idpar = idpar_tree(merged)
    assert idpar[t1.n_nodes] == 2


# ---------------------------------------------------------------------------
# resample_tree
# ---------------------------------------------------------------------------


def test_resample_tree_anchors_preserves_branch_and_terminal_count():
    """`method='anchors'` is defined by keeping these fixed.

    MATLAB's method deliberately does not: it deletes every original node,
    so branch and termination points move onto the sr grid and can merge.
    """
    tree = sample_tree()
    resampled = resample_tree(tree, sr=10.0, method="anchors")
    from pytrees import B_tree, T_tree

    assert B_tree(resampled).sum() == B_tree(tree).sum()
    assert T_tree(resampled).sum() == T_tree(tree).sum()
    assert ver_tree(resampled, quiet=True) == []


def test_resample_tree_on_simple_chain_gives_expected_node_count():
    tree = _chain_tree()  # single segment, root(X=0) to leaf(X=30), length 30
    resampled = resample_tree(tree, sr=10.0, extend_terminals=False)
    # anchors are root + terminal (2), plus grid points strictly within (0, 30):
    # 10, 20 -> 2 new nodes = 4 total
    assert resampled.n_nodes == 4
    np.testing.assert_allclose(sorted(resampled.X.tolist()), [0.0, 10.0, 20.0, 30.0])


def test_resample_tree_extends_terminals_by_half_sr():
    """The tip is stretched by sr/2 before resampling, in both methods.

    The two methods then differ in what survives: `anchors` keeps the
    stretched tip itself (35.0), `matlab` snaps to the last grid multiple
    at or below it (30.0).
    """
    tree = _chain_tree()
    anchors = resample_tree(tree, sr=10.0, method="anchors", extend_terminals=True)
    assert anchors.X.max() == pytest.approx(35.0)

    matlab = resample_tree(tree, sr=10.0, method="matlab", extend_terminals=True)
    assert matlab.X.max() == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# abel_tree / rootangle_tree (Phase 3 backlog, unblocked by this phase)
# ---------------------------------------------------------------------------


def test_abel_tree_matches_hand_computed_spacing():
    # continuation node 1 between root(0,0,0) and branch node 2(10,0,0);
    # after collapsing the continuation point, root-to-branch length is 10,
    # and the two branch-to-terminal legs are sqrt(50) each. abel_tree's
    # mean (matching MATLAB) is taken over len_tree of the *pruned* tree,
    # which includes the root's own trivial 0-length entry -- so this is
    # (0 + 10 + sqrt(50) + sqrt(50)) / 4, not a mean over the 3 real segments.
    dA = sparse.csr_matrix(
        ([1, 1, 1, 1], ([1, 2, 3, 4], [0, 1, 2, 2])), shape=(5, 5)
    )
    tree = Tree(
        dA=dA,
        X=np.array([0.0, 5.0, 10.0, 15.0, 15.0]),
        Y=np.array([0.0, 0.0, 0.0, 5.0, -5.0]),
        Z=np.zeros(5),
        D=np.ones(5),
        R=np.zeros(5, dtype=int),
        rnames=["a"],
    )
    expected = (0.0 + 10.0 + np.sqrt(50) + np.sqrt(50)) / 4
    assert abel_tree(tree) == pytest.approx(expected, rel=1e-6)


def test_rootangle_tree_root_is_zero_and_shape_matches_resampled_tree():
    tree = _chain_tree()
    rootangle = rootangle_tree(tree)
    assert rootangle[0] == 0.0
    assert np.all(np.isfinite(rootangle))


def test_rootangle_tree_straight_chain_is_zero_angle():
    tree = _chain_tree()  # perfectly straight along X: every segment points at the root's ray
    rootangle = rootangle_tree(tree)
    assert np.allclose(rootangle, 0.0, atol=1e-6)


def test_resample_tree_returns_single_node_tree_unchanged():
    # A single-node tree has no segments to resample. Without a guard the
    # section-based rebuild produces a zero-node tree and then fails deep
    # inside sort_tree with "expected exactly one root, found 0".
    from scipy import sparse

    single = Tree(
        dA=sparse.csr_matrix((1, 1)), X=np.array([0.0]), Y=np.array([0.0]),
        Z=np.array([0.0]), D=np.array([2.0]), R=np.array([0]), rnames=["dend"],
    )
    out = resample_tree(single, 5.0)
    assert out.n_nodes == 1


def test_rootangle_tree_handles_single_node_tree():
    # rootangle_tree resamples internally, so it hit the same failure
    from scipy import sparse

    single = Tree(
        dA=sparse.csr_matrix((1, 1)), X=np.array([0.0]), Y=np.array([0.0]),
        Z=np.array([0.0]), D=np.array([2.0]), R=np.array([0]), rnames=["dend"],
    )
    angles = rootangle_tree(single)
    assert len(angles) == 1
