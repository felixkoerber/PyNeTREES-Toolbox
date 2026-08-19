"""Tests for pytrees.graphtheory: pure dA/topology primitives.

Hand-verified against a small fixed tree (see `_branchy_tree`):

    0 (root)
    +-- 1
    |   +-- 3 (leaf)
    |   +-- 4 (leaf)
    +-- 2 (leaf)

plus sanity/integration checks against the real bundled reconstruction
(`sample_tree()`), mirroring the intent of treestoolbox-master's
tests/graphtheory/check_*.m (which mostly just call each function on
sample_tree and eyeball a plot -- here we assert concrete expected values).
"""

import numpy as np
import pytest
from scipy import sparse

from pytrees.graphtheory import _subtree_blocks
from pytrees import (
    hss_tree,
    B_tree,
    BO_tree,
    C_tree,
    LO_tree,
    NO_PARENT,
    PL_tree,
    Pvec_tree,
    T_tree,
    Tree,
    asym_tree,
    child_tree,
    dissect_tree,
    idchild_tree,
    idpar_tree,
    ipar_tree,
    ratio_tree,
    redirect_tree,
    rindex_tree,
    sample_tree,
    sort_tree,
    strahler_tree,
    sub_tree,
    typeN_tree,
)


def _branchy_tree() -> Tree:
    dA = sparse.csr_matrix(
        ([1, 1, 1, 1], ([1, 2, 3, 4], [0, 0, 1, 1])), shape=(5, 5)
    )
    return Tree(
        dA=dA,
        X=np.arange(5, dtype=float),
        Y=np.zeros(5),
        Z=np.zeros(5),
        D=np.array([2.0, 1.0, 1.0, 0.5, 0.5]),
        R=np.array([0, 0, 0, 0, 0]),
        rnames=["dend"],
    )


# ---------------------------------------------------------------------------
# node typing
# ---------------------------------------------------------------------------


def test_typeN_and_BCT_masks():
    tree = _branchy_tree()
    np.testing.assert_array_equal(typeN_tree(tree), [2, 2, 0, 0, 0])
    np.testing.assert_array_equal(B_tree(tree), [True, True, False, False, False])
    np.testing.assert_array_equal(C_tree(tree), [False, False, False, False, False])
    np.testing.assert_array_equal(T_tree(tree), [False, False, True, True, True])


def test_BCT_masks_are_mutually_exclusive_and_exhaustive_on_sample_tree():
    tree = sample_tree()
    B, C, T = B_tree(tree), C_tree(tree), T_tree(tree)
    assert np.all(B.astype(int) + C.astype(int) + T.astype(int) == 1)


# ---------------------------------------------------------------------------
# parent / child indices
# ---------------------------------------------------------------------------


def test_idpar_tree_self_referencing_root():
    tree = _branchy_tree()
    np.testing.assert_array_equal(idpar_tree(tree), [0, 0, 0, 1, 1])


def test_idpar_tree_root_self_false_gives_sentinel():
    tree = _branchy_tree()
    idpar = idpar_tree(tree, root_self=False)
    assert idpar[0] == NO_PARENT
    np.testing.assert_array_equal(idpar[1:], [0, 0, 1, 1])


def test_idchild_tree():
    tree = _branchy_tree()
    idchild = idchild_tree(tree)
    np.testing.assert_array_equal(idchild[0], [1, 2])
    np.testing.assert_array_equal(idchild[1], [3, 4])
    np.testing.assert_array_equal(idchild[2], [NO_PARENT, NO_PARENT])


def test_idchild_tree_first_only():
    tree = _branchy_tree()
    first = idchild_tree(tree, first_only=True)
    np.testing.assert_array_equal(first, [1, 3, NO_PARENT, NO_PARENT, NO_PARENT])


# ---------------------------------------------------------------------------
# path length / order
# ---------------------------------------------------------------------------


def test_PL_tree():
    tree = _branchy_tree()
    np.testing.assert_array_equal(PL_tree(tree), [0, 1, 1, 2, 2])


def test_ipar_tree():
    tree = _branchy_tree()
    ipar = ipar_tree(tree)
    # max depth is 2 (node 3/4), so rows are padded to 2 + 2 = 4 columns
    np.testing.assert_array_equal(ipar[0], [0, NO_PARENT, NO_PARENT, NO_PARENT])
    np.testing.assert_array_equal(ipar[1], [1, 0, NO_PARENT, NO_PARENT])
    np.testing.assert_array_equal(ipar[3], [3, 1, 0, NO_PARENT])


def test_BO_tree():
    tree = _branchy_tree()
    np.testing.assert_allclose(BO_tree(tree), [0, 1, 1, 2, 2])


def test_PL_and_ipar_consistent_on_sample_tree():
    tree = sample_tree()
    PL = PL_tree(tree)
    ipar = ipar_tree(tree)
    # PL[i] must equal the number of valid (non-sentinel) ancestors, minus self
    valid_counts = (ipar != NO_PARENT).sum(axis=1)
    np.testing.assert_array_equal(PL, valid_counts - 1)


def test_LO_tree_runs_on_sample_tree_and_root_is_minimal():
    tree = sample_tree()
    LO = LO_tree(tree)
    assert LO.shape == (tree.n_nodes,)
    assert np.all(np.isfinite(LO))


# ---------------------------------------------------------------------------
# meta-functions over an arbitrary vector
# ---------------------------------------------------------------------------


def test_child_tree_default_counts_descendants():
    tree = _branchy_tree()
    np.testing.assert_array_equal(child_tree(tree), [4, 2, 0, 0, 0])


def test_Pvec_tree_cumulative_sum_includes_self():
    tree = _branchy_tree()
    Pvec = Pvec_tree(tree, np.ones(5))
    np.testing.assert_array_equal(Pvec, [1, 2, 2, 3, 3])


def test_ratio_tree_default_diameter():
    tree = _branchy_tree()
    ratio = ratio_tree(tree)
    np.testing.assert_allclose(ratio, [1.0, 0.5, 0.5, 0.5, 0.5])


# ---------------------------------------------------------------------------
# regions, subtrees, rerooting, sorting, Strahler, asymmetry
# ---------------------------------------------------------------------------


def test_rindex_tree_resets_per_region():
    tree = _branchy_tree()
    tree.R = np.array([0, 0, 1, 1, 1])
    np.testing.assert_array_equal(rindex_tree(tree), [0, 1, 0, 1, 2])


def test_sub_tree_mask():
    tree = _branchy_tree()
    np.testing.assert_array_equal(
        sub_tree(tree, 1).mask, [False, True, False, True, True]
    )
    np.testing.assert_array_equal(
        sub_tree(tree, 0).mask, [True, True, True, True, True]
    )


def test_strahler_tree():
    tree = _branchy_tree()
    np.testing.assert_array_equal(strahler_tree(tree), [2, 2, 1, 1, 1])


def test_asym_tree_default_terminal_count():
    tree = _branchy_tree()
    asym = asym_tree(tree)
    assert np.isnan(asym[2]) and np.isnan(asym[3]) and np.isnan(asym[4])
    assert asym[1] == pytest.approx(0.5)
    assert asym[0] == pytest.approx(1 / 3)


def test_redirect_tree_reroots_at_leaf():
    tree = _branchy_tree()
    new_tree, order = redirect_tree(tree, 2, full_output=True)
    np.testing.assert_array_equal(order, [2, 0, 1, 3, 4])
    expected = np.zeros((5, 5))
    expected[1, 0] = 1  # old root (now idx 1) parented by old node2 (now idx 0)
    expected[2, 1] = 1  # old node1 (now idx 2) parented by old root (now idx 1)
    expected[3, 2] = 1
    expected[4, 2] = 1
    np.testing.assert_array_equal(new_tree.dA.toarray(), expected)


def test_redirect_tree_warns_when_new_root_is_a_branch_point():
    tree = _branchy_tree()
    with pytest.warns(UserWarning, match="trifurcation"):
        redirect_tree(tree, 1)


def test_sort_tree_hier_gives_contiguous_subtrees():
    # _branchy_tree's node 2 sits *between* node 1's subtree {1, 3, 4} in the
    # raw index order, so it is not itself BCT-conform (not every subtree is
    # a contiguous index range) -- sort_tree must fix that up.
    tree = _branchy_tree()
    resorted, order = sort_tree(tree, by="hier", full_output=True)
    np.testing.assert_array_equal(order, [0, 1, 3, 4, 2])

    idpar = idpar_tree(resorted, root_self=False)
    assert idpar[0] == NO_PARENT
    assert np.all(idpar[1:] < np.arange(1, 5))  # parent always precedes child


def test_sort_tree_recovers_valid_order_from_shuffled_tree():
    tree = _branchy_tree()
    # shuffle: old index -> new index mapping (permute nodes arbitrarily)
    shuffle = np.array([2, 4, 0, 1, 3])  # new_tree.node[i] = tree.node[shuffle[i]]
    shuffled = tree.reindexed(shuffle)

    for mode in ("hier", "lo", "lex"):
        resorted, order = sort_tree(shuffled, by=mode, full_output=True)
        # root must come first, and every parent must precede its child
        idpar = idpar_tree(resorted, root_self=False)
        assert idpar[0] == NO_PARENT
        assert np.all(idpar[1:] < np.arange(1, 5))
        # and it must be the same tree, just relabeled
        np.testing.assert_array_equal(
            sorted(resorted.D.tolist()), sorted(tree.D.tolist())
        )


def test_sort_tree_runs_on_sample_tree():
    tree = sample_tree()
    for mode in ("hier", "lo", "lex"):
        resorted, order = sort_tree(tree, by=mode, full_output=True)
        assert resorted.n_nodes == tree.n_nodes
        assert len(set(order.tolist())) == tree.n_nodes  # order is a permutation


def test_asym_tree_rejects_trifurcation():
    dA = sparse.csr_matrix(
        ([1, 1, 1], ([1, 2, 3], [0, 0, 0])), shape=(4, 4)
    )
    tree = Tree(
        dA=dA, X=np.zeros(4), Y=np.zeros(4), Z=np.zeros(4),
        D=np.ones(4), R=np.zeros(4, dtype=int), rnames=["a"],
    )
    with pytest.raises(ValueError, match="binary"):
        asym_tree(tree)


# ---------------------------------------------------------------------------
# dissect_tree
# ---------------------------------------------------------------------------


def _chain_to_branch_tree() -> Tree:
    # 0 --(C)--> 1 --(C)--> 2 --(B)--> {3 (T), 4 (T)}
    dA = sparse.csr_matrix(
        ([1, 1, 1, 1], ([1, 2, 3, 4], [0, 1, 2, 2])), shape=(5, 5)
    )
    return Tree(
        dA=dA, X=np.zeros(5), Y=np.zeros(5), Z=np.zeros(5),
        D=np.ones(5), R=np.zeros(5, dtype=int), rnames=["a"],
    )


def test_dissect_tree_groups_continuation_chain_into_one_section():
    tree = _chain_to_branch_tree()
    sect = dissect_tree(tree, by_region=False)
    np.testing.assert_array_equal(
        sorted(sect.tolist()), [[0, 2], [2, 3], [2, 4]]
    )


def test_dissect_tree_root_as_branch_point_has_no_degenerate_self_section():
    # a root that is itself a branch point (common in real reconstructions,
    # e.g. a soma branching directly into several dendrites) must not
    # produce a spurious (root, root) entry -- found via a real GC
    # morphology in build_neuron_model, where such an entry silently
    # became a self-loop edge in resample_tree's reconstructed dA
    dA = sparse.csr_matrix(([1, 1], ([1, 2], [0, 0])), shape=(3, 3))
    tree = Tree(
        dA=dA, X=np.array([0.0, 1.0, 1.0]), Y=np.array([0.0, 1.0, -1.0]),
        Z=np.zeros(3), D=np.ones(3), R=np.zeros(3, dtype=int), rnames=["a"],
    )
    sect = dissect_tree(tree, by_region=False)
    assert not any(s == e for s, e in sect.tolist())
    np.testing.assert_array_equal(sorted(sect.tolist()), [[0, 1], [0, 2]])


def test_dissect_tree_cuts_at_region_change():
    # region change strictly *inside* the chain (node1 -> node2), not at
    # the root's own boundary -- so this genuinely distinguishes
    # by_region=True from False (see the next test), unlike putting the
    # transition at the root's immediate child
    tree = _chain_to_branch_tree()
    tree.R = np.array([0, 0, 1, 1, 1])
    tree.rnames = ["soma", "dend"]
    sect = dissect_tree(tree, by_region=True)
    np.testing.assert_array_equal(
        sorted(sect.tolist()), [[0, 1], [1, 2], [2, 3], [2, 4]]
    )


def test_dissect_tree_region_change_at_roots_own_child_has_no_extra_boundary():
    # a region change exactly at the root's own child can't split the
    # root off on its own -- the resulting section just extends all the
    # way back to the root, same as if there were no region split at all
    tree = _chain_to_branch_tree()
    tree.R = np.array([0, 1, 1, 1, 1])
    tree.rnames = ["soma", "dend"]
    sect = dissect_tree(tree, by_region=True)
    np.testing.assert_array_equal(sorted(sect.tolist()), [[0, 2], [2, 3], [2, 4]])


def test_dissect_tree_ignores_region_change_when_disabled():
    tree = _chain_to_branch_tree()
    tree.R = np.array([0, 0, 1, 1, 1])
    tree.rnames = ["soma", "dend"]
    sect = dissect_tree(tree, by_region=False)
    np.testing.assert_array_equal(
        sorted(sect.tolist()), [[0, 2], [2, 3], [2, 4]]
    )


def test_dissect_tree_runs_on_sample_tree():
    tree = sample_tree()
    sect = dissect_tree(tree)
    assert sect.shape[1] == 2
    assert sect.shape[0] > 0


# ---------------------------------------------------------------------------
# _subtree_blocks (internal helper backing flatten_tree/morph_tree/smooth_tree)
# ---------------------------------------------------------------------------


def test_subtree_blocks_matches_sub_tree_on_real_reconstruction():
    # _subtree_blocks replaced a per-node `(ipar == node).any(axis=1)` scan
    # that was quadratic in (n_nodes x max_depth). It must return exactly
    # the same descendant sets -- cross-checked here against `sub_tree`, an
    # independent BFS implementation, on a real reconstruction (not just a
    # hand-built fixture, where a subtly wrong traversal could still
    # coincidentally agree). Sampled rather than exhaustive: `sub_tree` is
    # itself a fresh BFS, so checking all ~2250 nodes is quadratic and made
    # this file take ~48s on its own.
    tree = sample_tree()
    order, start, size = _subtree_blocks(tree.dA)
    rng = np.random.default_rng(0)
    nodes = [0, *rng.choice(tree.n_nodes, 60, replace=False).tolist()]
    for node in nodes:
        node = int(node)
        expected = set(np.flatnonzero(sub_tree(tree, node).mask).tolist())
        got = set(order[start[node] : start[node] + size[node]].tolist())
        assert got == expected, f"node {node}"


def test_subtree_blocks_sizes_are_globally_consistent():
    # cheap O(n) whole-tree invariant, complementing the sampled check
    # above: every node's subtree size must equal 1 + the sum of its
    # children's sizes, and each block must start with the node itself.
    tree = sample_tree()
    order, start, size = _subtree_blocks(tree.dA)
    idpar = idpar_tree(tree, root_self=False)
    counted = np.ones(tree.n_nodes, dtype=int)
    for node in order[::-1]:
        parent = idpar[node]
        if parent != NO_PARENT:
            counted[parent] += counted[node]
    np.testing.assert_array_equal(size, counted)
    assert np.all(order[start] == np.arange(tree.n_nodes))


def test_subtree_blocks_root_covers_whole_tree():
    tree = _chain_to_branch_tree()
    order, start, size = _subtree_blocks(tree.dA)
    root = 0
    assert size[root] == tree.n_nodes
    assert set(order[start[root] : start[root] + size[root]].tolist()) == set(
        range(tree.n_nodes)
    )


def test_subtree_blocks_leaf_is_just_itself():
    tree = _chain_to_branch_tree()  # 3 and 4 are terminals
    order, start, size = _subtree_blocks(tree.dA)
    for leaf in (3, 4):
        assert size[leaf] == 1
        assert order[start[leaf]] == leaf


# ---------------------------------------------------------------------------
# invariants pinning the O(n) rewrites of PL_tree / LO_tree / Pvec_tree /
# sub_tree (these replaced MATLAB's sparse matrix-power transliterations --
# see the functions' docstrings). Each checks the quantity's *definition*,
# independently of how it is computed.
# ---------------------------------------------------------------------------


def test_PL_tree_equals_depth_walked_through_parents():
    tree = sample_tree()
    PL = PL_tree(tree)
    idpar = idpar_tree(tree, root_self=False)
    for node in range(0, tree.n_nodes, 37):      # sample for speed
        depth, cur = 0, node
        while idpar[cur] != NO_PARENT:
            cur = idpar[cur]
            depth += 1
        assert PL[node] == depth, f"node {node}"


def test_PL_tree_root_is_zero_and_children_are_one():
    tree = _branchy_tree()
    PL = PL_tree(tree)
    assert PL[0] == 0
    np.testing.assert_array_equal(PL, [0, 1, 1, 2, 2])


def test_LO_tree_equals_PL_plus_descendant_PL_sum():
    # LO_tree's definition: own path length + path lengths of all descendants.
    # child_tree(tree, v) independently computes "sum of v over descendants",
    # so this cross-checks LO against a different implementation.
    for tree in (_branchy_tree(), sample_tree()):
        PL = PL_tree(tree)
        np.testing.assert_allclose(LO_tree(tree), PL + child_tree(tree, PL))


def test_LO_tree_leaf_equals_its_own_path_length():
    tree = _branchy_tree()
    LO, PL = LO_tree(tree), PL_tree(tree)
    for leaf in np.flatnonzero(T_tree(tree)):
        assert LO[leaf] == PL[leaf]     # no descendants to add


def test_Pvec_tree_matches_explicit_ancestor_walk():
    from pytrees import len_tree

    tree = sample_tree()
    v = len_tree(tree)
    P = Pvec_tree(tree, v)
    idpar = idpar_tree(tree, root_self=False)
    for node in range(0, tree.n_nodes, 53):
        total, cur = 0.0, node
        while cur != NO_PARENT:
            total += v[cur]
            cur = idpar[cur]
        assert P[node] == pytest.approx(total, rel=1e-9), f"node {node}"


def test_Pvec_tree_with_ones_is_depth_plus_one():
    tree = sample_tree()
    np.testing.assert_allclose(
        Pvec_tree(tree, np.ones(tree.n_nodes)), PL_tree(tree) + 1
    )


def test_sub_tree_is_consistent_with_parent_chain():
    # a node is in sub_tree(inode) iff inode lies on its path to the root
    tree = hss_tree()  # the big sample: enough depth to make this meaningful
    idpar = idpar_tree(tree, root_self=False)
    for inode in (0, 17, 400, 1500):
        mask = sub_tree(tree, inode).mask
        for node in range(0, tree.n_nodes, 101):
            cur, found = node, False
            while cur != NO_PARENT:
                if cur == inode:
                    found = True
                    break
                cur = idpar[cur]
            assert bool(mask[node]) == found, f"sub_tree({inode})[{node}]"


def test_sub_tree_of_root_is_everything():
    tree = sample_tree()
    assert sub_tree(tree, 0).mask.all()
