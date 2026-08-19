"""Tests for pytrees.construct: synthetic tree generation.

MST_tree is checked two ways: exact topology on a small deterministic
point set (a straight line, where the greedy nearest-neighbor result is
unambiguous), and a qualitative check on a larger random point cloud that
the balancing factor `bf` actually trades wiring cost against path length
the way the algorithm is supposed to (this is the whole point of the
Cuntz et al. algorithm, and a real way to catch a broken cost function
that unit tests on tiny trees wouldn't).
"""

import numpy as np
import pytest
from scipy import sparse

from pytrees.construct import _smoothbranch
from pytrees import (
    B_tree,
    BCT_tree,
    MST_tree,
    PL_tree,
    T_tree,
    Tree,
    allBCTs_tree,
    allBTs_tree,
    cap_tree,
    clean_tree,
    idpar_tree,
    isBCT_tree,
    jitter_tree,
    len_tree,
    quaddiameter_tree,
    quadfit_tree,
    resample_tree,
    sample_tree,
    smooth_tree,
    soma_tree,
    ver_tree,
)


# ---------------------------------------------------------------------------
# MST_tree
# ---------------------------------------------------------------------------


def test_MST_tree_straight_line_gives_a_chain():
    X = np.array([0.0, 10.0, 20.0, 30.0])
    Y = np.zeros(4)
    tree, connected = MST_tree(X, Y, start=0, bf=0.0, thr=100.0, full_output=True)[:2]
    assert np.all(connected)
    np.testing.assert_array_equal(idpar_tree(tree), [0, 0, 1, 2])
    assert ver_tree(tree, quiet=True) == []


def test_MST_tree_respects_distance_threshold():
    X = np.array([0.0, 5.0, 100.0])  # last point far beyond thr
    Y = np.zeros(3)
    tree, connected = MST_tree(X, Y, start=0, thr=10.0, full_output=True)[:2]
    assert connected.tolist() == [True, True, False]
    assert tree.n_nodes == 2


def test_MST_tree_respects_max_path_length():
    X = np.array([0.0, 40.0, 80.0])  # chain would need path length 80
    Y = np.zeros(3)
    tree, connected = MST_tree(X, Y, start=0, thr=100.0, mplen=50.0, full_output=True)[:2]
    # third point (path length 80 via the chain) must be excluded
    assert connected[2] == False  # noqa: E712
    assert PL_tree(tree).max() <= 50.0 + 1e-6 or tree.n_nodes < 3


def test_MST_tree_balancing_factor_trades_wire_for_path_length():
    rng = np.random.default_rng(0)
    X = np.concatenate([[0.0], rng.uniform(0, 400, 300)])
    Y = np.concatenate([[0.0], rng.uniform(0, 400, 300)])
    Z = np.zeros(301)

    tree_wire, _ = MST_tree(X, Y, Z, bf=0.0, thr=50.0, full_output=True)[:2]  # minimize wiring
    tree_path, _ = MST_tree(X, Y, Z, bf=1.0, thr=50.0, full_output=True)[:2]  # minimize path length

    assert len_tree(tree_wire).sum() < len_tree(tree_path).sum()
    assert PL_tree(tree_wire).max() > PL_tree(tree_path).max()


def test_MST_tree_avoid_multifurcations_caps_children_at_two():
    rng = np.random.default_rng(2)
    X = np.concatenate([[0.0], rng.uniform(0, 200, 150)])
    Y = np.concatenate([[0.0], rng.uniform(0, 200, 150)])
    tree, _ = MST_tree(X, Y, bf=0.4, thr=50.0, avoid_multifurcations=True, full_output=True)[:2]
    children_count = np.asarray(tree.dA.sum(axis=0)).ravel()
    assert children_count.max() <= 2


# ---------------------------------------------------------------------------
# BCT strings
# ---------------------------------------------------------------------------


def test_isBCT_tree_valid_and_invalid():
    assert isBCT_tree([1, 2, 1, 0, 2, 0, 0])
    assert not isBCT_tree([1, 1, 1, 1])  # no termination
    assert isBCT_tree([1, 1, 1, 1, 0])  # termination makes it valid


def test_BCT_tree_matches_input_sequence():
    bct = [1, 2, 1, 0, 2, 0, 0]
    tree = BCT_tree(bct)
    typeN = np.asarray(tree.dA.sum(axis=0)).ravel()
    np.testing.assert_array_equal(typeN, bct)
    assert ver_tree(tree, quiet=True) == []


def test_BCT_tree_rejects_non_conform_sequence():
    with pytest.raises(ValueError):
        BCT_tree([1, 1, 1, 1])


def test_allBCTs_tree_all_rows_are_valid_trees_and_unique():
    bcts = allBCTs_tree(6)
    assert bcts.shape[1] == 6
    assert len(np.unique(bcts, axis=0)) == bcts.shape[0]
    for row in bcts:
        assert isBCT_tree(row)
        assert row.sum() == 5  # N-1 edges


def test_allBTs_tree_only_branch_and_terminal():
    bts = allBTs_tree(9)
    assert set(np.unique(bts).tolist()) <= {0, 2}
    for row in bts:
        assert isBCT_tree(row)


def test_allBCTs_tree_with_trees_returns_matching_trees():
    bcts, trees = allBCTs_tree(5, with_trees=True)
    assert len(trees) == bcts.shape[0]
    for row, tree in zip(bcts, trees):
        np.testing.assert_array_equal(np.asarray(tree.dA.sum(axis=0)).ravel(), row)


# ---------------------------------------------------------------------------
# cleanup / shaping
# ---------------------------------------------------------------------------


def _branch_with_redundant_terminal() -> Tree:
    # 0 -> 1 -> 2 (main branch), and 1 -> 3, where node 3 is a short spurious
    # terminal ending right next to node 2's path
    dA = sparse.csr_matrix(([1, 1, 1], ([1, 2, 3], [0, 1, 1])), shape=(4, 4))
    return Tree(
        dA=dA,
        X=np.array([0.0, 10.0, 20.0, 10.5]),
        Y=np.array([0.0, 0.0, 0.0, 0.1]),
        Z=np.zeros(4),
        D=np.array([2.0, 2.0, 2.0, 2.0]),
        R=np.zeros(4, dtype=int),
        rnames=["a"],
    )


def test_clean_tree_removes_spurious_close_terminal():
    tree = _branch_with_redundant_terminal()
    cleaned = clean_tree(tree, radius=1.0)
    assert cleaned.n_nodes == 3
    assert ver_tree(cleaned, quiet=True) == []


def test_clean_tree_removes_short_terminal_branch():
    # root is a branch point: one healthy long branch, one short stub
    # ("short terminal branch" is measured as the whole branch from the
    # nearest branch point to the tip, not a single segment)
    dA = sparse.csr_matrix(([1, 1], ([1, 2], [0, 0])), shape=(3, 3))
    tree = Tree(
        dA=dA, X=np.array([0.0, 100.0, 0.3]), Y=np.zeros(3), Z=np.zeros(3),
        D=np.ones(3), R=np.zeros(3, dtype=int), rnames=["a"],
    )
    cleaned = clean_tree(tree, radius=1.0)
    assert cleaned.n_nodes == 2
    assert cleaned.X.max() == pytest.approx(100.0)  # the healthy branch survives


def test_clean_tree_leaves_well_separated_tree_unchanged():
    # two long branches, far apart in space and both well over `radius` in
    # length -- neither the "close to another branch" nor "too short"
    # criterion can trigger regardless of the (small) diameter-based term
    dA = sparse.csr_matrix(([1, 1], ([1, 2], [0, 0])), shape=(3, 3))
    tree = Tree(
        dA=dA, X=np.array([0.0, 100.0, -100.0]), Y=np.zeros(3), Z=np.zeros(3),
        D=np.full(3, 0.5), R=np.zeros(3, dtype=int), rnames=["a"],
    )
    cleaned = clean_tree(tree, radius=0.01)
    assert cleaned.n_nodes == tree.n_nodes


def test_clean_tree_runs_on_real_reconstruction_without_error():
    # D/2 alone (independent of `radius`) can legitimately flag naturally
    # close-together branches in a real, densely-branching reconstruction,
    # so this only checks it runs cleanly and never disconnects the tree,
    # not that it's a no-op.
    tree = sample_tree()
    cleaned = clean_tree(tree, radius=0.01)
    assert cleaned.n_nodes <= tree.n_nodes
    assert ver_tree(cleaned, quiet=True) == []


def test_soma_tree_cosine_profile_and_region_tagging():
    dA = sparse.csr_matrix(([1, 1], ([1, 2], [0, 1])), shape=(3, 3))
    tree = Tree(
        dA=dA, X=np.array([0.0, 10.0, 20.0]), Y=np.zeros(3), Z=np.zeros(3),
        D=np.array([1.0, 1.0, 1.0]), R=np.zeros(3, dtype=int), rnames=["dend"],
    )
    result = soma_tree(tree, maxD=10.0, length=20.0, tag_region=True)
    # at the root (Plen=0), cosine profile gives maxD * cos(0) = maxD
    assert result.D[0] == pytest.approx(10.0)
    assert result.rnames == ["dend", "soma"]
    assert result.rnames[result.R[0]] == "soma"
    # original diameter is never reduced below its own value
    assert np.all(result.D >= tree.D)


def test_cap_tree_adds_nodes_with_valid_topology():
    tree = soma_tree(
        resample_tree(sample_tree(), 2.0), maxD=20.0, tag_region=True
    )
    capped = cap_tree(tree, spacing=1.0)
    assert capped.n_nodes > tree.n_nodes
    assert ver_tree(capped, quiet=True) == []
    # new nodes share the root's region
    assert np.all(capped.R[tree.n_nodes:] == tree.R[0])


def test_cap_tree_no_op_when_diameter_too_small():
    dA = sparse.csr_matrix(([1], ([1], [0])), shape=(2, 2))
    tree = Tree(
        dA=dA, X=np.array([0.0, 1.0]), Y=np.zeros(2), Z=np.zeros(2),
        D=np.array([0.01, 0.01]), R=np.zeros(2, dtype=int), rnames=["a"],
    )
    result = cap_tree(tree, spacing=1.0)
    assert result.n_nodes == tree.n_nodes


# ---------------------------------------------------------------------------
# jitter_tree / smooth_tree / _smoothbranch
# ---------------------------------------------------------------------------


def test_jitter_tree_zero_noise_is_a_no_op():
    tree = resample_tree(sample_tree(), 5.0)
    jittered = jitter_tree(tree, stde=0.0, lam=5)
    np.testing.assert_allclose(jittered.X, tree.X)
    np.testing.assert_allclose(jittered.Y, tree.Y)
    np.testing.assert_allclose(jittered.Z, tree.Z)


def test_jitter_tree_root_never_moves_and_shape_changes():
    tree = resample_tree(sample_tree(), 5.0)
    jittered = jitter_tree(tree, stde=2.0, lam=5, rng=np.random.default_rng(0))
    assert jittered.X[0] == pytest.approx(tree.X[0])
    assert jittered.Y[0] == pytest.approx(tree.Y[0])
    assert jittered.Z[0] == pytest.approx(tree.Z[0])
    assert not np.allclose(jittered.X, tree.X)
    assert ver_tree(jittered, quiet=True) == []


def test_private_smoothbranch_straightens_a_zigzag_with_full_smoothing():
    X = np.array([0.0, 1.0, 2.0])
    Y = np.array([0.0, 1.0, 0.0])  # zigzag peak at the midpoint
    Z = np.zeros(3)
    Xs, Ys, Zs = _smoothbranch(X, Y, Z, p=1.0, n=1)
    # endpoints preserved, midpoint pulled fully onto the line X1-X3 (Y=0)
    assert Xs[0] == pytest.approx(0.0) and Xs[-1] == pytest.approx(2.0)
    assert Ys[1] == pytest.approx(0.0, abs=1e-9)


def test_private_smoothbranch_short_path_is_unchanged():
    X, Y, Z = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.zeros(2)
    Xs, Ys, Zs = _smoothbranch(X, Y, Z, p=1.0, n=3)
    np.testing.assert_array_equal(Xs, X)
    np.testing.assert_array_equal(Ys, Y)


def test_smooth_tree_preserves_endpoints_and_reduces_or_keeps_length():
    tree = resample_tree(sample_tree(), 5.0)
    smoothed = smooth_tree(tree, p=0.9, n=3)
    assert ver_tree(smoothed, quiet=True) == []
    assert smoothed.X[0] == pytest.approx(tree.X[0])
    # smoothing shortens (or at worst preserves) total cable length
    assert len_tree(smoothed).sum() <= len_tree(tree).sum() + 1e-6


# ---------------------------------------------------------------------------
# quaddiameter_tree / quadfit_tree
# ---------------------------------------------------------------------------


def test_quaddiameter_tree_produces_valid_positive_diameters():
    tree = resample_tree(sample_tree(), 10.0)
    qd = quaddiameter_tree(tree, scale=0.5, offset=0.5)
    assert np.all(qd.D > 0)
    assert ver_tree(qd, quiet=True) == []


def test_quaddiameter_tree_tapers_along_a_path():
    tree = resample_tree(sample_tree(), 10.0)
    qd = quaddiameter_tree(tree)
    # diameter at the root should exceed the mean diameter at terminals
    # (taper narrows toward the tips)
    terminal_D = qd.D[T_tree(qd)]
    assert qd.D[0] >= terminal_D.mean()


def test_quadfit_tree_returns_reasonable_fit():
    dA = sparse.csr_matrix(([1, 1], ([1, 2], [0, 1])), shape=(3, 3))
    tree = Tree(
        dA=dA, X=np.array([0.0, 20.0, 40.0]), Y=np.zeros(3), Z=np.zeros(3),
        D=np.array([5.0, 3.0, 1.0]), R=np.zeros(3, dtype=int), rnames=["a"],
    )
    scale, offset, fitted = quadfit_tree(tree)
    assert np.isfinite(scale) and np.isfinite(offset)
    assert fitted.n_nodes == tree.n_nodes
    assert ver_tree(fitted, quiet=True) == []
