"""Tests for pytrees.electrotonics.

`_cable_tree()` is a minimal, fully hand-computable fixture: a 2-node cable,
root (D=10um) -> child (D=2um) at 100um along X, with Ri=100 Ohm*cm and
Gm=1/2500 S/cm^2 (the exact values every MATLAB `check_*` fixture in this
folder uses). Every scalar below (surf/cvol/gi/gm/lambda/elen/M's four
entries) was computed by hand from the ported formulas -- see the PR/commit
message or PORT_STATUS.md Design Decision for the full derivation -- so
these are exact-value assertions, not just "runs without crashing" smoke
tests.

`M_tree`'s conductance-matrix construction itself is ported as a literal
transliteration of MATLAB's sparse matrix algebra, not re-derived from cable
theory (see electrotonics.py's module docstring): the tests here verify the
*translation* is faithful (exact hand-computed entries on the small fixture,
plus the Kirchhoff zero-row-sum invariant every valid conductance-Laplacian
must satisfy on the real bundled reconstruction), not an independent
re-derivation of the underlying physical convention.
"""

import numpy as np
import pytest
from scipy import sparse

from pytrees import (
    Tree,
    M_tree,
    cgin_tree,
    elen_tree,
    gi_tree,
    gm_tree,
    lambda_tree,
    loop_tree,
    sample_tree,
    sse_tree,
    ssecat_tree,
    syn_tree,
    syncat_tree,
)


def _cable_tree() -> Tree:
    dA = sparse.csr_matrix(([1], ([1], [0])), shape=(2, 2))
    tree = Tree(
        dA=dA,
        X=np.array([0.0, 100.0]),
        Y=np.zeros(2),
        Z=np.zeros(2),
        D=np.array([10.0, 2.0]),
        R=np.zeros(2, dtype=int),
        rnames=["dend"],
    )
    tree.Ri = 100.0
    tree.Gm = 1.0 / 2500.0
    return tree


# hand-computed reference values for _cable_tree(), Ri=100, Gm=1/2500
_SURF1_CM2 = np.pi * 2.0 * 100.0 / 1e8
_GM1 = (1.0 / 2500.0) * _SURF1_CM2
_CVOL1_CM = (4.0 * 100.0 / (np.pi * 2.0**2)) * 1e4
_GI1 = (1.0 / _CVOL1_CM) / 100.0
_GI0 = 1.0 / 100.0  # root: cvol clamped to 0.0001 [1/um] -> 1 [1/cm]
_LAMBDA1 = np.sqrt((2.0 / 4.0) / (10000 * (1.0 / 2500.0) * 100.0))
_ELEN1 = 100.0 / _LAMBDA1 / 10000.0


def test_gi_tree_hand_verified():
    tree = _cable_tree()
    gi = gi_tree(tree)
    np.testing.assert_allclose(gi, [_GI0, _GI1], rtol=1e-6)


def test_gm_tree_hand_verified():
    tree = _cable_tree()
    gm = gm_tree(tree)
    np.testing.assert_allclose(gm, [0.0, _GM1], rtol=1e-6)


def test_lambda_tree_hand_verified():
    tree = _cable_tree()
    lam = lambda_tree(tree)
    assert lam[1] == pytest.approx(_LAMBDA1, rel=1e-9)


def test_elen_tree_hand_verified():
    tree = _cable_tree()
    elen = elen_tree(tree)
    assert elen[0] == 0.0  # root segment has zero length
    assert elen[1] == pytest.approx(_ELEN1, rel=1e-6)


def test_cgin_tree_equals_sum_of_gm():
    # collapsed input conductance = Gm * total surface -- same quantity as
    # summing every segment's own membrane conductance
    tree = _cable_tree()
    assert cgin_tree(tree) == pytest.approx(gm_tree(tree).sum(), rel=1e-9)


def test_electrotonics_functions_require_Ri_Gm():
    tree = _cable_tree()
    tree.Ri = None
    with pytest.raises(ValueError, match="Ri"):
        gi_tree(tree)


def test_M_tree_hand_verified_entries():
    tree = _cable_tree()
    M = M_tree(tree).toarray()
    # off-diagonal (axial) term: -1/(Ri) * 1/cvol(root, clamped) = -0.01,
    # scaled by 1e6 -> -10000; diagonal picks up the same magnitude plus
    # each node's own membrane conductance (Mgm, scaled by 1e6)
    expected_axial = _GI0 * 1e6  # = 10000
    assert M[0, 1] == pytest.approx(-expected_axial, rel=1e-6)
    assert M[1, 0] == pytest.approx(-expected_axial, rel=1e-6)
    assert M[0, 0] == pytest.approx(expected_axial, rel=1e-6)  # gm[0] == 0
    assert M[1, 1] == pytest.approx(expected_axial + _GM1 * 1e6, rel=1e-6)


def test_M_tree_symmetric_with_scalar_Ri():
    tree = _cable_tree()
    M = M_tree(tree).toarray()
    np.testing.assert_allclose(M, M.T)


def test_M_tree_axial_rows_sum_to_zero_kirchhoff():
    # with no membrane leak (Gm=0), the axial part alone is a graph
    # Laplacian: current into every node from its neighbors must balance to
    # zero (Kirchhoff's current law) -- true regardless of the specific
    # per-segment cvol convention, so this is a check on the *translation*,
    # not on the underlying cable-theory choice (see module docstring).
    tree = sample_tree()
    tree.Ri = 100.0
    tree.Gm = 0.0
    M = M_tree(tree).toarray()
    row_sums = M.sum(axis=1)
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-6)


def test_sse_tree_is_the_matrix_inverse_of_M():
    tree = sample_tree()
    tree.Ri = 100.0
    tree.Gm = 1.0 / 2500.0
    M = M_tree(tree)
    sse = sse_tree(tree)
    assert sse.shape == (tree.n_nodes, tree.n_nodes)
    np.testing.assert_allclose(sse, sse.T, atol=1e-8)  # sse is symmetric
    identity = M.toarray() @ sse
    np.testing.assert_allclose(identity, np.eye(tree.n_nodes), atol=1e-6)


def test_sse_tree_scalar_I_is_onehot_column():
    tree = _cable_tree()
    sse_full = sse_tree(tree)
    sse_col1 = sse_tree(tree, I=1)
    np.testing.assert_allclose(sse_col1, sse_full[:, 1], rtol=1e-6)


def test_syn_tree_zero_input_gives_zero_potential():
    tree = _cable_tree()
    v = syn_tree(tree)
    np.testing.assert_allclose(v, 0.0, atol=1e-12)


def test_syn_tree_current_only_matches_sse_tree():
    tree = sample_tree()
    tree.Ri = 100.0
    tree.Gm = 1.0 / 2500.0
    v_syn = syn_tree(tree, I=10)
    v_sse = sse_tree(tree, I=10)
    np.testing.assert_allclose(v_syn, v_sse, rtol=1e-6)


def test_syn_tree_matches_independent_dense_solve():
    tree = _cable_tree()
    ge = np.array([0.0, 1.0])
    v = syn_tree(tree, ge=ge, Ee=60.0)
    M_dense = M_tree(tree).toarray()
    expected = np.linalg.solve(M_dense + np.diag(ge), ge * 60.0)
    np.testing.assert_allclose(v, expected, rtol=1e-6)


def test_loop_tree_adds_exact_conductance_at_four_entries():
    tree = sample_tree()
    tree.Ri = 100.0
    tree.Gm = 1.0 / 2500.0
    M = M_tree(tree)
    looped = loop_tree(tree, 0, 5, gelsyn=2.0)
    delta = (looped - M).toarray()
    assert delta[0, 5] == pytest.approx(-2.0)
    assert delta[5, 0] == pytest.approx(-2.0)
    assert delta[0, 0] == pytest.approx(2.0)
    assert delta[5, 5] == pytest.approx(2.0)


def test_loop_tree_zero_conductance_is_a_no_op():
    tree = _cable_tree()
    M = M_tree(tree).toarray()
    looped = loop_tree(tree, 0, 1, gelsyn=0.0).toarray()
    np.testing.assert_allclose(looped, M)


def test_ssecat_tree_single_tree_matches_sse_tree():
    tree = sample_tree()
    tree.Ri = 100.0
    tree.Gm = 1.0 / 2500.0
    sse = sse_tree(tree)
    sse_cat = ssecat_tree([tree], 0, 5, gelsyn=0.0)
    np.testing.assert_allclose(sse_cat, sse, rtol=1e-6)


def test_syncat_tree_single_tree_matches_syn_tree():
    tree = sample_tree()
    tree.Ri = 100.0
    tree.Gm = 1.0 / 2500.0
    v_syn = syn_tree(tree, I=10)
    v_cat = syncat_tree([tree], 0, 5, gelsyn=0.0, I=10)
    np.testing.assert_allclose(v_cat, v_syn, rtol=1e-6)


def test_ssecat_tree_offsets_index_into_concatenated_system():
    # two trees, gelsyn=0 (no real coupling): the block-diagonal structure
    # means node `t1.n_nodes + k` in the concatenated system is node k of
    # the second tree
    t1 = _cable_tree()
    t2 = _cable_tree()
    sse_cat = ssecat_tree([t1, t2], 0, 1, gelsyn=0.0)
    sse_single = sse_tree(t1)
    np.testing.assert_allclose(sse_cat[:2, :2], sse_single, rtol=1e-6)
    np.testing.assert_allclose(sse_cat[2:, 2:], sse_single, rtol=1e-6)
    np.testing.assert_allclose(sse_cat[:2, 2:], 0.0, atol=1e-12)
