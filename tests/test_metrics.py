"""Tests for pynetrees.metrics: coordinate-based metrics and transforms.

Hand-verified against a small fixed tree (`_geom_tree`), same topology as
graphtheory's `_branchy_tree` (0 -> {1, 2}, 1 -> {3, 4}) but with real 3D
coordinates chosen to give clean, hand-computable lengths/angles:

    node0 (root): (1, 2, 3),            D=4
    node1: node0 + (3, 4, 0)  -> len 5, D=2
    node2: node0 + (0, 0, 5)  -> len 5, D=2
    node3: node1 + (0, 0, 3)  -> len 3, D=1
    node4: node1 + (3, 0, 0)  -> len 3, D=1

Root is deliberately *not* at the coordinate origin, so tests can
distinguish "centered on root" from "centered on the origin" behavior.
"""

import numpy as np
import pytest
from scipy import sparse

from pynetrees import (
    Tree,
    angleB_tree,
    bin_tree,
    cvol_tree,
    cyl_tree,
    dist_tree,
    direction_tree,
    eucl_tree,
    flatten_tree,
    flip_tree,
    gene_tree,
    len_tree,
    morph_tree,
    rot_tree,
    scale_tree,
    surf_tree,
    tran_tree,
    vol_tree,
    zcorr_tree,
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


# ---------------------------------------------------------------------------
# segment geometry
# ---------------------------------------------------------------------------


def test_len_tree():
    tree = _geom_tree()
    np.testing.assert_allclose(len_tree(tree), [0, 5, 5, 3, 3])


def test_cyl_tree_root_is_degenerate_point():
    tree = _geom_tree()
    X1, X2, Y1, Y2, Z1, Z2 = cyl_tree(tree)
    assert X1[0] == X2[0] == pytest.approx(1.0)


def test_surf_and_vol_tree_cylinder():
    tree = _geom_tree()
    np.testing.assert_allclose(surf_tree(tree), np.pi * tree.D * len_tree(tree))
    np.testing.assert_allclose(vol_tree(tree), np.pi * len_tree(tree) * tree.D**2 / 4)


def test_surf_and_vol_tree_frustum():
    tree = _geom_tree()
    tree.frustum = True
    idpar = [0, 0, 0, 1, 1]
    D, Dp, length = tree.D, tree.D[idpar], len_tree(tree)
    expected_surf = np.pi * (D + Dp) / 2 * np.sqrt(length**2 + (D - Dp) ** 2 / 4)
    expected_vol = np.pi * length * (D**2 + D * Dp + Dp**2) / 12
    np.testing.assert_allclose(surf_tree(tree), expected_surf)
    np.testing.assert_allclose(vol_tree(tree), expected_vol)


def test_cvol_tree_clamps_zero_length_root():
    tree = _geom_tree()
    cvol = cvol_tree(tree)
    assert cvol[0] == pytest.approx(0.0001)
    assert np.all(cvol[1:] > 0)


# ---------------------------------------------------------------------------
# distances, directions, angles
# ---------------------------------------------------------------------------


def test_eucl_tree_default_is_distance_from_root():
    tree = _geom_tree()
    eucl = eucl_tree(tree)
    np.testing.assert_allclose(eucl, [0, 5, 5, np.sqrt(34), np.sqrt(52)])


def test_eucl_tree_explicit_point_and_2d():
    tree = _geom_tree()
    eucl2d = eucl_tree(tree, point=1, dim=2)
    # distance from node1 to itself is 0
    assert eucl2d[1] == pytest.approx(0.0)


def test_direction_tree_normalized():
    tree = _geom_tree()
    direction = direction_tree(tree)
    np.testing.assert_allclose(direction[1], [0.6, 0.8, 0.0])
    np.testing.assert_allclose(direction[2], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(direction[4], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(direction[0], direction[1])  # root placeholder


def test_angleB_tree_right_angles():
    tree = _geom_tree()
    angleB = angleB_tree(tree)
    assert angleB[0] == pytest.approx(np.pi / 2)
    assert angleB[1] == pytest.approx(np.pi / 2)
    assert np.isnan(angleB[2]) and np.isnan(angleB[3]) and np.isnan(angleB[4])


def test_angleB_tree_rejects_trifurcation():
    dA = sparse.csr_matrix(([1, 1, 1], ([1, 2, 3], [0, 0, 0])), shape=(4, 4))
    tree = Tree(
        dA=dA, X=np.zeros(4), Y=np.zeros(4), Z=np.zeros(4),
        D=np.ones(4), R=np.zeros(4, dtype=int), rnames=["a"],
    )
    with pytest.raises(ValueError, match="binary"):
        angleB_tree(tree)


# ---------------------------------------------------------------------------
# rigid transforms
# ---------------------------------------------------------------------------


def test_scale_tree_centered_on_root():
    tree = _geom_tree()
    scaled = scale_tree(tree, fac=2.0, center=True)
    np.testing.assert_allclose([scaled.X[0], scaled.Y[0], scaled.Z[0]], [1, 2, 3])
    np.testing.assert_allclose(
        [scaled.X[1], scaled.Y[1], scaled.Z[1]], [1 + 6, 2 + 8, 3]
    )
    assert scaled.D[1] == pytest.approx(4.0)


def test_scale_tree_not_centered():
    tree = _geom_tree()
    scaled = scale_tree(tree, fac=2.0, center=False)
    np.testing.assert_allclose([scaled.X[0], scaled.Y[0], scaled.Z[0]], [2, 4, 6])


def test_scale_tree_no_diameter_scaling():
    tree = _geom_tree()
    scaled = scale_tree(tree, fac=2.0, scale_diameter=False)
    np.testing.assert_allclose(scaled.D, tree.D)


def test_tran_tree_default_centers_root_at_origin():
    tree = _geom_tree()
    centered = tran_tree(tree)
    np.testing.assert_allclose([centered.X[0], centered.Y[0], centered.Z[0]], [0, 0, 0])
    np.testing.assert_allclose([centered.X[1], centered.Y[1]], [3, 4])


def test_tran_tree_by_vector():
    tree = _geom_tree()
    moved = tran_tree(tree, [5.0, 5.0, 5.0])
    np.testing.assert_allclose([moved.X[0], moved.Y[0], moved.Z[0]], [6, 7, 8])


def test_tran_tree_recenter_on_node():
    tree = _geom_tree()
    recentered = tran_tree(tree, 1)
    np.testing.assert_allclose(
        [recentered.X[0], recentered.Y[0], recentered.Z[0]], [-3, -4, 0]
    )
    np.testing.assert_allclose(
        [recentered.X[1], recentered.Y[1], recentered.Z[1]], [0, 0, 0]
    )


def test_rot_tree_2d_and_3d_agree():
    tree = tran_tree(_geom_tree())  # centered at origin for a clean check
    rotated_2d = rot_tree(tree, deg=90.0)
    rotated_3d = rot_tree(tree, deg=(0.0, 0.0, 90.0))
    np.testing.assert_allclose(rotated_2d.X[1], rotated_3d.X[1])
    np.testing.assert_allclose(rotated_2d.Y[1], rotated_3d.Y[1])
    np.testing.assert_allclose([rotated_3d.X[1], rotated_3d.Y[1]], [4.0, -3.0], atol=1e-10)
    np.testing.assert_allclose(rotated_3d.Z, tree.Z)  # z-only rotation leaves Z alone


def test_flip_tree_around_x():
    tree = tran_tree(_geom_tree())
    flipped = flip_tree(tree, axis="x")
    np.testing.assert_allclose(flipped.X, -tree.X)
    np.testing.assert_allclose(flipped.Y, tree.Y)


def test_flip_tree_rejects_bad_axis():
    tree = _geom_tree()
    with pytest.raises(ValueError):
        flip_tree(tree, axis="w")


# ---------------------------------------------------------------------------
# shape-preserving / shape-correcting transforms
# ---------------------------------------------------------------------------


def test_flatten_tree_preserves_segment_lengths():
    dA = sparse.csr_matrix(([1], ([1], [0])), shape=(2, 2))
    tree = Tree(
        dA=dA, X=np.array([0.0, 3.0]), Y=np.array([0.0, 4.0]), Z=np.array([0.0, 5.0]),
        D=np.array([1.0, 1.0]), R=np.zeros(2, dtype=int), rnames=["a"],
    )
    original_len = len_tree(tree)
    flat = flatten_tree(tree)
    np.testing.assert_allclose(flat.Z, 0.0, atol=1e-9)
    np.testing.assert_allclose(len_tree(flat), original_len, atol=1e-9)


def test_morph_tree_sets_uniform_segment_length():
    tree = _geom_tree()
    morphed = morph_tree(tree, v=np.full(5, 10.0))
    np.testing.assert_allclose(len_tree(morphed)[1:], 10.0, atol=1e-9)


def test_morph_tree_round_trip_recovers_original():
    tree = _geom_tree()
    original_len = len_tree(tree)
    morphed = morph_tree(tree, v=np.full(5, 10.0))
    recovered = morph_tree(morphed, v=original_len)
    np.testing.assert_allclose(recovered.X, tree.X, atol=1e-8)
    np.testing.assert_allclose(recovered.Y, tree.Y, atol=1e-8)
    np.testing.assert_allclose(recovered.Z, tree.Z, atol=1e-8)


def test_zcorr_tree_detects_and_corrects_z_jump():
    dA = sparse.csr_matrix(([1, 1], ([1, 2], [0, 1])), shape=(3, 3))
    tree = Tree(
        dA=dA,
        X=np.zeros(3), Y=np.zeros(3), Z=np.array([0.0, 2.0, 20.0]),
        D=np.ones(3), R=np.zeros(3, dtype=int), rnames=["a"],
    )
    corrected, jumped = zcorr_tree(tree, tz=5.0)
    np.testing.assert_array_equal(jumped, [2])
    assert corrected.Z[2] == pytest.approx(2.0)
    assert corrected.Z[1] == pytest.approx(2.0)  # untouched, jump was downstream


# ---------------------------------------------------------------------------
# dist_tree / bin_tree / gene_tree (filed under MATLAB's "graphtheory", live
# here because they need len_tree/eucl_tree -- see module docstring)
# ---------------------------------------------------------------------------


def _chain_branch_tree() -> Tree:
    # 0 --(C, len 5)--> 1 --(C, len 10)--> 2 --(B)--> {3 (T, len 10), 4 (T, len 10)}
    dA = sparse.csr_matrix(
        ([1, 1, 1, 1], ([1, 2, 3, 4], [0, 1, 2, 2])), shape=(5, 5)
    )
    return Tree(
        dA=dA,
        X=np.array([0.0, 5.0, 15.0, 15.0, 15.0]),
        Y=np.array([0.0, 0.0, 0.0, 10.0, -10.0]),
        Z=np.zeros(5),
        D=np.ones(5),
        R=np.zeros(5, dtype=int),
        rnames=["a"],
    )


def test_dist_tree_crossing_mask():
    tree = _chain_branch_tree()  # path lengths from root: 0, 5, 15, 25, 25
    crossing_10 = dist_tree(tree, 10.0)[:, 0]
    np.testing.assert_array_equal(crossing_10, [False, False, True, False, False])
    crossing_20 = dist_tree(tree, 20.0)[:, 0]
    np.testing.assert_array_equal(crossing_20, [False, False, False, True, True])


def test_bin_tree_explicit_v_and_bins_matches_digitize_contract():
    tree = _chain_branch_tree()
    v = np.array([0.0, 5.0, 15.0, 25.0, 25.0])
    bins = np.array([0.0, 10.0, 20.0, 30.0])
    bin_index, edges = bin_tree(tree, v=v, bins=bins)
    expected = np.digitize(v, bins)
    expected[(v < bins[0]) | (v > bins[-1])] = 0
    np.testing.assert_array_equal(bin_index, expected)
    np.testing.assert_array_equal(edges, bins)


def test_bin_tree_default_uses_euclidean_distance_and_bin_count():
    tree = _chain_branch_tree()
    bin_index, edges = bin_tree(tree, bins=4)
    assert len(edges) == 5
    assert bin_index.shape == (5,)
    assert np.all(bin_index >= 0)


def test_gene_tree_matches_hand_computed_branch_lengths():
    tree = _chain_branch_tree()
    gene = gene_tree(tree)
    rows = sorted(gene.tolist())
    np.testing.assert_allclose(rows, [[10.0, 0.0], [10.0, 0.0], [15.0, 2.0]])
