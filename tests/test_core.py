"""Tests for pynetrees.core: Tree construction and ver_tree validation.

Ports the intent of treestoolbox-master/tests/IO/check_ver_tree.m (test 1:
a well-formed tree has no issues; test 2: a mismatched-size field is flagged).
"""

import numpy as np
import pytest
from scipy import sparse

import pynetrees as pt
from pynetrees import Tree, sample_tree, ver_tree


def _tiny_tree() -> Tree:
    # 3-node tree: 0 is root, 1 and 2 are children of 0.
    dA = sparse.csr_matrix(([1, 1], ([1, 2], [0, 0])), shape=(3, 3))
    return Tree(
        dA=dA,
        X=[0.0, 1.0, -1.0],
        Y=[0.0, 0.0, 0.0],
        Z=[0.0, 0.0, 0.0],
        D=[1.0, 1.0, 1.0],
        R=[0, 0, 0],
        rnames=["soma"],
    )


def test_tree_basic_attributes():
    tree = _tiny_tree()
    assert tree.n_nodes == 3
    assert len(tree) == 3
    assert tree.dA.shape == (3, 3)
    assert isinstance(tree.dA, sparse.csr_matrix)
    np.testing.assert_array_equal(tree.X, [0.0, 1.0, -1.0])


def test_sample_tree_is_valid():
    tree = sample_tree()
    assert ver_tree(tree, quiet=True) == []


def test_ver_tree_flags_size_mismatch():
    tree = _tiny_tree()
    tree.X = np.array([0.0])  # now inconsistent with dA's 3 nodes
    issues = ver_tree(tree, quiet=True)
    assert any("X" in issue for issue in issues)


def test_ver_tree_flags_non_square_dA():
    tree = _tiny_tree()
    tree.dA = sparse.csr_matrix(np.zeros((3, 2)))
    issues = ver_tree(tree, quiet=True)
    assert any("square" in issue for issue in issues)


def test_ver_tree_flags_out_of_range_region_index():
    tree = _tiny_tree()
    tree.R = np.array([0, 5, 0])  # 5 is out of range for a single-region tree
    issues = ver_tree(tree, quiet=True)
    assert any("rnames" in issue for issue in issues)


def test_ver_tree_warns_by_default():
    tree = _tiny_tree()
    tree.X = np.array([0.0])
    with pytest.warns(UserWarning):
        ver_tree(tree)


def test_validate_method_matches_function():
    tree = _tiny_tree()
    assert tree.validate(quiet=True) == ver_tree(tree, quiet=True)


# ---------------------------------------------------------------------------
# region accessors
# ---------------------------------------------------------------------------


def test_region_accessors():
    tree = _tiny_tree()
    tree.R = np.array([0, 0, 1])
    tree.rnames = ["soma", "dend"]

    assert tree.region_index("soma") == 0
    assert tree.region_index("dend") == 1
    np.testing.assert_array_equal(tree.region_nodes("soma"), [0, 1])
    np.testing.assert_array_equal(tree.region_nodes("dend"), [2])
    np.testing.assert_array_equal(tree.region_nodes("soma", "dend"), [0, 1, 2])
    np.testing.assert_array_equal(
        tree.region_mask("dend"), [False, False, True]
    )


def test_region_index_raises_with_available_names():
    # MATLAB's find(strcmp(...)) silently returns empty here, which makes the
    # *next* line match nothing instead of failing -- we raise instead
    tree = _tiny_tree()
    tree.rnames = ["soma"]
    with pytest.raises(KeyError, match="soma"):
        tree.region_index("nonexistent")


def test_repr_includes_regions():
    tree = _tiny_tree()
    tree.rnames = ["soma", "dend"]
    assert "soma" in repr(tree)


# ---------------------------------------------------------------------------
# Tree.copy / ad-hoc meta / per-node fields surviving the transforms
# ---------------------------------------------------------------------------


def _electrical_sample():
    tree = pt.repair_tree(pt.sample_tree())
    tree.Ri, tree.Gm, tree.Cm = 100.0, 1 / 2500.0, 1.0
    tree.meta["per_node"] = np.arange(tree.n_nodes, dtype=float)
    tree.meta["whole_cell"] = "GC-2026"
    return tree


def test_copy_carries_every_field_it_was_not_asked_to_change():
    tree = _electrical_sample()
    moved = tree.copy(X=tree.X + 10.0)

    np.testing.assert_allclose(moved.X, tree.X + 10.0)
    assert (moved.Ri, moved.Gm, moved.Cm) == (tree.Ri, tree.Gm, tree.Cm)
    assert moved.frustum == tree.frustum
    assert moved.meta["whole_cell"] == "GC-2026"
    np.testing.assert_array_equal(moved.meta["per_node"], tree.meta["per_node"])


def test_copy_rejects_a_field_name_it_does_not_have():
    """A typo'd field would otherwise return an unmodified tree and be
    blamed on the transform rather than on the call."""
    with pytest.raises(TypeError, match="unexpected field"):
        _electrical_sample().copy(Xs=np.zeros(3))


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(lambda t: pt.delete_tree(t, [5, 6, 7]), id="delete_tree"),
        pytest.param(lambda t: pt.resample_tree(t, 10.0), id="resample_tree"),
        pytest.param(pt.root_tree, id="root_tree"),
        pytest.param(pt.repair_tree, id="repair_tree"),
        pytest.param(pt.elimt_tree, id="elimt_tree"),
        pytest.param(lambda t: pt.sort_tree(t, by="lo"), id="sort_tree"),
        pytest.param(lambda t: pt.redirect_tree(t, 50), id="redirect_tree"),
        pytest.param(lambda t: pt.tran_tree(t, [1.0, 2.0, 3.0]), id="tran_tree"),
        pytest.param(lambda t: pt.scale_tree(t, 2.0), id="scale_tree"),
    ],
)
def test_transforms_keep_the_electrical_properties_and_meta(transform):
    """These used to be dropped: every editing function built a bare `Tree(...)`
    and simply never mentioned Ri/Gm/Cm, so `delete_tree(tree, [...])` handed
    back a tree that failed later in `build_neuron_model` with "no Ri set"."""
    tree = _electrical_sample()
    out = transform(tree)

    assert out.Ri == 100.0
    assert out.Gm == 1 / 2500.0
    assert out.Cm == 1.0
    assert out.meta["whole_cell"] == "GC-2026"
    assert len(out.meta["per_node"]) == out.n_nodes


def test_per_node_meta_follows_the_nodes_through_a_subset():
    tree = _electrical_sample()
    sub = tree.reindexed(np.arange(0, tree.n_nodes, 2))
    np.testing.assert_array_equal(
        sub.meta["per_node"], np.arange(0, tree.n_nodes, 2, dtype=float)
    )


def test_meta_of_another_length_is_left_alone():
    """Only length-n_nodes entries are per-node; anything else is metadata
    about the cell and must not be silently re-ordered."""
    tree = _electrical_sample()
    tree.meta["three_numbers"] = np.array([1.0, 2.0, 3.0])
    sub = tree.reindexed(np.arange(10))
    np.testing.assert_array_equal(sub.meta["three_numbers"], [1.0, 2.0, 3.0])


def test_meta_dict_is_not_shared_between_derived_trees():
    tree = _electrical_sample()
    other = tree.copy()
    other.meta["whole_cell"] = "changed"
    assert tree.meta["whole_cell"] == "GC-2026"
