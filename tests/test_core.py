"""Tests for pytrees.core: Tree construction and ver_tree validation.

Ports the intent of treestoolbox-master/tests/IO/check_ver_tree.m (test 1:
a well-formed tree has no issues; test 2: a mismatched-size field is flagged).
"""

import numpy as np
import pytest
from scipy import sparse

from pytrees import Tree, sample_tree, ver_tree


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
