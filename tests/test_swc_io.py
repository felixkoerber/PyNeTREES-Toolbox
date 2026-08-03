"""Tests for pytrees.io.swc: SWC reading and writing.

Ports the intent of treestoolbox-master/tests/IO/check_swc_tree.m and
check_load_tree.m: load a real reconstruction, verify it's well-formed, and
round-trip it through a write/read cycle. ``fixtures/test02.swc`` mirrors a
real-world quirk documented in the MATLAB test suite: node indices that are
present but not contiguous/sorted (1-based, out of file order).
"""

from pathlib import Path

import numpy as np

from pytrees import Tree, ver_tree
from pytrees.io import load_swc, save_swc

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_swc_single_root_real_reconstruction():
    tree = load_swc(FIXTURES / "25HSS.swc")
    assert isinstance(tree, Tree)
    assert tree.n_nodes == 2252  # file has 2252 data rows (no comments)
    assert ver_tree(tree, quiet=True) == []
    # exactly one node has no parent (the root)
    root_rows = tree.dA.sum(axis=1)
    assert int((np.asarray(root_rows).ravel() == 0).sum()) == 1


def test_load_swc_handles_unsorted_noncontiguous_indices():
    # test02.swc node indices are out of order and not 1..N in file order;
    # this is the case MATLAB's load_tree.m explicitly corrects for.
    tree = load_swc(FIXTURES / "test02.swc")
    assert isinstance(tree, Tree)
    assert tree.n_nodes == 11
    assert ver_tree(tree, quiet=True) == []


def test_swc_round_trip(tmp_path):
    original = load_swc(FIXTURES / "25HSS.swc")
    out_path = tmp_path / "roundtrip.swc"
    save_swc(original, out_path)
    reloaded = load_swc(out_path)

    assert isinstance(reloaded, Tree)
    assert reloaded.n_nodes == original.n_nodes
    np.testing.assert_allclose(reloaded.X, original.X, atol=1e-5)
    np.testing.assert_allclose(reloaded.Y, original.Y, atol=1e-5)
    np.testing.assert_allclose(reloaded.Z, original.Z, atol=1e-5)
    np.testing.assert_allclose(reloaded.D, original.D, atol=1e-5)
    # topology (parent structure) must be preserved exactly
    np.testing.assert_array_equal(
        original.dA.toarray(), reloaded.dA.toarray()
    )


def test_load_swc_splits_multiple_roots_into_separate_trees(tmp_path):
    # two disconnected components, each with its own parent==-1 root row
    multi = tmp_path / "multi_root.swc"
    multi.write_text(
        "1 1 0 0 0 1 -1\n"
        "2 1 1 0 0 1 1\n"
        "3 1 0 0 0 1 -1\n"
        "4 1 1 0 0 1 3\n"
        "5 1 2 0 0 1 4\n"
    )
    trees = load_swc(multi)
    assert isinstance(trees, list)
    assert sorted(t.n_nodes for t in trees) == [2, 3]
    for tree in trees:
        assert ver_tree(tree, quiet=True) == []


def test_load_swc_rejects_file_without_root(tmp_path):
    bad = tmp_path / "no_root.swc"
    bad.write_text("1 1 0 0 0 1 2\n2 1 0 0 0 1 1\n")  # 1<->2, no parent -1
    try:
        load_swc(bad)
        assert False, "expected ValueError for a file with no root"
    except ValueError as exc:
        assert "root" in str(exc)
