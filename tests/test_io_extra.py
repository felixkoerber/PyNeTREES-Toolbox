"""Tests for Phase 5 I/O: neurolucida_tree, save_tree/load_tree, and the
upgraded save_swc.

The neurolucida tests use both a small hand-written synthetic .asc snippet
(for exact, hand-verifiable structure) and the real bundled sample file
`treestoolbox-master/sample/neurolucida/twop9purks.ASC` (for an end-to-end
sanity check against actual NeuroLucida export quirks the synthetic
snippet can't cover, e.g. inline "(Color RGB (...))" metadata, comments,
"Incomplete" markers, and marker/thumbnail blocks that must be excluded).
"""

from pathlib import Path

import numpy as np
import pytest

from pytrees import (
    B_tree,
    Tree,
    idpar_tree,
    load_neurolucida,
    load_swc,
    load_tree,
    save_swc,
    save_tree,
    ver_tree,
)

REPO_ROOT = Path(__file__).parents[2]
SAMPLE_ASC = REPO_ROOT / "treestoolbox-master" / "sample" / "neurolucida" / "twop9purks.ASC"

_SYNTHETIC_ASC = """\
; a tiny synthetic neurolucida file
("CellBody"
 (Color Blue)
 (CellBody)
 (0.0 0.0 0.0 2.0)
 (1.0 0.0 0.0 2.0)
)  ;  End of contour

( (Color Red)
  (Dendrite)
  (0.0 0.0 0.0 1.0)  ; Root
  (10.0 0.0 0.0 1.0)  ; 1, R
  (
    (20.0 5.0 0.0 0.5)  ; 1, R-1
    (30.0 5.0 0.0 0.5)
     Incomplete
  |
    (20.0 -5.0 0.0 0.5)  ; 1, R-2
    (30.0 -5.0 0.0 0.5)
     Incomplete
  )  ;  End of split
)  ;  End of tree

(FilledUpTriangle
  (Color Red)
  (Name "Marker 1")
  (5.0 5.0 5.0 0.1)
)  ;  End of markers
"""


# ---------------------------------------------------------------------------
# neurolucida_tree: synthetic fixture
# ---------------------------------------------------------------------------


def test_load_neurolucida_synthetic_topology_and_regions(tmp_path):
    asc = tmp_path / "synthetic.asc"
    asc.write_text(_SYNTHETIC_ASC)

    trees = load_neurolucida(asc, repair=False)
    assert isinstance(trees, list)
    assert len(trees) == 2  # CellBody + Dendrite; marker block excluded

    by_region = {t.rnames[0]: t for t in trees if len(t.rnames) == 1}
    soma = by_region["CellBody"]
    assert soma.n_nodes == 2
    np.testing.assert_allclose(soma.X, [0.0, 1.0])

    dend = by_region["Dendrite"]
    assert dend.n_nodes == 6
    assert ver_tree(dend, quiet=True) == []
    assert B_tree(dend).sum() == 1  # exactly one branch point, from the split

    # branch point is node 1 (path length 10 along X); its two children
    # should be the (20, 5, 0) and (20, -5, 0) points
    idpar = idpar_tree(dend, no_self=True)
    branch_node = int(np.flatnonzero(B_tree(dend))[0])
    assert dend.X[branch_node] == pytest.approx(10.0)
    children = np.flatnonzero(idpar == branch_node)
    np.testing.assert_allclose(sorted(dend.Y[children].tolist()), [-5.0, 5.0])


def test_load_neurolucida_excludes_marker_blocks(tmp_path):
    asc = tmp_path / "synthetic.asc"
    asc.write_text(_SYNTHETIC_ASC)
    trees = load_neurolucida(asc, repair=False)
    for t in trees:
        marker_point = np.isclose(t.X, 5.0) & np.isclose(t.Y, 5.0) & np.isclose(t.Z, 5.0)
        assert not marker_point.any()


def test_load_neurolucida_repair_produces_binary_tree(tmp_path):
    asc = tmp_path / "synthetic.asc"
    asc.write_text(_SYNTHETIC_ASC)
    trees = load_neurolucida(asc, repair=True)
    dend = next(t for t in trees if t.rnames == ["Dendrite"])
    children_count = np.asarray(dend.dA.sum(axis=0)).ravel()
    assert children_count.max() <= 2
    assert ver_tree(dend, quiet=True) == []


# ---------------------------------------------------------------------------
# neurolucida_tree: real bundled sample file
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_ASC.exists(), reason="bundled sample .ASC not found")
def test_load_neurolucida_real_sample_file():
    trees = load_neurolucida(SAMPLE_ASC, repair=False)
    assert isinstance(trees, list)
    # 2 CellBody contours, 2 Axon, 2 Dendrite, 2 Apical -- the marker
    # (FilledUpTriangle) and Thumbnail/ImageCoords blocks must be excluded
    regions = sorted(t.rnames[0] for t in trees)
    assert regions == sorted(
        ["CellBody", "CellBody", "Axon", "Axon", "Dendrite", "Dendrite", "Apical", "Apical"]
    )
    for t in trees:
        assert ver_tree(t, quiet=True) == []
        assert t.n_nodes > 0


@pytest.mark.skipif(not SAMPLE_ASC.exists(), reason="bundled sample .ASC not found")
def test_load_neurolucida_real_sample_file_repairs_to_binary():
    trees = load_neurolucida(SAMPLE_ASC, repair=True)
    for t in trees:
        children_count = np.asarray(t.dA.sum(axis=0)).ravel()
        assert children_count.max() <= 2
        assert ver_tree(t, quiet=True) == []


# ---------------------------------------------------------------------------
# save_tree / load_tree (native format)
# ---------------------------------------------------------------------------


def _sample_tree() -> Tree:
    from scipy import sparse

    dA = sparse.csr_matrix(([1, 1], ([1, 2], [0, 0])), shape=(3, 3))
    return Tree(
        dA=dA,
        X=np.array([0.0, 1.0, -1.0]),
        Y=np.array([0.0, 2.0, -2.0]),
        Z=np.array([0.0, 3.0, -3.0]),
        D=np.array([4.0, 1.0, 1.0]),
        R=np.array([0, 1, 1]),
        rnames=["soma", "dend"],
        name="roundtrip-me",
        frustum=True,
    )


def test_save_load_tree_round_trip_exact(tmp_path):
    tree = _sample_tree()
    path = tmp_path / "tree"
    save_tree(tree, path)
    reloaded = load_tree(path)

    np.testing.assert_array_equal(reloaded.dA.toarray(), tree.dA.toarray())
    np.testing.assert_array_equal(reloaded.X, tree.X)
    np.testing.assert_array_equal(reloaded.Y, tree.Y)
    np.testing.assert_array_equal(reloaded.Z, tree.Z)
    np.testing.assert_array_equal(reloaded.D, tree.D)
    np.testing.assert_array_equal(reloaded.R, tree.R)
    assert reloaded.rnames == tree.rnames
    assert reloaded.name == tree.name
    assert reloaded.frustum == tree.frustum


def test_save_tree_appends_npz_suffix_automatically(tmp_path):
    tree = _sample_tree()
    save_tree(tree, tmp_path / "no_extension_given")
    assert (tmp_path / "no_extension_given.npz").exists()
    # load_tree should find it whether or not the caller includes .npz
    reloaded = load_tree(tmp_path / "no_extension_given")
    assert reloaded.n_nodes == tree.n_nodes


# ---------------------------------------------------------------------------
# save_swc precision upgrade
# ---------------------------------------------------------------------------


def test_save_swc_full_precision_round_trip(tmp_path):
    tree = _sample_tree()
    tree.X = np.array([0.0, 1.123456789, -1.987654321])
    path = tmp_path / "precise.swc"
    save_swc(tree, path)
    reloaded = load_swc(path)
    np.testing.assert_allclose(reloaded.X, tree.X, atol=1e-7)
