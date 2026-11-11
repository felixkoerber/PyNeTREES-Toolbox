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

import pynetrees as pt
from pynetrees import (
    B_tree,
    T_tree,
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
    asc.write_text(_SYNTHETIC_ASC.replace("(1.0 0.0 0.0 2.0)", "(20.0 0.0 0.0 2.0)"))

    tree = load_neurolucida(asc, repair=False)
    assert isinstance(tree, Tree)
    assert set(tree.rnames) == {"CellBody", "Dendrite"}
    assert ver_tree(tree, quiet=True) == []

    # the soma is a chain along the contour, the dendrite joined onto it
    assert int(np.isclose(tree.X, 10.0).sum()) >= 1
    dendrite = tree.region_nodes("Dendrite")
    assert int(dendrite.sum()) == 6
    assert int((B_tree(tree) & dendrite).sum()) == 1
    assert sorted(tree.Y[dendrite & T_tree(tree)].tolist()) == [-5.0, 5.0]


def test_load_neurolucida_returns_markers_and_contours(tmp_path):
    asc = tmp_path / "synthetic.asc"
    asc.write_text(_SYNTHETIC_ASC.replace("(1.0 0.0 0.0 2.0)", "(20.0 0.0 0.0 2.0)"))
    result = load_neurolucida(asc, repair=False, full_output=True)
    np.testing.assert_allclose(result.markers, [[5.0, 5.0, 5.0, 0.1]])
    assert [c["name"] for c in result.contours] == ["CellBody"]
    marker_point = (np.isclose(result.trees.X, 5.0) & np.isclose(result.trees.Y, 5.0)
                    & np.isclose(result.trees.Z, 5.0))
    assert not marker_point.any()


def test_load_neurolucida_reads_spines(tmp_path):
    asc = tmp_path / "spiny.asc"
    asc.write_text("""\
( (Dendrite)
  (0.0 0.0 0.0 1.0)
  (10.0 0.0 0.0 1.0)
  <(10.0 2.0 0.0 0.5)>
  (20.0 0.0 0.0 1.0)
)
""")
    tree = load_neurolucida(asc, repair=False)
    spines = tree.region_nodes("spines")
    assert int(spines.sum()) == 1
    parent = pt.idpar_tree(tree)[np.flatnonzero(spines)[0]]
    assert tree.X[parent] == 10.0 and tree.Y[parent] == 0.0


def test_load_neurolucida_one_soma_keeps_the_largest(tmp_path):
    asc = tmp_path / "two.asc"
    contour = """("CellBody" (CellBody)
 (0.0 {y} 0.0 1.0) ({L} {y} 0.0 1.0) ({L} {w} 0.0 1.0) (0.0 {w} 0.0 1.0) (0.0 {y} 0.0 1.0))
"""
    asc.write_text(contour.format(y=0.0, w=4.0, L=12.0)
                   + contour.format(y=100.0, w=130.0, L=60.0))
    assert len(load_neurolucida(asc, repair=False)) == 2
    tree = load_neurolucida(asc, repair=False, one_soma=True)
    assert isinstance(tree, Tree) and tree.Y.max() > 50.0


def test_load_neurolucida_repair_produces_binary_tree(tmp_path):
    asc = tmp_path / "synthetic.asc"
    asc.write_text(_SYNTHETIC_ASC.replace("(1.0 0.0 0.0 2.0)", "(20.0 0.0 0.0 2.0)"))
    tree = load_neurolucida(asc, repair=True)
    children_count = np.asarray(tree.dA.sum(axis=0)).ravel()
    assert children_count.max() <= 2
    assert ver_tree(tree, quiet=True) == []


# ---------------------------------------------------------------------------
# neurolucida_tree: real bundled sample file
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SAMPLE_ASC.exists(), reason="bundled sample .ASC not found")
def test_load_neurolucida_real_sample_file():
    trees = load_neurolucida(SAMPLE_ASC, repair=False)
    # two cells, each a soma with its apical, axon and (for one) basal trees
    assert isinstance(trees, list) and len(trees) == 2
    assert set(trees[0].rnames) == {"Apical", "Axon", "CellBody", "Dendrite"}
    assert set(trees[1].rnames) == {"Apical", "Axon", "CellBody"}
    for t in trees:
        assert ver_tree(t, quiet=True) == []


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
