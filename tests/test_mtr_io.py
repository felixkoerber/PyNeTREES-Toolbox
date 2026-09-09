"""Tests for pynetrees.io.load_mtr, against the real bundled Active GC Model
morphology archives (MATLAB v5 .mtr files -- see io/mtr.py's module
docstring for why this exists despite .mtr being deferred back in Phase 1).
"""

from pathlib import Path

import pytest

import numpy as np
import pynetrees as pt
from pynetrees import load_mtr, ver_tree

REPO_ROOT = Path(__file__).parents[2]
MORPHOS = REPO_ROOT / "Active GC Model" / "morphos"
GC_MIDI = MORPHOS / "SH_07_all_repairedandsomaAIS_MLyzed-Midi.mtr"
V73_FILE = MORPHOS / "0dplaxonFitsoma.mtr"


@pytest.mark.skipif(not GC_MIDI.exists(), reason="bundled Active GC Model .mtr not found")
def test_load_mtr_real_granule_cell_population():
    trees = load_mtr(GC_MIDI)
    assert isinstance(trees, list)
    assert len(trees) == 8
    for tree in trees:
        assert ver_tree(tree, quiet=True) == []
        assert tree.n_nodes > 0
        assert set(tree.rnames) >= {"soma", "axon"}
        # R must be valid 0-based indices into rnames after the MATLAB
        # 1-based -> 0-based conversion
        assert tree.R.min() >= 0
        assert tree.R.max() < len(tree.rnames)


@pytest.mark.skipif(not V73_FILE.exists(), reason="bundled v7.3 .mtr not found")
def test_load_mtr_reads_v73_nested_cell_array():
    """v7.3 files load, including the 2-level nesting that broke pymatreader.

    This file has no v5 twin anywhere in `morphos/`, so before `mat73` was
    wired in (Design Decision #47) its 15 reconstructions were unreachable
    from Python -- and from Octave, which cannot read them either.
    """
    trees = load_mtr(V73_FILE)
    assert isinstance(trees, list)
    assert len(trees) == 15
    for tree in trees:
        assert ver_tree(tree, quiet=True) == []
        assert set(tree.rnames) >= {"soma", "axon"}
        # electrotonic fields carried through from the MATLAB struct
        assert tree.Ri is not None


@pytest.mark.skipif(not V73_FILE.exists(), reason="bundled v7.3 .mtr not found")
def test_v73_detection_reads_the_header_not_the_exception():
    """Format is sniffed from the 128-byte MATLAB header.

    Not from catching scipy's exception: scipy raises `NotImplementedError`
    for most v7.3 files but `ValueError: embedded null character` for this
    one, having misparsed HDF5 bytes as a v5 structure.
    """
    from pynetrees.io._matlab import is_v73

    assert is_v73(V73_FILE)
    assert not is_v73(GC_MIDI)


def test_load_mtr_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_mtr("does_not_exist.mtr")


# ---------------------------------------------------------------------------
# Tree.meta <-> MATLAB's own ad-hoc struct fields
# ---------------------------------------------------------------------------


def test_matlab_ad_hoc_fields_land_in_meta():
    """MATLAB's TREES hangs extra data straight off the tree struct, and the
    bundled granule cells carry `NID` (T2N's hoc identifier) and `col` (the
    plot colour). Both used to be dropped on the floor."""
    tree = pt.gc_tree()
    assert tree.meta["NID"] == "cell_SH07all2_1"
    np.testing.assert_allclose(np.ravel(tree.meta["col"]), [0.0, 0.0, 0.3])


def test_meta_round_trips_through_mtr(tmp_path):
    tree = pt.sample_tree()
    tree.meta.update({
        "NID": "SH07_cell4",                              # a string
        "col": np.array([1.0, 0.0, 0.0]),                 # tree.col
        "Rho_soma": 1.37,                                 # a scalar
        "per_node": np.arange(tree.n_nodes, dtype=float),  # a per-node array
    })

    back = pt.load_mtr(pt.save_mtr(tree, tmp_path / "cell.mtr"))

    assert back.meta["NID"] == "SH07_cell4"
    np.testing.assert_allclose(np.ravel(back.meta["col"]), [1.0, 0.0, 0.0])
    assert float(np.ravel(back.meta["Rho_soma"])[0]) == pytest.approx(1.37)
    np.testing.assert_allclose(
        np.ravel(back.meta["per_node"]), np.arange(tree.n_nodes)
    )


def test_meta_is_written_flat_so_matlab_functions_can_see_it(tmp_path):
    """A nested struct would round-trip just as well and be invisible to
    every TREES function that reads `tree.col` by name."""
    from scipy.io import loadmat

    tree = pt.sample_tree()
    tree.meta["col"] = np.array([1.0, 0.0, 0.0])
    path = pt.save_mtr(tree, tmp_path / "cell.mtr")

    raw = loadmat(str(path), simplify_cells=True)["tree"]
    assert "col" in raw                       # top level, beside X/Y/Z/dA
    assert "meta" not in raw


def test_a_meta_key_that_collides_with_a_tree_field_is_refused(tmp_path):
    """MATLAB has one flat namespace per struct, so `meta['X']` would either
    corrupt the morphology or vanish depending on dict ordering."""
    tree = pt.sample_tree()
    tree.meta["X"] = "not the coordinates"

    with pytest.raises(ValueError, match="collide"):
        pt.save_mtr(tree, tmp_path / "cell.mtr")


def test_a_meta_value_mtr_cannot_store_warns_rather_than_vanishing(tmp_path):
    tree = pt.sample_tree()
    tree.meta["a_function"] = len            # savemat cannot encode this

    with pytest.warns(UserWarning, match="cannot store"):
        back = pt.load_mtr(pt.save_mtr(tree, tmp_path / "cell.mtr"))
    assert "a_function" not in back.meta


def test_a_meta_key_that_is_not_a_matlab_field_name_warns(tmp_path):
    tree = pt.sample_tree()
    tree.meta["not a field"] = 1.0

    with pytest.warns(UserWarning, match="not a valid MATLAB field name"):
        pt.save_mtr(tree, tmp_path / "cell.mtr")


def test_editing_a_matlab_tree_here_keeps_its_identity(tmp_path):
    """The point of the whole thing: load from MATLAB, edit, write back, and
    the fields MATLAB cares about are still there."""
    tree = pt.gc_tree()
    edited = pt.delete_tree(tree, tree.region_nodes("axon"), keep_regions=True)

    back = pt.load_mtr(pt.save_mtr(edited, tmp_path / "edited.mtr"))
    assert back.meta["NID"] == tree.meta["NID"]
    np.testing.assert_allclose(
        np.ravel(back.meta["col"]), np.ravel(tree.meta["col"])
    )
