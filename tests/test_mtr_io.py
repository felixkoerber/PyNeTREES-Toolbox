"""Tests for pytrees.io.load_mtr, against the real bundled Active GC Model
morphology archives (MATLAB v5 .mtr files -- see io/mtr.py's module
docstring for why this exists despite .mtr being deferred back in Phase 1).
"""

from pathlib import Path

import pytest

from pytrees import load_mtr, ver_tree

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
    from pytrees.io._matlab import is_v73

    assert is_v73(V73_FILE)
    assert not is_v73(GC_MIDI)


def test_load_mtr_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_mtr("does_not_exist.mtr")
