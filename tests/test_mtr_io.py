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
def test_load_mtr_v73_file_raises_clear_error():
    with pytest.raises(ValueError, match="v7.3"):
        load_mtr(V73_FILE)


def test_load_mtr_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_mtr("does_not_exist.mtr")
