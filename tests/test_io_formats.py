"""B3: the `.neu` reader, the `.nmf` pair, the `.mtr` writer, and the
extension dispatcher behind `load_tree`/`save_tree`.

The `.neu` topology arithmetic **is** verified against MATLAB, unusually for
this cluster. Octave's `textscan` does not behave like MATLAB's, so
`load_tree.m` cannot be run there directly; the file parsing was redone in
plain Octave and the verbatim `load_tree.m` parent-id and region code run on
top of it. Parent indices and geometry came back bit-identical on all three
of the toolbox's own `.neu` fixtures. The reference values below are from
that run.

The `.mtr` writer is verified the other way round: files written here were
loaded by MATLAB's own `load_tree` under Octave and put through `len_tree`,
`B_tree`, `T_tree` and `PL_tree`, which agreed with Python to every digit
printed.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

import pytrees as pt

REPO_ROOT = Path(__file__).parents[2]
NEU_DIR = REPO_ROOT / "treestoolbox-master" / "tests" / "IO" / "test_neu_tree"

pytestmark = pytest.mark.filterwarnings("ignore:.*3D points.*:UserWarning")


def _neu(stem: str) -> Path:
    path = NEU_DIR / f"{stem}.neu"
    if not path.exists():
        pytest.skip(f"{path} not present")
    return path


@pytest.fixture(scope="module")
def tree():
    return pt.sample_tree()


# ---------------------------------------------------------------------------
# .neu -- values below are MATLAB's, from the harness described above
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stem, n_nodes, n_regions",
    [("GC1", 1214, 3), ("GC", 3220, 1), ("GCT", 6800, 7)],
)
def test_neu_fixtures_load(stem, n_nodes, n_regions):
    loaded = pt.load_neu(_neu(stem))
    assert loaded.n_nodes == n_nodes
    assert len(loaded.rnames) == n_regions


def test_neu_geometry_matches_matlab():
    """Column sums of X, Y, Z, D, printed by the Octave harness."""
    loaded = pt.load_neu(_neu("GC1"))
    assert loaded.X.sum() == pytest.approx(41817.798093, abs=1e-5)
    assert loaded.Y.sum() == pytest.approx(0.017707, abs=1e-5)
    assert loaded.Z.sum() == pytest.approx(121400.0, abs=1e-5)
    assert loaded.D.sum() == pytest.approx(1314.761489, abs=1e-5)


def test_neu_root_is_not_assumed_to_be_the_first_node():
    """`GC1.neu` puts its root at node 305 -- the `soma[0]` section is
    tenth in the file, not first.

    MATLAB's reader assumes otherwise and **crashes** on this, one of the
    three fixtures the toolbox ships for the format: its single-tree branch
    builds `dA` from `counter = 2 : N`, so with the root elsewhere it
    reaches `dA(row, -1)`. Reproduced in Octave.
    """
    loaded = pt.load_neu(_neu("GC1"))
    assert loaded.root == 305
    assert loaded.dA.nnz == loaded.n_nodes - 1  # connected, one edge short of N


def test_neu_sections_collapse_into_regions_by_default():
    loaded = pt.load_neu(_neu("GC1"))
    assert sorted(loaded.rnames) == ["axon[]", "section[]", "soma[]"]


def test_neu_keep_sections_makes_each_section_a_region():
    loaded = pt.load_neu(_neu("GC1"), keep_sections=True)
    assert len(loaded.rnames) == 60  # the file's section count
    assert "axon[0]" in loaded.rnames


def test_neu_compound_names_keep_their_anatomy():
    """MATLAB blanks from the *first* bracket to the end of the name, so
    `GCT.neu`'s 90 sections -- `GC7[0].adendGCL[3]`, `GC7[0].soma[0]`, ...
    -- all collapse to one region called `GC7[]`, losing every anatomical
    label. Blanking each bracket in place keeps them.
    """
    loaded = pt.load_neu(_neu("GCT"))
    assert "GC7[].adendGCL[]" in loaded.rnames
    assert "GC7[]" not in loaded.rnames
    assert len(loaded.rnames) == 7


def test_neu_warns_when_the_declared_point_count_is_wrong():
    """The NEURON-side writer computes it as sections x points-per-section.
    It is wrong on all three fixtures, so the section table is used and the
    header only warns."""
    with pytest.warns(UserWarning, match="3D points"):
        pt.io.neu.load_neu(_neu("GC1"))


def test_neu_rejects_a_section_attached_at_its_own_far_end(tmp_path):
    path = tmp_path / "bad.neu"
    path.write_text(
        "# section lines: 2\n"
        "a[0] 0 -1 0 2\n"
        "b[0] 1 a[0] 1 2\n"
        "# 3d points: 4\n"
        "0 0 0 1\n1 0 0 1\n2 0 0 1\n3 0 0 1\n"
    )
    with pytest.raises(ValueError, match="own '1' end"):
        pt.load_neu(path)


def test_neu_needs_the_section_marker(tmp_path):
    path = tmp_path / "empty.neu"
    path.write_text("just some text\n")
    with pytest.raises(ValueError, match="section lines"):
        pt.load_neu(path)


def test_neu_parent_end_zero_branches_from_the_parents_start(tmp_path):
    """The one thing SWC parent indices cannot express directly: NEURON
    lets a section hang off either end of its parent."""
    header = "# section lines: 2\na[0] 0 -1 0 3\nb[0] 0 a[0] {end} 2\n"
    points = "# 3d points: 5\n0 0 0 1\n1 0 0 1\n2 0 0 1\n9 9 0 1\n9 8 0 1\n"

    from pytrees.graphtheory import idpar_tree

    at_end = tmp_path / "end1.neu"
    at_end.write_text(header.format(end=1) + points)
    at_start = tmp_path / "end0.neu"
    at_start.write_text(header.format(end=0) + points)

    assert idpar_tree(pt.load_neu(at_end))[3] == 2  # parent's last point
    assert idpar_tree(pt.load_neu(at_start))[3] == 0  # parent's first point


# ---------------------------------------------------------------------------
# .nmf
# ---------------------------------------------------------------------------


def test_nmf_round_trip_is_lossless(tree, tmp_path):
    path = pt.save_nmf(tree, tmp_path / "cell.nmf")
    back = pt.load_nmf(path)
    np.testing.assert_allclose(back.X, tree.X)
    np.testing.assert_allclose(back.D, tree.D)
    np.testing.assert_array_equal(back.R, tree.R)
    assert (back.dA != tree.dA).nnz == 0


def test_nmf_keeps_region_names_where_matlab_would_lose_them(tree, tmp_path):
    """MATLAB's writer stores only the region *indices*, so a round trip
    through it renames `dendrite`/`subtree` to `1`/`2`. The names go into a
    group attribute here, which MATLAB's reader ignores rather than
    chokes on."""
    back = pt.load_nmf(pt.save_nmf(tree, tmp_path / "cell.nmf"))
    assert back.rnames == tree.rnames


def test_nmf_stores_radius_not_diameter(tree, tmp_path):
    """The factor-of-two trap in every SWC-family format."""
    h5py = pytest.importorskip("h5py")
    path = pt.save_nmf(tree, tmp_path / "cell.nmf")
    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(handle["/swc/r"][:].ravel(), tree.D / 2)


def test_nmf_writes_matlabs_one_based_parent_indices(tree, tmp_path):
    h5py = pytest.importorskip("h5py")
    path = pt.save_nmf(tree, tmp_path / "cell.nmf")
    with h5py.File(path, "r") as handle:
        idpar = handle["/swc/parent_index"][:].ravel()
    assert idpar[0] == -1
    assert idpar[1] == 1  # node 2's parent is node 1, in MATLAB's counting


def test_nmf_rejects_a_file_without_the_swc_group(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "other.nmf"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("something", data=[1, 2, 3])
    with pytest.raises(ValueError, match="no '/swc' group"):
        pt.load_nmf(path)


# ---------------------------------------------------------------------------
# .mtr writing -- the MATLAB interoperability path
# ---------------------------------------------------------------------------


def test_mtr_round_trip(tree, tmp_path):
    back = pt.load_mtr(pt.save_mtr(tree, tmp_path / "cell.mtr"))
    np.testing.assert_allclose(back.X, tree.X)
    assert back.rnames == tree.rnames
    assert back.name == tree.name
    assert (back.dA != tree.dA).nnz == 0


def test_mtr_carries_the_electrotonic_constants(tree, tmp_path):
    back = pt.load_mtr(pt.save_mtr(tree, tmp_path / "cell.mtr"))
    assert (back.Ri, back.Gm, back.Cm) == (tree.Ri, tree.Gm, tree.Cm)


def test_mtr_stores_a_list_of_trees(tmp_path):
    trees = [pt.sample_tree(), pt.sample2_tree()]
    back = pt.load_mtr(pt.save_mtr(trees, tmp_path / "many.mtr"))
    assert [t.n_nodes for t in back] == [197, 15]


def test_mtr_uses_matlabs_one_based_region_indices(tree, tmp_path):
    from scipy.io import loadmat

    path = pt.save_mtr(tree, tmp_path / "cell.mtr")
    raw = loadmat(str(path), struct_as_record=False, squeeze_me=True)["tree"]
    np.testing.assert_array_equal(np.ravel(raw.R), tree.R + 1)


# ---------------------------------------------------------------------------
# the dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".npz", ".mtr", ".swc", ".nmf"])
def test_every_writable_format_reloads_through_the_dispatcher(tree, tmp_path, suffix):
    path = pt.save_tree(tree, tmp_path / f"cell{suffix}")
    back = pt.load_tree(path)
    np.testing.assert_allclose(back.X, tree.X)
    np.testing.assert_allclose(back.D, tree.D)
    assert (back.dA != tree.dA).nnz == 0


def test_the_written_path_is_returned(tree, tmp_path):
    """The extension is appended when missing, so the path passed in is not
    always the path written."""
    path = pt.save_tree(tree, tmp_path / "cell")
    assert path.suffix == ".npz"
    assert path.exists()


def test_dispatcher_reads_neu(tmp_path):
    assert pt.load_tree(_neu("GC")).n_nodes == 3220


def test_dispatcher_passes_options_to_the_format(tmp_path):
    assert len(pt.load_tree(_neu("GC1"), keep_sections=True).rnames) == 60


def test_an_unknown_extension_lists_what_is_supported(tmp_path):
    with pytest.raises(ValueError, match=r"\.swc"):
        pt.load_tree(tmp_path / "cell.txt")


def test_writing_a_format_that_can_only_be_read_is_refused(tree, tmp_path):
    """`.neu` is written by NEURON, not by this port; say so rather than
    accepting the call and producing nothing."""
    with pytest.raises(ValueError, match="this port writes"):
        pt.save_tree(tree, tmp_path / "cell.neu")


def test_saving_several_trees_needs_a_format_that_holds_several(tree, tmp_path):
    with pytest.raises(ValueError, match="single tree"):
        pt.save_tree([tree, tree], tmp_path / "cells.swc")


def test_mat_is_accepted_as_an_alias_for_mtr(tree, tmp_path):
    """People rename these freely, and a `.mtr` is a `.mat`."""
    back = pt.load_tree(pt.save_tree(tree, tmp_path / "cell.mat"))
    np.testing.assert_allclose(back.X, tree.X)
