"""The remaining MATLAB options ported in W3.

`ipar_tree(terminals_only=)`, `dissect_tree(with_positions=)` and
`soma_tree(overlap_correction=)`. Where a MATLAB reference exists it was
produced by running the source in Octave 11; where it does not, the reason
is recorded in `MATLAB_TOOLBOX_BUGS.md`.
"""

from __future__ import annotations

import numpy as np
import pytest

import pytrees as pt


# ---------------------------------------------------------------------------
# ipar_tree(terminals_only=True)  -- MATLAB's '-T'
# ---------------------------------------------------------------------------


def test_terminals_only_has_one_row_per_terminal():
    tree = pt.sample_tree()
    rows = pt.ipar_tree(tree, terminals_only=True)
    assert rows.shape[0] == int(pt.T_tree(tree).sum()) == 26


def test_terminals_only_rows_start_at_a_terminal():
    tree = pt.sample_tree()
    rows = pt.ipar_tree(tree, terminals_only=True)
    assert pt.T_tree(tree)[rows[:, 0]].all()


def test_terminals_only_stops_before_the_first_branch_point():
    """That is the defining property: each row is one unbranched run."""
    tree = pt.sample_tree()
    branch = pt.B_tree(tree)
    for row in pt.ipar_tree(tree, terminals_only=True):
        path = row[row >= 0]
        assert not branch[path].any()


def test_terminals_only_paths_are_contiguous_parent_chains():
    tree = pt.sample_tree()
    idpar = pt.idpar_tree(tree, root_self=False)
    for row in pt.ipar_tree(tree, terminals_only=True):
        path = row[row >= 0]
        for child, parent in zip(path[:-1], path[1:]):
            assert idpar[child] == parent


def test_terminals_only_is_dramatically_smaller():
    """The point of the option: the full matrix is the toolbox's worst
    structure (2.3 MB for the HSS cell), and this is the same information
    for terminal segments at a fraction of the size."""
    tree = pt.hss_tree()
    full = pt.ipar_tree(tree)
    terminal = pt.ipar_tree(tree, terminals_only=True)
    assert terminal.nbytes < full.nbytes / 10


def test_nodes_argument_selects_terminals():
    tree = pt.sample_tree()
    terminals = np.flatnonzero(pt.T_tree(tree))
    chosen = terminals[:5]
    rows = pt.ipar_tree(tree, terminals_only=True, nodes=chosen)
    assert rows.shape[0] == 5
    np.testing.assert_array_equal(np.sort(rows[:, 0]), np.sort(chosen))


# ---------------------------------------------------------------------------
# dissect_tree(with_positions=True)  -- MATLAB's second output
# ---------------------------------------------------------------------------


def test_with_positions_returns_sections_and_positions():
    tree = pt.sample_tree()
    plain = pt.dissect_tree(tree)
    result = pt.dissect_tree(tree, with_positions=True)

    np.testing.assert_array_equal(result.sections, plain)
    assert result.positions.shape == (tree.n_nodes, 2)


def test_section_count_matches_matlab():
    """MATLAB's `dissect_tree(sample_tree)` yields 51 sections."""
    assert pt.dissect_tree(pt.sample_tree()).shape[0] == 51


def test_every_node_is_assigned_to_a_section():
    tree = pt.sample_tree()
    _, positions = pt.dissect_tree(tree, with_positions=True)
    assert (positions[:, 0] >= 0).all()


def test_relative_position_runs_zero_to_one_within_each_section():
    tree = pt.sample_tree()
    sections, positions = pt.dissect_tree(tree, with_positions=True)
    assert positions[:, 1].min() >= 0.0
    assert positions[:, 1].max() <= 1.0
    # Each section *owns* its end node, at fraction 1.0. It does not own its
    # start node: a branch point starts two sections and ends one, and it
    # belongs to the one it ends -- otherwise the later assignment would
    # silently overwrite the earlier. MATLAB draws the line the same way
    # (it assigns `DEC(1:end-1)`, excluding the start).
    root = tree.root
    for index, (start, end) in enumerate(sections.tolist()):
        if start == end:
            continue
        assert positions[end, 1] == pytest.approx(1.0)
        assert positions[end, 0] == index
        if start != root:
            assert positions[start, 0] != index


def test_positions_are_what_neuron_needs_to_place_a_synapse():
    """`(section, fraction along it)` is exactly NEURON's `sec(x)` addressing."""
    tree = pt.sample_tree()
    sections, positions = pt.dissect_tree(tree, with_positions=True)
    node = tree.n_nodes // 2
    section_index, fraction = positions[node]
    assert 0 <= section_index < len(sections)
    assert 0.0 <= fraction <= 1.0


# ---------------------------------------------------------------------------
# soma_tree(overlap_correction=True)  -- MATLAB's '-b'
# ---------------------------------------------------------------------------


def test_overlap_correction_reduces_surface_where_branches_are_enclosed():
    """No MATLAB reference exists for this: `soma_tree(...,'-b')` crashes
    there on any tree whose root has a single child, which includes its own
    `sample_tree` (see MATLAB_TOOLBOX_BUGS.md). The behaviour is checked
    against the physical property the option exists for instead."""
    tree = pt.hss_tree()
    plain = pt.soma_tree(tree, maxD=120.0)
    corrected = pt.soma_tree(tree, maxD=120.0, overlap_correction=True)
    assert corrected.total_surface < plain.total_surface


def test_overlap_correction_is_a_no_op_without_enclosed_branch_points():
    """Correct, not broken: with no branch point *passed* inside the soma
    profile there is no shared membrane to discount."""
    tree = pt.sample_tree()
    plain = pt.soma_tree(tree, maxD=30.0)
    corrected = pt.soma_tree(tree, maxD=30.0, overlap_correction=True)
    np.testing.assert_allclose(corrected.D, plain.D)


def test_overlap_factor_grows_by_sqrt2_per_branch_passed():
    from pytrees.construct import _overlap_factor

    tree = pt.sample_tree()
    factor = _overlap_factor(tree)
    ratios = factor[factor > 1]
    if len(ratios):
        # every value is an integer power of sqrt(2)
        powers = np.log(ratios) / np.log(np.sqrt(2.0))
        np.testing.assert_allclose(powers, np.round(powers), atol=1e-9)


def test_soma_tree_still_reaches_maxD_at_the_root():
    tree = pt.sample_tree()
    somatic = pt.soma_tree(tree, maxD=30.0)
    assert somatic.D[somatic.root] == pytest.approx(30.0)
