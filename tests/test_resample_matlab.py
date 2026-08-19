"""`resample_tree(method='matlab')` against the MATLAB reference.

The reference values below were produced by running the *actual*
`edit/resample_tree.m` in Octave 11 against the bundled `sample_tree`
(197 nodes), not by recording this implementation's own output. That
distinction is the point: these numbers can tell us we are wrong.

Reproducing them needs a patched toolbox copy (Octave has no `contains`,
which TREES' option parsing uses) -- the procedure is recorded in
`REVIEW_PLAN.md` §W7. Baking the outputs in as constants keeps the check in
CI without requiring Octave.
"""

from __future__ import annotations

import numpy as np
import pytest

import pytrees as pt

# case -> (kwargs, expected n_nodes, expected total length [um])
MATLAB_REFERENCE = {
    "default": ({}, 78, 724.5606),
    "sr5": ({"sr": 5.0}, 155, 753.8981),
    "interp_diameter": ({"interp_diameter": True}, 78, 724.5606),
    "conserve_length": ({"conserve_length": True}, 78, 770.0000),
    "no_collapse": ({"collapse_branches": False}, 91, 821.6374),
}


@pytest.mark.parametrize("case", MATLAB_REFERENCE)
def test_matches_matlab_node_count_and_length(case):
    kwargs, n_nodes, total_length = MATLAB_REFERENCE[case]
    result = pt.resample_tree(pt.sample_tree(), method="matlab", **kwargs)
    assert result.n_nodes == n_nodes
    assert result.total_length == pytest.approx(total_length, abs=1e-3)


def test_matlab_is_the_default_method():
    tree = pt.sample_tree()
    default = pt.resample_tree(tree, sr=10.0)
    explicit = pt.resample_tree(tree, sr=10.0, method="matlab")
    assert default.n_nodes == explicit.n_nodes == 78


def test_segments_come_out_no_longer_than_sr():
    """Grid points are placed at multiples of sr *of the original path*.

    Deleting the intermediate original nodes then replaces each polyline
    with a chord, so the resampled segments end up slightly **shorter** than
    `sr` rather than exactly `sr`. MATLAB's own source says so at the `'-l'`
    branch: "after deleting points on the way the length of an edge is not
    sr anymore (because we cut the paths short)". That is the entire reason
    `conserve_length` exists, and asserting exact multiples here would be
    asserting a property the algorithm does not have.
    """
    sr = 10.0
    result = pt.resample_tree(pt.sample_tree(), sr=sr, method="matlab",
                              collapse_branches=False)
    seg = np.delete(pt.len_tree(result), result.root)
    assert seg.max() <= sr + 1e-9
    assert seg.mean() > 0.8 * sr  # chords, but not wildly short


def test_conserve_length_makes_every_segment_exactly_sr():
    sr = 10.0
    result = pt.resample_tree(pt.sample_tree(), sr=sr, method="matlab",
                              conserve_length=True)
    seg = pt.len_tree(result)
    non_root = np.delete(seg, result.root)
    np.testing.assert_allclose(non_root, sr, atol=1e-9)


def test_interp_diameter_changes_diameters_but_not_geometry():
    tree = pt.sample_tree()
    plain = pt.resample_tree(tree, method="matlab")
    interp = pt.resample_tree(tree, method="matlab", interp_diameter=True)

    assert plain.n_nodes == interp.n_nodes
    np.testing.assert_allclose(plain.X, interp.X)
    assert not np.allclose(plain.D, interp.D)
    # ...and it does change surface/volume, which is why it is off by default
    assert plain.total_surface != pytest.approx(interp.total_surface)


def test_collapse_removes_grid_induced_near_coincident_branches():
    tree = pt.sample_tree()
    with_collapse = pt.resample_tree(tree, method="matlab")
    without = pt.resample_tree(tree, method="matlab", collapse_branches=False)
    assert with_collapse.n_nodes < without.n_nodes
    assert without.n_nodes - with_collapse.n_nodes == 13  # MATLAB collapses 13


def test_collapse_tie_break_keeps_the_first_daughter():
    """Pinned because it was found empirically, not read off the source.

    When two collapse candidates have equally large subtrees, which one
    survives is invisible in the geometry (both are moved to their midpoint
    first) but *is* visible in the diameters. Matching MATLAB required
    keeping the first daughter; the opposite choice reproduced everything
    else exactly and left ~7 of 78 diameters differing by up to 0.023 um.
    """
    tree = pt.sample_tree()
    result = pt.resample_tree(tree, method="matlab", interp_diameter=True)
    # MATLAB's diameter extremes for this exact configuration
    assert result.D.min() == pytest.approx(1.3, abs=1e-6)
    assert result.D.max() == pytest.approx(4.299, abs=1e-3)


def test_anchors_method_still_available_and_different():
    tree = pt.sample_tree()
    matlab = pt.resample_tree(tree, method="matlab")
    anchors = pt.resample_tree(tree, method="anchors")
    assert matlab.n_nodes != anchors.n_nodes
    assert pt.B_tree(anchors).sum() == pt.B_tree(tree).sum()


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="'matlab' or 'anchors'"):
        pt.resample_tree(pt.sample_tree(), method="grid")
