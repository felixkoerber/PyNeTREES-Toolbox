"""`rot_tree`'s automatic alignment modes, against the MATLAB reference.

Design Decision #56, reversing #20 (which deferred these as "niche").
Reference extents come from running `metrics/rot_tree.m` in Octave 11 on
`hss_tree` -- chosen because it has real `axon`/`dend`/`soma` regions, so
the `m3d` modes' axon exclusion actually does something.
"""

from __future__ import annotations

import numpy as np
import pytest

import pytrees as pt

# mode -> (x, y, z) bounding-box extent [um] produced by MATLAB
MATLAB_EXTENTS = {
    "pcaX": (791.00, 285.36, 67.69),
    "pcaY": (285.36, 791.00, 67.69),
    "pcaZ": (67.69, 285.36, 791.00),
    "m3dX": (766.91, 272.79, 223.73),
    "m3dY": (272.79, 766.91, 223.73),
    "m3dZ": (272.79, 223.73, 766.91),
}


@pytest.fixture(scope="module")
def cell():
    return pt.hss_tree()


@pytest.mark.parametrize("mode", MATLAB_EXTENTS)
def test_matches_matlab_extents(cell, mode):
    rotated = pt.rot_tree(cell, mode=mode)
    xyz = np.column_stack([rotated.X, rotated.Y, rotated.Z])
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    np.testing.assert_allclose(extent, MATLAB_EXTENTS[mode], atol=0.01)


@pytest.mark.parametrize("mode,axis", [("pcaX", 0), ("pcaY", 1), ("pcaZ", 2)])
def test_pca_puts_the_largest_extent_on_the_named_axis(cell, mode, axis):
    rotated = pt.rot_tree(cell, mode=mode)
    xyz = np.column_stack([rotated.X, rotated.Y, rotated.Z])
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    assert np.argmax(extent) == axis


def test_pca_sign_convention_matches_matlab(cell):
    """A PC's sign is arbitrary; MATLAB and numpy disagree by default.

    MATLAB's `pca` makes each component's largest-magnitude element
    positive. Without matching that, extents agree exactly while
    coordinates come out mirrored -- an error of the tree's whole width
    (307 um here) that no extent-based check would catch.
    """
    rotated = pt.rot_tree(cell, mode="pcaX")
    # MATLAB's own values for this tree, not this implementation's
    assert rotated.X.min() == pytest.approx(-640.4592, abs=0.01)
    assert rotated.X.max() == pytest.approx(150.5412, abs=0.01)


@pytest.mark.parametrize("mode,axis", [("m3dX", 0), ("m3dY", 1), ("m3dZ", 2)])
def test_m3d_aligns_the_mean_axis(cell, mode, axis):
    rotated = pt.rot_tree(cell, mode=mode)
    xyz = np.column_stack([rotated.X, rotated.Y, rotated.Z])
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    assert np.argmax(extent) == axis


def test_m3d_excludes_the_axon_by_default(cell):
    """The exclusion is not cosmetic: an axon drags the mean axis off."""
    with_axon = pt.rot_tree(cell, mode="m3dX", exclude_regions=())
    without = pt.rot_tree(cell, mode="m3dX")
    assert not np.allclose(with_axon.X, without.X)


def test_m3d_accepts_an_explicit_node_subset(cell):
    """`nodes=` is the parameter MATLAB's docstring promised but never read.

    MATLAB documents `DEG` as doubling for a node subset under `-m3d`, but
    that branch never references `DEG` at all.
    """
    dend = cell.region_nodes("dend")
    rotated = pt.rot_tree(cell, mode="m3dX", nodes=dend)
    assert rotated.n_nodes == cell.n_nodes
    assert not np.allclose(rotated.X, cell.X)


def test_rotation_preserves_shape(cell):
    """Whatever the mode, a rotation must not change any distance."""
    for mode in ("pcaX", "m3dY"):
        rotated = pt.rot_tree(cell, mode=mode)
        np.testing.assert_allclose(
            pt.len_tree(rotated).sum(), cell.total_length, rtol=1e-9
        )


def test_explicit_degrees_still_work(cell):
    """A 90-degree z-rotation sends Y into X, *not* -Y into X.

    MATLAB applies its rotation matrix on the right (`[X Y Z] * RM`, the
    row-vector convention), and so does this port; assuming the
    column-vector convention flips the sign. Checked against MATLAB rather
    than reasoned about: `rot_tree(hss_tree, [0 0 90]).X` equals
    `hss_tree.Y` element for element.
    """
    turned = pt.rot_tree(cell, (0.0, 0.0, 90.0))
    np.testing.assert_allclose(turned.X, cell.Y, atol=1e-9)
    np.testing.assert_allclose(turned.Y, -cell.X, atol=1e-9)


def test_scalar_degrees_rotate_in_the_xy_plane(cell):
    turned = pt.rot_tree(cell, 180.0)
    np.testing.assert_allclose(turned.X, -cell.X, atol=1e-9)
    np.testing.assert_allclose(turned.Z, cell.Z, atol=1e-9)


def test_unknown_mode_is_rejected(cell):
    with pytest.raises(ValueError, match="unknown mode"):
        pt.rot_tree(cell, mode="pcaW")


def test_align_region_needs_a_preceding_region(cell):
    with pytest.raises(ValueError, match="preceding region"):
        pt.rot_tree(cell, mode="m3dX", align_region=cell.rnames[0])


def test_align_region_warns_when_no_border_exists(cell):
    """`soma` and `axon` are adjacent in rnames but share no node border."""
    with pytest.warns(UserWarning, match="alignment skipped"):
        pt.rot_tree(cell, mode="m3dX", align_region="soma")
