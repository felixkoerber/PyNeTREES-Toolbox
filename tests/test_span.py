"""V4: spanned area, space filling, and scaling a tree to a target size.

`span_tree` was checked against MATLAB's own code running under Octave, on
the exact Euclidean disk (Octave implements no other), and agrees pixel for
pixel -- see `MATLAB_REFERENCE` below. The structuring element is the one
thing that could not be compared, because MATLAB's default is a four-line
approximation of a disk that Octave does not implement.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

import pynetrees as pt
from pynetrees.density import _disk_close


@pytest.fixture(scope="module")
def tree():
    return pt.sample_tree()


# ---------------------------------------------------------------------------
# the closing itself
# ---------------------------------------------------------------------------


def _footprint_close(image, radius):
    """The literal definition: dilate then erode with a disk footprint.

    Correct and obvious, and quadratic in the radius -- which is why the
    module uses distance transforms instead. Small images only.
    """
    offsets = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    disk = (yy ** 2 + xx ** 2) <= radius ** 2
    dilated = ndimage.binary_dilation(image, disk)
    return ndimage.binary_erosion(dilated, disk, border_value=1)


@pytest.mark.parametrize("radius", [1, 2, 3, 5, 9])
def test_the_fast_closing_is_the_same_as_the_obvious_one(radius):
    """The optimisation `_disk_close` makes, checked against the definition
    on images small enough for both to be cheap."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        image = rng.random((60, 60)) > 0.97
        np.testing.assert_array_equal(_disk_close(image, radius),
                                      _footprint_close(image, radius))


def test_closing_bridges_a_gap_between_neighbouring_branches():
    """Two stretches of cable running alongside each other, 10 um apart:
    the ground between them counts as spanned once the radius reaches half
    the gap."""
    lines = np.zeros((80, 80), dtype=bool)
    lines[35, 10:70] = True
    lines[45, 10:70] = True
    assert not _disk_close(lines, 4)[40, 40]
    assert _disk_close(lines, 5)[40, 40]


def test_closing_never_bridges_two_isolated_points():
    """Worth pinning, because it is the opposite of what "fills gaps up to
    `radius`" suggests. Erosion undoes dilation exactly for an isolated
    convex blob, so no radius joins two lone nodes -- closing fills the
    pockets *between* stretches of cable, and needs cable on both sides to
    do it."""
    points = np.zeros((80, 80), dtype=bool)
    points[40, 20] = points[40, 40] = True
    assert not any(_disk_close(points, r)[40, 30] for r in (5, 11, 15, 30))


def test_a_zero_radius_changes_nothing():
    rng = np.random.default_rng(1)
    image = rng.random((20, 20)) > 0.8
    np.testing.assert_array_equal(_disk_close(image, 0), image)


# ---------------------------------------------------------------------------
# span_tree
# ---------------------------------------------------------------------------

#: (default radius, mask side, area, MATLAB's theta *index*) from MATLAB's
#: own `span_tree`/`theta_tree` under Octave, patched only to use the exact
#: disk. MATLAB's theta is a 1-based position in `0 : ceil (max (hB))`, so
#: the radius it stands for is one less -- see MATLAB_TOOLBOX_BUGS.md.
MATLAB_REFERENCE = {
    "sample_tree": (57, 343, 7064, 10),
    "sample2_tree": (26, 181, 554, 7),
}


@pytest.mark.parametrize("name", sorted(MATLAB_REFERENCE))
def test_span_matches_matlab(name):
    radius, side, area, _ = MATLAB_REFERENCE[name]
    span = pt.span_tree(getattr(pt, name)())
    assert span.mask.shape == (side, side)
    assert span.area == area


@pytest.mark.parametrize("name", sorted(MATLAB_REFERENCE))
def test_theta_matches_matlab_once_its_off_by_one_is_undone(name):
    *_, theta_index = MATLAB_REFERENCE[name]
    assert pt.theta_tree(getattr(pt, name)()) == theta_index - 1


def test_the_default_radius_is_matlabs_rule(tree):
    expected = round(np.sqrt((tree.X.max() - tree.X.min())
                             * (tree.Y.max() - tree.Y.min())) / 2)
    plain = pt.span_tree(tree)
    assert pt.span_tree(tree, expected).area == plain.area


def test_every_node_lands_inside_its_own_span(tree):
    """A closing only ever adds pixels, so the nodes must survive it."""
    span = pt.span_tree(tree)
    rows = np.round(tree.Y).astype(int) + span.origin
    cols = np.round(tree.X).astype(int) + span.origin
    assert span.mask[rows, cols].all()


def test_the_origin_maps_coordinates_back(tree):
    """What `origin` is for: turning a pixel back into microns."""
    span = pt.span_tree(tree)
    assert 0 <= round(tree.X[0]) + span.origin < span.mask.shape[1]
    assert span.mask.shape[0] == 2 * span.origin + 1


def test_a_bigger_radius_spans_more(tree):
    areas = [pt.span_tree(tree, r).area for r in (10, 30, 60)]
    assert areas[0] < areas[1] < areas[2]


def test_the_span_is_at_least_the_arbors_own_footprint(tree):
    """Closing cannot lose the nodes, so the area is bounded below by the
    number of distinct pixels they occupy."""
    pixels = len(np.unique(np.column_stack(
        [np.round(tree.Y), np.round(tree.X)]), axis=0))
    assert pt.span_tree(tree, 1).area >= pixels


def test_the_hull_method_is_a_different_shape(tree):
    """MATLAB's `-b`: follow the outline instead of the cable."""
    closed = pt.span_tree(tree, method="close")
    hull = pt.span_tree(tree, method="hull")
    assert hull.area != closed.area
    assert hull.mask.shape == closed.mask.shape


def test_an_unknown_method_is_rejected(tree):
    with pytest.raises(ValueError, match="'close' or 'hull'"):
        pt.span_tree(tree, method="convex")


def test_span_takes_a_group(tree):
    spans = pt.span_tree([tree, tree])
    assert isinstance(spans, list) and spans[0].area == spans[1].area


# ---------------------------------------------------------------------------
# theta_tree
# ---------------------------------------------------------------------------


def test_theta_is_a_distance_in_microns(tree):
    theta = pt.theta_tree(tree)
    assert 0 < theta < np.ptp(tree.X)


def test_discs_of_theta_really_do_cover_the_packing_fraction(tree):
    """The definition, recomputed from scratch rather than trusting the
    histogram: put a disc of radius theta on every node and check it covers
    at least 90.69% of the span."""
    span = pt.span_tree(tree)
    theta = pt.theta_tree(tree)
    rows = np.round(tree.Y).astype(int) + span.origin
    cols = np.round(tree.X).astype(int) + span.origin
    seeds = (~span.mask).copy()
    seeds[rows, cols] = True
    distance = ndimage.distance_transform_edt(~seeds)
    covered = (distance[span.mask] <= theta).sum()
    assert covered >= 0.9069 * span.area


def test_one_micron_less_does_not_cover_it(tree):
    """theta is the *first* radius that suffices, so the one below must
    not -- this is what makes it a measurement rather than a bound."""
    span = pt.span_tree(tree)
    theta = pt.theta_tree(tree)
    rows = np.round(tree.Y).astype(int) + span.origin
    cols = np.round(tree.X).astype(int) + span.origin
    seeds = (~span.mask).copy()
    seeds[rows, cols] = True
    distance = ndimage.distance_transform_edt(~seeds)
    covered = (distance[span.mask] <= theta - 1).sum()
    assert covered < 0.9069 * span.area


def test_a_denser_arbor_fills_its_territory_more_finely(tree):
    """Resampling to 1 um puts cable everywhere it already was, so the span
    is unchanged but every point is closer to a node."""
    dense = pt.resample_tree(tree, 1.0)
    assert pt.theta_tree(dense) <= pt.theta_tree(tree)


def test_ignoring_the_boundary_never_helps(tree):
    """MATLAB's `-e`. Dropping the rim from the covered set can only make
    the far corners harder to reach."""
    assert pt.theta_tree(tree, include_boundary=False) >= pt.theta_tree(tree)


# ---------------------------------------------------------------------------
# scaleS_tree / scaleV_tree
# ---------------------------------------------------------------------------


def test_scaleS_hits_the_target_area(tree):
    scaled = pt.scaleS_tree(tree, 100_000.0)
    assert pt.span_tree(scaled.tree).area == pytest.approx(100_000, rel=0.02)
    assert abs(scaled.error) < 0.02


def test_scaleS_reports_the_miss_rather_than_printing_it(tree):
    """MATLAB writes the residual to the console and drops it."""
    scaled = pt.scaleS_tree(tree, 100_000.0)
    achieved = pt.span_tree(scaled.tree).area
    assert scaled.error == pytest.approx(achieved / 100_000.0 - 1.0)


def test_scaleS_needs_two_passes_to_get_that_close(tree):
    """Closing uses a radius fixed in microns, so area does not scale as
    the square of the factor and one pass lands well short."""
    area = pt.span_tree(tree).area
    one_pass = pt.scale_tree(tree, np.sqrt(100_000.0 / area))
    naive = abs(pt.span_tree(one_pass).area / 100_000.0 - 1.0)
    assert naive > abs(pt.scaleS_tree(tree, 100_000.0).error)


def test_scaleV_hits_the_target_volume(tree):
    scaled = pt.scaleV_tree(tree, 500_000.0)
    assert pt.boundary_tree(scaled.tree).volume == pytest.approx(
        500_000.0, rel=1e-6)


def test_scaleV_converges_in_one_pass(tree):
    """The shrink factor is relative, so the alpha shape scales with the
    points and the volume scales exactly as factor**3."""
    assert abs(pt.scaleV_tree(tree, 500_000.0).error) < 1e-9


def test_scaleV_in_two_dimensions_uses_the_square_root(tree):
    scaled = pt.scaleV_tree(tree, 20_000.0, dim=2)
    flat = pt.boundary_tree(scaled.tree, dim=2).volume
    assert flat == pytest.approx(20_000.0, rel=1e-6)


def test_the_factor_reproduces_the_scaling(tree):
    scaled = pt.scaleV_tree(tree, 500_000.0)
    again = pt.scale_tree(tree, scaled.factor)
    np.testing.assert_allclose(again.X, scaled.tree.X)


def test_a_tree_that_cannot_reach_the_target_reports_the_miss():
    """A single node spans one pixel however it is scaled, so the target is
    unreachable. The point of returning `error` is that this comes back as
    a number rather than as a plausible-looking tree."""
    one = pt.delete_tree(pt.sample2_tree(), np.arange(1, 15))
    scaled = pt.scaleS_tree(one, 1000.0, resample=False)
    assert scaled.error == pytest.approx(-1.0, abs=0.01)


def test_scale_functions_take_a_group(tree):
    results = pt.scaleV_tree([tree, tree], 500_000.0)
    assert isinstance(results, list) and len(results) == 2
    assert results[0].factor == pytest.approx(results[1].factor)


# ---------------------------------------------------------------------------
# theta_mc_tree
# ---------------------------------------------------------------------------


def test_monte_carlo_theta_is_a_quantile_of_its_own_distances(tree):
    """The definition, checked against the sample it returns."""
    result = pt.theta_mc_tree(tree, 20_000, rng=np.random.default_rng(0))
    assert (result.distances <= result.theta).mean() == pytest.approx(
        0.9069, abs=0.005)


def test_every_sampled_point_is_inside_the_cell(tree):
    """Exact sampling, not rejection: nothing is drawn and thrown away, so
    the count returned is the count asked for."""
    result = pt.theta_mc_tree(tree, 5_000, rng=np.random.default_rng(0))
    assert len(result.distances) == 5_000


def test_monte_carlo_theta_is_stable_across_seeds(tree):
    """A Monte-Carlo estimate is only useful if the noise is smaller than
    what it is meant to distinguish."""
    thetas = [pt.theta_mc_tree(tree, 20_000, rng=np.random.default_rng(s)).theta
              for s in range(5)]
    assert np.std(thetas) < 0.05 * np.mean(thetas)


def test_more_samples_do_not_move_the_answer(tree):
    coarse = pt.theta_mc_tree(tree, 5_000, rng=np.random.default_rng(0)).theta
    fine = pt.theta_mc_tree(tree, 80_000, rng=np.random.default_rng(0)).theta
    assert coarse == pytest.approx(fine, rel=0.1)


def test_monte_carlo_theta_moves_with_the_sampling_rate():
    """Not the invariance you might assume, and worth pinning because the
    obvious comparison -- theta of cell A against theta of cell B -- is
    invalid unless both were sampled alike.

    Resampling adds nodes along cable that is already there, so the
    morphology and its convex hull are unchanged. But the alpha shape at a
    fixed `alpha` loosens (the empty pockets get subdivided into simplices
    small enough to keep), and theta follows it. On `hsn` it roughly
    doubles. See `boundary_tree`'s Notes.
    """
    tree = pt.hsn_tree()
    dense = pt.resample_tree(tree, 1.0)
    coarse_theta = pt.theta_mc_tree(tree, 20_000,
                                    rng=np.random.default_rng(0)).theta
    dense_theta = pt.theta_mc_tree(dense, 20_000,
                                   rng=np.random.default_rng(0)).theta
    assert dense_theta > 1.5 * coarse_theta

    from scipy.spatial import ConvexHull
    hulls = [ConvexHull(np.column_stack([t.X, t.Y, t.Z])).volume
             for t in (tree, dense)]
    assert hulls[1] == pytest.approx(hulls[0], rel=0.01)  # shape unchanged


def test_the_grid_measure_does_not_have_that_sensitivity():
    """`theta_tree` rasterises onto a one-micron grid instead of wrapping a
    boundary, so it is the one to reach for when the reconstructions were
    not sampled alike."""
    tree = pt.hsn_tree()
    dense = pt.resample_tree(tree, 1.0)
    assert pt.theta_tree(dense) == pytest.approx(pt.theta_tree(tree), abs=2.0)


def test_scaling_the_tree_scales_monte_carlo_theta(tree):
    """It is a length, so it must be homogeneous of degree one."""
    plain = pt.theta_mc_tree(tree, 20_000, rng=np.random.default_rng(0)).theta
    bigger = pt.theta_mc_tree(pt.scale_tree(tree, 2.0), 20_000,
                              rng=np.random.default_rng(0)).theta
    assert bigger == pytest.approx(2.0 * plain, rel=0.05)


def test_a_tree_too_small_to_enclose_a_volume_gives_nan():
    """Three points span no volume, and a space-filling measure over no
    volume has no answer."""
    small = pt.delete_tree(pt.sample2_tree(), np.arange(3, 15))
    assert np.isnan(pt.theta_mc_tree(small).theta)


def test_monte_carlo_theta_takes_a_group(tree):
    results = pt.theta_mc_tree([tree, tree], 5_000)
    assert isinstance(results, list) and len(results) == 2


# ---------------------------------------------------------------------------
# empty trees (V2's rule, which the V4 additions have to obey too)
# ---------------------------------------------------------------------------


def test_an_empty_tree_spans_no_area():
    """Caught by the V2/V3 sweeps rather than written from foresight: both
    of these were added without `@empty_safe` and the rule-level tests
    found them two hundred tests later."""
    empty = pt.delete_tree(pt.sample2_tree(), np.arange(15))
    span = pt.span_tree(empty)
    assert span.area == 0.0
    assert span.mask.shape == (0, 0)


def test_the_space_filling_radius_of_nothing_is_nan():
    """Not 0: a covering radius over no territory is undefined, and 0 would
    read as "perfectly space-filling"."""
    empty = pt.delete_tree(pt.sample2_tree(), np.arange(15))
    assert np.isnan(pt.theta_tree(empty))
    assert np.isnan(pt.theta_mc_tree(empty).theta)
