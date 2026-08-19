"""B2: `r_mc_tree` and `dissectSholl_tree`.

Both rest on `boundary_tree`, hence on MATLAB's built-in ``boundary()``,
which Octave does not implement -- so, as with `boundary_tree` and
`convexity_tree` themselves, **no differential check against MATLAB was
possible on this machine**. They are checked here against the properties
their statistics must have, and against reference point sets whose answers
are known independently of any implementation.

The `r_mc_tree` tests lean on that second kind: the Clark-Evans ratio has
known limits for a Poisson cloud (1) and for a lattice (~2), which pin the
estimator without needing MATLAB at all.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.sparse import csr_matrix

import pytrees as pt


@pytest.fixture(scope="module")
def tree():
    return pt.sample_tree()


def _cloud(points: np.ndarray) -> pt.Tree:
    """A path tree through arbitrary points, so `r_mc_tree` can be pointed
    at a point set with a known answer rather than only at neurons."""
    n = len(points)
    rows = np.arange(1, n)
    dA = csr_matrix((np.ones(n - 1), (rows, rows - 1)), shape=(n, n))
    return pt.Tree(dA=dA, X=points[:, 0].copy(), Y=points[:, 1].copy(),
                   Z=points[:, 2].copy(), D=np.ones(n),
                   R=np.zeros(n, dtype=int), rnames=["dendrite"])


# ---------------------------------------------------------------------------
# r_mc_tree
# ---------------------------------------------------------------------------


def test_a_uniform_cloud_scores_about_one():
    """The whole point of the statistic: R = 1 means "no more ordered than
    chance". Points drawn uniformly are the null, so they must land there.

    This is the test that would catch a wrong volume, a wrong sampler, or a
    biased nearest-neighbour estimate -- none of which need MATLAB to
    detect.
    """
    rng = np.random.default_rng(4)
    cloud = _cloud(rng.random((400, 3)) * 100.0)
    assert pt.r_mc_tree(cloud, n_mc=30, rng=1).R == pytest.approx(1.0, abs=0.12)


def test_a_lattice_scores_well_above_one():
    """A perfectly regular grid is the opposite extreme; Clark & Evans put
    it near 2."""
    axis = np.linspace(0.0, 90.0, 7)
    grid = np.stack(np.meshgrid(axis, axis, axis), -1).reshape(-1, 3)
    assert pt.r_mc_tree(_cloud(grid), n_mc=20, rng=1).R > 1.6


def test_a_neurons_nodes_are_strongly_clustered(tree):
    """Reconstruction nodes sit a micron or two apart along cable, so as a
    point set they are far more clustered than chance -- R well below 1.
    That is a property of the sampling, not of the cell."""
    assert pt.r_mc_tree(tree, n_mc=20, rng=0).R < 0.7


def test_termination_points_are_more_regular_than_the_nodes(tree):
    """And this is the measurement the function exists for: strip out the
    sampling and the remaining extremities are spread out, not clumped."""
    nodes = pt.r_mc_tree(tree, n_mc=20, rng=0).R
    tips = pt.r_mc_tree(tree, n_mc=20, nodes="t", rng=0).R
    assert tips > nodes


def test_node_selectors_pick_the_documented_subsets(tree):
    counts = {
        sel: pt.r_mc_tree(tree, n_mc=2, nodes=sel, rng=0).n
        for sel in ("all", "bt", "b", "t")
    }
    assert counts["all"] == tree.n_nodes
    assert counts["bt"] == counts["b"] + counts["t"]
    assert counts["b"] == int(pt.B_tree(tree).sum())
    assert counts["t"] == int(pt.T_tree(tree).sum())


def test_an_unknown_selector_says_what_is_allowed(tree):
    with pytest.raises(ValueError, match="'all', 'bt', 'b', 't'"):
        pt.r_mc_tree(tree, n_mc=1, nodes="terminals")


def test_volume_correction_raises_the_null_estimate(tree):
    """A finite sample never reaches the boundary, so its own hull is
    smaller than the reference one; correcting for that spreads the points
    out and lengthens the expected nearest-neighbour distance.

    MATLAB's flag for this is inverted against both its own name (``-nv``,
    "no volume correction") and its own documented default -- see the
    function's Notes.
    """
    on = pt.r_mc_tree(tree, n_mc=20, rng=0, volume_correction=True)
    off = pt.r_mc_tree(tree, n_mc=20, rng=0, volume_correction=False)
    assert on.rE > off.rE
    assert on.R < off.R


def test_confidence_bounds_bracket_the_estimate(tree):
    result = pt.r_mc_tree(tree, n_mc=5, n_boot=200, confidence=True, rng=0)
    assert result.Rmin < result.R < result.Rmax


def test_bounds_are_nan_when_not_asked_for(tree):
    """Rather than silently returning the point estimate for all three."""
    result = pt.r_mc_tree(tree, n_mc=3, rng=0)
    assert np.isnan(result.Rmin) and np.isnan(result.Rmax)


def test_the_null_distribution_is_returned_not_just_summarised(tree):
    result = pt.r_mc_tree(tree, n_mc=12, rng=0)
    assert result.rEs.shape == (12,)
    assert result.rE == pytest.approx(result.rEs.mean())
    assert result.rEstd == pytest.approx(result.rEs.std(ddof=1))


def test_two_dimensional_analysis_ignores_z(tree):
    """Collapsing a tree onto the plane brings its points closer together,
    so the same cell reads as more clustered in 2D."""
    flat = pt.r_mc_tree(tree, n_mc=15, dim=2, rng=0)
    assert 0.0 < flat.R < 1.0


def test_a_single_point_has_no_nearest_neighbour(tree):
    with pytest.raises(ValueError, match="at least 2"):
        pt.r_mc_tree(tree, n_mc=1, nodes=[0])


def test_sampling_is_reproducible(tree):
    a = pt.r_mc_tree(tree, n_mc=5, rng=7).R
    b = pt.r_mc_tree(tree, n_mc=5, rng=7).R
    assert a == b


def test_uniform_sampling_really_is_uniform():
    """The sampler draws from the boundary's simplex decomposition rather
    than rejecting bounding-box points, so it is worth checking directly
    that it has not acquired a bias towards, say, small simplices."""
    from pytrees.density import _alpha_shape, _sample_in_simplices

    rng = np.random.default_rng(0)
    corners = np.array(np.meshgrid([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])).reshape(3, -1).T
    box = _alpha_shape(np.vstack([corners, rng.random((60, 3))]), 0.0)
    points = _sample_in_simplices(box.points, box.simplices, 40000, rng)
    # a uniform fill of the unit cube: mean 0.5 per axis, and each octant
    # holds an eighth of the points
    assert points.mean(axis=0) == pytest.approx(0.5, abs=0.02)
    octant = ((points > 0.5).astype(int) * [1, 2, 4]).sum(axis=1)
    assert np.bincount(octant, minlength=8) / len(points) == pytest.approx(
        0.125, abs=0.01
    )


# ---------------------------------------------------------------------------
# dissectSholl_tree
# ---------------------------------------------------------------------------


def test_every_profile_is_normalised(tree):
    """The comparison is of shape, not magnitude -- the magnitude is
    reported separately as `scale`."""
    result = pt.dissectSholl_tree(tree, density=True, rng=0)
    for profile in (result.observed, result.domain, result.angle, result.density):
        assert trapezoid(profile, result.radii) == pytest.approx(1.0)


def test_radii_span_the_whole_cell(tree):
    result = pt.dissectSholl_tree(tree, c=0.9, centripetal=False, rng=0)
    assert result.radii[0] == 0.0
    assert result.radii[-1] == pytest.approx(pt.eucl_tree(pt.tran_tree(tree)).max())
    assert len(result.radii) == 25


def test_optional_profiles_are_absent_rather_than_empty(tree):
    """Not zeros, not NaNs: the analysis simply was not run."""
    result = pt.dissectSholl_tree(tree, c=0.9, centripetal=False, rng=0)
    assert result.angle is None and result.k is None and result.bf is None
    assert result.err_angle is None
    assert result.density is None and result.err_density is None


def test_the_domain_prediction_is_in_the_right_ballpark(tree):
    """It is a null model, so it should be wrong -- but recognisably the
    same profile, not noise. A broken sphere-sampling step would show up
    here as an error near the size of the profile itself."""
    result = pt.dissectSholl_tree(tree, c=0.9, centripetal=False, rng=0)
    assert 0.0 < result.err_domain < 0.5 * trapezoid(result.observed, result.radii)


def test_the_domain_profile_vanishes_beyond_the_cell(tree):
    """No part of a sphere larger than the territory lies inside it."""
    result = pt.dissectSholl_tree(tree, c=0.9, centripetal=False, rng=0)
    assert result.domain[0] == 0.0
    assert result.domain[-1] < result.domain.max() * 0.2


def test_supplying_convexity_skips_recomputing_it(tree):
    assert pt.dissectSholl_tree(tree, c=0.42, centripetal=False, rng=0).c == 0.42


def test_convexity_sets_how_tightly_the_territory_is_drawn(tree):
    """`c` reaches the boundary as ``shrink = 1 - c``, so a cell declared
    convex gets the loose convex hull and a concave one gets a tight
    wrap."""
    convex = pt.dissectSholl_tree(tree, c=1.0, centripetal=False, rng=0)
    concave = pt.dissectSholl_tree(tree, c=0.0, centripetal=False, rng=0)
    assert convex.volume > concave.volume


def test_the_centripetal_fit_reports_its_parameters(tree):
    result = pt.dissectSholl_tree(tree, c=0.9, rng=0)
    assert result.k > 0
    assert 0.0 <= result.bf <= 1.0
    assert result.rootangle is not None
    assert (result.bf, result.k) == pt.bf_tree(result.rootangle, dim="3d")


def test_the_estimated_scale_approximates_the_measured_one(tree):
    """`est_scale` predicts the Sholl integral from cable length and root
    angles alone, without measuring a single intersection."""
    result = pt.dissectSholl_tree(tree, c=0.9, rng=0)
    assert result.est_scale == pytest.approx(result.scale, rel=0.25)


def test_scale_is_the_unnormalised_sholl_integral(tree):
    result = pt.dissectSholl_tree(tree, c=0.9, centripetal=False, rng=0)
    observed = pt.sholl_tree(pt.tran_tree(tree), 2 * result.radii,
                             warn_double=False).s
    assert result.scale == pytest.approx(trapezoid(observed, result.radii))


def test_the_size_fudge_is_reproduced_but_overridable(tree):
    """MATLAB's 3D branch silently doubles the estimated branch length for
    cells reaching past 500 um, with no explanation and no 2D counterpart.
    Kept for fidelity, but it must be possible to turn off."""
    default = pt.dissectSholl_tree(tree, c=0.9, rng=0)
    forced = pt.dissectSholl_tree(tree, c=0.9, scale_factor=2.0, rng=0)
    # the sample tree is well under 500 um, so the default is sf = 1
    assert default.err_angle != pytest.approx(forced.err_angle)


def test_two_dimensional_dissection(tree):
    result = pt.dissectSholl_tree(tree, c=0.9, dim=2, density=True, rng=0)
    assert trapezoid(result.domain, result.radii) == pytest.approx(1.0)
    assert result.volume > 0  # an area here, not a volume
    assert result.err_density is not None


def test_an_unsupported_dimension_is_rejected(tree):
    with pytest.raises(ValueError, match="dim must be 2 or 3"):
        pt.dissectSholl_tree(tree, c=0.9, dim=4)


def test_more_directions_do_not_change_the_answer_much(tree):
    """The domain profile is a Monte-Carlo estimate; if it were sensitive
    to the sample size at these settings the default would be too low."""
    coarse = pt.dissectSholl_tree(tree, c=0.9, centripetal=False,
                                  n_directions=4000, rng=0)
    fine = pt.dissectSholl_tree(tree, c=0.9, centripetal=False,
                                n_directions=40000, rng=0)
    assert coarse.err_domain == pytest.approx(fine.err_domain, abs=0.01)
