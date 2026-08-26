"""B2: `M_atten_tree`, `angleBd_tree`/`angleBd2_tree`, `boundary_tree`,
`convexity_tree`.

Verification status differs by function, and the docstrings say which:
`M_atten_tree` and the `angleBd` pair are pure computation on the tree and
are checked against their defining properties; `boundary_tree` and
`convexity_tree` rest on MATLAB's built-in `boundary()`, which Octave does
not implement, so **no differential check against MATLAB was possible** and
they are checked against geometric properties only.
"""

from __future__ import annotations

import numpy as np
import pytest

import pynetrees as pt


@pytest.fixture(scope="module")
def tree():
    return pt.sample_tree()


@pytest.fixture(scope="module")
def big():
    return pt.hsn_tree()


# ---------------------------------------------------------------------------
# angleBd_tree / angleBd2_tree
# ---------------------------------------------------------------------------


def test_one_angle_per_branch_point(tree):
    angles = pt.angleBd_tree(tree, dist=5)
    assert len(angles) == int(pt.B_tree(tree).sum())


def test_angles_are_radians_in_range(tree):
    angles = pt.angleBd_tree(tree, dist=5)
    finite = angles[np.isfinite(angles)]
    assert (finite >= 0).all() and (finite <= np.pi).all()


def test_walking_further_smooths_the_measurement(tree):
    """The reason these exist next to `angleB_tree`.

    `angleB_tree` measures from the immediate daughters, so a single
    jittered node swings it. Walking out several nodes first describes
    where the branches actually go.
    """
    immediate = np.nanmean(pt.angleB_tree(tree))
    walked = np.nanmean(pt.angleBd_tree(tree, dist=10))
    assert immediate != pytest.approx(walked)


def test_the_two_variants_genuinely_differ(big):
    """`angleBd` follows the bulkier branch, `angleBd2` the longer-reaching
    one. They agree only where bulk and reach happen to coincide."""
    a = pt.angleBd_tree(big, dist=15)
    b = pt.angleBd2_tree(big, dist=15)
    differing = np.nansum(np.abs(a - b) > 1e-9)
    assert differing > 0


def test_variants_coincide_on_a_tree_too_simple_to_separate_them(tree):
    """Not a bug: on the 197-node sample the walks rarely meet an
    intermediate branch point where bulk and reach disagree."""
    a = pt.angleBd_tree(tree, dist=5)
    b = pt.angleBd2_tree(tree, dist=5)
    np.testing.assert_allclose(a, b)


def test_dist_below_two_is_rejected(tree):
    with pytest.raises(ValueError, match="at least 2"):
        pt.angleBd_tree(tree, dist=1)


def test_walk_stops_cleanly_at_a_terminal(tree):
    """A `dist` longer than any branch must not run off the end."""
    angles = pt.angleBd_tree(tree, dist=500)
    assert np.isfinite(angles).any()


# ---------------------------------------------------------------------------
# M_atten_tree
# ---------------------------------------------------------------------------


def _passive(tree):
    tree = pt.Tree(dA=tree.dA, X=tree.X, Y=tree.Y, Z=tree.Z, D=tree.D,
                   R=tree.R, rnames=tree.rnames, name=tree.name)
    tree.Ri, tree.Gm = 100.0, 1.0 / 2500.0
    return tree


def test_compartment_count_is_at_least_one(tree):
    assert pt.M_atten_tree(_passive(tree)) >= 1


def test_a_stricter_threshold_finds_more_compartments(tree):
    """Higher `thr` means nodes must be more strongly coupled to count as
    one compartment, so the tree fragments into more of them."""
    passive = _passive(tree)
    loose = pt.M_atten_tree(passive, thr=0.05)
    strict = pt.M_atten_tree(passive, thr=0.9)
    assert strict > loose


def test_requires_electrotonic_properties(tree):
    """Note the bundled samples *do* carry Ri/Gm/Cm -- the `.mtr` loader
    reads them straight out of the MATLAB struct -- so a tree without them
    has to be built explicitly here."""
    bare = pt.Tree(dA=tree.dA, X=tree.X, Y=tree.Y, Z=tree.Z, D=tree.D,
                   R=tree.R, rnames=tree.rnames)
    assert bare.Ri is None
    with pytest.raises(ValueError):
        pt.M_atten_tree(bare)


# ---------------------------------------------------------------------------
# boundary_tree
# ---------------------------------------------------------------------------


def test_shrink_zero_is_the_convex_hull(tree):
    from scipy.spatial import ConvexHull

    bound = pt.boundary_tree(tree, shrink=0.0)
    expected = ConvexHull(np.column_stack([tree.X, tree.Y, tree.Z]))
    assert len(bound.vertices) == len(expected.vertices)
    assert bound.volume == pytest.approx(expected.volume, rel=1e-9)


def test_tighter_shrink_uses_more_boundary_points(tree):
    """A concave boundary follows the arbor, so it needs more vertices than
    the convex hull's handful of extremes."""
    loose = pt.boundary_tree(tree, shrink=0.0)
    tight = pt.boundary_tree(tree, shrink=1.0)
    assert len(tight.vertices) > len(loose.vertices)
    assert tight.volume < loose.volume


def test_shrink_sweeps_the_whole_family_not_just_its_end(tree):
    """The cutoff is interpolated by rank, not by radius.

    Interpolating the circumradius cutoff linearly left the shape
    indistinguishable from the convex hull until shrink was past 0.9 --
    the radii are skewed enough that most of the [0, 1] dial did nothing.
    """
    volumes = [pt.boundary_tree(tree, shrink=s).volume
               for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert volumes == sorted(volumes, reverse=True)
    # every quarter turn of the dial must actually move the shape
    assert all(a > b * 1.05 for a, b in zip(volumes, volumes[1:]))


def test_boundary_envelops_every_point(tree):
    """MATLAB documents even the tightest shrink as *enveloping* the points.

    An earlier cutoff rule kept only the single smallest simplex at
    shrink=1, returning a 4-vertex sliver of a 197-node tree.
    """
    from scipy.spatial import Delaunay

    coords = np.column_stack([tree.X, tree.Y, tree.Z])
    bound = pt.boundary_tree(tree, shrink=1.0)
    envelope = Delaunay(bound.vertices)
    assert (envelope.find_simplex(coords) >= 0).mean() > 0.95


def test_convexity_is_matlabs_spelling_of_the_shrink_factor(tree):
    """MATLAB parameterises `boundary_tree` by convexity, with
    ``shrink = 1 - c``, so a convex cell is wrapped loosely."""
    assert pt.boundary_tree(tree, c=0.4).volume == pytest.approx(
        pt.boundary_tree(tree, shrink=0.6).volume
    )


def test_shrink_and_c_together_are_rejected(tree):
    with pytest.raises(ValueError, match="not both"):
        pt.boundary_tree(tree, shrink=0.3, c=0.4)


def test_shrink_outside_the_unit_interval_is_rejected(tree):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        pt.boundary_tree(tree, shrink=1.5)


def test_two_dimensional_boundary(tree):
    bound = pt.boundary_tree(tree, shrink=0.0, dim=2)
    assert bound.vertices.shape[1] == 2
    assert bound.faces.shape[1] == 2  # edges, not triangles
    assert bound.polygon is not None


def test_two_dimensional_polygon_is_a_closed_ring(tree):
    """`dissectSholl_tree` casts rays against this, so consecutive rows
    have to be joined by an actual edge -- the raw edge list is not
    ordered."""
    bound = pt.boundary_tree(tree, shrink=0.6, dim=2)
    edges = {tuple(sorted(e)) for e in bound.faces.tolist()}
    ring = bound.polygon
    lookup = {tuple(v): i for i, v in enumerate(bound.vertices.tolist())}
    walked = [lookup[tuple(v)] for v in ring.tolist()]
    for a, b in zip(walked, walked[1:] + walked[:1]):
        assert tuple(sorted((a, b))) in edges


def test_three_dimensional_boundary_has_no_polygon(tree):
    assert pt.boundary_tree(tree).polygon is None


def test_simplex_fill_measures_the_enclosed_volume(tree):
    """`volume` sums the filled simplices rather than measuring the outer
    envelope, so a lobed or perforated shape is not over-counted."""
    from pynetrees.density import _simplex_volumes

    bound = pt.boundary_tree(tree, shrink=0.7)
    assert bound.volume == pytest.approx(
        _simplex_volumes(bound.points, bound.simplices).sum()
    )


# ---------------------------------------------------------------------------
# convexity_tree
# ---------------------------------------------------------------------------


def test_convexity_is_a_fraction(tree):
    value = pt.convexity_tree(tree, thr=25.0, rng=0)
    assert 0.0 <= value <= 1.0


def test_a_generously_dilated_arbor_becomes_convex(tree):
    """Grow the occupied volume enough and the gaps fill in."""
    assert pt.convexity_tree(tree, thr=300.0, rng=0) == pytest.approx(1.0)


def test_a_thin_arbor_is_not_convex(tree):
    """At a small threshold the cell is a thin branching structure, and
    most straight lines between its tips pass through empty space."""
    assert pt.convexity_tree(tree, thr=8.0, rng=0) < 0.9


def test_convexity_rises_with_threshold(tree):
    values = [pt.convexity_tree(tree, thr=t, rng=0) for t in (10.0, 25.0, 60.0)]
    assert values[0] < values[1] <= values[2]


def test_too_few_points_is_rejected(tree):
    with pytest.raises(ValueError, match="at least 2"):
        pt.convexity_tree(tree, nodes=[0])


def test_subsampling_is_reproducible(tree):
    a = pt.convexity_tree(tree, thr=12.0, max_pairs=50, rng=3)
    b = pt.convexity_tree(tree, thr=12.0, max_pairs=50, rng=3)
    assert a == b
