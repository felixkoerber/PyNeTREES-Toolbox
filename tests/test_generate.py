"""B4: the generative pipeline.

`gscale_tree`, `rpoints_tree`, `in_hull`, `clone_tree`, `dscam_tree`,
`spines_tree` and `PP_generator_tree`, plus `MST_tree`'s new
grow-onto-an-existing-tree mode, which `clone_tree` needs.

None of this can be diffed against MATLAB: every function here draws from a
random number generator, and MATLAB's stream cannot be reproduced from
numpy. What *can* be checked -- and is, below -- is that each function
delivers the statistic it promises: a clone lands inside the group's
measured spread, `rpoints_tree` reproduces the density it sampled,
`PP_generator_tree` hits the Clark-Evans ratio it was asked for, and
`spines_tree` puts spines perpendicular to the cable at the requested
length. Those are the claims the pipeline actually makes.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

import pynetrees as pt
from pynetrees.graphtheory import idpar_tree


@pytest.fixture(scope="module")
def tree():
    return pt.sample_tree()


@pytest.fixture(scope="module")
def group():
    """Ten HSN dendrites -- a real group, with soma/axon/dendrite regions."""
    return pt.dLPTCs_trees()["dhsn"]


# ---------------------------------------------------------------------------
# gscale_tree
# ---------------------------------------------------------------------------


def test_every_region_that_has_nodes_is_measured(group):
    spanning = pt.gscale_tree(group)
    assert set(spanning.names) == {"axon", "dendrite"}
    assert "dendrite" in spanning
    assert len(spanning.scaled_trees) == len(group)


def test_a_region_named_but_never_used_is_dropped(group):
    """These cells all declare a `soma` region that no node is assigned to.
    Keeping it would give `clone_tree` a soma of nothing to build from."""
    assert all("soma" in t.rnames for t in group)
    assert all((np.asarray(t.R) != t.rnames.index("soma")).all() for t in group)
    assert "soma" not in pt.gscale_tree(group)


def test_regions_are_looked_up_by_name_not_by_index(group):
    spanning = pt.gscale_tree(group)
    assert spanning["dendrite"].name == "dendrite"
    with pytest.raises(KeyError, match="apical"):
        spanning["apical"]


def test_one_row_per_tree_even_where_the_region_is_absent(tree):
    """Row `i` must always mean tree `i`, so an absent region is NaN rather
    than a shorter array."""
    other = pt.Tree(dA=tree.dA, X=tree.X, Y=tree.Y, Z=tree.Z, D=tree.D,
                    R=np.zeros(tree.n_nodes, dtype=int), rnames=["axon"])
    spanning = pt.gscale_tree([tree, other])
    dendrite = spanning["dendrite"]
    assert dendrite.extent.shape == (2, 3)
    assert np.isfinite(dendrite.extent[0]).all()
    assert np.isnan(dendrite.extent[1]).all()


def test_extents_are_measured_from_the_root(group):
    """Each tree is translated to the origin first, so the numbers describe
    the cell rather than its position in the slide's coordinate frame."""
    shifted = [pt.tran_tree(t, [1000.0, -500.0, 250.0]) for t in group]
    a = pt.gscale_tree(group)["dendrite"]
    b = pt.gscale_tree(shifted)["dendrite"]
    np.testing.assert_allclose(a.mean_extent, b.mean_extent)
    np.testing.assert_allclose(a.centre, b.centre)


def test_point_counts_exclude_continuation_points(group):
    """A continuation point says how finely the cell was traced, not how it
    branches, so including it would weight dense stretches more."""
    spanning = pt.gscale_tree(group)
    dendrite = spanning["dendrite"]
    for index, source in enumerate(group):
        nodes = dendrite.nodes[index]
        expected = int(((pt.B_tree(source) | pt.T_tree(source))[nodes]).sum())
        assert dendrite.n_points[index] == expected


def test_pooled_points_are_rescaled_to_the_group_mean(group):
    """The whole premise: after scaling, every cell's points describe the
    same territory, so pooling them is meaningful."""
    dendrite = pt.gscale_tree(group)["dendrite"]
    widths = np.array([np.ptp(p, axis=0) for p in dendrite.points if len(p)])
    # each cell's rescaled cloud now spans close to the group mean
    ratio = widths / dendrite.mean_extent
    assert 0.7 < np.median(ratio) < 1.3
    assert ratio.std() < np.median(widths / widths.mean(axis=0)) + 1.0


def test_a_flat_group_does_not_divide_by_zero(tree):
    """MATLAB substitutes 1 for a mean extent of 0, which is what a planar
    group's z axis gives."""
    flat = pt.flatten_tree(tree)
    spanning = pt.gscale_tree([flat, flat])
    assert spanning["dendrite"].mean_extent[2] == 1.0
    assert np.isfinite(spanning.scaled_trees[0].Z).all()


def test_wriggle_measures_excess_cable(tree):
    """Amplitude is how much longer the traced path is than the branch
    points alone would need, so a straightened tree scores ~0."""
    wriggly = pt.gscale_tree(tree).wriggle[0, 0]
    straight = pt.gscale_tree(
        pt.delete_tree(tree, np.flatnonzero(pt.C_tree(tree)))
    ).wriggle[0, 0]
    assert wriggly > straight
    assert straight == pytest.approx(0.0, abs=1e-9)


def test_a_single_tree_is_accepted(tree):
    assert len(pt.gscale_tree(tree).scaled_trees) == 1


# ---------------------------------------------------------------------------
# rpoints_tree
# ---------------------------------------------------------------------------


def _rebin(points, grid):
    """Bin `points` on `grid`'s own edges.

    Not `gdens_tree(points)`: that pads outward from whatever the data's
    own minimum happens to be, so two grids of the same `sr` need not share
    an origin and cannot be compared voxel by voxel.
    """
    counts, _ = np.histogramdd(points, bins=grid.edges)
    return counts


def test_sampled_points_reproduce_the_density(tree):
    """The point of the function. Sampling a tree's density and re-binning
    the sample on the same grid must give back the same picture."""
    grid = pt.gdens_tree(tree, sr=20.0)
    points = pt.rpoints_tree(grid, 20000, rng=0)
    resampled = _rebin(points, grid)
    assert np.corrcoef(grid.counts.ravel(), resampled.ravel())[0, 1] > 0.95


def test_points_land_inside_the_grid(tree):
    grid = pt.gdens_tree(tree, sr=20.0)
    points = pt.rpoints_tree(grid, 2000, rng=0)
    for axis, centres in enumerate((grid.x, grid.y, grid.z)):
        half = (centres[1] - centres[0]) / 2
        assert points[:, axis].min() >= centres[0] - half - 1e-9
        assert points[:, axis].max() <= centres[-1] + half + 1e-9


def test_empty_voxels_are_never_sampled(tree):
    """Weighted by count, so a voxel the cell never enters gets nothing."""
    grid = pt.gdens_tree(tree, sr=25.0)
    resampled = _rebin(pt.rpoints_tree(grid, 5000, rng=0), grid)
    assert resampled[grid.counts == 0].sum() == 0


def test_without_a_density_points_are_uniform():
    points = pt.rpoints_tree(None, 20000, x=[0, 10], y=[0, 10], rng=0)
    assert points.mean(axis=0)[:2] == pytest.approx([5.0, 5.0], abs=0.1)
    assert (points[:, 2] == 0).all()


def test_mismatched_box_sides_warn():
    with pytest.warns(UserWarning, match="isotropically"):
        pt.rpoints_tree(None, 10, x=[0, 10], y=[0, 100], rng=0)


def test_an_empty_density_is_rejected():
    with pytest.raises(ValueError, match="nothing to sample"):
        pt.rpoints_tree(np.zeros((4, 4, 4)), 10, rng=0)


def test_sampling_is_reproducible(tree):
    grid = pt.gdens_tree(tree, sr=20.0)
    np.testing.assert_array_equal(
        pt.rpoints_tree(grid, 500, rng=3), pt.rpoints_tree(grid, 500, rng=3)
    )


def test_a_boundary_filters_the_sample():
    ring = np.array([[np.cos(a), np.sin(a)] for a in
                     np.linspace(0, 2 * np.pi, 60)]) * 30
    points = pt.rpoints_tree(None, 4000, x=[-50, 50], boundary=[ring], rng=0)
    assert len(points) < 4000  # some were dropped
    assert (np.linalg.norm(points[:, :2], axis=1) <= 30.001).all()


# ---------------------------------------------------------------------------
# in_hull
# ---------------------------------------------------------------------------


def _circle(radius, n=64):
    angles = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([np.cos(angles), np.sin(angles)]) * radius


def test_the_largest_ring_is_the_outside_and_the_rest_are_holes():
    """The one piece of `in_c` that is not just unpacking MATLAB's packed
    contour format."""
    inside = pt.in_hull([[0, 0], [5, 0], [20, 0]],
                        [_circle(10, 64), _circle(3, 30)])
    assert inside.tolist() == [False, True, False]


def test_a_single_ring_has_no_holes():
    assert pt.in_hull([[0, 0]], [_circle(10)]).tolist() == [True]


def test_no_rings_means_nothing_is_inside():
    assert pt.in_hull([[0, 0], [1, 1]], []).tolist() == [False, False]


# ---------------------------------------------------------------------------
# MST_tree growing onto an existing tree
# ---------------------------------------------------------------------------


def test_growth_preserves_the_seed_tree(tree):
    rng = np.random.default_rng(0)
    cloud = rng.uniform(-100, 100, (120, 3))
    grown = pt.MST_tree(cloud[:, 0], cloud[:, 1], cloud[:, 2], start=tree,
                        bf=0.4, thr=60.0)
    assert grown.n_nodes > tree.n_nodes
    np.testing.assert_allclose(grown.X[:tree.n_nodes], tree.X)
    np.testing.assert_allclose(grown.D[:tree.n_nodes], tree.D)
    assert (grown.dA[:tree.n_nodes, :tree.n_nodes] != tree.dA).nnz == 0


def test_grown_nodes_land_in_their_own_region(tree):
    rng = np.random.default_rng(0)
    cloud = rng.uniform(-100, 100, (120, 3))
    grown = pt.MST_tree(cloud[:, 0], cloud[:, 1], cloud[:, 2], start=tree,
                        bf=0.4, thr=60.0)
    assert grown.rnames == list(tree.rnames) + ["new"]
    added = np.asarray(grown.R)[tree.n_nodes:]
    assert (added == grown.rnames.index("new")).all()
    np.testing.assert_array_equal(np.asarray(grown.R)[:tree.n_nodes], tree.R)


def test_the_seeds_path_length_carries_into_the_balancing_term(tree):
    """Without it the new material is grown as if the seed had no extent,
    so `bf` would stop meaning anything after the first branch."""
    rng = np.random.default_rng(1)
    cloud = rng.uniform(-100, 100, (150, 3))
    grown = pt.MST_tree(cloud[:, 0], cloud[:, 1], cloud[:, 2], start=tree,
                        bf=0.9, thr=60.0)
    assert pt.Pvec_tree(grown).max() > pt.Pvec_tree(tree).max()


def test_every_seed_node_is_a_valid_attachment_point(tree):
    """Not only its root -- otherwise the growth would fan out from one
    point instead of branching off the existing arbor."""
    rng = np.random.default_rng(2)
    cloud = rng.uniform(-100, 100, (200, 3))
    grown = pt.MST_tree(cloud[:, 0], cloud[:, 1], cloud[:, 2], start=tree,
                        bf=0.4, thr=40.0)
    parents = idpar_tree(grown)[tree.n_nodes:]
    attached_to_seed = parents[parents < tree.n_nodes]
    assert len(np.unique(attached_to_seed)) > 1


def test_several_seed_trees_compete_for_one_cloud(tree):
    left = pt.tran_tree(tree, [-300.0, 0.0, 0.0])
    right = pt.tran_tree(tree, [300.0, 0.0, 0.0])
    rng = np.random.default_rng(3)
    cloud = rng.uniform(-400, 400, (300, 3))
    grown = pt.MST_tree(cloud[:, 0], cloud[:, 1], cloud[:, 2],
                        start=[left, right], bf=0.4, thr=80.0)
    assert len(grown) == 2
    assert all(g.n_nodes > tree.n_nodes for g in grown)


def test_index_starts_still_behave_as_before():
    rng = np.random.default_rng(0)
    cloud = rng.uniform(-50, 50, (60, 3))
    plain = pt.MST_tree(cloud[:, 0], cloud[:, 1], cloud[:, 2], start=0,
                        bf=0.4, thr=40.0)
    assert plain.rnames == ["tree"]
    assert (plain.D == 1.0).all()


# ---------------------------------------------------------------------------
# clone_tree
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clones(group):
    return pt.clone_tree(group, n=2, bf=0.4, rng=0)


def test_clones_are_valid_trees(clones):
    for clone in clones:
        assert clone.dA.nnz == clone.n_nodes - 1  # connected, acyclic
        assert clone.n_nodes > 100
        assert (clone.D > 0).all()


def test_clones_land_inside_the_groups_measured_spread(clones, group):
    """The claim the whole pipeline makes. Not "identical to a real cell" --
    "drawn from the same distribution"."""
    real = np.array([
        int((pt.B_tree(t) | pt.T_tree(t)).sum()) for t in group
    ])
    for clone in clones:
        count = int((pt.B_tree(clone) | pt.T_tree(clone)).sum())
        assert real.min() * 0.5 < count < real.max() * 1.5


def test_clone_extents_track_the_group(clones, group):
    real = np.array([
        np.ptp(np.column_stack([t.X, t.Y, t.Z]), axis=0) for t in group
    ])
    for clone in clones:
        span = np.ptp(np.column_stack([clone.X, clone.Y, clone.Z]), axis=0)
        assert (span[:2] > real[:, :2].min(axis=0) * 0.5).all()
        assert (span[:2] < real[:, :2].max(axis=0) * 2.0).all()


def test_ignored_regions_are_not_grown(clones):
    """`spines` and `axon` are skipped by name, as in MATLAB."""
    for clone in clones:
        assert "axon" not in clone.rnames
        assert "spines" not in clone.rnames


def test_clones_differ_from_each_other(clones):
    assert clones[0].n_nodes != clones[1].n_nodes


def test_cloning_is_reproducible(group):
    a = pt.clone_tree(group[:3], n=1, rng=5)[0]
    b = pt.clone_tree(group[:3], n=1, rng=5)[0]
    assert a.n_nodes == b.n_nodes
    np.testing.assert_allclose(a.X, b.X)


def test_a_single_tree_can_be_cloned(tree):
    clone = pt.clone_tree(tree, n=1, rng=1)[0]
    assert clone.n_nodes > 10
    assert clone.dA.nnz == clone.n_nodes - 1


def test_the_requested_number_of_clones_comes_back(group):
    assert len(pt.clone_tree(group[:2], n=3, rng=0)) == 3


# ---------------------------------------------------------------------------
# dscam_tree
# ---------------------------------------------------------------------------


def test_dscam_pulls_branches_together(tree):
    """The whole point: without DSCAM, sibling branches stop avoiding each
    other, so nodes end up closer to their non-relatives."""
    before = np.column_stack([tree.X, tree.Y, tree.Z])
    after_tree = pt.dscam_tree(tree, 300, rng=0)
    after = np.column_stack([after_tree.X, after_tree.Y, after_tree.Z])
    assert (cKDTree(after).query(after, k=2)[0][:, 1].mean()
            < cKDTree(before).query(before, k=2)[0][:, 1].mean())


def test_dscam_changes_only_coordinates(tree):
    moved = pt.dscam_tree(tree, 100, rng=0)
    assert (moved.dA != tree.dA).nnz == 0
    np.testing.assert_array_equal(moved.D, tree.D)
    np.testing.assert_array_equal(moved.R, tree.R)


def test_dscam_moves_whole_subtrees_rigidly(tree):
    """A branch is dragged with its node, so its own internal geometry --
    and therefore the tree's total cable -- barely changes."""
    moved = pt.dscam_tree(tree, 200, rng=0)
    assert pt.len_tree(moved).sum() == pytest.approx(
        pt.len_tree(tree).sum(), rel=0.05
    )


def test_dscam_does_nothing_with_no_iterations(tree):
    still = pt.dscam_tree(tree, 0, rng=0)
    np.testing.assert_array_equal(still.X, tree.X)


def test_dscam_is_reproducible(tree):
    np.testing.assert_array_equal(
        pt.dscam_tree(tree, 50, rng=7).X, pt.dscam_tree(tree, 50, rng=7).X
    )


# ---------------------------------------------------------------------------
# spines_tree
# ---------------------------------------------------------------------------


def test_each_spine_is_a_neck_and_a_head(tree):
    spined = pt.spines_tree(tree, 50, rng=0)
    assert spined.n_nodes == tree.n_nodes + 100


def test_heads_hang_off_necks_which_hang_off_the_dendrite(tree):
    result = pt.spines_tree(tree, 40, rng=0, full_output=True)
    idpar = idpar_tree(result.tree)
    np.testing.assert_array_equal(idpar[result.heads], result.necks)
    assert (idpar[result.necks] < tree.n_nodes).all()


def test_all_spine_indices_come_back_not_just_the_last(tree):
    """MATLAB overwrites `indhead`/`indneck` each pass of its loop, so
    despite the plural in its docstring it returns two numbers."""
    result = pt.spines_tree(tree, 25, rng=0, full_output=True)
    assert len(result.heads) == 25
    assert len(result.necks) == 25
    assert len(np.unique(np.concatenate([result.heads, result.necks]))) == 50


def test_necks_stand_perpendicular_to_the_cable(tree):
    """A spine growing along the dendrite would be geometrically
    meaningless, and would double-count the cable's surface."""
    result = pt.spines_tree(tree, 100, rng=1, full_output=True)
    coords = np.column_stack([result.tree.X, result.tree.Y, result.tree.Z])
    parents = idpar_tree(result.tree)[result.necks]
    offset = coords[result.necks] - coords[parents]
    offset /= np.linalg.norm(offset, axis=1, keepdims=True)
    along = pt.direction_tree(tree)[parents]
    assert np.abs(np.einsum("ij,ij->i", offset, along)).max() < 1e-8


def test_the_head_continues_along_the_neck(tree):
    """Head and neck are collinear, so the head is a cylinder standing off
    the cable rather than a kink."""
    result = pt.spines_tree(tree, 40, head_diameter=2.0, rng=1,
                            full_output=True)
    coords = np.column_stack([result.tree.X, result.tree.Y, result.tree.Z])
    parents = idpar_tree(result.tree)[result.necks]
    neck = coords[result.necks] - coords[parents]
    head = coords[result.heads] - coords[result.necks]
    neck /= np.linalg.norm(neck, axis=1, keepdims=True)
    head /= np.linalg.norm(head, axis=1, keepdims=True)
    np.testing.assert_allclose(np.einsum("ij,ij->i", neck, head), 1.0, atol=1e-9)
    assert np.linalg.norm(
        coords[result.heads] - coords[result.necks], axis=1
    ) == pytest.approx(2.0)


def test_head_and_neck_diameters_are_set(tree):
    spined = pt.spines_tree(tree, 30, neck_diameter=0.3, head_diameter=1.5,
                            rng=0, full_output=True)
    assert (spined.tree.D[spined.necks] == 0.3).all()
    assert (spined.tree.D[spined.heads] == 1.5).all()


def test_spines_go_into_their_own_region(tree):
    spined = pt.spines_tree(tree, 10, rng=0)
    assert "spines" in spined.rnames
    assert int((np.asarray(spined.R) == spined.rnames.index("spines")).sum()) == 20


def test_necks_and_heads_can_be_separated(tree):
    spined = pt.spines_tree(tree, 10, rng=0, separate_regions=True)
    assert "spine_neck" in spined.rnames and "spine_head" in spined.rnames


def test_explicit_node_indices_are_honoured(tree):
    """MATLAB reads this branch only when every value happens to be below
    the node count, so it silently misreads coordinates as indices."""
    result = pt.spines_tree(tree, [5, 10, 15], rng=0, full_output=True)
    np.testing.assert_array_equal(
        idpar_tree(result.tree)[result.necks], [5, 10, 15]
    )


def test_explicit_coordinates_are_honoured(tree):
    """MATLAB's documented `XYZ` matrix input cannot be reached at all: it
    either falls into the node-index branch or leaves `indy` undefined."""
    where = np.array([[50.0, 50.0, 0.0], [80.0, 20.0, 10.0]])
    result = pt.spines_tree(tree, where, rng=0, full_output=True)
    coords = np.column_stack([result.tree.X, result.tree.Y, result.tree.Z])
    np.testing.assert_allclose(coords[result.necks], where)


def test_placement_can_be_restricted_to_part_of_the_tree(tree):
    subset = np.arange(20)
    result = pt.spines_tree(tree, 30, nodes=subset, rng=0, full_output=True)
    assert set(idpar_tree(result.tree)[result.necks].tolist()) <= set(subset.tolist())


def test_a_bad_spine_specification_says_what_is_allowed(tree):
    with pytest.raises(ValueError, match="count, a 1D array"):
        pt.spines_tree(tree, np.zeros((3, 3, 3)), rng=0)


# ---------------------------------------------------------------------------
# PP_generator_tree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [0.7, 1.0, 1.4])
def test_the_cloud_hits_the_requested_regularity(target):
    """Measured with an independent seed and a longer Monte-Carlo run than
    the generator itself used, so this is not just reading back the value
    the loop stopped on."""
    points = pt.PP_generator_tree(120, target, n_mc=10, rng=0, max_iter=60)
    assert pt.r_mc_tree(points, n_mc=40, dim=2, rng=1).R == pytest.approx(
        target, abs=0.1
    )


def test_clustered_and_regular_clouds_really_differ():
    """Independently of `r_mc_tree`: a clustered cloud has nearest-neighbour
    distances that are both shorter *and* more variable relative to their
    mean, because it is mostly tight groups with gaps between them."""
    clustered = pt.PP_generator_tree(120, 0.7, n_mc=10, rng=0, max_iter=60)
    regular = pt.PP_generator_tree(120, 1.4, n_mc=10, rng=0, max_iter=60)
    nn = lambda p: cKDTree(p).query(p, k=2)[0][:, 1]  # noqa: E731
    assert nn(clustered).mean() < nn(regular).mean()
    assert nn(clustered).std() / nn(clustered).mean() >         nn(regular).std() / nn(regular).mean()


def test_three_dimensional_clouds():
    points = pt.PP_generator_tree(80, 1.3, dim=3, n_mc=10, rng=0, max_iter=40)
    assert points.shape == (80, 3)
    assert pt.r_mc_tree(points, n_mc=40, dim=3, rng=1).R > 1.0


def test_the_exclusion_zone_is_respected():
    """Standing in for the physical size of whatever the points represent."""
    points = pt.PP_generator_tree(60, 1.2, epsilon=8.0, n_mc=10, rng=0,
                                  max_iter=40)
    assert cKDTree(points).query(points, k=2)[0][:, 1].min() >= 8.0 - 1e-9


def test_points_stay_inside_the_box():
    """R = 1.5 is out of reach for 100 points in a 100 um box, which is the
    point: the clamp has to hold even while the search is still pushing."""
    with pytest.warns(UserWarning, match="gave up"):
        points = pt.PP_generator_tree(100, 1.5, box=50.0, n_mc=10, rng=0,
                                      max_iter=30)
    assert np.abs(points).max() <= 50.0 + 1e-9


def test_an_existing_cloud_can_be_rearranged():
    rng = np.random.default_rng(0)
    start = rng.uniform(-100, 100, (100, 2))
    points = pt.PP_generator_tree(start, 1.3, n_mc=10, rng=0, max_iter=40)
    assert points.shape == start.shape
    assert not np.allclose(points, start)


def test_an_unreachable_target_gives_up_and_says_so():
    """MATLAB's loop has no bound and spins forever here."""
    with pytest.warns(UserWarning, match="gave up"):
        pt.PP_generator_tree(40, 5.0, n_mc=5, rng=0, max_iter=3)


def test_a_zero_step_is_rejected():
    with pytest.raises(ValueError, match="nonzero"):
        pt.PP_generator_tree(50, 1.2, a=0.0)


def test_full_output_reports_the_search():
    points, iterations, history = pt.PP_generator_tree(
        100, 1.2, n_mc=10, rng=0, max_iter=40, full_output=True
    )
    assert len(history) == iterations + 1
    assert abs(history[-1] - 1.2) < abs(history[0] - 1.2)
