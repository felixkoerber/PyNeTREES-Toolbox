"""V4: growing trees into a volume, and toy trees over random points.

`growth_tree` cannot be diffed against MATLAB -- its target cloud comes
from `boundary()`, which Octave does not implement, and the growth is
driven by that cloud. So it is pinned against its own defining properties
instead: what each parameter is supposed to do, and the invariants a grown
morphology must satisfy however it was produced.
"""

from __future__ import annotations

import numpy as np
import pytest

import pynetrees as pt
from pynetrees.construct import _transform_space_filling


@pytest.fixture(scope="module")
def seed():
    return pt.sample_tree()


@pytest.fixture(scope="module")
def grown(seed):
    return pt.growth_tree(seed, thr=60, n_target_points=20_000,
                          rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# random_tree
# ---------------------------------------------------------------------------


def test_random_tree_has_the_nodes_asked_for():
    tree = pt.random_tree(80, rng=np.random.default_rng(0))
    assert tree.n_nodes == 80
    assert pt.ver_tree(tree, quiet=True) == []


def test_random_tree_is_rooted_at_the_origin():
    tree = pt.random_tree(50, rng=np.random.default_rng(0))
    assert (tree.X[tree.root], tree.Y[tree.root], tree.Z[tree.root]) == (0, 0, 0)


def test_the_sphere_option_keeps_points_inside_the_radius():
    tree = pt.random_tree(200, 60.0, rng=np.random.default_rng(0))
    assert np.linalg.norm(
        np.column_stack([tree.X, tree.Y, tree.Z]), axis=1).max() <= 60.0


def test_the_box_option_reaches_the_corners():
    """A cube of half-width r has points further out than r."""
    tree = pt.random_tree(400, 60.0, shape="box", rng=np.random.default_rng(0))
    assert np.linalg.norm(
        np.column_stack([tree.X, tree.Y, tree.Z]), axis=1).max() > 60.0


def test_two_dimensional_clouds_are_flat():
    tree = pt.random_tree(60, dim=2, rng=np.random.default_rng(0))
    assert np.all(tree.Z == 0.0)


def test_anisotropy_stretches_one_axis():
    tree = pt.random_tree(300, 80.0, anisotropy=3.0,
                          rng=np.random.default_rng(0))
    assert np.ptp(tree.Y) > 4 * np.ptp(tree.X)


def test_random_tree_is_reproducible():
    a = pt.random_tree(60, rng=np.random.default_rng(7))
    b = pt.random_tree(60, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a.X, b.X)


def test_a_tree_needs_at_least_one_node():
    with pytest.raises(ValueError, match="at least 1"):
        pt.random_tree(0)


def test_an_unknown_shape_is_rejected():
    with pytest.raises(ValueError, match="'sphere' or 'box'"):
        pt.random_tree(20, shape="blob")


# ---------------------------------------------------------------------------
# growth_tree: the result is a tree
# ---------------------------------------------------------------------------


def test_a_grown_tree_is_valid(grown):
    assert pt.ver_tree(grown.tree, quiet=True) == []
    assert grown.tree.n_nodes > 1


def test_growth_lays_down_about_one_node_per_micron(grown):
    """What makes a grown tree usable without resampling."""
    lengths = pt.len_tree(grown.tree)
    assert np.median(lengths[lengths > 0]) == pytest.approx(1.0, abs=0.2)


def test_the_logs_line_up_with_the_steps(grown):
    n_steps = len(grown.length)
    assert len(grown.terminals) == n_steps
    assert len(grown.attached_to) == n_steps
    assert len(grown.target) == n_steps
    assert grown.attached_to[0] == -1 and grown.target[0] == -1


def test_cable_only_ever_grows(grown):
    assert np.all(np.diff(grown.length) > 0)
    assert np.all(np.diff(grown.terminals) >= 0)


def test_the_tracked_length_is_the_real_length(grown):
    """It is accumulated as cable is laid rather than recomputed, so it has
    to be checked against the tree it claims to describe."""
    assert grown.length[-1] == pytest.approx(
        float(pt.len_tree(grown.tree).sum()))
    assert grown.terminals[-1] == pytest.approx(
        float(pt.T_tree(grown.tree).sum()))


def test_the_tracked_history_matches_step_by_step(seed):
    """The incremental bookkeeping, checked against recomputing at every
    single step rather than only at the end."""
    result = pt.growth_tree(seed, thr=25, n_target_points=10_000,
                            history=True, rng=np.random.default_rng(0))
    np.testing.assert_allclose(
        result.length[1:], [float(pt.len_tree(t).sum()) for t in result.history])
    np.testing.assert_array_equal(
        result.terminals[1:], [float(pt.T_tree(t).sum()) for t in result.history])


def test_every_step_attaches_to_a_node_that_exists(grown):
    assert grown.attached_to[1:].max() < grown.tree.n_nodes
    assert grown.attached_to[1:].min() >= 0


def test_growth_is_reproducible(seed):
    kwargs = dict(thr=30, n_target_points=10_000)
    a = pt.growth_tree(seed, rng=np.random.default_rng(3), **kwargs)
    b = pt.growth_tree(seed, rng=np.random.default_rng(3), **kwargs)
    np.testing.assert_array_equal(a.tree.X, b.tree.X)
    np.testing.assert_array_equal(a.target, b.target)


# ---------------------------------------------------------------------------
# growth_tree: the parameters do what they say
# ---------------------------------------------------------------------------


def test_the_space_filling_term_reaches_for_the_furthest_point(seed):
    """`sp = 1` maps to exactly 1, so the cost term drops out and the first
    step must go to the open point furthest from the root. This is the
    sharpest available check that the space-filling term is wired up --
    and that tracking it incrementally rather than recomputing it, as
    MATLAB does, gives the same answer."""
    result = pt.growth_tree(seed, start=[0.0, 0.0, 0.0], thr=1, sp=1.0, k=0.0,
                            n_target_points=5_000,
                            rng=np.random.default_rng(0))
    distances = np.linalg.norm(result.targets, axis=1)
    assert result.target[1] == int(np.argmax(distances))


def test_no_space_filling_takes_the_cheapest_point_instead(seed):
    """`sp = 0` is an ordinary minimum spanning tree step: nearest wins."""
    result = pt.growth_tree(seed, start=[0.0, 0.0, 0.0], thr=1, sp=0.0, k=0.0,
                            n_target_points=5_000,
                            rng=np.random.default_rng(0))
    distances = np.linalg.norm(result.targets, axis=1)
    assert result.target[1] == int(np.argmin(distances))


def test_more_space_filling_spreads_further(seed):
    """The point of the parameter: the same number of steps covers more
    ground."""
    reach = []
    for sp in (0.0, 1.0):
        result = pt.growth_tree(seed, thr=40, sp=sp, k=0.0,
                                n_target_points=10_000,
                                rng=np.random.default_rng(0))
        reach.append(float(pt.eucl_tree(result.tree).max()))
    assert reach[1] > reach[0]


def test_stochasticity_changes_the_outcome(seed):
    kwargs = dict(thr=30, n_target_points=10_000)
    plain = pt.growth_tree(seed, k=0.0, rng=np.random.default_rng(0), **kwargs)
    noisy = pt.growth_tree(seed, k=0.8, rng=np.random.default_rng(0), **kwargs)
    assert not np.array_equal(plain.target, noisy.target)


def test_a_higher_balancing_factor_makes_cable_run_more_directly(seed):
    """`bf` trades wiring against conduction distance, exactly as in
    `MST_tree`.

    Measured as the **detour** -- path length along the tree over
    straight-line distance to the root -- not as raw path length, which
    also changes because a higher `bf` reaches further per step and so
    grows a bigger cell in the same number of steps.
    """
    detours = []
    for bf in (0.0, 0.25, 0.5, 1.0):
        result = pt.growth_tree(seed, thr=40, bf=bf, k=0.0,
                                n_target_points=10_000,
                                rng=np.random.default_rng(0))
        path = pt.Pvec_tree(result.tree, pt.len_tree(result.tree))
        straight = pt.eucl_tree(result.tree)
        far = straight > 1.0
        detours.append(float(np.mean(path[far] / straight[far])))

    assert detours == sorted(detours, reverse=True)
    assert detours[0] > 1.4          # bf = 0 wanders
    assert detours[-1] == pytest.approx(1.0, abs=1e-6)  # bf = 1 is a star


def test_the_space_filling_transform_is_matlabs(seed):
    """Kept warts and all, so numbers carry over from MATLAB scripts:
    0.5 does not map to 0.5."""
    assert _transform_space_filling(0.5) == pytest.approx(0.475)
    assert _transform_space_filling(1.0) == 1.0
    assert _transform_space_filling(0.0) == 0.0
    assert _transform_space_filling(0.001) == 0.0  # below the switch-off


# ---------------------------------------------------------------------------
# growth_tree: what it grows into, and where it starts
# ---------------------------------------------------------------------------


def test_growth_stays_inside_the_territory(seed):
    """Targets are drawn from the seed's boundary, so the grown cell must
    not sprawl outside the shape it was given."""
    result = pt.growth_tree(seed, thr=60, n_target_points=20_000,
                            rng=np.random.default_rng(0))
    for grown_axis, seed_axis in ((result.tree.X, seed.X),
                                  (result.tree.Y, seed.Y),
                                  (result.tree.Z, seed.Z)):
        margin = 0.05 * (seed_axis.max() - seed_axis.min())
        assert grown_axis.min() >= seed_axis.min() - margin
        assert grown_axis.max() <= seed_axis.max() + margin


def test_explicit_target_points_are_used_as_given():
    """MATLAB's `-P`."""
    points = np.random.default_rng(1).random((2_000, 3)) * 100
    result = pt.growth_tree(points, thr=20, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(result.targets, points)
    assert result.tree.n_nodes > 1


def test_two_dimensional_targets_are_lifted_to_three():
    points = np.random.default_rng(1).random((500, 2)) * 100
    result = pt.growth_tree(points, thr=10, rng=np.random.default_rng(0))
    assert result.targets.shape[1] == 3
    assert np.all(result.tree.Z == 0.0)


def test_growing_from_a_given_root(seed):
    result = pt.growth_tree(seed, start=[10.0, 20.0, 30.0], thr=5,
                            n_target_points=5_000, rng=np.random.default_rng(0))
    root = result.tree.root
    assert (result.tree.X[root], result.tree.Y[root],
            result.tree.Z[root]) == (10.0, 20.0, 30.0)


def test_continuing_an_existing_tree_keeps_it(seed):
    start = pt.sample2_tree()
    result = pt.growth_tree(seed, start=start, thr=20, n_target_points=10_000,
                            rng=np.random.default_rng(0))
    assert result.tree.n_nodes > start.n_nodes
    np.testing.assert_allclose(result.tree.X[: start.n_nodes], start.X)
    assert pt.ver_tree(result.tree, quiet=True) == []


def test_new_cable_gets_its_own_region(seed):
    """So "what did this growth add" survives into the result."""
    start = pt.sample2_tree()
    result = pt.growth_tree(seed, start=start, thr=20, n_target_points=10_000,
                            rng=np.random.default_rng(0))
    assert "new" in result.tree.rnames
    added = result.tree.R[start.n_nodes:]
    assert np.all(added == result.tree.rnames.index("new"))


def test_no_targets_says_so():
    with pytest.raises(ValueError, match="no target points"):
        pt.growth_tree(np.empty((0, 3)))


# ---------------------------------------------------------------------------
# growth_tree: stopping
# ---------------------------------------------------------------------------


def test_stopping_after_a_number_of_steps(seed):
    result = pt.growth_tree(seed, thr=15, n_target_points=10_000,
                            rng=np.random.default_rng(0))
    assert len(result.length) - 1 == 15


def test_stopping_at_a_cable_length(seed):
    result = pt.growth_tree(seed, thr=300.0, stop="length",
                            n_target_points=10_000,
                            rng=np.random.default_rng(0))
    assert result.length[-1] >= 300.0
    assert result.length[-2] < 300.0  # stopped as soon as it got there


def test_stopping_at_a_number_of_tips(seed):
    result = pt.growth_tree(seed, thr=12, stop="terminals",
                            n_target_points=10_000,
                            rng=np.random.default_rng(0))
    assert pt.T_tree(result.tree).sum() >= 12


def test_an_unknown_stopping_rule_is_rejected(seed):
    with pytest.raises(ValueError, match="'steps', 'length' or 'terminals'"):
        pt.growth_tree(seed, stop="forever")


def test_history_is_off_by_default(seed):
    plain = pt.growth_tree(seed, thr=10, n_target_points=5_000,
                           rng=np.random.default_rng(0))
    assert plain.history == []


def test_history_records_every_step(seed):
    result = pt.growth_tree(seed, thr=10, n_target_points=5_000, history=True,
                            rng=np.random.default_rng(0))
    assert len(result.history) == 10
    assert [t.n_nodes for t in result.history] == sorted(
        t.n_nodes for t in result.history)
    assert result.history[-1].n_nodes == result.tree.n_nodes


# ---------------------------------------------------------------------------
# jitter
# ---------------------------------------------------------------------------


def test_jitter_makes_new_cable_wander(seed):
    kwargs = dict(thr=25, n_target_points=10_000)
    straight = pt.growth_tree(seed, rng=np.random.default_rng(0), **kwargs)
    wobbly = pt.growth_tree(seed, jitter=[(0.5, 5)],
                            rng=np.random.default_rng(0), **kwargs)
    assert wobbly.tree.total_length > straight.tree.total_length


def test_jitter_on_a_stretch_shorter_than_its_own_window(seed):
    """Regression: `np.convolve(..., "same")` returns
    `max(len(signal), len(kernel))` rows, so a two-micron stretch smoothed
    over eleven came back the wrong length and the growth crashed."""
    result = pt.growth_tree(seed, thr=20, jitter=[(0.3, 5), (0.1, 12)],
                            n_target_points=10_000,
                            rng=np.random.default_rng(0))
    assert pt.ver_tree(result.tree, quiet=True) == []


def test_jitter_keeps_new_cable_attached(seed):
    """The wander is anchored at zero, so the first new node does not jump
    away from the node it grew out of."""
    result = pt.growth_tree(seed, thr=20, jitter=[(2.0, 5)],
                            n_target_points=10_000,
                            rng=np.random.default_rng(0))
    assert float(pt.len_tree(result.tree).max()) < 20.0
