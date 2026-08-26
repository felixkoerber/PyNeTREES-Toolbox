"""V3: a list of trees in, a list of results out.

The rule replaces W5's concatenation (`gene_tree` on a group used to return
one stacked array). List return is *reversible* -- `np.vstack(results)`
recovers the pooled form -- while a concatenated array cannot be split
again unless the caller kept every tree's node count separately, which is
the bookkeeping the concatenation was meant to save them.

Most of what follows is a **sweep** rather than one case per function: the
claim is a rule, and a rule guarded one instance at a time is a rule that
the eighty-ninth function will break.
"""

from __future__ import annotations

import inspect

import matplotlib
import numpy as np
import pytest
from scipy import sparse

matplotlib.use("Agg")

import pynetrees as pt
from pynetrees._population import (accepts_population, is_nested_population,
                                 is_population)


@pytest.fixture(scope="module")
def trees():
    return pt.dLPTCs_trees()["dhsn"][:3]


@pytest.fixture
def empty():
    return pt.Tree(dA=sparse.csr_matrix((0, 0)), X=np.zeros(0), Y=np.zeros(0),
                   Z=np.zeros(0), D=np.zeros(0), R=np.zeros(0, dtype=int),
                   rnames=[], name="empty")


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------


def test_a_list_of_trees_gives_a_list_of_results(trees):
    results = pt.gene_tree(trees)
    assert isinstance(results, list)
    assert len(results) == len(trees)
    np.testing.assert_array_equal(results[0], pt.gene_tree(trees[0]))


def test_the_list_form_can_still_be_pooled(trees):
    """The reason list return is the better rule: pooling is one call
    away, and the reverse is not."""
    pooled = np.vstack(pt.gene_tree(trees))
    assert pooled.shape[0] == sum(len(pt.gene_tree(t)) for t in trees)


def test_a_single_tree_is_untouched(trees):
    """The generalisation must not change the one-tree case at all."""
    assert pt.gene_tree(trees[0]).shape[1] == 2
    assert pt.dist_tree(trees[0], [50]).shape == (trees[0].n_nodes, 1)
    assert isinstance(pt.resample_tree(trees[0]), pt.Tree)


def test_editing_a_group_gives_a_group(trees):
    resampled = pt.resample_tree(trees, 10.0)
    assert isinstance(resampled, list)
    assert all(isinstance(t, pt.Tree) for t in resampled)
    assert [t.name for t in resampled] == [t.name for t in trees]


def test_order_is_preserved(trees):
    """Result i belongs to tree i -- the whole point of keeping the shape."""
    lengths = [float(np.sum(each)) for each in pt.len_tree(trees)]
    assert lengths == [float(np.sum(pt.len_tree(t))) for t in trees]


def test_a_tuple_of_trees_works_too(trees):
    assert len(pt.len_tree(tuple(trees))) == len(trees)


def test_an_empty_group_says_what_is_wrong():
    """An empty list would otherwise fall through to a per-tree
    computation and fail somewhere unhelpful."""
    for call in (lambda: pt.gene_tree([]), lambda: pt.dist_tree([], [50]),
                 lambda: pt.bin_tree([]), lambda: pt.sholl_tree([]),
                 lambda: pt.len_tree([]), lambda: pt.resample_tree([])):
        with pytest.raises(ValueError, match="empty list of trees"):
            call()


# ---------------------------------------------------------------------------
# nesting: groups of groups
# ---------------------------------------------------------------------------


def test_a_list_of_groups_gives_a_list_of_groups(trees):
    """`dLPTCs_trees()` is groups of cells, and a `.mtr` can hold a 2-deep
    cell array, so the structure has to survive."""
    groups = [list(trees[:2]), list(trees[2:])]
    results = pt.len_tree(groups)
    assert [len(g) for g in results] == [2, 1]
    np.testing.assert_array_equal(results[0][0], pt.len_tree(trees[0]))


def test_nesting_stops_at_one_level(trees):
    """Three deep is not a shape anything in the toolbox produces, and
    recursing into it would silently accept a mistake."""
    with pytest.raises((AttributeError, TypeError, ValueError)):
        pt.len_tree([[[trees[0]]]])


def test_the_two_population_shapes_are_told_apart(trees):
    assert is_population(trees)
    assert is_population([])  # vacuously, so the error above can be raised
    assert not is_population(trees[0])
    assert not is_population([1, 2, 3])
    assert not is_population(np.zeros((3, 3)))
    assert is_nested_population([list(trees), list(trees)])
    assert not is_nested_population(trees)
    assert not is_nested_population([])


# ---------------------------------------------------------------------------
# empty trees keep their slot
# ---------------------------------------------------------------------------


def test_an_empty_tree_in_a_group_keeps_its_slot(trees, empty):
    """Dropping it would renumber every cell after it, silently."""
    group = [trees[0], empty, trees[1]]
    results = pt.len_tree(group)
    assert len(results) == 3
    assert results[1].size == 0
    np.testing.assert_array_equal(results[2], pt.len_tree(trees[1]))


def test_an_empty_tree_in_a_group_is_edited_not_dropped(trees, empty):
    edited = pt.resample_tree([trees[0], empty], 10.0)
    assert len(edited) == 2 and edited[1].n_nodes == 0


def test_the_decorator_order_is_what_makes_that_work(empty):
    """`@accepts_population` sits *outside* `@empty_safe`. The other order
    maps to the undecorated function, and an empty cell raises."""
    assert pt.len_tree([empty])[0].size == 0


# ---------------------------------------------------------------------------
# per-tree arguments
# ---------------------------------------------------------------------------


def test_one_argument_is_shared_by_the_whole_group(trees):
    """"Resample all of these to 1 um" is the common case."""
    assert all(t.n_nodes > 0 for t in pt.resample_tree(trees, 20.0))


def test_a_sequence_per_tree_is_zipped(trees):
    deleted = pt.delete_tree(trees, [[1, 2], [1], [1, 2, 3]])
    assert [t.n_nodes for t in deleted] == [trees[0].n_nodes - 2,
                                            trees[1].n_nodes - 1,
                                            trees[2].n_nodes - 3]


def test_an_array_is_shared_rather_than_zipped(trees):
    """An `np.ndarray` never zips, which is the escape hatch for the
    ambiguous case below."""
    deleted = pt.delete_tree(trees, np.array([1, 2, 3]))
    assert [t.n_nodes for t in deleted] == [t.n_nodes - 3 for t in trees]


def test_a_flat_list_the_length_of_the_group_refuses_to_guess(trees):
    """`delete_tree(trees, [3, 7])` with two trees could mean either, and
    guessing wrong deletes the wrong nodes with nothing downstream to
    notice. So it raises, and says what to write instead."""
    with pytest.raises(ValueError, match="ambiguous"):
        pt.delete_tree(trees, [1, 2, 3])


def test_the_error_names_both_ways_out(trees):
    with pytest.raises(ValueError) as excinfo:
        pt.delete_tree(trees, [1, 2, 3])
    assert "np.asarray" in str(excinfo.value)
    assert "one sequence per tree" in str(excinfo.value)


def test_a_flat_list_of_another_length_is_shared(trees):
    """Only the length coincidence is ambiguous."""
    deleted = pt.delete_tree(trees, [1, 2])
    assert [t.n_nodes for t in deleted] == [t.n_nodes - 2 for t in trees]


def test_paired_names_must_exist_in_the_signature():
    """A typo in the decorator would otherwise silently never zip."""
    with pytest.raises(ValueError, match="not in its signature"):
        @accepts_population(paired="noduhs")
        def f(tree, nodes=None):
            return nodes


def test_per_node_values_zip_with_the_trees(trees):
    """`morph_tree(trees, values)` -- one value vector per tree, which can
    only be a per-tree argument since the node counts differ."""
    values = [np.full(t.n_nodes, 2.0) for t in trees]
    morphed = pt.morph_tree(trees, values)
    assert len(morphed) == len(trees)
    for one in morphed:
        # every segment rescaled to 2 um, the root excepted (no segment)
        assert np.allclose(pt.len_tree(one)[1:], 2.0)


# ---------------------------------------------------------------------------
# what is deliberately not mapped
# ---------------------------------------------------------------------------


def test_bin_tree_uses_one_edge_set_for_the_whole_group(trees):
    indices, edges = pt.bin_tree(trees, bins=8)
    assert len(indices) == len(trees)
    assert [len(a) for a in indices] == [t.n_nodes for t in trees]
    assert len(edges) == 9


def test_group_bin_edges_span_every_tree(trees):
    """Binning each cell separately would give each its own edges, so bin 3
    would mean a different distance in every cell."""
    _, edges = pt.bin_tree(trees, bins=8)
    assert edges[-1] >= max(pt.eucl_tree(t).max() for t in trees)
    assert edges[0] <= min(pt.eucl_tree(t).min() for t in trees)


def test_no_node_falls_outside_the_group_bins(trees):
    indices, _ = pt.bin_tree(trees, bins=8)
    assert all((a > 0).all() for a in indices)


def test_explicit_edges_are_used_as_given(trees):
    edges = np.linspace(0.0, 400.0, 5)
    _, back = pt.bin_tree(trees, bins=edges)
    np.testing.assert_allclose(back, edges)


def test_sholl_profiles_share_their_radii(trees):
    results = pt.sholl_tree(trees, 50.0, warn_double=False)
    assert len(results) == len(trees)
    assert all(np.array_equal(r.dd, results[0].dd) for r in results)


def test_sholl_profiles_stack_into_a_matrix(trees):
    """The reason for sharing the radii: the caller can average or sum the
    group's profiles column by column."""
    results = pt.sholl_tree(trees, 50.0, warn_double=False)
    profiles = np.array([r.s for r in results])
    assert profiles.shape == (len(trees), len(results[0].dd))


def test_group_radii_reach_the_furthest_cell(trees):
    results = pt.sholl_tree(trees, 50.0, warn_double=False)
    assert results[0].dd[-1] >= 2 * max(pt.eucl_tree(t).max() for t in trees)


def test_explicit_radii_are_honoured_for_a_group(trees):
    radii = np.arange(0.0, 400.0, 50.0)
    results = pt.sholl_tree(trees, radii, warn_double=False)
    assert all(np.array_equal(r.dd, radii) for r in results)


def test_sholl_is_not_pooled_for_you(trees):
    """Summing, averaging and per-cell normalisation all give different
    answers, so the reduction stays with the caller."""
    results = pt.sholl_tree(trees, 50.0, warn_double=False)
    assert isinstance(results, list)
    assert not isinstance(results, np.ndarray)


def test_vonmises_and_bf_pool_root_angles(trees):
    """These two pool by design -- the fit needs one distribution, not one
    per cell -- so they are the documented exception to the rule."""
    k_group, _ = pt.vonMises_tree(trees)
    k_one, _ = pt.vonMises_tree(trees[0])
    assert isinstance(k_group, float) and k_group != k_one


def test_stats_tree_still_takes_a_group(trees):
    stats = pt.stats_tree(list(trees))
    assert len(stats["summary"]) == len(trees)


def test_two_tree_functions_are_left_alone(trees):
    """"A list of trees" is ambiguous for these -- all pairs? zipped? -- so
    the caller writes the loop they mean."""
    assert not hasattr(pt.cat_tree, "__population_paired__")


def test_savers_taking_a_path_are_left_alone(tmp_path, trees):
    """Mapping them would write every tree to the same file. `save_tree`
    and `save_mtr` take a group already, because their formats hold one."""
    assert not hasattr(pt.save_swc, "__population_paired__")
    written = pt.save_mtr(list(trees), tmp_path / "group.mtr")
    assert written.exists()


# ---------------------------------------------------------------------------
# plot_tree draws a group into one scene
# ---------------------------------------------------------------------------


def test_plot_tree_draws_a_whole_group_into_one_plotter(trees):
    pv = pytest.importorskip("pyvista")
    plotter = pt.plot_tree(list(trees), show=False)
    assert isinstance(plotter, pv.Plotter)
    assert len(plotter.renderer.actors) == len(trees)


def test_a_group_cycles_colours_so_the_cells_differ(trees):
    pytest.importorskip("pyvista")
    from pynetrees.plotting import POPULATION_COLORS, _per_tree_colors
    assert _per_tree_colors(None, 3) == list(POPULATION_COLORS[:3])


def test_more_trees_than_colours_wraps_around():
    from pynetrees.plotting import POPULATION_COLORS, _per_tree_colors
    colors = _per_tree_colors(None, len(POPULATION_COLORS) + 2)
    assert colors[len(POPULATION_COLORS)] == POPULATION_COLORS[0]


def test_one_colour_applies_to_every_tree():
    from pynetrees.plotting import _per_tree_colors
    assert _per_tree_colors("red", 3) == ["red"] * 3


def test_an_rgb_triple_is_not_mistaken_for_three_colours():
    """`(1.0, 0.0, 0.0)` with three trees is red, not red/green/blue."""
    from pynetrees.plotting import _per_tree_colors
    assert _per_tree_colors((1.0, 0.0, 0.0), 3) == [(1.0, 0.0, 0.0)] * 3
    assert _per_tree_colors(("red", "green", "blue"), 3) == ["red", "green",
                                                             "blue"]


def test_offsets_can_be_given_one_per_tree(trees):
    from pynetrees.plotting import _per_tree_offsets
    laid_out = pt.spread_tree(list(trees))
    offsets = _per_tree_offsets(laid_out.offsets, len(trees))
    assert len(offsets) == len(trees)
    assert offsets[0] != offsets[1]


def test_a_single_offset_moves_the_whole_group():
    from pynetrees.plotting import _per_tree_offsets
    assert _per_tree_offsets((1.0, 2.0, 3.0), 3) == [(1.0, 2.0, 3.0)] * 3


def test_the_gallery_is_one_call(trees):
    """The case the offset rule exists for."""
    pytest.importorskip("pyvista")
    laid_out = pt.spread_tree(list(trees))
    plotter = pt.plot_tree(laid_out.trees, offset=laid_out.offsets, show=False)
    assert len(plotter.renderer.actors) == len(trees)


def test_an_empty_tree_in_a_group_does_not_take_the_figure_down(trees, empty):
    pytest.importorskip("pyvista")
    plotter = pt.plot_tree([trees[0], empty, trees[1]], show=False)
    assert len(plotter.renderer.actors) == 2  # the empty one drew nothing


def test_plot_tree_says_so_when_the_group_is_empty():
    pytest.importorskip("pyvista")
    with pytest.raises(ValueError, match="empty list of trees"):
        pt.plot_tree([])


# ---------------------------------------------------------------------------
# the sweep: the rule, guarded as a rule
# ---------------------------------------------------------------------------

#: Take the group and reduce or re-bin it, rather than mapping over it --
#: each documented in `pynetrees._population`.
NOT_MAPPED = {"bin_tree", "sholl_tree", "stats_tree", "vonMises_tree",
              "bf_tree", "gscale_tree", "clone_tree", "spread_tree",
              "save_tree", "save_mtr", "plot_tree", "dLPTCs_trees",
              "hsn_tree", "sample_tree", "sample2_tree"}


#: Maps over a group like everything else, but has nothing to hand back for
#: an empty cell: NEURON has no model with no sections, and returning a
#: placeholder would be a simulation object that cannot be simulated.
NEEDS_A_REAL_TREE = {"build_neuron_model"}

#: 3D hull needs scikit-image for its marching-cubes isosurface (see
#: density.py); optional extra, so skip these rather than fail when absent.
try:
    import skimage  # noqa: F401
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
NEEDS_SKIMAGE = {"hull_tree", "vhull_tree"}


def _mapped_entry_points():
    """Everything decorated with `@accepts_population`."""
    for name in sorted(pt.__all__):
        function = getattr(pt, name, None)
        if not callable(function) or isinstance(function, type):
            continue
        if hasattr(function, "__population_paired__"):
            yield name, function


def test_every_mapped_function_keeps_the_shape_it_was_given(empty):
    """One rule, checked on every function that claims to follow it.

    Called with a two-tree group that contains an empty tree, since that is
    the combination where both halves of the rule have to hold at once. The
    real tree is `sample_tree` and carries `Ri`/`Gm`/`Cm`, so the
    electrotonic functions fail (or not) on the *shape* rather than on
    missing constants -- and so that the sweep stays quick enough that
    nobody is tempted to delete it.
    """
    real = pt.sample_tree()
    real.Ri, real.Gm, real.Cm = 100.0, 1 / 2500.0, 1.0
    group = [real, empty]
    offenders = []
    for name, function in _mapped_entry_points():
        if name in NEEDS_A_REAL_TREE:
            continue
        if name in NEEDS_SKIMAGE and not HAS_SKIMAGE:
            continue
        parameters = list(inspect.signature(function).parameters.values())
        if any(p.default is inspect.Parameter.empty
               and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
               for p in parameters[1:]):
            continue  # needs an argument we cannot invent; covered by name
        try:
            result = function(group)
        except Exception as exc:  # noqa: BLE001 - that is what is under test
            offenders.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if not isinstance(result, list) or len(result) != 2:
            offenders.append(f"{name}: returned {type(result).__name__}")
    assert not offenders, "did not keep its shape:\n  " + "\n  ".join(offenders)


def test_the_sweep_actually_covers_something():
    """A guard that passes because it tested nothing is worse than none."""
    assert len(list(_mapped_entry_points())) > 60


def test_nothing_that_reduces_a_group_got_decorated_by_accident():
    """`sholl_tree` mapped instead of binned across the group would give
    every cell its own radii, and the profiles would silently stop being
    comparable."""
    for name in NOT_MAPPED:
        function = getattr(pt, name, None)
        if function is not None:
            assert not hasattr(function, "__population_paired__"), name


def test_the_docstring_says_it_accepts_a_group():
    assert "list of trees" in pt.gene_tree.__doc__
    assert "one value per tree" in pt.delete_tree.__doc__
