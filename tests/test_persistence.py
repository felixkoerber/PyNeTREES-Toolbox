"""V4: branch length order, barcodes and persistence images.

The numbers in `MATLAB_REFERENCE` come from running MATLAB's own
`BLO_tree`, `barcode_tree` and `realisations_tree` under Octave against the
same bundled trees (`scratchpad/blo_ref.m`). The port agrees with them
exactly, tie-breaking included -- which matters because the port computes
the decomposition a completely different way.
"""

from __future__ import annotations

import numpy as np
import pytest

import pynetrees as pt


@pytest.fixture(scope="module")
def tree():
    return pt.sample_tree()


# ---------------------------------------------------------------------------
# BLO_tree
# ---------------------------------------------------------------------------


def test_the_decomposition_covers_every_node_exactly_once(tree):
    order, _, _ = pt.BLO_tree(tree)
    assert order.min() >= 1  # 1-based: a rank, not an index
    assert set(order.tolist()) == set(range(1, order.max() + 1))


def test_branch_one_starts_at_the_root(tree):
    order, _, _ = pt.BLO_tree(tree)
    assert order[tree.root] == 1


def test_branch_one_is_not_the_longest_path_and_that_is_matlabs_doing():
    """The headline claim of MATLAB's `BLO_tree` -- "returns the primary
    branches by longest first" -- is not what it computes. It selects by
    node count, so on `hsn` branch 1 stops less than halfway to the
    furthest tip. Pinned here so the default cannot drift silently."""
    tree = pt.hsn_tree()
    path_length = pt.Pvec_tree(tree, pt.len_tree(tree))
    by_nodes = pt.BLO_tree(tree).order
    assert path_length[by_nodes == 1].max() == pytest.approx(319.5, abs=0.5)
    assert path_length.max() == pytest.approx(648.4, abs=0.5)


def test_by_length_does_reach_the_furthest_tip():
    """...and the option that does what the name says, does."""
    tree = pt.hsn_tree()
    path_length = pt.Pvec_tree(tree, pt.len_tree(tree))
    by_length = pt.BLO_tree(tree, by="length").order
    assert path_length[by_length == 1].max() == pytest.approx(path_length.max())


def test_each_branch_is_a_connected_path(tree):
    """Every node in a branch except its head has its parent in the same
    branch -- otherwise it is not a path."""
    order, _, _ = pt.BLO_tree(tree)
    idpar = pt.idpar_tree(tree, root_self=False)
    for label in range(1, order.max() + 1):
        nodes = np.flatnonzero(order == label)
        detached = [n for n in nodes
                    if idpar[n] != pt.NO_PARENT and order[idpar[n]] != label]
        assert len(detached) <= 1, f"branch {label} is not one path"


def test_branches_get_shorter(tree):
    """Not strictly monotonic -- selection is by node count, not by summed
    length (see BLO_tree's Notes) -- but the trend must hold."""
    order, length, _ = pt.BLO_tree(tree)
    per_branch = [length[order == b][0] for b in range(1, order.max() + 1)]
    first_half = np.mean(per_branch[: len(per_branch) // 2])
    second_half = np.mean(per_branch[len(per_branch) // 2:])
    assert first_half > second_half


def test_cumulative_reaches_the_branch_length(tree):
    order, length, cumulative = pt.BLO_tree(tree)
    for label in range(1, order.max() + 1):
        inside = order == label
        assert cumulative[inside].max() == pytest.approx(length[inside][0])


@pytest.mark.parametrize("name", ["sample_tree", "hsn_tree", "hss_tree"])
def test_v_does_not_select_anything_under_matlabs_rule(name):
    """MATLAB documents `V` as "values to be integrated to select longest
    path", but selection counts nodes with `V > 0` and never sums `V`. So
    any strictly positive `V` gives the identical ordering -- surprising
    enough to pin."""
    tree = getattr(pt, name)()
    baseline = pt.BLO_tree(tree).order
    for values in (np.ones(tree.n_nodes), pt.eucl_tree(tree) + 1.0):
        np.testing.assert_array_equal(pt.BLO_tree(tree, values).order, baseline)


def test_v_does_select_under_the_length_rule(tree):
    """Which is what makes `by="length"` the meta-function the docstring
    describes."""
    by_len = pt.BLO_tree(tree, by="length").order
    by_one = pt.BLO_tree(tree, np.ones(tree.n_nodes), by="length").order
    assert not np.array_equal(by_len, by_one)


def test_the_two_rules_disagree_about_most_of_the_tree(tree):
    """69% of nodes here, up to 97% on `hsn` -- not a rounding difference."""
    differ = (pt.BLO_tree(tree).order
              != pt.BLO_tree(tree, by="length").order).mean()
    assert differ > 0.5


def test_an_unknown_rule_is_rejected(tree):
    with pytest.raises(ValueError, match="'nodes' or 'length'"):
        pt.BLO_tree(tree, by="longest")


def test_v_must_match_the_tree(tree):
    with pytest.raises(ValueError, match="length n_nodes"):
        pt.BLO_tree(tree, np.ones(5))


# ---------------------------------------------------------------------------
# against MATLAB
# ---------------------------------------------------------------------------

#: (n_nodes, n_branches, realisations) from MATLAB under Octave.
MATLAB_REFERENCE = {
    "sample_tree": (197, 26, 2.81067102683136e+20),
    "sample2_tree": (15, 5, 12.0),
    "hsn_tree": (1290, 224, 6.55401973526678e+270),
}


@pytest.mark.parametrize("name", sorted(MATLAB_REFERENCE))
def test_the_branch_count_matches_matlab(name):
    n_nodes, n_branches, _ = MATLAB_REFERENCE[name]
    one = getattr(pt, name)()
    assert one.n_nodes == n_nodes
    assert pt.BLO_tree(one).order.max() == n_branches
    assert len(pt.barcode_tree(one)) == n_branches


@pytest.mark.parametrize("name", sorted(MATLAB_REFERENCE))
def test_realisations_matches_matlab(name):
    """MATLAB computes this in double precision; the port is exact, so the
    comparison is against MATLAB's float to its own precision."""
    _, _, expected = MATLAB_REFERENCE[name]
    exact = pt.realisations_tree(getattr(pt, name)())
    assert isinstance(exact, int)
    assert float(exact) == pytest.approx(expected, rel=1e-14)


def test_realisations_is_exact_where_matlab_would_lose_digits():
    """A 271-digit integer is not representable as a double. The point of
    returning `int` is that no digits are invented or lost -- round-tripping
    through a float is what MATLAB is stuck with, and it does not come
    back."""
    exact = pt.realisations_tree(pt.hsn_tree())
    assert isinstance(exact, int)
    assert len(str(exact)) == 271
    assert int(float(exact)) != exact


def test_a_zero_length_tip_terminates_and_gets_its_own_bar():
    """MATLAB's `BLO_tree` loops forever here: once the real branch is
    consumed, every remaining row scores zero, `max` returns row 1, row 1 is
    already empty, and nothing changes -- verified by running it (see
    MATLAB_TOOLBOX_BUGS.md). Four nodes are enough, and a repeated point is
    all it takes to build one, so this is not an exotic input."""
    from scipy import sparse

    dA = sparse.csr_matrix(([1, 1, 1], ([1, 2, 3], [0, 1, 2])), shape=(4, 4))
    tree = pt.Tree(dA=dA, X=np.array([0.0, 10.0, 20.0, 20.0]),
                   Y=np.zeros(4), Z=np.zeros(4), D=np.ones(4),
                   R=np.ones(4, dtype=int), rnames=["dendrite"],
                   name="zero-length tip")

    order, length, _ = pt.BLO_tree(tree)
    np.testing.assert_array_equal(order, [1, 1, 1, 2])
    assert length[3] == 0.0
    np.testing.assert_allclose(pt.barcode_tree(tree), [[0, 20], [20, 20]])


# ---------------------------------------------------------------------------
# barcode_tree
# ---------------------------------------------------------------------------


def test_bars_are_born_before_they_die(tree):
    bars = pt.barcode_tree(tree)
    assert (bars[:, 0] <= bars[:, 1]).all()
    assert (bars[:, 0] >= 0).all()


def test_the_first_bar_spans_the_whole_cell(tree):
    bars = pt.barcode_tree(tree)
    assert bars[0, 0] == pytest.approx(0.0)
    assert bars[0, 1] == pytest.approx(bars[:, 1].max())


def test_every_other_bar_is_born_inside_a_living_one(tree):
    """A branch has to branch off something. This is also exactly the
    condition that makes `realisations_tree` non-zero."""
    bars = pt.barcode_tree(tree)
    for index in range(1, len(bars)):
        birth = bars[index, 0]
        alive = (bars[:, 0] <= birth) & (birth <= bars[:, 1])
        alive[index] = False
        assert alive.any(), f"bar {index} is born at {birth} inside nothing"


def test_the_barcode_is_invariant_to_rotation(tree):
    """The property that makes it worth computing: it describes branching,
    not embedding."""
    turned = pt.rot_tree(tree, (30.0, 40.0, 50.0))
    np.testing.assert_allclose(pt.barcode_tree(turned), pt.barcode_tree(tree),
                               atol=1e-9)


def test_topological_mode_counts_nodes(tree):
    """With `v = 1` per node, a death is a node count, so it must be a
    whole number and match the tree's topological depth."""
    bars = pt.barcode_tree(tree, mode="topological")
    np.testing.assert_allclose(bars, np.round(bars))
    assert bars[:, 1].max() == pt.PL_tree(tree).max() + 1


def test_scaling_the_tree_scales_the_barcode(tree):
    bars = pt.barcode_tree(tree)
    bigger = pt.barcode_tree(pt.scale_tree(tree, 3.0))
    np.testing.assert_allclose(bigger, bars * 3.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# persistenceimage_tree
# ---------------------------------------------------------------------------


def test_the_image_is_square_and_sized_by_the_cell(tree):
    image = pt.persistenceimage_tree(tree)
    death = pt.barcode_tree(tree)[:, 1].max()
    assert image.shape == (round(1.25 * death), round(1.25 * death))


def test_two_cells_of_different_size_give_different_sized_images():
    """Which is why `size=` exists: comparing cells pixel by pixel needs
    one grid, and the caller has to choose it."""
    small = pt.persistenceimage_tree(pt.sample2_tree())
    large = pt.persistenceimage_tree(pt.hsn_tree())
    assert small.shape != large.shape
    fixed = [pt.persistenceimage_tree(t, size=400)
             for t in (pt.sample2_tree(), pt.hsn_tree())]
    assert fixed[0].shape == fixed[1].shape == (400, 400)


def test_the_mass_in_the_image_is_the_number_of_bars(tree):
    """Each bar contributes a kernel summing to 1, less whatever falls off
    the edge -- so the total is just under the bar count."""
    image = pt.persistenceimage_tree(tree)
    n_bars = len(pt.barcode_tree(tree))
    assert 0.8 * n_bars < image.sum() <= n_bars


def test_accumulating_counts_coincident_bars_and_matlab_does_not():
    """The documented departure. On `hsn` two bars share a pixel."""
    tree = pt.hsn_tree()
    accumulated = pt.persistenceimage_tree(tree)
    binary = pt.persistenceimage_tree(tree, accumulate=False)
    assert accumulated.sum() > binary.sum()
    bars = pt.barcode_tree(tree)
    pixels = np.unique(np.round(bars), axis=0)
    assert accumulated.sum() - binary.sum() == pytest.approx(
        len(bars) - len(pixels), abs=0.05)


def test_nothing_lands_below_the_diagonal(tree):
    """Indexed `[birth, death]` with birth <= death, every bar sits on or
    above the diagonal; the lower triangle can only hold what the kernel
    smears there."""
    image = pt.persistenceimage_tree(tree, sigma=1.0)
    assert np.tril(image, k=-10).sum() < 1e-6 * image.sum()
    assert np.triu(image).sum() > 0.9 * image.sum()


def test_a_wider_kernel_spreads_the_same_mass(tree):
    tight = pt.persistenceimage_tree(tree, sigma=2.0)
    loose = pt.persistenceimage_tree(tree, sigma=30.0)
    assert tight.max() > loose.max()
    assert tight.sum() > loose.sum()  # more falls off the edge when spread


def test_a_cell_too_small_to_render_says_so():
    tiny = pt.scale_tree(pt.sample2_tree(), 0.001)
    with pytest.raises(ValueError, match="too little to render"):
        pt.persistenceimage_tree(tiny)


# ---------------------------------------------------------------------------
# populations and empty trees, inherited from V3/V2
# ---------------------------------------------------------------------------


def test_a_group_gives_a_list_of_barcodes():
    trees = pt.dLPTCs_trees()["dhsn"][:3]
    bars = pt.barcode_tree(trees)
    assert isinstance(bars, list) and len(bars) == 3
    assert all(b.shape[1] == 2 for b in bars)


def test_an_empty_tree_has_an_empty_barcode(tree):
    empty = pt.delete_tree(tree, np.arange(tree.n_nodes))
    assert pt.barcode_tree(empty).shape == (0, 2)
    assert pt.BLO_tree(empty).order.size == 0
    assert pt.L_tree(empty) == 0.0


# ---------------------------------------------------------------------------
# L_tree
# ---------------------------------------------------------------------------


def test_L_tree_is_the_total_length(tree):
    assert pt.L_tree(tree) == pytest.approx(tree.total_length)
    assert pt.L_tree(tree) == pytest.approx(pt.len_tree(tree).sum())


def test_L_tree_in_two_dimensions_is_shorter(tree):
    """The XY projection cannot be longer than the 3D path."""
    assert pt.L_tree(tree, dim=2) < pt.L_tree(tree)


def test_L_tree_takes_a_group(tree):
    assert pt.L_tree([tree, tree]) == [pt.L_tree(tree)] * 2
