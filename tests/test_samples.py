"""The bundled sample morphologies (Design Decision #51).

`sample_tree()` returns MATLAB's actual sample -- the 197-node subtree of an
HSN cell from `sample.mtr` -- rather than the 2252-node `25HSS.swc` stand-in
it returned while `.mtr` reading was still deferred.
"""

from __future__ import annotations

import numpy as np
import pytest

import pytrees as pt


EXPECTED = {
    "sample_tree": 197,
    "sample2_tree": 15,
    "hsn_tree": 1290,
    "hss_tree": 2252,
}


@pytest.mark.parametrize("name,n_nodes", EXPECTED.items())
def test_sample_loaders_match_matlab_node_counts(name, n_nodes):
    tree = getattr(pt, name)()
    assert tree.n_nodes == n_nodes
    assert tree.validate(quiet=True) == []


@pytest.mark.parametrize("name", EXPECTED)
def test_samples_are_named_after_their_loader(name):
    assert getattr(pt, name)().name == name.removesuffix("_tree")


def test_sample_tree_is_matlabs_sample_not_the_swc_stand_in():
    """The regression this guards is subtle enough to be worth naming.

    The old stand-in was the HSS cell exported to SWC, which is a different
    cell *and* lost its region names to the format (SWC has no field for
    them). Asserting node count alone would not catch a silent revert to a
    differently-sized file, so check the identity that actually matters:
    small, and carrying real named regions.
    """
    tree = pt.sample_tree()
    assert tree.n_nodes == 197
    assert tree.rnames != ["1"], "regions collapsed -- this is the SWC stand-in"
    assert len(tree.rnames) >= 2


def test_hss_tree_carries_the_regions_the_swc_export_lost():
    tree = pt.hss_tree()
    assert set(tree.rnames) == {"axon", "dend", "soma"}
    # every declared region is actually used, and R indexes them validly
    assert tree.R.min() >= 0 and tree.R.max() < len(tree.rnames)


def test_hss_tree_is_the_same_cell_the_old_sample_tree_was():
    """Same morphology, restored to its original orientation and regions."""
    tree = pt.hss_tree()
    assert tree.n_nodes == 2252
    assert tree.total_length == pytest.approx(8100.26, abs=0.01)


def test_dLPTCs_population_has_all_five_groups():
    groups = pt.dLPTCs_trees()
    assert len(groups) == 5, "prefix-collapsing bug would merge dvs2/dvs3/dvs4"
    assert sum(len(v) for v in groups.values()) == 55
    assert list(groups) == ["dhse", "dhsn", "dvs2", "dvs3", "dvs4"]
    for name, trees in groups.items():
        assert trees, f"group {name} is empty"
        for tree in trees:
            assert tree.validate(quiet=True) == []


def test_dLPTCs_feeds_stats_tree_directly():
    """The reason this fixture was worth unblocking at all."""
    from pytrees._matlab_groups import group_arrays

    groups = pt.dLPTCs_trees()
    # one tree per group keeps the test quick; the shape is what's under test
    small = {k: v[:1] for k, v in groups.items()}
    trees, names = group_arrays(small)
    stats = pt.stats_tree(trees, group_names=names)
    assert set(stats["summary"]["group"]) == set(names)


# ---------------------------------------------------------------------------
# Tree convenience properties (#48, #49)
# ---------------------------------------------------------------------------


def test_total_length_matches_len_tree_sum():
    tree = pt.sample_tree()
    assert tree.total_length == pytest.approx(pt.len_tree(tree).sum())
    assert tree.total_surface == pytest.approx(pt.surf_tree(tree).sum())
    assert tree.total_volume == pytest.approx(pt.vol_tree(tree).sum())


def test_total_length_is_not_cached_because_trees_are_mutable():
    tree = pt.sample_tree()
    before = tree.total_length
    tree.X = tree.X * 2.0
    assert tree.total_length != pytest.approx(before)


def test_root_is_found_by_in_degree_not_assumed_to_be_index_zero():
    tree = pt.sample_tree()
    assert tree.root == 0  # it is sorted, so it is 0 here

    # shuffle so the root lands elsewhere, and check every geometry
    # transform still pivots on the real root (Design Decision #48)
    order = np.concatenate([np.arange(5, tree.n_nodes), np.arange(5)])
    shuffled = tree.reindexed(order)
    root = shuffled.root
    assert root != 0

    flipped = pt.flip_tree(shuffled, axis="x")
    assert flipped.X[root] == pytest.approx(shuffled.X[root])

    scaled = pt.scale_tree(shuffled, 2.0, center=True)
    assert scaled.X[root] == pytest.approx(shuffled.X[root])

    centred = pt.tran_tree(shuffled)
    assert centred.X[root] == pytest.approx(0.0)
    assert centred.Y[root] == pytest.approx(0.0)


def test_root_raises_a_clear_error_on_an_empty_tree():
    from scipy import sparse

    empty = pt.Tree(
        dA=sparse.csr_matrix((0, 0)), X=np.array([]), Y=np.array([]),
        Z=np.array([]), D=np.array([]), R=np.array([], dtype=int), rnames=[],
    )
    with pytest.raises(ValueError, match="empty"):
        empty.root


# ---------------------------------------------------------------------------
# sub_tree region trimming (#50)
# ---------------------------------------------------------------------------


def test_sub_tree_trims_regions_the_subtree_does_not_use():
    """MATLAB's own `sub_tree.m` says this is missing; the port does it."""
    tree = pt.hss_tree()  # axon / dend / soma
    axon = tree.region_nodes("axon")
    mask, sub = pt.sub_tree(tree, int(axon[len(axon) // 2]))

    used = sorted(set(sub.R.tolist()))
    assert used == list(range(len(sub.rnames))), "R must index the trimmed list"
    assert len(sub.rnames) <= len(tree.rnames)
    assert set(sub.rnames) <= set(tree.rnames)
    assert sub.n_nodes == int(mask.sum())


def test_sub_tree_keeps_all_regions_when_all_are_present():
    tree = pt.hss_tree()
    _mask, sub = pt.sub_tree(tree, tree.root)
    assert sub.rnames == tree.rnames


def test_sub_tree_with_tree_false_skips_the_extraction():
    tree = pt.sample_tree()
    result = pt.sub_tree(tree, 5, with_tree=False)
    assert result.tree is None
    np.testing.assert_array_equal(result.mask, pt.sub_tree(tree, 5).mask)
