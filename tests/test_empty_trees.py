"""V2: an empty tree is a value you can use, not just one you can create.

`delete_tree` can produce a tree with no nodes, and a population is allowed
to contain one -- dropping it instead would renumber every cell after it,
silently. Before this, 52 of the exported functions raised on such a tree.

The tests below are mostly **sweeps** rather than one case per function:
the claim is a *rule* (see `pynetrees._empty`), and a rule guarded one
instance at a time is a rule that the fifty-first function will break.
"""

from __future__ import annotations

import inspect

import matplotlib
import numpy as np
import pytest
from scipy import sparse

matplotlib.use("Agg")

import pynetrees as pt


@pytest.fixture
def empty():
    return pt.Tree(dA=sparse.csr_matrix((0, 0)), X=np.zeros(0), Y=np.zeros(0),
                   Z=np.zeros(0), D=np.zeros(0), R=np.zeros(0, dtype=int),
                   rnames=[], name="empty")


@pytest.fixture
def passive(empty):
    """An empty tree that also carries electrotonic constants, so those
    functions fail (or not) on emptiness rather than on missing Ri/Gm."""
    empty.Ri, empty.Gm, empty.Cm = 100.0, 1 / 2500.0, 1.0
    return empty


# ---------------------------------------------------------------------------
# how you get one in the first place
# ---------------------------------------------------------------------------


def test_deleting_every_node_gives_an_empty_tree():
    tree = pt.sample_tree()
    gone = pt.delete_tree(tree, np.arange(tree.n_nodes))
    assert gone.n_nodes == 0


# ---------------------------------------------------------------------------
# per-node quantities: empty in, empty out
# ---------------------------------------------------------------------------

PER_NODE = [
    "len_tree", "surf_tree", "vol_tree", "cvol_tree", "eucl_tree",
    "B_tree", "T_tree", "C_tree", "typeN_tree", "PL_tree", "BO_tree",
    "LO_tree", "Pvec_tree", "child_tree", "idpar_tree", "ratio_tree",
    "rindex_tree", "strahler_tree", "asym_tree", "rootangle_tree",
    "abel_tree", "angleB_tree", "angleBd_tree", "angleBd2_tree",
    "xdend_tree",
]


@pytest.mark.parametrize("name", PER_NODE)
def test_per_node_functions_return_an_empty_array(empty, name):
    """Not a length-1 array of zeros, which would put a phantom node into
    every downstream sum."""
    result = getattr(pt, name)(empty)
    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_direction_tree_keeps_its_second_axis(empty):
    """Shape, not just emptiness: `(0, 3)` still stacks with real trees."""
    assert pt.direction_tree(empty).shape == (0, 3)


def test_cyl_tree_returns_the_right_number_of_empty_arrays(empty):
    assert len(pt.cyl_tree(empty, dim=3)) == 6
    assert all(a.size == 0 for a in pt.cyl_tree(empty, dim=3))


def test_gene_tree_keeps_its_second_axis(empty):
    assert pt.gene_tree(empty).shape == (0, 2)


ELECTROTONIC_VECTORS = ["gi_tree", "gm_tree", "elen_tree", "lambda_tree"]
ELECTROTONIC_MATRICES = ["M_tree", "sse_tree", "syn_tree"]


@pytest.mark.parametrize("name", ELECTROTONIC_VECTORS)
def test_electrotonic_vectors_are_empty(passive, name):
    assert getattr(pt, name)(passive).size == 0


@pytest.mark.parametrize("name", ELECTROTONIC_MATRICES)
def test_electrotonic_matrices_are_empty(passive, name):
    """A node-by-node matrix over no nodes is `(0, 0)`, not `(0,)`."""
    assert np.asarray(getattr(pt, name)(passive)).shape == (0, 0)


# ---------------------------------------------------------------------------
# sums are zero, means are nan
# ---------------------------------------------------------------------------


def test_total_length_of_nothing_is_zero(empty):
    assert empty.total_length == 0.0
    assert empty.total_surface == 0.0
    assert empty.total_volume == 0.0


def test_input_conductance_of_nothing_is_zero_and_still_a_number(passive):
    """`cgin_tree` sums a surface, so it is a sum -- and it must return the
    same *kind* of thing it does for a real tree. It briefly returned a
    ``(0, 0)`` array here, which is a shape no caller of a scalar expects."""
    assert pt.cgin_tree(passive) == 0.0
    assert np.ndim(pt.cgin_tree(passive)) == 0


def test_an_empty_cell_has_no_electrotonic_compartments(passive):
    assert pt.M_atten_tree(passive) == 0


def test_simulating_an_empty_cell_gives_no_traces(passive):
    voltage, spikes = pt.LIF_tree(passive)
    assert voltage.shape == (0, 0) and spikes.size == 0


def test_convexity_of_nothing_is_nan(empty):
    """Not 0 and not 1 -- the fraction of visible pairs among no pairs is
    undefined, and returning either number would be a claim."""
    assert np.isnan(pt.convexity_tree(empty))


def test_a_population_mean_is_not_dragged_toward_zero(empty):
    """The reason the sum/mean split matters. If an empty tree's mean path
    length came back as 0 rather than nan, averaging a group containing one
    would quietly halve the answer."""
    stats = pt.stats_tree([pt.sample_tree(), empty])
    assert len(stats["summary"]) == 2  # the empty tree keeps its row
    assert np.isnan(stats["summary"]["mplen"].iloc[1])
    assert stats["summary"]["len"].iloc[1] == 0.0  # summing nothing IS zero
    assert not np.isnan(stats["summary"]["mplen"].mean())


def test_an_empty_tree_keeps_its_index_in_a_population(empty):
    """Dropping it would renumber every cell after it."""
    stats = pt.stats_tree([pt.sample_tree(), empty, pt.sample2_tree()])
    assert stats["summary"]["tree"].tolist() == [0, 1, 2]


# ---------------------------------------------------------------------------
# editing functions return the empty tree
# ---------------------------------------------------------------------------

EDITING = [
    "elim0_tree", "elimt_tree", "repair_tree", "restrain_tree", "root_tree",
    "resample_tree", "sort_tree", "flatten_tree", "clean_tree", "cap_tree",
    "smooth_tree", "quaddiameter_tree", "soma_tree", "jitter_tree",
    "insertp_tree", "spines_tree", "dscam_tree", "tran_tree", "scale_tree",
    "flip_tree", "rot_tree", "morph_tree",
]


@pytest.mark.parametrize("name", EDITING)
def test_editing_functions_return_an_empty_tree(empty, name):
    result = getattr(pt, name)(empty)
    assert isinstance(result, pt.Tree)
    assert result.n_nodes == 0


# ---------------------------------------------------------------------------
# richer results keep their shape
# ---------------------------------------------------------------------------


def test_sholl_of_nothing_has_no_spheres(empty):
    result = pt.sholl_tree(empty)
    assert result.s.size == 0 and result.dd.size == 0


def test_dissect_tree_returns_no_sections(empty):
    assert pt.dissect_tree(empty).shape == (0, 2)


def test_bin_tree_returns_no_bins_and_no_edges(empty):
    indices, edges = pt.bin_tree(empty)
    assert indices.size == 0 and edges.size == 0


def test_density_and_hulls_are_empty_not_broken(empty):
    assert pt.gdens_tree(empty).counts.size == 0
    assert pt.hull_tree(empty).vertices.size == 0
    assert pt.vhull_tree(empty).volumes.size == 0
    assert pt.boundary_tree(empty).volume == 0.0


def test_r_mc_of_nothing_is_nan_not_one(empty):
    """R = 1 would say "indistinguishable from random", which is a claim
    about a point set that does not exist."""
    result = pt.r_mc_tree(empty)
    assert result.n == 0
    assert np.isnan(result.R)


def test_an_empty_point_cloud_bins_to_an_empty_grid():
    """`gdens_tree` takes a bare array too, and `np.min` of an empty one has
    no identity to fall back on."""
    assert pt.gdens_tree(np.empty((0, 3))).counts.size == 0


# ---------------------------------------------------------------------------
# plotting draws nothing rather than raising
# ---------------------------------------------------------------------------


def test_matplotlib_plotters_return_none_rather_than_raising(empty):
    assert pt.plot_mpl_tree(empty) is None
    assert pt.dendrogram_tree(empty) is None
    assert pt.xplore_tree(empty) is None


def test_plot_tree_composes_a_population_containing_an_empty_tree(empty):
    """The case that matters: one bad cell in a group must not take the
    whole figure down."""
    pv = pytest.importorskip("pyvista")
    plotter = pv.Plotter(off_screen=True)
    pt.plot_tree(pt.sample_tree(), plotter=plotter, show=False)
    returned = pt.plot_tree(empty, plotter=plotter, show=False)
    assert returned is plotter


# ---------------------------------------------------------------------------
# what still raises, and should
# ---------------------------------------------------------------------------


def test_the_root_of_nothing_is_still_an_error(empty):
    """There is no right answer, and inventing one would hide a bug."""
    with pytest.raises(ValueError, match="tree is empty"):
        empty.root


def test_growing_from_no_points_is_still_an_error():
    with pytest.raises(ValueError):
        pt.MST_tree(np.empty(0), np.empty(0), np.empty(0))


# ---------------------------------------------------------------------------
# the sweep: nothing exported may raise on an empty tree
# ---------------------------------------------------------------------------

#: Functions with a documented reason to refuse -- see `pynetrees._empty`.
MAY_RAISE = {"MST_tree", "BCT_tree", "isBCT_tree", "allBCTs_tree",
             "allBTs_tree", "clone_tree", "gscale_tree", "ver_tree",
             "spread_tree"}


def _single_tree_entry_points():
    for name in sorted(pt.__all__):
        if name in MAY_RAISE or name[0].isupper():
            continue
        function = getattr(pt, name)
        if not callable(function):
            continue
        try:
            parameters = list(inspect.signature(function).parameters.values())
        except (TypeError, ValueError):
            continue
        if not parameters or parameters[0].name not in ("tree", "intree"):
            continue
        if any(p.default is inspect.Parameter.empty
               and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
               for p in parameters[1:]):
            continue
        yield name, function


def test_no_single_tree_function_raises_on_an_empty_tree(passive):
    """The rule, guarded as a rule.

    Every exported function whose first argument is a tree and whose other
    arguments are optional gets called with an empty one. Anything that
    raises is either a bug or belongs in `MAY_RAISE` with a reason.
    """
    offenders = []
    for name, function in _single_tree_entry_points():
        try:
            function(passive)
        except Exception as exc:  # noqa: BLE001 - that is what is under test
            offenders.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not offenders, "raised on an empty tree:\n  " + "\n  ".join(offenders)


def test_the_sweep_actually_covers_something():
    """A guard that passes because it tested nothing is worse than no guard."""
    assert len(list(_single_tree_entry_points())) > 40
