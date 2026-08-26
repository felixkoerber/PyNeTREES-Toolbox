"""`plot_tree`'s polymorphic `color` and MATLAB-matching argument order.

Design Decision #54. MATLAB overloads one `color` argument three ways; this
port had split it into `color=`/`scalars=`, which meant two parameters where
one was always `None`.
"""

from __future__ import annotations

import numpy as np
import pytest

import pynetrees as pt

pv = pytest.importorskip("pyvista")


@pytest.fixture
def tree():
    return pt.sample_tree()


def _render(tree, *args, **kwargs):
    """Render off-screen and return the plotter, so failures are real errors."""
    plotter = pt.plot_tree(tree, *args, mode="line", **kwargs)
    plotter.close()
    return plotter


def test_colour_name(tree):
    _render(tree, "red")


def test_rgb_triple(tree):
    _render(tree, (1.0, 0.0, 0.0))


def test_per_node_values_are_colour_mapped(tree):
    _render(tree, pt.BO_tree(tree).astype(float))


def test_per_node_rgb_array(tree):
    _render(tree, np.random.default_rng(0).random((tree.n_nodes, 3)))


def test_default_is_a_flat_colour(tree):
    _render(tree)


def test_scalars_keyword_still_works(tree):
    """Kept as the escape hatch, and for code written against the old form."""
    _render(tree, scalars=pt.BO_tree(tree).astype(float))


def test_positional_order_matches_matlab(tree):
    """MATLAB: plot_tree(intree, color, DD, ipart, res)."""
    _render(tree, "blue", (100.0, 0.0, 0.0), np.arange(50), 12)


def test_wrong_length_vector_is_rejected_with_a_useful_message(tree):
    with pytest.raises(ValueError, match=r"length-\d+ vector"):
        pt.plot_tree(tree, np.zeros(7), mode="line")


def test_three_node_tree_reads_a_length_three_vector_as_rgb():
    """The one genuine ambiguity, resolved MATLAB's way.

    On a 3-node tree `[1, 0, 0]` could be an RGB triple or three per-node
    values. It is read as RGB; `scalars=` forces the other reading.
    """
    from scipy import sparse

    dA = sparse.csr_matrix(([1, 1], ([1, 2], [0, 1])), shape=(3, 3))
    tiny = pt.Tree(dA=dA, X=np.arange(3.0), Y=np.zeros(3), Z=np.zeros(3),
                   D=np.ones(3), R=np.zeros(3, dtype=int), rnames=["d"])
    from pynetrees.plotting import _resolve_color

    flat, mapped, rgb = _resolve_color([1.0, 0.0, 0.0], tiny.n_nodes)
    assert flat is not None and mapped is None and rgb is None

    # values above 1.0 cannot be an RGB triple, so they map instead
    flat2, mapped2, _ = _resolve_color([1.0, 5.0, 9.0], tiny.n_nodes)
    assert flat2 is None and mapped2 is not None


def test_categories_flag_discretises_the_colormap(tree):
    _render(tree, tree.R.astype(float), cmap="tab10", categories=True)


def test_offset_moves_the_render_not_the_tree(tree):
    before = tree.X.copy()
    _render(tree, "black", (500.0, 0.0, 0.0))
    np.testing.assert_array_equal(tree.X, before)
