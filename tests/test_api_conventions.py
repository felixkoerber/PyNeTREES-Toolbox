"""The API conventions from Design Decisions #40-#42, #48-#52, pinned.

These tests exist to stop a *future* function from half-adopting a
convention -- which is the failure mode that actually worries me here, more
than any individual call site. Each asserts the rule, not one function's
implementation of it.
"""

from __future__ import annotations

import ast
import pathlib
import warnings

import numpy as np
import pytest
from scipy import sparse

import pytrees as pt
from pytrees.core import Tree

SRC = pathlib.Path(__file__).parents[1] / "src" / "pytrees"


# ---------------------------------------------------------------------------
# #42 -- primary result by default, extras behind full_output
# ---------------------------------------------------------------------------

# functions whose primary result is the Tree, and which carry extra
# bookkeeping output that cannot be recomputed from it
FULL_OUTPUT_FUNCS = ["sort_tree", "redirect_tree", "insertp_tree", "insert_tree"]


def _tiny_tree() -> Tree:
    dA = sparse.csr_matrix(([1, 1, 1, 1], ([1, 2, 3, 4], [0, 0, 1, 1])), shape=(5, 5))
    return Tree(
        dA=dA, X=np.arange(5.0), Y=np.zeros(5), Z=np.zeros(5),
        D=np.ones(5), R=np.zeros(5, dtype=int), rnames=["d"], name="tiny",
    )


def _call(name, tree, **kwargs):
    fn = getattr(pt, name)
    if name == "redirect_tree":
        return fn(tree, 4, **kwargs)
    if name == "insertp_tree":
        return fn(tree, inode=3, plens=[0.5], **kwargs)
    if name == "insert_tree":
        return fn(tree, X=[9.0], Y=[0.0], Z=[0.0], D=[1.0], parent=[0], **kwargs)
    return fn(tree, **kwargs)


@pytest.mark.parametrize("name", FULL_OUTPUT_FUNCS)
def test_bare_call_returns_a_tree(name):
    assert isinstance(_call(name, _tiny_tree()), Tree)


@pytest.mark.parametrize("name", FULL_OUTPUT_FUNCS)
def test_full_output_returns_a_named_tuple_whose_tree_matches(name):
    tree = _tiny_tree()
    bare = _call(name, tree)
    full = _call(name, tree, full_output=True)

    assert isinstance(full, tuple), "extras must come back as a tuple"
    assert hasattr(full, "tree"), "and be a NamedTuple, not a bare tuple"
    assert len(full) == 2
    np.testing.assert_array_equal(full.tree.X, bare.X)
    np.testing.assert_array_equal(full.tree.dA.toarray(), bare.dA.toarray())


def test_stale_tuple_unpacking_fails_loudly_at_the_call_site():
    """A caller written against the old tuple API must not fail silently.

    `Tree` defines `__len__` but neither `__iter__` nor `__getitem__`, so
    unpacking raises immediately and names the real problem, rather than
    silently binding two of something and blowing up somewhere unrelated.
    """
    with pytest.raises(TypeError, match="cannot unpack non-iterable Tree"):
        _tree, _order = pt.sort_tree(_tiny_tree())


def test_no_source_file_unpacks_a_full_output_function_without_asking():
    """Static sweep: no call site tuple-unpacks these without full_output=.

    Catches stale calls in modules the test suite never imports, which is
    where an unmigrated call site would otherwise survive.
    """
    offenders = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Tuple) for t in node.targets):
                continue
            call = node.value
            if not isinstance(call, ast.Call):
                continue
            fname = getattr(call.func, "id", getattr(call.func, "attr", None))
            if fname not in FULL_OUTPUT_FUNCS:
                continue
            if not any(kw.arg == "full_output" for kw in call.keywords):
                offenders.append(f"{path.name}:{node.lineno} {fname}")
    assert not offenders, "tuple-unpacking without full_output=True: " + str(offenders)


# ---------------------------------------------------------------------------
# #41 -- renamed booleans keep working for one release, with a warning
# ---------------------------------------------------------------------------


def test_deprecated_no_self_still_works_and_warns():
    tree = _tiny_tree()
    with pytest.warns(DeprecationWarning, match="root_self"):
        old = pt.idpar_tree(tree, no_self=True)
    np.testing.assert_array_equal(old, pt.idpar_tree(tree, root_self=False))


def test_deprecated_no_root_still_works_and_warns():
    tree = _tiny_tree()
    with pytest.warns(DeprecationWarning, match="at_root"):
        pt.elimt_tree(tree, no_root=True)


def test_new_boolean_spellings_do_not_warn():
    tree = _tiny_tree()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pt.idpar_tree(tree, root_self=False)
        pt.elimt_tree(tree, at_root=False)


# ---------------------------------------------------------------------------
# #40 -- dim is always the integer 2 or 3
# ---------------------------------------------------------------------------


def test_dim_int_replaces_dim2_bool():
    tree = _tiny_tree()
    np.testing.assert_allclose(pt.len_tree(tree, dim=2), pt.len_tree(tree, dim=2))
    assert len(pt.cyl_tree(tree, dim=2)) == 4
    assert len(pt.cyl_tree(tree, dim=3)) == 6


def test_deprecated_dim2_still_works_and_warns():
    tree = _tiny_tree()
    with pytest.warns(DeprecationWarning, match="dim=2"):
        old = pt.len_tree(tree, dim2=True)
    np.testing.assert_allclose(old, pt.len_tree(tree, dim=2))


def test_bad_dim_is_rejected_rather_than_silently_treated_as_3d():
    with pytest.raises(ValueError, match="dim must be 2 or 3"):
        pt.len_tree(_tiny_tree(), dim=4)


def test_passing_both_dim_spellings_is_an_error_not_a_preference():
    with pytest.raises(ValueError, match="not both"):
        pt.len_tree(_tiny_tree(), dim=2, dim2=True)


def test_deprecated_bf_tree_params_still_works_and_warns():
    """MATLAB calls the three published fit constants `params`, which reads
    like data. Renamed to `fit_constants`; the old name warns."""
    angles = np.linspace(0.0, np.pi, 200)
    with pytest.warns(DeprecationWarning, match="fit_constants"):
        old = pt.bf_tree(angles, dim="3d", params=(1e-6, 1.0, 1.0))
    assert old == pt.bf_tree(angles, dim="3d", fit_constants=(1e-6, 1.0, 1.0))


def test_bf_tree_rejects_both_spellings_at_once():
    with pytest.raises(ValueError, match="not both"):
        pt.bf_tree(np.linspace(0.0, np.pi, 200),
                   fit_constants=(1.0, 1.0, 1.0), params=(1.0, 1.0, 1.0))
