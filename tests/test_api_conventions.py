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

import pynetrees as pt
from pynetrees.core import Tree

SRC = pathlib.Path(__file__).parents[1] / "src" / "pynetrees"


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
# #67 -- the renamed parameters have no aliases at all
# ---------------------------------------------------------------------------
#
# Every rename from the W1 pass shipped with a one-release deprecation shim.
# The port has no users to keep compatible with, so the shims are gone and
# the old spellings must fail *loudly* -- a caller who kept the old name
# needs a TypeError, not a silently ignored keyword.


@pytest.mark.parametrize("call", [
    lambda t: pt.idpar_tree(t, no_self=True),
    lambda t: pt.elimt_tree(t, no_root=True),
    lambda t: pt.len_tree(t, dim2=True),
    lambda t: pt.cyl_tree(t, dim2=True),
    lambda t: pt.chull_tree(t, dim2=True),
])
def test_the_old_spellings_are_gone_not_ignored(call):
    with pytest.raises(TypeError, match="unexpected keyword"):
        call(_tiny_tree())


def test_bf_trees_params_alias_is_gone():
    angles = np.linspace(0.0, np.pi, 200)
    with pytest.raises(TypeError, match="unexpected keyword"):
        pt.bf_tree(angles, params=(1.0, 1.0, 1.0))


@pytest.mark.parametrize("name", ["plot_tree_mpl", "dA_tree_mpl", "spread_trees"])
def test_the_renamed_functions_are_gone(name):
    assert not hasattr(pt, name)
    assert name not in pt.__all__


def test_nothing_raises_a_deprecation_warning_any_more():
    """The whole point of #67: there is no deprecated surface left."""
    tree = _tiny_tree()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        pt.idpar_tree(tree, root_self=False)
        pt.elimt_tree(tree, at_root=False)
        pt.len_tree(tree, dim=2)
        pt.bf_tree(np.linspace(0.0, np.pi, 200), dim=3)


# ---------------------------------------------------------------------------
# #40 -- dim is always the integer 2 or 3
# ---------------------------------------------------------------------------


def test_dim_takes_an_integer():
    tree = _tiny_tree()
    assert len(pt.cyl_tree(tree, dim=2)) == 4
    assert len(pt.cyl_tree(tree, dim=3)) == 6


def test_bad_dim_is_rejected_rather_than_silently_treated_as_3d():
    with pytest.raises(ValueError, match="dim must be 2 or 3"):
        pt.len_tree(_tiny_tree(), dim=4)


def test_the_old_dim_strings_are_rejected():
    with pytest.raises(ValueError, match="dim must be 2 or 3"):
        pt.vonMises_tree(np.linspace(0.0, np.pi, 200), dim="3d")


def test_no_function_still_takes_a_dim_string():
    """Guards the rule rather than one instance of it."""
    import inspect

    offenders = []
    for name in pt.__all__:
        obj = getattr(pt, name)
        if not callable(obj) or isinstance(obj, type):
            continue
        try:
            parameter = inspect.signature(obj).parameters.get("dim")
        except (TypeError, ValueError):
            continue
        if parameter is not None and isinstance(parameter.default, str):
            offenders.append(name)
    assert not offenders, f"dim defaults to a string in: {offenders}"


def test_no_function_still_accepts_dim2():
    """Same, for the boolean spelling."""
    import inspect

    offenders = []
    for name in pt.__all__:
        obj = getattr(pt, name)
        if not callable(obj) or isinstance(obj, type):
            continue
        try:
            parameters = inspect.signature(obj).parameters
        except (TypeError, ValueError):
            continue
        if "dim2" in parameters:
            offenders.append(name)
    assert not offenders, f"dim2 still accepted by: {offenders}"
