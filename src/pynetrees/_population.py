"""Letting per-tree functions also take a population of trees (V3).

The toolbox is written around one cell at a time, but most questions are
asked of a group: what does *this cell type* look like. The rule is now
uniform across the package -- **a list in gives a list out** -- applied at
each definition site with :func:`accepts_population` so it is visible in
the source rather than hidden in a base class.

Why a list and not a pooled array
---------------------------------

W5 concatenated: ``gene_tree`` on a group returned one stacked array, on
the reasoning that pooling is what a population analysis wants. That is
replaced here, because **list return is reversible and concatenation is
not**. ``np.vstack(results)`` recovers the pooled form in one call, while a
concatenated array cannot be split again unless the caller separately kept
every tree's node count -- and if they kept it, they did the bookkeeping the
concatenation was supposed to save them.

Nesting
-------

``dLPTCs_trees()`` returns groups of cells and a ``.mtr`` file can hold a
cell array of cell arrays, so a list of lists of trees gives a list of
lists of results. **Any** depth works, and the result mirrors the input
exactly -- element ``[2][0][7]`` of the answer is the answer for tree
``[2][0][7]``, at every level. Nothing is flattened, so a caller who
organised their cells into groups and subgroups gets those groups back.

The structure has to be *regular*: at each level either every element is a
tree, or every element is a list. A list mixing the two (``[tree, [tree]]``)
raises rather than guessing which level the caller meant.

Empty trees
-----------

A population may contain a tree with no nodes, and it **keeps its slot**:
the mapping never filters. Dropping it would renumber every cell after it,
which is exactly the kind of off-by-a-cell that survives review. The empty
tree's own result is whatever :mod:`pynetrees._empty` says it is.

What is deliberately *not* mapped
---------------------------------

**Binning** -- ``sholl_tree``, ``bin_tree``. These take the group and
compute their bins *across* it, returning one result per tree on a common
axis. Mapping them would give every cell its own bins, and bin 3 would mean
a different distance in every cell.

**Distribution fits** -- ``vonMises_tree``, ``bf_tree``. These pool by
design: the fit needs one distribution, not one per cell.

**Two-tree functions** -- ``cat_tree``, ``peters_tree``,
``share_boundary_tree``, ``ssecat_tree``. "A list of trees" is ambiguous
for these (all pairs? zipped?), so the caller writes the loop they mean.

**Savers taking a path** -- ``save_swc``, ``save_hoc``, ``save_nmf``, ….
Mapping them would write every tree to the same file. ``save_tree`` and
``save_mtr`` take a population already, because their formats hold one.
"""

from __future__ import annotations

import functools
import inspect

import numpy as np

from .core import Tree

__all__ = ["accepts_population", "is_nested_population", "is_population",
           "require_population"]


def is_population(obj) -> bool:
    """Is this a flat list/tuple of trees rather than a single tree?

    An **empty** list counts: it is vacuously a list of trees, and treating
    it as one lets the caller get "empty list of trees" from
    :func:`require_population` instead of an ``AttributeError`` from deep
    inside a per-tree computation.
    """
    return isinstance(obj, (list, tuple)) and all(
        isinstance(t, Tree) for t in obj
    )


def is_nested_population(obj) -> bool:
    """Is this a list of populations -- groups of cells, nested to any depth?

    Recursive rather than one-level, so ``[[[t]], [[t], [t]]]`` counts. The
    ``not is_population(obj)`` guard is what stops a flat list of trees from
    also answering yes, which keeps the two predicates disjoint.
    """
    return (
        isinstance(obj, (list, tuple))
        and len(obj) > 0
        and not is_population(obj)
        and all(
            is_population(group) or is_nested_population(group) for group in obj
        )
    )


def _children(level, name: str) -> list:
    """The elements of one population level, checked for a regular shape.

    Either every element is a tree or every element is a list; a level that
    mixes them is ambiguous -- ``[tree, [tree]]`` could be a 2-cell group
    whose second cell was wrapped by mistake, or a group of groups whose
    first entry was not -- and guessing turns a typo into a silently
    mis-shaped result.
    """
    members = list(level)
    trees = [i for i, m in enumerate(members) if isinstance(m, Tree)]
    lists = [i for i, m in enumerate(members) if isinstance(m, (list, tuple))]
    if trees and lists:
        raise ValueError(
            f"{name}: population level mixes trees (at index {trees[0]}) with "
            f"nested lists (at index {lists[0]}). Every element of a level "
            f"must be a tree, or every element must be a list -- wrap the "
            f"bare trees in their own list to make the nesting explicit."
        )
    if len(trees) + len(lists) != len(members):
        bad = next(
            i for i, m in enumerate(members)
            if not isinstance(m, (Tree, list, tuple))
        )
        raise TypeError(
            f"{name}: population element {bad} is a "
            f"{type(members[bad]).__name__}, not a Tree or a list of them"
        )
    return members


def require_population(trees, name: str) -> list[Tree]:
    """Validate a population argument, or say what is wrong with it."""
    if len(trees) == 0:
        raise ValueError(f"{name}: empty list of trees")
    return list(trees)


# ---------------------------------------------------------------------------
# per-tree arguments
# ---------------------------------------------------------------------------


def _is_sequence(value) -> bool:
    return isinstance(value, (list, tuple, np.ndarray)) or value is None


def _pair(name: str, parameter: str, value, count: int):
    """Decide whether ``value`` is one value for every tree, or one each.

    Returns ``None`` to broadcast, or a list of ``count`` per-tree values.

    The rule, and the reason for it: a ``list``/``tuple`` of exactly
    ``count`` elements that are *themselves* sequences is one value per
    tree; anything else -- a scalar, an array, a list of the wrong length --
    is one value for all of them. ``np.ndarray`` never zips, so wrapping in
    ``np.asarray`` is the way to force broadcast.

    The genuinely ambiguous case is a **flat list whose length happens to
    equal the number of trees**: ``delete_tree(trees, [3, 7])`` with two
    trees could mean either. That raises rather than guessing, because
    guessing wrong here deletes the wrong nodes and nothing downstream
    would notice.
    """
    if not isinstance(value, (list, tuple)) or len(value) != count:
        return None
    if all(_is_sequence(each) for each in value):
        return list(value)
    raise ValueError(
        f"{name}: `{parameter}` is a flat list of {count} values and you "
        f"passed {count} trees, which is ambiguous. Write "
        f"`{parameter}=[[...], [...]]` (one sequence per tree) to give each "
        f"tree its own, or `{parameter}=np.asarray(...)` to give the same "
        f"value to every tree."
    )


def _rebuild(signature, arguments) -> tuple[tuple, dict]:
    """Turn a bound-arguments mapping back into ``(args, kwargs)``.

    Everything that can be passed by keyword is, so that a per-tree
    substitution does not have to track positions.
    """
    args: list = []
    kwargs: dict = {}
    for parameter in list(signature.parameters.values())[1:]:
        if parameter.name not in arguments:
            continue
        value = arguments[parameter.name]
        if parameter.kind is parameter.VAR_POSITIONAL:
            args.extend(value)
        elif parameter.kind is parameter.VAR_KEYWORD:
            kwargs.update(value)
        elif parameter.kind is parameter.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[parameter.name] = value
    return tuple(args), kwargs


def accepts_population(func=None, *, paired=()):
    """Let a per-tree function also take a list -- or a list of lists.

    Parameters
    ----------
    paired : str or sequence of str, optional
        Names of parameters that may carry **one value per tree** rather
        than one value shared by all of them. See :func:`_pair` for how the
        two are told apart, and why the ambiguous case raises.

    Notes
    -----
    Goes **outside** :func:`pynetrees._empty.empty_safe`, so that an empty
    tree inside a population still gets the empty-tree treatment:
    ``@accepts_population`` first, ``@empty_safe`` beneath it. The other
    order maps to the *undecorated* function and an empty cell in a group
    would raise.
    """
    if isinstance(paired, str):
        paired = (paired,)
    paired = tuple(paired)

    def decorate(function):
        signature = inspect.signature(function)
        unknown = set(paired) - set(signature.parameters)
        if unknown:
            raise ValueError(
                f"{function.__name__}: paired parameter(s) "
                f"{sorted(unknown)} are not in its signature"
            )

        @functools.wraps(function)
        def wrapper(tree, *args, **kwargs):
            if isinstance(tree, Tree):
                return function(tree, *args, **kwargs)
            if not isinstance(tree, (list, tuple)):
                return function(tree, *args, **kwargs)

            # A list at this level is mapped over, and each element goes back
            # through `wrapper` -- so a tree is computed, a list recurses, and
            # the depth of the result matches the depth of the input however
            # deep that is. `_children` is what rejects a ragged level.
            members = _children(tree, function.__name__)
            require_population(members, function.__name__)

            split = {}
            if paired:
                bound = signature.bind_partial(None, *args, **kwargs)
                for parameter in paired:
                    if parameter not in bound.arguments:
                        continue
                    each = _pair(function.__name__, parameter,
                                 bound.arguments[parameter], len(members))
                    if each is not None:
                        split[parameter] = each
            if not split:
                return [wrapper(m, *args, **kwargs) for m in members]

            results = []
            for index, one in enumerate(members):
                arguments = dict(bound.arguments)
                for parameter, values in split.items():
                    arguments[parameter] = values[index]
                call_args, call_kwargs = _rebuild(signature, arguments)
                results.append(wrapper(one, *call_args, **call_kwargs))
            return results

        wrapper.__population_paired__ = paired
        if wrapper.__doc__:
            wrapper.__doc__ = wrapper.__doc__.rstrip() + (
                _PAIRED_NOTE.format(names=", ".join(f"``{p}``" for p in paired))
                if paired else _NOTE
            )
        return wrapper

    return decorate if func is None else decorate(func)


_NOTE = """

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.
    """

_PAIRED_NOTE = """

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape. {names} may be given
    once for the whole group or as one value per tree -- see
    :mod:`pynetrees._population`.
    """
