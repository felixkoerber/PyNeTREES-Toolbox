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

Levels may be **ragged**: ``[tree, [tree, tree]]`` -- one cell beside a group
of two -- returns ``[value, [value, value]]``. Every element is handled on
its own terms, so there is no shape to infer and nothing to get wrong.

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

**Joined networks** -- ``ssecat_tree``, ``syncat_tree``. Their list *is* one
network of coupled cells, so the whole list is the input.

Two-tree functions -- ``cat_tree``, ``peters_tree``, ``share_boundary_tree`` --
take populations through :func:`accepts_tree_pairs`: every pairing by default,
or element by element with ``pairwise=False``.

**Savers taking a path** -- ``save_swc``, ``save_hoc``, ``save_nmf``, ….
Mapping them would write every tree to the same file. ``save_tree`` and
``save_mtr`` take a population already, because their formats hold one.
"""

from __future__ import annotations

import functools
import inspect

import numpy as np

from .core import Tree

__all__ = ["accepts_population", "accepts_tree_pairs", "is_nested_population",
           "is_population", "leaves", "map_leaves", "require_population"]


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
    """Is this a population with at least one level of nesting in it?

    Recursive rather than one-level, so ``[[[t]], [[t], [t]]]`` counts, and
    tolerant of ragged levels, so ``[t, [t]]`` counts too -- an element may
    be a tree or a group. The ``not is_population(obj)`` guard is what stops
    a *flat* list of trees from also answering yes, which keeps the two
    predicates disjoint.
    """
    return (
        isinstance(obj, (list, tuple))
        and len(obj) > 0
        and not is_population(obj)
        and all(
            isinstance(member, Tree)
            or is_population(member)
            or is_nested_population(member)
            for member in obj
        )
    )


def _children(level, name: str) -> list:
    """The elements of one population level, each a tree or a list of them.

    A level may **mix** the two. ``[tree, [tree, tree]]`` is a perfectly
    good structure -- one cell and a group of two -- and the result mirrors
    it position for position: ``[value, [value, value]]``. There is nothing
    to disambiguate, because the shape of the answer is read off the shape
    of the question rather than inferred from it.

    (An earlier version rejected mixed levels on the theory that they might
    be a typo. That was wrong: a rule which maps element to element cannot
    be confused by its input, and refusing a valid structure to guard
    against a hypothetical mistake made the common case worse.)

    What is still refused is an element that is neither -- a string, an
    array, a number -- because that one really has no per-tree meaning.
    """
    members = list(level)
    for index, member in enumerate(members):
        if not isinstance(member, (Tree, list, tuple)):
            raise TypeError(
                f"{name}: population element {index} is a "
                f"{type(member).__name__}, not a Tree or a list of them"
            )
    return members


def leaves(population) -> list:
    """Every element of a population, in order, however deeply it is nested."""
    if isinstance(population, (list, tuple)):
        return [leaf for member in population for leaf in leaves(member)]
    return [population]


def map_leaves(function, population):
    """Apply ``function`` to every tree, keeping the population's nesting."""
    if isinstance(population, (list, tuple)):
        return [map_leaves(function, member) for member in population]
    return function(population)


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
            # deep that is. An empty level is a group with nobody in it, and
            # comes back as an empty group so the indices around it hold.
            members = _children(tree, function.__name__)

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


def accepts_tree_pairs(*, empty=None):
    """Let a two-tree function take populations of trees on either side.

    With ``pairwise=True`` (the default) every tree of the first population
    meets every tree of the second. The result has the first population's
    nesting, and each of its leaves has the second population's nesting.
    With ``pairwise=False`` trees are matched element by element, and the
    two populations must have the same nesting.

    Parameters
    ----------
    empty : callable, optional
        ``empty(first, second)`` gives the result when either tree of a pair
        has no nodes, so that an empty cell keeps its slot.
    """

    def decorate(function):
        name = function.__name__

        def one(first, second, args, kwargs):
            if empty is not None and 0 in (first.n_nodes, second.n_nodes):
                return empty(first, second)
            return function(first, second, *args, **kwargs)

        def every_pair(first, second, args, kwargs):
            if isinstance(first, (list, tuple)):
                return [every_pair(m, second, args, kwargs)
                        for m in _children(first, name)]
            if isinstance(second, (list, tuple)):
                return [every_pair(first, m, args, kwargs)
                        for m in _children(second, name)]
            return one(first, second, args, kwargs)

        def matched(first, second, args, kwargs, where="population"):
            first_many = isinstance(first, (list, tuple))
            if first_many != isinstance(second, (list, tuple)):
                raise ValueError(
                    f"{name}: with pairwise=False both populations need the "
                    f"same nesting, but {where} is a tree on one side and a "
                    f"list on the other"
                )
            if not first_many:
                return one(first, second, args, kwargs)
            left, right = _children(first, name), _children(second, name)
            if len(left) != len(right):
                raise ValueError(
                    f"{name}: with pairwise=False both populations need the "
                    f"same nesting, but {where} has {len(left)} elements on "
                    f"one side and {len(right)} on the other"
                )
            return [matched(a, b, args, kwargs, f"{where}[{index}]")
                    for index, (a, b) in enumerate(zip(left, right))]

        @functools.wraps(function)
        def wrapper(first, second, *args, pairwise=True, **kwargs):
            if isinstance(first, Tree) and isinstance(second, Tree):
                return one(first, second, args, kwargs)
            combine = every_pair if pairwise else matched
            return combine(first, second, args, kwargs)

        signature = inspect.signature(function)
        wrapper.__signature__ = signature.replace(parameters=[
            *signature.parameters.values(),
            inspect.Parameter("pairwise", inspect.Parameter.KEYWORD_ONLY, default=True),
        ])
        if wrapper.__doc__:
            wrapper.__doc__ = wrapper.__doc__.rstrip() + _PAIRS_NOTE
        return wrapper

    return decorate


_PAIRS_NOTE = """

    Either tree may be a **population** (a list of trees, nested to any depth).
    By default every pairing is computed, with the result nested as the first
    population and then the second. ``pairwise=False`` matches them element by
    element instead, and raises if their nesting differs.
    """

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
