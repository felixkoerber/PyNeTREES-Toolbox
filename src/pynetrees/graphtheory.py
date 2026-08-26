"""Graph-theoretic primitives operating on a Tree's adjacency matrix alone.

Ports treestoolbox-master/graphtheory/*.m. Every function here assumes its
input is a single, connected, singly-rooted tree (``dA`` has exactly one
all-zero row, the root) -- the same standing assumption the MATLAB originals
make (there, the root is always node 1; here, node 0 once a tree has been
through ``sort_tree``). ``sort_tree`` itself is the one function that must
*establish* that invariant from an arbitrary starting order, so it alone
detects the root generically via ``_root_index`` rather than assuming it.

Where an original MATLAB algorithm was a compact but opaque sparse-matrix
trick that only worked because MATLAB is 1-based (root's own index, 1, could
never collide with the "no parent" sentinel, 0), this port re-derives the
same documented result using explicit 0-based bookkeeping and the ``-1``
"no parent" sentinel (see PORT_STATUS.md, Design Decisions #4/#5). Where the
original algorithm is just standard graph theory (BFS/DFS/post-order), this
port uses that directly instead of the MATLAB matrix-power formulation --
it's equivalent, simpler to verify, and easier to read.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import sparse

from ._population import accepts_population
from ._empty import empty_safe
from .core import NO_PARENT, Tree

# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


def _root_index(dA: sparse.spmatrix) -> int:
    """0-based index of the tree's root (the node with no parent).

    Detected via row-sum rather than "index 0", since 0 is a valid node
    index here (unlike MATLAB) and can't double as a sentinel.

    This is the ``dA``-only form, kept for the handful of internal callers
    that hold an adjacency matrix rather than a :class:`~pynetrees.Tree`.
    **Prefer :attr:`Tree.root`** in anything that has a Tree in hand -- it's
    public, needs no cross-module private import, and is therefore the one
    people actually reach for (see Design Decision #48; three geometry
    functions hardcoded ``[0]`` precisely because this helper was awkward to
    get at from ``metrics``/``construct``).
    """
    row_sums = np.asarray(dA.sum(axis=1)).ravel()
    roots = np.flatnonzero(row_sums == 0)
    if len(roots) != 1:
        raise ValueError(f"expected exactly one root, found {len(roots)}")
    return int(roots[0])


def _children_lists(dA: sparse.spmatrix) -> list[list[int]]:
    """``children[j]`` = list of direct children of node ``j``."""
    n = dA.shape[0]
    children: list[list[int]] = [[] for _ in range(n)]
    coo = dA.tocoo()
    for child, parent in zip(coo.row.tolist(), coo.col.tolist()):
        children[parent].append(child)
    return children


def _dfs_preorder(dA: sparse.spmatrix) -> np.ndarray:
    """Pre-order node visitation from the root, children in ascending index."""
    root = _root_index(dA)
    children = _children_lists(dA)
    for lst in children:
        lst.sort()
    order = [root]
    stack = list(reversed(children[root]))
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(reversed(children[node]))
    return np.array(order)


def _subtree_blocks(dA: sparse.spmatrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Descendant sets for *every* node, computed once in O(n_nodes) total.

    A pre-order traversal emits each node immediately followed by its entire
    subtree, so every node's descendants occupy one contiguous block of that
    order. Returns ``(order, start, size)`` where
    ``order[start[i] : start[i] + size[i]]`` is exactly node ``i``'s
    descendant set (including ``i`` itself).

    This exists because the obvious alternatives are both accidentally
    quadratic when a loop needs a descendant set at *every* node: calling
    :func:`sub_tree` per node is a fresh BFS each time, and scanning a
    prebuilt ``ipar_tree`` matrix (``(ipar == node).any(axis=1)``) rereads
    all ``n_nodes x max_depth`` entries per node -- which on a real granule
    cell (3765 nodes, 1625 deep) is ~2e10 element comparisons. Building the
    blocks once instead makes the whole sweep linear.
    """
    order = _dfs_preorder(dA)
    n = dA.shape[0]
    start = np.empty(n, dtype=int)
    start[order] = np.arange(len(order))

    children = _children_lists(dA)
    size = np.ones(n, dtype=int)
    for node in order[::-1]:  # children always precede parents in reverse pre-order
        for child in children[node]:
            size[node] += size[child]
    return order, start, size


# ---------------------------------------------------------------------------
# node typing: B / C / T
# ---------------------------------------------------------------------------


def _children_count(dA: sparse.spmatrix) -> np.ndarray:
    return np.asarray(dA.sum(axis=0)).ravel()


@accepts_population
def typeN_tree(tree: Tree) -> np.ndarray:
    """Node type per node: 0 terminal, 1 continuation, 2 (or more) branch."""
    typeN = _children_count(tree.dA)
    typeN = np.minimum(typeN, 2)
    return typeN.astype(int)


@accepts_population
def B_tree(tree: Tree) -> np.ndarray:
    """Boolean mask of branch points (more than one child)."""
    return _children_count(tree.dA) > 1


@accepts_population
def C_tree(tree: Tree) -> np.ndarray:
    """Boolean mask of continuation points (exactly one child)."""
    return _children_count(tree.dA) == 1


@accepts_population
def T_tree(tree: Tree) -> np.ndarray:
    """Boolean mask of termination points (no children)."""
    return _children_count(tree.dA) == 0


# ---------------------------------------------------------------------------
# parent / child indices
# ---------------------------------------------------------------------------


@accepts_population
def idpar_tree(tree: Tree, root_self: bool = True) -> np.ndarray:
    """0-based index of each node's direct parent.

    Parameters
    ----------
    tree : Tree
    root_self : bool, default True
        What the root gets, since it has no parent. ``True`` (MATLAB's
        default) makes the root its own parent, which lets expressions like
        ``v[idpar]`` be written without a special case -- ``ratio_tree``
        relies on it to give the root a ratio of exactly 1. ``False``
        (MATLAB's ``'-z'``) gives the root :data:`NO_PARENT` (``-1``)
        instead, which is what you want when you are about to *walk* the
        parent chain and need a stopping condition.

    Returns
    -------
    np.ndarray
        Integer array of length ``n_nodes``.
    """
    coo = tree.dA.tocoo()
    # start with every node as its own parent, then overwrite the ones that
    # actually have an incoming edge -- which leaves exactly the roots
    idpar = np.arange(tree.n_nodes, dtype=int)
    idpar[coo.row] = coo.col
    if not root_self:
        is_root = np.ones(tree.n_nodes, dtype=bool)
        is_root[coo.row] = False
        idpar[is_root] = NO_PARENT
    return idpar


@accepts_population
def idchild_tree(tree: Tree, nodes=None, first_only: bool = False) -> np.ndarray:
    """Direct child indices of each node in ``nodes`` (default: all nodes).

    Returns an ``(len(nodes), width)`` int array, :data:`NO_PARENT`-padded,
    where ``width`` is the largest number of children found (MATLAB hardcodes
    ``width=2``, silently truncating any trifurcation; this port doesn't).
    """
    dA = tree.dA.tocsc()
    N = tree.n_nodes
    nodes = np.arange(N) if nodes is None else np.asarray(nodes, dtype=int)

    children_lists = [np.flatnonzero(dA[:, j].toarray().ravel()) for j in nodes]
    width = max((len(c) for c in children_lists), default=0)
    width = max(width, 1)

    idchild = np.full((len(nodes), width), NO_PARENT, dtype=int)
    for row, children in enumerate(children_lists):
        idchild[row, : len(children)] = children

    return idchild[:, 0] if first_only else idchild


# ---------------------------------------------------------------------------
# path length / order
# ---------------------------------------------------------------------------


@accepts_population
@empty_safe("nodes", dtype=int)
def PL_tree(tree: Tree) -> np.ndarray:
    """Topological path length (number of edges) from each node to the root.

    The root has path length 0, its children 1, and so on -- so this is node
    depth, and it is computed as ``PL[node] = PL[parent] + 1`` in pre-order,
    which is O(n_nodes).

    MATLAB computes it by repeated sparse matrix-vector multiplication, one
    per depth level. That is idiomatic and fast in MATLAB, but transliterated
    to SciPy it costs one Python-level call per level -- 1624 of them on a
    real granule cell, ~37 ms against ~2 ms here. See also :func:`LO_tree`.
    """
    order = _dfs_preorder(tree.dA)
    idpar = idpar_tree(tree, root_self=False)

    PL = np.zeros(tree.n_nodes, dtype=float)
    for node in order:
        parent = idpar[node]
        if parent != NO_PARENT:
            PL[node] = PL[parent] + 1.0
    return PL


@accepts_population
def ipar_tree(tree: Tree, terminals_only: bool = False, nodes=None) -> np.ndarray:
    """Ancestor path of every node: ``[i, parent(i), ..., root]``.

    Parameters
    ----------
    tree : Tree
    terminals_only : bool, default False
        MATLAB's ``'-T'``. Return one row per **termination point**, each
        holding only the path back to (and excluding) its first branch
        point -- i.e. that terminal's own unbranched segment.
    nodes : array_like, optional
        Restrict to these nodes' rows. With ``terminals_only``, selects
        which terminals (MATLAB's ``ipart``).

    Returns
    -------
    np.ndarray
        ``(n_rows, max_depth + 2)`` int array, :data:`NO_PARENT`-padded.

    Notes
    -----
    The full matrix is this toolbox's worst-scaling structure: it is dense
    and ``n_nodes x max_depth``, which is 49 MB for a 3765-node granule cell
    and remains ~3x superlinear (see `docs/port-audit.md`). Most callers
    only need a traversal -- ``Pvec_tree``, ``PL_tree``, ``flatten_tree``,
    ``morph_tree`` and ``smooth_tree`` were all moved off it for exactly
    that reason. Reach for it when you genuinely need arbitrary ancestor
    queries, and prefer ``terminals_only=True`` when you need terminal
    segments, since that form is dramatically smaller.
    """
    N = tree.n_nodes
    idpar_noself = idpar_tree(tree, root_self=False)
    max_depth = int(PL_tree(tree).max()) if N > 1 else 0

    ipar = np.full((N, max_depth + 2), NO_PARENT, dtype=int)
    current = np.arange(N)
    for col in range(max_depth + 2):
        ipar[:, col] = current
        valid = current != NO_PARENT
        nxt = np.full(N, NO_PARENT, dtype=int)
        nxt[valid] = idpar_noself[current[valid]]
        current = nxt

    if terminals_only:
        terminals = np.flatnonzero(T_tree(tree))
        if nodes is not None:
            terminals = np.intersect1d(terminals, np.asarray(nodes, dtype=int))
        rows = ipar[terminals]
        # Walk out from each terminal and stop at the first branch point.
        # The branch point itself is excluded: it belongs to the parent
        # section, not to this terminal's unbranched run.
        is_branch = B_tree(tree)
        seen_branch = np.zeros(len(rows), dtype=bool)
        for col in range(rows.shape[1]):
            entry = rows[:, col]
            real = entry != NO_PARENT
            rows[seen_branch, col] = NO_PARENT
            hit = np.zeros(len(rows), dtype=bool)
            hit[real] = is_branch[entry[real]]
            rows[hit & ~seen_branch, col] = NO_PARENT
            seen_branch |= hit
        # trim the all-padding tail so the result is actually smaller
        keep = (rows != NO_PARENT).any(axis=0)
        return rows[:, keep] if keep.any() else rows[:, :1]

    return ipar if nodes is None else ipar[np.asarray(nodes, dtype=int)]


@accepts_population
@empty_safe("nodes", dtype=int)
def BO_tree(tree: Tree) -> np.ndarray:
    """Branch order of each node: how many branch points lie between it and
    the root (root itself has branch order 0)."""
    dA = tree.dA.tocsr()
    root = _root_index(dA)
    typeN = typeN_tree(tree).astype(float)
    # scale each column j by typeN[j]: passing through node j on the way up
    # multiplies the running total by 2 iff j is a branch point
    sdA = dA.multiply(typeN.reshape(1, -1)).tocsr()

    BO = np.asarray(sdA[:, root].todense()).ravel()
    resBO = BO.copy()
    while np.sum(resBO) != 0:
        resBO = np.asarray(sdA @ resBO).ravel()
        BO = BO + resBO
    BO[root] = 1.0
    return np.log2(BO)


@accepts_population
@empty_safe("nodes", dtype=int)
def LO_tree(tree: Tree) -> np.ndarray:
    """Level order: for each node, its own topological path length plus the
    path lengths of every node below it -- a near-unique isomorphism
    invariant, used by :func:`sort_tree`'s ``'lo'`` mode as a tie-breaker.

    Written as the O(n_nodes) recurrence it actually is: each node's
    descendant sum is its children's descendant sums plus their own path
    lengths, accumulated bottom-up.

    MATLAB reaches the same quantity by repeatedly multiplying a sparse
    matrix by ``dA`` until the root column empties -- effectively summing
    path lengths one generation at a time. That is a natural MATLAB idiom
    and fast there, but a literal transliteration performs one SciPy
    sparse matmul per tree *level* (1624 of them on a real granule cell)
    and the per-call overhead dominates: 512 ms against ~6 ms here. The
    equivalence ``LO == PL + (sum of PL over descendants)`` was verified
    exactly (max abs difference 0.0) on hand-built trees and on both
    bundled reconstructions before this replaced the transliteration.
    """
    PL = PL_tree(tree)
    order = _dfs_preorder(tree.dA)
    children = _children_lists(tree.dA)

    descendant_sum = np.zeros(tree.n_nodes, dtype=float)
    for node in order[::-1]:  # reverse pre-order: children before parents
        for child in children[node]:
            descendant_sum[node] += descendant_sum[child] + PL[child]
    return PL + descendant_sum


# ---------------------------------------------------------------------------
# meta-functions operating on an arbitrary per-node vector
# ---------------------------------------------------------------------------


@accepts_population(paired="v")
def child_tree(tree: Tree, v: np.ndarray | None = None) -> np.ndarray:
    """For each node, the sum of ``v`` over *all* of its descendants
    (excluding itself). Default ``v`` is all-ones, giving descendant counts.
    """
    N = tree.n_nodes
    v = np.ones(N) if v is None else np.asarray(v, dtype=float)

    ipar = ipar_tree(tree)
    ancestors = ipar[:, 1:]  # each node's ancestor chain, excluding itself
    valid = ancestors != NO_PARENT
    rows, cols = np.nonzero(valid)
    ancestor_idx = ancestors[rows, cols]

    child = np.zeros(N)
    np.add.at(child, ancestor_idx, v[rows])
    return child


@accepts_population(paired="v")
@empty_safe("nodes")
def Pvec_tree(tree: Tree, v: np.ndarray | None = None) -> np.ndarray:
    """Cumulative sum of ``v`` along the path from the root to each node
    (inclusive of the node itself).

    Parameters
    ----------
    tree : Tree
    v : np.ndarray, optional
        Per-node quantity to accumulate. Defaults to
        :func:`~pynetrees.len_tree`, giving **metric path length from the
        root [um]** -- overwhelmingly the intended meaning, and what six
        call sites inside this toolbox alone were spelling out longhand as
        ``Pvec_tree(tree, len_tree(tree))``. Pass ``np.ones(n_nodes)`` for
        topological depth + 1, or any other per-node array.

    Returns
    -------
    np.ndarray
        Float array of length ``n_nodes``.

    Notes
    -----
    Computed by the recurrence ``P[node] = P[parent] + v[node]`` in
    pre-order, which is O(n_nodes). The previous version summed a prebuilt
    ``ipar_tree`` matrix instead -- correct, but that matrix is
    ``n_nodes x max_depth`` (49 MB, 6.1M entries for a real granule cell),
    so it was the worst-scaling function in the toolbox at 3.4x superlinear.
    """
    if v is None:
        from .metrics import len_tree

        v = len_tree(tree)
    v = np.asarray(v, dtype=float)
    order = _dfs_preorder(tree.dA)
    idpar = idpar_tree(tree, root_self=False)

    out = np.zeros(tree.n_nodes, dtype=float)
    for node in order:
        parent = idpar[node]
        out[node] = v[node] + (out[parent] if parent != NO_PARENT else 0.0)
    return out


@accepts_population(paired="v")
def ratio_tree(tree: Tree, v: np.ndarray | None = None) -> np.ndarray:
    """Ratio of ``v`` at each node to ``v`` at its parent (root: 1.0)."""
    v = tree.D if v is None else np.asarray(v, dtype=float)
    idpar = idpar_tree(tree)  # self-referencing: root's ratio is v/v == 1
    return v / v[idpar]


# ---------------------------------------------------------------------------
# regions, subtrees, rerooting, sorting
# ---------------------------------------------------------------------------


@accepts_population
def rindex_tree(tree: Tree) -> np.ndarray:
    """0-based rank of each node within its own region, by node order."""
    R = tree.R
    rindex = np.zeros(len(R), dtype=int)
    for region in np.unique(R):
        mask = R == region
        rindex[mask] = np.arange(int(mask.sum()))
    return rindex


class SubTree(NamedTuple):
    """Result of :func:`sub_tree`: which nodes, and the tree they form."""

    mask: np.ndarray
    """Boolean mask over the *parent* tree, ``True`` for nodes in the subtree
    (matching MATLAB's ``sub``: "1 if part of subtree, 0 if not")."""
    tree: Tree | None
    """The extracted subtree, or ``None`` if ``with_tree=False``."""


@accepts_population
def sub_tree(tree: Tree, inode: int, with_tree: bool = True) -> SubTree:
    """The subtree rooted at ``inode``: which nodes it contains, and the tree.

    Parameters
    ----------
    tree : Tree
    inode : int
        0-based index of the subtree's root.
    with_tree : bool, default True
        Whether to build the extracted :class:`~pynetrees.Tree` as well as the
        mask. Pass ``False`` in a per-node loop -- it costs about 30% extra
        per call (2203 vs 1682 us on a 3765-node granule cell), which is
        cheap once but not free thousands of times over.

    Returns
    -------
    SubTree
        Named tuple ``(mask, tree)``. Unpacks like MATLAB's
        ``[sub, subtree] = sub_tree(...)``, and ``result.mask`` also works.

    Notes
    -----
    **Region names are trimmed** to those the subtree actually uses, with
    ``R`` reindexed to match. MATLAB does not do this -- ``sub_tree.m``
    carries the comment *"NOTE ! region update for tree output still
    missing!!!"* -- and the result there keeps the whole parent's region
    list. Cutting a purely dendritic branch out of a granule cell and being
    told it still has an ``axon`` region is not useful, so this port closes
    the gap rather than reproducing it (Design Decision #50).

    Traversal walks the child lists directly. An earlier version read each
    node's children as ``dA[:, node].toarray()``, which materialises a dense
    length-``n_nodes`` column *per visited node* and makes a single BFS
    O(n_nodes^2) -- 514 ms on that same granule cell, against ~1.7 ms here.
    """
    children = _children_lists(tree.dA)
    mask = np.zeros(tree.n_nodes, dtype=bool)
    mask[inode] = True
    stack = [inode]
    while stack:
        node = stack.pop()
        for child in children[node]:
            if not mask[child]:
                mask[child] = True
                stack.append(child)

    if not with_tree:
        return SubTree(mask, None)
    return SubTree(mask, _extract_subtree(tree, mask))


def _extract_subtree(tree: Tree, mask: np.ndarray) -> Tree:
    """Cut ``mask``'s nodes out as a standalone Tree, trimming unused regions.

    Splitting this out of :func:`sub_tree` keeps the traversal and the
    reindex separately testable, and gives ``delete_tree``'s forest split a
    single place to get the same region trimming.
    """
    nodes = np.flatnonzero(mask)
    sub = tree.reindexed(nodes)

    used = np.unique(sub.R)
    if len(used) == len(tree.rnames):
        return sub  # every region still present; nothing to trim

    # remap R so it indexes the trimmed rnames list rather than the parent's
    remap = np.full(len(tree.rnames), -1, dtype=int)
    remap[used] = np.arange(len(used))
    sub.R = remap[sub.R]
    sub.rnames = [tree.rnames[i] for i in used]
    return sub


class RedirectResult(NamedTuple):
    """Result of :func:`redirect_tree` with ``full_output=True``."""

    tree: Tree
    """The rerooted tree."""
    order: np.ndarray
    """``order[i]`` is the old node index now sitting at new position ``i``."""


@accepts_population
def redirect_tree(
    tree: Tree, new_root: int, name: str | None = None, *, full_output: bool = False
):
    """Reroot the tree at ``new_root``, reversing edge direction as needed.

    Parameters
    ----------
    tree : Tree
    new_root : int
        0-based index of the node to become the new root.
    name : str, optional
        Name for the returned tree; defaults to the input's.
    full_output : bool, default False
        If ``True``, return a :class:`RedirectResult` ``(tree, order)``
        instead of just the tree -- ``order`` being the only way to map old
        node indices onto new ones after the reindex (Design Decision #42).

    Returns
    -------
    Tree or RedirectResult

    Warns
    -----
    UserWarning
        If ``new_root`` is a branch point. Rerooting there leaves it a
        trifurcation, so the result is no longer binary -- matching the
        MATLAB original's documented restriction.
    """
    N = tree.n_nodes
    if B_tree(tree)[new_root]:
        import warnings

        warnings.warn(
            f"node {new_root} is a branch point; redirecting there creates "
            "a trifurcation",
            stacklevel=2,
        )

    undirected = (tree.dA + tree.dA.T).tocsr()
    order = [new_root]
    parent_of: dict[int, int] = {}
    visited = np.zeros(N, dtype=bool)
    visited[new_root] = True
    frontier = [new_root]
    while frontier:
        nxt = []
        for node in frontier:
            neighbors = undirected.indices[
                undirected.indptr[node] : undirected.indptr[node + 1]
            ]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    parent_of[nb] = node
                    order.append(int(nb))
                    nxt.append(int(nb))
        frontier = nxt

    if len(order) != N:
        raise ValueError("tree is disconnected; cannot redirect")

    order_arr = np.array(order)
    old_to_new = {old: new for new, old in enumerate(order_arr)}
    rows = [old_to_new[old] for old in order[1:]]
    cols = [old_to_new[parent_of[old]] for old in order[1:]]
    dA_new = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(N, N)
    ).tocsr()

    new_tree = Tree(
        dA=dA_new,
        X=tree.X[order_arr],
        Y=tree.Y[order_arr],
        Z=tree.Z[order_arr],
        D=tree.D[order_arr],
        R=tree.R[order_arr],
        rnames=tree.rnames,
        name=tree.name if name is None else name,
    )
    return RedirectResult(new_tree, order_arr) if full_output else new_tree


class SortResult(NamedTuple):
    """Result of :func:`sort_tree` with ``full_output=True``."""

    tree: Tree
    """The reindexed, BCT-conform tree."""
    order: np.ndarray
    """``order[i]`` is the old node index now sitting at new position ``i``."""


@accepts_population
@empty_safe("tree")
def sort_tree(tree: Tree, by: str = "hier", *, full_output: bool = False):
    """Reindex nodes to be BCT-conform: every parent precedes its children,
    and each subtree occupies a contiguous index block.

    Parameters
    ----------
    tree : Tree
    by : {'hier', 'lo', 'lex'}, default 'hier'
        Which of the many valid BCT orderings to produce.

        - ``'hier'`` -- keep nodes in their existing relative order, only
          fixing up parent/child adjacency. Arbitrary among the valid
          orderings, but the cheapest.
        - ``'lo'`` -- order by (topological path length, level order),
          giving a near-canonical ordering (MATLAB's ``'-LO'``).
        - ``'lex'`` -- order by number of children: terminals, then
          continuations, then branches (MATLAB's ``'-LEX'``).
    full_output : bool, default False
        If ``True``, return a :class:`SortResult` ``(tree, order)`` instead
        of just the tree (Design Decision #42).

    Returns
    -------
    Tree or SortResult

    Notes
    -----
    ``'hier'`` is a DFS pre-order rather than MATLAB's level-order-ish
    scheme (Design Decision #12). Both satisfy the BCT invariant and nothing
    downstream depends on which valid ordering it gets, but the consequence
    is worth stating plainly: **node indices are not comparable between
    MATLAB and pynetrees after a sort.** Do not cross-reference "node 417"
    between the two toolboxes.
    """
    N = tree.n_nodes
    if by == "hier":
        pre_order = np.arange(N)
    elif by == "lo":
        PL = PL_tree(tree)
        LO = LO_tree(tree)
        pre_order = np.lexsort((LO, PL))
    elif by == "lex":
        root = _root_index(tree.dA)
        ndaughters = _children_count(tree.dA)
        rest = [i for i in range(N) if i != root]
        rest.sort(key=lambda i: ndaughters[i])
        pre_order = np.array([root, *rest])
    else:
        raise ValueError(f"unknown sort mode {by!r}")

    dA_pre = tree.dA.tocsr()[pre_order][:, pre_order]
    hier_order = _dfs_preorder(dA_pre)
    order = pre_order[hier_order]
    sorted_tree = tree.reindexed(order)
    return SortResult(sorted_tree, order) if full_output else sorted_tree


@accepts_population
@empty_safe("nodes", dtype=int)
def strahler_tree(tree: Tree) -> np.ndarray:
    """Strahler number of each node (terminals are 1; a node is
    ``max(children) + 1`` if 2+ children tie for the max, else
    ``max(children)``)."""
    children = _children_lists(tree.dA)
    order = _dfs_preorder(tree.dA)
    strahler = np.zeros(tree.n_nodes, dtype=int)

    for node in reversed(order.tolist()):
        kids = children[node]
        if not kids:
            strahler[node] = 1
            continue
        kid_vals = sorted((strahler[k] for k in kids), reverse=True)
        strahler[node] = (
            kid_vals[0] + 1 if len(kid_vals) > 1 and kid_vals[0] == kid_vals[1]
            else kid_vals[0]
        )
    return strahler


class BranchLengthOrder(NamedTuple):
    """Result of :func:`BLO_tree`."""

    order: np.ndarray
    """``(n_nodes,)`` int: which branch each node belongs to, 1 = longest."""
    length: np.ndarray
    """``(n_nodes,)`` float: total ``v`` along that node's whole branch."""
    cumulative: np.ndarray
    """``(n_nodes,)`` float: ``v`` accumulated from the branch's start."""


def _empty_blo(tree):
    return BranchLengthOrder(np.empty(0, dtype=int), np.empty(0), np.empty(0))


@accepts_population(paired="v")
@empty_safe(_empty_blo)
def BLO_tree(tree: Tree, v: np.ndarray | None = None, *,
             by: str = "nodes") -> BranchLengthOrder:
    """Branch length order: decompose the tree into paths, deepest first.

    Repeatedly takes the deepest remaining root-to-tip path, calls it the
    next branch, and removes it; branch 2 is the deepest path hanging off
    branch 1, and so on. Unlike :func:`BO_tree` and :func:`strahler_tree`,
    which label nodes by local topology, this cuts the arbor into whole
    paths -- which is what makes it the foundation of the
    persistent-homology description of a morphology
    (:func:`pynetrees.barcode_tree`), where each branch becomes one bar.

    Parameters
    ----------
    tree : Tree
    v : array_like, optional
        Per-node values accumulated into ``length`` and ``cumulative``.
        Defaults to ``len_tree(tree)``, so branches are measured in
        microns. **With** ``by="nodes"`` **this does not affect the
        ordering** -- see the Notes.
    by : {'nodes', 'length'}, keyword-only, default 'nodes'
        What "deepest" means. ``'nodes'`` counts nodes carrying ``v > 0``,
        which is what MATLAB does. ``'length'`` maximises accumulated
        ``v``, which is what MATLAB's name and documentation describe.

    Returns
    -------
    BranchLengthOrder
        ``(order, length, cumulative)``, each ``(n_nodes,)``. ``order`` is
        **1-based** -- it is a rank, not an index.

    Notes
    -----
    A branch's length includes the segment joining it to its parent branch,
    which is what makes consecutive bars in the barcode abut rather than
    leave a gap.

    **The default is not what the MATLAB function's name suggests, and the
    default is deliberate.** MATLAB selects each branch with
    ``max (sum (V0 (ipar + 1) > 0, 2))`` -- the *count* of path nodes with
    a positive value. It never sums ``V``. Two consequences, both
    measurable:

    - Branch 1 is the path with the most nodes, not the longest one. On
      ``hsn_tree`` MATLAB's first branch ends 319.5 um from the root while
      the furthest tip is at 648.4 um.
    - Any strictly positive ``v`` gives an identical ordering, so ``v``
      selects nothing. Passing ``eucl_tree`` or ``ones`` changes only the
      measured lengths.

    ``by="nodes"`` is kept as the default so that barcodes match MATLAB's
    and published analyses reproduce; ``by="length"`` does what the name
    says. They disagree about where 69-97% of nodes belong on the bundled
    trees, so this is not a detail. Recorded in ``MATLAB_TOOLBOX_BUGS.md``.

    MATLAB rebuilds `ipar_tree`'s dense ``n_nodes x max_depth`` matrix and
    rescans it once per branch. This is the same decomposition computed
    with a heap in ``O(n log n)``; ``tests/test_persistence.py`` checks the
    two agree node-for-node against MATLAB's own output.
    """
    import heapq

    from .metrics import len_tree

    N = tree.n_nodes
    values = len_tree(tree) if v is None else np.asarray(v, dtype=float)
    if len(values) != N:
        raise ValueError(f"v must be length n_nodes ({N}), got {len(values)}")
    if by not in ("nodes", "length"):
        raise ValueError(f"by must be 'nodes' or 'length', got {by!r}")

    idpar = idpar_tree(tree, root_self=False)
    children = _children_lists(tree.dA)
    order_of_visit = _dfs_preorder(tree.dA).tolist()

    # `depth[x]`: how deep x is along its root path, in whichever currency
    # `by` names. Because the already-assigned nodes always form a
    # connected set containing the root, a node's *remaining* depth is just
    # `depth[x] - depth[anchor]`, which is what makes the heap work.
    weight = (values > 0).astype(float) if by == "nodes" else values
    depth = np.zeros(N)
    for node in order_of_visit:
        parent = idpar[node]
        depth[node] = weight[node] + (0.0 if parent == NO_PARENT
                                      else depth[parent])

    # `deepest[x]`: the largest `depth` anywhere in x's subtree, and
    # `tip[x]`: the lowest-numbered node achieving it -- MATLAB's `max`
    # returns the first index among ties, so reproducing that is what makes
    # the two implementations agree node-for-node.
    deepest = depth.copy()
    tip = np.arange(N)
    for node in reversed(order_of_visit):
        parent = idpar[node]
        if parent == NO_PARENT:
            continue
        if (deepest[node], -tip[node]) > (deepest[parent], -tip[parent]):
            deepest[parent], tip[parent] = deepest[node], tip[node]

    blo = np.zeros(N, dtype=int)
    branch_length = np.zeros(N)
    cumulative = np.zeros(N)

    root = _root_index(tree.dA)
    frontier = [(-deepest[root], tip[root], root, 0)]
    counter = 0
    while frontier:
        _, leaf, head, _ = heapq.heappop(frontier)
        counter += 1

        branch = [leaf]
        while branch[-1] != head:
            branch.append(idpar[branch[-1]])
        branch.reverse()  # head first, so the accumulation runs outwards

        along = values[branch]
        blo[branch] = counter
        branch_length[branch] = along.sum()
        cumulative[branch] = np.cumsum(along)

        for node in branch:
            for child in children[node]:
                if blo[child] == 0:
                    heapq.heappush(frontier, (-(deepest[child] - depth[node]),
                                              tip[child], child, node))

    return BranchLengthOrder(blo, branch_length, cumulative)


@accepts_population(paired="vec")
def asym_tree(
    tree: Tree, vec: np.ndarray | None = None, van_pelt: bool = False
) -> np.ndarray:
    """Asymmetry ratio at each branch point (NaN elsewhere): the smaller of
    the two daughter subtrees' summed ``vec`` over the total (default
    ``vec``: terminal count). Requires strictly binary branch points --
    run ``repair_tree`` first if the tree may have trifurcations.

    With ``van_pelt=True``, uses Van Pelt's tree-asymmetry index instead:
    ``abs(v1 - v2) / (v1 + v2 - 2)``.
    """
    vec = T_tree(tree).astype(float) if vec is None else np.asarray(vec, dtype=float)
    dA = tree.dA.tocsc()
    branch_nodes = np.flatnonzero(B_tree(tree))

    asym = np.full(tree.n_nodes, np.nan)
    for bp in branch_nodes:
        children = np.flatnonzero(dA[:, bp].toarray().ravel())
        if len(children) != 2:
            raise ValueError(
                f"node {bp} has {len(children)} children; asym_tree requires "
                "strictly binary branch points (run repair_tree first)"
            )
        # with_tree=False: this runs once per branch point, and the
        # extracted Tree would be built and discarded every time
        v1 = vec[sub_tree(tree, int(children[0]), with_tree=False).mask].sum()
        v2 = vec[sub_tree(tree, int(children[1]), with_tree=False).mask].sum()
        if van_pelt:
            asym[bp] = 0.0 if v1 + v2 <= 2 else abs(v1 - v2) / (v1 + v2 - 2)
        else:
            asym[bp] = min(v1, v2) / (v1 + v2)
    return asym


class SectionResult(NamedTuple):
    """Result of :func:`dissect_tree` with ``with_positions=True``."""

    sections: np.ndarray
    """``(n_sections, 2)`` array of ``(start_node, end_node)``."""
    positions: np.ndarray
    """``(n_nodes, 2)``: each node's section index, and how far along
    that section it sits as a fraction in ``[0, 1]``."""


@accepts_population
@empty_safe("pairs", dtype=int)
def dissect_tree(tree: Tree, by_region: bool = True, *,
                 with_positions: bool = False):
    """Group nodes into sections delimited by branch points, termination
    points, and (optionally) region changes.

    Returns an ``(n_sections, 2)`` array of ``(start_node, end_node)`` pairs.
    Reimplemented as a direct per-cut-point ancestor walk rather than
    MATLAB's `ipar`/`cumsum` index trick (which the MATLAB docstring itself
    flags as "isn't completely correct yet at the root") -- this version
    handles the root the same way as every other cut point, no `root_tree`
    workaround needed. The MATLAB second output (per-node section index and
    relative position, used for NEURON `nseg` bookkeeping) isn't ported yet;
    add it if/when Phase 11 (T2N) needs it.

    A region-change cut is placed at the *parent* of the first node in the
    new region, not at that node itself: the transitioning node's own
    segment already belongs entirely to the new region, so it starts the
    new section rather than ending the old one (matching MATLAB's
    `iR = idpar(tree.R ~= tree.R(idpar))` -- indexing by the *parent*).
    Getting this backwards (marking the transition node itself, as an
    earlier version of this port did) silently produces an extra, spurious
    section split at every region boundary -- caught while building Phase
    11's NEURON bridge, which exercises region-based sectioning for the
    first time.

    The root is never treated as the *end* of a section, regardless of why
    it was cut (a region change at its own child, or the root genuinely
    being a branch point -- common in real reconstructions, e.g. a soma
    branching directly into several dendrites). There's nothing before the
    root to split off, so a section reaching it just extends all the way
    back via the ancestor walk's own stopping condition; treating the root
    as its own degenerate end-of-section produced a spurious
    root-to-itself entry, which callers building a new `dA` from `sect`
    (e.g. `resample_tree`) turned into an actual self-loop -- caught by
    testing against a real reconstruction whose root does branch, which
    none of the earlier hand-built test fixtures did.
    """
    idpar = idpar_tree(tree)  # self-referencing: idpar[root] == root
    root = _root_index(tree.dA)
    cut = B_tree(tree) | T_tree(tree)
    if by_region:
        non_root = idpar != np.arange(tree.n_nodes)
        changed = np.zeros(tree.n_nodes, dtype=bool)
        changed[non_root] = tree.R[non_root] != tree.R[idpar[non_root]]
        region_change = np.zeros(tree.n_nodes, dtype=bool)
        region_change[idpar[changed]] = True
        cut = cut | region_change

    starts, ends = [], []
    for end in np.flatnonzero(cut):
        if end == root:
            continue
        node = idpar[end]
        while node != root and not cut[node]:
            node = idpar[node]
        starts.append(node)
        ends.append(end)
    sections = np.array(list(zip(starts, ends)), dtype=int).reshape(-1, 2)
    if not with_positions:
        return sections

    # Per-node (section index, relative position along it). This is what
    # NEURON needs to place a synapse: "section 12, 30% of the way along".
    Plen = Pvec_tree(tree)
    positions = np.zeros((tree.n_nodes, 2), dtype=float)
    positions[:, 0] = -1
    for index, (start, end) in enumerate(sections.tolist()):
        span = Plen[end] - Plen[start]
        node = end
        # Walk back to but *excluding* the start node. A branch point is the
        # end of one section and the start of the next two, so claiming it
        # here would overwrite the position it already holds as the previous
        # section's endpoint -- and it belongs to that one, at fraction 1.0.
        while node != start:
            positions[node, 0] = index
            positions[node, 1] = (
                (Plen[node] - Plen[start]) / span if span > 0 else 0.0
            )
            node = idpar[node]

    # The root starts the tree rather than continuing any section, so it is
    # never anyone's endpoint; give it the first section at fraction 0.
    root = tree.root
    if positions[root, 0] < 0:
        positions[root] = (0.0, 0.0)
    return SectionResult(sections, positions)
