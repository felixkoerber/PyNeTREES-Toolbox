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

import numpy as np
from scipy import sparse

from .core import NO_PARENT, Tree

# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


def _root_index(dA: sparse.spmatrix) -> int:
    """0-based index of the tree's root (the node with no parent).

    Detected via row-sum rather than "index 0", since 0 is a valid node
    index here (unlike MATLAB) and can't double as a sentinel.
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


def typeN_tree(tree: Tree) -> np.ndarray:
    """Node type per node: 0 terminal, 1 continuation, 2 (or more) branch."""
    typeN = _children_count(tree.dA)
    typeN = np.minimum(typeN, 2)
    return typeN.astype(int)


def B_tree(tree: Tree) -> np.ndarray:
    """Boolean mask of branch points (more than one child)."""
    return _children_count(tree.dA) > 1


def C_tree(tree: Tree) -> np.ndarray:
    """Boolean mask of continuation points (exactly one child)."""
    return _children_count(tree.dA) == 1


def T_tree(tree: Tree) -> np.ndarray:
    """Boolean mask of termination points (no children)."""
    return _children_count(tree.dA) == 0


# ---------------------------------------------------------------------------
# parent / child indices
# ---------------------------------------------------------------------------


def idpar_tree(tree: Tree, no_self: bool = False) -> np.ndarray:
    """0-based index of each node's direct parent.

    By default (``no_self=False``, matching MATLAB's default), the root is
    its own parent -- a convenience many downstream computations rely on.
    With ``no_self=True`` (MATLAB's ``'-z'``), the root instead gets
    :data:`NO_PARENT` (``-1``).
    """
    dA = tree.dA
    N = tree.n_nodes
    idpar = np.asarray(dA @ np.arange(N)).ravel().astype(int)
    is_root = np.asarray(dA.sum(axis=1)).ravel() == 0
    if no_self:
        idpar[is_root] = NO_PARENT
    else:
        idpar[is_root] = np.flatnonzero(is_root)
    return idpar


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
    idpar = idpar_tree(tree, no_self=True)

    PL = np.zeros(tree.n_nodes, dtype=float)
    for node in order:
        parent = idpar[node]
        if parent != NO_PARENT:
            PL[node] = PL[parent] + 1.0
    return PL


def ipar_tree(tree: Tree) -> np.ndarray:
    """Path to root: ``ipar[i] = [i, parent(i), grandparent(i), ..., root]``,
    :data:`NO_PARENT`-padded to the depth of the deepest node.
    """
    N = tree.n_nodes
    idpar_noself = idpar_tree(tree, no_self=True)
    max_depth = int(PL_tree(tree).max()) if N > 1 else 0

    ipar = np.full((N, max_depth + 2), NO_PARENT, dtype=int)
    current = np.arange(N)
    for col in range(max_depth + 2):
        ipar[:, col] = current
        valid = current != NO_PARENT
        nxt = np.full(N, NO_PARENT, dtype=int)
        nxt[valid] = idpar_noself[current[valid]]
        current = nxt
    return ipar


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


def Pvec_tree(tree: Tree, v: np.ndarray) -> np.ndarray:
    """Cumulative sum of ``v`` along the path from the root to each node
    (inclusive of the node itself).

    Pass any per-node quantity: with ``len_tree`` you get metric path
    length, with ``ones`` you get topological depth + 1, and so on.

    Computed by the recurrence ``P[node] = P[parent] + v[node]`` in
    pre-order, which is O(n_nodes). The previous version summed a prebuilt
    ``ipar_tree`` matrix instead -- correct, but that matrix is
    ``n_nodes x max_depth`` (49 MB, 6.1M entries for a real granule cell),
    so it was the worst-scaling function in the toolbox at 3.4x superlinear.
    """
    v = np.asarray(v, dtype=float)
    order = _dfs_preorder(tree.dA)
    idpar = idpar_tree(tree, no_self=True)

    out = np.zeros(tree.n_nodes, dtype=float)
    for node in order:
        parent = idpar[node]
        out[node] = v[node] + (out[parent] if parent != NO_PARENT else 0.0)
    return out


def ratio_tree(tree: Tree, v: np.ndarray | None = None) -> np.ndarray:
    """Ratio of ``v`` at each node to ``v`` at its parent (root: 1.0)."""
    v = tree.D if v is None else np.asarray(v, dtype=float)
    idpar = idpar_tree(tree)  # self-referencing: root's ratio is v/v == 1
    return v / v[idpar]


# ---------------------------------------------------------------------------
# regions, subtrees, rerooting, sorting
# ---------------------------------------------------------------------------


def rindex_tree(tree: Tree) -> np.ndarray:
    """0-based rank of each node within its own region, by node order."""
    R = tree.R
    rindex = np.zeros(len(R), dtype=int)
    for region in np.unique(R):
        mask = R == region
        rindex[mask] = np.arange(int(mask.sum()))
    return rindex


def sub_tree(tree: Tree, inode: int) -> np.ndarray:
    """Boolean mask selecting ``inode`` and all of its descendants.

    Walks the child lists directly. An earlier version read each node's
    children as ``dA[:, node].toarray()``, which materialises a dense
    length-``n_nodes`` column *per visited node* and makes a single BFS
    O(n_nodes^2) -- 514 ms on a 3765-node granule cell, against ~1 ms here.
    That mattered beyond this function, since `asym_tree`, `repair_tree` and
    `clean_tree` all call it in a loop.
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
    return mask


def redirect_tree(tree: Tree, new_root: int, name: str | None = None):
    """Reroot the tree at ``new_root``, reversing edge direction as needed
    along the path from the old root. Returns ``(new_tree, order)`` where
    ``order[i]`` is the old index now at new position ``i``.

    Only makes topological sense when ``new_root`` is not itself a branch
    point (rerooting there would leave it a trifurcation) -- matching the
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
    return new_tree, order_arr


def sort_tree(tree: Tree, by: str = "hier"):
    """Reindex nodes to be BCT-conform (parent always precedes its children,
    each subtree contiguous). Returns ``(new_tree, order)``.

    ``by``:
        ``'hier'`` (default) -- keep nodes in their existing relative order,
            only fixing up parent/child adjacency (many isomorphic BCT
            orders satisfy this; this one is arbitrary but cheap).
        ``'lo'`` -- order by (topological path length, level order) first,
            giving a near-canonical ordering (MATLAB's ``'-LO'``).
        ``'lex'`` -- order by number of children, terminals before
            continuations before branches (MATLAB's ``'-LEX'``).
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
    return tree.reindexed(order), order


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
        v1 = vec[sub_tree(tree, int(children[0]))].sum()
        v2 = vec[sub_tree(tree, int(children[1]))].sum()
        if van_pelt:
            asym[bp] = 0.0 if v1 + v2 <= 2 else abs(v1 - v2) / (v1 + v2 - 2)
        else:
            asym[bp] = min(v1, v2) / (v1 + v2)
    return asym


def dissect_tree(tree: Tree, by_region: bool = True) -> np.ndarray:
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
    return np.array(list(zip(starts, ends)), dtype=int).reshape(-1, 2)
