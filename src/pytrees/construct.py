"""Constructing synthetic trees.

Ports (part of) treestoolbox-master/construct/*.m. `MST_tree` is the
flagship: a greedy, path-length-balanced minimum-spanning-tree grower
(Cuntz, Borst & Segev 2007 / Cuntz et al. 2010 -- the algorithm the whole
toolbox is named after). MATLAB's version is a heavily hand-optimized
single function (~600 lines) that manually maintains a shrinking "vicinity
window" of candidate points per tree, sorted and re-sliced on every
iteration, to avoid an O(n^2) distance recomputation. This port gets the
same practical performance using two standard tools instead:
`scipy.spatial.cKDTree` for "which unclaimed points are within `thr` of
this node" queries, and a lazy-deletion min-heap for "what's the cheapest
still-valid candidate right now" -- the standard technique for Prim's-
algorithm-style growth where a node's best known cost can only improve as
the tree grows. See PORT_STATUS.md Design Decision for what this drops
(multiple simultaneous competing trees, the `DIST` cost-matrix term,
grow-from-cut-ends-only mode, time-lapse recording) and why.

Also **deliberately not ported**: `clone_tree` and `gscale_tree` (a
population-statistics-driven generative pipeline coupling MST_tree,
`rpoints_tree`, `gdens_tree`, and heavy per-region statistical resampling --
very high complexity, tightly coupled to a specific published dataset's
region-naming conventions, and low standalone value outside that pipeline);
`rpoints_tree`/`PP_generator_tree`/`in_c`/`cpoints`/`cplotter` (contour
point-generation utilities that only exist to feed that pipeline);
`fix_tree`/`finetune_fix_tree` (the todo list already flags these as
incomplete in MATLAB itself); `spines_tree` (a specialized add-on, low
priority relative to the core generative functions); `dscam_tree`
(research-specific, niche).
"""

from __future__ import annotations

import heapq
from importlib import resources

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .core import NO_PARENT, Tree
from .edit import delete_tree, insert_tree
from .graphtheory import (
    Pvec_tree,
    T_tree,
    _subtree_blocks,
    child_tree,
    dissect_tree,
    idpar_tree,
    ipar_tree,
    sort_tree,
)
from .metrics import direction_tree, eucl_tree, len_tree

# ---------------------------------------------------------------------------
# MST_tree: greedy path-length-balanced minimum spanning tree
# ---------------------------------------------------------------------------


def MST_tree(
    X,
    Y,
    Z=None,
    start: int = 0,
    bf: float = 0.4,
    thr: float = 50.0,
    mplen: float = 10000.0,
    avoid_multifurcations: bool = False,
):
    """Grow a synthetic tree connecting the given points.

    Greedily attaches the cheapest reachable point to the growing tree at
    each step, where the cost of attaching point ``p`` to tree node ``t``
    is ``distance(p, t) + bf * path_length(t)`` -- balancing minimal total
    wiring (distance) against minimal conduction path length (`bf`, 0..1).
    Points farther than ``thr`` from every tree node, or that would exceed
    ``mplen`` total path length, are never connected and are dropped from
    the output. ``avoid_multifurcations``, if set, refuses to attach a
    third child to any node (MATLAB's ``'-b'``); some points may then be
    left unconnected even if within range, if their only recorded
    candidate attachment saturates before they're processed -- see
    PORT_STATUS.md.

    Returns ``(tree, connected)``, where ``connected`` is a boolean mask
    (length ``len(X)``) of which input points ended up in the tree.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.zeros_like(X) if Z is None else np.asarray(Z, dtype=float)
    n = len(X)
    coords = np.column_stack([X, Y, Z])
    kdtree = cKDTree(coords)

    connected = np.zeros(n, dtype=bool)
    connected[start] = True
    parent = np.full(n, NO_PARENT, dtype=int)
    plen = np.zeros(n)
    children_count = np.zeros(n, dtype=int)
    order = [start]

    best_cost = np.full(n, np.inf)
    heap: list[tuple[float, int, int]] = []

    def push_from(node: int) -> None:
        if avoid_multifurcations and children_count[node] >= 2:
            return
        for p in kdtree.query_ball_point(coords[node], r=thr):
            if p == node or connected[p]:
                continue
            cost = float(np.linalg.norm(coords[p] - coords[node])) + bf * plen[node]
            if cost < best_cost[p]:
                best_cost[p] = cost
                heapq.heappush(heap, (cost, p, node))

    push_from(start)
    while heap:
        cost, p, attach = heapq.heappop(heap)
        if connected[p] or cost > best_cost[p]:
            continue  # stale heap entry (lazy deletion)
        if avoid_multifurcations and children_count[attach] >= 2:
            continue
        d = float(np.linalg.norm(coords[p] - coords[attach]))
        new_plen = plen[attach] + d
        if new_plen > mplen:
            continue
        connected[p] = True
        parent[p] = attach
        plen[p] = new_plen
        children_count[attach] += 1
        order.append(p)
        push_from(p)

    order_arr = np.array(order)
    old_to_new = {old: new for new, old in enumerate(order_arr)}
    rows, cols = [], []
    for new, old in enumerate(order_arr):
        if old == start:
            continue
        rows.append(new)
        cols.append(old_to_new[parent[old]])
    n_out = len(order_arr)
    dA = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_out, n_out)).tocsr()

    tree = Tree(
        dA=dA,
        X=X[order_arr], Y=Y[order_arr], Z=Z[order_arr],
        D=np.ones(n_out), R=np.zeros(n_out, dtype=int), rnames=["tree"],
        name="MST",
    )
    return tree, connected


# ---------------------------------------------------------------------------
# BCT strings: topology-only tree construction/validation/enumeration
# ---------------------------------------------------------------------------


def isBCT_tree(bct_or_tree) -> bool:
    """Check whether a B/C/T-type children-count sequence (2=branch,
    1=continuation, 0=terminal -- or a Tree, whose column sums are used)
    describes a single valid rooted tree."""
    bct = (
        np.asarray(bct_or_tree.dA.sum(axis=0)).ravel()
        if isinstance(bct_or_tree, Tree)
        else np.asarray(bct_or_tree, dtype=float)
    )
    if bct.size == 0:
        return False
    c = np.cumsum(bct - 1) + 1
    zero_crossings = np.flatnonzero(c == 0)
    return len(zero_crossings) > 0 and zero_crossings[0] == len(bct) - 1


def BCT_tree(bct) -> Tree:
    """Build a (topology-only) Tree from a B/C/T children-count sequence.

    Coordinates are all zero -- this constructs pure topology, useful for
    testing and enumerating isomorphism classes (:func:`allBCTs_tree`,
    :func:`allBTs_tree`). MATLAB's version optionally attaches a fake
    dendrogram layout via `xdend_tree`; that's a Phase 7 (graphical)
    concern and isn't ported here.
    """
    bct = np.asarray(bct)
    if not isBCT_tree(bct):
        raise ValueError("input sequence is not BCT-conform")
    n = len(bct)
    stack = [0]
    rows, cols = [], []
    prev = NO_PARENT
    for i in range(n):
        if prev != NO_PARENT:
            rows.append(i)
            cols.append(prev)
        prev = i
        if bct[i] == 0:
            prev = stack.pop()
        elif bct[i] == 2:
            stack.append(i)
    dA = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    return Tree(
        dA=dA, X=np.zeros(n), Y=np.zeros(n), Z=np.zeros(n),
        D=np.ones(n), R=np.zeros(n, dtype=int), rnames=["tree"], name="BCT",
    )


def _all_conform_sequences(n: int, digits: tuple[int, ...]):
    base = len(digits)
    for counter in range(base**n):
        seq = [digits[(counter // base ** (n - 1 - i)) % base] for i in range(n)]
        if isBCT_tree(seq):
            yield seq


def allBCTs_tree(n: int = 8, with_trees: bool = False):
    """All non-isomorphic B/C/T topologies with ``n`` nodes.

    Brute-force over all ``3**n`` sequences -- "gets very slow very
    quickly" per the MATLAB docstring; the small default matches that.
    """
    canon = []
    for seq in _all_conform_sequences(n, (0, 1, 2)):
        resorted, _ = sort_tree(BCT_tree(seq), by="lo")
        canon.append(np.asarray(resorted.dA.sum(axis=0)).ravel())
    bcts = np.unique(np.array(canon), axis=0) if canon else np.empty((0, n))
    if not with_trees:
        return bcts
    return bcts, [BCT_tree(row) for row in bcts]


def allBTs_tree(n: int = 15, with_trees: bool = False):
    """All non-isomorphic binary (branch/terminal only, no continuation)
    topologies with ``n`` nodes. Only achievable for select (odd) ``n``,
    by the definition of a full binary tree."""
    canon = []
    for seq in _all_conform_sequences(n, (0, 2)):
        resorted, _ = sort_tree(BCT_tree(seq), by="lo")
        canon.append(np.asarray(resorted.dA.sum(axis=0)).ravel())
    bts = np.unique(np.array(canon), axis=0) if canon else np.empty((0, n))
    if not with_trees:
        return bts
    return bts, [BCT_tree(row) for row in bts]


# ---------------------------------------------------------------------------
# cleanup / shaping
# ---------------------------------------------------------------------------


def clean_tree(tree: Tree, radius: float = 1.0) -> Tree:
    """Delete improbable terminal branches: ones that end within
    ``D/2 + radius/2`` of a node on a *different* branch (likely a
    reconstruction/generation artifact), or whose total length is under
    ``radius``. At most one terminal branch is removed per branch point
    per call -- run repeatedly for further cleanup, as the MATLAB
    docstring also recommends.
    """
    tree, _ = sort_tree(tree, by="lo")
    D = tree.D
    length = len_tree(tree)
    typeN = np.asarray(tree.dA.sum(axis=0)).ravel()
    idpar = idpar_tree(tree)

    to_delete: set[int] = set()
    resolved_branch_points: set[int] = set()
    for t in np.flatnonzero(typeN == 0):
        if t == 0:
            continue  # degenerate single-node tree: nothing to trim
        # the root (index 0) is never part of a deletable branch run, even
        # when it's itself typeN==1 (a single-child "stalk" root) -- so the
        # fallback boundary is right after the root, not the root itself
        non_continuation = np.flatnonzero(typeN[:t] != 1)
        branch_start = int(non_continuation[-1]) + 1 if non_continuation.size else 1
        branch = np.arange(branch_start, t + 1)
        branch_parent = int(idpar[branch[0]])
        if branch_parent in resolved_branch_points:
            continue

        close_by = np.flatnonzero(eucl_tree(tree, point=int(t)) < (D / 2 + radius / 2))
        overlaps_other_branch = np.setdiff1d(close_by, branch).size > 0
        too_short = length[branch].sum() < radius

        if overlaps_other_branch or too_short:
            resolved_branch_points.add(branch_parent)
            to_delete.update(branch.tolist())

    if not to_delete:
        return tree
    result = delete_tree(tree, sorted(to_delete), keep_regions=True)
    if isinstance(result, list):
        raise ValueError("clean_tree: deletion disconnected the tree")
    return result


def soma_tree(
    tree: Tree, maxD: float = 30.0, length: float | None = None, tag_region: bool = False
) -> Tree:
    """Reshape diameter near the root into a cosine soma profile of
    (approximate) target diameter ``maxD`` and length ``length``
    (default ``1.5 * maxD``). If ``tag_region``, affected nodes are
    (re)labeled with a ``"soma"`` region.

    MATLAB's ``'-b'`` overlap-correction option (reduce diameter past a
    branch point near the soma by sqrt(2), to compensate for double-
    counted surface area) is not ported -- a subtle, physiologically-
    motivated correction, not core to "add a soma".
    """
    if length is None:
        length = 1.5 * maxD
    Plen = Pvec_tree(tree, len_tree(tree))
    idx = np.flatnonzero(Plen < length / 2)

    D = tree.D.copy()
    D[idx] = np.maximum(D[idx], maxD * np.cos(np.pi * Plen[idx] / length))

    R, rnames = tree.R.copy(), list(tree.rnames)
    if tag_region:
        if "soma" not in rnames:
            rnames.append("soma")
        R[idx] = rnames.index("soma")

    return Tree(
        dA=tree.dA, X=tree.X, Y=tree.Y, Z=tree.Z, D=D, R=R, rnames=rnames,
        name=tree.name, frustum=tree.frustum,
    )


def cap_tree(tree: Tree, spacing: float = 1.0) -> Tree:
    """Cap the open end at the root with small segments approximating a
    rounded (spherical) cap, ``spacing`` um apart, so the soma doesn't end
    in a flat, artificial-looking cut.

    MATLAB's ``'-a'`` axon-adding option is not ported -- it draws its
    length/diameter/taper from hardcoded statistical parameters specific
    to one published dataset, not a general capping algorithm.
    """
    direction = direction_tree(tree, normalize=True)
    width = tree.D[0]
    X0, Y0, Z0 = tree.X[0], tree.Y[0], tree.Z[0]

    new_X, new_Y, new_Z, new_D, new_parent = [], [], [], [], []
    parent = 0
    for dist in np.arange(spacing, width, spacing):
        remaining = width**2 - 2 * dist**2
        d = float(np.sqrt(remaining)) if remaining > 0 else 0.0
        if d <= 0:
            continue
        new_X.append(X0 - dist * direction[1, 0])
        new_Y.append(Y0 - dist * direction[1, 1])
        new_Z.append(Z0 - dist * direction[1, 2])
        new_D.append(d)
        new_parent.append(parent)
        parent = tree.n_nodes + len(new_X) - 1

    if not new_X:
        return tree
    # R is passed explicitly (not left to insert_tree's tree.R[parent]
    # default) since `new_parent` chains through newly-inserted nodes,
    # which don't have an entry in the original tree.R yet.
    root_region = np.full(len(new_X), tree.R[0], dtype=int)
    return insert_tree(
        tree, X=new_X, Y=new_Y, Z=new_Z, D=new_D, parent=new_parent, R=root_region
    )


# ---------------------------------------------------------------------------
# jittering / smoothing
# ---------------------------------------------------------------------------


def _gauss(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma * sigma))


def jitter_tree(tree: Tree, stde: float = 1.0, lam: int = 10, ipart=None, rng=None) -> Tree:
    """Add spatially-correlated noise to node coordinates: each node's
    displacement is a Gaussian-weighted (kernel centered at 1 hop, width
    ``lam / 5``) blend of independent per-node noise, over nodes within
    ``lam`` topological hops.

    Reimplemented via per-node BFS over the undirected tree graph instead
    of MATLAB's precomputed dense adjacency-matrix powers (``A^k`` for
    ``k`` up to ``lam``) -- same result, and asymptotically cheaper for
    large trees with modest ``lam`` (BFS touches O(lam) nodes per source
    instead of a full matrix multiply). One behavioral difference: a
    node's distance to *itself* is computed as a true BFS distance (0),
    not MATLAB's value of 2 (an artifact of detecting self-reachability
    via "walk of length k" parity on a matrix power rather than shortest
    path) -- a deliberate correctness fix, see PORT_STATUS.md.
    """
    N = tree.n_nodes
    rng = np.random.default_rng() if rng is None else rng
    if ipart is None:
        ipart = np.arange(N)
    else:
        ipart = np.asarray(ipart)
        if ipart.dtype == bool:
            ipart = np.flatnonzero(ipart)

    undirected = (tree.dA + tree.dA.T).tocsr()
    R = np.zeros((N, 3))
    R[ipart] = rng.normal(size=(len(ipart), 3)) * stde * lam

    sigma = lam / 5
    R1 = np.zeros((N, 3))
    for node in range(N):
        dist = np.full(N, np.inf)
        dist[node] = 0
        frontier = [node]
        for hop in range(1, lam + 1):
            nxt = []
            for u in frontier:
                for v in undirected.indices[undirected.indptr[u] : undirected.indptr[u + 1]]:
                    if np.isinf(dist[v]):
                        dist[v] = hop
                        nxt.append(v)
            if not nxt:
                break
            frontier = nxt
        weight = np.where(np.isinf(dist), 0.0, _gauss(dist, 1.0, sigma))
        R1[node] = (R * weight[:, None]).sum(axis=0)

    return tree.with_coords(
        X=tree.X + R1[:, 0] - R1[0, 0],
        Y=tree.Y + R1[:, 1] - R1[0, 1],
        Z=tree.Z + R1[:, 2] - R1[0, 2],
    )


def smoothbranch(X, Y, Z, p: float, n: int):
    """Smooth a single consecutive path of 3D points (endpoints fixed),
    ``n`` iterations of pulling each interior point ``p`` (0..1) toward
    its projection onto the line between its neighbors."""
    X, Y, Z = np.asarray(X, dtype=float), np.asarray(Y, dtype=float), np.asarray(Z, dtype=float)
    if len(X) <= 2:
        return X, Y, Z
    for _ in range(n):
        X1, Y1, Z1 = X[:-2], Y[:-2], Z[:-2]
        X2, Y2, Z2 = X[1:-1], Y[1:-1], Z[1:-1]
        X3, Y3, Z3 = X[2:], Y[2:], Z[2:]
        denom = (X3 - X1) ** 2 + (Y3 - Y1) ** 2 + (Z3 - Z1) ** 2
        u = ((X2 - X1) * (X3 - X1) + (Y2 - Y1) * (Y3 - Y1) + (Z2 - Z1) * (Z3 - Z1)) / denom
        Xu, Yu, Zu = X1 + u * (X3 - X1), Y1 + u * (Y3 - Y1), Z1 + u * (Z3 - Z1)
        Xs = np.concatenate([[X[0]], X2 + p * (Xu - X2), [X[-1]]])
        Ys = np.concatenate([[Y[0]], Y2 + p * (Yu - Y2), [Y[-1]]])
        Zs = np.concatenate([[Z[0]], Z2 + p * (Zu - Z2), [Z[-1]]])
        X, Y, Z = Xs, Ys, Zs
    return X, Y, Z


def smooth_tree(tree: Tree, pwchild: float = 0.5, p: float = 0.9, n: int = 5) -> Tree:
    """Smooth a tree along its longest paths (see :func:`smoothbranch`).
    First merges dissected sections along "heavy" branches (where one
    child subtree carries more than ``pwchild`` of the descendant weight)
    into longer paths, so smoothing happens along natural long branches
    rather than independently on every short inter-branch-point segment.
    """
    sect = dissect_tree(tree).tolist()
    ipar = ipar_tree(tree)
    blocks = _subtree_blocks(tree.dA)
    idpar = idpar_tree(tree, no_self=True)
    nchild = child_tree(tree)

    counter = 0
    while counter < len(sect):
        start, end = sect[counter]
        dchildren = np.flatnonzero(idpar == end)
        if dchildren.size == 0:
            counter += 1
            continue
        continuing = [i for i, s in enumerate(sect) if i != counter and s[0] == end]
        if not continuing:
            counter += 1
            continue

        wchild = nchild[dchildren]
        if wchild.sum() == 0:  # all children are themselves leaves: no dominant one
            counter += 1
            continue
        rwchild = wchild / wchild.sum()
        if not np.any(rwchild > pwchild):
            counter += 1
            continue

        heavy_child = int(dchildren[np.argmax(rwchild)])
        # O(subtree) lookup rather than rescanning the whole ipar matrix --
        # this runs once per section, so the naive version was quadratic
        _order, _start, _size = blocks
        heavy_descendants = set(
            _order[_start[heavy_child] : _start[heavy_child] + _size[heavy_child]].tolist()
        )
        merge_idx = next(
            (i for i in continuing if sect[i][1] == heavy_child or sect[i][1] in heavy_descendants),
            None,
        )
        if merge_idx is None:
            counter += 1
            continue
        sect[counter][1] = sect[merge_idx][1]
        del sect[merge_idx]
        # re-examine the now-extended section (don't advance counter)

    X, Y, Z = tree.X.copy(), tree.Y.copy(), tree.Z.copy()
    for start, end in sect:
        row = ipar[end]
        path = row[: int(np.flatnonzero(row == start)[0]) + 1][::-1]
        Xs, Ys, Zs = smoothbranch(tree.X[path], tree.Y[path], tree.Z[path], p, n)
        X[path], Y[path], Z[path] = Xs, Ys, Zs

    return tree.with_coords(X=X, Y=Y, Z=Z)


# ---------------------------------------------------------------------------
# quadratic diameter tapering
# ---------------------------------------------------------------------------


def _load_quaddiameter_data():
    with resources.as_file(resources.files("pytrees") / "data" / "quaddiameter.npz") as path:
        with np.load(path) as data:
            return data["P"], data["ldend"]


def quaddiameter_tree(tree: Tree, scale: float = 0.5, offset: float = 0.5) -> Tree:
    """Apply a quadratic diameter taper (Cuntz, Borst & Segev 2007) along
    every root-to-terminal path, using the bundled best-fit parameters for
    optimal current transfer; nodes shared by multiple paths get the mean
    of each path's diameter at that point.
    """
    P, ldend = _load_quaddiameter_data()
    N = tree.n_nodes
    Plen = Pvec_tree(tree, len_tree(tree))
    ipar = ipar_tree(tree)

    D_sum = np.zeros(N)
    D_count = np.zeros(N)
    for t in np.flatnonzero(T_tree(tree)):
        row = ipar[t]
        path = row[row != NO_PARENT][::-1]
        pathlen = Plen[path]
        i2 = int(np.argmin((pathlen[-1] - ldend) ** 2))
        D_sum[path] += np.polyval(P[i2], pathlen) * scale
        D_count[path] += 1

    D = np.where(D_count > 0, D_sum / np.maximum(D_count, 1), 0.5) + offset
    return tree.with_coords(D=D)


def quadfit_tree(tree: Tree):
    """Fit a quadratic diameter taper (:func:`quaddiameter_tree`) to
    ``tree``'s existing diameters. Returns ``(scale, offset, fitted_tree)``."""

    def err(params):
        fitted = quaddiameter_tree(tree, scale=params[0], offset=params[1])
        return float(np.linalg.norm(tree.D - fitted.D))

    result = minimize(err, x0=np.random.rand(2), method="Nelder-Mead")
    scale, offset = result.x
    return scale, offset, quaddiameter_tree(tree, scale=scale, offset=offset)
