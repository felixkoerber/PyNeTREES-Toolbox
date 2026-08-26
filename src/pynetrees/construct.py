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
from typing import NamedTuple
from importlib import resources

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from ._compat import resolve_dim
from ._population import accepts_population
from ._empty import empty_safe
from .core import NO_PARENT, Tree
from .edit import delete_tree, insert_tree
from .graphtheory import (
    B_tree,
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


class MSTResult(NamedTuple):
    """Result of :func:`MST_tree` with ``full_output=True``."""

    trees: "Tree | list[Tree]"
    """The grown tree, or one per start point."""
    connected: np.ndarray
    """Boolean mask over the input points: did each end up in a tree?"""
    indx: np.ndarray
    """``(n_points, 2)``: for each input point, ``[tree index, node index
    within that tree]``, or ``[-1, -1]`` if it was never connected. MATLAB's
    second output."""
    history: np.ndarray | None
    """``(n_steps, 3)`` growth log ``[tree, point, parent point]`` in the
    order points were attached, or ``None`` unless ``record=True``."""


def MST_tree(
    X,
    Y,
    Z=None,
    start=0,
    bf: float = 0.4,
    thr: float = 50.0,
    mplen: float = 10000.0,
    avoid_multifurcations: bool = False,
    *,
    dist=None,
    cut_ends: bool = False,
    record: bool = False,
    full_output: bool = False,
):
    """Grow synthetic tree(s) connecting a cloud of points.

    At each step the cheapest available attachment is made, where connecting
    point ``p`` to tree node ``t`` costs::

        distance(p, t)  +  bf * path_length(t)  [+ dist penalty]

    balancing minimal total wiring against minimal conduction path length --
    the Cuntz/Borst/Segev construction the toolbox is named after.

    Parameters
    ----------
    X, Y, Z : array_like
        Coordinates of the points to connect. ``Z`` defaults to zeros.
    start : int, sequence of int, Tree, or list[Tree], default 0
        Where to grow from. An **index** picks one of the points in the
        cloud; **several indices** grow several trees at once, competing
        for the same cloud, so that territories fall out of the growth
        rather than being assigned -- this is how a population is grown
        into a shared field, and what MATLAB's multi-`msttrees` mode is
        for.

        A **Tree** (or a list of them) instead continues growing an
        existing morphology: every one of its nodes is a valid attachment
        point, its path lengths carry over into the balancing term, and its
        diameters and regions are preserved in the result. Nodes added by
        the growth are labelled with a region called ``"new"``, which is
        what :func:`~pynetrees.generate.clone_tree` renames per region.
    bf : float, default 0.4
        Balancing factor in ``[0, 1]``. ``0`` minimises wiring alone,
        giving long meandering paths to the root; ``1`` minimises path
        length, giving a star.
    thr : float, default 50.0
        Maximum span [um] of any single connection.
    mplen : float, default 10000.0
        Maximum path length [um] from the root; points beyond it stay
        unconnected.
    avoid_multifurcations : bool, default False
        MATLAB's ``'-b'``. Refuse a third child on any node. Some points may
        then stay unconnected even within ``thr``.
    dist : scipy.sparse matrix, optional
        MATLAB's ``DIST``: an ``(n_points, n_points)`` matrix of connection
        *preferences*, where larger means more likely and zero means "no
        particular reason to connect". Enters the cost as
        ``max(dist) * (1 - dist[t, p] / max(dist))``, so the most-preferred
        pairing pays nothing extra and an unlisted one pays the full range.

        Indexed over the **input points only** -- not over the nodes of a
        seed tree, which never need a preference because they are already
        connected. MATLAB instead requires the caller to index it over the
        growing trees' own nodes as well ("Don't forget to include input
        tree nodes into the distance matrix DIST!"), which is easy to get
        wrong and impossible to check.
    cut_ends : bool, default False
        MATLAB's ``'-c'``. Grow only from points that have at least one
        positive entry in ``dist`` -- the marked "cut ends". Requires
        ``dist``.
    record : bool, default False
        MATLAB's ``'-t'``. Also return the growth history.
    full_output : bool, default False
        Return :class:`MSTResult` rather than just the tree(s).

    Returns
    -------
    Tree or list[Tree] or MSTResult
        A single Tree for a single start point, a list for several.

    Notes
    -----
    Not a literal port: MATLAB's ~600-line version hand-maintains a
    shrinking "vicinity window" per tree, re-sorted and re-sliced every
    iteration, to avoid recomputing an O(n^2) distance matrix. This uses
    `scipy.spatial.cKDTree` for the radius queries and a lazy-deletion
    min-heap for "cheapest valid candidate", the standard formulation for
    Prim's-style growth where a node's best known cost only improves
    (Design Decision #27).

    ``record`` returns the growth **log**, not a list of intermediate trees
    as MATLAB does: any intermediate state is a prefix of the log, so
    storing whole trees per step would be quadratic in memory for
    information already present.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.zeros_like(X) if Z is None else np.asarray(Z, dtype=float)
    n_points = len(X)
    coords = np.column_stack([X, Y, Z])

    # growing onto existing trees: their nodes go in front of the cloud, so
    # that one index space covers both and the loop below needs no special
    # case for "is this an original node"
    seeds = _seed_trees(start)
    offsets = np.cumsum([0] + [t.n_nodes for t in seeds]) if seeds else np.array([0])
    n_seed = int(offsets[-1]) if seeds else 0
    if seeds:
        coords = np.vstack(
            [np.column_stack([t.X, t.Y, t.Z]) for t in seeds] + [coords]
        )
        X, Y, Z = coords[:, 0], coords[:, 1], coords[:, 2]
        starts = offsets[:-1].astype(int)
    else:
        starts = np.atleast_1d(np.asarray(start, dtype=int))
        if len(np.unique(starts)) != len(starts):
            raise ValueError(
                f"start points must be distinct, got {starts.tolist()}"
            )
        if starts.min() < 0 or starts.max() >= n_points:
            raise ValueError(f"start indices must lie in 0..{n_points - 1}")
    n = len(coords)
    kdtree = cKDTree(coords)

    dist_scale = 0.0
    if dist is not None:
        dist = sparse.csr_matrix(dist)
        if dist.shape != (n_points, n_points):
            raise ValueError(
                f"dist must be ({n_points}, {n_points}) -- indexed over the "
                f"input points; got {dist.shape}"
            )
        dist_scale = float(dist.max()) if dist.nnz else 0.0

    growable = None
    if cut_ends:
        if dist is None:
            raise ValueError("cut_ends=True needs dist= to mark the cut ends")
        growable = np.zeros(n, dtype=bool)
        growable[n_seed + np.unique(sparse.find(dist)[0])] = True
        growable[:n_seed] = True  # a seed tree is always grow-from-able
        growable[starts] = True

    connected = np.zeros(n, dtype=bool)
    owner = np.full(n, -1, dtype=int)
    parent = np.full(n, NO_PARENT, dtype=int)
    plen = np.zeros(n)
    children_count = np.zeros(n, dtype=int)
    order: list[list[int]] = [[s] for s in starts]
    history: list[tuple[int, int, int]] = []

    for tree_index, s in enumerate(starts):
        connected[s] = True
        owner[s] = tree_index

    for tree_index, seed in enumerate(seeds):
        block = np.arange(offsets[tree_index], offsets[tree_index + 1])
        connected[block] = True
        owner[block] = tree_index
        coo = seed.dA.tocoo()
        parent[block[0] + coo.row] = block[0] + coo.col
        # the balancing term needs metric path length, which the seed
        # already determines -- start it from there rather than from zero,
        # or the new material is grown as if the seed had no extent
        plen[block] = Pvec_tree(seed, len_tree(seed))
        children_count[block] = np.asarray(seed.dA.sum(axis=0)).ravel()
        order[tree_index] = block.tolist()

    best_cost = np.full(n, np.inf)
    heap: list[tuple[float, int, int, int]] = []

    def extra_cost(node: int, point: int) -> float:
        """The `dist` preference term, in the same units as distance."""
        if dist is None or dist_scale == 0.0 or node < n_seed:
            return 0.0
        return dist_scale * (
            1.0 - dist[node - n_seed, point - n_seed] / dist_scale
        )

    def push_from(node: int, tree_index: int) -> None:
        if avoid_multifurcations and children_count[node] >= 2:
            return
        if growable is not None and not growable[node]:
            return
        for p in kdtree.query_ball_point(coords[node], r=thr):
            if p == node or connected[p]:
                continue
            cost = (
                float(np.linalg.norm(coords[p] - coords[node]))
                + bf * plen[node]
                + extra_cost(node, p)
            )
            if cost < best_cost[p]:
                best_cost[p] = cost
                heapq.heappush(heap, (cost, p, node, tree_index))

    for tree_index, s in enumerate(starts):
        if seeds:
            # any node of a seed tree can take a new branch, not just its root
            for node in range(offsets[tree_index], offsets[tree_index + 1]):
                push_from(node, tree_index)
        else:
            push_from(int(s), tree_index)

    while heap:
        cost, p, attach, tree_index = heapq.heappop(heap)
        if connected[p] or cost > best_cost[p]:
            continue  # stale heap entry (lazy deletion)
        if avoid_multifurcations and children_count[attach] >= 2:
            continue
        span = float(np.linalg.norm(coords[p] - coords[attach]))
        new_plen = plen[attach] + span
        if new_plen > mplen:
            continue
        connected[p] = True
        owner[p] = tree_index
        parent[p] = attach
        plen[p] = new_plen
        children_count[attach] += 1
        order[tree_index].append(p)
        if record:
            history.append((tree_index, p, attach))
        push_from(p, tree_index)

    trees = []
    indx = np.full((n, 2), -1, dtype=int)
    for tree_index, member_order in enumerate(order):
        member_arr = np.array(member_order)
        old_to_new = {old: new for new, old in enumerate(member_arr)}
        indx[member_arr, 0] = tree_index
        indx[member_arr, 1] = np.arange(len(member_arr))

        rows, cols = [], []
        for new, old in enumerate(member_arr):
            if parent[old] == NO_PARENT:
                continue
            rows.append(new)
            cols.append(old_to_new[parent[old]])
        n_out = len(member_arr)
        dA = sparse.coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n_out, n_out)
        ).tocsr()
        D_out, R_out, rnames, name = _mst_attributes(
            seeds, tree_index, member_arr, offsets, n_out, len(starts)
        )
        trees.append(
            Tree(
                dA=dA,
                X=X[member_arr], Y=Y[member_arr], Z=Z[member_arr],
                D=D_out, R=R_out, rnames=rnames, name=name,
            )
        )

    result = trees[0] if len(starts) == 1 else trees
    if full_output:
        return MSTResult(
            result, connected[n_seed:], indx[n_seed:],
            np.array(history, dtype=int).reshape(-1, 3) if record else None,
        )
    return result


def _seed_trees(start) -> list[Tree]:
    """The seed trees in ``start``, or an empty list if it names indices."""
    if isinstance(start, Tree):
        return [start]
    if isinstance(start, (list, tuple)) and start and all(
        isinstance(t, Tree) for t in start
    ):
        return list(start)
    return []


#: Region name given to nodes a growth adds onto a seed tree.
MST_NEW_REGION = "new"


def _mst_attributes(seeds, tree_index, member_arr, offsets, n_out, n_trees):
    """Diameters, regions and name for one grown tree.

    Without a seed there is nothing to preserve, so everything is the
    default. With one, its nodes keep their own diameters and regions and
    the added nodes land in a region called ``"new"``.
    """
    if not seeds:
        return (
            np.ones(n_out), np.zeros(n_out, dtype=int), ["tree"],
            "MST" if n_trees == 1 else f"MST{tree_index}",
        )

    seed = seeds[tree_index]
    rnames = list(seed.rnames)
    if MST_NEW_REGION not in rnames:
        rnames.append(MST_NEW_REGION)
    new_region = rnames.index(MST_NEW_REGION)

    lo, hi = offsets[tree_index], offsets[tree_index + 1]
    from_seed = (member_arr >= lo) & (member_arr < hi)
    D_out = np.ones(n_out)
    R_out = np.full(n_out, new_region, dtype=int)
    local = member_arr[from_seed] - lo
    D_out[from_seed] = seed.D[local]
    R_out[from_seed] = np.asarray(seed.R)[local]
    return D_out, R_out, rnames, seed.name

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
        resorted = sort_tree(BCT_tree(seq), by="lo")
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
        resorted = sort_tree(BCT_tree(seq), by="lo")
        canon.append(np.asarray(resorted.dA.sum(axis=0)).ravel())
    bts = np.unique(np.array(canon), axis=0) if canon else np.empty((0, n))
    if not with_trees:
        return bts
    return bts, [BCT_tree(row) for row in bts]


# ---------------------------------------------------------------------------
# cleanup / shaping
# ---------------------------------------------------------------------------


@accepts_population
@empty_safe("tree")
def clean_tree(tree: Tree, radius: float = 1.0) -> Tree:
    """Delete improbable terminal branches: ones that end within
    ``D/2 + radius/2`` of a node on a *different* branch (likely a
    reconstruction/generation artifact), or whose total length is under
    ``radius``. At most one terminal branch is removed per branch point
    per call -- run repeatedly for further cleanup, as the MATLAB
    docstring also recommends.
    """
    tree = sort_tree(tree, by="lo")
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


def _overlap_factor(tree: Tree) -> np.ndarray:
    """Per-node ``sqrt(2) ** (branches passed)``, for soma surface correction.

    Two cylinders meeting at a branch point share membrane that neither
    NEURON nor this toolbox subtracts, so summed surface area over-counts
    at every bifurcation. Dividing each daughter's diameter by ``sqrt(2)``
    restores the total: two cylinders of diameter ``d/sqrt(2)`` have the
    same combined circumference-times-length as one of diameter ``d``.

    A branch directly at the root whose two daughters diverge by more than
    90 degrees is exempted: that is a soma sending off an axon in one
    direction and a dendrite in the other, not a dendrite splitting in two,
    so no membrane is being shared.
    """
    branch = B_tree(tree)
    # branch order: how many branch points lie on the path from the root
    passed = Pvec_tree(tree, branch.astype(float))
    # at a branch point itself the split has not happened yet
    passed[branch] -= 1

    root = tree.root
    children = np.flatnonzero(np.asarray(tree.dA.getcol(root).todense()).ravel())
    exempt = 0
    if len(children) >= 2:
        direction = direction_tree(tree, normalize=True)
        d0, d1 = direction[children[0]], direction[children[1]]
        angle = np.degrees(
            np.arctan2(np.linalg.norm(np.cross(d0, d1)), float(np.dot(d0, d1)))
        )
        if abs(angle) > 90.0:
            exempt = 1

    return np.sqrt(2.0) ** (passed - exempt)


@accepts_population
@empty_safe("tree")
def soma_tree(
    tree: Tree, maxD: float = 30.0, length: float | None = None,
    tag_region: bool = False, overlap_correction: bool = False,
) -> Tree:
    """Reshape diameter near the root into a cosine soma profile of
    (approximate) target diameter ``maxD`` and length ``length``
    (default ``1.5 * maxD``). If ``tag_region``, affected nodes are
    (re)labeled with a ``"soma"`` region.

    Parameters
    ----------
    tree : Tree
    maxD : float, default 30.0
        Peak soma diameter [um], reached at the root.
    length : float, optional
        Axial extent of the soma profile [um]; defaults to ``1.5 * maxD``.
        The cosine falls to zero at ``length / 2``, which is where the
        reshaping stops.
    tag_region : bool, default False
        Label the affected nodes with a ``"soma"`` region.
    overlap_correction : bool, default False
        MATLAB's ``'-b'``. Divide diameters by ``sqrt(2)`` for each branch
        point already passed, so that two cylinders meeting at a branch do
        not double-count the membrane they share. Neither NEURON nor this
        toolbox models overlapping surfaces, so without it the soma's
        surface area -- and hence its input conductance -- comes out too
        large wherever the soma spans a bifurcation.

        A branch straight off the root whose daughters diverge by more than
        90 degrees is treated as soma-to-axon plus soma-to-dendrite rather
        than a true bifurcation, and does not count.

    Returns
    -------
    Tree
    """
    if length is None:
        length = 1.5 * maxD
    Plen = Pvec_tree(tree)
    idx = np.flatnonzero(Plen < length / 2)

    D = tree.D.copy()
    profile = maxD * np.cos(np.pi * Plen[idx] / length)

    if overlap_correction:
        profile = profile / _overlap_factor(tree)[idx]

    D[idx] = np.maximum(D[idx], profile)

    R, rnames = tree.R.copy(), list(tree.rnames)
    if tag_region:
        if "soma" not in rnames:
            rnames.append("soma")
        R[idx] = rnames.index("soma")

    return Tree(
        dA=tree.dA, X=tree.X, Y=tree.Y, Z=tree.Z, D=D, R=R, rnames=rnames,
        name=tree.name, frustum=tree.frustum,
    )


@accepts_population
@empty_safe("tree")
def cap_tree(tree: Tree, spacing: float = 1.0) -> Tree:
    """Cap the tree's open root end with a rounded (hemispherical) profile.

    A flat-cut soma looks artificial and, more importantly, under-counts
    membrane area at the very place where input resistance is measured. This
    adds a short chain of tapering segments extending *backwards* from the
    root -- away from the tree -- whose diameters trace a spherical cap of
    the root's own diameter.

    Parameters
    ----------
    tree : Tree
    spacing : float, default 1.0
        Distance [um] between successive cap nodes.

    Returns
    -------
    Tree
        The tree with cap nodes appended, or the input unchanged if the root
        is too thin for even one cap node at this ``spacing``.

    Notes
    -----
    The cap grows from :attr:`Tree.root` along the *reverse* of that node's
    own segment direction. MATLAB's ``cap_tree.m`` hardcodes ``tree.X(1)``
    and ``direction(2, :)``, and this port transliterated both -- correct
    only after ``sort_tree``. On a tree whose root sits elsewhere it capped
    the wrong end entirely (Design Decision #48).

    MATLAB's ``'-a'`` axon-adding option is deliberately not ported here: it
    draws length, diameter and taper from constants fit to one published
    dataset, which makes it a dataset-specific generator rather than part of
    a capping algorithm, and folding it into this function makes it easy to
    apply by accident.
    """
    root = tree.root
    direction = direction_tree(tree, normalize=True)
    width = tree.D[root]
    X0, Y0, Z0 = tree.X[root], tree.Y[root], tree.Z[root]

    # The root's own `direction` entry is degenerate (it has no parent
    # segment), so the outward normal comes from its first child instead --
    # the cap must extend away from wherever the tree actually goes.
    children = np.flatnonzero(np.asarray(tree.dA.getcol(root).todense()).ravel())
    if len(children) == 0:
        return tree
    outward = direction[children[0]]

    new_X, new_Y, new_Z, new_D, new_parent = [], [], [], [], []
    parent = root
    for dist in np.arange(spacing, width, spacing):
        remaining = width**2 - 2 * dist**2
        d = float(np.sqrt(remaining)) if remaining > 0 else 0.0
        if d <= 0:
            continue
        new_X.append(X0 - dist * outward[0])
        new_Y.append(Y0 - dist * outward[1])
        new_Z.append(Z0 - dist * outward[2])
        new_D.append(d)
        new_parent.append(parent)
        # each cap node parents the next: `insert_tree` explicitly supports
        # new nodes parenting earlier new nodes, and validates the ordering
        parent = tree.n_nodes + len(new_X) - 1

    if not new_X:
        return tree
    root_region = np.full(len(new_X), tree.R[root], dtype=int)
    return insert_tree(
        tree, X=new_X, Y=new_Y, Z=new_Z, D=new_D, parent=new_parent, R=root_region
    )


# ---------------------------------------------------------------------------
# jittering / smoothing
# ---------------------------------------------------------------------------


def _gauss(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma * sigma))


@accepts_population
@empty_safe("tree")
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


def _smoothbranch(X, Y, Z, p: float, n: int):
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


@accepts_population
@empty_safe("tree")
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
    idpar = idpar_tree(tree, root_self=False)
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
        Xs, Ys, Zs = _smoothbranch(tree.X[path], tree.Y[path], tree.Z[path], p, n)
        X[path], Y[path], Z[path] = Xs, Ys, Zs

    return tree.with_coords(X=X, Y=Y, Z=Z)


# ---------------------------------------------------------------------------
# quadratic diameter tapering
# ---------------------------------------------------------------------------


def _load_quaddiameter_data():
    with resources.as_file(resources.files("pynetrees") / "data" / "quaddiameter.npz") as path:
        with np.load(path) as data:
            return data["P"], data["ldend"]


@accepts_population
@empty_safe("tree")
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


@accepts_population
def quadfit_tree(tree: Tree):
    """Fit a quadratic diameter taper (:func:`quaddiameter_tree`) to
    ``tree``'s existing diameters. Returns ``(scale, offset, fitted_tree)``."""

    def err(params):
        fitted = quaddiameter_tree(tree, scale=params[0], offset=params[1])
        return float(np.linalg.norm(tree.D - fitted.D))

    result = minimize(err, x0=np.random.rand(2), method="Nelder-Mead")
    scale, offset = result.x
    return scale, offset, quaddiameter_tree(tree, scale=scale, offset=offset)


# ---------------------------------------------------------------------------
# toy trees (V4)
# ---------------------------------------------------------------------------


def random_tree(n: int = 100, radius: float = 100.0, bf: float = 0.5,
                anisotropy: float = 1.0, *, dim: int = 3,
                shape: str = "sphere", avoid_multifurcations: bool = False,
                rng=None) -> Tree:
    """Grow a tree over randomly scattered points -- a toy cell.

    For sanity-checking an analysis, for a figure that needs a plausible
    arbor and nothing more, and as the null model a real morphology is
    compared against. Not a growth model: the points carry no biology, so
    what comes out shows what :func:`MST_tree`'s wiring rule alone produces.

    Parameters
    ----------
    n : int, default 100
        Number of nodes.
    radius : float, default 100
        Half-width of the region the points are drawn from [um].
    bf : float, default 0.5
        Balancing factor, passed to :func:`MST_tree`.
    anisotropy : float, default 1
        Stretch the cloud along Y and squeeze it along X by this factor,
        keeping the area roughly fixed. ``1`` is isotropic.
    dim : {2, 3}, keyword-only, default 3
        ``2`` flattens the cloud onto the XY plane.
    shape : {'sphere', 'box'}, keyword-only, default 'sphere'
        Draw from a ball of ``radius`` (MATLAB's ``'-sphere'``) or from the
        cube enclosing it.
    avoid_multifurcations : bool, keyword-only, default False
        Passed to :func:`MST_tree` (MATLAB's ``'-b'``).
    rng : numpy.random.Generator, optional
        For a reproducible cloud.

    Returns
    -------
    Tree
        Rooted at the origin, which is always one of the points.

    Notes
    -----
    MATLAB draws ``10 * n`` points, keeps those inside the sphere and takes
    the first ``n``. That is ~5.2n survivors for a ball inscribed in its
    cube, so it works, but it fails outright if the draw comes up short.
    This rejects and redraws until it has ``n``, which cannot.

    MATLAB's threshold and maximum path length are both 10000, i.e. no
    limit; the same is done here by passing them through to
    :func:`MST_tree` unchanged.
    """
    if shape not in ("sphere", "box"):
        raise ValueError(f"shape must be 'sphere' or 'box', got {shape!r}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    dim = resolve_dim(dim)
    rng = np.random.default_rng() if rng is None else rng

    points = np.empty((0, 3))
    while len(points) < n - 1:
        draw = rng.random((max(10 * n, 32), 3)) * 2 * radius - radius
        if dim == 2:
            draw[:, 2] = 0.0
        if shape == "sphere":
            draw = draw[np.linalg.norm(draw, axis=1) < radius]
        points = np.vstack([points, draw])

    # the root sits at the origin, as MATLAB's `X (1) = 0` arranges
    points = np.vstack([np.zeros((1, 3)), points[: n - 1]])
    points[:, 0] /= anisotropy
    points[:, 1] *= anisotropy

    return MST_tree(points[:, 0], points[:, 1], points[:, 2], start=0, bf=bf,
                    thr=10000.0, mplen=10000.0,
                    avoid_multifurcations=avoid_multifurcations)



# ---------------------------------------------------------------------------
# growth_tree (V4)
# ---------------------------------------------------------------------------


class Growth(NamedTuple):
    """Result of :func:`growth_tree`."""

    tree: Tree
    """The grown morphology."""
    length: np.ndarray
    """Total cable [um] after each step, starting from the seed."""
    terminals: np.ndarray
    """Number of termination points after each step."""
    attached_to: np.ndarray
    """Which existing node each step grew from (0-based; ``-1`` for the seed)."""
    target: np.ndarray
    """Which target point each step grew to (index into ``targets``)."""
    targets: np.ndarray
    """``(n, 3)`` the target cloud that was grown into."""
    history: list
    """The tree after every step, when ``history=True``; empty otherwise."""


#: Below this, ``sp`` is treated as zero and the space-filling term is
#: dropped entirely -- MATLAB's ``sig_limit``, which it uses both to clamp
#: the sigmoid and (after transforming) to choose the branch.
_SIG_LIMIT = 5e-4


def _transform_space_filling(sp: float) -> float:
    """MATLAB's sigmoid remap of ``sp``.

    A logit, rescaled so that the usable range of the raw parameter spreads
    across [0, 1]. Kept exactly, warts included: it is not symmetric, and
    ``sp = 0.5`` comes out as 0.475 rather than 0.5.
    """
    sp = min(max(float(sp), _SIG_LIMIT), 1.0 - _SIG_LIMIT)
    scale = np.log((1.0 - _SIG_LIMIT) / _SIG_LIMIT)
    return float(np.clip(np.log(sp / (1.0 - sp)) / scale / 1.75 + 0.475, 0.0, 1.0))


def _correlated_jitter(n: int, pairs, rng) -> np.ndarray:
    """Smooth random wander along a new stretch of cable.

    Each ``(stde, lam)`` pair contributes Gaussian noise smoothed over
    ``lam`` nodes, so the path meanders rather than shaking. Anchored at
    zero so the first new node stays attached to the tree.
    """
    from scipy import ndimage

    offsets = np.zeros((n, 3))
    for stde, lam in pairs:
        lam = int(lam)
        noise = rng.normal(size=(n, 3)) * stde * lam
        kernel = _gauss(np.arange(-lam, lam + 1), 0.0, lam / 5.0)
        # `ndimage`, not `np.convolve(..., "same")`: numpy returns
        # `max(len(signal), len(kernel))` rows, so a stretch shorter than
        # the smoothing window comes back the wrong length instead of
        # simply being smoothed less. Zero padding matches MATLAB's `convn`.
        smoothed = ndimage.convolve1d(noise, kernel, axis=0, mode="constant",
                                      cval=0.0)
        offsets += smoothed - smoothed[0]
    return offsets


def growth_tree(span, start=None, thr: float = 100, bf: float = 0.5,
                sp: float = 0.5, k: float = 0.0, *, alpha: float = 0.5,
                jitter=(), max_radius: float = 10_000.0,
                n_target_points: int = 100_000, stop: str = "steps",
                history: bool = False, rng=None) -> Growth:
    """Grow a tree into a volume, balancing wiring cost against filling space.

    :func:`MST_tree` connects a *given* cloud of points as cheaply as it
    can. This grows into a *region*: at each step it picks the next target
    by trading off three things -- the cost of reaching it (``bf``, as in
    `MST_tree`), how far it lies from the arbor so far (``sp``, which
    pushes growth into empty territory rather than thickening what is
    already there), and chance (``k``). The result is a cell that fills its
    domain rather than one that merely spans a sample of it.

    Parameters
    ----------
    span : Tree, array_like, or Boundary
        The territory to grow into. A **Tree** uses the alpha shape around
        its nodes, so "grow a new cell shaped like this one" is one call. An
        ``(n, 3)`` **array** is taken as the target points themselves
        (MATLAB's ``-P``). A :class:`~pynetrees.Boundary` is used directly.
    start : Tree or array_like, optional
        An existing morphology to continue growing, or a root coordinate.
        Defaults to the origin -- or to ``span``'s root if ``span`` is a
        tree.
    thr : float, default 100
        When to stop; read according to ``stop``.
    bf : float, default 0.5
        Balancing factor, exactly as in :func:`MST_tree`: how much the path
        length back to the root counts against the cost of a connection.
    sp : float in [0, 1], default 0.5
        Space-filling factor. ``0`` reduces this to an ordinary minimum
        spanning tree over the targets reached so far; higher values prefer
        targets far from any existing cable.
    k : float in [0, 1], default 0
        Stochasticity. ``0`` is deterministic given the target cloud.
    alpha : float, keyword-only, default 0.5
        Shrink factor for the boundary, when one is derived from a tree.
    jitter : sequence of (stde, lam) pairs, keyword-only
        Make new stretches of cable wander instead of running dead straight.
        Each pair is one scale of wobble, as in :func:`jitter_tree`.
    max_radius : float, keyword-only, default 10000
        Longest connection allowed [um]; also sets how far ahead of the
        growing tip target points are considered.
    n_target_points : int, keyword-only, default 100000
        How many points to scatter through the volume. MATLAB's default is
        1e6 drawn in the *bounding box*, most of which it discards; these
        all land inside.
    stop : {'steps', 'length', 'terminals'}, keyword-only, default 'steps'
        Whether ``thr`` counts growth steps, total cable [um] (MATLAB's
        ``-L``), or termination points (``-T``).
    history : bool, keyword-only, default False
        Keep the tree after every step. MATLAB always does, which for a
        long growth means hundreds of copies of a growing morphology.
    rng : numpy.random.Generator, optional

    Returns
    -------
    Growth
        ``(tree, length, terminals, attached_to, target, targets, history)``.

    Notes
    -----
    New cable is laid down at roughly **one node per micron**, so a grown
    tree is finely sampled by construction and does not need resampling
    before analysis.

    ``sp`` and ``k`` are passed through MATLAB's transforms unchanged -- a
    logit remap and ``tanh (4 k)`` respectively -- so numbers carry over
    from MATLAB scripts. One consequence worth knowing: ``sp = 0.5`` is not
    the midpoint of the effect, it maps to 0.475, and the space-filling
    term switches off entirely only below ``sp ~ 0.0018``.

    **Sampling is exact rather than by rejection**, as in
    :func:`pynetrees.theta_mc_tree`. MATLAB's ``mc3d`` fills the bounding cube
    with ``rand`` and keeps what lands inside the boundary, which for a flat
    or elongated domain throws away most of every draw.

    The nearest-node distances that drive selection are computed with a
    k-d tree rather than a full pairwise distance matrix per step. MATLAB
    recomputes ``pdist2`` between every tree node and every point in the
    vicinity on every iteration, which is what makes its version slow on a
    large domain.
    """
    from scipy.spatial import cKDTree

    from .density import _alpha_shape, _sample_in_simplices
    from .metrics import len_tree

    if stop not in ("steps", "length", "terminals"):
        raise ValueError(
            f"stop must be 'steps', 'length' or 'terminals', got {stop!r}"
        )
    generator = np.random.default_rng(rng)

    # --- the territory ----------------------------------------------------
    seed_tree = span if isinstance(span, Tree) else None
    if seed_tree is not None:
        coords = np.column_stack([span.X, span.Y, span.Z])
        shape = _alpha_shape(coords, alpha)
        targets = _sample_in_simplices(shape.points, shape.simplices,
                                       n_target_points, generator)
    elif hasattr(span, "simplices") and hasattr(span, "points"):
        targets = _sample_in_simplices(span.points, span.simplices,
                                       n_target_points, generator)
    else:
        targets = np.atleast_2d(np.asarray(span, dtype=float))
        if targets.shape[1] == 2:
            targets = np.column_stack([targets, np.zeros(len(targets))])
    if len(targets) == 0:
        raise ValueError("growth_tree: no target points to grow into")

    # --- the seed ---------------------------------------------------------
    if isinstance(start, Tree):
        X = list(start.X)
        Y = list(start.Y)
        Z = list(start.Z)
        parents = idpar_tree(start, root_self=False).tolist()
        plen = list(Pvec_tree(start, len_tree(start)))
        rnames = list(start.rnames)
        regions = list(start.R)
        diameters = list(start.D)
        # New cable gets its own region, matching `MST_tree`'s convention
        # when it continues an existing morphology -- so "what did this
        # growth add" survives into the result rather than being merged
        # into whatever the seed's regions were.
        if "new" not in rnames:
            rnames.append("new")
        new_region = rnames.index("new")
    else:
        if start is None:
            root = ([span.X[0], span.Y[0], span.Z[0]] if seed_tree is not None
                    else [0.0, 0.0, 0.0])
        else:
            root = list(np.asarray(start, dtype=float).ravel())
            if len(root) == 2:
                root = root + [0.0]
        X, Y, Z = [float(root[0])], [float(root[1])], [float(root[2])]
        parents = [NO_PARENT]
        plen = [0.0]
        rnames = ["tree"]
        regions = [0]  # 0-based into `rnames` in this port, unlike MATLAB
        diameters = [1.0]
        new_region = 0

    # --- parameter transforms --------------------------------------------
    sp = _transform_space_filling(sp)
    k = float(np.tanh(k * 4.0))

    # --- the open set -----------------------------------------------------
    # Only targets within `radius` of the root are in play; the radius grows
    # as the tree reaches further out, which is what keeps the cost of each
    # step bounded on a large domain.
    root_point = np.array([X[0], Y[0], Z[0]])
    root_distance = np.linalg.norm(targets - root_point, axis=1)
    order = np.argsort(root_distance, kind="stable")
    sorted_distance = root_distance[order]
    radius = max_radius

    to_tree, nearest = cKDTree(np.column_stack([X, Y, Z])).query(targets)

    in_play = int(np.searchsorted(sorted_distance, radius, side="left"))
    open_idx = list(order[:in_play])
    cost = list(np.where(to_tree[open_idx] > max_radius, np.inf,
                         to_tree[open_idx])
                + bf * np.asarray(plen)[nearest[open_idx]])
    attach = [int(i) for i in nearest[open_idx]]
    # Distance from each open point to the nearest node, kept separately
    # from `cost`: the two have different minima, since `cost` is weighted
    # by path length. Both are running minima over a node set that only
    # grows, so both update in place -- MATLAB recomputes the whole
    # `pdist2` every iteration for the identical answer.
    spread = list(to_tree[open_idx])

    def build() -> Tree:
        n = len(X)
        rows = [i for i in range(n) if parents[i] != NO_PARENT]
        cols = [parents[i] for i in rows]
        dA = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)),
                               shape=(n, n)).tocsr()
        return Tree(dA=dA, X=np.array(X), Y=np.array(Y), Z=np.array(Z),
                    D=np.array(diameters, dtype=float),
                    R=np.array(regions, dtype=int), rnames=rnames,
                    name="grown")

    seed = build()
    # Tracked incrementally rather than recomputed. MATLAB rebuilds the
    # tree and calls `len_tree`/`T_tree` on every step, which is O(n) work
    # per step and so O(n^2) over a growth; the additions are local and
    # exactly known, so there is nothing to recompute.
    total_length = float(len_tree(seed).sum())
    child_count = [0] * len(X)
    for node, parent_of in enumerate(parents):
        if parent_of != NO_PARENT:
            child_count[parent_of] += 1
    n_terminals = float(T_tree(seed).sum())

    lengths = [total_length]
    terminals = [n_terminals]
    attached_log = [-1]
    target_log = [-1]
    snapshots = []

    # --- grow -------------------------------------------------------------
    steps = 0
    while open_idx:
        costs = np.asarray(cost, dtype=float)
        if not np.isfinite(costs).any():
            break
        scale = costs[np.isfinite(costs)].max() or 1.0

        if sp > _SIG_LIMIT:
            # how far each candidate lies from *any* existing cable: the
            # term that pushes growth outward instead of thickening
            reaches = np.asarray(spread, dtype=float)
            spread_scale = reaches.max() or 1.0
            score = (1 - k) * ((1 - sp) * costs / scale
                               - sp * reaches / spread_scale)
        else:
            score = (1 - k) * costs / scale
        if k:
            score = score + k * generator.random(len(costs))
        score = np.where(np.isfinite(costs), score, np.inf)

        chosen = int(np.argmin(score))
        point = targets[open_idx[chosen]]
        parent = int(attach[chosen])

        origin = np.array([X[parent], Y[parent], Z[parent]])
        reach = float(np.linalg.norm(point - origin))
        if reach > max_radius:
            break
        radius = max(radius, max_radius + root_distance[open_idx[chosen]])

        # lay down roughly one node per micron along the new stretch
        n_new = max(int(round(reach)), 1)
        step = (point - origin) / reach if reach > 0 else np.zeros(3)
        path = origin + np.outer(np.arange(1, n_new + 1), step)
        if jitter:
            path = path + _correlated_jitter(n_new, jitter, generator)

        first_new = len(X)
        previous = origin
        running = plen[parent]
        # a branch point is created only if the parent already had a child;
        # otherwise the parent simply stops being the tip and the new node
        # becomes one, leaving the terminal count unchanged
        if child_count[parent] > 0:
            n_terminals += 1
        child_count[parent] += 1

        for offset, node in enumerate(path):
            X.append(float(node[0]))
            Y.append(float(node[1]))
            Z.append(float(node[2]))
            diameters.append(1.0)
            regions.append(new_region)
            parents.append(parent if offset == 0 else first_new + offset - 1)
            child_count.append(0)
            if offset > 0:
                child_count[first_new + offset - 1] = 1
            segment = float(np.linalg.norm(node - previous))
            running += segment
            total_length += segment
            plen.append(running)
            previous = node

        steps += 1
        lengths.append(total_length)
        terminals.append(n_terminals)
        attached_log.append(parent)
        target_log.append(int(open_idx[chosen]))
        if history:
            snapshots.append(build())

        if stop == "length" and lengths[-1] >= thr:
            break
        if stop == "terminals" and terminals[-1] >= thr:
            break
        if stop == "steps" and steps >= thr:
            break

        # --- update the open set -----------------------------------------
        del open_idx[chosen]
        del cost[chosen]
        del attach[chosen]
        del spread[chosen]

        if open_idx:
            fresh = np.arange(first_new, len(X))
            fresh_points = np.column_stack([X, Y, Z])[fresh]
            distance, which = cKDTree(fresh_points).query(targets[open_idx])
            closer = distance < np.asarray(spread)
            for position in np.flatnonzero(closer):
                spread[position] = float(distance[position])
            distance = np.where(distance > max_radius, np.inf, distance)
            candidate = distance + bf * np.asarray(plen)[fresh[which]]
            for position in np.flatnonzero(candidate < np.asarray(cost)):
                cost[position] = float(candidate[position])
                attach[position] = int(fresh[which[position]])

        wider = int(np.searchsorted(sorted_distance, radius, side="left"))
        if wider > in_play:
            arriving = order[in_play:wider]
            raw, which = cKDTree(np.column_stack([X, Y, Z])).query(
                targets[arriving])
            distance = np.where(raw > max_radius, np.inf, raw)
            open_idx.extend(arriving.tolist())
            cost.extend((distance + bf * np.asarray(plen)[which]).tolist())
            attach.extend(int(i) for i in which)
            spread.extend(raw.tolist())
            in_play = wider

    return Growth(tree=build(), length=np.array(lengths),
                  terminals=np.array(terminals),
                  attached_to=np.array(attached_log),
                  target=np.array(target_log), targets=targets,
                  history=snapshots)
