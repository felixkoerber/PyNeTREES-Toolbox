"""Structural tree editing: delete/insert nodes, repair, resample, concatenate.

Ports treestoolbox-master/edit/*.m. This is the phase that finally lands
`repair_tree`, deferred all the way back from Phase 1 (`load_swc` builds a
raw, possibly non-BCT-conform tree; `repair_tree` is what fixes that up).

Several functions here are deliberately *not* a literal translation -- see
PORT_STATUS.md's Design Decisions for the reasoning in each case:
`delete_tree` always splices deleted nodes' children to the nearest
surviving ancestor and always returns a list when that disconnects the tree
(MATLAB's version has an inconsistent, documented-as-broken version of this,
see the todo list: "delete_tree | multiple trees doesn't work yet"), and
`resample_tree` preserves branch/termination points exactly rather than
snapping them onto the resampling grid (the original docstring itself calls
that snapping rule "arbitrary", not a mathematical necessity).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .core import NO_PARENT, Tree
from .graphtheory import (
    C_tree,
    Pvec_tree,
    T_tree,
    dissect_tree,
    idpar_tree,
    ipar_tree,
    redirect_tree,
    sort_tree,
    sub_tree,
)
from .metrics import cyl_tree, direction_tree, eucl_tree, len_tree, morph_tree, tran_tree


# ---------------------------------------------------------------------------
# node deletion
# ---------------------------------------------------------------------------


def delete_tree(tree: Tree, inodes, keep_regions: bool = False) -> Tree | list[Tree]:
    """Delete nodes from a tree, splicing each deleted node's children to
    its nearest surviving ancestor so the remaining tree(s) stay connected.

    ``inodes`` is a boolean mask (length ``n_nodes``) or a list/array of
    node indices. If the deletion disconnects the tree (e.g. deleting a
    branching root), returns a **list** of Trees, one per resulting
    component, instead of a single Tree -- unlike MATLAB, whose forest-
    splitting only kicks in for a specific option combination and is
    documented as broken for the default case (todo list: "delete_tree |
    multiple trees doesn't work yet").
    """
    N = tree.n_nodes
    inodes = np.atleast_1d(np.asarray(inodes))
    if inodes.dtype == bool:
        delete_mask = inodes.copy()
    else:
        delete_mask = np.zeros(N, dtype=bool)
        delete_mask[inodes.astype(int)] = True

    if delete_mask.all():
        raise ValueError("delete_tree: cannot delete every node in a tree")

    ipar = ipar_tree(tree)
    keep = ~delete_mask
    keep_idx = np.flatnonzero(keep)

    new_parent = np.full(N, NO_PARENT, dtype=int)
    for node in keep_idx:
        for anc in ipar[node, 1:]:
            if anc == NO_PARENT:
                break
            if keep[anc]:
                new_parent[node] = anc
                break

    old_to_new = {old: new for new, old in enumerate(keep_idx)}
    n_new = len(keep_idx)
    rows, cols = [], []
    for node in keep_idx:
        p = new_parent[node]
        if p != NO_PARENT:
            rows.append(old_to_new[node])
            cols.append(old_to_new[p])
    dA = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_new, n_new)
    ).tocsr()

    R = tree.R[keep_idx]
    rnames = tree.rnames
    if not keep_regions:
        uniq, R = np.unique(R, return_inverse=True)
        rnames = [tree.rnames[i] for i in uniq]

    result = Tree(
        dA=dA,
        X=tree.X[keep_idx],
        Y=tree.Y[keep_idx],
        Z=tree.Z[keep_idx],
        D=tree.D[keep_idx],
        R=R,
        rnames=rnames,
        name=tree.name,
        frustum=tree.frustum,
    )

    roots = np.flatnonzero(np.asarray(dA.sum(axis=1)).ravel() == 0)
    if len(roots) <= 1:
        return result
    return [result.reindexed(np.flatnonzero(sub_tree(result, int(r)))) for r in roots]


# ---------------------------------------------------------------------------
# repair pipeline
# ---------------------------------------------------------------------------


def elim0_tree(tree: Tree, keep_regions: bool = False) -> Tree:
    """Delete zero-length segments (except the root's own trivial one)."""
    zero_idx = np.flatnonzero(len_tree(tree) == 0)
    if len(zero_idx) > 1:
        result = delete_tree(tree, zero_idx[1:], keep_regions=keep_regions)
        if isinstance(result, list):
            raise ValueError("elim0_tree: unexpectedly disconnected the tree")
        return result
    return tree


def elimt_tree(tree: Tree, no_root: bool = False):
    """Replace every multifurcation (3+ children) with a short chain of
    tiny bifurcations, each offset by ~0.0001 um. Returns ``(tree, changed)``.
    """
    dA = tree.dA.tocsr()
    N = tree.n_nodes
    idpar = idpar_tree(tree)
    is_root = np.asarray(dA.sum(axis=1)).ravel() == 0
    children_count = np.asarray(dA.sum(axis=0)).ravel()

    multif = np.flatnonzero(children_count > 2)
    if no_root:
        multif = multif[~is_root[multif]]
    if len(multif) == 0:
        return tree, False

    coo = dA.tocoo()
    edges = list(zip(coo.row.tolist(), coo.col.tolist()))
    X, Y, Z, D, R = (
        tree.X.tolist(), tree.Y.tolist(), tree.Z.tolist(), tree.D.tolist(), tree.R.tolist(),
    )
    n = N

    for bp in multif.tolist():
        children = sorted(c for c, p in edges if p == bp)
        n_spacers = len(children) - 2

        dX = tree.X[bp] - tree.X[idpar[bp]]
        dY = tree.Y[bp] - tree.Y[idpar[bp]]
        dZ = tree.Z[bp] - tree.Z[idpar[bp]]
        if dX == 0 and dY == 0 and dZ == 0:
            dX = np.mean(tree.X) - tree.X[0]
            dY = np.mean(tree.Y) - tree.Y[0]
            dZ = np.mean(tree.Z) - tree.Z[0]
        norm = np.sqrt(dX**2 + dY**2 + dZ**2)
        dX, dY, dZ = dX / norm, dY / norm, dZ / norm

        edges = [(c, p) for c, p in edges if not (p == bp and c in children)]

        prev = bp
        for i in range(n_spacers):
            spacer = n
            n += 1
            X.append(tree.X[bp] + 0.0001 * dX * (i + 1))
            Y.append(tree.Y[bp] + 0.0001 * dY * (i + 1))
            Z.append(tree.Z[bp] + 0.0001 * dZ * (i + 1))
            D.append(D[bp])
            R.append(R[bp])
            edges.append((children[i], prev))
            edges.append((spacer, prev))
            prev = spacer
        edges.append((children[n_spacers], prev))
        edges.append((children[n_spacers + 1], prev))

    rows = [c for c, _ in edges]
    cols = [p for _, p in edges]
    dA_new = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    new_tree = Tree(
        dA=dA_new, X=np.array(X), Y=np.array(Y), Z=np.array(Z), D=np.array(D),
        R=np.array(R), rnames=tree.rnames, name=tree.name, frustum=tree.frustum,
    )
    return new_tree, True


def repair_tree(tree: Tree, no_root_trifurcation: bool = False) -> Tree:
    """Rectify a tree to full BCT conformity: eliminate multifurcations,
    drop zero-length segments, and sort into canonical (level-order) index
    order. Most other functions in this toolbox assume their input has
    already been through this.
    """
    tree, _ = elimt_tree(tree, no_root=no_root_trifurcation)
    tree, _ = elimt_tree(tree)
    tree = elim0_tree(tree)

    if tree.n_nodes > 1 and T_tree(tree)[0]:
        dA = tree.dA.tolil()
        dA[1, 0] = 1
        tree = Tree(
            dA=dA, X=tree.X, Y=tree.Y, Z=tree.Z, D=tree.D, R=tree.R,
            rnames=tree.rnames, name=tree.name, frustum=tree.frustum,
        )

    tree, _ = sort_tree(tree, by="lo")
    return tree


# ---------------------------------------------------------------------------
# adding / moving nodes
# ---------------------------------------------------------------------------


def root_tree(tree: Tree) -> Tree:
    """Prepend a near-zero-length segment at the root (some downstream
    algorithms rely on the root having exactly one child)."""
    N = tree.n_nodes
    dA = sparse.lil_matrix((N + 1, N + 1))
    dA[1:, 1:] = tree.dA
    dA[1, 0] = 1

    def prepend(v, first_delta=0.0):
        return np.concatenate([[v[0] + first_delta], v])

    return Tree(
        dA=dA.tocsr(),
        X=prepend(tree.X, -0.0001),
        Y=prepend(tree.Y),
        Z=prepend(tree.Z),
        D=prepend(tree.D),
        R=prepend(tree.R),
        rnames=tree.rnames,
        name=tree.name,
        frustum=tree.frustum,
    )


def insert_tree(tree: Tree, X, Y, Z, D, parent, R=None) -> Tree:
    """Append new leaf nodes to a tree.

    Each new point ``i`` becomes a child of the *existing* node
    ``parent[i]`` (0-based). Regions default to their parent's region.
    Replaces MATLAB's ``[inode R X Y Z D idpar]`` SWC-tuple calling
    convention with explicit arrays.
    """
    X, Y, Z, D = (np.asarray(a, dtype=float) for a in (X, Y, Z, D))
    parent = np.asarray(parent, dtype=int)
    n_new = len(X)
    N = tree.n_nodes
    R = tree.R[parent] if R is None else np.asarray(R, dtype=int)

    coo = tree.dA.tocoo()
    rows = np.concatenate([coo.row, np.arange(N, N + n_new)])
    cols = np.concatenate([coo.col, parent])
    dA = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(N + n_new, N + n_new)
    ).tocsr()

    return Tree(
        dA=dA,
        X=np.concatenate([tree.X, X]),
        Y=np.concatenate([tree.Y, Y]),
        Z=np.concatenate([tree.Z, Z]),
        D=np.concatenate([tree.D, D]),
        R=np.concatenate([tree.R, R]),
        rnames=tree.rnames,
        name=tree.name,
        frustum=tree.frustum,
    )


def insertp_tree(tree: Tree, inode: int | None = None, plens=None):
    """Insert nodes at path lengths ``plens`` [um] on the path from the
    root to ``inode`` (default: every 10 um, or halfway if the path is
    shorter than 10 um). Returns ``(new_tree, added_mask)``.
    """
    N = tree.n_nodes
    Plen = Pvec_tree(tree, len_tree(tree))
    if inode is None:
        inode = N - 1

    if plens is None:
        plens = (
            np.arange(0.0, Plen[inode] + 1e-9, 10.0)
            if Plen[inode] > 10
            else np.array([Plen[inode] / 2])
        )
    else:
        plens = np.asarray(plens, dtype=float)

    ipar = ipar_tree(tree)
    row = ipar[inode]
    path = row[row != NO_PARENT][::-1].tolist()  # root ... inode
    plen_path = Plen[path].tolist()

    plens = np.setdiff1d(plens, plen_path)
    plens = np.sort(plens[plens < max(plen_path)])
    if len(plens) == 0:
        return tree, np.zeros(N, dtype=bool)

    X, Y, Z, D, R = tree.X.tolist(), tree.Y.tolist(), tree.Z.tolist(), tree.D.tolist(), tree.R.tolist()
    edges_to_remove, edges_to_add = [], []
    n = N
    for pl in plens.tolist():
        pos = max(i for i in range(len(plen_path)) if plen_path[i] < pl)
        p_node, c_node = path[pos], path[pos + 1]
        rpos = (pl - plen_path[pos]) / (plen_path[pos + 1] - plen_path[pos])
        new_node = n
        n += 1
        X.append(tree.X[p_node] + (tree.X[c_node] - tree.X[p_node]) * rpos)
        Y.append(tree.Y[p_node] + (tree.Y[c_node] - tree.Y[p_node]) * rpos)
        Z.append(tree.Z[p_node] + (tree.Z[c_node] - tree.Z[p_node]) * rpos)
        D.append(tree.D[p_node] + (tree.D[c_node] - tree.D[p_node]) * rpos)
        R.append(tree.R[c_node])
        edges_to_remove.append((c_node, p_node))
        edges_to_add.append((new_node, p_node))
        edges_to_add.append((c_node, new_node))
        path.insert(pos + 1, new_node)
        plen_path.insert(pos + 1, pl)

    coo = tree.dA.tocoo()
    rows, cols = coo.row.tolist(), coo.col.tolist()
    for c, p in edges_to_remove:
        i = next(i for i in range(len(rows)) if rows[i] == c and cols[i] == p)
        del rows[i]
        del cols[i]
    for c, p in edges_to_add:
        rows.append(c)
        cols.append(p)

    dA = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    new_tree = Tree(
        dA=dA, X=np.array(X), Y=np.array(Y), Z=np.array(Z), D=np.array(D),
        R=np.array(R), rnames=tree.rnames, name=tree.name, frustum=tree.frustum,
    )
    sorted_tree, order = sort_tree(new_tree, by="lo")
    return sorted_tree, order >= N


def interpd_tree(tree: Tree, node1: int, node2: int) -> Tree:
    """Linearly interpolate diameter between two nodes on the same root path."""
    Plen = Pvec_tree(tree, len_tree(tree))
    ipar = ipar_tree(tree)
    row1, row2 = ipar[node1], ipar[node2]

    if node2 in row1:
        path = row1[: int(np.flatnonzero(row1 == node2)[0]) + 1]
    elif node1 in row2:
        path = row2[: int(np.flatnonzero(row2 == node1)[0]) + 1]
    else:
        raise ValueError(f"nodes {node1} and {node2} do not lie on the same root path")

    m = (tree.D[path[-1]] - tree.D[path[0]]) / (Plen[path[-1]] - Plen[path[0]])
    D = tree.D.copy()
    D[path] = m * (Plen[path] - Plen[path[0]]) + tree.D[path[0]]
    return tree.with_coords(D=D)


def recon_tree(tree: Tree, ichilds, ipars, shift: bool = True) -> Tree:
    """Reconnect the subtrees rooted at ``ichilds`` to new parents ``ipars``.

    If ``shift`` (default), each subtree is translated so its root lands on
    its new parent's position.
    """
    ichilds = np.atleast_1d(np.asarray(ichilds, dtype=int))
    ipars = np.atleast_1d(np.asarray(ipars, dtype=int))
    X, Y, Z = tree.X.copy(), tree.Y.copy(), tree.Z.copy()

    if shift:
        for child, parent in zip(ichilds, ipars):
            mask = sub_tree(tree, int(child))
            dX, dY, dZ = X[child] - X[parent], Y[child] - Y[parent], Z[child] - Z[parent]
            X[mask] -= dX
            Y[mask] -= dY
            Z[mask] -= dZ

    idpar = idpar_tree(tree)
    dA = tree.dA.tolil()
    for child, parent in zip(ichilds, ipars):
        dA[child, idpar[child]] = 0
        dA[child, parent] = 1

    return Tree(
        dA=dA.tocsr(), X=X, Y=Y, Z=Z, D=tree.D, R=tree.R,
        rnames=tree.rnames, name=tree.name, frustum=tree.frustum,
    )


def restrain_tree(
    tree: Tree, maxpl: float = 400.0, interpolate: bool = True
) -> Tree:
    """Prune a tree so no node exceeds path length ``maxpl`` [um] from the
    root. If ``interpolate`` (default), terminal points beyond ``maxpl``
    are pulled back to exactly ``maxpl`` along their original direction
    rather than simply deleted.
    """
    Plen = Pvec_tree(tree, len_tree(tree))
    if not np.any(Plen > maxpl):
        return tree

    if not interpolate:
        result = delete_tree(tree, Plen > maxpl, keep_regions=True)
        if isinstance(result, list):
            raise ValueError("restrain_tree: pruning disconnected the tree")
        return result

    idpar = idpar_tree(tree)
    beyond = (Plen > maxpl) & (Plen[idpar] > maxpl)
    tree = delete_tree(tree, beyond, keep_regions=True)
    if isinstance(tree, list):
        raise ValueError("restrain_tree: pruning disconnected the tree")

    idpar = idpar_tree(tree)
    Plen = Pvec_tree(tree, len_tree(tree))
    over = Plen > maxpl
    direction = direction_tree(tree, normalize=True)
    X, Y, Z = tree.X.copy(), tree.Y.copy(), tree.Z.copy()
    reach = maxpl - Plen[idpar[over]]
    X[over] = X[idpar[over]] + direction[over, 0] * reach
    Y[over] = Y[idpar[over]] + direction[over, 1] * reach
    Z[over] = Z[idpar[over]] + direction[over, 2] * reach
    return tree.with_coords(X=X, Y=Y, Z=Z)


def cat_tree(
    tree1: Tree,
    tree2: Tree,
    inode1: int | None = None,
    inode2: int = 0,
    keep_regions: bool = False,
) -> Tree:
    """Concatenate ``tree2`` onto ``tree1``, connecting ``tree2``'s
    ``inode2`` (default: its root) to ``tree1``'s ``inode1`` (default:
    whichever node in ``tree1`` is closest to ``tree2``'s ``inode2``).
    """
    if inode1 is None:
        point = np.array([tree2.X[inode2], tree2.Y[inode2], tree2.Z[inode2]])
        inode1 = int(np.argmin(eucl_tree(tree1, point)))

    tree2, _ = redirect_tree(tree2, inode2)
    N1 = tree1.n_nodes
    dA = sparse.block_diag([tree1.dA, tree2.dA], format="lil")
    dA[N1, inode1] = 1

    X = np.concatenate([tree1.X, tree2.X])
    Y = np.concatenate([tree1.Y, tree2.Y])
    Z = np.concatenate([tree1.Z, tree2.Z])
    D = np.concatenate([tree1.D, tree2.D])

    if keep_regions:
        R = np.concatenate([tree1.R, tree2.R + len(tree1.rnames)])
        rnames = tree1.rnames + tree2.rnames
    else:
        names1 = np.asarray(tree1.rnames)
        names2 = np.asarray(tree2.rnames)
        combined_names = np.concatenate([names1[tree1.R], names2[tree2.R]])
        uniq, R = np.unique(combined_names, return_inverse=True)
        rnames = uniq.tolist()

    merged = Tree(
        dA=dA.tocsr(), X=X, Y=Y, Z=Z, D=D, R=R, rnames=rnames,
        name=tree1.name, frustum=tree1.frustum,
    )
    return sort_tree(merged, by="lo")[0]


def resample_tree(tree: Tree, sr: float = 10.0, extend_terminals: bool = True) -> Tree:
    """Redistribute nodes along the tree at approximately ``sr`` [um] spacing.

    Works section-by-section (see :func:`~pytrees.dissect_tree`): branch and
    termination points are preserved exactly at their original positions,
    and every original node strictly between two such anchors is replaced
    by new nodes at multiples of ``sr`` path length, interpolated along the
    original polyline through that section. This is a deliberately simpler
    convention than MATLAB's version, which also relocates branch/
    termination points onto the resampling grid via a delete-and-splice
    pass -- see PORT_STATUS.md Design Decisions (the original docstring
    itself calls that snapping rule "arbitrary", not a mathematical
    necessity, so this port picked the anchor-preserving alternative).

    If ``extend_terminals`` (default), each terminal segment is first
    stretched by ``sr / 2`` (via :func:`morph_tree`) to reduce truncation
    bias at branch tips, matching MATLAB's default behavior.

    A tree with a single node has no segments to resample and is returned
    unchanged. (Without this guard the section-based rebuild below produces
    a zero-node tree, which then fails deep inside `sort_tree` with a
    confusing "expected exactly one root, found 0".)
    """
    if tree.n_nodes < 2:
        return tree

    if extend_terminals:
        length = len_tree(tree)
        target = length.copy()
        term = T_tree(tree)
        target[term] = length[term] + 0.5 * sr
        tree = morph_tree(tree, target)

    idpar = idpar_tree(tree)
    Plen = Pvec_tree(tree, len_tree(tree))
    sections = dissect_tree(tree, by_region=True)

    anchors = sorted(set(sections[:, 0].tolist()) | set(sections[:, 1].tolist()))
    old_to_new = {old: new for new, old in enumerate(anchors)}
    X = [tree.X[old] for old in anchors]
    Y = [tree.Y[old] for old in anchors]
    Z = [tree.Z[old] for old in anchors]
    D = [tree.D[old] for old in anchors]
    R = [tree.R[old] for old in anchors]

    rows, cols = [], []
    n = len(anchors)
    for start, end in sections.tolist():
        # original polyline of nodes from `start` to `end` (via parent chain)
        chain = [end]
        node = end
        while node != start:
            node = idpar[node]
            chain.append(node)
        chain.reverse()
        chain_len = [Plen[c] for c in chain]

        prev_new = old_to_new[start]
        p0, p1 = chain_len[0], chain_len[-1]
        if p1 <= p0:
            rows.append(old_to_new[end])
            cols.append(prev_new)
            continue

        first_k = int(np.floor(p0 / sr)) + 1
        last_k = int(np.floor(p1 / sr))
        grid = [k * sr for k in range(first_k, last_k + 1) if p0 < k * sr < p1]

        seg = 0
        for g in grid:
            while chain_len[seg + 1] < g:
                seg += 1
            l0, l1 = chain_len[seg], chain_len[seg + 1]
            a, b = chain[seg], chain[seg + 1]
            rpos = (g - l0) / (l1 - l0) if l1 > l0 else 0.0
            X.append(tree.X[a] + (tree.X[b] - tree.X[a]) * rpos)
            Y.append(tree.Y[a] + (tree.Y[b] - tree.Y[a]) * rpos)
            Z.append(tree.Z[a] + (tree.Z[b] - tree.Z[a]) * rpos)
            D.append(tree.D[a] + (tree.D[b] - tree.D[a]) * rpos)
            R.append(tree.R[b])
            rows.append(n)
            cols.append(prev_new)
            prev_new = n
            n += 1
        rows.append(old_to_new[end])
        cols.append(prev_new)

    dA = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    resampled = Tree(
        dA=dA, X=np.array(X), Y=np.array(Y), Z=np.array(Z), D=np.array(D),
        R=np.array(R), rnames=tree.rnames, name=tree.name, frustum=tree.frustum,
    )
    return sort_tree(resampled, by="lo")[0]


# ---------------------------------------------------------------------------
# functions filed under MATLAB's "metrics" but placed here since they need
# delete_tree/resample_tree, which need metrics.py functions themselves --
# putting these in metrics.py would make it import edit.py, a cycle.
# ---------------------------------------------------------------------------


def abel_tree(tree: Tree) -> float:
    """Average segment length [um] between branch/termination points, after
    collapsing every continuation point (a measure of typical inter-branch
    spacing, independent of how densely the reconstruction was sampled)."""
    collapse = C_tree(tree).copy()
    collapse[0] = False  # never collapse the root
    pruned = delete_tree(tree, collapse, keep_regions=True)
    if isinstance(pruned, list):
        raise ValueError("abel_tree: collapsing continuation points disconnected the tree")
    return float(len_tree(pruned).mean())


def rootangle_tree(tree: Tree) -> np.ndarray:
    """Angle (radians) between each segment and the straight line from the
    root to that segment's end, computed on a 1 um resampling of the tree.

    Centers the tree on its root first -- MATLAB's version measures against
    the coordinate origin directly, which only equals "distance to root" if
    the tree happens to already be centered there; this port's explicit
    `tran_tree` call makes the "line to root" in the docstring correct
    regardless of the tree's absolute position.
    """
    rtree = tran_tree(resample_tree(tree, sr=1.0, extend_terminals=False))
    X1, X2, Y1, Y2, Z1, Z2 = cyl_tree(rtree)

    dX, dY, dZ = X2 - X1, Y2 - Y1, Z2 - Z1
    seg_len = np.sqrt(dX**2 + dY**2 + dZ**2)
    seg_len[seg_len == 0] = 1.0
    dX, dY, dZ = dX / seg_len, dY / seg_len, dZ / seg_len

    root_len = np.sqrt(X2**2 + Y2**2 + Z2**2)
    root_len[root_len == 0] = 1.0
    rX, rY, rZ = X2 / root_len, Y2 / root_len, Z2 / root_len

    cosang = dX * rX + dY * rY + dZ * rZ
    rootangle = np.arccos(np.clip(cosang, -1.0, 1.0))
    rootangle[0] = 0.0
    return rootangle
