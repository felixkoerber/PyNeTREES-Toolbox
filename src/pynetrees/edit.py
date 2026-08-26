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

import logging
from typing import NamedTuple

import numpy as np
from scipy import sparse

from ._population import accepts_population
from ._empty import empty_safe
from .core import NO_PARENT, Tree
from .graphtheory import (
    B_tree,
    C_tree,
    child_tree,
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

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# node deletion
# ---------------------------------------------------------------------------


@accepts_population(paired="inodes")
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

    if N and delete_mask.all():
        # Deleting everything gives an empty tree, as MATLAB does -- not an
        # error. An empty tree is a usable value throughout this port (see
        # `pynetrees._empty`), and refusing here would make "filter a
        # population down to the cells matching X" fail on the one cell
        # where nothing matches.
        return Tree(
            dA=sparse.csr_matrix((0, 0)),
            X=np.empty(0), Y=np.empty(0), Z=np.empty(0), D=np.empty(0),
            R=np.empty(0, dtype=int),
            rnames=[] if not keep_regions else list(tree.rnames),
            name=tree.name, frustum=tree.frustum,
            Ri=tree.Ri, Gm=tree.Gm, Cm=tree.Cm,
        )

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
    return [sub_tree(result, int(r)).tree for r in roots]


# ---------------------------------------------------------------------------
# repair pipeline
# ---------------------------------------------------------------------------


@accepts_population
def elim0_tree(tree: Tree, keep_regions: bool = False) -> Tree:
    """Delete zero-length segments (except the root's own trivial one)."""
    zero_idx = np.flatnonzero(len_tree(tree) == 0)
    if len(zero_idx) > 1:
        result = delete_tree(tree, zero_idx[1:], keep_regions=keep_regions)
        if isinstance(result, list):
            raise ValueError("elim0_tree: unexpectedly disconnected the tree")
        return result
    return tree


@accepts_population
def elimt_tree(tree: Tree, at_root: bool = True) -> Tree:
    """Replace every multifurcation (3+ children) with a short chain of
    bifurcations, each spacer offset by ~0.0001 um along the parent segment.

    Parameters
    ----------
    tree : Tree
    at_root : bool, default True
        Whether to also split a multifurcating *root*. ``True`` matches
        MATLAB's default. ``False`` (MATLAB's ``'-r'``) leaves the root
        alone, which is what you want for a soma that legitimately branches
        into several primary dendrites and shouldn't grow a spacer chain.

    Returns
    -------
    Tree
        The de-multifurcated tree, or the input unchanged if there was
        nothing to do.

    Notes
    -----
    Previously returned ``(tree, changed)``. The flag is gone (Design
    Decision #42): it is recomputable -- ``typeN_tree(result).max() <= 2``,
    or simply comparing ``n_nodes`` -- and every caller was unpacking a
    tuple for information almost none of them used. When nothing changes,
    that fact now goes to :mod:`logging` at debug level, so library code
    stays quiet by default but the information is still recoverable.
    """
    dA = tree.dA.tocsr()
    N = tree.n_nodes
    idpar = idpar_tree(tree)
    is_root = np.asarray(dA.sum(axis=1)).ravel() == 0
    children_count = np.asarray(dA.sum(axis=0)).ravel()

    multif = np.flatnonzero(children_count > 2)
    if not at_root:
        multif = multif[~is_root[multif]]
    if len(multif) == 0:
        _log.debug("elimt_tree: no multifurcations in %r, returning unchanged", tree.name)
        return tree

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
    return new_tree


@accepts_population
@empty_safe("tree")
def repair_tree(tree: Tree, no_root_trifurcation: bool = False) -> Tree:
    """Rectify a tree to full BCT conformity: eliminate multifurcations,
    drop zero-length segments, and sort into canonical (level-order) index
    order. Most other functions in this toolbox assume their input has
    already been through this.
    """
    tree = elimt_tree(tree, at_root=not no_root_trifurcation)
    tree = elimt_tree(tree)
    tree = elim0_tree(tree)

    if tree.n_nodes > 1 and T_tree(tree)[0]:
        dA = tree.dA.tolil()
        dA[1, 0] = 1
        tree = Tree(
            dA=dA, X=tree.X, Y=tree.Y, Z=tree.Z, D=tree.D, R=tree.R,
            rnames=tree.rnames, name=tree.name, frustum=tree.frustum,
        )

    tree = sort_tree(tree, by="lo")
    return tree


# ---------------------------------------------------------------------------
# adding / moving nodes
# ---------------------------------------------------------------------------


@accepts_population
@empty_safe("tree")
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


class InsertResult(NamedTuple):
    """Result of :func:`insert_tree` with ``full_output=True``."""

    tree: Tree
    """The tree with the new nodes appended."""
    inodes: np.ndarray
    """0-based indices of the newly added nodes, in input order."""


def insert_tree(tree: Tree, X, Y, Z, D, parent, R=None, *, full_output: bool = False):
    """Append new nodes to a tree.

    Parameters
    ----------
    tree : Tree
    X, Y, Z, D : array_like
        Coordinates [um] and diameters [um] of the new nodes.
    parent : array_like of int
        0-based parent index for each new node. Replaces MATLAB's
        ``[inode R X Y Z D idpar]`` SWC-tuple calling convention.
    R : array_like of int, optional
        Region index per new node; defaults to each node's parent's region.
    full_output : bool, default False
        If ``True``, return an :class:`InsertResult` ``(tree, inodes)``.

    Returns
    -------
    Tree or InsertResult

    Raises
    ------
    ValueError
        If any ``parent[i]`` is a *forward* reference -- i.e. points at a
        new node that has not been assigned an index yet
        (``parent[i] >= N + i``), or is out of range entirely. Left
        unchecked this silently produces a cycle or an orphan.

    Notes
    -----
    **New nodes may parent each other.** The parent index is written
    straight into the adjacency matrix, so ``parent[i]`` may refer either to
    an existing node (``0 <= p < n_nodes``) or to an *earlier* new node
    (``n_nodes <= p < n_nodes + i``). That is not incidental --
    :func:`~pynetrees.cap_tree` depends on it, chaining each cap segment onto
    the previous one:

    .. code-block:: python

        # three nodes in a chain hanging off existing node 0
        n = tree.n_nodes
        insert_tree(tree, X=[1., 2., 3.], Y=[0., 0., 0.], Z=[0., 0., 0.],
                    D=[1., 1., 1.], parent=[0, n, n + 1])

    The capability was previously undocumented and unvalidated; the
    forward-reference check above is what makes it safe to rely on.
    """
    X, Y, Z, D = (np.asarray(a, dtype=float) for a in (X, Y, Z, D))
    parent = np.asarray(parent, dtype=int)
    n_new = len(X)
    N = tree.n_nodes

    if n_new and (len(Y) != n_new or len(Z) != n_new or len(D) != n_new
                  or len(parent) != n_new):
        raise ValueError(
            f"insert_tree: X/Y/Z/D/parent must all have the same length "
            f"(got {n_new}, {len(Y)}, {len(Z)}, {len(D)}, {len(parent)})"
        )

    limits = N + np.arange(n_new)
    bad = np.flatnonzero((parent < 0) | (parent >= limits))
    if len(bad):
        i = int(bad[0])
        raise ValueError(
            f"insert_tree: parent[{i}] = {parent[i]} is not a valid parent for "
            f"new node {N + i}. Parents must be an existing node (0..{N - 1}) "
            f"or an *earlier* new node (up to {N + i - 1}); a forward "
            f"reference would create a cycle or an orphan."
        )

    if R is None:
        # Inherit each new node's region from its parent. A parent may itself
        # be a new node, whose region is only known once *it* has inherited --
        # so resolve in input order, which the forward-reference check above
        # guarantees is a valid topological order.
        R = np.empty(n_new, dtype=int)
        for i, p in enumerate(parent.tolist()):
            R[i] = tree.R[p] if p < N else R[p - N]
    else:
        R = np.asarray(R, dtype=int)

    coo = tree.dA.tocoo()
    rows = np.concatenate([coo.row, np.arange(N, N + n_new)])
    cols = np.concatenate([coo.col, parent])
    dA = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(N + n_new, N + n_new)
    ).tocsr()

    new_tree = Tree(
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
    if full_output:
        return InsertResult(new_tree, np.arange(N, N + n_new))
    return new_tree


class InsertpResult(NamedTuple):
    """Result of :func:`insertp_tree` with ``full_output=True``."""

    tree: Tree
    """The tree with the interpolated nodes inserted, re-sorted."""
    added: np.ndarray
    """Boolean mask over the *new* tree, ``True`` for inserted nodes."""


@accepts_population
@empty_safe("tree")
def insertp_tree(
    tree: Tree, inode: int | None = None, plens=None, *, full_output: bool = False
):
    """Insert nodes at given path lengths along the root-to-``inode`` path.

    Parameters
    ----------
    tree : Tree
    inode : int, optional
        0-based index of the node whose root path is subdivided. Defaults
        to the last node.
    plens : array_like, optional
        Path lengths [um] from the root at which to insert. Defaults to
        every 10 um, or a single node at the halfway point if the path is
        shorter than 10 um. Values already occupied by a node, or beyond
        the path's end, are dropped.
    full_output : bool, default False
        If ``True``, return an :class:`InsertpResult` ``(tree, added)``.
        The mask cannot be recomputed afterwards -- the result is re-sorted,
        so inserted nodes are no longer identifiable by index (Design
        Decision #42).

    Returns
    -------
    Tree or InsertpResult
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
        return InsertpResult(tree, np.zeros(N, dtype=bool)) if full_output else tree

    X, Y, Z, D, R = tree.X.tolist(), tree.Y.tolist(), tree.Z.tolist(), tree.D.tolist(), tree.R.tolist()
    # Edges are edited as we go rather than collected and applied at the
    # end: two insertions into the same original segment make the second
    # one split an edge the *first* one created, which is not in the
    # original `dA` at all.
    coo = tree.dA.tocoo()
    edges = list(zip(coo.row.tolist(), coo.col.tolist()))
    n = N
    for pl in plens.tolist():
        pos = max(i for i in range(len(plen_path)) if plen_path[i] < pl)
        p_node, c_node = path[pos], path[pos + 1]
        rpos = (pl - plen_path[pos]) / (plen_path[pos + 1] - plen_path[pos])
        new_node = n
        n += 1
        # Read from the growing lists, not from `tree`: two insertions into
        # the same original segment make the second one's parent a node
        # that only exists here, and `tree.X[p_node]` would run off the end
        # of the original arrays. MATLAB writes into the growing fields for
        # the same reason.
        X.append(X[p_node] + (X[c_node] - X[p_node]) * rpos)
        Y.append(Y[p_node] + (Y[c_node] - Y[p_node]) * rpos)
        Z.append(Z[p_node] + (Z[c_node] - Z[p_node]) * rpos)
        D.append(D[p_node] + (D[c_node] - D[p_node]) * rpos)
        R.append(R[c_node])
        edges.remove((c_node, p_node))
        edges.append((new_node, p_node))
        edges.append((c_node, new_node))
        path.insert(pos + 1, new_node)
        plen_path.insert(pos + 1, pl)

    rows = [c for c, _ in edges]
    cols = [p for _, p in edges]
    dA = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    new_tree = Tree(
        dA=dA, X=np.array(X), Y=np.array(Y), Z=np.array(Z), D=np.array(D),
        R=np.array(R), rnames=tree.rnames, name=tree.name, frustum=tree.frustum,
    )
    sorted_tree, order = sort_tree(new_tree, by="lo", full_output=True)
    if full_output:
        return InsertpResult(sorted_tree, order >= N)
    return sorted_tree


@accepts_population
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


@accepts_population(paired=("ichilds", "ipars",))
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
            mask = sub_tree(tree, int(child), with_tree=False).mask
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


@accepts_population
@empty_safe("tree")
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

    tree2 = redirect_tree(tree2, inode2)
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
    return sort_tree(merged, by="lo")


@accepts_population
def resample_tree(
    tree: Tree,
    sr: float = 10.0,
    method: str = "matlab",
    *,
    extend_terminals: bool = True,
    interp_diameter: bool = False,
    conserve_length: bool = False,
    collapse_branches: bool = True,
    preserve_branch_spacing: bool = False,
    trim_regions: bool = True,
) -> Tree:
    """Redistribute a tree's nodes to roughly ``sr`` [um] spacing.

    Parameters
    ----------
    tree : Tree
    sr : float, default 10.0
        Target internode spacing [um].
    method : {'matlab', 'anchors'}, default 'matlab'
        Which abstraction to use for the bits resampling leaves
        underdetermined -- MATLAB's own docstring says "some abstraction
        principles need to be arbitrarily set", and the two methods set
        them differently.

        - ``'matlab'`` -- a faithful port of `resample_tree.m`. Every node
          in the result sits at an exact multiple of ``sr`` path length
          from the root, because *all* original nodes are deleted after the
          grid points are inserted. Branch and termination points therefore
          **move** onto the grid.
        - ``'anchors'`` -- branch and termination points stay exactly where
          they were, and only the nodes between them are redistributed.
          Better when you care about branch-point positions (the NEURON
          bridge does), but it is not what MATLAB computes.

    extend_terminals : bool, default True
        Stretch each terminal segment by ``sr / 2`` first, so the grid
        does not systematically truncate branch tips. MATLAB does this
        unconditionally; here it is switchable.
    interp_diameter : bool, default False
        MATLAB's ``'-d'``. Interpolate diameters along each segment rather
        than inheriting the child node's. Changes total surface and volume,
        which is why it is off by default.
    conserve_length : bool, default False
        MATLAB's ``'-l'``. After resampling, stretch every segment back to
        exactly ``sr`` so total path lengths match the original. The tree
        grows slightly overall, so this is wrong for automated
        reconstruction pipelines and right for length-preserving analysis.
    collapse_branches : bool, default True
        Merge branch daughters that end up implausibly close together
        (within 0.75 * 2 * ``sr`` of path length of each other). MATLAB's
        ``'-v'`` switches this *off*; the sense is inverted here per Design
        Decision #41.
    preserve_branch_spacing : bool, default False
        MATLAB's ``'-b'``. Lengthen sub-``sr`` segments that run between two
        branch points, so consecutive branch points do not collapse into a
        multifurcation. MATLAB's own docstring warns this "does not
        preserve length" and "might give a mess with high sr".
    trim_regions : bool, default True
        Drop region names left unused after resampling. MATLAB's ``'-r'``
        switches this off; inverted here per #41.

    Returns
    -------
    Tree

    Notes
    -----
    ``method='matlab'`` is the default as of Design Decision #45, reversing
    #23. The port originally shipped only the anchor-preserving variant, on
    the grounds that MATLAB's snapping rule is arbitrary -- which is true,
    but "arbitrary" is not the same as "wrong", and defaulting to something
    other than the reference implementation makes every downstream number
    quietly incomparable.

    A single-node tree has nothing to resample and is returned unchanged.
    """
    if method not in ("matlab", "anchors"):
        raise ValueError(f"method must be 'matlab' or 'anchors', got {method!r}")
    if tree.n_nodes < 2:
        return tree

    if method == "matlab":
        return _resample_matlab(
            tree, sr,
            extend_terminals=extend_terminals,
            interp_diameter=interp_diameter,
            conserve_length=conserve_length,
            collapse_branches=collapse_branches,
            preserve_branch_spacing=preserve_branch_spacing,
            trim_regions=trim_regions,
        )

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
    return sort_tree(resampled, by="lo")



def _grid_points(plen_parent: float, plen_child: float, sr: float) -> list[float]:
    """Path lengths on the ``0, sr, 2*sr, ...`` grid lying inside one segment.

    MATLAB writes this as ``Gpath = 0 : sr : Plen(child)`` followed by
    ``Gpath(Gpath > Plen(parent))``. Building the whole ramp and discarding
    most of it is cheap in MATLAB and wasteful here, so this computes the
    index range directly -- identical values, no allocation proportional to
    the node's depth in the tree.
    """
    k_first = int(np.floor(plen_parent / sr)) + 1
    k_last = int(np.floor(plen_child / sr))
    return [k * sr for k in range(k_first, k_last + 1) if k * sr > plen_parent]


def _resample_matlab(
    tree: Tree,
    sr: float,
    *,
    extend_terminals: bool,
    interp_diameter: bool,
    conserve_length: bool,
    collapse_branches: bool,
    preserve_branch_spacing: bool,
    trim_regions: bool,
) -> Tree:
    """Faithful port of ``edit/resample_tree.m``.

    The original's five steps, kept visible because each is separately
    observable in the output:

    1. stretch terminal segments by ``sr / 2`` (and, with the branch-spacing
       option, stretch short branch-to-branch segments to ``sr``), then
       :func:`morph_tree`;
    2. on every edge, insert a node at each multiple of ``sr`` of path
       length from the root that falls strictly inside that edge;
    3. **delete every original node except the root** -- the step that moves
       branch and termination points onto the grid, and the one
       ``method='anchors'`` declines to do;
    4. collapse branch daughters the grid left implausibly close together;
    5. optionally re-stretch every segment to exactly ``sr``.
    """
    N = tree.n_nodes

    # --- 1. terminal extension, and optional branch-spacing protection ----
    if extend_terminals or preserve_branch_spacing:
        length = len_tree(tree)
        target = length.copy()
        if extend_terminals:
            term = T_tree(tree)
            target[term] = length[term] + 0.5 * sr
        if preserve_branch_spacing:
            idpar_pre = idpar_tree(tree)
            branch = B_tree(tree)
            target[(target < sr) & branch & branch[idpar_pre]] = sr
        tree = morph_tree(tree, target)

    Plen = Pvec_tree(tree)
    root = tree.root

    # --- 2. insert grid points along every edge ---------------------------
    X, Y, Z, D = (a.tolist() for a in (tree.X, tree.Y, tree.Z, tree.D))
    # `origin` is MATLAB's `nindy`: which original node each node inherits
    # its per-node fields (R, and D unless interpolating) from.
    origin = list(range(N))
    coo = tree.dA.tocoo()
    edges = [(int(c), int(p)) for c, p in zip(coo.row, coo.col)]

    rows: list[int] = []
    cols: list[int] = []
    n = N
    for child, parent in edges:
        grid = _grid_points(Plen[parent], Plen[child], sr)
        if not grid:
            rows.append(child)
            cols.append(parent)
            continue
        span = Plen[child] - Plen[parent]
        prev = parent
        for g in grid:
            rpos = (g - Plen[parent]) / span if span > 0 else 0.0
            X.append(tree.X[parent] + rpos * (tree.X[child] - tree.X[parent]))
            Y.append(tree.Y[parent] + rpos * (tree.Y[child] - tree.Y[parent]))
            Z.append(tree.Z[parent] + rpos * (tree.Z[child] - tree.Z[parent]))
            D.append(tree.D[parent] + rpos * (tree.D[child] - tree.D[parent]))
            origin.append(child)
            rows.append(n)
            cols.append(prev)
            prev = n
            n += 1
        rows.append(child)
        cols.append(prev)

    origin_arr = np.array(origin, dtype=int)
    dense = Tree(
        dA=sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr(),
        X=np.array(X), Y=np.array(Y), Z=np.array(Z),
        # Without diameter interpolation every node takes its origin node's
        # diameter, keeping the cable piecewise-constant; with it, the
        # interpolated values above are kept and surface/volume change.
        D=np.array(D) if interp_diameter else tree.D[origin_arr],
        R=tree.R[origin_arr],
        rnames=tree.rnames, name=tree.name, frustum=tree.frustum,
        Ri=tree.Ri, Gm=tree.Gm, Cm=tree.Cm,
    )

    # --- 3. delete every original node but the root -----------------------
    doomed = [i for i in range(N) if i != root]
    result = delete_tree(dense, doomed, keep_regions=not trim_regions)
    if isinstance(result, list):
        raise ValueError("resample_tree: resampling disconnected the tree")

    # Surviving nodes in the order delete_tree keeps them: the root, then
    # every inserted node. This is MATLAB's `iF`, and it is what lets the
    # collapse step ask questions of the pre-deletion tree.
    survivors = np.array([root] + list(range(N, n)), dtype=int)

    # --- 4. collapse grid-induced near-coincident branches -----------------
    if collapse_branches:
        result = _collapse_small_angle_branches(
            result, dense, survivors, sr, trim_regions=trim_regions
        )

    # --- 5. length conservation -------------------------------------------
    if conserve_length:
        result = morph_tree(result, np.full(result.n_nodes, sr))

    return result


def _collapse_small_angle_branches(
    pruned: Tree, dense: Tree, survivors: np.ndarray, sr: float, *, trim_regions: bool
) -> Tree:
    """Merge branch daughters that the resampling grid left too close together.

    After the delete step a branch point's daughters can end up within a
    fraction of ``sr`` of each other -- an artefact of snapping, not
    anatomy. MATLAB measures each daughter pair by the summed length, *in
    the pre-deletion tree*, of the two paths back to their branch point,
    normalised by ``2 * sr``; below 0.75 the pair is merged to its midpoint.

    Measuring in ``dense`` rather than ``pruned`` is the whole trick: the
    pruned tree no longer contains the intermediate nodes whose lengths are
    being summed.
    """
    ipar_dense = ipar_tree(dense)
    len_dense = len_tree(dense)
    n_children = np.asarray(pruned.dA.sum(axis=0)).ravel()
    branch_points = np.flatnonzero(n_children > 1)
    if len(branch_points) == 0:
        return pruned

    pairs: list[tuple[int, int]] = []
    for bp in branch_points:
        daughters = np.flatnonzero(
            np.asarray(pruned.dA.getcol(bp).todense()).ravel()
        )
        # path (in `dense`) from each daughter back to, but excluding, `bp`
        paths: dict[int, np.ndarray] = {}
        for d in daughters:
            chain = ipar_dense[survivors[d]]
            stop = np.flatnonzero(chain == survivors[bp])
            paths[int(d)] = chain[: stop[0]] if len(stop) else chain[chain >= 0]

        for i, d1 in enumerate(daughters):
            for d2 in daughters[i + 1:]:
                both = np.unique(np.concatenate([paths[int(d1)], paths[int(d2)]]))
                both = both[both >= 0]
                if len_dense[both].sum() / (2 * sr) < 0.75:
                    pairs.append((int(d1), int(d2)))

    if not pairs:
        return pruned

    # Of each pair keep whichever node carries more of the tree and delete
    # the other, first moving both to their midpoint so the branch point
    # does not visibly jump.
    subtree_size = child_tree(pruned)
    X, Y, Z = pruned.X.copy(), pruned.Y.copy(), pruned.Z.copy()
    dA = pruned.dA.tolil()
    to_delete: list[int] = []
    for d1, d2 in pairs:
        if d1 in to_delete or d2 in to_delete:
            continue  # already merged away by an overlapping pair
        drop, keep = (d1, d2) if subtree_size[d1] < subtree_size[d2] else (d2, d1)
        mx = (X[d1] + X[d2]) / 2
        my = (Y[d1] + Y[d2]) / 2
        mz = (Z[d1] + Z[d2]) / 2
        X[d1] = X[d2] = mx
        Y[d1] = Y[d2] = my
        Z[d1] = Z[d2] = mz
        grandchildren = np.flatnonzero(np.asarray(dA[:, drop].todense()).ravel())
        for gc in grandchildren:
            dA[gc, drop] = 0
            dA[gc, keep] = 1
        to_delete.append(drop)

    merged = Tree(
        dA=dA.tocsr(), X=X, Y=Y, Z=Z, D=pruned.D, R=pruned.R,
        rnames=pruned.rnames, name=pruned.name, frustum=pruned.frustum,
        Ri=pruned.Ri, Gm=pruned.Gm, Cm=pruned.Cm,
    )
    out = delete_tree(merged, to_delete, keep_regions=not trim_regions)
    if isinstance(out, list):
        raise ValueError("resample_tree: branch collapse disconnected the tree")
    return out


# ---------------------------------------------------------------------------
# functions filed under MATLAB's "metrics" but placed here since they need
# delete_tree/resample_tree, which need metrics.py functions themselves --
# putting these in metrics.py would make it import edit.py, a cycle.
# ---------------------------------------------------------------------------


@accepts_population
@empty_safe("nodes")
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


@accepts_population
@empty_safe("nodes")
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
