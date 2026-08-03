"""Coordinate-based metrics and geometry transforms.

Ports (part of) treestoolbox-master/metrics/*.m -- the functions that need
``X``/``Y``/``Z``/``D`` in addition to ``dA``. `abel_tree` and
`rootangle_tree` are MATLAB "metrics" functions but live in `edit.py`
instead, since they need `delete_tree`/`resample_tree`, which need this
module -- putting them here would be a circular import (see PORT_STATUS.md).
`rot_tree`'s PCA/`-m3d`/`-al` modes are not ported (niche, needs
region-plane fitting; the common degree-based rotation is implemented here).

`dist_tree`, `bin_tree`, `gene_tree` are filed under MATLAB's "graphtheory"
but live here for the same reason: they need `len_tree`/`eucl_tree`.

All geometry transforms (`scale_tree`, `tran_tree`, `rot_tree`, `flip_tree`,
`flatten_tree`, `morph_tree`, `zcorr_tree`) return a *new* Tree rather than
mutating in place -- unlike the MATLAB originals, which mutate a global
trees array when called without an output argument. Pure functions only.
"""

from __future__ import annotations

import numpy as np

from .core import Tree
from .graphtheory import (
    B_tree,
    C_tree,
    Pvec_tree,
    _subtree_blocks,
    idpar_tree,
    ipar_tree,
    sort_tree,
    sub_tree,
    typeN_tree,
)


def _descendants_of(blocks, node: int) -> np.ndarray:
    """Index array of ``node`` plus all its descendants -- the same set
    :func:`~pytrees.sub_tree` returns, looked up in O(subtree size) from the
    prebuilt :func:`~pytrees.graphtheory._subtree_blocks` decomposition.

    Used where a per-node loop needs a descendant set at every step
    (:func:`flatten_tree`, :func:`morph_tree`). See ``_subtree_blocks``'s
    docstring for why the two obvious alternatives (a per-node `sub_tree`
    BFS, or scanning an `ipar_tree` matrix) are both quadratic here.
    """
    order, start, size = blocks
    return order[start[node] : start[node] + size[node]]

# ---------------------------------------------------------------------------
# segment geometry
# ---------------------------------------------------------------------------


def cyl_tree(tree: Tree, dim2: bool = False):
    """Start/end coordinates of every segment (node-to-parent).

    Returns ``(X1, X2, Y1, Y2)`` if ``dim2`` else ``(X1, X2, Y1, Y2, Z1, Z2)``,
    each an ``(n_nodes,)`` array; the root's segment has ``point1 == point2``
    (its self-referencing "parent" under :func:`idpar_tree`'s default).
    """
    idpar = idpar_tree(tree)
    X1, X2 = tree.X[idpar], tree.X
    Y1, Y2 = tree.Y[idpar], tree.Y
    if dim2:
        return X1, X2, Y1, Y2
    Z1, Z2 = tree.Z[idpar], tree.Z
    return X1, X2, Y1, Y2, Z1, Z2


def len_tree(tree: Tree, dim2: bool = False) -> np.ndarray:
    """Length of every segment [um] (root: 0)."""
    if dim2:
        X1, X2, Y1, Y2 = cyl_tree(tree, dim2=True)
        return np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2)
    X1, X2, Y1, Y2, Z1, Z2 = cyl_tree(tree)
    return np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2 + (Z2 - Z1) ** 2)


def surf_tree(tree: Tree) -> np.ndarray:
    """Lateral surface area of every segment [um^2]."""
    D = tree.D
    length = len_tree(tree)
    if tree.frustum:
        idpar = idpar_tree(tree)
        Dp = D[idpar]
        return (np.pi * (D + Dp) / 2) * np.sqrt(length**2 + (D - Dp) ** 2 / 4)
    return np.pi * D * length


def vol_tree(tree: Tree) -> np.ndarray:
    """Volume of every segment [um^3]."""
    D = tree.D
    length = len_tree(tree)
    if tree.frustum:
        idpar = idpar_tree(tree)
        Dp = D[idpar]
        return (np.pi * length * (D**2 + D * Dp + Dp**2)) / 12
    return (np.pi * length * D**2) / 4


def cvol_tree(tree: Tree) -> np.ndarray:
    """Continuous volume [1/um] of every segment, for electrotonic calculations."""
    D = tree.D
    length = len_tree(tree)
    if tree.frustum:
        idpar = idpar_tree(tree)
        Dp = D[idpar]
        cvol = (12 * length) / (np.pi * (D**2 + D * Dp + Dp**2))
    else:
        cvol = (4 * length) / (np.pi * D**2)
    cvol[cvol == 0] = 0.0001  # numeric correction, matches MATLAB original
    return cvol


# ---------------------------------------------------------------------------
# distances, directions, angles
# ---------------------------------------------------------------------------


def eucl_tree(tree: Tree, point=None, dim: int = 3) -> np.ndarray:
    """Euclidean distance from every node to ``point`` (default: the root).

    ``point`` is a node index (int) or an explicit ``(x, y[, z])`` coordinate.
    ``dim`` is 2 or 3.
    """
    if point is None:
        point = 0
    if np.isscalar(point):
        px, py, pz = tree.X[point], tree.Y[point], tree.Z[point]
    else:
        point = np.asarray(point, dtype=float)
        px, py = point[0], point[1]
        pz = point[2] if point.size > 2 else 0.0

    if dim == 2:
        return np.sqrt((tree.X - px) ** 2 + (tree.Y - py) ** 2)
    return np.sqrt((tree.X - px) ** 2 + (tree.Y - py) ** 2 + (tree.Z - pz) ** 2)


def direction_tree(tree: Tree, normalize: bool = True) -> np.ndarray:
    """``(n_nodes, 3)`` vector from each node's parent to the node itself.

    The root has no real parent direction; it's set to node 1's direction as
    a placeholder, matching the MATLAB original.
    """
    idpar = idpar_tree(tree)
    direction = np.stack(
        [tree.X - tree.X[idpar], tree.Y - tree.Y[idpar], tree.Z - tree.Z[idpar]],
        axis=1,
    )
    if normalize:
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        direction = direction / norms
    if tree.n_nodes > 1:
        direction[0] = direction[1]
    return direction


def angleB_tree(tree: Tree) -> np.ndarray:
    """Angle (radians) between the two daughter branches at each branch
    point (NaN elsewhere). Requires strictly binary branch points."""
    dA = tree.dA.tocsc()
    angleB = np.full(tree.n_nodes, np.nan)
    for bp in np.flatnonzero(B_tree(tree)):
        children = np.flatnonzero(dA[:, bp].toarray().ravel())
        if len(children) != 2:
            raise ValueError(
                f"node {bp} has {len(children)} children; angleB_tree requires "
                "strictly binary branch points (run repair_tree first)"
            )
        Pr = np.array([tree.X[bp], tree.Y[bp], tree.Z[bp]])
        V1 = np.array([tree.X[children[0]], tree.Y[children[0]], tree.Z[children[0]]]) - Pr
        V2 = np.array([tree.X[children[1]], tree.Y[children[1]], tree.Z[children[1]]]) - Pr
        n1, n2 = V1 / np.linalg.norm(V1), V2 / np.linalg.norm(V2)
        angleB[bp] = 0.0 if np.allclose(n1, n2) else np.arccos(np.clip(n1 @ n2, -1.0, 1.0))
    return angleB


# ---------------------------------------------------------------------------
# rigid geometry transforms
# ---------------------------------------------------------------------------


def scale_tree(
    tree: Tree, fac=2.0, center: bool = True, scale_diameter: bool = True
) -> Tree:
    """Scale a tree's coordinates (and, by default, diameter) by ``fac``.

    ``fac`` is a scalar or an ``(fx, fy, fz)`` triple. If ``center`` (default),
    scaling is performed about the root's own position rather than the
    coordinate origin.
    """
    fac = np.atleast_1d(np.asarray(fac, dtype=float))
    ox, oy, oz = (tree.X[0], tree.Y[0], tree.Z[0]) if center else (0.0, 0.0, 0.0)
    X, Y, Z = tree.X - ox, tree.Y - oy, tree.Z - oz

    if fac.size > 1:
        X, Y, Z = X * fac[0], Y * fac[1], Z * fac[2]
        D = tree.D * fac[:2].mean() if scale_diameter else tree.D
    else:
        X, Y, Z = X * fac[0], Y * fac[0], Z * fac[0]
        D = tree.D * fac[0] if scale_diameter else tree.D

    return tree.with_coords(X=X + ox, Y=Y + oy, Z=Z + oz, D=D)


def tran_tree(tree: Tree, offset=0) -> Tree:
    """Translate a tree's coordinates.

    ``offset`` is a node index (int, default 0 == root) to recenter the tree
    on -- i.e. that node becomes the new origin -- or an explicit
    ``(dx, dy[, dz])`` vector to translate by.
    """
    if np.isscalar(offset):
        node = int(offset)
        dx, dy, dz = -tree.X[node], -tree.Y[node], -tree.Z[node]
    else:
        offset = np.asarray(offset, dtype=float)
        if offset.size == 2:
            offset = np.append(offset, 0.0)
        dx, dy, dz = offset

    return tree.with_coords(X=tree.X + dx, Y=tree.Y + dy, Z=tree.Z + dz)


def _rotation_matrix(degx, degy, degz, hand: str = "right") -> np.ndarray:
    """3x3 rotation matrix R = Rz @ Ry @ Rx, angles in radians."""
    if hand == "left":
        Rx = np.array([[1, 0, 0], [0, np.cos(degx), np.sin(degx)], [0, -np.sin(degx), np.cos(degx)]])
        Ry = np.array([[np.cos(degy), 0, -np.sin(degy)], [0, 1, 0], [np.sin(degy), 0, np.cos(degy)]])
        Rz = np.array([[np.cos(degz), np.sin(degz), 0], [-np.sin(degz), np.cos(degz), 0], [0, 0, 1]])
    else:
        Rx = np.array([[1, 0, 0], [0, np.cos(degx), -np.sin(degx)], [0, np.sin(degx), np.cos(degx)]])
        Ry = np.array([[np.cos(degy), 0, np.sin(degy)], [0, 1, 0], [-np.sin(degy), 0, np.cos(degy)]])
        Rz = np.array([[np.cos(degz), -np.sin(degz), 0], [np.sin(degz), np.cos(degz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rot_tree(tree: Tree, deg=(0.0, 0.0, 90.0)) -> Tree:
    """Rotate a tree by ``deg`` degrees: a scalar (XY-plane rotation) or an
    ``(x, y[, z])`` triple of axis rotations, applied x then y then z.

    MATLAB's PCA/``-m3d``/``-al`` automatic-alignment modes are not ported
    (niche; see PORT_STATUS.md).
    """
    deg = np.atleast_1d(np.asarray(deg, dtype=float))
    if deg.size == 1:
        theta = np.radians(deg[0])
        RM = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        XY = np.stack([tree.X, tree.Y], axis=1) @ RM
        return tree.with_coords(X=XY[:, 0], Y=XY[:, 1])

    if deg.size == 2:
        deg = np.append(deg, 0.0)
    RM = _rotation_matrix(*np.radians(deg))
    XYZ = np.stack([tree.X, tree.Y, tree.Z], axis=1) @ RM
    return tree.with_coords(X=XYZ[:, 0], Y=XYZ[:, 1], Z=XYZ[:, 2])


def flip_tree(tree: Tree, axis: str = "x") -> Tree:
    """Mirror a tree around the root, along one axis ('x', 'y', or 'z')."""
    if axis == "x":
        return tree.with_coords(X=2 * tree.X[0] - tree.X)
    if axis == "y":
        return tree.with_coords(Y=2 * tree.Y[0] - tree.Y)
    if axis == "z":
        return tree.with_coords(Z=2 * tree.Z[0] - tree.Z)
    raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")


# ---------------------------------------------------------------------------
# shape-preserving / shape-correcting transforms
# ---------------------------------------------------------------------------


def flatten_tree(tree: Tree) -> Tree:
    """Flatten a tree onto the XY plane, conserving each segment's length
    (subtrees are shifted outward in X/Y to compensate for the lost Z
    extent, exactly as the 3D segment length is preserved in 2D)."""
    idpar = idpar_tree(tree)
    blocks = _subtree_blocks(tree.dA)
    tree = tran_tree(tree, [0.0, 0.0, -tree.Z[0]])
    if np.all(np.abs(tree.Z) < 1e-3):
        return tree.with_coords(Z=np.zeros_like(tree.Z))

    X, Y, Z = tree.X.copy(), tree.Y.copy(), tree.Z.copy()
    for node in range(1, tree.n_nodes):
        parent = idpar[node]
        dX, dY, dZ = X[node] - X[parent], Y[node] - Y[parent], Z[node] - Z[parent]
        xy = np.hypot(dX, dY)
        xyz = np.sqrt(dX**2 + dY**2 + dZ**2)
        idx = _descendants_of(blocks, node)
        if xy != 0:
            u = xyz / xy
            X[idx] += (u - 1) * dX
            Y[idx] += (u - 1) * dY
        else:
            X[idx] += xyz
        Z[idx] -= dZ
        Z[node] = 0.0

    return tree.with_coords(X=X, Y=Y, Z=Z)


def morph_tree(tree: Tree, v: np.ndarray | None = None) -> Tree:
    """Rescale every segment's length to ``v`` (default: 10 um each) while
    preserving branch angles and topology -- a META-FUNCTION: e.g. passing
    the original ``len_tree`` output back in regrows the original geometry
    (except for originally-zero-length segments, which can't be recovered).
    """
    N = tree.n_nodes
    v = np.full(N, 10.0) if v is None else np.asarray(v, dtype=float)

    idpar = idpar_tree(tree)
    blocks = _subtree_blocks(tree.dA)
    x0, y0, z0 = tree.X[0], tree.Y[0], tree.Z[0]
    tree = tran_tree(tree)
    length = len_tree(tree)

    X, Y, Z = tree.X.copy(), tree.Y.copy(), tree.Z.copy()
    rng = np.random.default_rng()
    for node in range(1, N):
        if length[node] == v[node]:
            continue
        parent = idpar[node]
        dX, dY, dZ = X[node] - X[parent], Y[node] - Y[parent], Z[node] - Z[parent]
        xyz = np.sqrt(dX**2 + dY**2 + dZ**2)
        if xyz == 0:
            r = rng.normal(size=3)
            dX, dY, dZ = r / np.linalg.norm(r)
            xyz = 1.0
        idx = _descendants_of(blocks, node)
        X[idx] += -dX + v[node] * (dX / xyz)
        Y[idx] += -dY + v[node] * (dY / xyz)
        Z[idx] += -dZ + v[node] * (dZ / xyz)

    result = tree.with_coords(X=X, Y=Y, Z=Z)
    return tran_tree(result, [x0, y0, z0])


def zcorr_tree(tree: Tree, tz: float = 5.0):
    """Correct sudden Neurolucida-style Z jumps: any parent-child Z gap
    exceeding ``tz`` [um] is subtracted from the entire downstream subtree.

    Returns ``(new_tree, jumped_nodes)``.
    """
    idpar = idpar_tree(tree)
    dZ = tree.Z[idpar] - tree.Z
    jumped = np.flatnonzero(np.abs(dZ) > tz)

    Z = tree.Z.copy()
    for node in jumped:
        mask = sub_tree(tree, int(node))
        Z[mask] += dZ[node]

    return tree.with_coords(Z=Z), jumped


# ---------------------------------------------------------------------------
# functions filed under MATLAB's "graphtheory" but placed here since they
# need len_tree/eucl_tree (putting them in graphtheory.py would make it
# import metrics.py, which already imports graphtheory.py -- a cycle)
# ---------------------------------------------------------------------------


def dist_tree(tree: Tree, distances) -> np.ndarray:
    """Boolean ``(n_nodes, len(distances))`` matrix: True wherever a node's
    segment crosses a given path distance [um] from the root."""
    Plen = Pvec_tree(tree, len_tree(tree))
    idpar = idpar_tree(tree)
    distances = np.atleast_1d(np.asarray(distances, dtype=float))
    parent_len = Plen[idpar][:, None]
    node_len = Plen[:, None]
    return (distances[None, :] >= parent_len) & (distances[None, :] < node_len)


def bin_tree(tree: Tree, v: np.ndarray | None = None, bins=10):
    """Bin nodes by ``v`` (default: Euclidean distance to root).

    ``bins`` is either a bin count or explicit bin edges. Returns
    ``(bin_index, edges)``; ``bin_index[i]`` is 0 if node ``i`` falls
    outside every bin, else its 1-based bin number.
    """
    v = eucl_tree(tree) if v is None else np.asarray(v, dtype=float)
    bins_arr = np.asarray(bins, dtype=float)
    edges = (
        np.linspace(v.min(), v.max() * 1.0001, int(bins_arr) + 1)
        if bins_arr.ndim == 0
        else bins_arr
    )
    bin_index = np.digitize(v, edges)
    bin_index[(v < edges[0]) | (v > edges[-1])] = 0
    return bin_index, edges


def gene_tree(tree: Tree) -> np.ndarray:
    """Topological "gene" of a tree: an ``(n_branches, 2)`` array of each
    branch/terminal segment's own path length and its ending node type
    (2=branch, 0=terminal) -- a compact shape signature, useful for
    comparing topology across trees. Operates on one tree at a time;
    MATLAB's nested-cell-array batch/plotting wrapper is population-level
    tooling that belongs with Phase 9's comparison utilities instead.
    """
    sorted_tree, _ = sort_tree(tree, by="lo")
    idpar = idpar_tree(sorted_tree)
    is_cut = ~C_tree(sorted_tree)
    length = len_tree(sorted_tree)
    typeN = typeN_tree(sorted_tree)

    end_points = np.flatnonzero(is_cut)
    pathlen = np.zeros(len(end_points))
    for i, end in enumerate(end_points):
        node, total = end, 0.0
        while True:
            total += length[node]
            parent = idpar[node]
            if parent == node or is_cut[parent]:
                break
            node = parent
        pathlen[i] = total

    return np.stack([pathlen, typeN[end_points].astype(float)], axis=1)
