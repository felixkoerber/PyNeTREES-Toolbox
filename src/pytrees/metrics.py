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

import warnings

import numpy as np

from ._compat import resolve_dim
from .core import Tree
from .graphtheory import (
    B_tree,
    _children_lists,
    _dfs_preorder,
    child_tree,
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


def cyl_tree(tree: Tree, dim: int | None = None, *, dim2=None):
    """Start/end coordinates of every segment (node-to-parent).

    Parameters
    ----------
    tree : Tree
    dim : {2, 3}, optional
        Default 3 (see :func:`pytrees._compat.resolve_dim` for why the
        signature says ``None``).
        Work in 3D or project onto the XY plane (Design Decision #40).
    dim2 : bool, optional
        **Deprecated** boolean spelling; ``dim2=True`` means ``dim=2``.

    Returns
    -------
    tuple of np.ndarray
        ``(X1, X2, Y1, Y2)`` when ``dim == 2``, else
        ``(X1, X2, Y1, Y2, Z1, Z2)``, each of length ``n_nodes``. The root's
        segment has ``point1 == point2`` (it is its own parent under
        :func:`idpar_tree`'s default), so its length is 0.

    Notes
    -----
    MATLAB's ``'-dA'`` output form -- the same geometry as sparse matrices --
    is deliberately not ported: the MATLAB source's own comment on that
    branch reads "SLOW!!", and nothing in the toolbox calls it.
    """
    dim = resolve_dim(dim, dim2)
    idpar = idpar_tree(tree)
    X1, X2 = tree.X[idpar], tree.X
    Y1, Y2 = tree.Y[idpar], tree.Y
    if dim == 2:
        return X1, X2, Y1, Y2
    Z1, Z2 = tree.Z[idpar], tree.Z
    return X1, X2, Y1, Y2, Z1, Z2


def len_tree(tree: Tree, dim: int | None = None, *, dim2=None) -> np.ndarray:
    """Length of every node-to-parent segment [um].

    Parameters
    ----------
    tree : Tree
    dim : {2, 3}, optional
        Default 3 (see :func:`pytrees._compat.resolve_dim` for why the
        signature says ``None``).
        3D length, or the length of the segment's XY projection.
    dim2 : bool, optional
        **Deprecated** boolean spelling; ``dim2=True`` means ``dim=2``.

    Returns
    -------
    np.ndarray
        Length per node [um]. The root is 0, having no parent segment.
        ``tree.total_length`` is the sum of this.
    """
    dim = resolve_dim(dim, dim2)
    if dim == 2:
        X1, X2, Y1, Y2 = cyl_tree(tree, dim=2)
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


def eucl_tree(tree: Tree, point=None, dim: int | None = None) -> np.ndarray:
    """Euclidean ("as the crow flies") distance from every node to ``point``.

    Parameters
    ----------
    tree : Tree
    point : int or array_like, optional
        A node index, or an explicit ``(x, y[, z])`` coordinate. Defaults to
        the tree's root -- found via :attr:`Tree.root`, not assumed to be
        node 0.
    dim : {2, 3}, optional
        Default 3 (see :func:`pytrees._compat.resolve_dim` for why the
        signature says ``None``).
        Measure in 3D, or in the XY plane only.

    Returns
    -------
    np.ndarray
        Distance per node [um]. Contrast :func:`~pytrees.Pvec_tree`, which
        measures *along* the tree rather than through space.
    """
    dim = resolve_dim(dim)
    if point is None:
        point = tree.root
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



def _walk_larger_subtree(tree: Tree, first: int, sibling_rank: int,
                         steps: int, children, subtree_size) -> int:
    """Walk ``steps`` nodes down from ``first``, keeping to the bulkier branch.

    At an intermediate branch point the walk continues into whichever
    daughter carries more of the tree; when the two are equally big it falls
    back on ``sibling_rank`` (which side of the *original* branch point this
    walk started on), so the two walks do not both veer the same way.
    """
    node = first
    for _ in range(steps - 1):
        kids = children[node]
        if not kids:
            break  # ran off a terminal: report how far we got
        if len(kids) == 1:
            node = kids[0]
        else:
            sizes = [subtree_size[k] for k in kids]
            if len(set(sizes)) == 1:
                node = kids[min(sibling_rank, len(kids) - 1)]
            else:
                node = kids[int(np.argmax(sizes))]
    return node


def _walk_longest_path(tree: Tree, first: int, steps: int, children,
                       depth_to_tip) -> int:
    """Walk ``steps`` nodes down from ``first`` along the longest path to a tip."""
    node = first
    for _ in range(steps - 1):
        kids = children[node]
        if not kids:
            break
        node = kids[int(np.argmax([depth_to_tip[k] for k in kids]))]
    return node


def _angle_at_branch_points(tree: Tree, dist: int, mode: str) -> np.ndarray:
    """Shared body of :func:`angleBd_tree` and :func:`angleBd2_tree`."""
    if dist < 2:
        raise ValueError(f"dist must be at least 2 nodes, got {dist}")

    children = _children_lists(tree.dA)
    branch_points = np.flatnonzero(B_tree(tree))

    if mode == "bulk":
        subtree_size = child_tree(tree)
        depth_to_tip = None
    else:
        subtree_size = None
        # metric distance from each node to its furthest terminal
        length = len_tree(tree)
        depth_to_tip = np.zeros(tree.n_nodes)
        for node in _dfs_preorder(tree.dA)[::-1]:
            kids = children[node]
            if kids:
                depth_to_tip[node] = max(
                    depth_to_tip[k] + length[k] for k in kids
                )

    coords = np.column_stack([tree.X, tree.Y, tree.Z])
    angles = np.full(len(branch_points), np.nan)

    for i, bp in enumerate(branch_points):
        kids = children[bp]
        if len(kids) != 2:
            # angleB_tree raises here; this returns NaN instead, because a
            # multifurcation has no single pair of branches to measure and
            # failing the whole sweep over one node is unhelpful
            continue
        ends = [
            _walk_larger_subtree(tree, k, rank, dist, children, subtree_size)
            if mode == "bulk"
            else _walk_longest_path(tree, k, dist, children, depth_to_tip)
            for rank, k in enumerate(kids)
        ]
        v1 = coords[ends[0]] - coords[bp]
        v2 = coords[ends[1]] - coords[bp]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            angles[i] = 0.0
            continue
        cosang = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
        angles[i] = np.arccos(cosang)
    return angles


def angleBd_tree(tree: Tree, dist: int = 5) -> np.ndarray:
    """Branch angle measured ``dist`` nodes out, along the bulkier branch.

    :func:`angleB_tree` measures the angle at a branch point from its two
    immediate daughters, which makes it hypersensitive to how the
    reconstruction placed the very next point -- a single jittered node can
    swing it by tens of degrees. This instead walks ``dist`` nodes down each
    daughter first, so the angle describes where the branches actually
    *go* rather than how they leave.

    Parameters
    ----------
    tree : Tree
    dist : int, default 5
        How many nodes to walk before measuring. Larger values describe
        coarser branch geometry; the walk stops early at a terminal.

    Returns
    -------
    np.ndarray
        Angle [radians] per branch point, in ascending node order. ``NaN``
        at non-binary branch points, which have no single pair to measure.

    Notes
    -----
    Where a walk meets a further branch point it follows whichever daughter
    carries the larger subtree -- the "main" continuation. Compare
    :func:`angleBd2_tree`, which follows the longest path to a tip instead;
    the two disagree wherever a short bushy branch outweighs a long sparse
    one.

    From `new-functions/`, i.e. code the MATLAB maintainers had not yet
    folded into the toolbox proper. Neither variant has a documented
    default for ``dist``; 5 is this port's choice.
    """
    return _angle_at_branch_points(tree, dist, mode="bulk")


def angleBd2_tree(tree: Tree, dist: int = 5) -> np.ndarray:
    """Branch angle measured ``dist`` nodes out, along the longest path.

    As :func:`angleBd_tree`, but at intermediate branch points the walk
    follows the branch with the longest remaining path to a termination
    point rather than the one with the most nodes. That is the better choice
    when a branch's *reach* matters more than its bulk -- e.g. following the
    apparent trunk of a sparsely-sampled arbor.

    Parameters
    ----------
    tree : Tree
    dist : int, default 5

    Returns
    -------
    np.ndarray
        Angle [radians] per branch point; ``NaN`` at non-binary ones.
    """
    return _angle_at_branch_points(tree, dist, mode="reach")

def scale_tree(
    tree: Tree, fac=2.0, center: bool = True, scale_diameter: bool = True
) -> Tree:
    """Scale a tree's coordinates (and, by default, diameter) by ``fac``.

    Parameters
    ----------
    tree : Tree
    fac : float or tuple, default 2.0
        Scalar factor, or an ``(fx, fy, fz)`` triple for anisotropic scaling.
    center : bool, default True
        Scale about the **root's** position rather than the coordinate
        origin, so the root stays put.
    scale_diameter : bool, default True
        Also scale ``D``. For an anisotropic ``fac`` the diameter factor is
        the mean of ``fx`` and ``fy``, matching MATLAB.

    Returns
    -------
    Tree

    Notes
    -----
    The centre is :attr:`Tree.root`, not node 0. MATLAB's ``scale_tree.m``
    hardcodes ``tree.X(1)``, and this port transliterated that -- which is
    correct only for a tree that has been through ``sort_tree``. On a
    hand-built or freshly loaded tree it scaled about an arbitrary node and
    produced a plausible-looking but wrong result (Design Decision #48).
    """
    fac = np.atleast_1d(np.asarray(fac, dtype=float))
    root = tree.root
    ox, oy, oz = (
        (tree.X[root], tree.Y[root], tree.Z[root]) if center else (0.0, 0.0, 0.0)
    )
    X, Y, Z = tree.X - ox, tree.Y - oy, tree.Z - oz

    if fac.size > 1:
        X, Y, Z = X * fac[0], Y * fac[1], Z * fac[2]
        D = tree.D * fac[:2].mean() if scale_diameter else tree.D
    else:
        X, Y, Z = X * fac[0], Y * fac[0], Z * fac[0]
        D = tree.D * fac[0] if scale_diameter else tree.D

    return tree.with_coords(X=X + ox, Y=Y + oy, Z=Z + oz, D=D)


def tran_tree(tree: Tree, offset=None) -> Tree:
    """Translate a tree's coordinates.

    Parameters
    ----------
    tree : Tree
    offset : int or array_like, optional
        A **node index**, in which case the tree is shifted so that node
        lands on the origin; or an explicit ``(dx, dy[, dz])`` vector to
        translate *by*. Defaults to the root, i.e. **move the root to
        (0, 0, 0)** -- so on an already-centred tree the default is a no-op.

    Returns
    -------
    Tree

    Notes
    -----
    Verified against MATLAB (`tran_tree.m` run in Octave) in all three
    modes, max difference 2.7e-11 -- which is the precision of the reference
    values carried across, not a real discrepancy. MATLAB's default is
    ``DD = 1``; a *scalar* ``DD`` takes its ``tree.X - tree.X(DD)`` branch,
    so "per default sets tree root to origin" and "default DD = node 1" are
    the same statement, not two competing ones.
    """
    if offset is None:
        offset = tree.root
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


_PCA_AXIS_ORDER = {
    # mode -> which principal component supplies X, Y, Z
    "pcax": (0, 1, 2),
    "pcay": (1, 0, 2),
    "pcaz": (2, 1, 0),
}

_M3D_AXIS = {
    # mode -> (axis index to align onto, the other two axes, reference vector)
    "m3dx": (0, [1, 2], np.array([0.0, 1.0, 0.0])),
    "m3dy": (1, [2, 0], np.array([1.0, 0.0, 0.0])),
    "m3dz": (2, [0, 1], np.array([1.0, 0.0, 0.0])),
}


def _signed_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle a->b in degrees, signed by the cross product's component sum.

    This is MATLAB's ``sign(sum(cross(a, b))) * acosd(dot(a, b))`` verbatim.
    It is not a general signed angle -- summing the cross product's
    components only gives the right sign because both vectors have been
    flattened into a coordinate plane first -- but reproducing it exactly is
    the point.
    """
    cosang = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.sign(np.sum(np.cross(a, b))) * np.degrees(np.arccos(cosang)))


def _rotate_pca(tree: Tree, mode: str) -> Tree:
    """Replace coordinates by their principal-component scores."""
    root = tree.root
    XYZ = np.column_stack([tree.X, tree.Y, tree.Z])
    XYZ = XYZ - XYZ[root]  # MATLAB translates to the origin first
    centred = XYZ - XYZ.mean(axis=0)
    # scores, largest component first -- the same quantity MATLAB's `pca`
    # returns as its second output
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    coeff = vt.T
    # A principal component's sign is mathematically arbitrary, so SVD
    # implementations are free to differ -- and numpy's and MATLAB's do.
    # MATLAB's `pca` settles it by making each column's largest-magnitude
    # element positive; without matching that, the scores come out mirrored
    # (identical extents, coordinates off by the tree's whole width).
    dominant = np.argmax(np.abs(coeff), axis=0)
    coeff = coeff * np.sign(coeff[dominant, np.arange(coeff.shape[1])])
    scores = centred @ coeff
    ix, iy, iz = _PCA_AXIS_ORDER[mode]
    return tree.with_coords(X=scores[:, ix], Y=scores[:, iy], Z=scores[:, iz])


def _mean_axis_nodes(tree: Tree, nodes, exclude_regions) -> np.ndarray:
    """Which nodes define the mean axis.

    MATLAB hardcodes "every node whose region is not called ``axon``" -- an
    axon is long, thin and usually points somewhere unrelated to the
    dendritic field, so including it drags the mean axis off. Its docstring
    claims the ``DEG`` argument doubles as a node subset for this, but the
    ``-m3d`` branch never reads ``DEG``; ``nodes=`` is the parameter that
    promise should have been.
    """
    if nodes is not None:
        return np.asarray(nodes, dtype=int)
    mask = np.ones(tree.n_nodes, dtype=bool)
    for name in exclude_regions:
        if name in tree.rnames:
            mask &= tree.R != tree.rnames.index(name)
    return np.flatnonzero(mask) if mask.any() else np.arange(tree.n_nodes)


def _rotate_mean_axis(tree: Tree, mode: str, nodes, exclude_regions,
                      align_region) -> Tree:
    """Align a tree's mean dendritic axis onto one coordinate axis."""
    raxis, others, e_ref = _M3D_AXIS[mode]
    root = tree.root
    origin = np.array([tree.X[root], tree.Y[root], tree.Z[root]])
    tree = tran_tree(tree)  # work about the origin, restore at the end

    a = np.zeros(3)
    a[raxis] = 1.0
    subset = _mean_axis_nodes(tree, nodes, exclude_regions)
    d = list(others)

    # Two rotations bring the mean vector onto the target axis: each flattens
    # the mean into a plane and rotates within it, around a different axis.
    for _ in range(2):
        mXYZ = np.array([tree.X[subset].mean(), tree.Y[subset].mean(),
                         tree.Z[subset].mean()])
        mXYZ[d[0]] = 0.0
        norm = np.linalg.norm(mXYZ)
        if norm > 0:
            mXYZ /= norm
            rangle = np.zeros(3)
            rangle[d[0]] = _signed_angle_deg(a, mXYZ)
            tree = rot_tree(tree, rangle)
        d = d[::-1]

    # Then spin about the aligned axis so the arbor's widest spread lands in
    # the plane MATLAB expects (xy for x/y alignment, xz for z).
    XYZ = np.column_stack([tree.X[subset], tree.Y[subset], tree.Z[subset]])
    XYZ[:, raxis] = 0.0  # discard information along the axis just aligned
    cov = np.cov(XYZ, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    pc1 = eigvecs[:, order[0]]

    # PC sign is arbitrary, so take whichever of +-pc1 needs the smaller turn
    candidates = [_signed_angle_deg(e_ref, pc1), _signed_angle_deg(-e_ref, pc1)]
    rangle = np.zeros(3)
    rangle[raxis] = candidates[int(np.argmin(np.abs(candidates)))]
    tree = rot_tree(tree, rangle)

    if align_region is not None and raxis < 2:
        tree = _align_region_border(tree, align_region, others)

    return tran_tree(tree, origin)


def _align_region_border(tree: Tree, align_region, others) -> Tree:
    """Rotate so a region boundary lies flat.

    Layered tissue (cortex, dentate molecular layers) gives a tree natural
    horizontal landmarks: the surface where one region hands over to the
    next. Fitting a plane to the nodes on that boundary and levelling it
    puts the anatomy the way it is drawn in figures.

    Only defined for x/y alignment -- with the tree already lying along z
    there is no remaining degree of freedom to spend on this, which is why
    MATLAB guards it with ``raxis < 3``.
    """
    if isinstance(align_region, str):
        if align_region not in tree.rnames:
            raise ValueError(
                f"align_region {align_region!r} is not a region of this tree "
                f"({tree.rnames})"
            )
        index = tree.rnames.index(align_region)
    else:
        index = int(align_region)
    if index == 0:
        raise ValueError(
            "align_region must have a preceding region to form a border with"
        )

    idpar = idpar_tree(tree)
    on_border = np.flatnonzero((tree.R == index) & (tree.R[idpar] == index - 1))
    if len(on_border) < 3:
        warnings.warn(
            f"only {len(on_border)} node(s) lie between regions "
            f"{tree.rnames[index - 1]!r} and {tree.rnames[index]!r}; "
            f"need 3 to fit a plane -- alignment skipped",
            stacklevel=3,
        )
        return tree

    pts = np.column_stack([tree.X[on_border], tree.Y[on_border],
                           tree.Z[on_border]])
    _, _, vt = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
    normal = vt[2]  # smallest singular vector == plane normal

    axis = [ax for ax in others if ax != 2]
    if not axis:
        return tree
    plane = np.delete(normal, 2)
    plane = plane / np.linalg.norm(plane)
    if plane[0] < 0:
        plane = -plane

    rangle = np.zeros(3)
    rangle[axis[0]] = np.sign(plane[1]) * np.degrees(
        np.arccos(np.clip(np.dot([1.0, 0.0], plane), -1.0, 1.0))
    )
    return rot_tree(tree, rangle)


def rot_tree(tree: Tree, deg=(0.0, 0.0, 90.0), mode: str | None = None, *,
             nodes=None, exclude_regions=("axon",), align_region=None) -> Tree:
    """Rotate a tree, either by explicit angles or onto an automatic axis.

    Parameters
    ----------
    tree : Tree
    deg : float or tuple, default (0, 0, 90)
        Degrees of rotation. A scalar rotates in the XY plane; an
        ``(x, y[, z])`` tuple rotates about each axis in turn, x then y then
        z (see :func:`_rotation_matrix`). Ignored when ``mode`` is given.
    mode : str, optional
        Automatic alignment instead of explicit angles:

        - ``'pcaX'``/``'pcaY'``/``'pcaZ'`` -- replace coordinates by their
          principal-component scores, ordering the axes so the named one
          carries the largest geometric extent.
        - ``'m3dX'``/``'m3dY'``/``'m3dZ'`` -- "mean axis": rotate so the
          arbor's mean direction lies along the named axis, then spin about
          it so the widest spread falls in the expected plane.
    nodes : array_like, optional
        Which nodes define the mean axis for the ``m3d`` modes. Defaults to
        every node outside ``exclude_regions``.
    exclude_regions : tuple of str, default ('axon',)
        Regions ignored when computing the mean axis. An axon is long, thin
        and usually points somewhere unrelated to the dendritic field, so
        including it drags the axis off; MATLAB hardcodes this same
        exclusion.
    align_region : str or int, optional
        MATLAB's ``'-al'``. After ``m3d`` alignment, additionally level the
        boundary between this region and the one before it, so layered
        tissue sits horizontally. Only meaningful for ``m3dX``/``m3dY``.

    Returns
    -------
    Tree

    Notes
    -----
    Ported in Design Decision #56, reversing #20 (which deferred these as
    "niche"). ``m3d`` does **not** overload ``deg`` as a node subset the way
    MATLAB's docstring says it does -- the MATLAB ``-m3d`` branch never
    reads ``DEG`` at all, so that promise is unimplemented there. ``nodes=``
    is the parameter it should have been.
    """
    if mode is not None:
        key = mode.lower()
        if key in _PCA_AXIS_ORDER:
            return _rotate_pca(tree, key)
        if key in _M3D_AXIS:
            return _rotate_mean_axis(tree, key, nodes, exclude_regions,
                                     align_region)
        raise ValueError(
            f"unknown mode {mode!r}; expected one of "
            f"{sorted(_PCA_AXIS_ORDER) + sorted(_M3D_AXIS)}"
        )

    deg = np.atleast_1d(np.asarray(deg, dtype=float))
    if deg.size == 1:
        theta = np.radians(deg[0])
        RM = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
        XY = np.stack([tree.X, tree.Y], axis=1) @ RM
        return tree.with_coords(X=XY[:, 0], Y=XY[:, 1])

    if deg.size == 2:
        deg = np.append(deg, 0.0)
    RM = _rotation_matrix(*np.radians(deg))
    XYZ = np.stack([tree.X, tree.Y, tree.Z], axis=1) @ RM
    return tree.with_coords(X=XYZ[:, 0], Y=XYZ[:, 1], Z=XYZ[:, 2])

def flip_tree(tree: Tree, axis: str = "x") -> Tree:
    """Mirror a tree about its root along one axis.

    Parameters
    ----------
    tree : Tree
    axis : {'x', 'y', 'z'}, default 'x'

    Returns
    -------
    Tree
        A mirrored copy; the root keeps its position.

    Notes
    -----
    Mirrors about :attr:`Tree.root`, not node 0 -- see :func:`scale_tree`'s
    note and Design Decision #48.
    """
    root = tree.root
    if axis == "x":
        return tree.with_coords(X=2 * tree.X[root] - tree.X)
    if axis == "y":
        return tree.with_coords(Y=2 * tree.Y[root] - tree.Y)
    if axis == "z":
        return tree.with_coords(Z=2 * tree.Z[root] - tree.Z)
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
    tree = tree.with_coords(Z=tree.Z - tree.Z[tree.root])
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
    # Remember where the root was, work relative to it, then put it back at
    # the end. `with_coords` does the shift directly -- going through
    # `tran_tree` only to apply an offset we have already computed cost an
    # extra Tree allocation and a module-level dependency for nothing.
    root = tree.root
    x0, y0, z0 = tree.X[root], tree.Y[root], tree.Z[root]
    tree = tree.with_coords(X=tree.X - x0, Y=tree.Y - y0, Z=tree.Z - z0)
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
    return result.with_coords(X=result.X + x0, Y=result.Y + y0, Z=result.Z + z0)


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
        mask = sub_tree(tree, int(node), with_tree=False).mask
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
    sorted_tree = sort_tree(tree, by="lo")
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
