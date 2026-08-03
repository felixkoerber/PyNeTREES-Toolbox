"""Core tree data structure and validation.

Python counterpart of the MATLAB tree struct (``dA``, ``X``, ``Y``, ``Z``,
``D``, ``R``, ``rnames``) and of ``IO/ver_tree.m``. See ``PORT_PLAN.md`` for
the indexing/sentinel conventions this module commits to:

- All node indices are 0-based (MATLAB is 1-based).
- ``NO_PARENT`` (``-1``) marks "this node has no parent", replacing MATLAB's
  overloaded use of the bare integer ``0`` (which is not available as a
  sentinel here since ``0`` is a valid 0-based node index).
- ``dA[i, j] == 1`` means node ``j`` is the parent of node ``i`` — same
  child-row/parent-column orientation as the MATLAB original, just reindexed.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse

NO_PARENT = -1


def _reindex_electrical(value, order: np.ndarray):
    """Slice a Ri/Gm/Cm value along with the rest of a reindex, if it's per-node.

    A scalar (or ``None``) applies uniformly regardless of node order, so it
    passes through unchanged; only a per-node array needs to move with its
    nodes, exactly like ``X``/``D``/etc.
    """
    if value is None or np.isscalar(value):
        return value
    return np.asarray(value)[order]


class Tree:
    """A neuronal tree: nodes plus directed parent adjacency and metrics.

    Attributes
    ----------
    dA : scipy.sparse.csr_matrix, shape (n_nodes, n_nodes)
        Directed adjacency, ``dA[i, j] == 1`` iff node ``j`` is node ``i``'s
        parent. The root's row is all zero.
    X, Y, Z : np.ndarray, shape (n_nodes,)
        Node coordinates.
    D : np.ndarray, shape (n_nodes,)
        Node diameters.
    R : np.ndarray, shape (n_nodes,)
        0-based index into ``rnames`` giving each node's region.
    rnames : list[str]
        Region names, indexed by ``R``.
    name : str
        Optional human-readable label.
    frustum : bool
        If True, segments are treated as tapering cones (frustums) between a
        node and its parent's diameter rather than uniform cylinders, in
        :func:`surf_tree`, :func:`vol_tree` and :func:`cvol_tree`.
    Ri : float | np.ndarray | None
        Axial resistivity [Ohm*cm], scalar or one value per node (the
        segment ending at that node). Required by every function in
        ``electrotonics.py``; no universal default exists (unlike
        ``frustum``), so it's ``None`` until set explicitly.
    Gm : float | np.ndarray | None
        Specific membrane conductance [S/cm^2], scalar or per-node. Same
        "no default" reasoning as ``Ri``.
    Cm : float | np.ndarray | None
        Specific membrane capacitance [uF/cm^2], scalar or per-node. Only
        needed by the time-stepping functions (``LIF_tree``/``AdExLIF_tree``).
    """

    __slots__ = (
        "dA", "X", "Y", "Z", "D", "R", "rnames", "name", "frustum",
        "Ri", "Gm", "Cm",
    )

    def __init__(
        self, dA, X, Y, Z, D, R, rnames, name: str = "", frustum: bool = False,
        Ri=None, Gm=None, Cm=None,
    ):
        self.dA = sparse.csr_matrix(dA)
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float)
        self.Z = np.asarray(Z, dtype=float)
        self.D = np.asarray(D, dtype=float)
        self.R = np.asarray(R, dtype=int)
        self.rnames = list(rnames)
        self.name = name
        self.frustum = frustum
        self.Ri = Ri
        self.Gm = Gm
        self.Cm = Cm

    @property
    def n_nodes(self) -> int:
        return self.dA.shape[0]

    def validate(self, quiet: bool = False) -> list[str]:
        """Convenience wrapper around :func:`ver_tree`."""
        return ver_tree(self, quiet=quiet)

    def region_index(self, name: str) -> int:
        """The ``R`` value for region ``name``.

        Raises :class:`KeyError` naming the available regions if ``name``
        isn't one of them -- MATLAB's equivalent
        (``find(strcmp(tree.rnames, name))``) silently returns empty, which
        then makes the *next* line's comparison quietly match nothing
        instead of failing.
        """
        try:
            return self.rnames.index(name)
        except ValueError:
            raise KeyError(
                f"no region {name!r} in tree {self.name!r}; "
                f"available regions: {self.rnames}"
            ) from None

    def region_mask(self, *names: str) -> np.ndarray:
        """Boolean mask of nodes belonging to any of ``names``."""
        wanted = [self.region_index(n) for n in names]
        return np.isin(self.R, wanted)

    def region_nodes(self, *names: str) -> np.ndarray:
        """Indices of nodes belonging to any of ``names``.

        Replaces the MATLAB idiom
        ``find(tree.R == find(strcmp(tree.rnames, 'soma')))``, which appears
        dozens of times across the bundled GC model scripts::

            soma = tree.region_nodes("soma")
            dend = tree.region_nodes("adendIML", "adendMML", "adendOML")
        """
        return np.flatnonzero(self.region_mask(*names))

    def reindexed(self, order, name: str | None = None) -> "Tree":
        """Return a new Tree with nodes reordered/subset per ``order``.

        ``order[i]`` is the *old* node index placed at new position ``i``.
        Works both for permutations (``len(order) == n_nodes``, e.g. a
        canonical sort) and subsets (``len(order) < n_nodes``, e.g. cutting
        out a subtree). ``rnames`` is shared unchanged; ``R`` values carry
        over as-is since they still index into it.
        """
        order = np.asarray(order, dtype=int)
        dA = self.dA.tocsr()[order][:, order]
        return Tree(
            dA=dA,
            X=self.X[order],
            Y=self.Y[order],
            Z=self.Z[order],
            D=self.D[order],
            R=self.R[order],
            rnames=self.rnames,
            name=self.name if name is None else name,
            frustum=self.frustum,
            Ri=_reindex_electrical(self.Ri, order),
            Gm=_reindex_electrical(self.Gm, order),
            Cm=_reindex_electrical(self.Cm, order),
        )

    def with_coords(
        self,
        X: np.ndarray | None = None,
        Y: np.ndarray | None = None,
        Z: np.ndarray | None = None,
        D: np.ndarray | None = None,
        name: str | None = None,
    ) -> "Tree":
        """Return a copy with some coordinate/diameter arrays replaced.

        Used by the geometry transforms in ``metrics.py`` (``scale_tree``,
        ``tran_tree``, ``rot_tree``, ...), which all return a *new* Tree
        rather than mutating in place or relying on a global trees array the
        way the MATLAB originals optionally do.
        """
        return Tree(
            dA=self.dA,
            X=self.X if X is None else X,
            Y=self.Y if Y is None else Y,
            Z=self.Z if Z is None else Z,
            D=self.D if D is None else D,
            R=self.R,
            rnames=self.rnames,
            name=self.name if name is None else name,
            frustum=self.frustum,
            Ri=self.Ri,
            Gm=self.Gm,
            Cm=self.Cm,
        )

    def __repr__(self) -> str:
        # regions are the field you most often need to look up before doing
        # anything else (they drive region_nodes, plotting, and every
        # region-dependent biophysical setting), so surface them here rather
        # than making callers print tree.rnames separately every time
        regions = ", ".join(self.rnames) if self.rnames else "-"
        return f"Tree(name={self.name!r}, n_nodes={self.n_nodes}, regions=[{regions}])"

    def __len__(self) -> int:
        return self.n_nodes


def ver_tree(tree: Tree, quiet: bool = False) -> list[str]:
    """Verify internal consistency of a :class:`Tree`.

    Mirrors ``IO/ver_tree.m``: never raises, collects every problem found and
    (unless ``quiet``) emits each as a :func:`warnings.warn` call, matching
    the MATLAB original's "warn, don't fail" behavior so that intentionally
    incomplete trees mid-pipeline (e.g. before a future ``repair_tree``) don't
    break callers that only want a health check.

    Returns
    -------
    list[str]
        Human-readable problem descriptions; empty if the tree is well-formed.
    """
    issues: list[str] = []

    if tree.dA.ndim != 2:
        issues.append("adjacency matrix dA has incorrect dimensions")
        n = None
    elif tree.dA.shape[0] != tree.dA.shape[1]:
        issues.append("adjacency matrix dA is not square")
        n = None
    else:
        n = tree.dA.shape[0]

    for name in ("X", "Y", "Z", "D"):
        arr = getattr(tree, name)
        if arr.ndim != 1:
            issues.append(f"{name} is not a 1-D array")
        elif n is not None and arr.shape[0] != n:
            issues.append(f"{name} size ({arr.shape[0]}) does not match dA ({n})")

    if tree.R.ndim != 1:
        issues.append("R is not a 1-D array")
    elif n is not None and tree.R.shape[0] != n:
        issues.append(f"R size ({tree.R.shape[0]}) does not match dA ({n})")
    elif tree.R.size and (tree.R.min() < 0 or tree.R.max() >= len(tree.rnames)):
        issues.append("R contains an index out of range for rnames")

    if not quiet:
        for issue in issues:
            warnings.warn(f"Tree {tree.name!r}: {issue}", stacklevel=2)

    return issues
