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


def _reindex_meta(meta: dict, order: np.ndarray, n_nodes: int) -> dict:
    """Carry ad-hoc :attr:`Tree.meta` entries through a reindex.

    An entry is treated as **per-node** -- and so reordered/subset with the
    nodes -- exactly when it is a sequence of length ``n_nodes``. Anything
    else (a scalar, a string, a dict, an array of some other length) is
    metadata about the cell as a whole and passes through untouched.

    Length is the only signal available: MATLAB's tree struct carries ad-hoc
    fields with no type information either, and `delete_tree`/`resample_tree`
    there make the same length-based decision. Being explicit about the rule
    is what keeps it from being surprising.
    """
    out = {}
    for key, value in meta.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (n_nodes,):
            out[key] = value[order]
        elif isinstance(value, (list, tuple)) and len(value) == n_nodes:
            out[key] = type(value)(value[i] for i in order)
        else:
            out[key] = value
    return out


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
    meta : dict
        Ad-hoc, caller-owned data travelling with the tree -- the Python
        stand-in for tacking an extra field onto MATLAB's tree struct
        (``tree.myfield = ...``), which the MATLAB toolbox does freely and
        which nothing in a ``__slots__`` class would otherwise allow.

        Nothing in the package reads or writes it; it exists so that
        per-node annotations (a synapse count, a channel density, an
        experimental label) survive the transforms. Entries whose length is
        ``n_nodes`` are **reordered and subset with the nodes** by
        :meth:`reindexed` -- and so by ``delete_tree``, ``sort_tree``,
        ``resample_tree`` and everything else built on it; entries of any
        other shape are treated as whole-cell metadata and copied across
        unchanged. See :func:`_reindex_meta`.

        ``.npz`` round-trips it (:func:`~pynetrees.io.save_npz`); the other
        formats have nowhere to put it and drop it.
    """

    __slots__ = (
        "dA", "X", "Y", "Z", "D", "R", "rnames", "name", "frustum",
        "Ri", "Gm", "Cm", "meta",
    )

    def __init__(
        self, dA, X, Y, Z, D, R, rnames, name: str = "", frustum: bool = False,
        Ri=None, Gm=None, Cm=None, meta=None,
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
        # copied, not aliased: two trees sharing one dict would let an edit
        # to a derived tree reach back into the one it came from
        self.meta = {} if meta is None else dict(meta)

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the tree."""
        return self.dA.shape[0]

    @property
    def root(self) -> int:
        """Index of the root node — the one node with no parent.

        Found by in-degree (``dA`` row sum), *never* by assuming index 0.
        After ``sort_tree``/``repair_tree`` the root does sit at 0, but a
        freshly loaded, hand-built or externally generated tree carries no
        such guarantee, and silently transforming around the wrong node
        produces a plausible-looking but wrong result rather than an error.

        This is the public face of Design Decision #10, promoted to a
        property in #48 precisely because the private ``_root_index`` helper
        lived in ``graphtheory`` and was awkward to reach from
        ``metrics``/``construct`` — which is how ``scale_tree``,
        ``flip_tree`` and ``cap_tree`` all ended up hardcoding ``[0]``.
        Reach for ``tree.root``; it is both the obvious and the correct
        thing to write.

        Raises
        ------
        ValueError
            If the tree has no root or more than one (i.e. it is empty,
            cyclic, or a disconnected forest).
        """
        roots = np.flatnonzero(np.asarray(self.dA.sum(axis=1)).ravel() == 0)
        if len(roots) != 1:
            raise ValueError(
                f"expected exactly one root, found {len(roots)}"
                + (" (tree is empty)" if self.n_nodes == 0 else "")
            )
        return int(roots[0])

    @property
    def total_length(self) -> float:
        """Summed segment length of the whole tree [um].

        Equivalent to ``len_tree(tree).sum()``, which is how MATLAB spells
        it. Deliberately **not** cached: ``Tree`` is mutable in place
        (``tree.X[5] = ...`` is legal and used by the editing functions), so
        a cache would go stale silently. The recomputation is well under a
        millisecond even on a 4000-node reconstruction.
        """
        from .metrics import len_tree

        return float(len_tree(self).sum())

    @property
    def total_surface(self) -> float:
        """Summed membrane surface area of the whole tree [um^2].

        Equivalent to ``surf_tree(tree).sum()``. Uncached, for the reason
        given on :attr:`total_length`.
        """
        from .metrics import surf_tree

        return float(surf_tree(self).sum())

    @property
    def total_volume(self) -> float:
        """Summed enclosed volume of the whole tree [um^3].

        Equivalent to ``vol_tree(tree).sum()``. Uncached, for the reason
        given on :attr:`total_length`.
        """
        from .metrics import vol_tree

        return float(vol_tree(self).sum())

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
        over as-is since they still index into it. Per-node :attr:`meta`
        entries move with their nodes (see :func:`_reindex_meta`).
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
            meta=_reindex_meta(self.meta, order, self.n_nodes),
        )

    #: Every constructor argument, in order. :meth:`copy` uses it to carry
    #: attributes across, so adding a field to ``__init__`` extends ``copy``
    #: (and every transform built on it) without a second edit -- which is
    #: how ``Ri``/``Gm``/``Cm`` came to be silently dropped by a dozen
    #: editing functions in the first place.
    _FIELDS = (
        "dA", "X", "Y", "Z", "D", "R", "rnames", "name", "frustum",
        "Ri", "Gm", "Cm", "meta",
    )

    def copy(self, **overrides) -> "Tree":
        """Return a copy of this tree with the named attributes replaced.

        The single way to build a *derived* tree: everything not named in
        ``overrides`` is carried over, so a transform cannot silently lose a
        field it never thought about. Used by the geometry transforms in
        ``metrics.py`` (``scale_tree``, ``tran_tree``, ``rot_tree``, ...) and
        the editing functions in ``edit.py``, which all return a **new** Tree
        rather than mutating in place or writing into a global trees array
        the way the MATLAB originals optionally do::

            moved   = tree.copy(X=tree.X + 10)
            renamed = tree.copy(name="shifted")
            rebuilt = tree.copy(dA=new_dA, X=x, Y=y, Z=z, D=d, R=r)

        Note that this is a **shallow** copy of the arrays: overriding ``X``
        gives the new tree the array you pass, while the arrays you do not
        override are shared with the original. That is what makes it cheap,
        and it is safe as long as transforms build new arrays instead of
        writing into the ones they were given -- which is the convention
        throughout the package.

        Raises
        ------
        TypeError
            If ``overrides`` names something that is not a Tree field --
            catching ``tree.copy(Xs=...)`` at the call rather than silently
            returning an unmodified tree.
        """
        unknown = set(overrides) - set(self._FIELDS)
        if unknown:
            raise TypeError(
                f"copy() got unexpected field(s) {sorted(unknown)}; "
                f"Tree fields are {list(self._FIELDS)}"
            )
        fields = {name: getattr(self, name) for name in self._FIELDS}
        fields.update(overrides)
        return Tree(**fields)

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
