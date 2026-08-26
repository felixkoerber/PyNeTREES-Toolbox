"""Passive electrotonic properties and steady-state cable-theory analysis.

Ported from ``treestoolbox-master/electrotonics/``. Every function here
needs ``Ri`` (axial resistivity, Ohm*cm) and/or ``Gm`` (specific membrane
conductance, S/cm^2) set on the tree first (:attr:`Tree.Ri`/:attr:`Tree.Gm`,
scalar or one value per node) -- these are physical modeling choices with no
universal default, exactly as in MATLAB (every ``check_*`` fixture sets
``tree.Ri``/``tree.Gm``/``tree.Cm`` by hand before calling anything in this
module; there's no auto-populated default to fall back on here either).

The conductance-matrix construction in :func:`M_tree` is ported as a literal
transliteration of the sparse matrix algebra (0-based reindex only), not
re-derived from cable-theory first principles -- same reasoning as Phase 2's
``LO_tree`` (PORT_STATUS.md's Phase 2 table: "ported faithfully ... deliberately
*not* re-derived"): getting an established, widely-used (Cuntz et al.)
formula's index conventions "improved" from scratch is a real risk of
introducing a subtle sign/index bug that a faithful port avoids. Verified
instead via a property every valid conductance-Laplacian must have
regardless of implementation details: with no membrane leak, every row of
the *axial* part sums to zero (Kirchhoff's current law, no node accumulates
charge from pure axial coupling) -- see ``tests/test_electrotonics.py``.

MATLAB's ``'-s'`` show option is dropped throughout (Design Decision #33):
every function here returns a plain array/matrix; plotting it is exactly
what Phase 7's ``plot_tree(tree, scalars=result)`` already does.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from ._population import accepts_population
from ._empty import empty_safe
from .core import Tree
from .graphtheory import Pvec_tree
from .metrics import cvol_tree, len_tree, surf_tree

__all__ = [
    "M_tree",
    "gi_tree",
    "gm_tree",
    "lambda_tree",
    "elen_tree",
    "cgin_tree",
    "M_atten_tree",
    "sse_tree",
    "syn_tree",
    "loop_tree",
    "ssecat_tree",
    "syncat_tree",
    "LIF_tree",
    "AdExLIF_tree",
]


def _empty_matrix(tree):
    """A node-by-node matrix over no nodes."""
    return np.empty((0, 0))


def _no_compartments(tree):
    """A cell with no cable has no electrotonic compartments -- 0, and an
    `int` like the populated case, not `0.0`."""
    return 0


def _empty_lif(tree):
    """No nodes to integrate, so no voltage trace and no spikes. The
    voltage keeps its two axes so it still stacks with a real run."""
    return np.empty((0, 0)), np.empty(0)


def _empty_adex(tree):
    """As :func:`_empty_lif`, plus the adaptation trace."""
    return np.empty((0, 0)), np.empty(0), np.empty((0, 0))



def _require(tree: Tree, attr: str):
    value = getattr(tree, attr)
    if value is None:
        raise ValueError(
            f"tree {tree.name!r} has no {attr} set -- electrotonic functions "
            f"need a physical value, e.g. tree.{attr} = 100 (scalar) or a "
            f"length-{tree.n_nodes} per-node array; see Tree's docstring"
        )
    return value


def _onehot_or_array(value, n: int) -> np.ndarray:
    """MATLAB's dual-purpose convention for ``ge``/``gi``/``I`` arguments:

    a scalar means "put a canonical unit value (1) at this 0-based node
    index"; an array-like means the full per-node values directly. Ported
    as-is (not replaced with an explicit separate "index" parameter) since
    it's used consistently across every function in this module.
    """
    if value is None:
        return np.zeros(n, dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.size == 1:
        vec = np.zeros(n, dtype=float)
        vec[int(arr.reshape(()))] = 1.0
        return vec
    return arr


@accepts_population
@empty_safe(_empty_matrix)
def M_tree(tree: Tree) -> sparse.csr_matrix:
    """Conductance matrix of the tree's equivalent electric circuit [uS].

    Combines axial (inter-compartment) and membrane conductances into one
    sparse NxN matrix, the basis for :func:`sse_tree`, :func:`syn_tree` and
    every other function in this module. Requires ``tree.Ri`` and
    ``tree.Gm``.
    """
    Ri = _require(tree, "Ri")
    Gm = _require(tree, "Gm")
    dA = tree.dA
    N = tree.n_nodes

    surf = surf_tree(tree) / 1e8  # um^2 -> cm^2
    cvol = cvol_tree(tree) * 1e4  # 1/um -> 1/cm

    Msurf = sparse.diags(surf)
    Mlov = sparse.diags(1.0 / cvol)

    # symmetric graph Laplacian of the axial coupling (see module docstring)
    lA = dA @ Mlov + Mlov @ dA.T
    interm = np.asarray(lA.sum(axis=0)).ravel()
    Milov = sparse.diags(interm) - lA

    inv_Ri = 1.0 / Ri if np.ndim(Ri) == 0 else (1.0 / np.asarray(Ri)).reshape(-1, 1)
    Mgi = Milov.multiply(inv_Ri)
    Mgm = Msurf.multiply(Gm)
    M = (Mgm + Mgi) * 1_000_000

    return sparse.csr_matrix(M)


@accepts_population
@empty_safe("nodes")
def gi_tree(tree: Tree) -> np.ndarray:
    """Axial conductance of every segment [S]. Requires ``tree.Ri``."""
    Ri = _require(tree, "Ri")
    Hlov = 1.0 / (cvol_tree(tree) * 1e4)  # [cm]
    return Hlov / Ri


@accepts_population
@empty_safe("nodes")
def gm_tree(tree: Tree) -> np.ndarray:
    """Membrane conductance of every segment [S]. Requires ``tree.Gm``."""
    Gm = _require(tree, "Gm")
    return Gm * surf_tree(tree) / 1e8  # um^2 -> cm^2


@accepts_population
@empty_safe("nodes")
def lambda_tree(tree: Tree) -> np.ndarray:
    """Length constant of every segment [cm]. Requires ``tree.Ri``/``tree.Gm``."""
    Ri = _require(tree, "Ri")
    Gm = _require(tree, "Gm")
    return np.sqrt((tree.D / 4) / (10000 * Gm * Ri))


@accepts_population
@empty_safe("nodes")
def elen_tree(tree: Tree) -> np.ndarray:
    """Electrotonic length of every segment (length / lambda), unitless."""
    return len_tree(tree) / lambda_tree(tree) / 10000  # um -> cm


@accepts_population
@empty_safe("zero")
def cgin_tree(tree: Tree) -> float:
    """Collapsed (point-neuron) input conductance of the whole tree [S].

    Requires ``tree.Gm``, taken as a single scalar specific membrane
    conductance representative of the whole cell.
    """
    Gm = _require(tree, "Gm")
    total_surf = surf_tree(tree).sum() / 1e8  # um^2 -> cm^2
    return Gm * total_surf


@accepts_population
@empty_safe(_empty_matrix)
def sse_tree(tree: Tree, I: float | np.ndarray | None = None) -> np.ndarray:
    """Steady-state electrotonic signature: potential [mV] per node per input.

    With ``I=None`` (default), returns the full NxN matrix whose column ``i``
    is the potential distribution from injecting 1 nA at node ``i`` (the
    diagonal is each node's local input resistance). A scalar ``I`` injects
    1 nA at that 0-based node index (returning one column); an explicit
    per-node array injects those exact currents.
    """
    M = M_tree(tree).tocsc()
    N = tree.n_nodes
    if I is None:
        return np.linalg.inv(M.toarray())
    rhs = _onehot_or_array(I, N)
    return sparse_linalg.spsolve(M, rhs)


@accepts_population
@empty_safe(_no_compartments)
def M_atten_tree(tree: Tree, thr: float = 0.13995) -> int:
    """Number of electrotonically distinct compartments in a tree.

    Thresholds the steady-state matrix from :func:`sse_tree` at ``thr``
    times its maximum, giving a boolean "these two nodes see each other"
    relation, then counts how many separate runs of nodes that relation
    breaks the tree into. One compartment means the whole cell is
    electrotonically compact -- current injected anywhere is felt
    everywhere; more means the arbor behaves as several semi-independent
    units.

    Parameters
    ----------
    tree : Tree
        Needs ``Ri`` and ``Gm`` set (see :func:`M_tree`).
    thr : float, default 0.13995
        Fraction of the largest steady-state response above which two nodes
        count as coupled. MATLAB's default, carried over unchanged; it is
        not derived from anything in the source, so treat it as a
        convention rather than a principled cutoff.

    Returns
    -------
    int
        Compartment count, at least 1.

    Notes
    -----
    Cost is dominated by ``sse_tree``'s full N x N inverse, so this is
    O(n^3) and not something to sweep over a population without thought.

    **MATLAB ships this function with no documentation at all** -- no header
    comment, no description of the return value, and a stray ``clf;``
    (clear-figure) left mid-computation. The description above is derived
    from reading what the code does. The one behavioural difference is that
    the stray ``clf`` is not reproduced: a metrics function should not wipe
    the caller's current figure.
    """
    sse = sse_tree(tree)
    coupled = sse > thr * sse.max()

    # For each node, fill in the span between the first and last node it
    # couples to: MATLAB does this to turn a possibly ragged coupling
    # pattern into contiguous blocks along the (sorted) node order.
    n = tree.n_nodes
    blocked = np.zeros((n, n), dtype=bool)
    for column in range(n):
        rows = np.flatnonzero(coupled[:, column])
        if len(rows) == 0:
            continue
        blocked[rows[0] : rows[-1] + 1, rows[0] : rows[-1] + 1] = True

    # Count runs of nodes that fall inside some block: each maximal run is
    # one compartment.
    on_diagonal = np.diag(blocked).astype(int)
    starts = np.diff(on_diagonal, prepend=0)
    starts[starts == -1] = 0
    runs = np.cumsum(starts) * on_diagonal
    return int(runs.max()) + 1 if runs.size else 1


@accepts_population
@empty_safe(_empty_matrix)
def syn_tree(
    tree: Tree,
    ge: float | np.ndarray | None = None,
    gi: float | np.ndarray | None = None,
    Ee: float = 60.0,
    Ei: float = -20.0,
    I: float | np.ndarray | None = None,
) -> np.ndarray:
    """Steady-state potential [mV] per node under synaptic + current input.

    ``ge``/``gi`` are per-node synaptic conductances [uS] (scalar = inject a
    canonical unit conductance at that 0-based node index, matching
    :func:`sse_tree`'s ``I`` convention); ``Ee``/``Ei`` their reversal
    potentials [mV]; ``I`` an additional current injection [nA].
    """
    M = M_tree(tree)
    N = tree.n_nodes
    ge_v = _onehot_or_array(ge, N)
    gi_v = _onehot_or_array(gi, N)
    I_v = _onehot_or_array(I, N)

    M2 = (M + sparse.diags(ge_v) + sparse.diags(gi_v)).tocsc()
    rhs = ge_v * Ee + gi_v * Ei + I_v
    return sparse_linalg.spsolve(M2, rhs)


def loop_tree(
    tree: Tree,
    inodes1: int | np.ndarray,
    inodes2: int | np.ndarray,
    gelsyn: float | np.ndarray = 1.0,
) -> sparse.csr_matrix:
    """Conductance matrix with extra electrical-synapse loops added.

    Adds a conductance ``gelsyn`` [uS] directly between each
    ``(inodes1[k], inodes2[k])`` pair of 0-based node indices, on top of
    :func:`M_tree`'s ordinary tree connectivity -- the only way to represent
    a non-tree (loopy) circuit in this data model.
    """
    M = M_tree(tree).tolil()
    inodes1 = np.atleast_1d(np.asarray(inodes1, dtype=int))
    inodes2 = np.atleast_1d(np.asarray(inodes2, dtype=int))
    gelsyn_arr = np.asarray(gelsyn, dtype=float)
    if gelsyn_arr.size == 1:
        gelsyn_arr = np.full(inodes1.shape, float(gelsyn_arr))

    for n1, n2, g in zip(inodes1, inodes2, gelsyn_arr):
        M[n1, n2] -= g
        M[n2, n1] -= g
        M[n1, n1] += g
        M[n2, n2] += g

    return M.tocsr()


def _block_diag_M(trees: list[Tree]) -> tuple[sparse.spmatrix, np.ndarray]:
    sizes = np.array([t.n_nodes for t in trees])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    MM = sparse.block_diag([M_tree(t) for t in trees], format="lil")
    return MM, offsets


def ssecat_tree(
    trees: list[Tree],
    inodes1: int | np.ndarray,
    inodes2: int | np.ndarray,
    gelsyn: float | np.ndarray = 1.0,
    I: float | np.ndarray | None = None,
) -> np.ndarray:
    """:func:`sse_tree` for several trees joined by electrical synapses.

    ``trees`` are combined into one block-diagonal conductance matrix first
    (no coupling at all between them), then ``inodes1``/``inodes2`` (0-based
    node indices *into the concatenated system*, i.e. offset by the
    cumulative node counts of the preceding trees -- same convention as
    :func:`loop_tree`, generalized across trees) add electrical-synapse
    loops between them.
    """
    MM, offsets = _block_diag_M(trees)
    N = int(offsets[-1])

    inodes1 = np.atleast_1d(np.asarray(inodes1, dtype=int))
    inodes2 = np.atleast_1d(np.asarray(inodes2, dtype=int))
    gelsyn_arr = np.asarray(gelsyn, dtype=float)
    if gelsyn_arr.size == 1:
        gelsyn_arr = np.full(inodes1.shape, float(gelsyn_arr))

    for n1, n2, g in zip(inodes1, inodes2, gelsyn_arr):
        MM[n1, n2] -= g
        MM[n2, n1] -= g
        MM[n1, n1] += g
        MM[n2, n2] += g

    MM = MM.tocsc()
    if I is None:
        return np.linalg.inv(MM.toarray())
    rhs = _onehot_or_array(I, N)
    return sparse_linalg.spsolve(MM, rhs)


def syncat_tree(
    trees: list[Tree],
    inodes1: int | np.ndarray,
    inodes2: int | np.ndarray,
    gelsyn: float | np.ndarray = 1.0,
    ge: float | np.ndarray | None = None,
    gi: float | np.ndarray | None = None,
    Ee: float = 60.0,
    Ei: float = -20.0,
    I: float | np.ndarray | None = None,
) -> np.ndarray:
    """:func:`syn_tree` for several trees joined by electrical synapses.

    Same tree-concatenation convention as :func:`ssecat_tree`; ``ge``/``gi``/
    ``Ee``/``Ei``/``I`` behave exactly as in :func:`syn_tree`, indexed into
    the concatenated system.
    """
    MM, offsets = _block_diag_M(trees)
    N = int(offsets[-1])

    inodes1 = np.atleast_1d(np.asarray(inodes1, dtype=int))
    inodes2 = np.atleast_1d(np.asarray(inodes2, dtype=int))
    gelsyn_arr = np.asarray(gelsyn, dtype=float)
    if gelsyn_arr.size == 1:
        gelsyn_arr = np.full(inodes1.shape, float(gelsyn_arr))

    for n1, n2, g in zip(inodes1, inodes2, gelsyn_arr):
        MM[n1, n2] -= g
        MM[n2, n1] -= g
        MM[n1, n1] += g
        MM[n2, n2] += g

    ge_v = _onehot_or_array(ge, N)
    gi_v = _onehot_or_array(gi, N)
    I_v = _onehot_or_array(I, N)

    MMg = (MM + sparse.diags(ge_v) + sparse.diags(gi_v)).tocsc()
    rhs = ge_v * Ee + gi_v * Ei + I_v
    return sparse_linalg.spsolve(MMg, rhs)


# ---------------------------------------------------------------------------
# time-stepping (integrate-and-fire) simulations
# ---------------------------------------------------------------------------


def _default_time() -> np.ndarray:
    return np.linspace(0.0, 1000.0, 10001)  # [ms], matches MATLAB's 0:0.1:1000


def _M_and_capacitance(tree: Tree, dt: float) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Shared setup for :func:`LIF_tree`/:func:`AdExLIF_tree`: the passive
    conductance matrix plus an implicit-Euler capacitive term. Requires
    ``tree.Cm`` in addition to ``M_tree``'s ``Ri``/``Gm``.
    """
    Cm = _require(tree, "Cm")
    Msurf = sparse.diags(surf_tree(tree) / 1e8)  # um^2 -> cm^2
    Mcm = Msurf.multiply(Cm) / dt  # uF, given time step dt in seconds
    M = (M_tree(tree) + Mcm).tocsr()
    Mcm_vec = np.asarray(Mcm.diagonal()).ravel()
    return M, Mcm_vec


@accepts_population
@empty_safe(_empty_lif)
def LIF_tree(
    tree: Tree,
    time: np.ndarray | None = None,
    ge: np.ndarray | None = None,
    gi: np.ndarray | None = None,
    Ee: float = 60.0,
    Ei: float = -20.0,
    I: np.ndarray | None = None,
    iroot: int = 0,
    thr: float = 10.0,
    vreset: float = 0.0,
    Aspike: float = 75.0,
    partial_reset: bool = False,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Leaky integrate-and-fire simulation over the tree's full morphology.

    Implicit-Euler time-steps the passive cable equation (:func:`M_tree` plus
    a capacitive term from ``tree.Cm``) under synaptic (``ge``/``gi``,
    reversal potentials ``Ee``/``Ei``) and current (``I``) input, generating
    a spike (recorded in ``sp``, seconds) whenever node ``iroot``'s potential
    crosses ``thr``. ``ge``/``gi``/``I`` are ``(n_nodes, len(time))`` arrays
    (default: all zero -- purely passive).

    With ``partial_reset=False`` (default), a spike resets *every* node to
    ``vreset``. With ``partial_reset=True``, nodes are reset in proportion to
    their path distance from the root (a sigmoid of :func:`Pvec_tree`'s
    cumulative path length, ``lambda=100``, ``xoffset=600``, matching
    MATLAB): distal nodes keep more of their pre-spike potential than
    proximal ones, rather than every node snapping to the same value.

    Dropped: MATLAB's ``Vzone`` parameter, which is parsed but only ever
    referenced inside a commented-out line -- a confirmed dead parameter,
    not a real part of the reset dynamics (see MATLAB_TOOLBOX_BUGS.md).
    Also dropped: the docstring-vs-code mismatch where MATLAB's header
    comment claims options ``'-s'``/``'-p'`` but the actual binary flags
    parsed are ``'-t'``/``'-e'`` -- replaced here with explicit,
    correctly-named ``partial_reset``/``verbose`` keywords (Design
    Decision 1).
    """
    N = tree.n_nodes
    time = _default_time() if time is None else np.asarray(time, dtype=float)
    T = time.size
    dt = (time[1] - time[0]) / 1000.0  # ms -> s

    ge = np.zeros((N, T)) if ge is None else np.asarray(ge, dtype=float)
    gi = np.zeros((N, T)) if gi is None else np.asarray(gi, dtype=float)
    I = np.zeros((N, T)) if I is None else np.asarray(I, dtype=float)

    M, Mcm_vec = _M_and_capacitance(tree, dt)

    if partial_reset:
        Pvec = Pvec_tree(tree, len_tree(tree))
        plset = 1.0 / (1.0 + np.exp(-(Pvec - 600.0) / 100.0))
    else:
        plset = np.zeros(N)

    v = np.zeros((N, T))
    sp = []
    for k in range(T - 1):
        if verbose and k % 100 == 0:
            print(time[k])

        M1 = (M + sparse.diags(ge[:, k]) + sparse.diags(gi[:, k])).tocsc()
        rhs = ge[:, k] * Ee + gi[:, k] * Ei + I[:, k] + v[:, k] * Mcm_vec
        v[:, k + 1] = sparse_linalg.spsolve(M1, rhs)

        if v[iroot, k + 1] >= thr:
            v[iroot, k] = Aspike  # cosmetic spike upstroke marker
            v0 = v[:, k + 1].copy()
            v[:, k + 1] = vreset + (v0 - vreset) * plset
            sp.append(k * dt)

    return v, np.array(sp)


@accepts_population
@empty_safe(_empty_adex)
def AdExLIF_tree(
    tree: Tree,
    time: np.ndarray | None = None,
    I: np.ndarray | None = None,
    ge: np.ndarray | None = None,
    gi: np.ndarray | None = None,
    Ee: float = 60.0,
    Ei: float = -20.0,
    iroot: int = 0,
    EL: float = 0.0,
    DeltaT: float = 2.0,
    Vt: float = 10.0,
    thr: float = 80.0,
    vreset: float = 2.0,
    Aspike: float = 110.0,
    tauw: float = 0.4,
    a: float = 0.0,
    b: float = 1e-6,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adaptive exponential LIF simulation over the tree's full morphology.

    Same passive-cable time-stepping as :func:`LIF_tree`, plus an
    exponential spike-generating current at node ``iroot`` and an adaptation
    variable ``w`` (leaky with time constant ``tauw``, subthreshold-coupled
    via ``a``, incremented by ``b`` on every spike) -- the standard AdEx
    mechanism. Reset is a hard clip (every node above ``vreset`` is pulled
    down to it exactly) rather than :func:`LIF_tree`'s optional
    distance-weighted partial reset: a genuinely different modeling choice,
    not just a different default, which is why this stays a separate
    function rather than one more flag on :func:`LIF_tree` (the MATLAB todo
    list suggests consolidating the two; the shared :func:`M_tree` +
    capacitance setup here *is* factored out via ``_M_and_capacitance``, but
    forcing the reset/threshold logic itself into one function would risk
    a third, subtly-wrong behavior for a modest deduplication gain).

    Returns the full ``(n_nodes, len(time))`` voltage and adaptation traces
    (``v``, ``w``) plus spike times ``sp`` [s] -- MATLAB's version instead
    hardcodes its returned ``v`` to node index 1 regardless of ``iroot``,
    which silently returns the wrong node's trace whenever ``iroot != 1``; a
    confirmed bug (see MATLAB_TOOLBOX_BUGS.md), not reproduced here. Also
    dropped: MATLAB's ``Vrest`` parameter, defaulted onto the tree but never
    actually referenced by the dynamics -- another confirmed dead parameter.
    """
    N = tree.n_nodes
    time = _default_time() if time is None else np.asarray(time, dtype=float)
    T = time.size
    dt = (time[1] - time[0]) / 1000.0  # ms -> s

    I = np.zeros((N, T)) if I is None else np.asarray(I, dtype=float)
    ge = np.zeros((N, T)) if ge is None else np.asarray(ge, dtype=float)
    gi = np.zeros((N, T)) if gi is None else np.asarray(gi, dtype=float)

    M, Mcm_vec = _M_and_capacitance(tree, dt)

    v = np.zeros((N, T))
    w = np.zeros((N, T))
    sp = []
    for k in range(T - 1):
        if verbose and k % 500 == 0:
            print(time[k])

        M1 = (M + sparse.diags(ge[:, k]) + sparse.diags(gi[:, k])).tocsc()

        w[:, k + 1] = (a * (v[:, k] - EL) - w[:, k]) / tauw * dt + w[:, k]

        rhs = ge[:, k] * Ee + gi[:, k] * Ei + I[:, k] - w[:, k] + v[:, k] * Mcm_vec
        v[:, k + 1] = sparse_linalg.spsolve(M1, rhs)

        v[iroot, k + 1] += DeltaT * np.exp((v[iroot, k] - Vt) / DeltaT)

        if v[iroot, k + 1] >= thr:
            v[iroot, k] = Aspike
            above = v[:, k + 1] > vreset
            v[above, k + 1] = vreset
            w[:, k + 1] = w[:, k] + b
            sp.append(k * dt)

    return v, np.array(sp), w
