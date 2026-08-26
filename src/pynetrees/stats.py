"""Statistics, comparison, and morphometric analyses across trees (Phase 9).

Ported from ``treestoolbox-master/metrics/``. ``stats_tree`` is the one
substantial redesign here: MATLAB's version returns a nested struct of cell
arrays (``gstats``/``dstats``, indexed by group then tree then, for
per-branch fields, again by branch), which is exactly the "roll your own
container" pattern Design Decision 5 already ruled out in favor of
``pandas.DataFrame`` for population-level tooling. This port instead
returns three (four with ``extras=True``) tidy, long-format DataFrames --
see :func:`stats_tree`'s docstring.

MATLAB's ``dstats_tree`` (a large, ``stats_tree``-specific matplotlib-style
multi-panel figure) is **not** ported: it's pure visualization of exactly
the data ``stats_tree`` now returns as ordinary DataFrames, which plot
directly via ``df.hist()``/``seaborn``/etc. -- reproducing its specific
panel layout would just be a second, parallel plotting API for data a
caller already has in the most flexible form available (same reasoning as
Phase 7 dropping every function's inline ``'-s'`` option, Design Decision
#33).

Everything Phase 9 originally deferred (Design Decision #35) has since been
ported: ``boundary_tree``/``convexity_tree``/``share_boundary_tree`` and
``hull_tree`` live in ``density.py`` (B1), ``M_atten_tree`` in
``electrotonics.py``, and ``r_mc_tree``/``dissectSholl_tree`` are below
(B2). See Design Decision #61 for what each one had to diverge on.

The one exception stands: MATLAB's ``dstats_tree`` is still not ported, for
the reason above -- it is a figure, not an analysis.

**Verification status.** ``sholl_tree``, ``vonMises_tree`` and ``bf_tree``
are checked against MATLAB numerically. ``r_mc_tree`` and
``dissectSholl_tree`` are **not**, and cannot be on this machine: both bottom
out in MATLAB's built-in ``boundary()``, which Octave does not implement, so
the differential harness used everywhere else has nothing to run. They are
tested against the properties their statistics must have and against
reference point sets with independently known answers (a uniform cloud must
score R = 1, a lattice ~2), which is a weaker claim than "matches MATLAB"
and is stated as such rather than glossed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from scipy.special import iv

from ._compat import resolve_dim
from ._population import (accepts_population, is_population,
                          require_population)
from ._empty import empty_safe
from .core import Tree
from .edit import resample_tree, rootangle_tree
from .graphtheory import B_tree, BO_tree, Pvec_tree, T_tree, asym_tree, dissect_tree
from .metrics import angleB_tree, cyl_tree, eucl_tree, len_tree, tran_tree
from .density import vhull_tree
from .plotting import chull_tree

__all__ = [
    "ShollResult",
    "sholl_tree",
    "vonMises_tree",
    "bf_tree",
    "peters_tree",
    "stats_tree",
    "RMCResult",
    "r_mc_tree",
    "ShollDissection",
    "dissectSholl_tree",
]

def _empty_sholl(tree):
    """A Sholl result for a tree with no cable: no spheres, no crossings."""
    blank = np.empty(0)
    return ShollResult(s=blank, dd=blank, sd=blank, XP=blank, YP=blank,
                       ZP=blank, iD=np.empty(0, dtype=int))

def _empty_rmc(tree):
    """Clark-Evans on fewer than two points is undefined, not 1.0."""
    nan = float("nan")
    return RMCResult(R=nan, Rmin=nan, Rmax=nan, r0=nan, rE=nan, rEmin=nan,
                     rEmax=nan, rEstd=nan, n=0, rEs=np.empty(0))

def _empty_dissection(tree):
    blank = np.empty(0)
    nan = float("nan")
    return ShollDissection(
        c=nan, volume=0.0, total_length=0.0, scale=nan, radii=blank,
        observed=blank, domain=blank, angle=None, density=None,
        rootangle=None, k=None, bf=None, est_scale=None, err_domain=nan,
        err_angle=None, err_density=None,
    )




# ---------------------------------------------------------------------------
# sholl_tree
# ---------------------------------------------------------------------------


@dataclass
class ShollResult:
    """Output of :func:`sholl_tree`.

    Attributes
    ----------
    s : np.ndarray
        Number of intersections at each diameter in ``dd``.
    dd : np.ndarray
        Sphere diameters [um] the analysis was evaluated at.
    sd : np.ndarray
        Number of *double* intersections at each diameter (a single segment
        crossing the same sphere twice).
    XP, YP, ZP : np.ndarray
        Coordinates of every intersection point found (concatenated across
        all diameters).
    iD : np.ndarray
        For each intersection point, the index into ``dd`` it belongs to.
    """

    s: np.ndarray
    dd: np.ndarray
    sd: np.ndarray
    XP: np.ndarray
    YP: np.ndarray
    ZP: np.ndarray
    iD: np.ndarray


@empty_safe(_empty_sholl)
def sholl_tree(
    tree: Tree | list[Tree],
    dd: float | np.ndarray = 50.0,
    single_only: bool = False,
    warn_double: bool = True,
) -> ShollResult | list[ShollResult]:
    """Sholl analysis: intersections of the tree with concentric spheres.

    ``dd`` is either a step size (spheres from 0 up to a bit past the
    tree's farthest point, matching MATLAB's auto-range) or an explicit
    array of diameters. Ported line-for-line from the sphere/line-segment
    intersection algorithm (Bourke 1992) -- a standard, well-established
    geometric formula, not something to re-derive.

    ``single_only=True`` subtracts double-counted segments from ``s``
    (MATLAB's ``'-o'``); ``warn_double`` controls whether a
    :func:`warnings.warn` is raised when any segment crosses a sphere
    twice (MATLAB's ``'-e'``, default on). The MATLAB ``'-s'``/``'-s3'``
    plotting options are dropped -- see module docstring.

    **A list of trees** returns a list of results evaluated on **one shared
    set of radii**, taken from the furthest node in the whole group -- so
    ``np.array([r.s for r in results])`` is a well-formed matrix and the
    profiles can be averaged or summed column by column. Pooling them here
    instead would mean choosing between sum, mean and per-cell
    normalisation, and that choice changes what the answer means, so it
    stays with the caller.
    """
    if is_population(tree):
        tree = require_population(tree, "sholl_tree")
        if np.ndim(dd) == 0:
            step = float(dd)
            top = max(np.ceil(2 * eucl_tree(t).max() / step) * step for t in tree)
            dd = np.arange(0.0, top + step, step)
        return [sholl_tree(t, dd, single_only, warn_double) for t in tree]

    dd = np.asarray(dd, dtype=float)
    if dd.ndim == 0:
        step = float(dd)
        eucl = eucl_tree(tree)
        top = np.ceil(2 * eucl.max() / step) * step
        dd = np.arange(0.0, top + step, step)

    X1, X2, Y1, Y2, Z1, Z2 = cyl_tree(tree)
    N = X1.size
    X3, Y3, Z3 = X1[0], Y1[0], Z1[0]

    s = np.zeros(dd.shape)
    sd = np.zeros(dd.shape)
    XP_parts, YP_parts, ZP_parts, iD_parts = [], [], [], []

    a = (X2 - X1) ** 2 + (Y2 - Y1) ** 2 + (Z2 - Z1) ** 2
    for i, diam in enumerate(dd):
        b = 2 * (
            (X2 - X1) * (X1 - X3) + (Y2 - Y1) * (Y1 - Y3) + (Z2 - Z1) * (Z1 - Z3)
        )
        c = (
            X3**2 + Y3**2 + Z3**2 + X1**2 + Y1**2 + Z1**2
            - 2 * (X3 * X1 + Y3 * Y1 + Z3 * Z1)
            - (diam / 2) ** 2
        )
        squ = b * b - 4 * a * c
        iu = squ >= 0

        u1 = np.full(N, np.nan)
        u2 = np.full(N, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            u1[iu] = (-b[iu] + np.sqrt(squ[iu])) / (2 * a[iu])
            u2[iu] = (-b[iu] - np.sqrt(squ[iu])) / (2 * a[iu])
        u1[(u1 < 0) | (u1 > 1)] = np.nan
        u2[(u2 < 0) | (u2 > 1)] = np.nan

        iu1 = ~np.isnan(u1)
        iu2 = ~np.isnan(u2)
        XP_parts.append(X1[iu1] + u1[iu1] * (X2[iu1] - X1[iu1]))
        YP_parts.append(Y1[iu1] + u1[iu1] * (Y2[iu1] - Y1[iu1]))
        ZP_parts.append(Z1[iu1] + u1[iu1] * (Z2[iu1] - Z1[iu1]))
        iD_parts.append(np.full(iu1.sum(), i))
        XP_parts.append(X1[iu2] + u2[iu2] * (X2[iu2] - X1[iu2]))
        YP_parts.append(Y1[iu2] + u2[iu2] * (Y2[iu2] - Y1[iu2]))
        ZP_parts.append(Z1[iu2] + u2[iu2] * (Z2[iu2] - Z1[iu2]))
        iD_parts.append(np.full(iu2.sum(), i))

        s[i] = iu1.sum() + iu2.sum()
        sd[i] = (iu1 & ~iu2).sum()

    s[dd == 0] = 1
    sd[dd == 0] = 0

    if single_only:
        s = s - sd

    if warn_double and sd.sum() > 0:
        warnings.warn(f"{int(sd.sum())} segments were counted twice", stacklevel=2)

    return ShollResult(
        s=s,
        dd=dd,
        sd=sd,
        XP=np.concatenate(XP_parts) if XP_parts else np.array([]),
        YP=np.concatenate(YP_parts) if YP_parts else np.array([]),
        ZP=np.concatenate(ZP_parts) if ZP_parts else np.array([]),
        iD=np.concatenate(iD_parts) if iD_parts else np.array([]),
    )


# ---------------------------------------------------------------------------
# vonMises_tree / bf_tree
# ---------------------------------------------------------------------------


def _collect_rootangles(data: Tree | list[Tree] | np.ndarray) -> np.ndarray:
    if isinstance(data, Tree):
        return rootangle_tree(data)
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], Tree):
        return np.concatenate([rootangle_tree(t) for t in data])
    angles = np.asarray(data, dtype=float)
    if angles.size and (angles.min() < 0 or angles.max() > np.pi):
        raise ValueError("root angles must lie in [0, pi]")
    return angles


def vonMises_tree(
    data: Tree | list[Tree] | np.ndarray, dim: int = 3
) -> tuple[float, dict]:
    """Fit a (modified) von Mises distribution to a tree's root-angle
    distribution, returning the centripetal bias ``k`` (Bird & Cuntz 2019).

    ``data`` is a single :class:`Tree`, a list of trees (root angles
    pooled across all of them), or an array of root angles directly
    (matching MATLAB's polymorphic input). ``dim`` is ``2`` or ``3``,
    selecting which functional form is fit.

    Uses :func:`scipy.optimize.curve_fit` in place of MATLAB's Curve
    Fitting Toolbox (``fit``/``fittype``) -- same nonlinear least-squares
    fit, no toolbox dependency. Returns ``(k, gof)`` where ``gof`` is a
    dict with ``rmse``/``sse``/``r_square`` (MATLAB's ``fit`` returns a
    richer ``gof`` struct; these are the commonly-used fields).
    """
    dim = resolve_dim(dim)
    rootangle = _collect_rootangles(data)
    angv = np.linspace(0.0, np.pi, 25)
    counts, _ = np.histogram(rootangle, bins=angv)
    mid = (angv[1:] + angv[:-1]) / 2
    pdf = counts / trapezoid(counts, mid)

    if dim == 2:
        def model(x, k):
            return np.exp(k * np.cos(x)) / (np.pi * iv(0, k))
    else:
        def model(x, k):
            return k * np.sin(x) * np.exp(k * np.cos(x)) / (2 * np.sinh(k))

    (k,), _ = curve_fit(model, mid, pdf, p0=[2.0])

    residuals = pdf - model(mid, k)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((pdf - pdf.mean()) ** 2))
    gof = {
        "sse": ss_res,
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "r_square": (1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }
    return float(k), gof


# fit constants from Bird & Cuntz 2019, relating centripetal bias k to
# balancing factor bf; (a, b, c) per dimensionality
_BF_FIT_PARAMS = {
    2: (1.201, 4.39, 0.2857),
    3: (0.7331, 3.714, 0.3331),
}


def bf_tree(
    data: Tree | list[Tree] | np.ndarray,
    dim: int = 3,
    fit_constants: tuple[float, float, float] | None = None,
) -> tuple[float, float]:
    """Estimate a tree's MST balancing factor from its root-angle distribution.

    Fits the centripetal bias ``k`` via :func:`vonMises_tree`, then maps it
    to an estimated balancing factor ``bf`` (as used by :func:`MST_tree`)
    through the closed-form relationship fit in Bird & Cuntz 2019. Returns
    ``(bf, k)``, clamped to ``[0, 1]`` with a warning if the raw estimate
    falls outside that range (matching MATLAB).

    ``fit_constants`` overrides the three published constants of that
    relationship -- it is not data, which is what MATLAB's name for it
    (``params``) suggested.
    """
    dim = resolve_dim(dim)
    k, _ = vonMises_tree(data, dim=dim)
    p1, p2, p3 = fit_constants if fit_constants is not None else _BF_FIT_PARAMS[dim]
    bf = 1 - (1 + (k / p1) ** (1 / p3)) ** (-1 / p2)
    if bf < 0:
        bf = 0.0
        warnings.warn("balancing factor out of usual range", stacklevel=2)
    elif bf > 1:
        bf = 1.0
        warnings.warn("balancing factor out of usual range", stacklevel=2)
    return float(bf), float(k)


# ---------------------------------------------------------------------------
# peters_tree
# ---------------------------------------------------------------------------


def peters_tree(
    tree1: Tree,
    tree2: Tree,
    spinedis: float = 3.0,
    synapsedis: float = 3.0,
    resample: bool = True,
) -> np.ndarray:
    """Candidate synapses between two trees (Peters' rule).

    For every node of ``tree1``, finds nodes of ``tree2`` within
    ``spinedis`` [um] -- candidate oppositions. Candidates are then
    greedily accepted closest-first, each acceptance eliminating every
    remaining candidate whose *either* endpoint lies within
    ``synapsedis`` [um] of the accepted one (in its own tree) -- avoiding
    a cluster of near-duplicate "synapses" along the same stretch of
    contact. ``resample=True`` (default) resamples both trees to 1 um
    spacing first, matching MATLAB's default.

    Returns an ``(n_candidates, 3)`` array of ``(node1, node2, distance)``.
    """
    if resample:
        tree1 = resample_tree(tree1, 1.0)
        tree2 = resample_tree(tree2, 1.0)

    pts1 = np.column_stack([tree1.X, tree1.Y, tree1.Z])
    pts2 = np.column_stack([tree2.X, tree2.Y, tree2.Z])

    rows = []
    for i in range(pts1.shape[0]):
        d = np.linalg.norm(pts2 - pts1[i], axis=1)
        for j in np.flatnonzero(d < spinedis):
            rows.append((i, j, d[j]))

    if not rows:
        return np.empty((0, 3))

    cand = np.array(rows, dtype=float)
    cand = cand[np.argsort(cand[:, 2])]

    kept = []
    remaining = cand
    while remaining.shape[0] > 0:
        best = remaining[0]
        kept.append(best)
        rest = remaining[1:]
        if rest.shape[0] == 0:
            break
        d1 = np.linalg.norm(pts1[rest[:, 0].astype(int)] - pts1[int(best[0])], axis=1)
        d2 = np.linalg.norm(pts2[rest[:, 1].astype(int)] - pts2[int(best[1])], axis=1)
        remaining = rest[~((d1 < synapsedis) & (d2 < synapsedis))]

    return np.array(kept)


# ---------------------------------------------------------------------------
# stats_tree
# ---------------------------------------------------------------------------


def _normalize_groups(trees) -> list[list[Tree]]:
    if isinstance(trees, Tree):
        return [[trees]]
    trees = list(trees)
    if trees and isinstance(trees[0], Tree):
        return [trees]
    return [list(group) for group in trees]


def _safe_ratio(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def stats_tree(
    trees: Tree | list[Tree] | list[list[Tree]],
    group_names: list[str] | None = None,
    extras: bool = False,
    density_thr: float = 25.0,
) -> dict[str, pd.DataFrame]:
    """Collect comparable statistics across one or more groups of trees.

    ``trees`` accepts a single tree, a flat list of trees (one group), or
    a list of lists of trees (several named groups) -- matching MATLAB's
    polymorphic input. Rather than MATLAB's nested struct-of-cell-arrays
    (``gstats``/``dstats``, a bespoke container Design Decision 5 already
    ruled against), this returns a dict of tidy, long-format DataFrames,
    which `groupby`/`pandas`/`seaborn` already know how to filter, and
    plot however you like:

    - ``"summary"``: one row per tree -- total length, branch-point count,
      mean branch order, spanning-field aspect ratios, etc.
    - ``"points"``: one row per branch/termination point per tree --
      branch order, path length, direct/path ratio, branch angle.
    - ``"branches"``: one row per dissected branch per tree -- branch
      length (MATLAB drops branches shorter than 0.2 -- likely spacer
      artifacts from `elimt_tree`-style multifurcation handling -- kept
      here for fidelity).

    With ``extras=True``, adds a per-tree convex hull volume and mean
    branch-point asymmetry to ``"summary"``, plus a ``"sholl"`` DataFrame
    (intersection counts at a common set of radii shared across every
    tree, for direct between-tree comparison). MATLAB's density/Voronoi
    piece (`parea`/`mparea`) is **not** included: it depends on
    `hull_tree`/`vhull_tree`, deferred since Phase 7 pending the
    density-grid machinery neither has yet.
    """
    groups = _normalize_groups(trees)
    if group_names is None:
        group_names = [str(i) for i in range(len(groups))]
    if len(group_names) != len(groups):
        raise ValueError("group_names must have one entry per group of trees")

    dsholl = None
    if extras:
        maxlen = 0.0
        for group in groups:
            for tree in group:
                maxlen = max(maxlen, eucl_tree(tree).max())
        dsholl = np.arange(0.0, round(1.1 * 2 * maxlen) + 1.0)

    summary_rows: list[dict] = []
    point_rows: list[dict] = []
    branch_rows: list[dict] = []
    sholl_rows: list[dict] = []

    for gname, group in zip(group_names, groups):
        for tidx, tree in enumerate(group):
            length = len_tree(tree)
            plen = Pvec_tree(tree, length)
            bo = BO_tree(tree)
            eucl = eucl_tree(tree)
            iBB = B_tree(tree)
            iBT = T_tree(tree) | iBB
            with np.errstate(invalid="ignore"):
                # root (Plen == 0) divides 0/0 -> NaN, same as MATLAB;
                # filtered out by nanmean below, not an error
                peucl = eucl[iBT] / plen[iBT]
            try:
                angleB = angleB_tree(tree)
            except ValueError as exc:
                # angleB_tree/asym_tree deliberately refuse non-binary branch
                # points, but their message names *them*, which is confusing
                # when the caller asked for statistics. Real reconstructions
                # do have multifurcations (the bundled NeuroLucida sample has
                # 24), so point at the fix rather than the symptom.
                raise ValueError(
                    f"stats_tree cannot measure branch angles on tree "
                    f"{tree.name!r}: it has a non-binary branch point. "
                    f"Run repair_tree(tree) first to make it BCT-conform. "
                    f"(underlying error: {exc})"
                ) from exc

            sect = dissect_tree(tree)
            blen_d = plen[sect[:, 1]] - plen[sect[:, 0]]
            blen = blen_d[blen_d > 0.2]

            for pt_bo, pt_plen, pt_peucl, pt_angle in zip(
                bo[iBT], plen[iBT], peucl, angleB[iBT]
            ):
                point_rows.append(
                    {
                        "group": gname,
                        "tree": tidx,
                        "BO": pt_bo,
                        "Plen": pt_plen,
                        "peucl": pt_peucl,
                        "angleB": pt_angle,
                    }
                )
            for bl in blen:
                branch_rows.append({"group": gname, "tree": tidx, "blen": bl})

            # Summing nothing is zero; averaging nothing is not. An empty
            # tree in a population keeps its row -- dropping it would
            # renumber every cell after it -- but its means are `nan`, so a
            # population average over the column is not silently dragged
            # toward zero (see `pynetrees._empty`).
            def _amax(values):
                return float(np.max(values)) if np.size(values) else float("nan")

            def _amean(values):
                return float(np.nanmean(values)) if np.size(values) else float("nan")

            def _aptp(values):
                return float(np.ptp(values)) if np.size(values) else float("nan")

            row = {
                "group": gname,
                "tree": tidx,
                "len": float(length.sum()),
                "max_plen": _amax(plen),
                "bpoints": int(iBB.sum()),
                "mpeucl": _amean(peucl),
                "maxbo": _amax(bo[iBT]),
                "mangleB": _amean(angleB[iBT]),
                "mblen": float(blen.mean()) if blen.size else float("nan"),
                "mplen": _amean(plen),
                "mbo": _amean(bo[iBT]),
                "wh": _safe_ratio(_aptp(tree.X), _aptp(tree.Y)),
                "wz": _safe_ratio(_aptp(tree.X), _aptp(tree.Z)),
                "chullx": _amean(tree.X),
                "chully": _amean(tree.Y),
                "chullz": _amean(tree.Z),
            }

            if extras:
                _, hull = chull_tree(tree)
                row["hull_volume"] = hull.volume if hull is not None else float("nan")

                # Density: Voronoi territory per branch/termination point,
                # clipped to the *space-filling* hull. MATLAB's `parea`
                # (per point) and `mparea` (its mean). This is the piece
                # that was missing until hull_tree/vhull_tree existed.
                try:
                    territory = vhull_tree(
                        tree, nodes=np.flatnonzero(iBT), thr=density_thr
                    ).volumes
                    row["mparea"] = float(np.nanmean(territory))
                except (ValueError, RuntimeError) as exc:
                    # too few points to tessellate, or a degenerate hull --
                    # report it as missing rather than failing the whole
                    # statistics run over one tree
                    warnings.warn(
                        f"stats_tree: no density statistics for tree "
                        f"{tree.name!r} ({exc})",
                        stacklevel=2,
                    )
                    territory = np.full(int(iBT.sum()), np.nan)
                    row["mparea"] = float("nan")

                asym = asym_tree(tree)
                valid_asym = asym[~np.isnan(asym)]
                row["masym"] = (
                    float(valid_asym.mean()) if valid_asym.size else float("nan")
                )

                sholl = sholl_tree(tree, dsholl, warn_double=False)
                for radius, count in zip(sholl.dd, sholl.s):
                    sholl_rows.append(
                        {
                            "group": gname,
                            "tree": tidx,
                            "radius": radius,
                            "count": count,
                        }
                    )

            if extras:
                # `points` already has one row per branch/termination point,
                # in flatnonzero(iBT) order, so territory lines up directly
                for offset, area in enumerate(territory):
                    point_rows[len(point_rows) - len(territory) + offset][
                        "parea"
                    ] = float(area)

            summary_rows.append(row)

    result = {
        "summary": pd.DataFrame(summary_rows),
        "points": pd.DataFrame(point_rows),
        "branches": pd.DataFrame(branch_rows),
    }
    if extras:
        result["sholl"] = pd.DataFrame(sholl_rows)
    return result


# ---------------------------------------------------------------------------
# r_mc_tree -- Monte-Carlo test of spatial randomness
# ---------------------------------------------------------------------------


class RMCResult(NamedTuple):
    """Output of :func:`r_mc_tree`.

    ``R`` is the statistic; everything else is the working that produced
    it, kept because the sampling distribution is what tells you whether a
    given ``R`` means anything.
    """

    R: float
    """Clark-Evans ratio ``r0 / rE``. Below 1 the points are clustered,
    above 1 they are more regularly spaced than chance, 1 is random."""
    Rmin: float
    """Lower confidence bound, ``r0 / rEmax``. ``nan`` unless
    ``confidence=True``."""
    Rmax: float
    """Upper confidence bound, ``r0 / rEmin``."""
    r0: float
    """Observed mean nearest-neighbour distance [um]."""
    rE: float
    """Mean nearest-neighbour distance expected under uniform randomness in
    the same volume, averaged over the Monte-Carlo iterations."""
    rEmin: float
    rEmax: float
    rEstd: float
    """Spread of ``rEs`` -- how stable the null estimate is."""
    n: int
    """Number of points analysed."""
    rEs: np.ndarray
    """The per-iteration null estimates, for plotting the null
    distribution rather than trusting a single summary."""


@accepts_population
@empty_safe(_empty_rmc)
def r_mc_tree(tree, alpha: float = 0.5, n_mc: int = 100,
              level: float = 0.05, nodes="all", *,
              volume_correction: bool = True, confidence: bool = False,
              n_boot: int = 1000, dim: int = 3, rng=None) -> RMCResult:
    """Test whether a tree's points are spaced more regularly than chance.

    The Clark-Evans ratio: mean observed nearest-neighbour distance divided
    by the mean expected if the same number of points were scattered
    uniformly through the same volume. The null is estimated by Monte
    Carlo rather than from a closed form, because the volume in question is
    the cell's own concave boundary, not a box.

    Parameters
    ----------
    tree : Tree or (n, dim) array_like
        A tree, or a bare point cloud -- the statistic is about points, and
        :func:`~pynetrees.generate.PP_generator_tree` measures clouds that are
        not trees.
    alpha : float in [0, 1], default 0.5
        Shrink factor of the boundary enclosing the points -- ``0`` the
        convex hull, ``1`` the tightest enveloping shape. See
        :func:`boundary_tree`.
    n_mc : int, default 100
        Monte-Carlo iterations.
    level : float, default 0.05
        Confidence intervals are for level ``1 - level``.
    nodes : {'all', 'bt', 'b', 't'} or array_like, default 'all'
        Which points to analyse: every node, branch **and** termination
        points, branch points only, termination points only, or an explicit
        index array. MATLAB spells these ``''``/``-bt``/``-b``/``-t``.
    volume_correction : bool, default True
        Rescale each Monte-Carlo sample so that the volume its own points
        span matches the reference volume. A finite sample never quite
        reaches the boundary, which shrinks its apparent volume and
        therefore its nearest-neighbour distances, biasing ``R`` upwards.
    confidence : bool, default False
        Bootstrap a confidence interval within each iteration. Costs
        ``n_mc * n_boot`` resamples; without it ``Rmin``/``Rmax`` are
        ``nan``.
    n_boot : int, default 1000
        Bootstrap resamples per iteration.
    dim : {2, 3}, default 3
    rng : numpy Generator or int, optional
        Seed, for reproducibility.

    Returns
    -------
    RMCResult

    Notes
    -----
    Two deliberate divergences from MATLAB's `r_mc_tree`.

    **The volume-correction flag is inverted upstream.** MATLAB documents
    ``'-nv'`` as "no volume correction" and states "By default, a volume
    correction is applied", but the code reads ``if pars.nv % volume
    correction`` -- so passing the *disable* flag is what enables it, and
    the default (``nv = false``) applies no correction at all. Both halves
    of the documentation are contradicted by the one line. This port
    follows the documented intent: correction on by default, off via
    ``volume_correction=False``. See MATLAB_TOOLBOX_BUGS.md.

    **Sampling is exact rather than by rejection.** MATLAB fills the
    bounding box with uniform points and throws away everything outside the
    boundary, testing each candidate with a vendored point-in-polyhedron
    routine. For a neuron -- a thin arbor inside a large box -- most of
    every batch is discarded. Drawing from the boundary's own simplex
    decomposition, weighted by simplex volume, gives the identical uniform
    distribution with no rejection and no point-in-mesh test (see
    ``pynetrees.density._sample_in_simplices``).

    A third, smaller one: MATLAB's ``bootci`` defaults to bias-corrected
    accelerated intervals; this uses the percentile bootstrap, so intervals
    will differ slightly on skewed samples.
    """
    from scipy.spatial import cKDTree

    from .density import _alpha_shape, _sample_in_simplices

    generator = np.random.default_rng(rng)
    if isinstance(tree, Tree):
        idx = _select_points(tree, nodes)
        coords = np.column_stack(
            [tree.X, tree.Y] + ([tree.Z] if dim == 3 else [])
        )[idx]
    else:
        coords = np.atleast_2d(np.asarray(tree, dtype=float))[:, :dim]
        if not isinstance(nodes, str):
            coords = coords[np.asarray(nodes, dtype=int)]
    n = len(coords)
    if n < 2:
        raise ValueError(f"need at least 2 points for a nearest neighbour, got {n}")

    reference = _alpha_shape(coords, alpha)
    r0 = float(_mean_nn(coords, cKDTree))

    rEs = np.empty(n_mc)
    lows = np.full(n_mc, np.nan)
    highs = np.full(n_mc, np.nan)
    for i in range(n_mc):
        points = _sample_in_simplices(reference.points, reference.simplices,
                                      n, generator)
        if volume_correction:
            sampled_volume = _alpha_shape(points, alpha).volume
            if sampled_volume > 0:
                points = points * (reference.volume / sampled_volume) ** (1 / dim)
        distances = _nn_distances(points, cKDTree)
        rEs[i] = distances.mean()
        if confidence:
            draws = generator.integers(0, n, size=(n_boot, n))
            means = distances[draws].mean(axis=1)
            lows[i], highs[i] = np.quantile(means, [level / 2, 1 - level / 2])

    rE = float(rEs.mean())
    rEmin = float(np.nanmean(lows)) if confidence else np.nan
    rEmax = float(np.nanmean(highs)) if confidence else np.nan
    return RMCResult(
        R=r0 / rE,
        Rmin=r0 / rEmax if confidence else np.nan,
        Rmax=r0 / rEmin if confidence else np.nan,
        r0=r0,
        rE=rE,
        rEmin=rEmin,
        rEmax=rEmax,
        rEstd=float(rEs.std(ddof=1)),
        n=n,
        rEs=rEs,
    )


def _select_points(tree: Tree, nodes) -> np.ndarray:
    """Resolve `r_mc_tree`'s ``nodes`` selector to an index array."""
    if isinstance(nodes, str):
        if nodes == "all":
            return np.arange(tree.n_nodes)
        if nodes == "bt":
            return np.flatnonzero(B_tree(tree) | T_tree(tree))
        if nodes == "b":
            return np.flatnonzero(B_tree(tree))
        if nodes == "t":
            return np.flatnonzero(T_tree(tree))
        raise ValueError(
            f"nodes must be 'all', 'bt', 'b', 't' or an index array, got {nodes!r}"
        )
    return np.asarray(nodes, dtype=int)


def _nn_distances(points: np.ndarray, cKDTree) -> np.ndarray:
    """Distance from each point to its nearest *other* point."""
    distances, _ = cKDTree(points).query(points, k=2)
    return distances[:, 1]


def _mean_nn(points: np.ndarray, cKDTree) -> float:
    return float(_nn_distances(points, cKDTree).mean())


# ---------------------------------------------------------------------------
# dissectSholl_tree
# ---------------------------------------------------------------------------


class ShollDissection(NamedTuple):
    """Output of :func:`dissectSholl_tree`.

    The point of the analysis is the comparison between ``observed`` and
    the successively richer predictions ``domain``, ``angle`` and
    ``density``: whatever the simplest prediction already explains is not
    evidence of anything more interesting.
    """

    c: float
    """Convexity used to set the boundary's tightness."""
    volume: float
    """Volume [um^3] (3D) or area [um^2] (2D) the boundary encloses."""
    total_length: float
    """Total cable length [um]."""
    scale: float
    """Integral of the observed (unnormalised) Sholl profile."""
    radii: np.ndarray
    """Radii [um] every profile below is sampled at."""
    observed: np.ndarray
    """Measured Sholl profile, normalised to unit integral."""
    domain: np.ndarray
    """Profile predicted by the spanning domain alone -- i.e. by nothing but
    the shape of the territory the cell occupies."""
    angle: np.ndarray | None
    """Profile predicted by the domain *and* the centripetal bias."""
    density: np.ndarray | None
    """Profile predicted by the domain *and* a non-uniform branch-point
    density. ``None`` unless ``density=True``."""
    rootangle: np.ndarray | None
    """Per-segment root angles the bias estimate was fitted to."""
    k: float | None
    """Fitted von Mises concentration -- the strength of the centripetal
    bias."""
    bf: float | None
    """Balancing factor implied by ``k``, on :func:`MST_tree`'s scale."""
    est_scale: float | None
    """Sholl integral predicted from total length and root angles alone."""
    err_domain: float
    """RMS deviation of ``domain`` from ``observed``."""
    err_angle: float | None
    err_density: float | None


@accepts_population
@empty_safe(_empty_dissection)
def dissectSholl_tree(tree: Tree, c: float | None = None, dim: int = 3, *,
                      centripetal: bool = True, density: bool = False,
                      n_radii: int = 25, n_directions: int = 20000,
                      scale_factor: float | None = None,
                      rng=None) -> ShollDissection:
    """Decompose a Sholl profile into what explains it (Bird & Cuntz 2018).

    A Sholl profile counts how many branches cross each sphere around the
    soma, and is routinely read as a signature of a cell type. Much of its
    shape, though, follows from nothing more than the *shape of the region*
    the cell fills -- a sphere of radius R simply has more of its surface
    inside a wide territory than a narrow one. This function builds up that
    null prediction and two refinements of it, so the profile can be
    compared against what is already explained:

    ``domain``
        What the spanning territory alone predicts.
    ``angle``
        Domain plus the centripetal bias: real dendrites do not leave a
        branch point in a uniformly random direction, they tend outward.
    ``density``
        Domain plus a radially non-uniform branch-point density.

    Parameters
    ----------
    tree : Tree
    c : float, optional
        Convexity. Computed with :func:`convexity_tree` if omitted, which
        is the expensive part of the call -- pass it if you have it.
    dim : {2, 3}, default 3
    centripetal : bool, default True
        Compute the ``angle`` correction (MATLAB's ``'-a'``, also default
        on). Requires the root-angle fit, so it is the second-most
        expensive part.
    density : bool, default False
        Compute the ``density`` correction (MATLAB's ``'-n'``).
    n_radii : int, default 25
        Radii the profiles are evaluated at, spanning 0 to the furthest
        node.
    n_directions : int, default 20000
        Directions sampled when measuring how much of each sphere lies
        inside the boundary.
    scale_factor : float, optional
        Multiplier on the estimated mean branch length. Defaults to
        MATLAB's rule -- see Notes.
    rng : numpy Generator or int, optional

    Returns
    -------
    ShollDissection

    Notes
    -----
    **Sphere sampling is restructured.** MATLAB tests a million random
    points per radius for containment in the boundary mesh, using a
    vendored ray-casting routine -- 25 million point-in-mesh tests per
    call. But every one of those points lies on a ray from the root, and
    the radii differ only in how far along that ray the point sits. This
    port casts each ray *once*, records every distance at which it crosses
    the surface, and then reads off containment at all ``n_radii`` radii
    from the crossing parity. Same estimator, one pass instead of
    ``n_radii``, and no vendored code.

    **MATLAB's undocumented size fudge is reproduced but exposed.** Its 3D
    branch silently doubles the estimated mean branch length for cells
    reaching beyond 500 um (``if rmax > 500, sf = 2``) with no explanation
    anywhere in the file, and its 2D branch has no such rule. That is a
    discontinuity in a published measure, so it is kept for fidelity but
    surfaced as ``scale_factor`` -- pass ``1.0`` to switch it off.

    The 2D branch also extrapolates the first root-angle bin
    (``rVraw(1) = rVraw(2) + (rVraw(2) - rVraw(3))``, patching over the
    fact that no segment has a root angle of exactly zero) while the 3D
    branch does not, even though the ``Estscale`` helper both branches call
    *does*. Reproduced as-is; see MATLAB_TOOLBOX_BUGS.md.
    """
    from .density import boundary_tree, convexity_tree

    generator = np.random.default_rng(rng)
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    if c is None:
        c = convexity_tree(tree, dim=dim, rng=generator)

    total_length = float(len_tree(tree).sum())
    centred = tran_tree(tree)  # put the root at the origin: all radii are from it
    bound = boundary_tree(centred, c=c, dim=dim)

    eucl = eucl_tree(centred)
    rmax = float(eucl.max())
    radii = np.linspace(0.0, rmax, n_radii)

    observed = np.nan_to_num(sholl_tree(centred, 2 * radii, warn_double=False).s)
    domain = _spanning_domain_profile(bound, radii, dim, n_directions, generator)

    scale = float(trapezoid(observed, radii))
    domain_norm = domain / trapezoid(domain, radii)
    observed_norm = observed / scale

    angle_norm = k = bf = est_scale = rootangle = None
    if centripetal:
        rootangle = rootangle_tree(tree)
        bf, k = bf_tree(rootangle, dim=dim)
        if scale_factor is None:
            # MATLAB: 3D doubles the branch length above 500 um, 2D never does
            scale_factor = 2.0 if (dim == 3 and rmax > 500) else 1.0
        angle_norm = _centripetal_profile(
            radii, domain_norm, rootangle, total_length, bound.volume,
            dim, scale_factor,
        )
        est_scale = _estimated_scale(rootangle, total_length)

    density_norm = None
    if density:
        counts = _histax(eucl[B_tree(tree)], radii)
        density_norm = 0.5 * (counts / trapezoid(counts, radii) + domain_norm)

    def rms(prediction):
        if prediction is None:
            return None
        return float(np.sqrt(trapezoid((prediction - observed_norm) ** 2, radii)))

    return ShollDissection(
        c=float(c),
        volume=bound.volume,
        total_length=total_length,
        scale=scale,
        radii=radii,
        observed=observed_norm,
        domain=domain_norm,
        angle=angle_norm,
        density=density_norm,
        rootangle=rootangle,
        k=k,
        bf=bf,
        est_scale=est_scale,
        err_domain=rms(domain_norm),
        err_angle=rms(angle_norm),
        err_density=rms(density_norm),
    )


def _spanning_domain_profile(bound, radii: np.ndarray, dim: int,
                             n_directions: int, generator) -> np.ndarray:
    """Sholl profile a cell would have if it merely filled its territory.

    For each radius this is the part of the sphere (circle in 2D) of that
    radius that lies inside the boundary, scaled by the sphere's full area
    (circumference). Rays are cast from the origin once and reused at every
    radius: a point at distance R along a ray is inside the shape exactly
    when an odd number of the ray's surface crossings lie beyond R.
    """
    directions = _uniform_directions(n_directions, dim, generator)
    if dim == 3:
        hit_ray, hit_t = _ray_triangle_hits(directions, bound.vertices, bound.faces)
        area = 4 * np.pi * radii**2
    else:
        hit_ray, hit_t = _ray_segment_hits(directions, bound.polygon)
        area = 2 * np.pi * radii

    crossings = np.zeros((len(directions), len(radii)), dtype=np.int32)
    if len(hit_t):
        np.add.at(crossings, hit_ray, hit_t[:, None] > radii[None, :])
    inside = (crossings % 2 == 1).mean(axis=0)
    return np.nan_to_num(area * inside)


def _uniform_directions(n: int, dim: int, generator) -> np.ndarray:
    """``n`` directions spread uniformly over the sphere (circle in 2D)."""
    theta = 2 * np.pi * generator.random(n)
    if dim == 2:
        return np.column_stack([np.cos(theta), np.sin(theta)])
    # uniform in area, not in latitude -- otherwise the poles are oversampled
    z = 1 - 2 * generator.random(n)
    r = np.sqrt(np.maximum(1 - z**2, 0.0))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])


def _ray_triangle_hits(directions: np.ndarray, vertices: np.ndarray,
                       faces: np.ndarray, chunk: int = 2048):
    """Moeller-Trumbore: every forward crossing of every ray with the mesh.

    Returns ``(ray_index, distance)`` for each hit, unordered.
    """
    v0 = vertices[faces[:, 0]]
    edge1 = vertices[faces[:, 1]] - v0
    edge2 = vertices[faces[:, 2]] - v0

    rays, distances = [], []
    for lo in range(0, len(directions), chunk):
        d = directions[lo : lo + chunk]
        # h = d x edge2, over the full (rays, faces) outer product
        h = np.cross(d[:, None, :], edge2[None, :, :])
        det = np.einsum("nfd,fd->nf", h, edge1)
        parallel = np.abs(det) < 1e-12
        inv = np.where(parallel, 1.0, 1.0 / np.where(parallel, 1.0, det))

        # rays all start at the origin, so s = origin - v0 = -v0
        s = -v0
        u = inv * np.einsum("nfd,fd->nf", h, s)
        q = np.cross(s[None, :, :], edge1[None, :, :])
        v = inv * np.einsum("nd,nfd->nf", d, q)
        t = inv * np.einsum("nfd,fd->nf", q, edge2)

        ok = ~parallel & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 0)
        ray, _face = np.nonzero(ok)
        rays.append(ray + lo)
        distances.append(t[ok])
    return np.concatenate(rays), np.concatenate(distances)


def _ray_segment_hits(directions: np.ndarray, polygon: np.ndarray):
    """Every forward crossing of every ray with a closed 2D polygon."""
    a = polygon
    b = np.roll(polygon, -1, axis=0)
    edge = b - a

    # solve  t * d = a + s * edge  for t >= 0 and s in [0, 1]
    cross = directions[:, None, 0] * edge[None, :, 1] - \
        directions[:, None, 1] * edge[None, :, 0]
    parallel = np.abs(cross) < 1e-12
    safe = np.where(parallel, 1.0, cross)
    t = (a[None, :, 0] * edge[None, :, 1] - a[None, :, 1] * edge[None, :, 0]) / safe
    s = (directions[:, None, 0] * a[None, :, 1] -
         directions[:, None, 1] * a[None, :, 0]) / -safe

    ok = ~parallel & (t > 0) & (s >= 0) & (s <= 1)
    ray, _edge = np.nonzero(ok)
    return ray, t[ok]


def _histax(values: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Histogram of ``values`` into bins *centred* on ``centres``.

    MATLAB's ``histax`` utility: the caller supplies the x-axis it wants to
    plot against, and this places the bin edges half a step either side so
    the counts line up with it.
    """
    centres = np.asarray(centres, dtype=float)
    half = (centres[1] - centres[0]) / 2
    edges = np.append(centres - half, centres[-1] + half)
    return np.histogram(np.asarray(values), bins=edges)[0].astype(float)


def _centripetal_profile(radii, domain_norm, rootangle, total_length, volume,
                         dim, scale_factor):
    """Domain profile convolved with the root-angle distribution.

    A branch leaving a branch point at root angle ``theta`` displaces the
    material it carries by roughly one mean branch length in that
    direction, which along the radial axis is ``S * cos(theta)``. So the
    predicted profile is the domain profile smeared by the distribution of
    that radial displacement -- which is what the integral below is.
    """
    from scipy.interpolate import interp1d

    # estimated branch point count, hence mean branch length
    if dim == 2:
        n_branch = np.sqrt(total_length**3 * 4 / (3 * np.pi * volume))
    else:
        n_branch = np.sqrt(total_length**3 / (3 * np.pi * volume))
    branch_length = scale_factor * total_length / n_branch

    angles = np.linspace(0.0, np.pi, 25)
    raw = _histax(rootangle, angles)
    if dim == 2:
        # no segment has a root angle of exactly 0, so the first bin is
        # empty and gets linearly extrapolated from its neighbours
        raw[0] = raw[1] + (raw[1] - raw[2])
    weights = raw / trapezoid(raw, angles)
    if dim == 2:
        branch_length = branch_length * weights.max()

    keep = weights > 0
    kernel_x = branch_length * weights[keep] * np.cos(angles[keep])
    kernel_y = branch_length * weights[keep] * np.sin(angles[keep])

    # interp1 needs a monotonic abscissa; pol2cart does not guarantee one
    order = np.argsort(kernel_x)
    kernel_x, kernel_y = kernel_x[order], kernel_y[order]
    kernel_x, unique = np.unique(kernel_x, return_index=True)
    kernel_y = kernel_y[unique]

    lo, hi = kernel_x[0], kernel_x[-1]
    grid = np.linspace(lo, hi, 1000)
    kernel = np.interp(grid, kernel_x, kernel_y)

    profile = interp1d(radii, domain_norm, kind="cubic",
                       bounds_error=False, fill_value=0.0)
    smeared = np.array([
        trapezoid(profile(np.linspace(r + lo, r + hi, 1000)) * kernel, grid)
        for r in radii
    ])
    # the outermost radius has nothing beyond it to smear inwards, so the
    # convolution there is meaningless; MATLAB pins it to the domain value
    smeared[-1] = domain_norm[-1]

    smeared = smeared / trapezoid(smeared, radii) + domain_norm
    return smeared / trapezoid(smeared, radii)


def _estimated_scale(rootangle: np.ndarray, total_length: float) -> float:
    """Sholl integral predicted from cable length and root angles alone.

    Every micron of cable crosses spheres at a rate set by how radially it
    runs: a purely radial segment crosses one sphere per micron, a purely
    tangential one crosses none. Averaging ``|cos(theta)|`` over the
    root-angle distribution and multiplying by total length gives the
    expected total number of intersections without measuring any.
    """
    angles = np.linspace(0.0, np.pi, 25)
    raw = _histax(rootangle, angles)
    raw[0] = raw[1] + (raw[1] - raw[2])
    weights = raw / trapezoid(raw, angles)
    return float(total_length * trapezoid(np.abs(np.cos(angles) * weights), angles))
