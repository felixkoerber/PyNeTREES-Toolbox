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

Several Phase 9 MATLAB functions are deferred rather than ported --
substantial, poorly-specified, or independently-flagged-as-buggy machinery
disproportionate to their value; see PORT_STATUS.md's Phase 9 table and
Design Decision #35 for the full per-function reasoning:
``convexity_tree``/``boundary_tree``/``share_boundary_tree`` (self-flagged
buggy/unclear by the maintainers -- and ``boundary_tree`` in particular
turns out to unconditionally crash on its own documented default call
path, see MATLAB_TOOLBOX_BUGS.md), ``dissectSholl_tree`` (research-
contributed, needs the deferred boundary/convexity machinery plus a
vendored third-party point-in-mesh algorithm, maintainers' own todo list
says "rewrite, don't profile-and-keep"), ``r_mc_tree`` (needs alpha-shapes
and point-in-polyhedron testing -- real new dependencies for a niche,
not-fully-validated-upstream statistical test), and ``M_atten_tree`` (has
no docstring at all, and the maintainers' own todo list says its purpose
is unclear -- porting speculative code around an unknown intent risks
confidently porting something wrong).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import iv

from .core import Tree
from .edit import resample_tree, rootangle_tree
from .graphtheory import B_tree, BO_tree, Pvec_tree, T_tree, asym_tree, dissect_tree
from .metrics import angleB_tree, cyl_tree, eucl_tree, len_tree
from .plotting import chull_tree

__all__ = [
    "ShollResult",
    "sholl_tree",
    "vonMises_tree",
    "bf_tree",
    "peters_tree",
    "stats_tree",
]


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


def sholl_tree(
    tree: Tree,
    dd: float | np.ndarray = 50.0,
    single_only: bool = False,
    warn_double: bool = True,
) -> ShollResult:
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
    """
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
    data: Tree | list[Tree] | np.ndarray, dim: str = "3d"
) -> tuple[float, dict]:
    """Fit a (modified) von Mises distribution to a tree's root-angle
    distribution, returning the centripetal bias ``k`` (Bird & Cuntz 2019).

    ``data`` is a single :class:`Tree`, a list of trees (root angles
    pooled across all of them), or an array of root angles directly
    (matching MATLAB's polymorphic input). ``dim`` is ``"2d"`` or ``"3d"``,
    selecting which functional form is fit.

    Uses :func:`scipy.optimize.curve_fit` in place of MATLAB's Curve
    Fitting Toolbox (``fit``/``fittype``) -- same nonlinear least-squares
    fit, no toolbox dependency. Returns ``(k, gof)`` where ``gof`` is a
    dict with ``rmse``/``sse``/``r_square`` (MATLAB's ``fit`` returns a
    richer ``gof`` struct; these are the commonly-used fields).
    """
    rootangle = _collect_rootangles(data)
    angv = np.linspace(0.0, np.pi, 25)
    counts, _ = np.histogram(rootangle, bins=angv)
    mid = (angv[1:] + angv[:-1]) / 2
    pdf = counts / np.trapezoid(counts, mid)

    if dim == "2d":
        def model(x, k):
            return np.exp(k * np.cos(x)) / (np.pi * iv(0, k))
    elif dim == "3d":
        def model(x, k):
            return k * np.sin(x) * np.exp(k * np.cos(x)) / (2 * np.sinh(k))
    else:
        raise ValueError(f"dim must be '2d' or '3d', got {dim!r}")

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
    "2d": (1.201, 4.39, 0.2857),
    "3d": (0.7331, 3.714, 0.3331),
}


def bf_tree(
    data: Tree | list[Tree] | np.ndarray,
    dim: str = "3d",
    params: tuple[float, float, float] | None = None,
) -> tuple[float, float]:
    """Estimate a tree's MST balancing factor from its root-angle distribution.

    Fits the centripetal bias ``k`` via :func:`vonMises_tree`, then maps it
    to an estimated balancing factor ``bf`` (as used by :func:`MST_tree`)
    through the closed-form relationship fit in Bird & Cuntz 2019. Returns
    ``(bf, k)``, clamped to ``[0, 1]`` with a warning if the raw estimate
    falls outside that range (matching MATLAB).
    """
    k, _ = vonMises_tree(data, dim=dim)
    p1, p2, p3 = params if params is not None else _BF_FIT_PARAMS[dim]
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

            row = {
                "group": gname,
                "tree": tidx,
                "len": float(length.sum()),
                "max_plen": float(plen.max()),
                "bpoints": int(iBB.sum()),
                "mpeucl": float(np.nanmean(peucl)),
                "maxbo": float(bo[iBT].max()),
                "mangleB": float(np.nanmean(angleB[iBT])),
                "mblen": float(blen.mean()) if blen.size else float("nan"),
                "mplen": float(plen.mean()),
                "mbo": float(bo[iBT].mean()),
                "wh": _safe_ratio(
                    tree.X.max() - tree.X.min(), tree.Y.max() - tree.Y.min()
                ),
                "wz": _safe_ratio(
                    tree.X.max() - tree.X.min(), tree.Z.max() - tree.Z.min()
                ),
                "chullx": float(tree.X.mean()),
                "chully": float(tree.Y.mean()),
                "chullz": float(tree.Z.mean()),
            }

            if extras:
                _, hull = chull_tree(tree)
                row["hull_volume"] = hull.volume if hull is not None else float("nan")
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

            summary_rows.append(row)

    result = {
        "summary": pd.DataFrame(summary_rows),
        "points": pd.DataFrame(point_rows),
        "branches": pd.DataFrame(branch_rows),
    }
    if extras:
        result["sholl"] = pd.DataFrame(sholl_rows)
    return result
