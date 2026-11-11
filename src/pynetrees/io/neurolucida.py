"""NeuroLucida ASCII (.asc) format import.

Ports (part of) ``IO/neurolucida_tree.m``. NeuroLucida ASCII is essentially
an S-expression format: parenthesized lists containing numbers, bare words,
quoted strings, and a ``|`` token marking parallel branches at a split. A
tree entry looks like::

    ( (Color Blue)
      (Dendrite)
      (x y z d)          ; a point -- becomes a node, child of the previous one
      (x y z d)
      (                  ; a branch point: two (or more) sibling branches...
        (x y z d) ...    ; ...separated by '|', each continuing from the
      |                  ; point immediately before this '('
        (x y z d) ...
      )
    )

This port re-derives the format from the bundled real sample file
(``treestoolbox-master/sample/neurolucida/twop9purks.ASC``) with a proper
recursive tokenizer/parser, rather than translating MATLAB's line-by-line,
manual-paren-depth-counting state machine (`Plevel`/`Tflag`/`Zflag` in the
original) -- the underlying structure is a standard nested S-expression, and
parsing it that way is far easier to verify correct.

Everything MATLAB's reader extracts is kept: **soma contours** become a fitted
cylinder chain, **markers** and raw contours are available through
``full_output=True``, **spines** (``<(x y z d)>``) become a region called
``spines``, and each reconstructed tree is joined to its nearest soma.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy import sparse

from ..core import NO_PARENT, Tree
from ..edit import cat_tree, repair_tree
from ..metrics import vol_tree

# marker glyph names (spines/synapse markers etc.) -- a whole top-level
# block starting with one of these is a marker cloud, not branch geometry
_MARKER_NAMES = {
    n.lower()
    for n in (
        "Dot", "OpenStar", "FilledQuadStar", "CircleArrow", "OpenCircle",
        "DoubleCircle", "OpenQuadStar", "CircleCross", "Cross", "Circle1",
        "Flower3", "Plus", "Circle2", "Pinwheel", "OpenUpTriangle", "Circle3",
        "TexacoStar", "OpenDownTriangle", "Circle4", "ShadedStar", "OpenSquare",
        "Circle5", "SkiBasket", "Asterisk", "Circle6", "Clock", "OpenDiamond",
        "Circle7", "ThinArrow", "FilledStar", "Circle8", "ThickArrow", "FilledCircle",
        "Circle9", "SquareGunSight", "FilledUpTriangle", "Flower2", "GunSight",
        "FilledDownTriangle", "SnowFlake", "TriStar", "FilledSquare", "OpenFinial",
        "NinjaStar", "FilledDiamond", "FilledFinial", "KnightsCross", "Flower",
        "MalteseCross", "Splat",
    )
}
# whole top-level blocks to skip outright (large/irrelevant housekeeping data)
_SKIP_TOP_LEVEL = {"thumbnail", "imagecoords"}
# nested metadata sub-lists to skip wherever encountered (their contents are
# never branch geometry)
_METADATA_TAGS = {
    "color", "name", "description", "sections", "mbfobjecttype", "guid",
    "set", "filldensity", "resolution", "ssm", "dzi", "propertyactivitystates",
}
# bare annotation words that carry no geometric information
_IGNORE_ATOMS = {"normal", "low", "high", "generated", "incomplete"}

_TOKEN_RE = re.compile(r'\(|\)|\||"[^"]*"|[^\s()|]+')


class NeurolucidaResult(NamedTuple):
    """Result of :func:`load_neurolucida` with ``full_output=True``."""

    trees: "Tree | list[Tree]"
    """One tree per soma, or a single tree if there is only one."""
    markers: np.ndarray
    """``(n, 4)`` marker coordinates ``[x, y, z, d]``."""
    contours: list
    """The raw soma contours, each ``{"name", "region", "points"}``."""


def load_neurolucida(path: str | Path, repair: bool = True, *, one_soma: bool = False,
                     full_output: bool = False):
    """Load a NeuroLucida ASCII (.asc) file: one tree per soma.

    Each soma contour is turned into a chain of cylinders, and every
    reconstructed tree is attached to the soma nearest its first point. A file
    with no contour uses its first tree in place of a soma.

    Parameters
    ----------
    path : str or Path
    repair : bool, default True
        Pass each tree through :func:`~pynetrees.repair_tree`.
    one_soma : bool, default False
        Keep only the soma with the largest volume. MATLAB's ``'-o'``.
    full_output : bool, default False
        Return :class:`NeurolucidaResult`, which adds the markers and the raw
        contours.
    """
    path = Path(path)
    text = path.read_text(errors="replace")

    markers: list[list[float]] = []
    contours: list[dict] = []
    pieces: list[Tree] = []
    for block in _parse_top_level(_tokenize(text)):
        if block and isinstance(block[0], str) and block[0].lower() in _SKIP_TOP_LEVEL:
            continue
        points: list[list[float]] = []
        parents: list[int] = []
        regions: list[str] = []
        _walk([block], NO_PARENT, "noregion", points, parents, regions, markers, contours)
        if points:
            pieces.append(_build_tree(points, parents, regions,
                                      name=f"{path.stem}_{len(pieces)}"))

    somata = [_soma_from_contour(c, f"{path.stem}_{i}") for i, c in enumerate(contours)
              if len(c["points"]) >= 2]
    somata = [soma for soma in somata if soma is not None]
    if one_soma and somata:
        somata = [max(somata, key=lambda soma: float(np.sum(vol_tree(soma))))]
    if not somata:
        if not pieces:
            raise ValueError(f"{path}: found no trees and no soma contours")
        somata = [pieces.pop(0)]

    for piece in pieces:
        target, node = _nearest_soma(somata, piece)
        somata[target] = cat_tree(somata[target], piece, node, 0)

    trees = [repair_tree(soma) if repair else soma for soma in somata]
    if len(trees) == 1:
        trees[0].name = path.stem
    result = trees[0] if len(trees) == 1 else trees
    if full_output:
        return NeurolucidaResult(result, np.array(markers, dtype=float).reshape(-1, 4), contours)
    return result


def _nearest_soma(somata: list[Tree], piece: Tree) -> tuple[int, int]:
    """Which soma, and which of its nodes, a tree's first point is closest to."""
    start = np.array([piece.X[0], piece.Y[0], piece.Z[0]])
    best, target, node = 1000.0, 0, 0   # MATLAB's search radius and fallback
    for index, soma in enumerate(somata):
        xyz = np.column_stack([soma.X, soma.Y, soma.Z])
        distance = np.linalg.norm(xyz - start, axis=1) - soma.D / 2
        nearest = int(np.argmin(distance))
        if distance[nearest] < best:
            best, target, node = float(distance[nearest]), index, nearest
    return target, node


def _soma_from_contour(contour: dict, name: str) -> Tree | None:
    """A cylinder chain along a contour's long axis, wide as the contour is.

    MATLAB's algorithm: resample the outline every micrometre, rotate it onto
    its 2D principal axes, take its half-width at each 1 um step along the long
    axis, smooth that with a 10-point window, and rotate the axis back.
    """
    xyz = np.asarray(contour["points"], dtype=float)[:, :3]
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
    arc, keep = np.unique(arc, return_index=True)
    if arc[-1] < 1.0:
        return None
    even = np.arange(0.0, np.floor(arc[-1]) + 1.0)
    xyz = np.column_stack([np.interp(even, arc, xyz[keep, i]) for i in range(3)])

    centre = xyz.mean(axis=0)
    centred = xyz - centre
    values, vectors = np.linalg.eigh(np.cov(centred[:, :2], rowvar=False))
    p = vectors[:, np.argsort(values)[::-1]]
    rotated = centred[:, :2] @ p.T
    along, keep = np.unique(rotated[:, 0], return_index=True)
    width = np.abs(rotated[keep, 1])

    steps = np.arange(np.ceil(along.min()), np.floor(along.max()) + 1.0)
    if len(steps) == 0:
        return None
    half = np.interp(steps, along, width)
    # MATLAB's convn(...,'same') centres a 10-point window one sample later
    # than numpy's 'same' does
    smoothed = np.convolve(half, np.ones(10) / 10)[5:5 + len(half)]

    axis = np.column_stack([steps, np.zeros(len(steps))]) @ np.linalg.inv(p.T)
    X, Y, Z = axis[:, 0] + centre[0], axis[:, 1] + centre[1], np.full(len(steps), centre[2])
    n = len(steps)
    dA = sparse.coo_matrix((np.ones(n - 1), (np.arange(1, n), np.arange(n - 1))),
                           shape=(n, n)).tocsr()
    return Tree(dA=dA, X=X, Y=Y, Z=Z, D=2 * smoothed, R=np.zeros(n, int),
                rnames=[contour["region"]], name=name)


# ---------------------------------------------------------------------------
# tokenizing and parsing into nested lists
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    without_comments = re.sub(r";.*", "", text)
    return _TOKEN_RE.findall(without_comments)


def _parse_top_level(tokens: list[str]) -> list[list]:
    """Parse a flat token stream into a list of top-level nested lists."""
    pos = 0
    n = len(tokens)

    def parse_list() -> list:
        nonlocal pos
        items: list = []
        while pos < n:
            tok = tokens[pos]
            if tok == "(":
                pos += 1
                items.append(parse_list())
            elif tok == ")":
                pos += 1
                return items
            elif tok == "|":
                pos += 1
                items.append("|")
            else:
                pos += 1
                items.append(_parse_atom(tok))
        return items

    blocks = []
    while pos < n:
        if tokens[pos] == "(":
            pos += 1
            blocks.append(parse_list())
        else:
            pos += 1  # skip stray top-level tokens
    return blocks


class _Quoted(str):
    """A quoted string: a contour starts with one, a region name does not."""


def _parse_atom(tok: str) -> float | str:
    if tok.startswith('"'):
        return _Quoted(tok.strip('"'))
    try:
        return float(tok)
    except ValueError:
        return tok


# ---------------------------------------------------------------------------
# walking the parsed structure into points/parents/regions
# ---------------------------------------------------------------------------


def _classify(item) -> tuple[str, object]:
    if isinstance(item, str):
        return ("ignore", None) if item.lower() in _IGNORE_ATOMS else ("region", item)
    if isinstance(item, list):
        if not item:
            return ("ignore", None)
        if isinstance(item[0], str) and item[0].lower() in _METADATA_TAGS:
            return ("metadata", None)
        if 3 <= len(item) <= 4 and all(isinstance(x, float) for x in item):
            return ("point", item)
        if isinstance(item[0], _Quoted):
            return ("contour", item)
        if isinstance(item[0], str) and item[0].lower() in _MARKER_NAMES:
            return ("marker", item)
        if len(item) == 1 and isinstance(item[0], str):
            return ("region", item[0])
        return ("split", item)
    return ("ignore", None)


def _coordinates(items) -> list[list[float]]:
    """Every ``(x y z [d])`` list directly inside ``items``, padded to four."""
    found = []
    for item in items:
        if (isinstance(item, list) and 3 <= len(item) <= 4
                and all(isinstance(x, float) for x in item)):
            found.append(item + [0.0] * (4 - len(item)))
    return found


def _walk(items, parent_idx, region, points, parents, regions, markers, contours) -> None:
    """Walk a flat item sequence, threading `parent_idx`/`region` through
    it and appending each point found to `points`/`parents`/`regions`.

    Spines are written ``<(x y z d)>``, and consecutive ones form one spine.
    They hang off the point before them without moving the walk along.
    """
    idx = parent_idx
    in_spine = False
    spine_tip = None
    for item in items:
        if item == "<":
            in_spine = True
            continue
        if item == ">":
            in_spine = False
            continue
        kind, value = _classify(item)
        if kind == "point":
            d = value[3] if len(value) > 3 else 0.0
            points.append([value[0], value[1], value[2], d])
            if in_spine:
                parents.append(idx if spine_tip is None else spine_tip)
                regions.append("spines")
                spine_tip = len(points) - 1
                continue
            spine_tip = None
            parents.append(idx)
            regions.append(region)
            idx = len(points) - 1
        elif kind == "region":
            region = value
        elif kind == "marker":
            markers.extend(_coordinates(value))
        elif kind == "contour":
            named = [sub[0] for sub in value[1:] if isinstance(sub, list) and len(sub) == 1
                     and isinstance(sub[0], str)]
            contours.append({"name": str(value[0]), "region": named[0] if named else region,
                             "points": _coordinates(value)})
        elif kind == "split":
            branches: list[list] = [[]]
            for sub in value:
                if sub == "|":
                    branches.append([])
                else:
                    branches[-1].append(sub)
            for branch in branches:
                _walk(branch, idx, region, points, parents, regions, markers, contours)
        # "metadata" / "ignore": contributes nothing, walk continues


def _build_tree(points, parents, regions, name: str) -> Tree:
    n = len(points)
    arr = np.array(points, dtype=float)
    rows = [i for i in range(n) if parents[i] != NO_PARENT]
    cols = [parents[i] for i in range(n) if parents[i] != NO_PARENT]
    dA = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()

    uniq, R = np.unique(regions, return_inverse=True)
    return Tree(
        dA=dA,
        X=arr[:, 0], Y=arr[:, 1], Z=arr[:, 2], D=arr[:, 3],
        R=R, rnames=uniq.tolist(), name=name,
    )
