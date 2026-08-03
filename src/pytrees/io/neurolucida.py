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

**Deliberately not ported** (see PORT_STATUS.md Design Decisions): soma
contours are *not* reconstructed into a fitted cylinder (MATLAB's version
does PCA-based cylinder fitting -- its own docstring calls this "quite
arbitrary" and says the function "can be much further optimized or just
rewritten"); markers (small glyphs like synapse/marker points) are dropped
entirely; and reconstructed trees are *not* automatically concatenated onto
their nearest soma via nearest-point matching. All three are real, working
MATLAB features -- this port keeps the actual branch/region geometry (the
part every downstream analysis needs) and defers the rest.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy import sparse

from ..core import NO_PARENT, Tree
from ..edit import repair_tree

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


def load_neurolucida(path: str | Path, repair: bool = True) -> Tree | list[Tree]:
    """Load a NeuroLucida ASCII (.asc) file into a Tree, or a list of Trees
    if it contains more than one (typically one per soma/dendrite/axon).

    If ``repair`` (default), each resulting tree is passed through
    :func:`~pytrees.repair_tree` before being returned.
    """
    path = Path(path)
    text = path.read_text(errors="replace")
    tokens = _tokenize(text)
    top_level_blocks = _parse_top_level(tokens)

    trees = []
    for i, block in enumerate(top_level_blocks):
        if block and isinstance(block[0], str) and block[0].lower() in _SKIP_TOP_LEVEL:
            continue
        if block and isinstance(block[0], str) and block[0].lower() in _MARKER_NAMES:
            continue

        points: list[list[float]] = []
        parents: list[int] = []
        regions: list[str] = []
        _walk(block, NO_PARENT, "noregion", points, parents, regions)
        if not points:
            continue

        tree = _build_tree(points, parents, regions, name=f"{path.stem}_{i}")
        trees.append(repair_tree(tree) if repair else tree)

    if len(trees) == 1:
        return trees[0]
    return trees


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


def _parse_atom(tok: str) -> float | str:
    try:
        return float(tok)
    except ValueError:
        return tok.strip('"')


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
        if len(item) == 1 and isinstance(item[0], str):
            return ("region", item[0])
        return ("split", item)
    return ("ignore", None)


def _walk(items, parent_idx, region, points, parents, regions) -> None:
    """Walk a flat item sequence, threading `parent_idx`/`region` through
    it and appending each point found to `points`/`parents`/`regions`.
    """
    idx = parent_idx
    for item in items:
        kind, value = _classify(item)
        if kind == "point":
            d = value[3] if len(value) > 3 else 0.0
            points.append([value[0], value[1], value[2], d])
            parents.append(idx)
            regions.append(region)
            idx = len(points) - 1
        elif kind == "region":
            region = value
        elif kind == "split":
            branches: list[list] = [[]]
            for sub in value:
                if sub == "|":
                    branches.append([])
                else:
                    branches[-1].append(sub)
            for branch in branches:
                _walk(branch, idx, region, points, parents, regions)
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
