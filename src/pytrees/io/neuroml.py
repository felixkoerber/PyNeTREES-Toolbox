"""NeuroML morphology export.

Ports ``IO/neuroml_tree.m``, contributed to the MATLAB toolbox by Padraig
Gleeson in 2011. Two schema versions, as there: NeuroML **v2** (the living
standard, and the default) and **v1 Level 1** / MorphML, kept for tools
that never moved on.

A NeuroML segment is an *edge*, not a node: each carries a ``proximal`` and
a ``distal`` point, so a tree of N nodes exports as N-1 segments and the
root node appears only as the proximal end of its children's segments.

Built with :mod:`xml.etree.ElementTree` rather than by concatenating
strings, which is what MATLAB does. That is not stylistic: string
concatenation writes a broken file the moment a region name contains ``&``
or ``"``, and produces a document nothing can validate if a single quote is
misplaced. The tree is serialised once, correctly escaped.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..core import NO_PARENT, Tree

__all__ = ["save_neuroml"]

_V2_NS = "http://www.neuroml.org/schema/neuroml2"
_V2_SCHEMA = (
    "https://raw.githubusercontent.com/NeuroML/NeuroML2/development"
    "/Schemas/NeuroML2/NeuroML_v2.3.xsd"
)
_V1_NS = "http://morphml.org/neuroml/schema"
_V1_SCHEMA = (
    "http://www.neuroml.org/NeuroMLValidator/NeuroMLFiles/Schemata"
    "/v1.8.1/Level1/NeuroML_Level1_v1.8.1.xsd"
)
_XSI = "http://www.w3.org/2001/XMLSchema-instance"


def save_neuroml(tree: Tree, path: str | Path, version: str = "2",
                 *, segment_groups: bool = True) -> Path:
    """Write a tree's morphology as NeuroML.

    Parameters
    ----------
    tree : Tree
    path : str or Path
        ``.xml`` is appended if missing. The stem becomes the cell id.
    version : {'2', '1'}, default '2'
        NeuroML 2 (MATLAB's ``'-v2a'``, its default too) or NeuroML v1
        Level 1 / MorphML (``'-v1l1'``).
    segment_groups : bool, default True
        Emit one ``<segmentGroup>`` per region, so ``axon``/``dendrite``
        labels survive the export. NeuroML 2 only, and **an addition** --
        MATLAB writes no groups, so a round trip through it loses every
        region. Set ``False`` for output that matches MATLAB's structure.

    Returns
    -------
    Path

    Notes
    -----
    Three divergences from MATLAB's writer, each because its output is
    wrong rather than merely different.

    **Its NeuroML 2 ``schemaLocation`` is malformed.** The namespace and
    the schema URL are concatenated without the separating space that
    attribute's syntax requires, giving
    ``"http://www.neuroml.org/schema/neuroml2http://neuroml.svn..."`` --
    one token where two are needed, so no validator can resolve it. The
    URL it points at, on SourceForge's long-dead SVN viewer, has not
    existed for years either; this writes the current schema location.

    **It attaches root segments to segment 0.** A segment whose proximal
    point is the tree's root has no parent segment, and MATLAB's
    ``parentid = idpar0 (ward) - 2`` evaluates to ``-1`` there, which it
    then rewrites to ``0`` -- silently declaring the segment a child of the
    first one. Here such a segment simply carries no ``<parent>``, which is
    how NeuroML spells "this is where the cell starts".

    **It mixes line endings**, writing CR+LF from ``fwrite`` and bare LF
    from ``fprintf`` within the same file.

    **Segment ids are the distal node's index**, where MATLAB uses
    ``node - 2`` in its 1-based counting. Its scheme assumes the root is
    node 1; on a tree whose root sits elsewhere -- which its own ``.neu``
    reader produces, see ``GC1.neu`` -- node 1 is a real node and gets the
    id ``-1``. Naming a segment after its distal node is unique whatever
    the root is, and makes the parent lookup an identity rather than an
    offset.

    A fourth difference is deliberate on MATLAB's side and preserved here:
    a segment's ``proximal`` point takes the **distal** node's diameter,
    not the proximal node's, so each segment is a uniform cylinder rather
    than a frustum. MATLAB flags this in a comment (``% NOTE: dist
    diameter!!``); it is what makes the export agree with the toolbox's own
    non-frustum surface-area convention.
    """
    if str(version) not in ("1", "2"):
        raise ValueError(f"version must be '1' or '2', got {version!r}")
    version = str(version)

    path = Path(path)
    if path.suffix != ".xml":
        path = path.with_suffix(path.suffix + ".xml")
    cell_id = path.stem

    parent = _parents(tree)
    root = ET.Element("neuroml")
    if version == "2":
        root.set("xmlns", _V2_NS)
        root.set("xmlns:xsi", _XSI)
        root.set("xsi:schemaLocation", f"{_V2_NS} {_V2_SCHEMA}")
        root.set("id", cell_id)
        notes_tag = "notes"
    else:
        root.set("xmlns", _V1_NS)
        root.set("xmlns:meta", "http://morphml.org/metadata/schema")
        root.set("xmlns:mml", "http://morphml.org/morphml/schema")
        root.set("xmlns:xsi", _XSI)
        root.set("xsi:schemaLocation", f"{_V1_NS} {_V1_SCHEMA}")
        root.set("length_units", "micrometer")
        notes_tag = "meta:notes"

    ET.SubElement(root, notes_tag).text = (
        f"\n  TREES toolbox tree - {tree.name}"
        f"\n  written by pytrees, the Python port of the TREES toolbox"
        f"\n  Export version: {'nml_v2' if version == '2' else 'nml_v1_l1'}\n"
    )

    if version == "2":
        cell = ET.SubElement(root, "cell", {"id": cell_id})
        container = ET.SubElement(cell, "morphology",
                                  {"id": f"{cell_id}_morphology"})
    else:
        cells = ET.SubElement(root, "cells")
        cell = ET.SubElement(cells, "cell", {"name": cell_id})
        container = ET.SubElement(
            cell, "segments", {"xmlns": "http://morphml.org/morphml/schema"}
        )

    for node in range(tree.n_nodes):
        proximal = parent[node]
        if proximal == NO_PARENT:
            continue  # the root is a point, not a segment
        _segment(container, tree, node, proximal, parent, version)

    if version == "2" and segment_groups:
        _segment_groups(container, tree, parent)

    ET.indent(root, space="  ")
    path.write_bytes(
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        + ET.tostring(root, encoding="utf-8", xml_declaration=False)
        + b"\n"
    )
    return path


def _parents(tree: Tree) -> np.ndarray:
    parent = np.full(tree.n_nodes, NO_PARENT, dtype=int)
    coo = tree.dA.tocoo()
    parent[coo.row] = coo.col
    return parent


def _segment(container, tree: Tree, node: int, proximal: int,
             parent: np.ndarray, version: str) -> None:
    """One segment: the edge from ``proximal`` down to ``node``."""
    segment = ET.SubElement(container, "segment", {
        "id": str(node),
        "name": f"Seg_{node}_P{proximal}_to_P{node}",
    })
    if parent[proximal] != NO_PARENT:
        # a segment is named for its distal node, so the parent segment is
        # simply the one named for this segment's proximal node
        if version == "2":
            ET.SubElement(segment, "parent", {"segment": str(proximal)})
        else:
            segment.set("parent", str(proximal))

    # NOTE: the *distal* diameter on both ends -- see this module's Notes
    ET.SubElement(segment, "proximal", _point(tree, proximal, tree.D[node]))
    ET.SubElement(segment, "distal", _point(tree, node, tree.D[node]))


def _point(tree: Tree, node: int, diameter: float) -> dict[str, str]:
    return {
        "x": f"{tree.X[node]:.8f}",
        "y": f"{tree.Y[node]:.8f}",
        "z": f"{tree.Z[node]:.8f}",
        "diameter": f"{diameter:.8f}",
    }


def _segment_groups(container, tree: Tree, parent: np.ndarray) -> None:
    """One ``<segmentGroup>`` per region, listing its segments.

    Without these the export is anatomically blank: NeuroML has no other
    place to record that a stretch of cable is axon rather than dendrite,
    and every downstream tool that colours or filters by region reads them.
    """
    regions = np.asarray(tree.R, dtype=int)
    for index, rname in enumerate(tree.rnames):
        members = [
            node for node in range(tree.n_nodes)
            if parent[node] != NO_PARENT and regions[node] == index
        ]
        if not members:
            continue
        group = ET.SubElement(container, "segmentGroup",
                              {"id": _nml_identifier(rname)})
        for node in members:
            ET.SubElement(group, "member", {"segment": str(node)})


def _nml_identifier(rname: str) -> str:
    """A region name usable as an XML id (NMTOKEN-safe)."""
    import re

    clean = re.sub(r"[^A-Za-z0-9_.-]", "_", str(rname))
    return clean if clean and not clean[0].isdigit() else f"r_{clean}"
