"""NEURON export: ``.hoc`` cell files, ``.hoc`` templates, and ``.nrn``.

Ports ``IO/neuron_tree.m`` (both its ``.hoc`` and ``.nrn`` branches) and
``IO/neuron_template_tree.m``.

These write a morphology out for NEURON to read. Note that
:mod:`pytrees.neuron_bridge` builds the same sections *in process* through
NEURON's Python interface, which is the better route when NEURON is
importable -- no file, no round trip, no name mangling. These exist for the
cases the bridge cannot cover: handing a cell to a NEURON model that lives
outside Python, to a collaborator, or to T2N.

Three output shapes, because they are used differently:

``style="cell"``
    A flat ``.hoc`` file that creates its sections at the top level.
    Simplest to read; only one cell per NEURON process.
``style="template"``
    The same geometry wrapped in ``begintemplate``/``endtemplate``, so a
    network model can instantiate many copies. Also emits the SectionList
    subsets T2N expects.
``.nrn`` (:func:`save_nrn`)
    One NEURON section per *graph segment* rather than per unbranched
    section, with the geometry appended as a numeric block the hoc file
    reads back with ``fscan()``.

**Section, not segment.** A "section" here is an unbranched run of nodes
between branch points, which is what NEURON means by the word -- so the 197
nodes of ``sample_tree`` become 51 sections. That grouping comes from
:func:`~pytrees.dissect_tree`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..core import Tree
from ..edit import root_tree
from ..graphtheory import dissect_tree, ipar_tree
from ..metrics import len_tree

__all__ = ["save_hoc", "save_nrn", "t2n_interface"]

# NEURON's hoc parser refuses procedures beyond a few hundred statements, so
# both MATLAB and this split the generated topology and geometry procedures
# into numbered chunks called from a wrapper. 250 is MATLAB's number.
_CHUNK = 250

# MATLAB writes CR+LF explicitly (`nextline = [char(13), newline]`), and hoc
# files are routinely moved between platforms, so the line ending is fixed
# rather than left to the OS.
_EOL = "\r\n"


class _Layout(NamedTuple):
    """How a tree maps onto NEURON's create/connect model."""

    tree: Tree
    """The tree actually written -- `root_tree`d, so it has one more node."""
    sections: np.ndarray
    """``(n_sections, 2)`` of ``(start_node, end_node)``."""
    section_region: np.ndarray
    """Region index of each section, taken from its **end** node."""
    names: list[str]
    """hoc array name per region."""
    counts: np.ndarray
    """Sections per region -- the hoc array sizes."""
    index_in_region: np.ndarray
    """Each section's position within its region's hoc array."""
    ipar: np.ndarray


def _layout(tree: Tree, stem: str, prefix: bool) -> _Layout:
    """Work out sections, regions and hoc array indices.

    ``root_tree`` first, as MATLAB does: without it a tree whose root is
    already a branch point produces sections that all start at the same
    node, and NEURON has nowhere to attach them.
    """
    tree = root_tree(tree)
    sections = dissect_tree(tree)
    section_region = np.asarray(tree.R, dtype=int)[sections[:, 1]]

    used = np.unique(tree.R)
    names = [_hoc_identifier(tree.rnames[r], stem, prefix) for r in used.tolist()]
    if len(set(names)) != len(names):
        raise ValueError(
            f"region names collide once made hoc-safe: {names}; rename the "
            "tree's regions so they differ by more than punctuation"
        )
    counts = np.array([(section_region == r).sum() for r in used.tolist()])

    # position of each section within its own region's hoc array, and the
    # region renumbered to index into `names` (regions can be sparse)
    index_in_region = np.empty(len(sections), dtype=int)
    slot_of_section = np.empty(len(sections), dtype=int)
    for slot, region in enumerate(used.tolist()):
        members = np.flatnonzero(section_region == region)
        index_in_region[members] = np.arange(len(members))
        slot_of_section[members] = slot

    return _Layout(tree, sections, slot_of_section, names, counts,
                   index_in_region, ipar_tree(tree))


def _hoc_identifier(rname: str, stem: str, prefix: bool) -> str:
    """A region name hoc will accept as an array identifier.

    Everything that is not a letter or digit is dropped, and a leading
    digit is prefixed, because hoc identifiers follow C's rules. MATLAB
    sanitises this way in `neuron_template_tree` but **not** in
    `neuron_tree`, whose ``[name '_' rnames{i}]`` passes any punctuation in
    a region name straight into the file and produces hoc that will not
    parse. Sanitising in both is the only version that always writes a
    loadable file.
    """
    clean = re.sub(r"[^a-zA-Z0-9]", "", str(rname))
    if prefix:
        clean = f"{re.sub(r'[^a-zA-Z0-9]', '', stem)}_{clean}"
    if not clean or clean[0].isdigit():
        clean = f"reg{clean}"
    return clean


class _Writer:
    """Accumulates lines and splits long procedures into numbered chunks."""

    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, line: str = "") -> None:
        self.lines.append(line)

    def text(self) -> str:
        return _EOL.join(self.lines) + _EOL


def _write(path: Path, text: str) -> None:
    """Write already-CRLF-terminated text without letting Python translate it.

    `Path.write_text` defaults to universal newlines, which on Windows turns
    every ``
`` into ``
`` -- and since these lines already end in
    ``
``, that yields ``
`` and a file NEURON reads with a blank
    line between every statement.
    """
    with path.open("w", newline="") as handle:
        handle.write(text)


# ---------------------------------------------------------------------------
# .hoc
# ---------------------------------------------------------------------------


def save_hoc(tree: Tree, path: str | Path, *, style: str = "cell",
             electrotonics: bool = False, run_file: bool = False) -> Path:
    """Write a tree as a NEURON ``.hoc`` file.

    Parameters
    ----------
    tree : Tree
    path : str or Path
        ``.hoc`` is appended if missing. The file stem becomes the cell
        name inside the file, so it must be a usable hoc identifier.
    style : {'cell', 'template'}, default 'cell'
        ``'cell'`` creates sections at the top level (MATLAB's
        `neuron_tree`); ``'template'`` wraps them in
        ``begintemplate``/``endtemplate`` so the cell can be instantiated
        many times (MATLAB's `neuron_template_tree`), and additionally
        emits the ``allreg``/``alladendreg``/``allaxonreg`` SectionLists
        that T2N uses.
    electrotonics : bool, default False
        Emit ``insert pas`` and the tree's ``Ri``/``Gm``/``Cm`` as ``Ra``,
        ``g_pas`` and ``cm``. MATLAB's ``'-e'``. Raises if the tree carries
        no passive parameters, rather than writing a file that silently
        lacks them.
    run_file : bool, default False
        Also write ``run_<stem>.hoc`` next to it, which loads ``nrngui``
        and opens the cell -- MATLAB's ``'-s'``. MATLAB's further ``'->'``
        option, which calls ``winopen`` to launch the file on Windows, is
        not ported: a save function should not start programs.

    Returns
    -------
    Path
        The file written.

    Notes
    -----
    MATLAB's `neuron_tree` documents a ``res`` argument as "number of
    segments per compartment", but its ``.hoc`` branch never reads it --
    the emitted ``proc geom_nseg()`` is empty. Only the ``.nrn`` branch uses
    it, so here it is a parameter of :func:`save_nrn` alone rather than an
    argument that quietly does nothing.

    The template style additionally sets each section's **first** 3D point
    to its second point's diameter, unless the tree is marked ``frustum``.
    NEURON takes a section's first point from where it attaches to its
    parent, so leaving the parent's diameter there inflates the section's
    surface area by a step change at every branch point.
    """
    if style not in ("cell", "template"):
        raise ValueError(f"style must be 'cell' or 'template', got {style!r}")

    path = Path(path)
    if path.suffix != ".hoc":
        path = path.with_suffix(path.suffix + ".hoc")
    stem = path.stem

    if electrotonics and tree.Ri is None:
        raise ValueError(
            "electrotonics=True needs Ri/Gm/Cm on the tree; set them first "
            "or leave the passive properties out of the file"
        )

    layout = _layout(tree, stem, prefix=(style == "cell"))
    write = _Writer()
    _hoc_header(write, stem, style)
    if style == "template":
        _template_declarations(write, stem, layout)
    else:
        _cell_declarations(write, layout)

    _topology(write, layout)
    _shape3d(write, layout, taper_fix=(style == "template" and not tree.frustum))
    _subsets(write, stem, layout, style)

    write("proc geom() {")
    write("}")
    write("proc geom_nseg() {")
    write("}")
    write("proc biophys() {")
    write("}")
    write(f"access {layout.names[0]}")

    if style == "template":
        write("proc init() {")
        write("  celldef()")
        write("}")
        write("")
        write(f"endtemplate {_hoc_identifier(stem, stem, False)}")
    else:
        write("celldef()")
        write("")
        if electrotonics:
            _passive_block(write, tree, f"reg_{stem}_all")

    _write(path, write.text())
    if run_file:
        runner = path.with_name(f"run_{stem}.hoc")
        _write(runner, _EOL.join(
            ['load_file ("nrngui.hoc")', f'xopen ("{path.name}")']) + _EOL)
    return path


def _hoc_header(write: _Writer, stem: str, style: str) -> None:
    write("/*")
    write("This is a CellBuilder-like file written for the simulator NEURON")
    write("by pytrees, the Python port of the TREES toolbox")
    write(f"({'neuron_template_tree' if style == 'template' else 'neuron_tree'}"
          " in the MATLAB original)")
    write("*/")
    write("")


def _cell_declarations(write: _Writer, layout: _Layout) -> None:
    write("proc celldef() {")
    for proc in ("topol", "subsets", "geom", "biophys", "geom_nseg"):
        write(f"  {proc}()")
    write("}")
    write("")
    for name, count in zip(layout.names, layout.counts.tolist()):
        write(f"create {name}[{count}]")
    write("")


def _template_declarations(write: _Writer, stem: str, layout: _Layout) -> None:
    write(f"begintemplate {_hoc_identifier(stem, stem, False)}")
    write("")
    write("proc celldef() {")
    for proc in ("topol", "subsets", "geom", "biophys", "geom_nseg"):
        write(f"  {proc}()")
    write("  is_artificial = 0")
    write("}")
    write("")
    for name in layout.names:
        write(f"public {name}")
    write("")
    for public in ("allregobj", "allreg", "alladendreg", "allaxonreg"):
        write(f"public {public}")
    for name in layout.names:
        write(f"public reg{name}")
    write("public is_artificial")
    write("")
    for name, count in zip(layout.names, layout.counts.tolist()):
        write(f"create {name}[{count}]")
    write("")


def _topology(write: _Writer, layout: _Layout) -> None:
    """``connect child[i](0), parent[j](1)`` for every section but the first.

    A section's parent is the section whose *end* node is this section's
    *start* node -- which is exactly how `dissect_tree` lays them out.
    """
    ends = {int(e): i for i, e in enumerate(layout.sections[:, 1].tolist())}

    chunks = _Writer()
    n_chunks = 1
    written = 0
    chunks("proc topol_1() {")
    for child, (start, _end) in enumerate(layout.sections.tolist()):
        parent = ends.get(int(start))
        if parent is None:  # the root section attaches to nothing
            continue
        chunks(
            f"  connect "
            f"{layout.names[layout.section_region[child]]}"
            f"[{layout.index_in_region[child]}](0),"
            f"{layout.names[layout.section_region[parent]]}"
            f"[{layout.index_in_region[parent]}](1)"
        )
        written += 1
        if written % (_CHUNK - 1) == 0:
            n_chunks += 1
            chunks("}")
            chunks(f"proc topol_{n_chunks}() {{")
    chunks("}")

    write.lines.extend(chunks.lines)
    write("proc topol() {")
    for i in range(1, n_chunks + 1):
        write(f"  topol_{i}()")
    write("  basic_shape()")
    write("}")
    write("")


def _section_paths(layout: _Layout):
    """Nodes of each section, ordered start -> end."""
    for start, end in layout.sections.tolist():
        path = layout.ipar[end]
        path = path[: int(np.flatnonzero(path == start)[0]) + 1]
        yield int(start), int(end), path[::-1]


def _shape3d(write: _Writer, layout: _Layout, taper_fix: bool) -> None:
    tree = layout.tree
    chunks = _Writer()
    n_chunks = 1
    written = 0
    chunks("proc shape3d_1() {")
    for index, (_start, end, nodes) in enumerate(_section_paths(layout)):
        name = layout.names[layout.section_region[index]]
        slot = layout.index_in_region[index]
        chunks(f"  {name}[{slot}] {{pt3dclear()")
        diameters = tree.D[nodes].copy()
        if taper_fix and len(diameters) > 1:
            diameters[0] = diameters[1]
        for node, diameter in zip(nodes.tolist(), diameters.tolist()):
            chunks(f"    pt3dadd({tree.X[node]:.15g}, {tree.Y[node]:.15g}, "
                   f"{tree.Z[node]:.15g}, {diameter:.15g})")
            written += 1
            if written % (_CHUNK - 1) == 0:
                n_chunks += 1
                chunks("  }")
                chunks("}")
                chunks(f"proc shape3d_{n_chunks}() {{")
                chunks(f"  {name}[{slot}] {{")
        chunks("  }")
    chunks("}")

    write.lines.extend(chunks.lines)
    write("proc basic_shape() {")
    for i in range(1, n_chunks + 1):
        write(f"  shape3d_{i}()")
    write("}")
    write("")


def _subsets(write: _Writer, stem: str, layout: _Layout, style: str) -> None:
    if style == "template":
        write("objref allreg, allregobj, alladendreg, allaxonreg, sec")
        for name in layout.names:
            write(f"objref reg{name}")
        write("proc subsets() { local counter")
        write("  allregobj   = new List()")
        write("  allreg      = new SectionList()")
        write("  alladendreg = new SectionList()")
        write("  allaxonreg  = new SectionList()")
        for name, count in zip(layout.names, layout.counts.tolist()):
            write(f"  reg{name} = new SectionList()")
            write(f"  for counter = 0, {count - 1} {name}[counter] {{")
            write(f"    reg{name}.append()")
            write("    sec = new SectionRef()")
            write("    allregobj.append(sec)")
            write("    allreg.append()")
            if "adend" in name:
                write("    alladendreg.append()")
            if "axon" in name:
                write("    allaxonreg.append()")
            write("  }")
        write("}")
        return

    write(f"objref reg_{stem}_all")
    for name in layout.names:
        write(f"objref reg_{name}")
    write("proc subsets() { local counter")
    write(f"  reg_{stem}_all = new SectionList()")
    for name, count in zip(layout.names, layout.counts.tolist()):
        write(f"  reg_{name} = new SectionList()")
        write(f"  for counter = 0, {count - 1} {name}[counter] {{")
        write(f"    reg_{name}.append()")
        write(f"    reg_{stem}_all.append()")
        write("  }")
    write("}")


def _passive_block(write: _Writer, tree: Tree, subset: str) -> None:
    write(f"forsec {subset} insert pas")
    write(f"forsec {subset} g_pas = {_g(tree.Gm)}")
    write(f"forsec {subset} Ra = {_g(tree.Ri)}")
    write(f"forsec {subset} cm = {_g(tree.Cm)}")
    write(f"forsec {subset} e_pas = 0")
    write("")


def _g(value) -> str:
    """MATLAB's `num2str` default: 5 significant digits."""
    return "1" if value is None else f"{float(np.ravel(value)[0]):.5g}"


# ---------------------------------------------------------------------------
# T2N interface matrix
# ---------------------------------------------------------------------------


def t2n_interface(tree: Tree, stem: str = "cell") -> np.ndarray:
    """Where each tree node ends up in the exported hoc sections.

    MATLAB returns this as `neuron_template_tree`'s third output,
    ``minterf``, and T2N uses it to map a node of the TREES morphology onto
    a position inside a NEURON section. Split out here rather than returned
    alongside a file path, because it is a computation about the tree, not
    a detail of writing a file -- and because a caller usually wants one or
    the other, not both.

    Returns
    -------
    np.ndarray
        ``(n_pt3dadd, 3)``, one row per ``pt3dadd`` call the exporter
        emits, in the same order: the node index, the section's index
        across the whole cell, and how far along that section the node sits
        as a fraction in ``[0, 1]``.

    Notes
    -----
    There are more rows than nodes: a branch point is the last node of one
    section and the first node of each of its daughters, so it is written
    once per section it appears in. The count is ``n_nodes + n_sections - 1``
    on the ``root_tree``d tree, which is what MATLAB preallocates.

    Node indices are 0-based, unlike MATLAB's, and unlike MATLAB's they
    refer to the tree **after** the extra root node is prepended -- which
    MATLAB's also do, silently, since it calls ``root_tree`` before
    building the matrix.
    """
    layout = _layout(tree, stem, prefix=False)
    lengths = len_tree(layout.tree)

    # global section index = sections in earlier regions + index within region
    offsets = np.concatenate([[0], np.cumsum(layout.counts)[:-1]])
    global_index = offsets[layout.section_region] + layout.index_in_region

    rows = []
    for index, (_start, _end, nodes) in enumerate(_section_paths(layout)):
        # cumulative length along the section, excluding the shared first node
        along = np.cumsum(lengths[nodes]) - lengths[nodes[0]]
        span = along[-1]
        for node, distance in zip(nodes.tolist(), along.tolist()):
            rows.append((node, global_index[index],
                         round(distance / span, 5) if span > 0 else 0.0))
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# .nrn
# ---------------------------------------------------------------------------


def save_nrn(tree: Tree, path: str | Path, res=None, *,
             electrotonics: bool = False) -> Path:
    """Write a tree as a NEURON ``.nrn`` file: one section per *segment*.

    Where :func:`save_hoc` groups nodes into unbranched sections, this makes
    every graph edge its own NEURON section -- coarser control over the
    file, finer control inside NEURON. The geometry is not written as hoc
    statements but as a plain numeric block appended after the procedures,
    which the emitted ``geometry()`` reads back with ``fscan()``.

    Parameters
    ----------
    tree : Tree
    path : str or Path
    res : int or array_like, optional
        Segments (``nseg``) per section. A scalar applies to all; an array
        gives one value per node. Defaults to ``ceil(len_tree(tree))``, i.e.
        roughly one segment per micron, which is MATLAB's default.
    electrotonics : bool, default False
        Emit the tree's passive parameters.

    Returns
    -------
    Path

    Notes
    -----
    **MATLAB's ``.nrn`` branch cannot run as written**, in two independent
    ways, both fixed here:

    1. Its single-region path reads ``H1 (counterR)`` in the ``else`` arm of
       ``if luR > 1``, but ``counterR`` is the loop variable of the ``for``
       loop in the ``if`` arm -- never assigned when that arm is skipped. So
       exporting any tree with exactly one region raises *Undefined function
       or variable 'counterR'*.
    2. Its ``'-e'`` block reads ``tree.ri``, ``tree.rm`` and ``tree.cm``,
       lowercase. The TREES tree structure spells these ``Ri``, ``Gm`` and
       ``Cm``; there are no lowercase fields, so any ``.nrn`` export with
       passive parameters raises a missing-field error.

    See MATLAB_TOOLBOX_BUGS.md.
    """
    path = Path(path)
    if path.suffix != ".nrn":
        path = path.with_suffix(path.suffix + ".nrn")
    stem = path.stem

    if electrotonics and tree.Ri is None:
        raise ValueError("electrotonics=True needs Ri/Gm/Cm on the tree")

    rooted = root_tree(tree)
    n = rooted.n_nodes
    if res is None:
        res = np.ceil(len_tree(rooted))
    res = np.broadcast_to(np.asarray(res, dtype=float), (n,))

    parent = np.full(n, -1, dtype=int)
    coo = rooted.dA.tocoo()
    parent[coo.row] = coo.col

    regions = np.asarray(rooted.R, dtype=int)
    used = np.unique(regions)
    names = [_hoc_identifier(rooted.rnames[r], stem, prefix=True)
             for r in used.tolist()]

    # every node except the root becomes a section; number them per region
    is_section = np.ones(n, dtype=bool)
    is_section[rooted.root] = False
    slot = np.full(n, -1, dtype=int)
    region_slot = np.searchsorted(used, regions)
    counts = []
    for position in range(len(used)):
        members = np.flatnonzero((region_slot == position) & is_section)
        slot[members] = np.arange(len(members))
        counts.append(len(members))
    regions = region_slot

    write = _Writer()
    for name, count in zip(names, counts):
        write(f"create {name}[{count}]")

    nodes = np.flatnonzero(is_section)
    write("proc topolneuron() {")
    for count, node in enumerate(nodes.tolist(), start=1):
        up = parent[node]
        if up == rooted.root:
            continue  # the root is not a section; nothing to connect to
        write(f"{names[regions[up]]}[{slot[up]}] connect "
              f"{names[regions[node]]}[{slot[node]}] (0), 1")
        if count % 200 == 0:
            write(f"topolneuron{count}()")
            write("}")
            write(f"proc topolneuron{count}() {{")
    write("}")

    write("proc geometry() { local counter")
    for name, count in zip(names, counts):
        write(f"   for counter = 0,{count - 1} {{")
        write(f"      {name}[counter]{{")
        write("         pt3dclear()")
        write("         nseg = fscan()")
        write("         pt3dadd(fscan(),fscan(),fscan(),fscan())")
        write("         pt3dadd(fscan(),fscan(),fscan(),fscan())")
        if electrotonics:
            write("         insert pas")
            write(f"         Ra = {_g(tree.Ri)}")
            write(f"         g_pas = {_g(tree.Gm)}")
            write(f"         cm = {_g(tree.Cm)}")
            write("         e_pas = 0")
        write("      }")
        write("   }")
    write("topolneuron()")
    write("}")
    write("geometry()")

    # the numeric block geometry() reads back, region by region, in the same
    # order the create statements declared
    for position in range(len(names)):
        for node in nodes.tolist():
            if regions[node] != position:
                continue
            up = parent[node]
            write(" ".join(f"{v:.5g}" for v in (
                res[node],
                rooted.X[up], rooted.Y[up], rooted.Z[up], rooted.D[node],
                rooted.X[node], rooted.Y[node], rooted.Z[node], rooted.D[node],
            )))

    _write(path, write.text())
    return path
