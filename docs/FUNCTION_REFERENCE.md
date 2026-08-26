# Function reference

**Auto-generated** from live introspection of `pynetrees.__all__` -- every public name, its signature and its complete docstring, exactly as `help()` would show it. Regenerate after any docstring or signature change:

```
conda run -n pynetrees python scripts/gen_function_reference.py
```

**173 public names** as of this generation (156 functions, 16 result types).

For a curated, one-line-per-function skim, see [api-overview.md](api-overview.md). This page is the detailed counterpart: full docstrings, in full.

## Contents

- [Data model](#data-model) (3)
- [I/O — SWC](#io-swc) (2)
- [I/O — MATLAB `.mtr`](#io-matlab-mtr) (2)
- [I/O — NeuroLucida](#io-neurolucida) (1)
- [I/O — `.neu`](#io-neu) (1)
- [I/O — `.nmf`](#io-nmf) (2)
- [I/O — NeuroML](#io-neuroml) (1)
- [I/O — NEURON `.hoc`](#io-neuron-hoc) (3)
- [I/O — native (`.npz`, format dispatch)](#io-native-npz-format-dispatch) (4)
- [Bundled sample data](#bundled-sample-data) (5)
- [Topology](#topology) (22)
- [Geometry and metrics](#geometry-and-metrics) (24)
- [Editing](#editing) (14)
- [Construction](#construction) (15)
- [Generative pipeline](#generative-pipeline) (10)
- [Density, hulls and space-filling](#density-hulls-and-space-filling) (13)
- [Topological description (persistent homology)](#topological-description-persistent-homology) (3)
- [Electrotonics](#electrotonics) (14)
- [Statistics and comparison](#statistics-and-comparison) (10)
- [Plotting](#plotting) (12)
- [Image stacks](#image-stacks) (8)
- [NEURON simulation](#neuron-simulation) (4)
- [Blender export (optional)](#blender-export-optional) (5)

## Data model

The `Tree` container itself, plus validation.

### `NO_PARENT`

Value: `-1`

### `Tree`

A neuronal tree: nodes plus directed parent adjacency and metrics.

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

### `ver_tree(tree: 'Tree', quiet: 'bool' = False) -> 'list[str]'`

Verify internal consistency of a :class:`Tree`.

Mirrors ``IO/ver_tree.m``: never raises, collects every problem found and
(unless ``quiet``) emits each as a :func:`warnings.warn` call, matching
the MATLAB original's "warn, don't fail" behavior so that intentionally
incomplete trees mid-pipeline (e.g. before a future ``repair_tree``) don't
break callers that only want a health check.

Returns
-------
list[str]
    Human-readable problem descriptions; empty if the tree is well-formed.

---

## I/O — SWC

### `load_swc(path: 'str | Path') -> 'Tree | list[Tree]'`

Load an SWC file into one :class:`Tree`, or a list if it has >1 root.

### `save_swc(tree: 'Tree', path: 'str | Path') -> 'None'`

Write a single-root :class:`Tree` to an SWC file.

Node ``i`` is written at SWC index ``i + 1``; the root's parent column
is ``-1``. Region names round-trip through the SWC integer ``type``
column when they parse as integers (as produced by :func:`load_swc`),
otherwise they're replaced by a 1-based region index.

---

## I/O — MATLAB `.mtr`

### `load_mtr(path: 'str | Path', variable: 'str | None' = None) -> 'Tree | list[Tree]'`

Load a MATLAB `.mtr`/`.mat` file into a Tree, or a list of Trees.

Parameters
----------
path : str or Path
variable : str, optional
    Which workspace variable holds the tree(s). By default the loader
    takes the sole variable that looks like tree data, whatever it is
    called. Name it explicitly when a file holds several candidates.

Returns
-------
Tree or list[Tree]
    A single Tree if the file holds exactly one, else a list -- the
    common case for `.mtr` archives, most of which bundle a whole
    population of reconstructions.

Raises
------
ValueError
    If no tree-shaped variable is found, if ``variable`` names one that
    isn't there, or if several candidates exist (the message lists them
    so you can pick).

Notes
-----
**The name of the variable carries no weight.** Two earlier rules were
both wrong: requiring it to be called exactly ``tree`` rejected any
workspace saved by hand or by T2N, and *preferring* ``tree`` when
several candidates existed silently loaded one population out of a file
holding two, with nothing to say the rest had been dropped. What a
variable is called is not evidence about which one you meant, so a file
with one candidate loads it and a file with several refuses.

### `save_mtr(trees: 'Tree | list[Tree]', path: 'str | Path') -> 'Path'`

Write one tree, or a list of them, to a MATLAB-readable ``.mtr``.

Ports ``IO/save_tree.m``, which is a one-liner: ``save (name, 'tree',
'-v7.3')``. A ``.mtr`` is just a ``.mat`` workspace holding a variable
called ``tree``, which is either a struct or a cell array of structs.

Returns the path written.

Notes
-----
**This writes v5, not v7.3.** MATLAB's `save_tree` forces ``-v7.3``
(HDF5); this uses ``scipy.io.savemat``, which writes the older v5
container. MATLAB's `load_tree` calls plain ``load``, which reads
either, so nothing on the MATLAB side can tell the difference. The
reason to prefer v5 is that writing a MATLAB *struct* into v7.3 by hand
means reproducing undocumented ``MATLAB_class`` attributes and object
references, i.e. reimplementing a format MATLAB never specified --
whereas ``savemat`` is a maintained, tested writer for the format
MATLAB has documented for decades. The only real v5 limit, 2 GB per
variable, is orders of magnitude beyond any morphology.

Region names round-trip as ``rnames`` and the electrotonic constants as
``Ri``/``Gm``/``Cm``; the ``R`` indices are converted back to MATLAB's
1-based convention on the way out.

---

## I/O — NeuroLucida

### `load_neurolucida(path: 'str | Path', repair: 'bool' = True) -> 'Tree | list[Tree]'`

Load a NeuroLucida ASCII (.asc) file into a Tree, or a list of Trees
if it contains more than one (typically one per soma/dendrite/axon).

If ``repair`` (default), each resulting tree is passed through
:func:`~pynetrees.repair_tree` before being returned.

---

## I/O — `.neu`

### `load_neu(path: 'str | Path', keep_sections: 'bool' = False) -> 'Tree | list[Tree]'`

Load a NEURON ``.neu`` file.

Parameters
----------
path : str or Path
keep_sections : bool, default False
    Make every NEURON section its own region. By default the bracketed
    index is stripped, so ``axon[0]``, ``axon[1]``, ... collapse into a
    single ``axon[]`` region -- which is almost always what is wanted,
    since NEURON's section names are an implementation detail of the
    model, not an anatomical labelling. MATLAB spells this ``'-ks'``.

Returns
-------
Tree or list[Tree]
    A list when the file holds several unconnected cells.

Notes
-----
MATLAB skips the file header by reading exactly 16 whitespace-separated
tokens (``textscan (neufid, '%s', 16)``) -- the token count of the
three header lines its own writer happens to emit. Any other comment
text silently shifts the parse. This reader locates the
``# section lines:`` and ``# 3d points:`` markers instead, so a file
with a different preamble still loads.

MATLAB additionally rejects any file where a section attaches at its
*own* ``1`` end (``'sorry!! I assume that each new branch is attached
at 0 end'``); the same restriction applies here, with the reason spelled
out, because such a section would run backwards relative to its point
list.

---

## I/O — `.nmf`

### `load_nmf(path: 'str | Path') -> 'Tree'`

Load a ``.nmf`` file into a :class:`Tree`.

Notes
-----
``/swc/r`` is a radius; :attr:`Tree.D` is a diameter, so it is doubled
on the way in and halved on the way out -- the same convention MATLAB
uses, and the usual source of factor-of-two errors between SWC-family
formats.

### `save_nmf(tree: 'Tree', path: 'str | Path') -> 'Path'`

Write a :class:`Tree` to a ``.nmf`` file.

Returns the path written, so a caller that let the suffix be added
knows where the file went.

---

## I/O — NeuroML

### `save_neuroml(tree: 'Tree', path: 'str | Path', version: 'str' = '2', *, segment_groups: 'bool' = True) -> 'Path'`

Write a tree's morphology as NeuroML.

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

---

## I/O — NEURON `.hoc`

### `save_hoc(tree: 'Tree', path: 'str | Path', *, style: 'str' = 'cell', electrotonics: 'bool' = False, run_file: 'bool' = False) -> 'Path'`

Write a tree as a NEURON ``.hoc`` file.

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

### `save_nrn(tree: 'Tree', path: 'str | Path', res=None, *, electrotonics: 'bool' = False) -> 'Path'`

Write a tree as a NEURON ``.nrn`` file: one section per *segment*.

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

### `t2n_interface(tree: 'Tree', stem: 'str' = 'cell') -> 'np.ndarray'`

Where each tree node ends up in the exported hoc sections.

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

---

## I/O — native (`.npz`, format dispatch)

### `load_npz(path: 'str | Path') -> 'Tree'`

Load a Tree previously written by :func:`save_npz`.

### `load_tree(path: 'str | Path', **kwargs) -> 'Tree | list[Tree]'`

Load a tree from any format this port reads, chosen by extension.

==========  ============================================================
``.npz``    pynetrees' own lossless format (:func:`save_npz`)
``.mtr``    MATLAB tree archive -- also ``.mat``
``.swc``    the standard SWC text format
``.neu``    NEURON transfer format (see :func:`~pynetrees.io.load_neu`)
``.nmf``    the toolbox's HDF5 extended SWC
``.asc``    Neurolucida ASCII
==========  ============================================================

Extra keyword arguments are passed to the format's own loader, e.g.
``load_tree("cell.neu", keep_sections=True)``. A path with no extension
at all is taken as ``.npz``, which is what `save_tree` writes when given
the same.

Returns a single :class:`Tree`, or a list when the file holds several.

Notes
-----
Two things MATLAB's `load_tree` does that this does not.

It opens a **file dialog** when called with no argument. That is a
reasonable default for a GUI-first toolbox and a bad one for a library:
a function that blocks on a window cannot be called from a script, a
notebook running headless, or a test. Use a file dialog in your own
code if you want one; this function needs a path.

It also applies :func:`~pynetrees.repair_tree` automatically to
``.swc``/``.neu``/``.nmf`` (its ``'-r'`` default), silently altering
what was on disk. Loading and repairing are kept separate here so that
"what does this file contain" has an answer -- call
``repair_tree`` yourself when you want it.

### `save_npz(tree: 'Tree', path: 'str | Path') -> 'Path'`

Save a Tree to pynetrees' native ``.npz``-based format.

### `save_tree(tree: 'Tree | list[Tree]', path: 'str | Path', **kwargs) -> 'Path'`

Save a tree in the format named by ``path``'s extension.

==========  ============================================================
``.npz``    pynetrees' own lossless format -- the default and the only one
            that stores everything a :class:`Tree` holds
``.mtr``    MATLAB tree archive, for handing work back to MATLAB
``.swc``    standard SWC (loses region *names*, keeps their codes)
``.nmf``    HDF5 extended SWC
``.hoc``    NEURON cell file -- ``style="template"`` for a template
``.nrn``    NEURON, one section per graph segment
``.xml``    NeuroML
==========  ============================================================

These last three are **export only**: nothing here reads them back, so
a tree saved as ``.hoc`` cannot be reloaded with :func:`load_tree`.

Only ``.mtr`` accepts a list of trees; the rest hold one tree per file.
Returns the path actually written -- the extension is added if missing,
so this is not always the path passed in.

---

## Bundled sample data

### `dLPTCs_trees() -> 'dict[str, list[Tree]]'`

A population of 55 dipteran lobula-plate tangential cells, in 5 groups.

The fixture MATLAB's ``stats_tree`` examples use, and what this port's
group-comparison API was built for::

    groups = dLPTCs_trees()
    stats = stats_tree(list(groups.values()), group_names=list(groups))

Returns
-------
dict[str, list[Tree]]
    Group name -> trees. Group names follow the cell classes in the
    original archive (``HSE``, ``HSN``, ...); the archive itself stores
    them positionally, so they are recovered from each tree's own
    ``name`` field where possible and numbered otherwise.

Notes
-----
This file could not be read at all before Design Decision #52: at its
nesting depth ``scipy.io.loadmat`` returns ``mat_struct`` objects even
with ``simplify_cells=True``, which the ``.mtr`` flattener did not
recognise.

### `hsn_tree() -> 'Tree'`

A full HSN cell (1290 nodes) -- MATLAB's ``hsn_tree``.

Returns
-------
Tree

### `hss_tree() -> 'Tree'`

A full HSS cell (2252 nodes) -- MATLAB's ``hss_tree``.

This is the tree ``sample_tree()`` used to return, before Design
Decision #51 restored MATLAB's meaning of that name -- but loaded from
``.mtr``, so unlike the old SWC version it carries its real
``axon``/``dend``/``soma`` regions and its original orientation.

Returns
-------
Tree

### `sample2_tree() -> 'Tree'`

A minimal 15-node sample tree -- MATLAB's ``sample2_tree``.

Small enough that you can read its full node table at a glance, which
makes it the right fixture for doctests and for reasoning about an
algorithm by hand.

Returns
-------
Tree

### `sample_tree() -> 'Tree'`

A subtree of a sample HSN cell (197 nodes) -- MATLAB's ``sample_tree``.

The toolbox's default example morphology: small enough to plot, print
and step through, but a real reconstruction with real branch structure
and two regions (``dend``, ``soma``).

Returns
-------
Tree

---

## Topology

Needs only `dA` — parents, children, branch points, path length, ordering, sub-trees.

### `BLO_tree(tree: 'Tree', v: 'np.ndarray | None' = None, *, by: 'str' = 'nodes') -> 'BranchLengthOrder'`

Branch length order: decompose the tree into paths, deepest first.

Repeatedly takes the deepest remaining root-to-tip path, calls it the
next branch, and removes it; branch 2 is the deepest path hanging off
branch 1, and so on. Unlike :func:`BO_tree` and :func:`strahler_tree`,
which label nodes by local topology, this cuts the arbor into whole
paths -- which is what makes it the foundation of the
persistent-homology description of a morphology
(:func:`pynetrees.barcode_tree`), where each branch becomes one bar.

Parameters
----------
tree : Tree
v : array_like, optional
    Per-node values accumulated into ``length`` and ``cumulative``.
    Defaults to ``len_tree(tree)``, so branches are measured in
    microns. **With** ``by="nodes"`` **this does not affect the
    ordering** -- see the Notes.
by : {'nodes', 'length'}, keyword-only, default 'nodes'
    What "deepest" means. ``'nodes'`` counts nodes carrying ``v > 0``,
    which is what MATLAB does. ``'length'`` maximises accumulated
    ``v``, which is what MATLAB's name and documentation describe.

Returns
-------
BranchLengthOrder
    ``(order, length, cumulative)``, each ``(n_nodes,)``. ``order`` is
    **1-based** -- it is a rank, not an index.

Notes
-----
A branch's length includes the segment joining it to its parent branch,
which is what makes consecutive bars in the barcode abut rather than
leave a gap.

**The default is not what the MATLAB function's name suggests, and the
default is deliberate.** MATLAB selects each branch with
``max (sum (V0 (ipar + 1) > 0, 2))`` -- the *count* of path nodes with
a positive value. It never sums ``V``. Two consequences, both
measurable:

- Branch 1 is the path with the most nodes, not the longest one. On
  ``hsn_tree`` MATLAB's first branch ends 319.5 um from the root while
  the furthest tip is at 648.4 um.
- Any strictly positive ``v`` gives an identical ordering, so ``v``
  selects nothing. Passing ``eucl_tree`` or ``ones`` changes only the
  measured lengths.

``by="nodes"`` is kept as the default so that barcodes match MATLAB's
and published analyses reproduce; ``by="length"`` does what the name
says. They disagree about where 69-97% of nodes belong on the bundled
trees, so this is not a detail. Recorded in ``MATLAB_TOOLBOX_BUGS.md``.

MATLAB rebuilds `ipar_tree`'s dense ``n_nodes x max_depth`` matrix and
rescans it once per branch. This is the same decomposition computed
with a heap in ``O(n log n)``; ``tests/test_persistence.py`` checks the
two agree node-for-node against MATLAB's own output.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape. ``v`` may be given
    once for the whole group or as one value per tree -- see
    :mod:`pynetrees._population`.

### `BO_tree(tree: 'Tree') -> 'np.ndarray'`

Branch order of each node: how many branch points lie between it and
the root (root itself has branch order 0).

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `B_tree(tree: 'Tree') -> 'np.ndarray'`

Boolean mask of branch points (more than one child).

Accepts a **list of trees** (or a list of lists of trees) as well as a
single one, returning results in the same shape -- see
:mod:`pynetrees._population`.

### `BranchLengthOrder`

Result of :func:`BLO_tree`.

| Field | Description |
|---|---|
| `order` | Alias for field number 0 |
| `length` | Alias for field number 1 |
| `cumulative` | Alias for field number 2 |

### `C_tree(tree: 'Tree') -> 'np.ndarray'`

Boolean mask of continuation points (exactly one child).

Accepts a **list of trees** (or a list of lists of trees) as well as a
single one, returning results in the same shape -- see
:mod:`pynetrees._population`.

### `LO_tree(tree: 'Tree') -> 'np.ndarray'`

Level order: for each node, its own topological path length plus the
path lengths of every node below it -- a near-unique isomorphism
invariant, used by :func:`sort_tree`'s ``'lo'`` mode as a tie-breaker.

Written as the O(n_nodes) recurrence it actually is: each node's
descendant sum is its children's descendant sums plus their own path
lengths, accumulated bottom-up.

MATLAB reaches the same quantity by repeatedly multiplying a sparse
matrix by ``dA`` until the root column empties -- effectively summing
path lengths one generation at a time. That is a natural MATLAB idiom
and fast there, but a literal transliteration performs one SciPy
sparse matmul per tree *level* (1624 of them on a real granule cell)
and the per-call overhead dominates: 512 ms against ~6 ms here. The
equivalence ``LO == PL + (sum of PL over descendants)`` was verified
exactly (max abs difference 0.0) on hand-built trees and on both
bundled reconstructions before this replaced the transliteration.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `PL_tree(tree: 'Tree') -> 'np.ndarray'`

Topological path length (number of edges) from each node to the root.

The root has path length 0, its children 1, and so on -- so this is node
depth, and it is computed as ``PL[node] = PL[parent] + 1`` in pre-order,
which is O(n_nodes).

MATLAB computes it by repeated sparse matrix-vector multiplication, one
per depth level. That is idiomatic and fast in MATLAB, but transliterated
to SciPy it costs one Python-level call per level -- 1624 of them on a
real granule cell, ~37 ms against ~2 ms here. See also :func:`LO_tree`.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `Pvec_tree(tree: 'Tree', v: 'np.ndarray | None' = None) -> 'np.ndarray'`

Cumulative sum of ``v`` along the path from the root to each node
(inclusive of the node itself).

Parameters
----------
tree : Tree
v : np.ndarray, optional
    Per-node quantity to accumulate. Defaults to
    :func:`~pynetrees.len_tree`, giving **metric path length from the
    root [um]** -- overwhelmingly the intended meaning, and what six
    call sites inside this toolbox alone were spelling out longhand as
    ``Pvec_tree(tree, len_tree(tree))``. Pass ``np.ones(n_nodes)`` for
    topological depth + 1, or any other per-node array.

Returns
-------
np.ndarray
    Float array of length ``n_nodes``.

Notes
-----
Computed by the recurrence ``P[node] = P[parent] + v[node]`` in
pre-order, which is O(n_nodes). The previous version summed a prebuilt
``ipar_tree`` matrix instead -- correct, but that matrix is
``n_nodes x max_depth`` (49 MB, 6.1M entries for a real granule cell),
so it was the worst-scaling function in the toolbox at 3.4x superlinear.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape. ``v`` may be given
    once for the whole group or as one value per tree -- see
    :mod:`pynetrees._population`.

### `T_tree(tree: 'Tree') -> 'np.ndarray'`

Boolean mask of termination points (no children).

Accepts a **list of trees** (or a list of lists of trees) as well as a
single one, returning results in the same shape -- see
:mod:`pynetrees._population`.

### `asym_tree(tree: 'Tree', vec: 'np.ndarray | None' = None, van_pelt: 'bool' = False) -> 'np.ndarray'`

Asymmetry ratio at each branch point (NaN elsewhere): the smaller of
the two daughter subtrees' summed ``vec`` over the total (default
``vec``: terminal count). Requires strictly binary branch points --
run ``repair_tree`` first if the tree may have trifurcations.

With ``van_pelt=True``, uses Van Pelt's tree-asymmetry index instead:
``abs(v1 - v2) / (v1 + v2 - 2)``.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape. ``vec`` may be given
    once for the whole group or as one value per tree -- see
    :mod:`pynetrees._population`.

### `child_tree(tree: 'Tree', v: 'np.ndarray | None' = None) -> 'np.ndarray'`

For each node, the sum of ``v`` over *all* of its descendants
(excluding itself). Default ``v`` is all-ones, giving descendant counts.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape. ``v`` may be given
    once for the whole group or as one value per tree -- see
    :mod:`pynetrees._population`.

### `dissect_tree(tree: 'Tree', by_region: 'bool' = True, *, with_positions: 'bool' = False)`

Group nodes into sections delimited by branch points, termination
points, and (optionally) region changes.

Returns an ``(n_sections, 2)`` array of ``(start_node, end_node)`` pairs.
Reimplemented as a direct per-cut-point ancestor walk rather than
MATLAB's `ipar`/`cumsum` index trick (which the MATLAB docstring itself
flags as "isn't completely correct yet at the root") -- this version
handles the root the same way as every other cut point, no `root_tree`
workaround needed. The MATLAB second output (per-node section index and
relative position, used for NEURON `nseg` bookkeeping) isn't ported yet;
add it if/when Phase 11 (T2N) needs it.

A region-change cut is placed at the *parent* of the first node in the
new region, not at that node itself: the transitioning node's own
segment already belongs entirely to the new region, so it starts the
new section rather than ending the old one (matching MATLAB's
`iR = idpar(tree.R ~= tree.R(idpar))` -- indexing by the *parent*).
Getting this backwards (marking the transition node itself, as an
earlier version of this port did) silently produces an extra, spurious
section split at every region boundary -- caught while building Phase
11's NEURON bridge, which exercises region-based sectioning for the
first time.

The root is never treated as the *end* of a section, regardless of why
it was cut (a region change at its own child, or the root genuinely
being a branch point -- common in real reconstructions, e.g. a soma
branching directly into several dendrites). There's nothing before the
root to split off, so a section reaching it just extends all the way
back via the ancestor walk's own stopping condition; treating the root
as its own degenerate end-of-section produced a spurious
root-to-itself entry, which callers building a new `dA` from `sect`
(e.g. `resample_tree`) turned into an actual self-loop -- caught by
testing against a real reconstruction whose root does branch, which
none of the earlier hand-built test fixtures did.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `idchild_tree(tree: 'Tree', nodes=None, first_only: 'bool' = False) -> 'np.ndarray'`

Direct child indices of each node in ``nodes`` (default: all nodes).

Returns an ``(len(nodes), width)`` int array, :data:`NO_PARENT`-padded,
where ``width`` is the largest number of children found (MATLAB hardcodes
``width=2``, silently truncating any trifurcation; this port doesn't).

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `idpar_tree(tree: 'Tree', root_self: 'bool' = True) -> 'np.ndarray'`

0-based index of each node's direct parent.

Parameters
----------
tree : Tree
root_self : bool, default True
    What the root gets, since it has no parent. ``True`` (MATLAB's
    default) makes the root its own parent, which lets expressions like
    ``v[idpar]`` be written without a special case -- ``ratio_tree``
    relies on it to give the root a ratio of exactly 1. ``False``
    (MATLAB's ``'-z'``) gives the root :data:`NO_PARENT` (``-1``)
    instead, which is what you want when you are about to *walk* the
    parent chain and need a stopping condition.

Returns
-------
np.ndarray
    Integer array of length ``n_nodes``.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `ipar_tree(tree: 'Tree', terminals_only: 'bool' = False, nodes=None) -> 'np.ndarray'`

Ancestor path of every node: ``[i, parent(i), ..., root]``.

Parameters
----------
tree : Tree
terminals_only : bool, default False
    MATLAB's ``'-T'``. Return one row per **termination point**, each
    holding only the path back to (and excluding) its first branch
    point -- i.e. that terminal's own unbranched segment.
nodes : array_like, optional
    Restrict to these nodes' rows. With ``terminals_only``, selects
    which terminals (MATLAB's ``ipart``).

Returns
-------
np.ndarray
    ``(n_rows, max_depth + 2)`` int array, :data:`NO_PARENT`-padded.

Notes
-----
The full matrix is this toolbox's worst-scaling structure: it is dense
and ``n_nodes x max_depth``, which is 49 MB for a 3765-node granule cell
and remains ~3x superlinear (see `docs/port-audit.md`). Most callers
only need a traversal -- ``Pvec_tree``, ``PL_tree``, ``flatten_tree``,
``morph_tree`` and ``smooth_tree`` were all moved off it for exactly
that reason. Reach for it when you genuinely need arbitrary ancestor
queries, and prefer ``terminals_only=True`` when you need terminal
segments, since that form is dramatically smaller.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `ratio_tree(tree: 'Tree', v: 'np.ndarray | None' = None) -> 'np.ndarray'`

Ratio of ``v`` at each node to ``v`` at its parent (root: 1.0).

Accepts a **list of trees** (or a list of lists of trees) as well as a
single one, returning results in the same shape. ``v`` may be given
once for the whole group or as one value per tree -- see
:mod:`pynetrees._population`.

### `redirect_tree(tree: 'Tree', new_root: 'int', name: 'str | None' = None, *, full_output: 'bool' = False)`

Reroot the tree at ``new_root``, reversing edge direction as needed.

Parameters
----------
tree : Tree
new_root : int
    0-based index of the node to become the new root.
name : str, optional
    Name for the returned tree; defaults to the input's.
full_output : bool, default False
    If ``True``, return a :class:`RedirectResult` ``(tree, order)``
    instead of just the tree -- ``order`` being the only way to map old
    node indices onto new ones after the reindex (Design Decision #42).

Returns
-------
Tree or RedirectResult

Warns
-----
UserWarning
    If ``new_root`` is a branch point. Rerooting there leaves it a
    trifurcation, so the result is no longer binary -- matching the
    MATLAB original's documented restriction.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `rindex_tree(tree: 'Tree') -> 'np.ndarray'`

0-based rank of each node within its own region, by node order.

Accepts a **list of trees** (or a list of lists of trees) as well as a
single one, returning results in the same shape -- see
:mod:`pynetrees._population`.

### `sort_tree(tree: 'Tree', by: 'str' = 'hier', *, full_output: 'bool' = False)`

Reindex nodes to be BCT-conform: every parent precedes its children,
and each subtree occupies a contiguous index block.

Parameters
----------
tree : Tree
by : {'hier', 'lo', 'lex'}, default 'hier'
    Which of the many valid BCT orderings to produce.

    - ``'hier'`` -- keep nodes in their existing relative order, only
      fixing up parent/child adjacency. Arbitrary among the valid
      orderings, but the cheapest.
    - ``'lo'`` -- order by (topological path length, level order),
      giving a near-canonical ordering (MATLAB's ``'-LO'``).
    - ``'lex'`` -- order by number of children: terminals, then
      continuations, then branches (MATLAB's ``'-LEX'``).
full_output : bool, default False
    If ``True``, return a :class:`SortResult` ``(tree, order)`` instead
    of just the tree (Design Decision #42).

Returns
-------
Tree or SortResult

Notes
-----
``'hier'`` is a DFS pre-order rather than MATLAB's level-order-ish
scheme (Design Decision #12). Both satisfy the BCT invariant and nothing
downstream depends on which valid ordering it gets, but the consequence
is worth stating plainly: **node indices are not comparable between
MATLAB and pynetrees after a sort.** Do not cross-reference "node 417"
between the two toolboxes.

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `strahler_tree(tree: 'Tree') -> 'np.ndarray'`

Strahler number of each node (terminals are 1; a node is
``max(children) + 1`` if 2+ children tie for the max, else
``max(children)``).

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `sub_tree(tree: 'Tree', inode: 'int', with_tree: 'bool' = True) -> 'SubTree'`

The subtree rooted at ``inode``: which nodes it contains, and the tree.

Parameters
----------
tree : Tree
inode : int
    0-based index of the subtree's root.
with_tree : bool, default True
    Whether to build the extracted :class:`~pynetrees.Tree` as well as the
    mask. Pass ``False`` in a per-node loop -- it costs about 30% extra
    per call (2203 vs 1682 us on a 3765-node granule cell), which is
    cheap once but not free thousands of times over.

Returns
-------
SubTree
    Named tuple ``(mask, tree)``. Unpacks like MATLAB's
    ``[sub, subtree] = sub_tree(...)``, and ``result.mask`` also works.

Notes
-----
**Region names are trimmed** to those the subtree actually uses, with
``R`` reindexed to match. MATLAB does not do this -- ``sub_tree.m``
carries the comment *"NOTE ! region update for tree output still
missing!!!"* -- and the result there keeps the whole parent's region
list. Cutting a purely dendritic branch out of a granule cell and being
told it still has an ``axon`` region is not useful, so this port closes
the gap rather than reproducing it (Design Decision #50).

Traversal walks the child lists directly. An earlier version read each
node's children as ``dA[:, node].toarray()``, which materialises a dense
length-``n_nodes`` column *per visited node* and makes a single BFS
O(n_nodes^2) -- 514 ms on that same granule cell, against ~1.7 ms here.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `typeN_tree(tree: 'Tree') -> 'np.ndarray'`

Node type per node: 0 terminal, 1 continuation, 2 (or more) branch.

Accepts a **list of trees** (or a list of lists of trees) as well as a
single one, returning results in the same shape -- see
:mod:`pynetrees._population`.

---

## Geometry and metrics

Needs `X`/`Y`/`Z`/`D` — lengths, surfaces, volumes, angles, transforms, scaling to a target size.

### `L_tree(tree: 'Tree', dim: 'int | None' = None) -> 'float'`

Total cable length of the tree [um] -- ``len_tree(tree).sum()``.

The same number as :attr:`pynetrees.Tree.total_length`, under MATLAB's
name, so translated code reads the same. Use whichever fits: the
attribute in Python-first code, this in a line-by-line translation.

Parameters
----------
tree : Tree
dim : {2, 3}, optional
    Default 3. Pass ``2`` for the length of the XY projection, which is
    what to compare against a 2D reconstruction.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `Scaled`

Result of :func:`scaleS_tree` / :func:`scaleV_tree`.

| Field | Description |
|---|---|
| `tree` | Alias for field number 0 |
| `factor` | Alias for field number 1 |
| `error` | Alias for field number 2 |

### `angleB_tree(tree: 'Tree') -> 'np.ndarray'`

Angle (radians) between the two daughter branches at each branch
point (NaN elsewhere). Requires strictly binary branch points.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `angleBd2_tree(tree: 'Tree', dist: 'int' = 5) -> 'np.ndarray'`

Branch angle measured ``dist`` nodes out, along the longest path.

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

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `angleBd_tree(tree: 'Tree', dist: 'int' = 5) -> 'np.ndarray'`

Branch angle measured ``dist`` nodes out, along the bulkier branch.

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

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `bin_tree(tree: 'Tree | list[Tree]', v: 'np.ndarray | None' = None, bins=10)`

Bin nodes by ``v`` (default: Euclidean distance to root).

``bins`` is either a bin count or explicit bin edges. Returns
``(bin_index, edges)``; ``bin_index[i]`` is 0 if node ``i`` falls
outside every bin, else its 1-based bin number.

**A list of trees** returns ``(bin_indices, edges)`` where
``bin_indices`` is a list, one array per tree, and ``edges`` is the
single set of edges spanning the whole group. Binning each tree
separately would give every cell its own edges, and the resulting bin
numbers would not be comparable between cells -- which is the only
reason to bin a group at all.

### `cvol_tree(tree: 'Tree') -> 'np.ndarray'`

Continuous volume [1/um] of every segment, for electrotonic calculations.

Accepts a **list of trees** (or a list of lists of trees) as well as a
single one, returning results in the same shape -- see
:mod:`pynetrees._population`.

### `cyl_tree(tree: 'Tree', dim: 'int | None' = None)`

Start/end coordinates of every segment (node-to-parent).

Parameters
----------
tree : Tree
dim : {2, 3}, optional
    Default 3 (see :func:`pynetrees._compat.resolve_dim` for why the
    signature says ``None``).
    Work in 3D or project onto the XY plane (Design Decision #40).

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

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `direction_tree(tree: 'Tree', normalize: 'bool' = True) -> 'np.ndarray'`

``(n_nodes, 3)`` vector from each node's parent to the node itself.

The root has no real parent direction; it's set to node 1's direction as
a placeholder, matching the MATLAB original.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `dist_tree(tree: 'Tree', distances) -> 'np.ndarray'`

Boolean ``(n_nodes, len(distances))`` matrix: True wherever a node's
segment crosses a given path distance [um] from the root.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `eucl_tree(tree: 'Tree', point=None, dim: 'int | None' = None) -> 'np.ndarray'`

Euclidean ("as the crow flies") distance from every node to ``point``.

Parameters
----------
tree : Tree
point : int or array_like, optional
    A node index, or an explicit ``(x, y[, z])`` coordinate. Defaults to
    the tree's root -- found via :attr:`Tree.root`, not assumed to be
    node 0.
dim : {2, 3}, optional
    Default 3 (see :func:`pynetrees._compat.resolve_dim` for why the
    signature says ``None``).
    Measure in 3D, or in the XY plane only.

Returns
-------
np.ndarray
    Distance per node [um]. Contrast :func:`~pynetrees.Pvec_tree`, which
    measures *along* the tree rather than through space.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `flatten_tree(tree: 'Tree') -> 'Tree'`

Flatten a tree onto the XY plane, conserving each segment's length
(subtrees are shifted outward in X/Y to compensate for the lost Z
extent, exactly as the 3D segment length is preserved in 2D).

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `flip_tree(tree: 'Tree', axis: 'str' = 'x') -> 'Tree'`

Mirror a tree about its root along one axis.

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

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `gene_tree(tree: 'Tree') -> 'np.ndarray'`

Topological "gene" of a tree: an ``(n_branches, 2)`` array of each
branch/terminal segment's own path length and its ending node type
(2=branch, 0=terminal) -- a compact shape signature, useful for
comparing topology across trees.

MATLAB's nested-cell-array *plotting* wrapper is not reproduced -- it
draws a figure per group, which is `plot_tree`'s job here. The batch
half of it is: a list of trees gives a list of genes, and
``np.vstack`` pools them if that is what the comparison wants.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `len_tree(tree: 'Tree', dim: 'int | None' = None) -> 'np.ndarray'`

Length of every node-to-parent segment [um].

Parameters
----------
tree : Tree
dim : {2, 3}, optional
    Default 3 (see :func:`pynetrees._compat.resolve_dim` for why the
    signature says ``None``).
    3D length, or the length of the segment's XY projection.

Returns
-------
np.ndarray
    Length per node [um]. The root is 0, having no parent segment.
    ``tree.total_length`` is the sum of this.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `morph_tree(tree: 'Tree', v: 'np.ndarray | None' = None) -> 'Tree'`

Rescale every segment's length to ``v`` (default: 10 um each) while
preserving branch angles and topology -- a META-FUNCTION: e.g. passing
the original ``len_tree`` output back in regrows the original geometry
(except for originally-zero-length segments, which can't be recovered).

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape. ``v`` may be given
    once for the whole group or as one value per tree -- see
    :mod:`pynetrees._population`.

### `rot_tree(tree: 'Tree', deg=(0.0, 0.0, 90.0), mode: 'str | None' = None, *, nodes=None, exclude_regions=('axon',), align_region=None) -> 'Tree'`

Rotate a tree, either by explicit angles or onto an automatic axis.

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

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `scaleS_tree(tree: 'Tree', target: 'float' = 1000000.0, *, radius: 'int | None' = None, resample: 'bool' = True, method: 'str' = 'close', shrink: 'float' = 0.5) -> 'Scaled'`

Scale a tree until the area it spans equals ``target`` [um^2].

For putting cells of different sizes on a common footing before
comparing how they fill space -- the question :func:`pynetrees.theta_tree`
asks, which is only meaningful between arbors of the same extent.

Parameters
----------
tree : Tree
target : float, default 1e6
    Wanted spanned area [um^2].
radius : int, optional
    Passed to :func:`pynetrees.span_tree`.
resample : bool, keyword-only, default True
    Resample to 1 um afterwards (MATLAB's ``'-r'``, its default).
    Scaling up spreads the nodes out, and the spanned area is measured
    on a one-micron grid, so without this a stretched tree is measured
    from a sparser set of points than it was fitted on.
method, shrink
    Passed to :func:`pynetrees.span_tree`.

Returns
-------
Scaled
    ``(tree, factor, error)``.

Notes
-----
Scaled **twice**, as MATLAB does, because the fit does not converge in
one step: closing uses a radius fixed in microns, so a tree scaled by
``k`` does not have its spanned area scaled by ``k^2``. The second pass
corrects most of what the first misses; ``error`` reports what is left
rather than leaving the caller to discover it, which is the part MATLAB
prints to the console and discards.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape. ``target`` may be given
    once for the whole group or as one value per tree -- see
    :mod:`pynetrees._population`.

### `scaleV_tree(tree: 'Tree', target: 'float' = 1000000.0, *, shrink: 'float' = 0.5, dim: 'int | None' = None) -> 'Scaled'`

Scale a tree until the volume it encloses equals ``target``.

The 3D counterpart of :func:`scaleS_tree`, measuring with the alpha
shape around the nodes (:func:`pynetrees.boundary_tree`) rather than a
rasterised span.

Parameters
----------
tree : Tree
target : float, default 1e6
    Wanted enclosed volume [um^3], or area [um^2] when ``dim=2``.
shrink : float, keyword-only, default 0.5
    How tightly the boundary wraps; see :func:`pynetrees.boundary_tree`.
dim : {2, 3}, optional
    Default 3.

Returns
-------
Scaled
    ``(tree, factor, error)``.

Notes
-----
One pass is enough here, unlike :func:`scaleS_tree`. The shrink factor
is relative, so the alpha shape scales with the points and the volume
scales exactly as ``factor ** dim``; ``error`` is numerical rather than
systematic. MATLAB carries a ``% scale volume again`` comment followed
by no second scaling, which is consistent with the same conclusion.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape. ``target`` may be given
    once for the whole group or as one value per tree -- see
    :mod:`pynetrees._population`.

### `scale_tree(tree: 'Tree', fac=2.0, center: 'bool' = True, scale_diameter: 'bool' = True) -> 'Tree'`

Scale a tree's coordinates (and, by default, diameter) by ``fac``.

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

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `surf_tree(tree: 'Tree') -> 'np.ndarray'`

Lateral surface area of every segment [um^2].

Accepts a **list of trees** (or a list of lists of trees) as well as a
single one, returning results in the same shape -- see
:mod:`pynetrees._population`.

### `tran_tree(tree: 'Tree', offset=None) -> 'Tree'`

Translate a tree's coordinates.

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

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `vol_tree(tree: 'Tree') -> 'np.ndarray'`

Volume of every segment [um^3].

Accepts a **list of trees** (or a list of lists of trees) as well as a
single one, returning results in the same shape -- see
:mod:`pynetrees._population`.

### `zcorr_tree(tree: 'Tree', tz: 'float' = 5.0)`

Correct sudden Neurolucida-style Z jumps: any parent-child Z gap
exceeding ``tz`` [um] is subtracted from the entire downstream subtree.

Returns ``(new_tree, jumped_nodes)``.

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

---

## Editing

Structural changes: repair, resample, delete, insert, re-root.

### `abel_tree(tree: 'Tree') -> 'float'`

Average segment length [um] between branch/termination points, after
collapsing every continuation point (a measure of typical inter-branch
spacing, independent of how densely the reconstruction was sampled).

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `cat_tree(tree1: 'Tree', tree2: 'Tree', inode1: 'int | None' = None, inode2: 'int' = 0, keep_regions: 'bool' = False) -> 'Tree'`

Concatenate ``tree2`` onto ``tree1``, connecting ``tree2``'s
``inode2`` (default: its root) to ``tree1``'s ``inode1`` (default:
whichever node in ``tree1`` is closest to ``tree2``'s ``inode2``).

### `delete_tree(tree: 'Tree', inodes, keep_regions: 'bool' = False) -> 'Tree | list[Tree]'`

Delete nodes from a tree, splicing each deleted node's children to
its nearest surviving ancestor so the remaining tree(s) stay connected.

``inodes`` is a boolean mask (length ``n_nodes``) or a list/array of
node indices. If the deletion disconnects the tree (e.g. deleting a
branching root), returns a **list** of Trees, one per resulting
component, instead of a single Tree -- unlike MATLAB, whose forest-
splitting only kicks in for a specific option combination and is
documented as broken for the default case (todo list: "delete_tree |
multiple trees doesn't work yet").

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape. ``inodes`` may be given
    once for the whole group or as one value per tree -- see
    :mod:`pynetrees._population`.

### `elim0_tree(tree: 'Tree', keep_regions: 'bool' = False) -> 'Tree'`

Delete zero-length segments (except the root's own trivial one).

Accepts a **list of trees** (or a list of lists of trees) as well as a
single one, returning results in the same shape -- see
:mod:`pynetrees._population`.

### `elimt_tree(tree: 'Tree', at_root: 'bool' = True) -> 'Tree'`

Replace every multifurcation (3+ children) with a short chain of
bifurcations, each spacer offset by ~0.0001 um along the parent segment.

Parameters
----------
tree : Tree
at_root : bool, default True
    Whether to also split a multifurcating *root*. ``True`` matches
    MATLAB's default. ``False`` (MATLAB's ``'-r'``) leaves the root
    alone, which is what you want for a soma that legitimately branches
    into several primary dendrites and shouldn't grow a spacer chain.

Returns
-------
Tree
    The de-multifurcated tree, or the input unchanged if there was
    nothing to do.

Notes
-----
Previously returned ``(tree, changed)``. The flag is gone (Design
Decision #42): it is recomputable -- ``typeN_tree(result).max() <= 2``,
or simply comparing ``n_nodes`` -- and every caller was unpacking a
tuple for information almost none of them used. When nothing changes,
that fact now goes to :mod:`logging` at debug level, so library code
stays quiet by default but the information is still recoverable.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `insert_tree(tree: 'Tree', X, Y, Z, D, parent, R=None, *, full_output: 'bool' = False)`

Append new nodes to a tree.

Parameters
----------
tree : Tree
X, Y, Z, D : array_like
    Coordinates [um] and diameters [um] of the new nodes.
parent : array_like of int
    0-based parent index for each new node. Replaces MATLAB's
    ``[inode R X Y Z D idpar]`` SWC-tuple calling convention.
R : array_like of int, optional
    Region index per new node; defaults to each node's parent's region.
full_output : bool, default False
    If ``True``, return an :class:`InsertResult` ``(tree, inodes)``.

Returns
-------
Tree or InsertResult

Raises
------
ValueError
    If any ``parent[i]`` is a *forward* reference -- i.e. points at a
    new node that has not been assigned an index yet
    (``parent[i] >= N + i``), or is out of range entirely. Left
    unchecked this silently produces a cycle or an orphan.

Notes
-----
**New nodes may parent each other.** The parent index is written
straight into the adjacency matrix, so ``parent[i]`` may refer either to
an existing node (``0 <= p < n_nodes``) or to an *earlier* new node
(``n_nodes <= p < n_nodes + i``). That is not incidental --
:func:`~pynetrees.cap_tree` depends on it, chaining each cap segment onto
the previous one:

.. code-block:: python

    # three nodes in a chain hanging off existing node 0
    n = tree.n_nodes
    insert_tree(tree, X=[1., 2., 3.], Y=[0., 0., 0.], Z=[0., 0., 0.],
                D=[1., 1., 1.], parent=[0, n, n + 1])

The capability was previously undocumented and unvalidated; the
forward-reference check above is what makes it safe to rely on.

### `insertp_tree(tree: 'Tree', inode: 'int | None' = None, plens=None, *, full_output: 'bool' = False)`

Insert nodes at given path lengths along the root-to-``inode`` path.

Parameters
----------
tree : Tree
inode : int, optional
    0-based index of the node whose root path is subdivided. Defaults
    to the last node.
plens : array_like, optional
    Path lengths [um] from the root at which to insert. Defaults to
    every 10 um, or a single node at the halfway point if the path is
    shorter than 10 um. Values already occupied by a node, or beyond
    the path's end, are dropped.
full_output : bool, default False
    If ``True``, return an :class:`InsertpResult` ``(tree, added)``.
    The mask cannot be recomputed afterwards -- the result is re-sorted,
    so inserted nodes are no longer identifiable by index (Design
    Decision #42).

Returns
-------
Tree or InsertpResult

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `interpd_tree(tree: 'Tree', node1: 'int', node2: 'int') -> 'Tree'`

Linearly interpolate diameter between two nodes on the same root path.

Accepts a **list of trees** (or a list of lists of trees) as well as a
single one, returning results in the same shape -- see
:mod:`pynetrees._population`.

### `recon_tree(tree: 'Tree', ichilds, ipars, shift: 'bool' = True) -> 'Tree'`

Reconnect the subtrees rooted at ``ichilds`` to new parents ``ipars``.

If ``shift`` (default), each subtree is translated so its root lands on
its new parent's position.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape. ``ichilds``, ``ipars`` may be given
    once for the whole group or as one value per tree -- see
    :mod:`pynetrees._population`.

### `repair_tree(tree: 'Tree', no_root_trifurcation: 'bool' = False) -> 'Tree'`

Rectify a tree to full BCT conformity: eliminate multifurcations,
drop zero-length segments, and sort into canonical (level-order) index
order. Most other functions in this toolbox assume their input has
already been through this.

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `resample_tree(tree: 'Tree', sr: 'float' = 10.0, method: 'str' = 'matlab', *, extend_terminals: 'bool' = True, interp_diameter: 'bool' = False, conserve_length: 'bool' = False, collapse_branches: 'bool' = True, preserve_branch_spacing: 'bool' = False, trim_regions: 'bool' = True) -> 'Tree'`

Redistribute a tree's nodes to roughly ``sr`` [um] spacing.

Parameters
----------
tree : Tree
sr : float, default 10.0
    Target internode spacing [um].
method : {'matlab', 'anchors'}, default 'matlab'
    Which abstraction to use for the bits resampling leaves
    underdetermined -- MATLAB's own docstring says "some abstraction
    principles need to be arbitrarily set", and the two methods set
    them differently.

    - ``'matlab'`` -- a faithful port of `resample_tree.m`. Every node
      in the result sits at an exact multiple of ``sr`` path length
      from the root, because *all* original nodes are deleted after the
      grid points are inserted. Branch and termination points therefore
      **move** onto the grid.
    - ``'anchors'`` -- branch and termination points stay exactly where
      they were, and only the nodes between them are redistributed.
      Better when you care about branch-point positions (the NEURON
      bridge does), but it is not what MATLAB computes.

extend_terminals : bool, default True
    Stretch each terminal segment by ``sr / 2`` first, so the grid
    does not systematically truncate branch tips. MATLAB does this
    unconditionally; here it is switchable.
interp_diameter : bool, default False
    MATLAB's ``'-d'``. Interpolate diameters along each segment rather
    than inheriting the child node's. Changes total surface and volume,
    which is why it is off by default.
conserve_length : bool, default False
    MATLAB's ``'-l'``. After resampling, stretch every segment back to
    exactly ``sr`` so total path lengths match the original. The tree
    grows slightly overall, so this is wrong for automated
    reconstruction pipelines and right for length-preserving analysis.
collapse_branches : bool, default True
    Merge branch daughters that end up implausibly close together
    (within 0.75 * 2 * ``sr`` of path length of each other). MATLAB's
    ``'-v'`` switches this *off*; the sense is inverted here per Design
    Decision #41.
preserve_branch_spacing : bool, default False
    MATLAB's ``'-b'``. Lengthen sub-``sr`` segments that run between two
    branch points, so consecutive branch points do not collapse into a
    multifurcation. MATLAB's own docstring warns this "does not
    preserve length" and "might give a mess with high sr".
trim_regions : bool, default True
    Drop region names left unused after resampling. MATLAB's ``'-r'``
    switches this off; inverted here per #41.

Returns
-------
Tree

Notes
-----
``method='matlab'`` is the default as of Design Decision #45, reversing
#23. The port originally shipped only the anchor-preserving variant, on
the grounds that MATLAB's snapping rule is arbitrary -- which is true,
but "arbitrary" is not the same as "wrong", and defaulting to something
other than the reference implementation makes every downstream number
quietly incomparable.

A single-node tree has nothing to resample and is returned unchanged.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `restrain_tree(tree: 'Tree', maxpl: 'float' = 400.0, interpolate: 'bool' = True) -> 'Tree'`

Prune a tree so no node exceeds path length ``maxpl`` [um] from the
root. If ``interpolate`` (default), terminal points beyond ``maxpl``
are pulled back to exactly ``maxpl`` along their original direction
rather than simply deleted.

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `root_tree(tree: 'Tree') -> 'Tree'`

Prepend a near-zero-length segment at the root (some downstream
algorithms rely on the root having exactly one child).

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `rootangle_tree(tree: 'Tree') -> 'np.ndarray'`

Angle (radians) between each segment and the straight line from the
root to that segment's end, computed on a 1 um resampling of the tree.

Centers the tree on its root first -- MATLAB's version measures against
the coordinate origin directly, which only equals "distance to root" if
the tree happens to already be centered there; this port's explicit
`tran_tree` call makes the "line to root" in the docstring correct
regardless of the tree's absolute position.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

---

## Construction

Synthetic trees: `MST_tree`, BCT enumeration, growth, smoothing, soma/diameter models.

### `BCT_tree(bct) -> 'Tree'`

Build a (topology-only) Tree from a B/C/T children-count sequence.

Coordinates are all zero -- this constructs pure topology, useful for
testing and enumerating isomorphism classes (:func:`allBCTs_tree`,
:func:`allBTs_tree`). MATLAB's version optionally attaches a fake
dendrogram layout via `xdend_tree`; that's a Phase 7 (graphical)
concern and isn't ported here.

### `Growth`

Result of :func:`growth_tree`.

| Field | Description |
|---|---|
| `tree` | Alias for field number 0 |
| `length` | Alias for field number 1 |
| `terminals` | Alias for field number 2 |
| `attached_to` | Alias for field number 3 |
| `target` | Alias for field number 4 |
| `targets` | Alias for field number 5 |
| `history` | Alias for field number 6 |

### `MST_tree(X, Y, Z=None, start=0, bf: 'float' = 0.4, thr: 'float' = 50.0, mplen: 'float' = 10000.0, avoid_multifurcations: 'bool' = False, *, dist=None, cut_ends: 'bool' = False, record: 'bool' = False, full_output: 'bool' = False)`

Grow synthetic tree(s) connecting a cloud of points.

At each step the cheapest available attachment is made, where connecting
point ``p`` to tree node ``t`` costs::

    distance(p, t)  +  bf * path_length(t)  [+ dist penalty]

balancing minimal total wiring against minimal conduction path length --
the Cuntz/Borst/Segev construction the toolbox is named after.

Parameters
----------
X, Y, Z : array_like
    Coordinates of the points to connect. ``Z`` defaults to zeros.
start : int, sequence of int, Tree, or list[Tree], default 0
    Where to grow from. An **index** picks one of the points in the
    cloud; **several indices** grow several trees at once, competing
    for the same cloud, so that territories fall out of the growth
    rather than being assigned -- this is how a population is grown
    into a shared field, and what MATLAB's multi-`msttrees` mode is
    for.

    A **Tree** (or a list of them) instead continues growing an
    existing morphology: every one of its nodes is a valid attachment
    point, its path lengths carry over into the balancing term, and its
    diameters and regions are preserved in the result. Nodes added by
    the growth are labelled with a region called ``"new"``, which is
    what :func:`~pynetrees.generate.clone_tree` renames per region.
bf : float, default 0.4
    Balancing factor in ``[0, 1]``. ``0`` minimises wiring alone,
    giving long meandering paths to the root; ``1`` minimises path
    length, giving a star.
thr : float, default 50.0
    Maximum span [um] of any single connection.
mplen : float, default 10000.0
    Maximum path length [um] from the root; points beyond it stay
    unconnected.
avoid_multifurcations : bool, default False
    MATLAB's ``'-b'``. Refuse a third child on any node. Some points may
    then stay unconnected even within ``thr``.
dist : scipy.sparse matrix, optional
    MATLAB's ``DIST``: an ``(n_points, n_points)`` matrix of connection
    *preferences*, where larger means more likely and zero means "no
    particular reason to connect". Enters the cost as
    ``max(dist) * (1 - dist[t, p] / max(dist))``, so the most-preferred
    pairing pays nothing extra and an unlisted one pays the full range.

    Indexed over the **input points only** -- not over the nodes of a
    seed tree, which never need a preference because they are already
    connected. MATLAB instead requires the caller to index it over the
    growing trees' own nodes as well ("Don't forget to include input
    tree nodes into the distance matrix DIST!"), which is easy to get
    wrong and impossible to check.
cut_ends : bool, default False
    MATLAB's ``'-c'``. Grow only from points that have at least one
    positive entry in ``dist`` -- the marked "cut ends". Requires
    ``dist``.
record : bool, default False
    MATLAB's ``'-t'``. Also return the growth history.
full_output : bool, default False
    Return :class:`MSTResult` rather than just the tree(s).

Returns
-------
Tree or list[Tree] or MSTResult
    A single Tree for a single start point, a list for several.

Notes
-----
Not a literal port: MATLAB's ~600-line version hand-maintains a
shrinking "vicinity window" per tree, re-sorted and re-sliced every
iteration, to avoid recomputing an O(n^2) distance matrix. This uses
`scipy.spatial.cKDTree` for the radius queries and a lazy-deletion
min-heap for "cheapest valid candidate", the standard formulation for
Prim's-style growth where a node's best known cost only improves
(Design Decision #27).

``record`` returns the growth **log**, not a list of intermediate trees
as MATLAB does: any intermediate state is a prefix of the log, so
storing whole trees per step would be quadratic in memory for
information already present.

### `allBCTs_tree(n: 'int' = 8, with_trees: 'bool' = False)`

All non-isomorphic B/C/T topologies with ``n`` nodes.

Brute-force over all ``3**n`` sequences -- "gets very slow very
quickly" per the MATLAB docstring; the small default matches that.

### `allBTs_tree(n: 'int' = 15, with_trees: 'bool' = False)`

All non-isomorphic binary (branch/terminal only, no continuation)
topologies with ``n`` nodes. Only achievable for select (odd) ``n``,
by the definition of a full binary tree.

### `cap_tree(tree: 'Tree', spacing: 'float' = 1.0) -> 'Tree'`

Cap the tree's open root end with a rounded (hemispherical) profile.

A flat-cut soma looks artificial and, more importantly, under-counts
membrane area at the very place where input resistance is measured. This
adds a short chain of tapering segments extending *backwards* from the
root -- away from the tree -- whose diameters trace a spherical cap of
the root's own diameter.

Parameters
----------
tree : Tree
spacing : float, default 1.0
    Distance [um] between successive cap nodes.

Returns
-------
Tree
    The tree with cap nodes appended, or the input unchanged if the root
    is too thin for even one cap node at this ``spacing``.

Notes
-----
The cap grows from :attr:`Tree.root` along the *reverse* of that node's
own segment direction. MATLAB's ``cap_tree.m`` hardcodes ``tree.X(1)``
and ``direction(2, :)``, and this port transliterated both -- correct
only after ``sort_tree``. On a tree whose root sits elsewhere it capped
the wrong end entirely (Design Decision #48).

MATLAB's ``'-a'`` axon-adding option is deliberately not ported here: it
draws length, diameter and taper from constants fit to one published
dataset, which makes it a dataset-specific generator rather than part of
a capping algorithm, and folding it into this function makes it easy to
apply by accident.

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `clean_tree(tree: 'Tree', radius: 'float' = 1.0) -> 'Tree'`

Delete improbable terminal branches: ones that end within
``D/2 + radius/2`` of a node on a *different* branch (likely a
reconstruction/generation artifact), or whose total length is under
``radius``. At most one terminal branch is removed per branch point
per call -- run repeatedly for further cleanup, as the MATLAB
docstring also recommends.

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `growth_tree(span, start=None, thr: 'float' = 100, bf: 'float' = 0.5, sp: 'float' = 0.5, k: 'float' = 0.0, *, alpha: 'float' = 0.5, jitter=(), max_radius: 'float' = 10000.0, n_target_points: 'int' = 100000, stop: 'str' = 'steps', history: 'bool' = False, rng=None) -> 'Growth'`

Grow a tree into a volume, balancing wiring cost against filling space.

:func:`MST_tree` connects a *given* cloud of points as cheaply as it
can. This grows into a *region*: at each step it picks the next target
by trading off three things -- the cost of reaching it (``bf``, as in
`MST_tree`), how far it lies from the arbor so far (``sp``, which
pushes growth into empty territory rather than thickening what is
already there), and chance (``k``). The result is a cell that fills its
domain rather than one that merely spans a sample of it.

Parameters
----------
span : Tree, array_like, or Boundary
    The territory to grow into. A **Tree** uses the alpha shape around
    its nodes, so "grow a new cell shaped like this one" is one call. An
    ``(n, 3)`` **array** is taken as the target points themselves
    (MATLAB's ``-P``). A :class:`~pynetrees.Boundary` is used directly.
start : Tree or array_like, optional
    An existing morphology to continue growing, or a root coordinate.
    Defaults to the origin -- or to ``span``'s root if ``span`` is a
    tree.
thr : float, default 100
    When to stop; read according to ``stop``.
bf : float, default 0.5
    Balancing factor, exactly as in :func:`MST_tree`: how much the path
    length back to the root counts against the cost of a connection.
sp : float in [0, 1], default 0.5
    Space-filling factor. ``0`` reduces this to an ordinary minimum
    spanning tree over the targets reached so far; higher values prefer
    targets far from any existing cable.
k : float in [0, 1], default 0
    Stochasticity. ``0`` is deterministic given the target cloud.
alpha : float, keyword-only, default 0.5
    Shrink factor for the boundary, when one is derived from a tree.
jitter : sequence of (stde, lam) pairs, keyword-only
    Make new stretches of cable wander instead of running dead straight.
    Each pair is one scale of wobble, as in :func:`jitter_tree`.
max_radius : float, keyword-only, default 10000
    Longest connection allowed [um]; also sets how far ahead of the
    growing tip target points are considered.
n_target_points : int, keyword-only, default 100000
    How many points to scatter through the volume. MATLAB's default is
    1e6 drawn in the *bounding box*, most of which it discards; these
    all land inside.
stop : {'steps', 'length', 'terminals'}, keyword-only, default 'steps'
    Whether ``thr`` counts growth steps, total cable [um] (MATLAB's
    ``-L``), or termination points (``-T``).
history : bool, keyword-only, default False
    Keep the tree after every step. MATLAB always does, which for a
    long growth means hundreds of copies of a growing morphology.
rng : numpy.random.Generator, optional

Returns
-------
Growth
    ``(tree, length, terminals, attached_to, target, targets, history)``.

Notes
-----
New cable is laid down at roughly **one node per micron**, so a grown
tree is finely sampled by construction and does not need resampling
before analysis.

``sp`` and ``k`` are passed through MATLAB's transforms unchanged -- a
logit remap and ``tanh (4 k)`` respectively -- so numbers carry over
from MATLAB scripts. One consequence worth knowing: ``sp = 0.5`` is not
the midpoint of the effect, it maps to 0.475, and the space-filling
term switches off entirely only below ``sp ~ 0.0018``.

**Sampling is exact rather than by rejection**, as in
:func:`pynetrees.theta_mc_tree`. MATLAB's ``mc3d`` fills the bounding cube
with ``rand`` and keeps what lands inside the boundary, which for a flat
or elongated domain throws away most of every draw.

The nearest-node distances that drive selection are computed with a
k-d tree rather than a full pairwise distance matrix per step. MATLAB
recomputes ``pdist2`` between every tree node and every point in the
vicinity on every iteration, which is what makes its version slow on a
large domain.

### `isBCT_tree(bct_or_tree) -> 'bool'`

Check whether a B/C/T-type children-count sequence (2=branch,
1=continuation, 0=terminal -- or a Tree, whose column sums are used)
describes a single valid rooted tree.

### `jitter_tree(tree: 'Tree', stde: 'float' = 1.0, lam: 'int' = 10, ipart=None, rng=None) -> 'Tree'`

Add spatially-correlated noise to node coordinates: each node's
displacement is a Gaussian-weighted (kernel centered at 1 hop, width
``lam / 5``) blend of independent per-node noise, over nodes within
``lam`` topological hops.

Reimplemented via per-node BFS over the undirected tree graph instead
of MATLAB's precomputed dense adjacency-matrix powers (``A^k`` for
``k`` up to ``lam``) -- same result, and asymptotically cheaper for
large trees with modest ``lam`` (BFS touches O(lam) nodes per source
instead of a full matrix multiply). One behavioral difference: a
node's distance to *itself* is computed as a true BFS distance (0),
not MATLAB's value of 2 (an artifact of detecting self-reachability
via "walk of length k" parity on a matrix power rather than shortest
path) -- a deliberate correctness fix, see PORT_STATUS.md.

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `quaddiameter_tree(tree: 'Tree', scale: 'float' = 0.5, offset: 'float' = 0.5) -> 'Tree'`

Apply a quadratic diameter taper (Cuntz, Borst & Segev 2007) along
every root-to-terminal path, using the bundled best-fit parameters for
optimal current transfer; nodes shared by multiple paths get the mean
of each path's diameter at that point.

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `quadfit_tree(tree: 'Tree')`

Fit a quadratic diameter taper (:func:`quaddiameter_tree`) to
``tree``'s existing diameters. Returns ``(scale, offset, fitted_tree)``.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `random_tree(n: 'int' = 100, radius: 'float' = 100.0, bf: 'float' = 0.5, anisotropy: 'float' = 1.0, *, dim: 'int' = 3, shape: 'str' = 'sphere', avoid_multifurcations: 'bool' = False, rng=None) -> 'Tree'`

Grow a tree over randomly scattered points -- a toy cell.

For sanity-checking an analysis, for a figure that needs a plausible
arbor and nothing more, and as the null model a real morphology is
compared against. Not a growth model: the points carry no biology, so
what comes out shows what :func:`MST_tree`'s wiring rule alone produces.

Parameters
----------
n : int, default 100
    Number of nodes.
radius : float, default 100
    Half-width of the region the points are drawn from [um].
bf : float, default 0.5
    Balancing factor, passed to :func:`MST_tree`.
anisotropy : float, default 1
    Stretch the cloud along Y and squeeze it along X by this factor,
    keeping the area roughly fixed. ``1`` is isotropic.
dim : {2, 3}, keyword-only, default 3
    ``2`` flattens the cloud onto the XY plane.
shape : {'sphere', 'box'}, keyword-only, default 'sphere'
    Draw from a ball of ``radius`` (MATLAB's ``'-sphere'``) or from the
    cube enclosing it.
avoid_multifurcations : bool, keyword-only, default False
    Passed to :func:`MST_tree` (MATLAB's ``'-b'``).
rng : numpy.random.Generator, optional
    For a reproducible cloud.

Returns
-------
Tree
    Rooted at the origin, which is always one of the points.

Notes
-----
MATLAB draws ``10 * n`` points, keeps those inside the sphere and takes
the first ``n``. That is ~5.2n survivors for a ball inscribed in its
cube, so it works, but it fails outright if the draw comes up short.
This rejects and redraws until it has ``n``, which cannot.

MATLAB's threshold and maximum path length are both 10000, i.e. no
limit; the same is done here by passing them through to
:func:`MST_tree` unchanged.

### `smooth_tree(tree: 'Tree', pwchild: 'float' = 0.5, p: 'float' = 0.9, n: 'int' = 5) -> 'Tree'`

Smooth a tree along its longest paths (see :func:`smoothbranch`).
First merges dissected sections along "heavy" branches (where one
child subtree carries more than ``pwchild`` of the descendant weight)
into longer paths, so smoothing happens along natural long branches
rather than independently on every short inter-branch-point segment.

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `soma_tree(tree: 'Tree', maxD: 'float' = 30.0, length: 'float | None' = None, tag_region: 'bool' = False, overlap_correction: 'bool' = False) -> 'Tree'`

Reshape diameter near the root into a cosine soma profile of
(approximate) target diameter ``maxD`` and length ``length``
(default ``1.5 * maxD``). If ``tag_region``, affected nodes are
(re)labeled with a ``"soma"`` region.

Parameters
----------
tree : Tree
maxD : float, default 30.0
    Peak soma diameter [um], reached at the root.
length : float, optional
    Axial extent of the soma profile [um]; defaults to ``1.5 * maxD``.
    The cosine falls to zero at ``length / 2``, which is where the
    reshaping stops.
tag_region : bool, default False
    Label the affected nodes with a ``"soma"`` region.
overlap_correction : bool, default False
    MATLAB's ``'-b'``. Divide diameters by ``sqrt(2)`` for each branch
    point already passed, so that two cylinders meeting at a branch do
    not double-count the membrane they share. Neither NEURON nor this
    toolbox models overlapping surfaces, so without it the soma's
    surface area -- and hence its input conductance -- comes out too
    large wherever the soma spans a bifurcation.

    A branch straight off the root whose daughters diverge by more than
    90 degrees is treated as soma-to-axon plus soma-to-dendrite rather
    than a true bifurcation, and does not count.

Returns
-------
Tree

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

---

## Generative pipeline

Population-statistics-driven synthesis: cloning, DSCAM-style self-avoidance, spines.

### `PP_generator_tree(n=100, R: 'float' = 1.2, a: 'float' = 0.1, *, alpha: 'float' = 0.5, n_mc: 'int' = 20, level: 'float' = 0.05, epsilon: 'float' = 0.0, dim: 'int' = 2, box: 'float' = 100.0, tol: 'float' = 0.01, max_iter: 'int' = 200, rng=None, full_output: 'bool' = False)`

Scatter points with a prescribed degree of spatial order.

Produces a cloud whose Clark-Evans ratio (:func:`~pynetrees.r_mc_tree`)
matches ``R``: below 1 the points are clustered, 1 is Poisson, above 1
they are spaced more regularly than chance. Useful as a synaptic or
contact-point target for :func:`~pynetrees.MST_tree` when the arrangement
of the targets is itself the thing under study.

It works by repeatedly nudging every point along the line to its nearest
neighbour -- toward it to cluster, away to disperse -- by a step that
shrinks as the measured R approaches the target.

Parameters
----------
n : int or (n, dim) array_like, default 100
    How many points to place, or a starting cloud to rearrange.
R : float, default 1.2
    Target Clark-Evans ratio.
a : float, default 0.1
    Step size. The sign is chosen automatically; only the magnitude
    matters. Must be nonzero.
alpha, n_mc, level : see :func:`~pynetrees.r_mc_tree`
    ``n_mc`` defaults to 20 rather than that function's 100, because R
    is remeasured on every iteration.
epsilon : float, default 0.0
    Minimum separation between points [um] -- an exclusion zone
    standing in for the physical size of whatever the points represent.
    Moves that would violate it are refused.
dim : {2, 3}, default 2
box : float, default 100.0
    Points live in ``[-box, box]`` per axis and are clamped to it.
tol : float, default 0.01
    Stop once ``|measured - R|`` falls below this.
max_iter : int, default 200
    Give up after this many iterations and warn. MATLAB's loop has no
    bound at all and will spin forever on an unreachable target -- and
    many are unreachable, since R is capped by how tightly the exclusion
    zone and the box let points pack.
rng : numpy Generator or int, optional
full_output : bool, default False
    Also return the iteration count and the R measured at each step.

Returns
-------
np.ndarray or tuple
    ``(n, dim)`` points, or ``(points, n_iterations, R_history)``.

### `RegionSpan`

What one region looks like across a group of cells.

Every per-tree array has one row per input tree, with ``NaN`` where the
tree has no nodes in this region -- rather than silently shortening, so
row ``i`` always refers to tree ``i``.

### `Spanning`

Output of :func:`gscale_tree`: the group's measured envelope.

### `SpineResult`

Output of :func:`spines_tree` with ``full_output=True``.

| Field | Description |
|---|---|
| `tree` | Alias for field number 0 |
| `heads` | Alias for field number 1 |
| `necks` | Alias for field number 2 |

### `clone_tree(trees: 'Tree | list[Tree]', n: 'int' = 1, bf: 'float' = 0.4, *, dim: 'int' = 3, rng=None) -> 'list[Tree]'`

Grow synthetic trees resembling a measured group.

For each region in turn: pool the group's rescaled branch and
termination points (:func:`gscale_tree`), pick a size for this clone
from the group's spread, scatter fresh target points at that density
(:func:`rpoints_tree`), and wire them with
:func:`~pynetrees.MST_tree`. Then restore the group's taper, its wriggle,
and its soma.

Parameters
----------
trees : Tree or list[Tree]
    The group to imitate. One tree works, but the variability that
    makes clones differ from each other comes from the spread *across*
    the group, so a single input yields near-identical clones.
n : int, default 1
    How many clones. Each is an independent MST growth and is not fast.
bf : float, default 0.4
    Balancing factor handed to :func:`~pynetrees.MST_tree`: 0 minimises
    total wire, 1 minimises path length to the root.
dim : {2, 3}, default 3
    Grow flat clones (MATLAB's ``'-dim2'``).
rng : numpy Generator or int, optional
    Seed. **Required for reproducibility** -- every size, count and
    taper parameter is drawn from a normal distribution.

Returns
-------
list[Tree]

Notes
-----
Regions are handled by *name*, and the names are load-bearing:
``"soma"`` becomes a small MST blob at the origin that everything else
attaches to, ``"primary"`` is grown first and in two passes (far half
then near half), ``"spines"`` and ``"axon"`` are skipped entirely, and
anything else is grown in one pass onto whatever exists so far. That is
MATLAB's scheme; a group whose regions are named otherwise will still
clone, just without the special handling.

**The two-stage point count is not a heuristic that can be dropped.**
`MST_tree` connects some target points as continuation points rather
than as branch or termination points, so asking for ``N`` targets
yields fewer than ``N`` topological points. Both MATLAB and this grow a
throwaway tree with ``N`` points, count how many survived as branch or
termination points, and regrow with ``N * (N / survivors)`` -- capped at
``3.5 N``. It roughly doubles the growth cost and is why cloning is slow.

MATLAB additionally runs an outlier pass that repeatedly bins the pooled
cloud and deletes points sitting alone in a bin, halting once half the
cloud is gone. It is not ported: it deletes from the *pooled* cloud, so
what counts as an outlier depends on how many cells happen to be in the
group, and the "stop at half" guard means a sparse group can lose half
its points to it. Bin the cloud yourself before calling if you want
that.

### `dscam_tree(tree: 'Tree', iterations: 'int | None' = None, *, move: 'float' = 0.1, cluster: 'float' = 2.0, rng=None) -> 'Tree'`

Pull branches toward each other, as a DSCAM knockout does.

DSCAM lets a neurite recognise its own siblings and avoid them; without
it, branches that would normally repel each other clump together. This
reproduces the effect crudely, as Bird, Deters & Cuntz (2021) do: pick a
node at random, find the nearest node that is *not* one of its ancestors
or descendants, and slide that node -- with its whole subtree -- ten
percent of the way toward it. Repeat.

Parameters
----------
tree : Tree
iterations : int, optional
    Default ``5 * n_nodes``, as in MATLAB.
move : float, default 0.1
    Fraction of the gap closed per step.
cluster : float, default 2.0
    Ignore candidate partners closer than this [um]. Without it the
    nearest non-relative is usually a node a micron away on a branch
    that is already touching, and nothing moves.
rng : numpy Generator or int, optional

Returns
-------
Tree
    Same topology and diameters; only coordinates change.

Notes
-----
**Resample carefully afterwards** -- MATLAB's docstring says the same.
The operation moves nodes without adding any, so segments that were
evenly spaced no longer are.

One divergence: MATLAB picks the partner with
``find (distance == min (distance (iVector)))``, which searches the
*unmasked* distance vector for that minimum value -- so if an excluded
node (an ancestor, or one inside the subtree) happens to sit at exactly
the same distance, and comes first, it is chosen instead. Here the
partner is taken from the masked set directly, which is what the line
was evidently meant to do.

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `gscale_tree(trees: 'Tree | list[Tree]') -> 'Spanning'`

Measure a group of cells region by region, and rescale them to a
common size.

Parameters
----------
trees : Tree or list[Tree]

Returns
-------
Spanning
    Per-region extents, centres of mass, point counts, taper fits and
    rescaled point clouds, plus the group's wriggle statistics and the
    rescaled trees themselves.

Notes
-----
Each tree is translated to put its root at the origin first, so extents
and centres are measured relative to the soma rather than to whatever
coordinate frame the reconstruction happened to use.

**The rescaled point clouds and the rescaled trees do not agree**, and
that is MATLAB's behaviour, preserved deliberately. A region's *points*
are scaled about that region's own centre of mass, so the centre stays
put; the *trees* are scaled about the origin, i.e. the root. The point
cloud is what `clone_tree` samples, and holding each region's centre
fixed is what keeps the pooled cloud from smearing; the scaled trees are
for display and comparison. Do not expect
``spanning['dendrite'].points[i]`` to be a subset of
``spanning.scaled_trees[i]``.

MATLAB returns a ``spanning`` struct of fifteen parallel cell arrays
indexed by region and then by tree. This returns a list of
:class:`RegionSpan` objects, looked up by name
(``spanning['dendrite']``), because the parallel-array layout makes
every access a two-level index into containers that must be kept in
step by hand.

### `in_hull(points, polygons) -> 'np.ndarray'`

Which points lie inside a boundary made of several rings.

The largest polygon is taken as the outer boundary and every other one
as a hole, so a cell with a gap in its arbor is handled correctly.
Ports `construct/in_c.m`, but takes
:func:`~pynetrees.hull_tree`'s polygon list rather than MATLAB's packed
``contourc`` matrix.

Parameters
----------
points : (n, 2) array_like
polygons : list of (m, 2) array_like

Returns
-------
np.ndarray
    Boolean mask over ``points``.

### `rpoints_tree(density=None, n: 'int' = 1000, *, x=None, y=None, z=None, boundary=None, thr: 'float' = 0.0, rng=None) -> 'np.ndarray'`

Draw ``n`` random points from a density grid.

Each point picks a voxel with probability proportional to its count,
then lands uniformly inside that voxel -- so the result reproduces the
density at grid resolution while staying continuous within it.

Parameters
----------
density : DensityGrid or ndarray, optional
    Usually a :func:`~pynetrees.gdens_tree` result. A bare array needs
    ``x``/``y``/``z`` to say where its voxels are. Omit it entirely to
    scatter points uniformly through the box given by ``x``/``y``
    (default ``[-500, 500]`` on both, as in MATLAB).
n : int, default 1000
x, y, z : array_like, optional
    Voxel-centre coordinates per axis. Ignored when ``density`` is a
    :class:`~pynetrees.density.DensityGrid`, which carries its own.
boundary : list of (m, 2) arrays, optional
    Keep only points inside this 2D boundary -- a
    :func:`~pynetrees.hull_tree` polygon list. MATLAB takes its packed
    ``contourc`` matrix here instead; see this module's docstring.
thr : float, default 0.0
    With ``boundary``, also drop points within ``thr`` [um] of it.
rng : numpy Generator or int, optional

Returns
-------
np.ndarray
    ``(n, 3)``, or fewer rows when ``boundary`` rejects some. **The
    count is not guaranteed** when filtering -- MATLAB has the same
    behaviour, and `clone_tree` compensates by asking for four times
    what it needs.

Notes
-----
MATLAB picks each point with a Python-level loop over the cumulative
density, calling ``ind2sub`` once per point and showing a waitbar every
5000 -- which is what makes the waitbar worth having. The same thing is
one ``searchsorted`` over the whole batch here, so `clone_tree`'s
repeated calls stop dominating its runtime.

### `spines_tree(tree: 'Tree', spines=100, neck_diameter: 'float' = 0.5, head_diameter: 'float' = 1.0, neck_length: 'float' = 1.0, neck_length_std: 'float' = 1.0, nodes=None, *, separate_regions: 'bool' = False, rng=None, full_output: 'bool' = False)`

Attach spines to a tree.

Each spine is two nodes: a **neck** of diameter ``neck_diameter``
standing off the dendrite by a length drawn from
``N(neck_length, neck_length_std)``, and a **head** one
``head_diameter`` further out -- so the head is a cylinder as long as it
is wide, which is what makes its surface area come out roughly right.

The direction is perpendicular to the local dendrite, at a uniformly
random angle around it, so spines fan out around the cable rather than
all pointing the same way.

Parameters
----------
tree : Tree
spines : int or array_like, default 100
    How many spines to add at randomly chosen nodes; **or** an integer
    array of node indices to spine; **or** an ``(n, 3)`` array of
    explicit neck coordinates.
neck_diameter : float, default 0.5
head_diameter : float, default 1.0
    Also the head's length -- see above.
neck_length, neck_length_std : float, default 1.0
    Mean and spread of the neck length [um].
nodes : array_like, optional
    Restrict random placement to these nodes (MATLAB's ``ipart``).
separate_regions : bool, default False
    Label necks and heads as two regions (``spine_neck``,
    ``spine_head``) instead of one region called ``spines``. MATLAB's
    ``'-sr'``.
rng : numpy Generator or int, optional
full_output : bool, default False
    Return :class:`SpineResult` -- the tree plus the head and neck node
    indices -- instead of just the tree.

Returns
-------
Tree or SpineResult

Notes
-----
Three things MATLAB's version gets wrong here.

**Its documented coordinate input cannot be reached.** ``XYZ`` is
dispatched as ``numel (XYZ) == 1`` -> a count, ``elseif all (XYZ < N)``
-> node indices, and nothing else. An ``(n, 3)`` matrix of coordinates
therefore either falls into the *indices* branch (when the cell happens
to sit near the origin, so every coordinate is below the node count) or
matches neither, leaving ``indy`` undefined and raising. This port
dispatches on **shape** -- an ``(n, 3)`` array is coordinates, a 1D
integer array is indices -- so all three documented forms work.

**Its ``'-sr'`` branch reads an undefined variable.** ``flag`` is only
assigned when no ``spine_neck`` region exists; if one does but
``spine_head`` does not, ``iR (2) = ... + 1 + flag`` raises.

**It returns only the last spine's indices.** ``indhead`` and
``indneck`` are overwritten each pass of the loop, so despite being
documented as "node indices of spine heads"/"necks" they hold two
numbers, not two arrays. Here they are the full arrays.

A fourth, geometric: MATLAB draws the neck length from a normal
distribution and then places the head at ``neck + dhead * dXYZ``
regardless of sign. When the draw is negative -- 16% of the time at its
own defaults of mean 1 and standard deviation 1 -- the neck goes one way
and the head the other, so the head ends up between the neck and the
dendrite or inside it. Here the *direction* is flipped rather than the
length, which is distributionally identical (the direction is uniformly
random to begin with) and keeps the head beyond the neck.

    An **empty tree** is returned unchanged -- see :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

---

## Density, hulls and space-filling

Voxel grids, alpha-shape boundaries, spanned area, space-filling radius (grid and Monte Carlo).

### `Boundary`

A concave boundary (alpha shape) wrapped around a point cloud.

Attributes
----------
vertices : np.ndarray
    ``(v, dim)`` coordinates of the points lying on the surface.
faces : np.ndarray
    ``(f, dim)`` triangles (3D) or edges (2D) making up the surface,
    indexing into ``vertices``.
volume : float
    Enclosed volume [um^3] in 3D, enclosed area [um^2] in 2D. This is
    the sum over the filled simplices, so a boundary with holes or
    several disconnected lobes is measured correctly rather than as
    its outer envelope.
points : np.ndarray
    ``(n, dim)`` the full point set that was wrapped.
simplices : np.ndarray
    ``(s, dim + 1)`` the *filled* simplices -- tetrahedra in 3D,
    triangles in 2D -- indexing into ``points``. This is the interior,
    as opposed to ``faces`` which is only the shell. Needed to sample
    uniformly inside the boundary (see :func:`r_mc_tree`).
polygon : np.ndarray or None
    2D only: ``(p, 2)`` surface vertices walked into boundary order, so
    consecutive rows are joined by an edge. ``None`` in 3D, where no
    such ordering exists. MATLAB's ``bound.xv``/``bound.yv``.

| Field | Description |
|---|---|
| `vertices` | Alias for field number 0 |
| `faces` | Alias for field number 1 |
| `volume` | Alias for field number 2 |
| `points` | Alias for field number 3 |
| `simplices` | Alias for field number 4 |
| `polygon` | Alias for field number 5 |

### `Span`

Result of :func:`span_tree`: the area an arbor spans, as a mask.

| Field | Description |
|---|---|
| `mask` | Alias for field number 0 |
| `area` | Alias for field number 1 |
| `origin` | Alias for field number 2 |

### `ThetaMC`

Result of :func:`theta_mc_tree`.

| Field | Description |
|---|---|
| `theta` | Alias for field number 0 |
| `distances` | Alias for field number 1 |

### `boundary_tree(tree: 'Tree', shrink: 'float | None' = None, dim: 'int | None' = None, nodes=None, *, c: 'float | None' = None) -> 'Boundary'`

Concave boundary (alpha shape) around a tree's points.

Parameters
----------
tree : Tree
shrink : float in [0, 1], default 0.5
    How tightly the boundary wraps. ``0`` gives the convex hull; ``1``
    gives the tightest shape that still envelops every point. Matches
    the sense of MATLAB's ``boundary`` shrink factor.
dim : {2, 3}, optional
    Default 3.
nodes : array_like, optional
    Subset of nodes to wrap. Defaults to all.
c : float, optional
    Convexity, as returned by :func:`convexity_tree`. MATLAB's
    `boundary_tree` is parameterised this way and sets its shrink
    factor to ``1 - c``, so that a convex cell is wrapped loosely and a
    concave one tightly. Supplying ``c`` does exactly that. Mutually
    exclusive with ``shrink``.

Returns
-------
Boundary

Notes
-----
**A fixed shrink is not a fixed shape across node densities.** Adding
nodes along existing cable does not change the point cloud's envelope,
but it does subdivide the empty pockets between branches into simplices
small enough to survive the threshold, so the boundary loosens.
Resampling `hsn_tree` to 1 um leaves its convex hull unchanged while
this grows from 30% to 49% of it. Comparisons across cells therefore
need a common sampling rate; :func:`pynetrees.theta_mc_tree`,
:func:`pynetrees.r_mc_tree` and :func:`pynetrees.scaleV_tree` all inherit
this.

**Not verifiable against MATLAB here.** MATLAB's `boundary_tree` calls
its built-in ``boundary()``, which Octave does not implement, so no
differential check was possible on this machine. The algorithm below is
the standard alpha-shape construction that ``boundary()`` documents
itself as performing -- Delaunay triangulation with simplices discarded
above a circumradius cutoff -- but the exact mapping from shrink factor
to cutoff is undocumented on MATLAB's side, so **boundary vertices will
not match theirs exactly**.

Separately, MATLAB's `boundary_tree` cannot run at all unless ``c`` is
passed: its default branch does ``pars = convexity_tree (...)``, which
replaces the whole parsed-options struct with a bare scalar, so the very
next line's ``pars.c`` errors out. Its own documented example,
``boundary_tree (sample_tree, '-dim3')``, is one of the calls that
fails. See MATLAB_TOOLBOX_BUGS.md.

For "how much space does this cell occupy", prefer :func:`hull_tree`:
it wraps the *arbor* at a stated distance rather than wrapping its
*points* by a unitless tightness knob, so the result means something
physical.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `convexity_tree(tree: 'Tree', thr: 'float' = 25.0, nodes=None, samples: 'int' = 24, max_pairs: 'int' = 20000, dim: 'int' = 3, rng=None) -> 'float'`

How convex a tree's occupied volume is, in ``[0, 1]``.

Takes pairs of points on the tree and asks what fraction can "see" each
other -- i.e. the straight line between them never leaves the volume the
cell occupies. A convex shape scores 1. A cell that wraps around
something, or splits into lobes with a gap between them, scores lower,
because lines between its far parts pass through empty space.

Parameters
----------
tree : Tree
thr : float, default 25.0
    Distance [um] defining the occupied volume, as in :func:`hull_tree`.
nodes : array_like, optional
    Points to test between. Defaults to the termination points, which
    are the extremities and so the most informative.
samples : int, default 24
    How many points to check along each connecting line. More is
    stricter about narrow gaps.
dim : {2, 3}, default 3
    Measure in the plane rather than in space. MATLAB has separate 2D
    and 3D branches here; this is the same computation either way.
max_pairs : int, default 20000
    Cap on the number of pairs tested; above it, a random subset is
    used. The pair count grows quadratically, so a 500-terminal cell
    would otherwise be 125000 lines.
rng : numpy Generator or int, optional
    Seed for the subsampling, for reproducibility.

Returns
-------
float
    Fraction of visible pairs, in ``[0, 1]``.

Notes
-----
**This deliberately does not reproduce MATLAB's version**, whose 3D
branch contradicts both its own documentation and its own 2D branch in
two independent ways -- visible directly in the source, so this is a
confirmed defect and not merely a suspicion, even though Octave's
missing ``boundary`` meant it could not be executed here:

1. It wraps the terminals with ``boundary (X, Y, Z, 0)``, and shrink 0
   is documented by MathWorks as the **convex hull** -- not the
   "tightest boundary" the docstring promises, and one against which
   every segment between interior points lies inside by definition.
   The 2D branch correctly asks for shrink 1.
2. It then returns ``c = 1 - nnz (Inds) / (nS1 * nS2)`` where ``Inds``
   flags the pairs that *did not* cross the surface, i.e. the ones
   inside. The 2D branch returns ``nnz (Inds) / (nS1 * nS2)`` from the
   same flag. One of the two has the sign inverted, and it is the 3D
   one that disagrees with the documented meaning.

Between them, the 3D result is close to the fraction of terminal pairs
with an endpoint on the convex hull -- a measure of how many terminals
happen to be extremal, not of convexity. See MATLAB_TOOLBOX_BUGS.md.

This version tests against the **space-filling hull** instead, which is
the standard definition of convexity for a shape and the only version
that can distinguish a compact arbor from a lobed one.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `gdens_tree(tree, sr: 'float' = 5.0, nodes=None) -> 'DensityGrid'`

Bin a tree's nodes into a regular ``sr``-sized voxel grid.

Parameters
----------
tree : Tree or (n, 3) array_like
    A tree, or a bare point cloud. MATLAB accepts both here too, and
    the point-cloud form is what the generative pipeline
    (:mod:`pynetrees.generate`) bins.
sr : float, default 5.0
    Voxel edge length [um].
nodes : array_like, optional
    Restrict to a subset of nodes (MATLAB's ``ipart``).

Returns
-------
DensityGrid
    ``counts`` indexed ``[ix, iy, iz]``, plus the voxel-centre
    coordinates along each axis.

Notes
-----
Counts **nodes**, not length, exactly as MATLAB does. That makes the
result depend on how the morphology was sampled, so compare densities
only between trees resampled the same way -- `resample_tree` first if
they were not. (Length-weighted density would be the more robust
measure, but it is not what this function is, and silently changing the
quantity would make results incomparable with published MATLAB ones.)

Indexing is ``[x, y, z]``. MATLAB's is ``[y, x, z]``, following its
image convention; that transposition is deliberate here, since every
other array in this port is ``[x, y, z]`` and mixing the two silently
is exactly how axis bugs happen.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `hull_tree(tree: 'Tree', thr: 'float' = 25.0, bx=50, by=50, bz=50, dim: 'int | None' = None, *, return_distances: 'bool' = False) -> 'HullResult'`

The surface lying ``thr`` um from the tree -- a space-filling hull.

Samples distance-to-the-nearest-segment on a regular grid and extracts
the ``thr`` isosurface. Unlike a convex hull this follows concavities,
so it actually describes the volume a cell occupies rather than the
volume it spans.

Parameters
----------
tree : Tree
thr : float, default 25.0
    Distance [um] defining the surface. Smaller values track the arbor
    more tightly and need a finer grid to resolve.
bx, by, bz : int or array_like, default 50
    Grid resolution per axis: an integer is a number of *intervals*
    across the padded extent (MATLAB's convention); an array is used as
    explicit coordinates.
dim : {2, 3}, optional
    Default 3.
return_distances : bool, default False
    Also return the sampled distance field (MATLAB's ``'-F'``).

Returns
-------
HullResult

Raises
------
ImportError
    In 3D, if `scikit-image` is not installed (needed for marching
    cubes). Install the ``[plot]`` extra.

Notes
-----
Cost is ``grid points x segments``, and the grid grows cubically -- the
default 50 intervals per axis is 132651 points, which against a
2252-node cell is 3e8 distance evaluations. Halving `thr` usually means
doubling the resolution to resolve it, i.e. 8x the work; raise the
resolution deliberately rather than by default.

If the isosurface comes out empty, `thr` is likely smaller than the
grid spacing, so no cell straddles the level -- the warning says so.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `lego_tree(tree: 'Tree', sr: 'float' = 5.0, nodes=None, ax=None, cmap: 'str' = 'viridis', threshold: 'float' = 0.0)`

Draw a tree's density grid as opaque voxels -- MATLAB's "lego" plot.

Parameters
----------
tree : Tree
sr : float, default 5.0
    Voxel edge length [um].
nodes : array_like, optional
ax : matplotlib 3D Axes, optional
    Created if omitted.
cmap : str
threshold : float, default 0.0
    Only draw voxels holding more than this many nodes.

Returns
-------
matplotlib.axes.Axes

Notes
-----
A blunt instrument by design: it shows occupancy, not shape. For "where
does this cell reach", :func:`hull_tree` is the better tool -- but a
lego plot makes *density* differences legible in a way a smooth hull
cannot, which is why the original has both.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `share_boundary_tree(tree1: 'Tree', tree2: 'Tree', thr: 'float' = 25.0, sr: 'float' = 5.0) -> 'float'`

Volume [um^3] shared by two trees' space-filling hulls.

How much of the space one cell occupies is also occupied by the other --
the quantity behind questions about territorial overlap and tiling.

Parameters
----------
tree1, tree2 : Tree
thr : float, default 25.0
    Hull distance [um].
sr : float, default 5.0
    Voxel edge [um] of the shared grid the two hulls are rasterised
    onto. The result is a voxel count times ``sr ** 3``, so accuracy is
    set by ``sr`` and cost by ``sr ** -3``.

Returns
-------
float
    Shared volume [um^3]. Zero if the hulls do not meet.

Notes
-----
Voxelised rather than computed as an exact mesh intersection, which is
what MATLAB does too. Both hulls are sampled on **one common grid**
spanning the union of their extents -- rasterising each on its own grid
and comparing would compare different things.

### `span_tree(tree: 'Tree', radius: 'int | None' = None, *, method: 'str' = 'close', shrink: 'float' = 0.5) -> 'Span'`

The area a tree's arbor spans, as a filled mask [um^2].

Rasterises the nodes onto a one-micron grid and closes the gaps between
them, giving "the territory this cell covers" -- the quantity a
space-filling argument is about, and the one :func:`theta_tree` and
:func:`pynetrees.scaleS_tree` are built on.

Parameters
----------
tree : Tree
radius : int, optional
    How wide a pocket between stretches of cable still counts as
    spanned [um]; the ground between two branches is filled once
    ``radius`` reaches half the gap. Defaults to MATLAB's rule, half
    the geometric mean of the cell's X and Y extent. Note that closing
    needs cable on *both* sides of a gap: no radius joins two isolated
    nodes, because erosion undoes dilation exactly for a lone blob.
method : {'close', 'hull'}, keyword-only, default 'close'
    ``'close'`` fills gaps up to ``radius`` by morphological closing
    (MATLAB's default). ``'hull'`` fills the alpha shape of the nodes
    instead (MATLAB's ``'-b'``), which ignores ``radius`` and follows
    the outline rather than the cable.
shrink : float, keyword-only, default 0.5
    Passed to :func:`boundary_tree` when ``method='hull'``.

Returns
-------
Span
    ``(mask, area, origin)``. This is 2D throughout -- MATLAB's comment
    says "actually only uses X and Y", and so does this.

Notes
-----
**The disk differs from MATLAB's.** MATLAB closes with
``strel ('disk', radius, 4)``, a four-line periodic approximation of a
disk that is octagonal rather than round; this uses the exact Euclidean
disk. Absolute areas therefore differ a little. The size of that
difference is not measured here: Octave implements only the exact disk
(``N = 0``), so there was nothing to compare against, and guessing at
MATLAB's octagon would be inventing a number rather than measuring one.
Everything else in this function was checked against MATLAB's own code
running on the exact disk, and agrees pixel for pixel.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `theta_mc_tree(tree: 'Tree', n: 'int' = 100000, alpha: 'float' = 0.5, *, dim: 'int | None' = None, rng=None) -> 'ThetaMC'`

Space-filling radius by Monte Carlo, in 3D.

The same question as :func:`theta_tree` -- how far into its own
territory does the arbor reach -- but measured by scattering points
through the volume the cell encloses and asking how far each one is
from the nearest piece of cable. Works in 3D, where rasterising onto a
grid would be prohibitive, and needs no grid resolution chosen.

Parameters
----------
tree : Tree
n : int, default 100000
    Number of sample points, all of them **inside** the boundary. See
    the Notes: MATLAB's ``NN`` counts points drawn in the bounding box,
    most of which it then discards.
alpha : float, default 0.5
    How tightly the boundary wraps; see :func:`boundary_tree`.
dim : {2, 3}, optional
    Default 3.
rng : numpy.random.Generator, optional

Returns
-------
ThetaMC
    ``(theta, distances)``.

Notes
-----
**Sampling is exact rather than by rejection**, the same change made for
:func:`pynetrees.r_mc_tree` (Design Decision #61). MATLAB fills a cube
with ``rand`` and keeps whatever lands inside the alpha shape, so a flat
or elongated cell wastes most of every draw -- and the cube is sized by
the *largest* extent, so the flatter the cell the worse it gets.
Drawing from the boundary's own simplex decomposition, weighted by
volume, gives the identical uniform distribution with nothing thrown
away.

**theta is the exact quantile**, not a bin edge. MATLAB takes
``find (cyax < 0.9069, 1, 'last') / 100`` over 0.01-wide bins, which
reports the last bin *below* the crossing and adds one bin's width
through the same 1-based indexing slip as :func:`theta_tree` -- two
errors of 0.01 um in opposite directions. Immaterial at that width, but
a quantile needs no bins at all.

**Only compare cells sampled at the same rate.** This measures inside
the alpha shape, and a fixed ``alpha`` does not describe the same shape
at every node density: adding nodes along existing cable subdivides the
empty pockets between branches into simplices small enough to be kept,
so the boundary *loosens*. Resampling `hsn_tree` to 1 um leaves its
convex hull unchanged (6.21e6 um^3) while the ``alpha = 0.5`` boundary
goes from 30% to 49% of it, and theta with it from 24 um to 48 um.
Nothing about the morphology changed. Resample every cell in a
comparison to a common rate first -- or use :func:`theta_tree`, whose
one-micron grid does not have this sensitivity.

Distances are to the nearest **node**, as in MATLAB, not to the nearest
segment as elsewhere in this module. Measured both ways the answers
differ by under 0.5% on the bundled trees, far below the boundary
effect above, so the extra machinery would buy nothing here.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `theta_tree(tree: 'Tree', radius: 'int | None' = None, *, include_boundary: 'bool' = True) -> 'float'`

Space-filling radius: how far the arbor reaches into its own territory.

Grows a disc around every node until those discs cover 90.69% of the
area the cell spans, and returns that radius [um]. A small theta means
the arbor fills its territory finely; a large one means it spans the
same ground with gaps in it.

Parameters
----------
tree : Tree
radius : int, optional
    Passed to :func:`span_tree`.
include_boundary : bool, keyword-only, default True
    Count the edge of the spanned area as covered, so that pixels near
    the rim are not charged for being far from any node. MATLAB's
    default; its ``'-e'`` option is ``False``.

Returns
-------
float
    The radius [um] at which the discs first cover 90.69% of the span.

Notes
-----
0.9069 is the packing density of circles on a hexagonal lattice,
``pi / (2 * sqrt (3))`` -- the most of a plane that equal discs can
cover -- so theta is the disc radius at which the arbor would be doing
as well as a perfect packing.

**MATLAB returns an index where this returns a distance.** Its
``find (y > 0.9069 * S, 1, 'first')`` is a 1-based position in
``0 : ceil (max (hB))``, so its answer is one micron larger than the
radius it stands for, and it then compares that index against real
distances in the ``'-s'`` plot. Recorded in ``MATLAB_TOOLBOX_BUGS.md``.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `vhull_tree(tree: 'Tree', nodes=None, boundary=None, thr: 'float' = 25.0, dim: 'int | None' = None) -> 'VoronoiResult'`

Voronoi territory of every node, clipped to the tree's hull.

Each node gets the region of space closer to it than to any other node,
trimmed to the space-filling hull so the outermost nodes do not receive
unbounded territory. The per-node volumes are what density statistics
are built from.

Parameters
----------
tree : Tree
nodes : array_like, optional
    Subset of nodes to tessellate.
boundary : array_like, optional
    Explicit boundary points to clip against. Defaults to the vertices
    of :func:`hull_tree` at ``thr``.
thr : float, default 25.0
    Hull distance used when ``boundary`` is not given.
dim : {2, 3}, optional
    Default 3.

Returns
-------
VoronoiResult

Notes
-----
Unbounded Voronoi cells get ``NaN`` rather than a clipped guess.
MATLAB's version silently drops them, which quietly biases any mean
computed over the result -- the outermost nodes are exactly the ones
with the largest territories.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

---

## Topological description (persistent homology)

Branch-length decomposition, barcodes, persistence images.

### `barcode_tree(tree: 'Tree', v: 'np.ndarray | None' = None, *, mode: 'str' = 'length', by: 'str' = 'nodes') -> 'np.ndarray'`

Persistent-homology barcode: one bar per branch.

Parameters
----------
tree : Tree
v : array_like, optional
    Per-node values to accumulate along each path. Overrides ``mode``.
mode : {'length', 'euclidean', 'topological'}, keyword-only
    What to accumulate when ``v`` is not given: segment length in
    microns (MATLAB's ``'-l'``), Euclidean distance to the root
    (``'-E'``), or one per node, making the bars count nodes rather
    than measure distance (``'-t'``).
by : {'nodes', 'length'}, keyword-only, default 'nodes'
    Passed to :func:`pynetrees.BLO_tree`, which decides which paths
    become branches. The default reproduces MATLAB; read that
    function's Notes before changing it, because the two disagree
    substantially.

Returns
-------
np.ndarray
    ``(n_branches, 2)``, columns ``[birth, death]``: the distances from
    the root at which each branch starts and ends. Rows follow
    :func:`pynetrees.BLO_tree`'s order, so row 0 is the branch containing
    the root and its birth is 0.

Notes
-----
Bars nest rather than overlap: a branch is born on its parent branch,
so its birth always falls inside its parent's ``[birth, death]``. That
is what :func:`realisations_tree` counts.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape. ``v`` may be given
    once for the whole group or as one value per tree -- see
    :mod:`pynetrees._population`.

### `persistenceimage_tree(tree: 'Tree', v: 'np.ndarray | None' = None, *, mode: 'str' = 'length', sigma: 'float' = 17.5, size: 'int | None' = None, accumulate: 'bool' = True) -> 'np.ndarray'`

Render a barcode as a fixed-size 2D density -- the persistence image.

Every cell gives an image on the same axes, so two cells can be
compared, averaged or clustered pixel by pixel, which a barcode of
unequal length does not allow.

Parameters
----------
tree : Tree
v, mode
    As :func:`barcode_tree`.
sigma : float, keyword-only, default 17.5
    Width of the Gaussian each bar is smeared into, in microns.
    MATLAB's fixed kernel works out to exactly this. Note it is
    **absolute**, not a fraction of the cell: a small cell is smoothed
    proportionally more than a large one.
size : int, optional
    Side of the square image in pixels, one micron each. Defaults to
    ``round(1.25 * max(death))``, MATLAB's rule -- 25% of headroom past
    the furthest tip.
accumulate : bool, keyword-only, default True
    Add up bars that land on the same pixel. ``False`` reproduces
    MATLAB, which marks occupied pixels with a 1 and so counts
    coincident branches once. See the Notes.

Returns
-------
np.ndarray
    ``(size, size)``, indexed ``[birth, death]``. Only the **upper**
    triangle can be occupied, since a branch cannot die before it is
    born.

Notes
-----
**This differs from MATLAB in two places, both deliberate.**

MATLAB assigns ``M(...) = 1`` rather than accumulating, so branches
whose births and deaths round to the same micron contribute once
between them. Across the 55 cells in :func:`pynetrees.dLPTCs_trees` that
silently drops a median of **1.5% of bars, at worst 4.0%** -- small,
but it falls hardest on the densely branched cells, which are the ones
a density is supposed to distinguish. The published method sums a
kernel per bar, so this accumulates by default; pass
``accumulate=False`` to reproduce MATLAB's figures exactly.

MATLAB also offsets the two axes inconsistently -- ``round(birth) + 1``
against ``round(death)`` -- which shifts the image one pixel off the
diagonal. Here both axes share an origin. Against a 17.5 um kernel one
pixel is invisible, and it cancels between cells anyway, so this
changes no clustering; it only matters if you read coordinates off the
image.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape. ``v`` may be given
    once for the whole group or as one value per tree -- see
    :mod:`pynetrees._population`.

### `realisations_tree(tree: 'Tree | np.ndarray', v: 'np.ndarray | None' = None, *, mode: 'str' = 'length') -> 'int'`

How many distinct trees share this tree's barcode.

The barcode says which branches exist and where each begins and ends,
but not *which* branch each one hangs off: a bar born at distance 40
could have branched off any bar alive at 40. Multiplying those choices
together counts the trees the barcode cannot tell apart -- a measure of
how much shape information the description throws away.

Parameters
----------
tree : Tree or array_like
    A tree, or an ``(n_branches, 2)`` barcode from
    :func:`barcode_tree`, so a barcode computed once can be reused.
v, mode
    As :func:`barcode_tree`, ignored when a barcode is passed.

Returns
-------
int
    Exact, however large. For a real cell this number is astronomical
    -- a 1290-node cell runs to hundreds of digits -- which is the
    point being made. MATLAB computes it in double precision and
    returns ``Inf`` for anything past ~1e308; Python's integers do not
    overflow, so the value is usable (its logarithm, in practice).

Notes
-----
Returns 0 if any non-root bar is born outside every other bar, which
cannot happen for a barcode that came from a tree.

---

## Electrotonics

Passive cable analysis and integrate-and-fire simulation; needs `tree.Ri`/`tree.Gm` (and `tree.Cm` for time-stepping).

### `AdExLIF_tree(tree: 'Tree', time: 'np.ndarray | None' = None, I: 'np.ndarray | None' = None, ge: 'np.ndarray | None' = None, gi: 'np.ndarray | None' = None, Ee: 'float' = 60.0, Ei: 'float' = -20.0, iroot: 'int' = 0, EL: 'float' = 0.0, DeltaT: 'float' = 2.0, Vt: 'float' = 10.0, thr: 'float' = 80.0, vreset: 'float' = 2.0, Aspike: 'float' = 110.0, tauw: 'float' = 0.4, a: 'float' = 0.0, b: 'float' = 1e-06, verbose: 'bool' = False) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]'`

Adaptive exponential LIF simulation over the tree's full morphology.

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

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `LIF_tree(tree: 'Tree', time: 'np.ndarray | None' = None, ge: 'np.ndarray | None' = None, gi: 'np.ndarray | None' = None, Ee: 'float' = 60.0, Ei: 'float' = -20.0, I: 'np.ndarray | None' = None, iroot: 'int' = 0, thr: 'float' = 10.0, vreset: 'float' = 0.0, Aspike: 'float' = 75.0, partial_reset: 'bool' = False, verbose: 'bool' = False) -> 'tuple[np.ndarray, np.ndarray]'`

Leaky integrate-and-fire simulation over the tree's full morphology.

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

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `M_atten_tree(tree: 'Tree', thr: 'float' = 0.13995) -> 'int'`

Number of electrotonically distinct compartments in a tree.

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

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `M_tree(tree: 'Tree') -> 'sparse.csr_matrix'`

Conductance matrix of the tree's equivalent electric circuit [uS].

Combines axial (inter-compartment) and membrane conductances into one
sparse NxN matrix, the basis for :func:`sse_tree`, :func:`syn_tree` and
every other function in this module. Requires ``tree.Ri`` and
``tree.Gm``.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `cgin_tree(tree: 'Tree') -> 'float'`

Collapsed (point-neuron) input conductance of the whole tree [S].

Requires ``tree.Gm``, taken as a single scalar specific membrane
conductance representative of the whole cell.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `elen_tree(tree: 'Tree') -> 'np.ndarray'`

Electrotonic length of every segment (length / lambda), unitless.

An **empty tree** gives an empty result rather than an error -- see
:mod:`pynetrees._empty`.

Accepts a **list of trees** (or a list of lists of trees) as well as a
single one, returning results in the same shape -- see
:mod:`pynetrees._population`.

### `gi_tree(tree: 'Tree') -> 'np.ndarray'`

Axial conductance of every segment [S]. Requires ``tree.Ri``.

An **empty tree** gives an empty result rather than an error -- see
:mod:`pynetrees._empty`.

Accepts a **list of trees** (or a list of lists of trees) as well as a
single one, returning results in the same shape -- see
:mod:`pynetrees._population`.

### `gm_tree(tree: 'Tree') -> 'np.ndarray'`

Membrane conductance of every segment [S]. Requires ``tree.Gm``.

An **empty tree** gives an empty result rather than an error -- see
:mod:`pynetrees._empty`.

Accepts a **list of trees** (or a list of lists of trees) as well as a
single one, returning results in the same shape -- see
:mod:`pynetrees._population`.

### `lambda_tree(tree: 'Tree') -> 'np.ndarray'`

Length constant of every segment [cm]. Requires ``tree.Ri``/``tree.Gm``.

An **empty tree** gives an empty result rather than an error -- see
:mod:`pynetrees._empty`.

Accepts a **list of trees** (or a list of lists of trees) as well as a
single one, returning results in the same shape -- see
:mod:`pynetrees._population`.

### `loop_tree(tree: 'Tree', inodes1: 'int | np.ndarray', inodes2: 'int | np.ndarray', gelsyn: 'float | np.ndarray' = 1.0) -> 'sparse.csr_matrix'`

Conductance matrix with extra electrical-synapse loops added.

Adds a conductance ``gelsyn`` [uS] directly between each
``(inodes1[k], inodes2[k])`` pair of 0-based node indices, on top of
:func:`M_tree`'s ordinary tree connectivity -- the only way to represent
a non-tree (loopy) circuit in this data model.

### `sse_tree(tree: 'Tree', I: 'float | np.ndarray | None' = None) -> 'np.ndarray'`

Steady-state electrotonic signature: potential [mV] per node per input.

With ``I=None`` (default), returns the full NxN matrix whose column ``i``
is the potential distribution from injecting 1 nA at node ``i`` (the
diagonal is each node's local input resistance). A scalar ``I`` injects
1 nA at that 0-based node index (returning one column); an explicit
per-node array injects those exact currents.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `ssecat_tree(trees: 'list[Tree]', inodes1: 'int | np.ndarray', inodes2: 'int | np.ndarray', gelsyn: 'float | np.ndarray' = 1.0, I: 'float | np.ndarray | None' = None) -> 'np.ndarray'`

:func:`sse_tree` for several trees joined by electrical synapses.

``trees`` are combined into one block-diagonal conductance matrix first
(no coupling at all between them), then ``inodes1``/``inodes2`` (0-based
node indices *into the concatenated system*, i.e. offset by the
cumulative node counts of the preceding trees -- same convention as
:func:`loop_tree`, generalized across trees) add electrical-synapse
loops between them.

### `syn_tree(tree: 'Tree', ge: 'float | np.ndarray | None' = None, gi: 'float | np.ndarray | None' = None, Ee: 'float' = 60.0, Ei: 'float' = -20.0, I: 'float | np.ndarray | None' = None) -> 'np.ndarray'`

Steady-state potential [mV] per node under synaptic + current input.

``ge``/``gi`` are per-node synaptic conductances [uS] (scalar = inject a
canonical unit conductance at that 0-based node index, matching
:func:`sse_tree`'s ``I`` convention); ``Ee``/``Ei`` their reversal
potentials [mV]; ``I`` an additional current injection [nA].

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `syncat_tree(trees: 'list[Tree]', inodes1: 'int | np.ndarray', inodes2: 'int | np.ndarray', gelsyn: 'float | np.ndarray' = 1.0, ge: 'float | np.ndarray | None' = None, gi: 'float | np.ndarray | None' = None, Ee: 'float' = 60.0, Ei: 'float' = -20.0, I: 'float | np.ndarray | None' = None) -> 'np.ndarray'`

:func:`syn_tree` for several trees joined by electrical synapses.

Same tree-concatenation convention as :func:`ssecat_tree`; ``ge``/``gi``/
``Ee``/``Ei``/``I`` behave exactly as in :func:`syn_tree`, indexed into
the concatenated system.

---

## Statistics and comparison

Sholl analysis, von Mises fits, spatial-randomness tests, population summaries.

### `RMCResult`

Output of :func:`r_mc_tree`.

``R`` is the statistic; everything else is the working that produced
it, kept because the sampling distribution is what tells you whether a
given ``R`` means anything.

| Field | Description |
|---|---|
| `R` | Alias for field number 0 |
| `Rmin` | Alias for field number 1 |
| `Rmax` | Alias for field number 2 |
| `r0` | Alias for field number 3 |
| `rE` | Alias for field number 4 |
| `rEmin` | Alias for field number 5 |
| `rEmax` | Alias for field number 6 |
| `rEstd` | Alias for field number 7 |
| `n` | Alias for field number 8 |
| `rEs` | Alias for field number 9 |

### `ShollDissection`

Output of :func:`dissectSholl_tree`.

The point of the analysis is the comparison between ``observed`` and
the successively richer predictions ``domain``, ``angle`` and
``density``: whatever the simplest prediction already explains is not
evidence of anything more interesting.

| Field | Description |
|---|---|
| `c` | Alias for field number 0 |
| `volume` | Alias for field number 1 |
| `total_length` | Alias for field number 2 |
| `scale` | Alias for field number 3 |
| `radii` | Alias for field number 4 |
| `observed` | Alias for field number 5 |
| `domain` | Alias for field number 6 |
| `angle` | Alias for field number 7 |
| `density` | Alias for field number 8 |
| `rootangle` | Alias for field number 9 |
| `k` | Alias for field number 10 |
| `bf` | Alias for field number 11 |
| `est_scale` | Alias for field number 12 |
| `err_domain` | Alias for field number 13 |
| `err_angle` | Alias for field number 14 |
| `err_density` | Alias for field number 15 |

### `ShollResult`

Output of :func:`sholl_tree`.

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

### `bf_tree(data: 'Tree | list[Tree] | np.ndarray', dim: 'int' = 3, fit_constants: 'tuple[float, float, float] | None' = None) -> 'tuple[float, float]'`

Estimate a tree's MST balancing factor from its root-angle distribution.

Fits the centripetal bias ``k`` via :func:`vonMises_tree`, then maps it
to an estimated balancing factor ``bf`` (as used by :func:`MST_tree`)
through the closed-form relationship fit in Bird & Cuntz 2019. Returns
``(bf, k)``, clamped to ``[0, 1]`` with a warning if the raw estimate
falls outside that range (matching MATLAB).

``fit_constants`` overrides the three published constants of that
relationship -- it is not data, which is what MATLAB's name for it
(``params``) suggested.

### `dissectSholl_tree(tree: 'Tree', c: 'float | None' = None, dim: 'int' = 3, *, centripetal: 'bool' = True, density: 'bool' = False, n_radii: 'int' = 25, n_directions: 'int' = 20000, scale_factor: 'float | None' = None, rng=None) -> 'ShollDissection'`

Decompose a Sholl profile into what explains it (Bird & Cuntz 2018).

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

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `peters_tree(tree1: 'Tree', tree2: 'Tree', spinedis: 'float' = 3.0, synapsedis: 'float' = 3.0, resample: 'bool' = True) -> 'np.ndarray'`

Candidate synapses between two trees (Peters' rule).

For every node of ``tree1``, finds nodes of ``tree2`` within
``spinedis`` [um] -- candidate oppositions. Candidates are then
greedily accepted closest-first, each acceptance eliminating every
remaining candidate whose *either* endpoint lies within
``synapsedis`` [um] of the accepted one (in its own tree) -- avoiding
a cluster of near-duplicate "synapses" along the same stretch of
contact. ``resample=True`` (default) resamples both trees to 1 um
spacing first, matching MATLAB's default.

Returns an ``(n_candidates, 3)`` array of ``(node1, node2, distance)``.

### `r_mc_tree(tree, alpha: 'float' = 0.5, n_mc: 'int' = 100, level: 'float' = 0.05, nodes='all', *, volume_correction: 'bool' = True, confidence: 'bool' = False, n_boot: 'int' = 1000, dim: 'int' = 3, rng=None) -> 'RMCResult'`

Test whether a tree's points are spaced more regularly than chance.

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

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `sholl_tree(tree: 'Tree | list[Tree]', dd: 'float | np.ndarray' = 50.0, single_only: 'bool' = False, warn_double: 'bool' = True) -> 'ShollResult | list[ShollResult]'`

Sholl analysis: intersections of the tree with concentric spheres.

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

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

### `stats_tree(trees: 'Tree | list[Tree] | list[list[Tree]]', group_names: 'list[str] | None' = None, extras: 'bool' = False, density_thr: 'float' = 25.0) -> 'dict[str, pd.DataFrame]'`

Collect comparable statistics across one or more groups of trees.

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

### `vonMises_tree(data: 'Tree | list[Tree] | np.ndarray', dim: 'int' = 3) -> 'tuple[float, dict]'`

Fit a (modified) von Mises distribution to a tree's root-angle
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

---

## Plotting

PyVista 3D rendering, matplotlib previews, dendrograms.

### `SpreadResult`

Result of :func:`spread_tree`: the laid-out trees and how far each
one moved.

| Field | Description |
|---|---|
| `trees` | Alias for field number 0 |
| `offsets` | Alias for field number 1 |

### `chull_tree(tree: 'Tree', nodes=None, plotter=None, color='black', opacity: 'float' = 0.2, dim: 'int | None' = None)`

Convex hull around ``nodes`` (default: all).

Parameters
----------
tree : Tree
nodes : array_like, optional
    Subset of nodes to hull. Defaults to all of them.
plotter : pyvista.Plotter or matplotlib.axes.Axes, optional
    If given, the hull is drawn onto it -- as a translucent surface
    for a PyVista plotter (3D), or as a closed polyline for a
    matplotlib Axes (2D). The object type selects which, so 2D and 3D
    do not need two different parameters.
color, opacity
    Appearance of the drawn hull.
dim : {2, 3}, optional
    Default 3. ``dim=2`` hulls the XY projection, measuring enclosed
    *area* rather than volume (Design Decision #40).

Returns ``(points, scipy.spatial.ConvexHull | None)``. The hull is
``None`` whenever one cannot exist, which covers two cases:

- **too few points** (fewer than 3 in 2D / 4 in 3D), and
- **degenerate geometry** -- points that all lie on a plane (in 3D) or a
  line (in 2D) enclose no volume, so Qhull cannot build a simplex.

That second case is not exotic: many reconstructions are traced in 2D
with ``Z == 0``, and :func:`~pynetrees.flatten_tree` produces a planar tree
by construction. Returning ``None`` keeps those callable rather than
raising a raw ``QhullError`` from deep inside SciPy. For a planar tree
you almost certainly want the 2D hull instead -- pass ``dim=2``,
which measures the enclosed *area*.

If ``plotter`` is given (3D only), the hull surface is added to it.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `dA_tree(tree: 'Tree', ax=None)`

Display a tree's adjacency matrix as a sparsity image (matplotlib
`spy`) -- a quick structural/debugging view, not anatomy.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `dendrogram_tree(tree: 'Tree', yvec=None, ax=None, color='black', linewidth: 'float' = 1.0)`

A 2D dendrogram: each node at ``(xdend_tree(tree), yvec)`` (default
``yvec``: path length from the root), connected to its parent by an
L-shaped (horizontal-then-vertical) line, the standard dendrogram
convention. Rendered with matplotlib -- an abstract topological
diagram, not spatial anatomy, so PyVista's 3D machinery isn't the
right tool here.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `plot_mpl_tree(tree: 'Tree', ax=None, color='black', scalars=None, cmap: 'str' = 'viridis', linewidth: 'float' = 1.0, nodes=None)`

Quick line-only 3D render via matplotlib (no diameter, no GPU
acceleration -- see this module's docstring for why `plot_tree`
(PyVista) is the recommended path for anything but a fast preview).
Fixes matplotlib's well-known default 3D aspect-ratio distortion via
`set_box_aspect`, so anatomy isn't visually stretched. Returns the Axes.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `plot_tree(tree: 'Tree', color=None, offset=(0.0, 0.0, 0.0), nodes=None, res: 'int' = 8, *, mode: 'str' = 'tube', cmap: 'str' = 'viridis', categories: 'bool' = False, scalars=None, plotter=None, show: 'bool' = False, screenshot: 'str | None' = None, **mesh_kwargs)`

Render a tree in 3D with PyVista.

Parameters
----------
tree : Tree or list of Tree
    A **list of trees** (or a list of lists) is drawn into one plotter,
    which is the deliberate exception to this package's list-in/list-out
    rule -- see :func:`_plot_population`. With no ``color`` given the
    cells cycle :data:`POPULATION_COLORS` so they can be told apart.
color : optional
    Follows MATLAB's overloading:

    - a colour name (``"black"``) or RGB triple ``(r, g, b)`` -- one
      flat colour for the whole tree;
    - a length-``n_nodes`` vector -- per-node values mapped through
      ``cmap`` (branch order, region, path length, anything);
    - an ``(n_nodes, 3)`` array -- an explicit RGB colour per node.

    Defaults to black.
offset : tuple, default (0, 0, 0)
    Translate the rendered geometry, for laying several trees out side
    by side without moving the trees themselves. MATLAB's ``DD``. For a
    group, an ``(n_trees, 3)`` array gives one offset each, so
    ``plot_tree(trees, offset=spread_tree(trees).offsets)`` is the whole
    gallery in one call.
nodes : array_like, optional
    Render only these nodes' segments. MATLAB's ``ipart``.
res : int, default 8
    Number of sides on each tube. MATLAB's ``res``.
mode : {'tube', 'line'}, keyword-only, default 'tube'
    ``'tube'`` builds one diameter-tapered tube mesh for the whole tree
    -- realistic geometry, still a single fast mesh however many
    segments. ``'line'`` skips tubing for a faster, diameter-less
    preview.
cmap : str, keyword-only
    Colormap used when ``color`` is a value vector.
categories : bool, keyword-only, default False
    Treat mapped values as discrete categories (e.g. region indices)
    rather than a continuous scale.
scalars : array_like, keyword-only, optional
    Explicit per-node values, overriding any interpretation of
    ``color``. Retained for callers written against the previous
    two-argument form, and as the escape hatch for the 3-node
    ambiguity noted below.
plotter : pyvista.Plotter, keyword-only, optional
    Draw into an existing plotter, to overlay several trees (MATLAB's
    ``hold on``). One is created if omitted, with
    ``off_screen=not show`` so headless environments work.
show, screenshot : keyword-only
    Display the window / write a PNG.

Returns
-------
pyvista.Plotter

Notes
-----
**Positional order matches MATLAB** (``intree, color, DD, ipart, res``)
as of Design Decision #54, so translated code reads the same. Everything
this port adds beyond MATLAB's five is keyword-only, which is what keeps
the order matched: a future addition cannot wedge itself into a
positional slot.

``scalars`` must be length ``n_nodes`` -- the *whole* tree, even when
``nodes`` renders a subset -- since the mesh keeps every node's
coordinates regardless (``nodes`` only selects which line cells get
built, which is the cost that matters on large trees).

MATLAB's ``'-b'`` (flat "blatt" patches) and ``'-2q'``/``'-3q'``
(quiver) render modes are not reproduced: ``'-b'`` exists to dodge the
cost of real cylinders in MATLAB's renderer, which ``mode='tube'``
simply does not have, and quiver plots of a 4000-segment tree are
unreadable. ``'-2l'``/``'-3l'`` map onto ``mode='line'``.

### `plotsect_tree(tree: 'Tree', sect, color='black', offset=(0.0, 0.0, 0.0), ipar=None, ax=None, linewidth: 'float' = 2.0, *, full_output: 'bool' = False)`

Draw the path from one node down to another.

Parameters
----------
tree : Tree
sect : (start, end)
    Two node indices. ``start`` must be an **ancestor** of ``end``:
    the path is read off the tree's own parent chain, not searched
    for, so it always runs away from the root.
color : matplotlib color, default "black"
offset : (dx, dy, dz), default (0, 0, 0)
    Shift the drawn path, for overlaying several trees.
ipar : np.ndarray, optional
    A precomputed :func:`~pynetrees.ipar_tree`. Worth passing when
    drawing many sections of the same tree -- MATLAB's docstring calls
    computing it "the slow part of this function".
ax : matplotlib 3D Axes, optional
linewidth : float, default 2.0
full_output : bool, default False
    Also return the node indices along the path.

Returns
-------
Axes, or (Axes, indices)

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `pointer_tree(plotter, tree: 'Tree', nodes, style: 'str' = 'marker', color='red', size: 'float' = 8.0, offset=(0.0, 0.0, 0.0))`

Mark specific nodes on an existing ``plotter`` (electrodes,
points of interest, ...). ``style`` is ``"marker"`` (a point, fast) or
``"sphere"`` (a small rendered sphere, clearer at a distance but
heavier for many points). Returns the same plotter.

MATLAB's tapering-electrode modes (`'-l'`/`'-v'`, built from a tiny
synthetic frustum tree) aren't ported -- niche relative to marking a
location, which `"sphere"`/`"marker"` already cover.

### `spread_tree(trees: 'list[Tree]', dx: 'float' = 50.0, dy: 'float' = 50.0) -> 'SpreadResult'`

Lay trees out on a roughly square grid so they can be shown together
without overlapping.

Returns
-------
SpreadResult
    ``(trees, offsets)`` -- the translated copies *and* the offsets
    applied. MATLAB has two functions here, `spread_tree` returning the
    offsets and `spread_trees` returning the trees; they differ only in
    return type and one is a wrapper around the other, so this is one
    function returning both (REVIEW_PLAN P7).

Notes
-----
Reimplemented as a straightforward greedy row-packing bin layout
(accumulate widths along a row until a target row width is exceeded,
then wrap) instead of MATLAB's `cumsum`/`mod` index arithmetic -- same
"roughly square, no-overlap" goal, which is an aesthetic choice rather
than a uniquely-determined one, and much easier to follow.

### `vtext_tree(plotter, tree: 'Tree', values=None, nodes=None, color='red', font_size: 'int' = 14, offset=(0.0, 0.0, 0.0))`

Add text labels at node positions (default: node index) to an
existing ``plotter``. Returns the same plotter.

### `xdend_tree(tree: 'Tree')`

X-coordinate for each node useful for a dendrogram layout: each
node's position is the midpoint of its leftmost and rightmost
descendant terminal's rank (terminals ranked left-to-right in node
order). Returns ``xdend`` (length ``n_nodes``).

Reimplemented as an O(n_nodes) bottom-up tree accumulation (post-order:
every node's [min, max] leaf-rank range is its children's ranges
combined) instead of MATLAB's `ipar`-matrix sort-and-diff trick --
same result, standard "assign each internal dendrogram node the
average of its leaves' positions" algorithm, and avoids an O(n^2)
blowup from testing "is this terminal a descendant of that node" for
every (node, terminal) pair.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `xplore_tree(tree: 'Tree', mode: 'str' = 'nodes', color='black', offset=(0.0, 0.0, 0.0), fig=None)`

Diagnostic views of a tree, for looking at one rather than
presenting it.

Parameters
----------
tree : Tree
mode : {'nodes', 'regions', 'projections'}, default 'nodes'
    ``'nodes'`` draws the arbor with every node's index written on it,
    which is what you want when a function has just told you something
    about node 412. ``'regions'`` colours by region and labels each at
    its centre of mass. ``'projections'`` shows the xy, yz and xz views
    stacked, so a cell's depth is visible without rotating anything.
    MATLAB spells these ``'-1'``, ``'-2'`` and ``'-3'``.
color : matplotlib color, default "black"
offset : (dx, dy, dz), default (0, 0, 0)
fig : matplotlib Figure, optional

Returns
-------
Figure

Notes
-----
MATLAB's ``'-2'`` labels each region with ``tree.rnames{counter}``,
indexing by the *loop* counter rather than by the region value
``uR (counter)`` it is labelling. On a tree whose regions are not
``1 : n`` -- which any tree that has had a region deleted is -- the
labels are attached to the wrong regions. Fixed here.

MATLAB's arrow overlay (``plot_tree (..., '-3q')``, a quiver per
segment showing which way is away from the root) is not reproduced:
matplotlib's 3D quiver draws one cone per arrow and is unusable past a
few hundred segments. The node indices already carry the direction,
since they increase away from the root in a sorted tree.

    An **empty tree** gives an empty result rather than an error -- see
    :mod:`pynetrees._empty`.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

---

## Image stacks

Loading, skeletonising and diameter-fitting confocal/2-photon stacks.

### `Stack`

A tiled 3D image stack in micron coordinates.

Attributes
----------
tiles : list[np.ndarray]
    One ``(nx, ny, nz)`` array per field of view. MATLAB's ``stack.M``.
origin : np.ndarray
    ``(n_tiles, 3)`` position of each tile's first voxel [um].
    MATLAB's ``stack.coord``.
voxel : np.ndarray
    ``(3,)`` voxel size [um], shared by every tile. MATLAB's
    ``stack.voxel``.
names : list[str]
    One per tile. MATLAB's ``stack.sM``.

### `fitD_stack(tree: 'Tree', stack: 'Stack', max_radius: 'float' = 30.0, samples: 'int' = 5, sigma: 'float' = 3.0, selectivity: 'float' = 0.1) -> 'np.ndarray'`

Measure a tree's diameters from the image it was traced from.

For each segment: sample the fluorescence along a line perpendicular to
it, average those profiles along the segment, sharpen the edges by
convolving with a derivative of a Gaussian, and read the width between
the innermost turning points either side of the cable. Requires the tree
and the stack to share a coordinate frame -- which they do if the tree
was traced from it.

Parameters
----------
tree : Tree
stack : Stack
max_radius : float, default 30.0
    Half-width of the perpendicular sampling line, **in voxels**. Sets
    the largest diameter that can be found.
samples : int, default 5
    How many positions along each segment to average over. **MATLAB has
    no such parameter and effectively samples one** -- see Notes.
sigma : float, default 3.0
    Width [voxels] of the Gaussian whose derivative sharpens the edges.
selectivity : float, default 0.1
    How steep a turn has to be to count as an edge. MATLAB hard-codes
    0.1 and its own comment names this as the number to change.

Returns
-------
np.ndarray
    One diameter per node, **in microns**. Nodes whose segment could
    not be measured keep the tree's own diameter.

Notes
-----
**MATLAB measures every segment at a single point, and knows it.**
``stacks/fitD_stack.m:124`` reads::

    % TODO, CRITICAL: RIGHT NOW ONLY THE TERMINAL POINT IS TAKEN
    mPX = [(P1(1) + cV(1)) (P1(1) + cV(1)) (P2(1))];

Three sampling positions are built, but ``cV`` is ``P2 - P1``, so
``P1 + cV`` *is* ``P2`` -- all three collapse onto the segment's far
end. Every diameter is therefore read at one point rather than along
the cable, which is not what the surrounding code was written to do.
Here ``samples`` positions are spread evenly along the segment as
intended; pass ``samples=1`` for MATLAB's behaviour.

Whether that helps depends on the data, and it is worth knowing that it
is not free: averaging along a segment picks up the *sibling* branch
near a branch point, so on a clean synthetic phantom the single-point
measurement is actually the less variable of the two (spread 1.2 versus
1.8 voxels). On real, noisy fluorescence the averaging is the point.
Both are available; neither is asserted to be better.

**The result is converted to microns**, which MATLAB does not do. Its
width comes out in voxels along the sampling line and is returned as-is,
to be assigned to ``tree.D`` -- a field in microns everywhere else in
the toolbox. The two agree only when the in-plane voxel size happens to
be 1 um. Here the width is scaled by the length of one sampling step in
microns, which is exact even for anisotropic voxels.

One MATLAB idiosyncrasy **is** reproduced: the two edge indices are
offset by one relative to each other (``m_1 = ... + i_max`` against
``m_2 = ... + i_max - 1``), which comes from ``diff`` shortening the
array and makes every width one step narrower than the index gap. It is
a sub-voxel systematic offset in an already-approximate measurement, and
changing it would silently move numbers people have published.

See MATLAB_TOOLBOX_BUGS.md.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `load_folder(path: 'str | Path', voxel=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), pattern: 'str' = '*') -> 'Stack'`

Load every image in a folder as the z planes of one tile.

Ports `loaddir_stack`. Files are taken in sorted order, which is what
makes ``plane001.tif``-style naming work and ``plane1.tif`` not; MATLAB
uses ``dir`` order, which is the same trap.

### `load_stack(path: 'str | Path') -> 'Stack'`

Load a MATLAB ``.stk`` file.

A ``.stk`` is a ``.mat`` workspace holding one ``stack`` struct -- the
same relationship ``.mtr`` has to ``.mat``.

### `load_tiff(path: 'str | Path', voxel=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)) -> 'Stack'`

Load a multi-page TIFF as a one-tile stack.

MATLAB's `loadtifs_stack` reads pages one at a time through
``imread``; ``tifffile`` does the whole file in one call and handles
the compression schemes microscopes actually emit.

### `save_stack(stack: 'Stack', path: 'str | Path') -> 'Path'`

Write a :class:`Stack` to a MATLAB-readable ``.stk``.

v5 rather than v7.3, for the reasons in
:func:`~pynetrees.io.save_mtr` -- MATLAB's ``load`` reads either.

### `show_stack(stack: 'Stack', axis: 'int' = 2, ax=None, cmap: 'str' = 'gray', alpha: 'float' = 1.0)`

Maximum-intensity projection of every tile, in micron coordinates.

Parameters
----------
stack : Stack
axis : {0, 1, 2}, default 2
    Project along x, y or z. MATLAB draws all three at once as
    semi-transparent textured surfaces in a 3D axes; here you ask for
    the one you want, on ordinary 2D axes, because a projection *is*
    two-dimensional and stacking three translucent ones on top of each
    other is hard to read.
ax : matplotlib Axes, optional
cmap : str, default "gray"
alpha : float, default 1.0

Returns
-------
Axes

### `skeletonize_stack(stack: 'Stack', thr: 'float | None' = None, close: 'bool' = False) -> 'np.ndarray'`

Thin a stack to its centreline and return the points, in microns.

The output is what :func:`~pynetrees.MST_tree` wants: an ``(n, 3)`` array
of carrier points to wire into a tree.

Parameters
----------
stack : Stack
thr : float, optional
    Binarisation threshold. Defaults to Otsu's, computed per tile,
    which is a documented and reproducible choice; MATLAB instead
    walks a 100-bin histogram down from the top until it has counted
    30000 voxels, a fixed number that means something different for
    every stack size.
close : bool, default False
    Morphologically close the binary volume first, joining voxels
    separated by a single gap. MATLAB's ``'-c'``.

Returns
-------
np.ndarray
    ``(n, 3)`` carrier points [um].

Notes
-----
**This will not reproduce MATLAB's skeleton voxel for voxel.**
`skel_stack` is a hand-rolled 3D thinning -- its own header says
"hopefully correctly interpreted from their papers" of Palagyi and Kuba
-- while this delegates to `skimage.morphology.skeletonize`, which
implements Lee, Kashyap & Chu (1994). Both are medial-axis thinnings
and both preserve topology; the individual voxels they keep differ.
Reimplementing the toolbox's own reading of a paper would have been
reimplementing it worse, but the difference is real and this is a
reconstruction front-end, so it is said out loud rather than buried.

---

## NEURON simulation

Requires the `neuron` package.

### `NeuronModel`

A tree built into a live NEURON section tree.

Attributes
----------
tree : Tree
    The tree the model was built from -- node indices below are this
    tree's own (:func:`~pynetrees.dissect_tree` handles the root
    directly, no `root_tree` prepending needed here).
sections : list
    Every ``h.Section`` created, one per dissected branch.
node_section : dict[int, h.Section]
    Which section each node index lives on.
node_x : dict[int, float]
    That node's normalized position (0-1) along its section.
region_sections : dict[str, list[h.Section]]
    Sections grouped by the tree region name of their end node --
    for :func:`insert_mechanism`'s ``region=`` filter.

### `build_neuron_model(tree: 'Tree', freq: 'float' = 100.0, d_lambda: 'float' = 0.1, e_pas: 'float' = -70.0) -> 'NeuronModel'`

Build a live NEURON section tree from ``tree``.

Requires ``tree.Ri``/``tree.Gm``/``tree.Cm`` (see
:mod:`pynetrees.electrotonics`), used as each section's ``Ra``/passive
``g_pas``/``cm`` (resolved from the section's end node, scalar or
per-node). ``freq``/``d_lambda`` control segment count via NEURON's own
d_lambda rule (see module docstring); ``e_pas`` is the passive leak
reversal potential [mV], applied uniformly (``Tree`` has no per-node
resting-potential attribute of its own).

Note: a region occupying *only* the root node itself (i.e. the region
already changes at the root's own child) doesn't get a section of its
own -- it's absorbed into the section that follows, since there's no
boundary before the root to split it off with. A real biophysical
compartment (e.g. a soma you want to give its own active
conductances via :func:`insert_mechanism`) should be modeled with more
than one node in that region, which is normal practice anyway.

    Accepts a **list of trees** (or a list of lists of trees) as well as a
    single one, returning results in the same shape -- see
    :mod:`pynetrees._population`.

### `insert_mechanism(model: 'NeuronModel', mechanism: 'str', region: 'str | None' = None, **params)`

Insert a NEURON mechanism (e.g. ``"hh"``) on the model's sections.

``region=None`` (default) applies to every section; otherwise only
sections belonging to that tree region name (see
:attr:`NeuronModel.region_sections`). Keyword arguments are set as
``seg.<mechanism>.<param> = value`` on every segment of every affected
section (a uniform value across the section -- for a spatially-varying
parameter, insert first and then set values directly on
``model.node_section``/``model.node_x`` locations yourself).

### `run_current_clamp(model: 'NeuronModel', at_node: 'int', amp: 'float', delay: 'float', dur: 'float', tstop: 'float', record_nodes: 'list[int] | None' = None, v_init: 'float' = -70.0) -> 'tuple[np.ndarray, dict[int, np.ndarray]]'`

Inject a rectangular current step and record voltage.

``at_node``/``record_nodes`` are node indices into the tree ``model``
was built from. ``amp`` [nA],
``delay``/``dur``/``tstop`` [ms]. Returns ``(t, v)`` where ``t`` is the
time axis [ms] and ``v`` maps each recorded node to its voltage trace
[mV] (``record_nodes`` defaults to ``[at_node]``).

---

## Blender export (optional)

Not imported by `import pynetrees` -- opt in with `from pynetrees import blender`. Needs the `blender` extra (`bpy`), a ~300 MB wheel that pins `numpy < 2`.

### `REGION_COLORS`

Value: `{'soma': (0.85, 0.75, 0.35), 'axon': (0.35, 0.55, 0.85), 'dendrite': (0.85, 0.35, 0.35), 'basal': (0.85, 0.45, 0.3), 'apical': (0.9, 0.3, 0.45), 'spines': (0.55, 0.85, 0.45), 'primary': (0.8, 0.5, 0.2)}`

### `build_tree(tree: 'Tree', name: 'str | None' = None, *, region_colors: 'dict | None' = None, resolution: 'int' = 8, taper: 'bool' = True, min_diameter: 'float' = 0.1, reset: 'bool' = True, offset=(0.0, 0.0, 0.0)) -> 'list'`

Build a tree as Blender curve objects, one per region.

Parameters
----------
tree : Tree
name : str, optional
    Prefix for the created objects. Defaults to the tree's own name.
region_colors : dict, optional
    ``{region name: (r, g, b)}``, overriding :data:`REGION_COLORS`.
resolution : int, default 8
    Bevel resolution -- the swept circle gets ``4 * (resolution + 1)``
    sides. 8 is smooth at print size; drop to 2 or 3 for a cell with
    many thousands of nodes.
taper : bool, default True
    Vary the tube radius with each node's diameter. With ``False``
    every branch is drawn at the mean diameter, which reads better for
    a schematic and renders faster.
min_diameter : float, default 0.1
    Floor on the drawn diameter [um]. A reconstruction with zero
    diameters would otherwise collapse to invisible threads.
reset : bool, default True
    Clear the session first. Pass ``False`` to add this tree to a scene
    already built.
offset : (dx, dy, dz), default (0, 0, 0)
    Shift the tree, for laying several out in one scene -- pair with
    :func:`~pynetrees.spread_tree`.

Returns
-------
list
    The created ``bpy`` objects, one per region.

### `render_tree(tree: 'Tree | list[Tree]', path: 'str | Path', *, size=(1200, 900), view: 'str' = 'xy', samples: 'int' = 32, background=(0.05, 0.05, 0.06), margin: 'float' = 1.1, **kwargs) -> 'Path'`

Build a tree and render it to an image, with no Blender window.

Parameters
----------
tree : Tree or list[Tree]
path : str or Path
    ``.png`` is appended if missing.
size : (width, height), default (1200, 900)
view : {'xy', 'xz', 'yz'}, default 'xy'
    Which plane to look at. An **orthographic** camera is used, framed
    on the tree's bounding box -- a perspective one would make the near
    half of a cell look thicker than the far half, which is exactly the
    artefact a morphology figure must not have.
samples : int, default 32
    EEVEE sampling. Higher is cleaner and slower.
background : (r, g, b), default dark grey
margin : float, default 1.1
    How much room to leave around the tree.

Extra keyword arguments go to :func:`build_tree`.

Returns
-------
Path

### `reset_scene() -> 'None'`

Empty the shared Blender session.

``bpy`` starts with Blender's default scene -- a cube, a camera and a
light -- and keeps everything ever created in it, because there is only
one session per process. Every entry point here calls this first unless
told not to.

### `save_blend(tree: 'Tree | list[Tree]', path: 'str | Path', **kwargs) -> 'Path'`

Build a tree (or several) and save the scene as a ``.blend``.

Extra keyword arguments go to :func:`build_tree`.

Returns the path written. Open it in Blender and everything is a real,
editable object -- which is the whole reason for preferring this to
`pov_tree`'s one-shot scene file.

---
