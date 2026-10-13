# Changelog

Notable changes to `pynetrees`, the Python port of the MATLAB TREES toolbox.

Version numbers are pre-1.0 and follow the usual convention for that: the
minor number moves for features and for breaking changes alike, and there are
no deprecation shims between them — the port has no external users yet, so a
rename now costs one commit and a rename later costs everybody's scripts. Each
entry says what changed and, where it matters, *why*, since a port's value is
mostly in the decisions rather than the diff.

The blow-by-blow record lives in [PORT_STATUS.md](PORT_STATUS.md) (per-function
status and every design decision), [NOT_YET_PORTED.md](NOT_YET_PORTED.md) (what
is left, per MATLAB file) and [MATLAB_TOOLBOX_BUGS.md](MATLAB_TOOLBOX_BUGS.md)
(bugs found in the original). This file is the summary.

---

## 0.3.1 — 2026-09-09

Revalidation of everything that depended on MATLAB's `inpolyhedron`, now that
the real File Exchange routine is on the path. **1050 tests pass, 17 skip**;
0.3.0 was 992.

One real bug found and fixed, one long-standing "not verifiable against
MATLAB" claim retired, and two more upstream bugs recorded.

### Fixed — `boundary_tree` now matches MATLAB exactly

MATLAB's `boundary()` was the port's last unverifiable dependency: Octave
does not implement it, so the differential harness had nothing to run, and
the docstrings said so. It turns out `boundary.m` is a **readable m-file, not
a built-in**, and it states outright what the shrink factor is:

```matlab
shp       = alphaShape (P, Inf);
Acritical = criticalAlpha (shp, 'one-region');
Ahispec   = alphaSpectrum (shp);  Ahispec = Ahispec (Ahispec >= Acritical);
shp.Alpha = Ahispec (numA + 1 - max (ceil ((1 - S) * numA), 1));
```

The spectrum of a 3D alpha shape is just the sorted tetrahedron circumradii,
and the shrink factor picks a **rank** within it. The port already did that.
Its one error was the anchor: `shrink = 1` was pinned to the smallest radius
at which no point is left out, where MATLAB pins it to the smallest radius at
which the shape is also **connected** — `criticalAlpha(..., 'one-region')`
rather than `'all-points'`.

Those two coincide on `sample_tree` and `hss_tree`, which is exactly why the
error survived an earlier differential session; on `hsn_tree` they are 149.97
and 158.68, and the gap moved *every* shrink factor and cost up to 5% of the
enclosed volume. `_one_region_cutoff` computes MATLAB's number with a
union-find over the points — adding a simplex merges its vertices, a point no
simplex has reached is a component of its own, so "one region" is precisely
"one component", and both halves of the condition fall out of one loop.

The result is **exact parity**: identical enclosed volume to every digit
MATLAB prints, identical surface triangle count, and an identical
inside/outside verdict on 20000 query points, for `sample_tree`, `hsn_tree`
and `hss_tree` at shrink 0, 0.25, 0.5, 0.75 and 1, in 2D and 3D. 58 new tests
in `tests/test_matlab_boundary.py` pin MATLAB's own output;
`scripts/matlab/run_boundary_validation.m` regenerates it.

This lands on `boundary_tree`, `scaleV_tree`, `dissectSholl_tree.V`,
`r_mc_tree`, `theta_mc_tree` and `growth_tree`, all of which draw from this
shape.

### Verified — the three functions that reach it through `inpolyhedron`

MATLAB's `r_mc_tree`, `theta_mc_tree` and `growth_tree` all fill a bounding
box with `rand` and keep whatever `inpolyhedron` says is inside. This port
samples the boundary's own simplices instead, weighted by volume — the same
uniform distribution with nothing thrown away — so what needed checking was
whether the two agree end to end. They do:

| | MATLAB | port |
|---|---|---|
| `r_mc_tree` `r0` (`sample_tree`) | 2.82101 | 2.82101 — every digit |
| `r_mc_tree` `R`, 400 iterations | 0.619638 | 0.620159 (0.08%) |
| `r_mc_tree` `R`, volume-corrected | 0.566855 | 0.567399 (0.10%) |
| `theta_mc_tree` (`sample_tree`) | 15.85 | 15.8877 |
| `theta_mc_tree` (`hsn_tree`) | 24.24 | 24.3064 |

The 1–2% gap in `R` at 20 iterations is the two samplers' noise, not bias: it
collapses to 0.1% at 400. `theta_mc_tree`'s 0.25% is MC noise plus MATLAB's
0.01-wide binning, which by construction reports the last bin *below* the
crossing.

### Found — `convexity_tree` still is not measuring convexity (bugs 1.13, 1.14)

The 2026 release fixed the two lines this port had flagged and deviated from,
so the open question was whether the port should now reproduce it. Run
against the fixed version: no.

`convexity_tree` meshgrids its terminal set against **itself**, so every
unordered pair is tested twice and the `nS1` **self-pairs** are tested at all.
A self-pair makes the intersection matrix singular — MATLAB prints *"Matrix
is singular to working precision"* once per pair — and the resulting `NaN`
fails every comparison, so the pair is recorded as *visible*. That puts a hard
floor of `1 / nT` under the score. Compounding it, the boundary is built from
the terminals, so every terminal lies **on** it and each segment "crosses" at
its own endpoint.

On `sample_tree` MATLAB returns 0.0384615, which is exactly `1/26` for its 26
terminals: **all 26 pairs it scored as visible are self-pairs, and not one
genuine pair survived.** Confirmed by re-running its loop and comparing the
flagged indices against the diagonal. The port keeps its own estimator —
unordered pairs, no self-pairs, containment against the space-filling hull,
which terminals are inside rather than on — and returns 0.997.

Separately, **`-dim2` does nothing**: `parseArgs` sets `dim2` true but leaves
`dim3` at its default of true, and the function tests `pars.dim3` first.
`convexity_tree(sample_tree, '-dim2')` silently returns the 3D answer; only
the name-value form reaches the 2D branch (0.536 against 0.0384615).

### Worth knowing — `inpolyhedron` needs consistently wound triangles

Not a port bug, but it invalidated the first comparison attempted here and
would invalidate anyone else's. `inpolyhedron` presumes face normals point
outward. MATLAB's `boundary()` supplies that — on a convex hull its verdict
agrees with `pointLocation` on all 20000 query points. The port's
`_boundary_faces` does not, because nothing in the port ever reads a normal,
and feeding its mesh to `inpolyhedron` reports **3.5x too many points inside,
with no error at all**. Compare solids by volume or by
`Delaunay.find_simplex`. Recorded in `scripts/matlab/README.md`.

---

## 0.3.0 — 2026-09-09

Round 5 of the review, plus a diff against the 2026 TREES release. **991
tests pass, 17 skip** in the main environment; a second environment with
Blender runs 963 with 16 skips (see *Blender* below). 0.2.0 was 932.

### The 2026 TREES release, diffed

The upstream release differs from the reference this port was written
against in **exactly four files** — and two of them change what "correct"
means here.

| File | Change | What it meant for the port |
|---|---|---|
| `convexity_tree.m` | `boundary(X,Y,Z,0)` → `,1)`, and `c = 1 - nnz(Inds)/…` → `c = nnz(Inds)/…` | **Both defects this port documented and refused to reproduce are now fixed upstream.** The deviation is history; the docstring says so. |
| `dissectSholl_tree.m` | the `if rmax > 500; sf = 2; end` block deleted | The port reproduced that fudge behind `scale_factor=`. Upstream abandoned it, so **the default is now `1.0` on both branches**. Pass `scale_factor=2.0` for a pre-2026 number. |
| `LIF_tree.m` | `'-t'` sigmoid commented out; the `if pars.e` guard round the progress print commented out too | Feature disabled upstream and the print made unconditional. The port keeps both as options. |
| `histax.m` | `xax(:)'` orientation fix | Cosmetic. |

**A new upstream bug fell out of the second one.** Deleting the 500 µm block
also deleted `rmax = max (eucs (:))`, which the **3D** branch still reads.
`rmax` now exists only inside the mutually exclusive `dim2` branch, and
`dim3` defaults to `true` — so `dissectSholl_tree` raises on its own default
path. Recorded as 1.11 in `MATLAB_TOOLBOX_BUGS.md`, along with 1.12 for the
two `convexity_tree` fixes. The port is unaffected: it builds `radii` the
same way in both branches.

### Added — POV-Ray and X3D export

The last two unported IO functions.

- **`save_x3d`** (`IO/x3d_tree.m`) — cylinders as XML, which browsers render
  with no plugin and Blender/MeshLab/ParaView import. `spheres=True` closes
  the joints at sharp branch points; `offset=` lays several cells out in one
  scene; `diameters='thin'/'thick'` for visibility in a wide view. The
  rotation carrying X3D's Y-up cylinder onto each segment is computed with
  the degenerate cases handled *before* the `arccos` rather than patched
  after it.
- **`save_pov`** (`IO/pov_tree.m`) — a `#declare`d union of cylinders, plus a
  `sh<name>.pov` scene that includes it and adds camera, lights and
  background, in one of three styles (`plain`, `fluorescence`, `coral`).
  `blob=True` (MATLAB's default) fuses the cylinders into one smooth
  implicit surface.

  Colour-coding is written **into each cylinder**. MATLAB writes the colours
  to a side-car `.dat` and has POV-Ray `#read` them back positionally while
  parsing, which makes the `.pov` unrenderable alone and the pairing
  fragile: lose the `.dat`, or edit one cylinder out, and every subsequent
  colour is off by one.

Neither reproduces MATLAB's `'->'` option, which calls `winopen` on the
result: a save function should not start programs.

### Added — `save_model_hoc`: the built model, not just the morphology

`save_hoc` writes a morphology. This writes **everything that was done to
it** — segment counts, `Ra`, `cm`, every inserted mechanism and every
parameter value it currently holds, including ones that vary along the cell
— so a model assembled with `build_neuron_model` / `GC_init` /
`insert_mechanism` can be handed to plain `nrniv`, to a collaborator who
does not use Python, or to a cluster job.

The state is read back **from the live sections**, not re-derived from
whatever built it. A density set by a callable gradient, a value poked in by
hand, a mechanism inserted from the console — all of it is in the sections,
so all of it comes out. Verified by loading the file into a *fresh* NEURON
and comparing: identical section count, segment count, lengths, `Ra`, `cm`,
`g_pas` and the per-segment gradient.

Only settable PARAMETERs are written. `psection()` also reports a
mechanism's STATEs (`m`, `h`, `n`) and ASSIGNEDs (`ina`, `gk`); writing
those would be writing down the middle of a simulation and calling it a
model. A uniform value collapses to one line per section, so the file stays
readable and only real gradients are per-segment. The whole granule cell
comes out as 4750 lines with all 16 channels correctly distributed per
region.

What a section cannot be asked about — recording vectors, `NetStim`/`NetCon`
wiring, `h.` globals — is named in a comment block at the end of the file,
rather than quietly producing a model that runs and is not yours.

### Added — `Tree.meta` round-trips through MATLAB

No v7.3 needed, and no new container: MATLAB's TREES already hangs ad-hoc
data straight off the tree struct — `tree.col`, `tree.NID`,
`tree.Rho_soma`, `tree.jpoints` — which is exactly what `meta` is for. The
two are now the same thing:

- loading, every field that is not part of the `Tree` model lands in
  `tree.meta` under its own name;
- saving, every `meta` entry is written back as a **top-level** field.

Flat rather than nested on purpose. A nested `meta` struct would round-trip
just as well and be invisible to every MATLAB function that reads `tree.col`
by name.

This turned up that **the bundled granule cells carry `NID`
(`cell_SH07all2_1`, T2N's hoc identifier) and `col`** — both of which the
loader had been dropping since `.mtr` support landed. A key colliding with a
real `Tree` field is refused rather than silently winning or losing; a value
`savemat` cannot encode is dropped with a warning.

### Changed — `Tree` has one region accessor

`region_mask()` is now **`region_nodes()`**, returning the boolean mask;
the old `region_nodes()` (indices) and `region_index()` are gone, as is
`Tree.validate()` (`ver_tree(tree)` was always the same thing).

A mask because that is what the surrounding code wants: it composes with
`&`/`|`/`~`, indexes the per-node arrays directly, and lines up element-wise
with `B_tree`/`T_tree`/`len_tree`. `np.flatnonzero(...)` gives indices where
they are genuinely wanted, and is explicit about it.

**This changes the meaning of existing calls rather than breaking them
loudly**: `tree.region_nodes("soma")[0]` used to be the first somatic node
and is now a bool. Every call site in the package, tests, notebooks and docs
was updated.

### Changed — ragged populations are valid

`[tree, [tree, tree]]` — one cell beside a group of two — now returns
`[value, [value, value]]`. 0.2.0 refused it on the theory that it might be a
typo; that was wrong. A rule which maps element to element cannot be
confused by its input, and refusing a valid structure to guard against a
hypothetical mistake made the common case worse. An element that is neither
a tree nor a list is still refused, because that one really has no per-tree
meaning.

### Changed — other API consistency

| Was | Is |
|---|---|
| `resample_tree(method="matlab")` | `method="default"` — every function here reproduces MATLAB unless it says otherwise, so naming one option after that distinguishes nothing. What distinguishes the alternative is what it preserves, hence `"anchors"`. |
| `ShollResult` (dataclass) | a `NamedTuple`, so it unpacks in MATLAB's own order like every other multi-output result here |

`plot_mpl_tree`, `dA_tree`, the integer `dim` convention and the absence of
deprecation shims were all already correct from 0.1.4's V1 cleanup; the
review items covering them needed no work.

### Added — `insert_tree` takes an SWC matrix

`insert_tree(tree, swc)` with MATLAB's `[inode R X Y Z D idpar]`, passed
positionally the way MATLAB does it — a 2-D array in the `X` slot cannot
mean anything else. A 5-column `[R X Y Z D]` form hangs everything off the
root.

Column 6 is a **diameter**, not a radius. The layout is identical to a
standard `.swc` file, where that column *is* a radius, so feeding raw SWC
rows in would silently halve every diameter; `radius=True` doubles it for
you. MATLAB shouts the same warning in its docstring. A region index the
tree has no name for is refused rather than inserted — an out-of-range `R`
is exactly what `ver_tree` flags.

Called with nothing to insert, the tree comes back unchanged, which is what
MATLAB does (`if ~isempty (pars.swc)` guards its whole body).

### Blender is part of the standard install

`bpy` is now in the `all` extra, and a second environment
(`.venv-blender`, gitignored) confirms it works: **all 17 previously-skipped
Blender tests pass**, and the whole suite runs green on numpy 1.26 with
bpy 5.0.1. `pynetrees.blender` was never broken — only untestable.

The cost is real and is now stated where it is decided rather than in a
footnote: `bpy` requires **numpy < 2 and CPython 3.11**, so `all` holds the
whole environment there. **`all-no-blender`** is the identical set without
it, for numpy 2 or Python 3.12+.

### `soma_tree`: `maxD` and `l` are independent

Checked against the source rather than assumed. MATLAB does **not** tie them
together — `l` only *defaults* to `1.5 * maxD`. Its note *"recently corrected
to match maxD and l"* means the profile now *attains* both: the superseded
formula `maxD/4 * cos(pi*Plen/(l/2)) + maxD/4` peaked at `maxD/2` and ran a
full cosine period inside the selected region, so neither requested number
came out. The current `maxD * cos(pi*Plen/l)` reaches exactly `maxD` at the
root and zero at `l/2`.

This port already implemented the corrected formula. The docstring said
"(approximate) target diameter", which undersold it; it now states the
independence and has tests for it.

### Added — `docs/sentinel-options.md`

An evaluation, not a change. `-1` serves three different jobs — "no parent",
padding in `ipar_tree`/`idchild_tree`, and a NaN stand-in in integer arrays —
and it is a **valid NumPy index**, so a missing value silently becomes the
last element. Five options are compared (guardrails, masked arrays, `np.nan`
floats, a non-indexable sentinel, splitting the roles) with a
recommendation and a note on what would change it.

---

## 0.2.0 — 2026-09-08

Answers the third round of `PyNeTREES code review.md`. Four latent bugs were
found in the process, three of them silent. **932 tests pass, 17 skip** (all
17 need `bpy`); 0.1.4 was 868 passing.

Remaining work, prioritised and assessed, is in
[REVIEW_ROADMAP.md](REVIEW_ROADMAP.md).

### Fixed — the editing functions silently dropped `Ri`/`Gm`/`Cm`

An audit of all 32 direct `Tree(...)` constructions found that **every
node-count-changing transform built a bare `Tree(...)` and never mentioned the
electrical properties**, so they came back `None`:

    delete_tree · resample_tree · root_tree · repair_tree · elimt_tree
    redirect_tree · insert_tree · insertp_tree · cat_tree

`tree.Ri = 100; build_neuron_model(delete_tree(tree, doomed))` therefore failed
with *"tree has no Ri set"* on a tree that visibly had one, several steps away
from the function that lost it. All nine now route through
`reindexed(source).copy(...)`, where `source` names the original node each
result node derives from — information those functions already had, since they
use it for `R`.

`cat_tree` is the one genuinely ambiguous case (two trees, two sets of values):
it concatenates when both sides have a value, keeps a shared scalar, and
**warns** when only one side does rather than quietly returning `None`.

### Fixed — `build_neuron_model` built electrically disconnected cells

A section's parent is the one whose *end* node is its start node. Nothing ends
at the root, so with a multi-child root — the normal case for a TREES soma, a
short chain the root sits inside — every trunk section was parentless and the
cell came back as **two or more electrically isolated section trees**. NEURON
accepts that silently: `h.topology()` shows several roots, and a clamp on one
subtree sees nothing that happens on another.

Confirmed on the Beining granule cell: 2 parentless sections of 62. MATLAB's
`neuron_template_tree` avoids it by calling `root_tree` first, which this port
cannot (prepending a node would shift every index and break `model.loc`), so
the trunks are joined at their own `0` ends instead — NEURON's spelling of
"these all start at the same point".

Consequence for anyone who worked around it: `model.loc(soma_node)` on the true
root is correct again. On the granule cell the isolated stub fired 12 spikes to
a 0.09 nA step where the properly-loaded soma fires 3.

### Fixed — `GC_init` dropped every ion reversal potential

The `<ion>_ion` blocks were applied in the same pass as the channels, but a
segment only grows an `ek` attribute once something carrying potassium has been
inserted into it. So `ek` stayed at NEURON's default −77 mV instead of the
model's −93 mV, on a cell whose whole point is its potassium conductances. Now
a second pass, after the channels.

### Fixed — a mechanism library could only be loaded once per process

`nrn_load_dll` is not idempotent; a second load aborts with *"The user defined
name already exists: BK"*. Comparing two variants of a model in one session —
the main thing having variants is for — hit it immediately.
`GC_init(..., mech_library=)` now records what it has loaded by resolved path.

Related: the path is now `resolve()`d, because NEURON hands it to the platform
loader as-is and Windows' `LoadLibrary` does not interpret it relative to the
Python process's working directory — a relative path failed with a bare "module
not found".

### Added — `pynetrees.gc_model`: the Beining et al. (2017) granule cell

`GC_initModel.m` as one adjustable function, with everything it needs bundled
except a compiler.

| | |
|---|---|
| `GC_init(tree, mech=None, ...)` | morphology + biophysics → a live `GCModel` |
| `gc_spec(source=None, ...)` | resolve a specification; pure, needs no NEURON |
| `GCSpec` | frozen; `.scaled()`, `.overridden()`, `.without()`, `.diff()` |
| `GC_BIOPHYS` | the published table — 11 regions, 570 parameters |
| `gc_trees()` / `gc_tree()` | the eight reconstructions it was fitted to |
| `gc_mod_path()` | the 44 `.mod` sources, for `nrnivmodl` / `mknrndll` |

The split is the point. `GC_biophys.m` (656 lines) and `GC_spinedensity.m` are
a **data table written as control flow** — a chain of `strfind(options, '-x')`
tests that *computes* a nested struct — so this takes the table as data
(`_gc_biophys.py`, a separate module mirroring MATLAB's own file split) and
keeps the function for composing, scaling and applying it.

A variant is therefore a **value**, not a flag string:

```python
spec    = pt.gc_spec()                       # the published model
newborn = spec.scaled({"Kir21": 0.5, "HCN": 2.0})
passive = pt.gc_spec(passive_only=True)
spec.diff(newborn)                           # exactly what changed
```

`GCSpec.diff` answers "what does `-y` actually change?", which the original
cannot. Other details worth knowing:

- Passive parameters are folded into `Ri`/`Gm`/`Cm` **before** the sections
  exist, because NEURON's d-lambda rule reads `Ra` and `cm` to choose `nseg`.
  Setting them afterwards — the tempting order — leaves the compartment count
  computed from the wrong cable.
- A region with no passive block **raises** rather than getting `Ra = nan`,
  which NEURON accepts and turns into `nan` voltages seconds later with nothing
  pointing back to the cause.
- `GCModel.skipped` lists every `(region, mechanism, parameter)` the compiled
  library would not take; `strict=True` refuses to build at all. Without the
  model's `.mod` files NEURON knows none of its channels, and the difference
  between a model that is wrong and one that says it is wrong is the whole
  value here.
- `spine_scale` reproduces `scale_spines.m` and **refuses to run on a spec that
  already has it folded in** — a MATLAB dump made with `ostruct.scalespines = 1`
  does, which `adendMML.pas.cm = 1.71 = 0.9 × 1.9` gives away.

`GC_BIOPHYS` was generated once from a dump of `neuron.mech{1}` and is verified
exact against it (570 parameters) by a test that skips when the dump is not
checked out. Compiled mechanisms are deliberately **not** bundled: ~23 MB, and
valid only for the exact NEURON version and architecture they were built
against.

`ostruct.usemorph` — MATLAB's catalogue of 16 hard-coded `.mtr` paths — is not
ported. `load_tree` reads all of them, and a number that means "SH07 cells"
only in one lab's directory layout is not an API.

### Changed — `MST_tree` now reproduces MATLAB exactly, multi-tree included

Verified against MATLAB R2026a on four fixed clouds (150 points in 2-D, 200 in
3-D, 300 shared between three seeds, one with `mplen` binding), compared
edge-by-edge at matched coordinates.

| Case | 0.1.4 | 0.2.0 |
|---|---|---|
| single 2-D / 3-D / `mplen` | Jaccard **1.000** | unchanged |
| three competing seeds | 0.911 — sizes 94/124/85 | **1.000** — sizes **101/101/101**, same cable to 0.1 µm |

MATLAB's growth loop is

    while something_was_added
        for counter = 1 : n_trees        % strictly in index order
            take THIS tree's own cheapest available connection

so the trees **take turns**, one connection each per sweep. The port had a
single global heap popping the absolute cheapest connection anywhere, which
lets a tree in a denser patch run ahead and claim territory before its
neighbours have had a turn — and the territories are the reason multi-tree
growth exists. Now one heap per tree with `best_cost` per tree (two trees can
hold different best offers for the same point), swept round-robin.

**This changes multi-tree output.** Single-tree growth is unaffected: one tree
means one turn per sweep.

### Added — `Tree.meta`, ad-hoc data that survives the transforms

The Python stand-in for tacking an extra field onto MATLAB's tree struct, which
`__slots__` otherwise forbids outright. Entries whose length is `n_nodes` are
**reordered and subset with the nodes** by `reindexed` — and so by every
transform built on it; anything else is treated as whole-cell metadata and
copied across. Length is the only available signal, and MATLAB's own
`delete_tree`/`resample_tree` make the same length-based decision.

`save_npz` round-trips it (arrays as npz members, the rest as one JSON blob, so
`allow_pickle=False` still holds) — and now round-trips `Ri`/`Gm`/`Cm` too,
which the module docstring had been claiming for some time without doing.

### Added — populations nest to any depth

`len_tree` on a four-deep list returns a four-deep list with indices lining up
at every level, and nothing is flattened. A caller who organised cells into
groups and subgroups gets those groups back.

A *ragged* level (`[tree, [tree]]`) now **raises** rather than being guessed at:
it could be a two-cell group whose second cell was wrapped by mistake, or a
group of groups whose first entry was not, and guessing turns a typo into a
silently mis-shaped result.

### Added — NEURON: node-index targeting and distance-dependent densities

```python
# two parts of the same region, different densities, no morphology surgery
pt.insert_mechanism(model, "hh", nodes=distal_tips, gnabar=0.2)

# a gradient, measured along the real cable by h.distance
pt.insert_mechanism(model, "hh", region="adendOML",
                    gnabar=lambda x: 0.12 * np.exp(-x / 200.0))
```

Given both `region=` and `nodes=`, the two are intersected. A `GLOBAL`
parameter raises rather than being set per segment, where only the last value
would survive.

### Added — `path_to_root(tree, node)`

The review asked for the root to stop being looked up through a `-1` sentinel.
An audit found the **library itself is clean** — every internal `v[idpar]` uses
`root_self=True`, and every `root_self=False` call tests `!= NO_PARENT` — so
the hazard is in user code, where

```python
while par[x] > 0:            # WRONG in two ways
```

silently drops the root whenever the root is node 0 (which it is after
`sort_tree`), and `v[parent[root]]` reads the **last** node under NumPy's
negative indexing and returns a plausible number rather than raising.
`path_to_root` is the operation people were hand-writing; `idpar_tree`'s
docstring now states the hazard outright.

This turned up a real off-by-one in `scripts/dendritic_democracy.ipynb`, whose
hand-written walk stopped one node short of the soma. MATLAB's
`while par(x) ~= 0` **includes** the root, so the port had been walking a
shorter path than the original all along.

### Added — `missing_dependencies()` and a one-time import warning

Lazy imports keep `import pynetrees` cheap, at the cost of a missing dependency
surfacing as an `ImportError` deep inside the first call that needed it, named
after the transitive dependency rather than the feature. One `UserWarning` at
import now names every unavailable feature and the extra that fixes it. Silence
it with `PYNETREES_NO_DEPENDENCY_WARNING=1`.

`UserWarning`, not `ImportWarning`: Python's default filters *ignore* the
latter, so the category that reads as right would make it invisible to exactly
the people it is for.

### Changed — breaking

| Was | Is | Why |
|---|---|---|
| `Tree.with_coords(X=, Y=, Z=, D=, name=)` | `Tree.copy(**fields)` | Carries *every* field it was not asked to change, so a transform cannot silently lose one it never thought about — which is exactly how `Ri`/`Gm`/`Cm` came to be dropped. Raises `TypeError` on an unknown field name, catching `tree.copy(Xs=...)` at the call instead of returning an unmodified tree. |
| `dissect_tree(with_positions=)` | `dissect_tree(full_output=)` | The one function using a different name for "also return MATLAB's second output"; `MST_tree`, `insert_tree`, `insertp_tree`, `redirect_tree`, `sort_tree`, `spines_tree`, `dscam_tree` and `spread_tree` already said `full_output`. |
| `MST_tree(start=[a, b, c])` | same call, different territories | See the round-robin change above. |

`sub_tree(with_tree=)` deliberately keeps its name: it is a *cost* flag (skip
building the subtree in a hot loop), not an output-shape flag, and the result
is a `SubTree` either way.

### Documentation and examples

- **[examples/08_granule_cell_model.ipynb](examples/08_granule_cell_model.ipynb),
  new** — the `GC_init` tutorial, executed end to end so every number in it is
  real. Three parts with different floors: resolving a specification needs only
  `pynetrees`, building one adds `neuron`, and running the active model needs
  the bundled `.mod` files compiled. References no path outside the package.
- **[examples/04_neuron_simulation.ipynb](examples/04_neuron_simulation.ipynb)** —
  its "with a granule cell you would write…" placeholder replaced by the two
  capabilities that now exist. Re-executed.
- **[scripts/dendritic_democracy.ipynb](scripts/dendritic_democracy.ipynb)** —
  reads back at the soma again instead of `dendIDs[-1]`, and uses
  `path_to_root`. **Its stored outputs predate the connectivity fix and need a
  re-run**; a banner at the top says so.
- **[REVIEW_ROADMAP.md](REVIEW_ROADMAP.md), new** — everything left from the
  code review, ordered, with an assessment of each item rather than a bare list.
- `docs/api-overview.md` and the generated `docs/FUNCTION_REFERENCE.md` cover
  the new names; both are checked against `pynetrees.__all__` by
  `tests/test_docs_use_the_real_api.py` in both directions.

### Packaging

- **`ipykernel` pinned below 7.** 7.x changes the startup/debug handshake and
  intermittently leaves cells stuck on "connecting" in VS Code's notebook UI.
  Jupyter Lab/Notebook are fine with 7.x.
- `nbformat`/`nbclient` added to the `dev` extra — regenerating the executed
  outputs in `examples/` after an API change is a real maintenance task.
- Package data now includes `data/gc.mtr` and `data/gc_mod/*.mod`.
- `.gitignore` excludes compiled NEURON mechanisms (`*.dll`, `*.o`,
  `lib_mech/`, `x86_64/`, `arm64/`): rebuilt by `nrnivmodl`, and specific to one
  NEURON version and architecture.

### Known limitations carried into this release

- **Blender export is complete but untestable alongside numpy 2.**
  `bpy==5.0.1`, the newest release, still requires `numpy<2.0,>=1.26` and
  `python==3.11.*`. `pynetrees.blender` works; it needs its own environment,
  which is why 17 tests skip.
- **`boundary_tree`'s shrink → alpha mapping drifts.** Measured against
  MATLAB's `boundary()`: exact at shrink 0, ~2% low at the default 0.5, and
  16.7% low at shrink ≈0.23 on a concave arbor. `dissectSholl_tree.V` inherits
  this directly. See REVIEW_ROADMAP.md §1.1.
- **`.mtr` writing is MATLAB v5**, which caps file size; a large morphology
  cannot yet round-trip. See REVIEW_ROADMAP.md §1.3.

---

## 0.1.4 — 2026-08-26

Phase 12 and the first two review rounds. Consolidated here from
`PORT_STATUS.md`, which has the per-function detail.

- Generative pipeline: `clone_tree`, `dscam_tree`, `spines_tree`,
  `gscale_tree`, `rpoints_tree`, `PP_generator_tree`.
- Topological description: `barcode_tree`, `persistenceimage_tree`,
  `realisations_tree`, `BLO_tree`.
- Density and space-filling: `boundary_tree`, `convexity_tree`, `gdens_tree`,
  `hull_tree`, `vhull_tree`, `lego_tree`, `span_tree`, `theta_tree`,
  `theta_mc_tree`, `r_mc_tree`, `share_boundary_tree`.
- Image stacks: `load_stack`, `skeletonize_stack`, `fitD_stack` and friends —
  3-D skeletonisation via scikit-image rather than MATLAB's hand-rolled
  thinning.
- `MST_tree`'s deferred options: multi-tree growth, `DIST`, grow-from-cut-ends,
  growth history.
- V1 API cleanup: eight deprecated spellings removed
  (`plot_tree_mpl` → `plot_mpl_tree`, `spread_trees`, `idpar_tree(no_self=)` →
  `root_self=`, `elimt_tree(no_root=)` → `at_root=`, the `dim2=`/`dim="2d"`
  spellings), and `tests/test_docs_use_the_real_api.py` added because the
  README and three notebooks had gone on advertising them.
- MATLAB differential testing against R2026a for the functions Octave could not
  run: `boundary_tree`, `convexity_tree`, `r_ms_tree`, `dissectSholl_tree`,
  `growth_tree`.

## 0.1.3 and earlier

Phases 1–11: the `Tree` structure and `ver_tree`; graph primitives; metrics and
geometry; editing and repair; construction (`MST_tree`, `BCT_tree`,
`soma_tree`, `quaddiameter_tree`); I/O across `.swc`, `.mtr`, `.neu`, `.nmf`,
Neurolucida `.asc`, NeuroML, `.hoc`/`.nrn`; plotting; electrotonics
(`sse_tree`, `syn_tree`, `LIF_tree`, …); statistics (`sholl_tree`,
`stats_tree`, `vonMises_tree`, `peters_tree`); and the NEURON bridge.

See [PORT_STATUS.md](PORT_STATUS.md) for the phase-by-phase record and the
numbered design decisions, and [PORT_PLAN.md](PORT_PLAN.md) for the reasoning
behind the phase order.
