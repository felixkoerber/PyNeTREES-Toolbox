# What is not yet ported

A complete inventory across the three MATLAB codebases in this repo, gathered
by enumerating every `.m` file and checking it against `pynetrees.__all__` and
`gc_model`. Counts are measured, not estimated.

| Codebase | Functions | Ported | Not ported |
|---|---|---|---|
| `treestoolbox-master` | 160 | **153 (96%)** | 7 (~6,800 LoC) |
| `Active GC Model` | 184 `.m` + 9 `.mlx` | ~4 | ~180 |
| `Pattern_separation_toolbox-main` | 32 | 22 | 10 |
| `T2N-master` | 52 | 0 (superseded) | — |

**Work packages A, B1-B5 are all closed.** `edit/` (13), `electrotonics/`
(14), `graphtheory/` (24), `metrics/` (29), `graphical/` (15) and `IO/`
(10 of 12) are complete, verified by enumerating the `.m` files and matching
them against the port's defined names rather than by eye.

What remains, none of it analysis:

| Group | LoC | Status |
|---|---|---|
| `stacks/` — 8 functions | 1,005 | **CLOSED** (#65) — `pynetrees/stacks.py`; seven of eight delegated to `tifffile`/`imageio`/`scikit-image`, see [GUI_AND_STACKS.md](GUI_AND_STACKS.md) |
| `pov_tree`, `x3d_tree` | 1,341 | **PLANNED** (V6) — `pynetrees/blender.py` (#66) covers the same need without an external ray-tracer, but a `.pov`/`.x3d` file is a portable artefact needing no 300 MB dependency, and both were asked for in full |
| `fix_tree` / `fix_tree_UI` / `finetune_fix_tree` | 5,047 | a MATLAB figure-callback GUI plus a headless core |
| `classes/+trees/Tree.m`, `Trees.m` | 1,700 | MATLAB OOP shims giving method syntax over the same functions — `pynetrees.Tree` **is** this, natively |
| `colorme` | 33 | a colour-cycle helper |

**`cgui_tree` (9,022 lines) is not going to be ported**, and
[GUI_AND_STACKS.md](GUI_AND_STACKS.md) works through why: sorting its twelve
panels by what the user actually gets, every capability except
point-and-click morphology editing is already in this port, and napari ships
that as a core layer type. The deliverable there is a napari plugin in a
separate distribution, not a work package here.

Deliberately skipped with the reasoning recorded: `cpoints`/`cplotter`
(unpack MATLAB's packed `contourc` format, which this port never produces),
`dstats_tree` (a figure, not an analysis), `tlen_tree` (`Tree.total_length`),
`start_trees` (MATLAB path setup), plus `utilities/` and `gui/` in full.

Three categories below, and they matter in different ways:

- **[A. Silent gaps](#a-silent-gaps-ported-functions-missing-matlab-options)** —
  functions you *can* call that quietly do less than MATLAB's. The most
  likely to surprise you, and the highest priority.
- **[B. Missing functions worth porting](#b-missing-functions-worth-porting)**
- **[C. Deliberately out of scope](#c-deliberately-out-of-scope)** — with the
  reason in each case, so the decision can be revisited rather than
  rediscovered.

---

## A. Silent gaps: ported functions missing MATLAB options

These are the dangerous ones. The function exists, the call succeeds, and the
result is narrower than the MATLAB equivalent — with nothing at the call site
to say so.

**Status: the seven marked DONE below were closed in W3** (Design Decisions
#54–#58), each verified against the MATLAB source running in Octave where a
reference could be obtained.

| Function | Missing | Consequence |
|---|---|---|
| `resample_tree` | MATLAB's whole **snapping method**, plus `'-l'` length conservation, `'-d'` diameter interpolation, `'-b'`, `'-v'` | **DONE** (#54) — MATLAB's method is now the default, verified differentially |
| `MST_tree` | Multi-tree competitive growth, `DIST` cost matrix, `'-c'` grow-from-cut-ends, `'-t'` time-lapse | **DONE** (#58) — multi-tree, `dist`, cut-ends, time-lapse, `indx` |
| `rot_tree` | `'-pcaX/Y/Z'`, `'-m3dX/Y/Z'`, `'-al'` | **DONE** (#56) — all six alignment modes, verified to 1e-13 |
| `plot_tree` | `color` as scalar-vector / N×3 matrix; MATLAB's argument order; `'-b'`/`'-p'`/`'-2q'`/`'-3q'` render modes | **DONE** (#55) — `color` merged, MATLAB argument order |
| `stats_tree` | `parea`/`mparea` density statistics | **DONE** (#59) — unblocked by B1 |
| `sholl_tree` | `'-s'`/`'-s3'` intersection plotting | Analysis only; plotting to be a separate `plot_sholl` |
| `dissect_tree` | 2nd output: per-node section index + relative position | **DONE** (#57) — `with_positions=True` |
| `ipar_tree` | `'-T'` terminal-to-first-branch-point paths | **DONE** (#57) — `terminals_only=True`, 28x smaller |
| `soma_tree` | `'-b'` overlap correction (√2 diameter reduction past a branch point) | **DONE** (#57) — `overlap_correction=True` (MATLAB's crashes) |
| `cap_tree` | `'-a'` add-axon | Deliberate: dataset-specific constants; would become `add_axon_tree` |
| `cyl_tree` | `'-dA'` sparse output form | MATLAB's own comment says "SLOW!!"; nothing uses it |
| `load_neurolucida` | Soma-contour cylinder fitting, markers, concatenation to nearest soma | A multi-block `.asc` loads as **several disconnected fragments** rather than one cell |
| `asym_tree` | `'-m'` movie | MATLAB's was buggy per its own todo list |
| `insertp_tree` | `'-p'`/`'-pr'` | Don't exist in MATLAB either |

**Also missing: one `load_tree` front door.** `load_mtr`, `load_swc`,
`load_neurolucida` and `load_tree` are separate entry points; MATLAB
dispatches on extension and offers a file dialog. Planned as W2's remaining
item, together with `.neu`/`.nmf` and `save_tree(matlab_format=True)`.

---

## B. Missing functions worth porting

### B1. Density/hull machinery — **CLOSED** — 5 functions, one shared dependency

The single biggest *coherent* gap. All five need the same 3D binning +
isosurface extraction, so they are one piece of work, not five.

| Function | LoC | What it does |
|---|---|---|
| `hull_tree` | 274 | **DONE** (#59) — space-filling isosurface at a threshold distance |
| `vhull_tree` | 213 | **DONE** (#59) — unbounded cells return NaN rather than being dropped |
| `gdens_tree` | 117 | **DONE** (#59) — indexed `[x, y, z]`, not MATLAB's `[y, x, z]` |
| `lego_tree` | 167 | **DONE** (#59) |
| `share_boundary_tree` | 249 | **DONE** (#59) |

**Unblocks** `stats_tree`'s `parea`/`mparea`. Needs `scikit-image` (marching
cubes) as an optional dependency; 2D can use matplotlib's contour.

### B2. Statistics and metrics — **CLOSED** — 7 functions

| Function | LoC | Notes |
|---|---|---|
| `dissectSholl_tree` | 476 | **DONE** (#61) — one ray-cast per direction replaces 25M point-in-mesh tests; not MATLAB-verifiable (no Octave `boundary`) |
| `dstats_tree` | 370 | **Skipped deliberately** — a figure, not an analysis; `stats_tree` returns DataFrames that plot directly |
| `convexity_tree` | 176 | **DONE** (#60) — deliberately diverges; MATLAB's 3D branch is sign-inverted against its own 2D branch |
| `r_mc_tree` | 199 | **DONE** (#61) — Clark-Evans spatial-randomness test (nothing to do with regions); exact simplex sampling, no rejection loop |
| `boundary_tree` | 83 | **DONE** (#60, reworked in #61) — returns a `Boundary` with volume, filled simplices and an ordered 2D polygon |
| `M_atten_tree` | 31 | **DONE** (#60) — closes `electrotonics/`; MATLAB ships it with no documentation at all |
| `tlen_tree` | 21 | **Skipped deliberately** — `Tree.total_length` covers it |

`angleBd_tree` / `angleBd2_tree` (104/109 LoC, `new-functions/`) compute
branch angles at variable distance rather than at the branch point itself —
a genuinely different measurement from the ported `angleB_tree`. **DONE**
(#60).

With B1 and B2 closed, `treestoolbox-master/metrics/` (30 files),
`graphtheory/` (25) and `electrotonics/` (14) are fully ported apart from
three files that are not functions to port: each folder's `Contents.m`
index, `graphtheory/start_trees.m` (the toolbox's path setup — `pip
install` is the Python equivalent), and `metrics/dstats_tree.m`, skipped
deliberately above. Verified by enumerating the `.m` files and matching
them against the port's defined names, not by eye.

What remains below is I/O formats (B3), the generative pipeline (B4), two
plot helpers (B5), and the utilities/GUI folders that have no Python
purpose.

### B3. I/O — **CLOSED**

| Function | LoC | Notes |
|---|---|---|
| `neuron_tree` | 547 | **DONE** (#62) — `save_hoc(style='cell')` and `save_nrn`; MATLAB's `.nrn` branch does not run at all |
| `neuron_template_tree` | 365 | **DONE** (#62) — `save_hoc(style='template')`; `minterf` split out as `t2n_interface` |
| `nmf_tree` | 92 | **DONE** (#62) — `save_nmf`/`load_nmf`; region names now survive the round trip |
| `neuroml_tree` | 169 | **DONE** (#62) — `save_neuroml`, v1 and v2, with `<segmentGroup>` per region |

Plus the **`.neu` reader** (a branch of `load_tree.m`) — **DONE** (#62),
and verified bit-identical to MATLAB's own parent-index arithmetic on all
three shipped fixtures; MATLAB's version crashes on one of them.

`load_tree`/`save_tree` are now extension dispatchers covering `.npz`,
`.mtr`/`.mat`, `.swc`, `.neu`, `.nmf`, `.asc` for reading and `.npz`,
`.mtr`, `.swc`, `.nmf`, `.hoc`, `.nrn`, `.xml` for writing. A `.mtr`
written here loads through MATLAB's own `load_tree`, verified under Octave.

Not ported from `IO/` **yet**: `pov_tree` (POV-Ray) and `x3d_tree` (X3D),
both scheduled as V6 -- native Blender support (#66) covers the same
need, but does not replace a portable scene file. Not ported at all:
`start_trees.m`, which is MATLAB path setup, and the GUI file dialogs,
see #62.

### B4. Generative pipeline — **CLOSED**

| Function | LoC | Notes |
|---|---|---|
| `gscale_tree` | 417 | **DONE** (#63) — returns `RegionSpan` objects by name, not 15 parallel cell arrays |
| `clone_tree` | 454 | **DONE** (#63) — needed a new `MST_tree` mode: grow onto an existing tree |
| `rpoints_tree` | 226 | **DONE** (#63) — whole batch in one `searchsorted` instead of a per-point loop |
| `PP_generator_tree` | 385 | **DONE** (#63) — plus `max_iter`; MATLAB's search loop is unbounded |
| `in_c` | 76 | **DONE** (#63) — as `in_hull`, on polygon lists rather than packed contour matrices |
| `cpoints`, `cplotter` | 109 | **Skipped deliberately** — pure unpacking of MATLAB's `contourc` format, which this port never produces |
| `dscam_tree` | 115 | **DONE** (#63) |
| `spines_tree` | 186 | **DONE** (#63) — four MATLAB bugs fixed, see MATLAB_TOOLBOX_BUGS.md |

Still open from `construct/`: `fix_tree` (1000), `fix_tree_UI` (3672) and
`finetune_fix_tree` (375) — interactive reconstruction repair. `fix_tree_UI`
is a MATLAB figure-callback GUI and does not port; `fix_tree` has a headless
core worth extracting, and is grouped with the GUI question below.

### B5. Small graphical — **CLOSED**

| Function | LoC | Notes |
|---|---|---|
| `plotsect_tree` | 72 | **DONE** (#64) — raises where MATLAB silently draws an empty line |
| `xplore_tree` | 123 | **DONE** (#64) — region labels fixed; arrow overlay dropped, see #64 |

---

## C. Deliberately out of scope

Recorded so the reasoning survives, not to close the question.

| Group | LoC | Why |
|---|---|---|
| **`gui/` — `cgui_tree` et al.** | 9,009 | A full MATLAB GUI application. A Python equivalent would be a rewrite against a different toolkit, not a port, and none of the analysis depends on it |
| **`stacks/` — 8 functions** | 1,005 | Image-stack loading, skeletonisation, diameter fitting. This is image processing, well covered by `scikit-image`/`tifffile`; porting MATLAB's versions adds little |
| **`utilities/` — 11 unported of 18** | 1,274 | MATLAB-language plumbing with direct Python equivalents: `parseArgs`→keyword arguments, `deg2rad`→`np.radians`, `isBinary`→`bool`, `eucdist`→`scipy.spatial.distance`, `tprint`/`gifmaker`/`scalebar`/`shine`/`hotcold`→matplotlib. Only `rotation_matrix` and `gauss` had real content; both are ported privately |
| ~~**`pov_tree` + `pov_patch`**~~ | 1,560 | Was declined as "a rendering pipeline for a specific external ray-tracer"; scheduled as V6 after all, show-file variants included |
| ~~**`x3d_tree`**~~ | 251 | X3D mesh export for external 3D viewers; scheduled as V6 |
| **`fix_tree` / `finetune_fix_tree` / `fix_tree_UI`** | 5,047 | MATLAB's own todo list flags these as incomplete |
| ~~**`plotsect_tree`**~~ | 72 | Was listed here on the maintainers' "has no options to begin with" note; ported in B5 anyway (#64) since it is 30 lines with a real keyword signature |

`utilities` and `gui` alone account for **10,283 of the 23,601 unported
lines** — 44% of the apparent gap is code with no Python purpose.

---

## D. `Active GC Model` — the big one

`gc_model/` currently ports the *demonstration* layer: dendritic democracy,
the pattern-separation toolbox, mechanism compilation and parallel execution.
The model itself is largely untouched.

**The blocker, unchanged:** `GC_biophys.m` (~656 lines) is the channel
configuration matrix — which conductance at what density in which region,
loading 18-parameter Markov rate tables from `soma_st8.txt`/`axon_st8.txt`.
Everything active depends on it. All 44 NMODL mechanisms **are** compiled and
loadable, so this is a data/configuration port, not a blocked one.

Roughly, what is there:

| Group | Files | Status |
|---|---|---|
| `GC_ATP_pattern_*` (pattern-separation experiments) | ~22 | not ported |
| `aGC_*` (physiology validation: AHP, FI curves, Ca dynamics, EPSC) | ~24 | not ported |
| `GC_biophys` / `GC_initModel` / `GC_model2` / `t2n_setionchannels` | 4 | **not ported — the blocker** |
| `GC_dendritic_demo*` | ~7 | **ported** (`gcmodel/democracy.py`) |
| `TM_fit*`, `quantal_TM`, `GC_STP_*` (short-term plasticity) | ~8 | not ported |
| `Population_models`, `Pareto`, `makeNeurons`, `gradientwalk_*` | ~15 | not ported |
| Plotting/analysis helpers (`doChannelPlots`, `plot_cells`, `violinplot`, …) | ~20 | not ported |

### Pattern separation toolbox — 22/32

Missing, in rough priority order:

- **`temporally_correlated`** — the 4th spike-ensemble generator; the other
  three are ported. Small, and its absence is asymmetric.
- **`TE_function` / `compute_TE` / `joint_distribution_TE`** — transfer
  entropy. MI and redundancy reduction are ported; TE is the third measure.
- **`spike_thinner_nth` / `_temporal` / `_spatiotemporal`**,
  `spike_expansion_random` — only `spike_thinner_random` is ported.
- `bin_outputs_delay`, `all_rate_codes`, `plot_raster` — minor.

---

## E. `T2N-master` — superseded, not ported

52 `.m` files. T2N's purpose is to drive NEURON *from MATLAB* by generating
`.hoc` files and shelling out to `nrniv`. `pynetrees.neuron_bridge` talks to
NEURON directly through its Python API, so the file-generation and
process-management layers have no counterpart to port — the need they serve
does not exist here. `t2n_setionchannels.m` is the exception: it belongs with
`GC_biophys` in §D.

---

## Suggested order

Items 1-4 and 6 of the original list are done -- W3's silent gaps, the
`load_tree` dispatcher with `.neu`/`.nmf`, B1's density/hull machinery, W5
(since superseded by V3's list-in/list-out rule, Design Decision #69) and
B4's generative pipeline. `M_atten_tree`, then the only unported function
in an otherwise complete `electrotonics`, is ported too. **V4 is done as
well** (Design Decision #70): all twelve functions added to the MATLAB
toolbox on 2026-08-26 are ported.

What is left, in order (`REVIEW_PLAN_2.md` carries the detail):

1. **V5 -- I/O**: v7.3 writing to lift `save_tree`'s 2 GB ceiling, and
   `load_mtr`'s variable-selection rule.
2. **V6 -- `pov_tree`, `x3d_tree`, `cyl_tree -dA`**: POV-Ray and X3D export
   in full, show-file variants included. Wanted alongside
   `pynetrees.blender`, not instead of it -- a `.pov`/`.x3d` file is a
   portable artefact that needs no 300 MB dependency to produce.
3. **V7 -- renames and small fixes**, cheap and better done once the above
   settle the signatures.
4. **`GC_biophys`** -- the Active GC Model blocker. Substantial, and worth
   scoping separately.
