# Port Status

Living tracker for the Python port. See [PORT_PLAN.md](PORT_PLAN.md) for the reasoning
behind the phase order and the design decisions summarized here.

Status values: `done` · `in-progress` · `not-started` · `deferred` (intentionally later)
· `wont-port` (superseded by a Python-native replacement, see note).

---

## Phase 0 — Project scaffolding

| Item | Status | Notes |
|---|---|---|
| `pyproject.toml` (package `pytrees`, src layout) | done | numpy/scipy core deps; pytest dev dep |
| `src/pytrees/__init__.py` | done | |
| `tests/` + pytest config | done | |
| Fixture data copied from MATLAB `sample/swc`, `tests/IO/test files` | done | `25HSS.swc`, `test02.swc` |

## Phase 1 — Core data structure, validation, minimal I/O

| Item (MATLAB origin) | Status | Notes |
|---|---|---|
| `Tree` class (tree struct) | done | `src/pytrees/core.py`; 0-based indices, `-1` no-parent sentinel (see Design Decisions) |
| `ver_tree` → `ver_tree()` / `Tree.validate()` | done | non-raising, warning-collecting, matches MATLAB semantics |
| `load_tree`/`swc_tree` (read) → `load_swc()` | done | handles unsorted/non-contiguous indices and multi-root files (`test02.swc`); does **not** yet call `repair_tree` (deferred — Phase 4) |
| `swc_tree` (write) → `save_swc()` | done | minimal writer, round-trips `load_swc` |
| `sample_tree` → `sample_tree()` | done | loads bundled `25HSS.swc` fixture |
| Test suite (12 tests, `pytest`) | done | `Tree`/`ver_tree` checks, single-root + multi-root + unsorted-index SWC loading, write/read round-trip, malformed-file rejection — all green |
| `repair_tree` full port (elimt/elim0/sort) | deferred | Phase 4 |
| `.mtr` (MATLAB binary) support | done | landed post-Phase-7 as `src/pytrees/io/mtr.py`'s `load_mtr`, once the bundled `Active GC Model/morphos/*.mtr` granule-cell reconstructions gave a concrete reason (see Design Decision #32). MATLAB v5 (`scipy.io.loadmat`) only -- v7.3/HDF5 `.mtr` files raise a clear error naming the file, rather than being silently unsupported |
| `.neu` / `.nmf` support | deferred | niche formats (NEURON hoc transfer, HDF5 swc-extension); lower priority than `.swc`/NeuroLucida |

## Phase 2 — Graph primitives (`dA` only)

`src/pytrees/graphtheory.py`, tested in `tests/test_graphtheory.py` (36 tests total in
the suite, including a full smoke-test of every function below against the real
2252-node bundled reconstruction — see Design Decisions for details on what that
surfaced).

| MATLAB function | Status | Notes |
|---|---|---|
| `idpar_tree` | done | `no_self` kwarg replaces `'-z'`; root detected via row-sum (general, not position-dependent) |
| `idchild_tree` | done | output width is dynamic (max children actually found), not MATLAB's hardcoded 2 |
| `ipar_tree` | done | rewritten around the `-1` sentinel instead of MATLAB's ambiguous `0`-padding (see Design Decisions) |
| `child_tree` | done | |
| `B_tree` | done | branch points |
| `T_tree` | done | termination points |
| `C_tree` | done | continuation points |
| `sub_tree` | done | returns boolean mask only; the MATLAB second output (`subtree` struct) deferred — trivial to add via `tree.reindexed(np.flatnonzero(mask))` once a caller needs it |
| `dissect_tree` | done | landed in Phase 4 (needed `root_tree`) — see "Phase 2/3 backlog items unblocked by Phase 4" below |
| `dist_tree` | done | landed in Phase 4, placed in `metrics.py` — see below |
| `BO_tree` | done | branch order |
| `LO_tree` | done | ported faithfully as the literal sparse matrix-matrix recursion (see Design Decisions — deliberately *not* re-derived) |
| `PL_tree` | done | topological path length; root found generically, not assumed at index 0 |
| `Pvec_tree` | done | no default `v` yet (MATLAB defaults to `len_tree`, Phase 3) — `v` is a required argument for now |
| `ratio_tree` | done | default `v=tree.D` (available since Phase 1) |
| `redirect_tree` | done | reimplemented via plain BFS rerooting instead of MATLAB's undirected-adjacency matrix-power walk (see Design Decisions) |
| `rindex_tree` | done | rank is 0-based (first node in a region is rank 0), not MATLAB's 1-based |
| `sort_tree` | done | `by="hier"/"lo"/"lex"` kwarg replaces `'-LO'`/`'-LEX'` flags; default mode reimplemented as DFS pre-order (see Design Decisions) |
| `strahler_tree` | done | reimplemented as a standard post-order traversal instead of MATLAB's queue/fixed-point loop; also handles multifurcations (MATLAB assumes binary) |
| `typeN_tree` | done | |
| `gene_tree` | done | landed in Phase 4, placed in `metrics.py` — see below |
| `asym_tree` | done | `van_pelt` kwarg replaces `'-v'`; `'-m'` movie option dropped (Phase 7 concern, and MATLAB's was buggy per todo list); raises (rather than silently mis-measuring) on non-binary branch points |
| `bin_tree` | done | landed in Phase 4, placed in `metrics.py` — see below |

## Phase 3 — Coordinate-based metrics

`src/pytrees/metrics.py`, tested in `tests/test_metrics.py` (59 tests total in the
suite now, plus a manual smoke-test of every function against the real 2252-node
bundled reconstruction — see Design Decisions #17/#18 for what that surfaced).

| MATLAB function | Status | Notes |
|---|---|---|
| `len_tree` | done | `dim2` kwarg replaces `'-dim2'` |
| `eucl_tree` | done | single `dim` kwarg (2 or 3) replaces separate `'-dim2'`/`'-dim3'` flags; `point` may be a node index or an explicit coordinate |
| `surf_tree` | done | frustum mode driven by new explicit `Tree.frustum` attribute (see Design Decisions #17), not a dynamically-added MATLAB struct field |
| `vol_tree` | done | ditto |
| `cvol_tree` | done | ditto |
| `cyl_tree` | done | the `'-dA'` sparse-matrix output form dropped — MATLAB's own comment calls it "SLOW!!" and nothing else in the toolbox uses it |
| `angleB_tree` | done | raises on non-binary branch points instead of silently reading `BB(1)`/`BB(2)` (same reasoning as Phase 2's `asym_tree`, Design Decision #15) |
| `direction_tree` | done | `normalize` kwarg replaces `'-n'` |
| `rootangle_tree` | done | landed in Phase 4, placed in `edit.py` — see "Phase 2/3 backlog items unblocked by Phase 4" |
| `scale_tree` | done | `center`/`scale_diameter` kwargs replace `'-o'`/`'-d'`; always returns a new Tree (Design Decision #19) |
| `tran_tree` | done | `offset` overloads node-index-to-recenter-on vs. explicit vector, matching MATLAB's own `DD` overload |
| `rot_tree` | done | only the plain degree-based rotation is ported; MATLAB's PCA/`'-m3d'`/`'-al'` auto-alignment modes are not (see below) |
| `flip_tree` | done | `axis` kwarg (`'x'`/`'y'`/`'z'`) replaces the `dim` 1/2/3 magic numbers |
| `flatten_tree` | done | rewritten to use a precomputed `ipar_tree`-based descendant lookup instead of a fresh `sub_tree` BFS per node — see Design Decision #18 (100x speedup on the real reconstruction, same result) |
| `morph_tree` | done | same descendant-lookup fix as `flatten_tree` |
| `abel_tree` | done | landed in Phase 4, placed in `edit.py` — see "Phase 2/3 backlog items unblocked by Phase 4" |
| `zcorr_tree` | done | loop is over flagged jump nodes only (typically a small fraction of all nodes), so the `sub_tree`-per-node cost that motivated the `flatten_tree`/`morph_tree` fix doesn't apply here |
| `rot_tree` PCA / `'-m3d'` / `'-al'` modes | deferred | needs PCA (easy) + iterative region-plane alignment via SVD (fiddly, region-name-dependent); niche compared to the plain degree-based rotation — revisit if a concrete use case shows up |

## Phase 4 — Edit operations

`src/pytrees/edit.py`, tested in `tests/test_edit.py` (96 tests total in the suite
now, plus a timed smoke-test of every function below against the real 2252-node
bundled reconstruction — all well under a second; see Design Decisions #21-24).
This phase also unblocked four Phase 2/3 backlog items — see those phases' tables.

| MATLAB function | Status | Notes |
|---|---|---|
| `repair_tree` | done | the headline item deferred from Phase 1; `load_swc` output can now be brought to full BCT conformity |
| `elim0_tree` | done | |
| `elimt_tree` | done | reimplemented on explicit edge lists rather than MATLAB's growing-sparse-matrix-in-a-loop; verified by hand-trace against MATLAB's index arithmetic (Design Decision #21) |
| `delete_tree` | done | always splices to the nearest surviving ancestor and always returns a `list[Tree]` when deletion disconnects the tree — MATLAB's version is inconsistent here and the todo list documents it as broken for the default case ("delete_tree \| multiple trees doesn't work yet") (Design Decision #22) |
| `insert_tree` | done | dropped MATLAB's `[inode R X Y Z D idpar]` SWC-tuple calling convention for explicit `X, Y, Z, D, parent` arrays |
| `insertp_tree` | done | `'-p'`/`'-pr'` options dropped — todo list already notes they don't exist even in MATLAB |
| `interpd_tree` | done | |
| `resample_tree` | done | **not** a literal port — preserves branch/termination points exactly instead of relocating them onto the resampling grid via MATLAB's delete-and-splice pass; see Design Decision #23 (this is also where a real bug was caught and fixed: node count was going *up*, not down, until the section-level (`dissect_tree`) rewrite) |
| `restrain_tree` | done | |
| `root_tree` | done | unblocked `dissect_tree` (Phase 2) |
| `recon_tree` | done | |
| `cat_tree` | done | |

## Phase 2/3 backlog items unblocked by Phase 4

| Function | Placed in | Status | Notes |
|---|---|---|---|
| `dissect_tree` | `graphtheory.py` | done | reimplemented as a direct per-cut-point ancestor walk, not MATLAB's `ipar`/`cumsum` trick (which its own docstring flags as "isn't completely correct yet at the root") — this version needs no `root_tree` workaround. **Real bug found and fixed in Phase 11**: `by_region`'s cut point was placed on the region-transition node itself instead of its parent, producing a spurious extra section split at every region boundary — see Design Decision #36 |
| `dist_tree` | `metrics.py` | done | moved out of `graphtheory.py` to avoid a circular import (needs `len_tree`) |
| `bin_tree` | `metrics.py` | done | ditto (needs `eucl_tree`'s default) |
| `gene_tree` | `metrics.py` | done | ditto (needs `len_tree`); operates on one tree at a time — MATLAB's nested-cell-array batch/plotting wrapper is population-level tooling deferred to Phase 9 |
| `abel_tree` | `edit.py` | done | moved here (not `metrics.py`) since it needs `delete_tree`, which itself needs `metrics.py` — putting it in `metrics.py` would cycle |
| `rootangle_tree` | `edit.py` | done | ditto, needs `resample_tree`; also centers the tree on its root first, fixing a latent MATLAB assumption that root sits at the coordinate origin (Design Decision #24) |

## Phase 5 — Full I/O

`src/pytrees/io/{swc,neurolucida,native}.py`, tested in `tests/test_io_extra.py`
(104 tests total in the suite now) plus the existing `tests/test_swc_io.py`.

| MATLAB function | Status | Notes |
|---|---|---|
| `swc_tree` (full writer options) | done | `save_swc` now uses the real `idpar_tree` (Phase 2) instead of a private duplicate, and writes `%12.8f`-equivalent precision matching MATLAB instead of Phase 1's placeholder `.6g` |
| `neurolucida_tree` | done | **not** a full port — see Design Decision #25. Core branch/region geometry only, reimplemented as a proper recursive-descent S-expression parser instead of MATLAB's line-by-line paren-depth state machine; verified against both a hand-written synthetic `.asc` snippet and the real bundled `sample/neurolucida/twop9purks.ASC` (8 trees recovered: 2 CellBody, 2 Axon, 2 Dendrite, 2 Apical, all valid; the 1 marker block and Thumbnail/ImageCoords header blocks correctly excluded) |
| `save_tree` / `load_tree` | done | **not** MATLAB's extension-sniffing multi-format dispatcher (`.mtr`/`.swc`/`.neu`/`.nmf`) — a single native format instead, see Design Decision #26. Single-tree only; multi-tree batching is Phase 9's `list[Tree]` territory |
| `neuroml_tree` | deferred | complex external XML interop format; T2N (Phase 11) talks to NEURON directly via the Python `neuron` package, not via NeuroML, so there's no near-term consumer — revisit if a concrete interop need shows up |
| `x3d_tree` | deferred | this is a rendering/mesh export (cylinders for external 3D viewers), not tree-data interchange — moved conceptually to Phase 7 (graphical), where the 3D backend decision belongs and where this can reuse whatever geometry code that phase builds |
| `pov_tree` | deferred | same reasoning as `x3d_tree` — POV-Ray rendering export, belongs with Phase 7 |
| `nmf_tree` | wont-port | HDF5 export of the same data `save_swc`/`save_tree` already cover (todo list also flags its own check-function as incomplete in MATLAB); `save_tree`'s native format fills the "extended format" role this was for, without adding an `h5py` dependency |

## Phase 6 — Construct (synthetic trees)

`src/pytrees/construct.py`, tested in `tests/test_construct.py` (130 tests total in
the suite now, plus manual smoke-testing of `MST_tree` on random 300-1000-point
clouds — see Design Decision #27 for the balancing-factor sanity check that gave
real confidence the ported cost function is correct, not just "runs without error").

| MATLAB function | Status | Notes |
|---|---|---|
| `MST_tree` | done | **not** a literal port of the ~600-line hand-optimized MATLAB version — reimplemented on `scipy.spatial.cKDTree` + a lazy-deletion min-heap (Prim's-algorithm style); see Design Decision #27. Ports the core single-growing-tree algorithm (distance + `bf * path_length` cost, `thr` max-distance, `mplen` max-path-length, `avoid_multifurcations`); does **not** port simultaneous multi-tree competitive growing, the `DIST` cost-matrix term, grow-from-cut-ends-only mode, or time-lapse recording — all documented as deferred |
| `BCT_tree` | done | topology-only (all coordinates zero); MATLAB's optional fake-dendrogram-layout metrics (`xdend_tree`) not ported — Phase 7 concern |
| `isBCT_tree` | done | accepts a bare sequence or a `Tree` (uses its column sums) |
| `allBCTs_tree` / `allBTs_tree` | done | `with_trees` kwarg replaces the MATLAB optional second output |
| `clean_tree` | done | one real bug caught and fixed during testing — see Design Decision #28 (root was incorrectly eligible for deletion when it had exactly one child) |
| `clone_tree` / `gscale_tree` | wont-port | a population-statistics-driven generative pipeline (MST_tree with multiple competing starting trees + region-by-region statistical resampling + density-grid outlier removal), tightly coupled to one dataset's region-naming conventions; very high complexity, low standalone value — revisit only if a concrete `Active GC Model` population-generation need arises |
| `fix_tree` / `finetune_fix_tree` | wont-port | todo list already flags these as incomplete in MATLAB itself |
| `soma_tree` | done | core cosine-profile shaping + region tagging; MATLAB's `'-b'` overlap-correction option (branch-angle-dependent sqrt(2) diameter adjustment) not ported — niche physiological correction |
| `cap_tree` | done | core capping only; MATLAB's `'-a'` axon-adding option not ported (hardcoded statistical parameters specific to one published dataset, not a general algorithm) |
| `jitter_tree` | done | reimplemented via per-node BFS over the undirected tree graph instead of precomputed dense adjacency-matrix powers — see Design Decision #29 (also fixes a self-distance quirk) |
| `smooth_tree` / `smoothbranch` | done | |
| `quaddiameter_tree` / `quadfit_tree` | done | bundled `quaddiameter_P.mat`/`quaddiameter_ldend.mat` converted once to `data/quaddiameter.npz` (same native-format reasoning as Phase 5's `save_tree`) |
| `rpoints_tree` / `PP_generator_tree` / `in_c` / `cpoints` / `cplotter` | wont-port | exist only to feed the `clone_tree` pipeline (above) |
| `dscam_tree` | wont-port | research-specific, niche |
| `spines_tree` | deferred | specialized add-on (attach spine cylinders), moderate complexity, low priority relative to the core generative functions ported this phase; revisit if detailed-compartmental-modeling work needs it |

## Phase 7 — Graphical / plotting

**Backend decision: made — see Design Decision #30.** `pyvista` (VTK) is the
primary 3D engine (`plot_tree`, `vtext_tree`, `pointer_tree`, `chull_tree`,
added to `pyproject.toml`'s `plot` extra alongside `matplotlib`); `matplotlib`
provides a lighter `plot_tree_mpl` fallback plus the inherently-2D
`dendrogram_tree`/`dA_tree_mpl`. Both are optional, lazily imported inside
each function — `import pytrees` never requires either.

`src/pytrees/plotting.py`, tested in `tests/test_plotting.py` (150 tests total
in the suite now). PyVista tests run headless (`off_screen=True`, the default)
and were also manually verified by rendering to PNG and visually inspecting the
bundled reconstruction (tube mesh generation ~0.17s, full off-screen render
~0.2s more, for 2251 segments — see Design Decision #30).

| MATLAB function | Status | Notes |
|---|---|---|
| `plot_tree` | done | **not** a port — see Design Decision #30. `mode="tube"` (one radius-varying tube mesh via PyVista's `tube()` filter, replaces MATLAB's per-segment SVD-built patch geometry) or `mode="line"` (fast, no diameter); `scalars=`/`cmap=` replace the color-vector overloading; `plotter=` lets trees be overlaid (replaces MATLAB `hold on`) |
| `plot_tree_mpl` | done | matplotlib fallback (line-only, no tube primitive available); fixes matplotlib's default 3D aspect-ratio distortion via `set_box_aspect` so anatomy isn't visually stretched — documented as the lighter, non-recommended-for-large-trees option |
| `dendrogram_tree` / `xdend_tree` | done | `xdend_tree` reimplemented as an O(n_nodes) bottom-up tree accumulation instead of MATLAB's `ipar`-matrix sort/diff trick (same "average of leaf positions" result, standard algorithm, no risk of the O(n^2) blowup the naive per-node descendant-mask approach would have — see Design Decision #31); `dendrogram_tree` rendered via matplotlib (abstract 2D diagram, not spatial anatomy) |
| `dA_tree` (display) | done | as `dA_tree_mpl` (matplotlib `spy` plot) — kept separate from the 3D `plot_tree` naming to avoid confusion between "plot the tree" and "plot the tree's adjacency matrix" |
| `chull_tree` | done | 3D via `scipy.spatial.ConvexHull` + optional PyVista mesh overlay; 2D returns the hull object without a render (no 2D backend wired up for it yet) |
| `hull_tree` / `vhull_tree` | deferred | density-grid-based isosurface hulls — need 3D binning + marching-cubes-style extraction, substantially more machinery than `chull_tree`'s convex hull |
| `gdens_tree` / `lego_tree` | deferred | density-grid plots, same binning-machinery dependency as `hull_tree` |
| `plotsect_tree` | wont-port | todo list: "has no options to begin with" |
| `pointer_tree` | done | `style="marker"/"sphere"` covers the commonly-used MATLAB modes (`-o`/`-O`/`-s`); the tapering-electrode modes (`-l`/`-v`, built from a synthetic frustum tree) not ported — niche relative to marking a location |
| `spread_tree` | done | reimplemented as a straightforward greedy row-packing bin layout instead of MATLAB's `cumsum`/`mod` index arithmetic (same "roughly square, no-overlap" goal, an aesthetic layout choice with no unique correct answer); `spread_trees` added as a convenience that applies the offsets directly via `tran_tree` |
| `vtext_tree` | done | via PyVista `add_point_labels` |
| `xplore_tree` | wont-port | todo list: "not yet parsed" even in MATLAB; effectively an interactive exploration GUI — Phase 10 territory, not a plotting function |

## Phase 8 — Electrotonics

`src/pytrees/electrotonics.py`, tested in `tests/test_electrotonics.py` (19
tests: hand-computed exact values on a minimal 2-node cable fixture, plus
property-based checks — Kirchhoff zero-row-sum, `M @ sse_tree(tree) == I` —
on the real bundled 2252-node reconstruction) and `tests/test_lif.py` (7
tests for `LIF_tree`/`AdExLIF_tree`: an independent dense implicit-Euler
cross-check for the passive sub-threshold case, plus reset-policy checks
for the supra-threshold case). Required new `Tree.Ri`/
`Tree.Gm`/`Tree.Cm` optional slots (default `None`); see Design Decision #33.

| MATLAB function | Status | Notes |
|---|---|---|
| `M_tree` | done | conductance matrix — foundational for this phase. Ported as a literal transliteration of the sparse matrix algebra (0-based reindex only), not re-derived from cable theory — see Design Decision #33 |
| `gi_tree` / `gm_tree` | done | `'-s'` show option dropped (Design Decision #33) |
| `lambda_tree` | done | |
| `elen_tree` | done | |
| `loop_tree` | done | `inodes1`/`inodes2`/`gelsyn` are required keyword args, not MATLAB's implicit "last node"/"root" positional defaults (Design Decision 10: don't hardcode which node is root/last) |
| `sse_tree` | done | |
| `ssecat_tree` / `syncat_tree` | done | operate on `list[Tree]` per Design Decision 5; `inodes1`/`inodes2` index into the concatenated system exactly as in MATLAB (caller computes offsets via cumulative node counts) |
| `syn_tree` | done | real bug caught during testing: MATLAB's `ge`/`gi` scalar-means-"inject at this index" convention collides with a scalar *default* of `0` — fixed by defaulting to `None` (see Design Decision #33) |
| `LIF_tree` / `AdExLIF_tree` | done | kept as two separate functions (not literally merged) despite the todo list's "consolidate" suggestion — their reset policies are genuinely different (full/partial distance-weighted reset vs. hard clip + adaptation), not just different defaults; shared `M_tree`+capacitance setup *is* factored into one private helper. Dropped two confirmed dead parameters (`LIF_tree`'s `Vzone`, `AdExLIF_tree`'s `Vrest` — both parsed/defaulted but never referenced by the real dynamics) and fixed `AdExLIF_tree`'s hardcoded `v = v(1,:)` output (always node 1 regardless of `iroot`) by returning the full `(n_nodes, T)` trace instead — see Design Decision #34 |
| `cgin_tree` | done | |

## Phase 9 — Stats & comparison

`src/pytrees/stats.py`, tested in `tests/test_stats.py` (15 tests: hand-
verified sholl/summary-statistic values on small fixed fixtures, plus
cross-checks — e.g. `syn_tree`/`sse_tree`-style consistency, pooling
identical distributions, a forced out-of-range `bf_tree` clip). New
core dependency: `pandas` (see Design Decision #35).

| MATLAB function | Status | Notes |
|---|---|---|
| `stats_tree` | done | redesigned around `pandas.DataFrame` per Design Decision 5 — returns tidy `summary`/`points`/`branches`(/`sholl` if `extras=True`) DataFrames instead of MATLAB's nested struct-of-cell-arrays; see Design Decision #35 |
| `dstats_tree` | wont-port | pure visualization of exactly the data `stats_tree` now returns as ordinary DataFrames (`df.hist()`/seaborn/etc. already cover it) — same reasoning as Phase 7/8's dropped `'-s'` options |
| `sholl_tree` | done | `'-s'`/`'-s3'` plotting options dropped; `'-o'`/`'-e'` become explicit `single_only`/`warn_double` kwargs |
| `dissectSholl_tree` | deferred | research-contributed, needs the deferred `boundary_tree`/`convexity_tree` machinery plus a vendored third-party point-in-mesh algorithm; todo list itself says "rewrite, don't profile-and-keep" — high complexity, low value relative to the rest of this phase |
| `convexity_tree` / `boundary_tree` / `share_boundary_tree` | deferred | todo list flags all three buggy/unclear; confirmed concretely for `boundary_tree` — see MATLAB_TOOLBOX_BUGS.md (it unconditionally crashes on its own documented default call path) |
| `bf_tree` | done | uses `scipy.optimize.curve_fit` in place of MATLAB's Curve Fitting Toolbox |
| `r_mc_tree` | deferred | needs alpha-shapes + point-in-polyhedron testing (real new dependencies) for a niche, not-fully-validated-upstream (todo list: check-function incomplete) statistical test |
| `vonMises_tree` | done | ditto (`curve_fit`, not the Curve Fitting Toolbox) |
| `peters_tree` | done | the maintainers' own scope uncertainty ("is this a TREES function?") is about curation, not correctness — the algorithm itself (greedy candidate-synapse matching) is well-defined and self-contained, so it's ported |
| `M_atten_tree` | deferred | has no docstring at all in MATLAB, and the todo list says its own purpose is unclear — porting speculative code around an unknown intent risks confidently porting something wrong |
| Population tooling (replaces MATLAB `Trees` class) | done | `stats_tree`'s pandas-based redesign fills this role directly, per Design Decision 5 |

## Phase 10 — GUI

| MATLAB function | Status | Notes |
|---|---|---|
| `cgui_tree` + helpers | wont-port as-is | GUIDE-based; if a GUI is wanted later, build a small Jupyter/PyVista-Qt viewer on top of Phase 7 instead |

## Phase 11 — T2N + NEURON integration

`src/pytrees/neuron_bridge.py`, tested in `tests/test_neuron_bridge.py` (9
tests, skipped if `neuron` isn't importable). Requires the real `neuron`
package -- no Windows pip wheel exists, so this was gated on the user
installing NEURON's official Windows binary installer first (confirmed
working: NEURON 9.0.0). Foundational vertical slice only -- see Design
Decision #36 for the full scope reasoning and the real `dissect_tree` bug
this phase's testing surfaced.

| T2N source (MATLAB) | Status | Notes |
|---|---|---|
| `t2n_writeTrees.m` / `neuron_template_tree.m` (hoc generation) | done, not literally | replaced with direct `h.Section`/`pt3dadd`/`connect` construction via the Python `neuron` package — no intermediate `.hoc` text, subprocess, or SSH/cluster plumbing needed (`build_neuron_model`) |
| `t2n_makeNseg.m` (nseg / d_lambda rule) | done, not literally | uses NEURON's own `h.lambda_f` (loaded from `stdlib.hoc`) instead of reimplementing the arc3d/diam3d walk by hand |
| `t2n_getMech` / `t2n_getPP` / mechanism plumbing | done, narrower | exposed as general-purpose `insert_mechanism(model, name, region=None, **params)` layered directly on `Tree.Ri`/`Gm`/`Cm` (Phase 8) rather than introducing T2N's own separate per-region "mech" config struct |
| current-clamp simulation + recording | done | `run_current_clamp` — the minimum "inject current, get voltage back" loop, cross-validated against `sse_tree`'s exact steady-state solve (~1-3% agreement for uniform-diameter trees; see Design Decision #36 for why tapered trees diverge further, and why that's expected, not a bug) |
| `Protocols/t2n_*` (IV, FI, current steps, bAP, resonance, ...) | not-started | thin wrappers on top of `run_current_clamp`/`insert_mechanism` once there's a concrete need for a specific one |
| `t2n.m`'s server/cluster/SSH mode | wont-port | Python's direct in-process NEURON binding has no need for MATLAB's external-process/file-based execution model this exists to support |
| `Auxiliaries/*` (plotting, raster, spike gen) | not-started | mostly builds on Phase 7 |

## Stacks (image-stack reconstruction tools) — unscheduled

Not yet placed in the phase order; revisit once the core tree pipeline (Phases 1–6) is
solid. MATLAB functions: `load_stack`, `loaddir_stack` (todo list: `imread` error),
`loadtifs_stack`, `imload_stack`, `save_stack`, `show_stack`, `skel_stack` (todo list:
`histax` dependency), `fitD_stack`.

## Utilities — mostly `wont-port`

| MATLAB function | Status | Notes |
|---|---|---|
| `isBinary`, `parseArgs` | wont-port | superseded by native Python typed kwargs (Design Decision 1) |
| `bezahlt`, `tprint`, `tprint0` | wont-port | MATLAB print/formatting helpers; use Python `pathlib`/f-strings instead |
| `deg2rad`, `rad2deg` | wont-port | use `numpy.deg2rad`/`numpy.rad2deg` directly |
| `eucdist` | deferred | fold into Phase 3 metrics if a 2D-specific variant is still needed once `eucl_tree` exists |
| `rotation_matrix` | deferred | Phase 3, needed by `rot_tree` |
| `histax` | deferred | todo list flags a possible transpose bug; needed by several Phase 7/9 functions — rewrite correctly when first needed, not before |
| `gauss`, `hotcold` | deferred | pull in only when a Phase 7/9 function actually needs them |
| `gifmaker` | deferred | todo list: colormap/`histax` issue; Phase 7 |
| `pov_patch` | deferred | Phase 5 (`pov_tree`) |
| `scalebar`, `shine`, `roundshow` | deferred | Phase 7 plotting polish |

---

## Phase 12 — `Active GC Model` scripts (next up)

The bundled dentate granule-cell model. Scope agreed with the user: **GC
import, model setup, and the dendritic-democratization analysis only** — the
pattern-separation toolbox around it is explicitly out of scope.

Source material: `DGGC_prepare_for_t2n.m`, `DENDEMO.m`,
`Dem_Democracy_Only.m`, `GC_dendritic_demo_main_1_6.mlx` (a Live Script; its
code was extracted from the `.mlx` zip's `matlab/document.xml`).

### What the demo actually does

1. **Init** — `GC_initModel(ostruct)` loads morphologies and sets biophysics.
2. **Preprocess** — `resample_tree`, then rewrite `.hoc`/`minterf.dat`.
3. **Pick a synapse path** — take the furthest terminal by
   `T_tree .* PL_tree`, then walk `idpar_tree` up to the root, marking every
   node on the way. That ordered node list is the synapse placement.
4. **Place synapses** — one `Exp2Syn`-style conductance per node on that path,
   each driven by its own artificial spike source, fired in a staggered
   sequence (~90 ms apart) so each synapse's somatic response is isolated.
5. **Record** — voltage-clamp the soma, record the clamp current; peak current
   per stimulus vs. that synapse's distance to soma is the raw
   democratization curve.
6. **Democratize** — rescale each synapse's weight to equalise its somatic
   effect, by two methods: (a) per-region polynomial fit of peak current vs.
   distance, (b) direct inverse normalisation against the measured peak.
   Re-run and compare the three weight vectors.

### Port plan

| Piece | Approach | Blockers |
|---|---|---|
| GC morphology import | `load_mtr` (done) + `repair_tree` + `resample_tree` | none |
| `GC_initModel` biophysics | New `examples/gc_model.py` setting `Ri`/`Gm`/`Cm` and region-wise channel densities via `insert_mechanism` | needs the model's channel `.mod` files compiled — see below |
| `t2n_writeTrees`/`minterf` | **not needed** — `build_neuron_model` maps nodes to segments directly via `model.loc()` | none |
| Synapse path selection | Direct: `T_tree`, `PL_tree`, `idpar_tree`, `Tree.region_nodes` (Design Decision #38) | none |
| Synapse placement + staggered stimuli | `h.Exp2Syn(model.loc(n))` + `h.NetStim`/`h.NetCon` on the raw NEURON objects | none |
| Somatic voltage clamp + current recording | `h.SEClamp` + `h.Vector().record()` | none |
| Peak detection | `scipy.signal.find_peaks` (replaces MATLAB `findpeaks`) | none |
| Democratization fits | `numpy.polyfit` per region (replaces MATLAB `fit(...,'poly2')`) | none |
| `DGGC_place_syns` / Poisson background | Port if needed; `quantal_TM` stochastic-release model is only used by the *pattern-separation* path | out of agreed scope |

### Custom `.mod` mechanisms — **resolved**

Previously flagged as Phase 12's main open risk. All 44 of the GC model's
NMODL mechanisms now compile and load: 28 density mechanisms (channels), 13
point processes (synapses), and 2 charge-accounting mechanisms. See
`gc_model/build_mechanisms.py` (reproducible build) and
`gcmodel.mechanisms` (loader + inventory).

Two Windows-specific obstacles, both with misleading error messages, are
handled by the build script: NEURON's bundled MSYS shell inherits an
unwritable `TMP=C:\WINDOWS\` (g++ then fails *after* successful NMODL
translation, so it looks like a compiler fault), and paths handed to that
shell must be `/cygdrive/C/...` rather than `C:\...` or Git Bash's `/c/...`.

One source needed patching for NEURON >= 9: `ingauss.mod` forward-declared
`nrn_random_arg` returning `void*`, but NEURON 9 declares it itself returning
`Rand*`, making the old declaration an "ambiguating new declaration" hard
error. Recorded in `build_mechanisms.PATCHED_SOURCES`.

**Still outstanding:** `GC_biophys.m`'s ~656-line option matrix deciding
which channel sits where at what density (it branches on combined option
strings, applies interacting scale factors, and loads 18-parameter Markov
rate tables from `soma_st8.txt`/`axon_st8.txt`). The *mechanisms* are
available; only the published parameter assignment is untranslated, and it is
deliberately left undone rather than half-done. `democracy.py` meanwhile uses
the model's real passive parameters (`Ra=200`, `cm=0.9`, `Rm=36.4 kOhm*cm^2`,
`e_pas=-80`, read off `GC_biophys.m`'s default branch) and can use its real
`Exp2Synq2` synapse.

### Deliberately out of scope

`analyse_pattern_separation`, `quantal_TM` short-term plasticity, the
`GC_ATP_pattern_*` energy/ATP scripts, `phase_locked` input generation, and the
cluster/batch drivers.

---

## Design Decisions Log

Dated, append-only. Each entry: what was decided, why, and what it changes going forward.

### 2026-07-30 — Phase 0/1 kickoff

1. **Package name**: `pytrees`, installable from `python_port/` (src layout). Chosen for
   brevity and direct lineage from "TREES toolbox"; easy to rename later since nothing
   external depends on it yet.
2. **Don't port the options-string parser** (`'-s'`, `'-z'`, ...). Use explicit typed
   keyword arguments everywhere. Reason: the MATLAB maintainers' own todo list documents
   this parser as a repeated source of bugs (numeric-prefixed flags, switch/case
   first-match-only bugs). How to apply: every ported `verb_tree` function gets real
   Python parameters, not a `options: str` blob.
3. **Don't port `classes/+trees` `subsref` dispatch magic.** Reason: it's an incomplete,
   self-described-as-messy experiment ("Tree and Trees — Merge!!" in the MATLAB todo
   list). How to apply: `Tree` is a plain data container; behavior lives in free
   functions named `verb_tree(tree, ...)`, matching MATLAB naming 1:1 for familiarity.
4. **0-based indexing everywhere.** Reason: native to Python/NumPy; MATLAB is 1-based.
   How to apply: every index-returning/consuming function (parent indices, region
   indices, node selections) must be re-derived for 0-based semantics, not mechanically
   transliterated line-by-line.
5. **"No parent" sentinel is `-1`, not MATLAB's `0`.** Reason: MATLAB overloads `0`
   (an invalid 1-based index) as "no parent found"; in 0-based indexing `0` is a real
   node index and can't double as a sentinel. How to apply: any function that surfaces
   "does this node have a parent" (e.g. the future `idpar_tree` port) uses `-1` for
   "none" and only falls back to "root is its own parent" when that specific MATLAB
   default behavior is being reproduced on purpose.
6. **`dA` orientation kept from MATLAB**: `dA[child, parent] = 1`, sparse, square,
   N×N, 0-based now. Reason: preserves the existing mental model/documentation for
   anyone moving between the two codebases; only the index base changes, not the
   semantics.
7. **`Tree.validate()` mirrors `ver_tree`'s non-raising, warning-collecting behavior**
   (returns a list of problem strings; does not raise by default). Reason: the MATLAB
   code frequently builds intentionally-incomplete trees mid-pipeline (e.g. before
   `repair_tree` runs) and only warns rather than hard-failing; replicating that lets
   later phases (`repair_tree` et al.) work the same way. How to apply: call sites that
   want strict behavior can opt in via a `strict=True` argument once that's needed;
   not added preemptively (YAGNI).
8. **`load_swc` does not call `repair_tree` yet** (that function doesn't exist in the
   port until Phase 4). Reason: avoid a forward dependency from Phase 1 to Phase 4.
   How to apply: `load_swc` builds the raw tree faithfully (including handling
   unsorted/non-contiguous SWC indices and multi-root files, matching MATLAB's
   `load_tree.m` `.swc` branch), and callers needing BCT-conformity must wait for
   Phase 4 or call it manually once it lands.
9. **`.mtr`, `.neu`, `.nmf` import deferred out of Phase 1.** Reason: `.mtr` is a
   MATLAB v5 `.mat` file (readable via `scipy.io.loadmat` in principle, but only once
   there's a real reason to load the many existing `.mtr` archives in `Active GC
   Model/morphos/`); `.neu`/`.nmf` are niche formats not needed for the first vertical
   slice. Revisit if/when Phase 5 or a concrete `Active GC Model` migration needs them.

### 2026-07-30 — Phase 2

10. **Root is detected by row-sum of `dA` (in-degree), not by assuming index 0.**
    Reason: MATLAB's graphtheory functions get away with assuming the root is
    always node 1 (1-based) because a tree is only ever handed to them after
    `sort_tree`/`repair_tree` established that. Our Phase 4 (`repair_tree`) isn't
    built yet, and even once it is, `sort_tree` itself must work on arbitrary
    input by definition. Row-sum detection (`_root_index` in `graphtheory.py`)
    is unambiguous regardless of where the root happens to sit, and costs one
    cheap vector reduction, so every Phase 2 function uses it rather than
    special-casing "assume index 0". How to apply: any future function that
    needs "which node is the root" should reuse `_root_index`, not hardcode `0`.
11. **`ipar_tree`'s padding is a genuine `-1` sentinel, produced by an explicit
    stop condition — not carried over as MATLAB's implicit zero.** Reason:
    MATLAB's version walks `dA` as a matrix power and lets the padding *emerge*
    as `0`, which is safe only because `0` is never a valid 1-based node index.
    In 0-based indexing the root's own real entry in the path (`ipar[i, k] == 0`
    when the root is node 0) is indistinguishable from "no more ancestors" if
    padding also uses `0`. The port instead follows each node's parent chain
    explicitly (`idpar_tree(no_self=True)`, which itself yields `-1` at the
    root) and stops writing real values the step after the root is reached, so
    `-1` unambiguously means "past the root" and never collides with a real
    index. How to apply: any function consuming `ipar_tree`'s output (e.g.
    `child_tree`, `Pvec_tree`) must treat `NO_PARENT` (`-1`) as "stop/no
    contribution", exactly where MATLAB's version treated `0` that way.
12. **`sort_tree`'s default ("hier") mode is a DFS pre-order, not MATLAB's
    iterative index-shuffle.** Reason: MATLAB's algorithm (repeatedly moving
    each node next to its parent's position, re-deriving the permutation each
    step) produces *a* valid BCT-conform order, but the docstring itself says
    "many isomorphic BCT order structures exist" — there's no uniqueness
    contract to preserve for the default mode, only the two documented
    properties (parent precedes child; each subtree is a contiguous index
    range). A plain pre-order DFS from the root satisfies both directly, in a
    form any reader can verify by inspection, at the same asymptotic cost.
    The `'lo'`/`'lex'` modes, which *do* have a specific documented ordering
    contract, are ported faithfully (a presort by the documented sort key,
    then the same DFS pass to guarantee contiguity on top of it — mirroring
    MATLAB running its hierarchical fixup after its own `-LO`/`-LEX` presort).
13. **`redirect_tree` is a plain BFS rerooting, not MATLAB's undirected
    adjacency matrix-power walk.** Reason: MATLAB computes new topological
    distances from the new root via repeated sparse matrix multiplication
    purely to get a sort key; a direct BFS gives the same distances *and* the
    new parent-of-each-node relationship in one pass, which is what's actually
    needed to rebuild `dA` — the matrix-walk version had to separately re-derive
    edges via the reordered old `dA`. BFS is standard, easier to verify, and
    strictly less work.
14. **`strahler_tree` is a post-order traversal, not MATLAB's queue/fixed-point
    loop.** Reason: MATLAB's version maintains a work-queue of "pending parent
    ids" and repeatedly re-scans it until every node resolves, which is
    difficult to read and (per inspection) implicitly assumes binary branching
    to bound its own termination. A post-order traversal (children strictly
    before parents, via the existing `_dfs_preorder` reversed) computes the
    same Strahler numbers in one linear pass and drops the binary assumption
    for free (ties among 3+ children are handled the same way ties among 2
    are). How to apply: this is the first place the port's Strahler numbers
    could disagree with MATLAB's — only for trees with a genuine trifurcation,
    where MATLAB's own asymmetry function (`asym_tree`) *refuses* to operate
    anyway (see next point), so no real MATLAB behavior is being contradicted.
15. **`asym_tree` raises `ValueError` on a non-binary branch point instead of
    silently indexing `dA(:, iB(counter))` and getting `BB(1)`/`BB(2)` from
    whatever two elements happen to come first.** Reason: the MATLAB docstring
    says outright "Tree must be BCT... use repair_tree if necessary" — a
    trifurcation is already documented as out-of-contract input, not a case
    MATLAB handles correctly. Raising surfaces the precondition violation
    instead of quietly computing a number for the wrong pair of children.
16. **Deferred within Phase 2, not implemented yet:** `dissect_tree`,
    `dist_tree` (need Phase 3's `len_tree`/`Pvec_tree` default or Phase 4's
    `root_tree`), `bin_tree` (needs Phase 3's `eucl_tree` default; MATLAB's own
    docstring says it's replaceable by a plain histogram call), `gene_tree`
    (needs Phase 3's `len_tree`; fits better alongside Phase 9's comparison
    tools thematically anyway). All four are noted in PORT_STATUS.md's Phase 2
    table with their specific blocking dependency, not silently skipped.
    (Superseded 2026-07-30, Phase 4: all four landed once their blockers did
    -- see Design Decision #24 below and the "Phase 2/3 backlog items
    unblocked by Phase 4" table.)

### 2026-07-30 — Phase 3

17. **`frustum` is a real, explicit `Tree` attribute (`bool`, default `False`),
    not a dynamically-added field.** Reason: MATLAB's `surf_tree`/`vol_tree`/
    `cvol_tree` all branch on `isfield(intree, 'frustum') && intree.frustum==1`
    -- a struct can grow this field ad hoc, but our `Tree` deliberately has a
    fixed set of slots (Design Decision 2: no `subsref`-style dynamic-field
    magic). Since three separate functions genuinely need this one specific,
    well-documented flag, it earns a real slot rather than a generic
    arbitrary-field bag, which would reopen the door to the MATLAB-style
    dynamic-struct pattern this port deliberately avoids.
18. **`flatten_tree` and `morph_tree` use a precomputed-`ipar_tree` descendant
    lookup instead of calling `sub_tree` inside their per-node loop.** Reason:
    both loop over every node and, at each one, need "this node's entire
    downstream subtree" to apply a coordinate shift -- calling `sub_tree`
    (a fresh BFS) inside that loop makes the whole function O(n_nodes^2) with
    substantial per-call Python/sparse-matrix overhead. Measured on the
    bundled 2252-node reconstruction, that was 42.9s (`flatten_tree`) and
    59.4s (`morph_tree`) before the fix. Since `ipar_tree` is already computed
    once at the top of both functions anyway, checking `(ipar == node).any(axis=1)`
    gives the identical descendant set via one vectorized NumPy comparison per
    node instead of a BFS -- after the change, both dropped to ~0.44s (~100x),
    with byte-identical results (verified: segment lengths still preserved by
    `flatten_tree`, still driven to the target length by `morph_tree`). This
    is, incidentally, close to what MATLAB's own version does --
    `find(ipar == counter)`, a vectorized matrix search rather than a BFS --
    so the fix brings this port's approach *closer* to the original's actual
    algorithm, not further from it. `zcorr_tree` was left calling `sub_tree`
    because its loop only runs over flagged jump nodes (typically a small
    fraction of all nodes, e.g. 26 of 2252 in the bundled reconstruction), so
    the same quadratic blowup doesn't apply there.
    **(Superseded 2026-07-31 -- see Design Decision #37.)** This fix was
    real but incomplete: scanning the `ipar` matrix is still
    O(n_nodes x max_depth) *per node*, i.e. quadratic overall, and it only
    looked fast because the bundled 2252-node reconstruction is shallow.
    On a real granule cell (3765 nodes, 1625 deep) `flatten_tree` took
    **10.1 s**. Replaced with an O(n_nodes) pre-order subtree
    decomposition.
19. **Every geometry transform (`scale_tree`, `tran_tree`, `rot_tree`,
    `flip_tree`, `flatten_tree`, `morph_tree`, `zcorr_tree`) returns a new
    `Tree` unconditionally.** Reason: MATLAB's versions mutate a global
    `trees` array when called without an output argument, and return a new
    struct otherwise -- a dual-mode convenience that only makes sense given
    MATLAB's implicit global cell array of "current" trees, which this port
    doesn't have (and shouldn't: it's exactly the kind of hidden global state
    the rest of this port has avoided). `Tree.with_coords()` (new helper on
    `Tree`) factors out the "copy with some coordinate arrays replaced"
    pattern shared by all seven functions.
20. **`rot_tree`'s PCA-alignment and `'-m3d'`/`'-al'` region-alignment modes
    are not ported.** Reason: they're a materially different, much more
    involved feature (iterative alignment via cross products, PCA/SVD on a
    region's boundary points, region-name lookups) bolted onto the same
    function name in MATLAB via option strings, rather than a variation on
    the same algorithm. The common case -- rotate by an explicit degree
    triple -- covers everything else in the toolbox that calls `rot_tree`.
    Flagged in PORT_STATUS.md rather than silently dropped; revisit if a
    concrete alignment use case comes up (e.g. while porting a specific
    `Active GC Model` preprocessing script).

### 2026-07-30 — Phase 4

21. **`elimt_tree` is rewritten on plain Python edge lists instead of
    MATLAB's growing-sparse-matrix-inside-a-loop.** Reason: MATLAB's version
    concatenates zero blocks onto `dA` and reassigns `tree.(fieldname)` for
    every field on every multifurcation processed, which is hard to follow
    and relies on `fieldnames` iteration order. The port instead collects
    `(child, parent)` edge tuples in a list, appends new ones per spacer
    node, and builds the final sparse matrix once at the end. Verified by
    hand-tracing both versions' index arithmetic against a 4-children (k=4)
    multifurcation to confirm they produce the identical spacer-chain
    topology (child 0 stays with the branch point; children 1..k-2 each get
    their own spacer; the last two children share the final spacer) before
    trusting it on the real reconstruction.
22. **`delete_tree` always splices to the nearest surviving ancestor, and
    always returns a `list[Tree]` when that disconnects the tree.** Reason:
    MATLAB's version ties both behaviors to an `'-x'`/`append_children` flag
    in a way that's internally inconsistent -- deleting a branching root
    with default options produces a single struct with a broken multi-root
    `dA` rather than a forest, and the todo list documents this outright
    ("delete_tree | multiple trees doesn't work yet"). There's no
    MATLAB-correct behavior here to preserve; the port picks the one
    consistent, always-connected-or-explicitly-split contract and applies it
    unconditionally, dropping the flag entirely.
23. **`resample_tree` preserves branch/termination points exactly instead of
    relocating them onto the resampling grid.** MATLAB's version inserts grid
    nodes, then deletes *every* original non-root node (including branch and
    terminal points) and lets `delete_tree`'s splicing "snap" each one onto
    whichever grid node preceded it -- a real algorithm, but one the
    docstring itself calls an "arbitrary" abstraction choice, not a
    mathematical necessity. This port picked the alternative, equally valid
    convention of leaving branch/terminal points exactly where they are and
    only resampling the interior of each section (via the new `dissect_tree`).
    **This distinction was not academic**: the first implementation (ported
    more literally, inserting grid points onto every original parent/child
    edge without removing the original continuation nodes) had a real bug --
    node count went *up* under resampling instead of down (2252 -> 3313 nodes
    at `sr=10` on the bundled reconstruction, whose original spacing is
    already only ~1-4 um). Rewriting around `dissect_tree`'s section
    boundaries (anchor node, walk the true polyline via the parent chain,
    interpolate along cumulative arc length, connect anchor-to-anchor) fixed
    it: 2252 -> 2067 nodes, branch/terminal counts exactly preserved (502
    and 503 both before and after), median resampled segment length close to
    but under `sr` (consistent with `abel_tree` measuring this reconstruction's
    natural inter-branch spacing at ~7.4 um, well under the 10 um target, so
    many sections are naturally shorter than one full `sr` step and are left
    unsubdivided). Caught by testing against the real reconstruction, not the
    small hand-built fixture, which was too coarse to reveal the bug.
24. **`rootangle_tree` explicitly centers the tree on its root
    (`tran_tree`) before measuring angles to it.** Reason: MATLAB's version
    computes `sqrt(X2^2 + Y2^2 + Z2^2)` directly against the coordinate
    origin, which only equals "distance to root" (what the docstring
    promises) if the tree happens to already be sitting at the origin. This
    port adds the centering step so the function's actual behavior matches
    its documented contract regardless of the input tree's absolute
    position -- a small, targeted correctness fix, not a behavior change for
    any tree that was already centered (which most are, by convention).

### 2026-07-30 — Phase 5

25. **`neurolucida_tree` ports the branch/region geometry only, via a proper
    recursive-descent S-expression parser, not MATLAB's line-by-line
    paren-depth state machine.** Reason: NeuroLucida ASCII is a standard
    nested-parenthesis format (confirmed against the real bundled sample,
    `sample/neurolucida/twop9purks.ASC` -- there was no spec doc to work
    from, so the actual file was the source of truth); MATLAB's version
    tracks nesting via three mutable flags (`Plevel`/`Tflag`/`Zflag`)
    threaded through a single pass over file *lines*, which is difficult to
    verify by inspection and doesn't compose. This port instead tokenizes
    the whole file, parses it into ordinary nested Python lists, and walks
    that structure recursively -- splits (`|`-separated branches inside a
    `(...)`) become a straightforward "split the flat item list on `|`,
    recurse on each piece" step. Three real MATLAB features are *not*
    ported, each because the MATLAB docstring itself flags the omitted part
    as ad hoc: soma contours are kept as simple point-chain trees rather
    than being fitted to a fitted cylinder via PCA (docstring: "quite
    arbitrary algorithms... can be much further optimized or just
    rewritten"), markers (small glyphs) are dropped entirely, and
    reconstructed dendrite/axon trees are *not* automatically concatenated
    onto their nearest soma via nearest-point matching -- all returned as
    independent Trees instead, in a `list[Tree]` (Design Decision 3's
    "explicit typed data over implicit magic" applies here too: nearest-
    point soma attachment is a heuristic a caller can apply deliberately via
    `cat_tree`, not something that should happen silently on load).
    Verified two ways: a hand-written synthetic `.asc` snippet with a known
    branch point (exact node count, region names, and branch-point position
    asserted), and the real bundled sample file (recovers exactly 8 trees --
    2 CellBody, 2 Axon, 2 Dendrite, 2 Apical -- correctly excluding the 1
    marker block and the Thumbnail/ImageCoords header blocks, all valid
    per `ver_tree` and reducible to a strictly binary tree via
    `repair_tree`).
26. **`save_tree`/`load_tree` are a single native format (`numpy.savez`,
    effectively a zip of named arrays), not MATLAB's extension-sniffing
    dispatcher across `.mtr`/`.swc`/`.neu`/`.nmf`.** Reason: `.mtr`/`.neu`/
    `.nmf` support was already deferred in Phase 1 (Design Decision 9) with
    no real consumer since; with only one non-SWC format actually
    implemented, a dispatcher is pure overhead (`load_swc`/`save_swc` are
    already the explicit, named entry point for SWC). Deliberately **not**
    pickle-based despite that being the shortest implementation: pickle
    executes arbitrary code on load, a bad default for "open a tree file
    someone sent you." `numpy.savez`/`numpy.load` round-trip the sparse
    `dA` (stored as COO row/col/shape), all coordinate arrays, region names
    (numpy stores fixed-width string arrays natively, no `allow_pickle`
    needed), and the `frustum` flag exactly, verified by a round-trip test
    asserting every field. This single-tree native format is *also* what
    now fills the role MATLAB's `nmf_tree` (HDF5 export, and per the todo
    list a MATLAB check-function that's itself incomplete) was for --
    "extended format beyond swc" -- without adding an `h5py` dependency.

### 2026-07-30 — Phase 6

27. **`MST_tree` is reimplemented on `scipy.spatial.cKDTree` + a
    lazy-deletion min-heap, not MATLAB's hand-rolled "vicinity window."**
    Reason: MATLAB's ~600-line version manually maintains, per growing
    tree, a distance-sorted list of nearby unclaimed points and re-slices
    it every iteration purely to avoid an O(n^2) distance recomputation --
    a real performance concern, but solved today by a KD-tree radius query
    (`query_ball_point(node, thr)`) instead of hand-maintained sorted
    windows. The cost function itself is unchanged: connecting point `p`
    to tree node `t` costs `distance(p, t) + bf * path_length(t)`, and
    since a still-open point's best achievable cost can only *improve* as
    the tree grows (more candidate attachment nodes become available, never
    fewer), the standard lazy-deletion min-heap pattern applies directly --
    push a new `(cost, point, attach_node)` entry whenever a cheaper
    attachment is found, and treat a popped entry as stale (skip it) if a
    cheaper one was recorded since. This is the standard technique for
    Prim's-algorithm-style incremental growth, not something specific to
    this port. Not ported: simultaneous multi-tree competitive growing
    (MATLAB's `msttrees` can be several starting trees growing at once,
    claiming points from each other), the `DIST` sparse cost-matrix term
    (a connection-probability bias, rarely used), grow-from-cut-ends-only
    mode (tied to `DIST`), and time-lapse recording (an animation/debug
    feature). `avoid_multifurcations` (MATLAB's `'-b'`) is ported as a
    best-effort approximation: a popped candidate is rejected if its
    attachment node already has 2 children, but (unlike MATLAB, which
    actively excludes branch points from the search at push-time) a point
    whose only ever-recorded candidate saturates this way isn't
    automatically re-queued against another node -- documented as a known,
    rare-in-practice gap rather than silently claimed as exact.
    **Verified two ways**, since a small hand-built tree can't validate a
    balancing algorithm: exact topology on a 4-point straight line
    (unambiguous greedy result), and a qualitative check on 300 random
    points that `bf=0` minimizes total wiring length while `bf=1`
    minimizes path length (this is the actual documented purpose of the
    algorithm -- a broken cost function could easily still produce *a*
    valid tree while getting this trade-off backwards or flat, which the
    balancing-factor check would catch and a topology-only test wouldn't).
28. **A real bug in `clean_tree`, caught by testing, not just written
    down as "seems right":** the fallback branch-start index (when no
    branch/terminal point precedes a given terminal) was initially `0`
    (the root), which made the root itself eligible to be considered part
    of a "deletable branch run" whenever it had exactly one child (a
    single-stalk root, common in real reconstructions). This caused two
    concrete failures: a hand-built test tree where the whole tree's total
    length was compared against `radius` instead of just the short stub
    branch, and unexpectedly large deletions on the real bundled
    reconstruction even with `radius` near zero. Fixed by making the root
    an implicit boundary that's never itself included in a branch run (the
    fallback start becomes index 1, not 0). The MATLAB original has the
    same latent issue -- its `find(...,1,'last')+1 : iT(counter)` produces
    an ambiguous or MATLAB-version-dependent empty-range expression in
    this exact situation (empty `find` result concatenated into a colon
    range) rather than a deliberate fallback -- so this isn't a case of
    "the port introduced a bug MATLAB didn't have"; it's a real edge case
    neither version handled explicitly, made concrete and fixed here.
29. **`jitter_tree` is reimplemented via per-node BFS over the undirected
    tree graph instead of precomputing dense adjacency-matrix powers
    (`A^k` for `k` up to `lambda`).** Both compute the same thing --
    Gaussian-kernel-weighted (centered at 1 hop, width `lambda/5`) blending
    of independent per-node noise, restricted to nodes within `lambda`
    topological hops -- but MATLAB detects "is node v reachable from node u
    within k hops" via `(A^k * indicator) > 0`, a walk-existence test, and
    the *first* ascending `k` at which that's true only equals the true
    shortest-path distance because of a parity argument (odd/even walk
    lengths on a tree). One consequence: a node's distance to *itself* is
    2 under this scheme (the first `k>=1` admitting a length-`k` closed
    walk), not 0 -- an artifact of the method, not an intentional
    "de-emphasize self" design choice (nothing in the docstring or
    surrounding code suggests it was deliberate). This port uses direct
    BFS instead, which is both more standard/readable and gives the
    intuitively correct self-distance of 0; the Gaussian kernel itself
    (`mu=1`, not `mu=0`) is unchanged, so immediate neighbors still get
    slightly more weight than a node's own independent noise term -- that
    part of the formula's behavior is preserved exactly, only the distance
    metric feeding it was corrected. Also asymptotically cheaper for large
    trees with modest `lambda`: BFS touches O(lambda) nodes per source
    instead of computing full sparse matrix powers shared across all
    sources.

### 2026-07-30 — Phase 7

30. **3D plotting backend: `pyvista` (VTK), left open since Phase 6.**
    Reason, concretely measured rather than assumed: MATLAB's `plot_tree.m`
    builds cylinder geometry by hand, one singular-value decomposition per
    segment to find vectors orthogonal to that segment's direction (its
    own inline comment calls the loop a "BOTTLENECK"), then assembles one
    big vertex/face array for a single `patch` object -- and its line-mode
    fallback is documented as slower still. PyVista's `PolyDataFilters.tube()`
    does the equivalent job (turn a polyline into a radius-tapered tube
    mesh) as a single compiled VTK operation that accepts a per-point
    radius array directly -- so a per-segment frustum taper (matching
    MATLAB's `frustum` mode) comes for free, with no per-segment Python-level
    loop at all. Measured on the bundled 2252-node/2251-segment
    reconstruction: tube generation ~0.17s, a full off-screen render another
    ~0.2s, and the result is one mesh VTK can pan/zoom/rotate at native
    framerates -- not a static image, and not thousands of separate patch
    objects. `matplotlib`'s `mplot3d` has no tube primitive and no real GPU
    acceleration for large scenes, so it's kept only as a lighter, explicitly
    secondary option (`plot_tree_mpl`) for quick previews or environments
    without a VTK-capable display -- not a peer to `plot_tree`. Both are
    optional extras, imported lazily inside each function that needs them,
    so `import pytrees` never requires either (verified: the whole test
    suite, including every plotting test, imports and runs with `pyvista`
    installed but would `pytest.importorskip` cleanly without it).
    Off-screen/headless rendering was verified working in this environment
    before committing to the decision (a real risk for a VTK-based library,
    since some Linux setups need a virtual framebuffer -- Windows didn't
    need one here, but this is exactly the kind of assumption worth
    checking empirically rather than asserting).
31. **`xdend_tree` is an O(n_nodes) bottom-up tree accumulation (post-order:
    every node's descendant-terminal rank range is its children's ranges
    combined), not MATLAB's `ipar`-matrix sort/diff computation.** Both
    compute the same dendrogram-layout quantity -- each node's X position
    is the midpoint of its leftmost and rightmost descendant terminal's
    rank -- but MATLAB's version sorts and diffs the entire `ipar` matrix
    flattened to a vector, a somewhat opaque way to extract "for each node,
    the range of ranks among its descendant terminals." The first Python
    draft used the same `(ipar == node).any(axis=1)` per-node descendant
    lookup that fixed Phase 3's `flatten_tree`/`morph_tree` performance
    problem, but doing that *again* for all `n_nodes` nodes (rather than
    just the nodes touched by a bounded per-node loop, as those two
    functions do) would reintroduce an O(n^2) cost. Reusing the tree's own
    parent/child structure (already available via `graphtheory.py`'s
    `_children_lists`/`_dfs_preorder` helpers) to accumulate ranges
    bottom-up avoids that entirely: verified at ~8ms on the bundled
    2252-node reconstruction.

### 2026-07-30 — post-Phase-7: `.mtr` support added

32. **`.mtr` import (Design Decision 9's deferral) resolved, scoped to
    MATLAB v5.** Reason: Phase 1 deferred it for lack of a concrete need;
    demonstrating `plot_tree` against a real granule-cell model (the
    `Active GC Model/morphos/*.mtr` archives) is exactly that need.
    Checked every bundled `.mtr` file's format first (`scipy.io.loadmat`
    against each): most are MATLAB v5 (plain `.mat`, directly readable),
    a handful are v7.3 (HDF5-based, would need `h5py`). Rather than block
    on adding an `h5py` dependency for the few v7.3 files, `load_mtr`
    reads v5 only and raises a clear, file-naming error for v7.3 input
    instead of scipy's generic `NotImplementedError` -- a real gap, but a
    named and diagnosable one. Verified against the real bundled
    `SH_07_all_repairedandsomaAIS_MLyzed-Midi.mtr` (8 granule-cell
    reconstructions, 2931-4308 nodes each, all valid per `ver_tree` with
    zero issues) and against a real v7.3 file to confirm the error path.
    MATLAB's 1-based `R` region-index values are converted to 0-based on
    load (same convention as every other loader in this port); extra
    struct fields some `.mtr` files carry (`Ri`, `Gm`, `Cm`, `col`, `NID`,
    `jpoints`, `Rho_soma`, `Rho_AIS`, ...) are outside this port's `Tree`
    model and dropped -- `Ri`/`Gm`/`Cm` specifically belong to Phase 8's
    electrotonics work, not I/O.

### 2026-07-31 — Phase 8 (steady-state functions)

33. **`Tree.Ri`/`Tree.Gm`/`Tree.Cm` added as real, optional (`None`-default)
    slots**, same reasoning as Phase 3's `frustum` (Design Decision #17):
    enough separate functions (`M_tree` and everything built on it) need
    these specific physical parameters that they earn real slots rather
    than a generic dynamic-field bag. Unlike `frustum`, there's no sensible
    universal default -- MATLAB itself never auto-populates `Ri`/`Gm`/`Cm`
    either (every `check_*` fixture in `tests/electrotonics/` sets
    `tree.Ri = 100`, `tree.Gm = 1/2500`, `tree.Cm = 1` by hand before calling
    anything), so every function here raises a clear `ValueError` naming the
    missing attribute instead of silently assuming a value. `reindexed()`/
    `with_coords()` carry a per-node array through node reordering exactly
    like `X`/`D`/etc., but pass a scalar through unchanged (a uniform
    resistivity/conductance doesn't need reindexing).
    **`M_tree`'s conductance-matrix construction is a literal
    transliteration of MATLAB's sparse matrix algebra (0-based reindex
    only), deliberately not re-derived from cable-theory first
    principles** -- same posture as Phase 2's `LO_tree`. Working through the
    algebra by hand (see `electrotonics.py`'s module docstring) shows the
    axial conductance assigned to the edge between a child and its parent
    uses the *parent's* own `cvol` value, not the child's -- surprising on
    first read, but not something to "fix" by guessing at a different
    convention without a concrete failure to chase, given this is an
    established, widely-used (Cuntz et al.) formula. Verified instead via a
    property any valid conductance-Laplacian must satisfy regardless of
    that choice: with the membrane leak zeroed out (`Gm=0`), every row of
    the axial part sums to zero (Kirchhoff's current law) -- confirmed on
    the real bundled 2252-node reconstruction (max abs row sum ~1e-13) --
    plus exact hand-computed entries on a trivial 2-node cable fixture, both
    checking the *translation* is faithful rather than re-litigating the
    underlying convention.
    **A real bug caught by testing, not just written down as "seems
    right":** `syn_tree`'s `ge`/`gi` parameters initially defaulted to the
    Python scalar `0.0`. MATLAB's own convention (ported faithfully in
    `_onehot_or_array`) treats *any* scalar as "inject a canonical unit
    conductance at this 0-based node index" -- so a scalar *default* of
    `0.0` was silently read as "add 1 uS of excitatory conductance at node
    0" on every call, rather than "no synaptic input" (MATLAB avoids this
    collision because its own default is the zero *vector* `sparse(N,1)`,
    never a bare scalar). Caught by a test asserting `syn_tree(tree)` with
    no arguments returns all-zero potentials (the trivial no-input case);
    fixed by defaulting `ge`/`gi` to `None` (meaning the zero vector)
    instead of `0.0`.
    `LIF_tree`/`AdExLIF_tree` (time-stepping spiking simulations) are
    **not** part of this pass -- meaningfully more machinery (a time loop,
    spike/reset logic, several more physiological parameters with their own
    documented defaults) than the steady-state functions above, and
    deliberately left `not-started` in PORT_STATUS.md rather than rushed
    through, to revisit as a focused follow-up.

### 2026-07-31 — Phase 8 (LIF_tree / AdExLIF_tree)

34. **Kept as two separate functions, not merged into one despite the todo
    list's explicit "consolidate... instead of keeping two near-identical
    functions" suggestion.** Reason: reading both line by line (not just
    their parameter lists) shows their *reset policies* are genuinely
    different modeling choices, not just different default values --
    `LIF_tree` either fully resets every node to `vreset` or applies a
    distance-weighted partial reset (a sigmoid of cumulative path length
    from the root); `AdExLIF_tree` instead hard-clips every node whose
    voltage exceeds `vreset` down to it (leaving already-lower nodes
    untouched) and separately increments an adaptation variable `w`. These
    aren't reconcilable into one formula behind a boolean flag without
    either losing a real behavior or growing a third, subtly-different one
    to cover both -- so what *is* shared (the `M_tree` + implicit-Euler
    capacitance setup) was factored into one private `_M_and_capacitance`
    helper, while the genuinely different time-stepping bodies stay
    separate public functions. Dropped MATLAB's option-string-driven
    tree-field-fallback pattern (`if ~isfield(tree, 'Ri'): tree.Ri = ...`)
    throughout, per Design Decision 1 -- every parameter is a real, typed
    Python keyword argument with its documented default, not sourced from
    an ad hoc tree field.
    **Two confirmed dead parameters, dropped rather than ported:**
    `LIF_tree`'s `Vzone` is parsed and defaulted (`0.995`) but the *only*
    place it's referenced in the entire function is inside a commented-out
    line (`% v(v(ireset,...) > Vzone*thr, ...) = vreset;`) -- it does
    nothing in the actual (uncommented) code path. `AdExLIF_tree`'s
    `Vrest` is likewise defaulted onto the tree (`-70`) but never appears
    anywhere else in the function body; the resting potential the dynamics
    actually settle around is governed by `EL` instead. Both added to
    MATLAB_TOOLBOX_BUGS.md.
    **A confirmed bug, not reproduced:** MATLAB's `AdExLIF_tree` ends with
    `v = v(1, :)`, hardcoding the returned trace to node index 1 (1-based)
    regardless of what `iroot` was actually set to -- silently returning
    the wrong node's voltage trace whenever `iroot ~= 1`. This port
    returns the full `(n_nodes, len(time))` trace instead (matching
    `LIF_tree`'s own contract), letting the caller slice `v[iroot]`
    themselves; strictly more informative and not prone to the same
    footgun.
    **Verified via an independent cross-check, not just "runs without
    crashing":** the passive (never-spiking, `thr` set unreachably high)
    case was checked against a dense implicit-Euler stepper written
    directly in the test file (`tests/test_lif.py`), calling neither
    `LIF_tree` nor `AdExLIF_tree` -- confirming the time-stepping mechanics
    (capacitive term, backward-Euler solve) independently of whether
    `M_tree` itself is correct (already covered separately). The
    supra-threshold tests use a single-time-step current pulse rather than
    a sustained one specifically so exactly one spike occurs at a known
    step -- a sustained large current instead causes back-to-back spikes
    every step, where the "next" column is immediately overwritten by the
    *following* spike's marker before a test can inspect the clean
    post-reset value (caught while first writing these tests: an initial
    version using a sustained current intermittently asserted the wrong
    thing for exactly this reason).

### 2026-07-31 — Phase 9

35. **`stats_tree` redesigned around `pandas.DataFrame`, per Design
    Decision 5** (population tooling: `list[Tree]` + `pandas`, not a
    bespoke container). MATLAB's version returns a nested
    struct-of-cell-arrays (`gstats`/`dstats`, indexed by group, then tree,
    then -- for per-branch fields -- branch again), exactly the kind of
    ad hoc container Design Decision 5 already ruled out. This port
    instead returns a dict of tidy, long-format DataFrames: `"summary"`
    (one row per tree), `"points"` (one row per branch/termination point),
    `"branches"` (one row per dissected branch -- kept as a *separate*
    DataFrame rather than crammed into `"points"`, since MATLAB's own
    per-tree `BO`/`Plen`/`peucl`/`angleB` and `blen` arrays are genuinely
    different lengths -- one entry per topological point vs. one per
    branch -- so forcing them into one fixed-width row per tree would
    either silently truncate/pad one of them or require ragged
    list-valued cells, neither of which plays well with `groupby`/
    plotting). `dstats_tree` -- MATLAB's large, `stats_tree`-specific
    multi-panel plotting function -- is **not** ported: it visualizes
    exactly the data `stats_tree` now returns as ordinary DataFrames,
    plottable directly via `df.hist()`/seaborn/etc., so reproducing its
    specific panel layout would just be a second, parallel plotting API
    for data already available in a more flexible form (same reasoning as
    every `'-s'` option dropped since Phase 7).
    **`Tree.stats_tree`'s `extras=True` path is intentionally narrower
    than MATLAB's `'-x'` option**: hull volume (via Phase 7's
    `chull_tree`) and mean branch-point asymmetry (via `asym_tree`) are
    included, but the density/Voronoi piece (`parea`/`mparea`) is not --
    it depends on `hull_tree`/`vhull_tree`, deferred since Phase 7 pending
    density-grid machinery neither has yet. Also not ported: the
    upper/lower hull-area split (`uharea`/`lharea`) and derived
    "convexity index", which need a hull-vertex-ordering helper
    (`cpoints`) tied to that same deferred density-grid workflow.
    **Four functions deferred as disproportionate complexity/value for
    this phase, each for a distinct concrete reason** (not a blanket
    "too hard"): `dissectSholl_tree` (research-contributed, needs the
    deferred boundary/convexity machinery *and* a vendored third-party
    point-in-mesh algorithm; the maintainers' own todo list says
    "rewrite, don't profile-and-keep" -- they don't consider the current
    version worth preserving either); `convexity_tree`/`boundary_tree`/
    `share_boundary_tree` (self-flagged buggy/unclear -- and reading
    `boundary_tree` line by line surfaced a *confirmed* bug, not just an
    "unclear" one: `pars = convexity_tree(intree, 'dim2', pars.dim2,
    'dim3', pars.dim3)` overwrites the entire parsed-arguments struct with
    a bare scalar, so every later `pars.dim2`/`pars.dim3` reference
    crashes whenever `c` isn't explicitly supplied -- the documented
    default path; see MATLAB_TOOLBOX_BUGS.md); `r_mc_tree` (a Monte Carlo
    clustering statistic needing alpha-shapes and point-in-polyhedron
    testing -- real new dependencies -- for a niche test the todo list
    itself flags as having an incomplete check-function upstream);
    `M_atten_tree` (literally no docstring in the MATLAB source, and the
    todo list says its own purpose is unclear -- porting speculative code
    around an intent nobody, including the original maintainers, can
    currently explain risks confidently porting something wrong).
    **`peters_tree` ported despite the maintainers' own scope question**
    ("is this a TREES function?" in the todo list): re-reading the
    function shows that's a curation/scope question (does candidate-
    synapse detection between two trees belong in an anatomy-reconstruction
    toolbox?), not a correctness one -- the greedy nearest-candidate-first,
    eliminate-nearby-duplicates algorithm is well-defined, self-contained,
    and doesn't touch any of the functions deferred above.
    **New core dependency: `pandas`** (not an optional extra) -- unlike
    `pyvista`/`matplotlib` (heavy, genuinely optional plotting backends),
    population-level tooling is core functionality as of this phase, and
    `pandas` is lightweight and near-ubiquitous in the Python scientific
    stack.

### 2026-07-31 — Phase 11

36. **Environment gate, checked before writing any code**: NEURON's
    Python package has no Windows pip wheel (confirmed via PyPI and a
    real install attempt) -- Windows needs the official binary installer,
    a system-level install this port shouldn't perform silently. Asked
    the user first; they installed NEURON 9.0.0, confirmed importable and
    able to run a real two-section passive simulation before any bridge
    code was written, so the rest of this phase is verified against the
    actual simulator throughout, same as every other phase.
    **`t2n.m` (2447 lines) is not a literal-translation candidate**:
    reading its structure shows the overwhelming majority is `.hoc` text
    generation (`neuron_template_tree.m`) plus file/SSH/cluster plumbing
    to hand a simulation to an *external* NEURON process -- necessary only
    because MATLAB has no direct NEURON binding. Python's `neuron` package
    gives in-process `h.Section`/`pt3dadd`/mechanism access directly, so
    essentially none of that machinery has a reason to exist in the port
    (already anticipated in PORT_PLAN.md's phase table). What *is* ported
    faithfully is the geometric core of `neuron_template_tree.m`: each
    dissected branch becomes one `h.Section` built from real 3D points via
    `pt3dadd` (continuous frusta, not manually-set `L`/`diam`), connected
    parent-to-child per the tree's topology. Segment count uses NEURON's
    *own* d_lambda rule (`h.lambda_f`, loaded from `stdlib.hoc`) rather
    than reimplementing MATLAB's arc3d/diam3d walk by hand -- reusing
    NEURON's own verified implementation is strictly more robust than
    re-deriving it, the same posture this port has taken toward other
    established formulas (Phase 2's `LO_tree`, Phase 8's `M_tree`), just
    here the "established implementation" is NEURON's own C code rather
    than MATLAB's. Passive properties come directly from `Tree.Ri`/`Gm`/
    `Cm` (Phase 8) instead of introducing T2N's own separate per-region
    "mech" struct; T2N's generality (arbitrary mechanisms per named
    region) is exposed instead as a general-purpose `insert_mechanism`.
    Scope is deliberately narrow: `build_neuron_model` + `insert_mechanism`
    + `run_current_clamp` is the foundational "does a Tree become a real,
    running compartmental model" vertical slice -- matching this port's
    established practice (PORT_PLAN.md's own "recommended first vertical
    slice" reasoning from Phase 0/1) of proving the foundation solidly
    before building a large surface on top of it. T2N's full protocol
    library (IV/FI curves, resonance, bAP, synaptic/network protocols) and
    its cluster/SSH execution mode are **not** ported -- the former are
    thin wrappers on top of exactly this layer (add when a concrete one is
    needed), the latter has no purpose once execution is in-process.
    **A real, confirmed bug in `dissect_tree` (Phase 2), found only now
    because this is the first phase to exercise region-based sectioning
    in anger**: `by_region`'s cut was placed on the region-*transition*
    node itself (`region_change[node] = R[node] != R[parent(node)]`).
    MATLAB's own algorithm (`iR = idpar(tree.R ~= tree.R(idpar))`) places
    it on that node's *parent* instead -- the transitioning node's own
    segment already belongs to the new region, so it should *start* the
    new section, not end the old one. Marking the wrong node silently
    produces one extra, spurious section split at *every* region
    boundary in *every* function that calls `dissect_tree` with
    `by_region=True` (its default) -- which includes `resample_tree`
    (Phase 4) and `stats_tree` (Phase 9), neither of which had a
    multi-region test fixture to catch it. Fixed at the source in
    `graphtheory.py`, along with a guard for the edge case where the
    region change lands exactly at the root's own child: the root can't
    be split off on its own (nothing precedes it), so the section simply
    extends all the way back to the root, same as if there were no region
    split at all -- verified by hand-deriving MATLAB's true output
    (including its own `root_tree`-prepend-then-subtract-1 workaround) for
    a small tree and confirming the fixed Python version reproduces the
    same non-degenerate sections. An existing test
    (`test_dissect_tree_cuts_at_region_change`) had encoded the *buggy*
    behavior as its expected value (its region transition happened to sit
    at the root's own child, exactly the degenerate case where the bug
    and the fix coincidentally agree in that one test) -- corrected, and
    a new test added specifically for a transition that isn't at the
    root's boundary, which does distinguish the two.
    **Verification**: built a small branching tree, ran a real NEURON
    current-clamp simulation to steady state, and compared it against
    `sse_tree`'s exact linear-algebra steady-state solve (Phase 8) -- two
    independently-implemented solvers (an industry-standard simulator vs.
    hand-rolled sparse linear algebra) agreeing is strong evidence both are
    right. For a uniform-diameter tree they agree to ~1-3%. A first attempt
    with a *tapered* tree (diameter varying node to node) showed a ~24%
    discrepancy -- investigated rather than dismissed, and traced to a
    real, expected modeling difference: NEURON's `pt3dadd` always
    continuously interpolates diameter between consecutive 3D points,
    while `sse_tree`'s discretization (Phase 8, non-frustum mode) treats
    each segment as a uniform-diameter cylinder. These are two genuinely
    different (both valid) discretizations of a continuously-tapering
    cable, not a bug in the conversion -- confirmed by rerunning the same
    comparison with uniform diameter (removing the taper ambiguity
    entirely), where agreement returned to ~1%. Both cases are covered in
    `tests/test_neuron_bridge.py`: a quantitative cross-check for the
    uniform case, a qualitative ordering check for the tapered one.
    **A second, related `dissect_tree` fix, caught only by testing against
    a real bundled reconstruction** (the granule-cell morphology in
    `Active GC Model/`, whose root/soma genuinely is a branch point --
    none of the hand-built test fixtures happened to have that shape): the
    root was still being treated as the *end* of a section whenever it was
    itself a cut point (a real branch point, not just via the region-change
    guard above), producing a spurious root-to-itself entry. A caller
    building a new adjacency matrix from `dissect_tree`'s output (e.g.
    `resample_tree`) turned that into an actual self-loop edge, which then
    left the reconstructed tree with zero in-degree-zero nodes --
    `_root_index` failing with "expected exactly one root, found 0".
    Generalized the fix: the root is never treated as a section's end,
    for any reason it might be cut -- there's nothing before it to split
    off regardless of why it's a boundary.

### 2026-07-31 — review pass: performance, ergonomics, documentation

37. **`flatten_tree`/`morph_tree`/`smooth_tree`: the Design Decision #18 fix
    was real but incomplete, and a proper O(n_nodes) replacement landed.**
    Found by profiling every public entry point against a real granule cell
    (3765 nodes) rather than the bundled 2252-node reconstruction all
    previous timings used. `flatten_tree` took **10.1 s**. The cause: #18
    replaced a per-node `sub_tree` BFS with a per-node
    `(ipar == node).any(axis=1)` scan, which is faster per call but still
    rereads all `n_nodes x max_depth` entries *every* iteration — total
    O(n^2 x depth). It only looked fixed because the bundled reconstruction
    is shallow (the granule cell is 1625 nodes deep, so the `ipar` matrix
    alone is 49 MB).
    Replaced with `graphtheory._subtree_blocks`: one pre-order traversal,
    after which every node's descendants occupy a *contiguous block* of that
    order, so a lookup is O(subtree) and building the whole decomposition is
    O(n_nodes). `flatten_tree` went **10.07 s -> 0.058 s (173x)**, with
    `morph_tree` and `smooth_tree` similarly improved. Verified equivalent
    three ways before trusting it: identical descendant sets vs. the old
    `ipar` scan, identical vs. `sub_tree`'s independent BFS, and
    `flatten_tree`'s documented length-conservation contract still holds to
    3.4e-13. Regression tests added in `tests/test_graphtheory.py` (sampled
    rather than exhaustive — an all-nodes cross-check against `sub_tree` is
    itself quadratic and made that file take 48 s on its own).
    **Generalizable lesson recorded here deliberately:** the port's
    performance claims had all been measured on one shallow tree. Depth,
    not just node count, drives cost for anything touching `ipar_tree`.
    `ipar_tree` remains a dense `n_nodes x max_depth` matrix by design
    (it mirrors MATLAB's) — that's fine for ancestor queries, but it is the
    wrong tool for descendant queries, and this is now documented in both
    `_subtree_blocks`' docstring and `docs/guide.md`'s performance section.
38. **Ergonomics: `Tree.region_nodes`/`region_mask`/`region_index` added.**
    Motivated concretely rather than speculatively: the MATLAB idiom
    `find(tree.R == find(strcmp(tree.rnames,'soma')))` appears dozens of
    times across the bundled `Active GC Model` scripts that are next in line
    to be ported, and the port had no equivalent — every translated line
    would have restated that whole expression. Unlike MATLAB's `strcmp`
    (which returns empty for an unknown name, so the *following* comparison
    silently matches nothing), `region_index` raises a `KeyError` listing
    the available regions. `Tree.__repr__` now also shows the region names,
    since they're the first thing you need before doing anything
    region-dependent.
39. **Documentation and tutorials written** (`docs/`, `examples/01..05`).
    Four docs (`concepts`, `guide`, `matlab-migration`, `api-overview`) and
    five executable tutorial notebooks. Two process notes worth keeping:
    - Every notebook is **executed as part of authoring** (`jupyter
      nbconvert --execute`), and the prose was corrected wherever the real
      output contradicted it. This caught four wrong claims that would
      otherwise have shipped: `sample_tree`'s region is `'1'` not `'dend'`;
      `MST_tree` returns `(tree, connected)` not a bare `Tree`; 0.5 nA into
      the sample tree does *not* fire an action potential (2 nA does); and
      an HH spike is *larger* at the thin distal tip than at the soma, the
      opposite of the "attenuated backpropagation" story first written.
    - A genuine API gotcha surfaced the same way: `syn_tree(tree, ge=n,
      Ee=0.0)` correctly returns all zeros, because this module expresses
      voltages as deviations from rest, so `Ee=0` means "no driving force".
      Not a bug, but easy to trip over — now called out in `docs/guide.md`
      and in the electrotonics notebook.
    - Docs are checked against the code (a scratch script verified every
      backticked function name in `docs/` exists in `pytrees.__all__`, and
      that all 108 public symbols are mentioned somewhere).

### 2026-08-18 — review response W1/W2/W4/W7 (see `REVIEW_PLAN.md`)

Implementing the function-by-function review. Work packages W1 (naming and
signature cleanup), W2 (I/O parity), W4 (return-value contract) and W7
(correctness audit) landed together, since they touch the same signatures.

40. **Dimensionality is always `dim: int` in `{2, 3}`.** Reason: three
    spellings had accumulated for one concept -- `dim2: bool` (`cyl_tree`,
    `len_tree`, `chull_tree`), `dim: int` (`eucl_tree`), and `dim: str`
    (`"2d"`/`"3d"` in `vonMises_tree`, `bf_tree`). The integer form reads
    naturally at the call site (`len_tree(t, dim=2)`), was already the
    majority, and extends to `hull_tree` without a fourth spelling.
    How to apply: new functions take `dim`, never a boolean flag.
    Implementation note: the *signature* default is `None`, not `3`. Without
    a sentinel there is no way to distinguish an explicit `dim=3` from the
    default, so the contradiction `len_tree(t, dim=3, dim2=True)` would pass
    silently; with one, it raises. Docstrings state the effective default.
41. **No negated boolean parameters.** `idpar_tree(no_self=)` became
    `root_self=` and `elimt_tree(no_root=)` became `at_root=`, defaults
    flipped so behaviour is unchanged. Reason: a default of `no_self=False`
    makes the reader resolve a double negative to work out what happens.
    Both retired spellings still work for one release via
    `pytrees/_compat.py`, raising `DeprecationWarning` -- verified against
    `gc_model`, which was still calling `no_self=True` and kept working.
42. **Multi-output functions return their primary result only; extras are
    opted into with `full_output=True`.** Applies to `sort_tree`,
    `redirect_tree`, `insertp_tree` and `insert_tree`; extras come back as
    `NamedTuple`s (`SortResult`, `RedirectResult`, ...) so `result.order`
    and tuple unpacking both work. `elimt_tree`'s `changed` flag was
    **dropped entirely** (recomputable; now a `logging.debug` line).
    Reason: the common case stops paying for the rare one. The evidence was
    already in the codebase -- of the 9 internal call sites unpacking these
    tuples, **7 wrote `tree, _ = ...`**.
    Scope: only functions whose primary result is the `Tree`. `sholl_tree`,
    `chull_tree`, `spread_tree`, `vonMises_tree` and `bf_tree` return
    several co-equal results by nature and keep returning all of them.
    **Deliberately shipped without a compatibility shim**, unlike #40/#41:
    no shim can straddle "returns a tuple" and "returns a Tree", and none is
    needed -- `Tree` defines neither `__iter__` nor `__getitem__`, so a
    stale `tree, order = sort_tree(t)` raises `TypeError: cannot unpack
    non-iterable Tree object` immediately, at the call site. That is checked
    by a test, as is a static `ast` sweep asserting no source file unpacks
    these without asking (it would catch a stale call in a module no test
    imports).
43. **`sub_tree` returns `(mask, tree)` by default**, with
    `with_tree=False` to skip building the extracted Tree. Reason: the
    subtree is half of what the function is *for*, and the performance
    objection raised while planning turned out to be overstated -- measured,
    eager extraction costs 1.3x per call (2203 vs 1682 us on a 3765-node
    cell), and only `asym_tree` still calls it in a loop (34 times there),
    which passes `with_tree=False`. An earlier docstring claimed
    `repair_tree` and `clean_tree` also called it in a loop; they no longer
    do, and that claim has been corrected.
44. **`sub_tree`'s extracted subtree trims `rnames` to the regions actually
    present**, reindexing `R` to match. Reason: MATLAB does *not* --
    `sub_tree.m` carries the comment "NOTE ! region update for tree output
    still missing!!!" -- and being told a purely dendritic branch still has
    an `axon` region is useless. A deliberate improvement on the original,
    not a porting divergence.
45. **v7.3 (HDF5) `.mat`/`.mtr` reading, via `mat73`.** Reason: MATLAB's own
    `save_tree.m` writes `'-v7.3'` **unconditionally**, so every `.mtr` a
    current TREES install produces was unreadable. Not an edge case -- the
    default path, and the crux of MATLAB/Python interoperability.
    The library was chosen by **measurement, not documentation**, against
    the three real v7.3 files in `Active GC Model/morphos/`:
    `scipy.io` and `hdf5storage` raise on both layouts; `pymatreader` reads
    the flat cell array but returns **raw, undereferenced `h5py.Reference`
    objects** for the nested one; only `mat73` reads both. The nested layout
    is not obscure -- it is `load_tree.m`'s documented 2-level `cgui_tree`
    form, and two of the three files use it. Choosing `pymatreader` from its
    docs (it returns exactly scipy's shape, which is tempting) would have
    shipped a silent failure.
    Also established: **Octave cannot read these files either** (`load`
    warns "can't read 'tree' (unknown datatype)"), so there is no
    re-save-as-v5 escape hatch outside MATLAB.
    `mat73` is an optional `[matlab]` extra, imported lazily.
46. **Format is detected from the file header, not from scipy's exception
    type.** Reason: scipy raises `NotImplementedError` for most v7.3 files
    but `ValueError: embedded null character` for `0dplaxonFitsoma.mtr`,
    having got far enough into the HDF5 bytes to misparse them as a v5
    structure. The 128-byte MATLAB text header ("MATLAB 7.3 MAT-file") is
    unambiguous; the exception type is not.
47. **`_flatten` accepts `scipy.io.matlab.mat_struct`, not just
    `dict`/`list`/`ndarray`.** Reason: at certain nesting depths scipy
    returns `mat_struct` **even with `simplify_cells=True`**. Without this,
    `load_mtr` could not read `dLPTCs.mtr` at all -- the bundled 55-tree,
    5-group population, which is precisely the fixture `stats_tree`'s
    group-comparison API exists to consume.
48. **`Tree.root` is a public property; the root is never index 0.**
    Reason: Design Decision #10 established row-sum root detection, but was
    *phrased* as a Phase-2 report ("every Phase 2 function uses it"), so by
    Phase 5/6 it read as history rather than a live invariant -- and
    `_root_index` was module-private in `graphtheory`, so reaching it from
    `metrics`/`construct` meant a cross-module private import. The path of
    least resistance was `tree.X[0]`, and three functions took it:
    `scale_tree` (scaled about the wrong point), `flip_tree` (mirrored about
    the wrong point) and `cap_tree` (capped the wrong end). All three were
    faithful transliterations -- MATLAB hardcodes `tree.X(1)` in exactly
    these three files -- so the port inherited the assumption rather than
    inventing it. All fixed, and a shuffled-root tree added to the tests so
    the next violation fails a test rather than a review.
49. **`Tree.total_length`/`total_surface`/`total_volume` properties**,
    replacing `sum(len_tree(tree))`. Uncached, deliberately: `Tree` is
    mutable in place (`tree.X[5] = ...` is legal and used by the editing
    functions), so a cache would go stale silently.
50. **`Pvec_tree`'s `v` defaults to `len_tree`**, giving metric path length
    from the root. Reason: six call sites inside the toolbox alone spelled
    out `Pvec_tree(tree, len_tree(tree))`. Purely additive.
51. **`insert_tree` validates parent indices and can return the new node
    indices** (`full_output=True`). New nodes parenting *earlier* new nodes
    was always supported -- `cap_tree` depends on it, chaining cap segments
    -- but was undocumented and unvalidated, so a forward reference silently
    produced a cycle or an orphan. Now documented, with an example, and
    rejected with a clear error. Region inheritance follows the same chain.
52. **`sample_tree()` loads `sample.mtr`; `sample2_tree`/`hsn_tree`/
    `hss_tree`/`dLPTCs_trees` ported alongside.** Reverses a Phase-1
    stand-in: `.mtr` reading was deferred (#9) and `sample/swc/` holds
    exactly one file, so `25HSS.swc` became the fixture and kept the name
    `sample_tree`. `.mtr` support landed at #32 and the sample was never
    revisited.
    The substitution cost more than a node count (2252 vs MATLAB's 197).
    `25HSS.swc` *is* the HSS cell -- same node count and same 8100.26 um
    total length as `hss.mtr` -- but the SWC export is **X-mirrored**
    relative to it, and SWC cannot carry region names, so `axon`/`dend`/
    `soma` collapsed into a single region `'1'`. Region handling is exactly
    what `dissect_tree`, `stats_tree` and the NEURON bridge exercise, so the
    default sample exercised none of it. This is also the root cause of a
    documentation correction recorded under #39 ("`sample_tree`'s region is
    `'1'` not `'dend'`") -- a symptom, not an isolated slip.
    Nothing was lost: that tree is now `hss_tree()`, in regioned `.mtr`
    form. Side benefit: the smaller default sample cut the test suite's
    BLAS-bound `sse_tree` inversions and kept total runtime at ~14 s.
    `dLPTCs` group names come from the **longest common prefix** of each
    group's tree names, not the leading alphabetic run: the groups are
    `dvs2`/`dvs3`/`dvs4`, which share the alphabetic prefix `dvs` and differ
    only by the following digit. Stripping digits silently merged three
    groups into one and lost 20 of the 55 trees.
53. **`load_mtr` no longer requires the variable to be called `tree`.**
    It prefers that name, falls back to the sole tree-shaped variable, and
    takes an explicit `variable=` for the ambiguous case (listing the
    candidates in the error). Reason: a `.mtr` is just a MATLAB workspace;
    the old check rejected anything saved by hand or by T2N, which commonly
    stores `tree` alongside other variables.

### 2026-08-18 — W3 (silent gaps), continued

54. **`resample_tree(method='matlab')` ported and made the default**,
    superseding #23/#45. Verified **differentially against the MATLAB
    source** (`edit/resample_tree.m` run in Octave 11 on the 197-node
    `sample_tree`), across five option combinations:

    | case | MATLAB | pytrees |
    |---|---|---|
    | default, sr=10 | 78 nodes, 724.5606 um | identical |
    | sr=5 | 155 nodes, 753.8981 um | identical |
    | `-d` interp. diameters | 78 nodes | identical |
    | `-l` length conservation | 78 nodes, 770.0000 um | identical |
    | `-v` no collapse | 91 nodes, 821.6374 um | identical |

    Coordinates agree to **4.3e-14**, diameters to **6.7e-16**.

    Two things this pinned down that reading the source did not:
    - **The collapse tie-break.** When two collapse candidates have equally
      large subtrees, which survives is invisible in the geometry (both are
      moved to their midpoint first) but visible in the diameters. Keeping
      the *first* daughter matches; the opposite choice reproduced
      everything else exactly and left ~7 of 78 diameters differing by up to
      0.023 um. Found by bisecting on `-d -v` vs `-d`, not by reading
      MATLAB's `min(child(itodel))` indexing -- which is genuinely hard to
      read, since the orientation of `collab`'s entries decides whether
      `min` runs column-wise or over the whole array.
    - **Resampled segments are *shorter* than `sr`, not equal to it.**
      Grid points are placed at multiples of `sr` along the *original*
      path; deleting the intermediate nodes then replaces each polyline
      with a chord. MATLAB's source says so at the `'-l'` branch ("we cut
      the paths short"), and `conserve_length` exists precisely to undo it.
      An earlier version of the new test asserted exact grid multiples --
      i.e. asserted a property the algorithm does not have.

    `method='anchors'` (the previous behaviour, branch/termination points
    preserved exactly) remains available and is still the right choice when
    branch-point positions matter, as in the NEURON bridge.
55. **`plot_tree`'s `color` is polymorphic again, and the positional order
    matches MATLAB** (`intree, color, DD, ipart, res`). Reason: `color=` and
    `scalars=` were two parameters where one was always `None`, and the
    merged form is what a MATLAB user types anyway. Everything this port
    adds beyond MATLAB's five arguments is **keyword-only**, which is what
    keeps the order matched -- a future addition cannot wedge itself into a
    positional slot.
    The one genuine ambiguity is a 3-node tree, where a length-3 vector
    could be an RGB triple or three per-node values; read as RGB, matching
    MATLAB, with `scalars=` as the explicit override. `scalars=` is retained
    as a keyword, so every existing call site kept working unchanged.
    **Not reproduced**: MATLAB's `'-b'` flat-patch mode exists to dodge the
    cost of real cylinders in MATLAB's renderer, which `mode='tube'` does
    not have; and `'-2q'`/`'-3q'` quiver plots of a 4000-segment tree are
    unreadable. `'-2l'`/`'-3l'` map onto `mode='line'`.
57. **`ipar_tree(terminals_only=)`, `dissect_tree(with_positions=)` and
    `soma_tree(overlap_correction=)` ported** -- MATLAB's `'-T'`, second
    output, and `'-b'`.
    - `terminals_only` returns one row per termination point, each the
      unbranched run back to (excluding) its first branch point, with the
      all-padding tail trimmed. That trim is the point: 2.3 MB -> 0.08 MB on
      the HSS cell, a 28x reduction, while every one of the 26 paths matches
      MATLAB exactly.
    - `with_positions` adds MATLAB's per-node `(section index, fraction
      along that section)` -- precisely NEURON's `sec(x)` addressing. A
      branch point starts two sections and ends one; it belongs to the one
      it **ends**, at fraction 1.0. Getting that backwards lets the later
      assignment silently overwrite the earlier, which is what a first
      version did (and what a first version of the *test* wrongly asserted).
      MATLAB draws the same line, by assigning `DEC(1:end-1)`.
    - `overlap_correction` divides diameters by `sqrt(2)` per branch point
      already passed, so two cylinders meeting at a branch stop
      double-counting shared membrane. On `hss_tree` at `maxD=120` this cuts
      total surface by 46%; on `sample_tree` at `maxD=30` it correctly does
      nothing, since no branch point inside the soma profile has been passed
      yet.

    **Three MATLAB bugs found while doing this** (all reproduced in Octave,
    now in `MATLAB_TOOLBOX_BUGS.md`):
    - `soma_tree(..., '-b')` **crashes** on any tree whose root has a single
      child -- including the toolbox's own `sample_tree`. Its guard is
      `if 1 < numel(idchild_tree(tree, 1))`, but `idchild_tree` returns a
      fixed-width NaN-padded matrix, so `numel` is 2 even for one child and
      the next line evaluates `dr(NaN, :)`. There is therefore no MATLAB
      reference for this option; the port is checked against the physical
      property instead. This port's `idchild_tree` already sizes its output
      to the widest node found rather than hardcoding 2, so it cannot
      reproduce the cause.
    - `ipar_tree`'s `'-T'` is **unreachable by its documented call**: the
      docstring shows options third, but `parseArgs` registers only `ipart`
      as positional, so `ipar_tree(tree, '-T')` binds `'-T'` to `ipart` and
      returns a 2x27 matrix of nonsense instead of 26x37. `'T', true` works.
      A real keyword makes this unrepresentable here.
    - `dissect_tree`'s second output inherits the root handling its own
      docstring disclaims ("isn't completely correct yet at the root"):
      MATLAB prepends a fake root, slices it back off with `vec(3:end,:)`,
      then patches `vec(1,2) = 0`. The port computes positions from its
      already-root-clean sections, so none of that is needed.
58. **`MST_tree`'s remaining MATLAB options ported**: competitive multi-tree
    growth, the `DIST` cost term, grow-from-cut-ends, time-lapse recording,
    and the `indx` second output. `full_output=False` per #42, so the bare
    call now returns the tree (or a list, for several start points) rather
    than `(tree, connected)`.
    - **Multi-tree growth** is the one that mattered most: it is how the
      published construction is normally used -- several cells grown into
      one shared point cloud, each bidding for every point, so territories
      emerge from the competition instead of being assigned. The heap
      formulation generalised directly: seed from every start and tag each
      candidate with its owning tree.
    - **`dist`** is indexed over the **input points only**. MATLAB requires
      the caller to index it over the growing trees' own nodes as well
      ("Don't forget to include input tree nodes into the distance matrix
      DIST!"), which is easy to get wrong and impossible to validate. Note
      its values are in *distance units*: the penalty spans `0..max(dist)`,
      so a preference of 1.0 against 10 um spacings is correctly ignored.
    - **`record`** returns the growth **log** (`[tree, point, parent]` per
      attachment), not MATLAB's list of intermediate trees. Every
      intermediate state is a prefix of the log, so storing whole trees per
      step would be quadratic in memory for information already there.
59. **B1 landed: `hull_tree`, `gdens_tree`, `lego_tree`, `vhull_tree` and
    `share_boundary_tree`**, in a new `density.py`. One module rather than
    five scattered functions because they share two primitives -- bin a tree
    into voxels, and measure distance from arbitrary points to it -- and
    splitting them would have meant writing the second one three times.
    - **Distance is to the nearest segment, not the nearest node.** This is
      why it is not a `cKDTree` query: measuring to nodes would make a
      hull's shape depend on how finely the morphology happened to be
      sampled. MATLAB makes the same choice.
    - `hull_tree` gives the **space-filling** hull -- the surface at `thr`
      um from the arbor, following its concavities -- as opposed to
      `chull_tree`'s convex hull. On the sample tree at `thr=5` the
      space-filling hull occupies **39.5%** of the convex hull's volume,
      which is the difference between "the volume a cell spans" and "the
      volume it occupies".
    - `gdens_tree` is indexed `[x, y, z]`; MATLAB's is `[y, x, z]`, its
      image convention. Transposed deliberately: every other array in this
      port is `[x, y, z]`, and mixing the two silently is how axis bugs
      happen.
    - `vhull_tree` returns **NaN** for unbounded Voronoi cells rather than
      dropping them as MATLAB does. Dropping biases any mean over the
      result, because the outermost nodes are exactly the ones with the
      largest territories.
    - **`stats_tree`'s `parea`/`mparea` now work** -- the density statistics
      deferred since Phase 7 for want of these two functions. `mparea` is
      verified to equal `mean(parea)` exactly.
    - No MATLAB diff is possible for the hull itself: marching cubes
      produces a mesh whose vertex count and ordering are implementation
      details, so matching vertex-for-vertex would be matching an artefact.
      Tested against the properties the geometry must have instead --
      monotone growth in `thr`, containment of every node, zero overlap
      between distant trees, decreasing overlap with separation.
    - **Performance/precision trade, stated because it is a real one:**
      `_segment_distance` expands the squared distance rather than forming
      an explicit closest point, dropping the `(P, N, 3)` intermediates and
      landing the work in BLAS. About 3x faster and far lighter on memory,
      but it is the numerically unstable expansion: a distance that should
      be exactly zero returns ~`|coords| * sqrt(eps)`, measured at 2e-6 um.
      That is picometres against a 0.1 um reconstruction precision. The
      test asserts the bound rather than exact zero, and says why.
    - New optional dependency: `scikit-image` (marching cubes) under the
      `[plot]` extra. 2D contours need only matplotlib.
60. **B2 (part): `M_atten_tree`, `angleBd_tree`/`angleBd2_tree`,
    `boundary_tree`, `convexity_tree`.**
    - **`M_atten_tree` closes `electrotonics`** -- it was the only unported
      function in an otherwise complete folder. **MATLAB ships it with no
      documentation whatsoever**: no header comment, no description of the
      return value, and a stray `clf;` (clear-figure) left mid-computation.
      The docstring here is derived from reading what the code does: it
      counts how many electrotonically distinct compartments a tree breaks
      into at a coupling threshold. The `clf` is deliberately not
      reproduced -- a metrics function must not wipe the caller's figure.
    - **`angleBd_tree`/`angleBd2_tree`** measure branch angle `dist` nodes
      out rather than at the immediate daughters, so a single jittered
      reconstruction point cannot swing the result. The two differ in which
      branch they follow at an intervening branch point: the bulkier
      subtree, or the one reaching furthest. That distinction is real --
      they disagree at up to 90 of 223 branch points on `hsn_tree`, by as
      much as 81.5 degrees -- but **identical on `sample_tree`**, which is
      too small for the two rules ever to diverge. Both facts are pinned by
      tests, so neither looks like a bug later.
    - **`boundary_tree` and `convexity_tree` could not be verified against
      MATLAB.** Both rest on MATLAB's built-in `boundary()`, which Octave
      does not implement (`exist('boundary') == 0`), so the differential
      approach used everywhere else in W3 was unavailable. They are tested
      against geometric properties instead, and the docstrings say so
      rather than implying a fidelity that was not established.
    - `boundary_tree` interpolates between the convex hull (`shrink=0`) and
      the tightest alpha shape that still **envelops** every point
      (`shrink=1`), which is what MathWorks documents those endpoints to
      mean. A plain quantile cutoff -- the obvious first implementation --
      abandons most of the tree well before `shrink` reaches 1: it returned
      a 4-vertex sliver of a 197-node cell.
    - **`convexity_tree` deliberately does not follow MATLAB.** MATLAB tests
      visibility against `boundary(X, Y, Z, 0)`, and a shrink factor of 0
      is documented as the *convex hull* -- against which every segment
      between interior points is inside by definition, making the measure
      degenerate. This version tests against the space-filling hull from
      B1, the standard definition, which actually separates a compact arbor
      from a lobed one: on the sample tree it runs 0.60 at `thr=10` up to
      1.00 at `thr=60`. **Upgraded from "suspect" to confirmed in B2's
      second half** -- see #61; reading `convexity_tree.m` closely enough to
      port `dissectSholl_tree` showed its 3D branch also returns the
      *complement* of what its own 2D branch returns, which is a textual
      fact about the file and needs no execution to establish.

    Deliberately skipped from B2, per the "cleaner in Python" rule:
    `tlen_tree` (superseded by `Tree.total_length`) and `dstats_tree`'s
    display layer (superseded by returning DataFrames plus matplotlib).
61. **B2 (rest): `r_mc_tree`, `dissectSholl_tree`, and the `boundary_tree`
    rework they forced.** This closes B2 and, with it, every function in
    `treestoolbox-master/metrics/`.
    - **`boundary_tree` gained a real return type and a MATLAB-compatible
      knob.** Porting the two consumers showed the old `(vertices,
      simplices)` tuple was missing everything they needed: the enclosed
      volume, the filled simplices, and -- in 2D -- an *ordered* polygon
      rather than a bag of edges. It now returns a `Boundary` NamedTuple
      carrying all of it. It also accepts MATLAB's parameterisation, `c=`
      (convexity), which sets `shrink = 1 - c`; `shrink=` remains available
      because "how tight is the wrap" is the thing the algorithm actually
      takes, and hiding it behind a derived quantity helps nobody.
    - **The shrink family is interpolated by rank, not by radius.** The
      first version moved the circumradius cutoff linearly between the two
      documented endpoints. The circumradii are heavily skewed -- a few
      slivers spanning the arbor's concavities are orders of magnitude
      larger than the rest -- so on `sample_tree` the enclosed volume sat
      at 138871 um^3 for every shrink from 0 to 0.5 and only collapsed past
      0.9: most of the dial did nothing. Walking the *sorted* radii instead
      gives 138881 / 106942 / 76717 / 61250 / 50294 across shrink 0 to 1,
      and still hits both documented endpoints exactly. A test now asserts
      each quarter-turn moves the shape by at least 5%.
    - **`r_mc_tree` samples exactly instead of rejecting.** MATLAB fills the
      bounding box uniformly and discards everything outside the boundary,
      testing each candidate with a vendored point-in-polyhedron routine --
      for a thin arbor in a large box, most of every batch. Drawing from
      the boundary's own simplex decomposition weighted by simplex volume
      gives the identical uniform distribution with no rejection loop and
      no vendored code. Verified directly: 40000 draws from a unit cube
      land at mean 0.5 per axis with each octant holding 12.5% +/- 1%.
    - **`r_mc_tree`'s volume correction follows the docs, not the code.**
      MATLAB documents `-nv` as "no volume correction" and states the
      correction is on by default, but writes `if pars.nv % volume
      correction` -- so the flag inverts and the documented default never
      happens. This port takes the documented intent: `volume_correction`
      defaults to `True`. It matters -- R = 0.567 with, 0.621 without, on
      `sample_tree`. MATLAB_TOOLBOX_BUGS.md.
    - **`dissectSholl_tree` casts each ray once instead of per radius.**
      MATLAB tests a million random points *per radius* for containment in
      the boundary mesh -- 25 million point-in-mesh tests per call, through
      a vendored `intriangulation`/`voxelise` pair. But every one of those
      points lies on a ray from the root, differing only in how far along
      it sits. Casting each ray once (Moeller-Trumbore), recording every
      surface crossing, and reading containment off the crossing parity
      gives the same estimator at all 25 radii in one pass. The whole 3D
      dissection runs in 0.3 s on `sample_tree`.
    - **MATLAB's 500 um fudge is reproduced but exposed.** Its 3D branch
      silently doubles the estimated mean branch length for cells reaching
      past 500 um, with no comment, no docstring mention, and no 2D
      counterpart. Kept for fidelity -- changing it would move published
      numbers silently -- but surfaced as `scale_factor=`, which `1.0`
      disables. Same treatment for its 2D-only first-bin extrapolation of
      the root-angle histogram, reproduced and documented rather than
      quietly harmonised.
    - **Neither is verifiable against MATLAB, and the tests say so.** Both
      bottom out in `boundary()`, absent from Octave. `r_mc_tree` is
      instead pinned against point sets with answers known independently of
      any implementation -- a uniform cloud must score R = 1 (measured
      1.00 +/- 0.12), a lattice near 2 (measured > 1.6) -- which is a
      stronger check than a diff against one implementation would have
      been. `dissectSholl_tree` is pinned on the invariants its profiles
      must satisfy (unit integrals, radii spanning the cell, the domain
      profile vanishing outside the territory) plus agreement with the
      already-verified `sholl_tree` on `scale`.
    - **A biologically meaningful sanity check fell out for free**, and is
      kept as a test: on `sample_tree`, all nodes score R = 0.57 (strongly
      clustered -- that is the reconstruction's sampling, not the cell),
      while termination points alone score 1.33, i.e. spread more regularly
      than chance. That is the measurement the function exists to make.
    - **`bf_tree(params=)` renamed to `fit_constants=`** (REVIEW_PLAN S3,
      the last outstanding item from it). The argument holds the three
      published constants of the Bird & Cuntz 2019 k->bf relationship, not
      data; `params` read like the latter. Old spelling warns.
    - Housekeeping: `np.trapezoid` (numpy >= 2.0 only) replaced throughout
      `stats.py` by `scipy.integrate.trapezoid`, which works on the
      `numpy>=1.24` the package actually declares. That mismatch was
      already live in `vonMises_tree`.
62. **B3: file formats — the `.neu` reader, `.nmf`, the `.mtr` writer, the
    NEURON and NeuroML exporters, and the `load_tree`/`save_tree`
    dispatcher.** This closes `treestoolbox-master/IO/` apart from
    `pov_tree` and `x3d_tree`, which the Blender work supersedes.
    - **MATLAB interoperability now runs both ways, and is verified.** A
      `.mtr` written by `save_tree` here loads through MATLAB's own
      `load_tree` under Octave, and `len_tree`, `B_tree`, `T_tree` and
      `PL_tree` run on the result with values matching Python to every
      digit printed (total length 765.152557, 25 branch points, 26
      terminals, max path length 35). A list of trees round-trips as a
      MATLAB cell array. This was the stated point of the v7.3 reader work
      in #47; the writer completes it.
    - **`save_mtr` writes v5, not v7.3**, though MATLAB's `save_tree` forces
      `-v7.3`. MATLAB's `load_tree` calls plain `load`, which reads either,
      so nothing on the MATLAB side can tell. Writing a MATLAB *struct*
      into v7.3 by hand means reproducing undocumented `MATLAB_class`
      attributes and object references — reimplementing a format MATLAB has
      never specified — whereas `scipy.io.savemat` is a maintained, tested
      writer for a format MATLAB *has* documented for decades. The only
      real v5 limit, 2 GB per variable, is orders of magnitude past any
      morphology.
    - **The `.neu` reader is the rare case in this cluster that *could* be
      diffed against MATLAB.** Octave's `textscan` does not behave like
      MATLAB's, so `load_tree.m` will not run there; the file parsing was
      rewritten in plain Octave and everything from `d = zeros (nsec, 1)`
      onward copied verbatim, so the arithmetic under test is MATLAB's.
      Parent indices came back **bit-identical** and geometry identical to
      0.0 on all three shipped fixtures.
    - That harness also showed **MATLAB's `.neu` reader crashes on
      `GC1.neu`**, one of the three fixtures the toolbox ships for it: the
      file's root section is tenth of sixty, so the root is node 306, and
      the reader's `for counter = 2 : N` loop assumes it is node 1. The
      port finds roots wherever they sit and splits the forest on them.
      MATLAB_TOOLBOX_BUGS.md.
    - **`.neu` region names diverge deliberately.** MATLAB truncates a
      section name at its first `[`, which turns `GCT.neu`'s ninety
      sections — `GC7[0].adendGCL[3]`, `GC7[0].soma[0]`, ... — into one
      region called `GC7[]`. Blanking each bracket in place instead gives 7
      anatomically meaningful regions and is identical to MATLAB on the
      simple names in the other two fixtures. The divergence is a fix for
      an unintended vector-index truncation, not a preference.
    - **`load_tree` is now a dispatcher** over `.npz`, `.mtr`/`.mat`,
      `.swc`, `.neu`, `.nmf` and `.asc`, with `save_tree` covering the
      writable subset plus the export-only `.hoc`, `.nrn` and `.xml`. Two
      things MATLAB's version does are deliberately not carried over: it
      **opens a file dialog** when called with no argument (fine for a
      GUI-first toolbox, unusable from a script, a headless notebook or a
      test), and it **silently applies `repair_tree`** to `.swc`/`.neu`/
      `.nmf`. Loading and repairing are kept apart so that "what does this
      file contain" has an answer. `save_tree` returns the path it actually
      wrote, since the extension is appended when missing.
    - **`.nmf` keeps region names, where MATLAB loses them.** Its writer
      stores only `tree.R`, the region indices, so a round trip renames
      `dendrite`/`subtree` to `1`/`2`. Here the names go into an HDF5 group
      *attribute*, which MATLAB's reader ignores — it iterates datasets —
      so the file stays readable there while round-tripping losslessly
      here.
    - **NEURON export is one module with three shapes**: `save_hoc(...,
      style="cell")` for MATLAB's `neuron_tree`, `style="template"` for
      `neuron_template_tree`, and `save_nrn` for the per-segment `.nrn`
      form. `neuron_bridge` is still the better route when NEURON is
      importable; these are for handing a cell to something outside Python.
    - **MATLAB's `.nrn` branch cannot run at all**, in two independent
      ways, both fixed: its single-region path reads a loop variable that
      is only assigned in the *other* branch of the `if`, and its `'-e'`
      block reads `tree.ri`/`tree.rm`/`tree.cm`, which the tree structure
      spells `Ri`/`Gm`/`Cm`. So there was no reference output to diff
      against for that format; it is tested against the structure
      `geometry()` requires — nine numbers per section, `create` sizes
      summing to the node count.
    - **Region names are sanitised into valid hoc identifiers in both hoc
      styles.** MATLAB does this only in `neuron_template_tree`;
      `neuron_tree` interpolates `[name '_' rnames{i}]` directly, so a
      region called `basal dendrite` writes hoc that will not parse. Names
      that *collide* once sanitised now raise, rather than silently merging
      two regions into one.
    - **`minterf` is a function, not a second return value.** MATLAB
      returns the T2N interface matrix as `neuron_template_tree`'s third
      output. Here it is `t2n_interface(tree)`, because it is a computation
      about the tree rather than a detail of writing a file, and a caller
      normally wants one or the other. It shares the section layout with
      the writer, and a test asserts it has exactly one row per `pt3dadd`
      the file contains.
    - **The hoc writer controls its own line endings.** MATLAB emits CR+LF
      explicitly; `Path.write_text` on Windows then translates the `\n` of
      an already-CRLF line into `\r\r\n`, giving NEURON a blank line
      between every statement. Written through an explicit
      `open(..., newline="")`, and asserted at byte level.
    - **NeuroML is built with `ElementTree`, not string concatenation.**
      Not a style preference: MATLAB's string building writes an unparseable
      document the moment a region name contains `&` or `"`, which a test
      now exercises. Three of its outputs are also wrong and are fixed —
      a `schemaLocation` concatenated into one token where the attribute
      needs two, root segments rewritten to be children of segment 0, and
      mixed CR+LF / LF line endings. One `<segmentGroup>` per region is
      **added**: MATLAB writes none, so its export cannot say which cable
      is axon and which is dendrite.
    - Segment ids are the distal node's index rather than MATLAB's
      `node - 2`, whose 1-based arithmetic yields id `-1` on any tree whose
      root is not node 1 — which its own `.neu` reader produces.
    - New optional dependency: `h5py` for `.nmf`, under a `[nmf]` extra. It
      already arrives with `[matlab]` (mat73 depends on it); the separate
      extra exists so that "I want `.nmf`" does not read as "I want MATLAB
      support".
