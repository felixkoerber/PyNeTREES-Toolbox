# TREES Toolbox → Python: Port Approach

This document is the *approach* — why the port is structured the way it is.
For the living checklist of what's built vs. outstanding, see [PORT_STATUS.md](PORT_STATUS.md).

## Source material this plan is based on

- `treestoolbox-master/` — MATLAB toolbox, organized by folder (`IO`, `construct`, `edit`,
  `electrotonics`, `graphical`, `graphtheory`, `gui`, `metrics`, `stacks`, `utilities`),
  each with a `Contents.m` index and a mirrored `tests/check_*` suite.
- `treestoolbox-master/classes/+trees/{Tree,Trees}.m` — an incomplete, in-progress OOP
  wrapper around the struct-based tree, using MATLAB `subsref` magic to dispatch
  `tree.something` to either a stored property or a `something_tree(tree, ...)` function.
- `To do list for TREES Toolbox.md` (repo root) — a maintainer-curated list of known bugs,
  incomplete functions, and open design questions.
- `T2N-master/` — MATLAB↔NEURON bridge, built by generating `.hoc` text from a tree struct.

## What the MATLAB code actually does (load-bearing facts)

- A tree is a plain struct: `dA` (sparse N×N adjacency matrix, directed child→parent),
  `X`, `Y`, `Z`, `D` (diameter), `R` (1-based index into `rnames`), `rnames` (region names),
  plus arbitrary extra fields bolted on by individual functions.
- `dA(i, idpar(i)) = 1` for every non-root node `i`; the root's row is all zero.
- Nearly every function opens with `ver_tree(intree)` — a validator that warns (does not
  throw) on malformed trees. It is the single most-depended-on function in the toolbox.
- Functions follow the convention `verb_tree(intree, options_string)`, where the options
  string is parsed by a homemade `parseArgs`/`inputParser`/`isBinary` combo. The todo list
  documents this parser as a recurring source of bugs (numeric-prefixed flags like `'2d'`,
  "only first matching case wins" switch bugs, missing options, etc.).
- A full `tests/check_*` suite mirrors the source layout (one check file per function,
  grouped by folder) — this is effectively an executable spec we can port function-by-function.
- The todo list explicitly names known-broken or incomplete functions/check-functions.
  These should be **fixed while porting**, not faithfully reproduced.

## Decisions carried into the Python design (see PORT_STATUS.md "Design Decisions" for the
running, dated log — this section only records the ones made *before* Phase 1 started)

1. **Don't port the options-string parser.** Replace `'-s'`, `'-z'`, `'-ks'` etc. with
   explicit typed keyword arguments (`show=False`, `keep_sections=False`, ...). This was
   flagged by the maintainers themselves as buggy, and Python has no reason to imitate it.
2. **Don't port the `classes/+trees` `subsref`-dispatch OOP layer.** It's an unfinished
   experiment (the todo list itself says "Tree and Trees — Merge!!"). The underlying idea
   (uniform `tree.thing` access) is good; the mechanism (runtime magic dispatch through
   `subsref`) is not something Python needs — attribute access is already dynamic.
   Instead: `Tree` is a plain, explicit data container; transformations stay free functions
   named `verb_tree(tree, ...)`, exactly mirroring MATLAB naming so muscle memory transfers.
   A thin convenience layer (methods, `__repr__`, plotting helpers) can sit on top later,
   but the data/behavior split stays function-oriented.
3. **0-based indexing throughout.** MATLAB is 1-based; Python/NumPy is 0-based. Every
   node-index quantity (parent indices, region indices, `dA` rows/columns) is 0-based in
   the port. This is a pervasive, easy-to-get-wrong conversion — every ported function
   must be re-derived against 0-based indexing, not mechanically transliterated.
4. **"No parent" sentinel changes from `0` to `-1`.** MATLAB uses `0` as a sentinel meaning
   "no parent" (since 0 is not a valid 1-based index), then in most functions immediately
   remaps root nodes to be their own parent. In 0-based Python indexing, `0` is a valid
   node index, so it cannot double as a sentinel. The port uses `-1` for "no parent"
   wherever MATLAB used the bare `0`-sentinel, and keeps the same "root is its own parent
   by default" convenience behavior where the MATLAB function provided it.
5. **Population-level tooling uses `list[Tree]` + `pandas`, not a custom container class.**
   The MATLAB `Trees` class reimplements group-by/filter/map/stats via `containers.Map` and
   `subsref` tricks. `pandas.DataFrame` already solves aggregate stats/filtering; a plain
   Python list already solves iteration/mapping. No bespoke population class planned.
6. **Plotting backend: decided in Phase 7 — `PyVista` (VTK-backed) primary, `matplotlib`
   secondary.** Candidates considered: `matplotlib` (2D/publication, cheap, no tube
   primitive, no real GPU acceleration for large 3D scenes), `PyVista` (VTK-backed,
   interactive 3D at scale, accepts per-point radius directly via its `tube()` filter),
   `Plotly` (notebook-friendly, weaker at scale). Measured on the bundled 2252-node
   reconstruction before committing: PyVista tube-mesh generation ~0.17s, full off-screen
   render another ~0.2s. See PORT_STATUS.md Design Decision #30 for the full reasoning
   and the headless-rendering verification.

## Phase hierarchy (bottom-up; each phase unlocks the next)

| Phase | Scope | Depends on |
|---|---|---|
| 0 | Project scaffolding: package layout, packaging config, test runner | — |
| 1 | Core data structure (`Tree`) + validation (`ver_tree`) + minimal SWC I/O + `sample_tree` | 0 |
| 2 | Graph primitives on `dA` alone (`idpar_tree`, `ipar_tree`, `child_tree`, `B_tree`, `T_tree`, `C_tree`, `sub_tree`, `dissect_tree`, `BO_tree`, `PL_tree`, `Pvec_tree`, `typeN_tree`, `sort_tree`) | 1 |
| 3 | Coordinate-based metrics (`len_tree`, `eucl_tree`, `surf_tree`, `vol_tree`, `angleB_tree`, `direction_tree`, `rootangle_tree`, `scale_tree`, `tran_tree`, `rot_tree`, `flatten_tree`) | 2 |
| 4 | Edit operations (`repair_tree`, `delete_tree`, `insert_tree`, `resample_tree`, `elim0_tree`, `elimt_tree`, `cat_tree`, `redirect_tree`) | 2, 3 |
| 5 | Full I/O (`swc_tree` write, `neurolucida_tree`, `save_tree`/`load_tree` replacement, NeuroML/x3d/pov exporters) | 4 |
| 6 | Construct (`MST_tree`, `BCT_tree`, `clean_tree`, `soma_tree`, ...) | 2, 3, 4 |
| 7 | Graphical / plotting (backend decision here) | 3 |
| 8 | Electrotonics (`M_tree`, `gi_tree`, `gm_tree`, `lambda_tree`, `sse_tree`, `syn_tree`, ...) | 1, 3 |
| 9 | Stats & comparison (`stats_tree`, `dstats_tree`, `sholl_tree`, population tooling) | 2–8 |
| 10 | GUI (low priority; likely a light Jupyter/PyVista viewer, not a GUIDE port) | 7 |
| 11 | T2N + NEURON integration via the Python `neuron` package | 1, 3, 8 |

Recommended first vertical slice (= Phase 0 + Phase 1): `Tree` + validation + SWC
round-trip + `sample_tree`, with ported tests. This is the minimum that lets a real
neuron reconstruction be loaded and inspected end-to-end, and it exercises the full
plumbing (packaging, data model, I/O, testing) that every later phase builds on.

## Testing strategy

- Port `tests/check_*/*.m` → `tests/test_*.py`, one function at a time, alongside the
  function itself (not as a separate backlog).
- Skip porting the check-functions the todo list already documents as broken/incomplete;
  fix the underlying function first, then write a correct test.
- Real-world fixtures: `treestoolbox-master/sample/swc/25HSS.swc` and
  `treestoolbox-master/tests/IO/test files/test02.swc` (the latter has out-of-order,
  non-contiguous node indices — a good stress test for the SWC parser) are copied into
  `python_port/tests/fixtures/` so the Python package doesn't reach across into the
  MATLAB tree for its own test data.
