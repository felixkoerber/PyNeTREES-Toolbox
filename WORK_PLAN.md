# Work plan — round 5

Started 2026-09-09, from the review notes on `REVIEW_ROADMAP.md`. Ordered so
that the things which touch every file happen before the things built on top
of them. Struck through as they land; the record of what actually changed goes
to `PORT_STATUS.md` and `CHANGELOG.md` as usual.

---

## A. Upstream diff — done first because it changes what "correct" means

The new TREES release (`treestoolbox-master(3)`) differs from the reference
this port was written against in **exactly four files**, and two of them
matter:

| File | Change | Consequence here |
|---|---|---|
| `metrics/convexity_tree.m` | `boundary(X,Y,Z,0)` → `,1)` **and** `c = 1 - nnz(Inds)/…` → `c = nnz(Inds)/…` | **Both bugs this port documented and deliberately deviated from are now fixed upstream.** The port's version and current MATLAB now agree in intent; re-validate numerically. |
| `metrics/dissectSholl_tree.m` | the `if rmax > 500; sf = 2; end` block is **deleted** | The port reproduces that fudge behind `scale_factor=`. Upstream has abandoned it, so the default must become 1.0. |
| `electrotonics/LIF_tree.m` | `'-t'` sigmoid `plset` commented out; the `if pars.e` guard around the progress `disp` also commented out | Feature disabled upstream, and the print is now unconditional. Port keeps both as options. |
| `utilities/histax.m` | `xax(:)'` orientation fix, copyright year | Cosmetic. |

**New upstream bug found in the process:** deleting the 500 µm block also
deleted `rmax = max(eucs(:))`, which the **3D** branch still uses at line 204.
`rmax` is now assigned only inside the mutually exclusive `dim2` branch, and
`dim3` defaults to `true` — so `dissectSholl_tree` raises on its own default
path. Goes to `MATLAB_TOOLBOX_BUGS.md`.

- [x] A1. Record the diff; fix `scale_factor` default; rewrite the
  `convexity_tree` note now that it describes a fixed bug.
- [x] A2. Re-validate `convexity_tree` against the new MATLAB — done in
  round 6, together with everything else that depends on `inpolyhedron`.
  Two findings. **The boundary is now exact**: `boundary.m` turned out to be
  a readable m-file, the port was anchoring `shrink = 1` on the wrong
  critical alpha, and fixing that puts `boundary_tree` on MATLAB's numbers
  digit for digit (see REVIEW_ROADMAP.md 1.1). **`convexity_tree` is still
  not worth matching**: the 2026 fix corrected the two lines this port had
  flagged, but the pair loop tests every point against itself, and those
  degenerate self-pairs always score as visible — on `sample_tree` the
  reported 0.0384615 is exactly `1/26` and *not one genuine pair* survived.
  Recorded as MATLAB_TOOLBOX_BUGS.md 1.13, with 1.14 for `-dim2` never
  reaching the 2D branch.

## B. API consistency — one batch, one test pass

Confirmed by the user: rename `region_mask()` → `region_nodes()` (returning
the mask) and drop the other two.

- [x] B1. `Tree.region_mask()` → `region_nodes()`; remove the old
  `region_nodes()` and `region_index()`; remove `Tree.validate()`.
- [x] B2. Already done in 0.1.4's V1 cleanup: `plot_mpl_tree` and `dA_tree`
  are the current names and there is no `plot_sholl`.
- [x] B3. `resample_tree(method="matlab")` → `"default"`;
  `bf_tree(params=)` → `values=`.
- [x] B4. `ShollResult` → `NamedTuple`.
- [x] B5. Already uniform: every public `dim` parameter is an integer.
- [x] B6. None left -- V1 removed them all.

## C. Corrections the user raised directly

- [x] C1. **Ragged populations are valid.** `[tree, [tree]]` currently
  raises. The user is right that this is over-strict: the natural reading is
  that a bare tree at a nested level is a group of one. Map it that way and
  return the same shape.
- [x] C2. **`soma_tree`.** MATLAB does **not** tie them together; `l` only
  *defaults* to `1.5 * maxD`. Its "corrected to match maxD and l" note means
  the profile now *attains* both -- the superseded formula peaked at
  `maxD / 2` and ran a full cosine period inside the region. The port already
  implements the corrected formula; docstring and tests now say so.
- [x] C3. **`insert_tree` should accept a whole SWC-format matrix**
  (`[X Y Z R iparent]` / `[inode R X Y Z D idpar]`) as well as the separate
  arrays.

## D. `-1` as a sentinel — evaluate, do not act yet

`-1` is used for "no parent", for padding in `ipar_tree`/`idchild_tree`, and
as a NaN stand-in. It is also a **valid NumPy index**, which is what makes it
dangerous. Deliverable is a written comparison of the options (masked arrays,
`np.nan` in float outputs, a distinct sentinel per role, object dtype, keeping
`-1` with harder guards), with a recommendation — not a change.

- [x] D1. `docs/sentinel-options.md`.

## E. Features

- [x] E1. **NEURON: emit `.hoc` for the current in-process setup.** `save_hoc`
  writes the morphology; this writes the morphology *plus* the mechanisms,
  parameters and stimuli that `GC_init`/`insert_mechanism` applied, so a
  session can be handed to plain `nrniv`. Tractable now that `GCSpec` holds
  the mechanism half in serialisable form.
- [x] E2. **`pov_tree` and `x3d_tree`.** Port both.
- [x] E3. **Blender is now part of `all`.** `.venv-blender` was created and
  the whole suite run in it: **963 passed, 16 skipped** on numpy 1.26 with
  bpy 5.0.1, including all 17 Blender tests. `pynetrees.blender` was never
  broken, only untestable. Extras restructured so `all` includes Blender and
  `all-no-blender` is the numpy-2 / Python-3.12+ variant.
- [x] E4. **`Tree.meta` survives MATLAB.** No v7.3 needed: MATLAB's TREES
  already hangs ad-hoc data straight off the tree struct (`tree.col`,
  `tree.NID`, `tree.Rho_soma`), so `meta` maps onto **top-level struct
  fields** in both directions -- flat, not nested, because that is where
  MATLAB's own functions look. Turned up that the bundled granule cells
  carry `NID` and `col`, which the loader had been dropping.
  The separate "large morphologies need v7.3" item is untouched and still
  open (REVIEW_ROADMAP §1.3).

## Deferred by agreement

- Connectomic-data additions — separate conversation.
