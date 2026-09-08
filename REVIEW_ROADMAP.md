# Review roadmap

What is left from `PyNeTREES code review.md`, in the order I would do it, with
an assessment of each. Written 2026-09-08, after the Round 4 pass recorded at
the bottom of [PORT_STATUS.md](PORT_STATUS.md).

Status of the suite: **932 passed, 17 skipped** (the 17 need `bpy`, which
cannot coexist with numpy 2 — see §5).

---

## Done in this pass

| Review item | Where |
|---|---|
| Ad-hoc items on the TREES object | `Tree.meta` |
| Length of tree as a class property | `Tree.total_length` — **already existed**, now documented |
| `with_coords()` → `copy()` | `Tree.copy(**fields)`, generalised to carry every field |
| Population nesting beyond two levels | `_population.py`, any depth, ragged input refused |
| `insert_mechanism` by node index | `nodes=` |
| Range equations `f(x)` for channel densities | callable parameters, `h.distance`-based |
| `GC_init` | new `pynetrees.gc_model`; table, morphologies and `.mod` sources all bundled |
| Warn on import about unmet requirements | `missing_dependencies()`, one `UserWarning` |
| MST identical to MATLAB | verified; multi-tree loop rewritten to take turns |
| Multiple outputs handled consistently | `dissect_tree(with_positions=)` → `full_output=` |
| Root parent indexing (`-1`) | audited (library was already clean) + `path_to_root` |
| `ipykernel < 7` | `pyproject.toml`, `requirements.txt` |
| Tutorial for the GC model | `examples/08_granule_cell_model.ipynb`, executed |
| Example scripts brought up to date | `04_neuron_simulation.ipynb`, `scripts/dendritic_democracy.ipynb` |

Four genuine bugs fell out of that work and are fixed: the editing functions
silently dropped `Ri`/`Gm`/`Cm`; `build_neuron_model` built multi-child-root
cells as electrically **isolated** section trees; `GC_init` applied the ion
blocks before the channels that create them, so every reversal potential was
dropped; and `nrn_load_dll` was called more than once per process, so building
a second model in a session aborted. All are described in PORT_STATUS.md
Round 4.

---

## 1. Highest value next — correctness the review can already name

### 1.1 `boundary_tree` against MATLAB (Round 3)

The most concrete open correctness gap, and measurements already exist. From
the 2026-08-28 differential session: enclosed volume is **exact** at shrink 0,
~2% low at the default 0.5, and **16.7% low** at shrink ≈0.23 on a concave
arbor (hsn). The reverse-engineered shrink → circumradius-cutoff curve is
calibrated near the convex-hull end and drifts loose as shrink decreases.

Four sub-items, in dependency order:

1. **Fit the shrink → alpha mapping** against MATLAB's `boundary()` on a grid
   of shrink values across `sample`, `hsn` and a pyramidal cell. The reference
   volumes are cheap to generate and the current curve is a guess.
2. **Always produce a connected boundary.** Sparse sampling (pyramidal cells)
   currently yields disconnected components; MATLAB does not.
3. **Match MATLAB's region-selection thresholds**, and **recompute convexity
   when it is not given**, as MATLAB does.
4. **Simplify complex boundaries into same-volume equivalents**, which MATLAB
   does automatically and the port does not (it returns the raw boundary).

Knock-on: `dissectSholl_tree.V` inherits the error directly, `theta_mc_tree`
and `r_mc_tree` inherit ~2% of it.

**Prerequisite** for anyone reproducing this: `inpolyhedron.m` is a File
Exchange dependency of `r_mc_tree`/`theta_mc_tree`/`growth_tree`'s `mc3d` and
is **not installed on this machine**. A ray-casting shim was written for the
August session; re-create it before running those harnesses.

### 1.2 Empty-tree support beyond `delete_tree` (Round 2)

The review is explicit and the pattern is already established
(`pynetrees._empty`): `len_tree`, `B_tree` and friends should return an empty
output for an empty tree rather than raising. Population code depends on this
— an empty cell must keep its slot without poisoning the whole group. Mostly
mechanical: extend `@empty_safe` coverage and add the parameterised test.

### 1.3 `save_tree` for large morphologies (Round 2)

`.mtr` writing is MATLAB v5, which caps file size. The review's requirement is
that a large morphology must survive a round trip. Options, in preference
order: write v7.3 via `h5py` (the format is documented and `mat73` already
reads it); or fall back to `.swc` with a loud, explicit message. Silently
truncating is the one unacceptable outcome.

---

## 2. API consistency — cheap, and it gets cheaper the sooner it is done

The port has no external users yet, so these are free now and expensive later.
The review is right that no deprecation shims are warranted.

| Change | Note |
|---|---|
| Remove `Tree.region_nodes()`/`region_index()`; rename `region_mask()` → `region_nodes()` | **Needs your confirmation.** As written this leaves one method named `region_nodes` that returns a *boolean mask*. That is what MATLAB-shaped code wants (`tree.R == idx`), but the name then contradicts the return. I would rather keep both and rename for accuracy — say which you want. |
| Remove `Tree.validate()` | Agreed; `ver_tree(tree)` is the same thing. |
| `plot_tree_mpl` → `plot_mpl_tree`, `dA_tree_mpl` → `dA_tree` | Mechanical. |
| `plot_sholl` → `plot_sholl_tree` | Mechanical. |
| `resample_tree(method="matlab")` → `method="default"` | Mechanical. |
| `bf_tree(params=)` → `values=` | Mechanical. |
| Drop every deprecation shim | `spread_trees`, the old `dim` spellings, `no_root`, `params`. |
| `ShollResult` → NamedTuple | Round 2. The other result types already are. |
| One `dim` convention everywhere | Pick `dim=2`/`dim=3` (integer) and apply it across `chull_tree`, `vonMises_tree`, `hull_tree`. Currently a mix of `dim2=` booleans and strings. |

Doing these as one commit with one test pass is much less work than eight
separate ones.

---

## 3. Population tooling, second half (Round 2/3)

Nesting is now unlimited, but two pieces remain:

- **`pairwise=` for two-tree functions.** `peters_tree`, `cat_tree`,
  `share_boundary_tree`, `ssecat_tree` currently refuse populations because
  "a list of trees" is ambiguous. The review resolves the ambiguity: default
  `pairwise=True` (all pairs), `pairwise=False` runs element-wise and raises
  on a nesting mismatch. Now implementable, since `_children` already
  validates shape.
- **`plot_tree` on a list of trees.** Currently one tree at a time.

---

## 4. `MST_tree` leftovers (Round 3)

Multi-tree growth now matches MATLAB exactly, which was the substantive item.
Remaining, all small:

- **Intermediate steps.** `record=True` returns the growth log; reconstructing
  a tree at step *k* from a prefix of that log is a ~20-line helper that does
  not exist yet. The review asks for the trees, not the log — build them from
  the record rather than storing them.
- **`_seed_tree` reuse.** The review asks why it is a separate function used
  once. The answer should be that `growth_tree`/`local_growth_tree` use it
  too; make that true or inline it.
- **Plotting.** Post-hoc is fine, per the review.

---

## 5. Blender — still relevant, and blocked on one pin

`pynetrees.blender` is **complete**: `build_tree`, `save_blend`,
`render_tree`, one curve object per region, one POLY spline per section, real
per-node taper. It supersedes `pov_tree`/`x3d_tree` by writing a `.blend`
(the start of a pipeline) rather than a `.pov` (the end of one).

It cannot be exercised in the main environment, and this is not going to
resolve itself: **`bpy==5.0.1`, the newest release, still requires
`numpy<2.0,>=1.26` and `python==3.11.*`** (checked today). This venv has numpy
2.4.6, so the 17 skipped tests are all Blender.

Recommendation: **keep it out of the default install** (it already is) and
work on it in a dedicated environment:

```bash
python -m venv .venv-blender && .venv-blender/Scripts/activate
pip install -e ".[blender,dev]"      # pulls numpy 1.26
pytest tests/test_blender.py
```

Round 2 also asks for `pov_tree`/`x3d_tree` themselves. My view: only worth
it if you need to hand a scene to a collaborator who runs POV-Ray or a
browser-based X3D viewer. Otherwise the Blender path strictly dominates. Say
if the interop matters and it goes on the list.

---

## 6. Ports still missing (Round 2)

- **The rest of the generative pipeline**, excluding `fix_tree`,
  `finetune_fix_tree` and `fix_tree_UI` as the review directs.
- **`cyl_tree`'s `-dA` option.** MATLAB's own comment calls it "SLOW!!" and
  nothing else in the toolbox uses it; cheap to add for completeness.
- **Functions newly added to the TREES GitHub** since the port started —
  needs a diff of the upstream repo against `NOT_YET_PORTED.md` first.
- **`rot_tree`'s PCA / `-m3d` / `-al` modes.** Needs region-plane fitting.
- **`ipar_tree`'s `-T` option** (paths from termination point to first branch
  point only) — worth confirming anything wants it before porting.

---

## 7. NEURON, remaining (Round 3)

- **Generate the `.hoc` for the current in-process setup.** `save_hoc` writes
  the *morphology*; the review wants the morphology **plus** the mechanisms,
  parameters and stimuli that `GC_init`/`insert_mechanism` just applied, so a
  session can be handed to plain `nrniv`. Now much more tractable than it was:
  `GCSpec` is the mechanism half in serialisable form already, so this is
  "walk the spec and the point processes, emit hoc" rather than "re-derive
  what is in the model".
- **The remaining `GC_biophys.m` variants.** The published *active* model is
  now bundled as `GC_BIOPHYS` and is the default. The other option strings
  (`-k`, `-y`, `-ra`, `-ms`, AH99) are still only reachable by scaling values
  by hand. Dumping each from MATLAB once and adding them beside `GC_BIOPHYS`
  would make `gc_spec("newborn")` work, and `GCSpec.diff` would then answer
  what each flag changes — the question the 656-line original cannot. Cheap
  now that the mechanism for it exists; it is a MATLAB run per variant.

---

## 8. Open questions from the review I could not resolve alone

- **`soma_tree`: is `maxD` matched to `l`?** Needs a MATLAB run comparing the
  two arguments across a sweep; I have not done it.
- **`insert_tree` taking a whole SWC-format matrix.** Straightforward to add
  as an alternative first argument. Worth doing only if you actually have
  SWC-shaped input in hand — say the word.
- **The 500 um fudge in `dissectSholl_tree`.** To restate it plainly: MATLAB's
  3-D branch **doubles** the estimated mean branch length for cells reaching
  past 500 um, with no comment, no docstring mention, and no 2-D counterpart.
  It is almost certainly a hack fitted to one dataset. The port reproduces it
  so published numbers stay reproducible, but exposes it as `scale_factor=`,
  and `scale_factor=1.0` turns it off. Same treatment for its 2-D-only
  first-bin extrapolation of the root-angle histogram.
- **Useful additions for connectomic data.** Worth a separate conversation;
  the obvious candidates are contact/apposition detection between large
  populations (`peters_tree` at scale), synapse-density mapping onto
  morphology, and cross-population overlap statistics.

---

## Suggested order

1. §1.2 empty-tree support and §2 the rename batch — both cheap, both get more
   expensive the longer they wait.
2. §1.1 `boundary_tree` — the real correctness debt, and the measurements to
   drive it already exist.
3. §1.3 large `.mtr` — a data-loss bug in waiting.
4. §3 population `pairwise=` and `plot_tree` over lists.
5. §7 hoc generation + the variant JSONs, together.
6. §6 the remaining ports, §4 MST leftovers, §5 Blender in its own env.
