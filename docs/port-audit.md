# Port audit — faithfulness, performance, bugs

A systematic review of the `pytrees` port against the MATLAB TREES toolbox:
does it behave the same, is it idiomatic, where is MATLAB faster, and what's
broken. Everything below was measured or reproduced, not inferred.

Method: (1) static scan of all 118 public functions for docstring quality and
loop nesting; (2) an edge-case sweep running 52 functions against 6 degenerate
trees; (3) scaling profile at two tree sizes to separate algorithmic problems
from constant factors; (4) targeted differential checks against MATLAB source.

---

## 1. Bugs found and fixed

### 1.1 `sub_tree` was accidentally quadratic — 285× slower than necessary

`sub_tree` read each node's children as `dA[:, node].toarray()`, materialising
a **dense length-`n_nodes` column per visited node**. One BFS over a 3765-node
granule cell took **514 ms**; it should be ~2 ms.

This propagated: `asym_tree`, `repair_tree` and `clean_tree` all call it in a
loop. Rewritten to walk prebuilt child lists.

| | before | after |
|---|---|---|
| `sub_tree` (whole tree) | 514.1 ms | **1.8 ms** (285×) |
| `asym_tree` | 1624.3 ms | **60.3 ms** (27×) |

Verified identical on 41 sampled nodes of the real reconstruction.

### 1.2 `chull_tree` crashed on any planar tree

Only "too few points" was guarded, not *degenerate* geometry. A coplanar point
set has no 3D hull, and Qhull raises a bare `QhullError` from deep inside
SciPy.

This is not an exotic input:

- many reconstructions are traced in 2D with `Z == 0`;
- **`flatten_tree` produces a planar tree by construction** — so the toolbox
  generated input that crashed another of its own functions;
- `stats_tree(extras=True)` calls `chull_tree`, so **all extended statistics
  failed on any flat morphology**.

Now returns `(points, None)`, matching the documented "no hull possible"
contract, and the docstring points at `dim2=True` (the 2D hull, which is what
you actually want there).

### 1.3 `resample_tree` failed on a single-node tree

`dissect_tree` correctly reports *no sections* for a lone node (the root is
never a section end), so the rebuild produced a zero-node tree and then died
inside `sort_tree` with `expected exactly one root, found 0` — an error
pointing nowhere near the cause. Also broke `rootangle_tree`, which resamples
internally. Now returns the tree unchanged.

### 1.4 `stats_tree` reported multifurcation errors as someone else's problem

`angleB_tree`/`asym_tree` deliberately refuse non-binary branch points
(Design Decision #15 — correct, and matching MATLAB's documented
precondition). But `stats_tree` inherited that exception verbatim, so asking
for statistics produced an error about `angleB_tree`.

Multifurcations are common in real data — the bundled NeuroLucida sample has
**24 across 8 trees** as loaded. The error now names `stats_tree`, says which
tree, and tells you to run `repair_tree`.

---

## 2. Where MATLAB is faster — and why

The honest answer is **nowhere algorithmically, but MATLAB's idioms don't
survive transliteration**. Three functions were literal ports of MATLAB's
sparse matrix-power recursions. That idiom is natural and fast in MATLAB,
where sparse matrix algebra is the native vocabulary. In SciPy each iteration
is a Python-level call whose overhead dominates, and the loop runs once per
tree *level* — 1624 times on a granule cell.

| function | MATLAB idiom | ported cost | rewritten | speedup |
|---|---|---|---|---|
| `LO_tree` | sparse matrix powers until root column empties | 544 ms | 10.9 ms | **50×** |
| `PL_tree` | one sparse matvec per depth level | 37.1 ms | 4.8 ms | **7.7×** |
| `Pvec_tree` | sum over the dense `ipar` matrix | 165.4 ms | 5.1 ms | **32×** |

Each was replaced by the O(n) recurrence it actually expresses:

- `PL[node] = PL[parent] + 1` (node depth)
- `Pvec[node] = Pvec[parent] + v[node]`
- `LO[node] = PL[node] + Σ PL over descendants`

The `LO_tree` identity was **verified exactly** (max abs difference `0.0`) on
hand-built trees and both bundled reconstructions before replacing the
transliteration — this function was previously left as a literal port
specifically because re-deriving it was judged risky.

Knock-on effects:

| | before | after |
|---|---|---|
| `repair_tree` | 545.2 ms | **22.3 ms** (24×) |
| `clean_tree` | 572.8 ms | **27.4 ms** (21×) |
| `resample_tree` | 274.2 ms | **24.4 ms** (11×) |
| `stats_tree` | 193.0 ms | **40.3 ms** (4.8×) |
| whole test suite | ~31 s | **14 s** |

### Where MATLAB genuinely retains an edge

- **`ipar_tree`** builds a dense `n_nodes × max_depth` matrix (49 MB for the
  granule cell) and remains 3.1× superlinear. This mirrors MATLAB's own data
  structure, and MATLAB's dense-array performance is better. The port
  mitigates it by not *using* `ipar_tree` where a traversal suffices —
  `Pvec_tree`, `PL_tree`, `flatten_tree`, `morph_tree` and `smooth_tree` were
  all moved off it. It is still the right tool for genuine ancestor queries.
- **Tight elementwise numeric loops** (`jitter_tree` at 653 ms, `smooth_tree`
  at 176 ms) do per-node work that MATLAB's JIT handles better than
  interpreted Python. Both remain the slowest functions; neither is
  algorithmically wrong.

---

## 3. Faithfulness

Function names, argument order and semantics track MATLAB closely enough that
existing knowledge transfers — see [matlab-migration.md](matlab-migration.md).
Deliberate divergences are all recorded in `PORT_STATUS.md`'s design log with
reasoning, and every one is either a bug fix or a documented simplification:

- **Bug fixes vs. MATLAB**: `rootangle_tree` (measured from the origin, not
  the root), `delete_tree` (MATLAB's default splitting is documented-broken),
  `clean_tree` (root wrongly deletable), `jitter_tree` (self-distance of 2 as
  a method artefact), `dissect_tree` (region cut on the wrong node — a bug
  this port introduced and later fixed), plus 10 MATLAB bugs catalogued in
  `MATLAB_TOOLBOX_BUGS.md`.
- **Interface modernisation**: option strings → typed keywords, no global
  `trees` array, no `'-s'` side-effect plotting, `stats_tree` → DataFrames.

**Verified in this audit:** the degenerate-input sweep (52 functions × 6
degenerate trees) produced no unexplained failures after the fixes above. The
remaining non-`ok` results are all correct behaviour: `NaN` at non-branch
points, and documented `ValueError`s on non-binary branch points.

---

## 4. Docstrings

All 118 public functions have docstrings. The scan flagged:

- **7 "missing"** — all nested helper closures (`push_from`, `err`, `prepend`,
  `parse_list`, `model`) and the `n_nodes` property; not API surface.
- **6 thin (<8 words)** — one-liners on functions whose behaviour is fully
  described by name and return type (`len_tree`, `vol_tree`, `T_tree`).
  Expanded where a unit or convention was implicit.
- **5 with undocumented parameters** — noted; `Pvec_tree`, `PL_tree`,
  `LO_tree`, `sub_tree` and `chull_tree` docstrings were rewritten during this
  audit to explain both what they compute and why the implementation differs
  from MATLAB.

Docstrings now state units (`[um]`, `[S/cm^2]`), index conventions (0-based,
`-1` sentinel), and — where the port diverges — what MATLAB does differently.

---

## 5. Regression coverage added

- 8 invariant tests pinning the rewritten functions to their *definitions*
  rather than their implementations (e.g. `LO_tree == PL + child_tree(PL)`,
  `Pvec_tree` vs. an explicit ancestor walk, `sub_tree` vs. the parent chain).
- 3 tests for the planar-hull fix, including `stats_tree(extras=True)`.
- 2 tests for the single-node `resample_tree`/`rootangle_tree` fix.

Suite: **222 passing**, down from ~31 s to ~14 s.

---

## 6. Remaining known limitations

| Item | Status |
|---|---|
| `ipar_tree` dense matrix, 3.1× superlinear | Inherent to the ported data structure; avoided where possible |
| `jitter_tree` (653 ms), `smooth_tree` (176 ms) | Per-node Python loops; correct but interpreter-bound |
| `quaddiameter_tree` 1.7× superlinear | Not investigated |
| `sse_tree()` full inverse | O(n²) memory by definition; pass `I=node` for one site |
| Single-node trees in `sse_tree`/`M_tree` | Return non-finite values rather than raising |
