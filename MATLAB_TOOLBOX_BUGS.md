# Bugs found in the MATLAB TREES Toolbox

A catalog of concrete bugs, self-acknowledged incomplete features, and
correctness gaps found in `treestoolbox-master/` while building the Python
port (`python_port/`). This is a companion to two other documents, not a
replacement for either:

- **`To do list for TREES Toolbox.md`** (repo root) — the maintainers' own
  running list of known problems. Several entries below cite and confirm
  items from that list with concrete evidence (a source excerpt, a
  reproduction case, or both); the rest are findings made independently
  while porting every function to Python (Phases 0–7, see
  `python_port/PORT_STATUS.md`).
- **`python_port/PORT_STATUS.md`** — the "Design Decisions Log" there
  records, for each bug that mattered to the port, exactly how the Python
  version handles it (fixed, deliberately not replicated, or — rarely —
  faithfully preserved because a downstream behavior depends on it). Each
  entry below cross-references its Design Decision number where one exists.

**Methodology**: every claim here was checked against the actual source in
`treestoolbox-master/` (file + line cited), and where practical, reproduced
by tracing the algorithm by hand or exercising it during the Python port's
own testing. This is not a static-analysis sweep — it's what surfaced
naturally from reimplementing and testing every function against both
hand-built cases and the bundled real reconstructions.

---

## 1. Confirmed logic bugs (silently produce wrong results)

### 1.1 `delete_tree` — forest-splitting only works with non-default options

**File**: `edit/delete_tree.m`

Deleting a node normally splices its children onto its own parent
(`append_children`, true unless the `'-x'` option is given). The
forest-splitting logic that's supposed to handle "deleting a branching root
produces multiple disconnected trees" is gated on `~append_children`
(`edit/delete_tree.m:141`) — i.e. it **only** runs when `'-x'` is given,
which also means splicing is *off*. With the default options (splicing on,
which is the entire point of calling `delete_tree` normally), deleting a
branching root leaves `tree.dA` with multiple all-zero rows (multiple
roots) and returns it as a single, structurally-invalid tree — never as
the cell array of separate trees the situation actually calls for.

Confirmed directly in the maintainers' own todo list: *"delete_tree |
multiple trees doesn't work yet"*.

**Python port**: `edit.py`'s `delete_tree` always splices to the nearest
*surviving* ancestor and always returns a `list[Tree]` when that
disconnects the tree, regardless of any flag — see PORT_STATUS.md Design
Decision #22.

### 1.2 `clean_tree` — ambiguous/empty range when no branch point precedes a terminal

**File**: `construct/clean_tree.m:77-78`

```matlab
ibranch  = find  (abs (typeN (1 : iT (counter) - 1) - 1), ...
    1, 'last') + 1 : iT (counter);
```

When every node between the root and a given terminal is a continuation
point (`typeN == 1`) — i.e. there is no branch/terminal point anywhere
before it, which happens whenever the root has exactly one child and the
first real branch point is further down — `find(..., 1, 'last')` returns
`[]`. `[] + 1` is still `[]` in MATLAB, and `[] : iT(counter)` is a
degenerate colon expression whose behavior is at best fragile and at worst
an error, not the intended "branch starts right after the root" fallback.

Reproduced while porting: a synthetic tree with a healthy long branch and a
short stub both attached directly to a single-child root triggered exactly
this path; the fallback needed to explicitly treat the root as an implicit
boundary (see below) rather than fall through to whatever `[]:N` happens to
evaluate to in a given MATLAB version.

**Python port**: `construct.py`'s `clean_tree` makes the root an explicit,
never-deletable boundary — PORT_STATUS.md Design Decision #28 has the full
trace, including a second test case where this exact bug caused the
*wrong* branch's length to be compared against `radius`.

### 1.3 `insert_tree` — region-renumbering is broken by the author's own admission

**File**: `IO/insert_tree.m:90-94` (comment left in place by the original author):

```matlab
% my god! Handling regions is not easy!!!!!!
% AND IS WRONG!!!!! IF FIRST REGION DOES NOT EXIST, THERE IS A
% SHIFT OF REGION NAMES THAT ARE DELETED..!!!!
```

The `'-d'` option (delete obsolete regions after inserting new points) has
a known-wrong region-renumbering path, left in the source with the comment
above rather than fixed. Also flagged in the todo list under `insert_tree`.

**Python port**: `edit.py`'s `insert_tree` drops the MATLAB `[inode R X Y Z
D idpar]` SWC-tuple calling convention entirely in favor of explicit
`X, Y, Z, D, parent, R` arrays, and region bookkeeping is just "use the
given `R` or default to the parent's region" — there's no region-elimination
step to get wrong.

### 1.4 `elimt_tree` — `ntrif` is a boolean, not a count, despite its docstring

**File**: `edit/elimt_tree.m:24` (docstring) vs. `edit/elimt_tree.m:152`:

```matlab
% - ntrif    :: number of trifurcations          <- docstring promise
...
ntrif = ~isempty (itrif);                        <- actual implementation
```

`ntrif` is documented as "number of trifurcations" but is actually `0` or
`1` (whether *any* were found), not a count. Any caller relying on the
documented contract (e.g. to report "fixed N trifurcations") gets a wrong
number silently — no error, just `1` instead of the real count.

**Python port**: `edit.py`'s `elimt_tree` returns `(tree, changed)` where
`changed` is explicitly documented as a boolean, matching what the MATLAB
code actually returns rather than what it claims to return.

### 1.5 `pointer_tree` — mutually-exclusive option dispatch doesn't make sense

**File**: `graphical/pointer_tree.m:90-151`

The option handling is a chain of `if pars.v ... elseif pars.l ... elseif
pars.s ... elseif pars.o ... else ...`, meaning only the *first* true flag
in that fixed priority order ever takes effect — passing e.g. both `'-s'`
and `'-o'` silently drops one of them with no warning, and the priority
order itself has no documented rationale. Confirmed in the todo list:
*"pointer_tree | the switch case in does not make sense. '-O' is not
really used and it's only for 'otherwise'. Also its implementation means
that only the first true option will apply and it does not really make
sense to have them separate from each other."*

**Python port**: `plotting.py`'s `pointer_tree` takes a single
`style="marker"|"sphere"` argument — one explicit choice, not five
overlapping boolean flags with an undocumented precedence order.

### 1.6 `rootangle_tree` — measures distance from the coordinate origin, not the root

**File**: `metrics/rootangle_tree.m:57`

```matlab
eucl             = sqrt (X2.^2 + Y2.^2 + Z2.^2);
```

The function is documented as computing "the angle between its segment and
the straight line to the root," but this line computes distance from
`(0, 0, 0)` — the coordinate origin — not from `tree.X(1)`/`tree.Y(1)`/
`tree.Z(1)`, the tree's actual root position. The function never calls
`tran_tree` (or otherwise re-centers) first. For any tree not already
sitting exactly at the origin, every angle this function returns is simply
wrong relative to what the docstring promises.

**Python port**: `edit.py`'s `rootangle_tree` explicitly calls `tran_tree`
to center on the root before measuring — PORT_STATUS.md Design Decision
#24.

### 1.7 `LIF_tree` — the `Vzone` parameter has zero effect

**File**: `electrotonics/LIF_tree.m:61,189`

```matlab
p.addParameter('Vzone', 0.995)
...
%         v     (v     (ireset, counterT + 1) > Vzone * thr, counterT + 1) = ...
%             vreset;   % reset voltage
```

`Vzone` is parsed as a real parameter with a documented default (`0.995`),
but the *only* place it's referenced anywhere in the function is inside a
commented-out line. A caller who passes a different `Vzone` value gets
exactly the same behavior as the default — the parameter is silently a
no-op, not documented as such.

**Python port**: `electrotonics.py`'s `LIF_tree` drops `Vzone` entirely
rather than porting a parameter that can never affect the output —
PORT_STATUS.md Design Decision #34.

### 1.8 `AdExLIF_tree` — the `Vrest` parameter has zero effect

**File**: `electrotonics/AdExLIF_tree.m:94-96`

```matlab
if ~isfield (tree, 'Vrest')
tree.Vrest       = -70;
end
```

`Vrest` is documented (*"Resting potential {DEFAULT: -70 mV}"*) and given a
default value here, but `tree.Vrest` is never read anywhere else in the
function — the resting potential the actual dynamics settle around is
governed entirely by `EL`. Same class of issue as `Vzone` above: a
plausible-looking, documented, settable parameter that provably cannot
change the output.

**Python port**: `electrotonics.py`'s `AdExLIF_tree` drops `Vrest` — see
PORT_STATUS.md Design Decision #34.

### 1.9 `AdExLIF_tree` — returned voltage trace ignores `iroot`

**File**: `electrotonics/AdExLIF_tree.m:193`

```matlab
v                = v (1, :);
```

The function computes a full `(N, T)` voltage trace internally but returns
only row 1 (node 1, 1-based) — hardcoded, regardless of what `iroot` was
actually set to. For the default `iroot = 1` this happens to return the
right node; for any other `iroot`, the caller silently gets a different
node's trace than the one where the spike mechanism was actually inserted.

**Python port**: `electrotonics.py`'s `AdExLIF_tree` returns the full
`(n_nodes, len(time))` trace instead (matching `LIF_tree`'s own contract),
letting the caller slice `v[iroot]` themselves — PORT_STATUS.md Design
Decision #34.

### 1.10 `boundary_tree` — crashes on its own documented default call path

**File**: `metrics/boundary_tree.m:49`

```matlab
if isempty (pars.c)
    % {DEFAULT: convexity unknown}
    pars         = convexity_tree (intree, 'dim2', pars.dim2, 'dim3', pars.dim3);
end
...
if pars.dim2 % Two-dimensional case
```

`pars` is the `inputParser` result struct (holding `pars.dim2`, `pars.dim3`,
etc.). This line reassigns `pars` to the *return value of `convexity_tree`*
— a bare scalar (the convexity value `c`) — destroying the struct entirely.
Every subsequent `pars.dim2`/`pars.dim3` reference (lines 54, 58, 71) then
tries to access a field on a plain `double`, which errors in MATLAB. This
triggers whenever the caller doesn't supply `c` explicitly — the
documented default (*"{DEFAULT: Unknown, calculated using
convexity_tree}"*) and the only example call in the function's own
docstring (`boundary_tree (sample_tree, '-dim3')`). The evident intent was
`pars.c = convexity_tree(...)`, assigning just the convexity field.

**Python port**: `boundary_tree` is now ported (B2) and simply cannot fail
this way — `c` is a real keyword argument and computing it when absent
assigns to a local, not over the parameter block. The port keeps MATLAB's
`c` spelling (`shrink = 1 - c`) alongside a direct `shrink=` so callers can
say what they mean. Design Decision #61 supersedes #35 for this cluster.

---

## 2. Self-acknowledged incomplete or non-functional features

These are cases where the MATLAB source itself — via a comment, a docstring
caveat, or the maintainers' todo list — documents that a feature doesn't
work, without the function refusing to run or warning the caller at call
time.

| Function | File | What's acknowledged broken |
|---|---|---|
| `MST_tree` | `construct/MST_tree.m:33` | `mplen` (max path length parameter) docstring: *"(doesn't really work yet..)"* |
| `isBCT_tree` | `construct/isBCT_tree.m:9` | *"NOTE! does not always work (doesn't check for trifurcations...)"* |
| `xdend_tree` | `graphical/xdend_tree.m:68` | *"Now if you want to build a standard tree that disregards the existing spatial embedding use this, but this doesn't seem to work..."* (the optional "equivalent tree" second output) |
| `dissect_tree` | `metrics/dissect_tree.m:28` | *"NOTE! this function isn't completely correct yet at the root!"* |
| `insertp_tree` | `edit/insertp_tree.m` docstring | Documents `'-p'` (path length to direct parent) and `'-pr'` (+ relative position) options; neither `pars.p` nor `pars.pr` is referenced anywhere in the function body (verified: zero matches) — purely aspirational documentation. Also flagged in the todo list. |
| `asym_tree` | `graphtheory/asym_tree.m` | Todo list: *"-m does not work, nanmean replace"* (the `'-m'` explanatory-movie option, and a deprecated `nanmean` call site) |
| `boundary_tree` | `metrics/boundary_tree.m` | Todo list: *"What is the -s option doing in boundary_tree"* (unclear/possibly non-functional) |
| `share_boundary_tree` | `metrics/share_boundary_tree.m` | Todo list: *"has some bugs and unused stuff, make TREES function, check convexity_set function?"* |
| `stats_tree` | `metrics/stats_tree.m` | Todo list: the `'-x'` option is broken |
| `load_tree` | `IO/load_tree.m` | Todo list: *"Does not work for .neu files"* |
| `loaddir_stack` | `stacks/loaddir_stack.m` | Todo list: `imread` error |
| `rot_tree` | `metrics/rot_tree.m` | Todo list: *"check regexpi??"* (suspicious regex usage in the `'-al'` region-alignment path) |

Cross-reference: every function in this table that the Python port actually
implements either fixes the gap outright or explicitly documents *not*
porting the broken feature — see the per-function notes in
`python_port/PORT_STATUS.md`'s phase tables.

---

## 3. Numerical robustness gaps

### 3.1 Unclamped `acos` of a dot product of normalized vectors

**Files**: `metrics/angleB_tree.m:76`, `graphtheory/asym_tree.m` (same
pattern), `metrics/rootangle_tree.m:64-70`

```matlab
angleB (counter) = acos (dot (nV1, nV2));
```

`nV1`/`nV2` are unit vectors, so `dot(nV1, nV2)` should lie in `[-1, 1]` —
but floating-point rounding can push it very slightly outside that range
(e.g. `1.0000000000000002`), at which point `acos` returns a complex
number rather than raising an error. `angleB_tree` has no clamp and no
`real()`/`isnan` cleanup, so a nearly-parallel or nearly-antiparallel
branch pair can silently produce a complex-valued "angle" that then
propagates into anything downstream (a mean, a color scale, a histogram)
without any indication something went wrong.

Telling corroboration: `rootangle_tree` *does* have a workaround for
exactly this failure mode two lines later —
`rootangle(isnan(rootangle)) = 0; rootangle = real(rootangle);` — showing
the maintainers hit this exact issue at least once, but the fix was never
applied to the other functions with the identical pattern.

**Python port**: every `acos`/`arccos` call site in `graphtheory.py`,
`metrics.py`, and `edit.py` (`asym_tree`, `angleB_tree`, `rootangle_tree`)
clips the input to `[-1, 1]` via `np.clip` before calling `np.arccos`.

### 3.2 `jitter_tree`'s node-to-itself distance is an unintended artifact, not a choice

**File**: `construct/jitter_tree.m:98-104`

Topological "distance" for the noise-smoothing kernel is computed by
testing, for ascending `k = 1, 2, 3, ...`, whether `A^k * indicator > 0`
(a walk-of-length-`k` existence test) and taking the first `k` where a node
lights up. On a tree (no cycles), a walk of length `k` between `u` and `v`
exists iff `k >= shortest_path(u, v)` **and** `k - shortest_path(u, v)` is
even — so a node's distance to *itself* (true shortest path 0) isn't
detected until `k = 2` (the first even, positive `k`), not `k = 0`. Nothing
in the surrounding code or comments suggests this was intentional; it's a
side effect of using a matrix-power walk test to approximate BFS distance.

**Python port**: `construct.py`'s `jitter_tree` uses direct BFS, giving the
intuitively-correct self-distance of 0 — PORT_STATUS.md Design Decision
#29 has the full derivation of why the MATLAB version lands on 2.

---

## 4. Performance issues acknowledged in the source

### 4.1 `plot_tree`'s per-segment SVD is a self-documented bottleneck

**File**: `graphical/plot_tree.m:383-389`

```matlab
for counter  = 1 : N
    % singular value decomposition
    v        = null     (dP (counter, :)); %%% BOTTLENECK
    ...
```

Building cylinder geometry calls `null()` (a full SVD) once per tree
segment, in a loop, and the source comment names it a bottleneck directly.
For a multi-thousand-segment reconstruction this dominates render time; the
docstring separately warns that the line-mode fallback (`'-2l'`/`'-3l'`) is
slower still.

**Python port**: `plotting.py`'s `plot_tree` builds one PyVista
`tube()`-filtered mesh for the whole tree — no per-segment loop, no SVD.
Measured on the bundled 2252-node/2251-segment reconstruction: ~0.17s for
the whole mesh (vs. an O(segment count) SVD loop in MATLAB). See
PORT_STATUS.md Design Decision #30 for the full comparison.

### 4.2 `MST_tree`'s vicinity-window bookkeeping exists only to avoid an O(n²) MATLAB cost

**File**: `construct/MST_tree.m` (whole-file characteristic, not one line)

Not a bug, but worth recording alongside 4.1: roughly 600 lines of
hand-maintained, per-tree "vicinity window" sorting/re-slicing exist purely
to avoid recomputing all-pairs distances every iteration in native MATLAB.
The algorithm itself (greedy, path-length-balanced nearest-neighbor growth)
is sound; the scaffolding around it is where the complexity lives.

**Python port**: `construct.py`'s `MST_tree` gets the same practical
performance from `scipy.spatial.cKDTree` (radius queries) plus a standard
lazy-deletion min-heap (Prim's-algorithm pattern) — see PORT_STATUS.md
Design Decision #27.

---

## 5. Minor / cosmetic issues (recorded for completeness, low impact)

- **`isBinary`** (`utilities/isBinary.m`) — todo list: *"Works only for
  scalars, might be mistaken for a MATLAB function?"* (naming collision
  risk plus a scalar-only limitation).
- **`dA_tree`** (`graphical/dA_tree.m`) — todo list notes an
  over-strict `numel(intree) > 1` input check was already removed upstream
  (recorded as resolved, not an open bug).
- **`idpar_tree`** (`graphtheory/idpar_tree.m`) — todo list: *"'z'
  option"* flagged with no further detail; likely refers to ambiguity
  around the `'-z'` (root-self-reference) flag's naming history (the
  todo list elsewhere notes it "used to be called '-0'").
- **`check_soma_tree`** (test suite) — todo list: *"Wrong measures
  (apparently)"* — the toolbox's own validation test for `soma_tree`
  is flagged as producing incorrect reference measurements.
- **`check_pov_patch`** (test suite) — todo list: *"Test 3 does not
  work - incorrect input?"*
- **`histax`** (`utilities/histax.m`) and everything that depends on it
  (`gdens_tree`, `bin_tree`'s histogram path, `skel_stack`,
  `neuron_template_tree`, `neuron_tree`, `gifmaker`) — todo list:
  *"Possibly a transpose error"*, meaning the suspected bug's blast radius
  covers every one of those dependents, not just `histax` itself.

---

## Summary

| Category | Count | Typical fix in the Python port |
|---|---|---|
| Confirmed logic bugs | 30 | Redesigned around the bug (not replicated) |
| Self-acknowledged incomplete features | 12 | Either fixed, or explicitly not ported (documented) |
| Numerical robustness gaps | 2 | Explicit clamping / correct distance metric |
| Performance bottlenecks (self-documented) | 2 | Vectorized / replaced with a standard algorithm + library |
| Minor/cosmetic | 16 | N/A (informational) |

None of this is a knock on the original toolbox — `treestoolbox-master/` is
a substantial, working, widely-used piece of scientific software, and the
maintainers' own todo list shows they were already tracking most of this.
The point of this document is narrower: everywhere the Python port's
behavior visibly diverges from a literal line-by-line translation of the
MATLAB source, it should be traceable to *either* a deliberate design
decision (see `python_port/PORT_STATUS.md`) *or* one of the bugs cataloged
here — not an accident of the port itself.

---

## Found while porting the remaining MATLAB options (2026-08-18)

These three surfaced during W3, all confirmed by running the MATLAB source
in Octave 11 rather than by reading it.

### `soma_tree(..., '-b')` crashes on any tree whose root has one child

**Reproduced**, on the toolbox's own `sample_tree`:

```
>> soma_tree (sample_tree, 30, [], '-b')
error: dr(nan,_): subscripts must be either integers 1 to (2^63)-1 or logicals
```

`soma_tree.m` guards the branch-angle test with

```matlab
if 1 < numel (idchild_tree (tree, 1))
```

but `idchild_tree` always returns a **fixed-width, NaN-padded** matrix --
`idchild_tree(sample_tree, 1)` is `[2 NaN]` even though the root has exactly
one child. So `numel(...) == 2` is always true, the guard never fires, and
the next line indexes `dr(ch(2), :)` = `dr(NaN, :)`.

A single-child root is the *common* case (a soma leading into one primary
neurite), so the `'-b'` overlap correction is effectively unusable in
MATLAB. The port tests the number of real children instead, and the
correction works: on `hss_tree` with `maxD=120` it reduces total surface by
46%, which is the whole point of the option.

Two ported functions already avoid the underlying cause: this port's
`idchild_tree` sizes its output to the widest node actually found instead of
hardcoding 2 (so it never invents a padded entry at width 2), and
`Tree.root` is used rather than a hardcoded node 1.

### `ipar_tree`'s `'-T'` cannot be reached by the documented call

The docstring shows `ipar_tree (sample_tree, [], '-s')`, i.e. options as the
*third* argument, but the parser registers only `ipart` as positional:

```matlab
pars = parseArgs (p, varargin, {'ipart'}, {'T', 's'});
```

so `ipar_tree(tree, '-T')` silently binds `'-T'` to `ipart` and returns a
2x27 matrix of nonsense rather than the 26x37 terminal-path matrix. The
working invocation is `ipar_tree(tree, 'T', true)`. The port takes a real
keyword, `ipar_tree(tree, terminals_only=True)`, so there is nothing to get
wrong.

### `dissect_tree`'s own docstring: "isn't completely correct yet at the root"

Already noted in the port (Design Decision #36) but worth listing here for
completeness, since the second output `vec` inherits the same root handling:
MATLAB prepends a fake root (`[0; Pvec_tree(tree)]`) and then slices it back
off with `vec(3:end, :)`, plus a manual `vec(1, 2) = 0` fix-up. The port
computes per-node section positions directly from its own root-clean
sections, so no prepend-and-slice dance is needed.


---

## Found while porting the spatial statistics (2026-08-19)

Four more, all in the `convexity_tree` / `boundary_tree` / `r_mc_tree` /
`dissectSholl_tree` cluster (B2). The first is the most consequential:
it makes a published measure return something other than what it names.

### `convexity_tree`'s 3D branch contradicts both its docstring and its own 2D branch

**File**: `metrics/convexity_tree.m:55` and `:105` vs `:115` and `:173`

The docstring defines convexity as *"the proportion of direct paths between
termination points of a tree that lie entirely within the **tightest**
boundary that can be drawn around said tree."* The two branches implement
different things, and only the 2D one matches that sentence.

```matlab
% 3D branch                       % 2D branch
[k, ~] = boundary(X, Y, Z, 0);    [k, ~] = boundary(X, Y, 1);
...                               ...
c = 1 - nnz(Inds)/(nS1 * nS2);    c = nnz(Inds)/(nS1 * nS2);
```

Two independent defects, both visible without running anything:

1. **Wrong boundary.** MathWorks documents shrink factor `0` as the
   *convex hull* and `1` as the tightest enveloping shape. The 3D branch
   asks for `0` — the loosest boundary available, not the tightest. Every
   straight segment between points inside a convex hull is inside it by
   construction, so the test is close to vacuous. The 2D branch correctly
   asks for `1`.
2. **Inverted sign.** `Inds(i)` is set when the search finished without
   finding an intersection (`if t == 1`), i.e. for the pairs that *stayed
   inside*. The 2D branch returns that fraction; the 3D branch returns one
   minus it. They cannot both match a docstring that names one quantity.

Between them, the 3D return value is approximately the fraction of terminal
pairs having an endpoint *on* the convex hull (endpoints lying on a hull
facet register as an intersection at `X(1) = 0`) — a measure of how many
terminals happen to be extremal, which is not convexity in any sense.

Not executed here: Octave has no `boundary`, so this is established from
the source rather than by reproduction. That is sufficient — the 2D/3D
disagreement is a textual fact about the file.

**Python port**: `convexity_tree` deliberately does **not** reproduce this.
It tests visibility against the space-filling hull (`hull_tree`), which is
the standard definition and the only version that separates a compact
arbor from a lobed one; it returns the fraction *inside*, matching the
documented sense and the 2D branch. Design Decision #61.

### `r_mc_tree`'s volume-correction flag is inverted, and its documented default is wrong too

**File**: `metrics/r_mc_tree.m:11`, `:29` and `:130`

The option list says:

```matlab
%     	'-nv' : no volume correction
```

and the header says *"By default, a volume correction is applied to prevent
the R value from being positively biased."* The code:

```matlab
if pars.nv % volume correction
```

so the flag whose name and documentation both mean *disable* is what
**enables** the correction, and since `nv` defaults to `false`, the
documented default behaviour never happens. Both halves of the
documentation are contradicted by that one line. The correction matters:
without it a finite Monte-Carlo sample never quite reaches the boundary, so
its apparent volume is too small, its expected nearest-neighbour distance
too short, and `R` biased upward — exactly the bias the header says is
being prevented. Measured on `sample_tree` in the port: `R` = 0.57 with the
correction, 0.62 without.

**Python port**: follows the documented intent — `volume_correction=True`
by default, `False` to switch off. Design Decision #61.

### `dissectSholl_tree` doubles its branch-length estimate above 500 µm, in 3D only, with no explanation

**File**: `metrics/dissectSholl_tree.m:205-206` and `:247`

```matlab
if rmax > 500
    sf = 2;
end
...
S = sf * tL / (bp); % Estimated branch length
```

`sf` is initialised to 1 at the top of the function and set to 2 only in
the 3D branch, only for cells reaching past 500 µm. There is no comment,
no mention in the docstring, and no counterpart in the 2D branch (which
applies an entirely different adjustment, `S = S * max(rV)`). A hard
discontinuity at a round number, applied to one branch of a published
measure, is the shape of a fudge factor left in after a fit.

**Python port**: reproduced for fidelity but surfaced as
`dissectSholl_tree(..., scale_factor=)`; pass `1.0` to disable.

### `dissectSholl_tree` patches its first root-angle bin in 2D and in `Estscale`, but not in 3D

**File**: `metrics/dissectSholl_tree.m:155` and `:470` vs `:253`

```matlab
rVraw(1)  = rVraw(2) + (rVraw(2) - rVraw(3));   % 2D branch, and Estscale
rV        = rVraw / trapz(tV, rVraw);           % 3D branch: no such line
```

The first root-angle bin is empty (no segment runs exactly along the line
to the root), so it is linearly extrapolated from its two neighbours — in
the 2D branch and in the internal `Estscale` helper, but not in the 3D
branch, even though `Estscale` is called from both. So a 3D call normalises
one distribution one way and the other distribution the other way, within
the same invocation.

**Python port**: reproduced as-is, and documented in the function's Notes,
because changing it would silently move published numbers. Flagged here so
that it is a decision rather than an accident.

### Minor: `dissectSholl_tree`'s output fields are not the ones it documents

**File**: `metrics/dissectSholl_tree.m:29` and `:41` vs `:88` and `:304`

The header documents `'tL'` and `'estScale'`; the code writes
`Output.TotalLength` and `Output.EstScale`. A caller following the
docstring gets a missing-field error.

**Python port**: `ShollDissection` is a NamedTuple, so the field names are
part of the type and cannot drift from the documentation.


---

## Found while porting the file formats (2026-08-19)

B3: the `.neu` reader, `.nmf`, the NEURON exporters and NeuroML. The first
two below were **reproduced by running the MATLAB source**, which is worth
noting because several of this section's neighbours could only be
established by reading it.

### `load_tree`'s `.neu` branch crashes unless the root is the first node

**File**: `IO/load_tree.m:237-241`

Having built `parid` (1-based parent per node, `-1` at roots), the
single-tree branch does:

```matlab
N        = size (swc, 1);
dA       = sparse (N, N);
for counter = 2 : N
    dA (counter, parid (counter)) = 1;
end
```

The loop starts at 2 because node 1 is assumed to be the root, whose
`parid` is `-1` and therefore unusable as a column index. But nothing in
the format puts the root first: a `.neu` file lists NEURON sections in
whatever order the model declared them, and the root section can appear
anywhere. When it does, some `counter >= 2` has `parid (counter) == -1`
and the assignment fails.

**Reproduced** on `tests/IO/test_neu_tree/GC1.neu` — one of the three
fixtures the toolbox ships *for this very function*. Its `soma[0]` section
is tenth of sixty, putting the root at node 306:

```
n roots (parid==-1) = 1, at 306
single-tree dA FAILED: dA(_,-1): subscripts must be either integers 1 to (2^63)-1 or logicals
```

(Run under Octave with the file parsing rewritten — Octave's `textscan`
differs from MATLAB's — and everything from `d = zeros (nsec, 1)` onward
copied verbatim, so the arithmetic under test is MATLAB's.)

Note the multi-tree branch just above has the mirror-image assumption: it
slices nodes into trees by `treelimits`, i.e. it assumes each tree occupies
one *contiguous* block starting at a root. Interleaved cells would be cut
apart at the wrong places.

**Python port**: `load_neu` finds roots wherever they are, walks each one's
subtree, and returns one `Tree` per root — verified to produce parent
indices **bit-identical to MATLAB's** on all three fixtures, and geometry
identical to 0.0, which is the same arithmetic evaluated without the
indexing assumption. Design Decision #62.

### `load_tree`'s `.neu` region stripping discards compound section names

**File**: `IO/load_tree.m:173-175`

```matlab
insa     = strfind (sa, '[');
if ~isempty (insa)
    sa   = [(sa (1 : insa - 1)) '[]'];
end
```

`strfind` returns a **vector** of every `[` position; MATLAB's colon
operator silently uses only its first element, so the name is truncated at
the *first* bracket. For a plain `axon[0]` that is the intent. For
NEURON's compound names it is not: `GCT.neu`'s ninety sections
(`GC7[0].adendGCL[3]`, `GC7[0].soma[0]`, `GC7[0].axon[1]`, ...) all become
a single region called `GC7[]`, and every anatomical label in the file is
lost. Octave flags the construct at runtime — *warning: colon arguments
should be scalars* — which is how this surfaced.

**Python port**: each bracketed index is blanked in place, giving
`GC7[].adendGCL[]`, `GC7[].soma[]`, ... — 7 regions instead of 1, and
identical to MATLAB's result for the simple names in the other two
fixtures.

### `neu_tree.hoc` writes a `# 3d points:` count that is not the number of 3D points

**File**: `IO/neu_tree.hoc` (the NEURON-side writer), visible in every
fixture under `tests/IO/test_neu_tree/`

| file | declared | actual | sections | points in first section |
|---|---|---|---|---|
| `GC1.neu` | 780 | 1214 | 60 | 13 |
| `GC.neu` | 875 | 3220 | 35 | 25 |
| `GCT.neu` | 180 | 6800 | 90 | 2 |

The declared value is `sections x points-in-one-section` in all three
cases: the writer multiplied where it should have summed. MATLAB's reader
is unaffected only because it discards the number and reads numbers to
end-of-file.

**Python port**: the section table's per-section counts are authoritative —
they are self-consistent and sum to the real total — and the header is
checked only to raise a `UserWarning`.

### `neuron_tree`'s `.nrn` branch cannot export a single-region tree

**File**: `IO/neuron_tree.m:136-146`

```matlab
if luR       > 1
    for counterR = 1 : luR
        ...
else
    fwrite (neuron, [ ...
        'create ' name '[' (num2str (H1 (counterR))) ']', ...
```

`counterR` is the loop variable of the `for` inside the `if` arm. In the
`else` arm — reached exactly when there is one region — it was never
assigned, so the export raises *Undefined function or variable
'counterR'*. One region is the common case for anything not hand-labelled.

**Python port**: `save_nrn` numbers regions uniformly and has no
single-region special case to get wrong.

### `neuron_tree`'s `.nrn` `'-e'` option reads fields that do not exist

**File**: `IO/neuron_tree.m:257-261`

```matlab
    (num2str (tree.ri)),         nextline], 'char');
    (num2str (1 ./ tree.rm)),    nextline], 'char');
    (num2str (tree.cm * 1e6)),   nextline], 'char');
```

The TREES tree structure spells its passive parameters `Ri`, `Gm` and
`Cm`; `ri`, `rm` and `cm` exist nowhere in the toolbox. So
`neuron_tree (tree, 'x.nrn', [], '-e')` fails on a missing field. The
`.hoc` branch of the same function, two hundred lines later, uses the
correct names — so this is a stale copy, not a second convention.

**Python port**: reads `Ri`/`Gm`/`Cm`, and raises a clear error up front if
the tree does not carry them, rather than writing a file that silently
lacks the parameters that were asked for.

### `neuroml_tree`'s NeuroML 2 `schemaLocation` is a single unusable token

**File**: `IO/neuroml_tree.m:93-94`

```matlab
fwrite  (nmlfile, ['    xsi:schemaLocation="http://www.neuroml.org/schema/neuroml2', ...
    'http://neuroml.svn.sourceforge.net/viewvc/neuroml/DemoVer2.0/lems/Schemas/NeuroML2/NeuroML_v2alpha.xsd"', ...
```

`xsi:schemaLocation` is a whitespace-separated list of *pairs*: namespace,
then schema URL. The two strings here are concatenated with no separator,
giving `"...neuroml2http://neuroml.svn..."` — one token where two are
required, so no validator can resolve the schema. (The SourceForge SVN
viewer it points at has also not existed for years.) The v1 branch eleven
lines above gets this right, with a trailing space inside the first string.

**Python port**: writes the two as a proper pair, against the current
NeuroML 2 schema location.

### `neuroml_tree` declares root segments to be children of segment 0

**File**: `IO/neuroml_tree.m:129-132`

```matlab
parentid  = idpar0 (ward) -2;
if (parentid == -1)
    parentid = 0;
end
```

`parentid == -1` means "this segment's proximal point is the tree's root",
i.e. it has no parent segment. Rewriting that to `0` makes it a child of
the first segment written. For a tree whose root has a single child the two
coincide; for any tree whose root branches, the second and later root
segments are attached to a segment they do not touch.

**Python port**: such a segment carries no `<parent>` element, which is how
NeuroML denotes the start of a cell.

### Minor: `neuron_tree`'s documented `res` argument does nothing in `.hoc`

**File**: `IO/neuron_tree.m:17`, `:82`, `:487`

`res` is documented as "number of segments per compartment", defaults to
`ceil (len_tree (tree))`, and is computed on every call. The `.hoc` branch
never reads it — the `proc geom_nseg()` it emits is empty — so only `.nrn`
is affected.

**Python port**: `res` is a parameter of `save_nrn` alone, rather than an
argument on `save_hoc` that quietly does nothing.

### Minor: `neuroml_tree` mixes line endings within one file

**File**: `IO/neuroml_tree.m` throughout

Element lines are written with `fwrite (..., [..., char(13), newline])`
(CR+LF); the `<proximal>`, `<distal>` and `</segment>` lines use `fprintf`
with `\n` (LF). Harmless to an XML parser, but it makes the files noisy in
version control and inconsistent with what the function evidently intends.


---

## Found while porting the generative pipeline (2026-08-20)

B4: `gscale_tree`, `clone_tree`, `rpoints_tree`, `dscam_tree`,
`spines_tree`, `PP_generator_tree`. `spines_tree` accounts for four of
these on its own.

### `spines_tree`'s documented coordinate input cannot be reached

**File**: `construct/spines_tree.m:84-86`

```matlab
if     numel (pars.XYZ) == 1
    indy         = ceil (rand (pars.XYZ, 1) * length (pars.ipart));
elseif all   (pars.XYZ < N) % they are indices
    indy         = pars.XYZ;
end
```

The header documents ``XYZ`` as *"::matrix: [X Y Z] or just a number of
spines to add"*, and the whole first paragraph of the description is about
attaching a spine at given coordinates. But the dispatch has only two arms
and no third:

- an ``(n, 3)`` coordinate matrix whose values all happen to be smaller
  than the node count — any cell traced near the origin — takes the
  **indices** arm and is silently reinterpreted as node numbers;
- an ``(n, 3)`` matrix with any value at or above the node count matches
  neither arm, so `indy` is never assigned and the next line,
  `dXYZ = zeros (numel (indy), 3)`, raises *Undefined function or variable
  'indy'*.

Either way the documented input does not work. Note also that the test is
on *magnitude*, so whether a coordinate matrix is treated as coordinates or
as indices depends on where the cell sits in space.

**Python port**: dispatch is on **shape** — a 2D ``(n, 3)`` array is
coordinates, a 1D integer array is indices, a scalar is a count — so all
three documented forms work and none can be confused for another.

### `spines_tree`'s `'-sr'` option reads an unassigned variable

**File**: `construct/spines_tree.m:107-120`

```matlab
r    = find (strcmpi (tree.rnames, 'spine_neck'));
if ~isempty (r)
    iR (1) = r (1);
else
    iR (1) = max (max (tree.R), numel (tree.rnames)) + 1;
    flag   = 1;
end
r          = find (strcmpi (tree.rnames, 'spine_head'));
if ~isempty (r)
    iR (2) = r (1);
else
    iR (2) = max (max (tree.R), numel (tree.rnames)) + 1 + flag;
end
```

`flag` is assigned only in the first `else`. Call `spines_tree` with
`'-sr'` on a tree that already has a `spine_neck` region but no
`spine_head` — the natural case when spining a tree twice with different
parameters — and the last line raises.

### `spines_tree` returns two numbers where it documents two arrays

**File**: `construct/spines_tree.m:157`, `:163`

```matlab
for counter = 1 : size (pars.XYZ, 1)
    [tree, indneck]    = insert_tree (tree, ...
    [tree, indhead]    = insert_tree (tree, ...
end
```

Both are overwritten on every pass, so the function's second and third
outputs — documented as *"node indices of spine heads"* and *"of spine
necks"* — are the indices of the **last spine only**. Anything wanting to
select the spines afterwards has to recompute them from the region labels.

**Python port**: `spines_tree(..., full_output=True)` returns the complete
arrays.

### `spines_tree` puts the head on the wrong side of a negative-length neck

**File**: `construct/spines_tree.m:100-104`, `:163-168`

The neck offset is `randn * stdlneck + mlneck` — a normal draw, which at
the function's own defaults (mean 1, standard deviation 1) is **negative
about 16% of the time**. The head is then placed unconditionally at
`XYZ + dXYZ * dhead`, i.e. always on the `+dXYZ` side. So for those spines
the neck points one way from the dendrite and the head the other: the head
lands between the neck and the cable, or inside the cable itself. Surface
area and any spine-head distance measured off such a tree are wrong.

**Python port**: the *direction* is flipped when the draw comes out
negative, not the length. Since the direction is uniformly random around
the cable to begin with, that is distributionally identical and leaves the
head beyond the neck. A test asserts head and neck are collinear and
pointing the same way for every spine.

### Minor: `spines_tree`'s region index is a vector, not a scalar

**File**: `construct/spines_tree.m:126`, `:131`

```matlab
iR     = max (tree.R,numel (tree.rnames)) + 1;
```

Two-argument `max` in MATLAB is **elementwise**, so this returns a vector
as long as `tree.R`, not the intended scalar. The `'-sr'` branch twelve
lines above writes `max (max (tree.R), numel (tree.rnames))`, which is
correct. The consequence is masked — `numel (iR) == 1` is then false, the
`iR (2) = iR (1)` fallback is skipped, and `iR (1)` happens to hold the
right value whenever `numel (rnames) >= R (1)` — so it works by accident on
ordinary trees.

### `gscale_tree` deletes empty regions by an index that shifts underneath it

**File**: `construct/gscale_tree.m:207-217`

```matlab
emptyregion      = find (dR);
for counterR     = 1 : length (emptyregion)
    spanning.regions (emptyregion (counterR)) = [];
    ...
```

`emptyregion` holds ascending indices into the original list, but each
deletion shifts everything after it down by one. With one empty region this
is correct; with two or more, the second deletion removes the wrong entry —
and with the last region empty, it can index past the end. A group whose
cells declare two or more region names that no node uses is enough to
trigger it, and the bundled `dLPTCs.mtr` is one region name away from it
already (all fifteen cells declare a `soma` region that no node is assigned
to).

**Python port**: unused regions are simply never added, so there is nothing
to delete.

### Minor: `gscale_tree`'s two outputs rescale about different centres

**File**: `construct/gscale_tree.m:271-283`

```matlab
spanning.X{r}{t} = xmass + mxdiff * (Xpre - xmass) / diff (xlims);   % points
ctrees{t}.X(iR)  =         mxdiff *  ctrees{t}.X(iR) / diff (xlims); % trees
```

The rescaled **point cloud** is scaled about the region's own centre of
mass, so that centre stays put; the rescaled **tree** is scaled about the
origin, i.e. the root. Both are returned from the same call as though they
were the same rescaling, and they are not — a region's points will not
coincide with the same region of the corresponding scaled tree. It looks
deliberate (the point cloud is the density target, the trees are for
display), so the port keeps it, but the docstring says so out loud.

### `PP_generator_tree` can loop forever

**File**: `construct/PP_generator_tree.m:155`, `:304`

```matlab
while ((Ract < pars.R - 0.01) || (Ract > pars.R + 0.01))
```

No iteration bound, in either of the function's two copies of the search.
Many targets are simply unreachable — R is capped by how tightly the
exclusion zone and the fixed 200 um box let the points pack — and asking
for one hangs MATLAB with no output and no way to tell whether it is
converging. (`'-e'` echoes the current and target R each pass, which is the
only reason this is survivable in practice.)

**Python port**: `max_iter=200` by default, and giving up raises a
`UserWarning` naming the R actually reached.

### Minor: `dscam_tree` can pick a partner it just excluded

**File**: `construct/dscam_tree.m:75`

```matlab
iClose       = find (distance == min (distance (iVector)));
iClose       = iClose (1);
```

The minimum is taken over the eligible nodes, but the `find` that locates
it searches the **whole** distance vector. If an excluded node — an
ancestor of the start node, or one inside its own subtree — sits at exactly
that distance and comes first, it is selected instead, and the branch is
moved toward a part of itself. Exact ties are not as rare as they look
here: `iVector` excludes everything within 2 um, so the surviving distances
cluster tightly around that cutoff.

**Python port**: the partner is taken from the masked set directly.


---

## Found while porting the last plot helpers (2026-08-20)

### `xplore_tree` labels regions by loop position, not by region value

**File**: `graphical/xplore_tree.m:73-80`

```matlab
uR           = unique (tree.R);
for counter  = 1 : length(uR)
    if isfield   (tree, 'rnames')
        rname    = tree.rnames {counter};
    ...
    iR       = find (tree.R == uR (counter));
```

The region being drawn is `uR (counter)`, but the name written next to it
is `rnames {counter}` -- the *position in the loop*, not the region's own
value. The two coincide only when the regions in use are exactly
`1 : length (uR)`. Any tree that has had a region deleted, or that uses
region 2 without using region 1 -- which is what
`delete_tree`/`sub_tree`/an SWC import with non-contiguous type codes all
produce -- gets its labels attached to the wrong regions, silently, in a
figure whose entire purpose is to tell you which region is which.

**Python port**: `xplore_tree(tree, mode="regions")` indexes `rnames` by
the region value.

### Minor: `plotsect_tree` draws nothing when the path is not directed

**File**: `graphical/plotsect_tree.m:63-65`

```matlab
indy = pars.ipar (pars.sect (1, 2), ...
    1 : find (pars.ipar (pars.sect (1, 2), :) == pars.sect (1, 1)));
```

The docstring requires *"a directed path away from the root"*. When the
start node is not actually an ancestor of the end node, `find` returns
empty, `1 : []` is an empty range, and the function plots an empty line and
returns normally. The caller gets a figure with nothing added to it and no
indication of why.

**Python port**: raises, naming both nodes.


---

## Found while porting the image stacks (2026-08-20)

### `fitD_stack` measures every diameter at one point, and says so itself

**File**: `stacks/fitD_stack.m:124-126`

```matlab
% TODO, CRITICAL: RIGHT NOW ONLY THE TERMINAL POINT IS TAKEN
mPX          = [(P1 (1) + cV (1)) (P1 (1) + cV (1)) (P2 (1))];
mPY          = [(P1 (2) + cV (2)) (P1 (2) + cV (2)) (P2 (2))];
```

Three sampling positions along the segment are constructed, and all three
are the same point: `cV` is `P2 - P1`, so `P1 + cV` **is** `P2`, and the
third entry is `P2` outright. Every diameter is therefore read from a
perpendicular profile at the segment's far end rather than along the cable
the surrounding code was written to traverse -- the sampling grid it builds
next, `repmat (mPX, 2 * maxR + 1, 1)`, has three identical columns.

The author flagged it. It is listed here because the comment is inside the
function and nothing in its documentation or its output says the
measurement is single-point.

**Python port**: `fitD_stack(..., samples=5)` spreads the positions along
the segment as intended; `samples=1` reproduces MATLAB.

Worth adding, because measuring it contradicted the expectation: averaging
along a segment is **not** automatically less noisy. Near a branch point
the perpendicular profile picks up the *sibling* branch, and on a clean
synthetic phantom the single-point measurement came out the *less* variable
of the two (spread 1.2 against 1.8 voxels). The defect is that the code
does not do what it says, not that its number is necessarily worse.

### Minor: `fitD_stack` returns voxels where the caller expects microns

**File**: `stacks/fitD_stack.m:175, :179`

```matlab
d            = (m_2 - m_1);
...
D (counter)  = d;
```

`m_1` and `m_2` are indices into a profile sampled at one-voxel steps along
the perpendicular, so `d` is a width in **voxels**. It is returned as `D`,
whose only use is to be assigned to `tree.D` -- a field in **microns**
everywhere else in the toolbox, and the field every length, surface and
volume function reads. The two agree only when the in-plane voxel happens
to be 1 um across, which is common enough for the mismatch to go unnoticed
and wrong enough to matter on any other acquisition.

**Python port**: the width is scaled by the length of one sampling step in
microns, which is exact even for anisotropic voxels.

### Minor: `skel_stack`'s default threshold is a fixed voxel count

**File**: `stacks/skel_stack.m:50-55`

```matlab
c        = histax (reshape (double (iM), numel (iM), 1), ...);
cc       = cumsum (flipud (c));
ic       = find (cc > 30000, 1);
thr      = (99 - ic) * double ((maxM - minM) / 99) + minM;
```

The default threshold is whatever intensity leaves roughly **30000 voxels**
above it. That is an absolute count, not a fraction, so it means something
different for every stack size: on a small crop it keeps most of the
volume, on a large one a thin sliver of the brightest cable. Nothing in the
documentation mentions it, and the threshold is the single parameter that
decides what the reconstruction sees.

**Python port**: Otsu's method by default -- documented, reproducible, and
scale-free -- with an explicit `thr` still available.

### Minor: `fitD_stack`'s two edge indices are offset by one

**File**: `stacks/fitD_stack.m:172-173`

```matlab
m_1          = max (q (find (q < 0))) +  i_max;
m_2          = min (q (find (q > 0))) + (i_max - 1);
```

Both come from the same `diff`-shortened array and both are re-anchored to
`i_max`, but one gets `i_max` and the other `i_max - 1`, so every width is
one sampling step narrower than the gap between the edges it found. It
comes from `diff` shortening the array by one and being corrected on only
one side.

**Python port**: reproduced, deliberately. A sub-voxel systematic offset in
an already-approximate measurement is not worth silently moving numbers
people have published over; it is noted in `fitD_stack`'s docstring.

## Found while porting the persistence functions (2026-08-26)

### `BLO_tree` orders branches by node count, not by length

**File**: `graphtheory/BLO_tree.m:75`

The function is called *branch **length** order*, its header says it
"returns the primary branches by longest first", and its `V` argument is
documented as "values to be integrated to select longest path". It selects
with:

```matlab
[~, i2]      = max (sum (V0 (ipar + 1) > 0, 2));
```

`sum (... > 0, 2)` **counts** the path nodes carrying a positive value. It
never sums `V`. Two things follow, both measured against the toolbox's own
bundled trees:

- **Branch 1 is the path with the most nodes, not the longest path.** On
  `hsn` MATLAB's first branch ends 319.5 um from the root while the
  furthest tip is at 648.4 um — less than halfway.
- **`V` selects nothing.** Any strictly positive `V` gives a bit-identical
  ordering, so passing `eucl_tree`, `ones` or `len_tree` changes only the
  lengths reported, never the decomposition. The documented meta-function
  behaviour does not exist.

The two rules disagree about where **69% to 97%** of nodes belong
(`sample` 69%, `hss` 81%, `hsn` 97%), so this is not a tie-breaking
detail. It propagates into `barcode_tree`, `persistenceimage_tree` and
`realisations_tree`, which are all built on the ordering.

Whether the count rule is *worse* is a separate question this port does not
claim to have settled: resampling each bundled tree to 1 um and comparing
barcodes, the count-based ordering was, if anything, the more stable of the
two. It is simply not what the function says it does.

**Python port**: `BLO_tree(..., by="nodes")` is the default and reproduces
MATLAB exactly, node for node, so barcodes and published analyses
reproduce. `by="length"` maximises accumulated `v`, which is what the name
and documentation describe. Both are documented in the function's Notes.

### Minor: `persistenceimage_tree` counts coincident bars once

**File**: `graphtheory/persistenceimage_tree.m:82`

```matlab
M (sub2ind (size (M), round (death), round (birth) + 1)) = 1;
```

Assignment, not accumulation, so any number of branches whose births and
deaths round to the same micron contribute a single 1 between them. The
published method (Kanari et al., 2018) sums one kernel per bar. Across the
55 cells in `dLPTCs.mtr` this silently drops a median of **1.5% of bars,
at worst 4.0%**, and it falls hardest on the densely branched cells — the
ones a density image exists to tell apart.

**Python port**: accumulates by default; `accumulate=False` reproduces
MATLAB's figures exactly.

### Minor: `persistenceimage_tree` offsets its two axes differently

**File**: `graphtheory/persistenceimage_tree.m:82`

`round (death)` against `round (birth) + 1` — the same coordinate treated
two ways, which shifts the whole image one pixel off the diagonal it is
plotted against. Invisible under the 17.5 um kernel and identical for every
cell, so it changes no comparison; it matters only if coordinates are read
off the image.

**Python port**: both axes share an origin.

### Minor: `BLO_tree` loops forever on a tree whose remaining segments are all zero-length

**File**: `graphtheory/BLO_tree.m:74-85`

When every remaining path scores zero, `max` returns row 1. If node 1 has
already been consumed its row is all zeros, `branch` comes back empty,
`ipar (ismember (ipar, branch)) = 0` changes nothing, and the `while` loop
never terminates.

**Verified by running it**, not by reading it. Four nodes are enough: a
chain `1 -> 2 -> 3` plus a node 4 duplicating node 3's coordinates
exactly, which any reconstruction containing a repeated point produces.
`len_tree` is `[0 10 10 0]`. Step 1 takes the branch `[3 2 1]` and zeroes
it; node 4's row is then `[4 0 0 0]`, which keeps `sum (sum (ipar))`
non-zero, but node 4 carries `V = 0` so every row now scores 0, `max`
returns row 1, and row 1 is already empty:

```
  step 1: picked row 3, branch = [3 2 1]
  step 2: picked row 1, branch = []
  -> branch is empty; nothing gets zeroed this iteration
  ... identical forever
```

Since `barcode_tree`, `persistenceimage_tree` and `realisations_tree` all
call it, they hang too.

**Python port**: the decomposition is driven by a frontier of unassigned
branch heads rather than by rescanning `ipar`, so it terminates by
construction — there is no state in which it has work left and cannot make
progress. On the tree above it returns `order = [1 1 1 2]`, giving the
zero-length tip its own empty bar `[20, 20]`, which is the right answer:
the branch exists and has no length.

## Found while porting the space-filling functions (2026-08-26)

### `theta_tree` returns a bin index where a distance is meant

**File**: `metrics/theta_tree.m:47-50`

```matlab
xax   = 0 : ceil  (max   (hB));
y     = cumsum    (histax (hB, xax));
theta = find      (y > 0.9069 * S, 1, 'first');
```

`find` returns a **1-based position** in `xax`, and `xax` starts at 0, so
the value returned is one micron larger than the radius it stands for. The
function then uses it as a distance — the `'-s'` branch thresholds the real
distance map with `BW < theta` and contours it — so the plot is drawn one
micron out too. Writing `xax (find (...))` would have been the fix.

Verified against MATLAB's own code under Octave: on `sample` it returns
**10** where the covering radius is **9 um**, and on `sample2` **7** where
it is **6 um**.

The absolute error is a fixed 1 um, so it hurts most exactly where theta is
smallest — a finely space-filling arbor with theta near 5 um is overstated
by 20%.

**Python port**: returns the distance, 9 and 6 respectively. Everything else
in the function agrees with MATLAB pixel for pixel.

### Minor: `span_tree` closes with an approximate disk

**File**: `metrics/span_tree.m:83`

```matlab
se = strel ('disk', radius, 4); % resolution
```

The third argument makes `strel` approximate the disk with four periodic
lines — an octagon, not a circle — which is a speed trade MATLAB offers and
which changes the spanned area slightly.

**Python port**: uses the exact Euclidean disk. **The size of the
difference is not measured here**: Octave implements only `N = 0`, so there
was no approximate disk to compare against, and constructing a guess at
MATLAB's octagon would produce a number rather than a measurement. With the
exact disk on both sides, the port reproduces MATLAB's mask and area
exactly (`sample` 343x343, 7064 um^2; `sample2` 181x181, 554 um^2).

### Minor: a debug `disp` left in `histax`

**File**: `graphtheory/utilities/histax.m:8`

```matlab
disp(size(xax)), disp(xax)
```

One of the two copies of `histax` in the toolbox prints its entire bin
vector to the console on every call. `theta_tree` calls it once per tree,
so a loop over a population fills the console with hundreds of lines. The
copy in `utilities/` does not have it, so which one runs depends on path
order.

**Python port**: not applicable — binning is `np.histogram`.
