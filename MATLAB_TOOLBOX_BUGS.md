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
| Confirmed logic bugs | 23 | Redesigned around the bug (not replicated) |
| Self-acknowledged incomplete features | 12 | Either fixed, or explicitly not ported (documented) |
| Numerical robustness gaps | 2 | Explicit clamping / correct distance metric |
| Performance bottlenecks (self-documented) | 2 | Vectorized / replaced with a standard algorithm + library |
| Minor/cosmetic | 9 | N/A (informational) |

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
