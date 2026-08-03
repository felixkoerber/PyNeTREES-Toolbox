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

**Python port**: `boundary_tree` (and the `convexity_tree`/
`share_boundary_tree` cluster it belongs to) is deferred rather than
fixed-and-ported — see PORT_STATUS.md Design Decision #35 for the full
reasoning (the maintainers' own todo list already flags this cluster as
buggy/unclear, and the underlying O(n²) custom line/triangle-intersection
algorithm would need a genuine re-engineering pass, not a one-line fix, to
be worth relying on).

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
| Confirmed logic bugs | 10 | Redesigned around the bug (not replicated) |
| Self-acknowledged incomplete features | 12 | Either fixed, or explicitly not ported (documented) |
| Numerical robustness gaps | 2 | Explicit clamping / correct distance metric |
| Performance bottlenecks (self-documented) | 2 | Vectorized / replaced with a standard algorithm + library |
| Minor/cosmetic | 5 | N/A (informational) |

None of this is a knock on the original toolbox — `treestoolbox-master/` is
a substantial, working, widely-used piece of scientific software, and the
maintainers' own todo list shows they were already tracking most of this.
The point of this document is narrower: everywhere the Python port's
behavior visibly diverges from a literal line-by-line translation of the
MATLAB source, it should be traceable to *either* a deliberate design
decision (see `python_port/PORT_STATUS.md`) *or* one of the bugs cataloged
here — not an accident of the port itself.
