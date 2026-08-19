# Response to `pyTREES function-by-function.md` — proposed plan

Every point in the review, with a verdict, a concrete implementation sketch,
and an effort estimate. Written after re-reading the relevant Python source
*and* the corresponding MATLAB source for each item, so the verdicts are
grounded rather than assumed.

**How to read this**

| Verdict | Meaning |
| --- | --- |
| **Agree** | Real gap or wart; I'd fix it as described. |
| **Agree; decided** | Fix confirmed in your review of this plan; it changes results or breaks callers, so the rollout is spelled out. |
| **Already true** | Checked the code; the behaviour you asked for is already there. |
| **Premise doesn't hold** | I checked against MATLAB and the concern doesn't apply — details given. |

Items are grouped into work packages **W1–W7** ordered by dependency and by
how expensive they get if deferred. Per-item IDs (`I1`, `G3`, …) match the
review's own ordering so you can diff the two documents side by side.

---

## 0. Summary and recommended order

| # | Work package | Why this position | Breaking? | Effort |
| --- | --- | --- | --- | --- |
| **W1** | Naming and signature cleanup | Breaking changes are cheapest now, before the API spreads further into `gc_model` and your notebooks | Yes | ~0.5 d |
| **W2** | I/O parity (`load_tree` dispatcher, v7.3, multi-tree, `.neu`/`.nmf`, **sample trees**) | Largest functional gap, and **it currently blocks the Active GC Model port** | Additive, except `sample_tree()` | ~2.5 d |
| **W3** | Missing MATLAB options (`rot_tree`, `MST_tree`, `resample_tree`, …) | The bulk of the remaining fidelity debt | Mostly additive; `resample_tree` is not | ~3 d |
| **W4** | Return-value contract (primary result by default, extras behind a flag) | Touches many signatures; wants to land after W1/W3 settle them | **Yes** — 25 call sites | ~1 d |
| **W5** | Population-level tooling (`list[Tree]` where it makes sense) | Needs W4's return conventions decided first | Additive | ~1 d |
| **W6** | New functions: `hull_tree`, `vhull_tree` | Unblocks `stats_tree`'s density/Voronoi statistics | Additive | ~1.5 d |
| **W7** | Correctness audit (root-index assumptions, empty trees, `cgin_tree`) | Independent; can run in parallel with anything | No | ~0.5 d |

Most decisions are now settled; two remain — see
[§8 Decisions](#8-decisions--settled-and-outstanding). Everything else I would
implement as described.

### Three cross-cutting rules I'd adopt

The review flags the same three inconsistencies in a dozen different places.
Rather than patch each site, I'd fix the underlying rule once and apply it
mechanically:

**R1 — the dimensionality argument is `dim: int` ∈ {2, 3}, always.**
Today it is spelled three different ways: `dim2: bool` (`cyl_tree`,
`len_tree`, `chull_tree`), `dim: int` (`eucl_tree`), and `dim: str`
(`vonMises_tree`, `bf_tree` — `"2d"`/`"3d"`). `dim: int` wins: it reads
naturally at the call site (`len_tree(t, dim=2)`), it is already the
majority convention, and it extends to `hull_tree` in W6 without inventing
a fourth spelling.

**R2 — no negated boolean parameters.** `no_self=False` and `no_root=False`
both make the reader resolve a double negative to work out the default.
Rename to the positive form and flip the default so behaviour is unchanged
(`root_self=True`, `at_root=True`).

**R3 — multi-output functions return only their primary result by default;
extra outputs are opted into with a flag.** Your call, and it is the better
design: the common case (`tree = sort_tree(t)`) stops paying for the rare
one, and nobody has to name a variable they will discard.

```python
def sort_tree(tree, by="hier", *, full_output=False):
    """...With ``full_output=True``, returns ``(tree, order)`` instead of
    just the sorted tree."""
```

**Spelling: `full_output=False`** (settled). It is SciPy's idiom for exactly
this (`fsolve`, `leastsq`, `brentq`), it stays positive per R2, and it reads
the same whether the extra output is one object or five.

**Scope: only functions whose primary result is the `Tree`.** Your call, and
it draws the line in the right place — the flag exists to spare callers an
unwanted `Tree`-plus-bookkeeping tuple, which is not the situation
`sholl_tree`, `chull_tree` or `spread_tree` are in. Those return several
co-equal results by nature, so they keep returning all of them, as a
`NamedTuple` (rule R3's second half). Concretely:

| Gets `full_output=` | Returns everything, always |
| --- | --- |
| `sort_tree`, `redirect_tree`, `insertp_tree`, `elimt_tree`, `MST_tree`, `insert_tree` | `sholl_tree`, `chull_tree`, `spread_tree`, `vonMises_tree`, `bf_tree`, `dissect_tree` |

`sub_tree` sits deliberately in neither column — see G3.

Where extras *are* returned, they come back as a `NamedTuple` so
`result.order` works alongside `tree, order = ...` unpacking.

The evidence that this is the right default is already in the codebase:
of the **9** internal call sites that unpack one of these tuples, **7 write
`tree, _ = ...`** — they are discarding the second output right now.

#### Making sure it lands consistently downstream

You flagged this specifically. The good news, which I checked rather than
assumed: **stale call sites cannot fail silently.** `Tree` defines
`__len__` but neither `__getitem__` nor `__iter__`, so `tree, order =
sort_tree(t)` raises immediately and at the right place:

```text
TypeError: cannot unpack non-iterable Tree object
```

(I had assumed `__len__` would make it unpackable and that a guard was
needed. It is not — the failure mode is already loud.) So the migration is
mechanical rather than risky, and the safeguards are about *completeness*,
not about catching silent corruption:

1. **Sweep, do not grep.** Convert all 9 internal sites plus the ~16 in
   `tests/`/`gc_model/`, then run the full suite **and** re-execute all
   five tutorial notebooks and both `gc_model` notebooks — notebooks are
   not covered by pytest and are where stale API use survives longest.
2. **Assert the contract in tests**, once per affected function: the bare
   call returns a `Tree`, and `full_output=True` returns a `NamedTuple`
   whose `.tree` equals it. That pins the *convention*, so a future
   function cannot half-adopt it — which is the failure mode that actually
   worries me here, more than any individual call site.
3. **One `ast`-based lint** asserting no tuple-unpacking assignment targets
   any of the six converted functions. Cheap, and it reaches call sites in
   files nothing imports.

The same three apply to the W1 renames, except those *do* need the
deprecation shims, since `idpar_tree(no_self=True)` would otherwise become
a silent `TypeError` on an unexpected keyword rather than a behaviour
change.

**This is a breaking change**, though. Today `sort_tree`, `redirect_tree`,
`insertp_tree`, `elimt_tree` and `MST_tree` all return bare tuples, so an
existing `tree, order = sort_tree(t)` would silently start unpacking a
`Tree` (into its first two nodes' worth of nothing) or raise a confusing
`TypeError` downstream. A deprecation shim cannot straddle "returns a
tuple" and "returns a Tree", so I would convert all 9 internal sites plus
the ~16 in `tests/` and `gc_model/` in the same commit, and add a release
note. This is the strongest argument for doing W1/W4 **now** rather than
after the API spreads further.

---

## W1 — Naming and signature cleanup

Cheap, mechanical, and breaking — so it should land first. Each rename ships
with a one-release compatibility shim that accepts the old spelling and
raises `DeprecationWarning`, so `gc_model` and your notebooks keep running
while you migrate.

### G1 · `idpar_tree(no_self=)` → `root_self=` — **Agree** (rule R2)

```python
def idpar_tree(tree, root_self: bool = True, *, no_self=None):
    if no_self is not None:            # removed after one release
        warnings.warn("no_self= is deprecated; use root_self=not no_self",
                      DeprecationWarning, stacklevel=2)
        root_self = not no_self
```

Roughly 20 internal call sites pass `no_self=True`; a single mechanical pass.

### E2a · `elimt_tree(no_root=)` → `at_root=` — **Agree** (rule R2)

Same shim. `at_root=True` (process the root too) preserves MATLAB's default.

### S3 · `bf_tree(params=)` → `values=` — **Agree**

Agreed that `params` is vague. One caveat: the argument holds the three
*fit constants* `(a, b, c)` of the Bird & Cuntz 2019 k→bf relationship, not
data values — so I would go one better and call it `fit_constants=`, which
says what it is. Happy to use `values=` if you prefer the shorter name; tell
me which and I will use that.

### P4/P5 · `plot_tree_mpl` → `plot_mpl_tree`, `dA_tree_mpl` → `dA_tree` — **Agree**

`plot_mpl_tree` restores the `<verb>_tree` shape shared by every other
function. `dA_tree` has no PyVista counterpart to disambiguate from, so the
suffix is noise. Old names kept as deprecated aliases.

### R1 rollout · `dim2: bool` → `dim: int`

Affects `cyl_tree`, `len_tree`, `chull_tree` (`dim2=`) and `vonMises_tree`,
`bf_tree` (`dim="3d"`). The shim accepts the old form:

```python
def len_tree(tree, dim: int = 3, *, dim2=None):
    if dim2 is not None:
        warnings.warn(...); dim = 2 if dim2 else 3
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim!r}")
```

### C1 · `Tree.total_length` property — **Agree**

```python
@property
def total_length(self) -> float:
    """Summed segment length of the whole tree [um] (== ``len_tree(t).sum()``)."""
    from .metrics import len_tree
    return float(len_tree(self).sum())
```

Deliberately *not* cached: `Tree` is mutable in place (`tree.X[5] = ...` is
legal and used), so a cache would silently go stale. The computation is a
few hundred microseconds even on a 3765-node reconstruction.

While there, I would add the two obvious siblings for the same reason —
`total_surface` and `total_volume` — since `sum(surf_tree(t))` reads exactly
as awkwardly as the `sum(len_tree(t))` you flagged.

### B5 · `smoothbranch` as a private helper — **Agree**

It operates on raw `X, Y, Z` arrays, not a `Tree`, so it does not belong in
the public `*_tree` surface. Rename to `_smoothbranch`, drop from `__all__`.
Not a `Tree` method — it never touches a tree. MATLAB exports it only
because MATLAB has no notion of a module-private function.

### G8/M1/M8 · Functions moved between modules — **Agree, with a discoverability fix**

`gene_tree`, `bin_tree`, `dist_tree` moved to `metrics.py`; `abel_tree`,
`rootangle_tree` to `edit.py`. The moves themselves are right (they follow
the dependency graph, not MATLAB's folder layout), but they cost you the
ability to guess where something lives.

Fix without moving anything back: **everything is already re-exported from
the top-level `pytrees` namespace**, so `pt.gene_tree` works regardless.
I would make that the documented contract — "import from `pytrees`, never
from `pytrees.metrics`" — and add a MATLAB-folder → Python-module table to
`docs/matlab-migration.md`, so a reader coming from `graphtheory/gene_tree.m`
lands in the right place.

---

## W2 — I/O parity

The biggest functional gap, and the one with a concrete consumer: **three of
the bundled `Active GC Model/morphos/*.mtr` files are MATLAB v7.3 and cannot
be loaded at all today.** Two of them — `0dplaxonFitsoma.mtr` and
`90dplaxonFitSH07_2soma.mtr`, 30 trees between them — have **no v5
equivalent anywhere in `morphos/`**, so that data is currently unreachable
from Python (and from Octave, which cannot read these files either). This
package is a prerequisite for finishing the Active GC Model port, not merely
tidiness, and it is the crux of MATLAB↔Python interoperability generally,
since MATLAB's `save_tree` writes v7.3 unconditionally.

### I1/I5 · One `load_tree` dispatching on extension — **Agree; reverses Design Decision #26**

DD#26 argued a dispatcher was pure overhead "with only one non-`.mtr` format
actually implemented". That reasoning has expired: `.mtr`, `.swc` and `.asc`
all have real loaders now, and `.neu`/`.nmf` are cheap to add (below).
Calling the split "embarrassing" is fair.

```python
_LOADERS = {".mtr": _load_mtr, ".swc": _load_swc, ".neu": _load_neu,
            ".nmf": _load_nmf, ".asc": _load_neurolucida, ".npz": _load_native}

def load_tree(path=None, *, repair=None, keep_sections=False):
    """Load one or more trees, dispatching on file extension."""
    if path is None:
        path = _pick_file_dialog()                            # I2
    suffix = Path(path).suffix.lower()
    if suffix not in _LOADERS:
        raise ValueError(
            f"{path}: unsupported extension {suffix!r}; "
            f"expected one of {', '.join(sorted(_LOADERS))}")
    if repair is None:                        # MATLAB's default: '-r' for
        repair = suffix in (".swc", ".neu", ".nmf", ".asc")   # all but .mtr
    ...
```

`load_mtr`, `load_swc`, `load_neurolucida` stay as public named entry
points — they are useful when you know the format and want the type checker
to know it too. `load_tree` becomes the polymorphic front door.

The design decision this reverses is recorded in `PORT_STATUS.md`; I would
supersede rather than delete it, with a dated "reversed in W2, because …"
note, so the reasoning trail stays intact.

### I2 · File-selection dialog when no filename is given — **Agree, with a caveat**

MATLAB pops `uigetfile`. The Python equivalent is `tkinter.filedialog`,
which is in the standard library:

```python
def _pick_file_dialog():
    from tkinter import Tk, filedialog
    root = Tk(); root.withdraw()
    name = filedialog.askopenfilename(
        title="Pick a tree file",
        filetypes=[("TREES formats", "*.mtr *.swc *.neu *.nmf *.asc *.npz")])
    root.destroy()
    if not name:
        raise ValueError("load_tree: no file selected")
    return name
```

**Caveat worth knowing before you ask for it:** a GUI dialog is a hard
failure in exactly the contexts this toolbox is otherwise good in — headless
servers, CI, and the `ProcessPoolExecutor` workers `gc_model` uses for
parallel NEURON runs. I would guard it so it only fires in an interactive
session and raises a clear error otherwise, rather than hanging a worker
process forever:

```python
if not sys.stdin.isatty() and "ipykernel" not in sys.modules:
    raise ValueError("load_tree: no path given and no interactive session "
                     "to prompt in")
```

### I6 · MATLAB v7.3 (HDF5) support — **Agree; highest-value item in this package. Use `mat73`.**

MATLAB's own `save_tree.m` writes `'-v7.3'` **unconditionally**, so every
`.mtr` produced by a current TREES install is unreadable by
`scipy.io.loadmat`. This is not an edge case, it is the default path — which
is exactly why it matters for MATLAB↔Python interoperability.

#### Which library — tested, not assumed

I ran all four candidates against the three real v7.3 files in
`Active GC Model/morphos/`. The results reordered my expectations:

| Library | Flat `{t1..t8}` cell array | Nested `{{t1..t15}}` cell array | Verdict |
| --- | --- | --- | --- |
| `scipy.io.loadmat` | ✗ `NotImplementedError` | ✗ | v5 only, by design |
| `hdf5storage` | ✗ `NotImplementedError` | ✗ | delegates to scipy for `.mat`; no help here |
| `pymatreader` | ✓ scipy-shaped dicts | **✗ returns raw, undereferenced `h5py.Reference` objects** | fails silently on the deeper nesting |
| **`mat73`** | ✓ | ✓ | **works on every file** |

The nested case is not obscure — it is the 2-level `cgui_tree` layout
`load_tree.m` documents, and it is what `0dplaxonFitsoma.mtr` and
`90dplaxonFitSH07_2soma.mtr` actually use. `pymatreader` looked like the
winner on the flat file (it returns exactly scipy's `simplify_cells=True`
shape) and then handed back unresolved HDF5 references on the other two.
Had I recommended it from the docs, that would have shipped.

**Also worth knowing: Octave cannot read these files either.** I tried to
use it as a conversion fallback and it warns `can't read 'tree' (unknown
datatype)` and loads nothing. So there is no "just re-save it as v5"
escape hatch outside MATLAB itself.

#### It already works with the existing code

`mat73`'s output flows through `mtr.py`'s **existing, unmodified**
`_flatten`/`_struct_to_tree` helpers — `_flatten`'s recursion already
handles the extra nesting level `mat73` adds. Verified end to end:

```text
0dplaxonFitsoma.mtr           -> 15 trees, ver_tree all clean
90dplaxonFitSH07_2soma.mtr    -> 15 trees, ver_tree all clean
SH_07_...MLyzed.mtr           ->  8 trees, ver_tree all clean
```

All 38 trees load with correct sparse `dA`, region names, and the
`Ri`/`Gm`/`Cm` electrotonic fields. Cross-checked field by field against
`pymatreader` on the one file both can read: **exact agreement**
(max difference `0.0` on `X`, `Y`, `Z`, `D`, `R`, `dA`, and `rnames`).

So the implementation is small:

```python
def _load_matlab(path):
    """Read a .mat/.mtr of either vintage into scipy-shaped nested dicts."""
    try:
        return loadmat(str(path), simplify_cells=True)
    except NotImplementedError:            # v7.3 / HDF5
        try:
            import mat73
        except ImportError:
            raise ValueError(
                f"{path} is a MATLAB v7.3 (HDF5) file; install the optional "
                f"dependency with `pip install pytrees[matlab]`") from None
        return mat73.loadmat(str(path))
```

`mat73` goes in a `[matlab]` extra (it pulls in `h5py`, nothing heavier),
imported lazily so the core package stays dependency-light. `_flatten`
needs no change; I would add a test asserting that, so a future refactor
cannot quietly break the nested case again.

**Writing** v7.3 is not needed: `scipy.io.savemat` emits v5, which MATLAB
reads fine (see I3).

#### Correction to my earlier claim about which data is blocked

I previously said `SH_07_all_repairedandsomaAIS_MLyzed.mtr` was the
full-resolution original and `-Midi` a downsampled copy. **That was wrong** —
checked, and they are the *same* morphologies (identical node counts and
total lengths to 0.00% across all 8 trees), just saved in different
formats. So that file is currently reachable via its v5 twin.

The real blockers are the other two. `0dplaxonFitsoma.mtr` and
`90dplaxonFitSH07_2soma.mtr` hold **30 trees between them with no v5
equivalent anywhere in `morphos/`** — their nearest-named siblings have
entirely different node counts (2301 vs 3939, 1123 vs 2812). That data is
unreachable from Python today, and unreachable from Octave too.

### I3 · `save_tree(..., matlab_format=True)` writing real `.mtr` — **Agree**

`scipy.io.savemat` writes v5, which MATLAB reads fine, so the *export*
direction needs no HDF5 writer:

```python
def save_tree(tree, path, matlab_format=None):
    """`matlab_format=None` (default) infers from the extension:
    `.mtr` → MATLAB v5, anything else → pytrees' native `.npz`."""
```

The conversion is the mirror of `_struct_to_tree`: `R + 1` back to 1-based,
`rnames` as a cell array of strings, `dA` as a v5 sparse matrix, and
`Ri`/`Gm`/`Cm` written out when set (they round-trip through MATLAB's own
electrotonics functions).

Round-trip test: `load → save → load`, asserting field equality. A true
MATLAB-side round-trip needs MATLAB or Octave; the `oct2py` env on this
machine could do it as a manual, non-CI check.

### I4 · Multiple trees per file for `load_tree`/`save_tree` — **Agree**

Both directions:

- `load_tree` already returns `list[Tree]` from `.mtr` and `.swc` when the
  file holds several. Making that uniform across every format is mostly
  documentation, plus making `.asc` follow suit.
- `save_tree(trees, path)` accepts a `Tree`, a `list[Tree]`, or a
  `list[list[Tree]]` (MATLAB's 2-deep `cgui_tree` nesting) and writes the
  matching structure. For the native `.npz` format that means prefixed keys
  (`t0_X`, `t0_dA_row`, …) plus an `n_trees` scalar.

The module docstring's "population-level tooling out of scope here" line was
never a principled boundary — it was the state of Phase 1. I would delete it
rather than defend it.

### I7 · `mtr.py:49` requires a variable literally named `tree` — **Agree, real bug**

```python
if "tree" not in data:
    raise ValueError(f"{path}: no 'tree' variable found in this .mtr file")
```

A `.mtr` is just a MATLAB workspace. `save_tree.m` happens to name the
variable `tree`, but any `.mat` saved by hand (or by T2N, which saves `tree`
alongside other variables, or under a different name entirely) is rejected.
Fix: prefer `tree`, and if it is absent fall back to the sole variable that
looks like a tree struct:

```python
candidates = {k: v for k, v in data.items()
              if not k.startswith("__") and _looks_like_tree(v)}
if "tree" in candidates:
    raw = candidates["tree"]
elif len(candidates) == 1:
    raw = next(iter(candidates.values()))
else:
    raise ValueError(
        f"{path}: expected a 'tree' variable; found {sorted(candidates)}. "
        f"Pass variable='<name>' to choose one.")
```

plus an explicit `variable=` parameter for the ambiguous case.
`_looks_like_tree` just checks for `dA` and `X` — the same fields
`_struct_to_tree` already requires.

### I8 · `load_swc` should call `repair_tree` — **Agree**

The docstring already admits the only reason it does not is that
`repair_tree` did not exist yet in Phase 1. Fix: add `repair: bool = True`
to `load_swc`/`load_neurolucida`/`load_tree`, matching MATLAB's `'-r'`
default for these formats (and MATLAB's *no*-repair default for `.mtr`,
which is already presumed repaired).

I would make it an explicit parameter rather than unconditional, because
"faithfully build whatever topology the file encodes" is genuinely the right
behaviour when you are debugging a malformed reconstruction — you want to
see the trifurcation, not have it silently split.

### I9 · `load_swc` handling multiple neurons — **Already true**

`load_swc` finds every parent-less row, walks each connected component, and
returns a `list[Tree]` — see [swc.py:59-77](src/pytrees/io/swc.py#L59-L77).
Worth adding to the docs (and a test against a real multi-neuron file), but
no code change needed.

### I10 · NeuroLucida: port the remaining MATLAB features — **Agree, in priority order**

Three features are missing, and they are not equally worth having:

1. **Markers** (synapse/spine glyphs) — *port this first.* Purely additive:
   the S-expression parser already walks past them, so it is a matter of
   collecting `(Marker ...)` blocks into a `tree.markers` dict rather than
   discarding them. Genuinely useful data that is currently thrown away.
2. **Concatenating trees onto their nearest soma** — *port second.* This is
   what makes a multi-block `.asc` load as one cell instead of several
   disconnected fragments, so it changes the *default* result meaningfully.
   Implementable with `cKDTree` over soma contour points plus `cat_tree`.
3. **Soma contour → fitted cylinder** — *port last, if at all.* MATLAB's own
   docstring calls its PCA fit "quite arbitrary" and says the function "can
   be much further optimized or just rewritten". I would rather expose the
   raw contour points as `tree.soma_contour` and let the caller decide
   (`soma_tree` already exists for building a soma profile). Tell me if you
   want the MATLAB fit reproduced bug-for-bug for comparability instead —
   that is a legitimate goal, just a different one.

### New · Port the MATLAB sample trees — **Agree; `sample_tree()` must load `sample.mtr`**

#### Why `pt.sample_tree()` is the wrong cell

Traced it. At **Phase 1 only SWC loading existed** — `.mtr` was deferred by
Design Decision #9. MATLAB's `sample_tree` loads `sample.mtr`, which the
port could not read, and the toolbox's `sample/swc/` folder contains
exactly **one** file, `25HSS.swc`. So that became the fixture and kept the
name `sample_tree()`. `PORT_STATUS.md` line 28 records the substitution
plainly ("loads bundled `25HSS.swc` fixture") — it just was not flagged as
a *semantic* difference. `.mtr` support arrived later (Design Decision #32)
and the sample was never revisited. A Phase-1 workaround that outlived its
reason.

`python_port/src/pytrees/data/sample.swc` is byte-identical to
`treestoolbox-master/sample/swc/25HSS.swc`.

#### What was lost, concretely

`25HSS.swc` is the **HSS** cell — same 2252 nodes and the same total length
(8100.26 µm) as `hss.mtr`, so it is genuinely that cell. But the SWC export
is **X-mirrored and translated** (bounding box `[1.1, 772.46]` against
`hss.mtr`'s `[-573.74, 197.62]` — identical 771.36 µm width, opposite
orientation), and **SWC cannot carry region names**, so:

```text
hss.mtr          regions: ['axon', 'dend', 'soma']
pt.sample_tree() regions: ['1']
```

That collapse is not cosmetic. Region handling is exactly what
`dissect_tree`, `stats_tree` and the NEURON bridge exercise, and the port's
default sample has had none of it. It also explains an earlier notebook
correction — *"sample_tree's region is `'1'` not `'dend'`"*
(`PORT_STATUS.md` line 1141) — which was a symptom of this substitution,
not an isolated documentation slip.

#### Plan: port all four, with MATLAB's names and semantics

Now trivially possible, since each is a one-line `load_tree` in MATLAB and
`load_mtr` already reads them:

| Function | File | Nodes | What it is |
| --- | --- | --- | --- |
| `sample_tree()` | `sample.mtr` | 197 | subtree of an HSN cell — **MATLAB's actual sample** |
| `sample2_tree()` | `sample2.mtr` | 15 | minimal tree, ideal for doctests |
| `hsn_tree()` | `hsn.mtr` | 1290 | full HSN cell |
| `hss_tree()` | `hss.mtr` | 2252 | full HSS cell, **with `axon`/`dend`/`soma` regions** |

Ship the `.mtr` files in `pytrees/data/` (6 KB + 0.9 KB + 36 KB + 33 KB —
smaller together than the 254 KB `sample.swc` they replace).

**Migration.** Nothing is lost: today's `sample_tree()` becomes
`hss_tree()`, in its properly-regioned `.mtr` form. But `sample_tree` is
referenced by **77 assertions across 8 test files**, 5 notebooks, 6 docs
pages and `gc_model/democracy.py`, and it goes from 2252 nodes to 197. So:

1. Add the four new loaders first, keeping `sample_tree()` as-is.
2. Migrate every call site to whichever tree it actually wants — most
   tests want *a* tree and can take the 197-node one (faster suite);
   anything wanting the big cell moves to `hss_tree()`.
3. Flip `sample_tree()` to `sample.mtr` and regenerate baselines in a
   separate labelled commit, same discipline as E5.
4. Re-execute all notebooks, and re-check their prose — the earlier region
   correction now needs *un*-correcting, since `sample.mtr` does carry real
   region names.

#### Related bug found while checking: `load_mtr` cannot read `dLPTCs.mtr`

The toolbox also bundles `dLPTCs.mtr` — **55 reconstructions in 5 named
groups**, which is precisely the fixture `stats_tree`'s group-comparison
API was built for, and the one MATLAB's own `stats_tree` examples use.
`pt.load_mtr` fails on it:

```text
ValueError: unexpected 'tree' contents:
    <class 'scipy.io.matlab._mio5_params.mat_struct'>
```

`_flatten` handles `dict`, `list` and `ndarray`, but scipy returns
`mat_struct` objects for this nesting depth **even with
`simplify_cells=True`**. One-line fix (accept anything with `_fieldnames`,
converting via `{k: getattr(o, k) for k in o._fieldnames}`), and it folds
naturally into I4's multi-tree work. Worth a fifth loader,
`dLPTCs_trees() -> dict[str, list[Tree]]`, feeding `stats_tree` directly.

### New · `.neu` and `.nmf` loaders — **Agree** (needed by I1)

- `.nmf` is HDF5 with a fixed `/swc/{x,y,z,r,type,parent_index}` layout —
  ~40 lines on top of the `h5py` dependency I6 already introduces.
- `.neu` is NEURON's section-based transfer format; MATLAB parses it with
  `textscan`. Test fixtures exist in the repo
  (`treestoolbox-master/tests/IO/test_neu_tree/GC.neu` and two others), so
  this is verifiable rather than speculative. The `'-ks'` "keep sections as
  regions" option maps onto `rnames` directly.

---

## W3 — Missing MATLAB options

### E5 · `resample_tree`: MATLAB's method as the default — **Agree; decided**

Currently the port preserves branch/termination points exactly
(Design Decision #23) and skips MATLAB's delete-and-splice snapping pass.
You want MATLAB's method as the default with the port's available as an
option.

Plan:

```python
def resample_tree(tree, sr=10.0, method="matlab", *,
                  conserve_length=False,       # MATLAB '-l'
                  interp_diameter=False,       # MATLAB '-d'
                  collapse_branches=True,      # MATLAB '-b', inverted (R2)
                  collapse_small_angles=True,  # MATLAB '-v', inverted (R2)
                  trim_regions=True,           # MATLAB '-r', inverted (R2)
                  extend_terminals=True):
```

`method="matlab"` is a faithful port of `resample_tree.m`'s
`insertp_tree` → `morph_tree` → `delete_tree` pipeline; `method="anchors"`
is today's implementation, kept because anchor-preservation is the better
behaviour when you care about branch-point positions (which the NEURON
bridge does).

**Decided: `method="matlab"` is the default.** That changes numeric output
for everything that resamples — `rootangle_tree`, `peters_tree`, and the
`gc_model` democracy experiment. So the rollout is:

1. Implement `method="matlab"` and pin it against MATLAB's own output. The
   `oct2py`/Octave setup on this machine runs the real `resample_tree.m`
   (I used it to verify `tran_tree` and `cgin_tree` in W7), so this can be
   a genuine differential test rather than a self-consistency check —
   modulo the `contains()` shim Octave needs for TREES' option parsing.
2. Flip the default and **regenerate the affected baselines in a separate,
   clearly labelled commit**, so the diff shows exactly which numbers moved
   and by how much. Baselines shifting silently inside a feature commit is
   how a real regression hides.
3. Re-run the `gc_model` democracy experiment and record the before/after,
   since that is the result you actually care about.

### M5 · `rot_tree` PCA / `-m3d` / `-al` modes — **Agree**

Reverses Design Decision #20 ("niche"). Three separate modes:

- **`-pcaX/Y/Z`** — the easy one. Centre the coordinates, take
  `np.linalg.svd` of the `n×3` coordinate matrix, permute the resulting
  principal axes so the named axis carries the largest extent. ~15 lines, no
  new dependency.
- **`-m3dX/Y/Z`** — "mean axis, 3D": in MATLAB the first argument `DEG`
  switches meaning and becomes a *node subset* to compute the axis from. I
  would **not** reproduce that overloading — it is the kind of thing that
  makes MATLAB code hard to read. Instead:
  `rot_tree(t, mode="m3dX", nodes=...)`, with `deg` ignored (and a warning
  if both are passed).
- **`-al`** — align region borders. The most involved: fit a plane to each
  region boundary and rotate so the boundaries lie horizontal. This is the
  region-plane fitting DD#20 flagged; a least-squares plane fit per boundary
  plus a rotation onto the mean normal covers it.

API shape: `rot_tree(tree, deg=(0,0,90), mode=None, nodes=None)`, since
`mode` and `deg` are mutually exclusive in MATLAB too.

### B1 · `MST_tree` full options — **Agree**

Four missing capabilities, in the order I would add them:

1. **`indx` second output** (`[itree, inode]` per point) — trivial, and it
   is the output you need to map input points back to tree nodes. Folds into
   W4's `NamedTuple`.
2. **Multi-tree growth** (`msttrees` as several start points, or as existing
   `Tree` structures to keep growing) — the current code hardcodes one
   integer `start`. The greedy heap loop generalises directly: seed the heap
   from every start node and tag each node with its owning tree. This is the
   option with the most real use, since it is how you grow a whole
   population into one shared point cloud with competition between cells.
3. **`DIST` sparse connection-probability matrix** — an additive term in the
   cost function. Mechanically simple; the subtlety is that MATLAB requires
   the caller to include the *tree's own* nodes in `DIST`'s indexing, which
   is easy to get wrong. I would instead accept `dist` indexed over the
   input points only, and document that difference explicitly.
4. **`-c` grow-from-cut-ends-only** and **`-t` time-lapse recording** —
   niche; `-t` returns a list of intermediate trees, cheap to add as
   `record=True` once the loop is otherwise refactored.

### B2 · `BCT_tree` should lay out coordinates — **Agree**

Right now every coordinate is zero, which makes a `BCT_tree` result
unplottable and slightly surprising. MATLAB attaches an `xdend_tree` layout.
`xdend_tree` *is* ported (Phase 7), so:

```python
def BCT_tree(bct, layout: bool = True):
    """...``layout=True`` (default) assigns dendrogram coordinates via
    :func:`xdend_tree` so the result is plottable; ``layout=False`` returns
    pure topology with all coordinates zero."""
```

`X = xdend_tree(tree)`, `Y = PL_tree(tree)`, `Z = 0`, `D = 1`.

### B3 · `soma_tree` overlap correction, and does it match `maxD`/`length`? — **Agree, plus a real check**

The `'-b'` option is a ~10-line addition: past the first branch point inside
the soma region, divide diameters by `sqrt(2)` so the doubled surface of two
overlapping cylinders does not inflate total membrane area.

On "does the port still match `maxD` and `length`" — I have **not** verified
this numerically, and it deserves a proper check rather than a guess. The
port applies `D = max(D, maxD·cos(π·Plen/length))` over nodes with
`Plen < length/2`, which yields exactly `maxD` at the root and tapers to
zero at `Plen = length/2`. MATLAB's is the same formula, but I want to
confirm the node-selection boundary and the `max`-versus-assignment
semantics against `soma_tree.m` line by line, and add a test asserting
`D.max() == maxD` and that the soma's axial extent equals `length`. I will
report the result rather than assert it now.

### B4 · `cap_tree`: `'-a'` axon option, and the root-at-0 assumption — **Split verdict**

- **`'-a'` (add axon): I would still leave it out.** It draws length,
  diameter and taper from hardcoded constants fit to one published dataset.
  That is not a capping algorithm, it is a dataset-specific generator, and
  burying it inside `cap_tree` makes it easy to apply by accident. If you
  want it, I would rather it be its own function — `add_axon_tree(tree, ...)`
  with those constants as visible, overridable defaults. Say the word.
- **Root-at-0: agree, and it is broader than `cap_tree`.** See W7 below.

### G2 · `ipar_tree`'s `'-T'` option — **Agree, low priority**

`'-T'` restricts paths to "termination point → first branch point". It is a
convenience filter over the full `ipar` matrix, not new information, and
nothing in the toolbox calls it. Cheap to add
(`ipar_tree(tree, terminals_only=True)`, masking rows to terminals and
truncating each at the first branch point). I would do it in the same pass
as G7 rather than schedule it separately.

Worth flagging while we are here: `ipar_tree` is the port's worst-scaling
function (dense `n × max_depth`, 49 MB for a granule cell — see
`docs/port-audit.md` §2). If `'-T'` is what you actually want most of the
time, the terminals-only variant is also dramatically smaller, and I would
implement it as a direct walk rather than by masking the full matrix.

### G6 · `Pvec_tree` should default to `len_tree` — **Agree**

`Pvec_tree(tree, len_tree(tree))` appears verbatim at six call sites inside
the toolbox alone. Metric path length is overwhelmingly the intended
meaning.

```python
def Pvec_tree(tree, v=None):
    """...``v`` defaults to :func:`len_tree`, giving metric path length
    from the root [um]."""
    v = len_tree(tree) if v is None else np.asarray(v, dtype=float)
```

Purely additive — `v` was positional with no default, so no existing call
changes meaning.

### P1 · `plot_tree`: merge `color`/`scalars`, match MATLAB's argument order — **Agree; decided**

MATLAB's signature is `plot_tree(intree, color, DD, ipart, res, options)`
where `color` is polymorphic: an RGB triple, a per-node scalar vector
(mapped through the colormap), or an `N×3` matrix of per-node RGB. The port
split that into `color=` and `scalars=` and reordered the rest.

Merging is clearly right — two arguments where one is always `None` is the
classic sign of a bad split, and the merged form is what a MATLAB user will
type anyway:

```python
def plot_tree(tree, color="black", offset=(0,0,0), nodes=None, res=8,
              mode="tube", *, cmap="viridis", plotter=None, show=False, ...):
    """``color`` is a colour name/RGB triple (flat), a length-``n_nodes``
    vector (mapped through ``cmap``), or an ``(n_nodes, 3)`` RGB array."""
```

Dispatch by shape: scalar/string/3-tuple → flat colour; `(n_nodes,)` →
scalars; `(n_nodes, 3)` → per-node RGB. The one genuine ambiguity is a
3-node tree, where a length-3 vector could be either an RGB triple or
per-node scalars — I would resolve it in favour of RGB (matching MATLAB) and
document it.

**Decided: match MATLAB's order.** So the positional sequence becomes
`(tree, color, offset, nodes, res)` — MATLAB's `(intree, color, DD, ipart,
res)` — with everything the port adds (`mode`, `cmap`, `plotter`, `show`,
`screenshot`) moved behind a `*` as keyword-only. That is what makes the
order *stay* matched: a future addition cannot quietly wedge itself into a
positional slot.

`mode="tube"`/`"line"` is the one judgement call, since it stands in for
MATLAB's `options` string (`'-p'`, `'-b'`, `'-2l'`, …), which is MATLAB's
6th positional argument. Keyword-only is better here — nobody should be
writing `plot_tree(t, "black", (0,0,0), None, 8, "line")`.

This breaks existing positional calls. `plot_tree` appears throughout the
tutorial notebooks; I would update them and re-execute all five in the same
commit, so the rendered outputs stay consistent with the code.

### P2 · `pointer_tree` electrode rendering — **Agree**

MATLAB's `'-l'`/`'-v'` modes build a tapering electrode from a small
synthetic frustum tree — a recognisable visual idiom in TREES figures.
Implementable as `style="electrode"` using a `pyvista.Cone`/`Tube` pointing
away from the tree's centroid, with `length` and `angle` parameters.
Additive; the existing `"marker"`/`"sphere"` styles are untouched.

### P3 · `chull_tree` in 2D, and `dim` consistency — **Agree**

Two parts:

- The `dim=2` branch currently computes the hull but never draws it (the
  `plotter is not None and not dim2` guard skips it). Fix: when `dim == 2`,
  draw the hull as a closed polyline via matplotlib, since a 2D hull on a 3D
  PyVista scene is the wrong pairing. That means `chull_tree` needs to
  accept either an `Axes` or a `Plotter` — I would key it off the object
  type rather than add a second parameter.
- `dim2=` → `dim: int` per rule R1.

### P6 · `dendrogram_tree`: coordinate offset, BCT check — **Agree**

- `offset=(dx, dy)` added, matching MATLAB's `DD` (used to lay several
  dendrograms side by side). Consistent with `plot_tree`'s existing
  `offset=`.
- BCT-conformity check: I would **warn**, not raise. A non-BCT tree still
  produces a readable (if oddly ordered) dendrogram, and `xdend_tree` does
  not crash on one. Raising would make the plotting function stricter than
  the analysis functions, which is backwards.

```python
if np.any(idpar_tree(t, root_self=False)[1:] > np.arange(1, t.n_nodes)):
    warnings.warn("tree is not BCT-conform; dendrogram order may look "
                  "scrambled — run repair_tree first", stacklevel=2)
```

### S1 · `sholl_tree`: drop `ShollResult`, add plotting — **Partly agree**

Agreed the dataclass is doing very little. Two sub-points, different
answers:

- **Return type: agree it should go, but to a `NamedTuple`, not a dict.**
  MATLAB returns seven positional outputs `[s, dd, sd, XP, YP, ZP, iD]`. A
  `NamedTuple` gives you MATLAB-style unpacking
  (`s, dd, sd, *_ = sholl_tree(t)`), attribute access (`.s`), *and*
  `._asdict()` for the dict you want — with none of a plain dict's
  typo-silence (`result["dd"]` vs `result["DD"]`). It is also rule R3,
  applied consistently with the rest of the toolbox. If you specifically
  want a plain dict I will do that instead; I just think this is the version
  you would rather use in six months.
- **Plotting: agree, port it.** MATLAB's `'-s'`/`'-s3'` draw the
  intersection points on the morphology. I would put it in `plotting.py` as
  `plot_sholl(tree, result, ax_or_plotter)` rather than as a side effect of
  the analysis function — the port has deliberately kept computation and
  rendering separate everywhere else, and mixing them back in here would be
  the only exception.

### S4 · `stats_tree` should run `hull_tree`/`vhull_tree` — **Agree** (see W6)

Blocked on those two functions existing. Once W6 lands, the `parea`/`mparea`
density statistics drop into `stats_tree(extras=True)` as two more `summary`
columns.

---

## W4 — The return-value contract

Applying rule R3, plus a per-function decision about *whether* a second
output should exist at all. My rule: **a second output survives if it cannot
be cheaply recomputed from the first.**

| Item | Function | Second output | Recomputable? | Decision |
| --- | --- | --- | --- | --- |
| G4 | `redirect_tree` | `order` | No | Keep, as `NamedTuple` |
| G5 | `sort_tree` | `order` | No | Keep, as `NamedTuple` |
| E4 | `insertp_tree` | `added_mask` | No | Keep, as `NamedTuple` |
| B1 | `MST_tree` | `connected` / `indx` | No | Keep, and add `indx` |
| E2b | `elimt_tree` | `changed` | **Yes** | **Drop** — return `Tree` |
| G3 | `sub_tree` | subtree `Tree` | Yes, from the mask | Add as `as_tree=` keyword |
| G7 | `dissect_tree` | per-node section index | Yes, O(n) | Add as `with_positions=` keyword |

### E2b · Drop `elimt_tree`'s `changed` flag — **Agree**

It is recomputable (`typeN_tree(result).max() <= 2`, or just comparing
`n_nodes`), and it forces every caller to unpack a tuple for information
almost none of them use. Return the `Tree`; emit a `logging.debug` when
nothing changed rather than a print, so library code stays quiet by default
but the information is recoverable when you are debugging.

### G3 · `sub_tree` returns both by default — **Agree; my earlier objection was wrong**

You want the mask *and* the subtree by default, with a flag to suppress the
tree. In my first pass I argued against that on performance grounds — that
eager construction would "undo a good chunk of the 285× speedup". **I
measured it, and that was overstated.** On the 3765-node granule cell:

| | per call |
| --- | --- |
| mask only | 1682 µs |
| mask + subtree | 2203 µs (**1.3×**) |

And the hot-caller argument barely applies any more: only **`asym_tree`**
still calls `sub_tree` in a loop, 34 times on that cell (`repair_tree` and
`clean_tree` no longer call it at all — they were refactored during the
performance audit, and `sub_tree`'s own docstring still claims otherwise,
which I would fix). Eager construction would add ~20 ms to `asym_tree`'s
83 ms. Affordable, and the internal caller opts out anyway.

So, as you asked — both by default:

```python
def sub_tree(tree, inode, with_tree: bool = True):
    """Indices of the subtree starting at ``inode``, and the subtree itself.

    Returns ``(mask, subtree)``. ``mask`` is a boolean array over the parent
    tree's nodes, matching MATLAB's ``sub`` ("1 if part of subtree, 0 if
    not"). Pass ``with_tree=False`` to skip building the extracted
    :class:`Tree` — worth doing in a per-node loop, where it costs ~30%.
    """
```

Note this uses a `with_tree=` flag rather than R3's `full_output=`, because
here the *primary* output is the mask and the tree is the extra — naming it
after what it controls is clearer than a generic flag. Both spellings mean
"give me less by default"; they just point at different outputs.

**One improvement over MATLAB while we are here.** `sub_tree.m` carries the
comment *"NOTE ! region update for tree output still missing!!!"* — its
subtree keeps the parent's full region list. `Tree.reindexed` has the same
gap today: I checked, and a subtree carrying only region `0` still lists all
8 of the granule cell's `rnames`. The port should trim `rnames` to the
regions actually present and reindex `R` accordingly, which is what anyone
expects from "cut this branch out".

### G7 · `dissect_tree`'s second output — **Agree, and yes, re-verify the bug fix**

MATLAB's second output is a per-node `[section_index, relative_position]`
pair used for NEURON `nseg` bookkeeping. Add as
`dissect_tree(tree, with_positions=True)`, computed inside the ancestor walk
that already builds the sections — no extra traversal.

On re-checking the region-cut fix: **agreed, and I would do it properly** —
against MATLAB rather than against my own reasoning. The plan is to run
`dissect_tree.m` in the `oct2py` environment on this machine over the
bundled `sample_tree` and the granule cell, and diff the section lists
element by element. That is a stronger check than the current test, which
only pins the Python behaviour. If MATLAB's `ipar`/`cumsum` trick differs
anywhere except the root case its own docstring already disclaims, I want to
know which of us is wrong before writing it down as settled.

### G5 · `sort_tree` uses a different method (DD#12) — **Keep the divergence, document it better**

`'hier'` is a DFS pre-order rather than MATLAB's level-order-ish scheme.
Both produce valid BCT orderings, and no downstream function depends on
*which* valid ordering it gets — the orderings are isomorphic. Changing it
would churn every index in every saved result for no functional gain.

What I *would* do: state explicitly in the docstring that node indices are
**not** comparable between MATLAB and pytrees after a sort, so nobody
cross-references node 417 between the two toolboxes and gets quietly wrong
answers. That is the actual risk here, and it is a documentation fix.

---

## W5 — Population-level tooling

### S2 · `vonMises_tree`'s pattern generalised — **Agree**

`vonMises_tree` already accepts `Tree | list[Tree] | np.ndarray`. You are
right that this should be the norm rather than one function's special case.

Rather than hand-writing polymorphic entry points, one decorator:

```python
def accepts_population(reduce="concat"):
    """Let a per-tree function also accept a list of trees.

    ``reduce="concat"``: results are concatenated (per-node quantities).
    ``reduce="list"``:   results are returned as a list (per-tree values).
    ``reduce="pool"``:   inputs are pooled before a single computation
                         (distribution fits like vonMises_tree/bf_tree).
    """
```

Applied to the functions where pooling is meaningful — `gene_tree` (M7),
`sholl_tree`, `stats_tree` (already), `bf_tree`, `dist_tree`, `bin_tree`.
**Not** applied to editing functions, where "apply to each tree" is already
trivially a list comprehension and a decorator would only obscure it.

### M7 · `gene_tree`'s population wrapper — **Agree**, via the decorator above.

MATLAB's version also plots the pooled gene distribution; per the S1
reasoning, that goes in `plotting.py` as `plot_gene(...)`, not inside the
analysis function.

### P7 · Merge `spread_tree` and `spread_trees` — **Agree**

Two functions differing only in return type is exactly the wart you
describe. One function, `NamedTuple` return (rule R3):

```python
def spread_tree(trees, dx=50.0, dy=50.0) -> SpreadResult:
    """Returns ``(trees, offsets)``: the translated copies *and* the
    ``(dx, dy, dz)`` offset applied to each."""
```

`trees` first, since it is what most callers want. `spread_trees` becomes a
deprecated alias for one release.

---

## W6 — New functions: `hull_tree` and `vhull_tree`

Neither is ported. `stats_tree`'s density/Voronoi statistics (S4) depend on
them, and both are useful on their own.

**`hull_tree`** — space-filling isosurface at a threshold distance from the
tree. Build a grid over the bounding box, compute distance from every grid
point to the nearest tree node (`cKDTree.query`, far faster than MATLAB's
loop), then extract the contour:

- 2D: `matplotlib.pyplot.contour` at the threshold level — no new dependency.
- 3D: `skimage.measure.marching_cubes`, which means **scikit-image as a new
  optional dependency** (it is not installed in the `pytrees` env today). I
  would put it under the existing `[plot]` extra rather than create a new one.

Returns `(contour, mask)` where `mask` is the boolean in/out grid — the
`'-F'` full-distance-matrix option becomes `return_distances=True`.

**`vhull_tree`** — Voronoi subdivision of the tree's points, clipped to the
hull boundary, giving a per-node territory volume. `scipy.spatial.Voronoi`
does the tessellation; the work is clipping unbounded cells against the
`hull_tree` boundary. In 2D that is straightforward polygon clipping; in 3D
I would clip each cell against each boundary half-space, then take
`ConvexHull(...).volume` per cell.

Those per-node volumes are what `stats_tree` needs for `parea`/`mparea`.

---

## W7 — Correctness audit

Independent of everything above; can run in parallel.

### X1 + M4 · Double-checking `cgin_tree` and `tran_tree` — **both confirmed, numerically**

You asked me to double-check these two. Reading the source was not enough,
so I ran the **real MATLAB code** — TREES' own `.m` files, executed in
Octave 11 — and compared against `pytrees` on the *identical* tree.

Method, so it can be repeated: copy `treestoolbox-master` to a scratch
directory, `sed` every `contains(` to a small `xcontains.m` shim (Octave
does not implement `contains`, and TREES' option parsing needs it), run
`start_trees`, and export the MATLAB `sample_tree` as a v5 `.mtr` so the
Python side loads bit-identical inputs.

**Result — `tran_tree`**, on a sample tree deliberately shifted off the
origin by `(10, 20, 30)` so the default is not a no-op:

| Case | max abs difference |
| --- | --- |
| default (`DD=1` / `offset=0`) — root to origin | `0.0` |
| centre on node 5 / node 4 (0-based) | `2.4e-11` |
| explicit vector `[1 2 3]` | `2.7e-11` |

Those residuals are the 12-significant-digit `printf` I used to carry the
reference values across, not a real discrepancy. **The port matches MATLAB
in all three modes**, confirming the earlier reading: MATLAB's default
`DD = 1` takes the *scalar* branch (`tree.X - tree.X(DD)`), which centres on
node 1 — exactly what `offset=0` does.

**Result — `cgin_tree`**, with `Gm = 1/2500` S/cm²:

| | value |
| --- | --- |
| MATLAB (Octave) | `1.5346102318177e-08` |
| pytrees | `1.5346102318177e-08` |
| relative difference | `4.3e-16` (machine epsilon) |

Surface area agrees to `1.3e-14` relative. The algebra was identical as
expected — MATLAB's `1/((1/Gm)/(Σsurf/1e8))` is just `Gm·Σsurf/1e8` — and
now the arithmetic is confirmed too.

**Two things this turned up along the way:**

- **MATLAB's `cgin_tree` docstring has the units wrong.** It declares
  `cgin ::value: ... in [nS]`, but the function's own plot title prints
  `num2str(1000000000 * cgin)` followed by `' nS'` — the returned value is
  in **siemens**. The port returns siemens (correct) and says so. Anyone
  trusting the MATLAB docstring while comparing numbers is off by 10⁹.
  → `MATLAB_TOOLBOX_BUGS.md`.
- **`pt.sample_tree()` is not MATLAB's `sample_tree`.** The Python one is a
  2252-node SWC; MATLAB's is a 197-node `.mtr` — different cells entirely
  (total surface 103081 µm² vs 3837 µm²). That is a portability trap for
  anyone following the tutorials side by side, and it invalidated my first
  comparison run until I spotted it. I would either ship MATLAB's sample as
  `sample_tree_matlab()` for cross-checking, or state the difference
  prominently in `docs/matlab-migration.md`. My preference is both.

### New · Root-index assumptions (raised by your B4 note) — **Agree, and it is systemic**

Design Decision #10 states the root is found by `dA` in-degree, never by
assuming index 0. Three functions violate it:

| Function | Site | Effect on a non-index-0 root |
| --- | --- | --- |
| `cap_tree` | `tree.X[0]`, `direction[1, …]` | Caps the wrong end of the tree |
| `scale_tree` | `center=True` uses `tree.X[0]` | Scales about the wrong point |
| `flip_tree` | `2*tree.X[0] - tree.X` | Mirrors about the wrong point |

In practice everything downstream of `repair_tree`/`sort_tree` has its root
at 0, so this is latent rather than active — but it is exactly the kind of
assumption that produces a silently wrong figure when someone loads a
hand-built or externally generated tree.

#### Why the decision was violated in the first place

You asked, and it is worth answering properly, because the answer decides
whether the fix sticks. Three causes compounded:

**1. MATLAB's own source hardcodes node 1 in exactly these three
functions** — and nowhere else that matters:

```matlab
flip_tree.m:47    tree.X = 2 * tree.X (1) - tree.X;
scale_tree.m:54   ORI    = [tree.X(1) tree.Y(1) tree.Z(1)];
cap_tree.m:87     [(tree.X (1)) (tree.Y (1)) (tree.Z (1))] - (l * direction (2, :))
```

These were transliterated faithfully — including the assumption. The port
did not introduce a bug so much as inherit one, and inheriting is the
default outcome whenever a line ports cleanly.

**2. Design Decision #10 was written with Phase-2 scope, and read that
way.** Its wording is *"every Phase 2 function uses it rather than
special-casing"*, with the general rule tucked into a closing "How to apply"
sentence. `scale_tree`/`flip_tree` are Phase 5 and `cap_tree` is Phase 6 —
written later, in different modules, by which point the decision read as a
historical note about `graphtheory.py` rather than a live invariant. A
decision phrased as a description of what was done does not defend itself
against what gets done next.

**3. `_root_index` is module-private, in a module those files do not
otherwise depend on.** Using it from `metrics.py` or `construct.py` means
importing a leading-underscore name across module boundaries — friction,
and a smell. Writing `tree.X[0]` was the path of least resistance, and
nothing flagged it.

#### So the fix has two halves

Patching the three call sites addresses cause (1) only. To stop it
recurring, I would also:

- **Make the root a first-class property**, `Tree.root`, alongside
  `n_nodes` and the new `total_length` (C1). Then the obvious thing to
  write — `tree.X[tree.root]` — is also the correct thing, and no
  cross-module private import is involved. `_root_index` stays as the
  internal implementation.
- **Rewrite DD#10 as a standing invariant** rather than a Phase-2 report,
  and add it to whatever checklist new functions go through.
- **Add a root-at-index-5 tree to the degenerate-input sweep**, so the
  next violation is caught by a test rather than by a code review. This is
  the part that actually enforces it: the sweep already runs 52 functions
  against 6 degenerate trees, and a non-zero-root tree is a natural
  seventh.

### E1 · `delete_tree` on a full-tree deletion — **Agree, with a prerequisite**

You are right that MATLAB returns an empty tree where the port raises. The
prerequisite is that **nothing else in the toolbox currently handles a
zero-node `Tree`**: `_root_index` raises, `sort_tree` raises,
`sse_tree`/`M_tree` return non-finite values (already logged in
`docs/port-audit.md` §6).

So the plan is two steps, in order:

1. Define the empty-tree contract: a `Tree` with `n_nodes == 0` is
   constructible (it already is), `ver_tree` accepts it, and the ~15
   functions that assume a root either return an empty result or raise a
   *clear* error naming the empty tree. Add a degenerate-input row for it
   alongside the six the last audit already sweeps.
2. Then change `delete_tree` to return the empty tree.

Doing (2) without (1) just moves the crash somewhere less obvious.

### E3 · How does `insert_tree` work, and can new nodes parent each other? — **Answered, plus a fix**

The docstring says new points attach to *existing* nodes, but the
implementation is more permissive: parent indices are written straight into
the adjacency matrix, so a `parent[i]` pointing at another newly added node
works fine as long as it refers to an already-assigned index (`N + j`).
`cap_tree` relies on exactly this — it chains each cap segment onto the
previous one.

So the capability exists; it is just undocumented and unvalidated. Plan:

- Document it explicitly, with `cap_tree`'s chaining as the example.
- Validate it: reject `parent[i] >= N + i` (a forward reference, which today
  silently produces a cycle or a disconnected node) with a clear error.
- **Return the new node indices**, as you ask — `NamedTuple(tree, inodes)`
  where `inodes = np.arange(N, N + n_new)`. Trivially derivable, but you
  should not have to know that, and it makes `insert_tree` self-describing.

### M4 · `tran_tree`'s default — **Premise doesn't hold; the port matches MATLAB**

Checked `metrics/tran_tree.m`. MATLAB's default is `DD = 1`, and the
`numel(DD) > 1` branch is what treats `DD` as an offset vector; a scalar
`DD` takes the *other* branch, `tree.X - tree.X(DD)`. So MATLAB's default
translates the tree so **node 1 (the root) sits at the origin** — exactly
what the port's `offset=0` (node 0 = root) does. The docstring line "per
default sets tree root to origin" describes the scalar branch, not a
separate `[0,0,0]` default.

And yes, it does do something: it shifts the whole tree so the root lands at
`(0, 0, 0)`. On an already-centred tree it is a no-op, which may be where
the doubt came from.

No change needed. I would only sharpen the port's docstring to say "default:
move the root to the origin", which is the useful phrasing, and note the
MATLAB equivalence.

### M3 · `direction_tree`'s root direction — **Agree it is odd; I would document, not change**

The root has no parent, so it has no direction. MATLAB assigns it the
direction *of its first child* (equivalently, the root→child vector), which
is the least-bad convention: it keeps the output length `n_nodes` and
aligned with every other per-node array, and it gives `cap_tree` a sensible
"which way is out of the soma" vector — which is precisely what `cap_tree`
uses it for.

The alternative, `NaN` at the root, would be more honest but would propagate
`NaN` into every consumer. Since this matches MATLAB and has a real
consumer, I would keep the behaviour and make the docstring state it plainly
rather than leaving the reader to discover it.

### M6 · `morph_tree` should not depend on `tran_tree` — **Agree**

Correct — `morph_tree` calls `tran_tree` only to apply a translation it has
already computed, and `with_coords` does that directly without the extra
call, the extra `Tree` allocation, or the module-level coupling. Small,
safe, no behaviour change.

### M2 · `cyl_tree`'s `'-dA'` output form — **Keep dropped**

MATLAB's own comment on that branch says "SLOW!!", nothing in the toolbox
calls it, and it returns the same geometry in a less convenient layout.
Reinstating it would mean porting a form the original author warned against.
Flagged here only so the decision is explicit rather than accidental — say
so if you want it and I will add it.

---

## 8. Decisions — settled and outstanding

**Settled in your review of this plan:**

| Item | Decision |
| --- | --- |
| E5 · `resample_tree` default | **MATLAB's method**, with `method="anchors"` kept as an option |
| P1 · `plot_tree` argument order | **Match MATLAB**: `(tree, color, offset, nodes, res)`, everything else keyword-only |
| R3 · multi-output functions | Primary result only by default; extras behind a flag |
| G3 · `sub_tree` | Returns **mask and subtree** by default; `with_tree=False` suppresses the tree |
| I6 · v7.3 reader | **`mat73`**, in a `[matlab]` extra, imported lazily |
| R3 · flag spelling and scope | **`full_output=False`**, only on functions whose primary result is the `Tree` |
| Samples | `sample_tree()` loads `sample.mtr`; `sample2_tree`/`hsn_tree`/`hss_tree` ported too |

**Still open:**

1. **`bf_tree`'s renamed parameter** — `fit_constants=` (what it holds) or
   your `values=` (shorter). The only thing still genuinely open.

Two I will decide myself unless you object: `sholl_tree`'s return type
(`NamedTuple`, so `s, dd, sd, *_ = sholl_tree(t)` matches MATLAB's seven
outputs *and* `._asdict()` gives you the dict), and moving `cap_tree`'s
`'-a'` into a separate `add_axon_tree` rather than leaving it as a flag on
a capping function.

---

## 9. Design decisions to record

New entries for `PORT_STATUS.md`'s log, continuing from #39:

- **#40** — Dimensionality is always `dim: int` ∈ {2, 3} (rule R1);
  supersedes the ad-hoc `dim2: bool` / `dim: str` spellings.
- **#41** — No negated boolean parameters (rule R2).
- **#42** — Multi-output functions return their primary result only; extras
  are opted into with a flag and come back as a `NamedTuple` (rule R3).
  `sub_tree` is the deliberate exception — both outputs by default, since
  the extracted subtree is half of what the function is *for*.
- **#43** — **Supersedes #26**: `load_tree`/`save_tree` become
  extension-dispatching front doors over the now-several real loaders, with
  format-specific functions kept as named entry points.
- **#44** — **Supersedes #20**: `rot_tree`'s PCA/`-m3d`/`-al` modes ported,
  but `-m3d`'s argument overloading (where `DEG` silently becomes a node
  subset) is replaced by an explicit `nodes=` parameter.
- **#45** — **Supersedes #23**: `resample_tree` defaults to MATLAB's
  snapping method, with the anchor-preserving method available as
  `method="anchors"`.
- **#46** — Plotting stays out of analysis functions: MATLAB's `'-s'` show
  options become separate `plot_*` functions in `plotting.py`, applied
  consistently to `sholl_tree`, `gene_tree` and `hull_tree`.
- **#47** — `mat73` (and `scikit-image` for W6) as optional dependencies,
  imported lazily so the core package stays dependency-light. `mat73` is
  chosen over `pymatreader`/`hdf5storage` on measured behaviour, not
  reputation: it is the only one that resolves MATLAB's nested cell arrays
  of structs rather than returning raw HDF5 references — see W2/I6.
- **#48** — **Restates #10 as a standing invariant, not a Phase-2 report**:
  the root is `Tree.root` (new public property), never index 0. Adds a
  non-zero-root tree to the degenerate-input sweep so violations are caught
  by a test rather than a review.
- **#49** — `Tree` gains `total_length`/`total_surface`/`total_volume`
  properties (uncached, because `Tree` is mutable in place).
- **#50** — `sub_tree`'s extracted subtree trims `rnames` to the regions
  actually present — fixing MATLAB's own acknowledged gap ("NOTE ! region
  update for tree output still missing!!!") rather than porting it.
- **#51** — **Reverses the Phase-1 sample substitution.** `sample_tree()`
  loads `sample.mtr` (MATLAB's actual sample, 197 nodes), and
  `sample2_tree`/`hsn_tree`/`hss_tree` are ported alongside it from the
  bundled `.mtr` files. The 2252-node `25HSS.swc` that `sample_tree()`
  returned was a stand-in from when only SWC loading existed; it is the
  HSS cell, X-mirrored and stripped of its `axon`/`dend`/`soma` regions by
  the SWC format. It becomes `hss_tree()`, in regioned `.mtr` form.
- **#52** — `_flatten` accepts `scipy.io.matlab.mat_struct`, not just
  `dict`/`list`/`ndarray`. Without it `load_mtr` cannot read `dLPTCs.mtr`,
  the bundled 55-tree/5-group population that `stats_tree`'s group API
  exists to consume.
