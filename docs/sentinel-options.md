# The `-1` sentinel: what it costs, and what we could do instead

**Status: evaluation, not a proposal to act on yet.** Nothing in the package
changes because of this document. It exists so the decision is made once,
deliberately, with the trade-offs written down — rather than drifting one
function at a time.

---

## The problem in one line

`-1` is a **valid NumPy index**, so every place we use it to mean "there
isn't one" is a place where a missing value silently becomes *the last
element* instead of an error.

```python
parent = idpar_tree(tree, root_self=False)   # root -> -1
depth  = some_per_node_array
depth[parent]        # the root quietly gets depth[-1]: the LAST node's value
```

Nothing raises. Nothing warns. The result has the right shape and the right
dtype, and one entry in it is nonsense. That is the worst failure mode a
numerical library has, and it is the entire case against the status quo.

## Where it is used, and for what

The one spelling covers three genuinely different jobs. That is worth
separating before choosing a fix, because the right answer is not obviously
the same for all three.

| Role | Where | Shape of the problem |
|---|---|---|
| **1. "no parent"** | `idpar_tree(root_self=False)`, `Tree.dA` construction, every `.swc`/`.neu`/`.nmf`/Neurolucida reader | One distinguished node per tree. Consumers almost always want to *branch* on it, not index with it. |
| **2. Padding** | `ipar_tree` (ragged ancestor paths in a rectangular array), `idchild_tree` (ragged child lists likewise) | Whole trailing regions of an array are "not a node". Consumers mask, or iterate until they hit it. |
| **3. "no value"** | `MST_tree`'s `indx` for unconnected points, `dissect_tree(full_output=True)`'s section index | A stand-in for NaN in an integer array. |

Roles 2 and 3 are the ones that get *indexed with* by accident. Role 1 is
mostly consumed by comparison, and the library is already careful about it
(audited 2026-09-08: every internal `v[idpar]` uses `root_self=True`, where
the root refers to itself and indexing is well defined).

So the exposure is real but concentrated: it is in **user** code and in any
**new** library code written without that context.

---

## Options

### A. Keep `-1`, add guardrails

Do nothing to the representation. Instead:

- keep the docstring warnings that now exist on `idpar_tree`;
- keep `path_to_root` and friends so the common walks never need the sentinel
  in the first place;
- add a debug-mode checker (`PYNETREES_STRICT=1`) that wraps the returning
  functions and raises if a sentinel array is used as an index.

**For.** Zero churn. `-1` is what MATLAB's `0` means, so the mental mapping
from the original stays direct. Integer dtype throughout, so no allocation or
speed cost anywhere.

**Against.** The failure mode survives. A guard you have to remember to turn
on is a guard for people who already knew.

### B. Masked arrays (`numpy.ma`)

Return `np.ma.masked_array` with the sentinel positions masked.

**For.** This is exactly what masked arrays are for. Indexing with a masked
value propagates the mask rather than silently wrapping, and `.filled(-1)`
recovers today's behaviour in one call for anyone who wants it.

**Against.** `numpy.ma` is semi-deprecated in practice — poorly supported by
SciPy, pandas and numba, slower, and full of sharp edges (`masked_array` in a
`scipy.sparse` constructor silently drops the mask). Handing one to a user
who then passes it to `scipy.optimize` creates a *new* class of silent bug to
replace the old one. It also does not survive `.npz`.

### C. `np.nan` in a float array

Return floats with `nan` where there is no value.

**For.** The idiomatic "no value" in numeric Python. `nan` propagates loudly
through arithmetic, and `arr[np.isnan(x)]` raises `IndexError` rather than
wrapping — the exact failure we want.

**Against.** These *are* indices; making them floats means every consumer
casts back to int to use them, and a cast is another place to lose the
sentinel. It doubles memory on the padded arrays, which are the biggest ones
we produce (`ipar_tree` is `n_nodes × max_depth`). And `nan != nan` breaks
the equality tests that read naturally today.

### D. A different sentinel that cannot index

Use a large negative number — `np.iinfo(np.int64).min`, or a named
`NO_NODE = -(2**62)`.

**For.** One-line change to a constant. Indexing with it raises `IndexError`
immediately, which is precisely the improvement wanted. Stays integer, stays
fast, stays `.npz`-storable, and every existing `!= NO_PARENT` comparison
keeps working unchanged.

**Against.** Arithmetic on it overflows rather than raising
(`NO_NODE - 1` wraps). `-1` is conventional and readable in a debugger where
`-4611686018427387904` is not. Any user code doing `parent >= 0` — a
reasonable thing to have written — keeps working, but `parent == -1` breaks
silently, and that spelling is out there.

### E. Split the roles

Keep `-1` for role 1 ("no parent"), where it is well understood and mostly
compared rather than indexed; use a distinct representation for roles 2 and 3.
Concretely: `ipar_tree`/`idchild_tree` return `(values, valid)` pairs, or a
list of ragged arrays, so padding is never a magic number at all.

**For.** Fixes the roles that actually get mis-indexed, leaves alone the one
that does not. A ragged return for a ragged concept is honest, and
`idchild_tree`'s rectangular output is already awkward (its width is "the
largest number of children found", which is data-dependent).

**Against.** The biggest API churn of any option here, and a list-of-arrays
return is slower to consume than one rectangular block for the vectorised
uses that motivated the rectangle.

---

## Recommendation

**D for the padding and no-value roles, plus A's guardrails, and leave role 1
alone.** Reasoning:

1. Role 1 is not where the bug lives. `NO_PARENT = -1` is compared, not
   indexed, in every internal use, it maps directly onto MATLAB's `0`, and it
   is what `.swc` readers naturally produce. Changing it buys little and
   breaks the mental model.
2. Roles 2 and 3 are where a stray index lands, and there the sentinel is
   never meaningfully compared to `-1` by users — they test `>= 0` or mask.
   A non-indexable sentinel there is close to free and removes the failure.
3. E is the *most* correct answer and the one to reach for if `idchild_tree`
   is ever reworked for other reasons. It is not worth doing on its own.
4. B and C both trade a silent wrap for a different silent failure
   (mask-dropping, float-cast) in a library whose users hand arrays straight
   to SciPy.

**Cost to do it:** one constant, a `dtype` audit of the padded producers, and
a sweep of the `== -1` / `!= NO_PARENT` comparisons — which are already
centralised behind the named constant, so the sweep is mechanical. Roughly a
day, most of it re-reading tests.

**What would change my mind:** if we ever return these arrays to callers who
serialise them to `.mat` for MATLAB, a huge negative integer round-trips
worse than `-1` does, and role 1's mapping to MATLAB's `0` becomes the more
valuable property. Worth checking against the `Tree.meta` MATLAB round-trip
work before committing.

---

## What is already true, and worth keeping either way

- `path_to_root(tree, node)` exists so the commonest parent walk never
  touches a sentinel.
- `idpar_tree`'s docstring states the indexing hazard outright.
- Every internal `v[idpar]` uses `root_self=True`. Verified by audit
  2026-09-08; worth a test that keeps it true rather than a comment that
  hopes so.
