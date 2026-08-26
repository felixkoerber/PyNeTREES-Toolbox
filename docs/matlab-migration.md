# Migrating from the MATLAB TREES toolbox

For people who already know `treestoolbox-master`. Function names are kept
identical wherever possible, so most of your knowledge transfers directly —
but a few conventions changed deliberately.

## The six things that will bite you

### 1. Indexing is 0-based

```matlab
tree.X(1)            % MATLAB: first node
idpar_tree(tree)(1)  % root's parent == 1 (itself)
```
```python
tree.X[0]            # Python: first node
idpar_tree(tree)[0]  # root's parent == 0 (itself)
```

Anywhere you had a literal node index, subtract one.

### 2. "No parent" is `-1`, not `0`

MATLAB uses `0` as the sentinel, which is safe only because `0` isn't a valid
1-based index. In Python it *is* a valid index, so the port uses `-1`:

```python
from pynetrees import NO_PARENT, idpar_tree, ipar_tree   # NO_PARENT == -1

idpar_tree(tree, root_self=False)[0]   # -1
ipar_tree(tree)                     # padded with -1, not 0
```

If you translate a MATLAB loop that checks `if idpar(n) ~= 0`, it becomes
`if idpar[n] != NO_PARENT`.

### 3. Option strings became keyword arguments

MATLAB's `'-s'`/`'-2d'`/`'-LO'` option strings are gone entirely — the port
uses real named parameters. (The MATLAB maintainers' own todo list documents
that parser as a recurring source of bugs.)

| MATLAB | pynetrees |
|---|---|
| `len_tree(tree, '-dim2')` | `len_tree(tree, dim=2)` |
| `sort_tree(tree, '-LO')` | `sort_tree(tree, by="lo")` |
| `idpar_tree(tree, '-0')` | `idpar_tree(tree, root_self=False)` |
| `flip_tree(tree, 1)` | `flip_tree(tree, axis="x")` |
| `sholl_tree(tree, dd, '-o')` | `sholl_tree(tree, dd, single_only=True)` |
| `asym_tree(tree, [], '-v')` | `asym_tree(tree, van_pelt=True)` |
| `delete_tree(tree, i, '-r')` | `delete_tree(tree, i, keep_regions=True)` |

### 4. Sample trees: `sample_tree()` is MATLAB's sample again

All four MATLAB sample loaders are ported, with MATLAB's meanings:

| pynetrees | file | nodes | |
|---|---|---|---|
| `sample_tree()` | `sample.mtr` | 197 | subtree of an HSN cell |
| `sample2_tree()` | `sample2.mtr` | 15 | minimal tree |
| `hsn_tree()` | `hsn.mtr` | 1290 | full HSN cell |
| `hss_tree()` | `hss.mtr` | 2252 | full HSS cell |
| `dLPTCs_trees()` | `dLPTCs.mtr` | 55 trees | 5 named groups, for `stats_tree` |

**If you used `sample_tree()` before pynetrees 0.0.2**, it returned a different
cell — the 2252-node HSS reconstruction, loaded from SWC, back when `.mtr`
reading wasn't implemented. That tree is now `hss_tree()`, and in its `.mtr`
form it also regains what SWC had stripped: its real `axon`/`dend`/`soma`
region names, and its original (un-mirrored) orientation.

### 5. `'-s'` (show) is gone everywhere — plot the result yourself

No function draws a figure as a side effect. This is more flexible:

```matlab
BO_tree(tree, '-s');
```
```python
plot_tree(tree, BO_tree(tree), cmap="viridis")
```

### 6. Nothing mutates a global `trees` array

MATLAB functions modify a global when called without an output. Here, editing
functions always return a new tree and never touch the input:

```python
tree = resample_tree(tree, 5.0)    # assign it, or the result is lost
```

## Common idioms translated

**Find nodes in a region.** The MATLAB incantation appears dozens of times in
the GC model scripts:

```matlab
nodes_soma = find(tree.R == find(strcmp(tree.rnames,'soma')));
soma_node  = nodes_soma(round(length(nodes_soma)/2));
```
```python
nodes_soma = tree.region_nodes("soma")
soma_node  = int(nodes_soma[len(nodes_soma) // 2])
```

Unlike MATLAB's `strcmp` (which silently returns empty for an unknown name and
then quietly matches nothing), `region_nodes` raises a `KeyError` listing the
available regions.

**Metric path length to the root:**

```matlab
Plen = Pvec_tree(tree);            % defaults to len_tree
```
```python
plen = Pvec_tree(tree, len_tree(tree))   # v is explicit -- no implicit default
```

**Walk from a terminal up to the root:**

```matlab
idpar = idpar_tree(tree);
n = start_node;
while idpar(n) ~= 1
    n = idpar(n);
end
```
```python
idpar = idpar_tree(tree, root_self=False)
n = start_node
while idpar[n] != NO_PARENT:
    n = idpar[n]
```

**Furthest terminal:**

```matlab
[~, i] = max(T_tree(tree) .* PL_tree(tree));
```
```python
i = int(np.argmax(T_tree(tree) * PL_tree(tree)))
```

## Renamed functions

Mostly to avoid ambiguity, since Python has no `'-s'` to disambiguate:

| MATLAB | pynetrees | Why |
|---|---|---|
| `dA_tree` | `dA_tree` | avoids reading as "plot the tree" |
| `plot_tree` (matplotlib path) | `plot_mpl_tree` | `plot_tree` is the PyVista one |
| `stats_tree` + `dstats_tree` | `stats_tree` returning DataFrames | see below |
| `neuron_template_tree`/`t2n` | `build_neuron_model` etc. | no `.hoc` text involved |

## Structural differences worth knowing

**`stats_tree` returns DataFrames, not nested structs.** MATLAB returns
`gstats`/`dstats` structs of cell arrays; the port returns a dict of tidy
pandas DataFrames (`summary`, `points`, `branches`, optionally `sholl`).
`dstats_tree` isn't ported — plotting DataFrames is a solved problem.

**Populations are plain lists.** There is no `Trees` class. Use `list[Tree]`
plus pandas, which is what the MATLAB class was reimplementing by hand.

**NEURON integration talks to NEURON directly.** No `.hoc` files, no
subprocess, no exchange folder — `build_neuron_model(tree)` constructs
`h.Section` objects in-process. T2N's cluster/SSH machinery has no purpose
here and isn't ported.

**Trees carry `Ri`/`Gm`/`Cm` as real attributes**, rather than fields bolted on
ad hoc. They default to `None` and raise if a function needs them unset.

## Behaviour that deliberately differs

These are places where the port does something *different*, not just
differently spelled — each is documented in `PORT_STATUS.md`'s design log.

| Function | Difference |
|---|---|
| `resample_tree` | Preserves branch/termination points exactly instead of snapping them onto the grid (#23) |
| `rootangle_tree` | Centres on the root first; MATLAB measures from the coordinate origin, which only matches its own docstring if the tree happens to sit there (#24) |
| `delete_tree` | Always splices to the nearest surviving ancestor and returns a forest when disconnected; MATLAB's default is documented-broken (#22) |
| `strahler_tree` | Handles multifurcations; MATLAB assumes binary (#14) |
| `asym_tree`, `angleB_tree` | Raise on non-binary branch points instead of silently using the first two children (#15) |
| `clean_tree` | Root is never itself deletable (#28) |
| `jitter_tree` | Node-to-itself distance is 0, not MATLAB's artefactual 2 (#29) |
| `dissect_tree` | Region cuts land on the parent of the transition node — matching MATLAB, after a bug in an earlier version of this port (#36) |
| `LIF_tree` | Drops MATLAB's dead `Vzone` parameter |
| `AdExLIF_tree` | Returns the full voltage trace; MATLAB hardcodes node 1 regardless of `iroot` |

Bugs found in the MATLAB original are catalogued separately in
[`MATLAB_TOOLBOX_BUGS.md`](../MATLAB_TOOLBOX_BUGS.md).

## Not ported

`clone_tree`/`gscale_tree`, `hull_tree`/`vhull_tree`/`gdens_tree`/`lego_tree`,
`convexity_tree`/`boundary_tree`/`dissectSholl_tree`/`r_mc_tree`/`M_atten_tree`,
`BLO_tree`/`barcode_tree`, `growth_tree`/`random_tree`, `span_tree`/`theta_tree`
and `scaleS_tree`/`scaleV_tree` were all missing here at one point; every one of
them is ported now. What's left, with the reasoning in `PORT_STATUS.md` and the
itemised inventory in `NOT_YET_PORTED.md`:

- **`cgui_tree`** and the GUIDE GUI — a full MATLAB GUI application; a Python
  equivalent would be a rewrite against a different toolkit, not a port.
- **`pov_tree`/`x3d_tree`** (POV-Ray/X3D scene export) — planned;
  `pynetrees.blender` covers the same rendering need today.
- **`fix_tree`/`fix_tree_UI`/`finetune_fix_tree`** — a MATLAB figure-callback
  GUI; MATLAB's own todo list flags these as incomplete.
- **`GC_biophys`** (the Active GC Model's active-conductance fitting) —
  substantial, scoped separately.
- **`parseArgs`/`isBinary`** and other MATLAB plumbing — superseded by real
  Python arguments.
- **T2N protocol library** (IV/FI/resonance/bAP) and its cluster/SSH execution
  mode — thin wrappers over `run_current_clamp`; add on demand.
- **v7.3 (HDF5) writing** for `save_tree`/`save_mtr` — the current write path
  is MATLAB v5, capped at ~2 GB per variable.

### `plot_tree`'s `color` argument

`color` is polymorphic exactly as in MATLAB, and the positional order now
matches too — `plot_tree(tree, color, DD, ipart, res)`:

```python
plot_tree(tree, "red")                       # flat colour name
plot_tree(tree, (1.0, 0.0, 0.0))             # flat RGB triple
plot_tree(tree, BO_tree(tree))               # per-node values -> colormap
plot_tree(tree, rgb_array)                   # (n_nodes, 3) explicit RGB
plot_tree(tree, "blue", (100, 0, 0), nodes)  # MATLAB's DD and ipart
```

The one ambiguity is a **3-node tree**, where a length-3 vector could be an
RGB triple or three per-node values; it is read as RGB, matching MATLAB.
Pass `scalars=` to force the other reading.
