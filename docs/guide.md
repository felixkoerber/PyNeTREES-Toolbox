# User guide

Task-oriented walkthrough. Assumes you've skimmed [concepts.md](concepts.md).

- [Loading and saving](#loading-and-saving)
- [Inspecting a tree](#inspecting-a-tree)
- [Measuring](#measuring)
- [Selecting parts of a tree](#selecting-parts-of-a-tree)
- [Editing](#editing)
- [Generating synthetic trees](#generating-synthetic-trees)
- [Plotting](#plotting)
- [Passive cable analysis](#passive-cable-analysis)
- [Simulating with NEURON](#simulating-with-neuron)
- [Comparing populations](#comparing-populations)
- [Performance notes](#performance-notes)

## Loading and saving

```python
from pytrees import (load_swc, save_swc, load_mtr, load_neurolucida,
                     sample_tree, hss_tree)

tree  = sample_tree()                    # MATLAB's 197-node sample
big   = hss_tree()                       # or the full 2252-node HSS cell
tree  = load_swc("cell.swc")             # standard SWC
cells = load_mtr("population.mtr")       # MATLAB .mtr archive
cells = load_neurolucida("cell.ASC")     # NeuroLucida ASCII
```

Loaders return a single `Tree` when the file holds one, or a `list[Tree]` when
it holds several — `.mtr` and NeuroLucida files usually hold several, so check:

```python
cells = load_mtr(path)
gc = cells[0] if isinstance(cells, list) else cells
```

For round-tripping trees *within* Python, `save_tree`/`load_tree` use a native
`.npz` format that preserves everything (including regions and `frustum`),
unlike SWC which is lossy:

```python
from pytrees import save_tree, load_tree
save_tree(tree, "cell.npz")
tree = load_tree("cell.npz")
```

> `.mtr` support covers MATLAB v5 files. A v7.3 (HDF5) file raises a clear
> error naming the file rather than failing obscurely.

## Inspecting a tree

```python
tree                       # Tree(name='25HSS', n_nodes=2252, regions=[dend])
tree.n_nodes
tree.rnames
tree.validate()            # list of structural problems; [] means healthy
```

`validate()` (a.k.a. `ver_tree`) never raises — it returns problems and warns.
Fix them with `repair_tree`.

## Measuring

All of these return one value per node, aligned with `tree.X` etc.

```python
from pytrees import (
    len_tree, surf_tree, vol_tree, eucl_tree, PL_tree, Pvec_tree,
    BO_tree, strahler_tree, angleB_tree, T_tree, B_tree, C_tree,
)

length = len_tree(tree)        # segment length [um]
surf   = surf_tree(tree)       # segment lateral surface [um^2]
eucl   = eucl_tree(tree)       # straight-line distance to root [um]
plen   = Pvec_tree(tree, length)   # *metric* path length to root [um]
topo   = PL_tree(tree)         # *topological* path length (node count)
order  = BO_tree(tree)         # branch order
```

`Pvec_tree` is the general "accumulate a per-node quantity along the path to
the root" operation — pass it any vector, not just lengths.

Node-type masks are booleans, so they compose with NumPy directly:

```python
terminals = T_tree(tree)              # mask
n_tips    = terminals.sum()
tip_depth = plen[terminals].max()     # furthest tip, in path length
```

Whole-tree summaries are just NumPy reductions over these:

```python
total_length = tree.total_length          # == len_tree(tree).sum()
total_surface = surf_tree(tree).sum()
n_branch_points = B_tree(tree).sum()
```

## Selecting parts of a tree

By region:

```python
soma = tree.region_nodes("soma")
dend = tree.region_nodes("adendIML", "adendMML", "adendOML")
```

By subtree — everything downstream of a node:

```python
from pytrees import sub_tree
mask, subtree = sub_tree(tree, node)   # mask incl. the node, plus the
                                       # subtree cut out as a real Tree
mask = sub_tree(tree, node, with_tree=False).mask   # in a hot loop
```

By section — the stretches between branch/termination points:

```python
from pytrees import dissect_tree
sections = dissect_tree(tree)      # (n_sections, 2) array of (start, end) nodes
```

Sections are the natural unit for anything compartmental — they're exactly what
becomes a NEURON `Section`.

## Editing

Every editing function returns a **new** tree.

```python
from pytrees import repair_tree, resample_tree, delete_tree, cat_tree

clean     = repair_tree(tree)              # enforce BCT conformity
resampled = resample_tree(tree, 5.0)       # ~5 um internode spacing
pruned    = delete_tree(tree, nodes_to_drop)
joined    = cat_tree(tree1, tree2)
```

`resample_tree` **preserves branch and termination points exactly** and only
redistributes nodes along the interior of each section. (MATLAB's version
snaps them onto the resampling grid instead — a deliberate difference, see
PORT_STATUS.md Design Decision #23.)

`delete_tree` returns a `list[Tree]` if the deletion disconnects the tree,
otherwise a single `Tree`.

Geometric transforms:

```python
from pytrees import tran_tree, rot_tree, scale_tree, flip_tree, flatten_tree

tran_tree(tree, [100, 0, 0])    # translate; or pass a node index to centre on it
rot_tree(tree, (0, 0, 90))      # rotate by degrees about x, y, z
scale_tree(tree, 2.0)
flip_tree(tree, axis="x")
flatten_tree(tree)              # project to XY, conserving segment lengths
```

## Generating synthetic trees

```python
from pytrees import MST_tree, BCT_tree, soma_tree, quaddiameter_tree, jitter_tree
import numpy as np

pts = np.random.rand(300, 3) * 100
tree, connected = MST_tree(pts[:, 0], pts[:, 1], pts[:, 2], bf=0.4, full_output=True)[:2]
```

> **Gotcha:** `MST_tree` is the one tree-producing function that returns a
> **tuple**, not a bare `Tree`. `connected` is a boolean mask over the input
> points saying which ones actually made it in — points farther than `thr`
> from everything, or beyond `mplen` of path length, get dropped silently
> otherwise.

`bf` is the **balancing factor**: `0` minimises total wiring length, `1`
minimises path length to the root. It's the main knob controlling how a
synthetic arbor looks.

```python
tree = quaddiameter_tree(tree)   # give it a realistic tapering diameter
tree = soma_tree(tree, 30.0)     # add a soma
tree = jitter_tree(tree, 1.0)    # add correlated positional noise
```

## Plotting

3D, via PyVista (the recommended path):

```python
from pytrees import plot_tree, BO_tree

pl = plot_tree(tree, BO_tree(tree), cmap="viridis", mode="tube")
pl.show()
```

- `mode="tube"` renders true diameters; `mode="line"` is much faster for
  large scenes or many cells.
- `scalars=` accepts **any** per-node array — colour by any measurement.
- `plotter=` overlays onto an existing plot (MATLAB's `hold on`).

> **Gotcha:** `scalars` must always be the *full-tree* array of length
> `n_nodes`, even when `nodes=` restricts what's drawn. The underlying mesh
> keeps every node's coordinates regardless. Passing a sliced array raises a
> clear `ValueError`.

For categorical data like regions, pass `categories=True` (any extra keyword
flows through to PyVista's `add_mesh`):

```python
plot_tree(tree, tree.R.astype(float), cmap="tab10", categories=True)
```

2D and diagrams, via matplotlib:

```python
from pytrees import plot_mpl_tree, dendrogram_tree, dA_tree

plot_mpl_tree(tree)        # lighter 2D/3D line plot
dendrogram_tree(tree)      # abstract topology diagram
dA_tree(tree)          # sparsity pattern of the adjacency matrix
```

Laying out several cells side by side:

```python
from pytrees import spread_trees
for cell in spread_trees(cells, dx=100, dy=100):
    pl = plot_tree(cell, mode="line", plotter=pl)
```

## Passive cable analysis

You can do a lot of electrotonics *without* a simulator. Set the physical
constants first:

```python
tree.Ri, tree.Gm, tree.Cm = 100.0, 1/2500, 1.0
```

```python
from pytrees import M_tree, sse_tree, syn_tree, lambda_tree, elen_tree, cgin_tree

M   = M_tree(tree)          # sparse conductance matrix of the equivalent circuit
sse = sse_tree(tree)        # full steady-state matrix (exact, by inversion)
```

> **Voltages are deviations from rest, not absolute membrane potentials.**
> Rest is `0`, and `Ee`/`Ei` are driving forces relative to it (defaults `+60`
> and `-20` mV, matching MATLAB). Passing `Ee=0` means "no driving force" and
> correctly returns an all-zero solution — it is not a bug.

`sse_tree()` with no current argument returns the **whole `n × n` transfer
matrix**: column `i` is the voltage everywhere when 1 nA is injected at node
`i`, so the diagonal is each node's input resistance. For one injection site,
pass its index — much cheaper:

```python
v = sse_tree(tree, I=soma_node)      # voltage everywhere from 1 nA at soma_node
```

Steady-state synaptic input:

```python
v = syn_tree(tree, ge=synapse_node, Ee=0.0)
```

Time-stepping (integrate-and-fire over the full morphology):

```python
from pytrees import LIF_tree
import numpy as np

t = np.linspace(0, 200, 2001)
I = np.zeros((tree.n_nodes, t.size))
I[soma_node, 100:600] = 0.5
v, spikes = LIF_tree(tree, time=t, I=I, iroot=soma_node)
```

## Simulating with NEURON

For active conductances and anything beyond passive cable theory, hand the tree
to the real NEURON simulator.

**Install:** on Linux/macOS `pip install neuron`. On **Windows there is no pip
wheel** — install the official binary from
[neuronsimulator.github.io](https://www.neuronsimulator.org/en/latest/install/install_instructions.html)
and make sure it's linked against the same Python.

```python
from pytrees import build_neuron_model, insert_mechanism, run_current_clamp

tree.Ri, tree.Gm, tree.Cm = 100.0, 1/2500, 1.0

model = build_neuron_model(tree)         # one h.Section per dissected section
insert_mechanism(model, "hh", region="soma")   # active soma
t, v = run_current_clamp(model, at_node=soma_node, amp=0.3,
                         delay=5, dur=50, tstop=100,
                         record_nodes=[soma_node, tip_node])
```

- Sections are built from the real 3D points via `pt3dadd`, so diameters taper
  exactly as in the morphology.
- Segment counts use NEURON's own d-lambda rule.
- `model.loc(node)` gives the NEURON segment for any tree node, so you can
  attach any NEURON object yourself:

```python
from neuron import h
syn = h.Exp2Syn(model.loc(dend_node))
```

> A region occupying only the root node gets absorbed into the following
> section — model a soma with more than one node if you want to give it its
> own mechanisms.

## Comparing populations

`stats_tree` collects comparable measurements across groups of cells and
returns tidy pandas DataFrames:

```python
from pytrees import stats_tree

res = stats_tree([control_cells, treated_cells], group_names=["control", "treated"])
res["summary"]     # one row per tree: total length, branch points, ...
res["points"]      # one row per branch/termination point
res["branches"]    # one row per dissected branch
```

Because these are ordinary DataFrames, comparison and plotting are just pandas:

```python
res["summary"].groupby("group")["len"].describe()
res["points"].boxplot(column="Plen", by="group")
```

`extras=True` adds convex-hull volume, mean asymmetry, and a `"sholl"` frame
with intersection counts on a common radius grid across all cells.

Other population-level tools:

```python
from pytrees import sholl_tree, bf_tree, peters_tree

sholl_tree(tree, 25.0).s     # Sholl intersection counts
bf_tree(tree)                # (balancing factor, centripetal bias) estimates
peters_tree(axon, dendrite)  # candidate synapse sites between two cells
```

## Performance notes

Typical timings on a real 3765-node granule cell (see
`PORT_STATUS.md` for details):

| Operation | Time |
|---|---|
| `len_tree`, `surf_tree`, `eucl_tree` | < 1 ms |
| `dissect_tree`, `sort_tree`, `idpar_tree` | a few ms |
| `BO_tree`, `PL_tree`, `sholl_tree` | tens of ms |
| `flatten_tree`, `morph_tree`, `M_tree`, `sse_tree(I=k)` | tens of ms |
| `repair_tree`, `resample_tree`, `smooth_tree` | ~0.5–1 s |
| `sse_tree()` (full `n × n` inverse) | seconds, and `O(n²)` memory |

Two things to be aware of:

- **`ipar_tree` builds a dense `n_nodes × max_depth` matrix** (~49 MB for the
  granule cell above). It's the right tool for ancestor queries, but if you
  only need parents use `idpar_tree`, and if you need *descendants* in a loop
  prefer `sub_tree` or `dissect_tree` — see the note in
  `graphtheory._subtree_blocks`.
- **`sse_tree()` with no argument inverts a dense `n × n` matrix.** For a
  single injection site pass `I=node` instead.
