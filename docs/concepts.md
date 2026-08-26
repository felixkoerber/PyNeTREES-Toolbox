# Core concepts

Everything in `pynetrees` is built on one data structure and a handful of
conventions. Learn these and the ~90 functions in the toolbox become mostly
self-explanatory.

## 1. A tree is nodes plus a parent pointer

A neuron morphology is stored as a `Tree`: a flat list of **nodes** (points in
3D with a diameter), plus an adjacency matrix saying which node is whose
parent.

```python
from pynetrees import sample_tree

tree = sample_tree()
tree                      # Tree(name='25HSS', n_nodes=2252, regions=[dend])
tree.n_nodes              # 2252
tree.X, tree.Y, tree.Z    # float arrays, one entry per node -- coordinates [um]
tree.D                    # float array -- diameter at each node [um]
tree.R                    # int array -- which region each node belongs to
tree.rnames               # list[str] -- region names, indexed by R
tree.dA                   # sparse matrix -- the topology (see below)
```

Every per-node quantity is a plain NumPy array of length `n_nodes`, in the same
order. This is the single most important thing to internalise: **node `i`'s
coordinate is `tree.X[i]`, its diameter is `tree.D[i]`, its branch order is
`BO_tree(tree)[i]`** — everything lines up by index, so you can freely combine
results with ordinary NumPy operations.

```python
import numpy as np
from pynetrees import BO_tree, len_tree

length = len_tree(tree)          # length of each node's segment
order  = BO_tree(tree)           # branch order of each node
total_length_of_high_order = length[order > 3].sum()   # just NumPy
```

### The adjacency matrix `dA`

`dA` is a sparse `n_nodes × n_nodes` matrix where `dA[i, j] == 1` means
**"node `j` is node `i`'s parent"**. Each row has at most one entry; the root's
row is empty. This is inherited unchanged from the MATLAB toolbox.

You rarely touch `dA` directly — `idpar_tree(tree)` gives you the parent index
of every node as a plain array, which is what you actually want:

```python
from pynetrees import idpar_tree

parent = idpar_tree(tree)   # parent[i] is node i's parent index
parent[0]                   # 0 -- the root is its own parent by default
idpar_tree(tree, root_self=False)[0]   # -1 -- explicit "no parent" instead
```

## 2. Indexing is 0-based, and "no parent" is `-1`

MATLAB is 1-based and overloads `0` to mean "no parent". Python is 0-based, so
`0` is a perfectly valid node index and can't double as a sentinel.

- **All node indices are 0-based.** Node `0` is normally the root.
- **`NO_PARENT` is `-1`.** Anywhere a function reports "there is no node here"
  it uses `-1`, never `0`.

```python
from pynetrees import NO_PARENT   # == -1
```

If you're translating MATLAB code, this is the single most common source of
off-by-one bugs. See [matlab-migration.md](matlab-migration.md).

## 3. Regions group nodes into named compartments

`tree.R[i]` is an integer index into `tree.rnames`, tagging each node as soma,
axon, a particular dendritic layer, and so on.

```python
gc.rnames
# ['axon', 'GCL', 'adendIML', 'adendMML', 'adendOML', 'adendOMLout', 'soma', 'axonh']

gc.region_nodes("soma")                              # indices of somatic nodes
gc.region_nodes("adendIML", "adendMML", "adendOML")  # all dendritic layers
gc.region_mask("axon")                               # boolean mask instead
gc.region_index("soma")                              # the raw R value
```

Regions drive plotting colours, per-compartment biophysics, and section
splitting when you build a NEURON model — so they matter more than they first
appear.

## 4. Functions are `verb_tree(tree, ...)` and return arrays, not pictures

The toolbox is deliberately **function-oriented**, mirroring MATLAB's naming so
existing knowledge transfers directly:

```python
len_tree(tree)      # segment lengths
surf_tree(tree)     # segment surface areas
eucl_tree(tree)     # straight-line distance to root
PL_tree(tree)       # topological path length (number of nodes)
BO_tree(tree)       # branch order
T_tree(tree)        # boolean mask: is this node a terminal?
B_tree(tree)        # boolean mask: is this node a branch point?
```

Three rules hold almost everywhere:

1. **The first argument is the tree.**
2. **Return values are per-node NumPy arrays** (or boolean masks) of length
   `n_nodes`, unless documented otherwise.
3. **Nothing plots as a side effect.** MATLAB's functions take a `'-s'` "show"
   option that draws a figure; the port drops it everywhere. Pass the result to
   a plotting function instead — which is more flexible anyway:

```python
from pynetrees import plot_tree, BO_tree
plot_tree(tree, BO_tree(tree), cmap="viridis")   # color = per-node values
```

## 5. Trees are immutable in practice

Editing functions return a **new** tree rather than modifying in place:

```python
from pynetrees import resample_tree, tran_tree

resampled = resample_tree(tree, 5.0)   # tree is unchanged
moved     = tran_tree(tree, [100, 0, 0])
```

MATLAB's versions mutate a global `trees` array when called without an output
argument. There's no hidden global state here — if you want the result, assign
it.

## 6. BCT conformity: what `repair_tree` is for

Most analysis functions assume the tree is *BCT-conform*: strictly binary
branching, no zero-length segments, and nodes ordered so a parent always
precedes its children. Real reconstructions often aren't.

```python
from pynetrees import ver_tree, repair_tree

ver_tree(tree)          # returns a list of problems; never raises
clean = repair_tree(tree)   # fix them
```

`ver_tree` **warns rather than failing**, matching MATLAB — a tree can be
legitimately half-built in the middle of a pipeline. Call `repair_tree` before
serious analysis if your data came from an unknown source. Functions with a
hard requirement (e.g. `angleB_tree`, which needs binary branch points) raise a
clear `ValueError` telling you to run `repair_tree` first, rather than silently
computing something wrong.

## 7. Physical parameters for simulation live on the tree

Cable-theory and simulation functions need three physical constants that aren't
part of the morphology:

```python
tree.Ri = 100.0        # axial resistivity     [Ohm*cm]
tree.Gm = 1 / 2500.0   # membrane conductance  [S/cm^2]
tree.Cm = 1.0          # membrane capacitance  [uF/cm^2]
```

Each may be a scalar (uniform) or a per-node array. They default to `None`, and
any function needing one raises a `ValueError` naming the missing attribute
rather than silently assuming a value — there is no universally correct default,
and quietly picking one would produce plausible-looking but meaningless numbers.

## 8. Optional dependencies

`import pynetrees` only needs NumPy, SciPy and pandas. Heavier things are
optional and imported lazily, so they only fail if you actually call into them:

| Feature | Needs | Install |
|---|---|---|
| 3D plotting | `pyvista` | `pip install -e ".[plot]"` |
| 2D/dendrogram plots | `matplotlib` | `pip install -e ".[plot]"` |
| NEURON simulation | `neuron` | see [guide.md](guide.md#simulating-with-neuron) |

## Where to go next

- [guide.md](guide.md) — doing actual work with these pieces
- [api-overview.md](api-overview.md) — the full function list, grouped by purpose
- [matlab-migration.md](matlab-migration.md) — if you're coming from the MATLAB toolbox
