# PyNeTREES Toolbox

A Python port of the [TREES toolbox](https://www.treestoolbox.org/) — load, edit,
measure, simulate and visualise neuronal branching structures.

`pynetrees` reads neuronal reconstructions (SWC, NeuroLucida `.ASC`, MATLAB `.mtr`),
gives you the toolbox's graph-theoretic and morphometric analyses on top of a plain
NumPy/SciPy data structure, does passive cable analysis without a simulator, builds
synthetic trees, and — where you do want a simulator — turns a morphology into a live
NEURON model in-process.

```python
import pynetrees as pt

tree = pt.sample_tree()
print(tree)
# Tree(name='sample', n_nodes=197, regions=[dendrite, subtree])
```

It is a **port, not a binding**: no MATLAB installation, license, or bridge is
involved. Function names and argument order follow the MATLAB original closely
enough that existing TREES knowledge transfers directly — see
[docs/matlab-migration.md](docs/matlab-migration.md).

---

## Requirements

- **Python 3.10+** (developed and tested on 3.11)
- **NumPy, SciPy, pandas** — installed automatically
- **PyVista, matplotlib, scikit-image** — optional, for plotting
- **mat73** — optional, to read MATLAB v7.3 `.mtr`/`.mat` files, which is
  what a current TREES install writes
- **NEURON 8+** — optional, only for `pynetrees.neuron_bridge`

## Installation

### 1. Create an isolated environment

Nothing in this repo does this for you, and installing a scientific stack straight
into a global or `base` Python is easy to regret later.

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd.exe):
.venv\Scripts\activate.bat
```

Prefer conda? `conda create -n pynetrees python=3.11 -y && conda activate pynetrees`
works just as well — the rest of these notes are the same either way.

### 2. Install the package

```bash
git clone https://github.com/<your-user>/pynetrees.git
cd pynetrees

# Linux / macOS — everything:
pip install -e ".[all]"

# Windows — everything except NEURON (no pip wheel; see step 4):
pip install -e ".[all-no-neuron]"
```

`all` pulls in every runtime dependency — plotting, NEURON, all the I/O
formats, image stacks, the notebook stack (ipykernel + trame), and pytest —
**except Blender** (`bpy` is a ~300 MB wheel that pins `numpy < 2`; add it
with `".[all,blender]"` only if you need it, and see the note below).
`all-no-neuron` is the same set with the `neuron` line dropped, so the
install succeeds on Windows where `pip install neuron` cannot.

The extras are separable if you want a smaller install:

| Command | What you get |
|---|---|
| `pip install -e .` | Core only — NumPy, SciPy, pandas |
| `pip install -e ".[all]"` | **Everything except Blender** — the default on Linux/macOS |
| `pip install -e ".[all-no-neuron]"` | Same as `all` minus NEURON — the default on Windows |
| `pip install -e ".[plot]"` | + PyVista, matplotlib, scikit-image — `pynetrees.plotting`, `hull_tree`'s 3D isosurface, and the tutorials |
| `pip install -e ".[matlab]"` | + mat73 — **v7.3 `.mtr`/`.mat` files**, which is what current MATLAB writes |
| `pip install -e ".[nmf]"` | + h5py — `.nmf` files |
| `pip install -e ".[stacks]"` | + tifffile, imageio, scikit-image — `pynetrees.stacks` |
| `pip install -e ".[notebook]"` | + ipykernel, ipywidgets, trame — run the notebooks in `examples/` |
| `pip install -e ".[blender]"` | + bpy — `pynetrees.blender`. A ~300 MB wheel that **pins `numpy < 2`**; see the note below |
| `pip install -e ".[neuron]"` | + NEURON (Linux/macOS only — see step 4) |
| `pip install -e ".[dev]"` | + pytest |
| `pip install -e ".[plot,dev]"` | A lean working set for hacking on the package |

To reproduce the exact versions the test suite is run against instead, use the
pinned set — then install the package itself:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

> **`bpy` holds numpy back.** Installing the `blender` extra pins `numpy < 2`,
> so the pinned set carries numpy 1.26 rather than 2.x. The suite passes on
> both — the package itself only requires `numpy>=1.24` — but if you want
> numpy 2 and Blender export in the same environment, you cannot have them
> yet. Keep `bpy` in a separate environment if that matters to you.

### 3. Verify

```bash
python -c "import pynetrees as pt; print(pt.sample_tree())"
# Tree(name='sample', n_nodes=197, regions=[dendrite, subtree])
```

### 4. NEURON (optional)

Only needed for [`neuron_bridge.py`](src/pynetrees/neuron_bridge.py). `import pynetrees`
never requires it, and the NEURON tests skip automatically when it's absent.

- **Linux / macOS:** `pip install neuron` (or `pip install -e ".[neuron]"`)
- **Windows:** there is no pip wheel. Use the official binary installer from
  [neuronsimulator.github.io](https://neuronsimulator.github.io/) — this port was
  verified against NEURON 9.0.0 installed that way.

Check it took:

```bash
python -c "import neuron; print(neuron.__version__)"
```

## Running the tests

```bash
pytest
```

Two test modules read fixtures from the MATLAB source tree this port was developed
alongside (`treestoolbox-master/sample/`, `Active GC Model/morphos/`). Those files
aren't redistributed here, so those tests **skip** on a fresh clone rather than fail,
as do the NEURON and PyVista tests when those packages are missing.

## Quickstart

Every snippet below is a real, executed result on the bundled sample reconstruction.

**Load and inspect**

```python
import pynetrees as pt

tree = pt.sample_tree()               # bundled 197-node reconstruction
tree = pt.load_swc("cell.swc")        # or your own: SWC,
tree = pt.load_neurolucida("cell.ASC")  # NeuroLucida,
trees = pt.load_mtr("population.mtr")   # or a MATLAB .mtr archive

print(f"{tree.n_nodes} nodes, "
      f"{pt.len_tree(tree).sum():.0f} um of cable, "
      f"{pt.B_tree(tree).sum()} branch points")
# 197 nodes, 765 um of cable, 25 branch points
```

**Measure**

```python
bo = pt.BO_tree(tree)                       # branch order per node
pl = pt.PL_tree(tree)                       # topological path length
pd = pt.Pvec_tree(tree, pt.len_tree(tree))  # metric path length [um]

print(bo.max(), pd.max().round(1))
# 9.0 142.7
```

**Population statistics** — returned as tidy pandas DataFrames, not MATLAB's nested
struct-of-cell-arrays, so `groupby`/seaborn work directly:

```python
stats = pt.stats_tree([group_a, group_b], group_names=["control", "lesion"])
stats["summary"]    # one row per tree
stats["points"]     # one row per branch/termination point
stats["branches"]   # one row per branch
```

**Passive cable analysis, no simulator required**

```python
tree.Ri, tree.Gm, tree.Cm = 100.0, 1 / 20000, 1.0   # [Ohm cm], [S/cm^2], [uF/cm^2]

v = pt.sse_tree(tree, I=0)      # steady state from 1 nA injected at node 0
print(f"input resistance at root: {v[0]:.1f} MOhm")
# input resistance at root: 526.0 MOhm
```

**Plot**

```python
pt.plot_tree(tree, scalars=bo)       # PyVista, 3D tube mesh
pt.plot_mpl_tree(tree, scalars=bo)   # matplotlib, fast line preview
pt.dendrogram_tree(tree)             # 2D topological dendrogram
```

Nothing plots as a side effect — `plot_tree` returns a `pyvista.Plotter` you call
`.show()` on, so you can overlay several trees, mark nodes and add hulls to one
scene first. [`examples/plot.ipynb`](examples/plot.ipynb) covers the interactive
workflow.

**Simulate in NEURON**

```python
model = pt.build_neuron_model(tree)
t, rec = pt.run_current_clamp(model, at_node=0, amp=0.1,
                              delay=5, dur=50, tstop=80)
print(f"{len(model.sections)} sections; peak V = {rec[0].max():.2f} mV")
# 51 sections; peak V = -21.74 mV
```

The tree becomes real `h.Section`s built from its 3D points via `pt3dadd`, with
segment counts from NEURON's own d_lambda rule. No `.hoc` text generation, no
subprocess, no file exchange — see [Simulation](#simulation) below.

## What's in the package

| Module | Contents |
|---|---|
| [`core.py`](src/pynetrees/core.py) | The `Tree` data structure, validation (`ver_tree`) |
| [`io/`](src/pynetrees/io/) | SWC, NeuroLucida `.ASC`, MATLAB `.mtr`/`.mat`, `.neu`, `.nmf`, NeuroML, `.hoc`, native `.npz` |
| [`graphtheory.py`](src/pynetrees/graphtheory.py) | Topology: parents, children, branch/termination points, branch order, path length, sorting, sub-trees, branch-length decomposition (`BLO_tree`) |
| [`metrics.py`](src/pynetrees/metrics.py) | Geometry: lengths, surfaces, volumes, angles, transformations, scaling to a target size |
| [`edit.py`](src/pynetrees/edit.py) | Repair, resample, delete, insert, concatenate, re-root |
| [`construct.py`](src/pynetrees/construct.py) | Synthetic trees: `MST_tree`, `BCT_tree`, `growth_tree`, `random_tree`, smoothing, soma and diameter models |
| [`generate.py`](src/pynetrees/generate.py) | Population-statistics-driven generative pipeline: cloning, DSCAM-style self-avoidance, spines |
| [`density.py`](src/pynetrees/density.py) | Voxel-grid density, alpha-shape boundaries, spanned area, space-filling radius |
| [`persistence.py`](src/pynetrees/persistence.py) | Topological description of a morphology: barcodes, persistence images (persistent homology) |
| [`electrotonics.py`](src/pynetrees/electrotonics.py) | Passive cable analysis, steady-state solves, LIF/AdEx-LIF |
| [`stats.py`](src/pynetrees/stats.py) | Sholl analysis, von Mises fits, spatial-randomness tests, population statistics |
| [`plotting.py`](src/pynetrees/plotting.py) | PyVista 3D rendering, matplotlib previews, dendrograms |
| [`stacks.py`](src/pynetrees/stacks.py) | Image-stack loading, skeletonisation, diameter fitting |
| [`neuron_bridge.py`](src/pynetrees/neuron_bridge.py) | Tree → live NEURON compartmental model |
| [`blender.py`](src/pynetrees/blender.py) | Native Blender export and headless rendering (opt-in, `[blender]` extra) |

All 173 public names are listed:

- one line each, grouped by purpose, in [docs/api-overview.md](docs/api-overview.md);
- in full — every signature and complete docstring — in
  [docs/FUNCTION_REFERENCE.md](docs/FUNCTION_REFERENCE.md).

## Documentation

| Page | What's in it |
|---|---|
| [docs/concepts.md](docs/concepts.md) | **Start here.** The data model, indexing conventions, the ideas everything builds on |
| [docs/guide.md](docs/guide.md) | Task-oriented walkthrough: loading, measuring, editing, plotting, simulating |
| [docs/matlab-migration.md](docs/matlab-migration.md) | For MATLAB TREES users: what's renamed, what changed, what isn't ported |
| [docs/api-overview.md](docs/api-overview.md) | Every public function, grouped by purpose — one line each |
| [docs/FUNCTION_REFERENCE.md](docs/FUNCTION_REFERENCE.md) | Every public name in full: signature and complete docstring, auto-generated from the live package |
| [docs/port-audit.md](docs/port-audit.md) | Faithfulness, performance and bug audit against the original |

Runnable tutorial notebooks in [`examples/`](examples/), each executed end-to-end so
the outputs you see are real:

| Notebook | Topic |
|---|---|
| [01_basics](examples/01_basics.ipynb) | Load a tree, inspect it, measure it, plot it |
| [02_editing_and_construction](examples/02_editing_and_construction.ipynb) | Repair, resample, prune, generate synthetic trees |
| [03_electrotonics](examples/03_electrotonics.ipynb) | Passive cable analysis without a simulator |
| [04_neuron_simulation](examples/04_neuron_simulation.ipynb) | Build a real NEURON model from a tree and run it |
| [05_populations_and_stats](examples/05_populations_and_stats.ipynb) | Compare groups of cells with `stats_tree` |
| [06_topology_and_growth](examples/06_topology_and_growth.ipynb) | Barcodes and persistence images; growing a tree into a volume with `growth_tree` |
| [07_generative_pipeline](examples/07_generative_pipeline.ipynb) | `clone_tree`, `dscam_tree` and `spines_tree`: synthesising trees from group statistics |
| [plot](examples/plot.ipynb) | Interactive 3D plotting: live widgets, overlays, annotation, hulls |

## How this differs from the MATLAB toolbox

Deliberate divergences, each recorded with its reasoning in the design log in
[PORT_STATUS.md](PORT_STATUS.md):

**Interface modernisation**

- Option strings (`'-s'`, `'-LO'`) → typed keyword arguments
- No global `trees` array — a `Tree` is an ordinary object you pass around
- No side-effect plotting; plotting functions return their axes/mesh
- 0-based indexing throughout, with `-1` as the no-parent sentinel
- `stats_tree` returns DataFrames instead of nested structs

**Bug fixes** — the port declines to reproduce 10 confirmed logic bugs found in the
MATLAB source while porting (`rootangle_tree` measuring from the coordinate origin
rather than the root, `LIF_tree`'s `Vzone` having no effect, `boundary_tree` crashing
on its own documented default call path, and others), catalogued in
[MATLAB_TOOLBOX_BUGS.md](MATLAB_TOOLBOX_BUGS.md).

<a name="simulation"></a>
**Simulation** — MATLAB reaches NEURON through T2N, whose core (`t2n.m`, 2447 lines)
is mostly `.hoc` text generation plus file/SSH/cluster plumbing, necessary only
because MATLAB has no NEURON binding. Python has one, so `neuron_bridge.py` builds
sections in-process and none of that machinery exists here. The geometric core of
`neuron_template_tree.m` is ported faithfully; T2N's protocol library (IV/FI curves,
resonance, bAP) is not — those are thin wrappers over this layer.

## Project status

Work in progress, and honest about it. [PORT_STATUS.md](PORT_STATUS.md) tracks every
MATLAB function with a status of `done` / `deferred` / `wont-port` and a reason, plus
a dated design-decisions log.

**Ported:** core data structure, I/O (SWC, NeuroLucida, MATLAB `.mtr`/`.mat`, `.neu`,
`.nmf`, NeuroML, `.hoc`), graph theory, metrics, editing, construction (including
`growth_tree`'s space-filling growth), the generative pipeline (`clone_tree`,
`gscale_tree`, `dscam_tree`, `spines_tree`), density/hull/space-filling machinery,
a topological (persistent-homology) description of morphology, electrotonics,
statistics, image-stack loading and diameter fitting, native Blender export, and a
foundational NEURON integration. Every exported function accepts a population
(a list of trees) as well as a single one, and handles a tree with no nodes as a
value rather than an error. See [NOT_YET_PORTED.md](NOT_YET_PORTED.md) for the
complete, itemised inventory.

**Not ported:**

- `cgui_tree` and the GUIDE-based GUI — no direct equivalent planned; a Python
  equivalent would be a rewrite against a different toolkit, not a port
- `pov_tree` / `x3d_tree` (POV-Ray / X3D export) — planned (V6); `pynetrees.blender`
  covers the same rendering need today
- T2N's protocol library (IV/FI curves, resonance, bAP) and its cluster/SSH
  execution mode — thin wrappers over `run_current_clamp`; add on demand
- `fix_tree` / `fix_tree_UI` / `finetune_fix_tree` — a MATLAB figure-callback GUI;
  MATLAB's own todo list flags these as incomplete
- `GC_biophys` (the Active GC Model's active-conductance fitting) — substantial,
  scoped separately
- v7.3 (HDF5) writing for `save_tree`/`save_mtr` — current write path is MATLAB v5,
  which caps a single variable at ~2 GB

## License

GPL-3.0-or-later, inherited from the MATLAB TREES toolbox this is a port of.
See [LICENSE](LICENSE).

## Citation

If you use this in published work, please cite the original TREES toolbox paper as
its authors ask:

> Cuntz H, Forstner F, Borst A, Häusser M (2010). One rule to grow them all: A
> general theory of neuronal branching and its practical application.
> *PLoS Computational Biology* 6(8): e1000877.

## Acknowledgements

The TREES toolbox is developed by Hermann Cuntz, Felix Effenberger and Marcel
Beining (Ernst Strüngmann Institute, Frankfurt), supervised by Alexander Borst (MPI
of Neurobiology) and Michael Häusser (UCL). Upstream source:
[cuntzlab/treestoolbox](https://github.com/cuntzlab/treestoolbox).

The bundled sample reconstruction in [`src/pynetrees/data/`](src/pynetrees/data/) comes
from the MATLAB toolbox's own sample set.
