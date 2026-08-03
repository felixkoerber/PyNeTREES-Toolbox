# pytrees

A Python port of the [TREES toolbox](https://www.treestoolbox.org/) — load, edit,
measure, simulate and visualise neuronal branching structures.

`pytrees` reads neuronal reconstructions (SWC, NeuroLucida `.ASC`, MATLAB `.mtr`),
gives you the toolbox's graph-theoretic and morphometric analyses on top of a plain
NumPy/SciPy data structure, does passive cable analysis without a simulator, builds
synthetic trees, and — where you do want a simulator — turns a morphology into a live
NEURON model in-process.

```python
import pytrees as pt

tree = pt.sample_tree()
print(tree)
# Tree(name='sample', n_nodes=2252, regions=[1])
```

It is a **port, not a binding**: no MATLAB installation, license, or bridge is
involved. Function names and argument order follow the MATLAB original closely
enough that existing TREES knowledge transfers directly — see
[docs/matlab-migration.md](docs/matlab-migration.md).

---

## Requirements

- **Python 3.10+** (developed and tested on 3.13)
- **NumPy, SciPy, pandas** — installed automatically
- **PyVista + matplotlib** — optional, for plotting
- **NEURON 8+** — optional, only for `pytrees.neuron_bridge`

## Installation

### 1. Create an isolated environment

Nothing in this repo does this for you, and installing a scientific stack straight
into a global or `base` Python is easy to regret later.

```bash
conda create -n pytrees python=3.11 -y
conda activate pytrees
```

or with `venv`:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
```

### 2. Install the package

```bash
git clone https://github.com/<your-user>/pytrees.git
cd pytrees
pip install -e ".[plot,dev]"
```

The extras are separable if you want a smaller install:

| Command | What you get |
|---|---|
| `pip install -e .` | Core only — NumPy, SciPy, pandas |
| `pip install -e ".[plot]"` | + PyVista, matplotlib (needed for `pytrees.plotting` and the tutorials) |
| `pip install -e ".[dev]"` | + pytest |
| `pip install -e ".[plot,dev]"` | Everything above |

To reproduce the exact versions this was developed against instead, use the pinned
set — then install the package itself:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

### 3. Verify

```bash
python -c "import pytrees as pt; print(pt.sample_tree())"
# Tree(name='sample', n_nodes=2252, regions=[1])
```

### 4. NEURON (optional)

Only needed for [`neuron_bridge.py`](src/pytrees/neuron_bridge.py). `import pytrees`
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
import pytrees as pt

tree = pt.sample_tree()               # bundled 2252-node reconstruction
tree = pt.load_swc("cell.swc")        # or your own: SWC,
tree = pt.load_neurolucida("cell.ASC")  # NeuroLucida,
trees = pt.load_mtr("population.mtr")   # or a MATLAB .mtr archive

print(f"{tree.n_nodes} nodes, "
      f"{pt.len_tree(tree).sum():.0f} um of cable, "
      f"{pt.B_tree(tree).sum()} branch points")
# 2252 nodes, 8100 um of cable, 502 branch points
```

**Measure**

```python
bo = pt.BO_tree(tree)                       # branch order per node
pl = pt.PL_tree(tree)                       # topological path length
pd = pt.Pvec_tree(tree, pt.len_tree(tree))  # metric path length [um]

print(bo.max(), pd.max().round(1))
# 60.0 976.0
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
# input resistance at root: 27.2 MOhm
```

**Plot**

```python
pt.plot_tree(tree, scalars=bo)       # PyVista, 3D tube mesh
pt.plot_tree_mpl(tree, scalars=bo)   # matplotlib, fast line preview
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
# 1005 sections; peak V = -67.46 mV
```

The tree becomes real `h.Section`s built from its 3D points via `pt3dadd`, with
segment counts from NEURON's own d_lambda rule. No `.hoc` text generation, no
subprocess, no file exchange — see [Simulation](#simulation) below.

## What's in the package

| Module | Contents |
|---|---|
| [`core.py`](src/pytrees/core.py) | The `Tree` data structure, validation (`ver_tree`) |
| [`io/`](src/pytrees/io/) | SWC, NeuroLucida `.ASC`, MATLAB `.mtr`, native formats |
| [`graphtheory.py`](src/pytrees/graphtheory.py) | Topology: parents, children, branch/termination points, branch order, path length, sorting, sub-trees |
| [`metrics.py`](src/pytrees/metrics.py) | Geometry: lengths, surfaces, volumes, angles, transformations |
| [`edit.py`](src/pytrees/edit.py) | Repair, resample, delete, insert, concatenate, re-root |
| [`construct.py`](src/pytrees/construct.py) | Synthetic trees: `MST_tree`, `BCT_tree`, smoothing, soma and diameter models |
| [`electrotonics.py`](src/pytrees/electrotonics.py) | Passive cable analysis, steady-state solves, LIF/AdEx-LIF |
| [`plotting.py`](src/pytrees/plotting.py) | PyVista 3D rendering, matplotlib previews, dendrograms |
| [`stats.py`](src/pytrees/stats.py) | Sholl analysis, von Mises fits, population statistics |
| [`neuron_bridge.py`](src/pytrees/neuron_bridge.py) | Tree → live NEURON compartmental model |

All 108 public functions are listed by purpose in
[docs/api-overview.md](docs/api-overview.md).

## Documentation

| Page | What's in it |
|---|---|
| [docs/concepts.md](docs/concepts.md) | **Start here.** The data model, indexing conventions, the ideas everything builds on |
| [docs/guide.md](docs/guide.md) | Task-oriented walkthrough: loading, measuring, editing, plotting, simulating |
| [docs/matlab-migration.md](docs/matlab-migration.md) | For MATLAB TREES users: what's renamed, what changed, what isn't ported |
| [docs/api-overview.md](docs/api-overview.md) | Every public function, grouped by purpose |
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

**Ported:** core data structure, I/O, graph theory, metrics, editing, construction,
plotting, electrotonics, statistics, and a foundational NEURON integration.

**Not ported:**

- `cgui_tree` and the GUIDE-based GUI — no direct equivalent planned
- Image-stack reconstruction tools (`load_stack`, `skel_stack`, …) — unscheduled
- T2N's protocol library and its cluster/SSH execution mode
- `clone_tree` / `gscale_tree` — high complexity, tightly coupled to one dataset's
  region conventions

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

The bundled sample reconstruction in [`src/pytrees/data/`](src/pytrees/data/) comes
from the MATLAB toolbox's own sample set.
