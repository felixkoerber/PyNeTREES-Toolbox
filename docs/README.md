# pytrees documentation

Python port of the [TREES toolbox](https://www.treestoolbox.org/) — load, edit,
analyse, simulate and visualise neuronal morphologies.

| Page | What's in it |
|---|---|
| [concepts.md](concepts.md) | **Start here.** The data model, indexing conventions, and the handful of ideas everything else builds on. |
| [guide.md](guide.md) | Task-oriented walkthrough: loading, measuring, editing, plotting, simulating. |
| [matlab-migration.md](matlab-migration.md) | For people who know the MATLAB toolbox: what changed, what's named differently, what deliberately isn't ported. |
| [port-audit.md](port-audit.md) | Faithfulness/performance/bug audit against the MATLAB original. |
| [api-overview.md](api-overview.md) | Every public function grouped by what it's for. |

Runnable tutorials live in [`../examples/`](../examples/):

| Notebook | Topic |
|---|---|
| `01_basics.ipynb` | Load a tree, inspect it, measure it, plot it |
| `02_editing_and_construction.ipynb` | Repair, resample, prune, and generate synthetic trees |
| `03_electrotonics.ipynb` | Passive cable analysis without a simulator |
| `04_neuron_simulation.ipynb` | Build a real NEURON model from a tree and run it |
| `05_populations_and_stats.ipynb` | Compare groups of cells with `stats_tree` |
| `plot.ipynb` | Interactive 3D plotting: live widgets, overlays, annotation, hulls |

## Project status

`pytrees` is a work in progress. See [`../PORT_STATUS.md`](../PORT_STATUS.md)
for a function-by-function record of what's ported, what's deliberately not,
and why — plus a dated design-decisions log. Bugs found in the original MATLAB
code while porting are catalogued in
[`../MATLAB_TOOLBOX_BUGS.md`](../MATLAB_TOOLBOX_BUGS.md).

## Install

```bash
cd python_port
conda create -n pytrees python=3.11 -y
conda activate pytrees
pip install -e ".[plot,dev]"
```

NEURON (for `pytrees.neuron_bridge`) is a separate install — see
[guide.md](guide.md#simulating-with-neuron).
