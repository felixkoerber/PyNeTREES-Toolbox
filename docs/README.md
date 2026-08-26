# PyNeTREES Toolbox documentation

Python port of the [TREES toolbox](https://www.treestoolbox.org/) — load, edit,
analyse, simulate and visualise neuronal morphologies.

| Page | What's in it |
|---|---|
| [concepts.md](concepts.md) | **Start here.** The data model, indexing conventions, and the handful of ideas everything else builds on. |
| [guide.md](guide.md) | Task-oriented walkthrough: loading, measuring, editing, plotting, simulating. |
| [matlab-migration.md](matlab-migration.md) | For people who know the MATLAB toolbox: what changed, what's named differently, what deliberately isn't ported. |
| [port-audit.md](port-audit.md) | Faithfulness/performance/bug audit against the MATLAB original. |
| [../REVIEW_PLAN.md](../REVIEW_PLAN.md) | The function-by-function review and the plan responding to it. |
| [../NOT_YET_PORTED.md](../NOT_YET_PORTED.md) | Complete inventory of what is not ported, and why. |
| [api-overview.md](api-overview.md) | Every public function grouped by what it's for. |
| [FUNCTION_REFERENCE.md](FUNCTION_REFERENCE.md) | Every public name in full — signature and complete docstring, auto-generated. |

Runnable tutorials live in [`../examples/`](../examples/):

| Notebook | Topic |
|---|---|
| `01_basics.ipynb` | Load a tree, inspect it, measure it, plot it |
| `02_editing_and_construction.ipynb` | Repair, resample, prune, and generate synthetic trees |
| `03_electrotonics.ipynb` | Passive cable analysis without a simulator |
| `04_neuron_simulation.ipynb` | Build a real NEURON model from a tree and run it |
| `05_populations_and_stats.ipynb` | Compare groups of cells with `stats_tree` |
| `06_topology_and_growth.ipynb` | Barcodes and persistence images; growing a tree into a volume with `growth_tree` |
| `plot.ipynb` | Interactive 3D plotting: live widgets, overlays, annotation, hulls |

## Project status

`pynetrees` is a work in progress. See [`../PORT_STATUS.md`](../PORT_STATUS.md)
for a function-by-function record of what's ported, what's deliberately not,
and why — plus a dated design-decisions log. Bugs found in the original MATLAB
code while porting are catalogued in
[`../MATLAB_TOOLBOX_BUGS.md`](../MATLAB_TOOLBOX_BUGS.md).

## Recent API changes

If you have code written against an earlier version, several conventions
changed (full reasoning in `PORT_STATUS.md`, decisions #40-#42 and #67):

| Was | Now |
|---|---|
| `tree, order = sort_tree(t)` | `tree = sort_tree(t)`, or `full_output=True` for both |
| `idpar_tree(t, no_self=True)` | `idpar_tree(t, root_self=False)` |
| `len_tree(t, dim2=True)` | `len_tree(t, dim=2)` |
| `bf_tree(t, params=True)` | `bf_tree(t, fit_constants=True)` |
| `vonMises_tree(t, dim="3d")` | `vonMises_tree(t, dim=3)` |
| `plot_tree_mpl` | `plot_mpl_tree` |
| `spread_trees(trees)` | `spread_tree(trees).trees` |
| `mask = sub_tree(t, n)` | `mask, subtree = sub_tree(t, n)` |
| `sample_tree()` → 2252 nodes | `sample_tree()` → 197 nodes; the old tree is `hss_tree()` |

**None of these keep working with a warning.** An earlier version of the port
carried the old spellings for one release behind a `DeprecationWarning`; #67
(2026-08-26) removed that entirely, reasoning that a port with no users yet
gains nothing from a compatibility period and only accumulates branches that
have to be kept working and tested. A renamed keyword now fails loudly and
immediately — `TypeError: unexpected keyword argument` — rather than warning
and silently doing nothing, which is the failure mode a caller upgrading
actually needs to see. The return-value changes were never soft either:
unpacking a single `Tree` raises `TypeError: cannot unpack non-iterable Tree
object` right at the call site.

## Install

```bash
cd python_port
conda create -n pynetrees python=3.11 -y
conda activate pynetrees
pip install -e ".[plot,matlab,dev]"
```

The `matlab` extra pulls in `mat73`, needed to read MATLAB **v7.3** `.mat`
/`.mtr` files — which is what MATLAB's own `save_tree` writes, always. Without
it, `.mtr` files from a current TREES install cannot be opened.

NEURON (for `pynetrees.neuron_bridge`) is a separate install — see
[guide.md](guide.md#simulating-with-neuron).
