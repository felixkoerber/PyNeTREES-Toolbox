# API overview

Every public name in `pytrees`, grouped by what it's for. All are importable
directly: `from pytrees import len_tree`.

Conventions: functions take the tree first, return per-node NumPy arrays of
length `n_nodes` unless noted, and never plot as a side effect. See
[concepts.md](concepts.md).

## Data model

| Name | Purpose |
|---|---|
| `Tree` | The morphology container (`dA`, `X`, `Y`, `Z`, `D`, `R`, `rnames`, `name`, `frustum`, `Ri`, `Gm`, `Cm`) |
| `Tree.region_nodes(*names)` | Indices of nodes in the named region(s) |
| `Tree.region_mask(*names)` | Same, as a boolean mask |
| `Tree.region_index(name)` | The `R` value for a region name |
| `Tree.validate(quiet=False)` | Structural check; returns a list of problems |
| `Tree.reindexed(order)` | New tree with nodes reordered/subset |
| `Tree.with_coords(...)` | Copy with some coordinate arrays replaced |
| `ver_tree(tree)` | Free-function form of `validate` |
| `NO_PARENT` | The `-1` sentinel |

## I/O

| Name | Purpose |
|---|---|
| `load_swc` / `save_swc` | Standard SWC |
| `load_mtr` | MATLAB `.mtr` archives (v5) |
| `load_neurolucida` | NeuroLucida `.ASC` |
| `load_tree` / `save_tree` | Native `.npz` — lossless round-trip |
| `sample_tree()` | Bundled 2252-node reconstruction, for examples/tests |

## Topology (needs only `dA`)

| Name | Returns |
|---|---|
| `idpar_tree` | Parent index of each node |
| `idchild_tree` | Child indices of each node |
| `ipar_tree` | Full ancestor path per node (dense, `-1`-padded) |
| `child_tree` | Number of descendants per node |
| `B_tree` / `C_tree` / `T_tree` | Masks: branch / continuation / termination point |
| `typeN_tree` | Node type as an integer code |
| `PL_tree` | Topological path length to root |
| `BO_tree` | Branch order |
| `LO_tree` | Level order |
| `strahler_tree` | Strahler number |
| `asym_tree` | Asymmetry at each branch point |
| `sub_tree` | Mask of a node's whole subtree |
| `dissect_tree` | `(start, end)` node pairs, one per section |
| `Pvec_tree(tree, v)` | Cumulative sum of `v` along the path to the root |
| `ratio_tree` | Ratio of a value to its parent's |
| `rindex_tree` | Rank of each node within its region |
| `sort_tree` | Canonical node ordering (`by="hier"/"lo"/"lex"`) |
| `redirect_tree` | Re-root the tree |

## Geometry and metrics

| Name | Returns |
|---|---|
| `len_tree` | Segment length [µm] |
| `cyl_tree` | Segment start/end coordinates |
| `surf_tree` / `vol_tree` / `cvol_tree` | Segment surface / volume / continuous volume |
| `eucl_tree` | Straight-line distance to root (or any point) |
| `dist_tree` | Distance to a set of nodes |
| `angleB_tree` | Branching angle at each branch point |
| `direction_tree` | Unit direction vector per segment |
| `rootangle_tree` | Angle between segment and the line to the root |
| `bin_tree` / `gene_tree` | Binned / genealogical measures |
| `tran_tree` / `rot_tree` / `scale_tree` / `flip_tree` | Rigid & scaling transforms |
| `flatten_tree` | Project to XY, conserving segment lengths |
| `morph_tree` | Rescale every segment to a target length |
| `zcorr_tree` | Correct Z jumps |

## Editing

| Name | Purpose |
|---|---|
| `repair_tree` | Enforce BCT conformity (the one to call after loading) |
| `elim0_tree` / `elimt_tree` | Remove zero-length segments / multifurcations |
| `delete_tree` | Delete nodes, splicing to the nearest surviving ancestor |
| `insert_tree` / `insertp_tree` | Add nodes |
| `interpd_tree` | Interpolate diameters |
| `resample_tree` | Redistribute nodes at a target spacing |
| `restrain_tree` | Constrain geometry |
| `root_tree` | Prepend a tiny root segment |
| `recon_tree` | Reconstruct a subtree |
| `cat_tree` | Concatenate two trees |
| `abel_tree` | Inter-branch-point spacing measure |

## Construction (synthetic trees)

| Name | Purpose |
|---|---|
| `MST_tree` | Grow a tree over a point cloud (`bf` balances wiring vs. path length). **Returns `(tree, connected)`** — the only tree-builder returning a tuple |
| `BCT_tree` / `isBCT_tree` | Build from / validate a BCT string |
| `allBCTs_tree` / `allBTs_tree` | Enumerate all topologies of a given size |
| `clean_tree` | Prune short spurious branches |
| `soma_tree` | Add a soma |
| `cap_tree` | Cap terminal ends |
| `jitter_tree` | Add spatially-correlated noise |
| `smooth_tree` / `smoothbranch` | Smooth along long paths |
| `quaddiameter_tree` / `quadfit_tree` | Assign realistic tapering diameters |

## Plotting

| Name | Purpose |
|---|---|
| `plot_tree` | 3D via PyVista — `mode="tube"/"line"`, `scalars=`, `plotter=` |
| `plot_tree_mpl` | Lighter matplotlib fallback |
| `vtext_tree` / `pointer_tree` | Label / mark nodes on a PyVista plot |
| `chull_tree` | Convex hull (+ optional overlay) |
| `dendrogram_tree` / `xdend_tree` | Abstract topology diagram / its layout |
| `dA_tree_mpl` | Adjacency-matrix sparsity plot |
| `spread_tree` / `spread_trees` | Lay several cells out on a grid |

## Electrotonics (no simulator needed)

Require `tree.Ri` / `tree.Gm` (and `tree.Cm` for time-stepping).

| Name | Returns |
|---|---|
| `M_tree` | Sparse conductance matrix of the equivalent circuit |
| `gi_tree` / `gm_tree` | Axial / membrane conductance per segment |
| `lambda_tree` / `elen_tree` | Length constant / electrotonic length |
| `cgin_tree` | Collapsed (point-neuron) input conductance |
| `sse_tree` | Steady-state voltage — full matrix, or one injection site |
| `syn_tree` | Steady state with synaptic conductances |
| `loop_tree` | Conductance matrix with electrical-synapse loops |
| `ssecat_tree` / `syncat_tree` | The above across several coupled trees |
| `LIF_tree` | Leaky integrate-and-fire over the full morphology |
| `AdExLIF_tree` | Adaptive exponential I&F |

## Statistics and comparison

| Name | Returns |
|---|---|
| `stats_tree` | dict of tidy DataFrames: `summary`, `points`, `branches`, `sholl` |
| `sholl_tree` | `ShollResult` — intersection counts vs. radius |
| `vonMises_tree` | Fitted centripetal bias `k` |
| `bf_tree` | `(balancing factor, k)` estimated from root angles |
| `peters_tree` | Candidate synapse sites between two trees |

## NEURON simulation

Requires the `neuron` package — see [guide.md](guide.md#simulating-with-neuron).

| Name | Purpose |
|---|---|
| `build_neuron_model` | Tree → live `h.Section` tree (`NeuronModel`) |
| `NeuronModel.loc(node)` | The NEURON segment for a tree node |
| `NeuronModel.region_sections` | Sections grouped by region name |
| `insert_mechanism` | Insert a mechanism, optionally per region |
| `run_current_clamp` | Inject a current step, record voltage |
