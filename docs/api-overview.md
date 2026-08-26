# API overview

All 173 public names in `pynetrees`, grouped by what they're for, one line each.
Everything below `import`s directly: `from pynetrees import len_tree`.

Conventions: functions take the tree first, return per-node NumPy arrays of
length `n_nodes` unless noted, accept a **list of trees** as well as a single
one (returning a list back), and never plot as a side effect. See
[concepts.md](concepts.md).

For the full signature and complete docstring of every name here, see
[FUNCTION_REFERENCE.md](FUNCTION_REFERENCE.md) — generated from the live
package, so it cannot drift from what's actually importable.

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
| `load_tree` / `save_tree` | Dispatch by extension across every format below — the front door |
| `load_swc` / `save_swc` | Standard SWC |
| `load_mtr` / `save_mtr` | MATLAB `.mtr`/`.mat` archives (v5 write; v5 and v7.3/HDF5 read) |
| `load_neurolucida` | NeuroLucida `.ASC` |
| `load_neu` | NEURON transfer format `.neu` |
| `load_nmf` / `save_nmf` | The toolbox's HDF5 extended-SWC `.nmf` |
| `save_neuroml` | NeuroML `.xml` (export only) |
| `save_hoc` / `save_nrn` | NEURON `.hoc` cell file / one-section-per-segment `.nrn` (export only) |
| `t2n_interface` | T2N-style per-node section/segment index array, for hand-rolled `.hoc` work |
| `load_npz` / `save_npz` | pynetrees' own lossless native format — the only one that round-trips every `Tree` field |
| `sample_tree()` | MATLAB's sample: 197-node subtree of an HSN cell |
| `sample2_tree()` | 15-node minimal tree, for doctests |
| `hsn_tree()` / `hss_tree()` | Full HSN (1290) / HSS (2252) cells |
| `dLPTCs_trees()` | Population: 55 cells in 5 named groups |

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
| `BLO_tree` | Decompose into paths (branches), deepest first — `BranchLengthOrder(order, length, cumulative)`; foundation of `barcode_tree` |
| `asym_tree` | Asymmetry at each branch point |
| `sub_tree` | `(mask, tree)` for a node's whole subtree; `with_tree=False` for the mask alone |
| `dissect_tree` | `(start, end)` node pairs, one per section |
| `Pvec_tree(tree, v)` | Cumulative sum of `v` along the path to the root |
| `ratio_tree` | Ratio of a value to its parent's |
| `rindex_tree` | Rank of each node within its region |
| `sort_tree` | Canonical node ordering (`by="hier"/"lo"/"lex"`); `full_output=True` also returns the permutation |
| `redirect_tree` | Re-root the tree |

## Geometry and metrics

| Name | Returns |
|---|---|
| `len_tree` | Segment length [µm] |
| `L_tree` | Total cable length [µm] — `len_tree(tree).sum()`, under MATLAB's name |
| `cyl_tree` | Segment start/end coordinates |
| `surf_tree` / `vol_tree` / `cvol_tree` | Segment surface / volume / continuous volume |
| `eucl_tree` | Straight-line distance to root (or any point) |
| `dist_tree` | Distance to a set of nodes |
| `angleB_tree` | Branching angle at each branch point |
| `angleBd_tree` / `angleBd2_tree` | Branching angle measured a fixed distance from the branch point |
| `direction_tree` | Unit direction vector per segment |
| `rootangle_tree` | Angle between segment and the line to the root |
| `bin_tree` / `gene_tree` | Binned / genealogical measures |
| `tran_tree` / `rot_tree` / `scale_tree` / `flip_tree` | Rigid & scaling transforms |
| `scaleS_tree` / `scaleV_tree` | Scale to a target spanned area / enclosed volume — `Scaled(tree, factor, error)` |
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
| `MST_tree` | Grow a tree over a point cloud (`bf` balances wiring vs. path length); `full_output=True` returns `MSTResult(trees, connected, indx, history)` |
| `random_tree` | Toy tree: `MST_tree` over a scattered point cloud |
| `growth_tree` | Grow into a *volume* rather than a fixed cloud, trading wiring cost against space-filling (`sp`) and stochasticity (`k`) — returns `Growth(tree, length, terminals, attached_to, target, targets, history)` |
| `BCT_tree` / `isBCT_tree` | Build from / validate a BCT string |
| `allBCTs_tree` / `allBTs_tree` | Enumerate all topologies of a given size |
| `clean_tree` | Prune short spurious branches |
| `soma_tree` | Add a soma |
| `cap_tree` | Cap terminal ends |
| `jitter_tree` | Add spatially-correlated noise |
| `smooth_tree` | Smooth along long paths (helper `_smoothbranch` is private: it takes raw X/Y/Z arrays, not a Tree) |
| `quaddiameter_tree` / `quadfit_tree` | Assign realistic tapering diameters |

## Generative pipeline

Population-statistics-driven synthesis, built on `construct.py`/`density.py`.

| Name | Purpose |
|---|---|
| `clone_tree` | Grow a new cell matching a real one's per-region statistics (spans, angles, density) |
| `gscale_tree` | Fit the per-region spanning statistics `clone_tree` draws from — `Spanning` of `RegionSpan`s |
| `dscam_tree` | DSCAM-style self-avoidant growth (siblings repel each other) |
| `spines_tree` | Scatter dendritic spines along a tree — `SpineResult` |
| `rpoints_tree` | Random points respecting a region's density profile |
| `PP_generator_tree` | Poisson-process point generator used by `rpoints_tree` |
| `in_hull` | Which points lie inside a (possibly multi-ring) 2D boundary |

## Density, hulls and space-filling

| Name | Purpose |
|---|---|
| `gdens_tree` | Bin nodes into a regular voxel grid — `DensityGrid` |
| `hull_tree` | Space-filling surface at a fixed distance from the arbor (not the convex hull) |
| `vhull_tree` | Voronoi-cell volume per node within a boundary |
| `boundary_tree` | Concave boundary (alpha shape) around a tree's points — `Boundary`; shrink `0`=convex hull, `1`=tightest wrap |
| `convexity_tree` | Fraction of node pairs with a clear line of sight (0=concave, 1=convex) |
| `share_boundary_tree` | Overlap between two trees' territories |
| `span_tree` | 2D spanned area (morphological closing of the rasterised arbor) — `Span` |
| `theta_tree` | Space-filling radius on `span_tree`'s grid |
| `theta_mc_tree` | Space-filling radius by Monte Carlo, in 3D — `ThetaMC`; only comparable across cells sampled alike (see docstring) |
| `lego_tree` | Plot `gdens_tree`'s voxel counts as a 3D bar chart |

## Topological description (persistent homology)

A description of branching structure independent of embedding — see
[persistence.py](../src/pynetrees/persistence.py)'s module docstring.

| Name | Purpose |
|---|---|
| `barcode_tree` | `(birth, death)` per branch, from `BLO_tree`'s decomposition |
| `persistenceimage_tree` | Barcode rendered as a fixed-size 2D density, for clustering across cells |
| `realisations_tree` | Exact count of trees indistinguishable from this one by their barcode |

## Plotting

| Name | Purpose |
|---|---|
| `plot_tree` | 3D via PyVista — `mode="tube"/"line"`, `scalars=`, `plotter=`; also takes a **list of trees**, drawn into one scene |
| `plot_mpl_tree` | Lighter matplotlib fallback |
| `vtext_tree` / `pointer_tree` | Label / mark nodes on a PyVista plot |
| `chull_tree` | Convex hull (+ optional overlay) |
| `dendrogram_tree` / `xdend_tree` | Abstract topology diagram / its layout |
| `dA_tree` | Adjacency-matrix sparsity plot |
| `plotsect_tree` | Draw the single path between two nodes |
| `xplore_tree` | 2D node/region explorer with hover labels |
| `spread_tree` | Lay several cells out on a non-overlapping grid — `SpreadResult(trees, offsets)` |

## Electrotonics (no simulator needed)

Require `tree.Ri` / `tree.Gm` (and `tree.Cm` for time-stepping).

| Name | Returns |
|---|---|
| `M_tree` | Sparse conductance matrix of the equivalent circuit |
| `gi_tree` / `gm_tree` | Axial / membrane conductance per segment |
| `lambda_tree` / `elen_tree` | Length constant / electrotonic length |
| `cgin_tree` | Collapsed (point-neuron) input conductance |
| `M_atten_tree` | Number of electrotonically distinct compartments |
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
| `dissectSholl_tree` | Sholl profile decomposed by direction — `ShollDissection` |
| `r_mc_tree` | Clark-Evans spatial-randomness test (Monte Carlo null) — `RMCResult` |
| `peters_tree` | Candidate synapse sites between two trees |

## Image stacks

Confocal/2-photon volumes: loading, skeletonising, diameter fitting. See
[GUI_AND_STACKS.md](../GUI_AND_STACKS.md) for the design reasoning.

| Name | Purpose |
|---|---|
| `load_stack` / `save_stack` | A `Stack` (voxel array + calibration) to/from `.stk` |
| `load_tiff` | A multi-page TIFF stack |
| `load_folder` | A folder of single-slice images as one stack |
| `show_stack` | Quick-look 3D rendering of a stack |
| `skeletonize_stack` | Thin a segmented stack to a 1-voxel-wide skeleton |
| `fitD_stack` | Fit each node's diameter from the stack intensity around it |

## NEURON simulation

Requires the `neuron` package — see [guide.md](guide.md#simulating-with-neuron).

| Name | Purpose |
|---|---|
| `build_neuron_model` | Tree → live `h.Section` tree (`NeuronModel`) |
| `NeuronModel.loc(node)` | The NEURON segment for a tree node |
| `NeuronModel.region_sections` | Sections grouped by region name |
| `insert_mechanism` | Insert a mechanism, optionally per region |
| `run_current_clamp` | Inject a current step, record voltage |

## Blender export (optional)

Not part of `pynetrees.__all__` and **not imported by `import pynetrees`** — opt in with
`from pynetrees import blender`. Needs the `blender` extra (`bpy`), a ~300 MB wheel
that pins `numpy < 2`; see [src/pynetrees/blender.py](../src/pynetrees/blender.py) and
its entries in [FUNCTION_REFERENCE.md](FUNCTION_REFERENCE.md).

| Name | Purpose |
|---|---|
| `reset_scene` | Clear a Blender scene to a known empty state |
| `build_tree` | Tree → a real Blender curve object, diameter-tapered |
| `save_blend` | Write a `.blend` file |
| `render_tree` | Headless render to PNG, orthographic camera |
