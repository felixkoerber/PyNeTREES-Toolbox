# The GUI and the image stacks: what to do with them

Two parts of `treestoolbox-master` were left out of the port on the grounds
that they are "image processing" and "a GUI". Both deserve a real answer
rather than a category label, because between them they are **10,027 lines**
— more than a third of everything not yet ported — and because one of them
is the toolbox's entire reconstruction front-end.

This document is the analysis. Its conclusions, up front:

- **`stacks/` should be ported, and is small.** Of its 1,005 lines, roughly
  700 are file loading and a hand-rolled 3D thinning algorithm, both of
  which have maintained Python equivalents. What is left — the `.stk`
  container and `fitD_stack`'s diameter fitting — is genuinely
  tree-specific, is about 250 lines of Python, and is worth having.
- **`cgui_tree` should not be ported, and its capability should not be
  rebuilt from scratch either.** It is a 7,203-line single function
  dispatching 332 string cases over one global struct. But **napari already
  is the Python `cgui_tree`** — the same stack/threshold/skeleton/points
  pipeline, as layers — so the honest deliverable is a napari plugin, and it
  is a separate project from this port, not a work package inside it.

---

## Part 1 — `stacks/`

### What is actually in it

| File | LoC | What it does | Python equivalent |
|---|---|---|---|
| `load_stack` | 85 | Read a `.stk` file | `scipy.io.loadmat` — a `.stk` is a `.mat` |
| `save_stack` | 52 | Write one | `scipy.io.savemat` |
| `imload_stack` | 94 | Read one image into a 3D matrix | `imageio.imread` |
| `loadtifs_stack` | 126 | Read a multi-page TIFF | `tifffile.imread` — one call |
| `loaddir_stack` | 121 | Read every image in a folder | `tifffile` + `pathlib.glob` |
| `show_stack` | 87 | Maximum-intensity projections on three planes | matplotlib / PyVista |
| `skel_stack` | 233 | 3D thinning to a skeleton | `skimage.morphology.skeletonize` |
| `fitD_stack` | 207 | **Fit tree diameters from the fluorescence** | *nothing* |

Seven of the eight are I/O or generic image processing. The eighth is the
only one that knows what a tree is.

### The data structure is the part worth keeping

A "stack" in this toolbox is not one image volume, it is a **tiled set** of
them — a struct with

```
stack.M      cell array of 3D matrices    the tiles
stack.sM     names of the tiles
stack.coord  (n, 3) origin of each tile in um
stack.voxel  (3,)   voxel size in um
```

That layout exists because two-photon stacks of a whole neuron are acquired
as overlapping fields of view, and every downstream function has to map a
micron coordinate onto *whichever tile contains it*. `fitD_stack` spends its
first thirty lines doing exactly that. Nothing in `tifffile` or `skimage`
knows about it, so a small `Stack` dataclass carrying those four fields —
with the tile lookup as a method — is the piece of `stacks/` with no Python
equivalent, and it is what the rest should be built on.

Reading and writing `.stk` matters for the same reason `.mtr` did: it is how
a MATLAB user hands over their data.

### `skel_stack` should be delegated, not ported

`skel_stack` is a hand-rolled 3D thinning, described in its own header as
*"Inspired by algorithms described by Palagyi and Kuba ... hopefully
correctly interpreted from their papers"*. It iteratively erodes a binary
volume, checking a 26-neighbourhood template per surviving voxel, in chunks
of 20 million voxels because MATLAB cannot hold the neighbour index array
otherwise.

`skimage.morphology.skeletonize` implements Lee, Kashyap & Chu (1994) for 3D
input — the same class of algorithm, maintained, tested, and orders of
magnitude faster. Reimplementing the toolbox's own interpretation of a paper
would be reimplementing it worse. **Delegate, and say in the docstring that
the skeleton will not be voxel-identical to MATLAB's** — because it will
not, and a silent difference in a reconstruction front-end is exactly the
kind of thing that must be stated.

### `fitD_stack` is the one to port properly — and it has a self-flagged bug

Given an already-traced tree and the stack it was traced from, `fitD_stack`
recovers a diameter per segment: it samples the fluorescence along a line
perpendicular to each segment, convolves with a derivative-of-Gaussian to
sharpen the edges, and reads the width between the turning points. That is a
real measurement and there is no library that does it.

Its source carries this, at `stacks/fitD_stack.m:124`:

```matlab
% TODO, CRITICAL: RIGHT NOW ONLY THE TERMINAL POINT IS TAKEN
mPX          = [(P1 (1) + cV (1)) (P1 (1) + cV (1)) (P2 (1))];
mPY          = [(P1 (2) + cV (2)) (P1 (2) + cV (2)) (P2 (2))];
```

The three sampling positions are meant to run along the segment; two of the
three are the same point, and `P1 + cV` **is** `P2`, so all three collapse
onto the segment's far end. Every diameter is therefore measured at one
point rather than along the cable — not what the surrounding code was
written to do. The author knew: hence the comment.

A port should sample along the segment as intended, and expose the sample
count. That is a behaviour change from MATLAB and belongs in
`MATLAB_TOOLBOX_BUGS.md` and a Design Decision, not in silence.

It is worth saying that averaging is not automatically better, because
measuring it showed otherwise: near a branch point the perpendicular
profile picks up the *sibling* branch, so on a clean synthetic phantom the
single-point measurement came out **less** variable (spread 1.2 versus 1.8
voxels). The fix is to make the choice available and default to the
documented intent, not to assert that MATLAB's number is wrong.

### Recommendation

A single `pynetrees/stacks.py`, roughly 250 lines:

- `Stack` dataclass — the four fields above, plus `tile_at(point)` and
  `to_microns`/`to_voxels`.
- `load_stack` / `save_stack` — `.stk` compatibility, both directions.
- `load_tiff` / `load_folder` — thin wrappers around `tifffile`, with the
  tile geometry attached.
- `show_stack` — three maximum-intensity projections.
- `skeletonize_stack` — threshold plus `skimage`, returning carrier points
  in **microns**, ready to hand to `MST_tree`.
- `fitD_stack` — ported properly, TODO fixed.

New optional dependency: `tifffile` (and `scikit-image`, already present
under `[plot]`), under a `[stacks]` extra.

That closes `stacks/` at about a quarter of MATLAB's line count, with the
generic half delegated to libraries that are someone else's job to maintain.

---

## Part 2 — `cgui_tree`

### What it is

| File | LoC |
|---|---|
| `cgui_tree.m` | 7,203 |
| `cgui_tree_initialize.m` | 1,688 |
| `cgui_tree_keys.m` | 76 |
| `cgui_mousewheel_tree.m` | 41 |

`cgui_tree.m` is **one function**. It takes a string, `action`, and
dispatches through **332 `case` labels** in a chain of `switch` blocks,
mutating a single `global cgui` struct that holds every graphics handle and
every piece of application state. Its own header advises:

> to better read the code we recommend to fold the code on "switch" and
> "if" clauses. Really! It's worth it!

There are twelve panels, and the case counts show where the weight sits:

| Prefix | Cases | Panel |
|---|---|---|
| `vis_` | 45 | figure and overall graphics |
| `ged_` | 43 | orientation and positioning within a group |
| `mtr_` | 38 | tree construction and **manual editing** |
| `stk_` | 28 | image stacks |
| `plx_` | 23 | outside plots |
| `plt_` | 23 | graphical elements and their handles |
| `slt_` | 22 | selection and statistics |
| `cat_` | 20 | tree sorter |
| `ui_` | 18 | user-interface plumbing |
| `skl_` | 11 | skeletonisation / carrier points |
| `thr_` | 8 | thresholded stacks |
| `ele_` | 4 | electrotonic properties |

### What of it is actually missing from this port

Line count is misleading here. Sorted by what the *user* gets:

| Capability | Panels | Already covered by `pynetrees`? |
|---|---|---|
| Figure, colours, views, element handles | `vis_`, `plt_`, `plx_`, `cat_` | **Yes** — matplotlib and PyVista, and this is 111 of the 332 cases |
| Selection and statistics readout | `slt_`, `ele_` | **Yes** — `stats_tree` returns DataFrames; `electrotonics` is complete |
| Positioning trees in a group | `ged_` | **Yes** — `tran_tree`, `rot_tree`, `flip_tree`, `spread_tree` |
| Stack loading, thresholding, skeletonising, carrier points | `stk_`, `thr_`, `skl_` | **After Part 1** |
| Automatic reconstruction from carrier points | `mtr_` | **Yes** — `MST_tree` |
| **Manual tree editing by clicking** | `mtr_` | **No** |

So of a 9,022-line application, the only capability with no answer in the
library is *point-and-click editing of a morphology*: add a node here,
delete that branch, drag this point onto the fluorescence. Everything else
is a front-end onto functions that already exist.

That is worth stating plainly, because it reframes the question. This is not
"port 9,022 lines". It is "provide interactive editing", and the other
8,000-odd lines are a 2009-era MATLAB figure wrapped around functions this
port already has.

### Why porting it is the wrong shape of work

Three reasons, in order of weight:

1. **There is nothing to port.** A `switch` on 332 strings over a global
   handle struct is not an architecture that survives translation; it is a
   description of MATLAB's `uicontrol` callback model. The Python version of
   every one of those cases is "call the function directly". Rewriting it
   against Qt or Tk would be authoring a new application whose only
   relationship to the original is the feature list.
2. **The dependency does not belong in the library.** `pynetrees` is a
   computation package; the analysis half is complete and testable without a
   display. Making a GUI toolkit a dependency of `import pynetrees` would be a
   real cost paid by every headless and notebook user for a feature most of
   them will not open.
3. **It would not be finished.** A half-built reconstruction GUI is worse
   than none: people would trace real data in it.

### The recommendation: a napari plugin, as a separate project

[napari](https://napari.org) is the standard Python n-dimensional image
viewer. The mapping to `cgui_tree`'s pipeline is close enough to be
uncomfortable:

| `cgui_tree` | napari |
|---|---|
| `stk_` — load and display tiled stacks | `Image` layer, with `scale` and `translate` per tile |
| `thr_` — threshold, view the binary volume | `Labels` layer, live threshold widget |
| `skl_` — skeleton, carrier points | `Points` layer |
| `mtr_` — **click to add/delete/move nodes** | `Points` layer, **built in** |
| `mtr_` — connect points into a tree | `MST_tree`, one call, redraw as a `Shapes`/`Tracks` layer |
| `vis_`/`plt_` — 3D view, colours, handles | native |

The one capability the library lacks — interactive node editing — is the one
napari ships as a core layer type. The work is a plugin that wires
`pynetrees`' functions to napari layers, not an application.

**It should be a separate distribution** (`pynetrees-napari` or similar), for
the reason in point 2 above: napari plus Qt is a large dependency, and
`pynetrees` should not require it. That also lets it release on its own
schedule, and means an unfinished plugin cannot destabilise the port.

**It should not block anything.** The port's analysis, editing, construction,
electrotonics, I/O and generation are complete and tested without it.

### If interactive editing is wanted sooner

An intermediate option worth knowing about: a Jupyter workflow. The stack
pipeline is functions; `plot_tree` already renders to PyVista, which has a
working notebook backend and point-picking callbacks. That covers "inspect
and correct a reconstruction in a notebook" without a plugin or a new
distribution — less capable than napari, but available immediately and with
no dependency `pynetrees` does not already have under `[plot]`.

---

## Summary

| | Verdict | Effort | Where |
|---|---|---|---|
| `stacks/` | **Port** — delegate the generic half | ~250 lines | `pynetrees/stacks.py`, `[stacks]` extra |
| `skel_stack` | Delegate to `skimage`, document the difference | — | above |
| `fitD_stack` | Port properly, fix its own CRITICAL TODO | — | above |
| `cgui_tree` | **Do not port** | — | — |
| Interactive editing | napari plugin | separate project | `pynetrees-napari` |
| Everything else in the GUI | Already in the library | — | — |
