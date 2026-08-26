"""Image stacks: the reconstruction front-end.

Ports `stacks/` -- but not line for line. Of MATLAB's eight functions, seven
are file loading or generic image processing, and Python already has better
versions of both; see `GUI_AND_STACKS.md` for the full reasoning. What is
kept here is the part that knows about neurons:

- the **tiled stack** container, which is why this module exists at all,
- `.stk` read/write, so a MATLAB user can hand over their data,
- carrier-point extraction, which feeds :func:`~pynetrees.MST_tree`,
- :func:`fitD_stack`, which measures a tree's diameters from the
  fluorescence it was traced from.

**A stack is a set of tiles, not one volume.** Two-photon stacks of a whole
neuron are acquired as overlapping fields of view, so a "stack" here is
several 3D arrays each with its own origin in microns, sharing one voxel
size. Every operation that takes a micron coordinate has to find the tile
containing it first, which is what :meth:`Stack.tile_at` is for.

Needs `tifffile` for TIFF input and `scikit-image` for skeletonisation, both
under the ``[stacks]`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._population import accepts_population
from .core import Tree
from .metrics import cyl_tree

__all__ = [
    "Stack",
    "load_stack",
    "save_stack",
    "load_tiff",
    "load_folder",
    "show_stack",
    "skeletonize_stack",
    "fitD_stack",
]


@dataclass
class Stack:
    """A tiled 3D image stack in micron coordinates.

    Attributes
    ----------
    tiles : list[np.ndarray]
        One ``(nx, ny, nz)`` array per field of view. MATLAB's ``stack.M``.
    origin : np.ndarray
        ``(n_tiles, 3)`` position of each tile's first voxel [um].
        MATLAB's ``stack.coord``.
    voxel : np.ndarray
        ``(3,)`` voxel size [um], shared by every tile. MATLAB's
        ``stack.voxel``.
    names : list[str]
        One per tile. MATLAB's ``stack.sM``.
    """

    tiles: list[np.ndarray]
    origin: np.ndarray
    voxel: np.ndarray
    names: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.tiles = [np.asarray(t) for t in self.tiles]
        self.origin = np.atleast_2d(np.asarray(self.origin, dtype=float))
        self.voxel = np.asarray(self.voxel, dtype=float).ravel()
        if not self.names:
            self.names = [f"tile{i + 1}" for i in range(len(self.tiles))]
        if len(self.origin) != len(self.tiles):
            raise ValueError(
                f"{len(self.tiles)} tiles but {len(self.origin)} origins"
            )
        if self.voxel.size != 3:
            raise ValueError(f"voxel must be (3,), got {self.voxel.shape}")

    def __len__(self) -> int:
        return len(self.tiles)

    def extent(self, index: int) -> np.ndarray:
        """``(2, 3)`` low and high corner of one tile [um]."""
        span = (np.array(self.tiles[index].shape) - 1) * self.voxel
        return np.array([self.origin[index], self.origin[index] + span])

    def tile_at(self, point) -> int:
        """Index of the tile containing ``point`` [um], or the nearest one.

        Tiles overlap and a traced node can fall in the gap between them, so
        this never fails: containment first, nearest tile centre otherwise.
        MATLAB picks by squared distance to the tile's coordinate axes,
        which comes to the same thing for a point inside a tile and to
        something arbitrary for one outside.
        """
        point = np.asarray(point, dtype=float)
        for index in range(len(self.tiles)):
            low, high = self.extent(index)
            if np.all(point >= low) and np.all(point <= high):
                return index
        centres = np.array([self.extent(i).mean(axis=0)
                            for i in range(len(self.tiles))])
        return int(np.argmin(np.linalg.norm(centres - point, axis=1)))

    def to_voxels(self, points, index: int) -> np.ndarray:
        """Micron coordinates to fractional voxel indices within a tile."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        return (points - self.origin[index]) / self.voxel

    def to_microns(self, voxels, index: int) -> np.ndarray:
        """Voxel indices within a tile back to micron coordinates."""
        voxels = np.atleast_2d(np.asarray(voxels, dtype=float))
        return voxels * self.voxel + self.origin[index]

    def projection(self, axis: int = 2) -> list[np.ndarray]:
        """Maximum-intensity projection of each tile along ``axis``."""
        return [tile.max(axis=axis) for tile in self.tiles]


# ---------------------------------------------------------------------------
# reading and writing
# ---------------------------------------------------------------------------


def load_stack(path: str | Path) -> Stack:
    """Load a MATLAB ``.stk`` file.

    A ``.stk`` is a ``.mat`` workspace holding one ``stack`` struct -- the
    same relationship ``.mtr`` has to ``.mat``.
    """
    from .io._matlab import read_matlab, to_plain

    path = Path(path)
    data = to_plain(read_matlab(path))
    if "stack" not in data:
        raise ValueError(
            f"{path}: no 'stack' variable; found {sorted(k for k in data)}"
        )
    struct = to_plain(data["stack"])

    tiles = [np.asarray(m) for m in _cells(struct["M"])]
    names = [str(s) for s in np.ravel(struct.get("sM", []))]
    return Stack(
        tiles=tiles,
        origin=np.atleast_2d(np.asarray(struct["coord"], dtype=float)),
        voxel=np.asarray(struct["voxel"], dtype=float).ravel(),
        names=names[: len(tiles)],
    )


def _cells(value) -> list:
    """MATLAB cell array to a Python list, whichever way it decoded.

    A one-element cell array comes back squeezed to a bare numeric array,
    so the object dtype -- not the dimensionality -- is what distinguishes
    "several tiles" from "one tile".
    """
    if isinstance(value, list):
        return value
    array = np.asarray(value)
    if array.dtype != object:
        return [array]
    return [np.asarray(tile) for tile in array.ravel()]


def save_stack(stack: Stack, path: str | Path) -> Path:
    """Write a :class:`Stack` to a MATLAB-readable ``.stk``.

    v5 rather than v7.3, for the reasons in
    :func:`~pynetrees.io.save_mtr` -- MATLAB's ``load`` reads either.
    """
    from scipy.io import savemat

    path = Path(path)
    if path.suffix != ".stk":
        path = path.with_suffix(path.suffix + ".stk")

    cells = np.empty((1, len(stack.tiles)), dtype=object)
    labels = np.empty((1, len(stack.tiles)), dtype=object)
    for index, tile in enumerate(stack.tiles):
        cells[0, index] = tile
        labels[0, index] = stack.names[index]

    savemat(str(path), {"stack": {
        "M": cells,
        "sM": labels,
        "coord": stack.origin,
        "voxel": stack.voxel.reshape(1, 3),
    }}, do_compression=True)
    return path


def load_tiff(path: str | Path, voxel=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)
              ) -> Stack:
    """Load a multi-page TIFF as a one-tile stack.

    MATLAB's `loadtifs_stack` reads pages one at a time through
    ``imread``; ``tifffile`` does the whole file in one call and handles
    the compression schemes microscopes actually emit.
    """
    tifffile = _tifffile()
    path = Path(path)
    volume = np.asarray(tifffile.imread(str(path)))
    return Stack(tiles=[_as_xyz(volume)], origin=np.atleast_2d(origin),
                 voxel=np.asarray(voxel), names=[path.stem])


def load_folder(path: str | Path, voxel=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0), pattern: str = "*") -> Stack:
    """Load every image in a folder as the z planes of one tile.

    Ports `loaddir_stack`. Files are taken in sorted order, which is what
    makes ``plane001.tif``-style naming work and ``plane1.tif`` not; MATLAB
    uses ``dir`` order, which is the same trap.
    """
    from imageio import v3 as iio

    path = Path(path)
    files = sorted(f for f in path.glob(pattern) if f.is_file())
    if not files:
        raise ValueError(f"{path}: no files matching {pattern!r}")

    planes = [np.asarray(iio.imread(f)) for f in files]
    shapes = {p.shape for p in planes}
    if len(shapes) > 1:
        raise ValueError(
            f"{path}: images differ in size ({sorted(shapes)}); a stack "
            "needs one shape throughout"
        )
    # stacked page-first, i.e. (z, y, x), so one orientation rule covers
    # both this and a multi-page TIFF
    volume = np.stack(planes, axis=0)
    return Stack(tiles=[_as_xyz(volume)], origin=np.atleast_2d(origin),
                 voxel=np.asarray(voxel), names=[path.name])


def _as_xyz(volume: np.ndarray) -> np.ndarray:
    """TIFF pages come in as ``(z, y, x)``; this port is ``(x, y, z)``.

    Same transposition `gdens_tree` makes, and for the same reason: mixing
    image order with coordinate order silently is how axis bugs happen.
    """
    volume = np.asarray(volume)
    if volume.ndim == 2:
        return volume.T[:, :, None]
    if volume.ndim == 3:
        return np.transpose(volume, (2, 1, 0))
    raise ValueError(f"expected a 2D or 3D image, got shape {volume.shape}")


def _tifffile():
    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover - dependency message
        raise ImportError(
            "reading TIFF stacks needs tifffile: `pip install pynetrees[stacks]`"
        ) from exc
    return tifffile


# ---------------------------------------------------------------------------
# viewing
# ---------------------------------------------------------------------------


def show_stack(stack: Stack, axis: int = 2, ax=None, cmap: str = "gray",
               alpha: float = 1.0):
    """Maximum-intensity projection of every tile, in micron coordinates.

    Parameters
    ----------
    stack : Stack
    axis : {0, 1, 2}, default 2
        Project along x, y or z. MATLAB draws all three at once as
        semi-transparent textured surfaces in a 3D axes; here you ask for
        the one you want, on ordinary 2D axes, because a projection *is*
        two-dimensional and stacking three translucent ones on top of each
        other is hard to read.
    ax : matplotlib Axes, optional
    cmap : str, default "gray"
    alpha : float, default 1.0

    Returns
    -------
    Axes
    """
    import matplotlib.pyplot as plt

    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
    if ax is None:
        ax = plt.figure().add_subplot()

    horizontal, vertical = [a for a in (0, 1, 2) if a != axis]
    for index, tile in enumerate(stack.tiles):
        low, high = stack.extent(index)
        ax.imshow(
            tile.max(axis=axis).T,
            extent=(low[horizontal], high[horizontal],
                    high[vertical], low[vertical]),
            cmap=cmap, alpha=alpha, origin="upper",
        )
    ax.set_xlabel("xyz"[horizontal] + " [um]")
    ax.set_ylabel("xyz"[vertical] + " [um]")
    ax.autoscale_view()
    ax.set_aspect("equal")
    return ax


# ---------------------------------------------------------------------------
# carrier points
# ---------------------------------------------------------------------------


def skeletonize_stack(stack: Stack, thr: float | None = None,
                      close: bool = False) -> np.ndarray:
    """Thin a stack to its centreline and return the points, in microns.

    The output is what :func:`~pynetrees.MST_tree` wants: an ``(n, 3)`` array
    of carrier points to wire into a tree.

    Parameters
    ----------
    stack : Stack
    thr : float, optional
        Binarisation threshold. Defaults to Otsu's, computed per tile,
        which is a documented and reproducible choice; MATLAB instead
        walks a 100-bin histogram down from the top until it has counted
        30000 voxels, a fixed number that means something different for
        every stack size.
    close : bool, default False
        Morphologically close the binary volume first, joining voxels
        separated by a single gap. MATLAB's ``'-c'``.

    Returns
    -------
    np.ndarray
        ``(n, 3)`` carrier points [um].

    Notes
    -----
    **This will not reproduce MATLAB's skeleton voxel for voxel.**
    `skel_stack` is a hand-rolled 3D thinning -- its own header says
    "hopefully correctly interpreted from their papers" of Palagyi and Kuba
    -- while this delegates to `skimage.morphology.skeletonize`, which
    implements Lee, Kashyap & Chu (1994). Both are medial-axis thinnings
    and both preserve topology; the individual voxels they keep differ.
    Reimplementing the toolbox's own reading of a paper would have been
    reimplementing it worse, but the difference is real and this is a
    reconstruction front-end, so it is said out loud rather than buried.
    """
    from skimage.filters import threshold_otsu
    from skimage.morphology import closing, skeletonize

    points = []
    for index, tile in enumerate(stack.tiles):
        volume = np.asarray(tile)
        cutoff = threshold_otsu(volume) if thr is None else thr
        binary = volume > cutoff
        if close:
            binary = closing(binary, np.ones((2, 2, 2), dtype=bool))
        if not binary.any():
            continue
        voxels = np.argwhere(skeletonize(binary))
        if len(voxels):
            points.append(stack.to_microns(voxels, index))
    return np.vstack(points) if points else np.empty((0, 3))


# ---------------------------------------------------------------------------
# fitD_stack
# ---------------------------------------------------------------------------


@accepts_population
def fitD_stack(tree: Tree, stack: Stack, max_radius: float = 30.0,
               samples: int = 5, sigma: float = 3.0,
               selectivity: float = 0.1) -> np.ndarray:
    """Measure a tree's diameters from the image it was traced from.

    For each segment: sample the fluorescence along a line perpendicular to
    it, average those profiles along the segment, sharpen the edges by
    convolving with a derivative of a Gaussian, and read the width between
    the innermost turning points either side of the cable. Requires the tree
    and the stack to share a coordinate frame -- which they do if the tree
    was traced from it.

    Parameters
    ----------
    tree : Tree
    stack : Stack
    max_radius : float, default 30.0
        Half-width of the perpendicular sampling line, **in voxels**. Sets
        the largest diameter that can be found.
    samples : int, default 5
        How many positions along each segment to average over. **MATLAB has
        no such parameter and effectively samples one** -- see Notes.
    sigma : float, default 3.0
        Width [voxels] of the Gaussian whose derivative sharpens the edges.
    selectivity : float, default 0.1
        How steep a turn has to be to count as an edge. MATLAB hard-codes
        0.1 and its own comment names this as the number to change.

    Returns
    -------
    np.ndarray
        One diameter per node, **in microns**. Nodes whose segment could
        not be measured keep the tree's own diameter.

    Notes
    -----
    **MATLAB measures every segment at a single point, and knows it.**
    ``stacks/fitD_stack.m:124`` reads::

        % TODO, CRITICAL: RIGHT NOW ONLY THE TERMINAL POINT IS TAKEN
        mPX = [(P1(1) + cV(1)) (P1(1) + cV(1)) (P2(1))];

    Three sampling positions are built, but ``cV`` is ``P2 - P1``, so
    ``P1 + cV`` *is* ``P2`` -- all three collapse onto the segment's far
    end. Every diameter is therefore read at one point rather than along
    the cable, which is not what the surrounding code was written to do.
    Here ``samples`` positions are spread evenly along the segment as
    intended; pass ``samples=1`` for MATLAB's behaviour.

    Whether that helps depends on the data, and it is worth knowing that it
    is not free: averaging along a segment picks up the *sibling* branch
    near a branch point, so on a clean synthetic phantom the single-point
    measurement is actually the less variable of the two (spread 1.2 versus
    1.8 voxels). On real, noisy fluorescence the averaging is the point.
    Both are available; neither is asserted to be better.

    **The result is converted to microns**, which MATLAB does not do. Its
    width comes out in voxels along the sampling line and is returned as-is,
    to be assigned to ``tree.D`` -- a field in microns everywhere else in
    the toolbox. The two agree only when the in-plane voxel size happens to
    be 1 um. Here the width is scaled by the length of one sampling step in
    microns, which is exact even for anisotropic voxels.

    One MATLAB idiosyncrasy **is** reproduced: the two edge indices are
    offset by one relative to each other (``m_1 = ... + i_max`` against
    ``m_2 = ... + i_max - 1``), which comes from ``diff`` shortening the
    array and makes every width one step narrower than the index gap. It is
    a sub-voxel systematic offset in an already-approximate measurement, and
    changing it would silently move numbers people have published.

    See MATLAB_TOOLBOX_BUGS.md.
    """
    from scipy.ndimage import map_coordinates

    if samples < 1:
        raise ValueError(f"samples must be at least 1, got {samples}")

    X1, X2, Y1, Y2, Z1, Z2 = cyl_tree(tree)
    starts = np.column_stack([X1, Y1, Z1])
    ends = np.column_stack([X2, Y2, Z2])

    projections = [np.asarray(t, dtype=float).max(axis=2) for t in stack.tiles]
    offsets = np.arange(-max_radius, max_radius + 1)
    kernel = np.diff(_gaussian(offsets, sigma))

    diameters = np.asarray(tree.D, dtype=float).copy()
    for node in range(tree.n_nodes):
        width = _fit_one(starts[node], ends[node], stack, projections,
                         offsets, samples, kernel, selectivity,
                         map_coordinates)
        if width is not None:
            diameters[node] = width
    return diameters


def _gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(x**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def _fit_one(start, end, stack, projections, offsets, samples, kernel,
             selectivity, map_coordinates):
    """One segment's diameter in microns, or ``None`` if unmeasurable."""
    index = stack.tile_at((start + end) / 2)
    image = projections[index]

    p1 = stack.to_voxels(start, index)[0, :2]
    p2 = stack.to_voxels(end, index)[0, :2]
    along = p2 - p1
    length = np.linalg.norm(along)
    if length == 0:
        return None

    # perpendicular in the projection plane, unit length in voxel space
    normal = np.array([-along[1], along[0]]) / length

    # positions along the segment -- the part MATLAB collapses onto one
    fractions = (np.linspace(0.0, 1.0, samples + 2)[1:-1] if samples > 1
                 else np.array([1.0]))
    centres = p1[None, :] + fractions[:, None] * along[None, :]

    grid = centres[:, None, :] + offsets[None, :, None] * normal[None, None, :]
    profile = map_coordinates(image, grid.reshape(-1, 2).T, order=1,
                              mode="nearest")
    profile = profile.reshape(len(fractions), -1).mean(axis=0)

    width = _width_between_turns(profile, kernel, selectivity)
    if width is None:
        return None
    # one sampling step is `normal` in voxels; its length in microns
    return width * float(np.linalg.norm(normal * stack.voxel[:2]))


def _width_between_turns(profile: np.ndarray, kernel: np.ndarray,
                         selectivity: float):
    """Distance in sampling steps between the cable's two edges.

    The profile is anchored on its own peak near the centre, then the
    innermost turning point on each side of that peak is taken -- not the
    outermost, and not a threshold crossing, both of which would return the
    whole sampling window on a smooth profile.
    """
    sharpened = np.convolve(profile, kernel, mode="same")
    turning = np.diff(sharpened)

    # MATLAB anchors on the brightest point in the middle half-radius
    half = len(profile) // 2
    window = profile[half : half + max(half // 2, 1)]
    peak = half + int(np.argmax(window))

    steep = np.flatnonzero(turning > selectivity) - peak
    left, right = steep[steep < 0], steep[steep > 0]
    if len(left) == 0 or len(right) == 0:
        return None
    # the -1 is MATLAB's own off-by-one; see fitD_stack's Notes
    return float(right.min() - left.max() - 1)
