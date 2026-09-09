"""X3D export: a tree as cylinders in a viewable XML scene.

Ports ``IO/x3d_tree.m``. X3D is the ISO successor to VRML: plain XML that
browsers render through `X_ITE <https://create3000.github.io/x_ite/>`_ or
`x3dom <https://www.x3dom.org/>`_ with no plugin, and that Blender, MeshLab
and ParaView all import. That makes it the format to reach for when the
result has to be *looked at by someone else* -- a link, not an install.

Each segment becomes one ``<Cylinder>`` inside a ``<Transform>`` that rotates
it from X3D's Y-up default onto the segment's own direction and translates it
to the segment's midpoint. Optionally a ``<Sphere>`` at each node closes the
joints, which matters at sharp branch points where two cylinders would
otherwise leave a wedge of empty space.

Relation to the other exporters
-------------------------------

- :func:`~pynetrees.save_pov` writes a POV-Ray scene: photorealistic, but
  needs POV-Ray installed and is a one-way trip to a raster image.
- :mod:`pynetrees.blender` builds real Blender objects, which is the right
  answer when you want to keep editing the scene.
- This is the one that opens in a browser.

MATLAB's ``'->'`` option calls ``winopen`` to launch a viewer. Not ported: a
save function should not start programs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .._population import accepts_population
from ..core import Tree
from ..graphtheory import idpar_tree

__all__ = ["save_x3d"]

#: MATLAB writes CR+LF explicitly (``[char(13), newline]``) and these files
#: are routinely mailed between platforms, so the line ending is fixed rather
#: than left to the OS -- same reasoning as ``io/hoc.py``.
_EOL = "\r\n"

#: The axis a cylinder points along before it is rotated. X3D's ``<Cylinder>``
#: is Y-up by definition, which is why the rotation below is computed from
#: ``+Y`` rather than from ``+Z`` as it would be in POV-Ray.
_CYLINDER_AXIS = np.array([0.0, 1.0, 0.0])


def _rotations(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis-angle taking ``+Y`` onto each segment direction.

    Returns ``(axis, angle)`` with ``axis`` unit-length. A segment already
    parallel to ``+Y`` has no unique rotation axis and gets ``+Y`` with a
    zero angle; one anti-parallel gets ``+Y`` with angle pi, which is a
    half-turn about the cylinder's own axis and therefore correct as well as
    arbitrary.

    MATLAB reaches the same result by cross- and dot-products written out
    longhand, then patches the degenerate case afterwards by testing for an
    all-zero axis. Doing it in that order means the ``acos`` runs on a
    quantity that can be slightly outside ``[-1, 1]`` from rounding; this
    clips first.
    """
    lengths = np.linalg.norm(vectors, axis=1)
    safe = np.where(lengths[:, None] > 0, lengths[:, None], 1.0)
    unit = vectors / safe

    axis = np.cross(_CYLINDER_AXIS, unit)
    axis_norm = np.linalg.norm(axis, axis=1)
    angle = np.arccos(np.clip(unit @ _CYLINDER_AXIS, -1.0, 1.0))

    # parallel or anti-parallel to +Y: the cross product vanishes and any
    # perpendicular axis will do, so pick one deterministically
    degenerate = axis_norm < 1e-12
    axis[degenerate] = _CYLINDER_AXIS
    axis_norm[degenerate] = 1.0
    angle[degenerate & (unit @ _CYLINDER_AXIS < 0)] = np.pi
    angle[degenerate & (unit @ _CYLINDER_AXIS >= 0)] = 0.0

    return axis / axis_norm[:, None], angle


def _colors(color, n_nodes: int) -> np.ndarray:
    """An ``(n_nodes, 3)`` RGB array from whatever the caller gave.

    One triple is broadcast to every node; a per-node array of triples is
    used as-is. MATLAB additionally accepts a scalar *value* per node and
    maps it through the current colormap -- not reproduced, because there is
    no "current colormap" outside a MATLAB figure and guessing one would
    make the file depend on invisible state. Map values to colours yourself
    (``matplotlib.cm.viridis(norm(values))[:, :3]``) and pass the result.
    """
    if color is None:
        color = (0.0, 0.0, 0.0)
    color = np.asarray(color, dtype=float)
    if color.ndim == 1:
        if color.size == n_nodes and n_nodes != 3:
            raise ValueError(
                f"color has one value per node, which MATLAB would push "
                f"through the current colormap. There is no current colormap "
                f"outside a MATLAB figure, so map it to RGB first -- e.g. "
                f"matplotlib.cm.viridis(values / values.max())[:, :3] -- and "
                f"pass the ({n_nodes}, 3) result."
            )
        if color.size != 3:
            raise ValueError(
                f"color must be an RGB triple or one per node, got "
                f"{color.size} values"
            )
        return np.tile(color, (n_nodes, 1))
    if color.shape != (n_nodes, 3):
        raise ValueError(
            f"per-node color must be ({n_nodes}, 3), got {color.shape}. A "
            f"scalar value per node is not accepted -- map it to RGB first, "
            f"e.g. matplotlib.cm.viridis(values)[:, :3]."
        )
    return color


def _triple(values) -> str:
    """Three numbers as X3D writes them: space-separated, no brackets."""
    return " ".join(f"{v:.6g}" for v in values)


@accepts_population
def save_x3d(
    tree: Tree,
    path: str | Path,
    *,
    color=None,
    offset=(0.0, 0.0, 0.0),
    nodes=None,
    spheres: bool = False,
    transparency: float | None = None,
    diameters: str | None = None,
    viewpoint: bool = True,
) -> Path:
    """Write a tree as X3D cylinders.

    Parameters
    ----------
    tree : Tree
    path : str or Path
        ``.x3d`` is appended if missing.
    color : RGB triple or (n_nodes, 3) array, optional
        One colour for the whole tree, or one per node. Components in
        ``[0, 1]``. Defaults to black. Unlike MATLAB this does **not** take
        a scalar per node to push through a colormap -- see
        :func:`_colors`.
    offset : (3,) array_like, default (0, 0, 0)
        Added to every coordinate. MATLAB's ``DD``, for laying several
        exported cells out side by side in one scene.
    nodes : array_like, optional
        Boolean mask or indices selecting which nodes to write. MATLAB's
        ``ipart``. Default: all of them.
    spheres : bool, default False
        Also put a ``<Sphere>`` at each node. MATLAB's ``'-o'``. Worth it at
        sharp branch points, where two cylinders alone leave a visible wedge
        of empty space.
    transparency : float, optional
        ``0`` opaque, ``1`` invisible. Defaults to ``0.3`` without
        ``spheres`` and ``0`` with them, which is what MATLAB does -- though
        only as a side effect of the two branches of its ``'-o'`` test
        having drifted apart, and its docstring mentions neither.
    diameters : {'thin', 'thick'}, optional
        ``'thin'`` draws everything at 1 um; ``'thick'`` adds 3 um to every
        diameter. MATLAB's ``'-thin'``/``'-thick'``, for making a thin arbor
        visible in a wide view.
    viewpoint : bool, default True
        Write a ``<Viewpoint>`` framing the cell, so a viewer opens looking
        at it rather than at the origin. MATLAB derives this from the
        current MATLAB axis (``'-v'``); with no figure to read, this
        computes it from the tree's own bounding box.

    Returns
    -------
    Path
        The file written.

    Notes
    -----
    The root has no segment of its own -- there is nothing behind it to draw
    a cylinder to -- so it contributes a sphere and nothing else. MATLAB
    instead lets the root's cylinder collapse to zero length and relies on
    the ``NaN`` that its normalisation produces to skip it, which works but
    means the file's shape depends on a division by zero.
    """
    path = Path(path)
    if path.suffix.lower() != ".x3d":
        path = path.with_suffix(path.suffix + ".x3d")

    n = tree.n_nodes
    rgb = _colors(color, n)
    offset = np.asarray(offset, dtype=float)
    if offset.size != 3:
        raise ValueError(f"offset must be 3 numbers, got {offset.size}")

    D = tree.D.astype(float).copy()
    if diameters == "thin":
        D[:] = 1.0
    elif diameters == "thick":
        D = D + 3.0
    elif diameters is not None:
        raise ValueError(
            f"diameters must be 'thin', 'thick' or None, got {diameters!r}"
        )

    if transparency is None:
        transparency = 0.0 if spheres else 0.3

    selected = np.ones(n, dtype=bool)
    if nodes is not None:
        nodes = np.asarray(nodes)
        selected = nodes.astype(bool) if nodes.dtype == bool else np.isin(
            np.arange(n), nodes
        )

    xyz = np.column_stack([tree.X, tree.Y, tree.Z])
    idpar = idpar_tree(tree)                      # root is its own parent
    vectors = xyz - xyz[idpar]
    lengths = np.linalg.norm(vectors, axis=1)
    axis, angle = _rotations(vectors)
    midpoints = xyz[idpar] + 0.5 * vectors + offset

    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<X3D>",
        "  <head>",
        f"    <meta content='pynetrees tree' name='{path.stem}'/>",
        "  </head>",
        "  <Scene>",
        "    <Background skyColor='1 1 1'/>",
    ]
    if viewpoint:
        out.extend(_viewpoint(xyz + offset))

    root = tree.root
    for node in np.flatnonzero(selected).tolist():
        if node != root and lengths[node] > 0:
            out.extend(_cylinder(
                axis[node], angle[node], midpoints[node],
                lengths[node], D[node] / 2, rgb[node], transparency,
            ))
        if spheres or node == root:
            out.extend(_sphere(xyz[node] + offset, D[node] / 2, rgb[node]))

    out.extend(["  </Scene>", "</X3D>"])
    _write(path, out)
    return path


def _cylinder(axis, angle, centre, height, radius, rgb, transparency) -> list[str]:
    material = f"<Material diffuseColor='{_triple(rgb)}'"
    if transparency:
        material += f" transparency='{transparency:.6g}'"
    material += "/>"
    return [
        f"    <Transform rotation='{_triple(axis)} {angle:.6g}' "
        f"translation='{_triple(centre)}'>",
        "      <Shape>",
        f"        <Cylinder height='{height:.6g}' radius='{radius:.6g}' "
        f"bottom='false' top='false' side='true' solid='false'/>",
        f"        <Appearance>{material}</Appearance>",
        "      </Shape>",
        "    </Transform>",
    ]


def _sphere(centre, radius, rgb) -> list[str]:
    return [
        f"    <Transform translation='{_triple(centre)}'>",
        "      <Shape>",
        f"        <Sphere radius='{radius:.6g}'/>",
        f"        <Appearance><Material diffuseColor='{_triple(rgb)}'/>"
        f"</Appearance>",
        "      </Shape>",
        "    </Transform>",
    ]


def _viewpoint(xyz: np.ndarray) -> list[str]:
    """A camera far enough back to frame the whole cell.

    Placed down ``+Z`` looking at the bounding-box centre, at a distance of
    the widest horizontal extent over ``0.81`` -- MATLAB's constant, which
    corresponds to the default 45-degree field of view with a little margin.
    """
    if xyz.size == 0:
        return []
    centre = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
    spread = float(np.max(xyz.max(axis=0)[:2] - xyz.min(axis=0)[:2]))
    distance = (spread / 0.81) if spread > 0 else 1.0
    camera = [centre[0], centre[1], centre[2] + distance]
    return [
        f"    <Viewpoint position='{_triple(camera)}' "
        f"centerOfRotation='{_triple(centre)}'/>"
    ]


def _write(path: Path, lines: list[str]) -> None:
    """Write with explicit CRLF, unmolested by universal newlines."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(_EOL.join(lines) + _EOL)
