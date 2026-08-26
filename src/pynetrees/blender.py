"""Blender export and headless rendering.

Supersedes MATLAB's `IO/pov_tree.m` (POV-Ray) and `IO/x3d_tree.m`. Both of
those write a text scene file for an external renderer to interpret; this
builds **real Blender objects** in a live Blender session through the
``bpy`` module, so the result can be saved as a ``.blend`` and opened,
lit, animated and re-rendered by hand afterwards. A ``.pov`` file is the end
of the pipeline; a ``.blend`` is the start of one.

What a tree becomes:

- one **curve object per region**, so ``axon`` and ``dendrite`` are separate
  objects in the outliner with their own materials, and can be hidden,
  recoloured or animated independently;
- one **POLY spline per unbranched section**, so the geometry follows the
  morphology's own topology rather than being a soup of cylinders;
- **real taper** -- each control point carries its node's radius, and
  Blender's bevel sweeps a circle scaled by it, so a node's diameter is
  geometry and not a shader trick.

``bpy`` is **not** a dependency of ``pynetrees`` and this module is not
imported by ``import pynetrees``. It is a 300 MB wheel that pins
``numpy < 2``, which is far too much to ask of someone who wants to measure
a Sholl profile. Install it with ``pip install pynetrees[blender]`` and import
this module explicitly::

    from pynetrees import blender

**One Blender session, one process.** ``bpy`` embeds a single global
Blender; every function here operates on that shared session and clears it
first by default. Pass ``reset=False`` to build several trees into one
scene.

**A .blend is a figure, not an archive.** Blender stores coordinates as
single-precision floats, so a node at 120 um comes back within about
1e-5 um of where it was put -- irrelevant against a 0.1 um reconstruction,
but it does mean this is not a lossless container for a morphology. Use
:func:`~pynetrees.save_tree` for that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import Tree
from .graphtheory import dissect_tree, ipar_tree

__all__ = [
    "reset_scene",
    "build_tree",
    "save_blend",
    "render_tree",
    "REGION_COLORS",
]

#: Default region colours. Names are matched case-insensitively and by
#: prefix, so ``apical_dendrite`` picks up ``dendrite``'s colour. Anything
#: unmatched cycles through :data:`_FALLBACK`.
REGION_COLORS: dict[str, tuple[float, float, float]] = {
    "soma": (0.85, 0.75, 0.35),
    "axon": (0.35, 0.55, 0.85),
    "dendrite": (0.85, 0.35, 0.35),
    "basal": (0.85, 0.45, 0.30),
    "apical": (0.90, 0.30, 0.45),
    "spines": (0.55, 0.85, 0.45),
    "primary": (0.80, 0.50, 0.20),
}

_FALLBACK = [
    (0.75, 0.35, 0.75), (0.35, 0.75, 0.70), (0.60, 0.60, 0.30),
    (0.45, 0.45, 0.80), (0.70, 0.70, 0.70),
]


def _bpy():
    try:
        import bpy
    except ImportError as exc:  # pragma: no cover - dependency message
        raise ImportError(
            "Blender export needs the bpy module: "
            "`pip install pynetrees[blender]`. Note bpy is a ~300 MB wheel, "
            "requires the exact Python version it was built for, and pins "
            "numpy < 2 -- which is why it is not a dependency of pynetrees."
        ) from exc
    return bpy


# ---------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------


def reset_scene() -> None:
    """Empty the shared Blender session.

    ``bpy`` starts with Blender's default scene -- a cube, a camera and a
    light -- and keeps everything ever created in it, because there is only
    one session per process. Every entry point here calls this first unless
    told not to.
    """
    bpy = _bpy()
    for collection in (bpy.data.objects, bpy.data.curves, bpy.data.meshes,
                       bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for item in list(collection):
            collection.remove(item)


def build_tree(tree: Tree, name: str | None = None, *,
               region_colors: dict | None = None, resolution: int = 8,
               taper: bool = True, min_diameter: float = 0.1,
               reset: bool = True, offset=(0.0, 0.0, 0.0)) -> list:
    """Build a tree as Blender curve objects, one per region.

    Parameters
    ----------
    tree : Tree
    name : str, optional
        Prefix for the created objects. Defaults to the tree's own name.
    region_colors : dict, optional
        ``{region name: (r, g, b)}``, overriding :data:`REGION_COLORS`.
    resolution : int, default 8
        Bevel resolution -- the swept circle gets ``4 * (resolution + 1)``
        sides. 8 is smooth at print size; drop to 2 or 3 for a cell with
        many thousands of nodes.
    taper : bool, default True
        Vary the tube radius with each node's diameter. With ``False``
        every branch is drawn at the mean diameter, which reads better for
        a schematic and renders faster.
    min_diameter : float, default 0.1
        Floor on the drawn diameter [um]. A reconstruction with zero
        diameters would otherwise collapse to invisible threads.
    reset : bool, default True
        Clear the session first. Pass ``False`` to add this tree to a scene
        already built.
    offset : (dx, dy, dz), default (0, 0, 0)
        Shift the tree, for laying several out in one scene -- pair with
        :func:`~pynetrees.spread_tree`.

    Returns
    -------
    list
        The created ``bpy`` objects, one per region.
    """
    bpy = _bpy()
    if reset:
        reset_scene()

    name = name or tree.name or "tree"
    palette = dict(REGION_COLORS)
    palette.update(region_colors or {})

    coords = np.column_stack([tree.X, tree.Y, tree.Z]) + np.asarray(offset)
    radii = np.maximum(np.asarray(tree.D, dtype=float), min_diameter) / 2
    largest = float(radii.max())

    regions = np.asarray(tree.R, dtype=int)
    objects = []
    for slot, index in enumerate(np.unique(regions).tolist()):
        label = tree.rnames[index] if index < len(tree.rnames) else str(index)
        sections = _region_splines(tree, regions, index)
        if not sections:
            continue

        curve = bpy.data.curves.new(f"{name}_{label}", type="CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = largest
        curve.bevel_resolution = int(resolution)
        # `radius` is a *fraction* of bevel_depth, so normalising here is
        # what makes a node's diameter come out in the tree's own units
        for nodes in sections:
            spline = curve.splines.new("POLY")
            spline.points.add(len(nodes) - 1)
            for slot_index, node in enumerate(nodes):
                point = spline.points[slot_index]
                point.co = (*coords[node], 1.0)
                point.radius = (
                    radii[node] / largest if taper else radii.mean() / largest
                )

        obj = bpy.data.objects.new(f"{name}_{label}", curve)
        obj.data.materials.append(
            _material(f"{name}_{label}", _color(label, palette, slot))
        )
        bpy.context.collection.objects.link(obj)
        objects.append(obj)
    return objects


def _region_splines(tree: Tree, regions: np.ndarray, index: int) -> list:
    """Node runs to draw for one region, as lists of node indices.

    Sections come from :func:`~pynetrees.dissect_tree`, so a spline is an
    unbranched run between branch points and a branch point appears at the
    end of one and the start of its daughters -- which is what makes the
    tubes meet rather than float apart.
    """
    ipar = ipar_tree(tree)
    out = []
    for start, end in dissect_tree(tree).tolist():
        chain = ipar[end]
        hit = np.flatnonzero(chain == start)
        if len(hit) == 0:
            continue
        nodes = chain[: hit[0] + 1][::-1]
        # a section belongs to the region of its end node, matching how
        # `save_hoc` assigns NEURON sections
        if regions[end] != index or len(nodes) < 2:
            continue
        out.append(nodes.tolist())
    return out


def _color(label: str, palette: dict, slot: int):
    """A colour for a region, by name where one is known."""
    lowered = str(label).lower()
    for key, value in palette.items():
        if key in lowered:
            return value
    return _FALLBACK[slot % len(_FALLBACK)]


def _material(name: str, color, roughness: float = 0.45):
    """A Principled BSDF material of the given base colour."""
    bpy = _bpy()
    material = bpy.data.materials.new(name)
    if not material.node_tree:
        material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
    material.diffuse_color = (*color, 1.0)  # viewport fallback
    return material


# ---------------------------------------------------------------------------
# saving and rendering
# ---------------------------------------------------------------------------


def save_blend(tree: Tree | list[Tree], path: str | Path, **kwargs) -> Path:
    """Build a tree (or several) and save the scene as a ``.blend``.

    Extra keyword arguments go to :func:`build_tree`.

    Returns the path written. Open it in Blender and everything is a real,
    editable object -- which is the whole reason for preferring this to
    `pov_tree`'s one-shot scene file.
    """
    bpy = _bpy()
    path = Path(path).resolve()
    if path.suffix != ".blend":
        path = path.with_suffix(path.suffix + ".blend")

    _build_all(tree, kwargs)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    return path


def render_tree(tree: Tree | list[Tree], path: str | Path, *,
                size=(1200, 900), view: str = "xy", samples: int = 32,
                background=(0.05, 0.05, 0.06), margin: float = 1.1,
                **kwargs) -> Path:
    """Build a tree and render it to an image, with no Blender window.

    Parameters
    ----------
    tree : Tree or list[Tree]
    path : str or Path
        ``.png`` is appended if missing.
    size : (width, height), default (1200, 900)
    view : {'xy', 'xz', 'yz'}, default 'xy'
        Which plane to look at. An **orthographic** camera is used, framed
        on the tree's bounding box -- a perspective one would make the near
        half of a cell look thicker than the far half, which is exactly the
        artefact a morphology figure must not have.
    samples : int, default 32
        EEVEE sampling. Higher is cleaner and slower.
    background : (r, g, b), default dark grey
    margin : float, default 1.1
        How much room to leave around the tree.

    Extra keyword arguments go to :func:`build_tree`.

    Returns
    -------
    Path
    """
    bpy = _bpy()
    path = Path(path).resolve()
    if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".exr", ".tif"):
        path = path.with_suffix(path.suffix + ".png")

    trees = _build_all(tree, kwargs)

    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = (int(s) for s in size)
    # resolution first: the camera has to know the aspect ratio to frame
    # the tree without cropping its shorter axis
    _frame(trees, view, margin, background, size)

    scene.render.filepath = str(path)
    scene.render.image_settings.file_format = (
        "PNG" if path.suffix.lower() == ".png" else path.suffix[1:].upper()
    )
    if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = int(samples)
    bpy.ops.render.render(write_still=True)
    return path


def _build_all(tree, kwargs) -> list[Tree]:
    """Build one tree or a list of them into the session."""
    trees = [tree] if isinstance(tree, Tree) else list(tree)
    reset = kwargs.pop("reset", True)
    for index, one in enumerate(trees):
        build_tree(one, reset=(reset and index == 0), **kwargs)
    return trees


def _frame(trees, view: str, margin: float, background, size) -> None:
    """Point an orthographic camera at everything, and light it."""
    bpy = _bpy()
    if view not in ("xy", "xz", "yz"):
        raise ValueError(f"view must be 'xy', 'xz' or 'yz', got {view!r}")

    points = np.vstack([np.column_stack([t.X, t.Y, t.Z]) for t in trees])
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    spans = np.ptp(points, axis=0)
    depth = float(spans.max()) or 1.0

    # Blender's `ortho_scale` is the view's *larger* dimension, so the
    # shorter screen axis has to be scaled by the aspect ratio or it gets
    # cropped -- which is what a tall cell in a wide frame does.
    horizontal, vertical = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[view]
    aspect = size[0] / size[1]
    extent = max(spans[horizontal], spans[vertical] * aspect, 1.0) * margin
    distance = depth * 2 + extent + 10.0

    directions = {"xy": (0, 0, 1), "xz": (0, -1, 0), "yz": (1, 0, 0)}
    rotations = {"xy": (0.0, 0.0, 0.0),
                 "xz": (np.pi / 2, 0.0, 0.0),
                 "yz": (np.pi / 2, 0.0, np.pi / 2)}

    camera_data = bpy.data.cameras.new("camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = extent
    # the default clip range is metres-scale; a cell is hundreds of
    # microns across and would fall outside it
    camera_data.clip_start = 0.01
    camera_data.clip_end = distance * 4
    camera = bpy.data.objects.new("camera", camera_data)
    camera.location = tuple(centre + np.array(directions[view]) * distance)
    camera.rotation_euler = rotations[view]
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    # a key light from the camera's side plus a soft fill, so depth reads
    # without any part of the arbor going black
    for factor, energy in ((np.array([1.0, 1.0, 1.0]), 4.0),
                           (np.array([-1.0, -0.5, 0.5]), 1.5)):
        light_data = bpy.data.lights.new("sun", type="SUN")
        light_data.energy = energy
        light = bpy.data.objects.new("sun", light_data)
        light.location = tuple(centre + factor * distance)
        light.rotation_euler = (np.pi / 4, 0.0, np.pi / 4)
        bpy.context.collection.objects.link(light)

    world = bpy.data.worlds[0] if bpy.data.worlds else \
        bpy.data.worlds.new("world")
    bpy.context.scene.world = world
    if world.node_tree is not None:
        node = world.node_tree.nodes.get("Background")
        if node is not None:
            node.inputs["Color"].default_value = (*background, 1.0)
