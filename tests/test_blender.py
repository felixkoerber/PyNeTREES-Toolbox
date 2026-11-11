"""Blender export -- supersedes MATLAB's `pov_tree` and `x3d_tree`.

Skipped entirely without `bpy`, which is not a dependency of `pynetrees`
(a ~300 MB wheel that pins `numpy < 2`).

The claim under test is that the **geometry is right**: a node's diameter
must come out as that diameter in Blender units, regions must be separate
objects, splines must follow the tree's own sections, and the camera must
frame the whole cell whatever the aspect ratio. Rendering is checked only
far enough to know the pipeline runs end to end and produces a lit image --
a pixel comparison against a reference render would be testing Blender's
sampler, not this module.

`bpy` is one global Blender session per process, so every test here starts
from a cleared scene.
"""

from __future__ import annotations

import numpy as np
import pytest

import pynetrees as pt

bpy = pytest.importorskip("bpy")
from pynetrees import blender  # noqa: E402  - after the skip


@pytest.fixture(autouse=True)
def clean():
    """One session, so each test clears it rather than inheriting."""
    blender.reset_scene()
    yield
    blender.reset_scene()


@pytest.fixture(scope="module")
def tree():
    return pt.sample_tree()


# ---------------------------------------------------------------------------
# the scene
# ---------------------------------------------------------------------------


def test_reset_empties_blenders_default_scene():
    """`bpy` starts with a cube, a camera and a light, and keeps everything
    ever made -- there is only one session per process."""
    bpy.ops.mesh.primitive_cube_add()
    blender.reset_scene()
    assert len(bpy.data.objects) == 0
    assert len(bpy.data.materials) == 0


def test_one_object_per_region(tree):
    objects = blender.build_tree(tree)
    assert len(objects) == len(np.unique(tree.R))
    assert [o.name for o in objects] == [
        f"{tree.name}_{r}" for r in tree.rnames
    ]


def test_each_region_gets_its_own_material(tree):
    objects = blender.build_tree(tree)
    colors = {
        tuple(o.data.materials[0].node_tree.nodes["Principled BSDF"]
              .inputs["Base Color"].default_value[:3])
        for o in objects
    }
    assert len(colors) == len(objects)


def test_known_region_names_get_their_documented_colour(tree):
    """So an axon is the same blue across every figure in a paper."""
    labelled = pt.Tree(dA=tree.dA, X=tree.X, Y=tree.Y, Z=tree.Z, D=tree.D,
                       R=np.asarray(tree.R), rnames=["axon", "soma"],
                       name="c")
    objects = blender.build_tree(labelled)
    axon = next(o for o in objects if o.name.endswith("axon"))
    base = axon.data.materials[0].node_tree.nodes["Principled BSDF"] \
        .inputs["Base Color"].default_value[:3]
    assert tuple(round(c, 6) for c in base) == blender.REGION_COLORS["axon"]


def test_region_colours_can_be_overridden(tree):
    objects = blender.build_tree(
        tree, region_colors={"dendrite": (0.0, 1.0, 0.0)}
    )
    dendrite = next(o for o in objects if o.name.endswith("dendrite"))
    base = dendrite.data.materials[0].node_tree.nodes["Principled BSDF"] \
        .inputs["Base Color"].default_value[:3]
    assert tuple(round(c, 6) for c in base) == (0.0, 1.0, 0.0)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_splines_follow_the_trees_own_sections(tree):
    """Not a soup of one cylinder per node: an unbranched run is one
    spline, so the tubes join instead of abutting."""
    objects = blender.build_tree(tree)
    assert sum(len(o.data.splines) for o in objects) == len(
        pt.dissect_tree(tree)
    )


def test_a_nodes_diameter_comes_out_as_that_diameter(tree):
    """The measurement that matters. Blender's per-point `radius` is a
    *fraction* of `bevel_depth`, so the two have to be normalised against
    each other or every render is silently mis-scaled."""
    objects = blender.build_tree(tree)
    widths = [
        2 * o.data.bevel_depth * point.radius
        for o in objects for spline in o.data.splines for point in spline.points
    ]
    assert min(widths) == pytest.approx(tree.D.min(), rel=1e-6)
    assert max(widths) == pytest.approx(tree.D.max(), rel=1e-6)


def test_taper_can_be_switched_off(tree):
    objects = blender.build_tree(tree, taper=False)
    radii = {round(point.radius, 9)
             for o in objects for s in o.data.splines for point in s.points}
    assert len(radii) == 1


def test_zero_diameters_do_not_collapse_to_nothing(tree):
    """A reconstruction with no diameter data would otherwise render as
    invisible threads."""
    flat = pt.Tree(dA=tree.dA, X=tree.X, Y=tree.Y, Z=tree.Z,
                   D=np.zeros(tree.n_nodes), R=np.asarray(tree.R),
                   rnames=list(tree.rnames), name="flat")
    objects = blender.build_tree(flat, min_diameter=0.5)
    widths = [2 * o.data.bevel_depth * p.radius
              for o in objects for s in o.data.splines for p in s.points]
    assert min(widths) == pytest.approx(0.5)


def test_control_points_sit_at_the_nodes_coordinates(tree):
    """To single precision -- Blender stores coordinates as float32, so a
    node at 120 um lands within about 1e-5 um of where it was put. That is
    far below reconstruction precision, but it does mean a `.blend` is not
    a lossless container for a morphology."""
    objects = blender.build_tree(tree)
    placed = np.array([
        tuple(point.co)[:3]
        for o in objects for s in o.data.splines for point in s.points
    ])
    nodes = np.column_stack([tree.X, tree.Y, tree.Z])
    for point in placed[::17]:
        assert np.abs(nodes - point).sum(axis=1).min() < 1e-4


def test_an_offset_moves_the_whole_tree(tree):
    objects = blender.build_tree(tree, offset=(100.0, 0.0, 0.0))
    first = tuple(objects[0].data.splines[0].points[0].co)[:3]
    plain = blender.build_tree(tree, name="plain")
    baseline = tuple(plain[0].data.splines[0].points[0].co)[:3]
    assert first[0] == pytest.approx(baseline[0] + 100.0)


def test_several_trees_can_share_one_scene(tree):
    blender.build_tree(tree, name="a")
    blender.build_tree(tree, name="b", reset=False, offset=(200.0, 0.0, 0.0))
    assert {o.name.split("_")[0] for o in bpy.data.objects} == {"a", "b"}


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def test_saving_produces_an_openable_blend(tree, tmp_path):
    """A `.blend` is where a figure *starts*: real objects to light and
    re-render, not `pov_tree`'s one-shot scene description. So the test is
    that Blender can open it again and the geometry is still there -- not
    that the bytes look right, which they would even if the scene were
    empty. (Blender 5 zstd-compresses these by default, so there is no
    `BLENDER` magic to check for anyway.)"""
    path = blender.save_blend(tree, tmp_path / "cell.blend")
    assert path.suffix == ".blend"
    assert path.stat().st_size > 10_000

    blender.reset_scene()
    bpy.ops.wm.open_mainfile(filepath=str(path))
    curves = [o for o in bpy.data.objects if o.type == "CURVE"]
    assert len(curves) == len(np.unique(tree.R))
    assert sum(len(o.data.splines) for o in curves) == len(pt.dissect_tree(tree))


def test_the_extension_is_added(tree, tmp_path):
    assert blender.save_blend(tree, tmp_path / "cell").name == "cell.blend"


@pytest.mark.slow
def test_rendering_produces_a_lit_image(tree, tmp_path):
    from PIL import Image

    path = blender.render_tree(tree, tmp_path / "cell.png", size=(240, 180),
                               samples=4)
    pixels = np.asarray(Image.open(path).convert("RGB"))
    assert pixels.shape == (180, 240, 3)
    # something brighter than the background is actually in frame
    assert (pixels.max(axis=2) > 120).sum() > 200


@pytest.mark.slow
def test_the_whole_tree_is_in_frame(tmp_path):
    """Blender's `ortho_scale` sets the view's *larger* dimension, so a
    tall cell in a wide frame gets cropped unless the aspect ratio is
    folded in -- which it was not, at first."""
    from PIL import Image

    tall = pt.rot_tree(pt.sample_tree(), [0.0, 0.0, 90.0])
    path = blender.render_tree(tall, tmp_path / "tall.png", size=(320, 160),
                               samples=4)
    pixels = np.asarray(Image.open(path).convert("RGB")).max(axis=2)
    lit = pixels > 120
    # nothing bright may touch the border, or the cell is running off it
    assert not lit[0].any() and not lit[-1].any()
    assert not lit[:, 0].any() and not lit[:, -1].any()


@pytest.mark.slow
@pytest.mark.parametrize("size", [(160, 320), (240, 320), (320, 320)])
def test_the_whole_tree_is_in_frame_when_the_image_is_not_landscape(tmp_path, size):
    """`ortho_scale` is the *height* of a portrait view, not its width, so a
    formula written for landscape frames clips the top and bottom."""
    from PIL import Image

    path = blender.render_tree(pt.sample_tree(), tmp_path / "cell.png", size=size,
                               samples=4)
    lit = np.asarray(Image.open(path).convert("RGB")).max(axis=2) > 120
    assert lit.any()
    assert not lit[0].any() and not lit[-1].any()
    assert not lit[:, 0].any() and not lit[:, -1].any()


def test_tube_ends_are_closed(tree):
    """Open shells show as hollow tips and tear apart under voxel remeshing."""
    assert all(obj.data.use_fill_caps for obj in blender.build_tree(tree))


def _spine_object(cell, **options):
    objects = blender.build_tree(cell, **options)
    return next(o for o in objects if "spine" in o.name)


def _radii(spline, curve):
    return [point.radius * curve.bevel_depth for point in spline.points]


def test_a_spine_is_a_thin_neck_and_a_wide_head(tree):
    """Each segment takes its end node's diameter, so the head is not a cone."""
    spiny = pt.spines_tree(tree, spines=5, neck_diameter=0.3, head_diameter=0.9, rng=1)
    obj = _spine_object(spiny)
    assert len(obj.data.splines) == 5
    for spline in obj.data.splines:
        assert np.allclose(_radii(spline, obj.data), [0.15, 0.15, 0.45, 0.45], atol=1e-5)


def test_spine_regions_are_stepped_when_labelled_separately(tree):
    spiny = pt.spines_tree(tree, spines=3, neck_diameter=0.3, head_diameter=0.9,
                           separate_regions=True, rng=1)
    objects = {o.name.split("_", 1)[1]: o for o in blender.build_tree(spiny)}
    # separate regions are separate sections, so each part is its own cylinder
    for region, radius in (("spine_neck", 0.15), ("spine_head", 0.45)):
        curve = objects[region].data
        assert np.allclose(_radii(curve.splines[0], curve), radius, atol=1e-5)


def test_stepping_can_be_switched_off(tree):
    spiny = pt.spines_tree(tree, spines=3, rng=1)
    obj = _spine_object(spiny, stepped=())
    assert all(len(spline.points) == 3 for spline in obj.data.splines)


def test_dendrites_of_a_spiny_cell_still_taper(tree):
    """Stepping is for spines: a dendrite keeps one control point per node."""
    spiny = pt.spines_tree(tree, spines=5, rng=1)

    def dendrite_points(**options):
        objects = blender.build_tree(spiny, **options)
        dendrite = next(o for o in objects if "dendrite" in o.name)
        return sum(len(spline.points) for spline in dendrite.data.splines)

    assert dendrite_points() == dendrite_points(stepped=())


def test_an_unknown_view_is_rejected(tree, tmp_path):
    with pytest.raises(ValueError, match="view must be"):
        blender.render_tree(tree, tmp_path / "x.png", view="top")
