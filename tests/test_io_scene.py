"""Tests for the scene exporters: `save_x3d` and `save_pov`.

Neither format has a reader here, so these check the *file* rather than a
round trip. X3D is XML, so it is parsed and asserted on as a document --
counting elements and reading attributes, not matching strings. POV-Ray is
not, so the checks there are structural: the declaration is well formed, the
braces balance, and the numbers are the tree's own.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

import pynetrees as pt


@pytest.fixture
def tree():
    return pt.sample_tree()


# ---------------------------------------------------------------------------
# X3D
# ---------------------------------------------------------------------------


def test_x3d_is_well_formed_xml(tmp_path, tree):
    path = pt.save_x3d(tree, tmp_path / "cell")
    root = ET.parse(path).getroot()
    assert root.tag == "X3D"
    assert root.find("Scene") is not None


def test_x3d_writes_one_cylinder_per_segment(tmp_path, tree):
    """The root has nothing behind it to draw a cylinder to, so it is the
    one node that contributes only a sphere."""
    root = ET.parse(pt.save_x3d(tree, tmp_path / "cell")).getroot()
    assert len(root.findall(".//Cylinder")) == tree.n_nodes - 1
    assert len(root.findall(".//Sphere")) == 1


def test_x3d_spheres_option_closes_every_joint(tmp_path, tree):
    root = ET.parse(pt.save_x3d(tree, tmp_path / "cell", spheres=True)).getroot()
    assert len(root.findall(".//Sphere")) == tree.n_nodes


def test_x3d_cylinder_radius_is_half_the_diameter(tmp_path, tree):
    root = ET.parse(pt.save_x3d(tree, tmp_path / "cell")).getroot()
    radii = sorted(float(c.get("radius")) for c in root.findall(".//Cylinder"))
    expected = sorted(tree.D[np.arange(tree.n_nodes) != tree.root] / 2)
    np.testing.assert_allclose(radii, expected, rtol=1e-5)


def test_x3d_cylinder_heights_are_the_segment_lengths(tmp_path, tree):
    root = ET.parse(pt.save_x3d(tree, tmp_path / "cell")).getroot()
    heights = sorted(float(c.get("height")) for c in root.findall(".//Cylinder"))
    lengths = pt.len_tree(tree)
    expected = sorted(lengths[np.arange(tree.n_nodes) != tree.root])
    np.testing.assert_allclose(heights, expected, rtol=1e-5)


def test_x3d_rotation_axes_are_unit_length(tmp_path, tree):
    """A non-unit axis is silently renormalised by some viewers and taken
    literally by others, so it has to be right in the file."""
    root = ET.parse(pt.save_x3d(tree, tmp_path / "cell")).getroot()
    for transform in root.findall(".//Transform"):
        rotation = transform.get("rotation")
        if rotation is None:
            continue                       # a sphere's transform: translation only
        axis = np.array([float(v) for v in rotation.split()[:3]])
        assert np.linalg.norm(axis) == pytest.approx(1.0)


def test_x3d_rotation_actually_points_the_cylinder_along_the_segment():
    """The whole job of the transform: X3D cylinders are Y-up, so the
    rotation has to carry +Y onto the segment direction."""
    tree = pt.sample_tree()
    root = ET.parse(pt.save_x3d(tree, __import__("tempfile").mkdtemp()
                                + "/c.x3d")).getroot()
    transform = next(t for t in root.findall(".//Transform")
                     if t.get("rotation") is not None)
    values = [float(v) for v in transform.get("rotation").split()]
    axis, angle = np.array(values[:3]), values[3]

    # Rodrigues' rotation of +Y about `axis` by `angle`
    y = np.array([0.0, 1.0, 0.0])
    turned = (y * np.cos(angle)
              + np.cross(axis, y) * np.sin(angle)
              + axis * (axis @ y) * (1 - np.cos(angle)))

    centre = np.array([float(v) for v in transform.get("translation").split()])
    xyz = np.column_stack([tree.X, tree.Y, tree.Z])
    idpar = pt.idpar_tree(tree)
    node = int(np.argmin(np.linalg.norm(
        (xyz[idpar] + xyz) / 2 - centre, axis=1)))
    direction = xyz[node] - xyz[idpar[node]]

    # the file stores 6 significant digits, so that is the accuracy to ask for
    np.testing.assert_allclose(
        turned, direction / np.linalg.norm(direction), atol=1e-5
    )


def test_x3d_per_node_colour(tmp_path, tree):
    colors = np.tile([1.0, 0.0, 0.0], (tree.n_nodes, 1))
    root = ET.parse(pt.save_x3d(tree, tmp_path / "cell", color=colors)).getroot()
    assert all(m.get("diffuseColor") == "1 0 0"
               for m in root.findall(".//Material"))


def test_x3d_refuses_a_scalar_per_node_and_says_what_to_do(tree, tmp_path):
    """MATLAB pushes it through the current colormap; there is no such thing
    outside a MATLAB figure, and guessing one would make the file depend on
    invisible state."""
    with pytest.raises(ValueError, match="map it to RGB first"):
        pt.save_x3d(tree, tmp_path / "cell", color=pt.PL_tree(tree))


def test_x3d_offset_moves_every_coordinate(tmp_path, tree):
    plain = ET.parse(pt.save_x3d(tree, tmp_path / "a")).getroot()
    moved = ET.parse(pt.save_x3d(tree, tmp_path / "b",
                                 offset=(1000.0, 0.0, 0.0))).getroot()

    def first_translation(root):
        return np.array([
            float(v) for v in
            root.findall(".//Transform")[0].get("translation").split()
        ])

    np.testing.assert_allclose(
        first_translation(moved) - first_translation(plain), [1000.0, 0.0, 0.0]
    )


def test_x3d_thin_and_thick(tmp_path, tree):
    thin = ET.parse(pt.save_x3d(tree, tmp_path / "t", diameters="thin")).getroot()
    assert {c.get("radius") for c in thin.findall(".//Cylinder")} == {"0.5"}

    thick = ET.parse(pt.save_x3d(tree, tmp_path / "k", diameters="thick")).getroot()
    radii = sorted(float(c.get("radius")) for c in thick.findall(".//Cylinder"))
    expected = sorted((tree.D[np.arange(tree.n_nodes) != tree.root] + 3) / 2)
    np.testing.assert_allclose(radii, expected, rtol=1e-5)


def test_x3d_nodes_selects_a_subset(tmp_path, tree):
    chosen = np.zeros(tree.n_nodes, dtype=bool)
    chosen[10:20] = True
    root = ET.parse(pt.save_x3d(tree, tmp_path / "c", nodes=chosen)).getroot()
    assert len(root.findall(".//Cylinder")) == 10


def test_x3d_viewpoint_frames_the_cell(tmp_path, tree):
    root = ET.parse(pt.save_x3d(tree, tmp_path / "c")).getroot()
    viewpoint = root.find(".//Viewpoint")
    centre = np.array([float(v) for v in
                       viewpoint.get("centerOfRotation").split()])
    xyz = np.column_stack([tree.X, tree.Y, tree.Z])
    np.testing.assert_allclose(centre, (xyz.min(0) + xyz.max(0)) / 2, rtol=1e-4)


def test_x3d_maps_over_a_population(tmp_path, tree):
    paths = pt.save_x3d([tree, tree], tmp_path / "c")
    assert len(paths) == 2


# ---------------------------------------------------------------------------
# POV-Ray
# ---------------------------------------------------------------------------


def test_pov_declares_a_union_named_after_the_file(tmp_path, tree):
    path = pt.save_pov(tree, tmp_path / "mycell")
    text = path.read_text(encoding="utf-8")
    assert "#declare mycell = union {" in text
    assert text.count("{") == text.count("}")


def test_pov_writes_one_cylinder_per_real_segment(tmp_path, tree):
    text = pt.save_pov(tree, tmp_path / "c").read_text(encoding="utf-8")
    lengths = pt.len_tree(tree)
    assert text.count("cylinder {") == int((lengths > 1e-4).sum())


def test_pov_blob_is_the_default_and_can_be_turned_off(tmp_path, tree):
    blobbed = pt.save_pov(tree, tmp_path / "a").read_text(encoding="utf-8")
    plain = pt.save_pov(tree, tmp_path / "b", blob=False).read_text(encoding="utf-8")
    assert "blob { threshold .15" in blobbed
    assert "blob" not in plain
    # the per-component strength a blob needs, and only a blob
    assert ", 1 " in blobbed or blobbed.count(", 1}") or ", 1 }" in blobbed


def test_pov_writes_a_scene_file_beside_the_geometry(tmp_path, tree):
    path = pt.save_pov(tree, tmp_path / "cell", style="fluorescence")
    scene = path.parent / "shcell.pov"
    assert scene.exists()

    text = scene.read_text(encoding="utf-8")
    assert '#include "cell.pov"' in text     # geometry is included, not inlined
    assert "camera {" in text and "light_source" in text
    assert "object { cell" in text


def test_pov_style_none_writes_only_the_geometry(tmp_path, tree):
    path = pt.save_pov(tree, tmp_path / "cell", style=None)
    assert not (path.parent / "shcell.pov").exists()


def test_pov_colour_coding_is_written_into_each_cylinder(tmp_path, tree):
    """MATLAB writes the colours to a side-car .dat and has POV-Ray #read
    them back positionally, which makes the pairing fragile and the .pov
    unrenderable alone."""
    text = pt.save_pov(
        tree, tmp_path / "c", v=pt.PL_tree(tree)
    ).read_text(encoding="utf-8")
    assert text.count("pigment { color rgb") == text.count("cylinder {")
    assert "#read" not in text
    assert not list(tmp_path.glob("*.dat"))


def test_pov_v_must_have_one_value_per_node(tmp_path, tree):
    with pytest.raises(ValueError, match="one value per node"):
        pt.save_pov(tree, tmp_path / "c", v=[1.0, 2.0])


def test_pov_refuses_a_filename_that_is_not_an_identifier(tmp_path, tree):
    """The stem becomes the declared object's name, so quietly sanitising it
    would break the #include the caller is about to write by hand."""
    with pytest.raises(ValueError, match="identifier"):
        pt.save_pov(tree, tmp_path / "my cell")
    with pytest.raises(ValueError, match="letter or underscore"):
        pt.save_pov(tree, tmp_path / "2cells")


def test_pov_rejects_an_unknown_style(tmp_path, tree):
    with pytest.raises(ValueError, match="style must be one of"):
        pt.save_pov(tree, tmp_path / "c", style="neon")


def test_pov_every_style_produces_a_balanced_scene(tmp_path, tree):
    for style in pt.POV_STYLES:
        path = pt.save_pov(tree, tmp_path / f"s_{style}", style=style)
        text = (path.parent / f"shs_{style}.pov").read_text(encoding="utf-8")
        assert text.count("{") == text.count("}"), style


def test_pov_coordinates_are_the_trees_own(tmp_path, tree):
    """No scaling, no recentring: the scene is in micrometres."""
    text = pt.save_pov(
        tree, tmp_path / "c", blob=False, style=None
    ).read_text(encoding="utf-8")
    first = next(line for line in text.splitlines() if "cylinder" in line)
    start = [float(v) for v in first.split("<")[1].split(">")[0].split(",")]
    idpar = pt.idpar_tree(tree)
    xyz = np.column_stack([tree.X, tree.Y, tree.Z])
    assert np.isclose(np.linalg.norm(xyz[idpar] - start, axis=1), 0, atol=1e-4).any()
