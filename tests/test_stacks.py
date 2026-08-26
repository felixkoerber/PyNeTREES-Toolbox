"""Image stacks: the reconstruction front-end.

Seven of MATLAB's eight `stacks/` functions are file loading or generic 3D
image processing and are delegated to `tifffile`/`imageio`/`scikit-image`
(see GUI_AND_STACKS.md); this covers what was actually ported -- the tiled
`Stack` container, `.stk` compatibility both directions, carrier-point
extraction, and `fitD_stack`.

The tests run against a **synthetic phantom**: a Y-shaped fluorescent
"neuron" of known geometry drawn into a voxel volume. That is deliberate.
The real question about this module is whether microns and voxels are kept
straight through four coordinate transforms, and a phantom whose cable
length and width are known by construction answers it, where a real stack
would only say "plausible".
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import pynetrees as pt
from pynetrees.stacks import (Stack, fitD_stack, load_folder, load_stack,
                            load_tiff, save_stack, show_stack,
                            skeletonize_stack)

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("skimage")

# the phantom's three straight segments, in voxels
_SEGMENTS = [((5, 30, 10), (30, 30, 10)),
             ((30, 30, 10), (52, 45, 10)),
             ((30, 30, 10), (52, 15, 10))]
_CABLE = sum(float(np.linalg.norm(np.subtract(b, a))) for a, b in _SEGMENTS)


def _phantom(sigma: float = 3.0, shape=(60, 60, 20)) -> np.ndarray:
    """A Y-shaped fluorescent cell of known length and width."""
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1)
    volume = np.zeros(shape, dtype=np.float32)
    for a, b in _SEGMENTS:
        a, b = np.array(a, float), np.array(b, float)
        d = b - a
        t = np.clip(((grid - a) @ d) / (d @ d), 0, 1)
        closest = a + t[..., None] * d
        volume = np.maximum(volume, 200 * np.exp(
            -(np.linalg.norm(grid - closest, axis=-1) ** 2) / (2 * sigma**2)))
    return volume


@pytest.fixture(scope="module")
def volume():
    return _phantom()


@pytest.fixture(scope="module")
def stack(volume):
    return Stack(tiles=[volume], origin=[[0.0, 0.0, 0.0]], voxel=[1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# the container
# ---------------------------------------------------------------------------


def test_extent_is_in_microns_not_voxels(volume):
    """The whole reason the container exists: everything downstream works
    in microns and has to be told what a voxel is worth."""
    stack = Stack(tiles=[volume], origin=[[10.0, 20.0, 5.0]],
                  voxel=[0.5, 0.5, 2.0])
    low, high = stack.extent(0)
    np.testing.assert_allclose(low, [10.0, 20.0, 5.0])
    np.testing.assert_allclose(high, [10 + 59 * 0.5, 20 + 59 * 0.5, 5 + 19 * 2])


def test_micron_and_voxel_coordinates_round_trip(volume):
    stack = Stack(tiles=[volume], origin=[[10.0, 20.0, 5.0]],
                  voxel=[0.5, 0.25, 2.0])
    points = np.array([[10.0, 20.0, 5.0], [25.0, 30.0, 15.0]])
    np.testing.assert_allclose(
        stack.to_microns(stack.to_voxels(points, 0), 0), points
    )


def test_a_point_finds_the_tile_it_lies_in(volume):
    """Stacks are tiled fields of view, and the traced tree spans them."""
    two = Stack(tiles=[volume, volume],
                origin=[[0.0, 0.0, 0.0], [200.0, 0.0, 0.0]],
                voxel=[1.0, 1.0, 1.0])
    assert two.tile_at((30, 30, 10)) == 0
    assert two.tile_at((230, 30, 10)) == 1


def test_a_point_outside_every_tile_still_gets_one(volume):
    """Tiles do not tile the plane: a node can fall in the gap between two
    fields of view, and refusing to measure it would be worse than
    measuring it against the nearest."""
    two = Stack(tiles=[volume, volume],
                origin=[[0.0, 0.0, 0.0], [500.0, 0.0, 0.0]],
                voxel=[1.0, 1.0, 1.0])
    assert two.tile_at((490, 30, 10)) == 1
    assert two.tile_at((100, 30, 10)) == 0


def test_mismatched_origins_are_rejected(volume):
    with pytest.raises(ValueError, match="origins"):
        Stack(tiles=[volume, volume], origin=[[0.0, 0.0, 0.0]],
              voxel=[1.0, 1.0, 1.0])


def test_a_bad_voxel_size_is_rejected(volume):
    with pytest.raises(ValueError, match=r"voxel must be"):
        Stack(tiles=[volume], origin=[[0.0, 0.0, 0.0]], voxel=[1.0, 1.0])


# ---------------------------------------------------------------------------
# .stk
# ---------------------------------------------------------------------------


def test_stk_round_trip(stack, tmp_path):
    back = load_stack(save_stack(stack, tmp_path / "cell.stk"))
    np.testing.assert_allclose(back.tiles[0], stack.tiles[0])
    np.testing.assert_allclose(back.voxel, stack.voxel)
    np.testing.assert_allclose(back.origin, stack.origin)


def test_stk_round_trip_with_several_tiles(volume, tmp_path):
    """A one-tile cell array comes back squeezed to a bare array and a
    multi-tile one as an object array; both have to decode."""
    many = Stack(tiles=[volume, volume * 2],
                 origin=[[0.0, 0.0, 0.0], [80.0, 0.0, 0.0]],
                 voxel=[0.5, 0.5, 2.0], names=["left", "right"])
    back = load_stack(save_stack(many, tmp_path / "two.stk"))
    assert len(back) == 2
    np.testing.assert_allclose(back.tiles[1], many.tiles[1])
    assert back.names == ["left", "right"]


def test_a_mat_file_without_a_stack_says_so(tmp_path):
    from scipy.io import savemat

    path = tmp_path / "other.stk"
    savemat(str(path), {"something": np.zeros(3)})
    with pytest.raises(ValueError, match="no 'stack' variable"):
        load_stack(path)


# ---------------------------------------------------------------------------
# image input -- the axis-order trap
# ---------------------------------------------------------------------------


def test_tiff_pages_come_back_in_xyz_order(volume, tmp_path):
    """TIFF pages are (z, y, x); everything in this port is (x, y, z).
    Getting this backwards would rotate every reconstruction."""
    path = tmp_path / "cell.tif"
    tifffile.imwrite(path, np.transpose(volume, (2, 1, 0)).astype(np.uint16))
    loaded = load_tiff(path)
    assert loaded.tiles[0].shape == volume.shape
    np.testing.assert_allclose(loaded.tiles[0], volume.astype(np.uint16))


def test_a_folder_of_planes_loads_the_same_way(volume, tmp_path):
    folder = tmp_path / "planes"
    folder.mkdir()
    for z in range(volume.shape[2]):
        tifffile.imwrite(folder / f"p{z:03d}.tif",
                         volume[:, :, z].T.astype(np.uint16))
    loaded = load_folder(folder)
    assert loaded.tiles[0].shape == volume.shape
    np.testing.assert_allclose(loaded.tiles[0], volume.astype(np.uint16))


def test_voxel_size_and_origin_are_carried_in(volume, tmp_path):
    path = tmp_path / "cell.tif"
    tifffile.imwrite(path, np.transpose(volume, (2, 1, 0)).astype(np.uint16))
    loaded = load_tiff(path, voxel=(0.3, 0.3, 1.5), origin=(100.0, 0.0, 0.0))
    np.testing.assert_allclose(loaded.voxel, [0.3, 0.3, 1.5])
    assert loaded.extent(0)[0][0] == 100.0


def test_planes_of_different_sizes_are_refused(tmp_path):
    folder = tmp_path / "ragged"
    folder.mkdir()
    tifffile.imwrite(folder / "a.tif", np.zeros((10, 10), np.uint16))
    tifffile.imwrite(folder / "b.tif", np.zeros((12, 10), np.uint16))
    with pytest.raises(ValueError, match="differ in size"):
        load_folder(folder)


def test_an_empty_folder_says_so(tmp_path):
    with pytest.raises(ValueError, match="no files matching"):
        load_folder(tmp_path)


# ---------------------------------------------------------------------------
# carrier points
# ---------------------------------------------------------------------------


def test_the_skeleton_traces_the_phantom(stack):
    """Points must land on the cable, not scattered through the volume."""
    points = skeletonize_stack(stack)
    assert len(points) > 20
    distance = _distance_to_phantom(points)
    assert distance.max() < 2.0


def test_the_skeleton_recovers_the_cable_length(stack):
    """End to end: threshold, thin, wire with `MST_tree`, and the total
    length should come back close to the phantom's known 76.4 um."""
    points = skeletonize_stack(stack)
    tree = pt.MST_tree(points[:, 0], points[:, 1], points[:, 2], start=0,
                       bf=0.3, thr=10.0)
    assert pt.len_tree(tree).sum() == pytest.approx(_CABLE, rel=0.1)


def test_carrier_points_come_back_in_microns(volume):
    """The skeleton is computed in voxels; handing voxel indices to
    `MST_tree` would silently scale the whole reconstruction."""
    scaled = Stack(tiles=[volume], origin=[[100.0, 0.0, 0.0]],
                   voxel=[2.0, 2.0, 2.0])
    points = skeletonize_stack(scaled)
    assert points[:, 0].min() >= 100.0
    assert points[:, 0].max() > 110.0  # would be < 60 if left in voxels


def test_an_explicit_threshold_is_used(stack):
    """Otsu is the default because it is documented and reproducible;
    MATLAB counts down a histogram until it has 30000 voxels, a fixed
    number that means something different for every stack size."""
    assert len(skeletonize_stack(stack, thr=100.0)) > 0
    assert len(skeletonize_stack(stack, thr=1e9)) == 0


def test_closing_joins_a_broken_cable(volume):
    """MATLAB's `-c`: a one-voxel gap in the fluorescence should not split
    the reconstruction in two."""
    broken = volume.copy()
    broken[20:22, :, :] = 0
    stack = Stack(tiles=[broken], origin=[[0.0, 0.0, 0.0]],
                  voxel=[1.0, 1.0, 1.0])
    assert len(skeletonize_stack(stack, close=True)) >= \
        len(skeletonize_stack(stack, close=False)) - 2


def _distance_to_phantom(points: np.ndarray) -> np.ndarray:
    out = np.full(len(points), np.inf)
    for a, b in _SEGMENTS:
        a, b = np.array(a, float), np.array(b, float)
        d = b - a
        t = np.clip(((points - a) @ d) / (d @ d), 0, 1)
        out = np.minimum(out, np.linalg.norm(points - (a + t[:, None] * d), axis=1))
    return out


# ---------------------------------------------------------------------------
# fitD_stack
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def traced(stack):
    points = skeletonize_stack(stack)
    return pt.MST_tree(points[:, 0], points[:, 1], points[:, 2], start=0,
                       bf=0.3, thr=10.0)


def test_measured_diameters_match_the_phantoms_width(traced, stack):
    """The phantom is a Gaussian tube of sigma 3, so its full width at half
    maximum is 7.1 voxels. The turning-point method reads between the
    sharpened profile's inflections, which sits a little wider -- but it
    must be in that neighbourhood, not at the sampling window's edge."""
    D = fitD_stack(traced, stack, max_radius=15)
    assert 6.0 < np.median(D) < 12.0


def test_diameters_come_back_in_microns(traced, volume):
    """MATLAB returns the width in *voxels* and assigns it to `tree.D`,
    which is microns everywhere else in the toolbox. The two agree only
    when the voxel happens to be 1 um across."""
    fine = Stack(tiles=[volume], origin=[[0.0, 0.0, 0.0]],
                 voxel=[1.0, 1.0, 1.0])
    coarse = Stack(tiles=[volume], origin=[[0.0, 0.0, 0.0]],
                   voxel=[4.0, 4.0, 1.0])
    scaled_tree = pt.Tree(dA=traced.dA, X=traced.X * 4, Y=traced.Y * 4,
                          Z=traced.Z, D=traced.D, R=np.asarray(traced.R),
                          rnames=list(traced.rnames))
    a = np.median(fitD_stack(traced, fine, max_radius=15))
    b = np.median(fitD_stack(scaled_tree, coarse, max_radius=15))
    assert b == pytest.approx(4 * a, rel=0.25)


def test_sampling_along_the_segment_is_available(traced, stack):
    """MATLAB builds three sampling positions that all collapse onto the
    segment's far end, and flags it CRITICAL in its own source."""
    one = fitD_stack(traced, stack, max_radius=15, samples=1)
    many = fitD_stack(traced, stack, max_radius=15, samples=7)
    assert not np.allclose(one, many)


def test_at_least_one_sample_is_required(traced, stack):
    with pytest.raises(ValueError, match="at least 1"):
        fitD_stack(traced, stack, samples=0)


def test_unmeasurable_segments_keep_their_original_diameter(traced):
    """An empty image has no edges to find; returning zeros would look like
    a measurement."""
    blank = Stack(tiles=[np.zeros((60, 60, 20), np.float32)],
                  origin=[[0.0, 0.0, 0.0]], voxel=[1.0, 1.0, 1.0])
    np.testing.assert_allclose(fitD_stack(traced, blank), traced.D)


# ---------------------------------------------------------------------------
# viewing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_projections_render_on_micron_axes(stack, axis):
    ax = show_stack(stack, axis=axis)
    assert len(ax.images) == len(stack)
    assert "um" in ax.get_xlabel()


def test_each_tile_is_drawn_at_its_own_position(volume):
    two = Stack(tiles=[volume, volume],
                origin=[[0.0, 0.0, 0.0], [200.0, 0.0, 0.0]],
                voxel=[1.0, 1.0, 1.0])
    ax = show_stack(two)
    lefts = sorted(image.get_extent()[0] for image in ax.images)
    assert lefts == [0.0, 200.0]


def test_a_bad_axis_is_rejected(stack):
    with pytest.raises(ValueError, match="axis must be"):
        show_stack(stack, axis=3)
