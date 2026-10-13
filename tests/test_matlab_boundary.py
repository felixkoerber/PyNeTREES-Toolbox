"""`boundary_tree` against MATLAB's built-in ``boundary()``.

Octave has no ``boundary``, so for a long time this was the one part of the
port with no differential check at all, and the docstrings said so. MATLAB
R2026a is installed on the development machine, and `boundary.m` turns out
to be a readable m-file rather than a built-in, which settles the shrink
factor exactly:

    shp       = alphaShape (P, Inf)
    Acritical = criticalAlpha (shp, 'one-region')
    Ahispec   = alphaSpectrum (shp) restricted to >= Acritical   % descending
    shp.Alpha = Ahispec (numA + 1 - max (ceil ((1 - S) * numA), 1))

The numbers below are MATLAB's own output on the bundled cells, captured
with ``scripts/matlab``'s harness (see that directory's README). Everything
matched to every digit printed once the port anchored on the *one-region*
critical alpha rather than the *all-points* one -- see
`pynetrees.density._one_region_cutoff`.

`r_mc_tree`, `theta_mc_tree` and `growth_tree` all draw their sample points
from this boundary, and in MATLAB all three reach it through the File
Exchange routine ``inpolyhedron``. So this file is what stands behind them:
if the shape is MATLAB's shape, the rest is sampling.
"""

from __future__ import annotations

import numpy as np
import pytest

import pynetrees as pt
from pynetrees.density import _alpha_shape, _one_region_cutoff

SHRINKS = (0.0, 0.25, 0.5, 0.75, 1.0)

#: ``[~, V] = boundary (X, Y, Z, shrink)`` in MATLAB R2026a.
MATLAB_VOLUME = {
    "sample_tree": {
        0.0: 138880.95274606274,
        0.25: 106941.65745979703,
        0.5: 76716.81745722183,
        0.75: 61249.878146540395,
        1.0: 50293.715706508505,
    },
    "hsn_tree": {
        0.0: 6213550.637112729,
        0.25: 3257756.6500151088,
        0.5: 1920978.6541610144,
        0.75: 1656815.8879035283,
        1.0: 1338783.3301466347,
    },
    "hss_tree": {
        0.0: 4621652.116323459,
        0.25: 1811954.2284794461,
        0.5: 1092423.8051965241,
        0.75: 774386.4273817084,
        1.0: 614251.0958965165,
    },
}

#: ``size (boundary (X, Y, Z, shrink), 1)`` -- the surface triangle count.
MATLAB_FACES = {
    "sample_tree": {0.0: 52, 0.25: 110, 0.5: 166, 0.75: 226, 1.0: 280},
    "hsn_tree": {0.0: 100, 0.25: 228, 0.5: 362, 0.75: 510, 1.0: 626},
    "hss_tree": {0.0: 130, 0.25: 346, 0.5: 578, 0.75: 800, 1.0: 1014},
}

#: ``[~, A] = boundary (X, Y, shrink)`` -- the 2D branch, which reaches the
#: same `boundary.m` and so is the same rule one dimension down.
MATLAB_AREA_2D = {
    "sample_tree": {
        0.0: 8404.09815981627,
        0.25: 8394.61696291394,
        0.5: 8160.51497019795,
        0.75: 8008.64258262995,
        1.0: 7779.99291609901,
    },
    "hsn_tree": {
        0.0: 149887.184764723,
        0.25: 137030.623113051,
        0.5: 123350.160500355,
        0.75: 98105.4064363876,
        1.0: 89825.02228262,
    },
    "hss_tree": {
        0.0: 130996.4262076,
        0.25: 109152.0136016,
        0.5: 83958.3021361997,
        0.75: 74993.9018583997,
        1.0: 68606.9678457998,
    },
}

#: ``criticalAlpha (alphaShape (P, Inf), 'one-region')`` and, second, the
#: ``'all-points'`` value the port used to anchor on by mistake. They agree
#: on two of these three cells, which is exactly why the error survived.
MATLAB_CRITICAL_ALPHA = {
    "sample_tree": (39.19080538511447, 39.19080538511447),
    "hsn_tree": (158.68267550501986, 149.9715697798764),
    "hss_tree": (115.26983812522042, 115.26983812522042),
}


def coords_of(name):
    tree = getattr(pt, name)()
    return np.column_stack([tree.X, tree.Y, tree.Z])


@pytest.mark.parametrize("name", sorted(MATLAB_VOLUME))
@pytest.mark.parametrize("shrink", SHRINKS)
def test_enclosed_volume_is_matlabs(name, shrink):
    """The number `scaleV_tree`, `r_mc_tree` and `theta_mc_tree` all rest on."""
    volume = pt.boundary_tree(getattr(pt, name)(), shrink=shrink).volume
    assert volume == pytest.approx(MATLAB_VOLUME[name][shrink], rel=1e-9)


@pytest.mark.parametrize("name", sorted(MATLAB_FACES))
@pytest.mark.parametrize("shrink", SHRINKS)
def test_the_surface_has_matlabs_triangles(name, shrink):
    """Same solid *and* same shell -- a shape can enclose the right volume
    with a different triangulation, and `save_x3d` would show the difference."""
    faces = pt.boundary_tree(getattr(pt, name)(), shrink=shrink).faces
    assert len(faces) == MATLAB_FACES[name][shrink]


@pytest.mark.parametrize("name", sorted(MATLAB_AREA_2D))
@pytest.mark.parametrize("shrink", SHRINKS)
def test_the_2d_branch_encloses_matlabs_area(name, shrink):
    """`boundary(X, Y, s)` is the same rule one dimension down, and
    `dissectSholl_tree` and `r_mc_tree` both have a 2D path into it."""
    area = pt.boundary_tree(getattr(pt, name)(), shrink=shrink, dim=2).volume
    assert area == pytest.approx(MATLAB_AREA_2D[name][shrink], rel=1e-9)


@pytest.mark.parametrize("name", sorted(MATLAB_CRITICAL_ALPHA))
def test_the_tightest_shrink_is_matlabs_one_region_critical_alpha(name):
    """`boundary.m` anchors shrink = 1 here, so an error in this one number
    moves every shrink factor, not just the tightest."""
    one_region, _all_points = MATLAB_CRITICAL_ALPHA[name]
    coords = coords_of(name)

    from scipy.spatial import Delaunay

    from pynetrees.density import _circumradii

    triangulation = Delaunay(coords)
    radii = _circumradii(coords, triangulation.simplices)
    critical = _one_region_cutoff(triangulation.simplices, radii, len(coords))
    assert critical == pytest.approx(one_region, rel=1e-9)


def test_one_region_is_stricter_than_all_points_where_they_differ():
    """The distinction is not academic: on `hsn_tree` requiring a connected
    shape costs 8.7 um of alpha, 43 tetrahedra and 3.6% of the volume, and
    getting it wrong silently shifted every shrink factor."""
    one_region, all_points = MATLAB_CRITICAL_ALPHA["hsn_tree"]
    assert one_region > all_points

    coords = coords_of("hsn_tree")
    from scipy.spatial import Delaunay

    from pynetrees.density import _circumradii

    triangulation = Delaunay(coords)
    radii = _circumradii(coords, triangulation.simplices)

    # the smallest radius at which no point is left out -- MATLAB's
    # criticalAlpha(..., 'all-points'), and what the port used to use
    reached = np.zeros(len(coords), dtype=bool)
    for index in np.argsort(radii):
        reached[triangulation.simplices[index]] = True
        if reached.all():
            covering = float(radii[index])
            break
    assert covering == pytest.approx(all_points, rel=1e-9)

    # every point is covered there, and the shape is still in two pieces --
    # which is the whole distinction
    assert (np.unique(triangulation.simplices[radii <= covering]).size
            == len(coords))
    assert _one_region_cutoff(triangulation.simplices, radii,
                              len(coords)) > covering


@pytest.mark.parametrize("name", sorted(MATLAB_VOLUME))
def test_shrink_only_ever_tightens(name):
    volumes = [pt.boundary_tree(getattr(pt, name)(), shrink=s).volume
               for s in SHRINKS]
    assert volumes == sorted(volumes, reverse=True)


@pytest.mark.parametrize("name", sorted(MATLAB_VOLUME))
def test_shrink_zero_is_the_convex_hull(name):
    """MATLAB documents it, and it is the one shrink factor with an
    independent reference: scipy's own hull volume."""
    from scipy.spatial import ConvexHull

    coords = coords_of(name)
    assert pt.boundary_tree(getattr(pt, name)(), shrink=0.0).volume == (
        pytest.approx(ConvexHull(coords).volume, rel=1e-9)
    )


@pytest.mark.parametrize("name", sorted(MATLAB_VOLUME))
def test_every_point_is_inside_even_at_the_tightest_shrink(name):
    """"Envelops the points" is the documented promise, and the union-find
    that finds the critical alpha is what enforces it."""
    coords = coords_of(name)
    shape = _alpha_shape(coords, 1.0)
    assert len(np.unique(shape.simplices)) == len(coords)
