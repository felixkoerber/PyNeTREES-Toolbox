"""POV-Ray export: a tree as a renderable scene.

Ports ``IO/pov_tree.m``. POV-Ray is a scene-description language plus a
ray-tracer, so what is written here is *source*, not an image: text you can
read, diff, edit by hand, and render at any resolution later with

.. code-block:: bash

    povray +Ish_mycell.pov +W1600 +H1200 +A

Two files come out, as in MATLAB. ``<name>.pov`` holds the geometry as a
single ``#declare``d union, and is meant to be ``#include``d rather than
rendered; ``sh<name>.pov`` is a small scene that includes it and adds a
camera, lights and a background. Keeping them apart is what lets one
morphology be dropped into several different scenes, and lets a scene be
re-lit without re-exporting the cell.

Blobs
-----

``blob=True`` (the default, as in MATLAB) wraps the cylinders in a POV-Ray
``blob``: an implicit surface that fuses overlapping components into one
smooth skin instead of leaving the visible seams and interpenetration you get
from a pile of separate cylinders. It renders faster too, which is why MATLAB
makes it the default and says so.

The cost is that a blob's surface sits *inside* its components -- the
threshold is a level set of the summed field, not the geometry itself -- so
diameters come out slightly thinner than the morphology says. Use
``blob=False`` when the picture is a measurement rather than an illustration.

Relation to the other exporters
-------------------------------

- :func:`~pynetrees.save_x3d` writes a browser-viewable scene: worse pictures,
  no install, opens on a phone.
- :mod:`pynetrees.blender` builds real Blender objects, which is where to go
  if the scene needs further editing or animation.
- This one produces the best still image, and needs POV-Ray to do it.

MATLAB's ``'->'`` option calls ``winopen`` on the result. Not ported: a save
function should not start programs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .._population import accepts_population
from ..core import Tree
from ..metrics import cyl_tree, len_tree

__all__ = ["POV_STYLES", "save_pov"]

_EOL = "\r\n"

#: Segments shorter than this are dropped. MATLAB's threshold, and the
#: reason for it is POV-Ray rather than aesthetics: a zero-length cylinder is
#: a degenerate object that the parser accepts and the ray-tracer then warns
#: about once per instance.
_MIN_LENGTH = 1e-4

#: Horizontal field of view of the preset scenes' camera [degrees].
_CAMERA_ANGLE = 65

#: Room left around the cell in the preset scenes, as a fraction of its size.
_FRAME_MARGIN = 1.1

#: Words the POV-Ray parser reserves, so a file stem cannot be one of them.
#: The union of the keyword tables in POV-Ray 3.7's ``reswords.cpp`` and
#: 3.8's ``reservedwords.cpp``. All of them are lower-case, which is why
#: POV-Ray's manual recommends a capital letter in every identifier.
_RESERVED_WORDS = frozenset("""
aa_level aa_threshold abs absorption accuracy acos acosh adaptive
adc_bailout agate agate_turb albedo all all_intersections alpha altitude
always_sample ambient ambient_light angle anisotropy aoi aperture append
arc_angle area_illumination area_light array asc ascii asin asinh
assumed_gamma atan atan2 atanh autostop average b_spline background
bezier_spline bicubic_patch bitwise_and bitwise_or bitwise_xor
black_hole blend_gamma blend_mode blob blue blur_samples bmp bokeh
bounded_by box boxed bozo break breakpoint brick brick_size brightness
brilliance bt2020 bt709 bump_map bump_size bumps camera case caustics
ceil cells charset checker chr circular clipped_by clock clock_on cmap
collect color color_map colour colour_map component composite concat
cone confidence conic_sweep conserve_energy contained_by control0
control1 coords cos cosh count crackle crand cube cubic cubic_spline
cubic_wave cutaway_textures cylinder cylindrical datetime debug declare
default defined degrees density density_file density_map dents
deprecated df3 dictionary difference diffuse dimension_size dimensions
direction disc dispersion dispersion_samples dist_exp distance
distance_maximum div double_illuminate dtag eccentricity else elseif
emission end error error_bound evaluate exp expand_thresholds exponent
exr exterior extinction face_indices facets fade_color fade_colour
fade_distance fade_power falloff falloff_angle false fclose file_exists
filter finish fisheye flatness flip floor focal_point fog fog_alt
fog_offset fog_type fopen for form frequency fresnel function gamma
gather gif global global_lights global_settings gradient granite gray
gray_threshold green grey grey_threshold hdr height_field hexagon
hf_gray_16 hf_grey_16 hierarchy hollow hypercomplex if ifdef iff ifndef
image_map image_pattern importance include inside inside_vector int
interior interior_texture internal interpolate intersection intervals
inverse ior irid irid_wavelength isosurface jitter jpeg julia
julia_fractal lambda lathe lemon leopard light_group light_source
linear_spline linear_sweep ln load_file local location log look_at
looks_like low_error_factor macro magnet major_radius mandel map_type
marble material material_map matrix max max_extent max_gradient
max_intersections max_iteration max_sample max_trace max_trace_level
maximum_reuse media media_attenuation media_interaction merge mesh mesh2
mesh_camera metallic method metric min min_extent minimum_reuse mixed
mm_per_unit mod mortar natural_spline nearest_count no no_bump_scale
no_image no_radiosity no_reflection no_shadow noise_generator normal
normal_indices normal_map normal_vectors now number_of_sides
number_of_tiles number_of_waves obj object octaves off offset omega
omnimax on once onion open optional orient orientation orthographic ovus
panoramic parallel parametric pass_through pattern pavement perspective
pgm phase phong phong_size photons pi pigment pigment_map
pigment_pattern planar plane png point_at polarity poly poly_wave
polygon polynomial pot potential pow ppm precision precompute prefix
premultiplied pretrace_end pretrace_start prism prod projected_through
pwr quadratic_spline quadric quartic quaternion quick_color quick_colour
quilted radial radians radiosity radius rainbow ramp_wave rand range
ratio read reciprocal recursion_limit red reflection reflection_exponent
refraction render repeat rgb rgbf rgbft rgbt right ripples rotate
roughness samples save_file sblue scale scallop_wave scattering seed
select sgreen shadowless sin sine_wave sinh sint16be sint16le sint32be
sint32le sint8 size sky sky_sphere slice slope slope_map smooth
smooth_triangle solid sor spacing specular sphere sphere_sweep spherical
spiral1 spiral2 spline split_union spotlight spotted sqr sqrt square
sred srgb srgbf srgbft srgbt statistics steps str strcmp strength strlen
strlwr strupr sturm substr subsurface suffix sum superellipsoid switch
sys t tan tanh target tau text texture texture_list texture_map tga
thickness threshold tiff tightness tile2 tiles tiling tolerance toroidal
torus trace transform translate translucency transmit triangle
triangle_wave triangular true ttf turb_depth turbulence type u u_steps
uint16be uint16le uint8 ultra_wide_angle undef union unofficial up
use_alpha use_color use_colour use_index user_defined utf8 uv_indices
uv_mapping uv_vectors v v_steps val variance vaxis_rotate vcross vdot
version vertex_vectors vlength vnormalize vrotate vstr vturbulence
warning warp water_level waves while width wood wrinkles write x xyz y
yes z
""".split())

#: The scene presets MATLAB spells ``-s1`` .. ``-s3``. Each is
#: ``(background, ambient, description)``; the lights and camera are shared.
POV_STYLES: dict[str, tuple[str, float, str]] = {
    "plain": ("rgbt <1, 1, 1, 0>", 1.0,
              "black on white, no shading -- the one for a figure panel"),
    "fluorescence": ("rgbt <0, 0, 0, 0.75>", 0.0,
                     "glowing cell on black, as a confocal stack looks"),
    "coral": ("rgbt <0.7, 0.7, 0.7, 0.75>", 0.0,
              "lit from above over a grey ground plane"),
}


def _triple(values) -> str:
    return ", ".join(f"{v:.4f}" for v in values)


def _colors(v, n_nodes: int, minmax: bool):
    """Per-node RGB from a scalar field, or ``None`` for a single colour.

    ``v`` is normalised to ``[0, 1]`` -- from zero to its maximum by default,
    or between its own min and max with ``minmax=True``, which is MATLAB's
    ``'-minmax'`` -- and then mapped through matplotlib's ``viridis``.

    MATLAB instead writes the colours to a side-car ``.dat`` file and has
    POV-Ray ``#read`` them back three floats at a time while parsing the
    geometry. That works, but it makes the ``.pov`` unrenderable on its own
    and the pairing positional: lose the ``.dat``, or edit one cylinder out
    of the ``.pov``, and every subsequent colour is off by one. Here the
    colour is written into the cylinder that has it.
    """
    if v is None:
        return None
    values = np.asarray(v, dtype=float).ravel()
    if values.size != n_nodes:
        raise ValueError(
            f"v must have one value per node ({n_nodes}), got {values.size}"
        )

    low = values.min() if minmax else 0.0
    high = values.max()
    span = high - low
    normalised = (values - low) / span if span > 0 else np.zeros_like(values)

    try:
        from matplotlib import colormaps
    except ImportError as error:  # pragma: no cover - exercised by hand
        raise ImportError(
            "colour-coding a POV export needs matplotlib for the colormap; "
            "install pynetrees[plot], or pass v=None and set the colour in "
            "the scene file instead"
        ) from error
    return colormaps["viridis"](normalised)[:, :3]


@accepts_population
def save_pov(
    tree: Tree,
    path: str | Path,
    *,
    v=None,
    nodes=None,
    blob: bool = True,
    minmax: bool = False,
    color=(0.0, 0.0, 0.0),
    style: str | None = "plain",
    width: int = 1200,
    height: int = 900,
) -> Path:
    """Write a tree as a POV-Ray scene.

    Parameters
    ----------
    tree : Tree
    path : str or Path
        ``.pov`` is appended if missing. The stem becomes the POV-Ray
        identifier the geometry is declared under, so it has to be a legal
        one -- letters, digits and underscores, not starting with a digit,
        and not a POV-Ray reserved word such as ``v`` or ``box``. A name
        containing a capital letter is always safe.
    v : array_like, optional
        One value per node, colour-coded through ``viridis``. Without it
        every segment gets ``color``.
    nodes : array_like, optional
        Boolean mask or indices selecting which segments to write. MATLAB's
        ``ipart``.
    blob : bool, default True
        Fuse the cylinders into one smooth implicit surface. Faster to
        render and free of seams, at the cost of a slightly thinner cell --
        see the module docstring.
    minmax : bool, default False
        Normalise ``v`` between its own minimum and maximum rather than from
        zero. MATLAB's ``'-minmax'``.
    color : RGB triple, default black
        Used when ``v`` is ``None``.
    style : {'plain', 'fluorescence', 'coral'} or None, default 'plain'
        Also write the ``sh<name>.pov`` scene file in this style; ``None``
        writes only the geometry. See :data:`POV_STYLES`.
    width, height : int
        The image size suggested in the scene file's render command. The
        camera adapts to whatever size is actually rendered, keeping the
        whole cell in frame.

    Returns
    -------
    Path
        The **geometry** file. The scene file, when written, is
        ``sh<name>.pov`` beside it.

    Notes
    -----
    Coordinates go out unchanged, so the scene is in micrometres and the
    camera distances in the scene file are too.
    """
    path = Path(path)
    if path.suffix.lower() != ".pov":
        path = path.with_suffix(path.suffix + ".pov")
    name = _identifier(path.stem)

    if style is not None and style not in POV_STYLES:
        raise ValueError(
            f"style must be one of {sorted(POV_STYLES)} or None, got {style!r}"
        )

    # cyl_tree gives the six coordinate arrays separately, where MATLAB
    # returns them as one Nx6 matrix; stack them back into per-node endpoints
    X1, X2, Y1, Y2, Z1, Z2 = cyl_tree(tree)
    starts = np.column_stack([X1, Y1, Z1])
    ends = np.column_stack([X2, Y2, Z2])
    lengths = len_tree(tree)
    rgb = _colors(v, tree.n_nodes, minmax)

    selected = lengths > _MIN_LENGTH
    if nodes is not None:
        nodes = np.asarray(nodes)
        chosen = nodes.astype(bool) if nodes.dtype == bool else np.isin(
            np.arange(tree.n_nodes), nodes
        )
        selected &= chosen

    lines = [f"// {tree.name or 'tree'}: {int(selected.sum())} segments, "
             f"written by pynetrees",
             f"#declare {name} = union {{"]
    if blob:
        lines.append("  blob { threshold .15")

    indent = "    " if blob else "  "
    for node in np.flatnonzero(selected).tolist():
        piece = (f"{indent}cylinder {{ <{_triple(starts[node])}>, "
                 f"<{_triple(ends[node])}>, {tree.D[node] / 2:.5f}")
        if blob:
            # the per-component strength a blob needs; 1 is MATLAB's
            piece += ", 1"
        if rgb is not None:
            piece += (f" texture {{ pigment {{ color rgb <"
                      f"{_triple(rgb[node])}> }} }}")
        lines.append(piece + " }")

    if blob:
        lines.append("  }")
    if rgb is None:
        lines.append(f"  pigment {{ color rgb <{_triple(color)}> }}")
    lines.append("}")

    _write(path, lines)

    if style is not None:
        _write(path.with_name(f"sh{path.name}"),
               _scene(name, path.name, tree, style, width, height))
    return path


def _identifier(stem: str) -> str:
    """The file stem as a POV-Ray identifier, or a clear refusal.

    Not sanitised silently: the name is how the caller refers to the object
    from their own scene file, so quietly changing it would break the
    ``#include`` they are about to write by hand. MATLAB does not check at
    all and produces a file POV-Ray cannot parse.
    """
    if not stem or not (stem[0].isalpha() or stem[0] == "_"):
        raise ValueError(
            f"POV-Ray identifiers start with a letter or underscore, so "
            f"{stem!r} cannot name the declared object. Rename the file."
        )
    bad = sorted({c for c in stem if not (c.isalnum() or c == "_")})
    if bad:
        raise ValueError(
            f"{stem!r} contains {bad}, which POV-Ray does not allow in an "
            f"identifier. Rename the file to letters, digits and underscores."
        )
    if stem in _RESERVED_WORDS:
        raise ValueError(
            f"{stem!r} is a POV-Ray reserved word, so it cannot be used as an "
            f"identifier. Rename the file; a name containing a capital letter "
            f"is always safe."
        )
    return stem


def _scene(name: str, geometry_file: str, tree: Tree, style: str,
           width: int, height: int) -> list[str]:
    """The ``sh<name>.pov`` scene: include the geometry, then light it."""
    background, ambient, description = POV_STYLES[style]

    xyz = np.column_stack([tree.X, tree.Y, tree.Z])
    low, high = xyz.min(axis=0), xyz.max(axis=0)
    centre = (low + high) / 2
    span_x, span_y = float(max(high[0] - low[0], 1.0)), float(max(high[1] - low[1], 1.0))
    # lights and the ground plane sit this far from the cell
    reach = max(span_x, span_y)
    near = float(low[2])

    # Everything is placed relative to the cell, and the distance is left for
    # POV-Ray to work out when it renders, because only then is the image's
    # aspect ratio known. It is measured to the cell's nearest face, where a
    # perspective camera sees the least, so the whole cell fits in any frame.
    # MATLAB uses the X extent at a fixed Z, which crops a tall cell and loses
    # one that sits away from Z = 0.
    distance = (
        f"{_FRAME_MARGIN} * max({span_x:.4f}, "
        f"{span_y:.4f} * image_width / image_height) "
        f"/ (2 * tan(radians({_CAMERA_ANGLE / 2})))"
    )

    finish = (f"finish {{ ambient {ambient:.2g} }}" if ambient
              else "finish { ambient 0.15 diffuse 0.8 specular 0.2 }")

    lines = [
        f"// {description}",
        f"// render:  povray +I{{this file}} +W{width} +H{height} +A",
        "",
        f'#include "{geometry_file}"',
        '#include "colors.inc"',
        "",
        f"background {{ {background} }}",
        "",
        "camera {",
        "  right     x * image_width / image_height",
        f"  location  <{centre[0]:.4f}, {centre[1]:.4f}, {near:.4f} - {distance}>",
        f"  look_at   <{centre[0]:.4f}, {centre[1]:.4f}, {centre[2]:.4f}>",
        f"  angle     {_CAMERA_ANGLE}",
        "}",
        "",
    ]

    if style == "fluorescence":
        lines += [
            "light_source { <%.4f, %.4f, %.4f> color rgb <0.4, 1.0, 0.4> "
            "shadowless }" % (centre[0], centre[1], near - reach),
            f"object {{ {name} pigment {{ color rgb <0.15, 1.0, 0.15> }} "
            f"finish {{ ambient 0.9 }} }}",
        ]
    elif style == "coral":
        lines += [
            "light_source { <%.4f, %.4f, %.4f> color White }"
            % (centre[0], high[1] + reach, near - reach),
            "light_source { <%.4f, %.4f, %.4f> color rgb <0.3,0.3,0.4> "
            "shadowless }" % (centre[0], low[1] - reach, centre[2] + reach),
            f"object {{ {name} pigment {{ color rgb <1.0, 0.45, 0.35> }} "
            f"{finish} }}",
            "plane { z, %.4f pigment { color rgb <0.6,0.65,0.7> } }"
            % (high[2] + reach,),
        ]
    else:
        lines += [
            "light_source { <%.4f, %.4f, %.4f> color White shadowless }"
            % (centre[0], centre[1], near - reach),
            f"object {{ {name} {finish} }}",
        ]
    return lines


def _write(path: Path, lines: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(_EOL.join(lines) + _EOL)
