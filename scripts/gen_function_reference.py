"""Generate docs/FUNCTION_REFERENCE.md: every public pynetrees name, its
signature and its full docstring, grouped by the module it lives in.

Run from the `python_port/` directory after any change to a docstring or a
public signature:

    conda run -n pynetrees python scripts/gen_function_reference.py

Every entry is pulled live via `inspect` -- nothing here is transcribed by
hand, so the reference cannot describe a signature or a docstring that does
not exist. It can still go stale by not being *re-run*, and there is no
pytest guard for that (unlike the README's numbers, which
`tests/test_docs_use_the_real_api.py` does check) -- re-run it as part of
any change that touches a public docstring or signature.
"""
from __future__ import annotations

import inspect
import pathlib
import re

import pynetrees as pt

try:
    import pynetrees.blender as pt_blender
    BLENDER_ERROR = None
except Exception as exc:  # pragma: no cover - bpy may be absent
    pt_blender = None
    BLENDER_ERROR = f"{type(exc).__name__}: {exc}"

OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "FUNCTION_REFERENCE.md"

# Module -> (section title, one-line description), in README's module order
# plus the extras that have grown since that table was last touched.
SECTIONS = [
    ("pynetrees.core", "Data model",
     "The `Tree` container itself, plus validation."),
    ("pynetrees.io.swc", "I/O — SWC", ""),
    ("pynetrees.io.mtr", "I/O — MATLAB `.mtr`", ""),
    ("pynetrees.io.neurolucida", "I/O — NeuroLucida", ""),
    ("pynetrees.io.neu", "I/O — `.neu`", ""),
    ("pynetrees.io.nmf", "I/O — `.nmf`", ""),
    ("pynetrees.io.neuroml", "I/O — NeuroML", ""),
    ("pynetrees.io.hoc", "I/O — NEURON `.hoc`", ""),
    ("pynetrees.io.native", "I/O — native (`.npz`, format dispatch)", ""),
    ("pynetrees.sample", "Bundled sample data", ""),
    ("pynetrees.graphtheory", "Topology",
     "Needs only `dA` — parents, children, branch points, path length, "
     "ordering, sub-trees."),
    ("pynetrees.metrics", "Geometry and metrics",
     "Needs `X`/`Y`/`Z`/`D` — lengths, surfaces, volumes, angles, "
     "transforms, scaling to a target size."),
    ("pynetrees.edit", "Editing",
     "Structural changes: repair, resample, delete, insert, re-root."),
    ("pynetrees.construct", "Construction",
     "Synthetic trees: `MST_tree`, BCT enumeration, growth, smoothing, "
     "soma/diameter models."),
    ("pynetrees.generate", "Generative pipeline",
     "Population-statistics-driven synthesis: cloning, DSCAM-style "
     "self-avoidance, spines."),
    ("pynetrees.density", "Density, hulls and space-filling",
     "Voxel grids, alpha-shape boundaries, spanned area, space-filling "
     "radius (grid and Monte Carlo)."),
    ("pynetrees.persistence", "Topological description (persistent homology)",
     "Branch-length decomposition, barcodes, persistence images."),
    ("pynetrees.electrotonics", "Electrotonics",
     "Passive cable analysis and integrate-and-fire simulation; needs "
     "`tree.Ri`/`tree.Gm` (and `tree.Cm` for time-stepping)."),
    ("pynetrees.stats", "Statistics and comparison",
     "Sholl analysis, von Mises fits, spatial-randomness tests, "
     "population summaries."),
    ("pynetrees.plotting", "Plotting",
     "PyVista 3D rendering, matplotlib previews, dendrograms."),
    ("pynetrees.stacks", "Image stacks",
     "Loading, skeletonising and diameter-fitting confocal/2-photon stacks."),
    ("pynetrees.neuron_bridge", "NEURON simulation",
     "Requires the `neuron` package."),
]

BLENDER_SECTION = ("pynetrees.blender", "Blender export (optional)",
    "Not imported by `import pynetrees` -- opt in with `from pynetrees import "
    "blender`. Needs the `blender` extra (`bpy`), a ~300 MB wheel that "
    "pins `numpy < 2`.")


def github_anchor(title: str) -> str:
    """GitHub's markdown-heading-to-anchor rule: lowercase, spaces to
    hyphens, strip everything that is not alphanumeric/hyphen/space first."""
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    return re.sub(r"[\s]+", "-", slug).strip("-")


def clean_doc(obj) -> str:
    doc = inspect.getdoc(obj)
    return doc.strip() if doc else "*(no docstring)*"


def signature_of(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"


def render_callable(name: str, obj) -> str:
    sig = signature_of(obj)
    doc = clean_doc(obj)
    lines = [f"### `{name}{sig}`", "", doc, ""]
    return "\n".join(lines)


def render_namedtuple(name: str, obj) -> str:
    doc = clean_doc(obj)
    fields = getattr(obj, "_fields", ())
    lines = [f"### `{name}`", "", doc, ""]
    if fields:
        lines.append("| Field | Description |")
        lines.append("|---|---|")
        field_docs = getattr(obj, "__doc__", "") or ""
        for field in fields:
            field_doc = inspect.getdoc(getattr(obj, field, None)) or ""
            field_doc = field_doc.strip().replace("\n", " ")
            lines.append(f"| `{field}` | {field_doc} |")
        lines.append("")
    return "\n".join(lines)


def render_constant(name: str, obj) -> str:
    return f"### `{name}`\n\nValue: `{obj!r}`\n"


def collect(module_names: dict[str, list[str]], get_all, get_attr) -> None:
    for name in sorted(get_all()):
        obj = get_attr(name)
        mod = getattr(obj, "__module__", "?")
        module_names.setdefault(mod, []).append(name)


#: Names whose module cannot be recovered by introspection (plain values
#: like `NO_PARENT = -1` carry no `__module__`), mapped by hand.
MANUAL_MODULE = {"NO_PARENT": "pynetrees.core"}


def build_module_map():
    names_by_module: dict[str, list[str]] = {}
    for name in sorted(pt.__all__):
        obj = getattr(pt, name)
        mod = MANUAL_MODULE.get(name) or getattr(obj, "__module__", "?")
        names_by_module.setdefault(mod, []).append(name)
    return names_by_module


def render_entry(name: str, obj) -> str:
    if isinstance(obj, type):
        return render_namedtuple(name, obj)
    if callable(obj):
        return render_callable(name, obj)
    return render_constant(name, obj)


def main():
    names_by_module = build_module_map()
    total = len(pt.__all__)

    out = []
    out.append("# Function reference")
    out.append("")
    out.append(
        "**Auto-generated** from live introspection of `pynetrees.__all__` -- "
        "every public name, its signature and its complete docstring, "
        "exactly as `help()` would show it. Regenerate after any docstring "
        "or signature change:"
    )
    out.append("")
    out.append("```")
    out.append("conda run -n pynetrees python scripts/gen_function_reference.py")
    out.append("```")
    out.append("")
    out.append(
        f"**{total} public names** as of this generation "
        f"({len([n for n in pt.__all__ if callable(getattr(pt, n)) and not isinstance(getattr(pt, n), type)])} "
        f"functions, "
        f"{len([n for n in pt.__all__ if isinstance(getattr(pt, n), type)])} "
        "result types)."
    )
    out.append("")
    out.append(
        "For a curated, one-line-per-function skim, see "
        "[api-overview.md](api-overview.md). This page is the detailed "
        "counterpart: full docstrings, in full."
    )
    out.append("")

    out.append("## Contents")
    out.append("")
    all_sections = list(SECTIONS)
    if pt_blender is not None:
        all_sections.append(BLENDER_SECTION)
    for mod, title, _ in all_sections:
        count = len(names_by_module.get(mod, []))
        if mod == "pynetrees.blender" and pt_blender is not None:
            count = len(pt_blender.__all__)
        anchor = github_anchor(title)
        out.append(f"- [{title}](#{anchor}) ({count})")
    out.append("")

    covered = set()
    for mod, title, blurb in SECTIONS:
        names = names_by_module.get(mod, [])
        out.append(f"## {title}")
        out.append("")
        if blurb:
            out.append(blurb)
            out.append("")
        if not names:
            out.append("*(none)*")
            out.append("")
            continue
        for name in names:
            covered.add(name)
            obj = getattr(pt, name)
            out.append(render_entry(name, obj))
        out.append("---")
        out.append("")

    # anything not accounted for by the section map
    missing = [n for n in pt.__all__ if n not in covered]
    if missing:
        out.append("## Uncategorised")
        out.append("")
        out.append(
            "Present in `pynetrees.__all__` but not assigned to a section "
            "above -- the generator's module map is missing an entry."
        )
        out.append("")
        for name in sorted(missing):
            obj = getattr(pt, name)
            out.append(render_entry(name, obj))
        out.append("---")
        out.append("")

    if pt_blender is not None:
        mod, title, blurb = BLENDER_SECTION
        out.append(f"## {title}")
        out.append("")
        out.append(blurb)
        out.append("")
        for name in sorted(pt_blender.__all__):
            obj = getattr(pt_blender, name)
            out.append(render_entry(name, obj))
        out.append("---")
        out.append("")
    else:
        out.append(f"## {BLENDER_SECTION[1]}")
        out.append("")
        out.append(
            f"Not importable in the environment this reference was "
            f"generated in ({BLENDER_ERROR}). See "
            "[../src/pynetrees/blender.py](../src/pynetrees/blender.py) directly."
        )
        out.append("")

    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT} ({sum(b.count(chr(10)) + 1 for b in out)} lines, {total} names, "
          f"{len(missing)} uncategorised)")


if __name__ == "__main__":
    main()
