"""The README and the example notebooks have to call functions that exist.

V1 removed eight deprecated spellings and the test suite stayed green,
because nothing executes the documentation. The README went on advertising
`pt.plot_tree_mpl`, and three notebooks called `pt.spread_trees`,
`idpar_tree(no_self=)` and `plot_tree(dim2=)` -- every one of them a
`TypeError` or `AttributeError` for the first person to copy a snippet.

Executing the notebooks in CI would catch this and much else, but it needs
PyVista rendering and minutes per run. Reading them is cheap, catches the
failure mode that actually occurred, and cannot go stale itself.
"""

from __future__ import annotations

import pathlib
import re

import pytest

import pynetrees as pt

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md"] + sorted((ROOT / "examples").glob("*.ipynb"))

#: `pt.<something>` -- in a notebook the source lines are JSON strings, so
#: this reads code and prose alike, which is what we want: a name that is
#: wrong in a sentence is just as wrong.
CALL = re.compile(r"\bpt\.([A-Za-z_][A-Za-z0-9_]*)")

#: Keyword arguments removed in V1 (Design Decision #67). A notebook can
#: mention one in prose while explaining the change; none currently does,
#: and if that day comes this is one line to relax.
REMOVED_KEYWORDS = ["dim2=", "no_self=", "no_root=", 'dim="2d"', 'dim="3d"',
                    "dim=\\\"2d\\\"", "dim=\\\"3d\\\""]


def _documents():
    for path in DOCS:
        yield path, path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_every_documented_name_exists(path):
    text = path.read_text(encoding="utf-8")
    missing = sorted({name for name in CALL.findall(text)
                      if not hasattr(pt, name)})
    assert not missing, f"{path.name} uses names pynetrees does not have: {missing}"


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_no_document_uses_a_removed_keyword(path):
    text = path.read_text(encoding="utf-8")
    found = [keyword for keyword in REMOVED_KEYWORDS if keyword in text]
    assert not found, f"{path.name} uses keywords removed in V1: {found}"


# ---------------------------------------------------------------------------
# the README's numbers, not just its names
# ---------------------------------------------------------------------------
#
# "Every snippet below is a real, executed result" -- and it was not. The
# node count and cable length belonged to `hss_tree` while the snippet said
# `sample_tree`, and the branch count, `BO_tree` maximum, path length and
# input resistance matched no bundled tree at all: numbers captured against
# some earlier state of the port and never recomputed. A claim like that
# has to be enforced or dropped.


def test_the_readme_load_and_inspect_block_is_true():
    tree = pt.sample_tree()
    line = (f"{tree.n_nodes} nodes, {pt.len_tree(tree).sum():.0f} um of "
            f"cable, {pt.B_tree(tree).sum()} branch points")
    assert line in (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_readme_verify_snippet_prints_what_it_says():
    """Step 3 of the install instructions is the first thing a new user
    runs; it printed a tree with the wrong node count and the wrong regions."""
    line = f"# {pt.sample_tree()!r}"
    assert line in (ROOT / "README.md").read_text(encoding="utf-8")


def test_every_sample_tree_repr_comment_in_the_readme_is_correct():
    """`in` on its own only proves the *right* string appears somewhere --
    it does not catch a *wrong* one sitting elsewhere in the same file.
    That is exactly how the intro snippet at the top of the README (`print
    (tree)` right under `import pynetrees`) stayed wrong -- claiming 2252
    nodes -- after the Quickstart/Verify copies further down had already
    been fixed to the real 197. Every occurrence has to match, not just one.
    """
    correct = repr(pt.sample_tree())
    pattern = re.compile(r"Tree\(name='sample'[^)]*\)")
    found = pattern.findall((ROOT / "README.md").read_text(encoding="utf-8"))
    assert found, "no sample_tree() repr comment found in README.md"
    wrong = sorted(set(found) - {correct})
    assert not wrong, f"README.md has a stale sample_tree() repr: {wrong}"


def test_the_readme_documents_every_extra_that_exists():
    """A user who cannot read a v7.3 `.mtr` -- which is what current MATLAB
    writes -- needs to know the `matlab` extra exists. Five of the seven
    were missing."""
    import re

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = pyproject.split("[project.optional-dependencies]")[1]
    block = block.split("\n[", 1)[0]
    extras = set(re.findall(r"^(\w+) = \[", block, re.M))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    undocumented = {name for name in extras if f'".[{name}' not in readme
                    and f'.[{name}]' not in readme}
    assert not undocumented, f"extras missing from the README: {undocumented}"


def test_the_readme_measure_block_is_true():
    tree = pt.sample_tree()
    path_length = pt.Pvec_tree(tree, pt.len_tree(tree))
    line = f"# {pt.BO_tree(tree).max()} {path_length.max().round(1)}"
    assert line in (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_readme_cable_analysis_block_is_true():
    tree = pt.sample_tree()
    tree.Ri, tree.Gm, tree.Cm = 100.0, 1 / 20000, 1.0
    voltage = pt.sse_tree(tree, I=0)
    line = f"# input resistance at root: {voltage[0]:.1f} MOhm"
    assert line in (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_scan_actually_reads_something():
    """A guard that passes because it found no documents is worse than
    none -- and the notebooks are found by glob, so an empty match is a
    plausible accident rather than a hypothetical one."""
    assert len(DOCS) >= 5
    names = {name for _, text in _documents() for name in CALL.findall(text)}
    assert len(names) > 40


# ---------------------------------------------------------------------------
# the two API references -- every public name has to be in both, and
# neither may describe a name that no longer exists
# ---------------------------------------------------------------------------

#: A name in backticks, optionally followed by a call or attribute --
#: `len_tree`, `` `len_tree(tree, dim)` ``, `` `Tree.region_nodes(...)` ``.
#: The leading component before a dot is what we match against `__all__`.
BACKTICKED = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)")

#: A generated reference entry: `### \`name(` or `### \`name\``.
HEADING = re.compile(r"^### `([A-Za-z_][A-Za-z0-9_]*)", re.M)


def test_api_overview_mentions_every_public_name():
    """`docs/api-overview.md` is hand-curated, so it drifts the moment a
    function is added or renamed and nobody remembers to update it -- which
    is exactly what happened to the V4 additions (63 of 173 names, more
    than a third of the API) before this test existed."""
    text = (ROOT / "docs" / "api-overview.md").read_text(encoding="utf-8")
    mentioned = set(BACKTICKED.findall(text))
    missing = sorted(set(pt.__all__) - mentioned)
    assert not missing, f"api-overview.md does not mention: {missing}"


def test_function_reference_matches_pytrees_all_exactly():
    """`docs/FUNCTION_REFERENCE.md` is generated (`scripts/gen_function_reference.py`)
    rather than hand-written, so it cannot describe a signature that does
    not exist -- but it can still fall behind if someone changes the
    package and forgets to re-run the generator. Checked both ways: every
    public name must have an entry, and no entry may name something that
    is no longer public.
    """
    text = (ROOT / "docs" / "FUNCTION_REFERENCE.md").read_text(encoding="utf-8")
    documented = set(HEADING.findall(text))

    # pynetrees.blender gets its own section (it is deliberately not part of
    # pynetrees.__all__ -- see its module docstring) but is still generated
    # from a real, importable module, so its names count as expected too.
    expected = set(pt.__all__)
    try:
        import pynetrees.blender as pt_blender
        expected |= set(pt_blender.__all__)
    except Exception:
        pass

    missing = sorted(expected - documented)
    stale = sorted(documented - expected)
    assert not missing, (
        f"FUNCTION_REFERENCE.md is missing {missing} -- re-run "
        f"scripts/gen_function_reference.py"
    )
    assert not stale, (
        f"FUNCTION_REFERENCE.md documents names that no longer exist: "
        f"{stale} -- re-run scripts/gen_function_reference.py"
    )
