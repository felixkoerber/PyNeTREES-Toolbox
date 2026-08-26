"""B3 exporters: NEURON `.hoc` (cell and template), `.nrn`, and NeuroML.

These are write-only formats with no reader on this side, so there is no
round trip to lean on. They are checked instead against the structure the
consuming tool requires -- one `create` per region sized to its section
count, one `connect` per non-root section, `pt3dadd` counts that match the
node/section arithmetic, well-formed XML -- and against the arithmetic that
`t2n_interface` shares with the `.hoc` writer.

MATLAB's `.nrn` branch could not be used as a reference because it does not
run: see the notes on `save_nrn`.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import pynetrees as pt

NML2 = "{http://www.neuroml.org/schema/neuroml2}"
MML = "{http://morphml.org/morphml/schema}"


@pytest.fixture(scope="module")
def tree():
    return pt.sample_tree()


@pytest.fixture(scope="module")
def rooted(tree):
    """What the exporters actually write: `root_tree` is applied first."""
    return pt.root_tree(tree)


def _lines(path):
    return path.read_text().splitlines()


# ---------------------------------------------------------------------------
# .hoc -- shared structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("style", ["cell", "template"])
def test_one_create_per_region_sized_by_section_count(tree, rooted, tmp_path, style):
    path = pt.save_hoc(tree, tmp_path / "cell.hoc", style=style)
    sizes = [int(m.group(1))
             for m in re.finditer(r"^create \w+\[(\d+)\]$",
                                  path.read_text(), re.M)]
    assert len(sizes) == len(np.unique(rooted.R))
    assert sum(sizes) == len(pt.dissect_tree(rooted))


@pytest.mark.parametrize("style", ["cell", "template"])
def test_every_section_but_the_first_is_connected(tree, rooted, tmp_path, style):
    path = pt.save_hoc(tree, tmp_path / "cell.hoc", style=style)
    connects = [line for line in _lines(path) if "connect" in line]
    assert len(connects) == len(pt.dissect_tree(rooted)) - 1


@pytest.mark.parametrize("style", ["cell", "template"])
def test_pt3dadd_count_is_nodes_plus_sections_minus_one(tree, rooted, tmp_path, style):
    """A branch point ends one section and starts each daughter, so it is
    written once per section it belongs to."""
    path = pt.save_hoc(tree, tmp_path / "cell.hoc", style=style)
    written = sum(1 for line in _lines(path) if "pt3dadd" in line)
    assert written == rooted.n_nodes + len(pt.dissect_tree(rooted)) - 1


def test_procedures_are_chunked_so_hoc_can_parse_them(tmp_path):
    """NEURON's parser refuses very long procedures, so both MATLAB and
    this split topology and geometry into numbered procs called from a
    wrapper. The sample tree is large enough to need more than one."""
    path = pt.save_hoc(pt.hsn_tree(), tmp_path / "big.hoc")
    text = path.read_text()
    assert "proc shape3d_2()" in text
    called = re.findall(r"^  (shape3d_\d+)\(\)$", text, re.M)
    defined = re.findall(r"^proc (shape3d_\d+)\(\) \{$", text, re.M)
    assert called == defined  # every chunk defined is called, and vice versa


def test_line_endings_are_crlf_and_not_doubled(tree, tmp_path):
    """`Path.write_text` would translate the `\\n` of an already-CRLF line
    into `\\r\\r\\n` on Windows, giving NEURON a blank line between every
    statement."""
    path = pt.save_hoc(tree, tmp_path / "cell.hoc")
    raw = path.read_bytes()
    assert b"\r\r\n" not in raw
    assert raw.count(b"\r\n") == raw.count(b"\n")


def test_region_names_become_valid_hoc_identifiers(tmp_path, tree):
    """MATLAB sanitises these in `neuron_template_tree` but not in
    `neuron_tree`, where a region called `basal dendrite` writes hoc that
    will not parse."""
    awkward = pt.Tree(dA=tree.dA, X=tree.X, Y=tree.Y, Z=tree.Z, D=tree.D,
                      R=tree.R, rnames=["basal dendrite", "2nd-order"],
                      name="awkward")
    path = pt.save_hoc(awkward, tmp_path / "awkward.hoc")
    for name in re.findall(r"^create (\w+)\[", path.read_text(), re.M):
        assert re.fullmatch(r"[A-Za-z_]\w*", name)


def test_region_names_that_collide_once_sanitised_are_refused(tmp_path, tree):
    """Silently merging two regions would be worse than failing."""
    clashing = pt.Tree(dA=tree.dA, X=tree.X, Y=tree.Y, Z=tree.Z, D=tree.D,
                       R=tree.R, rnames=["a-b", "a_b"], name="clash")
    with pytest.raises(ValueError, match="collide"):
        pt.save_hoc(clashing, tmp_path / "clash.hoc")


def test_an_unknown_style_is_rejected(tree, tmp_path):
    with pytest.raises(ValueError, match="'cell' or 'template'"):
        pt.save_hoc(tree, tmp_path / "cell.hoc", style="nrn")


def test_the_extension_is_added_when_missing(tree, tmp_path):
    assert pt.save_hoc(tree, tmp_path / "cell").name == "cell.hoc"


# ---------------------------------------------------------------------------
# .hoc -- style-specific
# ---------------------------------------------------------------------------


def test_cell_style_creates_sections_at_the_top_level(tree, tmp_path):
    text = pt.save_hoc(tree, tmp_path / "flat.hoc").read_text()
    assert "begintemplate" not in text
    assert text.rstrip().endswith("celldef()")
    assert "create flat_dendrite[" in text  # region names carry the cell name


def test_template_style_wraps_the_cell(tree, tmp_path):
    text = pt.save_hoc(tree, tmp_path / "cellT.hoc", style="template").read_text()
    assert "begintemplate cellT" in text
    assert "endtemplate cellT" in text
    assert "proc init() {" in text
    assert "public is_artificial" in text


def test_template_style_emits_the_t2n_section_lists(tree, tmp_path):
    text = pt.save_hoc(tree, tmp_path / "cellT.hoc", style="template").read_text()
    for expected in ("allregobj   = new List()", "allreg      = new SectionList()",
                     "alladendreg = new SectionList()"):
        assert expected in text


def test_template_style_fixes_the_diameter_step_at_branch_points(tree, tmp_path):
    """NEURON takes a section's first 3D point from where it attaches to
    its parent, so leaving the parent's diameter there steps the surface
    area up at every branch point. The template writer copies the second
    point's diameter over it; the cell writer, like MATLAB, does not.
    """
    flat = pt.save_hoc(tree, tmp_path / "a.hoc", style="cell").read_text()
    wrapped = pt.save_hoc(tree, tmp_path / "b.hoc", style="template").read_text()

    def first_two(text):
        block = text.split("pt3dclear()")[2]  # a section past the first
        return re.findall(r"pt3dadd\(.*?, ([0-9.eE+-]+)\)", block)[:2]

    assert first_two(flat)[0] != first_two(flat)[1]
    assert first_two(wrapped)[0] == first_two(wrapped)[1]


def test_frustum_trees_keep_their_taper(tree, tmp_path):
    tapered = pt.Tree(dA=tree.dA, X=tree.X, Y=tree.Y, Z=tree.Z, D=tree.D,
                      R=tree.R, rnames=tree.rnames, name="tap", frustum=True)
    text = pt.save_hoc(tapered, tmp_path / "tap.hoc", style="template").read_text()
    block = text.split("pt3dclear()")[2]
    diameters = re.findall(r"pt3dadd\(.*?, ([0-9.eE+-]+)\)", block)[:2]
    assert diameters[0] != diameters[1]


# ---------------------------------------------------------------------------
# electrotonics and the run file
# ---------------------------------------------------------------------------


def test_passive_parameters_are_written(tree, tmp_path):
    text = pt.save_hoc(tree, tmp_path / "e.hoc", electrotonics=True).read_text()
    assert "insert pas" in text
    assert f"Ra = {tree.Ri:.5g}" in text
    assert f"g_pas = {tree.Gm:.5g}" in text


def test_asking_for_passive_parameters_a_tree_lacks_is_an_error(tree, tmp_path):
    bare = pt.Tree(dA=tree.dA, X=tree.X, Y=tree.Y, Z=tree.Z, D=tree.D,
                   R=tree.R, rnames=tree.rnames)
    with pytest.raises(ValueError, match="Ri/Gm/Cm"):
        pt.save_hoc(bare, tmp_path / "e.hoc", electrotonics=True)


def test_run_file_loads_the_cell(tree, tmp_path):
    pt.save_hoc(tree, tmp_path / "cell.hoc", run_file=True)
    runner = tmp_path / "run_cell.hoc"
    assert runner.exists()
    assert 'xopen ("cell.hoc")' in runner.read_text()


# ---------------------------------------------------------------------------
# t2n_interface
# ---------------------------------------------------------------------------


def test_interface_has_one_row_per_pt3dadd(tree, rooted, tmp_path):
    """The matrix has to line up with the file, so it is derived from the
    same section layout the writer uses."""
    matrix = pt.t2n_interface(tree)
    written = sum(1 for line in _lines(pt.save_hoc(tree, tmp_path / "c.hoc"))
                  if "pt3dadd" in line)
    assert len(matrix) == written
    assert len(matrix) == rooted.n_nodes + len(pt.dissect_tree(rooted)) - 1


def test_interface_positions_run_from_zero_to_one_within_each_section(tree):
    matrix = pt.t2n_interface(tree)
    assert matrix[:, 2].min() == 0.0
    assert matrix[:, 2].max() == 1.0
    for section in np.unique(matrix[:, 1]):
        positions = matrix[matrix[:, 1] == section, 2]
        assert positions[0] == 0.0
        assert positions[-1] == pytest.approx(1.0)
        assert np.all(np.diff(positions) >= 0)


def test_interface_covers_every_section_exactly_once(tree, rooted):
    matrix = pt.t2n_interface(tree)
    assert sorted(np.unique(matrix[:, 1]).astype(int)) == list(
        range(len(pt.dissect_tree(rooted)))
    )


def test_interface_node_indices_are_valid(tree, rooted):
    nodes = pt.t2n_interface(tree)[:, 0].astype(int)
    assert nodes.min() >= 0
    assert nodes.max() < rooted.n_nodes
    assert len(np.unique(nodes)) == rooted.n_nodes  # every node appears


# ---------------------------------------------------------------------------
# .nrn
# ---------------------------------------------------------------------------


def test_nrn_makes_one_section_per_graph_segment(tree, rooted, tmp_path):
    path = pt.save_nrn(tree, tmp_path / "flat.nrn")
    sizes = [int(m.group(1))
             for m in re.finditer(r"^create \w+\[(\d+)\]$", path.read_text(), re.M)]
    assert sum(sizes) == rooted.n_nodes - 1  # every node but the root


def test_nrn_geometry_block_has_nine_numbers_per_section(tree, rooted, tmp_path):
    """`geometry()` reads `nseg` then two `pt3dadd(fscan() x4)` per
    section, so each data row must carry exactly nine values."""
    text = pt.save_nrn(tree, tmp_path / "flat.nrn").read_text()
    data = [line for line in text.splitlines()
            if line and re.fullmatch(r"[-0-9.eE+ ]+", line)]
    assert len(data) == rooted.n_nodes - 1
    assert {len(line.split()) for line in data} == {9}


def test_nrn_works_for_a_single_region_tree(tree, tmp_path):
    """MATLAB's `.nrn` branch raises here: its one-region path reads
    `H1 (counterR)` where `counterR` is the loop variable of a loop that
    only runs in the *other* branch."""
    one = pt.Tree(dA=tree.dA, X=tree.X, Y=tree.Y, Z=tree.Z, D=tree.D,
                  R=np.zeros(tree.n_nodes, dtype=int), rnames=["dend"],
                  name="one")
    text = pt.save_nrn(one, tmp_path / "one.nrn").read_text()
    assert len(re.findall(r"^create ", text, re.M)) == 1


def test_nrn_passive_parameters_use_the_right_field_names(tree, tmp_path):
    """MATLAB's `-e` block reads `tree.ri`/`tree.rm`/`tree.cm`, lowercase;
    the tree structure has `Ri`/`Gm`/`Cm` and no lowercase counterparts, so
    that path raises a missing-field error."""
    text = pt.save_nrn(tree, tmp_path / "e.nrn", electrotonics=True).read_text()
    assert f"Ra = {tree.Ri:.5g}" in text
    assert f"g_pas = {tree.Gm:.5g}" in text


def test_nrn_honours_an_explicit_segment_count(tree, tmp_path):
    text = pt.save_nrn(tree, tmp_path / "res.nrn", res=7).read_text()
    data = [line for line in text.splitlines()
            if line and re.fullmatch(r"[-0-9.eE+ ]+", line)]
    assert {line.split()[0] for line in data} == {"7"}


# ---------------------------------------------------------------------------
# NeuroML
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version, tag", [("2", NML2), ("1", MML)])
def test_neuroml_is_well_formed_and_has_one_segment_per_edge(
    tree, tmp_path, version, tag
):
    path = pt.save_neuroml(tree, tmp_path / "cell.xml", version=version)
    root = ET.fromstring(path.read_text())
    segments = list(root.iter(f"{tag}segment"))
    assert len(segments) == tree.n_nodes - 1


def test_neuroml_schema_location_is_two_tokens(tree, tmp_path):
    """MATLAB concatenates the namespace and the schema URL with no space,
    so no validator can resolve either half."""
    root = ET.fromstring(
        pt.save_neuroml(tree, tmp_path / "cell.xml").read_text()
    )
    location = root.get(f"{{{ 'http://www.w3.org/2001/XMLSchema-instance' }}}"
                        "schemaLocation")
    assert location is not None and len(location.split()) == 2


def test_neuroml_root_segments_have_no_parent(tree, tmp_path):
    """MATLAB rewrites the root segment's parent from -1 to 0, declaring it
    a child of the first segment. A segment with no `<parent>` is how
    NeuroML says "the cell starts here"."""
    root = ET.fromstring(
        pt.save_neuroml(tree, tmp_path / "cell.xml").read_text()
    )
    parentless = [s for s in root.iter(f"{NML2}segment")
                  if s.find(f"{NML2}parent") is None]
    assert len(parentless) == 1


def test_neuroml_segment_parents_follow_the_tree(tree, tmp_path):
    from pynetrees.graphtheory import idpar_tree

    root = ET.fromstring(
        pt.save_neuroml(tree, tmp_path / "cell.xml").read_text()
    )
    idpar = idpar_tree(tree)
    for segment in root.iter(f"{NML2}segment"):
        node = int(segment.get("id"))
        parent = segment.find(f"{NML2}parent")
        if parent is not None:
            assert int(parent.get("segment")) == idpar[node]


def test_neuroml_uses_the_distal_diameter_at_both_ends(tree, tmp_path):
    """Deliberate, and MATLAB's own choice -- a segment is a uniform
    cylinder, not a frustum. Flagged in its source as `% NOTE: dist
    diameter!!`."""
    root = ET.fromstring(
        pt.save_neuroml(tree, tmp_path / "cell.xml").read_text()
    )
    for segment in root.iter(f"{NML2}segment"):
        node = int(segment.get("id"))
        for end in ("proximal", "distal"):
            point = segment.find(f"{NML2}{end}")
            assert float(point.get("diameter")) == pytest.approx(tree.D[node])


def test_neuroml_segment_groups_carry_the_regions(tree, tmp_path):
    """An addition: MATLAB writes no groups at all, so its export has no
    way to say which cable is axon and which is dendrite."""
    root = ET.fromstring(
        pt.save_neuroml(tree, tmp_path / "cell.xml").read_text()
    )
    groups = {g.get("id"): len(list(g.iter(f"{NML2}member")))
              for g in root.iter(f"{NML2}segmentGroup")}
    assert set(groups) == set(tree.rnames)
    assert sum(groups.values()) == tree.n_nodes - 1


def test_neuroml_segment_groups_can_be_switched_off(tree, tmp_path):
    root = ET.fromstring(
        pt.save_neuroml(tree, tmp_path / "c.xml", segment_groups=False).read_text()
    )
    assert not list(root.iter(f"{NML2}segmentGroup"))


def test_neuroml_escapes_region_names_that_would_break_a_string_writer(tree, tmp_path):
    """The reason this uses ElementTree: MATLAB concatenates strings, so a
    region called `a&b` writes a document nothing can parse."""
    nasty = pt.Tree(dA=tree.dA, X=tree.X, Y=tree.Y, Z=tree.Z, D=tree.D,
                    R=tree.R, rnames=['a&b<c"d', "sub'tree"], name="nasty")
    ET.fromstring(pt.save_neuroml(nasty, tmp_path / "n.xml").read_text())


def test_an_unknown_neuroml_version_is_rejected(tree, tmp_path):
    with pytest.raises(ValueError, match="version must be"):
        pt.save_neuroml(tree, tmp_path / "c.xml", version="2a")


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".hoc", ".nrn", ".xml"])
def test_export_formats_reach_the_dispatcher(tree, tmp_path, suffix):
    assert pt.save_tree(tree, tmp_path / f"cell{suffix}").exists()


def test_export_only_formats_cannot_be_loaded_back(tree, tmp_path):
    path = pt.save_tree(tree, tmp_path / "cell.hoc")
    with pytest.raises(ValueError, match="this port reads"):
        pt.load_tree(path)
