"""Tests for pynetrees.gc_model: the Beining et al. (2017) granule cell.

Split deliberately in two. Everything about *resolving a specification* --
composing variants, scaling conductances, folding spine membrane, diffing
two models -- is pure and runs anywhere; only the handful of tests that
actually build sections need NEURON, and those skip without it, matching
`test_neuron_bridge.py`.

`TOY` below is a deliberately small hand-written spec, so the pure tests read
as documentation of the structure rather than as assertions about 570 numbers.
The bundled `GC_BIOPHYS` -- the real published table -- gets its own group at
the bottom, including a check against the original MATLAB dump when that dump
happens to be checked out at `tests/Active GC Model/gc_mech.json`; nothing
else in the suite needs that directory.
"""

import json

import numpy as np
import pytest

import pynetrees as pt
from pynetrees.gc_model import GC_BIOPHYS, GC_SPINE_DENSITY, GCSpec, gc_spec

# ---------------------------------------------------------------------------
# a small hand-written spec, so the pure tests do not depend on the bundled
# 40 kB dump and read as documentation of the structure
# ---------------------------------------------------------------------------

TOY = {
    "all": {
        "pas": {"cm": 1.0, "Ra": 200.0, "g": 1e-5, "e": -80.0},
        "k_ion": {"ek": -93.0},
    },
    "soma": {
        "pas": {"cm": 0.9, "Ra": 200.0, "g": 2e-5, "e": -80.0},
        "na8st": {"gbar": 0.088},
        "Kv21": {"gkbar": 0.007},
    },
    "adendMML": {
        "pas": {"cm": 0.9, "Ra": 200.0, "g": 2e-5, "e": -80.0},
        "Kv42": {"gkbar": 0.00435, "vshift": -20.0},
    },
    "axon": {
        "pas": {"cm": 0.9, "Ra": 100.0, "g": 6.6e-6, "e": -70.0},
        "na8st": {"gbar": 0.088},
    },
}


@pytest.fixture
def toy():
    return gc_spec(TOY)


# ---------------------------------------------------------------------------
# resolving a specification -- no simulator involved
# ---------------------------------------------------------------------------


def test_channels_excludes_passive_and_ion_blocks(toy):
    assert toy.channels == ["Kv21", "Kv42", "na8st"]
    assert "pas" in toy.mechanisms and "k_ion" in toy.mechanisms


def test_a_region_inherits_all_then_overrides_it(toy):
    soma = toy.for_region("soma")
    assert soma["k_ion"] == {"ek": -93.0}      # inherited from `all`
    assert soma["pas"]["g"] == 2e-5            # soma's own, not `all`'s 1e-5


def test_an_unknown_region_gets_exactly_the_all_block(toy):
    assert toy.for_region("no_such_region") == toy.regions["all"]


def test_scaling_touches_conductances_and_leaves_kinetics_alone(toy):
    scaled = toy.scaled({"Kv42": 2.0})
    mml = scaled.regions["adendMML"]["Kv42"]
    assert mml["gkbar"] == pytest.approx(0.0087)
    assert mml["vshift"] == -20.0  # a shift is not a conductance


def test_scaling_can_name_the_parameter_explicitly(toy):
    silenced = toy.scaled({"na8st": {"gbar": 0.0}})
    assert silenced.regions["soma"]["na8st"]["gbar"] == 0.0
    assert silenced.regions["axon"]["na8st"]["gbar"] == 0.0


def test_scaling_can_be_restricted_to_one_region(toy):
    scaled = toy.scaled({"na8st": 2.0}, region="axon")
    assert scaled.regions["axon"]["na8st"]["gbar"] == pytest.approx(0.176)
    assert scaled.regions["soma"]["na8st"]["gbar"] == pytest.approx(0.088)


def test_a_spec_is_frozen_so_a_variant_cannot_change_its_parent(toy):
    before = toy.regions["soma"]["na8st"]["gbar"]
    toy.scaled({"na8st": 10.0}).overridden({"soma": {"Kv21": {"gkbar": 1.0}}})
    assert toy.regions["soma"]["na8st"]["gbar"] == before
    assert toy.regions["soma"]["Kv21"]["gkbar"] == 0.007


def test_passive_only_is_the_passive_model(toy):
    passive = gc_spec(TOY, passive_only=True)
    assert passive.channels == []
    assert passive.regions["soma"]["pas"] == toy.regions["soma"]["pas"]


def test_diff_names_exactly_what_a_variant_changed(toy):
    newborn = toy.scaled({"Kv42": 0.5})
    difference = newborn.diff(toy)
    assert list(difference) == ["adendMML"]
    before_after = difference["adendMML"]["Kv42"]["gkbar"]
    assert before_after == pytest.approx((0.002175, 0.00435))


def test_diff_of_a_spec_with_itself_is_empty(toy):
    assert toy.diff(toy) == {}


def test_spine_scaling_multiplies_the_passive_membrane(toy):
    unscaled = gc_spec(TOY, spine_scale_applied=False)
    scaled = gc_spec(TOY, spine_scale_applied=False, spine_scale=0.9)

    factor = 1.0 + GC_SPINE_DENSITY["adendMML"] * 0.9   # 1.9
    assert scaled.regions["adendMML"]["pas"]["cm"] == pytest.approx(
        unscaled.regions["adendMML"]["pas"]["cm"] * factor
    )
    assert scaled.regions["adendMML"]["pas"]["g"] == pytest.approx(
        unscaled.regions["adendMML"]["pas"]["g"] * factor
    )
    # the soma has no spines, so it must come through untouched
    assert scaled.regions["soma"]["pas"]["cm"] == pytest.approx(0.9)
    # and Ra is a cytoplasmic property, not membrane -- never scaled
    assert scaled.regions["adendMML"]["pas"]["Ra"] == 200.0


def test_spine_scaling_an_already_scaled_spec_is_refused():
    """A MATLAB dump made with scalespines = 1 already has it folded in;
    applying it again is the mistake the flag exists to catch."""
    with pytest.raises(ValueError, match="already has spine scaling"):
        gc_spec(TOY, spine_scale=0.9)


def test_spine_scaling_warns_about_a_region_it_has_no_density_for():
    spec = dict(TOY, unknown_layer={"pas": {"cm": 1.0, "Ra": 200.0, "g": 1e-5}})
    with pytest.warns(UserWarning, match="no spine density"):
        gc_spec(spec, spine_scale_applied=False, spine_scale=0.9)


def test_a_spec_reads_from_json(tmp_path):
    path = tmp_path / "toy_mech.json"
    path.write_text(json.dumps(TOY), encoding="utf-8")
    spec = gc_spec(path)
    assert spec.name == "toy_mech"
    assert spec.channels == ["Kv21", "Kv42", "na8st"]


def test_gc_spec_accepts_another_spec_and_applies_the_options(toy):
    derived = gc_spec(toy, scale={"na8st": 0.0}, celsius=37.0, name="hot")
    assert derived.celsius == 37.0
    assert derived.name == "hot"
    assert derived.regions["soma"]["na8st"]["gbar"] == 0.0


def test_the_source_dict_is_not_mutated():
    """gc_spec deep-copies, so a caller's own dict is theirs afterwards."""
    source = {"soma": {"na8st": {"gbar": 1.0}}}
    gc_spec(source, scale={"na8st": 0.0})
    assert source["soma"]["na8st"]["gbar"] == 1.0


# ---------------------------------------------------------------------------
# building it -- needs NEURON
# ---------------------------------------------------------------------------

neuron = pytest.importorskip("neuron")


@pytest.fixture
def gc_tree():
    """A small tree carrying the granule-cell region names."""
    tree = pt.repair_tree(pt.sample_tree())
    rnames = ["soma", "adendMML"]
    R = np.where(np.arange(tree.n_nodes) < 8, 0, 1)
    return tree.copy(R=R, rnames=rnames, name="toy_gc")


def test_GC_init_folds_the_passive_parameters_per_region(gc_tree):
    gc = pt.GC_init(gc_tree, TOY)

    soma = next(iter(gc.region_sections["soma"][0]))
    mml = next(iter(gc.region_sections["adendMML"][0]))
    assert gc.region_sections["soma"][0].cm == pytest.approx(0.9)
    assert soma.pas.g == pytest.approx(2e-5)
    assert mml.pas.e == pytest.approx(-80.0)


def test_GC_init_reports_channels_the_library_does_not_have(gc_tree):
    """Without the model's compiled .mod files, every granule-cell channel
    is missing -- which must be said, not silently ignored."""
    gc = pt.GC_init(gc_tree, TOY)
    missing = {mech for _, mech, _ in gc.skipped}
    assert {"na8st", "Kv21", "Kv42"} <= missing


def test_GC_init_strict_refuses_to_build_a_half_applied_model(gc_tree):
    with pytest.raises(RuntimeError, match="could not be applied"):
        pt.GC_init(gc_tree, TOY, strict=True)


def test_GC_init_passive_only_needs_no_compiled_mechanisms(gc_tree):
    gc = pt.GC_init(gc_tree, TOY, passive_only=True, strict=True)
    assert gc.skipped == []
    assert gc.spec.channels == []


def test_GC_init_refuses_a_region_it_has_no_passive_block_for(gc_tree):
    """nan for Ra would integrate to nan seconds later with nothing pointing
    back here, so it has to be caught while the region name is still known."""
    stray = gc_tree.copy(rnames=["soma", "not_in_the_spec"])
    no_fallback = {k: v for k, v in TOY.items() if k != "all"}
    with pytest.raises(ValueError, match="no passive parameters"):
        pt.GC_init(stray, no_fallback)


def test_an_all_block_covers_a_region_the_spec_does_not_name(gc_tree):
    """The flip side: with an `all` pas block there is no such thing as an
    uncovered region, which is why the check above has to remove it."""
    stray = gc_tree.copy(rnames=["soma", "not_in_the_spec"])
    gc = pt.GC_init(stray, TOY, passive_only=True, strict=True)
    unnamed = next(iter(gc.region_sections["not_in_the_spec"][0]))
    assert unnamed.pas.g == pytest.approx(1e-5)   # `all`'s value


def test_GC_init_maps_over_a_population(gc_tree):
    models = pt.GC_init([gc_tree, gc_tree], TOY, passive_only=True)
    assert len(models) == 2
    assert all(m.spec.channels == [] for m in models)


def test_the_built_model_is_one_connected_section_tree(gc_tree):
    """A TREES soma is a short chain the root sits inside, so the root
    usually has two children and every trunk section starts there. Left
    alone that gives two electrically isolated cells, and a clamp on one
    sees nothing that happens on the other."""
    gc = pt.GC_init(gc_tree, TOY, passive_only=True)
    parentless = [s for s in gc.sections if s.parentseg() is None]
    assert len(parentless) == 1


def test_a_clamp_at_the_root_soma_is_felt_at_a_distal_tip(gc_tree):
    """The end-to-end consequence of the connectivity fix above."""
    gc = pt.GC_init(gc_tree, TOY, passive_only=True)
    tips = np.flatnonzero(pt.T_tree(gc_tree))
    distal = int(tips[-1])

    t, v = pt.run_current_clamp(
        gc.model, at_node=gc_tree.root, amp=0.1, delay=5, dur=50, tstop=80,
        record_nodes=[gc_tree.root, distal], v_init=-80.0,
    )
    assert v[distal].max() > v[distal][0] + 1.0


def test_a_mechanism_library_loads_only_once_per_process(tmp_path, monkeypatch):
    """`nrn_load_dll` is not idempotent -- a second load aborts with "The user
    defined name already exists: <mechanism>". Comparing two variants of a
    model in one session is the normal thing to do, so the second call has to
    be a no-op rather than a crash."""
    from pynetrees import gc_model

    library = tmp_path / "nrnmech.dll"
    library.write_bytes(b"not really a library")
    monkeypatch.setattr(gc_model, "_LOADED_LIBRARIES", set())

    calls = []

    class FakeH:
        def nrn_load_dll(self, path):
            calls.append(path)
            return 1

    fake = FakeH()
    gc_model._load_mechanisms(fake, library)
    gc_model._load_mechanisms(fake, library)
    gc_model._load_mechanisms(fake, str(library))       # same file, other spelling
    assert len(calls) == 1


def test_a_missing_mechanism_library_says_how_to_build_one(tmp_path):
    from pynetrees import gc_model

    with pytest.raises(FileNotFoundError, match="nrnivmodl"):
        gc_model._load_mechanisms(None, tmp_path / "absent.dll")


# ---------------------------------------------------------------------------
# the bundled model: table, morphology and .mod sources
# ---------------------------------------------------------------------------


def test_the_bundled_table_matches_the_matlab_dump_exactly():
    """`_gc_biophys.py` was transcribed from `gc_mech.json` once. If that dump
    is still around, every one of its 570 parameters must still agree -- a
    hand-edit to the table that drifts from the published model is exactly
    what this catches."""
    from pathlib import Path

    dump = Path(__file__).parent / "Active GC Model" / "gc_mech.json"
    if not dump.exists():
        pytest.skip("the MATLAB dump is not checked in; nothing to compare against")

    reference = json.loads(dump.read_text(encoding="utf-8"))
    assert set(GC_BIOPHYS) == set(reference)
    for region, block in reference.items():
        assert set(GC_BIOPHYS[region]) == set(block), region
        for mech, params in block.items():
            assert set(GC_BIOPHYS[region][mech]) == set(params), f"{region}.{mech}"
            for name, value in params.items():
                assert GC_BIOPHYS[region][mech][name] == float(value), \
                    f"{region}.{mech}.{name}"


def test_gc_spec_with_no_arguments_is_the_published_model():
    spec = gc_spec()
    assert spec.name == "GC_Beining2017"
    assert spec.regions == GC_BIOPHYS
    assert "na8st" in spec.channels and "Kir21" in spec.channels


def test_the_bundled_table_is_not_handed_out_to_be_mutated():
    """`gc_spec()` deep-copies, so scaling a spec cannot rewrite the module
    constant every later call reads."""
    before = GC_BIOPHYS["soma"]["na8st"]["gbar"]
    gc_spec().scaled({"na8st": 0.0})
    gc_spec().regions["soma"]["na8st"]["gbar"] = -1.0
    assert GC_BIOPHYS["soma"]["na8st"]["gbar"] == before


def test_the_bundled_table_already_has_spine_scaling_folded_in():
    """adendMML.pas.cm = 0.9 * (1 + 1.0 * 0.9) = 1.71 gives it away, and the
    flag on the spec has to say so or `spine_scale` would double-apply it."""
    spec = gc_spec()
    assert spec.spine_scaled
    assert spec.regions["adendMML"]["pas"]["cm"] == pytest.approx(
        0.9 * (1 + GC_SPINE_DENSITY["adendMML"] * 0.9)
    )
    with pytest.raises(ValueError, match="already has spine scaling"):
        gc_spec(spine_scale=0.9)


def test_the_bundled_morphologies_carry_the_regions_the_table_names():
    trees = pt.gc_trees()
    assert len(trees) == 8

    named = set(GC_BIOPHYS) - {"all"}
    for tree in trees:
        used = {tree.rnames[r] for r in np.unique(tree.R)}
        unknown = used - named - {"adendOMLout"}
        assert not unknown, f"{tree.name}: no biophysics for {sorted(unknown)}"


def test_gc_tree_picks_one_of_them():
    assert pt.gc_tree().name == "GC1"
    assert pt.gc_tree(3).name == "GC4"


def test_the_mod_sources_cover_every_channel_the_table_names():
    """Without these, GC_init can only report every channel as skipped."""
    available = {path.stem for path in pt.gc_mod_path().glob("*.mod")}
    named = set(gc_spec().channels)
    assert named <= available, f"no .mod for {sorted(named - available)}"


def test_GC_init_maps_over_a_nested_population():
    """The generic sweep in `test_population.py` skips GC_init, because it
    needs a tree whose regions the granule-cell table covers. This is that
    coverage, on trees that qualify."""
    cells = pt.gc_trees()[:2]
    nested = [[cells[0]], [cells[1], cells[0]]]

    models = pt.GC_init(nested, passive_only=True, cut_axon=True, strict=True)
    assert [len(group) for group in models] == [1, 2]
    assert models[1][0].model.tree.n_nodes != models[0][0].model.tree.n_nodes
