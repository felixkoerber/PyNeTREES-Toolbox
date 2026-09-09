"""Tests for `save_model_hoc`: a built NEURON model written back out as hoc.

`save_hoc` writes a morphology and is tested in `test_io_export.py` by
reading the text. This writes a *model* -- segment counts, passive
properties, every inserted mechanism and its current values -- so the test
that matters is whether loading the file reproduces the model, not whether
the text looks right.

That check runs in a **subprocess**. NEURON is one global session per
process: sections built by other tests are still in `h.allsec()`, and
deleting them to get a clean slate would break whichever test runs next.
A fresh interpreter is the honest isolation, and it is also exactly what a
collaborator receiving the file would do.
"""

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import pynetrees as pt

pytest.importorskip("neuron")


@pytest.fixture
def model():
    tree = pt.repair_tree(pt.sample_tree())
    tree.Ri, tree.Gm, tree.Cm = 100.0, 1 / 2500.0, 1.0
    built = pt.build_neuron_model(tree)
    # a gradient, so the export has something that cannot collapse to one
    # section-wide value
    pt.insert_mechanism(built, "hh",
                        gnabar=lambda x: 0.12 * np.exp(-x / 120.0))
    return built


def _reload_in_a_fresh_neuron(path):
    """Load `path` in a new interpreter and report what NEURON ended up with."""
    script = textwrap.dedent(f"""
        import json
        from neuron import h
        h.load_file({str(path).replace(chr(92), '/')!r})
        sections = list(h.allsec())
        print("RESULT" + json.dumps({{
            "nsec": len(sections),
            "nseg": sum(s.nseg for s in sections),
            "roots": sum(1 for s in sections if s.parentseg() is None),
            "L": sorted(round(s.L, 6) for s in sections),
            "Ra": sorted(round(s.Ra, 6) for s in sections),
            "cm": sorted(round(sg.cm, 9) for s in sections for sg in s),
            "gnabar": sorted(round(sg.hh.gnabar, 9)
                             for s in sections for sg in s),
            "g_pas": sorted(round(sg.pas.g, 12) for s in sections for sg in s),
        }}))
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    line = next(l for l in completed.stdout.splitlines() if l.startswith("RESULT"))
    return json.loads(line[len("RESULT"):])


def _live(model):
    sections = list(model.sections)
    return {
        "nsec": len(sections),
        "nseg": sum(s.nseg for s in sections),
        "roots": sum(1 for s in sections if s.parentseg() is None),
        "L": sorted(round(s.L, 6) for s in sections),
        "Ra": sorted(round(s.Ra, 6) for s in sections),
        "cm": sorted(round(sg.cm, 9) for s in sections for sg in s),
        "gnabar": sorted(round(sg.hh.gnabar, 9) for s in sections for sg in s),
        "g_pas": sorted(round(sg.pas.g, 12) for s in sections for sg in s),
    }


def test_the_exported_hoc_rebuilds_the_same_model(tmp_path, model):
    """The whole point: geometry, topology, segment counts, passive
    properties and every mechanism value come back identically."""
    path = pt.save_model_hoc(model, tmp_path / "mycell")
    assert _reload_in_a_fresh_neuron(path) == _live(model)


def test_a_density_gradient_survives_the_round_trip(tmp_path, model):
    """A value that varies along the cell is the case a spec-based export
    would lose -- this reads the live sections, so it cannot."""
    reloaded = _reload_in_a_fresh_neuron(
        pt.save_model_hoc(model, tmp_path / "mycell")
    )
    gnabar = reloaded["gnabar"]
    assert len(set(gnabar)) > 1, "the fixture's gradient should not be uniform"
    assert gnabar == _live(model)["gnabar"]


def test_the_reloaded_cell_is_one_connected_tree(tmp_path, model):
    assert _reload_in_a_fresh_neuron(
        pt.save_model_hoc(model, tmp_path / "mycell")
    )["roots"] == 1


# ---------------------------------------------------------------------------
# the text itself, where that is the cheaper check
# ---------------------------------------------------------------------------


def test_one_create_and_one_insert_per_section(tmp_path, model):
    lines = pt.save_model_hoc(
        model, tmp_path / "c"
    ).read_text(encoding="utf-8").splitlines()

    n = len(model.sections)
    assert sum(1 for l in lines if l.startswith("create ")) == n
    assert sum(1 for l in lines if "insert hh" in l) == n
    assert sum(1 for l in lines if "insert pas" in l) == n
    # every section but the root attaches to exactly one parent
    assert sum(1 for l in lines if "connect " in l) == n - 1


def test_states_and_assigned_variables_are_not_written(tmp_path, model):
    """psection() reports a mechanism's STATEs (m, h, n) and ASSIGNEDs
    (ina, gk) alongside its PARAMETERs. Writing those would be writing down
    the middle of a simulation and calling it a model."""
    text = pt.save_model_hoc(model, tmp_path / "c").read_text(encoding="utf-8")
    for assigned in ("m_hh =", "h_hh =", "n_hh =", "ina_hh", "gk_hh", "i_pas"):
        assert assigned not in text
    for parameter in ("gnabar_hh", "gkbar_hh", "gl_hh", "el_hh", "g_pas", "e_pas"):
        assert parameter in text


def test_a_uniform_parameter_collapses_to_one_line_per_section(tmp_path, model):
    """Most of a model is uniform; writing it per segment would bury the
    part that is not."""
    text = pt.save_model_hoc(model, tmp_path / "c").read_text(encoding="utf-8")
    # gkbar is uniform, so one plain assignment per section and no `(x)` form
    assert text.count("gkbar_hh = ") == len(model.sections)
    assert "gkbar_hh(" not in text


def test_stimuli_are_written_at_the_right_place(tmp_path, model):
    soma = int(model.tree.root)
    text = pt.save_model_hoc(
        model, tmp_path / "c",
        stimuli=[("IClamp", soma, {"delay": 5, "dur": 50, "amp": 2.0})],
    ).read_text(encoding="utf-8")

    assert "new IClamp(" in text
    assert "stim[0].amp = 2" in text
    assert "stim[0].delay = 5" in text


def test_a_stimulus_on_a_node_that_is_not_on_the_model_is_refused(tmp_path, model):
    with pytest.raises(KeyError, match="not on this model"):
        pt.save_model_hoc(model, tmp_path / "c",
                          stimuli=[("IClamp", 10**6, {"amp": 1.0})])


def test_run_conditions_travel_with_the_model(tmp_path, model):
    """celsius changes every rate constant, so leaving it in whoever's
    memory silently changes the answer."""
    text = pt.save_model_hoc(
        model, tmp_path / "c", v_init=-65.0, tstop=80.0, dt=0.025
    ).read_text(encoding="utf-8")
    assert "celsius = " in text
    assert "v_init = -65" in text
    assert "tstop = 80" in text
    assert "dt = 0.025" in text


def test_it_says_what_it_could_not_export(tmp_path, model):
    """Better than quietly producing a model that runs and is not yours."""
    text = pt.save_model_hoc(model, tmp_path / "c").read_text(encoding="utf-8")
    assert "Not exported" in text
    assert "NetStim" in text and "recording vectors" in text


def test_a_mechanism_library_is_loaded_first(tmp_path, model):
    library = tmp_path / "nrnmech.dll"
    library.write_bytes(b"")
    text = pt.save_model_hoc(
        model, tmp_path / "c", mech_library=library
    ).read_text(encoding="utf-8")
    assert text.index("nrn_load_dll") < text.index("create ")


def test_the_extension_is_added(tmp_path, model):
    assert pt.save_model_hoc(model, tmp_path / "c").name == "c.hoc"
