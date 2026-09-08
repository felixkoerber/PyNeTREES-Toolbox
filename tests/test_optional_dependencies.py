"""Tests for pynetrees._optional: saying up front what an install cannot do.

Lazy imports keep `import pynetrees` cheap, at the cost of a missing
dependency surfacing as an ImportError deep inside the first call that
needed it -- named after the transitive dependency rather than the feature.
These cover the check that answers the question at import instead.
"""

import warnings

import pytest

import pynetrees as pt
from pynetrees import _optional


def test_every_feature_names_an_extra_that_pyproject_actually_offers():
    """A warning telling the reader to `pip install pynetrees[typo]` is
    worse than no warning, so the table is checked against pyproject."""
    import re
    from pathlib import Path

    text = (Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    block = text.split("[project.optional-dependencies]")[1].split("\n[")[0]
    declared = set(re.findall(r"^([A-Za-z0-9_-]+)\s*=", block, re.MULTILINE))

    named = {extra for extra, _, _ in _optional.OPTIONAL_FEATURES.values()}
    assert named <= declared, f"not in pyproject: {sorted(named - declared)}"


def test_missing_dependencies_reports_only_what_is_absent():
    missing = pt.missing_dependencies()
    for feature, (extra, modules, effect) in missing.items():
        assert modules, f"{feature} listed with nothing missing"
        for module in modules:
            assert not _optional._installed(module)
        assert extra and effect


def test_an_installed_feature_is_not_reported():
    """numpy is a hard dependency, so nothing may claim it is missing."""
    absent = {m for _, mods, _ in pt.missing_dependencies().values() for m in mods}
    assert "numpy" not in absent
    assert "scipy" not in absent


def test_the_warning_names_the_feature_the_module_and_the_fix(monkeypatch):
    monkeypatch.delenv("PYNETREES_NO_DEPENDENCY_WARNING", raising=False)
    monkeypatch.setattr(
        _optional, "OPTIONAL_FEATURES",
        {"time travel": ("delorean", ("flux_capacitor",), "going back to 1955")},
    )
    with pytest.warns(UserWarning) as record:
        _optional.warn_about_missing()

    message = str(record[0].message)
    assert "time travel" in message
    assert "flux_capacitor" in message
    assert "pynetrees[delorean]" in message
    assert "going back to 1955" in message


def test_it_is_one_warning_and_not_one_per_feature(monkeypatch):
    """A stack of near-identical warnings gets filtered out wholesale,
    taking the line that mattered with it."""
    monkeypatch.delenv("PYNETREES_NO_DEPENDENCY_WARNING", raising=False)
    monkeypatch.setattr(
        _optional, "OPTIONAL_FEATURES",
        {f"feature {i}": (f"extra{i}", (f"nope{i}",), "nothing")
         for i in range(5)},
    )
    with pytest.warns(UserWarning) as record:
        _optional.warn_about_missing()
    assert len(record) == 1
    assert str(record[0].message).count("feature ") == 5


def test_the_environment_variable_silences_it(monkeypatch):
    monkeypatch.setenv("PYNETREES_NO_DEPENDENCY_WARNING", "1")
    monkeypatch.setattr(
        _optional, "OPTIONAL_FEATURES",
        {"anything": ("extra", ("definitely_not_installed",), "nothing")},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _optional.warn_about_missing()
    assert caught == []


def test_a_complete_install_says_nothing(monkeypatch):
    monkeypatch.delenv("PYNETREES_NO_DEPENDENCY_WARNING", raising=False)
    monkeypatch.setattr(_optional, "OPTIONAL_FEATURES", {})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _optional.warn_about_missing()
    assert caught == []


def test_checking_does_not_import_the_module_it_checks():
    """`find_spec`, not `import` -- importing bpy alone starts a Blender
    session, which is the cost this whole module exists to avoid."""
    import sys

    before = set(sys.modules)
    pt.missing_dependencies()
    new = set(sys.modules) - before
    assert not {m for m in new if m.split(".")[0] in ("bpy", "neuron", "pyvista")}
