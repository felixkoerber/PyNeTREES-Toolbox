"""Tests for pynetrees.neuron_bridge: Tree -> real NEURON simulation.

Requires the `neuron` package (see pyproject.toml's `neuron` extra / the
module docstring for why Windows needs the official binary installer
instead of a pip wheel) -- skipped entirely if it isn't importable, same
pattern as pyvista/matplotlib in test_plotting.py.

The core correctness check (`test_...matches_sse_tree_for_uniform_diameter`)
cross-validates two *independent* implementations of the same passive
cable equations: NEURON itself (a real, widely-used, industry-standard
simulator) against `sse_tree`'s exact linear-algebra steady-state solve
(Phase 8). They're built on genuinely different discretizations (NEURON's
continuous pt3d-interpolated frusta vs `sse_tree`'s one-uniform-cylinder-
per-node model), so exact agreement isn't expected in general -- but for a
tree with *uniform* diameter (removing the taper-vs-uniform-cylinder
discrepancy), both discretizations describe the same physical cable, and
the two independently-implemented solvers should agree closely. See
PORT_STATUS.md Design Decision #36 for the fuller explanation of a real,
larger (~24%) mismatch found and understood -- not fixed, because there
was nothing to fix -- while first writing this test with a tapered tree.
"""

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip("neuron")

from pynetrees import (
    Tree,
    build_neuron_model,
    insert_mechanism,
    run_current_clamp,
    sse_tree,
)


def _branchy_tree(uniform_diam: bool = True) -> Tree:
    # root(0) -> mid(1) -> {leafA(2), leafB(3)}, symmetric leaves
    dA = sparse.csr_matrix(([1, 1, 1], ([1, 2, 3], [0, 1, 1])), shape=(4, 4))
    D = np.array([4.0, 4.0, 4.0, 4.0]) if uniform_diam else np.array([10.0, 4.0, 2.0, 2.0])
    tree = Tree(
        dA=dA,
        X=np.array([0.0, 100.0, 200.0, 200.0]),
        Y=np.array([0.0, 0.0, 50.0, -50.0]),
        Z=np.zeros(4),
        D=D,
        R=np.zeros(4, dtype=int),
        rnames=["dend"],
    )
    tree.Ri = 100.0
    tree.Gm = 1.0 / 15000.0
    tree.Cm = 1.0
    return tree


def _cable_tree() -> Tree:
    dA = sparse.csr_matrix(([1], ([1], [0])), shape=(2, 2))
    tree = Tree(
        dA=dA,
        X=np.array([0.0, 100.0]),
        Y=np.zeros(2),
        Z=np.zeros(2),
        D=np.array([4.0, 4.0]),
        R=np.zeros(2, dtype=int),
        rnames=["dend"],
    )
    tree.Ri = 100.0
    tree.Gm = 1.0 / 15000.0
    tree.Cm = 1.0
    return tree


def test_build_neuron_model_requires_Ri_Gm_Cm():
    tree = _cable_tree()
    tree.Cm = None
    with pytest.raises(ValueError, match="Cm"):
        build_neuron_model(tree)


def test_build_neuron_model_section_count_and_geometry():
    tree = _branchy_tree()
    model = build_neuron_model(tree)
    # one section per dissected branch: root->mid, mid->leafA, mid->leafB
    assert len(model.sections) == 3
    lengths = sorted(sec.L for sec in model.sections)
    assert lengths[0] == pytest.approx(100.0, rel=1e-3)
    assert lengths[1] == pytest.approx(np.hypot(100.0, 50.0), rel=1e-3)
    assert lengths[2] == pytest.approx(np.hypot(100.0, 50.0), rel=1e-3)


def test_build_neuron_model_topology_is_connected_tree():
    tree = _branchy_tree()
    model = build_neuron_model(tree)
    from neuron import h

    # exactly one section (the root's) should have no parent
    n_roots = sum(1 for sec in model.sections if not h.SectionRef(sec=sec).has_parent())
    assert n_roots == 1


def test_run_current_clamp_zero_amp_stays_at_rest():
    tree = _cable_tree()
    model = build_neuron_model(tree)
    t, v = run_current_clamp(model, at_node=0, amp=0.0, delay=5, dur=10, tstop=20)
    np.testing.assert_allclose(v[0], -70.0, atol=1e-6)


def test_run_current_clamp_depolarizes_toward_steady_state():
    tree = _cable_tree()
    model = build_neuron_model(tree)
    t, v = run_current_clamp(
        model, at_node=0, amp=0.05, delay=5, dur=1000, tstop=1000, record_nodes=[0, 1]
    )
    assert v[0][-1] > -70.0  # depolarized from rest
    assert v[0][-1] > v[1][-1]  # more depolarized nearer the injection site


def test_run_current_clamp_matches_sse_tree_for_uniform_diameter():
    tree = _branchy_tree(uniform_diam=True)
    model = build_neuron_model(tree)
    amp = 0.01
    _, v = run_current_clamp(
        model, at_node=0, amp=amp, delay=5, dur=3000, tstop=3000,
        record_nodes=[0, 1, 2, 3],
    )
    steady = {node: v[node][-1] - (-70.0) for node in [0, 1, 2, 3]}

    plain_tree = Tree(
        dA=tree.dA, X=tree.X, Y=tree.Y, Z=tree.Z, D=tree.D, R=tree.R, rnames=tree.rnames
    )
    plain_tree.Ri = tree.Ri
    plain_tree.Gm = tree.Gm
    expected = sse_tree(plain_tree, I=0) * amp

    for node in [0, 1, 2, 3]:
        assert steady[node] == pytest.approx(expected[node], rel=0.03)


def test_run_current_clamp_tapered_tree_still_orders_correctly():
    # with real diameter tapering, exact match against sse_tree isn't
    # expected (see module docstring) -- but the qualitative physics must
    # still hold: depolarization decreases with distance from the
    # injection site, and the two symmetric leaves match each other
    tree = _branchy_tree(uniform_diam=False)
    model = build_neuron_model(tree)
    _, v = run_current_clamp(
        model, at_node=0, amp=0.01, delay=5, dur=2000, tstop=2000,
        record_nodes=[0, 1, 2, 3],
    )
    steady = {node: v[node][-1] for node in [0, 1, 2, 3]}
    assert steady[0] >= steady[1] >= steady[2]
    assert steady[2] == pytest.approx(steady[3], rel=1e-6)


def test_insert_mechanism_hh_produces_a_spike():
    tree = _cable_tree()
    model = build_neuron_model(tree)
    insert_mechanism(model, "hh")
    t, v = run_current_clamp(
        model, at_node=0, amp=0.3, delay=5, dur=5, tstop=25, v_init=-65.0
    )
    assert v[0].max() > 0.0  # a real action potential overshoots 0 mV


def test_insert_mechanism_region_filter():
    # soma needs *more than one* node before the region transition to get
    # its own section: a region occupying only the root node itself gets
    # absorbed into the following section rather than split off on its
    # own (see build_neuron_model's docstring / PORT_STATUS.md Design
    # Decision #36) -- a real biophysical soma is modeled with more than
    # a single point anyway, so this is a reasonable modeling constraint,
    # not a limitation worth working around.
    dA = sparse.csr_matrix(([1, 1, 1], ([1, 2, 3], [0, 1, 2])), shape=(4, 4))
    tree = Tree(
        dA=dA,
        X=np.array([0.0, 10.0, 20.0, 30.0]),
        Y=np.zeros(4),
        Z=np.zeros(4),
        D=np.array([10.0, 10.0, 4.0, 4.0]),
        R=np.array([0, 0, 1, 1]),
        rnames=["soma", "dend"],
    )
    tree.Ri = 100.0
    tree.Gm = 1.0 / 15000.0
    tree.Cm = 1.0
    model = build_neuron_model(tree)
    insert_mechanism(model, "hh", region="soma", gnabar=0.12)

    soma_secs = model.region_sections["soma"]
    dend_secs = model.region_sections["dend"]
    assert all(hasattr(seg, "hh") for sec in soma_secs for seg in sec)
    assert all(not hasattr(seg, "hh") for sec in dend_secs for seg in sec)
