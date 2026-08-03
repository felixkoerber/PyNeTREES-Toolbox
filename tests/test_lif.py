"""Tests for pytrees.electrotonics.LIF_tree / AdExLIF_tree.

The passive (sub-threshold, thr set unreachably high) case is cross-checked
against an independent dense implicit-Euler stepper written directly in the
test (not reusing LIF_tree's own loop) -- this verifies the time-stepping
*mechanics* (capacitive term, backward-Euler solve) independently of
M_tree's own correctness (already covered by test_electrotonics.py). The
supra-threshold cases check the two functions' genuinely different reset
policies (LIF_tree: hard reset everywhere, or optional distance-weighted
partial reset; AdExLIF_tree: hard clip + adaptation increment).
"""

import numpy as np
import pytest
from scipy import sparse

from pytrees import AdExLIF_tree, LIF_tree, M_tree, Tree, surf_tree


def _cable_tree() -> Tree:
    dA = sparse.csr_matrix(([1], ([1], [0])), shape=(2, 2))
    tree = Tree(
        dA=dA,
        X=np.array([0.0, 100.0]),
        Y=np.zeros(2),
        Z=np.zeros(2),
        D=np.array([10.0, 2.0]),
        R=np.zeros(2, dtype=int),
        rnames=["dend"],
    )
    tree.Ri = 100.0
    tree.Gm = 1.0 / 2500.0
    tree.Cm = 1.0
    return tree


def _chain_tree() -> Tree:
    # root -> mid -> leaf, well-separated path lengths so Pvec_tree
    # (cumulative path length) differs meaningfully node to node
    dA = sparse.csr_matrix(([1, 1], ([1, 2], [0, 1])), shape=(3, 3))
    tree = Tree(
        dA=dA,
        X=np.array([0.0, 100.0, 500.0]),
        Y=np.zeros(3),
        Z=np.zeros(3),
        D=np.array([10.0, 2.0, 2.0]),
        R=np.zeros(3, dtype=int),
        rnames=["dend"],
    )
    tree.Ri = 100.0
    tree.Gm = 1.0 / 2500.0
    tree.Cm = 1.0
    return tree


def test_LIF_tree_requires_Cm():
    tree = _cable_tree()
    tree.Cm = None
    with pytest.raises(ValueError, match="Cm"):
        LIF_tree(tree, time=np.array([0.0, 0.1]))


def test_LIF_tree_passive_subthreshold_matches_independent_stepper():
    tree = _cable_tree()
    time = np.array([0.0, 0.1, 0.2, 0.3])
    T = time.size
    N = tree.n_nodes
    I = np.zeros((N, T))
    I[1, :] = 0.05  # constant current injection at the leaf

    v, sp = LIF_tree(tree, time=time, I=I, thr=1e9)
    assert sp.size == 0  # threshold unreachable -> no spikes

    # independent dense backward-Euler cross-check
    dt = (time[1] - time[0]) / 1000.0
    M0 = M_tree(tree).toarray()
    Mcm_vec = (surf_tree(tree) / 1e8) * tree.Cm / dt
    M = M0 + np.diag(Mcm_vec)
    v_expected = np.zeros((N, T))
    for k in range(T - 1):
        rhs = I[:, k] + v_expected[:, k] * Mcm_vec
        v_expected[:, k + 1] = np.linalg.solve(M, rhs)

    np.testing.assert_allclose(v, v_expected, rtol=1e-6)


def test_LIF_tree_spike_resets_every_node_when_not_partial():
    tree = _cable_tree()
    # a single-step current pulse (not a sustained drive): guarantees exactly
    # one isolated spike at a known step (k=0), so the post-reset column
    # (k=1) can be checked directly. A sustained huge current instead causes
    # back-to-back spikes every step, and the "next" column is immediately
    # overwritten by the *following* spike's marker before it can be read.
    I = np.zeros((tree.n_nodes, 51))
    I[0, 0] = 5000.0
    v, sp = LIF_tree(
        tree, time=np.linspace(0.0, 5.0, 51), I=I, iroot=0, thr=1.0, vreset=0.0
    )
    assert sp.size == 1
    np.testing.assert_allclose(v[:, 1], 0.0, atol=1e-9)


def test_LIF_tree_partial_reset_differs_by_path_distance():
    tree = _chain_tree()
    N = tree.n_nodes
    time = np.linspace(0.0, 5.0, 51)
    I = np.zeros((N, time.size))
    I[0, 0] = 5000.0  # single-step pulse -> one isolated, deterministically-timed spike

    v, sp = LIF_tree(
        tree, time=time, I=I, iroot=0, thr=1.0, vreset=0.0, partial_reset=True
    )
    # the pulse only ever occurs at step 0, so the *first* spike (whichever
    # step it lands on) always resets column 1 -- a partial reset isn't
    # guaranteed to land below threshold in one shot (unlike the full-reset
    # case above), so more than one spike can follow as it settles; that
    # doesn't affect what column 1 itself should look like
    assert sp.size >= 1
    # mid (closer to root) should be reset closer to vreset than the distal leaf
    post_reset = v[:, 1]
    assert abs(post_reset[1] - 0.0) <= abs(post_reset[2] - 0.0) + 1e-9


def test_AdExLIF_tree_requires_Cm():
    tree = _cable_tree()
    tree.Cm = None
    with pytest.raises(ValueError, match="Cm"):
        AdExLIF_tree(tree, time=np.array([0.0, 0.1]))


def test_AdExLIF_tree_shapes_and_no_spike_when_undriven():
    # AdEx's exponential term (`DeltaT * exp((v - Vt) / DeltaT)`) is active
    # even exactly at rest (a small but nonzero value at v=0, Vt=10) -- so
    # voltage settles to a small nonzero equilibrium with zero external
    # drive, it does not stay pinned at exactly 0 like LIF_tree's plain
    # leaky integrator does. That's the real, intended model behavior, not
    # a bug -- this test only checks it stays small and never spikes.
    tree = _cable_tree()
    time = np.linspace(0.0, 5.0, 51)
    v, sp, w = AdExLIF_tree(tree, time=time)
    assert v.shape == (tree.n_nodes, time.size)
    assert w.shape == (tree.n_nodes, time.size)
    assert sp.size == 0
    assert np.all(np.abs(v) < 1.0)  # nowhere near thr=80


def test_AdExLIF_tree_spikes_and_adapts_when_driven():
    tree = _cable_tree()
    time = np.linspace(0.0, 20.0, 201)
    I = np.zeros((tree.n_nodes, time.size))
    I[0, :] = 5000.0
    v, sp, w = AdExLIF_tree(tree, time=time, I=I, a=0.1, b=1e-3)
    assert sp.size > 0
    # adaptation should have accumulated something by the end
    assert w[:, -1].sum() > 0.0
