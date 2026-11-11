"""`MST_tree` against MATLAB's own output, pinned from R2026a.

Regenerate `tests/fixtures/mst_matlab.mat` with
`scripts/matlab/run_mst_validation.m`. Every case compares the parent of every
point, so a single different connection fails.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.io import loadmat

import pynetrees as pt

REFERENCE = loadmat(Path(__file__).parent / "fixtures" / "mst_matlab.mat")
CLOUDS = (1, 2, 3)


def _cloud(s):
    P = REFERENCE[f"P_{s}"]
    return P[:, 0], P[:, 1], P[:, 2]


def _expected(case, s):
    return REFERENCE[f"par_{case}_{s}"].ravel().astype(int)


@pytest.mark.parametrize("s", CLOUDS)
@pytest.mark.parametrize("case, options", [
    ("ctrl", dict(bf=0.4)),
    ("b", dict(bf=0.4, avoid_multifurcations=True)),
    ("m02", dict(bf=0.2, mplen=60.0)),
    ("m07", dict(bf=0.7, mplen=60.0)),
])
def test_single_start_matches_matlab(case, options, s):
    result = pt.MST_tree(*_cloud(s), start=0, thr=50.0, record=True, full_output=True, **options)
    parents = np.zeros(200, dtype=int)
    parents[0] = -1
    for _, point, parent in result.history:
        parents[point] = parent + 1
    np.testing.assert_array_equal(parents, _expected(case, s))


@pytest.mark.parametrize("s", CLOUDS)
@pytest.mark.parametrize("case, options", [
    ("sd", dict()),
    ("sb", dict(avoid_multifurcations=True)),
    ("sm", dict(mplen=90.0)),
])
def test_growth_onto_a_seed_matches_matlab(case, options, s):
    A = REFERENCE[f"seed_{s}"]
    n = len(A)
    children = np.flatnonzero(A[:, 3] > 0)
    dA = sparse.coo_matrix((np.ones(len(children)), (children, A[children, 3].astype(int) - 1)),
                           shape=(n, n))
    seed = pt.Tree(dA, A[:, 0], A[:, 1], A[:, 2], np.ones(n), np.zeros(n, int), ["tree"])

    X, Y, Z = _cloud(s)
    result = pt.MST_tree(X[30:], Y[30:], Z[30:], start=seed, bf=0.4, thr=50.0, record=True,
                         full_output=True, **options)
    parents = np.zeros(170, dtype=int)
    for _, point, parent in result.history:
        parents[point - n] = -(parent + 1) if parent < n else parent - n + 1
    np.testing.assert_array_equal(parents, _expected(case, s))


@pytest.mark.parametrize("s", CLOUDS)
@pytest.mark.parametrize("case, options", [("mt", dict()), ("mb", dict(avoid_multifurcations=True))])
def test_several_start_points_match_matlab(case, options, s):
    result = pt.MST_tree(*_cloud(s), start=[0, 1, 2], bf=0.4, thr=50.0, record=True,
                         full_output=True, **options)
    parents = np.zeros(197, dtype=int)
    for tree, point, parent in result.history:
        parents[point - 3] = -(tree + 1) if parent < 3 else parent - 2
    np.testing.assert_array_equal(parents, _expected(case, s))
