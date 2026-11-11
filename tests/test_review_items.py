"""The items closed from the code-review list, one behaviour per test."""

import numpy as np
import pytest
from scipy import sparse

import pynetrees as pt
from pynetrees.io import _matlab_v73, mtr, native


@pytest.fixture(scope="module")
def tree():
    return pt.sample_tree()


@pytest.fixture(scope="module")
def empty(tree):
    return pt.delete_tree(tree, np.arange(tree.n_nodes))


def _shape(value):
    return [_shape(v) for v in value] if isinstance(value, list) else value.n_nodes


# ------------------------------ MST_steps ------------------------------


def _run(n=40, **options):
    rng = np.random.default_rng(0)
    P = rng.uniform(0, 100, (n, 3))
    return pt.MST_tree(P[:, 0], P[:, 1], P[:, 2], thr=80.0, record=True, full_output=True, **options)


def test_mst_steps_starts_at_the_seed_and_ends_at_the_result():
    result = _run()
    states = pt.MST_steps(result)
    assert states[0].n_nodes == 1
    assert states[-1].n_nodes == result.trees.n_nodes
    assert (states[-1].dA != result.trees.dA).nnz == 0
    assert [s.n_nodes for s in states] == list(range(1, result.trees.n_nodes + 1))


def test_mst_steps_picks_steps_and_counts_from_the_end():
    result = _run()
    five, last = pt.MST_steps(result, [5, -1])
    assert five.n_nodes == 6 and last.n_nodes == result.trees.n_nodes
    with pytest.raises(ValueError, match="outside"):
        pt.MST_steps(result, 10_000)


def test_mst_steps_with_several_trees_gives_one_state_per_tree():
    result = _run(start=[0, 1])
    state = pt.MST_steps(result, 3)[0]
    assert len(state) == 2 and sum(t.n_nodes for t in state) == 5


def test_mst_steps_needs_a_record():
    with pytest.raises(ValueError, match="record=True"):
        pt.MST_steps(pt.MST_tree(*np.random.default_rng(0).uniform(0, 9, (3, 4)), full_output=True))


def test_strict_mplen_never_exceeds_the_limit():
    result = _run(n=120, mplen=40.0, strict_mplen=True)
    assert pt.Pvec_tree(result.trees, pt.len_tree(result.trees)).max() <= 40.0 + 1e-9


# ------------------------------ bf_tree ------------------------------


def test_bf_tree_survives_a_negative_fitted_bias():
    with pytest.warns(UserWarning, match="usual range"):
        bf, k = pt.bf_tree(np.linspace(0.0, np.pi, 200), dim=3)
    assert k < 0 and bf == 0.0


# ------------------------------ MATLAB files ------------------------------


@pytest.mark.parametrize("version", ["5", "7.3"])
def test_mtr_keeps_nesting_and_empty_slots(tmp_path, tree, empty, version):
    population = [[tree, [tree]], empty, [], [tree]]
    back = pt.load_mtr(pt.save_mtr(population, tmp_path / "pop.mtr", version=version))
    assert _shape(back) == [[197, [197]], 0, [], [197]]


def test_a_cell_of_one_tree_stays_a_list(tmp_path, tree):
    assert isinstance(pt.load_mtr(pt.save_mtr([tree], tmp_path / "one.mtr")), list)


def test_v73_round_trips_every_field(tmp_path, tree):
    labelled = tree.copy(Ri=100.0, Gm=1 / 20000, frustum=True,
                         meta={"label": "abc", "vals": np.arange(tree.n_nodes, dtype=float)})
    path = pt.save_mtr(labelled, tmp_path / "t.mtr", version="7.3")
    assert path.read_bytes().startswith(b"MATLAB 7.3 MAT-file")
    back = pt.load_tree(path)
    np.testing.assert_allclose(back.X, tree.X)
    np.testing.assert_array_equal(back.R, tree.R)
    assert (back.dA != tree.dA).nnz == 0
    assert back.rnames == tree.rnames and back.Ri == 100.0 and back.frustum
    assert back.meta["label"] == "abc"
    np.testing.assert_allclose(back.meta["vals"], labelled.meta["vals"])


def test_large_data_switches_to_v73_automatically(tmp_path, tree, monkeypatch):
    monkeypatch.setattr(mtr, "_V5_LIMIT", 10)
    path = pt.save_mtr(tree, tmp_path / "big.mtr")
    assert path.read_bytes().startswith(b"MATLAB 7.3")


def test_save_mtr_rejects_an_unknown_version(tmp_path, tree):
    with pytest.raises(ValueError, match="version"):
        pt.save_mtr(tree, tmp_path / "x.mtr", version="6")


def test_v73_sparse_shape_survives(tmp_path):
    path = tmp_path / "s.mat"
    _matlab_v73.write_v73(path, "m", sparse.csr_matrix(np.diag([1.0, 0.0, 3.0])))
    read = _matlab_v73.read_v73(path)["m"]
    assert read.shape == (3, 3) and read.nnz == 2


# ------------------------------ load_tree / save_tree ------------------------------


def test_swc_is_repaired_by_default_and_not_on_request(tmp_path, tree):
    path = pt.save_tree(tree, tmp_path / "t.swc")
    assert pt.load_tree(path, repair=False).n_nodes == tree.n_nodes
    assert pt.load_swc(path).n_nodes == pt.repair_tree(pt.load_swc(path, repair=False)).n_nodes


def test_load_dialog_loads_every_selected_file(tmp_path, tree, monkeypatch):
    a, b = (pt.save_tree(tree, tmp_path / name) for name in ("a.npz", "b.npz"))
    monkeypatch.setattr(native, "_choose_files_to_load", lambda: [str(a), str(b)])
    assert len(pt.load_tree()) == 2
    monkeypatch.setattr(native, "_choose_files_to_load", lambda: [str(a)])
    assert isinstance(pt.load_tree(), pt.Tree)


def test_cancelling_a_dialog_returns_none(tree, monkeypatch):
    monkeypatch.setattr(native, "_choose_files_to_load", lambda: [])
    monkeypatch.setattr(native, "_choose_file_to_save", lambda: "")
    assert pt.load_tree() is None and pt.save_tree(tree) is None


def test_save_dialog_writes_where_it_is_told(tmp_path, tree, monkeypatch):
    target = tmp_path / "chosen.npz"
    monkeypatch.setattr(native, "_choose_file_to_save", lambda: str(target))
    assert pt.save_tree(tree) == target and target.exists()


# ------------------------------ geometry ------------------------------


def test_cyl_tree_adjacency_form_holds_the_same_coordinates(tree):
    X1, X2, Y1, Y2, Z1, Z2 = pt.cyl_tree(tree, adjacency=True)
    plain = pt.cyl_tree(tree)
    assert sparse.issparse(X1) and X1.shape == tree.dA.shape
    children, parents = tree.dA.nonzero()
    np.testing.assert_allclose(np.asarray(X1[children, parents]).ravel(), tree.X[parents])
    np.testing.assert_allclose(np.asarray(X2[children, parents]).ravel(), plain[1][children])


def test_boundary_fills_an_enclosed_void():
    rng = np.random.default_rng(8)
    direction = rng.normal(size=(1200, 2))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    ring = direction * rng.uniform(40, 50, (1200, 1))
    cell = pt.Tree(sparse.csr_matrix((1200, 1200)), ring[:, 0], ring[:, 1], np.zeros(1200),
                   np.ones(1200), np.zeros(1200, int), ["t"])
    area = pt.boundary_tree(cell, 1.0, dim=2).volume
    assert area == pytest.approx(np.pi * 50 ** 2, rel=0.05)    # a disc, not a ring


def test_boundary_reduce_halves_the_faces_and_keeps_the_volume(tree):
    pytest.importorskip("pyvista")
    full = pt.boundary_tree(tree, 0.5)
    reduced = pt.boundary_tree(tree, 0.5, reduce=0.5)
    assert len(reduced.faces) == pytest.approx(len(full.faces) / 2, rel=0.1)
    assert reduced.volume == full.volume
    with pytest.raises(ValueError, match="3D"):
        pt.boundary_tree(tree, 0.5, dim=2, reduce=0.5)


# ------------------------------ plotting ------------------------------


def _pv():
    return pytest.importorskip("pyvista")


def test_plot_tree_has_no_lighting_by_default(tree):
    _pv()
    plotter = pt.plot_tree(tree)
    assert not next(iter(plotter.renderer.actors.values())).prop.lighting
    lit = pt.plot_tree(tree, lighting=True)
    assert next(iter(lit.renderer.actors.values())).prop.lighting


@pytest.mark.parametrize("plane, flat", [(1, None), (2, None), (3, None)])
def test_blatt_draws_one_strip_per_segment(tree, plane, flat):
    _pv()
    plotter = pt.plot_tree(tree, mode="blatt", blatt=plane, color=pt.BO_tree(tree))
    mesh = next(iter(plotter.renderer.actors.values())).mapper.dataset
    assert mesh.n_cells == tree.n_nodes - 1
    assert mesh.n_points == 4 * (tree.n_nodes - 1)


def test_blatt_plane_is_validated(tree):
    _pv()
    with pytest.raises(ValueError, match="blatt"):
        pt.plot_tree(tree, mode="blatt", blatt=4)


def test_quiver_draws_arrows(tree):
    _pv()
    plotter = pt.plot_tree(tree, mode="quiver", dim=2)
    mesh = next(iter(plotter.renderer.actors.values())).mapper.dataset
    assert mesh.n_points > tree.n_nodes
    assert np.isclose(mesh.points[:, 2].max() - mesh.points[:, 2].min(), 0.0) or mesh.n_cells > 0


def test_plot_tree_takes_any_nesting(tree):
    _pv()
    plotter = pt.plot_tree([[tree, [tree]], tree], color="red")
    assert len(plotter.renderer.actors) == 3


def test_plot_sholl_tree_draws_in_two_and_three_dimensions(tree):
    _pv()
    ax = pt.plot_sholl_tree(tree, 40.0)
    assert ax.get_xlabel() == "x [um]"
    assert len(pt.plot_sholl_tree(tree, 40.0, dim=3).renderer.actors) > 3


@pytest.mark.parametrize("style", ["electrode", "wide_electrode"])
def test_pointer_tree_draws_electrodes(tree, style):
    pv = _pv()
    plotter = pv.Plotter(off_screen=True)
    pt.pointer_tree(plotter, tree, [0, 5], style=style, rng=1)
    assert len(plotter.renderer.actors) == 2


def test_dendrogram_offset_shifts_and_scales(tree):
    import matplotlib.pyplot as plt

    _, plain = plt.subplots()
    _, moved = plt.subplots()
    pt.dendrogram_tree(tree, ax=plain)
    pt.dendrogram_tree(tree, ax=moved, offset=(10.0, 5.0), wscale=2.0)
    a, b = plain.lines[0].get_xydata(), moved.lines[0].get_xydata()
    np.testing.assert_allclose(b[:, 0], a[:, 0] * 2 + 10.0)
    np.testing.assert_allclose(b[:, 1], a[:, 1] + 5.0)


def test_dendrogram_can_drop_horizontals_and_warns_on_multifurcations(tree):
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    pt.dendrogram_tree(tree, ax=ax, horizontal=False)
    assert len(ax.lines[0].get_xydata()) == 2
    star = pt.Tree([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], [0, 1, 2, 3], [0] * 4,
                   [0] * 4, [1] * 4, [0] * 4, ["t"])
    with pytest.warns(UserWarning, match="multifurcations"):
        pt.dendrogram_tree(star)


# ------------------------------ granule-cell variants ------------------------------


def test_published_variant_is_the_bundled_model():
    table, variant = pt.gc_spec(), pt.gc_spec("published")
    for region in table.regions:
        one, other = table.for_region(region), variant.for_region(region)
        assert set(one) == set(other), region
        for mechanism in one:
            for name, value in one[mechanism].items():
                assert other[mechanism][name] == pytest.approx(value), (region, mechanism, name)


def test_variants_differ_as_their_flags_say():
    published = pt.gc_spec("published")
    newborn = pt.gc_spec("newborn")
    assert newborn.for_region("soma")["pas"]["e"] == -70
    assert published.for_region("soma")["pas"]["e"] == -80
    assert (pt.gc_spec("ratadjust").for_region("soma")["Kir21"]["gkbar"]
            == pytest.approx(2.5 * published.for_region("soma")["Kir21"]["gkbar"]))


@pytest.mark.parametrize("name", pt.GC_VARIANTS)
def test_every_variant_builds_a_spec(name):
    spec = pt.gc_spec(name)
    assert spec.name == f"GC_{name}" and spec.regions


def test_ion_settings_skip_sections_without_that_ion():
    neuron = pytest.importorskip("neuron")
    from pynetrees.gc_model import _set_parameter

    with_channel, without = neuron.h.Section(name="a"), neuron.h.Section(name="b")
    with_channel.insert("hh")
    assert _set_parameter(neuron.h, [with_channel, without], "k_ion", "ek", -91.0)
    assert with_channel(0.5).ek == -91.0


# ------------------------------ node-level mechanisms ------------------------------


def test_only_nodes_sets_just_the_segments_holding_them(tree):
    pytest.importorskip("neuron")
    cell = tree.copy(Ri=100.0, Gm=1 / 20000, Cm=1.0)
    model = pt.build_neuron_model(cell, d_lambda=0.02)
    section = max(model.sections, key=lambda sec: sec.nseg)
    assert section.nseg >= 3
    node = next(n for n, sec in model.node_section.items()
                if sec is section and 0.05 < model.node_x[n] < 0.95)

    pt.insert_mechanism(model, "hh", nodes=[node], only_nodes=True, gnabar=0.5)
    index = min(int(model.node_x[node] * section.nseg), section.nseg - 1)
    values = [seg.hh.gnabar for seg in section]
    assert values[index] == 0.5
    assert all(v == 0.0 for i, v in enumerate(values) if i != index)


def test_only_nodes_needs_nodes_and_parameters(tree):
    pytest.importorskip("neuron")
    model = pt.build_neuron_model(tree.copy(Ri=100.0, Gm=1 / 20000, Cm=1.0))
    with pytest.raises(ValueError, match="nodes="):
        pt.insert_mechanism(model, "hh", only_nodes=True, gnabar=0.1)
    with pytest.raises(ValueError, match="parameter"):
        pt.insert_mechanism(model, "hh", nodes=[1], only_nodes=True)
