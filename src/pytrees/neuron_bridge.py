"""Bridge from a :class:`Tree` to a real, running NEURON simulation (Phase 11).

Ported from ``T2N-master/`` -- but not as a literal translation. T2N's own
core (``t2n.m``, 2447 lines) spends the overwhelming majority of its bulk
generating ``.hoc`` *text* (``neuron_template_tree.m``) to hand off to an
external NEURON process via files/SSH/cluster plumbing, because MATLAB has
no direct NEURON binding. Python does: the ``neuron`` package gives direct
in-process access to ``h.Section``, ``pt3dadd``, mechanism insertion, and
recording vectors, so none of that text-generation/subprocess/server
machinery has a reason to exist here -- this was already the plan recorded
in PORT_PLAN.md's phase table before this phase started.

What *is* ported faithfully is the geometric heart of
``neuron_template_tree.m``: each dissected branch (:func:`dissect_tree`)
becomes one NEURON ``Section``, built from the tree's real 3D points via
``pt3dadd`` (not manually-set ``L``/``diam``, so NEURON handles
varying-diameter segments the same way it would for a hand-built
morphology), connected parent-to-child exactly as the tree's topology
dictates. Segment count (``nseg``) uses NEURON's own standard "d_lambda
rule" (``h.lambda_f``, loaded from ``stdlib.hoc``) rather than
reimplementing that arc-length/diameter walk by hand in Python --
reusing NEURON's own verified implementation instead of re-deriving it is
strictly more robust, not a shortcut (same posture as trusting an
established formula in Phase 2/8, just here the "established
implementation" is NEURON's own C code instead of MATLAB's).

Passive properties come directly from :attr:`Tree.Ri`/:attr:`Tree.Gm`/
:attr:`Tree.Cm` (Phase 8) -- no separate per-region "mech" configuration
struct is introduced; T2N's own generality there (arbitrary mechanisms
per named region) is instead exposed as the general-purpose
:func:`insert_mechanism`, layered on top of the same three attributes
this port already uses for its own passive-cable functions
(:mod:`pytrees.electrotonics`).

Unlike ``neuron_template_tree.m``, this module does **not** call
``root_tree`` first: Phase 2's :func:`dissect_tree` already handles the
root the same way as every other cut point (see its own docstring), so
MATLAB's dummy-root-prepending workaround (needed only because *its*
region-cut logic couldn't otherwise treat the true root as an internal
boundary) isn't needed here. Building this module's region-aware section
splitting is in fact what surfaced a real, confirmed bug in
:func:`dissect_tree`'s ``by_region`` logic: it was marking a region
*transition* node itself as the cut point, when MATLAB's own algorithm
(`iR = idpar(tree.R ~= tree.R(idpar))`) marks that node's *parent*
instead. The difference is silent but real -- every region boundary
produced one extra, spurious section split. Fixed at the source
(`graphtheory.py`); see PORT_STATUS.md Design Decision #36.

This module is optional and lazily imported -- `import pytrees` never
requires `neuron` to be installed, matching Phase 7's `pyvista`/
`matplotlib` precedent. NEURON has no Windows pip wheel; Windows needs the
official binary installer (see `pyproject.toml`'s `neuron` extra).

Scope: this is the foundational "build and run a real compartmental model
from a Tree" vertical slice -- section/segment construction, arbitrary
mechanism insertion, and a current-clamp simulation runner. T2N's full
protocol library (IV/FI curves, resonance, synaptic/network protocols,
cluster execution) is not ported; those are all thin wrappers on top of
exactly this layer and can be added once there's a concrete need for a
specific one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import Tree
from .graphtheory import dissect_tree, ipar_tree

__all__ = ["NeuronModel", "build_neuron_model", "insert_mechanism", "run_current_clamp"]


def _require(tree: Tree, attr: str):
    value = getattr(tree, attr)
    if value is None:
        raise ValueError(
            f"tree {tree.name!r} has no {attr} set -- build_neuron_model needs "
            f"a physical value, e.g. tree.{attr} = 100 (scalar) or a "
            f"length-{tree.n_nodes} per-node array"
        )
    return value


def _at(value, node: int) -> float:
    """Resolve a scalar-or-per-node Ri/Gm/Cm value at one node index."""
    return float(value) if np.isscalar(value) else float(np.asarray(value)[node])


@dataclass
class NeuronModel:
    """A tree built into a live NEURON section tree.

    Attributes
    ----------
    tree : Tree
        The tree the model was built from -- node indices below are this
        tree's own (:func:`~pytrees.dissect_tree` handles the root
        directly, no `root_tree` prepending needed here).
    sections : list
        Every ``h.Section`` created, one per dissected branch.
    node_section : dict[int, h.Section]
        Which section each node index lives on.
    node_x : dict[int, float]
        That node's normalized position (0-1) along its section.
    region_sections : dict[str, list[h.Section]]
        Sections grouped by the tree region name of their end node --
        for :func:`insert_mechanism`'s ``region=`` filter.
    """

    tree: Tree
    sections: list = field(default_factory=list)
    node_section: dict = field(default_factory=dict)
    node_x: dict = field(default_factory=dict)
    region_sections: dict = field(default_factory=dict)

    def loc(self, node: int):
        """The ``sec(x)`` NEURON segment for a given node index."""
        return self.node_section[node](self.node_x[node])


def build_neuron_model(
    tree: Tree, freq: float = 100.0, d_lambda: float = 0.1, e_pas: float = -70.0
) -> NeuronModel:
    """Build a live NEURON section tree from ``tree``.

    Requires ``tree.Ri``/``tree.Gm``/``tree.Cm`` (see
    :mod:`pytrees.electrotonics`), used as each section's ``Ra``/passive
    ``g_pas``/``cm`` (resolved from the section's end node, scalar or
    per-node). ``freq``/``d_lambda`` control segment count via NEURON's own
    d_lambda rule (see module docstring); ``e_pas`` is the passive leak
    reversal potential [mV], applied uniformly (``Tree`` has no per-node
    resting-potential attribute of its own).

    Note: a region occupying *only* the root node itself (i.e. the region
    already changes at the root's own child) doesn't get a section of its
    own -- it's absorbed into the section that follows, since there's no
    boundary before the root to split it off with. A real biophysical
    compartment (e.g. a soma you want to give its own active
    conductances via :func:`insert_mechanism`) should be modeled with more
    than one node in that region, which is normal practice anyway.
    """
    from neuron import h

    h.load_file("stdlib.hoc")

    Ri = _require(tree, "Ri")
    Gm = _require(tree, "Gm")
    Cm = _require(tree, "Cm")

    ipar = ipar_tree(tree)
    sect = dissect_tree(tree)

    sections = [h.Section(name=f"sec{i}") for i in range(len(sect))]
    end_to_secidx = {int(e): i for i, (_, e) in enumerate(sect)}
    node_section: dict[int, object] = {}
    node_x: dict[int, float] = {}
    region_sections: dict[str, list] = {}

    for i, (s, e) in enumerate(sect):
        s, e = int(s), int(e)
        chain_e = ipar[e]
        idx = int(np.flatnonzero(chain_e == s)[0])
        indy = chain_e[: idx + 1][::-1]  # node chain from s to e, inclusive

        X, Y, Z = tree.X[indy], tree.Y[indy], tree.Z[indy]
        D = tree.D[indy].copy()
        if not tree.frustum and len(D) > 1:
            # the shared boundary point's diameter belongs to the parent
            # section, not this one -- use the next point's instead
            D[0] = D[1]

        sec = sections[i]
        sec.pt3dclear()
        for x, y, z, d in zip(X, Y, Z, D):
            sec.pt3dadd(float(x), float(y), float(z), float(d))

        sec.Ra = _at(Ri, e)
        sec.cm = _at(Cm, e)
        sec.insert("pas")
        for seg in sec:
            seg.pas.g = _at(Gm, e)
            seg.pas.e = e_pas

        lam = h.lambda_f(freq, sec=sec)
        nseg = int((sec.L / (d_lambda * lam) + 0.9) / 2) * 2 + 1 if lam > 0 else 1
        sec.nseg = max(nseg, 1)

        seg_len = np.concatenate(
            [[0.0], np.cumsum(np.hypot(np.hypot(np.diff(X), np.diff(Y)), np.diff(Z)))]
        )
        total = seg_len[-1] if seg_len[-1] > 0 else 1.0
        for node, pos in zip(indy, seg_len):
            node = int(node)
            if node in node_section:
                continue  # already mapped (shared boundary node, electrically equivalent)
            node_section[node] = sec
            node_x[node] = float(np.clip(pos / total, 0.0, 1.0))

        region = tree.rnames[tree.R[e]]
        region_sections.setdefault(region, []).append(sec)

    for i, (s, _e) in enumerate(sect):
        s = int(s)
        parent_idx = end_to_secidx.get(s)
        if parent_idx is not None and parent_idx != i:
            sections[i].connect(sections[parent_idx](1), 0)

    return NeuronModel(
        tree=tree,
        sections=sections,
        node_section=node_section,
        node_x=node_x,
        region_sections=region_sections,
    )


def insert_mechanism(model: NeuronModel, mechanism: str, region: str | None = None, **params):
    """Insert a NEURON mechanism (e.g. ``"hh"``) on the model's sections.

    ``region=None`` (default) applies to every section; otherwise only
    sections belonging to that tree region name (see
    :attr:`NeuronModel.region_sections`). Keyword arguments are set as
    ``seg.<mechanism>.<param> = value`` on every segment of every affected
    section (a uniform value across the section -- for a spatially-varying
    parameter, insert first and then set values directly on
    ``model.node_section``/``model.node_x`` locations yourself).
    """
    sections = model.region_sections[region] if region is not None else model.sections
    for sec in sections:
        sec.insert(mechanism)
        for seg in sec:
            target = getattr(seg, mechanism)
            for name, value in params.items():
                setattr(target, name, value)


def run_current_clamp(
    model: NeuronModel,
    at_node: int,
    amp: float,
    delay: float,
    dur: float,
    tstop: float,
    record_nodes: list[int] | None = None,
    v_init: float = -70.0,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Inject a rectangular current step and record voltage.

    ``at_node``/``record_nodes`` are node indices into the tree ``model``
    was built from. ``amp`` [nA],
    ``delay``/``dur``/``tstop`` [ms]. Returns ``(t, v)`` where ``t`` is the
    time axis [ms] and ``v`` maps each recorded node to its voltage trace
    [mV] (``record_nodes`` defaults to ``[at_node]``).
    """
    from neuron import h

    h.load_file("stdrun.hoc")

    if record_nodes is None:
        record_nodes = [at_node]

    ic = h.IClamp(model.loc(at_node))
    ic.delay = delay
    ic.dur = dur
    ic.amp = amp

    t_vec = h.Vector().record(h._ref_t)
    v_vecs = {node: h.Vector().record(model.loc(node)._ref_v) for node in record_nodes}

    h.finitialize(v_init)
    h.continuerun(tstop)

    t = np.array(t_vec)
    v = {node: np.array(vec) for node, vec in v_vecs.items()}
    return t, v
