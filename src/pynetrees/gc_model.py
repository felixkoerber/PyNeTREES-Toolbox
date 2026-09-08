"""The Beining et al. (2017) granule-cell model, as one adjustable function.

Ports the *purpose* of ``GC_initModel.m`` from the Active GC Model package:
turn a reconstructed granule cell plus a biophysics specification into a
live NEURON model, with the published variants reachable by argument rather
than by editing the source.

What is ported and what is not
------------------------------

MATLAB splits the job across three files. ``GC_biophys.m`` (656 lines) is a
chain of ``strfind(options, '-x')`` tests that *computes* a nested
region -> mechanism -> parameter struct; ``GC_spinedensity.m`` adds a
per-region spine factor; ``scale_spines.m`` folds that factor into ``pas``
and deletes it again. ``GC_initModel.m`` then picks a morphology by number,
calls those three, and hands the result to T2N.

Only the last of those is a *function* in any useful sense -- the other two
are a data table written as control flow. So this module takes the table as
**data** (:func:`gc_spec`, fed from a JSON dump of ``neuron.mech{1}``, a
plain dict, or another spec) and keeps the function for what functions are
good at: composing that table, scaling it, and applying it to a tree.

The consequence is that a variant is a value here, not a flag string. Where
MATLAB writes ``GC_biophys('-a-p-n-y')`` and you must read 656 lines to find
out what ``-y`` changed, this writes::

    spec    = gc_spec("gc_mech.json")
    newborn = spec.scaled({"Kir21": 0.5, "HCN": 2.0})

and the difference is visible at the call site. :meth:`GCSpec.diff` reports
exactly what separates two variants, which the MATLAB original cannot do at
all.

Morphology selection (``ostruct.usemorph``, 16 hard-coded ``.mtr`` paths) is
deliberately **not** ported: :func:`~pynetrees.load_tree` already loads any
of them, and a number that means "SH07 cells" only in one lab's directory
layout is not an API.

Spine scaling
-------------

``GC_spinedensity.m`` gives each region a factor and ``scale_spines.m``
multiplies ``pas.cm`` and ``pas.g`` by it, modelling the extra membrane of
spines the reconstruction does not resolve. :func:`gc_spec` does the same
through ``spine_scale=``, but note that a spec dumped from a MATLAB run of
``GC_initModel`` with ``ostruct.scalespines = 1`` **already has it folded
in** -- ``adendMML.pas.cm = 1.71 = 0.9 * 1.9`` gives it away. Applying it
twice is the obvious mistake, so :func:`gc_spec` refuses to unless told the
source is unscaled.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path

from ._gc_biophys import GC_BIOPHYS
from ._population import accepts_population
from .core import Tree
from .neuron_bridge import NeuronModel, build_neuron_model

__all__ = [
    "GCModel",
    "GCSpec",
    "GC_BIOPHYS",
    "GC_SPINE_DENSITY",
    "GC_init",
    "gc_mod_path",
    "gc_spec",
]

#: Per-region spine factor from ``GC_spinedensity.m``: the *added* membrane
#: as a fraction, so a region scales by ``1 + factor * spine_scale``. The
#: proximal regions get nothing -- granule-cell spines are a molecular-layer
#: phenomenon -- which is why they are listed at ``0.0`` rather than omitted,
#: so that a region missing from this table is a real question rather than a
#: silent zero.
GC_SPINE_DENSITY: dict[str, float] = {
    "soma": 0.0,
    "axon": 0.0,
    "axonh": 0.0,
    "GCL": 0.0,
    "SGCL": 0.0,
    "adendIML": 0.5,
    "adendMML": 1.0,
    "adendOML": 1.0,
}

#: Mechanism names that describe the passive cable rather than a channel.
#: Folded into ``Ri``/``Gm``/``Cm`` before the sections are built, because
#: NEURON needs them at ``h.Section`` construction time to run the d-lambda
#: rule -- setting them afterwards leaves ``nseg`` computed from the wrong
#: cable.
_PASSIVE = frozenset({"pas", "pasq"})

#: Mechanism libraries already loaded into this process, by resolved path.
#: ``nrn_load_dll`` is not idempotent -- loading the same library twice
#: aborts with "The user defined name already exists: <mechanism>" -- and
#: comparing two variants of a model in one session is the normal thing to
#: do, so the second `GC_init` has to be a no-op rather than a crash.
_LOADED_LIBRARIES: set[str] = set()

#: The pseudo-region every other region inherits from, as in the MATLAB
#: struct.
_ALL = "all"


def _is_conductance(param: str) -> bool:
    """Does this parameter name a conductance density?

    ``scale_spines.m`` tests ``strfind(name, 'bar')``; ``g`` and ``gbar``
    are added because the passive mechanism spells it without the suffix and
    is the one entry spine scaling most has to catch.
    """
    return param.endswith("bar") or param in ("g", "gbar")


def _deep_copy_regions(regions) -> dict:
    return {
        rname: {mech: dict(params) for mech, params in block.items()}
        for rname, block in regions.items()
    }


@dataclass(frozen=True)
class GCSpec:
    """A granule-cell biophysics specification: what goes where, and how much.

    Attributes
    ----------
    regions : dict
        ``{region_name: {mechanism: {parameter: value}}}``, mirroring
        ``neuron.mech{t}`` in T2N. The pseudo-region ``"all"`` applies to
        every section and is overridden by a named region.
    celsius : float
        Simulation temperature [degC]. ``GC_initModel.m`` uses 24.
    v_init : float
        Initial membrane potential [mV].
    name : str
        Label carried through to the built model, for reporting.

    Notes
    -----
    Frozen on purpose: every adjustment returns a new spec, so a variant
    derived from another cannot quietly change the one it came from. Use
    :meth:`scaled`, :meth:`overridden` and :meth:`without` to derive.
    """

    regions: dict[str, dict[str, dict[str, float]]]
    celsius: float = 24.0
    v_init: float = -80.0
    name: str = "GC"
    spine_scaled: bool = True

    # ----------------------------------------------------------------- views

    @property
    def mechanisms(self) -> list[str]:
        """Every mechanism named anywhere in the spec, passive included."""
        return sorted({m for block in self.regions.values() for m in block})

    @property
    def channels(self) -> list[str]:
        """Active mechanisms only -- what has to be compiled from ``.mod``."""
        return [
            m for m in self.mechanisms
            if m not in _PASSIVE and not m.endswith("_ion")
        ]

    @property
    def ion_blocks(self) -> list[str]:
        """``<ion>_ion`` pseudo-mechanisms: reversals and concentrations.

        Separate from :attr:`channels` because they are not inserted --
        NEURON creates them as a side effect of inserting something that
        carries that ion, and they are only settable afterwards.
        """
        return [m for m in self.mechanisms if m.endswith("_ion")]

    def for_region(self, region: str) -> dict[str, dict[str, float]]:
        """The mechanisms applying to ``region``, with ``"all"`` merged in.

        Merged per *mechanism*, not per parameter: a region that names
        ``Cabuffer`` replaces the ``"all"`` entry outright rather than
        inheriting the parameters it did not mention. That is what MATLAB's
        struct assignment does, and half-inherited channel kinetics would be
        worse than either alternative.
        """
        merged = dict(self.regions.get(_ALL, {}))
        merged.update(self.regions.get(region, {}))
        return merged

    # ------------------------------------------------------------ derivation

    def scaled(self, factors, region: str | None = None) -> "GCSpec":
        """A copy with conductances multiplied by ``factors``.

        Parameters
        ----------
        factors : dict
            Either ``{mechanism: factor}``, scaling every parameter of that
            mechanism that names a conductance density (see
            :func:`_is_conductance` -- the same test ``scale_spines.m``
            uses), or ``{mechanism: {parameter: factor}}`` to be explicit
            about which.
        region : str, optional
            Restrict the scaling to one region; the default touches every
            region that has the mechanism.

        Examples
        --------
        A variant as a value rather than a flag string::

            newborn = spec.scaled({"Kir21": 0.5, "HCN": 2.0})
            no_na   = spec.scaled({"na8st": {"gbar": 0.0}})
        """
        regions = _deep_copy_regions(self.regions)
        for rname, block in regions.items():
            if region is not None and rname != region:
                continue
            for mech, factor in factors.items():
                if mech not in block:
                    continue
                if isinstance(factor, dict):
                    for param, value in factor.items():
                        if param in block[mech]:
                            block[mech][param] *= value
                    continue
                for param in block[mech]:
                    if _is_conductance(param):
                        block[mech][param] *= factor
        return replace(self, regions=regions)

    def overridden(self, changes) -> "GCSpec":
        """A copy with parameters replaced outright.

        ``changes`` is ``{region: {mechanism: {parameter: value}}}``, merged
        parameter by parameter so that naming one value leaves the rest of
        that mechanism alone -- the opposite of :meth:`for_region`'s
        whole-mechanism replacement, because here the caller is editing a
        spec rather than resolving an inheritance.
        """
        regions = _deep_copy_regions(self.regions)
        for rname, block in changes.items():
            target = regions.setdefault(rname, {})
            for mech, params in block.items():
                target.setdefault(mech, {}).update(params)
        return replace(self, regions=regions)

    def without(self, *mechanisms: str) -> "GCSpec":
        """A copy with these mechanisms removed from every region.

        ``spec.without(*spec.channels, *spec.ion_blocks)`` is the passive
        model (``ostruct.vmodel = 3``), which is what
        :func:`gc_spec`'s ``passive_only`` spells.
        """
        drop = set(mechanisms)
        return replace(self, regions={
            name: {m: p for m, p in block.items() if m not in drop}
            for name, block in self.regions.items()
        })

    def diff(self, other: "GCSpec") -> dict:
        """Every parameter that differs between two specs.

        Returns ``{region: {mechanism: {parameter: (mine, theirs)}}}``, with
        ``None`` on either side for a parameter only one spec has. This is
        the question "what does ``-y`` actually change?" that
        ``GC_biophys.m``'s option strings cannot answer.
        """
        out: dict = {}
        for rname in sorted(set(self.regions) | set(other.regions)):
            mine = self.regions.get(rname, {})
            theirs = other.regions.get(rname, {})
            for mech in sorted(set(mine) | set(theirs)):
                a, b = mine.get(mech, {}), theirs.get(mech, {})
                changed = {
                    param: (a.get(param), b.get(param))
                    for param in sorted(set(a) | set(b))
                    if a.get(param) != b.get(param)
                }
                if changed:
                    out.setdefault(rname, {})[mech] = changed
        return out


def gc_mod_path() -> Path:
    """Directory holding the granule-cell model's ``.mod`` sources.

    The 44 NEURON mechanism files the published model is built from, bundled
    because :data:`GC_BIOPHYS` names channels that no NEURON installation has
    -- ``na8st``, ``Kir21``, ``Kv42``, ``BK`` and the rest are specific to
    this model, and without them :func:`GC_init` can only report them all as
    :attr:`GCModel.skipped`.

    They are **sources, not a library**: NEURON loads compiled mechanisms, so
    build them once and pass the result as ``mech_library=``::

        import subprocess, pynetrees as pt

        # Linux / macOS -- writes ./x86_64 (or ./arm64)
        subprocess.run(["nrnivmodl", str(pt.gc_mod_path())], check=True)
        gc = pt.GC_init(pt.gc_tree(), mech_library="x86_64")

    On Windows the equivalent is ``mknrndll`` (a GUI shipped with NEURON;
    point it at this directory), which writes ``nrnmech.dll`` beside the
    ``.mod`` files.

    The compiled products are deliberately **not** bundled: they are ~23 MB
    and valid only for the exact NEURON version and architecture they were
    built against, so shipping one would be shipping something that silently
    refuses to load for most people.

    Returns
    -------
    Path
        The directory. Passing it to ``nrnivmodl`` compiles every file in it.

    Notes
    -----
    The returned path is only guaranteed to exist for the duration of the
    call when the package is installed as a zip; copy the files out if you
    need it to persist. A normal ``pip install`` or editable install gives a
    real directory on disk.
    """
    with resources.as_file(resources.files("pynetrees") / "data" / "gc_mod") as path:
        return Path(path)


# ---------------------------------------------------------------------------
# building a spec -- pure, so a variant can be checked without NEURON
# ---------------------------------------------------------------------------


def gc_spec(
    source=None,
    *,
    passive_only: bool = False,
    spine_scale: float | None = None,
    spine_scale_applied: bool = True,
    spine_density=None,
    scale=None,
    override=None,
    celsius: float = 24.0,
    v_init: float = -80.0,
    name: str | None = None,
) -> GCSpec:
    """Resolve a granule-cell biophysics specification.

    Pure: touches no simulator, so a variant can be built, diffed and
    asserted on in a test with NEURON absent. :func:`GC_init` is what turns
    one into a running model.

    Parameters
    ----------
    source : str or Path or dict or GCSpec, optional
        Where the table comes from. Defaults to :data:`GC_BIOPHYS`, the
        published active model bundled with the package, so ``gc_spec()``
        with no arguments is the Beining et al. (2017) granule cell. Also
        takes a JSON file of ``neuron.mech{1}`` as dumped from MATLAB, the
        same structure as a dict, or an existing spec to derive from.
    passive_only : bool, default False
        Drop every active channel and ion block, leaving the passive cable --
        ``ostruct.vmodel = 3``. Equivalent to
        ``spec.without(*spec.channels, *spec.ion_blocks)``, offered here
        because it is the most-used variant.
    spine_scale : float, optional
        Fold spine membrane into ``pas.cm``/``pas.g`` region by region: each
        is multiplied by ``1 + GC_SPINE_DENSITY[region] * spine_scale``.
        MATLAB's ``ostruct.scalespines``, which passes ``scalespines * 0.9``
        -- so ``spine_scale=0.9`` reproduces its default.
    spine_scale_applied : bool, default True
        Whether ``source`` already carries spine scaling. Passing
        ``spine_scale`` while this is ``True`` raises: scaling an
        already-scaled spec is the mistake this argument exists to catch.
    spine_density : dict, optional
        Per-region spine factors, defaulting to :data:`GC_SPINE_DENSITY`.
    scale : dict, optional
        Passed to :meth:`GCSpec.scaled` -- the usual way to express a variant
        that differs only in channel densities.
    override : dict, optional
        Passed to :meth:`GCSpec.overridden`, applied after ``scale``.
    celsius, v_init : float
        Simulation temperature [degC] and initial potential [mV].
    name : str, optional
        Label for the spec; defaults to the JSON file's stem.

    Returns
    -------
    GCSpec
    """
    if source is None:
        spec = GCSpec(
            regions=_deep_copy_regions(GC_BIOPHYS), celsius=celsius,
            v_init=v_init, name="GC_Beining2017", spine_scaled=True,
        )
    elif isinstance(source, GCSpec):
        spec = replace(source, celsius=celsius, v_init=v_init)
    elif isinstance(source, dict):
        spec = GCSpec(
            regions=_deep_copy_regions(source), celsius=celsius,
            v_init=v_init, spine_scaled=spine_scale_applied,
        )
    else:
        path = Path(source)
        with path.open(encoding="utf-8") as handle:
            spec = GCSpec(
                regions=_deep_copy_regions(json.load(handle)), celsius=celsius,
                v_init=v_init, name=path.stem, spine_scaled=spine_scale_applied,
            )

    if name is not None:
        spec = replace(spec, name=name)
    if passive_only:
        # the ion blocks go too: with no channel carrying potassium there is
        # no `ek` on any segment to set, so keeping them would report every
        # reversal potential as unapplied on a model that is working fine
        spec = spec.without(*spec.channels, *spec.ion_blocks)

    if spine_scale is not None:
        if spec.spine_scaled:
            raise ValueError(
                "this spec already has spine scaling folded in (a MATLAB dump "
                "made with ostruct.scalespines = 1 does), so spine_scale would "
                "apply it twice. Pass spine_scale_applied=False if the source "
                "really is unscaled."
            )
        spec = _scale_spines(spec, spine_scale, spine_density or GC_SPINE_DENSITY)

    if scale:
        spec = spec.scaled(scale)
    if override:
        spec = spec.overridden(override)
    return spec


def _scale_spines(spec: GCSpec, amount: float, density: dict) -> GCSpec:
    """``scale_spines.m``: fold the per-region spine factor into ``pas``.

    Passive parameters only, matching the MATLAB default -- its ``'-a'``
    option also scales active conductances, which the published model does
    not use.
    """
    regions = _deep_copy_regions(spec.regions)
    unknown = sorted(set(regions) - set(density) - {_ALL})
    if unknown:
        warnings.warn(
            f"no spine density for region(s) {unknown}; they are left "
            f"unscaled. Pass spine_density= to give them one.",
            stacklevel=3,
        )
    for rname, block in regions.items():
        factor = 1.0 + density.get(rname, 0.0) * amount
        for mech in _PASSIVE & set(block):
            for param in block[mech]:
                if _is_conductance(param) or param == "cm":
                    block[mech][param] *= factor
    return replace(spec, regions=regions, spine_scaled=True)


# ---------------------------------------------------------------------------
# building the model
# ---------------------------------------------------------------------------


@dataclass
class GCModel:
    """A granule cell built into NEURON, plus the spec that produced it.

    Attributes
    ----------
    model : NeuronModel
        The sections themselves -- ``loc(node)``, ``sections``,
        ``region_sections`` and everything else
        :mod:`pynetrees.neuron_bridge` provides. Attribute access falls
        through to it, so ``gc.loc(node)`` works directly.
    spec : GCSpec
        The resolved specification, so a finished model can still say what
        it was built from.
    skipped : list[tuple[str, str, str]]
        ``(region, mechanism, parameter)`` for everything the compiled
        mechanism library would not accept. Empty is what you want; entries
        here almost always mean the ``.mod`` files were compiled from a
        different version of the model than the spec came from.
    """

    model: NeuronModel
    spec: GCSpec
    skipped: list = field(default_factory=list)

    def __getattr__(self, name):
        # the NeuronModel is what callers actually use, so let this stand in
        # for one rather than making every line say `.model.`
        return getattr(self.model, name)

    def report(self) -> str:
        """One-line summary, including anything that did not apply."""
        line = (
            f"{self.spec.name}: {self.model.tree.n_nodes} nodes -> "
            f"{len(self.model.sections)} sections, "
            f"{len(self.spec.channels)} channel(s)"
        )
        if self.skipped:
            missing = sorted({mech for _, mech, _ in self.skipped})
            line += f"; {len(self.skipped)} parameter(s) skipped in {missing}"
        return line


def _passive_arrays(tree: Tree, spec: GCSpec):
    """Per-node ``Ri``/``Gm``/``Cm`` from the spec's ``pas`` entries.

    Region by region, ``all`` first so a named region overrides it. A node
    whose region names no passive block would keep ``nan``, which is
    reported rather than silently simulated -- NEURON takes ``nan`` for
    ``Ra`` without complaint and produces a cell that integrates to ``nan``
    seconds later, with nothing pointing back to here.
    """
    import numpy as np

    n = tree.n_nodes
    Ri, Gm, Cm = (np.full(n, np.nan) for _ in range(3))

    def assign(params, mask):
        Ri[mask], Gm[mask], Cm[mask] = params["Ra"], params["g"], params["cm"]

    shared = spec.regions.get(_ALL, {})
    for mech in _PASSIVE & set(shared):
        assign(shared[mech], slice(None))

    for index, rname in enumerate(tree.rnames):
        block = spec.regions.get(rname, {})
        for mech in _PASSIVE & set(block):
            assign(block[mech], tree.R == index)

    unset = np.flatnonzero(np.isnan(Ri))
    if unset.size:
        missing = sorted({tree.rnames[r] for r in tree.R[unset]})
        raise ValueError(
            f"GC_init: no passive parameters for region(s) {missing} -- "
            f"{unset.size} of {n} nodes would get Ra=nan and the simulation "
            f"would silently produce nan. Add a 'pas' block for them to the "
            f"spec, or an 'all' block covering everything."
        )
    return Ri, Gm, Cm


def _default_e_pas(spec: GCSpec) -> float:
    """The leak reversal to build with, before the per-region pass.

    ``build_neuron_model`` takes one value; regions that disagree with it
    are corrected in :func:`_apply_reversals` immediately afterwards. Taking
    the ``all`` block's value avoids doing that work at all in the common
    case where every region agrees.
    """
    shared = spec.regions.get(_ALL, {})
    for mech in _PASSIVE & set(shared):
        if "e" in shared[mech]:
            return float(shared[mech]["e"])
    for block in spec.regions.values():
        for mech in _PASSIVE & set(block):
            if "e" in block[mech]:
                return float(block[mech]["e"])
    return -80.0


def _sections_for(model: NeuronModel, region: str) -> list:
    """The sections a spec region names, ``all`` meaning every one."""
    if region == _ALL:
        return list(model.sections)
    return model.region_sections.get(region, [])


def _set_parameter(h, sections, mech: str, param: str, value: float) -> bool:
    """Set one parameter, per segment if NEURON allows it, globally if not.

    A ``.mod`` file's ``RANGE`` variables live on the segment and may differ
    along the cell; its ``GLOBAL`` ones live on the mechanism and may not.
    The spec does not record which is which -- T2N does the same probe on
    the MATLAB side -- so ask the object rather than keeping a table that
    would go stale against the ``.mod`` files.
    """
    probe = getattr(next(iter(sections[0])), mech, None)
    if probe is not None and hasattr(probe, param):
        for sec in sections:
            for seg in sec:
                setattr(getattr(seg, mech), param, value)
        return True

    # an ion block's reversal potential (ek/ena/eca) sits on the segment
    # itself rather than on a mechanism object
    if mech.endswith("_ion") and hasattr(next(iter(sections[0])), param):
        for sec in sections:
            for seg in sec:
                setattr(seg, param, value)
        return True

    global_name = f"{param}_{mech}"
    if hasattr(h, global_name):
        setattr(h, global_name, value)
        return True
    return False


def _apply_mechanisms(h, model: NeuronModel, spec: GCSpec, skipped: list) -> None:
    """Insert and parameterise every non-passive mechanism in the spec.

    Two passes, and the order matters in both directions.

    Across regions, ``all`` goes first so that a named region's value wins --
    the inheritance :meth:`GCSpec.for_region` describes.

    Across mechanism *kinds*, channels go first and ``<ion>_ion`` blocks
    last. A segment only grows an ``ek`` attribute once something that uses
    potassium has been inserted into it, so setting the reversal potentials
    in the same pass silently drops every one of them -- which is exactly
    what happened here before this was split: ``ek`` stayed at NEURON's
    default -77 mV instead of the model's -93 mV, on a cell whose whole
    point is its potassium conductances.
    """
    ordered = [_ALL] + [r for r in spec.regions if r != _ALL]

    for pass_is_ion in (False, True):
        for rname in ordered:
            block = spec.regions.get(rname, {})
            sections = _sections_for(model, rname)
            if not sections:
                continue  # a region the spec knows and this morphology lacks
            for mech, params in block.items():
                if mech in _PASSIVE:
                    continue  # already folded into Ri/Gm/Cm
                if mech.endswith("_ion") is not pass_is_ion:
                    continue
                if not pass_is_ion:
                    try:
                        for sec in sections:
                            sec.insert(mech)
                    except ValueError:
                        # not in the compiled library at all -- record every
                        # parameter of it rather than one line per section
                        skipped.extend((rname, mech, p) for p in params)
                        continue
                for param, value in params.items():
                    if not _set_parameter(h, sections, mech, param, float(value)):
                        skipped.append((rname, mech, param))


def _apply_reversals(model: NeuronModel, spec: GCSpec) -> None:
    """Per-region ``e_pas``, for the variants where the axon differs.

    Run after the sections exist because ``e`` is the one passive parameter
    that does not feed the d-lambda rule, so it costs nothing to set late --
    unlike ``Ra`` and ``cm``, which have to be right before ``nseg`` is
    chosen.
    """
    for rname, block in spec.regions.items():
        sections = _sections_for(model, rname)
        if not sections:
            continue
        for mech in _PASSIVE & set(block):
            if "e" not in block[mech]:
                continue
            value = float(block[mech]["e"])
            for sec in sections:
                for seg in sec:
                    seg.pas.e = value


def _load_mechanisms(h, library) -> None:
    """Load a compiled ``.mod`` library once per process.

    Says clearly what is wrong when the library is absent or unloadable, and
    does nothing at all when it has already been loaded -- see
    :data:`_LOADED_LIBRARIES` for why the second call must not be an error.
    """
    # resolved, because NEURON hands the path to the platform loader as-is
    # and Windows' LoadLibrary does not interpret it relative to the Python
    # process's working directory -- a relative path fails with a bare
    # "module not found" that says nothing about why
    path = Path(library).resolve()
    if str(path) in _LOADED_LIBRARIES:
        return
    if not path.exists():
        raise FileNotFoundError(
            f"mech_library={str(path)!r} does not exist. Compile the model's "
            f".mod files (nrnivmodl on Linux/macOS, mknrndll on Windows) and "
            f"point this at the resulting nrnmech.dll or x86_64 directory."
        )
    if not h.nrn_load_dll(str(path)):
        raise RuntimeError(
            f"NEURON refused to load {str(path)!r}. It is usually built for a "
            f"different NEURON version or architecture than the one running "
            f"here; recompile the .mod files against this NEURON."
        )
    _LOADED_LIBRARIES.add(str(path))


@accepts_population
def GC_init(
    tree: Tree,
    mech=None,
    *,
    cut_axon: bool = False,
    mech_library=None,
    freq: float = 100.0,
    d_lambda: float = 0.1,
    strict: bool = False,
    **spec_options,
) -> GCModel:
    """Build a granule cell into NEURON from a morphology and a spec.

    The Python counterpart of ``GC_initModel.m``, minus its morphology
    catalogue: pass the tree you want, since :func:`~pynetrees.load_tree`
    already reads every ``.mtr`` that function hard-codes a path for.

    Parameters
    ----------
    tree : Tree or list[Tree]
        The morphology. Its region names have to match the spec's --
        ``soma``, ``GCL``, ``adendIML``/``adendMML``/``adendOML``, ``axon``,
        ``axonh`` for the published model. A list of trees gives a list of
        models, nested to whatever depth was passed in.
    mech : str, Path, dict or GCSpec, optional
        The biophysics, in any form :func:`gc_spec` takes. Defaults to
        :data:`GC_BIOPHYS`, so ``GC_init(tree)`` builds the published model
        and the argument is there for when you want a different one.
    cut_axon : bool, default False
        Delete the ``axon`` region first, keeping the hillock ``axonh`` --
        ``ostruct.cut``, and what the dendritic-democracy scripts do.
    mech_library : str or Path, optional
        Compiled mechanism library to load: ``nrnmech.dll`` on Windows, the
        ``x86_64``/``arm64`` directory elsewhere. Without it NEURON knows
        only its built-in mechanisms and every granule-cell channel lands in
        :attr:`GCModel.skipped`.
    freq, d_lambda : float
        Passed to :func:`~pynetrees.build_neuron_model` for the segment count.
    strict : bool, default False
        Raise if any parameter could not be applied, rather than collecting
        it in :attr:`GCModel.skipped`. Worth turning on in a test; off by
        default because a partly-compiled library is a normal state to be
        exploring in.
    **spec_options
        Forwarded to :func:`gc_spec` (``passive_only``, ``scale``,
        ``override``, ``spine_scale``, ...), so the common case stays one
        call::

            gc = GC_init(tree)                        # the published model
            gc = GC_init(tree, passive_only=True)     # its passive variant
            gc = GC_init(tree, "my_dump.json")        # something else entirely

    Returns
    -------
    GCModel

    Notes
    -----
    Passive parameters are folded into ``tree.Ri``/``Gm``/``Cm`` **before**
    the sections are built, because NEURON's d-lambda rule reads ``Ra`` and
    ``cm`` to choose ``nseg``. Setting them afterwards -- the tempting
    order, since everything else is set on finished sections -- leaves the
    compartment count computed from the wrong cable.
    """
    from neuron import h

    spec = gc_spec(mech, **spec_options)

    if mech_library is not None:
        _load_mechanisms(h, mech_library)

    if cut_axon and "axon" in tree.rnames:
        from .edit import delete_tree

        tree = delete_tree(tree, tree.region_nodes("axon"), keep_regions=True)

    h.celsius = spec.celsius

    Ri, Gm, Cm = _passive_arrays(tree, spec)
    tree = tree.copy(Ri=Ri, Gm=Gm, Cm=Cm)
    model = build_neuron_model(
        tree, freq=freq, d_lambda=d_lambda, e_pas=_default_e_pas(spec)
    )

    skipped: list[tuple[str, str, str]] = []
    _apply_mechanisms(h, model, spec, skipped)
    _apply_reversals(model, spec)

    if strict and skipped:
        raise RuntimeError(
            f"GC_init: {len(skipped)} parameter(s) could not be applied, "
            f"starting with {skipped[0]}. The compiled mechanism library is "
            f"missing them -- compile the model's .mod files and pass "
            f"mech_library=, or leave strict=False to build anyway."
        )
    return GCModel(model=model, spec=spec, skipped=skipped)
