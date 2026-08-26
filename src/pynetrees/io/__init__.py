"""File formats: readers, writers, and the extension dispatcher.

:func:`load_tree` and :func:`save_tree` pick a format from the path's
extension; the per-format functions below are there for when you want to be
explicit, or need an option only one format has.
"""

from .hoc import save_hoc, save_nrn, t2n_interface
from .mtr import load_mtr, save_mtr
from .native import load_npz, load_tree, save_npz, save_tree
from .neu import load_neu
from .neuroml import save_neuroml
from .neurolucida import load_neurolucida
from .nmf import load_nmf, save_nmf
from .swc import load_swc, save_swc

__all__ = [
    "load_tree",
    "save_tree",
    "load_npz",
    "save_npz",
    "load_swc",
    "save_swc",
    "load_neurolucida",
    "load_mtr",
    "save_mtr",
    "load_neu",
    "load_nmf",
    "save_nmf",
    "save_hoc",
    "save_nrn",
    "save_neuroml",
    "t2n_interface",
]
