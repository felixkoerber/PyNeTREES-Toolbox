"""Bundled sample tree, mirroring MATLAB's ``sample/mtr/sample_tree.m``.

The MATLAB version loads a subtree of an HS cell from ``sample.mtr``. Since
``.mtr`` (MATLAB binary) support is deferred (see PORT_STATUS.md), this port
bundles the same HS-cell reconstruction as an SWC file instead
(``treestoolbox-master/sample/swc/25HSS.swc``).
"""

from __future__ import annotations

from importlib import resources

from .core import Tree
from .io.swc import load_swc


def sample_tree() -> Tree:
    """Return the bundled sample neuron (a subtree of an HS cell)."""
    with resources.as_file(
        resources.files("pytrees") / "data" / "sample.swc"
    ) as path:
        tree = load_swc(path)
    if isinstance(tree, list):
        tree = tree[0]
    tree.name = "sample"
    return tree
