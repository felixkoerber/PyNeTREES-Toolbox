"""Writing MATLAB v7.3 files, which are HDF5 with MATLAB's own conventions.

Only what a tree struct needs: numbers, logicals, strings, sparse matrices,
structs and cell arrays. MATLAB stores arrays column-first, so every dimension
list is reversed relative to NumPy's.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
from scipy import sparse

_USERBLOCK = 512
_GZIP_ABOVE = 1024    # elements; small datasets are not worth compressing


def _header() -> bytes:
    """The 512-byte preamble that marks an HDF5 file as a MAT-file."""
    stamp = datetime.datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    text = f"MATLAB 7.3 MAT-file, Platform: pynetrees, Created on: {stamp} HDF5 schema 1.00 ."
    return text.encode("ascii").ljust(116, b" ") + b"\0" * 8 + b"\x00\x02" + b"IM"


def write_v73(path: str | Path, name: str, value) -> None:
    """Write one variable to a new v7.3 MAT-file at ``path``."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            "writing MATLAB v7.3 files needs h5py: pip install 'pynetrees[matlab]'. "
            "Saving as .swc is an alternative that MATLAB also reads."
        ) from error

    with h5py.File(path, "w", userblock_size=_USERBLOCK, libver="earliest") as handle:
        _Writer(handle, h5py).write(handle, name, value)
    with open(path, "r+b") as raw:
        raw.write(_header())


class _Writer:
    def __init__(self, handle, h5py):
        self.h5py = h5py
        self.refs = handle.require_group("#refs#")
        self.count = 0

    def _class(self, node, name: str) -> None:
        node.attrs["MATLAB_class"] = np.bytes_(name)

    def _dataset(self, parent, name, data, matlab_class):
        data = np.asarray(data)
        options = {}
        if data.size > _GZIP_ABOVE:
            options = {"compression": "gzip", "compression_opts": 3, "chunks": True}
        node = parent.create_dataset(name, data=data, **options)
        self._class(node, matlab_class)
        return node

    def _empty(self, parent, name, shape, matlab_class):
        node = parent.create_dataset(name, data=np.array(shape[::-1], dtype=np.uint64))
        self._class(node, matlab_class)
        node.attrs["MATLAB_empty"] = np.uint8(1)
        return node

    def write(self, parent, name, value):
        """Write ``value`` as ``parent[name]`` and return the new node."""
        if sparse.issparse(value):
            return self._sparse(parent, name, value)
        if isinstance(value, dict):
            return self._struct(parent, name, value)
        if isinstance(value, str):
            return self._string(parent, name, value)
        if isinstance(value, (list, tuple)):
            cells = np.empty((1, len(value)), dtype=object)
            for index, member in enumerate(value):
                cells[0, index] = member
            return self._cell(parent, name, cells)
        array = np.asarray(value)
        if array.dtype == object:
            return self._cell(parent, name, array)
        if array.dtype.kind in "US":
            return self._string(parent, name, str(array.item()))
        return self._numeric(parent, name, array)

    def _numeric(self, parent, name, array):
        matlab_shape = (1, 1) if array.ndim == 0 else array.shape
        if array.ndim == 1:
            matlab_shape = (array.size, 1)
        if array.size == 0:
            return self._empty(parent, name, matlab_shape, "double")
        data = array.reshape(matlab_shape).T
        if array.dtype.kind == "b":
            node = self._dataset(parent, name, data.astype(np.uint8), "logical")
            node.attrs["MATLAB_int_decode"] = np.int32(1)
            return node
        return self._dataset(parent, name, data.astype(float), "double")

    def _string(self, parent, name, text: str):
        if not text:
            return self._empty(parent, name, (0, 0), "char")
        codes = np.frombuffer(text.encode("utf-16-le"), dtype=np.uint16).reshape(-1, 1)
        node = self._dataset(parent, name, codes, "char")
        node.attrs["MATLAB_int_decode"] = np.int32(2)
        return node

    def _sparse(self, parent, name, matrix):
        matrix = sparse.csc_matrix(matrix)
        matrix.sort_indices()
        group = parent.create_group(name)
        self._class(group, "double")
        group.attrs["MATLAB_sparse"] = np.uint64(matrix.shape[0])
        group.create_dataset("data", data=matrix.data.astype(float))
        group.create_dataset("ir", data=matrix.indices.astype(np.uint64))
        group.create_dataset("jc", data=matrix.indptr.astype(np.uint64))
        return group

    def _struct(self, parent, name, fields: dict):
        group = parent.create_group(name)
        self._class(group, "struct")
        for field, value in fields.items():
            self.write(group, field, value)
        names = np.empty(len(fields), dtype=self.h5py.vlen_dtype(np.dtype("S1")))
        for index, field in enumerate(fields):
            names[index] = np.array(list(field), dtype="S1")
        group.attrs["MATLAB_fields"] = names
        return group

    def _cell(self, parent, name, cells: np.ndarray):
        if cells.size == 0:
            return self._empty(parent, name, cells.shape or (0, 0), "cell")
        references = np.empty(cells.shape[::-1], dtype=self.h5py.ref_dtype)
        flat_refs = references.T
        for position in np.ndindex(cells.shape):
            label = format(self.count, "x")
            self.count += 1
            flat_refs[position] = self.write(self.refs, label, cells[position]).ref
        node = parent.create_dataset(name, data=references, dtype=self.h5py.ref_dtype)
        self._class(node, "cell")
        return node


def read_v73(path: str | Path) -> dict:
    """Read a v7.3 MAT-file into plain Python values, keeping cell structure.

    Structs become dicts, cell arrays lists, strings ``str``, sparse matrices
    SciPy sparse, and numeric arrays NumPy arrays (single numbers as scalars).
    """
    import h5py

    with h5py.File(path, "r") as handle:
        return {name: _read(handle, handle[name], h5py) for name in handle
                if not name.startswith("#")}


def _read(handle, node, h5py):
    attrs = node.attrs
    matlab_class = attrs.get("MATLAB_class", b"")
    matlab_class = matlab_class.decode() if isinstance(matlab_class, bytes) else str(matlab_class)

    if isinstance(node, h5py.Group):
        if "MATLAB_sparse" in attrs:
            return _read_sparse(node)
        fields = {key: node[key] for key in node}
        lengths = {f.shape for f in fields.values()
                   if isinstance(f, h5py.Dataset) and f.dtype == h5py.ref_dtype}
        if len(lengths) == 1 and len(lengths) and matlab_class != "cell" and all(
                isinstance(f, h5py.Dataset) and f.dtype == h5py.ref_dtype
                for f in fields.values()) and np.prod(next(iter(lengths))) > 1:
            count = int(np.prod(next(iter(lengths))))   # a struct array
            return [{key: _read(handle, handle[np.ravel(f[()], order="F")[i]], h5py)
                     for key, f in fields.items()} for i in range(count)]
        return {key: _read(handle, value, h5py) for key, value in fields.items()}

    if attrs.get("MATLAB_empty", 0):
        return "" if matlab_class == "char" else ([] if matlab_class == "cell"
                                                  else np.empty((0, 0)))
    if node.dtype == h5py.ref_dtype:
        cells = np.ravel(node[()], order="F")
        return [_read(handle, handle[reference], h5py) for reference in cells]

    data = np.asarray(node[()]).T
    if matlab_class == "char":
        return data.astype(np.uint16).ravel(order="F").tobytes().decode("utf-16-le")
    if matlab_class == "logical":
        data = data.astype(bool)
    if data.size == 1:
        return data.ravel()[0].item()
    return np.squeeze(data)


def _read_sparse(group):
    rows = int(group.attrs["MATLAB_sparse"])
    pointers = np.asarray(group["jc"][()], dtype=np.int64)
    if "data" not in group:
        return sparse.csr_matrix((rows, len(pointers) - 1))
    return sparse.csc_matrix(
        (np.asarray(group["data"][()], dtype=float),
         np.asarray(group["ir"][()], dtype=np.int64), pointers),
        shape=(rows, len(pointers) - 1)).tocsr()
