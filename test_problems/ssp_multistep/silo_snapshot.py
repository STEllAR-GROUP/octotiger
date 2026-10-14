# Copyright (c) 2026 AUTHORS
# Distributed under the Boost Software License, Version 1.0.
"""Read and compare exact Octo-Tiger restart payloads with the Silo C library."""

import ctypes as C
import ctypes.util
from pathlib import Path


class SiloSnapshot:
    def __init__(self, library=None):
        # Some Linux Silo packages leave C++ runtime symbols unresolved.
        cxx = ctypes.util.find_library("stdc++")
        if cxx:
            self.cxx = C.CDLL(cxx, mode=C.RTLD_GLOBAL)
        library = library or ctypes.util.find_library("siloh5") or ctypes.util.find_library("silo")
        if not library:
            raise RuntimeError("Silo shared library not found; pass --silo-library")
        self.silo = C.CDLL(str(library))
        signatures = {
            "DBOpenReal": ([C.c_char_p, C.c_int, C.c_int], C.c_void_p),
            "DBClose": ([C.c_void_p], C.c_int),
            "DBInqVarExists": ([C.c_void_p, C.c_char_p], C.c_int),
            "DBGetVarByteLength": ([C.c_void_p, C.c_char_p], C.c_int),
            "DBGetVarType": ([C.c_void_p, C.c_char_p], C.c_int),
            "DBReadVar": ([C.c_void_p, C.c_char_p, C.c_void_p], C.c_int),
            "DBGetVarDims": ([C.c_void_p, C.c_char_p, C.c_int, C.POINTER(C.c_int)], C.c_int),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.silo, name)
            function.argtypes = arguments
            function.restype = result

    def read(self, path, names):
        """Return each raw variable's datatype, dimensions and exact value bytes."""
        handle = self.silo.DBOpenReal(str(Path(path).resolve()).encode(), 5, 1)  # UNKNOWN, READ
        if not handle:
            raise RuntimeError(f"Cannot open Silo file: {path}")
        try:
            result = {}
            for name in names:
                encoded = name.encode()
                if self.silo.DBInqVarExists(handle, encoded) != 1:
                    raise RuntimeError(f"Missing checkpoint variable: {path}:{name}")
                length = self.silo.DBGetVarByteLength(handle, encoded)
                if length <= 0:
                    raise RuntimeError(f"Empty checkpoint variable: {path}:{name}")
                datatype = self.silo.DBGetVarType(handle, encoded)
                dimensions = (C.c_int * 32)()
                rank = self.silo.DBGetVarDims(handle, encoded, 32, dimensions)
                if rank <= 0 or rank > 32:
                    raise RuntimeError(f"Invalid checkpoint dimensions: {path}:{name}")
                buffer = C.create_string_buffer(length)
                if self.silo.DBReadVar(handle, encoded, buffer) != 0:
                    raise RuntimeError(f"Cannot read checkpoint variable: {path}:{name}")
                result[name] = (datatype, tuple(dimensions[:rank]), buffer.raw)
            return result
        finally:
            self.silo.DBClose(handle)

    @staticmethod
    def integers(variable):
        datatype, dimensions, contents = variable
        integer_types = {16: C.c_int, 17: C.c_short, 18: C.c_long, 22: C.c_longlong}
        if datatype not in integer_types or len(dimensions) != 1:
            raise RuntimeError("Expected a one-dimensional Silo integer array")
        value_type = integer_types[datatype]
        count = dimensions[0]
        if len(contents) != count * C.sizeof(value_type):
            raise RuntimeError("Silo integer array length disagrees with its dimensions")
        return tuple((value_type * count).from_buffer_copy(contents))

    def checkpoint(self, path):
        """Return all node payloads keyed by node id, independent of file grouping."""
        path = Path(path)
        metadata = self.read(path, ("restart_version", "restart_groups", "node_list"))
        if self.integers(metadata["restart_version"]) not in ((1,), (2,)):
            raise RuntimeError(f"Unsupported exact restart version in {path}")
        node_ids = self.integers(metadata["node_list"])
        groups = self.integers(metadata["restart_groups"])
        if len(groups) != len(node_ids) or len(set(node_ids)) != len(node_ids):
            raise RuntimeError(f"Invalid exact restart node/group inventory in {path}")
        grouped_nodes = {}
        for node_id, group in zip(node_ids, groups):
            if node_id <= 0 or group < 0:
                raise RuntimeError(f"Invalid node id or restart group in {path}")
            grouped_nodes.setdefault(group, []).append(node_id)
        payloads = {}
        for group, nodes in grouped_nodes.items():
            names = [f"restart/{node_id:o}/state" for node_id in nodes]
            values = self.read(path.parent / (path.name + ".data") / f"{group}.silo", names)
            for node_id, name in zip(nodes, names):
                datatype, dimensions, contents = values[name]
                if datatype != 21 or dimensions != (len(contents),):  # DB_CHAR
                    raise RuntimeError(f"Invalid opaque restart payload: node {node_id:o} in {path}")
                payloads[node_id] = contents
        return payloads


def compare_checkpoints(reader, left, right, expected_nodes=9):
    metadata_names = ("restart_layout", "restart_globals",
                      "restart_solver_integers", "restart_solver_reals")
    before_metadata = reader.read(left, metadata_names)
    after_metadata = reader.read(right, metadata_names)
    for name in metadata_names:
        a, b = before_metadata[name], after_metadata[name]
        if a != b:
            if a[:2] != b[:2]:
                detail = f"datatype/dimensions {a[:2]} != {b[:2]}"
            else:
                offset = next((i for i, (x, y) in enumerate(zip(a[2], b[2])) if x != y),
                              min(len(a[2]), len(b[2])))
                detail = f"first differing byte {offset}, lengths {len(a[2])}/{len(b[2])}"
            raise AssertionError(f"Exact restart master metadata mismatch: {name}: {detail}")
    before, after = reader.checkpoint(left), reader.checkpoint(right)
    if before.keys() != after.keys():
        raise AssertionError(f"Restart node inventories differ: {left} vs {right}")
    # Level-one test meshes contain eight leaves and their root. Checking the
    # root as well matters: it owns FMM state and multistep timestep metadata.
    if expected_nodes is not None and len(before) != expected_nodes:
        raise AssertionError(f"Expected all {expected_nodes} node archives, found {len(before)} in {left}")
    differences = []
    for node_id in sorted(before):
        a, b = before[node_id], after[node_id]
        if a != b:
            offset = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                          min(len(a), len(b)))
            differences.append(f"node {node_id:o}: first differing byte {offset}, "
                               f"lengths {len(a)}/{len(b)}")
    if differences:
        raise AssertionError("Exact restart payload mismatch:\n" + "\n".join(differences))
    return len(before), sum(len(value) for value in before.values())
