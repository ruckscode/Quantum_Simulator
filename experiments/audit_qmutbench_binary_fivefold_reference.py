"""Validate the historical five-fold QMutBench OOF split using stdlib only.

Reads only numeric ID/label/fold NPY members from the trusted local NPZ and
joins those values to the development OOF CSV. Object arrays are never read.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import struct
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/mutation_diff_oof.npz"
OOF = ROOT / "data/qmutbench_ml/master_optimization_v1/oof_predictions.csv"
N_META = 723_079
EXPECTED_ROWS = 578_117


def numeric_member(archive: zipfile.ZipFile, name: str) -> tuple[tuple[int, ...], str, bytes]:
    with archive.open(name) as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError(f"Invalid NPY member {name}")
        major, minor = stream.read(2)
        if major == 1:
            header_size = struct.unpack("<H", stream.read(2))[0]
        elif major in (2, 3):
            header_size = struct.unpack("<I", stream.read(4))[0]
        else:
            raise ValueError(f"Unsupported NPY version {major}.{minor}")
        header = ast.literal_eval(stream.read(header_size).decode("latin1").strip())
        descr = header["descr"]
        if header["fortran_order"]:
            raise ValueError(f"Unexpected Fortran-ordered numeric member {name}")
        shape = tuple(header["shape"])
        if not descr.endswith(("i8", "i4", "i2", "i1", "u8", "u4", "u2", "u1")):
            raise ValueError(f"Non-integer or object member rejected: {name}: {descr}")
        return shape, descr, stream.read()


def values(payload: bytes, descr: str):
    endian = "<" if descr.startswith("<") or descr.startswith("=") else ">"
    code = descr[-2:]
    formats = {"i8": "q", "i4": "i", "i2": "h", "i1": "b",
               "u8": "Q", "u4": "I", "u2": "H", "u1": "B"}
    fmt = endian + formats[code]
    width = struct.calcsize(fmt)
    if len(payload) % width:
        raise ValueError("NPY numeric payload has a truncated element")
    return (value[0] for value in struct.iter_unpack(fmt, payload))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit() -> dict:
    fold_by_id = bytearray(N_META)
    y_by_id = bytearray([255]) * N_META
    with zipfile.ZipFile(ARCHIVE) as archive:
        available = set(archive.namelist())
        required = {"record_id.npy", "y.npy", "fold.npy"}
        if not required <= available:
            raise ValueError(f"Missing numeric arrays: {required - available}")
        members = {name: numeric_member(archive, name) for name in required}
    shapes = {name: member[0] for name, member in members.items()}
    if any(shape != (EXPECTED_ROWS,) for shape in shapes.values()):
        raise ValueError(f"Unexpected reference array shapes: {shapes}")
    ids = values(members["record_id.npy"][2], members["record_id.npy"][1])
    ys = values(members["y.npy"][2], members["y.npy"][1])
    folds = values(members["fold.npy"][2], members["fold.npy"][1])
    for rid, y, fold in zip(ids, ys, folds):
        if not (0 <= rid < N_META and y in (0, 1) and fold in (1, 2, 3, 4, 5)):
            raise ValueError(f"Invalid reference tuple id={rid}, y={y}, fold={fold}")
        if fold_by_id[rid]:
            raise ValueError(f"Duplicate record ID in reference arrays: {rid}")
        fold_by_id[rid] = fold
        y_by_id[rid] = y

    fold_counts: Counter[int] = Counter()
    class_counts: Counter[tuple[int, int]] = Counter()
    group_fold: dict[str, int] = {}
    seen = bytearray(N_META)
    rows = 0
    with OOF.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            rows += 1
            rid = int(row["record_id"]); y = int(row["y_true"]); origin = str(row["origin_program"])
            if rid < 0 or rid >= N_META or not fold_by_id[rid] or seen[rid]:
                raise ValueError(f"Missing/duplicate development reference ID {rid}")
            if y != y_by_id[rid] or y not in (0, 1) or not origin:
                raise ValueError(f"Reference label/origin mismatch for record {rid}")
            fold = int(fold_by_id[rid])
            if origin in group_fold and group_fold[origin] != fold:
                raise ValueError(f"Origin {origin!r} crosses five-fold reference splits")
            group_fold[origin] = fold
            seen[rid] = 1
            fold_counts[fold] += 1
            class_counts[(fold, y)] += 1
    if rows != EXPECTED_ROWS or sum(seen) != EXPECTED_ROWS or len(group_fold) != 305:
        raise ValueError(f"Coverage mismatch: rows={rows}, unique={sum(seen)}, origins={len(group_fold)}")
    return {"status": "verified", "archive": str(ARCHIVE.relative_to(ROOT)).replace("\\", "/"),
        "archive_sha256": sha256(ARCHIVE), "oof_csv": str(OOF.relative_to(ROOT)).replace("\\", "/"),
        "records": rows, "origins": len(group_fold),
        "fold_counts": {str(k): fold_counts[k] for k in sorted(fold_counts)},
        "class_counts_by_fold": {f"fold={f},label={y}": class_counts[(f, y)]
                                  for f in range(1, 6) for y in (0, 1)},
        "origin_cross_fold_count": 0, "numeric_arrays_only": True,
        "frozen_test_accessed": False}


if __name__ == "__main__":
    result = audit()
    out = ROOT / "reports/experiments/full_data_qml_optimization_v3/binary_reference_fold_audit.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
