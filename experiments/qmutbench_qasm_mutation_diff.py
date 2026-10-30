"""Derive bounded original/mutant QASM mutation-alignment features.

This operates only on syntax and source mutation metadata. It does not load
outcomes, group labels, or row IDs as model features. Row IDs are retained only
for exact provenance joins. A linear common-prefix/common-suffix alignment
avoids SequenceMatcher's worst-case behavior on long repetitive circuits.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qmutbench_validation import parse_qasm_file  # noqa: E402
from experiments.qmutbench_qasm_features import (BASELINE_OOF, EXPECTED_DATA_SHA, EXPECTED_REPO_COMMIT,
    OFFICIAL_CHECKOUT, PARSER_ARITY, QASM_ROOT, SOURCE_CSV, sha256_file)
from qmutbench_ml.qasm_mutation_diff import UnsupportedQASMGateError, mutation_diff_features

OUTPUT = ROOT / "data/qmutbench_ml/qasm_mutation_diff_v1"
FEATURE_COLUMNS = [
    "qasm_diff_orig_span", "qasm_diff_mut_span", "qasm_diff_total_span",
    "qasm_diff_common_prefix_fraction", "qasm_diff_common_suffix_fraction",
    "qasm_diff_position_fraction", "qasm_diff_expected_gate_match",
    "qasm_diff_qubit_set_equal", "qasm_diff_qubits_touched",
    "qasm_diff_removed_arity", "qasm_diff_added_arity",
    "qasm_diff_exact_single_edit", "qasm_diff_is_local",
    "qasm_diff_removed_gate", "qasm_diff_added_gate", "qasm_diff_alignment_class",
    "qasm_diff_local_prev_gate", "qasm_diff_local_next_gate",
]
USECOLS = ["Unnamed: 0", "algorithm", "qubits", "operation", "gate", "Position_int"]
_ORIGINAL_OPERATIONS: dict[str, list[tuple[str, tuple[int, ...]]]] = {}


def _ops(path: Path) -> list[tuple[str, tuple[int, ...]]]:
    parsed, _ = parse_qasm_file(path)
    result = []
    for op in parsed:
        name = str(op[0]).lower()
        if name in {"measure", "barrier"}:
            continue
        if name not in PARSER_ARITY:
            raise UnsupportedQASMGateError(
                f"Unsupported OpenQASM gate {name!r}; mutation features support {len(PARSER_ARITY)} gate names."
            )
        arity, parameter_count = PARSER_ARITY[name]
        if len(op) != 1 + arity + parameter_count:
            raise ValueError(f"Unexpected parsed operation representation for {name!r}")
        wires = tuple(int(q) for q in op[1:1 + arity])
        result.append((name, wires))
    return result


def _safe_paths(row: dict[str, Any], tracked_paths: set[str]) -> tuple[int, str, str]:
    rid, alg, qubits = int(row["Unnamed: 0"]), str(row["algorithm"]), int(row["qubits"])
    operation, gate, pos = str(row["operation"]), str(row["gate"]), int(row["Position_int"])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", alg) or not re.fullmatch(r"[A-Za-z0-9_-]+", gate):
        raise ValueError(f"Record {rid}: unsafe algorithm/gate path token")
    original = f"Origin_programs/{alg}_{qubits}_qubits.qasm"
    if operation == "Add":
        name = f"AddGate_{gate}_inGap_{pos}_.qasm"
    elif operation == "Remove":
        name = f"RemoveGate_{pos}_.qasm"
    elif operation == "Replace":
        name = f"ReplaceGate_{gate}_inPositionOfGate_{pos}.qasm"
    else:
        raise ValueError(f"Record {rid}: unsupported operation {operation!r}")
    mutant = f"Mutated_programs/Mutants_{alg}_{qubits}_qubits/{name}"
    if original not in tracked_paths or mutant not in tracked_paths:
        raise FileNotFoundError(f"Record {rid}: source circuit absent from verified pinned tree")
    return rid, original, mutant


def _initialize_worker(cache_path: str) -> None:
    global _ORIGINAL_OPERATIONS
    raw = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    _ORIGINAL_OPERATIONS = {name: [(gate, tuple(wires)) for gate, wires in operations]
                            for name, operations in raw.items()}


def _read_development_metadata(ids: list[int]) -> pd.DataFrame:
    """Scan row identifiers first; materialize mutation fields only for dev IDs."""
    wanted_ids = set(map(int, ids))
    selected = []
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        id_name = "Unnamed: 0" if "Unnamed: 0" in header else ""
        id_index = header.index(id_name)
        field_indices = {name: header.index(name) for name in USECOLS
                         if name not in {"Unnamed: 0", id_name}}
        for row in reader:
            rid = int(row[id_index])
            if rid in wanted_ids:
                selected.append({"Unnamed: 0": rid,
                                 **{name: row[index] for name, index in field_indices.items()}})
    if len(selected) != len(wanted_ids):
        found = {int(row["Unnamed: 0"]) for row in selected}
        raise ValueError(f"Development reference IDs missing in source metadata: {len(wanted_ids - found)}")
    frame = pd.DataFrame(selected).set_index("Unnamed: 0", drop=False)
    if frame.index.duplicated().any():
        raise ValueError("Development source metadata contains duplicate record IDs")
    return frame


def _iter_development_metadata_chunks(ids: np.ndarray, chunk_rows: int):
    """Yield only development metadata in bounded source-order chunks."""
    wanted = np.asarray(ids, dtype=np.int64)
    if chunk_rows < 1 or (len(wanted) > 1 and np.any(wanted[1:] <= wanted[:-1])):
        raise ValueError("Development record IDs must be unique, increasing, and chunk size positive")
    selected: list[dict[str, Any]] = []
    pointer = 0
    last_id = -1
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        id_name = "Unnamed: 0" if "Unnamed: 0" in header else ""
        id_index = header.index(id_name)
        field_indices = {name: header.index(name) for name in USECOLS
                         if name not in {"Unnamed: 0", id_name}}
        for row in reader:
            rid = int(row[id_index])
            if rid <= last_id:
                raise ValueError("Source metadata IDs are not strictly increasing")
            last_id = rid
            if pointer >= len(wanted):
                continue
            expected = int(wanted[pointer])
            if rid > expected:
                raise ValueError(f"Development source metadata is missing record ID {expected}")
            if rid == expected:
                selected.append({"Unnamed: 0": rid,
                    **{name: row[index] for name, index in field_indices.items()}})
                pointer += 1
                if len(selected) == chunk_rows:
                    yield pd.DataFrame(selected)
                    selected = []
    if pointer != len(wanted):
        raise ValueError(f"Development source metadata is missing {len(wanted) - pointer} record IDs")
    if selected:
        yield pd.DataFrame(selected)


def _worker(item: tuple[int, str, str, str, str]) -> tuple[int, dict[str, Any] | None, str | None]:
    rid, original, mutant, operation, gate = item
    try:
        if original not in _ORIGINAL_OPERATIONS:
            raise KeyError(f"Original circuit absent from checked signature cache: {original}")
        features = mutation_diff_features(_ORIGINAL_OPERATIONS[original], _ops(QASM_ROOT / mutant), operation, gate)
        return rid, features, None
    except Exception as exc:
        return rid, None, f"{type(exc).__name__}: {exc}"


def extract(output: Path = OUTPUT, workers: int | None = None, chunk_rows: int = 10000) -> dict[str, Any]:
    if sha256_file(SOURCE_CSV) != EXPECTED_DATA_SHA:
        raise ValueError("Source metadata checksum mismatch")
    rev = subprocess.run(["git", "-C", str(OFFICIAL_CHECKOUT), "rev-parse", "HEAD"], check=True,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(OFFICIAL_CHECKOUT), "status", "--porcelain"], check=True,
                           capture_output=True, text=True).stdout.strip()
    if rev != EXPECTED_REPO_COMMIT or dirty:
        raise ValueError("Official QMutBench checkout is not at the verified clean commit")
    oof = pd.read_csv(BASELINE_OOF, usecols=["record_id"])
    ids = oof.record_id.astype("int64").to_numpy()
    if len(ids) != 578117 or len(np.unique(ids)) != len(ids):
        raise ValueError("Development row IDs violate the verified OOF contract")
    ids.sort()
    tracked_raw = subprocess.run(["git", "-C", str(OFFICIAL_CHECKOUT), "ls-files", "-z", "--",
                                  "data/Origin_programs", "data/Mutated_programs"], check=True,
                                 capture_output=True).stdout
    tracked = {os.fsdecode(path).removeprefix("data/") for path in tracked_raw.split(b"\0") if path.endswith(b".qasm")}
    script_sha = sha256_file(Path(__file__))
    parser_sha = sha256_file(ROOT / "qmutbench_validation.py")
    key_data = {"source_sha256": EXPECTED_DATA_SHA, "qasm_commit": rev,
                "development_oof_sha256": sha256_file(BASELINE_OOF),
                "extractor_sha256": script_sha, "parser_sha256": parser_sha,
                "runtime_features_sha256": sha256_file(ROOT / "qmutbench_ml/qasm_mutation_diff.py"),
                "features": FEATURE_COLUMNS, "alignment": "linear common-prefix/common-suffix on gate name and wire tuple"}
    cache_key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cache_key") != cache_key:
            raise ValueError("Stale QASM mutation-diff cache; select a new output version")
    else:
        manifest = {"schema": "qmutbench-qasm-mutation-diff-v1", **key_data,
                    "cache_key": cache_key, "feature_columns": FEATURE_COLUMNS,
                    "development_rows": len(ids), "chunks": [], "status": "in_progress"}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    original_names: set[str] = set()
    for metadata in _iter_development_metadata_chunks(ids, chunk_rows):
        for row in metadata.to_dict("records"):
            _, original, _ = _safe_paths(row, tracked)
            original_names.add(original)
    original_cache_path = output / "original_signatures.json"
    if not original_cache_path.exists():
        originals = {}
        for relative in sorted(original_names):
            originals[relative] = [[name, list(wires)] for name, wires in _ops(QASM_ROOT / relative)]
        temp_cache = original_cache_path.with_suffix(".tmp")
        temp_cache.write_text(json.dumps(originals, separators=(",", ":")), encoding="utf-8")
        temp_cache.replace(original_cache_path)
    else:
        originals = json.loads(original_cache_path.read_text(encoding="utf-8"))
        if set(originals) != original_names:
            raise ValueError("Original signature cache does not cover the current development origins")
    manifest["original_signatures_sha256"] = sha256_file(original_cache_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    completed = {item["file"] for item in manifest.get("chunks", [])}
    chunk_info = list(manifest.get("chunks", []))
    workers = workers or max(1, min(3, os.cpu_count() or 1))
    total_errors = 0
    metadata_chunks = _iter_development_metadata_chunks(ids, chunk_rows)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=_initialize_worker,
                                                 initargs=(str(original_cache_path),)) as pool:
        for index, start in enumerate(range(0, len(ids), chunk_rows)):
            filename, errname = f"diff_{index:05d}.csv.gz", f"errors_{index:05d}.jsonl"
            metadata = next(metadata_chunks)
            if filename in completed:
                continue
            current = [_safe_paths(row, tracked) for row in metadata.to_dict("records")]
            task_rows = [(rid, orig, mut, str(row.operation), str(row.gate))
                         for (rid, orig, mut), row in zip(current, metadata.itertuples(index=False))]
            rows, errors = [], []
            for rid, features, error in pool.map(_worker, task_rows, chunksize=16):
                if error:
                    errors.append({"record_id": rid, "error": error})
                else:
                    rows.append({"record_id": rid, **features})
            part = pd.DataFrame(rows)
            temp = output / (filename + ".tmp")
            part.to_csv(temp, index=False, compression="gzip")
            temp.replace(output / filename)
            (output / errname).write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in errors), encoding="utf-8")
            total_errors += len(errors)
            entry = {"file": filename, "error_file": errname, "rows": len(rows),
                     "errors": len(errors), "sha256": sha256_file(output / filename)}
            chunk_info = [x for x in chunk_info if x["file"] != filename] + [entry]
            manifest.update({"chunks": chunk_info, "status": "in_progress"})
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"completed chunk {index+1}/{(len(ids)+chunk_rows-1)//chunk_rows}: {len(rows)} rows, {len(errors)} errors", flush=True)
    row_count = sum(x["rows"] for x in chunk_info)
    errors_count = sum(x["errors"] for x in chunk_info)
    manifest.update({"status": "complete" if row_count + errors_count == len(ids) and errors_count == 0 else "complete_with_errors",
                     "feature_rows": row_count, "error_rows": errors_count,
                     "worker_count": workers, "chunk_rows": chunk_rows})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunk-rows", type=int, default=10000)
    args = parser.parse_args()
    result = extract(args.output_dir, args.workers, args.chunk_rows)
    print(json.dumps({key: result.get(key) for key in ("status", "development_rows", "feature_rows", "error_rows", "worker_count")}, indent=2))
