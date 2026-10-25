"""Extract outcome-independent QASM structure for QMutBench development rows.

The parser reads only circuit syntax. The extraction input intentionally omits
outcome, group, and row-label columns; record_id is retained only for lineage.
The official local checkout, metadata CSV, and grouped OOF row IDs are verified
before a cache can be reused.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qmutbench_validation import QMutBenchParseError, parse_qasm_file  # noqa: E402

EXPECTED_DATA_SHA = "3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778"
EXPECTED_REPO_COMMIT = "01f5e7700c051fe715fc368b307deaa7e6869ba8"
SOURCE_CSV = ROOT / "data/qmutbench_ingestion/raw/official_all_data_20261015.csv"
OFFICIAL_CHECKOUT = ROOT / "data/qmutbench_full"
QASM_ROOT = OFFICIAL_CHECKOUT / "data"
BASELINE_OOF = ROOT / "data/qmutbench_ml/master_optimization_v1/oof_predictions.csv"
DEFAULT_OUTPUT = ROOT / "data/qmutbench_ml/qasm_structural_v4"
PARSER_ARITY = {
    "id": (1, 0), "x": (1, 0), "y": (1, 0), "z": (1, 0), "h": (1, 0),
    "s": (1, 0), "sdg": (1, 0), "t": (1, 0), "tdg": (1, 0), "sx": (1, 0),
    "sxdg": (1, 0), "rx": (1, 1), "ry": (1, 1), "rz": (1, 1), "p": (1, 1),
    "u1": (1, 1), "u2": (1, 2), "u3": (1, 3), "u": (1, 3),
    "cx": (2, 0), "cy": (2, 0), "cz": (2, 0), "ch": (2, 0), "swap": (2, 0),
    "cp": (2, 1), "cu1": (2, 1), "cu2": (2, 2), "cu3": (2, 3), "cu": (2, 4),
    "crx": (2, 1), "cry": (2, 1), "crz": (2, 1), "rxx": (2, 1), "ryy": (2, 1),
    "rzz": (2, 1), "rzx": (2, 1), "ccx": (3, 0), "rccx": (3, 0), "cswap": (3, 0),
}
COUNT_GATES = ("cx", "cz", "cp", "u", "h", "ry", "rx", "rz", "ccx", "swap", "rzz", "rxx")
FEATURE_COLUMNS = [
    "qasm_mut_gate_count", "qasm_mut_depth", "qasm_mut_depth_per_qubit",
    "qasm_mut_single_qubit_count", "qasm_mut_two_qubit_count", "qasm_mut_three_plus_count",
    "qasm_mut_two_plus_fraction", "qasm_mut_distinct_gate_types",
    "qasm_orig_gate_count", "qasm_orig_depth", "qasm_orig_depth_per_qubit",
    "qasm_orig_single_qubit_count", "qasm_orig_two_qubit_count", "qasm_orig_three_plus_count",
    "qasm_orig_two_plus_fraction", "qasm_orig_distinct_gate_types",
    "qasm_delta_gate_count", "qasm_delta_depth", "qasm_delta_two_qubit_fraction",
    *[f"qasm_mut_{name}_fraction" for name in COUNT_GATES],
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def circuit_features(path: str | Path) -> dict[str, float | int]:
    """Measure native QASM operation counts and ASAP qubit-layer depth.

    Barriers synchronize their listed wires; measurements are excluded from
    gate count and depth. No gate decomposition or simulator execution occurs.
    """
    operations, width = parse_qasm_file(path)
    if width < 1:
        raise QMutBenchParseError(f"{Path(path).name}: invalid zero-width circuit")
    latest = [0] * width
    counts: dict[str, int] = {}
    one = two = three = gate_count = 0
    for op in operations:
        name = str(op[0]).lower()
        if name == "measure":
            continue
        if name == "barrier":
            wires = tuple(int(q) for q in op[1:])
            if wires:
                layer = max(latest[q] for q in wires)
                for q in wires:
                    latest[q] = layer
            continue
        if name not in PARSER_ARITY:
            raise QMutBenchParseError(f"{Path(path).name}: unsupported gate arity for {name!r}")
        arity, parameter_count = PARSER_ARITY[name]
        if len(op) != 1 + arity + parameter_count:
            raise QMutBenchParseError(f"{Path(path).name}: unexpected parsed tuple for {name!r}")
        wires = tuple(int(q) for q in op[1:1 + arity])
        if any(q < 0 or q >= width for q in wires) or len(set(wires)) != len(wires):
            raise QMutBenchParseError(f"{Path(path).name}: invalid operands for {name!r}")
        layer = 1 + max((latest[q] for q in wires), default=0)
        for q in wires:
            latest[q] = layer
        gate_count += 1
        counts[name] = counts.get(name, 0) + 1
        if arity == 1:
            one += 1
        elif arity == 2:
            two += 1
        else:
            three += 1
    depth = max(latest, default=0)
    return {
        "gate_count": gate_count, "depth": depth,
        "depth_per_qubit": depth / width,
        "single_qubit_count": one, "two_qubit_count": two,
        "three_plus_count": three,
        "two_plus_fraction": (two + three) / gate_count if gate_count else 0.0,
        "distinct_gate_types": len(counts),
        **{f"{name}_fraction": counts.get(name, 0) / gate_count if gate_count else 0.0
           for name in COUNT_GATES},
    }


def _add_source_paths(row: dict[str, Any], available_paths: set[str]) -> tuple[int, str, str]:
    rid = int(row["Unnamed: 0"])
    alg = str(row["algorithm"])
    gate = str(row["gate"])
    if not re.fullmatch(r"[A-Za-z0-9_-]+", alg) or not re.fullmatch(r"[A-Za-z0-9_-]+", gate):
        raise ValueError(f"record {rid}: unsafe algorithm or gate token in source metadata")
    qubits = int(row["qubits"])
    pos = int(row["Position_int"])
    operation = str(row["operation"])
    original = f"Origin_programs/{alg}_{qubits}_qubits.qasm"
    if operation == "Add":
        mutant_name = f"AddGate_{gate}_inGap_{pos}_.qasm"
    elif operation == "Remove":
        mutant_name = f"RemoveGate_{pos}_.qasm"
    elif operation == "Replace":
        mutant_name = f"ReplaceGate_{gate}_inPositionOfGate_{pos}.qasm"
    else:
        raise ValueError(f"Unknown mutation operation for record {rid}: {operation}")
    mutant = f"Mutated_programs/Mutants_{alg}_{qubits}_qubits/{mutant_name}"
    for rel in (original, mutant):
        if ".." in Path(rel).parts or rel.startswith(("/", "\\")):
            raise ValueError(f"QASM path escaped the verified source tree: {rel}")
        if rel not in available_paths:
            raise FileNotFoundError(f"record {rid}: source circuit not found: {rel}")
    return rid, original, mutant


def _source_snapshot() -> dict[str, Any]:
    checkout = subprocess.run(["git", "-C", str(OFFICIAL_CHECKOUT), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(OFFICIAL_CHECKOUT), "status", "--porcelain"],
                           check=True, capture_output=True, text=True).stdout.strip()
    if checkout != EXPECTED_REPO_COMMIT or dirty:
        raise ValueError(f"QMutBench source checkout is not the verified clean commit: {checkout}; dirty={bool(dirty)}")
    data_hash = sha256_file(SOURCE_CSV)
    if data_hash != EXPECTED_DATA_SHA:
        raise ValueError(f"Official metadata checksum mismatch: {data_hash}")
    return {"metadata_csv": str(SOURCE_CSV.relative_to(ROOT)).replace("\\", "/"),
            "metadata_csv_sha256": data_hash,
            "official_repository": "https://github.com/EnautMendi/QMutBench",
            "official_repository_commit": checkout,
            "official_repository_worktree_clean": True,
            "qasm_root": str(QASM_ROOT.relative_to(ROOT)).replace("\\", "/")}


def _worker_extract(item: tuple[int, str]) -> tuple[int, dict[str, Any] | None, str | None]:
    rid, relative = item
    try:
        return rid, circuit_features(QASM_ROOT / relative), None
    except Exception as exc:  # recorded per row; never silently fill a feature
        return rid, None, f"{type(exc).__name__}: {exc}"


def _source_feature_hash(parser_sha: str) -> str:
    return hashlib.sha256((Path(__file__).read_bytes() + bytes.fromhex(parser_sha)).hexdigest().encode()).hexdigest()


def extract_development(output: Path = DEFAULT_OUTPUT, workers: int | None = None,
                        chunk_rows: int = 10000) -> dict[str, Any]:
    snapshot = _source_snapshot()
    parser_sha = sha256_file(ROOT / "qmutbench_validation.py")
    script_sha = sha256_file(Path(__file__))
    data_sha = snapshot["metadata_csv_sha256"]
    # Only IDs are loaded from saved OOF predictions; no labels or probabilities enter extraction.
    oof_sha = sha256_file(BASELINE_OOF)
    oof = pd.read_csv(BASELINE_OOF, usecols=["record_id"])
    ids = oof["record_id"].astype("int64")
    if len(ids) != 578117 or ids.duplicated().any():
        raise ValueError("Development OOF record IDs failed the verified 578,117-row contract")
    cache_key = hashlib.sha256(json.dumps({
        "source": snapshot, "dataset_sha256": data_sha, "development_oof_sha256": oof_sha,
        "parser_sha256": parser_sha, "extractor_sha256": script_sha,
        "feature_columns": FEATURE_COLUMNS,
        "depth_definition": "ASAP per-wire native gate layers; barriers synchronize; measurements excluded",
    }, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("cache_key") != cache_key:
            raise ValueError(f"Stale extraction cache at {output}; choose a new versioned output directory")
    else:
        manifest_path.write_text(json.dumps({"schema": "qmutbench-qasm-structure-v1", "cache_key": cache_key,
            "source": snapshot, "source_dataset_sha256": data_sha,
            "development_oof_sha256": oof_sha, "parser_sha256": parser_sha,
            "extractor_sha256": script_sha, "development_record_ids": len(ids),
            "feature_columns": FEATURE_COLUMNS, "status": "in_progress",
            "depth_definition": "ASAP per-wire native gate layers; barriers synchronize; measurements excluded",
            "chunks": []}, indent=2) + "\n", encoding="utf-8")

    # Read only prediction-time source fields; target, Origin_program and row-label data are omitted.
    # Index the pinned QASM tree once; avoid hundreds of thousands of repeated
    # filesystem stat calls while still proving exact path coverage.
    tracked_qasm = subprocess.run(
        ["git", "-C", str(OFFICIAL_CHECKOUT), "ls-files", "-z", "--",
         "data/Origin_programs", "data/Mutated_programs"],
        check=True, capture_output=True,
    ).stdout
    available_paths = {os.fsdecode(raw).removeprefix("data/")
                       for raw in tracked_qasm.split(b"\0") if raw.endswith(b".qasm")}
    original_path_count = sum(path.startswith("Origin_programs/") for path in available_paths)
    mutant_path_count = sum(path.startswith("Mutated_programs/") for path in available_paths)
    if original_path_count != 375 or mutant_path_count != 723079:
        raise ValueError(f"Unexpected pinned QASM tree coverage: {original_path_count} originals, {mutant_path_count} mutants")
    usecols = ["Unnamed: 0", "algorithm", "qubits", "operation", "gate", "Position_int", "gates"]
    source = pd.read_csv(SOURCE_CSV, usecols=usecols).set_index("Unnamed: 0", drop=False)
    missing_ids = ids[~ids.isin(source.index)]
    if len(missing_ids):
        raise ValueError(f"OOF IDs absent from metadata: {missing_ids.head().tolist()}")
    frame = source.loc[ids.to_numpy()].reset_index(drop=True)
    paths = [_add_source_paths(row, available_paths) for row in frame.to_dict("records")]
    expected_gate_counts = dict(zip(frame["Unnamed: 0"].astype(int), frame["gates"].astype(int)))
    if len({rid for rid, _, _ in paths}) != len(paths):
        raise ValueError("Source record IDs are not unique")
    originals = sorted({original for _, original, _ in paths})
    original_cache: dict[str, dict[str, Any] | None] = {}
    original_errors: list[dict[str, Any]] = []
    for relative in originals:
        try:
            original_cache[relative] = circuit_features(QASM_ROOT / relative)
        except Exception as exc:
            original_cache[relative] = None
            original_errors.append({"record_id": -1, "qasm_path": relative,
                                    "error": f"{type(exc).__name__}: {exc}"})

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = {item["file"] for item in manifest.get("chunks", [])}
    workers = workers or max(1, min(3, os.cpu_count() or 1))
    errors_total = len(original_errors)
    success_total = 0
    chunk_meta = list(manifest.get("chunks", []))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(paths), chunk_rows):
            chunk_index = start // chunk_rows
            chunk_name = f"features_{chunk_index:05d}.csv.gz"
            error_name = f"errors_{chunk_index:05d}.jsonl"
            if chunk_name in completed:
                continue
            current = paths[start:start + chunk_rows]
            extracted = pool.map(_worker_extract, [(rid, mutant) for rid, _, mutant in current], chunksize=32)
            rows: list[dict[str, Any]] = []
            errors = list(original_errors) if start == 0 else []
            orig_for = {rid: original for rid, original, _ in current}
            mut_for = {rid: mutant for rid, _, mutant in current}
            for rid, mutant_features, error in extracted:
                original_rel = orig_for[rid]
                original_features = original_cache[original_rel]
                if error or original_features is None:
                    errors.append({"record_id": rid, "qasm_path": original_rel if original_features is None else mut_for[rid],
                                   "error": error or "original source circuit parsing failed"})
                    continue
                row = {"record_id": rid, "original_qasm_path": original_rel,
                       "mutant_qasm_path": mut_for[rid]}
                for prefix, feats in (("qasm_mut", mutant_features), ("qasm_orig", original_features)):
                    for key, value in feats.items():
                        row[f"{prefix}_{key}"] = value
                row["qasm_delta_gate_count"] = row["qasm_mut_gate_count"] - row["qasm_orig_gate_count"]
                row["qasm_delta_depth"] = row["qasm_mut_depth"] - row["qasm_orig_depth"]
                row["qasm_delta_two_qubit_fraction"] = row["qasm_mut_two_plus_fraction"] - row["qasm_orig_two_plus_fraction"]
                row["source_gate_count_match"] = int(row["qasm_mut_gate_count"] == expected_gate_counts[rid])
                rows.append(row)
            part = pd.DataFrame(rows)
            tmp = output / (chunk_name + ".tmp")
            part.to_csv(tmp, index=False, compression="gzip")
            tmp.replace(output / chunk_name)
            (output / error_name).write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in errors), encoding="utf-8")
            success_total += len(rows)
            errors_total += len([e for e in errors if e.get("record_id", -1) >= 0])
            chunk_info = {"file": chunk_name, "error_file": error_name,
                          "rows": len(rows), "errors": len(errors), "sha256": sha256_file(output / chunk_name)}
            chunk_meta = [x for x in chunk_meta if x["file"] != chunk_name] + [chunk_info]
            manifest.update({"chunks": chunk_meta, "status": "in_progress"})
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(f"completed chunk {chunk_index + 1}/{(len(paths) + chunk_rows - 1)//chunk_rows}: {len(rows)} rows, {len(errors)} errors", flush=True)
    total_rows = sum(item["rows"] for item in chunk_meta)
    total_errors = sum(item["errors"] for item in chunk_meta)
    manifest.update({"status": "complete" if total_rows + total_errors == len(paths) and total_errors == 0 else "complete_with_parse_errors",
                     "records_requested": len(paths), "feature_rows": total_rows,
                     "parse_error_rows": total_errors, "original_circuit_count": len(originals),
                     "original_parse_failures": len(original_errors), "worker_count": workers,
                     "chunk_rows": chunk_rows, "chunks": chunk_meta})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunk-rows", type=int, default=10000)
    args = parser.parse_args()
    result = extract_development(args.output_dir, args.workers, args.chunk_rows)
    print(json.dumps({k: result[k] for k in ("status", "records_requested", "feature_rows", "parse_error_rows", "original_circuit_count", "worker_count")}, indent=2))


if __name__ == "__main__":
    main()
