"""Memory-bounded, resumable QMutBench DEVELOPMENT-only QASM feature extraction.

Frozen test origins are excluded using only record_id values from saved grouped
development OOF predictions. The extractor never reads the Killed target column
and never opens a frozen-test QASM file. Feature chunks contain no labels.

Run from the repository root:
  python experiments/qmutbench_full_corpus_streaming.py
  python experiments/qmutbench_full_corpus_streaming.py --chunk-rows 250
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.qmutbench_qasm_features import FEATURE_COLUMNS, circuit_features  # noqa: E402

DEFAULT_METADATA = ROOT / "data/qmutbench_ingestion/raw/official_all_data_20261015.csv"
DEFAULT_QASM_ROOT = ROOT / "data/qmutbench_full/data"
DEFAULT_DEV_IDS = ROOT / "data/qmutbench_ml/master_optimization_v1/oof_predictions.csv"
DEFAULT_OUTPUT = ROOT / "data/qmutbench_ml/qmutbench_dev_qasm_stream_v2"
EXPECTED_DEV_ROWS = 578_117
EXPECTED_DEV_ORIGINS = 305
REQUIRED = {"Unnamed: 0", "Origin_program", "algorithm", "qubits", "gates", "operation", "gate", "Position_int"}
MUTATION_OPERATIONS = {"Add", "Remove", "Replace"}
OUTPUT_COLUMNS = ["record_id", "origin_program", "original_qasm_path", "mutant_qasm_path",
                  *FEATURE_COLUMNS, "source_gate_count_match"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def available_ram_bytes() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        if os.name == "nt":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                    ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_extended", ctypes.c_ulonglong),
                ]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.avail_phys)
    return 0


def require_ram_floor(min_available_mib: int) -> int:
    available = available_ram_bytes()
    if available < min_available_mib * 1024 * 1024:
        raise MemoryError(
            f"Available RAM is {available / 1024**2:.1f} MiB; safety floor is "
            f"{min_available_mib} MiB. No new chunk was started; resume after freeing memory."
        )
    return available


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.reindex(columns=OUTPUT_COLUMNS).to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


def safe_relative_paths(row: dict[str, Any]) -> tuple[str, str]:
    algorithm, gate = str(row["algorithm"]), str(row["gate"])
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not algorithm or not gate or any(c not in allowed for c in algorithm + gate):
        raise ValueError("Unsafe algorithm/gate token in metadata")
    qubits, position = int(row["qubits"]), int(row["Position_int"])
    operation = str(row["operation"])
    original = f"Origin_programs/{algorithm}_{qubits}_qubits.qasm"
    if operation == "Add":
        mutant_name = f"AddGate_{gate}_inGap_{position}_.qasm"
    elif operation == "Remove":
        mutant_name = f"RemoveGate_{position}_.qasm"
    elif operation == "Replace":
        mutant_name = f"ReplaceGate_{gate}_inPositionOfGate_{position}.qasm"
    else:
        raise ValueError(f"Unsupported mutation operation: {operation}")
    mutant = f"Mutated_programs/Mutants_{algorithm}_{qubits}_qubits/{mutant_name}"
    for rel in (original, mutant):
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError(f"Path escaped QASM root: {rel}")
    return original, mutant


def make_feature_row(record_id: int, origin: str, original_rel: str, mutant_rel: str,
                     original: dict[str, Any], mutant: dict[str, Any], expected_gate_count: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "record_id": record_id, "origin_program": origin,
        "original_qasm_path": original_rel, "mutant_qasm_path": mutant_rel,
    }
    for prefix, features in (("qasm_mut", mutant), ("qasm_orig", original)):
        for name, value in features.items():
            row[f"{prefix}_{name}"] = value
    row["qasm_delta_gate_count"] = row["qasm_mut_gate_count"] - row["qasm_orig_gate_count"]
    row["qasm_delta_depth"] = row["qasm_mut_depth"] - row["qasm_orig_depth"]
    row["qasm_delta_two_qubit_fraction"] = row["qasm_mut_two_plus_fraction"] - row["qasm_orig_two_plus_fraction"]
    row["source_gate_count_match"] = int(row["qasm_mut_gate_count"] == expected_gate_count)
    return row


def extract(metadata: Path, qasm_root: Path, dev_ids_path: Path, output: Path, *,
            chunk_rows: int, min_available_mib: int, max_chunks: int | None = None,
            expected_dev_rows: int = EXPECTED_DEV_ROWS) -> dict[str, Any]:
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")
    for path in (metadata, dev_ids_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not qasm_root.is_dir():
        raise FileNotFoundError(f"QASM root not found: {qasm_root}")
    output.mkdir(parents=True, exist_ok=True)

    # Read only record_id from saved development OOF predictions; no labels or probabilities.
    dev_id_frame = pd.read_csv(dev_ids_path, usecols=["record_id"])
    dev_id_series = pd.to_numeric(dev_id_frame["record_id"], errors="raise").astype("int64")
    if dev_id_series.duplicated().any():
        raise ValueError("Saved development OOF record_id values are not unique")
    dev_ids = set(dev_id_series.tolist())
    if len(dev_ids) != expected_dev_rows:
        raise ValueError(f"Expected {expected_dev_rows} development IDs, found {len(dev_ids)}")

    config = {
        "schema": "qmutbench-development-qasm-features-v2",
        "metadata_sha256": sha256_file(metadata),
        "development_ids_sha256": sha256_file(dev_ids_path),
        "parser_sha256": sha256_file(ROOT / "qmutbench_validation.py"),
        "extractor_sha256": sha256_file(Path(__file__)),
        "metadata_path": str(metadata.resolve()), "qasm_root": str(qasm_root.resolve()),
        "development_rows_expected": len(dev_ids), "chunk_rows": chunk_rows,
        "feature_columns": OUTPUT_COLUMNS,
        "label_policy": "metadata Killed column is never read; feature chunks exclude labels",
        "test_policy": "only record_ids in saved grouped-development OOF predictions are eligible for QASM file access",
        "depth_definition": "ASAP per-wire native gate layers; barriers synchronize; measurements excluded",
    }
    cache_key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cache_key") != cache_key:
            raise ValueError("Output manifest has different source/config hashes; choose a new output directory")
    else:
        manifest = {**config, "cache_key": cache_key, "status": "in_progress", "chunks": [],
                    "development_rows_seen": 0, "feature_rows": 0, "parse_error_rows": 0,
                    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        atomic_json(manifest_path, manifest)

    completed: dict[int, dict[str, Any]] = {}
    for item in manifest.get("chunks", []):
        path = output / item["file"]
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise ValueError(f"Missing or corrupted checkpoint chunk: {path}")
        completed[int(item["chunk_index"])] = item

    original_cache: dict[str, dict[str, Any] | None] = {}
    original_failures: dict[str, str] = {}
    seen_dev_ids: set[int] = set()
    origins_seen: set[str] = set()
    started = time.monotonic()
    usecols = sorted(REQUIRED)
    try:
        for chunk_index, frame in enumerate(pd.read_csv(metadata, usecols=usecols, chunksize=chunk_rows, low_memory=False)):
            if max_chunks is not None and chunk_index >= max_chunks:
                break
            require_ram_floor(min_available_mib)
            missing = sorted(REQUIRED - set(frame.columns))
            if missing:
                raise ValueError(f"Metadata schema missing columns: {missing}")
            numeric_ids = pd.to_numeric(frame["Unnamed: 0"], errors="raise").astype("int64")
            selected_mask = numeric_ids.isin(dev_ids)
            selected = frame.loc[selected_mask].copy()
            selected_ids = set(numeric_ids.loc[selected_mask].tolist())
            overlap = selected_ids & seen_dev_ids
            if overlap:
                raise ValueError(f"Duplicate development record IDs encountered: {sorted(overlap)[:5]}")
            seen_dev_ids.update(selected_ids)
            origins_seen.update(selected["Origin_program"].astype(str).unique())
            chunk_name = f"features_{chunk_index:05d}.csv.gz"
            error_name = f"errors_{chunk_index:05d}.jsonl"
            if chunk_index in completed:
                continue

            rows: list[dict[str, Any]] = []
            errors: list[dict[str, Any]] = []
            for row_index, raw in enumerate(selected.to_dict(orient="records")):
                if row_index % 25 == 0:
                    require_ram_floor(min_available_mib)
                record_id = int(raw["Unnamed: 0"])
                origin = str(raw["Origin_program"])
                try:
                    original_rel, mutant_rel = safe_relative_paths(raw)
                    original_path, mutant_path = qasm_root / original_rel, qasm_root / mutant_rel
                    if not original_path.is_file():
                        raise FileNotFoundError(f"Missing original QASM: {original_rel}")
                    if not mutant_path.is_file():
                        raise FileNotFoundError(f"Missing mutant QASM: {mutant_rel}")
                    if original_rel not in original_cache:
                        try:
                            original_cache[original_rel] = circuit_features(original_path)
                        except Exception as exc:
                            original_cache[original_rel] = None
                            original_failures[original_rel] = f"{type(exc).__name__}: {exc}"
                    original_features = original_cache[original_rel]
                    if original_features is None:
                        raise ValueError(f"Original QASM parse failure: {original_failures[original_rel]}")
                    mutant_features = circuit_features(mutant_path)
                    rows.append(make_feature_row(record_id, origin, original_rel, mutant_rel,
                                                 original_features, mutant_features, int(raw["gates"])))
                except Exception as exc:
                    errors.append({"record_id": record_id, "origin_program": origin,
                                   "error_type": type(exc).__name__, "error": str(exc)})

            atomic_csv_gzip(pd.DataFrame(rows), output / chunk_name)
            error_path = output / error_name
            tmp_error = error_path.with_suffix(error_path.suffix + ".tmp")
            tmp_error.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in errors), encoding="utf-8")
            os.replace(tmp_error, error_path)
            completed[chunk_index] = {
                "chunk_index": chunk_index, "file": chunk_name, "error_file": error_name,
                "metadata_rows_scanned": int(len(frame)), "development_rows_seen": int(len(selected)),
                "feature_rows": int(len(rows)), "parse_error_rows": int(len(errors)),
                "sha256": sha256_file(output / chunk_name),
                "available_ram_mib_after": round(available_ram_bytes() / 1024**2, 1),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            manifest["chunks"] = [completed[i] for i in sorted(completed)]
            manifest.update({
                "development_rows_seen": sum(x["development_rows_seen"] for x in completed.values()),
                "feature_rows": sum(x["feature_rows"] for x in completed.values()),
                "parse_error_rows": sum(x["parse_error_rows"] for x in completed.values()),
                "development_origins_seen_so_far": len(origins_seen),
                "original_parse_failures": len(original_failures),
                "original_parse_failure_examples": dict(list(original_failures.items())[:25]),
                "elapsed_seconds_this_run": round(time.monotonic() - started, 3),
                "status": "in_progress",
            })
            atomic_json(manifest_path, manifest)
            print(f"source_chunk={chunk_index} dev_rows={len(selected)} features={len(rows)} "
                  f"errors={len(errors)} ram={available_ram_bytes()/1024**2:.1f}MiB "
                  f"elapsed={time.monotonic()-started:.1f}s", flush=True)

    except MemoryError as exc:
        manifest.update({"status": "paused_resource_guard", "pause_reason": str(exc),
                         "chunks": [completed[i] for i in sorted(completed)]})
        atomic_json(manifest_path, manifest)
        print(f"PAUSED SAFELY: {exc}", file=sys.stderr, flush=True)
        return manifest

    finished_scan = max_chunks is None
    total_dev_rows = sum(item["development_rows_seen"] for item in completed.values())
    total_features = sum(item["feature_rows"] for item in completed.values())
    total_errors = sum(item["parse_error_rows"] for item in completed.values())
    missing_dev_ids = len(dev_ids - seen_dev_ids) if finished_scan else None
    if finished_scan and missing_dev_ids:
        raise ValueError(f"{missing_dev_ids} saved development IDs were absent from the metadata scan")
    status = "complete" if finished_scan and total_dev_rows == len(dev_ids) and total_features == len(dev_ids) and total_errors == 0 else (
        "complete_with_errors" if finished_scan and total_dev_rows == len(dev_ids) else "partial")
    manifest.update({
        "status": status, "development_rows_seen": total_dev_rows, "feature_rows": total_features,
        "parse_error_rows": total_errors, "development_origins_seen": len(origins_seen),
        "missing_development_ids": missing_dev_ids,
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds_this_run": round(time.monotonic() - started, 3),
        "chunks": [completed[i] for i in sorted(completed)],
    })
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--qasm-root", type=Path, default=DEFAULT_QASM_ROOT)
    parser.add_argument("--development-ids", type=Path, default=DEFAULT_DEV_IDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-rows", type=int, default=500)
    parser.add_argument("--min-available-mib", type=int, default=1024)
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional smoke run; leaves manifest partial.")
    args = parser.parse_args()
    result = extract(args.metadata, args.qasm_root, args.development_ids, args.output_dir,
                     chunk_rows=args.chunk_rows, min_available_mib=args.min_available_mib,
                     max_chunks=args.max_chunks)
    print(json.dumps({key: result.get(key) for key in (
        "status", "development_rows_seen", "feature_rows", "parse_error_rows",
        "development_origins_seen", "missing_development_ids", "cache_key", "output_dir",
    )}, indent=2))


if __name__ == "__main__":
    main()
