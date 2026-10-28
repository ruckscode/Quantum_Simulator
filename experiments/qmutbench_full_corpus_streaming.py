"""Memory-bounded, resumable QMutBench full-corpus QASM feature extraction.

This phase extracts syntax-only features for every metadata row. It deliberately
keeps labels out of feature files; labels are joined later by record_id. The
source corpus and metadata remain local-only and are never uploaded.

Example (PowerShell, from repository root):
  python experiments/qmutbench_full_corpus_streaming.py
  python experiments/qmutbench_full_corpus_streaming.py --chunk-rows 500 --min-available-mib 1024

The extractor stops safely when available RAM is below the configured floor.
Completed chunks are checksum-verified and resumable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.qmutbench_qasm_features import circuit_features  # noqa: E402

DEFAULT_METADATA = ROOT / "data/qmutbench_ingestion/raw/official_all_data_20261015.csv"
DEFAULT_QASM_ROOT = ROOT / "data/qmutbench_full/data"
DEFAULT_OUTPUT = ROOT / "data/qmutbench_ml/full_corpus_qasm_stream_v1"
REQUIRED = {
    "Unnamed: 0", "Origin_program", "algorithm", "qubits", "gates",
    "operation", "gate", "Position_int", "Killed",
}
EXPECTED_ROWS = 723_079
EXPECTED_ORIGINS = 375
LABELS = {"WOO", "OPO", "Survivor"}
MUTATION_OPERATIONS = {"Add", "Remove", "Replace"}


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
    floor = min_available_mib * 1024 * 1024
    if available < floor:
        raise MemoryError(
            f"Available RAM is {available / 1024**2:.1f} MiB; "
            f"configured safety floor is {min_available_mib} MiB. "
            "No new chunk was started. Close memory-heavy applications and resume."
        )
    return available


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


def safe_relative_paths(row: dict[str, Any]) -> tuple[str, str]:
    algorithm = str(row["algorithm"])
    gate = str(row["gate"])
    if not algorithm or not gate or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in algorithm + gate):
        raise ValueError("Unsafe algorithm/gate token in metadata")
    qubits = int(row["qubits"])
    position = int(row["Position_int"])
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


def _feature_row(record_id: int, origin: str, original_rel: str, mutant_rel: str,
                 original: dict[str, Any], mutant: dict[str, Any], expected_gate_count: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "record_id": record_id,
        "origin_program": origin,
        "original_qasm_path": original_rel,
        "mutant_qasm_path": mutant_rel,
    }
    for prefix, features in (("qasm_mut", mutant), ("qasm_orig", original)):
        for name, value in features.items():
            result[f"{prefix}_{name}"] = value
    result["qasm_delta_gate_count"] = result["qasm_mut_gate_count"] - result["qasm_orig_gate_count"]
    result["qasm_delta_depth"] = result["qasm_mut_depth"] - result["qasm_orig_depth"]
    result["qasm_delta_two_qubit_fraction"] = result["qasm_mut_two_plus_fraction"] - result["qasm_orig_two_plus_fraction"]
    result["source_gate_count_match"] = int(result["qasm_mut_gate_count"] == expected_gate_count)
    return result


def extract(metadata: Path, qasm_root: Path, output: Path, *, chunk_rows: int,
            min_available_mib: int, max_chunks: int | None = None) -> dict[str, Any]:
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be >= 1")
    if not metadata.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata}")
    if not qasm_root.is_dir():
        raise FileNotFoundError(f"QASM root not found: {qasm_root}")
    output.mkdir(parents=True, exist_ok=True)

    metadata_sha = sha256_file(metadata)
    parser_path = ROOT / "qmutbench_validation.py"
    parser_sha = sha256_file(parser_path)
    extractor_sha = sha256_file(Path(__file__))
    cache_payload = {
        "schema": "qmutbench-full-qasm-features-v1",
        "metadata_sha256": metadata_sha,
        "parser_sha256": parser_sha,
        "extractor_sha256": extractor_sha,
        "metadata_path": str(metadata.resolve()),
        "qasm_root": str(qasm_root.resolve()),
        "feature_extractor": "experiments.qmutbench_qasm_features.circuit_features",
        "depth_definition": "ASAP per-wire native gate layers; barriers synchronize; measurements excluded",
        "chunk_rows": chunk_rows,
    }
    cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cache_key") != cache_key:
            raise ValueError(
                f"Existing manifest in {output} belongs to different inputs/configuration. "
                "Choose a new output directory; existing chunks were not modified."
            )
    else:
        manifest = {
            **cache_payload, "cache_key": cache_key, "status": "in_progress",
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "expected_rows": EXPECTED_ROWS, "expected_origins": EXPECTED_ORIGINS,
            "chunks": [], "records_requested": 0, "feature_rows": 0,
            "parse_error_rows": 0, "original_parse_failures": 0,
        }
        atomic_json(manifest_path, manifest)

    completed: dict[int, dict[str, Any]] = {}
    for item in manifest.get("chunks", []):
        path = output / item["file"]
        if path.exists() and sha256_file(path) == item.get("sha256"):
            completed[int(item["chunk_index"])] = item
        else:
            raise ValueError(f"Completed chunk missing or checksum mismatch: {path}")

    original_cache: dict[str, dict[str, Any] | None] = {}
    original_failures: dict[str, str] = {}
    records_requested = feature_rows = parse_error_rows = 0
    labels_seen: dict[str, int] = {}
    origins_seen: set[str] = set()
    started = time.monotonic()
    reader = pd.read_csv(metadata, chunksize=chunk_rows, low_memory=False)
    try:
        for chunk_index, frame in enumerate(reader):
            if max_chunks is not None and chunk_index >= max_chunks:
                break
            require_ram_floor(min_available_mib)
            missing = sorted(REQUIRED - set(frame.columns))
            if missing:
                raise ValueError(f"Metadata schema missing columns: {missing}")
            if frame["Unnamed: 0"].isna().any() or frame["Unnamed: 0"].duplicated().any():
                raise ValueError(f"Missing or duplicate record IDs in metadata chunk {chunk_index}")
            if not set(frame["operation"].astype(str).unique()) <= MUTATION_OPERATIONS:
                raise ValueError(f"Unexpected mutation operation in chunk {chunk_index}")
            if not set(frame["Killed"].astype(str).unique()) <= LABELS:
                raise ValueError(f"Unexpected label in chunk {chunk_index}")

            for label, count in frame["Killed"].astype(str).value_counts().items():
                labels_seen[label] = labels_seen.get(label, 0) + int(count)
            origins_seen.update(frame["Origin_program"].astype(str).unique())
            chunk_name = f"features_{chunk_index:05d}.csv.gz"
            error_name = f"errors_{chunk_index:05d}.jsonl"
            if chunk_index in completed:
                records_requested += int(completed[chunk_index]["records_requested"])
                feature_rows += int(completed[chunk_index]["feature_rows"])
                parse_error_rows += int(completed[chunk_index]["parse_error_rows"])
                continue

            output_rows: list[dict[str, Any]] = []
            errors: list[dict[str, Any]] = []
            for raw in frame.to_dict(orient="records"):
                require_ram_floor(min_available_mib)
                record_id = int(raw["Unnamed: 0"])
                origin_name = str(raw["Origin_program"])
                try:
                    original_rel, mutant_rel = safe_relative_paths(raw)
                    original_path = qasm_root / original_rel
                    mutant_path = qasm_root / mutant_rel
                    if not original_path.is_file():
                        raise FileNotFoundError(f"Missing original circuit: {original_rel}")
                    if not mutant_path.is_file():
                        raise FileNotFoundError(f"Missing mutant circuit: {mutant_rel}")
                    if original_rel not in original_cache:
                        try:
                            original_cache[original_rel] = circuit_features(original_path)
                        except Exception as exc:
                            original_cache[original_rel] = None
                            original_failures[original_rel] = f"{type(exc).__name__}: {exc}"
                    original_features = original_cache[original_rel]
                    if original_features is None:
                        raise ValueError(f"Original circuit parse failure: {original_failures[original_rel]}")
                    mutant_features = circuit_features(mutant_path)
                    output_rows.append(_feature_row(
                        record_id, origin_name, original_rel, mutant_rel, original_features,
                        mutant_features, int(raw["gates"]),
                    ))
                except Exception as exc:
                    errors.append({
                        "record_id": record_id, "origin_program": origin_name,
                        "error_type": type(exc).__name__, "error": str(exc),
                    })

            part = pd.DataFrame(output_rows)
            atomic_csv_gzip(part, output / chunk_name)
            error_path = output / error_name
            tmp_error = error_path.with_suffix(error_path.suffix + ".tmp")
            tmp_error.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in errors),
                encoding="utf-8",
            )
            os.replace(tmp_error, error_path)
            chunk_record = {
                "chunk_index": chunk_index, "file": chunk_name, "error_file": error_name,
                "records_requested": int(len(frame)), "feature_rows": int(len(output_rows)),
                "parse_error_rows": int(len(errors)), "sha256": sha256_file(output / chunk_name),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "available_ram_mib_after": round(available_ram_bytes() / 1024**2, 1),
            }
            completed[chunk_index] = chunk_record
            records_requested += len(frame)
            feature_rows += len(output_rows)
            parse_error_rows += len(errors)
            manifest["chunks"] = [completed[i] for i in sorted(completed)]
            manifest.update({
                "records_requested": records_requested,
                "feature_rows": feature_rows,
                "parse_error_rows": parse_error_rows,
                "labels_seen": labels_seen,
                "origins_seen_so_far": len(origins_seen),
                "original_parse_failures": len(original_failures),
                "original_parse_failure_examples": dict(list(original_failures.items())[:25]),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "status": "in_progress",
            })
            atomic_json(manifest_path, manifest)
            print(
                f"chunk={chunk_index} rows={len(frame)} features={len(output_rows)} "
                f"errors={len(errors)} elapsed={time.monotonic()-started:.1f}s "
                f"available_ram={available_ram_bytes()/1024**2:.1f}MiB",
                flush=True,
            )

    except MemoryError as exc:
        manifest.update({
            "status": "paused_resource_guard",
            "pause_reason": str(exc),
            "last_updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "chunks": [completed[i] for i in sorted(completed)],
        })
        atomic_json(manifest_path, manifest)
        print(f"PAUSED SAFELY: {exc}", file=sys.stderr, flush=True)
        return manifest

    # Resume validation uses all committed chunk rows and their hashes; no target labels enter feature files.
    all_chunk_rows = sum(item["feature_rows"] for item in completed.values())
    all_requested = sum(item["records_requested"] for item in completed.values())
    all_errors = sum(item["parse_error_rows"] for item in completed.values())
    manifest.update({
        "status": "complete" if all_requested == EXPECTED_ROWS and all_errors == 0 else
                  "complete_with_errors" if all_requested == EXPECTED_ROWS else "partial",
        "records_requested": all_requested,
        "feature_rows": all_chunk_rows,
        "parse_error_rows": all_errors,
        "labels_seen": labels_seen,
        "origins_seen": len(origins_seen),
        "original_parse_failures": len(original_failures),
        "original_parse_failure_examples": dict(list(original_failures.items())[:25]),
        "elapsed_seconds_this_run": round(time.monotonic() - started, 3),
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--qasm-root", type=Path, default=DEFAULT_QASM_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-rows", type=int, default=500)
    parser.add_argument("--min-available-mib", type=int, default=1024)
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional smoke test; leaves manifest partial.")
    args = parser.parse_args()
    result = extract(args.metadata, args.qasm_root, args.output_dir,
                     chunk_rows=args.chunk_rows,
                     min_available_mib=args.min_available_mib,
                     max_chunks=args.max_chunks)
    print(json.dumps({
        "status": result.get("status"),
        "records_requested": result.get("records_requested"),
        "feature_rows": result.get("feature_rows"),
        "parse_error_rows": result.get("parse_error_rows"),
        "cache_key": result.get("cache_key"),
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
