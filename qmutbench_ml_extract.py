"""Representative, bounded QMutBench extraction for the current simulator.

Reads the source CSV without modifying it, stratifies eligible rows, runs a
200-record gate first, then continues toward 10,000 successful mutants only
when pilot success is at least 80 percent. Metadata and ML features are saved
separately so mutation metadata cannot be mistaken for model inputs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from qmutbench_validation import parse_qasm_file  # noqa: E402
from quantum_simulator import QuantumSimulator  # noqa: E402

PROJECT_SOURCE_CSV = ROOT / "data" / "qmutbench_full" / "all_data.csv"
SOURCE_CSV = PROJECT_SOURCE_CSV
QASM_ROOT = ROOT / "data" / "qmutbench_full" / "data"
OUTPUT_ROOT = ROOT / "data" / "qmutbench_ml" / "representative_10k"
OFFICIAL_BASE = "https://enautmendi.github.io/QMutBench/data/"
RANDOM_STATE = 42
MAX_QUBITS = 12
PILOT_SIZE = 200
SUCCESS_THRESHOLD = 0.80
TARGET_SUCCESSFUL_MUTANTS = 10_000
CSV_CHUNK_SIZE = 100_000
EXECUTION_BATCH_SIZE = 100

SOURCE_COLUMNS = [
    "algorithm", "qubits", "gates", "operation", "gate", "Position_int",
    "position", "Killed", "Origin_program",
]
METADATA_COLUMNS = [
    "record_type", "label", "algorithm", "qubits", "gates", "operation",
    "gate", "Position_int", "position", "Killed", "original_circuit_id",
    "qmutbench_row_index", "original_qasm_path", "mutant_qasm_path",
]
FEATURE_COLUMNS = [
    "trace_max_probability_difference",
    "trace_mean_probability_difference",
    "trace_median_probability_difference",
    "trace_deviation_concentration",
    "trace_high_deviation_fraction",
]
FEATURE_FILE_COLUMNS = ["record_type", "label", "qmutbench_row_index", *FEATURE_COLUMNS]
KEY_METADATA = [
    "qmutbench_row_index", "algorithm", "qubits", "gates", "operation",
    "gate", "Position_int", "position", "Killed", "original_circuit_id",
    "original_qasm_path", "mutant_qasm_path",
]


class UnsupportedOperationError(ValueError):
    pass


def _read_eligible_rows(source: Path) -> tuple[pd.DataFrame, dict[str, Any], str]:
    counts: Counter[str] = Counter()
    eligible_parts: list[pd.DataFrame] = []
    offset = 0
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    for chunk in pd.read_csv(source, usecols=SOURCE_COLUMNS, chunksize=CSV_CHUNK_SIZE):
        n = len(chunk)
        chunk.insert(0, "qmutbench_row_index", np.arange(offset, offset + n, dtype=np.int64))
        offset += n
        counts["source_rows"] += n
        qubits = pd.to_numeric(chunk["qubits"], errors="coerce")
        valid = qubits.notna() & (qubits > 0)
        over = valid & (qubits > MAX_QUBITS)
        counts["excluded_over_12_qubits"] += int(over.sum())
        counts["excluded_invalid_qubit_count"] += int((~valid).sum())
        supported_mutation = chunk["operation"].isin(["Add", "Remove", "Replace"])
        counts["excluded_unsupported_mutation_type"] += int((valid & ~over & ~supported_mutation).sum())
        mask = valid & ~over & supported_mutation
        if mask.any():
            part = chunk.loc[mask].copy()
            part["qubits"] = qubits.loc[mask].astype(int)
            part["gates"] = pd.to_numeric(part["gates"], errors="coerce").fillna(0).astype(int)
            part["Position_int"] = pd.to_numeric(part["Position_int"], errors="coerce")
            part = part[part["Position_int"].notna()].copy()
            part["Position_int"] = part["Position_int"].astype(int)
            eligible_parts.append(part)
            counts["eligible_rows_qubits_1_to_12"] += len(part)
    eligible = pd.concat(eligible_parts, ignore_index=True)
    # Five normalized position bands preserve early, middle, late, and boundary variety.
    denominator = eligible["gates"].clip(lower=1)
    normalized_position = (eligible["Position_int"] / denominator).clip(0, 1)
    eligible["_position_band"] = np.minimum((normalized_position * 5).astype(int), 4)
    counts["excluded_invalid_position"] = counts["source_rows"] - counts[
        "excluded_over_12_qubits"
    ] - counts["excluded_invalid_qubit_count"] - counts[
        "excluded_unsupported_mutation_type"
    ] - counts["eligible_rows_qubits_1_to_12"]
    counts["eligible_original_circuits"] = int(eligible["Origin_program"].nunique())
    counts["eligible_algorithms"] = int(eligible["algorithm"].nunique())
    counts["eligible_mutation_operations"] = {
        str(key): int(value)
        for key, value in eligible["operation"].value_counts().sort_index().items()
    }
    counts["eligible_qubit_counts"] = {
        str(int(key)): int(value)
        for key, value in eligible["qubits"].value_counts().sort_index().items()
    }
    return eligible, dict(counts), digest.hexdigest()


def _apportion(size: int, groups: list[pd.DataFrame]) -> list[int]:
    """Return deterministic proportional integer quotas, bounded by group size."""
    if not groups or size <= 0:
        return [0] * len(groups)
    capacities = np.array([len(group) for group in groups], dtype=np.int64)
    size = min(size, int(capacities.sum()))
    quotas = np.zeros(len(groups), dtype=np.int64)
    if size >= len(groups):
        quotas[:] = 1
        remaining = size - len(groups)
        weights = capacities - 1
    else:
        remaining = size
        weights = capacities.copy()
    while remaining:
        available = np.flatnonzero((capacities - quotas) > 0)
        if not len(available):
            break
        local_weights = weights[available].astype(float)
        if local_weights.sum() == 0:
            local_weights = (capacities[available] - quotas[available]).astype(float)
        exact = remaining * local_weights / local_weights.sum()
        add = np.minimum(np.floor(exact).astype(np.int64), capacities[available] - quotas[available])
        if add.sum():
            quotas[available] += add
            remaining -= int(add.sum())
            continue
        order = sorted(range(len(available)), key=lambda j: (-float(exact[j] % 1), j))
        for j in order:
            if not remaining:
                break
            idx = available[j]
            if quotas[idx] < capacities[idx]:
                quotas[idx] += 1
                remaining -= 1
    return [int(quota) for quota in quotas]


def select_representative(frame: pd.DataFrame, size: int, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Stratify by operation/algorithm, then qubits/gate/position band."""
    if size > len(frame):
        size = len(frame)
    primary = [group for _, group in frame.groupby(["operation", "algorithm"], sort=True, dropna=False)]
    primary_quotas = _apportion(size, primary)
    selected: list[pd.DataFrame] = []
    group_number = 0
    for primary_group, primary_quota in zip(primary, primary_quotas):
        if not primary_quota:
            group_number += 1
            continue
        secondary = [
            group for _, group in primary_group.groupby(
                ["qubits", "gate", "_position_band"], sort=True, dropna=False
            )
        ]
        secondary_quotas = _apportion(primary_quota, secondary)
        for stratum, quota in zip(secondary, secondary_quotas):
            if quota:
                selected.append(stratum.sample(n=quota, random_state=seed + group_number))
            group_number += 1
    result = pd.concat(selected, ignore_index=True)
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


def original_rel(row: pd.Series) -> str:
    # QMutBench's source ID has a suffix such as ``_indep_qiskit``; repository
    # filenames use the algorithm and width instead.
    return f"Origin_programs/{row.algorithm}_{int(row.qubits)}_qubits.qasm"


def mutant_rel(row: pd.Series) -> str:
    stem = f"Mutated_programs/Mutants_{row.algorithm}_{int(row.qubits)}_qubits/"
    gate, position, operation = str(row.gate), int(row.Position_int), str(row.operation)
    if operation == "Add":
        name = f"AddGate_{gate}_inGap_{position}_.qasm"
    elif operation == "Remove":
        name = f"RemoveGate_{position}_.qasm"
    elif operation == "Replace":
        name = f"ReplaceGate_{gate}_inPositionOfGate_{position}.qasm"
    else:
        raise ValueError(f"Unexpected QMutBench mutation operation: {operation}")
    return stem + name


def _resolve_qasm(relative: str) -> Path:
    source_path = (QASM_ROOT / relative).resolve()
    qasm_root = QASM_ROOT.resolve()
    if qasm_root not in source_path.parents:
        raise ValueError(f"Invalid QASM relative path: {relative}")
    if source_path.is_file():
        return source_path
    cache_root = OUTPUT_ROOT / "qasm_cache"
    cached = (cache_root / relative).resolve()
    if cache_root.resolve() not in cached.parents:
        raise ValueError(f"Invalid cache path: {relative}")
    cached.parent.mkdir(parents=True, exist_ok=True)
    url = OFFICIAL_BASE + relative.replace("\\", "/")
    request = urllib.request.Request(url, headers={"User-Agent": "QMutBench-ML-extract/2.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response, cached.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except (OSError, urllib.error.URLError) as exc:
        cached.unlink(missing_ok=True)
        raise FileNotFoundError(f"Unable to retrieve {relative}: {exc}") from exc
    return cached


def _trace_features(trace: list[dict[str, Any]]) -> dict[str, float]:
    differences = [float(item["probability_difference"]) for item in trace]
    if not differences:
        raise ValueError("Simulator produced an empty trace")
    total = sum(differences)
    ordered = sorted(differences)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "trace_max_probability_difference": max(differences),
        "trace_mean_probability_difference": total / len(differences),
        "trace_median_probability_difference": float(median),
        "trace_deviation_concentration": max(differences) / total if total else 0.0,
        "trace_high_deviation_fraction": sum(value > 0.05 for value in differences) / len(differences),
    }


def _normal_metadata(row: pd.Series, org_rel: str) -> dict[str, Any]:
    return {
        "record_type": "original", "label": "NORMAL", "algorithm": str(row.algorithm),
        "qubits": int(row.qubits), "gates": int(row.gates), "operation": "",
        "gate": "", "Position_int": "", "position": "", "Killed": "",
        "original_circuit_id": str(row.Origin_program),
        "qmutbench_row_index": int(row.qmutbench_row_index),
        "original_qasm_path": org_rel, "mutant_qasm_path": "",
    }


def _mutant_metadata(row: pd.Series, org_rel: str, mut_rel: str) -> dict[str, Any]:
    return {
        "record_type": "mutant", "label": "PROGRAM_FAULT", "algorithm": str(row.algorithm),
        "qubits": int(row.qubits), "gates": int(row.gates), "operation": str(row.operation),
        "gate": str(row.gate), "Position_int": int(row.Position_int),
        "position": row.position, "Killed": row.Killed,
        "original_circuit_id": str(row.Origin_program),
        "qmutbench_row_index": int(row.qmutbench_row_index),
        "original_qasm_path": org_rel, "mutant_qasm_path": mut_rel,
    }


def _failure_category(exc: Exception) -> tuple[str, str]:
    match = re.search(r"(?:unsupported simulator operation|Unsupported operation in circuit):?\s*['\"]?([a-zA-Z0-9_]+)", str(exc))
    if match:
        return "unsupported_operation", match.group(1).lower()
    if isinstance(exc, (FileNotFoundError, OSError)):
        return "qasm_unavailable", ""
    return "execution_or_parse_failure", ""


def _execute_row(
    row: pd.Series,
    original_cache: dict[str, tuple[list[tuple[Any, ...]], int]],
    normal_cache: dict[str, dict[str, float]],
) -> dict[str, Any]:
    org_rel, mut_rel = original_rel(row), mutant_rel(row)
    try:
        if org_rel not in original_cache:
            org_ops, org_n = parse_qasm_file(_resolve_qasm(org_rel))
            if org_n > MAX_QUBITS:
                raise ValueError(f"original QASM declares {org_n} qubits, over safe limit {MAX_QUBITS}")
            if org_n != int(row.qubits):
                raise ValueError(f"original QASM width {org_n} != metadata width {int(row.qubits)}")
            original_cache[org_rel] = (org_ops, org_n)
        org_ops, org_n = original_cache[org_rel]
        mut_ops, mut_n = parse_qasm_file(_resolve_qasm(mut_rel))
        if mut_n > MAX_QUBITS:
            raise ValueError(f"mutant QASM declares {mut_n} qubits, over safe limit {MAX_QUBITS}")
        if mut_n != org_n:
            raise ValueError(f"QASM width mismatch: original={org_n}, mutant={mut_n}")
        _, _, _, trace = QuantumSimulator(mut_n, seed=RANDOM_STATE + int(row.qmutbench_row_index)).run_circuit_traced(
            org_ops, mut_ops, seed=RANDOM_STATE + int(row.qmutbench_row_index)
        )
        mutant_features = _trace_features(trace)
        if org_rel not in normal_cache:
            _, _, _, normal_trace = QuantumSimulator(org_n, seed=RANDOM_STATE).run_circuit_traced(
                org_ops, org_ops, seed=RANDOM_STATE
            )
            normal_cache[org_rel] = _trace_features(normal_trace)
        return {
            "status": "success", "metadata": _mutant_metadata(row, org_rel, mut_rel),
            "features": mutant_features, "normal_metadata": _normal_metadata(row, org_rel),
            "normal_features": normal_cache[org_rel],
        }
    except Exception as exc:
        category, unsupported = _failure_category(exc)
        return {
            "status": category, "unsupported_operation": unsupported,
            "failure": {
                "qmutbench_row_index": int(row.qmutbench_row_index),
                "original_circuit_id": str(row.Origin_program),
                "qubits": int(row.qubits), "operation": str(row.operation),
                "gate": str(row.gate), "Position_int": int(row.Position_int),
                "original_qasm_path": org_rel, "mutant_qasm_path": mut_rel,
                "failure_type": type(exc).__name__, "failure_category": category,
                "unsupported_operation": unsupported, "error": f"{type(exc).__name__}: {exc}",
            },
        }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_pilot(results: list[dict[str, Any]], selected: pd.DataFrame, source_stats: dict[str, Any], source_hash: str) -> dict[str, Any]:
    pilot_dir = OUTPUT_ROOT / "pilot_200"
    successful = [item for item in results if item["status"] == "success"]
    features: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    normal_seen: set[str] = set()
    for item in successful:
        mid = int(item["metadata"]["qmutbench_row_index"])
        features.append({"record_type": "mutant", "label": "PROGRAM_FAULT", "qmutbench_row_index": mid, **item["features"]})
        metadata.append(item["metadata"])
        org = str(item["metadata"]["original_qasm_path"])
        if org not in normal_seen:
            normal_seen.add(org)
            features.append({"record_type": "original", "label": "NORMAL", "qmutbench_row_index": mid, **item["normal_features"]})
            metadata.append(item["normal_metadata"])
    failures = [item["failure"] for item in results if item["status"] != "success"]
    _write_csv(pilot_dir / "features.csv", features, FEATURE_FILE_COLUMNS)
    _write_csv(pilot_dir / "metadata.csv", metadata, METADATA_COLUMNS)
    _write_csv(pilot_dir / "failures.csv", failures, [
        "qmutbench_row_index", "original_circuit_id", "qubits", "operation", "gate", "Position_int",
        "original_qasm_path", "mutant_qasm_path", "failure_type", "failure_category",
        "unsupported_operation", "error",
    ])
    pilot_unsupported = Counter(item["unsupported_operation"] for item in results if item.get("unsupported_operation"))
    op_counts = Counter(str(item["metadata"]["operation"]) for item in successful)
    qubit_counts = Counter(str(item["metadata"]["qubits"]) for item in successful)
    summary = {
        "selected_records": len(selected),
        "processed_records": len(results),
        "pilot_selected_records": len(selected), "pilot_processed_records": len(results),
        "pilot_successful_mutants": len(successful), "pilot_failed_records": len(failures),
        "pilot_success_rate": len(successful) / len(results) if results else 0.0,
        "pilot_failure_breakdown": dict(Counter(item["status"] for item in results if item["status"] != "success")),
        "pilot_unsupported_operation_counts": dict(pilot_unsupported),
        "unsupported_operation_record_count": sum(pilot_unsupported.values()),
        "pilot_mutation_operation_counts": dict(sorted(op_counts.items())),
        "pilot_qubit_counts": dict(sorted(qubit_counts.items(), key=lambda pair: int(pair[0]))),
        "pilot_unique_original_circuits": len(normal_seen),
        "pilot_feature_columns": FEATURE_COLUMNS,
        "feature_exclusion_policy": "Only FEATURE_COLUMNS enter features.csv. Mutation operation, gate, position, Killed, algorithm, and qubit metadata are stored in metadata.csv.",
        "source_csv": str(SOURCE_CSV), "source_csv_sha256": source_hash,
        "random_state": RANDOM_STATE, "safe_qubit_limit": MAX_QUBITS,
        "population_counts": source_stats,
        "over_12_qubit_exclusion_count": source_stats["excluded_over_12_qubits"],
        "pilot_selected_row_indices": [int(value) for value in selected.qmutbench_row_index],
        "pilot_features_path": str((pilot_dir / "features.csv").resolve()),
        "pilot_metadata_path": str((pilot_dir / "metadata.csv").resolve()),
        "pilot_failures_path": str((pilot_dir / "failures.csv").resolve()),
    }
    (pilot_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _select_target_pool(eligible: pd.DataFrame, pilot: pd.DataFrame, wanted: int) -> pd.DataFrame:
    wanted = min(wanted, len(eligible))
    pilot_ids = set(int(value) for value in pilot.qmutbench_row_index)
    remainder = eligible[~eligible.qmutbench_row_index.isin(pilot_ids)]
    fill = select_representative(remainder, max(0, wanted - len(pilot)), RANDOM_STATE)
    combined = pd.concat([pilot, fill], ignore_index=True)
    return combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def run(source: Path = SOURCE_CSV) -> dict[str, Any]:
    global SOURCE_CSV
    SOURCE_CSV = source
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    eligible, source_stats, source_hash = _read_eligible_rows(source)
    if len(eligible) < PILOT_SIZE:
        raise ValueError(f"Only {len(eligible)} eligible records are available for a {PILOT_SIZE}-record pilot")

    pilot = select_representative(eligible, PILOT_SIZE, RANDOM_STATE)
    original_cache: dict[str, tuple[list[tuple[Any, ...]], int]] = {}
    normal_cache: dict[str, dict[str, float]] = {}
    pilot_results: list[dict[str, Any]] = []
    for start in range(0, len(pilot), EXECUTION_BATCH_SIZE):
        for _, row in pilot.iloc[start:start + EXECUTION_BATCH_SIZE].iterrows():
            pilot_results.append(_execute_row(row, original_cache, normal_cache))
        print(f"pilot processed {min(start + EXECUTION_BATCH_SIZE, len(pilot))}/{len(pilot)}")

    pilot_manifest = _save_pilot(pilot_results, pilot, source_stats, source_hash)
    print(json.dumps({key: value for key, value in pilot_manifest.items() if key.startswith("pilot_")}, indent=2))
    if pilot_manifest["pilot_success_rate"] < SUCCESS_THRESHOLD:
        pilot_manifest["full_extraction_started"] = False
        (OUTPUT_ROOT / "pilot_200" / "manifest.json").write_text(json.dumps(pilot_manifest, indent=2) + "\n", encoding="utf-8")
        print(f"STOP: pilot success rate {pilot_manifest['pilot_success_rate']:.1%} is below 80%; full extraction was not started.")
        return pilot_manifest

    # The current user instruction is pilot-only; even a passing pilot must stop here.
    pilot_manifest["full_extraction_started"] = False
    pilot_manifest["pilot_gate_passed"] = True
    pilot_manifest["next_step"] = "Pilot passed; full extraction was not started because this run is explicitly pilot-only."
    (OUTPUT_ROOT / "pilot_200" / "manifest.json").write_text(json.dumps(pilot_manifest, indent=2) + "\n", encoding="utf-8")
    print("Pilot passed. Stopped after 200 records as requested; full extraction was not started.")
    return pilot_manifest

    estimated_candidates = min(
        len(eligible),
        max(PILOT_SIZE, math.ceil(TARGET_SUCCESSFUL_MUTANTS / pilot_manifest["pilot_success_rate"])),
    )
    selected = _select_target_pool(eligible, pilot, estimated_candidates)
    results_by_index = {
        int(row.qmutbench_row_index): result for row, result in zip(pilot.itertuples(index=False), pilot_results)
    }
    selected_ids = set(int(value) for value in selected.qmutbench_row_index)
    # Pilot outcomes are reused only when a pilot row is in the final reproducible pool.
    results_by_index = {key: value for key, value in results_by_index.items() if key in selected_ids}
    for start in range(0, len(selected), EXECUTION_BATCH_SIZE):
        batch = selected.iloc[start:start + EXECUTION_BATCH_SIZE]
        for _, row in batch.iterrows():
            idx = int(row.qmutbench_row_index)
            if idx not in results_by_index:
                results_by_index[idx] = _execute_row(row, original_cache, normal_cache)
        print(f"full candidate pool processed {min(start + EXECUTION_BATCH_SIZE, len(selected))}/{len(selected)}")

    ordered_results = [results_by_index[int(row.qmutbench_row_index)] for _, row in selected.iterrows()]
    successful = [item for item in ordered_results if item["status"] == "success"]
    # If the pilot estimate was optimistic, draw a further reproducible stratum-balanced tranche.
    next_seed = RANDOM_STATE + 1
    while len(successful) < TARGET_SUCCESSFUL_MUTANTS and len(selected) < len(eligible):
        remaining = eligible[~eligible.qmutbench_row_index.isin(set(int(x) for x in selected.qmutbench_row_index))]
        need = TARGET_SUCCESSFUL_MUTANTS - len(successful)
        tranche_size = min(len(remaining), max(500, math.ceil(need / max(pilot_manifest["pilot_success_rate"], 0.01))))
        tranche = select_representative(remaining, tranche_size, next_seed)
        next_seed += 1
        new_results = []
        for start in range(0, len(tranche), EXECUTION_BATCH_SIZE):
            for _, row in tranche.iloc[start:start + EXECUTION_BATCH_SIZE].iterrows():
                result = _execute_row(row, original_cache, normal_cache)
                idx = int(row.qmutbench_row_index)
                results_by_index[idx] = result
                new_results.append(result)
            print(f"reserve tranche processed {min(start + EXECUTION_BATCH_SIZE, len(tranche))}/{len(tranche)}")
        selected = pd.concat([selected, tranche], ignore_index=True)
        ordered_results.extend(new_results)
        successful = [item for item in ordered_results if item["status"] == "success"]

    # Preserve every attempted candidate outcome. No rejected row disappears from the ledger.
    successes = [item for item in ordered_results if item["status"] == "success"]
    failure_results = [item for item in ordered_results if item["status"] != "success"]
    feature_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    normal_seen: set[str] = set()
    for item in successes:
        metadata = item["metadata"]
        idx = int(metadata["qmutbench_row_index"])
        feature_rows.append({"record_type": "mutant", "label": "PROGRAM_FAULT", "qmutbench_row_index": idx, **item["features"]})
        metadata_rows.append(metadata)
        org_rel = str(metadata["original_qasm_path"])
        if org_rel not in normal_seen:
            normal_seen.add(org_rel)
            feature_rows.append({"record_type": "original", "label": "NORMAL", "qmutbench_row_index": idx, **item["normal_features"]})
            metadata_rows.append(item["normal_metadata"])

    _write_csv(OUTPUT_ROOT / "features.csv", feature_rows, FEATURE_FILE_COLUMNS)
    _write_csv(OUTPUT_ROOT / "metadata.csv", metadata_rows, METADATA_COLUMNS)
    failures = [item["failure"] for item in failure_results]
    _write_csv(OUTPUT_ROOT / "failures.csv", failures, [
        "qmutbench_row_index", "original_circuit_id", "qubits", "operation", "gate", "Position_int",
        "original_qasm_path", "mutant_qasm_path", "failure_type", "failure_category",
        "unsupported_operation", "error",
    ])
    unsupported_counts = Counter(item.get("unsupported_operation", "") for item in failure_results if item.get("unsupported_operation"))
    operation_counts = Counter(str(item["metadata"]["operation"]) for item in successes)
    qubit_counts = Counter(str(item["metadata"]["qubits"]) for item in successes)
    ledger_rows = []
    for row, outcome in zip(selected.itertuples(index=False), ordered_results):
        ledger_rows.append({
            "qmutbench_row_index": int(row.qmutbench_row_index),
            "original_circuit_id": str(row.Origin_program),
            "algorithm": str(row.algorithm), "qubits": int(row.qubits),
            "operation": str(row.operation), "gate": str(row.gate),
            "Position_int": int(row.Position_int), "position": str(row.position),
            "status": outcome["status"],
            "unsupported_operation": outcome.get("unsupported_operation", ""),
            "error": outcome.get("failure", {}).get("error", ""),
        })
    _write_csv(OUTPUT_ROOT / "selection_ledger.csv", ledger_rows, [
        "qmutbench_row_index", "original_circuit_id", "algorithm", "qubits", "operation",
        "gate", "Position_int", "position", "status", "unsupported_operation", "error",
    ])
    status_counts = Counter(item["status"] for item in ordered_results)
    full_manifest = {
        "dataset": "QMutBench", "source_csv": str(source), "source_csv_sha256": source_hash,
        "random_state": RANDOM_STATE, "safe_qubit_limit": MAX_QUBITS,
        "pilot_success_rate": pilot_manifest["pilot_success_rate"],
        "pilot_threshold": SUCCESS_THRESHOLD, "pilot_manifest": str((OUTPUT_ROOT / "pilot_200" / "manifest.json").resolve()),
        "target_successful_mutants": TARGET_SUCCESSFUL_MUTANTS,
        "candidate_target_from_pilot": estimated_candidates,
        "population_counts": source_stats,
        "selected_candidate_records": len(selected),
        "processed_candidate_records": len(ordered_results),
        "successful_mutants": len(successes),
        "failed_candidate_records": len(failure_results),
        "processed_status_counts": dict(status_counts),
        "unsupported_operation_record_count": sum(unsupported_counts.values()),
        "unsupported_operation_counts": dict(sorted(unsupported_counts.items())),
        "over_12_qubit_exclusion_count": source_stats["excluded_over_12_qubits"],
        "unique_original_circuits_with_successful_mutants": len(normal_seen),
        "successful_mutation_operation_distribution": dict(sorted(operation_counts.items())),
        "successful_qubit_count_distribution": dict(sorted(qubit_counts.items(), key=lambda pair: int(pair[0]))),
        "feature_columns_only": FEATURE_COLUMNS,
        "label_column": "label (target only)", "record_id_column": "qmutbench_row_index (key only)",
        "metadata_only_columns": [column for column in METADATA_COLUMNS if column not in {"label", "record_type", "qmutbench_row_index"}],
        "label_policy": "Every successfully executed mutant is PROGRAM_FAULT. Killed remains metadata. Unique corresponding originals are emitted once as NORMAL.",
        "outputs": {
            "features": str((OUTPUT_ROOT / "features.csv").resolve()),
            "metadata": str((OUTPUT_ROOT / "metadata.csv").resolve()),
            "failures": str((OUTPUT_ROOT / "failures.csv").resolve()),
            "selection_ledger": str((OUTPUT_ROOT / "selection_ledger.csv").resolve()),
        },
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(full_manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(full_manifest, indent=2))
    return full_manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_CSV)
    args = parser.parse_args()
    run(args.source)
