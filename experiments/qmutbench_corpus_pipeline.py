"""Verified, bounded QMutBench ingestion for benchmark-derived mutant data.

This pipeline keeps raw source bytes, normalized metadata, and simulated
observations in separate locations. It does not turn QMutBench's ``Killed``
field into diagnosis truth or claim independent family provenance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from qmutbench_validation import (  # noqa: E402
    QMutBenchParseError,
    QMutBenchUnsupportedError,
    _normalize_circuit,
    parse_qasm_file,
)
from quantum_simulator import QuantumSimulator  # noqa: E402

EXPECTED_COLUMNS = [
    "", "Origin_program", "algorithm", "qubits", "gates", "operation",
    "Position_int", "Killed", "gate", "algorithm_group", "dominant_state", "position",
]
REQUIRED_COLUMNS = set(EXPECTED_COLUMNS) - {""}
DEFAULT_SOURCE_MANIFEST = ROOT / "data/qmutbench_ingestion/source_manifest.json"
MAX_QUBITS = 12
DEFAULT_SHOTS = 256
DEFAULT_SEED = 1701
EQUIVALENCE_TOLERANCE = 1e-12
KNOWN_LEAKAGE_FIELDS = {
    "label", "benchmark_label", "target", "killed", "operation", "gate",
    "position", "position_int", "mutation_id", "mutant_id", "record_type",
    "artifact_role", "original_circuit_id", "source_row_index", "record_id",
    "qmutbench_row_index", "original_qasm_path", "mutant_qasm_path",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_and_verify_source(manifest_path: Path) -> tuple[dict[str, Any], Path, Path, str]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = manifest.get("metadata_csv", {})
    source = (ROOT / dataset.get("path", "")).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"QMutBench metadata CSV is missing: {source}")
    expected_hash = dataset.get("sha256")
    expected_size = dataset.get("bytes")
    actual_hash = sha256_file(source)
    if not expected_hash or actual_hash.lower() != str(expected_hash).lower():
        raise ValueError("QMutBench metadata CSV SHA-256 does not match source_manifest.json")
    if expected_size is not None and source.stat().st_size != int(expected_size):
        raise ValueError("QMutBench metadata CSV byte size does not match source_manifest.json")
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        actual_schema = next(csv.reader(stream), None)
    if actual_schema != EXPECTED_COLUMNS or dataset.get("schema") != actual_schema:
        raise ValueError("QMutBench metadata CSV schema does not match source_manifest.json")
    qasm_folder = manifest.get("qasm_artifacts", {})
    commit = str(qasm_folder.get("pinned_commit", ""))
    qasm_root = (manifest_path.parent / "raw" / f"qasm_{commit}").resolve()
    if not commit or not qasm_root.is_dir():
        raise ValueError("Pinned QMutBench QASM source root is missing")
    return manifest, source, qasm_root, actual_hash


def source_statistics(source: Path) -> dict[str, Any]:
    origins: set[str] = set()
    algorithms: set[str] = set()
    killed: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    qubit_min: int | None = None
    qubit_max: int | None = None
    rows = 0
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected QMutBench columns: {reader.fieldnames!r}")
        for row in reader:
            rows += 1
            origins.add(row["Origin_program"])
            algorithms.add(row["algorithm"])
            operations[row["operation"]] += 1
            killed[row["Killed"]] += 1
            try:
                qubits = int(row["qubits"])
            except (TypeError, ValueError):
                continue
            qubit_min = qubits if qubit_min is None else min(qubit_min, qubits)
            qubit_max = qubits if qubit_max is None else max(qubit_max, qubits)
    return {
        "source_rows": rows,
        "origin_program_ids": len(origins),
        "algorithms": len(algorithms),
        "qubit_range": [qubit_min, qubit_max],
        "operation_counts": dict(sorted(operations.items())),
        "raw_Killed_counts": dict(sorted(killed.items())),
    }


def _safe_part(value: str, field: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_+-]+", value or ""):
        raise ValueError(f"invalid {field} in QMutBench metadata: {value!r}")
    return value


def qasm_relative_paths(row: dict[str, str]) -> tuple[str, str]:
    algorithm = _safe_part(row["algorithm"], "algorithm")
    operation = row["operation"]
    gate = _safe_part(row["gate"], "gate") if row["gate"] else ""
    qubits = int(row["qubits"])
    position = int(row["Position_int"])
    if qubits <= 0 or position < 0:
        raise ValueError("qubit count and mutation position must be non-negative valid integers")
    if operation == "Add":
        filename = f"AddGate_{gate}_inGap_{position}_.qasm"
    elif operation == "Remove":
        filename = f"RemoveGate_{position}_.qasm"
    elif operation == "Replace":
        filename = f"ReplaceGate_{gate}_inPositionOfGate_{position}.qasm"
    else:
        raise ValueError(f"unsupported QMutBench mutation type {operation!r}")
    original = f"Origin_programs/{algorithm}_{qubits}_qubits.qasm"
    mutant = f"Mutated_programs/Mutants_{algorithm}_{qubits}_qubits/{filename}"
    return original, mutant


def _source_file_hash(relative: str, qasm_root: Path, manifest: dict[str, Any]) -> tuple[Path, str]:
    path = (qasm_root / relative).resolve()
    if qasm_root.resolve() not in path.parents:
        raise ValueError("QASM path escaped the pinned source directory")
    if not path.is_file():
        raise FileNotFoundError(f"QMutBench QASM source is missing: {relative}")
    artifacts = manifest.get("qasm_artifacts", {}).get("files", [])
    expected = next((item.get("sha256") for item in artifacts if item.get("path") == relative), None)
    actual = sha256_file(path)
    if not expected:
        raise ValueError(f"QASM source has no pinned hash attestation in source manifest: {relative}")
    if actual.lower() != str(expected).lower():
        raise ValueError(f"QASM source SHA-256 mismatch: {relative}")
    return path, actual


def _gate_only(operations: Iterable[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return [operation for operation in operations if operation and str(operation[0]).lower() != "measure"]


def _validate_supported_operations(operations: list[tuple[Any, ...]], path: Path) -> list[tuple[Any, ...]]:
    """Validate gate names, arity, parameters, and operands before execution."""
    return _normalize_circuit(operations, path.name)


def _simulate_distribution(operations: list[tuple[Any, ...]], qubits: int, seed: int) -> list[float]:
    simulator = QuantumSimulator(qubits, seed=seed)
    simulator.run_circuit(_gate_only(operations))
    return [float(item) for item in simulator.get_probabilities().tolist()]


def _sample_counts(operations: list[tuple[Any, ...]], qubits: int, shots: int, seed: int) -> dict[str, int]:
    simulator = QuantumSimulator(qubits, seed=seed)
    return {str(key): int(value) for key, value in sorted(
        simulator.run_shots(_gate_only(operations), shots=shots, seed=seed).items()
    )}


def _tvd(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("distribution dimensions differ")
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right))


def _counts_tvd(left: dict[str, int], right: dict[str, int]) -> float:
    total_left, total_right = sum(left.values()), sum(right.values())
    if not total_left or not total_right:
        return 0.0
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(k, 0) / total_left - right.get(k, 0) / total_right) for k in keys)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(8):
            try:
                os.replace(temp_name, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(0.1 * (2 ** attempt), 1.0))
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _append_jsonl(path: Path, value: dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        return stream.tell()


def _truncate(path: Path, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    with path.open("r+b") as stream:
        stream.truncate(offset)
        stream.flush()
        os.fsync(stream.fileno())


def _primary_feature_blockers(features: Iterable[str]) -> list[str]:
    return [f"label-leaking or construction metadata is prohibited as a primary feature: {name}"
            for name in features if str(name).strip().lower() in KNOWN_LEAKAGE_FIELDS]


def evaluation_preflight(
    records: Iterable[dict[str, Any]],
    *,
    partitions: dict[str, list[str]],
    primary_features: list[str],
    task_target: str,
    minimum_independent_families: int | None,
    independent_roots_verified: bool,
    labels_independently_verified: bool,
) -> dict[str, Any]:
    blockers: list[str] = []
    required_partitions = {"train", "calibration", "test"}
    if set(partitions) != required_partitions:
        blockers.append("train, calibration, and test root assignments are all required")
    memberships: dict[str, set[str]] = {name: set(partitions.get(name, [])) for name in required_partitions}
    for left, right in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
        overlap = memberships[left] & memberships[right]
        if overlap:
            blockers.append(f"lineage-group overlap between {left} and {right}: {sorted(overlap)}")
    groups: set[str] = set()
    record_count = 0
    missing_hash_count = 0
    for row in records:
        record_count += 1
        if row.get("lineage_group_id"):
            groups.add(str(row["lineage_group_id"]))
        if (not row.get("source_csv_sha256") or not row.get("source_record_sha256")
                or (row.get("qasm_sha256") is None and row.get("artifact_role") == "generated_mutant")):
            missing_hash_count += 1
    assigned_groups = set.union(*memberships.values()) if memberships else set()
    if groups - assigned_groups:
        blockers.append(f"lineage groups are not assigned to a locked partition: {sorted(groups - assigned_groups)}")
    if missing_hash_count:
        blockers.append(f"required source hashes/provenance missing for {missing_hash_count} records")
    if not record_count:
        blockers.append("no normalized records are available")
    if task_target != "benchmark_mutant_artifact_detection":
        blockers.append("QMutBench labels are only compatible with benchmark mutant-artifact detection, not diagnosis categories")
    if not primary_features:
        blockers.append("a preregistered primary feature schema is required")
    if not independent_roots_verified:
        blockers.append("independent circuit-family/recipe roots are not externally verified")
    if not labels_independently_verified:
        blockers.append("labels are benchmark construction roles, not independently adjudicated behavioral truth")
    if minimum_independent_families is None or minimum_independent_families < 1:
        blockers.append("a predeclared minimum eligible-family count is required")
    elif len(groups) < minimum_independent_families:
        blockers.append(f"eligible independent family count {len(groups)} is below required minimum {minimum_independent_families}")
    blockers.extend(_primary_feature_blockers(primary_features))
    if not any(memberships.values()):
        blockers.append("no root assignments have been frozen")
    return {
        "ready": not blockers,
        "records_checked": record_count,
        "task_target": task_target,
        "eligible_independent_families": len(groups) if independent_roots_verified else 0,
        "observed_lineage_groups": sorted(groups),
        "primary_features": primary_features,
        "blockers": blockers,
        "fitting_started": False,
        "independent_generalization_claim_authorized": False,
        "hardware_claim_authorized": False,
    }


def _base_record_id(source_hash: str, source_row_index: int) -> str:
    return f"qmutbench:{source_hash[:16]}:row:{source_row_index:06d}"


def _original_record_id(source_hash: str, origin_id: str) -> str:
    return f"qmutbench:{source_hash[:16]}:origin:{hashlib.sha256(origin_id.encode('utf-8')).hexdigest()[:16]}"


def _failure_status(exc: Exception) -> str:
    message = str(exc).lower()
    if isinstance(exc, FileNotFoundError):
        return "SOURCE_MISSING"
    if isinstance(exc, QMutBenchUnsupportedError):
        return "UNSUPPORTED_OPERATION"
    if isinstance(exc, QMutBenchParseError):
        return "PARSE_REJECTED"
    if "over safe limit" in message or "resource" in message:
        return "RESOURCE_LIMIT_REJECTED"
    if "unsupported operation" in message:
        return "UNSUPPORTED_OPERATION"
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return "INVALID_OR_UNSUPPORTED_RECORD"
    return "SIMULATOR_FAILURE"


def _source_row_digest(row: dict[str, str]) -> str:
    return canonical_sha256(row)


def _source_contract_hashes() -> dict[str, str]:
    return {
        "pipeline_sha256": sha256_file(Path(__file__).resolve()),
        "qasm_parser_sha256": sha256_file(ROOT / "qmutbench_validation.py"),
        "simulator_sha256": sha256_file(ROOT / "src/quantum_simulator/quantum_simulator.py"),
    }


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _process_row(
    row: dict[str, str], source_index: int, manifest: dict[str, Any], source_hash: str,
    qasm_root: Path, shots: int, seed_base: int,
    original_cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    record_id = _base_record_id(source_hash, source_index)
    origin_id = row.get("Origin_program", "")
    row_digest = _source_row_digest(row)
    normalized: dict[str, Any] = {
        "schema": "qmutbench-normalized-record-v1",
        "record_id": record_id,
        "source": "QMutBench",
        "source_commit": manifest.get("source_commit"),
        "source_csv_sha256": source_hash,
        "source_row_index": source_index,
        "source_record_sha256": row_digest,
        "original_circuit_id": origin_id or None,
        "lineage_group_id": origin_id or None,
        "lineage_group_basis": "official Origin_program-to-Mutants_<origin> grouping",
        "family_root_independence_verified": False,
        "label_provenance": "benchmark construction role; not independent behavioral adjudication",
        "diagnosis_category_compatible": False,
        "source_metadata": row,
        "artifact_role": "generated_mutant",
        "benchmark_label": "generated_mutant",
        "execution_status": "PENDING",
        "original_record_id": None,
        "original_qasm_path": None,
        "mutant_qasm_path": None,
        "qasm_sha256": None,
    }
    failure: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None
    original_record: dict[str, Any] | None = None
    try:
        if not origin_id:
            raise ValueError("missing Origin_program")
        if row.get("") not in (None, "") and int(row[""]) != source_index:
            raise ValueError("metadata source index does not match CSV row order")
        metadata_qubits = int(row["qubits"])
        if metadata_qubits <= 0:
            raise ValueError("invalid metadata qubit count")
        if metadata_qubits > MAX_QUBITS:
            raise RuntimeError(f"metadata width {metadata_qubits} over safe limit {MAX_QUBITS}")
        original_rel, mutant_rel = qasm_relative_paths(row)
        normalized.update({
            "algorithm": row["algorithm"], "qubits": metadata_qubits,
            "gate_count_metadata": int(row["gates"]) if str(row.get("gates", "")).isdigit() else None,
            "mutation_operator_metadata": row["operation"],
            "mutation_gate_metadata": row["gate"],
            "mutation_position_index_metadata": int(row["Position_int"]),
            "killed_raw_metadata": row["Killed"],
        })
        original_path, original_hash = _source_file_hash(original_rel, qasm_root, manifest)
        mutant_path, mutant_hash = _source_file_hash(mutant_rel, qasm_root, manifest)
        normalized.update({"original_qasm_path": original_rel, "mutant_qasm_path": mutant_rel,
                           "qasm_sha256": mutant_hash, "original_qasm_sha256": original_hash,
                           "original_record_id": _original_record_id(source_hash, origin_id)})

        if origin_id in original_cache:
            original_cache[origin_id] = original_cache.pop(origin_id)
        if origin_id not in original_cache:
            original_ops, original_width = parse_qasm_file(original_path)
            if original_width != metadata_qubits:
                raise ValueError(f"original QASM width {original_width} differs from metadata {metadata_qubits}")
            original_ops = _validate_supported_operations(original_ops, original_path)
            original_bytes = original_path.stat().st_size
            original_record = {
                "schema": "qmutbench-normalized-record-v1",
                "record_id": _original_record_id(source_hash, origin_id), "source": "QMutBench",
                "source_commit": manifest.get("source_commit"), "source_csv_sha256": source_hash,
                # This source-record digest anchors the original-parent identity
                # to the first metadata row that names it; the original QASM hash
                # remains the direct byte-level identity for the circuit itself.
                "source_record_sha256": row_digest, "source_row_index": source_index,
                "original_circuit_id": origin_id, "lineage_group_id": origin_id,
                "lineage_group_basis": "official Origin_program-to-Mutants_<origin> grouping",
                "family_root_independence_verified": False,
                "label_provenance": "benchmark construction role; not independent behavioral adjudication",
                "diagnosis_category_compatible": False, "artifact_role": "original_reference",
                "benchmark_label": "original_reference", "execution_status": "SUPPORTED",
                "qasm_path": original_rel, "qasm_sha256": original_hash,
                "source_metadata_fields": ["Origin_program", "algorithm", "qubits"],
            }
            original_cache[origin_id] = {
                "ops": original_ops, "width": original_width, "hash": original_hash,
                "path": original_path, "distribution": _simulate_distribution(original_ops, original_width, seed_base),
            }
            # Keep memory bounded if ingestion eventually processes many roots.
            if len(original_cache) > 8:
                original_cache.pop(next(iter(original_cache)))
        parent = original_cache[origin_id]
        mutant_ops, mutant_width = parse_qasm_file(mutant_path)
        mutant_ops = _validate_supported_operations(mutant_ops, mutant_path)
        if mutant_width != parent["width"] or mutant_width != metadata_qubits:
            raise ValueError("mutant/original/metadata QASM widths do not agree")
        estimated_bytes = QuantumSimulator.estimate_working_array_bytes(mutant_width)
        if estimated_bytes > 64 * 1024 * 1024:
            raise RuntimeError(f"estimated working array {estimated_bytes} bytes over 64 MiB cap")

        # Running both circuits validates supported gate semantics before sampling.
        mutant_distribution = _simulate_distribution(mutant_ops, mutant_width, seed_base + source_index)
        original_distribution = parent["distribution"]
        seed = seed_base + source_index
        original_counts = _sample_counts(parent["ops"], parent["width"], shots, seed)
        mutant_counts = _sample_counts(mutant_ops, mutant_width, shots, seed)
        exact_tvd = _tvd(original_distribution, mutant_distribution)
        sampled_tvd = _counts_tvd(original_counts, mutant_counts)
        observation = {
            "schema": "qmutbench-simulated-observation-v1",
            "record_id": record_id,
            "original_record_id": normalized["original_record_id"],
            "source_csv_sha256": source_hash,
            "source_record_sha256": row_digest,
            "source_row_index": source_index,
            "original_qasm_sha256": original_hash,
            "mutant_qasm_sha256": mutant_hash,
            "source_commit": manifest.get("source_commit"),
            "observation_kind": "ideal_statevector_probabilities_and_seeded_shot_counts",
            "noise_settings": {"noise_model": "none", "readout_error": 0.0, "hardware_model": None},
            "shots_per_circuit": shots,
            "seed": seed,
            "seed_strategy": "seed_base_plus_source_row_index; same seed used for paired reference and mutant sampling",
            "measurement": "computational basis; QASM measurement instructions removed before simulation and final state sampled",
            "exact_original_probabilities": original_distribution,
            "exact_mutant_probabilities": mutant_distribution,
            "exact_tvd": exact_tvd,
            "exact_observable_status": "changed" if exact_tvd > EQUIVALENCE_TOLERANCE else "equivalent_at_computational_basis",
            "sampled_original_counts": original_counts,
            "sampled_mutant_counts": mutant_counts,
            "sampled_tvd": sampled_tvd,
            "simulator": {
                "python": platform.python_version(),
                "numpy": _numpy_version(),
                "platform": platform.platform(),
                "source_hashes": _source_contract_hashes(),
                "max_qubits": MAX_QUBITS,
                "preflight_memory_budget_bytes": 64 * 1024 * 1024,
                "simulator_memory_budget_bytes": None,
                "estimated_working_array_bytes": estimated_bytes,
            },
            "interpretation_limit": "simulator-only observation for this measurement basis and configuration; not hardware validity or independent behavioral truth",
        }
        normalized["execution_status"] = "SIMULATED"
    except Exception as exc:  # Every unavailable/rejected source row is retained explicitly.
        status = _failure_status(exc)
        normalized["execution_status"] = status
        normalized["failure_reason"] = f"{type(exc).__name__}: {exc}"
        failure = {
            "schema": "qmutbench-rejection-v1",
            "record_id": record_id, "source_row_index": source_index,
            "original_circuit_id": origin_id or None, "status": status,
            "exception_type": type(exc).__name__, "reason": str(exc),
            "source_csv_sha256": source_hash, "source_record_sha256": row_digest,
        }
    return normalized, original_record, observation, failure


def _numpy_version() -> str:
    import numpy as np
    return str(np.__version__)


def run_ingestion(
    *,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    output_dir: Path,
    start_row_index: int = 0,
    limit: int = 100,
    shots: int = DEFAULT_SHOTS,
    seed_base: int = DEFAULT_SEED,
    resume: bool = False,
    minimum_independent_families: int | None = None,
) -> dict[str, Any]:
    if start_row_index < 0 or limit < 1 or shots < 1:
        raise ValueError("start row must be non-negative and limit/shots must be positive")
    manifest, source, qasm_root, source_hash = load_and_verify_source(source_manifest_path)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = output_dir / "normalized_records.jsonl"
    observations_path = output_dir / "simulated_observations.jsonl"
    failures_path = output_dir / "rejections_and_failures.jsonl"
    checkpoint_path = output_dir / "checkpoint.json"
    config = {
        "source_csv_sha256": source_hash, "source_commit": manifest.get("source_commit"),
        "start_row_index": start_row_index, "limit": limit, "shots": shots,
        "seed_base": seed_base, "max_qubits": MAX_QUBITS,
        "measurement": "computational basis; no noise", "equivalence_tolerance": EQUIVALENCE_TOLERANCE,
        "code_hashes": _source_contract_hashes(),
    }
    config_hash = canonical_sha256(config)
    if resume:
        if not checkpoint_path.is_file():
            raise ValueError("--resume requested but no checkpoint exists")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("config_sha256") != config_hash:
            raise ValueError("resume configuration/source differs from checkpoint")
        offsets = checkpoint.get("output_offsets", {})
        for key, path in (("normalized", normalized_path), ("observations", observations_path), ("failures", failures_path)):
            _truncate(path, int(offsets.get(key, 0)))
        last_completed = int(checkpoint.get("last_completed_source_row_index", start_row_index - 1))
        summary = checkpoint.get("counts", {})
        original_record_ids = set(checkpoint.get("original_record_ids", []))
        selected_groups = set(checkpoint.get("selected_lineage_groups", []))
    else:
        if checkpoint_path.exists() or any(p.exists() and p.stat().st_size for p in (normalized_path, observations_path, failures_path)):
            raise FileExistsError("output directory contains previous ingestion state; choose a new path or use --resume")
        for path in (normalized_path, observations_path, failures_path):
            path.touch()
        last_completed = start_row_index - 1
        summary = {"processed_source_records": 0, "parsed_and_supported_mutant_cases": 0,
                   "simulated_mutant_cases": 0, "successful_pair_executions": 0,
                   "failures": 0, "exact_behavior_changed": 0,
                   "equivalent_at_computational_basis": 0,
                   "max_simulator_working_array_bytes": 0}
        original_record_ids = set()
        selected_groups = set()
    original_cache: dict[str, dict[str, Any]] = {}
    processed_here = 0
    hashes = _source_contract_hashes()
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError("source CSV header does not match the pinned QMutBench schema")
        for source_index, row in enumerate(reader):
            if source_index < start_row_index or source_index <= last_completed:
                continue
            if source_index >= start_row_index + limit:
                break
            normalized, original_record, observation, failure = _process_row(
                row, source_index, manifest, source_hash, qasm_root, shots, seed_base, original_cache
            )
            if original_record and original_record["record_id"] not in original_record_ids:
                _append_jsonl(normalized_path, original_record)
                original_record_ids.add(original_record["record_id"])
            _append_jsonl(normalized_path, normalized)
            if observation:
                _append_jsonl(observations_path, observation)
                summary["parsed_and_supported_mutant_cases"] = int(summary.get("parsed_and_supported_mutant_cases", 0)) + 1
                summary["simulated_mutant_cases"] = int(summary.get("simulated_mutant_cases", 0)) + 1
                summary["successful_pair_executions"] = int(summary.get("successful_pair_executions", 0)) + 2
                summary["max_simulator_working_array_bytes"] = max(
                    int(summary.get("max_simulator_working_array_bytes", 0)),
                    int(observation["simulator"]["estimated_working_array_bytes"]),
                )
                key = "exact_behavior_changed" if observation["exact_observable_status"] == "changed" else "equivalent_at_computational_basis"
                summary[key] = int(summary.get(key, 0)) + 1
            if failure:
                _append_jsonl(failures_path, failure)
                summary["failures"] = int(summary.get("failures", 0)) + 1
            summary["processed_source_records"] = int(summary.get("processed_source_records", 0)) + 1
            if normalized.get("lineage_group_id"):
                selected_groups.add(normalized["lineage_group_id"])
            processed_here += 1
            offsets = {"normalized": normalized_path.stat().st_size,
                       "observations": observations_path.stat().st_size,
                       "failures": failures_path.stat().st_size}
            checkpoint = {
                "config": config, "config_sha256": config_hash,
                "last_completed_source_row_index": source_index,
                "output_offsets": offsets, "counts": summary,
                "original_record_ids": sorted(original_record_ids),
                "selected_lineage_groups": sorted(selected_groups),
            }
            _atomic_json(checkpoint_path, checkpoint)

    # Re-read normalized rows as a stream; do not materialize the corpus in RAM.
    def iter_normalized_records():
        with normalized_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)

    preflight = evaluation_preflight(
        iter_normalized_records(), partitions={"train": [], "calibration": [], "test": []},
        primary_features=[], task_target="benchmark_mutant_artifact_detection",
        minimum_independent_families=minimum_independent_families,
        independent_roots_verified=False, labels_independently_verified=False,
    )
    observations_count = sum(1 for line in observations_path.open("r", encoding="utf-8") if line.strip())
    failure_counts: Counter[str] = Counter()
    with failures_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                failure_counts[json.loads(line)["status"]] += 1
    stats = source_statistics(source)
    result = {
        "pipeline": "qmutbench_corpus_ingestion_v1", "status": "completed_bounded_run",
        "output_schema": {
            "normalized_records": {"version": "qmutbench-normalized-record-v1", "fields": [
                "schema", "record_id", "source", "source_commit", "source_csv_sha256", "source_row_index",
                "source_record_sha256", "original_circuit_id", "lineage_group_id", "lineage_group_basis",
                "family_root_independence_verified", "label_provenance", "diagnosis_category_compatible",
                "source_metadata", "artifact_role", "benchmark_label", "execution_status",
                "original_record_id", "original_qasm_path", "mutant_qasm_path", "qasm_sha256",
                "original_qasm_sha256", "algorithm", "qubits", "gate_count_metadata",
                "mutation_operator_metadata", "mutation_gate_metadata", "mutation_position_index_metadata",
                "killed_raw_metadata", "failure_reason", "qasm_path", "source_metadata_fields",
            ]},
            "simulated_observations": {"version": "qmutbench-simulated-observation-v1", "fields": [
                "schema", "record_id", "original_record_id", "source_csv_sha256", "source_record_sha256",
                "source_row_index", "original_qasm_sha256", "mutant_qasm_sha256", "source_commit",
                "observation_kind", "noise_settings", "shots_per_circuit", "seed", "seed_strategy",
                "measurement", "exact_original_probabilities", "exact_mutant_probabilities", "exact_tvd",
                "exact_observable_status", "sampled_original_counts", "sampled_mutant_counts",
                "sampled_tvd", "simulator", "interpretation_limit",
            ]},
            "rejections_and_failures": {"version": "qmutbench-rejection-v1", "fields": [
                "schema", "record_id", "source_row_index", "original_circuit_id", "status", "exception_type",
                "reason", "source_csv_sha256", "source_record_sha256",
            ]},
            "storage": "UTF-8 JSON Lines; one normalized record per original or mutant artifact, one paired observation per mutant record, one rejection row per unsuccessful source record",
        },
        "source": {"url": manifest["metadata_csv"]["url"], "commit": manifest.get("source_commit"),
                   "csv_sha256": source_hash, "source_manifest": str(source_manifest_path),
                   "source_manifest_sha256": sha256_file(source_manifest_path.resolve()),
                   "full_metadata_summary": stats},
        "run_config": config, "processed_this_invocation": processed_here,
        "counts": summary, "observation_records": observations_count,
        "rejection_breakdown": dict(sorted(failure_counts.items())),
        "lineage": {"distinct_QMutBench_Origin_program_groups": len(selected_groups),
                    "group_ids": sorted(selected_groups), "independent_roots_verified": False,
                    "reason": "Official parent grouping is retained but does not establish independent recipe roots."},
        "behavioral_interpretation": {
            "exact_distribution_change_count": summary.get("exact_behavior_changed", 0),
            "equivalent_or_indistinguishable_count": summary.get("equivalent_at_computational_basis", 0),
            "criterion": f"ideal computational-basis statevector TVD > {EQUIVALENCE_TOLERANCE}; equality does not establish full semantic equivalence",
            "Killed_field_used_as_ground_truth": False,
        },
        "memory_estimate": {"basis": "maximum simulator estimated working-array bytes per circuit; excludes Python/CSV/runtime overhead",
                            "max_observed_simulator_working_array_bytes": int(
                                summary.get("max_simulator_working_array_bytes", 0))},
        "ml_preflight": preflight,
        "outputs": {"normalized_records": _display_path(normalized_path),
                    "simulated_observations": _display_path(observations_path),
                    "failures": _display_path(failures_path),
                    "checkpoint": _display_path(checkpoint_path)},
        "provenance_hashes": hashes,
        "claim_limits": ["one QMutBench original group in the requested smoke run",
                         "benchmark-derived mutant artifact roles only; no independent behavioral labels",
                         "simulator-only computational-basis outputs; no hardware validity or independent ML generalization"],
    }
    _atomic_json(output_dir / "run_manifest.json", result)
    return result


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-row-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--minimum-independent-families", type=int)
    args = parser.parse_args()
    result = run_ingestion(
        source_manifest_path=args.source_manifest, output_dir=args.output_dir,
        start_row_index=args.start_row_index, limit=args.limit, shots=args.shots,
        seed_base=args.seed_base, resume=args.resume,
        minimum_independent_families=args.minimum_independent_families,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
