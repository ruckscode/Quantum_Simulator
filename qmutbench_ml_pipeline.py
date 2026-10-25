"""Build a reproducible QMutBench feature dataset in resumable chunks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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

from qmutbench_validation import QMutBenchParseError, parse_qasm_file  # noqa: E402
from quantum_simulator.quantum_simulator import QuantumSimulator  # noqa: E402


SOURCE_CSV = ROOT / "data" / "qmutbench_full" / "all_data.csv"
LOCAL_QASM_ROOT = ROOT / "data" / "qmutbench_full" / "data"
OUTPUT_ROOT = ROOT / "data" / "qmutbench_ml"
OFFICIAL_BASE = "https://enautmendi.github.io/QMutBench/"
SEED = 42
BATCH_SIZE = 100
CHUNK_SIZE = 1000
MAX_SIMULATOR_QUBITS = 12  # Protect the statevector simulator from exponential allocations.
HIGH_DEVIATION_THRESHOLD = 0.05

META_COLUMNS = [
    "record_type",
    "label",
    "algorithm",
    "qubits",
    "gates",
    "operation",
    "gate",
    "Position_int",
    "Killed",
    "original_circuit_id",
    "qmutbench_row_index",
    "original_qasm_path",
    "mutant_qasm_path",
]

FEATURE_COLUMNS = [
    "trace_max_probability_difference",
    "trace_mean_probability_difference",
    "trace_median_probability_difference",
    "trace_deviation_concentration",
    "trace_high_deviation_fraction",
]


def _stratified_sample(
    frame: pd.DataFrame,
    size: int,
    seed: int,
    strata: list[str],
) -> pd.DataFrame:
    """Exact-size proportional stratified sample with one per nonempty stratum."""
    if size > len(frame):
        raise ValueError(f"Requested {size} rows from only {len(frame)} rows")

    groups = [
        (key, group)
        for key, group in frame.groupby(
            strata,
            sort=True,
            dropna=False,
        )
    ]

    if size < len(groups):
        raise ValueError(
            f"Sample size {size} is too small to cover {len(groups)} strata"
        )

    counts = np.array(
        [len(group) for _, group in groups],
        dtype=np.int64,
    )

    quotas = np.ones(len(groups), dtype=np.int64)
    remaining = size - len(groups)
    capacity = counts - 1

    while remaining > 0:
        available = np.flatnonzero(capacity > 0)

        if not len(available):
            break

        weights = capacity[available].astype(float)
        exact = remaining * weights / weights.sum()

        add = np.minimum(
            np.floor(exact).astype(np.int64),
            capacity[available],
        )

        if add.sum() == 0:
            order = sorted(
                range(len(available)),
                key=lambda j: (
                    -exact[j] % 1,
                    str(groups[available[j]][0]),
                ),
            )

            for j in order:
                if remaining == 0:
                    break

                idx = available[j]

                if capacity[idx] > 0:
                    quotas[idx] += 1
                    capacity[idx] -= 1
                    remaining -= 1

            continue

        quotas[available] += add
        capacity[available] -= add
        remaining -= int(add.sum())

    if remaining:
        raise ValueError("Could not allocate exact sample quotas")

    pieces = []

    for group_index, ((_, group), quota) in enumerate(
        zip(groups, quotas)
    ):
        if quota:
            pieces.append(
                group.sample(
                    n=int(quota),
                    random_state=seed + group_index,
                )
            )

    result = pd.concat(
        pieces,
        ignore_index=True,
    )

    return result.sample(
        frac=1,
        random_state=seed,
    ).reset_index(drop=True)


def select_records(
    source: Path,
    count: int = 10_000,
) -> pd.DataFrame:
    """
    Select QMutBench records that can actually be executed by the
    configured statevector simulator.

    The simulator-qubit filter is applied BEFORE stratified sampling.
    This prevents the smoke/full selection from wasting samples on
    circuits that exceed the simulator limit.
    """

    columns = [
        "algorithm",
        "qubits",
        "gates",
        "operation",
        "gate",
        "Position_int",
        "position",
        "Killed",
        "Origin_program",
    ]

    frame = pd.read_csv(
        source,
        usecols=columns,
    )

    frame.insert(
        0,
        "qmutbench_row_index",
        np.arange(
            len(frame),
            dtype=np.int64,
        ),
    )

    frame["qubits"] = frame["qubits"].astype(int)
    frame["Position_int"] = frame["Position_int"].astype(int)

    # IMPORTANT:
    # Filter BEFORE sampling so that selected records are compatible
    # with the simulator's configured statevector limit.
    frame = frame[
        frame["qubits"] <= MAX_SIMULATOR_QUBITS
    ].copy()

    if frame.empty:
        raise ValueError(
            f"No QMutBench records have <= "
            f"{MAX_SIMULATOR_QUBITS} qubits"
        )

    if count > len(frame):
        raise ValueError(
            f"Requested {count} rows from only {len(frame)} "
            f"eligible QMutBench rows "
            f"(<= {MAX_SIMULATOR_QUBITS} qubits)"
        )

    return _stratified_sample(
        frame,
        count,
        SEED,
        [
            "operation",
            "algorithm",
            "qubits",
        ],
    )


def original_rel(row: pd.Series) -> str:
    return (
        f"Origin_programs/"
        f"{row.algorithm}_"
        f"{int(row.qubits)}_qubits.qasm"
    )


def mutant_rel(row: pd.Series) -> str:
    prefix = (
        f"Mutated_programs/"
        f"Mutants_{row.algorithm}_"
        f"{int(row.qubits)}_qubits/"
    )

    index = int(row.Position_int)
    gate = str(row.gate)
    operation = str(row.operation)

    if operation == "Replace":
        name = (
            f"ReplaceGate_{gate}_"
            f"inPositionOfGate_{index}.qasm"
        )

    elif operation == "Remove":
        name = f"RemoveGate_{index}_.qasm"

    elif operation == "Add":
        name = f"AddGate_{gate}_inGap_{index}_.qasm"

    else:
        raise ValueError(
            f"Unknown QMutBench operation: {operation}"
        )

    return prefix + name


def _safe_cache_path(relative: str) -> Path:
    path = (
        OUTPUT_ROOT / "qasm_cache" / relative
    ).resolve()

    cache_root = (
        OUTPUT_ROOT / "qasm_cache"
    ).resolve()

    if path != cache_root and cache_root not in path.parents:
        raise ValueError(
            f"Invalid QASM cache path: {relative}"
        )

    return path


def retrieve_qasm(relative: str) -> Path:
    """Use checked-in local cache first, then local output cache, then official URL."""

    local_source = LOCAL_QASM_ROOT / relative

    if local_source.is_file():
        return local_source

    cached = _safe_cache_path(relative)

    if cached.is_file():
        return cached

    cached.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    url = (
        OFFICIAL_BASE
        + "data/"
        + relative.replace("\\", "/")
    )

    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "User-Agent": "QMutBench-ML-pipeline/1.0"
        },
    )

    try:
        with (
            urllib.request.urlopen(
                request,
                timeout=30,
            ) as response,
            cached.open("wb") as target,
        ):
            shutil.copyfileobj(
                response,
                target,
                length=1024 * 1024,
            )

    except (
        OSError,
        urllib.error.URLError,
    ) as exc:
        cached.unlink(
            missing_ok=True
        )

        raise FileNotFoundError(
            f"Could not retrieve {relative} "
            f"from official QMutBench: {exc}"
        ) from exc

    return cached


def _run_and_trace(
    reference: list[tuple[Any, ...]],
    mutant: list[tuple[Any, ...]],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:

    _, _, _, trace = (
        QuantumSimulator(
            n,
            seed=seed,
        ).run_circuit_traced(
            reference,
            mutant,
            seed=seed,
        )
    )

    return trace


def _trace_features(
    trace: list[dict[str, Any]],
) -> dict[str, float]:

    diffs = [
        float(row["probability_difference"])
        for row in trace
    ]

    if not diffs:
        raise ValueError(
            "Simulator produced an empty execution trace"
        )

    total = sum(diffs)
    sorted_diffs = sorted(diffs)
    mid = len(sorted_diffs) // 2

    if len(sorted_diffs) % 2:
        median = sorted_diffs[mid]
    else:
        median = (
            sorted_diffs[mid - 1]
            + sorted_diffs[mid]
        ) / 2

    return {
        "trace_max_probability_difference": max(diffs),

        "trace_mean_probability_difference":
            total / len(diffs),

        "trace_median_probability_difference":
            float(median),

        "trace_deviation_concentration":
            max(diffs) / total
            if total > 0
            else 0.0,

        "trace_high_deviation_fraction":
            sum(
                value > HIGH_DEVIATION_THRESHOLD
                for value in diffs
            ) / len(diffs),
    }


def _read_parse(
    path: Path,
) -> tuple[list[tuple[Any, ...]], int]:
    return parse_qasm_file(path)


def _row_metadata(
    row: pd.Series,
    kind: str,
    label: str,
    org_path: str,
    mut_path: str | None,
) -> dict[str, Any]:

    return {
        "record_type": kind,
        "label": label,
        "algorithm": str(row.algorithm),
        "qubits": int(row.qubits),
        "gates": int(row.gates),
        "operation": str(row.operation),
        "gate": str(row.gate),
        "Position_int": int(row.Position_int),
        "Killed": str(row.Killed),
        "original_circuit_id": str(row.Origin_program),
        "qmutbench_row_index": int(
            row.qmutbench_row_index
        ),
        "original_qasm_path": org_path,
        "mutant_qasm_path": mut_path or "",
    }


def _record_output(
    metadata: dict[str, Any],
    features: dict[str, float],
) -> dict[str, Any]:

    return {
        **metadata,
        **features,
    }


def process_records(
    frame: pd.DataFrame,
    *,
    batch_size: int = BATCH_SIZE,
    checkpoint_path: Path | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
    resume_checkpoint: bool = True,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    start_position = 0
    if checkpoint_path and resume_checkpoint and checkpoint_path.is_file():
        try:
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            expected = {**(checkpoint_metadata or {}), "row_indices": frame.qmutbench_row_index.astype(int).tolist()}
            if all(saved.get(key) == value for key, value in expected.items()):
                start_position = int(saved["next_position"])
                if not 0 <= start_position <= len(frame):
                    raise ValueError("invalid checkpoint position")
                records, failures = saved["records"], saved["failures"]
            else:
                print("Ignoring stale chunk checkpoint", flush=True)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            print("Ignoring invalid chunk checkpoint", flush=True)

    original_cache: dict[
        str,
        tuple[
            Path,
            list[tuple[Any, ...]],
            int,
            dict[str, float],
        ],
    ] = {}

    mutant_cache: dict[
        str,
        tuple[
            Path,
            list[tuple[Any, ...]],
            int,
        ],
    ] = {}

    original_written: set[str] = {str(r["original_qasm_path"]) for r in records if r.get("record_type") == "original"}

    unsupported_qasm_constructs: Counter[str] = Counter()
    failure_classes: Counter[str] = Counter()

    unique_circuits: set[str] = set(frame.iloc[:start_position].Origin_program.astype(str))
    unique_mutants: set[str] = set(frame.iloc[:start_position].apply(mutant_rel, axis=1))

    successful_mutants = sum(r.get("record_type") == "mutant" for r in records)
    successful_originals = sum(r.get("record_type") == "original" for r in records)
    failure_classes.update(f["failure_type"] for f in failures)
    for failure in failures:
        unsupported_op = re.search(r"Unsupported operation in circuit: '([^']+)'", failure.get("error", ""))
        if unsupported_op:
            unsupported_qasm_constructs[unsupported_op.group(1)] += 1

    for batch_start in range(
        start_position,
        len(frame),
        batch_size,
    ):

        batch = frame.iloc[
            batch_start:
            batch_start + batch_size
        ]

        for _, row in batch.iterrows():

            org_rel = original_rel(row)
            mut_rel = mutant_rel(row)

            circuit_id = str(
                row.Origin_program
            )

            unique_circuits.add(
                circuit_id
            )

            unique_mutants.add(
                mut_rel
            )

            context = {
                "qmutbench_row_index":
                    int(row.qmutbench_row_index),

                "original_circuit_id":
                    circuit_id,

                "mutant_qasm_path":
                    mut_rel,

                "original_qasm_path":
                    org_rel,
            }

            try:

                if org_rel not in original_cache:

                    org_path = retrieve_qasm(
                        org_rel
                    )

                    org_ops, org_n = _read_parse(
                        org_path
                    )

                    if org_n > MAX_SIMULATOR_QUBITS:
                        raise QMutBenchUnsupportedError(
                            f"Simulator statevector limit is "
                            f"{MAX_SIMULATOR_QUBITS} qubits; "
                            f"circuit declares {org_n}"
                        )

                    org_trace = _run_and_trace(
                        org_ops,
                        org_ops,
                        org_n,
                        SEED,
                    )

                    org_features = _trace_features(
                        org_trace
                    )

                    original_cache[org_rel] = (
                        org_path,
                        org_ops,
                        org_n,
                        org_features,
                    )

                (
                    org_path,
                    org_ops,
                    org_n,
                    org_features,
                ) = original_cache[org_rel]

                if org_rel not in original_written:

                    normal_meta = _row_metadata(
                        row,
                        "original",
                        "NORMAL",
                        org_rel,
                        None,
                    )

                    records.append(
                        _record_output(
                            normal_meta,
                            org_features,
                        )
                    )

                    original_written.add(
                        org_rel
                    )

                    successful_originals += 1

                if mut_rel not in mutant_cache:

                    mut_path = retrieve_qasm(
                        mut_rel
                    )

                    mut_ops, mut_n = _read_parse(
                        mut_path
                    )

                    if mut_n != org_n:
                        raise ValueError(
                            f"QASM width mismatch: "
                            f"original={org_n}, "
                            f"mutant={mut_n}"
                        )

                    if mut_n > MAX_SIMULATOR_QUBITS:
                        raise QMutBenchUnsupportedError(
                            f"Simulator statevector limit is "
                            f"{MAX_SIMULATOR_QUBITS} qubits; "
                            f"mutant declares {mut_n}"
                        )

                    mutant_cache[mut_rel] = (
                        mut_path,
                        mut_ops,
                        mut_n,
                    )

                (
                    mut_path,
                    mut_ops,
                    mut_n,
                ) = mutant_cache[mut_rel]

                trace = _run_and_trace(
                    org_ops,
                    mut_ops,
                    mut_n,
                    SEED
                    + int(
                        row.qmutbench_row_index
                    ),
                )

                features = _trace_features(
                    trace
                )

                metadata = _row_metadata(
                    row,
                    "mutant",
                    "PROGRAM_FAULT",
                    org_rel,
                    mut_rel,
                )

                records.append(
                    _record_output(
                        metadata,
                        features,
                    )
                )

                successful_mutants += 1

            except Exception as exc:

                error_type = type(exc).__name__

                message = (
                    f"{error_type}: {exc}"
                )

                failure_classes[
                    error_type
                ] += 1

                unsupported_op = re.search(
                    r"Unsupported operation in circuit: '([^']+)'",
                    str(exc),
                )

                if unsupported_op:
                    unsupported_qasm_constructs[
                        unsupported_op.group(1)
                    ] += 1

                failures.append(
                    {
                        **context,
                        "failure_type":
                            error_type,
                        "error":
                            message,
                    }
                )

            processed = batch_start + batch.index.get_loc(_) + 1
            if processed == 1 or processed % 25 == 0 or processed == len(frame):
                print(f"Processed {processed}/{len(frame)} records in current chunk", flush=True)
            if checkpoint_path and (processed % 25 == 0 or processed == len(frame)):
                payload = {**(checkpoint_metadata or {}), "row_indices": frame.qmutbench_row_index.astype(int).tolist(), "next_position": processed, "records": records, "failures": failures}
                temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
                os.replace(temporary, checkpoint_path)

    # Count only actual statevector-limit failures.
    simulator_limit_failures = sum(
        1
        for failure in failures
        if failure["failure_type"]
        == "QMutBenchUnsupportedError"
        and "statevector limit"
        in failure["error"]
    )

    summary = {
        "selected_records":
            len(frame),

        "successful_mutant_executions":
            successful_mutants,

        "successful_original_executions":
            successful_originals,

        "failed_mutant_records":
            len(failures),

        "failure_types":
            dict(failure_classes),

        "unsupported_qasm_constructs":
            dict(unsupported_qasm_constructs),

        "simulator_limit_failures":
            simulator_limit_failures,

        "unique_original_circuits_selected":
            len(unique_circuits),

        "unique_mutants_selected":
            len(unique_mutants),

        "originals_emitted_as_normal":
            successful_originals,

        "generated_feature_columns":
            FEATURE_COLUMNS,

        "metadata_columns":
            META_COLUMNS,

        "max_simulator_qubits":
            MAX_SIMULATOR_QUBITS,

        "batch_size":
            batch_size,
    }

    return (
        records,
        failures,
        summary,
    )


class QMutBenchUnsupportedError(
    NotImplementedError
):
    pass


def write_outputs(
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    summary: dict[str, Any],
    prefix: str,
    *,
    output_dir: Path | None = None,
) -> None:

    output_dir = output_dir or OUTPUT_ROOT
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir / f"{prefix}.csv"
    )

    failure_path = (
        output_dir / f"{prefix}_failures.csv"
    )

    manifest_path = (
        output_dir / f"{prefix}_manifest.json"
    )

    columns = (
        META_COLUMNS
        + FEATURE_COLUMNS
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(records)

    failure_columns = [
        "qmutbench_row_index",
        "original_circuit_id",
        "original_qasm_path",
        "mutant_qasm_path",
        "failure_type",
        "error",
    ]

    with failure_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=failure_columns,
        )

        writer.writeheader()
        writer.writerows(failures)

    digest = hashlib.sha256()

    with SOURCE_CSV.open(
        "rb"
    ) as source_stream:

        for block in iter(
            lambda:
                source_stream.read(
                    1024 * 1024
                ),
            b"",
        ):
            digest.update(block)

    manifest = {
        "dataset": "QMutBench",
        "selection_seed": SEED,
        "source_csv": str(SOURCE_CSV),
        "source_csv_sha256":
            digest.hexdigest(),

        "feature_schema":
            FEATURE_COLUMNS,

        "features_source":
            "QuantumSimulator.run_circuit_traced probability_difference trace",

        "feature_threshold":
            HIGH_DEVIATION_THRESHOLD,

        "label_policy":
            "Mutants are PROGRAM_FAULT; original circuit is emitted once per unique original QASM as NORMAL; Killed is retained only as metadata.",

        "summary":
            summary,

        "output_csv":
            str(csv_path.resolve()),

        "failures_csv":
            str(failure_path.resolve()),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                **summary,
                "output_csv":
                    str(
                        csv_path.resolve()
                    ),
                "failures_csv":
                    str(
                        failure_path.resolve()
                    ),
                "manifest":
                    str(
                        manifest_path.resolve()
                    ),
            },
            indent=2,
        )
    )


def _source_digest() -> str:
    digest = hashlib.sha256()
    with SOURCE_CSV.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _merge_chunks(chunk_dir: Path, expected_chunks: int) -> None:
    result_parts = []
    failure_parts = []
    manifests = []
    for chunk_id in range(expected_chunks):
        prefix = f"chunk_{chunk_id:05d}"
        manifest_path = chunk_dir / f"{prefix}_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Missing completed chunk manifest: {manifest_path}")
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        result_parts.append(pd.read_csv(chunk_dir / f"{prefix}_results.csv"))
        failure_parts.append(pd.read_csv(chunk_dir / f"{prefix}_failures.csv"))
    results = pd.concat(result_parts, ignore_index=True) if result_parts else pd.DataFrame(columns=META_COLUMNS + FEATURE_COLUMNS)
    failures = pd.concat(failure_parts, ignore_index=True) if failure_parts else pd.DataFrame(columns=["qmutbench_row_index", "original_circuit_id", "original_qasm_path", "mutant_qasm_path", "failure_type", "error"])
    if not results.empty:
        results["_dedupe"] = np.where(results.record_type == "original", results.original_circuit_id, results.qmutbench_row_index.astype(str))
        results = results.drop_duplicates(["record_type", "_dedupe"], keep="first").drop(columns="_dedupe")
    if not failures.empty:
        failures = failures.drop_duplicates(["qmutbench_row_index"], keep="first")
    results.to_csv(OUTPUT_ROOT / "qmutbench_results.csv", index=False)
    failures.to_csv(OUTPUT_ROOT / "qmutbench_failures.csv", index=False)
    summary = {
        "selected_records": sum(m["summary"]["selected_records"] for m in manifests),
        "successful_mutant_executions": sum(m["summary"]["successful_mutant_executions"] for m in manifests),
        "successful_original_executions": int(results.record_type.eq("original").sum()) if not results.empty else 0,
        "failed_mutant_records": len(failures),
        "failure_types": dict(Counter(failures.failure_type)) if not failures.empty else {},
        "simulator_limit_failures": sum(m["summary"]["simulator_limit_failures"] for m in manifests),
        "chunk_count": int(expected_chunks),
        "chunk_size": manifests[0]["chunk_size"] if manifests else CHUNK_SIZE,
        "max_simulator_qubits": MAX_SIMULATOR_QUBITS,
    }
    merged = {"dataset": "QMutBench", "selection_seed": SEED, "source_csv": str(SOURCE_CSV), "source_csv_sha256": _source_digest(), "feature_schema": FEATURE_COLUMNS, "summary": summary, "chunks": [m["chunk_id"] for m in manifests], "output_csv": str((OUTPUT_ROOT / "qmutbench_results.csv").resolve()), "failures_csv": str((OUTPUT_ROOT / "qmutbench_failures.csv").resolve())}
    (OUTPUT_ROOT / "qmutbench_manifest.json").write_text(json.dumps(merged, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def process_chunks(selected: pd.DataFrame, *, chunk_size: int, batch_size: int, resume: bool) -> None:
    """Process a stable selection, checkpointing each complete chunk."""
    chunk_dir = OUTPUT_ROOT / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    source_hash = _source_digest()
    total_chunks = (len(selected) + chunk_size - 1) // chunk_size
    for chunk_id in range(total_chunks):
        chunk = selected.iloc[chunk_id * chunk_size:(chunk_id + 1) * chunk_size].reset_index(drop=True)
        prefix = f"chunk_{chunk_id:05d}"
        manifest_path = chunk_dir / f"{prefix}_manifest.json"
        result_path = chunk_dir / f"{prefix}_results.csv"
        failure_path = chunk_dir / f"{prefix}_failures.csv"
        if resume and manifest_path.is_file() and result_path.is_file() and failure_path.is_file():
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old.get("source_csv_sha256") == source_hash and old.get("row_indices") == chunk.qmutbench_row_index.astype(int).tolist() and old.get("chunk_size") == chunk_size and old.get("max_simulator_qubits") == MAX_SIMULATOR_QUBITS:
                print(f"Skipping completed chunk {chunk_id + 1}/{total_chunks}", flush=True)
                continue
        print(f"Starting chunk {chunk_id + 1}/{total_chunks} ({len(chunk)} records)", flush=True)
        checkpoint_path = chunk_dir / f"{prefix}_checkpoint.json"
        checkpoint_meta = {"source_csv_sha256": source_hash, "chunk_size": chunk_size, "max_simulator_qubits": MAX_SIMULATOR_QUBITS}
        records, failures, summary = process_records(chunk, batch_size=batch_size, checkpoint_path=checkpoint_path, checkpoint_metadata=checkpoint_meta, resume_checkpoint=resume)
        staging = chunk_dir / f".{prefix}_staging"
        staging.mkdir(parents=True, exist_ok=True)
        write_outputs(records, failures, summary, f"{prefix}_results", output_dir=staging)
        generated_manifest = staging / f"{prefix}_results_manifest.json"
        generated_failures = staging / f"{prefix}_results_failures.csv"
        manifest = json.loads(generated_manifest.read_text(encoding="utf-8"))
        generated_manifest.unlink()
        manifest_path.unlink(missing_ok=True)
        (staging / f"{prefix}_results.csv").replace(result_path)
        generated_failures.replace(failure_path)
        shutil.rmtree(staging)
        manifest.update({"chunk_id": chunk_id, "chunk_size": chunk_size, "row_indices": chunk.qmutbench_row_index.astype(int).tolist(), "source_csv_sha256": source_hash, "max_simulator_qubits": MAX_SIMULATOR_QUBITS})
        manifest_tmp = manifest_path.with_suffix(".json.tmp")
        manifest_tmp.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(manifest_tmp, manifest_path)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    _merge_chunks(chunk_dir, total_chunks)


def main() -> int:

    global MAX_SIMULATOR_QUBITS, SOURCE_CSV

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_CSV,
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Process the stratified "
            "10,000-record selection. "
            "Default runs only the "
            "100-record smoke test."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--max-simulator-qubits",
        type=int,
        default=MAX_SIMULATOR_QUBITS,
    )

    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true", help="Resume completed chunks (default).")
    resume.add_argument("--no-resume", dest="resume", action="store_false", help="Reprocess and replace chunks.")
    parser.set_defaults(resume=True)

    parser.add_argument(
        "--max-smoke-failure-rate",
        type=float,
        default=0.20,
        help=(
            "Stop instead of full processing "
            "when smoke failures exceed "
            "this fraction."
        ),
    )

    args = parser.parse_args()

    SOURCE_CSV = args.source

    MAX_SIMULATOR_QUBITS = (
        args.max_simulator_qubits
    )

    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if args.max_simulator_qubits < 1:
        parser.error("--max-simulator-qubits must be positive")

    # Selection now happens only from records
    # that are within the configured simulator limit.
    selected = select_records(
        SOURCE_CSV,
        10_000,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUTPUT_ROOT / "selected_10000_metadata.csv", index=False)
    if not args.full:
        selected = _stratified_sample(selected, 100, SEED, ["operation", "algorithm"])
    process_chunks(selected, chunk_size=args.chunk_size, batch_size=args.batch_size, resume=args.resume)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
