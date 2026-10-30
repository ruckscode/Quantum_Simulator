"""Prepare a semantics-preserving QBugs QASM task with duplicate-safe groups.

The original normalized task discarded gate parameters and therefore mapped
many distinct circuits to the same text. This variant retains OpenQASM 2 text,
rejects complete source pairs when an exact QASM fingerprint has contradictory
labels, and keeps each remaining source pair in one grouped fold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data/external_quantum_datasets/qbugs_hornbro/hornbro/datasets/Qbugs"
OUT = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/qbugs_qasm_v2"
SEED = 20261029


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return sha256(path)


def atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> str:
    import io as textio
    buffer = textio.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows)
    return atomic_write(path, buffer.getvalue().encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-pickles", type=int, help="prepare a bounded source-pickle development sample")
    args = parser.parse_args()
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(SOURCE_ROOT)
    if args.limit_pickles is not None and args.limit_pickles < 1:
        raise ValueError("--limit-pickles must be positive")

    helper_path = ROOT / "experiments/prepare_qbugs_circuit_pairs.py"
    spec = importlib.util.spec_from_file_location("qbugs_pair_extractor", helper_path)
    helper = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(helper)
    from qiskit import qasm2

    pickles = sorted(SOURCE_ROOT.rglob("qcs.pkl"))
    if args.limit_pickles:
        pickles = pickles[:args.limit_pickles]
    records: list[dict] = []
    source_files = []
    for path in pickles:
        payload = path.read_bytes()
        circuits = helper.RestrictedQiskitUnpickler(io.BytesIO(payload)).load()
        if not isinstance(circuits, list) or len(circuits) % 2:
            raise ValueError(f"Unexpected source pair layout: {path}")
        half = len(circuits) // 2
        scenario = path.parent.relative_to(SOURCE_ROOT).as_posix()
        relpath = path.relative_to(ROOT).as_posix()
        source_files.append({"path": relpath, "sha256": digest(payload), "pairs": half})
        for i in range(half):
            pair_id = f"{scenario}:{i}"
            for circuit, label, kind in ((circuits[i], 0, "correct"), (circuits[i + half], 1, "buggy")):
                qasm = qasm2.dumps(circuit)
                qasm_hash = digest(qasm.encode("utf-8"))
                rid = f"qbugs-qasm-{hashlib.sha256(pair_id.encode()).hexdigest()[:10]}-{kind}"
                records.append({"record_id": rid, "bug_id": pair_id, "label": label,
                                "code": qasm, "scenario": scenario, "pair_index": i,
                                "circuit_qubits": circuit.num_qubits, "circuit_depth": circuit.depth(),
                                "instruction_count": len(circuit.data), "qasm_sha256": qasm_hash,
                                "source_pickle": relpath})
        del circuits

    by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_hash[row["qasm_sha256"]].append(row)
    conflicting = {key for key, values in by_hash.items()
                   if len({int(row["label"]) for row in values}) > 1}
    rejected_pairs = {row["bug_id"] for key in conflicting for row in by_hash[key]}
    rejected = []
    for row in records:
        if row["bug_id"] in rejected_pairs:
            rejected.append({"record_id": row["record_id"], "source_pair_id": row["bug_id"],
                             "qasm_sha256": row["qasm_sha256"], "label": row["label"],
                             "reason": "source pair excluded because an exact OpenQASM fingerprint has conflicting correct/buggy labels"})
    accepted = [row for row in records if row["bug_id"] not in rejected_pairs]
    if not accepted:
        raise RuntimeError("No rows remain after exact-QASM semantic conflict filtering")

    # Connect source pairs and any duplicate QASM fingerprints. Exact QASM
    # cross-label collisions have already caused their complete source pairs
    # to be rejected; same-label duplicates remain together in one component.
    parent = list(range(len(accepted)))
    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    seen_pair: dict[str, int] = {}
    seen_qasm: dict[str, int] = {}
    for i, row in enumerate(accepted):
        for mapping, key in ((seen_pair, row["bug_id"]), (seen_qasm, row["qasm_sha256"])):
            if key in mapping:
                union(i, mapping[key])
            else:
                mapping[key] = i
    roots = sorted({find(i) for i in range(len(accepted))})
    component_id = {root: f"qbugs-qasm-component-{n:05d}" for n, root in enumerate(roots)}
    for i, row in enumerate(accepted):
        row["bug_id"] = component_id[find(i)]
    groups = [row["bug_id"] for row in accepted]
    labels = [int(row["label"]) for row in accepted]
    if len(set(groups)) < 5:
        raise RuntimeError("Fewer than five duplicate-safe groups remain")
    folds = [None] * len(accepted)
    component_rows: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        component_rows[group].append(index)
    rng = random.Random(SEED)
    randomized = list(component_rows.items())
    rng.shuffle(randomized)
    randomized.sort(key=lambda item: len(item[1]), reverse=True)
    class_totals = {label: labels.count(label) for label in (0, 1)}
    fold_classes = [{0: 0, 1: 0} for _ in range(5)]
    fold_sizes = [0] * 5
    for _, indices in randomized:
        counts = {label: sum(labels[i] == label for i in indices) for label in (0, 1)}
        def cost(fold_index: int) -> tuple[int, float, int]:
            score = sum((fold_classes[fold_index][label] + counts[label] - class_totals[label] / 5) ** 2
                        for label in (0, 1))
            return (fold_sizes[fold_index], score, fold_index)
        selected = min(range(5), key=cost)
        fold_sizes[selected] += len(indices)
        for label in (0, 1):
            fold_classes[selected][label] += counts[label]
        for index in indices:
            folds[index] = selected + 1
    if any(value is None for value in folds):
        raise RuntimeError("Some accepted rows did not receive a fold")
    fold_rows = [{"record_id": row["record_id"], "group_id": row["bug_id"],
                  "fold": folds[i], "model": "qbugs_qasm_v2_grouped"}
                 for i, row in enumerate(accepted)]
    for row in accepted:
        row.pop("source_pickle")

    suffix = "_dev2" if args.limit_pickles else ""
    output = OUT.with_name(OUT.name + suffix)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_bytes = json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode()
    source_hash = digest(source_bytes)
    fields = list(accepted[0])
    input_hash = atomic_csv(output / "normalized_manifest.csv", accepted, fields)
    fold_hash = atomic_csv(output / "group_folds.csv", fold_rows, list(fold_rows[0]))
    rejection_hash = atomic_csv(output / "rejected_rows.csv", rejected,
                                ["record_id", "source_pair_id", "qasm_sha256", "label", "reason"])
    from collections import Counter
    fold_counts = {str(k): {"rows": sum(int(folds[i]) == k for i in range(len(accepted))),
                        "groups": len({groups[i] for i in range(len(accepted)) if int(folds[i]) == k}),
                        "classes": dict(Counter(str(labels[i]) for i in range(len(accepted)) if int(folds[i]) == k))}
                   for k in range(1, 6)}
    if any(item["rows"] == 0 or set(item["classes"]) != {"0", "1"} for item in fold_counts.values()):
        raise RuntimeError(f"Invalid grouped fold coverage or class balance: {fold_counts}")
    report = {"status": "PREPARED_FOR_SOURCE_ISOLATED_VALIDATION", "task_id": "correct_buggy_circuit_state_qasm_v2",
              "source_file_count": len(source_files), "source_files_sha256": source_hash,
              "input_sha256": input_hash, "fold_manifest_sha256": fold_hash,
              "rejection_manifest_sha256": rejection_hash, "input_rows": len(records),
              "accepted_rows": len(accepted), "rejected_rows": len(rejected),
              "rejected_source_pairs": len(rejected_pairs), "conflicting_exact_qasm_fingerprints": len(conflicting),
              "groups": len(set(groups)), "exact_qasm_fingerprints": len({r['qasm_sha256'] for r in accepted}),
              "cross_fold_exact_qasm_fingerprints": 0,
              "folds": fold_counts, "source_files": source_files,
              "label_policy": "Correct=0 and buggy=1 follows upstream first-half/second-half source pairing. Complete source pairs are rejected only when exact QASM repeats with contradictory labels.",
              "feature_policy": "OpenQASM 2 serialization retains gate parameters and operation sequence; pair IDs, scenario, and labels are excluded from model text.",
              "group_policy": "Connect paired examples and all exact-QASM repeats before stratified grouped splitting.",
              "split_seed": SEED, "frozen_test_accessed": False}
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    contract = {"source_id": "qbugs_hornbro", "task_id": "correct_buggy_circuit_state_qasm_v2",
                "target_semantics": "Upstream paired correct (0) versus buggy (1) QBugs circuits represented as OpenQASM 2; not hardware diagnosis.",
                "required": fields, "record_id_column": "record_id", "group_column": "bug_id",
                "label_column": "label", "feature_column": "code", "allowed_labels": ["0", "1"],
                "fold_record_column": "record_id", "fold_group_column": "group_id", "fold_column": "fold",
                "fold_model_column": "model", "fold_model_value": "qbugs_qasm_v2_grouped",
                "input_path": f"{output.relative_to(ROOT).as_posix()}/normalized_manifest.csv",
                "fold_path": f"{output.relative_to(ROOT).as_posix()}/group_folds.csv",
                "output_path": f"data/processed/qbugs_hornbro/correct_buggy_circuit_state_qasm_v2{suffix}"}
    contract_path = ROOT / f"experiments/source_task_contracts/qbugs_correct_buggy_circuit_state_qasm_v2{suffix}.json"
    if contract_path.exists():
        raise FileExistsError(f"Refusing to overwrite contract: {contract_path}")
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "contract": str(contract_path.relative_to(ROOT)),
                      "input_rows": len(records), "accepted_rows": len(accepted), "rejected_rows": len(rejected),
                      "groups": len(set(groups)), "folds": fold_counts}, indent=2))


if __name__ == "__main__":
    main()
