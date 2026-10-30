"""Validate the safely loadable Mute-Alg QBugs subset without hiding failures.

Run with the isolated Qiskit 1.4.5 environment. The upstream HornBro README
defines each qcs.pkl as original circuits followed by buggy circuits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import pickle
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data/external_quantum_datasets/hornbro_mute_alg/hornbro/datasets/Mute-Alg"
OUT = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/mute_alg_compatible_subset"
SEED = 20261029


class LegacyParameterTable(dict):
    """Compatibility container for the retired Qiskit ParameterTable class."""


_QISKIT_QUANTUM_CIRCUIT = None
_QISKIT_OUTER_SCOPE = None
_QISKIT_CIRCUIT_INSTRUCTION = None
_PURE_LOAD_BUILD = pickle._Unpickler.load_build


class RestrictedQiskitUnpickler(pickle._Unpickler):
    # The pure-Python unpickler lets us handle two retired Qiskit slot states
    # explicitly while retaining a restrictive global allowlist.
    dispatch = pickle._Unpickler.dispatch.copy()
    ALLOWED = ("qiskit", "numpy", "sympy", "builtins", "collections", "enum", "types",
               "copyreg", "re", "fractions", "decimal", "math", "uuid", "datetime",
               "operator", "functools")

    def find_class(self, module: str, name: str):
        if module == "qiskit.circuit.parametertable" and name == "ParameterTable":
            return LegacyParameterTable
        if module.startswith(self.ALLOWED):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Blocked pickle global: {module}.{name}")

    def load_build(self):
        state, instance = self.stack[-1], self.stack[-2]
        if _QISKIT_OUTER_SCOPE is not None and isinstance(instance, _QISKIT_OUTER_SCOPE):
            state = self.stack.pop()
            if (isinstance(state, tuple) and len(state) == 2 and state[0] is None and
                    isinstance(state[1], dict) and set(state[1]) == {"circuit"} and
                    isinstance(state[1]["circuit"], _QISKIT_QUANTUM_CIRCUIT)):
                object.__setattr__(instance, "circuit", state[1]["circuit"])
                return
            raise pickle.UnpicklingError("Unsupported legacy OuterCircuitScopeInterface slot state")
        if _QISKIT_CIRCUIT_INSTRUCTION is not None and type(instance) is _QISKIT_CIRCUIT_INSTRUCTION:
            state = self.stack.pop()
            if isinstance(state, tuple) and len(state) == 3:
                self.stack[-1] = _QISKIT_CIRCUIT_INSTRUCTION(state[0], state[1], state[2])
                return
            raise pickle.UnpicklingError("Unsupported legacy CircuitInstruction state")
        return _PURE_LOAD_BUILD(self)


RestrictedQiskitUnpickler.dispatch[ord("b")] = RestrictedQiskitUnpickler.load_build


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return sha256(path)


def atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> str:
    import io as textio
    buffer = textio.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows)
    return atomic_write(path, buffer.getvalue().encode("utf-8"))


def assign_folds(rows: list[dict]) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    labels = [int(row["label"]) for row in rows]
    for i, row in enumerate(rows):
        groups[row["bug_id"]].append(i)
    if len(groups) < 5:
        raise RuntimeError(f"Only {len(groups)} duplicate-safe groups; five folds require at least five")
    order = list(groups.items())
    rng = random.Random(SEED)
    rng.shuffle(order)
    order.sort(key=lambda item: len(item[1]), reverse=True)
    totals = {label: labels.count(label) for label in (0, 1)}
    fold_labels = [{0: 0, 1: 0} for _ in range(5)]
    fold_sizes = [0] * 5
    result = [0] * len(rows)
    for _, indices in order:
        counts = {label: sum(labels[i] == label for i in indices) for label in (0, 1)}
        def cost(fold: int) -> tuple[int, float, int]:
            deviation = sum((fold_labels[fold][label] + counts[label] - totals[label] / 5) ** 2
                            for label in (0, 1))
            return fold_sizes[fold], deviation, fold
        selected = min(range(5), key=cost)
        fold_sizes[selected] += len(indices)
        for label in (0, 1):
            fold_labels[selected][label] += counts[label]
        for index in indices:
            result[index] = selected + 1
    for fold in range(1, 6):
        fold_labels_actual = {str(label) for i, label in enumerate(labels) if result[i] == fold}
        if not fold_labels_actual == {"0", "1"}:
            raise RuntimeError(f"Fold {fold} is empty or single-class")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-pickles", type=int, help="bounded validation sample; writes a separate _dev folder")
    parser.add_argument("--algorithm", help="restrict a development sample to one Mute-Alg algorithm directory")
    args = parser.parse_args()
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(SOURCE_ROOT)
    if args.limit_pickles is not None and args.limit_pickles < 1:
        raise ValueError("--limit-pickles must be positive")
    if args.algorithm and not args.limit_pickles:
        raise ValueError("--algorithm is intended for a bounded --limit-pickles development sample")
    from qiskit import QuantumCircuit, qasm2
    from qiskit.circuit import CircuitInstruction
    from qiskit.circuit.quantumcircuit import _OuterCircuitScopeInterface
    global _QISKIT_QUANTUM_CIRCUIT, _QISKIT_OUTER_SCOPE, _QISKIT_CIRCUIT_INSTRUCTION
    _QISKIT_QUANTUM_CIRCUIT = QuantumCircuit
    _QISKIT_OUTER_SCOPE = _OuterCircuitScopeInterface
    _QISKIT_CIRCUIT_INSTRUCTION = CircuitInstruction

    pickles = sorted(SOURCE_ROOT.rglob("qcs.pkl"))
    if args.algorithm:
        pickles = [path for path in pickles if path.parent.parent.parent.name == args.algorithm]
    if args.limit_pickles:
        pickles = pickles[:args.limit_pickles]
    if not pickles:
        raise RuntimeError("No source pickle files selected")
    file_audit, records, failed_files = [], [], []
    for file_index, path in enumerate(pickles):
        payload = path.read_bytes()
        relative = path.relative_to(ROOT).as_posix()
        file_item = {"path": relative, "sha256": sha_bytes(payload), "bytes": len(payload)}
        try:
            circuits = RestrictedQiskitUnpickler(io.BytesIO(payload)).load()
            if (not isinstance(circuits, list) or len(circuits) % 2 or
                    not all(isinstance(circuit, QuantumCircuit) for circuit in circuits)):
                raise TypeError("Expected an even list containing only QuantumCircuit objects")
            half = len(circuits) // 2
            scenario = path.parent.relative_to(SOURCE_ROOT).as_posix()
            file_item.update({"status": "LOADED", "circuits": len(circuits), "source_pairs": half})
            for pair_index in range(half):
                pair_key = f"{scenario}:{pair_index}"
                pair_id = f"mute-{hashlib.sha256(pair_key.encode()).hexdigest()[:12]}"
                for circuit, label, kind in ((circuits[pair_index], 0, "original"),
                                             (circuits[pair_index + half], 1, "buggy")):
                    qasm = qasm2.dumps(circuit)
                    qasm_hash = sha_bytes(qasm.encode("utf-8"))
                    records.append({"record_id": f"{pair_id}-{kind}", "bug_id": pair_id,
                        "label": label, "code": qasm, "scenario": scenario, "pair_index": pair_index,
                        "algorithm": path.parent.parent.parent.name,
                        "qubits": circuit.num_qubits, "depth": circuit.depth(),
                        "instruction_count": len(circuit.data), "qasm_sha256": qasm_hash,
                        "source_file_id": f"mute-file-{file_index:04d}"})
            del circuits
        except Exception as exc:
            file_item.update({"status": "BLOCKED_LOAD_OR_NORMALIZATION", "error_type": type(exc).__name__,
                              "error": str(exc)[:400]})
            failed_files.append({"path": relative, "sha256": file_item["sha256"],
                                 "bytes": len(payload), "reason": f"{type(exc).__name__}: {str(exc)[:400]}"})
        file_audit.append(file_item)

    by_qasm: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_qasm[row["qasm_sha256"]].append(row)
    bad_hashes = {key for key, items in by_qasm.items()
                  if len({int(item["label"]) for item in items}) > 1}
    bad_pairs = {item["bug_id"] for key in bad_hashes for item in by_qasm[key]}
    rejected = [{"record_id": row["record_id"], "source_pair_id": row["bug_id"],
                 "qasm_sha256": row["qasm_sha256"], "label": row["label"],
                 "reason": "complete source pair excluded because identical QASM has contradictory labels"}
                for row in records if row["bug_id"] in bad_pairs]
    accepted = [row.copy() for row in records if row["bug_id"] not in bad_pairs]
    # Union paired examples and every exact-QASM recurrence before deciding
    # whether the source supports a useful grouped split.
    groups_count = 0
    split_failure = None
    folds: list[int] = []
    if accepted:
        parent = list(range(len(accepted)))
        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        def union(a: int, b: int) -> None:
            a, b = find(a), find(b)
            if a != b: parent[b] = a
        pair_seen, qasm_seen = {}, {}
        for i, row in enumerate(accepted):
            for mapping, key in ((pair_seen, row["bug_id"]), (qasm_seen, row["qasm_sha256"])):
                if key in mapping: union(i, mapping[key])
                else: mapping[key] = i
        roots = sorted({find(i) for i in range(len(accepted))})
        groups_count = len(roots)
        component_ids = {root: f"mute-component-{n:06d}" for n, root in enumerate(roots)}
        for i, row in enumerate(accepted):
            row["bug_id"] = component_ids[find(i)]
            row.pop("source_file_id", None)
        try:
            folds = assign_folds(accepted)
        except RuntimeError as exc:
            split_failure = str(exc)
    else:
        split_failure = "No rows remain after excluding complete pairs with contradictory exact-QASM labels"
    is_ready = split_failure is None
    fold_model = "mute_alg_compatible_subset_qasm_v2"
    fold_rows = ([{"record_id": row["record_id"], "group_id": row["bug_id"], "fold": folds[i],
                   "model": fold_model} for i, row in enumerate(accepted)] if is_ready else [])

    suffix = (f"_dev_{args.algorithm}" if args.algorithm else "_dev") if args.limit_pickles else "_qiskit_compat_v2"
    output = OUT.with_name(OUT.name + suffix)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    fields = list((accepted or records)[0]) if (accepted or records) else [
        "record_id", "bug_id", "label", "code", "scenario", "pair_index", "algorithm",
        "qubits", "depth", "instruction_count", "qasm_sha256", "source_file_id"]
    input_hash = atomic_csv(output / "normalized_manifest.csv", accepted, fields)
    fold_hash = atomic_csv(output / "group_folds.csv", fold_rows,
        ["record_id", "group_id", "fold", "model"])
    reject_hash = atomic_csv(output / "rejected_rows.csv", rejected,
        ["record_id", "source_pair_id", "qasm_sha256", "label", "reason"])
    failed_hash = atomic_csv(output / "unreadable_source_files.csv", failed_files,
        ["path", "sha256", "bytes", "reason"])
    prep = {"status": "READY_FOR_SOURCE_RUNNER" if is_ready else "INELIGIBLE_DUPLICATE_SAFE_GROUP_SPLIT",
        "source_semantics": "Mute-Alg generated quantum program repair pairs; first half originals, second half buggy per HornBro upstream README.",
        "upstream_dataset_readme": "https://github.com/JanusQ/HornBro",
        "upstream_pair_order_evidence": "The official HornBro README states the first half of qcs.pkl contains originals and the second half contains buggy programs.",
        "source_file_count_scanned": len(file_audit), "source_files": file_audit,
        "successfully_loaded_files": sum(x["status"] == "LOADED" for x in file_audit),
        "unreadable_or_unusable_files": len(failed_files), "unreadable_files_sha256": failed_hash,
        "input_rows_before_semantic_rejections": len(records), "accepted_rows": len(accepted),
        "rejected_semantic_rows": len(rejected), "rejected_complete_pairs": len(bad_pairs),
        "conflicting_exact_qasm_fingerprints": len(bad_hashes), "groups": groups_count,
        "class_counts": dict(Counter(str(r["label"]) for r in accepted)), "folds": dict(Counter(str(x) for x in folds)),
        "split_failure_reason": split_failure,
        "input_sha256": input_hash, "fold_manifest_sha256": fold_hash, "rejection_manifest_sha256": reject_hash,
        "feature_policy": "OpenQASM 2 circuit text only; pair, algorithm, qubit, and label metadata are excluded from model text.",
        "group_policy": "Connect each upstream correct/buggy pair and exact QASM repeats before five-fold grouped splitting.",
        "limitations": (["only files safely loaded and normalized by the restricted Qiskit 1.4.5 compatibility loader were included"] +
                        (["not all Mute-Alg source files could be deserialized or normalized"] if failed_files else [])),
        "split_seed": SEED, "frozen_test_accessed": False}
    (output / "preparation_report.json").write_text(json.dumps(prep, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if is_ready:
      contract = {"source_id": "mute_alg_hornbro", "task_id": "compatible_subset_qasm_v2",
        "target_semantics": "Upstream Mute-Alg original (0) versus synthetically buggy (1) circuits represented as OpenQASM 2; compatibility-selected subset only, not hardware diagnosis.",
        "required": fields, "record_id_column": "record_id", "group_column": "bug_id", "label_column": "label",
        "feature_column": "code", "model_feature_fields": ["code"], "allowed_labels": ["0", "1"],
        "fold_record_column": "record_id", "fold_group_column": "group_id", "fold_column": "fold",
        "fold_model_column": "model", "fold_model_value": fold_model,
        "input_path": f"{output.relative_to(ROOT).as_posix()}/normalized_manifest.csv",
        "fold_path": f"{output.relative_to(ROOT).as_posix()}/group_folds.csv",
        "output_path": f"data/processed/mute_alg/compatible_subset_qasm_v2{suffix}",
        "feature_policy": prep["feature_policy"], "grouping_policy": prep["group_policy"],
        "split_seed": SEED}
      contract_path = ROOT / f"experiments/source_task_contracts/mute_alg_compatible_subset_qasm_v2{suffix}.json"
      if contract_path.exists():
          raise FileExistsError(f"Refusing to overwrite contract: {contract_path}")
      contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
      contract_path = None
    print(json.dumps({"output": str(output.relative_to(ROOT)), "contract": str(contract_path.relative_to(ROOT)),
        "source_files": len(file_audit), "loaded_files": prep["successfully_loaded_files"],
        "blocked_files": len(failed_files), "input_rows": len(records), "accepted_rows": len(accepted),
        "rejected_rows": len(rejected), "groups": prep["groups"], "folds": prep["folds"],
        "status": prep["status"], "split_failure_reason": split_failure,
        "contract": str(contract_path.relative_to(ROOT)) if contract_path else None}, indent=2))


if __name__ == "__main__":
    main()
