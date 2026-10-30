"""Extract QBugs correct/buggy circuit pairs from the HornBro Zenodo artifact."""
from __future__ import annotations

import hashlib
import io
import json
import pickle
import os
import csv
from pathlib import Path

from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("QBUGS_DATASET_ROOT", ROOT / "data/external_quantum_datasets/qbugs_hornbro/hornbro/datasets/Qbugs"))
OUT = Path(os.environ.get("QBUGS_OUTPUT", ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/qbugs"))
DATASET_NAME = os.environ.get("QBUGS_DATASET_NAME", "QBugs circuit pairs distributed with HornBro v3")


class RestrictedQiskitUnpickler(pickle.Unpickler):
    ALLOWED = ("qiskit", "numpy", "sympy", "builtins", "collections", "enum", "types",
               "copyreg", "re", "fractions", "decimal", "math", "uuid", "datetime",
               "operator", "functools")

    def find_class(self, module: str, name: str):
        if module.startswith(self.ALLOWED):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Blocked pickle global: {module}.{name}")


def circuit_tokens(circuit: QuantumCircuit) -> str:
    tokens = [f"qubits_{circuit.num_qubits}", f"clbits_{circuit.num_clbits}",
              f"depth_{circuit.depth()}", f"ops_{len(circuit.data)}"]
    try:
        tokens.append(f"parameters_{len(circuit.parameters)}")
    except Exception:
        pass
    for item in circuit.data:
        op = item.operation
        qids = [circuit.find_bit(q).index for q in item.qubits]
        cids = [circuit.find_bit(c).index for c in item.clbits]
        operands = "_".join(f"q{i}" for i in qids) or "none"
        cbits = "_".join(f"c{i}" for i in cids) or "none"
        tokens.append(f"gate_{op.name}_on_{operands}_cbits_{cbits}")
    return " ".join(tokens)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, source_files = [], []
    for path in sorted(SRC.rglob("qcs.pkl")):
        payload = path.read_bytes()
        try:
            circuits = RestrictedQiskitUnpickler(io.BytesIO(payload)).load()
        except Exception as exc:
            raise RuntimeError(f"Could not load documented QBugs/Mute-Alg pickle under Qiskit compatibility env: {path}") from exc
        if not isinstance(circuits, list) or len(circuits) % 2:
            raise ValueError(f"Unexpected QBugs qcs.pkl layout: {path}")
        half = len(circuits) // 2
        if not all(isinstance(c, QuantumCircuit) for c in circuits):
            raise TypeError(f"Non-QuantumCircuit entry in {path}")
        scenario = path.parent.relative_to(SRC).as_posix()
        digest = hashlib.sha256(payload).hexdigest()
        source_files.append({"path": str(path.relative_to(ROOT)), "sha256": digest,
                             "bytes": len(payload), "circuits": len(circuits), "pairs": half,
                             "scenario": scenario})
        for i in range(half):
            gid = str(int(hashlib.sha256(f"{scenario}:{i}".encode()).hexdigest()[:12], 16))
            for circuit, label in ((circuits[i], 0), (circuits[i + half], 1)):
                kind = "correct" if label == 0 else "buggy"
                rows.append({"record_id": f"qbugs-{hashlib.sha256(scenario.encode()).hexdigest()[:8]}-{i}-{kind}",
                             "bug_id": gid, "label": label, "code": circuit_tokens(circuit),
                             "scenario": scenario, "pair_index": i,
                             "source_pickle": str(path.relative_to(ROOT)), "circuit_qubits": circuit.num_qubits,
                             "circuit_depth": circuit.depth(), "instruction_count": len(circuit.data)})
        del circuits
    fields = list(rows[0])
    with (OUT / "normalized_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    class_counts = {str(k): sum(int(r["label"]) == k for r in rows) for k in (0, 1)}
    (OUT / "preparation_manifest.json").write_text(json.dumps({
        "dataset": DATASET_NAME,
        "source_archive": "data/external_quantum_datasets/hornbro_qbugs_source.rar",
        "source_readme": "data/external_quantum_datasets/qbugs_hornbro/provenance/README.md",
        "label_map": {"correct": 0, "buggy": 1},
        "pair_rule": "The README states first half of each qcs.pkl list is original, second half buggy; pair index i in first half is paired with i+half in second half.",
        "group_rule": "scenario path plus pair index; both paired circuits stay in one fold",
        "rows": len(rows), "groups": len({r["bug_id"] for r in rows}),
        "source_files": source_files,
        "class_counts": class_counts,
        "feature_policy": "Qiskit circuit gate/operand sequence plus depth and size tokens; no label, scenario, or pair identifiers enter text features.",
        "qiskit_pickle_environment": "Qiskit 1.4.5 isolated environment; restricted unpickler allows only Qiskit and numeric/container modules.",
        "limitations": "This QBugs delivery is the repair paper's circuit-pair derivative, not a distinct issue-level catalogue count. Group by generated pair origin and do not mix with Bugs4Q.",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "groups": len({r["bug_id"] for r in rows}),
                      "files": len(source_files),
                      "counts": class_counts}, indent=2))


if __name__ == "__main__":
    main()
