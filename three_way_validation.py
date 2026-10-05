"""Three-environment circuit validation runner, separate from simulator core."""

from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ibm_validation import (
    check_ibm_access_status,
    discover_ibm_backends,
    execute_on_ibm_backend,
)
from quantum_simulator import QuantumCircuit
from statistics import compare_distributions


SHOTS = 1000
SEED = 1729


@dataclass(frozen=True)
class CircuitDefinition:
    name: str
    n_qubits: int
    instructions: tuple[tuple[Any, ...], ...]


def _definitions() -> dict[str, CircuitDefinition]:
    pi = math.pi
    return {
        "X": CircuitDefinition("X", 1, (("x", 0),)),
        "H": CircuitDefinition("H", 1, (("h", 0),)),
        "Bell": CircuitDefinition("Bell", 2, (("h", 0), ("cx", 0, 1))),
        "GHZ": CircuitDefinition("GHZ", 3, (("h", 0), ("cx", 0, 1), ("cx", 1, 2))),
        "RX(pi)": CircuitDefinition("RX(pi)", 1, (("rx", 0, pi),)),
        "RY(pi)": CircuitDefinition("RY(pi)", 1, (("ry", 0, pi),)),
    }


EXPERIMENTS = _definitions()


def _measurements(n_qubits: int) -> tuple[tuple[Any, ...], ...]:
    qubits = list(range(n_qubits))
    return (("measure", qubits, qubits.copy()),)


def _canonical_counts(raw: Mapping[str, int], *, reverse_bits: bool) -> dict[str, int]:
    counts: dict[str, int] = {}
    for outcome, count in raw.items():
        key = "".join(str(outcome).split())
        if reverse_bits:
            key = key[::-1]
        counts[key] = counts.get(key, 0) + int(count)
    return dict(sorted(counts.items()))


def _distribution(counts: Mapping[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("count output must have a positive total")
    return {key: value / total for key, value in sorted(counts.items())}


def _environment(status: str, reason: str | None = None, **fields: Any) -> dict[str, Any]:
    return {"status": status, "reason": reason, "raw_counts": None,
            "normalized_counts": None, "probability_distribution": None, **fields}


def _run_aer(instructions: Sequence[Any], n_qubits: int, shots: int, seed: int) -> dict[str, Any]:
    if importlib.util.find_spec("qiskit") is None:
        return _environment("UNAVAILABLE", "Qiskit is not installed.", dependencies={"qiskit": False, "qiskit_aer": importlib.util.find_spec("qiskit_aer") is not None})
    aer_available = importlib.util.find_spec("qiskit_aer") is not None
    if not aer_available:
        return _environment("UNAVAILABLE", "Qiskit Aer is not installed.", dependencies={"qiskit": True, "qiskit_aer": False})
    try:
        from qiskit import QuantumCircuit as QiskitCircuit
        from qiskit_aer import AerSimulator

        qc = QiskitCircuit(n_qubits, n_qubits)
        for instruction in instructions:
            name, *args = instruction
            if name == "cx":
                qc.cx(*args)
            elif name in {"rx", "ry"}:
                theta, qubit = args[1], args[0]
                getattr(qc, name)(theta, qubit)
            else:
                getattr(qc, name)(*args)
        qc.measure(range(n_qubits), range(n_qubits))
        backend = AerSimulator()
        result = backend.run(qc, shots=shots, seed_simulator=seed).result()
        raw = {str(key): int(value) for key, value in result.get_counts(qc).items()}
        normalized = _canonical_counts(raw, reverse_bits=True)
        return _environment("COMPLETED", dependencies={"qiskit": True, "qiskit_aer": True},
                            raw_counts=raw, normalized_counts=normalized,
                            probability_distribution=_distribution(normalized), seed_used=seed)
    except Exception as exc:
        return _environment("FAILED", f"{type(exc).__name__}: {exc}", dependencies={"qiskit": True, "qiskit_aer": True})


def run_three_way_validation(
    name: str,
    *,
    n_qubits: int | None = None,
    instructions: Sequence[Sequence[Any]] | None = None,
    backend_name: str | None = None,
    shots: int = SHOTS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run a preset or caller-supplied operation-list circuit."""
    if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    if name in EXPERIMENTS and n_qubits is None and instructions is None:
        definition = EXPERIMENTS[name]
        circuit_name, width, gates = definition.name, definition.n_qubits, definition.instructions
        measurement_mapping = {str(q): q for q in range(width)}
    else:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits <= 0:
            raise ValueError("n_qubits must be a positive integer for a custom circuit")
        if instructions is None:
            raise ValueError("instructions are required for a custom circuit")
        circuit_name, width = name, n_qubits
        gates = tuple(tuple(item) for item in instructions)
        measurement_mapping = {str(q): q for q in range(width)}
    measurements = _measurements(width)
    # Use a full, fixed mapping for comparable n-bit outcomes across all targets.
    gate_instructions = tuple(item for item in gates if str(item[0]).lower() != "measure")
    full_instructions = [*gate_instructions, *measurements]

    circuit = QuantumCircuit(width)
    circuit.instructions.extend(full_instructions)
    our_raw = circuit.run(shots=shots, seed=seed).get_counts()
    our_counts = _canonical_counts(our_raw, reverse_bits=False)
    our = _environment("COMPLETED", raw_counts=our_raw, normalized_counts=our_counts,
                       probability_distribution=_distribution(our_counts), seed_used=seed)

    aer = _run_aer(gate_instructions, width, shots, seed)

    access = check_ibm_access_status()
    ibm_information: dict[str, Any] = {"access": access.to_dict(), "backend_name": backend_name}
    if access.status != "ACCESS_CONFIGURED":
        ibm = _environment("UNAVAILABLE", access.reason, **ibm_information)
    else:
        if backend_name is None:
            discovery = discover_ibm_backends()
            candidates = [b for b in discovery.get("backends", []) if b.get("operational") is not False]
            if not candidates:
                ibm = _environment("UNAVAILABLE", discovery.get("error") or "No operational IBM backend was discovered.",
                                   **{**ibm_information, "backend_discovery": discovery})
            else:
                backend_name = str(candidates[0]["name"])
                ibm_information["backend_name"] = backend_name
                ibm = None
        else:
            ibm = None
        if ibm is None:
            result = execute_on_ibm_backend(full_instructions, backend_name=backend_name, shots=shots)
            result_dict = result.to_dict()
            if result.real_hardware_executed and result.observed_counts is not None:
                raw = dict(result.observed_counts)
                normalized = _canonical_counts(raw, reverse_bits=True)
                ibm = _environment("COMPLETED", **ibm_information, raw_counts=raw,
                                   normalized_counts=normalized,
                                   probability_distribution=_distribution(normalized),
                                   seed_used=None, seed_note="IBM hardware execution is not seeded.",
                                   adapter_report=result_dict)
            else:
                ibm = _environment("UNAVAILABLE" if result.overall_status == "BLOCKED" else "FAILED",
                                   "; ".join(result.errors) or result.overall_status,
                                   **ibm_information, adapter_report=result_dict)

    environments = {"our_simulator": our, "qiskit_aer": aer, "ibm_quantum": ibm}
    comparisons: dict[str, Any] = {}
    pairs = (("qiskit_vs_our_simulator", "qiskit_aer", "our_simulator"),
             ("ibm_vs_our_simulator", "ibm_quantum", "our_simulator"),
             ("ibm_vs_qiskit", "ibm_quantum", "qiskit_aer"))
    for label, left, right in pairs:
        a, b = environments[left], environments[right]
        if a["probability_distribution"] is None or b["probability_distribution"] is None:
            comparisons[label] = None
            continue
        keys = sorted(set(a["probability_distribution"]) | set(b["probability_distribution"]))
        da = {key: a["probability_distribution"].get(key, 0.0) for key in keys}
        db = {key: b["probability_distribution"].get(key, 0.0) for key in keys}
        comparisons[label] = asdict(compare_distributions(da, db, shots=shots))

    return {
        "schema_version": "1.0",
        "circuit": {"name": circuit_name, "n_qubits": width,
                    "instructions": [list(item) for item in gates],
                    "shots": shots, "seed": seed,
                    "measurement_mapping": measurement_mapping,
                    "measurement_instructions": [list(item) for item in measurements],
                    "canonical_bit_order": "left-to-right qubit index order (q0 first)"},
        "environments": environments,
        "pairwise_comparisons": comparisons,
    }


def run_validation_suite(*, backend_name: str | None = None, shots: int = SHOTS, seed: int = SEED) -> dict[str, Any]:
    return {"schema_version": "1.0", "experiments": [
        run_three_way_validation(name, backend_name=backend_name, shots=shots, seed=seed)
        for name in EXPERIMENTS
    ]}


def write_json_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Write a generated report; this does not trigger any additional execution."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


__all__ = ["CircuitDefinition", "EXPERIMENTS", "run_three_way_validation", "run_validation_suite", "write_json_report"]
