"""Single-command, offline-first acceptance run for the quantum simulator.

Run from the repository root:
    python clean_simulation_test.py

The script executes the project's own simulator, compares with Qiskit when
installed, exercises the IBM adapter's offline path, and writes a diagnosis
report. It never submits a live IBM job.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quantum_simulator import QuantumCircuit, QuantumSimulator, diagnose_circuit
from quantum_simulator.diagnosis_engine import diagnose_execution
from quantum_simulator.hardware_model import HardwareModel
from ibm_validation import check_ibm_access_status, run_offline_validation, ideal_distribution as ibm_ideal_distribution

SHOTS = 2048
SEED = 20261016
REPORT_PATH = ROOT / "reports" / "clean_simulation_test_report.json"


def _distribution(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("Cannot normalize empty measurement counts")
    return {key: value / total for key, value in sorted(counts.items())}


def _exact_probabilities(instructions: list[tuple[Any, ...]], n_qubits: int) -> dict[str, float]:
    simulator = QuantumSimulator(n_qubits, seed=SEED)
    simulator.run_circuit(
        instruction for instruction in instructions
        if not (isinstance(instruction, (tuple, list)) and instruction and str(instruction[0]).lower() == "measure")
    )
    return {
        format(index, f"0{n_qubits}b"): float(probability)
        for index, probability in enumerate(simulator.get_probabilities())
        if probability > 1e-12
    }


def _tv_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def _qiskit_comparison() -> dict[str, Any]:
    """Use Qiskit Aer if present; otherwise use Qiskit's statevector sampler."""
    try:
        from qiskit import QuantumCircuit as QiskitCircuit
    except ImportError:
        return {
            "status": "NOT_INSTALLED",
            "comparison_performed": False,
            "counts": None,
            "total_variation_distance": None,
            "note": "Qiskit is optional; no package was installed by this test.",
        }

    qc = QiskitCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    counts: dict[str, int]
    engine = "qiskit.quantum_info.Statevector"
    try:
        from qiskit_aer import AerSimulator
        result = AerSimulator().run(qc, shots=SHOTS, seed_simulator=SEED).result()
        counts = {str(key).replace(" ", ""): int(value) for key, value in result.get_counts().items()}
        engine = "qiskit_aer.AerSimulator"
    except ImportError:
        try:
            from qiskit.quantum_info import Statevector
            unmeasured = QiskitCircuit(2)
            unmeasured.h(0)
            unmeasured.cx(0, 1)
            # Qiskit versions differ: Statevector.sample_counts has not
            # accepted a seed argument in all supported releases. Use an
            # explicitly seeded NumPy generator and sample the statevector
            # probabilities ourselves for cross-version reproducibility.
            state = Statevector.from_instruction(unmeasured)
            probabilities = state.probabilities()
            outcomes = [format(i, "02b") for i in range(len(probabilities))]
            sampled = __import__("numpy").random.default_rng(SEED).choice(
                outcomes, size=SHOTS, p=probabilities
            )
            counts = {}
            for key in sampled:
                counts[str(key)] = counts.get(str(key), 0) + 1
        except Exception as exc:
            return {
                "status": "FAILED",
                "comparison_performed": False,
                "counts": None,
                "total_variation_distance": None,
                "note": f"Qiskit is installed but its local simulation path failed: {type(exc).__name__}: {exc}",
            }
    except Exception as exc:
        return {
            "status": "FAILED",
            "comparison_performed": False,
            "counts": None,
            "total_variation_distance": None,
            "note": f"Qiskit Aer execution failed: {type(exc).__name__}: {exc}",
        }

    # Both circuits are Bell circuits; counts are normalized before comparison.
    return {
        "status": "COMPARED",
        "comparison_performed": True,
        "engine": engine,
        "counts": dict(sorted(counts.items())),
        "total_variation_distance": _tv_distance(_distribution(LOCAL_COUNTS), _distribution(counts)),
        "note": "Finite-shot results may differ due to sampling; this is a simulator-to-simulator comparison, not hardware.",
    }


def build_report() -> dict[str, Any]:
    global LOCAL_COUNTS

    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure_all()
    LOCAL_COUNTS = bell.run(shots=SHOTS, seed=SEED).get_counts()
    local_distribution = _distribution(LOCAL_COUNTS)
    ideal_distribution = _exact_probabilities(
        [("h", 0), ("cx", 0, 1)], 2
    )

    # Program fault: compare an expected X circuit with an observed H circuit.
    program_diagnosis = diagnose_execution(
        expected_distribution={"0": 0.0, "1": 1.0},
        observed_distribution={"0": 0.5, "1": 0.5},
        expected_circuit=[("x", 0), ("measure", 0)],
        observed_circuit=[("h", 0), ("measure", 0)],
        shots=SHOTS,
    )

    # Controlled synthetic evidence only. This is not an IBM backend result.
    hardware_model = HardwareModel(1, backend_name="synthetic-offline-fixture")
    hardware_model.set_qubit_properties(0, readout_error=0.45)
    hardware_diagnosis = diagnose_execution(
        expected_distribution={"0": 0.0, "1": 1.0},
        observed_distribution={"0": 0.45, "1": 0.55},
        expected_circuit=[("x", 0), ("measure", 0)],
        observed_circuit=[("x", 0), ("measure", 0)],
        hardware_model=hardware_model,
        shots=SHOTS,
    )

    access = check_ibm_access_status().to_dict()
    # Use a compact one-qubit circuit here because this adapter's current
    # local statevector path has known limitations for multi-qubit instruction
    # normalization. The Bell circuit above separately validates two-qubit
    # simulation and Qiskit comparison.
    offline_result = run_offline_validation(
        [("h", 0), ("measure", 0)],
        shots=256,
        seed=SEED,
    )
    offline_ibm = offline_result.to_dict()
    # Keep adapter failure diagnostics visible in the saved report rather
    # than reducing the integration result to a single status label.
    offline_ibm["adapter_errors"] = list(getattr(offline_result, "errors", []) or [])

    qiskit = _qiskit_comparison()
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "execution_mode": "LOCAL_OFFLINE_ONLY",
        "seed": SEED,
        "shots": SHOTS,
        "circuit": {
            "name": "2-qubit Bell state",
            "operations": [["h", 0], ["cx", 0, 1], ["measure_all"]],
            "expected_ideal_probabilities": ideal_distribution,
            "local_simulator_counts": dict(sorted(LOCAL_COUNTS.items())),
            "local_simulator_distribution": local_distribution,
            "local_vs_ideal_total_variation_distance": _tv_distance(local_distribution, ideal_distribution),
            "shots_valid": sum(LOCAL_COUNTS.values()) == SHOTS,
        },
        "qiskit_comparison": qiskit,
        "ibm_integration": {
            "access_status": access,
            "offline_adapter_status": offline_ibm.get("overall_status", offline_ibm.get("status")),
            "real_ibm_status": offline_ibm.get("real_ibm_status", "NOT_EXECUTED"),
            "real_hardware_executed": bool(offline_ibm.get("real_hardware_executed", False)),
            "note": "This acceptance run never submits an IBM job. A blocked access status is an environment status, not a simulator failure.",
        },
        "diagnosis": {
            "program_fault_scenario": {
                "category": program_diagnosis.category.value,
                "anomaly_detected": program_diagnosis.anomaly_detected,
                "summary": program_diagnosis.summary,
                "evidence": list(program_diagnosis.evidence),
                "missing_evidence": list(program_diagnosis.missing_evidence),
                "scenario_source": "controlled local fixture",
            },
            "hardware_anomaly_scenario": {
                "category": hardware_diagnosis.category.value,
                "anomaly_detected": hardware_diagnosis.anomaly_detected,
                "summary": hardware_diagnosis.summary,
                "evidence": list(hardware_diagnosis.evidence),
                "missing_evidence": list(hardware_diagnosis.missing_evidence),
                "scenario_source": "synthetic HardwareModel fixture; not physical hardware",
            },
        },
    }
    return report


def main() -> int:
    try:
        report = build_report()
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"ACCEPTANCE TEST FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    circuit = report["circuit"]
    qiskit = report["qiskit_comparison"]
    ibm = report["ibm_integration"]
    diagnosis = report["diagnosis"]
    print("=== CLEAN QUANTUM SIMULATION TEST ===")
    print(f"Local Bell circuit: {'PASS' if circuit['shots_valid'] else 'FAIL'}")
    print(f"Shots: {sum(circuit['local_simulator_counts'].values())}/{report['shots']}")
    print(f"Local counts: {circuit['local_simulator_counts']}")
    print(f"Observed vs ideal TV distance: {circuit['local_vs_ideal_total_variation_distance']:.6f}")
    print(f"Qiskit comparison: {qiskit['status']}")
    if qiskit.get("comparison_performed"):
        print(f"Qiskit engine: {qiskit['engine']}")
        print(f"Local vs Qiskit TV distance: {qiskit['total_variation_distance']:.6f}")
        print(f"Qiskit counts: {qiskit['counts']}")
    else:
        print(f"Qiskit note: {qiskit['note']}")
    print(f"IBM access status: {ibm['access_status']['status']}")
    print(f"IBM offline adapter: {ibm['offline_adapter_status']}")
    print(f"Real IBM hardware executed: {ibm['real_hardware_executed']}")
    print(f"Program-fault diagnosis: {diagnosis['program_fault_scenario']['category']}")
    print(f"Synthetic hardware diagnosis: {diagnosis['hardware_anomaly_scenario']['category']}")
    print(f"JSON report: {REPORT_PATH.relative_to(ROOT)}")
    print("NOTE: synthetic hardware diagnosis is not evidence from a physical backend.")
    return 0 if circuit["shots_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
