"""Offline demonstration of the Quantum Simulator diagnosis workflow."""

from __future__ import annotations

from pathlib import Path
import sys

# Allow running this script directly from a source checkout without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantum_simulator import QuantumCircuit, diagnose_circuit
from quantum_simulator.diagnosis_engine import (
    DiagnosisCategory,
    diagnose_execution,
    format_program_fault_diagnosis,
)
from quantum_simulator.hardware_model import HardwareModel


def _gate_label(signature: object) -> str:
    """Render the public localizer signature as GATE(qubits)."""
    if isinstance(signature, (tuple, list)) and signature:
        name = str(signature[0]).upper()
        # Localization signatures are (name, qubits, parameters).
        qubits = signature[1] if len(signature) > 1 else ()
        if not isinstance(qubits, (tuple, list)):
            qubits = (qubits,)
        return f"{name}({', '.join(str(q) for q in qubits)})"
    return str(signature)


def _diagnosis_instructions(circuit: QuantumCircuit) -> list[tuple[object, ...]]:
    """Express QuantumCircuit.measure_all() with explicit classical bits.

    This equivalent instruction shape is accepted by the current diagnosis
    canonicalizer, which does not handle the shorthand emitted by measure_all.
    """
    instructions: list[tuple[object, ...]] = []
    for instruction in circuit.instructions:
        if (instruction and instruction[0] == "measure" and len(instruction) == 2
                and isinstance(instruction[1], list)):
            qubits = instruction[1]
            instructions.append(("measure", qubits, list(range(len(qubits)))))
        else:
            instructions.append(instruction)
    return instructions


def main() -> None:
    print("Hardware-Aware Data-Driven Testing and Fault Localization")
    print("Final Project Demonstration")

    summaries: list[tuple[str, DiagnosisCategory]] = []

    print("\n==================================================")
    print("CASE 1: HEALTHY CIRCUIT")
    print("==================================================")
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure_all()
    bell_result = bell.run(shots=1000, seed=1729)
    healthy = diagnose_circuit(_diagnosis_instructions(bell), shots=1000, seed=1729)
    healthy_category = healthy.category
    summaries.append(("Healthy circuit", healthy_category))
    print(f"Diagnosis: {healthy_category.value.replace('_', ' ')}")
    print(f"Observed counts (1000 shots): {bell_result.get_counts()}")
    if healthy.statistics_result is not None:
        print(f"Observed distribution: {healthy.statistics_result.observed_distribution}")

    print("\n==================================================")
    print("CASE 2: PROGRAM FAULT")
    print("==================================================")
    expected = QuantumCircuit(1)
    expected.x(0)
    expected.measure(0)
    observed = QuantumCircuit(1)
    observed.h(0)
    observed.measure(0)
    expected_counts = expected.run(shots=1000, seed=1729).get_counts()
    observed_counts = observed.run(shots=1000, seed=1730).get_counts()
    all_outcomes = set(expected_counts) | set(observed_counts)
    expected_distribution = {
        outcome: expected_counts.get(outcome, 0) / 1000 for outcome in all_outcomes
    }
    observed_distribution = {
        outcome: observed_counts.get(outcome, 0) / 1000 for outcome in all_outcomes
    }
    program = diagnose_execution(
        expected_distribution,
        observed_distribution,
        expected_circuit=expected.instructions,
        observed_circuit=observed.instructions,
        shots=1000,
    )
    program_category = program.category
    summaries.append(("Program fault", program_category))
    print(f"Diagnosis: {program_category.value.replace('_', ' ')}")
    candidates = program.program_evidence_candidates
    if not candidates and program.program_result is not None:
        candidates = program.program_result.candidates
    if candidates:
        fault = candidates[0]
        print(f"Fault location: Step {fault.index}")
        print(f"Expected gate: {_gate_label(fault.expected_signature)}")
        print(f"Observed gate: {_gate_label(fault.actual_signature)}")
        print(f"Mismatch score: {fault.score:.2f}")
        print(f"Reason: {fault.reason}")
        print("\nDiagnosis formatter:")
        print(format_program_fault_diagnosis(program))
    else:
        print(program.summary)

    print("\n==================================================")
    print("CASE 3: HARDWARE FAULT")
    print("==================================================")
    ghz = QuantumCircuit(3)
    ghz.h(0)
    ghz.cx(0, 1)
    ghz.cx(1, 2)
    ghz.measure_all()
    hardware = HardwareModel(3, coupling_map=[(0, 1), (1, 2)])
    hardware.set_qubit_properties(0, readout_error=0.0)
    hardware.set_qubit_properties(1, readout_error=0.0)
    hardware.set_qubit_properties(2, readout_error=1.0)
    hardware_result = diagnose_circuit(
        _diagnosis_instructions(ghz), hardware_model=hardware, shots=1000, seed=1729
    )
    hardware_category = hardware_result.category
    summaries.append(("Hardware fault", hardware_category))
    print(f"Diagnosis: {hardware_category.value.replace('_', ' ')}")
    localized = next(
        (candidate for candidate in hardware_result.hardware_evidence_candidates
         if candidate.component_type == "qubit"),
        None,
    )
    if localized is not None:
        qubit = localized.component
        readout_error = hardware.get_qubit_properties(qubit).readout_error
        print(f"Localized component: Qubit {qubit}")
        print(f"Readout error: {readout_error}")
    elif hardware_result.hardware_result and hardware_result.hardware_result.candidates:
        candidate = hardware_result.hardware_result.candidates[0]
        print(f"Localized component: {candidate.component_type} {candidate.component}")
        if candidate.component_type == "qubit":
            print(f"Readout error: {hardware.get_qubit_properties(candidate.component).readout_error}")
    else:
        print(hardware_result.summary)

    print("\n==================================================")
    print("DEMO SUMMARY")
    print("==================================================")
    for label, category in summaries:
        print(f"{label:<21} -> {category.value.replace('_', ' ')}")


if __name__ == "__main__":
    main()
