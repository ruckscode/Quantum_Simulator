"""Integration checks for traced program-fault evidence."""

import pytest

from quantum_simulator.diagnosis_engine import diagnose_execution
from quantum_simulator.quantum_simulator import QuantumSimulator


@pytest.mark.parametrize(
    "n_qubits,reference,faulty,expected_operation,expected_qubits",
    [
        (2, [("h", 0), ("x", 1), ("z", 0)], [("h", 0), ("y", 1), ("z", 0)], "y", (1,)),
        (2, [("h", 0), ("x", 1), ("z", 0)], [("h", 0), ("z", 0)], "x", (1,)),
        (2, [("h", 0), ("z", 0)], [("h", 0), ("x", 1), ("z", 0)], "x", (1,)),
        (1, [("h", 0), ("ry", 0, 0.2), ("x", 0)], [("h", 0), ("ry", 0, 1.1), ("x", 0)], "ry", (0,)),
        (2, [("h", 0), ("h", 1), ("cx", 0, 1)], [("h", 0), ("h", 1), ("cz", 0, 1)], "cz", (0, 1)),
    ],
    ids=["gate-replacement", "gate-deletion", "gate-insertion", "parameter-change", "two-qubit-replacement"],
)
def test_known_program_fault_is_localized_from_traced_execution(
    n_qubits, reference, faulty, expected_operation, expected_qubits
):
    _, _, _, trace = QuantumSimulator(n_qubits).run_circuit_traced(reference, faulty)
    result = diagnose_execution(
        expected_circuit=reference,
        observed_circuit=faulty,
        precomputed_trace=trace,
    )

    assert result.program_result is not None
    assert result.program_result.candidates
    assert any(
        candidate.operation == expected_operation
        and candidate.qubits == expected_qubits
        for candidate in result.program_result.candidates
    ), result.program_result.candidates


def test_identical_traced_circuit_does_not_report_program_fault():
    circuit = [("h", 0), ("cx", 0, 1), ("rz", 1, 0.4)]
    _, _, _, trace = QuantumSimulator(2).run_circuit_traced(circuit, circuit)
    result = diagnose_execution(
        expected_circuit=circuit,
        observed_circuit=circuit,
        precomputed_trace=trace,
    )

    assert result.program_result is not None
    assert result.program_result.anomaly_metric == pytest.approx(0.0)
    assert result.program_result.candidates == []

