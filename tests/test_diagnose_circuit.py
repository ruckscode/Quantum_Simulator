from quantum_simulator import QuantumCircuit, diagnose_circuit
from quantum_simulator.diagnosis_engine import DiagnosisCategory
from quantum_simulator.hardware_model import HardwareModel


def test_diagnose_circuit_localizes_hardware_readout_anomaly():
    hardware = HardwareModel(3)
    hardware.set_qubit_properties(2, readout_error=1.0)
    circuit = [
        ("h", 0),
        ("cx", 0, 1),
        ("cx", 1, 2),
        ("measure", 0, [0]),
        ("measure", 1, [1]),
        ("measure", 2, [2]),
    ]

    result = diagnose_circuit(circuit, hardware_model=hardware, shots=1000, seed=42)

    assert result.anomaly_detected is True
    assert result.category is DiagnosisCategory.HARDWARE_ANOMALY
    assert any(candidate.component_type == "qubit" and candidate.component == 2
               for candidate in result.hardware_evidence_candidates)


def test_diagnose_circuit_without_hardware_compares_ideal_execution():
    result = diagnose_circuit([("h", 0), ("measure", 0, [0])], shots=100, seed=42)

    assert result.anomaly_detected is False
    assert result.category is DiagnosisCategory.NO_ANOMALY


def test_diagnose_circuit_accepts_quantum_circuit_measure_all():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    result = diagnose_circuit(circuit, shots=1000, seed=42)

    assert result.category is DiagnosisCategory.NO_ANOMALY
