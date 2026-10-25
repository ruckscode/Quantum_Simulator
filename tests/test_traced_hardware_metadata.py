"""Hardware calibration evidence emitted by operation-level traces."""

import numpy as np

from quantum_simulator.hardware_model import HardwareModel
from quantum_simulator.quantum_simulator import QuantumSimulator


def _trace(circuit, model, n_qubits=1):
    return QuantumSimulator(n_qubits).run_circuit_traced(
        circuit, circuit, hardware_model=model
    )


def test_healthy_calibration_metadata_reports_resolved_values_and_no_failure():
    model = HardwareModel(1)
    model.add_gate_calibration("h", qubits=(0,), error=0.0, duration=5.0)
    model.set_qubit_properties(0, t1=120.0, t2=90.0)

    _, _, _, trace = _trace([("h", 0)], model)
    row = trace[0]
    assert row["resolved_calibration_name"] == "h"
    assert row["gate_error"] == 0.0
    assert row["gate_duration"] == 5.0
    assert row["gate_failed"] is False
    assert row["affected_hardware_qubits"] == [0]
    assert row["t1"] == {0: 120.0}
    assert row["t2"] == {0: 90.0}
    assert row["readout_error"] is None


def test_nonzero_gate_error_records_actual_calibrated_failure():
    model = HardwareModel(1)
    model.add_gate_calibration("x", qubits=(0,), error=1.0, duration=2.0)

    _, actual_states, _, trace = _trace([("x", 0)], model)
    row = trace[0]
    assert row["gate_error"] == 1.0
    assert row["gate_failed"] is True
    np.testing.assert_allclose(actual_states[-1], [1.0, 0.0])


def test_measurement_trace_reports_readout_error_for_affected_qubit():
    model = HardwareModel(1)
    model.set_qubit_properties(0, readout_error=0.35)

    _, _, _, trace = _trace([("measure", 0)], model)
    row = trace[0]
    assert row["affected_hardware_qubits"] == [0]
    assert row["readout_error"] == {0: 0.35}
    assert row["resolved_calibration_name"] is None
    assert row["gate_error"] is None
    assert row["gate_duration"] is None
    assert row["gate_failed"] is None
    assert row["t1"] is None
    assert row["t2"] is None


def test_gate_trace_reports_relevant_t1_t2_values():
    model = HardwareModel(1)
    model.add_gate_calibration("rx", qubits=(0,), error=0.02, duration=8.0)
    model.set_qubit_properties(0, t1=44.0, t2=31.0)

    _, _, _, trace = _trace([("rx", 0, 0.6)], model)
    assert trace[0]["t1"] == {0: 44.0}
    assert trace[0]["t2"] == {0: 31.0}
    assert trace[0]["gate_duration"] == 8.0


def test_multi_qubit_gate_metadata_uses_pair_calibration_and_both_qubits():
    model = HardwareModel(2)
    model.add_gate_calibration("cx", qubits=(0, 1), error=0.07, duration=12.0)
    model.set_qubit_properties(0, t1=100.0, t2=70.0)
    model.set_qubit_properties(1, t1=110.0, t2=80.0)

    _, _, _, trace = _trace([("cnot", 0, 1)], model, n_qubits=2)
    row = trace[0]
    assert row["resolved_calibration_name"] == "cx"
    assert row["gate_error"] == 0.07
    assert row["gate_duration"] == 12.0
    assert row["affected_hardware_qubits"] == [0, 1]
    assert row["t1"] == {0: 100.0, 1: 110.0}
    assert row["t2"] == {0: 70.0, 1: 80.0}
    assert isinstance(row["gate_failed"], bool)


def test_seeded_traced_execution_matches_untraced_hardware_execution():
    model = HardwareModel(2, default_gate_error=0.2, default_gate_duration=0.3)
    model.set_qubit_properties(0, t1=1.1, t2=0.8, readout_error=0.15)
    model.set_qubit_properties(1, t1=1.4, t2=0.9, readout_error=0.25)
    circuit = [("h", 0), ("cx", 0, 1), ("rz", 1, 0.37), ("measure", [0, 1])]
    seed = 174

    traced = QuantumSimulator(2, seed=seed)
    _, traced_states, _, trace = traced.run_circuit_traced(
        circuit, circuit, hardware_model=model, seed=seed
    )
    direct = QuantumSimulator(2, seed=seed)
    results = direct.run_circuit(circuit, hardware_model=model)

    np.testing.assert_allclose(traced_states[-1], direct.state, atol=1e-12)
    assert trace[-1]["actual"] == circuit[-1]
    assert len(results) == 1
    assert len(results[0]) == 2

