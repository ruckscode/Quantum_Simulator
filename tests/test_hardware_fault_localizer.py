import pytest

from quantum_simulator.hardware_fault_localizer import localize_hardware_fault
from quantum_simulator.hardware_model import HardwareModel


def test_healthy_hardware_model_has_no_false_faults():
    model = HardwareModel(2, coupling_map=[(0, 1)])
    expected = [("h", 0), ("cnot", 0, 1)]
    actual = [("h", 0), ("cnot", 0, 1)]

    result = localize_hardware_fault(model, expected, actual, threshold=0.05)

    assert result.candidates == []


def test_unsupported_circuit_marks_hardware_trace_metric_unavailable():
    model = HardwareModel(1)
    circuit = [("unsupported_gate", 0)]

    result = localize_hardware_fault(model, circuit, circuit)

    assert result.anomaly_metric == 0.0  # Existing numeric field retained for compatibility.
    assert "trace localization was skipped" in result.summary.lower()
    assert any("not measured" in item.lower() for item in result.evidence)
    assert result.candidates == []


def test_one_degraded_qubit_is_localized():
    model = HardwareModel(2, coupling_map=[(0, 1)])
    model.set_qubit_properties(1, t1=8e-6, t2=12e-6, readout_error=0.25, status="available")
    expected = [("h", 0), ("cnot", 0, 1)]
    actual = [("h", 0), ("x", 1), ("cnot", 0, 1)]

    result = localize_hardware_fault(model, expected, actual, threshold=0.05)

    assert any(candidate.component_type == "qubit" and candidate.component == 1 for candidate in result.candidates)


def test_degraded_coupling_edge_is_localized():
    model = HardwareModel(2, coupling_map=[(0, 1)])
    edge = model.get_edge(0, 1)
    assert edge is not None
    edge.error = 0.45
    edge.duration = 9.0
    expected = [("h", 0), ("cnot", 0, 1)]
    actual = [("h", 0), ("swap", 0, 1)]

    result = localize_hardware_fault(model, expected, actual, threshold=0.05)

    assert any(candidate.component_type == "edge" and set(candidate.component) == {0, 1} for candidate in result.candidates)


def test_readout_error_localizes_relevant_qubit():
    model = HardwareModel(2, coupling_map=[(0, 1)])
    model.set_qubit_properties(0, readout_error=0.40)
    expected = [("h", 0), ("cnot", 0, 1), ("measure",)]
    actual = [("h", 0), ("cnot", 0, 1), ("measure",)]

    result = localize_hardware_fault(model, expected, actual, threshold=0.05)

    assert any(candidate.component_type == "qubit" and candidate.component == 0 for candidate in result.candidates)


def test_gate_error_is_localized_to_relevant_component():
    model = HardwareModel(2, coupling_map=[(0, 1)])
    model.add_gate_calibration("cx", qubits=(0, 1), error=0.30, duration=2.0, supported=True)
    expected = [("h", 0), ("cnot", 0, 1)]
    actual = [("h", 0), ("swap", 0, 1)]

    result = localize_hardware_fault(model, expected, actual, threshold=0.05)

    assert any(candidate.component_type in {"edge", "gate"} and (0 in getattr(candidate, "component", ()) or set(getattr(candidate, "component", ())) == {0, 1}) for candidate in result.candidates)


@pytest.mark.parametrize("duration", [0.001, 2.0, 100.0])
def test_gate_duration_without_error_is_not_a_fault_candidate(duration):
    model = HardwareModel(1)
    model.add_gate_calibration("i", qubits=(0,), error=0.0, duration=duration)

    result = localize_hardware_fault(model, [("i", 0)], [("i", 0)])

    assert result.candidates == []
    assert "no hardware fault candidates" in result.summary.lower()


@pytest.mark.parametrize("duration", [0.001, 2.0, 100.0])
def test_edge_duration_without_error_is_not_a_fault_candidate(duration):
    model = HardwareModel(2, coupling_map=[(0, 1)])
    edge = model.get_edge(0, 1)
    assert edge is not None
    edge.duration = duration

    result = localize_hardware_fault(model, [("cnot", 0, 1)], [("cnot", 0, 1)])

    assert result.candidates == []
    assert "no hardware fault candidates" in result.summary.lower()


def test_multiple_hardware_issues_returns_interpretable_candidates():
    model = HardwareModel(3, coupling_map=[(0, 1), (1, 2)])
    model.set_qubit_properties(0, readout_error=0.20, t1=50e-6)
    model.set_qubit_properties(1, t1=8e-6, t2=10e-6)
    model.add_edge(1, 2, error=0.35, duration=5.0)
    expected = [("h", 0), ("cnot", 0, 1), ("cnot", 1, 2)]
    actual = [("h", 0), ("x", 1), ("cnot", 0, 1), ("swap", 1, 2)]

    result = localize_hardware_fault(model, expected, actual, threshold=0.05)

    assert len(result.candidates) >= 2
    assert all(hasattr(candidate, "evidence") for candidate in result.candidates)


def test_missing_or_partial_calibration_data_is_handled_without_crashing():
    model = HardwareModel(2, coupling_map=[(0, 1)])
    model._qubits[0].metadata["last_calibration"] = "2025-01-01"
    expected = [("h", 0), ("cnot", 0, 1)]
    actual = [("h", 0), ("swap", 0, 1)]

    result = localize_hardware_fault(model, expected, actual, threshold=0.05)

    assert isinstance(result, type(localize_hardware_fault(model, expected, actual, threshold=0.05)))


def test_trace_candidate_without_calibration_is_not_summarized_as_likely_hardware_fault():
    model = HardwareModel(1)
    expected = [("h", 0), ("measure", 0)]
    actual = [("x", 0), ("measure", 0)]

    result = localize_hardware_fault(model, expected, actual, threshold=0.05)

    assert result.candidates
    assert "without calibration-backed fault evidence" in result.summary.lower()
    assert "likely hardware component" not in result.summary.lower()
    assert any("no gate calibration" in candidate.reason.lower() for candidate in result.candidates)


def test_invalid_hardware_reference_raises_value_error():
    model = HardwareModel(2)
    expected = [("h", 0), ("cnot", 0, 2)]
    actual = [("h", 0), ("cnot", 0, 2)]

    with pytest.raises(ValueError):
        localize_hardware_fault(model, expected, actual, threshold=0.05)


def test_phase_a_regression_gate_module():
    import quantum_simulator.gates as gates

    assert gates.get_gate("h") is gates.H
    assert gates.get_gate_qubit_count("cx") == 2
    assert gates.is_parameterized_gate("rx") is True


def test_phase_b_regression_simulator_module():
    from quantum_simulator import QuantumSimulator

    sim = QuantumSimulator(2)
    sim.run_circuit([("h", 0), ("cnot", 0, 1)])
    expected = [1 / 2**0.5, 0, 0, 1 / 2**0.5]
    assert abs(sim.state[0] - expected[0]) < 1e-12
    assert abs(sim.state[3] - expected[3]) < 1e-12
