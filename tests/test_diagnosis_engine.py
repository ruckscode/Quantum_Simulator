from quantum_simulator.diagnosis_engine import (
    DiagnosisCategory,
    diagnose_execution,
    format_program_fault_diagnosis,
)
from quantum_simulator.hardware_model import HardwareModel


EXPECTED = {"00": 1.0, "11": 0.0}
ANOMALOUS = {"00": 0.0, "11": 1.0}


def test_normal_execution_is_no_anomaly():
    result = diagnose_execution(
        expected_distribution=EXPECTED,
        observed_distribution=EXPECTED,
        expected_circuit=[("h", 0)],
        observed_circuit=[("h", 0)],
        hardware_model=HardwareModel(1),
    )

    assert result.anomaly_detected is False
    assert result.category is DiagnosisCategory.NO_ANOMALY
    assert result.statistics_result.total_variation_distance == 0.0


def test_precomputed_trace_is_passed_to_program_localizer(monkeypatch):
    import quantum_simulator.fault_localizer as fault_localizer

    trace = (0.2, [{
        "step": 0,
        "operation": "h",
        "expected": ("h", 0),
        "actual": ("h", 0),
        "difference": 0.7,
    }])

    def unexpected_trace_call(*args, **kwargs):
        raise AssertionError("_trace_deviation must not run for a supplied trace")

    monkeypatch.setattr(fault_localizer, "_trace_deviation", unexpected_trace_call)
    result = diagnose_execution(
        expected_circuit=[("h", 0)],
        observed_circuit=[("h", 0)],
        precomputed_trace=trace,
    )

    assert result.program_result.anomaly_metric == 0.2
    assert result.program_result.candidates[0].score == 0.7


def test_program_fault_on_healthy_hardware_is_localized_as_program_fault():
    result = diagnose_execution(
        expected_distribution=EXPECTED,
        observed_distribution=ANOMALOUS,
        expected_circuit=[("x", 0)],
        observed_circuit=[("h", 0)],
        hardware_model=HardwareModel(1),
    )

    assert result.anomaly_detected is True
    assert result.category is DiagnosisCategory.PROGRAM_FAULT
    assert result.program_result.candidates
    assert not result.hardware_evidence_candidates


def test_direct_circuit_mismatch_is_program_fault_without_distribution_anomaly():
    result = diagnose_execution(
        expected_circuit=[("x", 0), ("measure", 0, [0])],
        observed_circuit=[("h", 0), ("measure", 0, [0])],
        shots=1000,
    )

    assert result.category is DiagnosisCategory.PROGRAM_FAULT
    assert result.anomaly_detected is True
    assert result.program_result.candidates
    assert result.program_result.candidates[0].index == 0


def test_program_fault_formatter_uses_candidate_details():
    result = diagnose_execution(
        expected_distribution=EXPECTED,
        observed_distribution=ANOMALOUS,
        expected_circuit=[("x", 0)],
        observed_circuit=[("h", 0)],
        hardware_model=HardwareModel(1),
    )

    assert format_program_fault_diagnosis(result) == (
        "=== PROGRAM FAULT DETECTED ===\n"
        "\n"
        "Fault location : Step 0\n"
        "Faulty gate    : H(0)\n"
        "Expected gate  : X(0)\n"
        "Mismatch score : 0.50\n"
        "\n"
        "Reason:\n"
        "Gate operation differs from expected circuit\n"
        "\n"
        "Suggested correction:\n"
        "Replace H(0) with X(0)"
    )


def test_hardware_degradation_with_correct_program_is_hardware_anomaly():
    hardware = HardwareModel(1)
    hardware.set_qubit_properties(0, readout_error=0.45)
    circuit = [("measure",)]

    result = diagnose_execution(
        expected_distribution=EXPECTED,
        observed_distribution=ANOMALOUS,
        expected_circuit=circuit,
        observed_circuit=circuit,
        hardware_model=hardware,
    )

    assert result.category is DiagnosisCategory.HARDWARE_ANOMALY
    assert result.program_result.candidates == []
    assert result.hardware_evidence_candidates


def test_simultaneous_program_and_hardware_evidence_is_ambiguous():
    hardware = HardwareModel(1)
    hardware.set_qubit_properties(0, readout_error=0.45)

    result = diagnose_execution(
        expected_distribution=EXPECTED,
        observed_distribution=ANOMALOUS,
        expected_circuit=[("x", 0)],
        observed_circuit=[("h", 0)],
        hardware_model=hardware,
    )

    assert result.category is DiagnosisCategory.AMBIGUOUS
    assert result.program_result.candidates
    assert result.hardware_evidence_candidates


def test_missing_hardware_calibration_is_reported_without_blocking_diagnosis():
    result = diagnose_execution(
        expected_distribution=EXPECTED,
        observed_distribution=ANOMALOUS,
        expected_circuit=[("x", 0)],
        observed_circuit=[("x", 0)],
    )

    assert result.category is DiagnosisCategory.INSUFFICIENT_EVIDENCE
    assert result.hardware_result is None
    assert any("hardware calibration" in item.lower() for item in result.missing_evidence)


def test_missing_distribution_stays_insufficient_and_reports_missing_evidence():
    result = diagnose_execution(
        expected_circuit=[("h", 0)],
        observed_circuit=[("h", 0)],
        hardware_model=HardwareModel(1),
    )

    assert result.anomaly_detected is None
    assert result.category is DiagnosisCategory.INSUFFICIENT_EVIDENCE
    assert any("distribution" in item.lower() for item in result.missing_evidence)


def test_multiple_program_candidates_are_preserved_with_evidence():
    result = diagnose_execution(
        expected_distribution=EXPECTED,
        observed_distribution=ANOMALOUS,
        expected_circuit=[("x", 0), ("h", 0)],
        observed_circuit=[("h", 0), ("x", 0)],
    )

    assert result.category is DiagnosisCategory.PROGRAM_FAULT
    assert len(result.program_result.candidates) >= 2
    assert all(candidate.evidence for candidate in result.program_result.candidates)


def test_multi_qubit_execution_integrates_hardware_localizer():
    hardware = HardwareModel(2, coupling_map=[(0, 1)])
    hardware.add_edge(0, 1, error=0.4)
    circuit = [("h", 0), ("cnot", 0, 1)]

    result = diagnose_execution(
        expected_distribution={"00": 0.5, "11": 0.5},
        observed_distribution={"00": 1.0, "11": 0.0},
        expected_circuit=circuit,
        observed_circuit=circuit,
        hardware_model=hardware,
    )

    assert result.category is DiagnosisCategory.HARDWARE_ANOMALY
    assert result.hardware_result is not None
    assert any(candidate.component_type == "edge" for candidate in result.hardware_evidence_candidates)


def test_multiple_hardware_candidates_are_preserved():
    hardware = HardwareModel(2, coupling_map=[(0, 1)])
    hardware.set_qubit_properties(0, readout_error=0.4)
    hardware.add_edge(0, 1, error=0.4)
    circuit = [("h", 0), ("cnot", 0, 1), ("measure",)]

    result = diagnose_execution(
        expected_distribution={"00": 1.0, "11": 0.0},
        observed_distribution={"00": 0.0, "11": 1.0},
        expected_circuit=circuit,
        observed_circuit=circuit,
        hardware_model=hardware,
    )

    assert result.category is DiagnosisCategory.HARDWARE_ANOMALY
    assert {candidate.component_type for candidate in result.hardware_evidence_candidates} == {"qubit", "edge"}


def test_integer_valued_gate_parameter_is_not_a_qubit_reference():
    result = diagnose_execution(
        expected_distribution={"0": 1.0, "1": 0.0},
        observed_distribution={"0": 1.0, "1": 0.0},
        expected_circuit=[("rx", 0, 1)],
        observed_circuit=[("rx", 0, 1)],
        hardware_model=HardwareModel(1),
    )

    assert result.category is DiagnosisCategory.NO_ANOMALY
    assert result.program_result is not None


def test_partial_calibration_reports_unavailable_gate_evidence():
    result = diagnose_execution(
        expected_distribution=EXPECTED,
        observed_distribution=EXPECTED,
        expected_circuit=[("h", 0)],
        observed_circuit=[("h", 0)],
        hardware_model=HardwareModel(1),
    )

    assert any("gate calibration" in item.lower() for item in result.missing_evidence)


def test_invalid_hardware_qubit_reference_is_rejected():
    try:
        diagnose_execution(
            expected_distribution=EXPECTED,
            observed_distribution=EXPECTED,
            expected_circuit=[("x", 1)],
            observed_circuit=[("x", 1)],
            hardware_model=HardwareModel(1),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid hardware reference should raise ValueError")
