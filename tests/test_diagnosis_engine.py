import pytest

from quantum_simulator.diagnosis_engine import (
    DiagnosisCategory,
    diagnose_circuit,
    diagnose_execution,
    format_program_fault_diagnosis,
)
from quantum_simulator.diagnosis_engine import _canonicalize_circuit
from quantum_simulator.hardware_model import HardwareModel


EXPECTED = {"00": 1.0, "11": 0.0}
ANOMALOUS = {"00": 0.0, "11": 1.0}


@pytest.mark.parametrize("gate", ["cp", "crx", "cry", "crz", "rxx", "rzz"])
def test_canonicalize_two_qubit_parameterized_gates_preserves_both_qubits(gate):
    operation = (gate, 0, 1, 0.375)
    assert _canonicalize_circuit([operation]) == [operation]


def test_calibration_warning_without_execution_anomaly_does_not_change_category():
    hardware = HardwareModel(1)
    hardware.set_qubit_properties(0, readout_error=0.30)
    circuit = [("x", 0), ("measure", 0)]

    result = diagnose_execution(
        expected_distribution={"0": 0.0, "1": 1.0},
        observed_distribution={"0": 0.98, "1": 0.02},
        expected_circuit=circuit,
        observed_circuit=circuit,
        hardware_model=hardware,
    )

    assert result.anomaly_detected is False
    assert result.category is DiagnosisCategory.NO_ANOMALY
    assert result.hardware_evidence_candidates
    assert "risk evidence" in result.summary


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


@pytest.mark.parametrize("duration", [0.001, 2.0, 100.0])
def test_healthy_short_or_long_gate_duration_does_not_override_no_anomaly(duration):
    hardware = HardwareModel(1)
    hardware.add_gate_calibration("i", qubits=(0,), error=0.0, duration=duration)

    result = diagnose_execution(
        expected_distribution={"0": 1.0, "1": 0.0},
        observed_distribution={"0": 1.0, "1": 0.0},
        expected_circuit=[("i", 0)],
        observed_circuit=[("i", 0)],
        hardware_model=hardware,
    )

    assert result.anomaly_detected is False
    assert result.category is DiagnosisCategory.NO_ANOMALY
    assert result.hardware_evidence_candidates == []


def test_degraded_gate_error_remains_localizable_with_anomaly_evidence():
    hardware = HardwareModel(1)
    hardware.add_gate_calibration("i", qubits=(0,), error=0.30, duration=2.0)

    result = diagnose_execution(
        expected_distribution=EXPECTED,
        observed_distribution=ANOMALOUS,
        expected_circuit=[("i", 0)],
        observed_circuit=[("i", 0)],
        hardware_model=hardware,
    )

    assert result.anomaly_detected is True
    assert result.category is DiagnosisCategory.HARDWARE_ANOMALY
    assert any(candidate.component_type == "gate" and candidate.component == (0,) for candidate in result.hardware_evidence_candidates)


def test_long_duration_alone_cannot_explain_execution_anomaly():
    hardware = HardwareModel(1)
    hardware.add_gate_calibration("i", qubits=(0,), error=0.0, duration=2.0)

    result = diagnose_execution(
        expected_distribution=EXPECTED,
        observed_distribution=ANOMALOUS,
        expected_circuit=[("i", 0)],
        observed_circuit=[("i", 0)],
        hardware_model=hardware,
    )

    assert result.anomaly_detected is True
    assert result.category is DiagnosisCategory.INSUFFICIENT_EVIDENCE
    assert result.hardware_evidence_candidates == []


def test_edge_duration_alone_cannot_explain_execution_anomaly():
    hardware = HardwareModel(2, coupling_map=[(0, 1)])
    edge = hardware.get_edge(0, 1)
    assert edge is not None
    edge.duration = 100.0

    result = diagnose_execution(
        expected_distribution=EXPECTED,
        observed_distribution=ANOMALOUS,
        expected_circuit=[("cnot", 0, 1)],
        observed_circuit=[("cnot", 0, 1)],
        hardware_model=hardware,
    )

    assert result.anomaly_detected is True
    assert result.category is DiagnosisCategory.INSUFFICIENT_EVIDENCE
    assert result.hardware_result is not None
    assert result.hardware_result.candidates == []
    assert result.hardware_evidence_candidates == []
    assert any("gate calibration entry" in item.lower() for item in result.missing_evidence)


def test_error_candidate_ranks_without_promoting_duration_only_qubit():
    hardware = HardwareModel(2)
    hardware.add_gate_calibration("x", qubits=(0,), error=0.30, duration=0.1)
    hardware.add_gate_calibration("i", qubits=(1,), error=0.0, duration=100.0)

    result = diagnose_execution(
        expected_distribution=EXPECTED,
        observed_distribution=ANOMALOUS,
        expected_circuit=[("x", 0), ("i", 1)],
        observed_circuit=[("x", 0), ("i", 1)],
        hardware_model=hardware,
    )

    assert result.category is DiagnosisCategory.HARDWARE_ANOMALY
    assert result.hardware_evidence_candidates
    assert result.hardware_evidence_candidates[0].component_type == "gate"
    assert result.hardware_evidence_candidates[0].component == (0,)
    assert all(candidate.component != (1,) for candidate in result.hardware_result.candidates)
    assert "calibration-backed hardware fault candidate(s): gate (0,)" in result.hardware_result.summary.lower()


def test_reported_t1_t2_in_seconds_are_not_misclassified_as_zero_microseconds():
    hardware = HardwareModel(1)
    hardware.set_qubit_properties(0, t1=158.084e-6, t2=214.086e-6)
    circuit = [("x", 0), ("measure", 0)]

    result = diagnose_execution(
        expected_distribution={"0": 0.0, "1": 1.0},
        observed_distribution={"0": 0.98, "1": 0.02},
        expected_circuit=circuit,
        observed_circuit=circuit,
        hardware_model=hardware,
    )

    assert result.anomaly_detected is False
    assert result.category is DiagnosisCategory.NO_ANOMALY
    assert result.hardware_evidence_candidates == []
    assert not any("T1 is short" in item or "T2 is short" in item for item in result.evidence)


def test_missing_gate_calibration_is_not_fault_evidence():
    result = diagnose_execution(
        expected_distribution={"0": 1.0, "1": 0.0},
        observed_distribution={"0": 1.0, "1": 0.0},
        expected_circuit=[("i", 0)],
        observed_circuit=[("i", 0)],
        hardware_model=HardwareModel(1),
    )

    assert result.category is DiagnosisCategory.NO_ANOMALY
    assert result.hardware_evidence_candidates == []
    assert any("no gate calibration entry" in message.lower() for message in result.missing_evidence)


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


def test_simulator_trace_list_is_accepted_as_precomputed_diagnostic_trace():
    from quantum_simulator.quantum_simulator import QuantumSimulator

    _, _, _, trace = QuantumSimulator(1).run_circuit_traced(
        [('x', 0)], [('h', 0)]
    )
    result = diagnose_execution(
        expected_circuit=[('x', 0)], observed_circuit=[('h', 0)],
        precomputed_trace=trace,
    )
    assert result.program_result is not None
    assert result.program_result.anomaly_metric == trace[-1]['difference']
    assert result.program_result.candidates


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


def test_anomaly_with_behaviorally_equivalent_source_mismatch_abstains_on_cause():
    result = diagnose_execution(
        expected_distribution={"0": 0.0, "1": 1.0},
        observed_distribution={"0": 1.0, "1": 0.0},
        expected_circuit=[("x", 0), ("measure", 0)],
        observed_circuit=[("y", 0), ("measure", 0)],
    )

    assert result.anomaly_detected is True
    assert result.program_evidence_candidates
    assert result.program_behavioral_candidates == []
    assert result.category is DiagnosisCategory.INSUFFICIENT_EVIDENCE
    assert "not linked to the measured outputs" in result.summary


def test_mismatch_on_unmeasured_disconnected_qubit_cannot_explain_anomaly():
    result = diagnose_execution(
        expected_distribution={"0": 1.0, "1": 0.0},
        observed_distribution={"0": 0.0, "1": 1.0},
        expected_circuit=[("x", 1), ("measure", 0)],
        observed_circuit=[("h", 1), ("measure", 0)],
    )

    assert result.anomaly_detected is True
    assert result.program_evidence_candidates
    assert result.program_behavioral_candidates == []
    assert result.category is DiagnosisCategory.INSUFFICIENT_EVIDENCE


def test_unrelated_hardware_degradation_does_not_localize_anomalous_output():
    hardware = HardwareModel(2)
    hardware.set_qubit_properties(1, readout_error=0.45)
    result = diagnose_execution(
        expected_distribution={"0": 1.0, "1": 0.0},
        observed_distribution={"0": 0.0, "1": 1.0},
        expected_circuit=[("measure", 0)],
        observed_circuit=[("measure", 0)],
        hardware_model=hardware,
    )

    assert result.anomaly_detected is True
    assert result.hardware_evidence_candidates == []
    assert result.category is DiagnosisCategory.INSUFFICIENT_EVIDENCE


def test_program_fault_formatter_does_not_promote_unproven_source_candidate():
    result = diagnose_execution(
        expected_circuit=[("x", 0)],
        observed_circuit=[("y", 0)],
    )

    formatted = format_program_fault_diagnosis(result)
    assert "SOURCE MISMATCH CANDIDATE" in formatted
    assert "Causal attribution is not established." in formatted
    assert "PROGRAM FAULT DETECTED" not in formatted


def test_direct_circuit_mismatch_without_distributions_localizes_but_abstains():
    result = diagnose_execution(
        expected_circuit=[("x", 0), ("measure", 0, [0])],
        observed_circuit=[("h", 0), ("measure", 0, [0])],
        shots=1000,
    )

    assert result.category is DiagnosisCategory.INSUFFICIENT_EVIDENCE
    assert result.anomaly_detected is None
    assert result.program_result.candidates
    assert result.program_result.candidates[0].index == 0
    assert any("cannot establish behavioral impact" in item for item in result.missing_evidence)


def test_identical_unsupported_operations_return_insufficient_evidence():
    unsupported = [("unsupported_gate", 0)]
    hardware = HardwareModel(1)
    hardware.set_qubit_properties(0, readout_error=0.45)
    result = diagnose_execution(
        expected_distribution=EXPECTED,
        observed_distribution=ANOMALOUS,
        expected_circuit=unsupported,
        observed_circuit=unsupported,
        hardware_model=hardware,
    )

    assert result.category is DiagnosisCategory.INSUFFICIENT_EVIDENCE
    assert result.program_evidence_candidates == []
    assert result.hardware_evidence_candidates == []
    assert any("unsupported" in item.lower() for item in result.missing_evidence)
    assert not any("gate calibration" in item.lower() for item in result.missing_evidence)
    assert "skipped" in result.program_result.summary.lower()


def test_unsupported_operation_mismatch_is_not_attributed_as_program_fault():
    result = diagnose_execution(
        expected_distribution=EXPECTED,
        observed_distribution=ANOMALOUS,
        expected_circuit=[("unsupported_gate", 0)],
        observed_circuit=[("other_unsupported_gate", 0)],
        hardware_model=HardwareModel(1),
    )

    assert result.category is DiagnosisCategory.INSUFFICIENT_EVIDENCE
    assert result.program_evidence_candidates == []
    assert result.hardware_evidence_candidates == []
    assert any("unsupported" in item.lower() for item in result.missing_evidence)


def test_supported_direct_mismatch_survives_alongside_unsupported_mismatch():
    result = diagnose_execution(
        expected_distribution=EXPECTED,
        observed_distribution=EXPECTED,
        expected_circuit=[("x", 0), ("unsupported_gate", 1)],
        observed_circuit=[("h", 0), ("other_unsupported_gate", 1)],
        hardware_model=HardwareModel(2),
    )

    # A localized syntactic mismatch does not prove behavioral impact when
    # the observed distribution is unchanged.
    assert result.category is DiagnosisCategory.AMBIGUOUS
    assert [candidate.operation for candidate in result.program_evidence_candidates] == ["h"]
    assert "skipped" in result.program_result.summary.lower()
    assert any("unsupported" in item.lower() for item in result.missing_evidence)


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


def test_nonfinite_observation_cannot_become_a_no_anomaly_diagnosis():
    with pytest.raises(ValueError, match="finite"):
        diagnose_execution(
            expected_distribution=[1.0, 0.0],
            observed_distribution=[float("nan"), 1.0],
        )


def test_multiple_source_candidates_without_modeled_output_effect_do_not_cause_anomaly():
    result = diagnose_execution(
        expected_distribution=EXPECTED,
        observed_distribution=ANOMALOUS,
        expected_circuit=[("x", 0), ("h", 0)],
        observed_circuit=[("h", 0), ("x", 0)],
    )

    assert result.category is DiagnosisCategory.INSUFFICIENT_EVIDENCE
    assert len(result.program_result.candidates) >= 2
    assert all(candidate.evidence for candidate in result.program_result.candidates)
    assert result.program_behavioral_candidates == []


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


def test_high_level_diagnosis_without_reference_keeps_historical_behavior():
    result = diagnose_circuit([("x", 0), ("measure", 0)], shots=64, seed=19)

    assert result.category is DiagnosisCategory.NO_ANOMALY
    assert result.anomaly_detected is False
    assert result.program_evidence_candidates == []


def test_matching_reference_circuit_is_normal():
    circuit = [("x", 0), ("measure", 0)]
    result = diagnose_circuit(
        circuit, reference_circuit=circuit, shots=64, seed=19,
    )

    assert result.category is DiagnosisCategory.NO_ANOMALY
    assert result.anomaly_detected is False
    assert result.program_evidence_candidates == []


def test_reference_circuit_localizes_submitted_program_mismatch():
    result = diagnose_circuit(
        [("h", 0), ("measure", 0)],
        reference_circuit=[("x", 0), ("measure", 0)],
        shots=256,
        seed=7,
    )

    assert result.category is DiagnosisCategory.PROGRAM_FAULT
    assert result.anomaly_detected is True
    assert result.program_evidence_candidates
    assert result.program_evidence_candidates[0].index == 0


def test_reference_equivalent_outputs_keep_localization_but_abstain_on_cause():
    result = diagnose_circuit(
        [("y", 0), ("measure", 0, [0])],
        reference_circuit=[("x", 0), ("measure", 0, [0])],
        shots=128,
        seed=11,
    )

    # The source mismatch is localizable, but these measured outputs agree,
    # so it is insufficient evidence for a behavioral fault classification.
    assert result.statistics_result.anomaly_detected is False
    assert result.category is DiagnosisCategory.INSUFFICIENT_EVIDENCE
    assert result.program_evidence_candidates


def test_execution_api_does_not_promote_syntactic_diff_to_behavioral_fault():
    result = diagnose_execution(
        expected_distribution={"1": 1.0},
        observed_distribution={"1": 1.0},
        expected_circuit=[("x", 0)],
        observed_circuit=[("y", 0)],
        shots=128,
    )

    assert result.anomaly_detected is False
    assert result.program_evidence_candidates
    assert result.category is DiagnosisCategory.INSUFFICIENT_EVIDENCE


def test_reference_measurement_classical_destinations_must_match():
    with pytest.raises(ValueError, match="compatible measurement schemas"):
        diagnose_circuit(
            [("x", 0), ("measure", 0, [1])],
            reference_circuit=[("x", 0), ("measure", 0, [0])],
            shots=32,
            seed=3,
        )


def test_reference_is_ideal_and_hardware_model_applies_to_submitted_circuit():
    hardware = HardwareModel(1)
    hardware.set_qubit_properties(0, readout_error=1.0)
    result = diagnose_circuit(
        [("x", 0), ("measure", 0)],
        reference_circuit=[("x", 0), ("measure", 0)],
        hardware_model=hardware,
        shots=64,
        seed=5,
    )

    assert result.statistics_result.total_variation_distance == 1.0
    assert result.category is DiagnosisCategory.HARDWARE_ANOMALY
    assert result.program_evidence_candidates == []
    assert result.hardware_evidence_candidates


def test_reference_comparison_preserves_ambiguous_mixed_evidence():
    hardware = HardwareModel(1)
    hardware.set_qubit_properties(0, readout_error=0.45)
    result = diagnose_circuit(
        [("h", 0), ("measure", 0)],
        reference_circuit=[("x", 0), ("measure", 0)],
        hardware_model=hardware,
        shots=256,
        seed=23,
    )

    assert result.category is DiagnosisCategory.AMBIGUOUS
    assert result.program_evidence_candidates
    assert result.hardware_evidence_candidates


def test_diagnosis_excludes_reported_hardware_candidates_when_age_policy_fails():
    hardware = HardwareModel(
        1,
        calibration_timestamp="2000-01-01T00:00:00Z",
        calibration_timestamp_provenance="reported",
    )
    hardware.set_qubit_properties(0, available=False, status="offline")
    result = diagnose_execution(
        EXPECTED,
        ANOMALOUS,
        expected_circuit=[("x", 0)],
        observed_circuit=[("x", 0)],
        hardware_model=hardware,
        max_calibration_age_seconds=60,
    )

    assert result.calibration_freshness["status"] == "STALE"
    assert result.hardware_result.candidates  # candidate remains visible for review
    assert result.hardware_evidence_candidates == []
    assert any("freshness is not established" in item for item in result.missing_evidence)


def test_diagnosis_does_not_apply_generic_freshness_gate_to_synthetic_configuration():
    hardware = HardwareModel(1, calibration_timestamp="synthetic-config")
    hardware.set_qubit_properties(0, available=False, status="offline")
    result = diagnose_execution(
        EXPECTED,
        ANOMALOUS,
        expected_circuit=[("x", 0)],
        observed_circuit=[("x", 0)],
        hardware_model=hardware,
        max_calibration_age_seconds=1,
    )

    assert result.calibration_freshness["status"] == "SYNTHETIC"
    assert result.hardware_evidence_candidates


def test_diagnosis_unknown_provenance_does_not_support_hardware_attribution():
    hardware = HardwareModel(
        1, calibration_timestamp_provenance="unknown",
    )
    hardware.set_qubit_properties(0, available=False, status="offline")
    result = diagnose_execution(
        EXPECTED,
        ANOMALOUS,
        expected_circuit=[("x", 0)],
        observed_circuit=[("x", 0)],
        hardware_model=hardware,
    )

    assert result.calibration_freshness["status"] == "UNKNOWN_PROVENANCE"
    assert result.hardware_evidence_candidates == []
    assert result.hardware_result.candidates
