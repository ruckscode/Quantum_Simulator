import numpy as np

from quantum_simulator.fault_localizer import DiagnosticTrace, _infer_n_qubits, _normalize_operation, localize_program_fault


def test_normalize_single_qubit_operation_preserves_qubit_index():
    assert _normalize_operation(("h", 2)) == ("h", (2,), ())


def test_normalize_two_qubit_operation_preserves_both_qubit_indices():
    assert _normalize_operation(["cx", 0, 2]) == ("cx", (0, 2), ())


def test_normalize_parameterized_operations_preserves_qubits_and_parameters():
    assert _normalize_operation(("rx", 2, 0.5)) == ("rx", (2,), (0.5,))
    assert _normalize_operation(("crx", 0, 2, 0.75)) == ("crx", (0, 2), (0.75,))


def test_qubit_inference_does_not_count_angles_as_qubits():
    assert _infer_n_qubits([("cp", 0, 3, 19.6), ("rxx", 1, 2, 0.4)]) == 4


def test_normalize_existing_operation_formats():
    assert _normalize_operation({"name": "ry", "qubits": [1], "parameters": [0.25]}) == (
        "ry", (1,), (0.25,)
    )
    assert _normalize_operation({"op": "cx", "qubits": [0, 1], "parameters": []}) == (
        "cx", (0, 1), ()
    )
    assert _normalize_operation(("measure", [0, 1])) == ("measure", (0, 1), ())
    assert _normalize_operation(("barrier", 0, 1)) == ("barrier", (0, 1), ())


def test_identical_measurement_mappings_do_not_mismatch():
    circuit = [("measure", (0,), (1,))]
    assert localize_program_fault(circuit, circuit).candidates == []


def test_measurement_classical_destination_change_is_localized():
    result = localize_program_fault(
        [("measure", (0,), (0,))], [("measure", (0,), (1,))]
    )
    assert len(result.candidates) == 1
    assert result.candidates[0].operation == "measure"
    assert result.candidates[0].qubits == (0,)
    assert result.candidates[0].expected_signature != result.candidates[0].actual_signature


def test_no_program_fault_has_no_candidates():
    expected = [("h", 0), ("cnot", 0, 1)]
    actual = [("h", 0), ("cnot", 0, 1)]
    result = localize_program_fault(expected, actual, threshold=0.05)
    assert result.candidates == []


def test_wrong_single_qubit_gate_is_localized():
    expected = [("h", 0), ("x", 0)]
    actual = [("h", 0), ("z", 0)]
    result = localize_program_fault(expected, actual, threshold=0.05)
    assert len(result.candidates) >= 1
    assert result.candidates[0].index == 1
    assert result.candidates[0].operation in {"z", "x"}


def test_multiple_gate_circuit_localizes_relevant_step():
    expected = [("h", 0), ("cnot", 0, 1), ("measure",)]
    actual = [("h", 0), ("x", 1), ("cnot", 0, 1), ("measure",)]
    result = localize_program_fault(expected, actual, threshold=0.05)
    assert any(candidate.index == 1 for candidate in result.candidates)
    assert any("x" in candidate.operation for candidate in result.candidates)


def test_single_qubit_operation_localization_evidence():
    expected = [("h", 0), ("x", 0)]
    actual = [("h", 0), ("y", 0)]
    result = localize_program_fault(expected, actual, threshold=0.05)
    assert result.summary
    assert any("Step 1" in evidence for candidate in result.candidates for evidence in candidate.evidence)


def test_multi_qubit_operation_localization():
    expected = [("h", 0), ("cnot", 0, 1), ("measure",)]
    actual = [("h", 0), ("swap", 0, 1), ("measure",)]
    result = localize_program_fault(expected, actual, threshold=0.05)
    assert any(candidate.operation == "swap" for candidate in result.candidates)


def test_trace_information_is_used_for_evidence():
    expected = [("h", 0), ("cnot", 0, 1)]
    actual = [("h", 0), ("x", 0), ("cnot", 0, 1)]
    result = localize_program_fault(expected, actual, threshold=0.05)
    assert result.evidence
    assert any("Mismatch" in text or "Step" in text for text in result.evidence)


def test_trace_deviation_is_computed_once_and_reused(monkeypatch):
    import quantum_simulator.fault_localizer as fault_localizer

    calls = []

    def fake_trace_deviation(expected, actual, n_qubits):
        calls.append((expected, actual, n_qubits))
        return 0.2, [{
            "step": 0,
            "operation": "h",
            "expected": expected[0],
            "actual": actual[0],
            "difference": 0.0,
        }]

    monkeypatch.setattr(fault_localizer, "_trace_deviation", fake_trace_deviation)
    circuit = [("h", 0)]

    result = localize_program_fault(circuit, circuit)

    assert len(calls) == 1
    assert result.anomaly_metric == 0.2
    assert len(result.candidates) == 1
    assert result.candidates[0].index == 0


def test_precomputed_trace_preserves_metric_and_step_evidence_without_recomputation(monkeypatch):
    import quantum_simulator.fault_localizer as fault_localizer

    circuit = [("h", 0)]
    trace: DiagnosticTrace = (0.2, [{
        "step": 0,
        "operation": "h",
        "expected": circuit[0],
        "actual": circuit[0],
        "difference": 0.7,
    }])

    def unexpected_trace_call(*args, **kwargs):
        raise AssertionError("_trace_deviation must not run for a supplied trace")

    monkeypatch.setattr(fault_localizer, "_trace_deviation", unexpected_trace_call)
    result = localize_program_fault(circuit, circuit, precomputed_trace=trace)

    assert result.anomaly_metric == 0.2
    assert result.candidates[0].index == 0
    assert result.candidates[0].score == 0.7


def test_inserted_single_qubit_gate_is_localized_without_cascade():
    expected = [("h", 0), ("cnot", 0, 1), ("measure",)]
    actual = [("h", 0), ("x", 2), ("cnot", 0, 1), ("measure",)]
    result = localize_program_fault(expected, actual, threshold=0.05)
    assert len(result.candidates) == 1
    assert result.candidates[0].operation == "x"
    assert result.candidates[0].qubits == (2,)
    assert "Inserted" in result.candidates[0].reason


def test_removed_single_qubit_gate_uses_reference_operation_and_qubit():
    expected = [("h", 0), ("x", 2), ("cnot", 0, 1), ("measure",)]
    actual = [("h", 0), ("cnot", 0, 1), ("measure",)]
    result = localize_program_fault(expected, actual, threshold=0.05)
    assert len(result.candidates) == 1
    assert result.candidates[0].operation == "x"
    assert result.candidates[0].qubits == (2,)
    assert "missing" in result.candidates[0].reason.lower()


def test_single_replacement_remains_a_substitution_candidate():
    result = localize_program_fault([("h", 1)], [("x", 1)], threshold=0.05)
    assert len(result.candidates) == 1
    assert result.candidates[0].operation == "x"
    assert result.candidates[0].qubits == (1,)
    assert result.candidates[0].expected_signature[0] == "h"
    assert result.candidates[0].actual_signature[0] == "x"
