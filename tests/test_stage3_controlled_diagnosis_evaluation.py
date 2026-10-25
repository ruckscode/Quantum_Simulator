"""Controlled evaluation of existing trace-based diagnosis (no ML involved)."""

import numpy as np
import pytest

from quantum_simulator.diagnosis_engine import DiagnosisCategory, diagnose_execution
from quantum_simulator.hardware_model import HardwareModel
from quantum_simulator.quantum_simulator import QuantumSimulator


SHOTS = 2048
SEED = 7319


def _distribution(circuit, n_qubits, hardware_model=None):
    counts = QuantumSimulator(n_qubits).run_shots(
        circuit, shots=SHOTS, seed=SEED, hardware_model=hardware_model
    )
    return {key: value / SHOTS for key, value in counts.items()}


def _run_case(name, family, reference, actual, n_qubits, hardware_model, ground_truth):
    expected_distribution = _distribution(reference, n_qubits)
    observed_distribution = _distribution(actual, n_qubits, hardware_model)
    outcome_keys = set(expected_distribution) | set(observed_distribution)
    expected_distribution = {key: expected_distribution.get(key, 0.0) for key in outcome_keys}
    observed_distribution = {key: observed_distribution.get(key, 0.0) for key in outcome_keys}
    _, _, _, trace = QuantumSimulator(n_qubits).run_circuit_traced(
        reference, actual, hardware_model=hardware_model
    )
    diagnosis = diagnose_execution(
        expected_distribution=expected_distribution,
        observed_distribution=observed_distribution,
        expected_circuit=reference,
        observed_circuit=actual,
        hardware_model=hardware_model,
        shots=SHOTS,
        precomputed_trace=trace,
    )
    # Ground truth is retained in this evaluation record only. It is not passed
    # to either execution or diagnosis.
    return {
        "name": name,
        "family": family,
        "reference": reference,
        "actual": actual,
        "ground_truth": ground_truth,
        "predicted_anomaly": diagnosis.anomaly_detected,
        "predicted_category": diagnosis.category,
        "localized_operations": [
            {"operation": item.operation, "qubits": item.qubits, "index": item.index, "reason": item.reason}
            for item in (diagnosis.program_result.candidates if diagnosis.program_result else [])
        ],
        "localized_hardware": [
            {"component_type": item.component_type, "component": item.component, "score": item.score, "reason": item.reason}
            for item in diagnosis.hardware_evidence_candidates
        ],
        "diagnosis": diagnosis,
        "trace": trace,
        "expected_distribution": expected_distribution,
        "observed_distribution": observed_distribution,
    }


def _cases():
    healthy = HardwareModel(2, backend_name="stage3_healthy")
    yield (
        "normal_identical",
        "NORMAL",
        [("x", 0), ("measure", 0)],
        [("x", 0), ("measure", 0)],
        2,
        healthy,
        {"condition": "identical program and healthy hardware"},
    )

    program_cases = [
        ("program_gate_replacement", [("x", 0), ("measure", 0)], [("h", 0), ("measure", 0)], 1, "h", (0,)),
        ("program_gate_deletion", [("x", 0), ("measure", 0)], [("measure", 0)], 1, "x", (0,)),
        ("program_gate_insertion", [("measure", 0)], [("x", 0), ("measure", 0)], 1, "x", (0,)),
        ("program_parameter_change", [("rx", 0, 0.0), ("measure", 0)], [("rx", 0, float(np.pi)), ("measure", 0)], 1, "rx", (0,)),
        (
            "program_two_qubit_replacement",
            [("x", 0), ("x", 1), ("cx", 0, 1), ("measure", 1)],
            [("x", 0), ("x", 1), ("cz", 0, 1), ("measure", 1)],
            2,
            "cz",
            (0, 1),
        ),
    ]
    for name, reference, actual, n_qubits, operation, qubits in program_cases:
        yield (
            name,
            "PROGRAM_FAULT",
            reference,
            actual,
            n_qubits,
            HardwareModel(n_qubits, backend_name=f"healthy_{name}"),
            {"operation": operation, "qubits": qubits},
        )

    readout = HardwareModel(1, backend_name="stage3_readout")
    readout.set_qubit_properties(0, readout_error=1.0)
    yield (
        "hardware_readout_q0",
        "HARDWARE_ANOMALY",
        [("x", 0), ("measure", 0)],
        [("x", 0), ("measure", 0)],
        1,
        readout,
        {"component_type": "qubit", "component": 0, "readout_error": 1.0},
    )

    gate_error = HardwareModel(1, backend_name="stage3_gate_error")
    gate_error.add_gate_calibration("x", qubits=(0,), error=1.0, duration=0.0)
    yield (
        "hardware_gate_error_x_q0",
        "HARDWARE_ANOMALY",
        [("x", 0), ("measure", 0)],
        [("x", 0), ("measure", 0)],
        1,
        gate_error,
        {"component_type": "gate", "component": (0,), "gate": "x", "qubits": (0,)},
    )

    coherence = HardwareModel(1, backend_name="stage3_coherence")
    # A short calibrated duration still overwhelms these very short coherence
    # times, while avoiding a separately flagged long-duration gate condition.
    coherence.add_gate_calibration("x", qubits=(0,), error=0.0, duration=1.0)
    coherence.set_qubit_properties(0, t1=0.0001, t2=0.0001)
    yield (
        "hardware_t1_t2_q0",
        "HARDWARE_ANOMALY",
        [("x", 0), ("measure", 0)],
        [("x", 0), ("measure", 0)],
        1,
        coherence,
        {"component_type": "qubit", "component": 0, "t1": 0.0001, "t2": 0.0001},
    )

    two_qubit = HardwareModel(2, backend_name="stage3_two_qubit_gate")
    two_qubit.add_gate_calibration("cx", qubits=(0, 1), error=1.0, duration=0.0)
    yield (
        "hardware_multiqubit_gate_cx_0_1",
        "HARDWARE_ANOMALY",
        [("x", 0), ("x", 1), ("cx", 0, 1), ("measure", 1)],
        [("x", 0), ("x", 1), ("cx", 0, 1), ("measure", 1)],
        2,
        two_qubit,
        {"component_type": "gate", "component": (0, 1), "gate": "cx", "qubits": (0, 1)},
    )


CASES = list(_cases())


@pytest.fixture(scope="module")
def evaluation_records():
    return {
        case[0]: _run_case(*case)
        for case in CASES
    }


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_controlled_case_runs_and_records_all_four_evaluation_dimensions(case, evaluation_records):
    record = evaluation_records[case[0]]
    diagnosis = record["diagnosis"]
    assert diagnosis.statistics_result is not None
    assert record["predicted_anomaly"] is not None
    assert record["predicted_category"] in set(DiagnosisCategory)
    assert isinstance(record["localized_operations"], list)
    assert isinstance(record["localized_hardware"], list)
    assert record["trace"]


def test_anomaly_detection_evaluation(evaluation_records):
    expected = {"normal_identical": False}
    expected.update({name: True for name, family, *_ in CASES if family != "NORMAL"})
    observed = {name: record["predicted_anomaly"] for name, record in evaluation_records.items()}
    assert observed == expected


def test_program_vs_hardware_classification_evaluation(evaluation_records):
    expected = {
        name: DiagnosisCategory.NO_ANOMALY if family == "NORMAL"
        else DiagnosisCategory.PROGRAM_FAULT if family == "PROGRAM_FAULT"
        else DiagnosisCategory.HARDWARE_ANOMALY
        for name, family, *_ in CASES
    }
    observed = {name: record["predicted_category"] for name, record in evaluation_records.items()}
    assert observed == expected


@pytest.mark.parametrize("case", [case for case in CASES if case[1] == "PROGRAM_FAULT"], ids=[case[0] for case in CASES if case[1] == "PROGRAM_FAULT"])
def test_program_fault_localization_evaluation(case, evaluation_records):
    record = evaluation_records[case[0]]
    truth = record["ground_truth"]
    assert any(
        item["operation"] == truth["operation"] and tuple(item["qubits"]) == truth["qubits"]
        for item in record["localized_operations"]
    )


def test_program_localizer_false_positive_evaluation_on_hardware_cases(evaluation_records):
    hardware_records = [
        evaluation_records[name]
        for name, family, *_ in CASES
        if family == "HARDWARE_ANOMALY"
    ]
    fallback_candidates = [
        operation
        for record in hardware_records
        for operation in record["localized_operations"]
        if operation["reason"] == "Largest traced deviation without direct operation mismatch"
    ]
    assert len(hardware_records) == 4
    # State-trace localization is available for gate/coherence effects. A
    # readout-only flip occurs after collapse, so it has no pre-measurement
    # state deviation to localize.
    assert len(fallback_candidates) == 3
    assert all(not record["diagnosis"].program_evidence_candidates for record in hardware_records)
    readout_record = next(record for record in hardware_records if record["name"] == "hardware_readout_q0")
    assert readout_record["diagnosis"].trace_deviation_candidates == []
    assert all(record["diagnosis"].trace_deviation_candidates for record in hardware_records if record["name"] != "hardware_readout_q0")


def test_identical_circuit_hardware_deviation_is_hardware_not_program_fault(evaluation_records):
    record = evaluation_records["hardware_gate_error_x_q0"]
    diagnosis = record["diagnosis"]

    assert record["reference"] == record["actual"]
    assert record["ground_truth"]["gate"] == "x"
    assert record["ground_truth"]["qubits"] == (0,)
    assert diagnosis.anomaly_detected is True
    assert diagnosis.category is DiagnosisCategory.HARDWARE_ANOMALY
    assert diagnosis.program_result is not None
    assert diagnosis.program_result.candidates  # fallback retained for diagnostics
    assert diagnosis.trace_deviation_candidates
    assert diagnosis.program_evidence_candidates == []
    assert diagnosis.hardware_evidence_candidates
    assert max(step["difference"] for step in record["trace"]) > 0.0
    assert record["ground_truth"] is not None


@pytest.mark.parametrize("case", [case for case in CASES if case[1] == "HARDWARE_ANOMALY"], ids=[case[0] for case in CASES if case[1] == "HARDWARE_ANOMALY"])
def test_hardware_component_localization_evaluation(case, evaluation_records):
    record = evaluation_records[case[0]]
    truth = record["ground_truth"]
    assert record["localized_hardware"]
    assert record["localized_hardware"][0]["component_type"] == truth["component_type"]
    assert record["localized_hardware"][0]["component"] == truth["component"]
