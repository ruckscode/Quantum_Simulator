"""End-to-end pipeline architecture checks with injected, deterministic labels.

Classifier labels are selected by scenario solely to exercise routing. These
tests measure neither classifier performance nor ML predictive accuracy.
"""
import numpy as np
import pytest

import quantum_simulator.diagnosis_engine as diagnosis_engine
from quantum_simulator.diagnosis_engine import DiagnosisCategory
from quantum_simulator.hardware_model import HardwareModel
from quantum_simulator.ml_diagnosis import (
    CLASSES, FEATURE_NAMES, ClassifierPrediction, diagnose_with_ml,
)
from quantum_simulator.quantum_simulator import QuantumSimulator

SHOTS = 4096
SEED = 831


class ScenarioClassifier:
    feature_names = FEATURE_NAMES

    def __init__(self, label):
        self.label = label

    def predict(self, features):
        assert tuple(features) == FEATURE_NAMES
        return ClassifierPrediction(self.label, {
            name: (0.98 if name == self.label else 0.01) for name in CLASSES
        })


def _hardware_cases():
    readout = HardwareModel(1, backend_name="e2e_readout")
    readout.set_qubit_properties(0, readout_error=1.0)
    yield ("readout_error", [("x", 0), ("measure", 0)], 1, readout,
           {"component_type": "qubit", "component": 0})

    gate = HardwareModel(1, backend_name="e2e_gate")
    gate.add_gate_calibration("x", qubits=(0,), error=1.0, duration=0.0)
    yield ("gate_error", [("x", 0), ("measure", 0)], 1, gate,
           {"component_type": "gate", "component": (0,)})

    coherence = HardwareModel(1, backend_name="e2e_t1_t2")
    coherence.add_gate_calibration("x", qubits=(0,), error=0.0, duration=1.0)
    coherence.set_qubit_properties(0, t1=0.0001, t2=0.0001)
    yield ("t1_t2_degradation", [("x", 0), ("measure", 0)], 1, coherence,
           {"component_type": "qubit", "component": 0})

    combined = HardwareModel(1, backend_name="e2e_combined")
    combined.set_qubit_properties(0, readout_error=0.35, t1=0.0001, t2=0.0001)
    combined.add_gate_calibration("x", qubits=(0,), error=0.4, duration=1.0)
    yield ("combined_degradation", [("x", 0), ("measure", 0)], 1, combined,
           {"component_type": "qubit", "component": 0})


def _execute(reference, actual, n_qubits, hardware):
    ideal = QuantumSimulator(n_qubits).run_shots(reference, shots=SHOTS, seed=SEED)
    observed = QuantumSimulator(n_qubits).run_shots(
        actual, shots=SHOTS, seed=SEED, hardware_model=hardware,
    )
    keys = set(ideal) | set(observed)
    expected = {key: ideal.get(key, 0) / SHOTS for key in keys}
    observed_dist = {key: observed.get(key, 0) / SHOTS for key in keys}
    _, _, _, trace = QuantumSimulator(n_qubits).run_circuit_traced(
        reference, actual, hardware_model=hardware,
    )
    return expected, observed_dist, trace


def _diagnose(reference, actual, n_qubits, hardware, label, monkeypatch):
    calls = {"program": 0, "hardware": 0}
    original_program = diagnosis_engine.localize_program_fault
    original_hardware = diagnosis_engine.localize_hardware_fault

    def program_spy(*args, **kwargs):
        calls["program"] += 1
        return original_program(*args, **kwargs)

    def hardware_spy(*args, **kwargs):
        calls["hardware"] += 1
        return original_hardware(*args, **kwargs)

    monkeypatch.setattr(diagnosis_engine, "localize_program_fault", program_spy)
    monkeypatch.setattr(diagnosis_engine, "localize_hardware_fault", hardware_spy)
    expected, observed, trace = _execute(reference, actual, n_qubits, hardware)
    result = diagnose_with_ml(
        expected, observed, expected_circuit=reference, observed_circuit=actual,
        hardware_model=hardware, shots=SHOTS, precomputed_trace=trace,
        classifier=ScenarioClassifier(label),
    )
    return result, calls


def _program_cases():
    healthy = lambda n: HardwareModel(n, backend_name="e2e_healthy")
    return [
        ("gate_replacement", [("x", 0), ("measure", 0)], [("h", 0), ("measure", 0)], 1, "h", (0,)),
        ("gate_deletion", [("x", 0), ("measure", 0)], [("measure", 0)], 1, "x", (0,)),
        ("gate_insertion", [("measure", 0)], [("x", 0), ("measure", 0)], 1, "x", (0,)),
        ("parameter_modification", [("rx", 0, 0.0), ("measure", 0)], [("rx", 0, float(np.pi)), ("measure", 0)], 1, "rx", (0,)),
        ("two_qubit_gate_fault", [("x", 0), ("x", 1), ("cx", 0, 1), ("measure", 1)],
         [("x", 0), ("x", 1), ("cz", 0, 1), ("measure", 1)], 2, "cz", (0, 1)),
    ]


@pytest.mark.parametrize("name,reference,actual,n_qubits,operation,qubits", _program_cases(), ids=lambda x: x if isinstance(x, str) else None)
def test_end_to_end_program_fault_localizes_injected_operation(name, reference, actual, n_qubits, operation, qubits, monkeypatch):
    result, calls = _diagnose(reference, actual, n_qubits, HardwareModel(n_qubits), "PROGRAM_FAULT", monkeypatch)
    candidates = result.localization_result.program_evidence_candidates
    assert calls["program"] >= 1
    assert any(item.operation == operation and tuple(item.qubits) == qubits for item in candidates), name
    assert result.final_diagnosis is DiagnosisCategory.PROGRAM_FAULT
    assert tuple(result.ml_features) == FEATURE_NAMES
    assert not hasattr(result, "feature_importance")


@pytest.mark.parametrize("name,circuit,n_qubits,hardware,truth", list(_hardware_cases()), ids=lambda x: x if isinstance(x, str) else None)
def test_end_to_end_hardware_fault_localizes_affected_component(name, circuit, n_qubits, hardware, truth, monkeypatch):
    result, calls = _diagnose(circuit, circuit, n_qubits, hardware, "HARDWARE_ANOMALY", monkeypatch)
    candidates = result.localization_result.hardware_evidence_candidates
    assert calls["hardware"] >= 1
    assert any(item.component_type == truth["component_type"] and item.component == truth["component"] for item in candidates), name
    assert result.final_diagnosis is DiagnosisCategory.HARDWARE_ANOMALY
    assert result.localization_result.program_evidence_candidates == []
    if name == "readout_error":
        # Readout noise changes reported classical bits after state collapse;
        # it must not be fabricated as a pre-measurement state-trace deviation.
        assert result.localization_result.trace_deviation_candidates == []
    else:
        assert result.localization_result.trace_deviation_candidates


def test_end_to_end_normal_executes_without_positive_localization(monkeypatch):
    circuit = [("h", 0), ("measure", 0)]
    result, calls = _diagnose(circuit, circuit, 1, HardwareModel(1), "NORMAL", monkeypatch)
    assert result.predicted_class == "NORMAL"
    assert result.final_diagnosis is DiagnosisCategory.NO_ANOMALY
    assert result.localization_result is None
    assert calls == {"program": 0, "hardware": 0}


def test_hardware_trace_deviation_cannot_become_program_fault(monkeypatch):
    circuit = [("x", 0), ("measure", 0)]
    hardware = HardwareModel(1)
    hardware.add_gate_calibration("x", qubits=(0,), error=1.0, duration=0.0)
    result, _ = _diagnose(circuit, circuit, 1, hardware, "HARDWARE_ANOMALY", monkeypatch)
    assert result.localization_result.trace_deviation_candidates
    assert result.localization_result.program_evidence_candidates == []
    assert result.final_diagnosis is DiagnosisCategory.HARDWARE_ANOMALY


def test_ml_classification_and_localization_are_distinct_layers(monkeypatch):
    reference = [("x", 0), ("measure", 0)]
    actual = [("h", 0), ("measure", 0)]
    result, _ = _diagnose(reference, actual, 1, HardwareModel(1), "PROGRAM_FAULT", monkeypatch)
    assert result.predicted_class == "PROGRAM_FAULT"
    assert result.ml_features.keys() == set(FEATURE_NAMES)
    assert result.localization_result.program_evidence_candidates
    assert all(hasattr(candidate, "evidence") for candidate in result.localization_result.program_evidence_candidates)
