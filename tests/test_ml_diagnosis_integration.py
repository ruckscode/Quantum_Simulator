from quantum_simulator.diagnosis_engine import DiagnosisCategory
import quantum_simulator.diagnosis_engine as diagnosis_engine
from quantum_simulator.hardware_model import HardwareModel
import pytest
from quantum_simulator.ml_diagnosis import (
    CLASSES, FEATURE_NAMES, ClassifierPrediction, diagnose_with_ml,
)


class MockClassifier:
    feature_names = FEATURE_NAMES

    def __init__(self, label, confidence=0.95):
        self.label, self.confidence = label, confidence

    def predict(self, features):
        assert tuple(features) == FEATURE_NAMES
        remainder = (1.0 - self.confidence) / (len(CLASSES) - 1)
        return ClassifierPrediction(self.label, {
            label: self.confidence if label == self.label else remainder for label in CLASSES
        })


def test_mock_normal_preserves_genuinely_normal_execution():
    result = diagnose_with_ml(
        {"0": 1.0, "1": 0.0}, {"0": 1.0, "1": 0.0},
        expected_circuit=[("x", 0)], observed_circuit=[("x", 0)],
        classifier=MockClassifier("NORMAL"),
    )
    assert result.predicted_class == "NORMAL"
    assert result.final_diagnosis is DiagnosisCategory.NO_ANOMALY
    assert result.localization_result is None


def test_mock_normal_does_not_bypass_unsupported_operation_uncertainty():
    circuit = [("unsupported_gate", 0)]
    result = diagnose_with_ml(
        {"0": 1.0, "1": 0.0}, {"0": 1.0, "1": 0.0},
        expected_circuit=circuit, observed_circuit=circuit,
        classifier=MockClassifier("NORMAL"),
    )

    assert result.final_diagnosis is DiagnosisCategory.INSUFFICIENT_EVIDENCE
    assert result.localization_result is not None
    assert any("unsupported" in item.lower() for item in result.localization_result.missing_evidence)


def _spy_localizers(monkeypatch):
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
    return calls


def test_normal_direct_circuit_mismatch_runs_full_diagnosis(monkeypatch):
    calls = _spy_localizers(monkeypatch)
    result = diagnose_with_ml(
        {"0": 1.0, "1": 0.0}, {"0": 1.0, "1": 0.0},
        expected_circuit=[("x", 0)], observed_circuit=[("h", 0)],
        hardware_model=HardwareModel(1), classifier=MockClassifier("NORMAL"),
    )
    assert calls["program"] == 1
    assert calls["hardware"] == 1
    assert result.localization_result is not None
    assert result.localization_result.program_evidence_candidates


def test_normal_relevant_degraded_calibration_runs_full_diagnosis(monkeypatch):
    calls = _spy_localizers(monkeypatch)
    hardware = HardwareModel(1)
    hardware.set_qubit_properties(0, readout_error=0.45)
    circuit = [("measure",)]
    result = diagnose_with_ml(
        {"0": 1.0, "1": 0.0}, {"0": 1.0, "1": 0.0},
        expected_circuit=circuit, observed_circuit=circuit,
        hardware_model=hardware, classifier=MockClassifier("NORMAL"),
    )
    assert calls["program"] == 1
    assert calls["hardware"] == 1
    assert result.localization_result is not None
    assert result.localization_result.hardware_evidence_candidates


def test_normal_measurement_only_trace_difference_does_not_run_localizers(monkeypatch):
    calls = _spy_localizers(monkeypatch)
    circuit = [("h", 0), ("measure", 0)]
    result = diagnose_with_ml(
        {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5},
        expected_circuit=circuit, observed_circuit=circuit,
        hardware_model=HardwareModel(1),
        precomputed_trace=[{
            "step": 1, "operation": "measure", "expected": 0,
            "actual": 1, "difference": 1.0,
        }],
        classifier=MockClassifier("NORMAL"),
    )
    assert result.final_diagnosis is DiagnosisCategory.NO_ANOMALY
    assert result.localization_result is None
    assert calls == {"program": 0, "hardware": 0}


def test_mock_normal_cannot_override_cancellation_fault_evidence():
    # Keep this regression fixture self-contained: pytest's importlib mode
    # intentionally does not guarantee that sibling test modules are importable.
    CIRCUIT = [("measure",)]
    PROGRAM_FAULT_CIRCUIT = [("x", 0), ("measure",)]
    IDEAL_REFERENCE = {"0": 1.0, "1": 0.0}
    from quantum_simulator.fault_localizer import _trace_deviation

    faulty_hardware = HardwareModel(1, backend_name="adapter_cancellation_regression")
    faulty_hardware.set_qubit_properties(0, readout_error=1.0)
    trace = _trace_deviation(CIRCUIT, PROGRAM_FAULT_CIRCUIT, n_qubits=1)
    result = diagnose_with_ml(
        IDEAL_REFERENCE, IDEAL_REFERENCE,
        expected_circuit=CIRCUIT, observed_circuit=PROGRAM_FAULT_CIRCUIT,
        hardware_model=faulty_hardware, precomputed_trace=trace,
        classifier=MockClassifier("NORMAL"),
    )

    assert result.predicted_class == "NORMAL"
    assert result.final_diagnosis is DiagnosisCategory.AMBIGUOUS
    assert result.localization_result is not None
    assert result.localization_result.anomaly_detected is False
    assert result.localization_result.program_evidence_candidates
    assert result.localization_result.hardware_evidence_candidates


def test_mock_program_fault_uses_existing_program_localizer():
    result = diagnose_with_ml(
        {"0": 1.0, "1": 0.0}, {"0": 0.0, "1": 1.0},
        expected_circuit=[("x", 0)], observed_circuit=[("h", 0)],
        hardware_model=HardwareModel(1), classifier=MockClassifier("PROGRAM_FAULT"),
    )
    assert result.final_diagnosis is DiagnosisCategory.PROGRAM_FAULT
    assert result.localization_result.program_result.candidates
    assert result.localization_result.program_evidence_candidates


def test_program_prediction_abstains_when_source_diff_has_no_observed_effect():
    result = diagnose_with_ml(
        {"1": 1.0}, {"1": 1.0},
        expected_circuit=[("x", 0)], observed_circuit=[("y", 0)],
        classifier=MockClassifier("PROGRAM_FAULT"),
    )

    assert result.localization_result is not None
    assert result.localization_result.program_evidence_candidates
    assert result.localization_result.anomaly_detected is False
    assert result.final_diagnosis is DiagnosisCategory.INSUFFICIENT_EVIDENCE
    assert "abstaining on program-fault attribution" in result.summary


def test_program_prediction_cannot_explain_unrelated_observation_anomaly():
    result = diagnose_with_ml(
        {"0": 0.0, "1": 1.0}, {"0": 1.0, "1": 0.0},
        expected_circuit=[("x", 0), ("measure", 0)],
        observed_circuit=[("y", 0), ("measure", 0)],
        classifier=MockClassifier("PROGRAM_FAULT"),
    )

    assert result.localization_result is not None
    assert result.localization_result.anomaly_detected is True
    assert result.localization_result.program_evidence_candidates
    assert result.localization_result.program_behavioral_candidates == []
    assert result.final_diagnosis is DiagnosisCategory.INSUFFICIENT_EVIDENCE
    assert "does not link it to the measured outputs" in result.summary


def test_mock_hardware_anomaly_uses_existing_hardware_localizer():
    hardware = HardwareModel(1)
    hardware.set_qubit_properties(0, readout_error=0.45)
    circuit = [("measure",)]
    result = diagnose_with_ml(
        {"0": 1.0, "1": 0.0}, {"0": 0.0, "1": 1.0},
        expected_circuit=circuit, observed_circuit=circuit, hardware_model=hardware,
        classifier=MockClassifier("HARDWARE_ANOMALY"),
    )
    assert result.final_diagnosis is DiagnosisCategory.HARDWARE_ANOMALY
    assert result.localization_result.hardware_evidence_candidates


def test_low_confidence_abstains_without_localization():
    result = diagnose_with_ml(
        {"0": 1.0, "1": 0.0}, {"0": 0.0, "1": 1.0},
        expected_circuit=[("x", 0)], observed_circuit=[("h", 0)],
        classifier=MockClassifier("PROGRAM_FAULT", confidence=0.40),
    )
    assert result.final_diagnosis is DiagnosisCategory.INSUFFICIENT_EVIDENCE
    assert result.localization_result is None


def test_trace_only_deviation_never_supports_ml_program_fault():
    circuit = [("h", 0)]
    trace = [{"step": 0, "operation": "h", "expected": ("h", 0),
              "actual": ("h", 0), "difference": 0.8}]
    result = diagnose_with_ml(
        {"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5},
        expected_circuit=circuit, observed_circuit=circuit,
        hardware_model=HardwareModel(1), precomputed_trace=trace,
        classifier=MockClassifier("PROGRAM_FAULT"),
    )
    assert result.final_diagnosis is not DiagnosisCategory.PROGRAM_FAULT
    assert result.localization_result.program_evidence_candidates == []


def test_inference_backend_is_explicitly_unavailable_without_adapter():
    from quantum_simulator.ml_diagnosis import ModelBackendUnavailable
    import pytest
    with pytest.raises(ModelBackendUnavailable, match="No trained observable-only"):
        diagnose_with_ml({"0": 1.0}, {"0": 1.0})


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_min_confidence_is_rejected_before_model_routing(invalid):
    with pytest.raises(ValueError, match="min_confidence must be between 0 and 1"):
        diagnose_with_ml(
            {"0": 1.0}, {"0": 1.0}, classifier=MockClassifier("NORMAL"),
            min_confidence=invalid,
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True, "0.9"])
def test_malformed_model_probability_values_fail_closed(invalid):
    class MalformedClassifier:
        feature_names = FEATURE_NAMES

        def predict(self, features):
            return ClassifierPrediction("NORMAL", {
                "NORMAL": invalid, "PROGRAM_FAULT": 0.0, "HARDWARE_ANOMALY": 0.0,
            })

    with pytest.raises(ValueError, match="probabilities must be finite values"):
        diagnose_with_ml({"0": 1.0}, {"0": 1.0}, classifier=MalformedClassifier())


def test_model_without_feature_schema_fails_with_contract_error():
    class MissingSchemaClassifier:
        def predict(self, features):
            return ClassifierPrediction("NORMAL", {"NORMAL": 1.0, "PROGRAM_FAULT": 0.0, "HARDWARE_ANOMALY": 0.0})

    with pytest.raises(ValueError, match="must declare the approved feature names"):
        diagnose_with_ml({"0": 1.0}, {"0": 1.0}, classifier=MissingSchemaClassifier())
