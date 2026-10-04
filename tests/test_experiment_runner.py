import json

import pytest

from diagnosis_engine import DiagnosisCategory
from experiment_config import ExperimentConfig, load_experiment_config
from experiments.run_experiment import run_experiment


def _config(**overrides):
    values = {
        "experiment_name": "test_local_experiment",
        "validation_label": "SYNTHETIC_VALIDATION",
        "num_qubits": 1,
        "shots": 64,
        "random_seed": 41,
        "anomaly_threshold": 0.05,
        "circuit": [["h", 0], ["measure", 0]],
        "hardware_parameters": {"coupling_map": [], "qubits": {"0": {"readout_error": 0.0}}},
        "noise_parameters": None,
    }
    values.update(overrides)
    return ExperimentConfig.from_mapping(values)


def test_valid_configuration_loads_and_runs_local_simulator(tmp_path):
    config_file = tmp_path / "valid.json"
    config_file.write_text(json.dumps(_config().to_dict()), encoding="utf-8")

    loaded = load_experiment_config(config_file)
    result = run_experiment(loaded, output_path=tmp_path / "result.json")

    assert result.status == "COMPLETED"
    assert result.execution_metadata["execution_target"] == "LOCAL_SIMULATOR"
    assert result.execution_metadata["random_seed"] == 41
    assert result.execution_metadata["backend_api_used"] is False
    assert (tmp_path / "result.json").is_file()


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="shots"):
        _config(shots=0)
    with pytest.raises(ValueError, match="SYNTHETIC_VALIDATION"):
        _config(validation_label="IBM_RESULT")


def test_missing_required_configuration_fields_are_reported():
    with pytest.raises(ValueError, match="missing required"):
        ExperimentConfig.from_mapping({"experiment_name": "incomplete"})


def test_same_configuration_and_seed_produce_same_fingerprint_and_counts(tmp_path):
    config = _config(observed_counts={"0": 32, "1": 32})

    first = run_experiment(config, output_path=tmp_path / "first.json")
    second = run_experiment(config, output_path=tmp_path / "second.json")

    assert first.reproducibility["fingerprint"] == second.reproducibility["fingerprint"]
    assert first.simulator_observed_counts == second.simulator_observed_counts
    assert first.observed_distribution == second.observed_distribution
    assert first.diagnosis.category is DiagnosisCategory.NO_ANOMALY


def test_serialization_preserves_complete_diagnosis_and_evidence(tmp_path):
    config = _config(
        experiment_name="readout_anomaly_test",
        circuit=[["x", 0], ["measure", 0]],
        observed_counts={"0": 25, "1": 39},
        hardware_parameters={"coupling_map": [], "qubits": {"0": {"readout_error": 0.4}}},
    )
    result = run_experiment(config, output_path=tmp_path / "diagnosis.json")
    saved = json.loads((tmp_path / "diagnosis.json").read_text(encoding="utf-8"))

    assert result.status == "COMPLETED"
    assert result.diagnosis.category is DiagnosisCategory.HARDWARE_ANOMALY
    assert saved["diagnosis"]["statistics_result"]["total_variation_distance"] > 0.3
    assert saved["diagnosis"]["hardware_evidence_candidates"]
    assert saved["hardware_evidence"]["evidence"]
    assert saved["execution_metadata"]["observation_source"] == "CONTROLLED_OBSERVATION_FIXTURE"
    assert saved["execution_metadata"]["noise_parameters_applied"] is False


def test_malformed_circuit_returns_saved_failure_result(tmp_path):
    config = _config(circuit=[["h"]])

    result = run_experiment(config, output_path=tmp_path / "malformed.json")

    assert result.status == "FAILED"
    assert result.errors
    assert "h" in result.errors[0].lower()
    assert json.loads((tmp_path / "malformed.json").read_text(encoding="utf-8"))["status"] == "FAILED"


def test_unsupported_operation_returns_structured_failure(tmp_path):
    config = _config(circuit=[["not_a_supported_gate", 0]])

    result = run_experiment(config, output_path=tmp_path / "unsupported.json")

    assert result.status == "FAILED"
    assert any("Unsupported operation" in error for error in result.errors)

