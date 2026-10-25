import json

import hashlib

import pytest

from diagnosis_engine import DiagnosisCategory
from experiment_config import ExperimentConfig, load_experiment_config
import experiments.run_experiment as experiment_runner
from experiments.run_experiment import _json_value, run_config_batch, run_experiment
from quantum_simulator.diagnosis_engine import diagnose_execution
from quantum_simulator.hardware_model import HardwareModel


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


def test_ibm_runtime_is_an_explicit_optional_integration_extra():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10 remains a supported runtime.
        pytest.skip("stdlib TOML reader is available from Python 3.11")

    metadata = tomllib.loads((experiment_runner.PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]

    assert "qiskit-ibm-runtime" not in project["dependencies"]
    assert project["optional-dependencies"]["ibm"] == [
        "qiskit>=2.5,<3", "qiskit-ibm-runtime>=0.50,<0.51"
    ]


def test_qiskit_oracle_is_a_separate_optional_dependency():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10 remains a supported runtime.
        pytest.skip("stdlib TOML reader is available from Python 3.11")

    metadata = tomllib.loads((experiment_runner.PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]

    assert "qiskit" not in project["dependencies"]
    assert project["optional-dependencies"]["oracle"] == ["qiskit>=2.5,<3"]


def test_atomic_artifact_writer_replaces_complete_json(tmp_path):
    destination = tmp_path / "nested" / "result.json"
    experiment_runner._atomic_write_text(destination, '{"version":1}\n')
    experiment_runner._atomic_write_text(destination, '{"version":2}\n')

    assert json.loads(destination.read_text(encoding="utf-8")) == {"version": 2}
    assert list(destination.parent.iterdir()) == [destination]


def test_atomic_artifact_writer_preserves_old_file_if_replace_fails(tmp_path, monkeypatch):
    destination = tmp_path / "result.json"
    destination.write_text('{"preserved":true}\n', encoding="utf-8")

    def fail_replace(_source, _destination):
        raise PermissionError("simulated Windows file lock")

    monkeypatch.setattr(experiment_runner.os, "replace", fail_replace)
    with pytest.raises(PermissionError, match="file lock"):
        experiment_runner._atomic_write_text(destination, '{"new":true}\n')

    assert json.loads(destination.read_text(encoding="utf-8")) == {"preserved": True}
    assert list(tmp_path.iterdir()) == [destination]


def test_batch_reproducibility_report_records_source_hashes(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config().to_dict()), encoding="utf-8")
    report = run_config_batch(
        [config_path],
        result_dir=tmp_path / "results",
        reproducibility_report_path=tmp_path / "reproducibility.json",
    )

    relative = "src/quantum_simulator/quantum_simulator.py"
    expected = hashlib.sha256((experiment_runner.PROJECT_ROOT / relative).read_bytes()).hexdigest()
    assert report["repeated_runs_matched"] is True
    assert report["environment"]["source_sha256"][relative] == expected
    saved = json.loads((tmp_path / "reproducibility.json").read_text(encoding="utf-8"))
    assert saved["environment"]["source_sha256"] == report["environment"]["source_sha256"]


def test_valid_configuration_loads_and_runs_local_simulator(tmp_path):
    config_file = tmp_path / "valid.json"
    config_file.write_text(json.dumps(_config().to_dict()), encoding="utf-8")

    loaded = load_experiment_config(config_file)
    result = run_experiment(loaded, output_path=tmp_path / "result.json")

    assert result.status == "COMPLETED"
    assert result.execution_metadata["execution_target"] == "LOCAL_SIMULATOR"
    assert result.execution_metadata["random_seed"] == 41
    assert result.execution_metadata["backend_api_used"] is False
    assert result.reproducibility["source_sha256"] == experiment_runner._source_fingerprints()
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


def test_serialization_marks_skipped_trace_metrics_as_unavailable():
    circuit = [("unsupported_gate", 0)]
    result = diagnose_execution(
        {"0": 1.0, "1": 0.0}, {"0": 0.0, "1": 1.0},
        expected_circuit=circuit, observed_circuit=circuit,
        hardware_model=HardwareModel(1),
    )

    saved = _json_value(result)

    assert saved["category"] == DiagnosisCategory.INSUFFICIENT_EVIDENCE.value
    assert saved["program_result"]["anomaly_metric"] == 0.0
    assert "partial" in saved["program_result"]["summary"].lower()
    assert any("not a measured deviation" in item.lower() for item in saved["program_result"]["evidence"])
    assert "not measured" in saved["hardware_result"]["summary"].lower()
    assert any("not measured" in item.lower() for item in saved["hardware_result"]["evidence"])
    assert any("unsupported" in item.lower() for item in saved["missing_evidence"])


def test_ground_truth_metadata_and_hardware_condition_reach_shot_execution(tmp_path):
    common = {
        "circuit": [["x", 0], ["measure", 0]],
        "num_qubits": 1,
        "shots": 2048,
        "random_seed": 8201,
        "circuit_family_id": "hardware_effect_test",
        "circuit_lineage_id": "hardware_effect_recipe",
    }
    normal = _config(
        **common,
        experiment_name="hardware_effect_normal",
        ground_truth_label="NORMAL",
        hardware_parameters=None,
    )
    anomaly = _config(
        **common,
        experiment_name="hardware_effect_anomaly",
        ground_truth_label="HARDWARE_ANOMALY",
        hardware_condition_id="readout_flip_0p25_v1",
        hardware_affected_component="qubit 0 readout",
        hardware_parameters={"coupling_map": [], "qubits": {"0": {"readout_error": 0.25}}},
    )

    normal_result = run_experiment(normal, output_path=tmp_path / "normal.json")
    anomaly_result = run_experiment(anomaly, output_path=tmp_path / "anomaly.json")

    assert normal_result.status == anomaly_result.status == "COMPLETED"
    assert normal_result.to_dict()["experiment_metadata"]["ground_truth_label"] == "NORMAL"
    assert anomaly_result.to_dict()["experiment_metadata"]["ground_truth_label"] == "HARDWARE_ANOMALY"
    assert normal_result.to_dict()["experiment_metadata"]["circuit_lineage_id"] == "hardware_effect_recipe"
    assert normal_result.execution_metadata["hardware_model_applied"] is False
    assert anomaly_result.execution_metadata["hardware_model_applied"] is True
    assert normal_result.observed_distribution != anomaly_result.observed_distribution
    assert anomaly_result.execution_metadata["observation_source"] == "SEEDED_LOCAL_SIMULATOR_SHOTS_WITH_HARDWARE_MODEL"


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

