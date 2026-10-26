from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from experiments.fault_isolation_phase1 import (
    HARDWARE_CLASS,
    NORMAL_CLASS,
    PROGRAM_CLASS,
    UNCERTAIN_CLASS,
    build_scenarios,
    case_record,
    finalize_existing_experiment,
    render_report,
    summarize,
)
import experiments.fault_isolation_phase1 as phase1
from experiments.run_experiment import run_experiment
from quantum_simulator.diagnosis_engine import DiagnosisCategory
from quantum_simulator.hardware_fault_localizer import HardwareCandidate


@pytest.fixture(scope="module")
def executed(tmp_path_factory):
    directory = tmp_path_factory.mktemp("fault-isolation-cases")
    scenarios = build_scenarios()
    results = {}
    for scenario in scenarios:
        results[scenario.scenario_id] = run_experiment(
            scenario.config,
            output_path=directory / f"{scenario.scenario_id}.json",
        )
    return {scenario.scenario_id: (scenario, results[scenario.scenario_id]) for scenario in scenarios}


def test_scenario_ground_truth_and_matched_pair_controls_are_independent():
    scenarios = build_scenarios()
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    pairs = {}
    for scenario in scenarios:
        for pair_id, factor in scenario.pair_memberships:
            pairs.setdefault(pair_id, []).append((scenario, factor))

    assert len(scenarios) == 9
    assert set(pairs) == {
        "healthy_vs_readout_q0",
        "readout_location_q0_vs_q1",
        "healthy_vs_gate_noise",
        "healthy_vs_coherence_degradation",
        "healthy_vs_program_mutation",
        "program_fault_ideal_vs_plus_hardware",
    }
    assert all(len(members) == 2 and len({factor for _, factor in members}) == 1 for members in pairs.values())

    normal, readout = by_id["healthy_ideal"].config, by_id["readout_noise_q0"].config
    assert normal.circuit == readout.circuit and normal.observed_circuit == readout.observed_circuit
    assert normal.random_seed == readout.random_seed and normal.shots == readout.shots
    assert normal.hardware_parameters["qubits"]["0"]["readout_error"] == 0.0
    assert readout.hardware_parameters["qubits"]["0"]["readout_error"] == 0.4

    normal_gate, noisy_gate = by_id["healthy_ideal"].config, by_id["gate_noise_x_q0"].config
    assert normal_gate.circuit == noisy_gate.circuit and normal_gate.observed_circuit == noisy_gate.observed_circuit
    assert normal_gate.hardware_parameters["qubits"] == noisy_gate.hardware_parameters["qubits"]
    assert normal_gate.hardware_parameters["gates"] == []
    assert noisy_gate.hardware_parameters["gates"] == [{"gate": "x", "qubits": [0], "error": 0.4, "duration": 0.0}]

    q0, q1 = by_id["readout_noise_q0"].config, by_id["readout_noise_q1"].config
    assert q0.circuit == q1.circuit and q0.observed_circuit == q1.observed_circuit
    assert q0.random_seed == q1.random_seed and q0.shots == q1.shots
    assert q0.hardware_parameters["qubits"]["0"]["readout_error"] == 0.4
    assert q0.hardware_parameters["qubits"]["1"]["readout_error"] == 0.0
    assert q1.hardware_parameters["qubits"]["0"]["readout_error"] == 0.0
    assert q1.hardware_parameters["qubits"]["1"]["readout_error"] == 0.4

    coherence_ok = by_id["coherence_healthy_control"].config
    coherence_bad = by_id["coherence_degradation_q0"].config
    assert coherence_ok.circuit == coherence_bad.circuit
    assert coherence_ok.shots == coherence_bad.shots and coherence_ok.random_seed == coherence_bad.random_seed
    assert coherence_ok.hardware_parameters["gates"] == coherence_bad.hardware_parameters["gates"]
    assert coherence_ok.hardware_parameters["qubits"]["1"] == coherence_bad.hardware_parameters["qubits"]["1"]
    assert coherence_ok.hardware_parameters["qubits"]["0"]["readout_error"] == coherence_bad.hardware_parameters["qubits"]["0"]["readout_error"]
    assert (coherence_ok.hardware_parameters["qubits"]["0"]["t1"], coherence_ok.hardware_parameters["qubits"]["0"]["t2"]) == (1e6, 1e6)
    assert (coherence_bad.hardware_parameters["qubits"]["0"]["t1"], coherence_bad.hardware_parameters["qubits"]["0"]["t2"]) == (2.0, 1.0)

    healthy, faulty = by_id["healthy_ideal"].config, by_id["program_fault_ideal"].config
    assert healthy.circuit == faulty.circuit
    assert healthy.observed_circuit is None or healthy.observed_circuit == healthy.circuit
    assert faulty.observed_circuit != faulty.circuit
    assert healthy.hardware_parameters == faulty.hardware_parameters
    assert healthy.shots == faulty.shots and healthy.random_seed == faulty.random_seed
    faulty_ideal = by_id["program_fault_ideal"].config
    faulty_noisy = by_id["program_fault_plus_readout_noise"].config
    assert faulty_ideal.circuit == faulty_noisy.circuit
    assert faulty_ideal.observed_circuit == faulty_noisy.observed_circuit
    assert faulty_ideal.shots == faulty_noisy.shots and faulty_ideal.random_seed == faulty_noisy.random_seed
    assert {q: v for q, v in faulty_ideal.hardware_parameters["qubits"].items()} != {
        q: v for q, v in faulty_noisy.hardware_parameters["qubits"].items()
    }
    assert faulty_ideal.hardware_parameters["gates"] == faulty_noisy.hardware_parameters["gates"]
    assert faulty_ideal.hardware_parameters["qubits"]["0"]["readout_error"] == 0.0
    assert faulty_noisy.hardware_parameters["qubits"]["0"]["readout_error"] == 0.4

    # The record builder uses the predeclared Scenario truth, never the output category.
    scenario = by_id["healthy_ideal"]
    fake_result = SimpleNamespace(
        status="COMPLETED",
        errors=[],
        execution_metadata={},
        diagnosis=SimpleNamespace(
            category=DiagnosisCategory.PROGRAM_FAULT,
            anomaly_detected=True,
            statistics_result=None,
            program_evidence_candidates=[],
            hardware_evidence_candidates=[],
            evidence=[],
            missing_evidence=[],
            summary="injected test prediction",
        ),
    )
    record = case_record(scenario, fake_result)
    assert record["ground_truth_class"] == NORMAL_CLASS
    assert record["predicted_class"] == PROGRAM_CLASS


def test_injections_change_execution_and_localization_tracks_targets(executed):
    records = {name: case_record(scenario, result) for name, (scenario, result) in executed.items()}
    baseline = executed["healthy_ideal"][1]
    assert baseline.status == "COMPLETED"
    assert executed["readout_noise_q0"][1].observed_counts != baseline.observed_counts
    assert executed["readout_noise_q1"][1].observed_counts != baseline.observed_counts
    assert executed["gate_noise_x_q0"][1].observed_counts != baseline.observed_counts
    assert executed["coherence_degradation_q0"][1].observed_counts != executed["coherence_healthy_control"][1].observed_counts
    assert executed["program_fault_ideal"][1].observed_counts != baseline.observed_counts

    assert records["program_fault_ideal"]["ground_truth_class"] == PROGRAM_CLASS
    assert records["program_fault_ideal"]["localization_correct"] is True
    for name in ("readout_noise_q0", "readout_noise_q1", "gate_noise_x_q0", "coherence_degradation_q0"):
        assert records[name]["localization_correct"] is True, (name, records[name]["hardware_candidates"])


def test_overlap_and_missing_causal_evidence_remain_uncertain(executed):
    combined_scenario, combined_result = executed["program_fault_plus_readout_noise"]
    combined = case_record(combined_scenario, combined_result)
    assert combined["predicted_class"] == UNCERTAIN_CLASS
    assert combined["localization_correct"] is True

    unknown_scenario, unknown_result = executed["deviation_without_cause_evidence"]
    unknown = case_record(unknown_scenario, unknown_result)
    assert unknown["predicted_class"] == UNCERTAIN_CLASS
    assert unknown["abstained"] is True
    assert unknown["missing_evidence"]


def test_fixed_seed_execution_is_reproducible(executed, tmp_path):
    scenario = executed["readout_noise_q0"][0]
    repeated = run_experiment(scenario.config, output_path=tmp_path / "readout-repeat.json")
    first = executed["readout_noise_q0"][1]
    assert repeated.observed_counts == first.observed_counts
    assert repeated.diagnosis.category == first.diagnosis.category
    assert repeated.diagnosis.statistics_result.total_variation_distance == first.diagnosis.statistics_result.total_variation_distance


def test_saved_case_finalization_serializes_localizers_without_rerunning(executed, tmp_path, monkeypatch):
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    for name, (_, result) in executed.items():
        (case_dir / f"{name}.json").write_text(json.dumps(result.to_dict()), encoding="utf-8")
    monkeypatch.setattr(phase1, "run_experiment", lambda *args, **kwargs: pytest.fail("finalizer reran a scenario"))

    report = finalize_existing_experiment(tmp_path)
    assert report["case_count"] == 9
    assert (tmp_path / "REPORT.md").is_file()
    assert (tmp_path / "phase1_results.json").is_file()
    assert (tmp_path / "case_summary.csv").is_file()
    assert phase1._json_plain(HardwareCandidate("qubit", 1, 0.8, "readout", ["calibration"])) == {
        "component_type": "qubit", "component": 1, "score": 0.8,
        "reason": "readout", "evidence": ["calibration"],
    }


def test_saved_report_preserves_historical_coherence_false_positive():
    saved_path = phase1.OUTPUT_DIR / "cases" / "coherence_healthy_control.json"
    if not saved_path.is_file():
        pytest.skip(
            "The historical Phase 1 case is a local-only ignored artifact; "
            "the tracked Phase 1.4 seed-sensitivity cases are not substituted for it."
        )
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved["diagnosis"]["category"] == HARDWARE_CLASS
    assert saved["diagnosis"]["anomaly_detected"] is False
    assert any(
        "gate_duration=2.0000" in candidate["reason"]
        for candidate in saved["diagnosis"]["hardware_evidence_candidates"]
    )


def test_phase1_healthy_coherence_control_is_no_anomaly_after_fix(tmp_path):
    scenario = next(item for item in build_scenarios() if item.scenario_id == "coherence_healthy_control")
    result = run_experiment(scenario.config, output_path=tmp_path / "coherence-healthy.json")

    assert result.diagnosis.anomaly_detected is False
    assert result.diagnosis.category is DiagnosisCategory.NO_ANOMALY
    assert result.diagnosis.hardware_evidence_candidates == []


def test_phase1_gate_fault_localization_remains_intact(tmp_path):
    scenario = next(item for item in build_scenarios() if item.scenario_id == "gate_noise_x_q0")
    result = run_experiment(scenario.config, output_path=tmp_path / "gate-fault.json")
    record = case_record(scenario, result)

    assert record["predicted_class"] == HARDWARE_CLASS
    assert record["localization_correct"] is True


def test_experiment_and_finalizer_refuse_existing_outputs(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        phase1.run_experiment_set(tmp_path)
    (tmp_path / "cases").mkdir()
    for scenario in build_scenarios():
        (tmp_path / "cases" / f"{scenario.scenario_id}.json").write_text("{}", encoding="utf-8")
    (tmp_path / "REPORT.md").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        finalize_existing_experiment(tmp_path)
    assert (tmp_path / "REPORT.md").read_text(encoding="utf-8") == "preserve"
