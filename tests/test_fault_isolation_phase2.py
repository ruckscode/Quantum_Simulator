from __future__ import annotations

from types import SimpleNamespace

from experiments.fault_isolation_phase1 import (
    HARDWARE_CLASS,
    NORMAL_CLASS,
    PROGRAM_CLASS,
    UNCERTAIN_CLASS,
    build_scenarios as build_phase1_scenarios,
)
from experiments.fault_isolation_phase2 import (
    SEEDS,
    SHOTS,
    build_scenarios,
    _case_outcome,
)
from quantum_simulator.diagnosis_engine import DiagnosisCategory


def test_phase2_scenarios_are_reproducible_and_structurally_new():
    first = build_scenarios()
    second = build_scenarios()
    prior = build_phase1_scenarios()
    prior_circuits = {
        (tuple(map(str, scenario.config.circuit)), tuple(map(str, scenario.config.observed_circuit or scenario.config.circuit)))
        for scenario in prior
    }

    assert first == second
    assert len(first) == 12
    assert len({scenario.circuit_family_id for scenario in first}) == 6
    assert all(scenario.config.num_qubits == 4 and scenario.config.shots == SHOTS for scenario in first)
    assert SEEDS == (20262001, 20262002, 20262003)
    assert all(
        (tuple(map(str, scenario.config.circuit)), tuple(map(str, scenario.config.observed_circuit or scenario.config.circuit)))
        not in prior_circuits
        for scenario in first
    )
    assert all(any(qubit >= 2 for operation in scenario.config.circuit for qubit in _operation_qubits_for_test(operation)) for scenario in first)


def _operation_qubits_for_test(operation):
    if operation[0] == "measure":
        return operation[1]
    if operation[0] in {"rzz", "cnot", "cx"}:
        return operation[1:3]
    return operation[1:2]


def test_program_mutations_are_predeclared_at_new_operation_positions():
    scenarios = {scenario.scenario_id: scenario for scenario in build_scenarios()}
    rotation = scenarios["program_fault_ry_to_rz_q2"]
    assert rotation.config.program_fault_operation_index == 2
    assert rotation.config.program_fault_qubits == [2]
    assert rotation.config.circuit[2][0] == "ry"
    assert rotation.config.observed_circuit[2][0] == "rz"
    assert rotation.ground_truth_class == PROGRAM_CLASS

    phase = scenarios["program_fault_remove_rzz_q0_q2"]
    assert phase.config.program_fault_operation_index == 2
    assert phase.config.circuit[2][0] == "rzz"
    assert len(phase.config.observed_circuit) == len(phase.config.circuit) - 1
    assert phase.config.observed_circuit[2][0] == "h"


def test_new_hardware_locations_strengths_and_localization_targets_are_declared():
    scenarios = {scenario.scenario_id: scenario for scenario in build_scenarios()}
    assert scenarios["hardware_readout_q3_moderate"].localization_targets[0]["component"] == 3
    assert scenarios["hardware_cx_coupler_q1_q3"].localization_targets[0]["component"] == [1, 3]
    assert scenarios["hardware_rzz_error_near_threshold"].localization_targets[0]["component"] == [0, 2]
    strengths = {
        scenarios[name].config.hardware_parameters["qubits"]["3"]["readout_error"]
        for name in (
            "hardware_readout_q3_moderate",
            "hardware_readout_q3_near_threshold",
            "hardware_readout_q3_subthreshold",
        )
    }
    assert strengths == {0.04, 0.055, 0.20}
    assert scenarios["hardware_rzz_error_near_threshold"].config.hardware_parameters["gates"]


def test_matched_program_and_hardware_cases_share_the_same_reference_circuit():
    scenarios = {scenario.scenario_id: scenario for scenario in build_scenarios()}
    program = scenarios["program_fault_ry_to_rz_q2"]
    hardware = scenarios["hardware_readout_inversion_q2"]
    assert program.pair_id == hardware.pair_id
    assert program.config.circuit == hardware.config.circuit
    assert program.config.observed_circuit != hardware.config.observed_circuit
    assert hardware.config.hardware_parameters["qubits"]["2"]["readout_error"] == 1.0


def test_ambiguous_scenarios_and_label_sources_are_independent_of_predictions():
    scenarios = {scenario.scenario_id: scenario for scenario in build_scenarios()}
    overlap = scenarios["overlap_ry_to_rz_plus_readout_q2"]
    unknown = scenarios["cause_unknown_phase_observation"]
    assert overlap.ground_truth_class == UNCERTAIN_CLASS
    assert overlap.config.observed_circuit != overlap.config.circuit
    assert overlap.config.hardware_parameters["qubits"]["2"]["readout_error"] == 0.20
    assert {target["kind"] for target in overlap.localization_targets} == {"program", "hardware"}
    assert unknown.ground_truth_class == UNCERTAIN_CLASS
    assert unknown.config.hardware_parameters is None
    assert unknown.config.observed_counts == {"0": SHOTS}

    hardware_truth = scenarios["hardware_readout_q3_moderate"]
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
            summary="deliberately contradictory fake prediction",
        ),
    )
    row = _case_outcome(hardware_truth, fake_result, SEEDS[1])
    assert row["ground_truth_class"] == HARDWARE_CLASS
    assert row["predicted_class"] == PROGRAM_CLASS
    assert row["category_correct"] is False
    assert row["evaluation_seed"] == row["settings"]["seed"] == SEEDS[1]


def test_healthy_controls_have_explicit_normal_labels_and_complete_profiles():
    controls = [scenario for scenario in build_scenarios() if scenario.ground_truth_class == NORMAL_CLASS]
    assert len(controls) == 2
    assert all(scenario.config.ground_truth_label == NORMAL_CLASS for scenario in controls)
    assert all(scenario.config.hardware_parameters["qubits"]["3"]["t1"] == 1e6 for scenario in controls)
