from pathlib import Path

import pytest

from bugs4q_integration import (
    INTEGRATED_CASES,
    OFFICIAL_ROOT,
    _extract_source,
    discover_bugs4q_cases,
    run_bugs4q_integration,
)


@pytest.fixture(scope="module")
def integration_report():
    return run_bugs4q_integration(shots=128, seed=317)


def test_discovers_only_requested_official_cases():
    assert tuple(discover_bugs4q_cases()) == (8, 12, 17, 25, 26, 30, 31, 39)


@pytest.mark.parametrize("case_number", INTEGRATED_CASES)
def test_translates_official_sources_and_preserves_measurement_mapping(case_number):
    case_dir = OFFICIAL_ROOT / str(case_number)
    buggy, n_buggy = _extract_source((case_dir / f"buggy_{case_number}.py").read_text(encoding="utf-8"), case_number)
    fixed, n_fixed = _extract_source((case_dir / f"fixed_{case_number}.py").read_text(encoding="utf-8"), case_number)
    expected_qubits = {8: 4, 12: 2, 17: 3, 25: 3, 26: 2, 30: 8, 31: 3, 39: 4}
    assert n_buggy == n_fixed == expected_qubits[case_number]
    assert buggy and fixed
    if case_number == 12:
        assert buggy[-1] == fixed[-1] == ("measure", [0, 1], [0, 1])
    elif case_number == 17:
        assert buggy[-1] == ("measure", [1, 0, 2], [1, 0, 2])
        assert fixed[-1] == ("measure", [1, 0, 2], [0, 1, 2])
    elif case_number == 31:
        assert buggy[-1] == ("measure", [0, 1, 2], [0, 1, 2])
        assert fixed[-1] == ("measure", [0, 1, 2], [2, 1, 0])


def test_cases_execute_compare_and_keep_evidence_layers_separate(integration_report):
    assert [case["case_number"] for case in integration_report.cases] == list(INTEGRATED_CASES)
    for case in integration_report.cases:
        assert case["expected_distribution"]
        assert case["observed_distribution"]
        assert case["behavioral_comparison"]["total_variation_distance"] >= 0
        assert case["system_diagnosis"]["category"]
        assert case["execution_status"] == "COMPLETED"
        assert case["case_id"] == f"Bugs4Q-{case['case_number']}"
        assert "bug_type" not in case["system_diagnosis"]
        assert case["ground_truth"]["available"] is False
        assert case["bugs4q_metadata"]["bug_type"]


def test_classical_bit_localization_preserves_both_mappings(integration_report):
    cases = {case["case_number"]: case for case in integration_report.cases}
    for case_number in (17, 31):
        evidence = cases[case_number]["localization_evidence"]
        assert evidence["status"] == "AVAILABLE"
        assert evidence["expected_measurements"]
        assert evidence["observed_measurements"]
        assert "classical-bit destinations" in evidence["note"]
        assert all("measured_qubits" in item and "classical_destinations" in item
                   for item in evidence["expected_measurements"] + evidence["observed_measurements"])


def test_source_level_localization_limitations_are_reported(integration_report):
    cases = {case["case_number"]: case for case in integration_report.cases}
    assert any("source-level CRZ" in item and "internal CX" in item
               for item in cases[8]["limitations"])
    assert any("X(2) candidate" in item and "multi-operation mutation" in item
               for item in cases[30]["limitations"])


def test_report_sources_are_official(integration_report):
    for case in integration_report.cases:
        metadata = case["bugs4q_metadata"]
        for key in ("buggy_source", "fixed_source", "modify_file"):
            assert Path(metadata[key]).is_file()
