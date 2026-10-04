import json

from quantum_simulator.diagnosis_engine import DiagnosisCategory
from quantum_simulator.hardware_model import HardwareModel
from quantum_simulator.validation_framework import (
    ValidationCase,
    run_validation_case,
    run_validation_suite,
    standard_validation_cases,
)


def test_standard_synthetic_scenarios_are_distinguished():
    cases = standard_validation_cases()
    report = run_validation_suite(cases)

    assert len(report.records) >= 8
    assert report.all_passed
    assert all(record.validation_satisfied for record in report.records)
    assert all(record.source == "controlled synthetic validation" for record in report.records)
    assert all(record.expected_behavior and record.observed_behavior for record in report.records)
    assert all(set(record.program) == {"expected", "observed"} for record in report.records)
    assert {record.expected_category for record in report.records} >= {
        DiagnosisCategory.NO_ANOMALY,
        DiagnosisCategory.PROGRAM_FAULT,
        DiagnosisCategory.HARDWARE_ANOMALY,
        DiagnosisCategory.INSUFFICIENT_EVIDENCE,
        DiagnosisCategory.AMBIGUOUS,
    }
    assert {record.case_id for record in report.records} >= {
        "three_qubit_operation",
        "finite_shot_guard_low_shots",
        "finite_shot_guard_high_shots",
        "partial_calibration_missing_gate_entry",
    }


def test_validation_record_preserves_candidates_evidence_and_serializes():
    case = next(case for case in standard_validation_cases() if case.case_id == "program_fault_single_qubit")

    record = run_validation_case(case)

    assert record.validation_satisfied
    assert record.diagnosis_result is not None
    assert record.relevant_candidates
    assert record.evidence
    serialized = json.dumps(record.to_dict())
    assert case.case_id in serialized


def test_invalid_case_inputs_become_failed_records_without_crashing_suite():
    malformed = ValidationCase(
        case_id="malformed_distribution_pair",
        expected_category=DiagnosisCategory.NO_ANOMALY,
        expected_distribution={"0": 1.0},
    )
    invalid_reference = ValidationCase(
        case_id="invalid_qubit_reference",
        expected_category=DiagnosisCategory.NO_ANOMALY,
        expected_distribution={"0": 1.0},
        observed_distribution={"0": 1.0},
        expected_circuit=[("x", 2)],
        observed_circuit=[("x", 2)],
        hardware_model=HardwareModel(1),
    )

    report = run_validation_suite([malformed, invalid_reference])

    assert not report.all_passed
    assert len(report.records) == 2
    assert all(not record.validation_satisfied for record in report.records)
    assert all(record.error for record in report.records)


def test_case_rejects_unrecognized_expected_category_cleanly():
    case = ValidationCase(case_id="bad_category", expected_category="NOT_A_CATEGORY")

    record = run_validation_case(case)

    assert not record.validation_satisfied
    assert "category" in record.error.lower()


def test_suite_rejects_non_case_entries_as_failed_records():
    report = run_validation_suite([object()])

    assert len(report.records) == 1
    assert not report.records[0].validation_satisfied
    assert report.records[0].error
