import json
import csv
from pathlib import Path

from bugs4q_validation import (
    Bugs4QCase,
    discover_bugs4q_files,
    load_bugs4q_dataset,
    run_bugs4q_validation,
    validate_bugs4q_case,
    validate_bugs4q_dataset,
    write_bugs4q_report,
)
from diagnosis_engine import DiagnosisCategory


def _case(case_id, **overrides):
    raw = {
        "case_id": case_id,
        "expected_circuit": [["x", 0]],
        "observed_circuit": [["h", 0]],
        "expected_distribution": {"0": 1.0, "1": 0.0},
        "observed_distribution": {"0": 0.0, "1": 1.0},
        "references": [
            {
                "source": "independent publication",
                "category": "PROGRAM_FAULT",
                "independent": True,
                "reliable": True,
            }
        ],
    }
    raw.update(overrides)
    return raw


def test_loads_bugs4q_json_case_and_runs_existing_diagnosis(tmp_path):
    dataset_dir = tmp_path / "Bugs4Q"
    dataset_dir.mkdir()
    (dataset_dir / "cases.json").write_text(json.dumps({"cases": [_case("B4Q-001")]}), encoding="utf-8")

    dataset = load_bugs4q_dataset(tmp_path)
    report = validate_bugs4q_dataset(dataset)

    assert dataset.found
    assert len(dataset.cases) == 1
    result = report.records[0]
    assert result.case_id == "B4Q-001"
    assert result.support_status == "SUPPORTED_VERIFIABLE"
    assert result.reference_status == "RELIABLE"
    assert result.observed_category is DiagnosisCategory.PROGRAM_FAULT
    assert result.case_level_category_match is True
    assert result.program_candidates
    assert result.evidence


def test_parses_gate_operation_objects_and_calibration(tmp_path):
    path = tmp_path / "Bugs4Q" / "operations.json"
    path.parent.mkdir()
    case = _case(
        "B4Q-object-op",
        expected_circuit=[{"name": "rx", "qubits": [0], "parameters": [0.5]}],
        observed_circuit=[{"name": "rx", "qubits": [0], "parameters": [0.5]}],
        hardware_calibration={
            "n_qubits": 1,
            "qubits": {"0": {"readout_error": 0.01}},
            "gates": {"rx": [{"qubits": [0], "error": 0.01, "duration": 0.02}]},
        },
        expected_distribution={"0": 1.0, "1": 0.0},
        observed_distribution={"0": 1.0, "1": 0.0},
        references=[],
    )
    path.write_text(json.dumps([case]), encoding="utf-8")

    record = validate_bugs4q_dataset(load_bugs4q_dataset(tmp_path)).records[0]

    assert record.supported
    assert record.observed_category is DiagnosisCategory.NO_ANOMALY
    assert record.reference_status == "MISSING"
    assert record.case_level_category_match is None


def test_supported_but_uncertain_reference_is_not_scored(tmp_path):
    path = tmp_path / "Bugs4Q" / "uncertain.json"
    path.parent.mkdir()
    path.write_text(json.dumps([_case(
        "B4Q-uncertain",
        references=[{"source": "report A", "category": "PROGRAM_FAULT", "reliable": False}],
    )]), encoding="utf-8")

    record = validate_bugs4q_dataset(load_bugs4q_dataset(tmp_path)).records[0]

    assert record.support_status == "SUPPORTED_UNCERTAIN"
    assert record.reference_status == "UNCERTAIN"
    assert record.case_level_category_match is None
    assert any("not counted as confirmed" in evidence for evidence in record.evidence)


def test_conflicting_reference_sources_are_preserved_and_unresolved():
    case = _case(
        "B4Q-conflict",
        references=[
            {"source": "report A", "category": "PROGRAM_FAULT", "reliable": True, "independent": True},
            {"source": "report B", "category": "HARDWARE_ANOMALY", "reliable": True, "independent": True},
        ],
    )
    result = validate_bugs4q_case(Bugs4QCase("B4Q-conflict", case, "Bugs4Q/cases.json"))

    assert result.reference_status == "CONFLICTING"
    assert result.case_level_category_match is None
    assert result.expected_reference["conflicting"] is True
    assert len(result.expected_reference["references"]) == 2


def test_unsupported_operation_is_retained_without_forced_diagnosis():
    case = _case("B4Q-unsupported", expected_circuit=[["gadget_x", 0]])
    result = validate_bugs4q_case(Bugs4QCase("B4Q-unsupported", case, "Bugs4Q/cases.json"))

    assert not result.supported
    assert result.support_status == "UNSUPPORTED"
    assert result.observed_category is None
    assert result.validation_status == "NOT_ASSESSED"
    assert "Unsupported" in result.unsupported_or_error_reason


def test_parsing_and_execution_failures_are_distinguished():
    bad_circuit = _case("B4Q-bad-circuit", expected_circuit="not-a-circuit")
    bad_distribution = _case("B4Q-bad-distribution", expected_distribution={"0": -1.0})

    parsing = validate_bugs4q_case(Bugs4QCase("B4Q-bad-circuit", bad_circuit, "cases.json"))
    execution = validate_bugs4q_case(Bugs4QCase("B4Q-bad-distribution", bad_distribution, "cases.json"))

    assert parsing.support_status == "PARSING_FAILURE"
    assert execution.support_status == "EXECUTION_FAILURE"
    assert parsing.validation_status == execution.validation_status == "NOT_ASSESSED"


def test_reference_disagreement_does_not_override_pipeline_diagnosis():
    case = _case(
        "B4Q-reference-mismatch",
        expected_circuit=[["x", 0]],
        observed_circuit=[["x", 0]],
        expected_distribution={"0": 1.0, "1": 0.0},
        observed_distribution={"0": 1.0, "1": 0.0},
        references=[{
            "source": "independent reference",
            "category": "PROGRAM_FAULT",
            "reliable": True,
            "independent": True,
        }],
    )
    result = validate_bugs4q_case(Bugs4QCase("B4Q-reference-mismatch", case, "cases.json"))

    assert result.observed_category is DiagnosisCategory.NO_ANOMALY
    assert result.case_level_category_match is False
    assert result.reference_status == "RELIABLE"


def test_empty_dataset_is_explicit_and_report_is_written(tmp_path):
    report = run_bugs4q_validation(tmp_path)
    output = write_bugs4q_report(report, tmp_path / "out" / "report.json")

    serialized = json.loads(output.read_text(encoding="utf-8"))
    assert report.dataset_found is False
    assert report.discovered_cases == 0
    assert report.reliable_ground_truth_cases == 0
    assert serialized["metrics"]["aggregate_accuracy"] is None
    assert serialized["dataset_issues"]


def test_csv_json_fields_load_as_case_data(tmp_path):
    path = tmp_path / "Bugs4Q" / "cases.csv"
    path.parent.mkdir()
    row = _case("B4Q-csv")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=row)
        writer.writeheader()
        writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})

    dataset = load_bugs4q_dataset(tmp_path)
    result = validate_bugs4q_dataset(dataset).records[0]

    assert len(dataset.cases) == 1
    assert dataset.cases[0].case_id == "B4Q-csv"
    assert result.support_status == "SUPPORTED_VERIFIABLE"
    assert result.case_level_category_match is True


def test_bad_json_file_is_a_dataset_parse_failure(tmp_path):
    path = tmp_path / "Bugs4Q" / "broken.json"
    path.parent.mkdir()
    path.write_text("{not-json", encoding="utf-8")

    dataset = load_bugs4q_dataset(tmp_path)
    report = validate_bugs4q_dataset(dataset)

    assert dataset.files
    assert report.discovered_cases == 0
    assert report.execution_failures == 1
    assert report.dataset_issues[0].status == "PARSING_FAILURE"


def test_repository_source_files_are_not_reported_as_dataset_files():
    project_root = Path(__file__).resolve().parents[1]

    supported, unsupported = discover_bugs4q_files(project_root)

    assert all(path.suffix.lower() != ".py" for path in supported + unsupported)

