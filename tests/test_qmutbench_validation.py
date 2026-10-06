import json
from pathlib import Path

from benchmark_validation import run_benchmark_validation
from quantum_simulator.diagnosis_engine import DiagnosisCategory
from qmutbench_validation import (
    QMutBenchCase,
    determine_qmutbench_ground_truth,
    diagnose_saved_qmutbench_case,
    discover_qmutbench_files,
    load_qmutbench_dataset,
    normalize_qmutbench_case,
    parse_qasm_file,
    validate_qmutbench_case,
    validate_qmutbench_dataset,
)


QMB_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "qmutbench"


def test_saved_case_adapter_passes_full_distributions_and_localization_circuits(monkeypatch):
    saved_case = {
        "execution_status": "COMPLETED",
        "behavioral_comparison": {
            "expected_distribution": {"00": 0.75, "01": 0.25},
            "observed_distribution": {"00": 0.25, "01": 0.75},
        },
        "original_output": {"distribution": {"00": 1.0}},
        "mutant_output": {"distribution": {"01": 1.0}},
        "reference_circuit": [["h", 1], ["measure", 1]],
        "mutant_circuit": [["x", 1], ["measure", 1]],
        "source_level_information": {"num_qubits": 2},
    }
    captured = {}
    expected_result = object()

    def fake_diagnose_execution(**kwargs):
        captured.update(kwargs)
        return expected_result

    monkeypatch.setattr("qmutbench_validation.diagnose_execution", fake_diagnose_execution)
    result = diagnose_saved_qmutbench_case(saved_case)

    assert result is expected_result
    assert captured == {
        "expected_distribution": {"00": 0.75, "01": 0.25},
        "observed_distribution": {"00": 0.25, "01": 0.75},
        "expected_circuit": [("h", 1), ("barrier", 1)],
        "observed_circuit": [("x", 1), ("barrier", 1)],
        "threshold": 0.05,
        "shots": 1024,
    }


def test_saved_case_adapter_rejects_missing_full_distributions():
    saved_case = {
        "execution_status": "COMPLETED",
        "behavioral_comparison": None,
        "original_output": {"distribution": {"00": 1.0}},
        "mutant_output": {"distribution": {"01": 1.0}},
        "reference_circuit": [["x", 0]],
        "mutant_circuit": [["h", 0]],
    }
    try:
        diagnose_saved_qmutbench_case(saved_case)
    except ValueError as exc:
        assert "behavioral_comparison" in str(exc)
    else:
        raise AssertionError("adapter unexpectedly used sparse distributions")


def test_parses_real_original_qasm():
    operations, num_qubits = parse_qasm_file(QMB_ROOT / "Origin_programs" / "ghz_2_qubits.qasm")

    assert num_qubits == 2
    assert operations[:2] == [("h", 1), ("cx", 1, 0)]
    assert operations[-2:] == [("measure", 0), ("measure", 1)]


def test_parses_supported_real_mutant_qasm():
    operations, num_qubits = parse_qasm_file(QMB_ROOT / "Mutated_programs" / "Mutants_ghz_2_qubits" / "AddGate_ry_inGap_1_.qasm")

    assert num_qubits == 2
    assert operations[0] == ("ry", 1, 1.5707963267948966)


def test_parses_whole_register_measurement(tmp_path):
    source = tmp_path / "register_measure.qasm"
    source.write_text(
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; measure q -> c;',
        encoding="utf-8",
    )
    operations, width = parse_qasm_file(source)
    assert width == 2
    assert operations == [("measure", 0), ("measure", 1)]


def test_discovers_pairs_and_executes_supported_real_qasm_mutants():
    dataset = discover_qmutbench_files(QMB_ROOT.parents[1])

    assert len(dataset.cases) == 12
    assert len(dataset.data_files) == 13
    assert all(len(case.source_files) == 2 for case in dataset.cases)
    assert all(case.source_files[0].endswith("ghz_2_qubits.qasm") for case in dataset.cases)
    report = validate_qmutbench_dataset(dataset).to_dict()
    assert report["executed_cases"] == 12
    assert report["unsupported_cases"] == 0
    assert report["diagnosed_cases"] == 12
    assert sum(case["execution_status"] == "COMPLETED" for case in report["cases"]) == 12
    assert next(case for case in report["cases"] if case["case_id"] == "ReplaceGate_id_inPositionOfGate_1")["execution_status"] == "COMPLETED"
    assert all(sum(case["original_output"]["counts"].values()) == 1024 for case in report["cases"] if case["execution_status"] == "COMPLETED")
    assert next(case for case in report["cases"] if case["case_id"] == "AddGate_rxx_inGap_1_")["execution_status"] == "COMPLETED"


def _write_qasm_program(path, gate):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\n{gate} q[0];\nmeasure q[0] -> c[0];\n', encoding="utf-8")


def test_multiple_programs_pair_only_with_matching_mutant_directory(tmp_path):
    base = tmp_path / "data" / "qmutbench"
    _write_qasm_program(base / "Origin_programs" / "alpha.qasm", "h")
    _write_qasm_program(base / "Origin_programs" / "beta.qasm", "x")
    _write_qasm_program(base / "Mutated_programs" / "Mutants_alpha" / "AddGate_x.qasm", "x")
    _write_qasm_program(base / "Mutated_programs" / "Mutants_beta" / "ReplaceGate_h.qasm", "h")

    dataset = discover_qmutbench_files(tmp_path)
    pairs = {case.case_id: case.source_files[0] for case in dataset.cases}

    assert pairs == {
        "AddGate_x": str(Path("data/qmutbench/Origin_programs/alpha.qasm")),
        "ReplaceGate_h": str(Path("data/qmutbench/Origin_programs/beta.qasm")),
    }


def test_unmatched_mutant_directory_becomes_unsupported_case(tmp_path):
    base = tmp_path / "data" / "qmutbench"
    _write_qasm_program(base / "Origin_programs" / "alpha.qasm", "h")
    _write_qasm_program(base / "Mutated_programs" / "Mutants_missing" / "AddGate_x.qasm", "x")

    dataset = discover_qmutbench_files(tmp_path)
    assert len(dataset.cases) == 1
    assert dataset.cases[0].source_files == [str(Path("data/qmutbench/Mutated_programs/Mutants_missing/AddGate_x.qasm"))]
    assert any(issue["status"] == "UNMATCHED_MUTANT_DIRECTORY" for issue in dataset.issues)

    result = validate_qmutbench_case(dataset.cases[0])
    assert result.support_status == "UNSUPPORTED"
    assert result.execution_status == "NOT_EXECUTED"


def test_rxx_mutant_executes_and_is_diagnosed():
    dataset = discover_qmutbench_files(QMB_ROOT.parents[1])
    case = next(case for case in dataset.cases if "AddGate_rxx" in case.case_id)
    result = validate_qmutbench_case(case)

    assert result.support_status == "SUPPORTED"
    assert result.execution_status == "COMPLETED"
    assert result.system_diagnosis is not None
    assert result.original_output is not None
    assert result.mutant_output is not None


def test_real_reference_and_supported_mutant_execution_are_recorded():
    dataset = discover_qmutbench_files(QMB_ROOT.parents[1])
    case = next(case for case in dataset.cases if case.case_id == "AddGate_ry_inGap_1_")
    result = validate_qmutbench_case(case, shots=128, seed=43)

    assert result.execution_status == "COMPLETED"
    assert result.original_program.endswith("ghz_2_qubits.qasm")
    assert result.mutant_filename == "AddGate_ry_inGap_1_.qasm"
    assert sum(result.original_output["counts"].values()) == 128
    assert sum(result.mutant_output["counts"].values()) == 128
    assert result.behavioral_comparison["shots"] == 128
    assert result.system_diagnosis is not None
    assert result.program_candidates
    assert result.independent_ground_truth_available is False


def test_behavioral_comparison_reports_equivalent_and_different_outputs():
    equivalent = QMutBenchCase(
        "same-output", {
            "original_circuit": [["x", 0]], "mutant_circuit": [["x", 0]],
            "num_qubits": 1, "_qasm_defer_execution": True,
        }, ["Origin_programs/ref.qasm", "Mutated_programs/Mutants_ref/ReplaceGate_i.qasm"],
    )
    different = QMutBenchCase(
        "different-output", {
            "original_circuit": [["x", 0]], "mutant_circuit": [["h", 0]],
            "num_qubits": 1, "_qasm_defer_execution": True,
        }, ["Origin_programs/ref.qasm", "Mutated_programs/Mutants_ref/ReplaceGate_h.qasm"],
    )

    same_result = validate_qmutbench_case(equivalent, shots=128, seed=17)
    different_result = validate_qmutbench_case(different, shots=128, seed=17)

    assert same_result.behaviorally_different is False
    assert same_result.behavioral_comparison["total_variation_distance"] == 0.0
    assert different_result.behaviorally_different is True
    assert different_result.behavioral_comparison["total_variation_distance"] > 0.0
    assert same_result.system_diagnosis is not None
    assert different_result.system_diagnosis is not None


def test_execution_failure_is_recorded_without_diagnosis(monkeypatch):
    case = QMutBenchCase(
        "execution-failure", {
            "original_circuit": [["x", 0]], "mutant_circuit": [["h", 0]],
            "num_qubits": 1, "_qasm_defer_execution": True,
        }, ["Origin_programs/ref.qasm", "Mutated_programs/Mutants_ref/AddGate_h.qasm"],
    )
    calls = 0
    def fail_mutant_execution(circuit, num_qubits, shots, seed):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulator failure")
        return {"0": shots}

    monkeypatch.setattr("qmutbench_validation._sample_measurement_counts", fail_mutant_execution)
    result = validate_qmutbench_case(case)

    assert result.execution_status == "EXECUTION_FAILURE"
    assert "simulator failure" in result.errors[0]
    assert result.system_diagnosis is None
    assert result.original_output["counts"] == {"0": 1024}
    assert result.mutant_output is None


def test_malformed_qasm_is_reported_as_parse_failure(tmp_path):
    malformed = tmp_path / "broken.qasm"
    malformed.write_text("OPENQASM 2.0; qreg q[2]; h q[3];", encoding="utf-8")

    try:
        parse_qasm_file(malformed)
    except ValueError as exc:
        assert "undeclared or out-of-range qubit" in str(exc)
    else:
        raise AssertionError("malformed QASM unexpectedly parsed")


def _qmut_case(case_id="QMB-TEST-1", **overrides):
    raw = {
        "case_id": case_id,
        "original_circuit": [["x", 0]],
        "mutant_circuit": [["h", 0]],
        "mutation": {
            "operator": "replace_gate",
            "changed_operation": 0,
            "qubits": [0],
        },
        "source_code": "def reference_program(): ...",
        "expected_behavior": "mutant differs from reference",
    }
    raw.update(overrides)
    return raw


def _write_fixture(tmp_path, cases, metadata=None):
    data_dir = tmp_path / "TEST_FIXTURES" / "QMutBench"
    data_dir.mkdir(parents=True)
    document = {"metadata": {"benchmark": "QMutBench", "fixture_classification": "TEST_FIXTURE", **(metadata or {})}, "cases": cases}
    path = data_dir / "cases.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_discovers_verified_qmutbench_case_file_and_labels_fixture(tmp_path):
    path = _write_fixture(tmp_path, [_qmut_case()])

    dataset = load_qmutbench_dataset(tmp_path)
    report = validate_qmutbench_dataset(dataset).to_dict()

    assert dataset.provenance_verified
    assert dataset.data_files == [str(path.relative_to(tmp_path))]
    assert len(dataset.cases) == 1
    assert report["dataset_type"] == "TEST_FIXTURE"
    assert report["cases"][0]["case_id"] == "QMB-TEST-1"


def test_generic_mutation_filename_without_qmutbench_provenance_is_ignored(tmp_path):
    path = tmp_path / "mutations.json"
    path.write_text(json.dumps({"cases": [_qmut_case()]}), encoding="utf-8")

    dataset = discover_qmutbench_files(tmp_path)

    assert not dataset.provenance_verified
    assert dataset.data_files == []
    assert dataset.cases == []
    assert dataset.issues[0]["status"] == "DATASET_UNAVAILABLE"


def test_normalization_preserves_circuits_mutation_source_and_missing_fields(tmp_path):
    _write_fixture(tmp_path, [_qmut_case("QMB-NORMALIZE", source_location="src/example.py:4")])
    normalized = normalize_qmutbench_case(load_qmutbench_dataset(tmp_path).cases[0])

    assert normalized.benchmark_name == "QMutBench"
    assert normalized.case_id == "QMB-NORMALIZE"
    assert normalized.reference_circuit == [["x", 0]]
    assert normalized.mutant_circuit == [["h", 0]]
    assert normalized.mutation_metadata["operator"] == "replace_gate"
    assert normalized.source_level_information["source_location"] == "src/example.py:4"
    assert normalized.ground_truth is None
    assert normalized.ground_truth_status == "MISSING"


def test_ground_truth_statuses_distinguish_reliable_uncertain_missing_and_conflicting():
    reliable = _qmut_case(
        "QMB-RELIABLE-FIXTURE",
        ground_truth={"mutant_killed": True, "label_semantics": "mutation_outcome"},
        ground_truth_reliable=True,
        reference_independent=True,
    )
    uncertain = _qmut_case(
        "QMB-UNCERTAIN-FIXTURE",
        references=[{"source": "fixture note", "mutant_killed": True, "reliable": False, "independent": False}],
    )
    missing = _qmut_case("QMB-MISSING-FIXTURE")
    conflicting = _qmut_case(
        "QMB-CONFLICT-FIXTURE",
        references=[
            {"source": "fixture source A", "mutant_killed": True, "reliable": True, "independent": True},
            {"source": "fixture source B", "mutant_killed": False, "reliable": True, "independent": True},
        ],
    )

    statuses = []
    for raw in (reliable, uncertain, missing, conflicting):
        info = determine_qmutbench_ground_truth(normalize_qmutbench_case(QMutBenchCase(raw["case_id"], raw, ["TEST_FIXTURES/QMutBench/cases.json"])))
        statuses.append(info["status"])

    assert statuses == ["RELIABLE", "UNCERTAIN", "MISSING", "CONFLICTING"]


def test_reliable_mutation_outcome_is_not_mapped_to_system_category_metrics():
    from qmutbench_validation import QMutBenchDataset, validate_qmutbench_dataset

    raw = _qmut_case(
        "QMB-OUTCOME-SEPARATION-FIXTURE",
        ground_truth={"mutant_killed": True, "label_semantics": "mutation_outcome"},
        ground_truth_reliable=True,
        reference_independent=True,
    )
    case = QMutBenchCase(raw["case_id"], raw, ["TEST_FIXTURES/QMutBench/cases.json"])
    report = validate_qmutbench_dataset(QMutBenchDataset(
        root="TEST_FIXTURES",
        data_files=["QMutBench/cases.json"],
        cases=[case],
        provenance_verified=True,
        test_fixture_data=True,
    )).to_dict()

    result = report["cases"][0]
    assert result["ground_truth_status"] == "RELIABLE"
    assert result["diagnosis_ground_truth_compatible"] is False
    assert result["case_level_category_match"] is None
    assert report["reliable_ground_truth_cases"] == 1
    assert report["metrics"]["accuracy"] is None


def test_mutation_label_is_not_injected_into_diagnosis():
    case_raw = _qmut_case(
        "QMB-ISOLATION-FIXTURE",
        original_circuit=[["x", 0]],
        mutant_circuit=[["x", 0]],
        mutation={"operator": "replace_gate", "label": "PROGRAM_FAULT"},
        ground_truth={"mutant_killed": True},
    )

    result = validate_qmutbench_case(QMutBenchCase(case_raw["case_id"], case_raw, ["TEST_FIXTURES/QMutBench/cases.json"]))

    assert result.mutation_metadata["label"] == "PROGRAM_FAULT"
    assert result.system_diagnosis["category"] == DiagnosisCategory.NO_ANOMALY.value
    assert result.program_candidates == []
    assert result.ground_truth_status == "UNCERTAIN"
    assert result.case_level_category_match is None


def test_unsupported_and_malformed_cases_are_not_forced_into_diagnosis():
    unsupported = _qmut_case("QMB-UNSUPPORTED-FIXTURE", mutant_circuit=[["unsupported_op", 0]])
    malformed = _qmut_case("QMB-MALFORMED-FIXTURE", original_circuit="not-a-circuit")

    unsupported_result = validate_qmutbench_case(QMutBenchCase(unsupported["case_id"], unsupported, ["TEST_FIXTURES/QMutBench/cases.json"]))
    malformed_result = validate_qmutbench_case(QMutBenchCase(malformed["case_id"], malformed, ["TEST_FIXTURES/QMutBench/cases.json"]))

    assert unsupported_result.support_status == "UNSUPPORTED"
    assert unsupported_result.system_diagnosis is None
    assert malformed_result.support_status == "PARSING_FAILURE"
    assert malformed_result.system_diagnosis is None


def test_missing_reference_or_mutant_circuit_is_explicit():
    raw = {"case_id": "QMB-MISSING-CIRCUIT-FIXTURE", "mutation_operator": "delete_gate"}
    normalized = normalize_qmutbench_case(QMutBenchCase(raw["case_id"], raw, ["TEST_FIXTURES/QMutBench/cases.json"]))
    result = validate_qmutbench_case(QMutBenchCase(raw["case_id"], raw, ["TEST_FIXTURES/QMutBench/cases.json"]))

    assert normalized.reference_circuit is None
    assert normalized.mutant_circuit is None
    assert normalized.ground_truth_status == "MISSING"
    assert result.support_status == "PARSING_FAILURE"
    assert result.execution_status == "NOT_EXECUTED"
    assert result.system_diagnosis is None


def test_combined_report_keeps_benchmark_results_separate(tmp_path):
    _write_fixture(tmp_path, [_qmut_case("QMB-COMBINED-FIXTURE")])
    output = tmp_path / "reports" / "benchmarks"

    combined = run_benchmark_validation(tmp_path, report_directory=output)

    assert set(combined["benchmark_summary"]) == {"Bugs4Q", "QMutBench"}
    assert combined["benchmark_summary"]["Bugs4Q"]["status"] == "PARTIAL / DATASET_UNAVAILABLE"
    assert combined["benchmark_summary"]["QMutBench"]["cases"] == 1
    assert combined["combined_metrics"]["accuracy"] is None
    assert (output / "bugs4q_validation_report.json").is_file()
    assert (output / "qmutbench_validation_report.json").is_file()
    assert (output / "combined_validation_report.json").is_file()
