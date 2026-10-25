"""Sanity checks for the dedicated deterministic robustness sweep."""

from experiments.robustness_evaluation import SEEDS, _run_case, generate_cases


def test_sweep_covers_seeded_sizes_depths_families_faults_and_hardware_strengths():
    cases = generate_cases()
    assert len(cases) == 1188
    assert {case["seed"] for case in cases} == set(SEEDS)
    assert {case["size"] for case in cases} == {"small", "medium", "larger"}
    assert {case["depth"] for case in cases} == {2, 4}
    assert {case["family"] for case in cases} == {"ghz", "parity", "qft_like"}
    assert {case["fault_type"] for case in cases if case["scenario"] == "PROGRAM_FAULT"} == {
        "gate_replacement", "gate_deletion", "gate_insertion", "parameter_modification",
        "single_qubit_fault", "two_qubit_fault",
    }
    assert {case["hardware_type"] for case in cases if case["scenario"] == "HARDWARE_ANOMALY"} == {
        "readout_error", "gate_error", "t1_degradation", "t2_degradation", "combined",
    }
    assert {case["strength"] for case in cases if case["scenario"] == "HARDWARE_ANOMALY"} == {0.15, 0.45, 0.8}
    assert all(case["reference"] != case["observed_program"]
               for case in cases if case["scenario"] == "PROGRAM_FAULT")
    assert all(case["reference"] == case["observed_program"]
               for case in cases if case["scenario"] != "PROGRAM_FAULT")


def test_runner_uses_deterministic_diagnosis_and_records_real_candidates():
    case = next(case for case in generate_cases() if case["scenario"] == "PROGRAM_FAULT" and case["fault_type"] == "two_qubit_fault")
    record = _run_case(case)
    assert record["anomaly_truth"] is True
    assert record["category_truth"] == "PROGRAM_FAULT"
    assert "category_predicted" in record
    assert isinstance(record["program_evidence"], list)
    assert isinstance(record["hardware_candidates"], list)
