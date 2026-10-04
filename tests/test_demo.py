import json

from demo import build_demo_report, render_human_report, run_demo


def test_offline_demo_matches_all_five_expected_categories():
    report = build_demo_report()

    assert report["offline"] is True
    assert report["real_ibm_execution"] is False
    assert report["bugs4q_dataset_used"] is False
    assert report["scenario_count"] == 5
    assert report["passed"] == 5
    assert report["failed"] == 0
    assert [scenario["expected_category"] for scenario in report["scenarios"]] == [
        "NO_ANOMALY",
        "PROGRAM_FAULT",
        "HARDWARE_ANOMALY",
        "AMBIGUOUS",
        "INSUFFICIENT_EVIDENCE",
    ]
    assert all(scenario["pass"] for scenario in report["scenarios"])


def test_demo_json_retains_full_diagnosis_and_evidence(tmp_path):
    destination = tmp_path / "final_demo_report.json"
    report = run_demo(destination)
    saved = json.loads(destination.read_text(encoding="utf-8"))

    assert saved["passed"] == 5
    for scenario in saved["scenarios"]:
        diagnosis = scenario["diagnosis"]
        assert {"statistics_result", "program_result", "hardware_result", "evidence", "missing_evidence"} <= set(diagnosis)
        assert "statistical_evidence" in scenario
        assert "execution_metadata" in scenario
        assert "errors" in scenario
    assert report["scenarios"][2]["hardware_evidence"]["candidates"]


def test_human_report_names_fixture_data_and_avoids_confidence_percentages():
    output = render_human_report(build_demo_report())

    assert "Scenarios passed: 5/5" in output
    assert "no confidence score calculated" in output
    assert "no IBM hardware was used" in output
    assert "%" not in output
