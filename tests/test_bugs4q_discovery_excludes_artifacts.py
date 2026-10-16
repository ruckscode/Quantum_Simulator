import json
from pathlib import Path

from bugs4q_validation import discover_bugs4q_files, load_bugs4q_dataset


def test_bugs4q_discovery_ignores_generated_reports_and_documentation(tmp_path):
    raw = tmp_path / "data" / "Bugs4Q_official" / "cases.json"
    raw.parent.mkdir(parents=True)
    raw.write_text(json.dumps([{"case_id": "raw-1", "expected_circuit": [["h", 0]]}]), encoding="utf-8")

    reports = tmp_path / "reports" / "benchmarks"
    reports.mkdir(parents=True)
    (reports / "bugs4q_full_results.json").write_text(
        json.dumps({"report_name": "generated report, not dataset"}),
        encoding="utf-8",
    )
    (reports / "bugs4q_validation_report.json").write_text("{}", encoding="utf-8")

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "BUGS4Q_OFFICIAL_DATASET.md").write_text(
        "# Bugs4Q dataset setup instructions", encoding="utf-8"
    )

    supported, unsupported = discover_bugs4q_files(tmp_path)

    assert supported == [raw]
    assert unsupported == []

    dataset = load_bugs4q_dataset(tmp_path)
    assert len(dataset.cases) == 1
    assert dataset.cases[0].case_id == "raw-1"
    assert all("reports" not in path.lower() for path in dataset.files)
    assert all("docs" not in path.lower() for path in dataset.files)
