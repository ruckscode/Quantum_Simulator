from collections import Counter
import json
from pathlib import Path

import ml_evaluation
import pytest
from ml_evaluation import (
    LABELS,
    OBSERVABLE_ONLY,
    WITH_PROGRAM_COMPARISON,
    _split_feasibility,
    load_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
FULL_DATASET_DIR = ROOT / "reports" / "experiments" / "full_dataset"


def _load_local_full_dataset_or_skip():
    if not FULL_DATASET_DIR.is_dir() or not any(FULL_DATASET_DIR.rglob("*.json")):
        pytest.skip(
            "Generated full-dataset reports are local-only and excluded from Git; "
            "run experiments/full_dataset.py to enable this integration check."
        )
    return load_dataset()


def test_pilot_reports_have_balanced_explicit_labels_and_grouped_repetitions():
    records = load_dataset(ROOT / "reports" / "experiments" / "pilot")

    assert len(records) == 45
    assert Counter(row["ground_truth_label"] for row in records) == {label: 15 for label in LABELS}
    assert len({row["circuit_family_id"] for row in records}) == 5
    assert len({row["experimental_group_id"] for row in records}) == 15
    assert len({row["run_id"] for row in records}) == 45
    repetitions = Counter(row["experimental_group_id"] for row in records)
    assert set(repetitions.values()) == {3}
    assert all(not row["duplicate_fingerprint"] for row in records)


def test_pilot_without_lineage_has_no_safe_split_and_separate_feature_sets():
    records = load_dataset(ROOT / "reports" / "experiments" / "pilot")
    feasible, explanation = _split_feasibility(records)

    assert not feasible
    assert "lack circuit_lineage_id" in explanation
    assert "gate_mismatch_fraction" not in OBSERVABLE_ONLY
    assert "gate_mismatch_fraction" in WITH_PROGRAM_COMPARISON


def test_full_dataset_has_balanced_groups_repetitions_and_family_coverage():
    records = _load_local_full_dataset_or_skip()

    assert len(records) == 270
    assert Counter(row["ground_truth_label"] for row in records) == {label: 90 for label in LABELS}
    assert len({row["circuit_family_id"] for row in records}) == 30
    assert len({row["experimental_group_id"] for row in records}) == 90
    assert len({row["run_id"] for row in records}) == 270
    assert Counter(row["experimental_group_id"] for row in records).most_common()[0][1] == 3
    assert all(not row["duplicate_fingerprint"] for row in records)
    for family in {row["circuit_family_id"] for row in records}:
        assert {row["ground_truth_label"] for row in records if row["circuit_family_id"] == family} == set(LABELS)


def test_historical_full_dataset_lacks_lineage_for_safe_split():
    records = _load_local_full_dataset_or_skip()
    feasible, explanation = _split_feasibility(records)

    assert not feasible
    assert "lack circuit_lineage_id" in explanation


def test_load_dataset_accepts_relative_report_directory(tmp_path, monkeypatch):
    # Exercise relative-path handling with a temporary fixture rather than
    # requiring the generated, Git-ignored 270-record research output.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ml_evaluation, "ROOT", tmp_path)
    report_dir = Path("reports/experiments/full_dataset")
    report_dir.mkdir(parents=True)
    report = {
        "status": "COMPLETED",
        "experiment_metadata": {
            "ground_truth_label": LABELS[0],
            "circuit_family_id": "fixture-family",
            "experimental_group_id": "fixture-group",
            "run_id": "fixture-run",
            "circuit_lineage_id": "fixture-lineage",
        },
        "reproducibility": {"source_sha256": "a" * 64},
        "observed_counts": {"0": 8},
        "configuration": {},
        "anomaly": {},
        "circuit": {},
    }
    (report_dir / "run.json").write_text(json.dumps(report), encoding="utf-8")

    records = load_dataset(report_dir)

    assert len(records) == 1
    assert records[0]["report_path"] == "reports/experiments/full_dataset/run.json"


def test_load_dataset_preserves_missing_run_provenance_for_strict_preflight(tmp_path):
    report = {
        "status": "COMPLETED",
        "experiment_metadata": {"ground_truth_label": "NORMAL"},
        "reproducibility": None,
        "observed_counts": {"0": 8},
        "configuration": {},
        "anomaly": {},
        "circuit": {},
    }
    (tmp_path / "unprovenanced.json").write_text(json.dumps(report), encoding="utf-8")

    rows = load_dataset(tmp_path)

    assert len(rows) == 1
    assert rows[0]["_source_sha256"] is None
