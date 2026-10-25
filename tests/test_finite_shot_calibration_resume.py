"""Focused recovery checks for the long-running finite-shot calibration."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import finite_shot_calibration_plan as calibration


def _payload(row):
    probabilities = {format(i, "04b"): (1.0 if i == 0 else 0.0) for i in range(16)}
    result = {
        "split": row["split"], "template_id": row["template_id"],
        "template_seed": row["template_seed"],
        "reference_fingerprint": "test-fingerprint",
        "configuration_id": row["configuration_id"], "scenario": row["scenario"],
        "fault_strength_stratum": row["fault_strength_stratum"],
        "shot_count": row["shot_count"], "sampling_seed": row["seed"],
        "expected_label": row["expected_label"], "intervention": {"kind": "test"},
        "generation_metadata": {}, "current_detected": False,
        "permutation_p": 1.0, "permutation_detected": False, "abstained": False,
        "diagnosis_category": "NO_ANOMALY", "tvd": 0.0,
        "finite_shot_guard": 0.0, "effective_threshold": 0.05,
        "expected_distribution": probabilities, "observed_distribution": probabilities,
        "status": "completed",
    }
    return {"status": "completed", "configuration_id": row["configuration_id"], "result": result}


def _write_rows(root: Path, rows):
    for row in rows:
        path = root / Path(row["output_location"])
        path.parent.mkdir(parents=True, exist_ok=True)
        calibration._atomic_create_json(path, _payload(row))


def test_stopped_state_reconciles_326_complete_and_3274_remaining(tmp_path):
    rows = calibration.build_ledger()
    _write_rows(tmp_path, rows[:326])
    state = calibration.inspect_resume_state(repository_root=tmp_path)
    assert state["planned_configurations"] == 3600
    assert state["valid_configurations"] == 326
    assert state["missing_configurations"] == 3274
    assert state["invalid_configurations"] == 0


def test_resume_preflight_skips_valid_and_identifies_missing_rows(tmp_path):
    rows = calibration.build_ledger()
    _write_rows(tmp_path, rows[:2])
    preflight = calibration.resume_preflight(repository_root=tmp_path)
    assert preflight["ready"] is True
    assert preflight["valid_configurations"] == 2
    assert preflight["remaining_configurations"] == 3598
    assert {row["configuration_id"] for row in preflight["state"]["missing"]}.isdisjoint(
        {rows[0]["configuration_id"], rows[1]["configuration_id"]}
    )


def test_incomplete_file_is_invalid_and_preserved_for_recovery(tmp_path):
    # Keep the test-owned output tree short enough for Windows MAX_PATH when
    # pytest adds a fresh basetemp and per-test directory to the full path.
    spec = calibration.load_spec()
    spec["proposed_next_calibration_plan"]["design"]["calibration_output_directory"] = "calibration"
    row = calibration.build_ledger(spec)[0]
    path = tmp_path / Path(row["output_location"])
    path.parent.mkdir(parents=True)
    path.write_text("{truncated", encoding="utf-8")
    original = path.read_bytes()
    preflight = calibration.resume_preflight(spec=spec, repository_root=tmp_path)
    assert preflight["ready"] is True
    assert preflight["invalid_configurations"] == 1
    assert preflight["remaining_configurations"] == 3600
    assert path.read_bytes() == original
    recovered = path.with_name(f"{row['configuration_id']}.recovered.json")
    calibration._atomic_create_json(recovered, _payload(row))
    state = calibration.inspect_resume_state(spec=spec, repository_root=tmp_path)
    assert state["valid_configurations"] == 1
    assert state["invalid_configurations"] == 0
    assert path.read_bytes() == original


def test_create_only_writer_uses_short_staging_name(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "recovered.json"
    real_mkstemp = calibration.tempfile.mkstemp
    prefixes = []

    def record_prefix(**kwargs):
        prefixes.append(kwargs["prefix"])
        return real_mkstemp(**kwargs)

    monkeypatch.setattr(calibration.tempfile, "mkstemp", record_prefix)
    calibration._atomic_create_json(target, {"status": "recovered"})
    assert prefixes == [".q."]
    assert __import__("json").loads(target.read_text(encoding="utf-8")) == {"status": "recovered"}


def test_create_only_writer_never_overwrites_existing_output(tmp_path):
    target = tmp_path / "existing.json"
    target.write_bytes(b'{"preserve": true}')
    with pytest.raises(FileExistsError):
        calibration._atomic_create_json(target, {"replace": True})
    assert target.read_bytes() == b'{"preserve": true}'


def test_manifest_writer_retries_transient_windows_replace_lock(tmp_path, monkeypatch):
    target = tmp_path / "run_manifest.json"
    target.write_text('{"status":"running"}', encoding="utf-8")
    real_replace = calibration.os.replace
    attempts = 0

    def transient_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("temporary sync lock")
        return real_replace(source, destination)

    monkeypatch.setattr(calibration.os, "replace", transient_replace)
    calibration._atomic_replace_owned_json(target, {"status": "resuming"})

    assert attempts == 3
    assert __import__("json").loads(target.read_text(encoding="utf-8")) == {"status": "resuming"}


def test_ledger_is_reproducible_with_identical_configuration_ids():
    first = calibration.build_ledger()
    second = calibration.build_ledger()
    assert first == second
    assert len(first) == len({row["configuration_id"] for row in first}) == 3600


def test_resume_preflight_blocks_existing_summary_without_touching_results(tmp_path):
    rows = calibration.build_ledger()
    _write_rows(tmp_path, rows[:1])
    output_root = tmp_path / "reports/experiments/finite_shot_distribution_calibration_run"
    output_root.mkdir(parents=True, exist_ok=True)
    summary = output_root / "summary.json"
    summary.write_text('{"status":"completed"}', encoding="utf-8")
    result_path = tmp_path / Path(rows[0]["output_location"])
    original = result_path.read_bytes()

    preflight = calibration.resume_preflight(repository_root=tmp_path)

    assert preflight["ready"] is False
    assert any("final summary already exists" in blocker for blocker in preflight["blockers"])
    assert result_path.read_bytes() == original
    assert summary.read_text(encoding="utf-8") == '{"status":"completed"}'

