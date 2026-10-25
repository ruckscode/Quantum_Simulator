import json

import pytest

from experiments.ml_provenance_recovery import build_record_recovery_table, write_report


def _fixture_dir(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "dataset_manifest.json").write_text(json.dumps({
        "records_written": 1,
        "families": [{"circuit_family_id": "family-1", "program_fault_id": "fault-1"}],
    }), encoding="utf-8")
    (root / "run-one.json").write_text(json.dumps({
        "status": "COMPLETED",
        "configuration": {
            "run_id": "run-one", "circuit_id": "circuit-one",
            "circuit_family_id": "family-1", "experimental_group_id": "group-one",
            "ground_truth_label": "PROGRAM_FAULT", "random_seed": 71,
            "shots": 64, "repetition_index": 1,
            "circuit": [["h", 0]], "observed_circuit": [["x", 0]],
            "hardware_parameters": {}, "noise_parameters": None,
        },
        "execution_metadata": {
            "execution_kind": "SYNTHETIC_VALIDATION", "execution_target": "LOCAL_SIMULATOR",
            "random_seed": 71, "shots": 64, "python_version": "3.12.10", "numpy_version": "2.5.3",
        },
        "experiment_metadata": {"validation_label": "SYNTHETIC_VALIDATION"},
        "reproducibility": {"fingerprint": "f" * 64, "simulator_version": "unversioned-current"},
    }), encoding="utf-8")
    return root


def test_recovery_table_keeps_saved_fields_but_does_not_infer_source_or_lineage(tmp_path):
    report = build_record_recovery_table(_fixture_dir(tmp_path))
    assert report["summary"]["record_count"] == 1
    row = report["records"][0]
    assert row["stable_artifact_id"] == "run-one"
    assert row["random_seed"] == 71
    assert row["configuration_sha256_recomputed_from_saved_json"]
    assert row["report_sha256_current_artifact_integrity_only"]
    assert row["source_sha256_recorded"] is None
    assert row["family_recipe_root"] is None
    assert row["missing_or_unproven"]["exact_generation_provenance"] is True
    assert row["available_field_status"]["random_seed"] == "RECORDED_IN_SAVED_REPORT"
    assert row["available_field_status"]["label_origin"] == "INFERRED_FROM_RECORDED_EXECUTION_AND_VALIDATION_METADATA"
    assert row["available_field_status"]["per_run_source_sha256"] == "MISSING"
    assert row["available_field_status"]["recipe_root_or_ancestry"] == "MISSING"
    assert row["eligibility"]["primary_training"] == "INELIGIBLE"
    assert "synthetic" in row["label_origin"]


def test_recovery_table_fails_closed_on_oversized_or_manifest_mismatched_input(tmp_path):
    root = _fixture_dir(tmp_path)
    with pytest.raises(ValueError, match="exceed cap"):
        build_record_recovery_table(root, max_total_bytes=10)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest["records_written"] = 2
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="record count disagrees"):
        build_record_recovery_table(root)


def test_recovery_report_writer_never_overwrites_prior_evidence(tmp_path):
    destination = tmp_path / "recovery.json"
    destination.write_text("historical", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_report({"new": "evidence"}, destination)
    assert destination.read_text(encoding="utf-8") == "historical"
