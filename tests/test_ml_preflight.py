from pathlib import Path
import platform

import numpy as np
import pytest

from ml_evaluation import LABELS, REQUIRED_METADATA
from ml_training import (
    OUT,
    PRODUCTION_OBSERVABLES,
    _preflight_summary,
    main,
)


def _fixture():
    rows = []
    generation_hashes = {"generator.py": "a" * 64}
    for family_index in range(5):
        for class_index, label in enumerate(LABELS):
            row = {
                "run_id": f"run-{family_index}-{class_index}",
                "circuit_id": f"circuit-{family_index}",
                "circuit_family_id": f"family-{family_index}",
                "circuit_lineage_id": f"lineage-{family_index}",
                "experimental_group_id": f"group-{family_index}-{class_index}",
                "ground_truth_label": label,
                "program_fault_id": f"fault-{family_index}" if label == "PROGRAM_FAULT" else None,
                "program_fault_description": "known injected operation" if label == "PROGRAM_FAULT" else None,
                "program_fault_operation_index": 0 if label == "PROGRAM_FAULT" else None,
                "program_fault_qubits": [0] if label == "PROGRAM_FAULT" else None,
                "hardware_condition_id": f"hw-{family_index}" if label == "HARDWARE_ANOMALY" else None,
                "hardware_affected_component": "readout q0" if label == "HARDWARE_ANOMALY" else None,
                "repetition_index": 0,
                "_metadata_fields_present": sorted(REQUIRED_METADATA),
                "validation_label": "SYNTHETIC_VALIDATION",
                "execution_kind": "SYNTHETIC_VALIDATION",
                "observation_source": "SEEDED_LOCAL_SIMULATOR_SHOTS_WITH_HARDWARE_MODEL",
                "hardware_model_applied": True,
                "report_fingerprint": f"hash-{family_index}-{class_index}",
                "_source_sha256": generation_hashes,
            }
            row.update({feature: float(family_index + class_index) for feature in PRODUCTION_OBSERVABLES})
            rows.append(row)
    folds = []
    for held_out in range(5):
        test = np.asarray([i for i, row in enumerate(rows) if row["circuit_family_id"] == f"family-{held_out}"])
        train = np.asarray([i for i in range(len(rows)) if i not in set(test)])
        folds.append((train, test))
    provenance = {
        "generation_provenance": {
            "generator": "generator.py",
            "source_sha256": generation_hashes,
            "runtime": {"python_version": platform.python_version(), "numpy_version": np.__version__},
        }
    }
    return rows, folds, provenance, generation_hashes


def test_preflight_checks_schemas_scenarios_family_folds_and_separation():
    rows, folds, provenance, generation_hashes = _fixture()
    dispositions = [
        {"disposition": "COMPLETED", "evaluation_eligibility": "ELIGIBLE"}
        for _ in rows
    ]
    result = _preflight_summary(
        rows, dispositions, folds,
        dataset_fingerprint_sha256="dataset-hash",
        source_hashes={"source.py": "source-hash"},
        dataset_provenance=provenance,
        expected_generation_source_hashes=generation_hashes,
        output_dir=OUT,
    )

    assert result["ready"] is True
    assert result["counts"] == {
        "eligible": 15, "completed": 15, "failed": 0, "unsupported": 0,
        "ineligible": 0, "other_source_dispositions": {},
    }
    assert result["feature_schema"] == list(PRODUCTION_OBSERVABLES)
    assert result["class_order"] == list(LABELS)
    assert result["family_count"] == 5
    assert len(result["folds"]) == 5
    assert all(fold["family_overlap"] == [] for fold in result["folds"])
    assert all(fold["lineage_overlap"] == [] for fold in result["folds"])
    assert result["dataset_fingerprint_sha256"] == "dataset-hash"
    assert result["source_hashes"] == {"source.py": "source-hash"}
    assert Path(result["proposed_output_directory"]) == OUT.resolve()
    assert result["output_is_outside_historical_reports"] is True
    assert result["writes_performed"] is False
    assert result["estimator_fitting_started"] is False
    assert result["prediction_started"] is False
    assert result["model_export_started"] is False


def test_fold_validator_rejects_related_families_split_across_folds():
    from ml_training import fold_assignment_records

    rows, folds, _, _ = _fixture()
    # Two different circuit_family_id values share a design lineage.
    for row in rows:
        if row["circuit_family_id"] == "family-1":
            row["circuit_lineage_id"] = "lineage-0"

    with pytest.raises(ValueError, match="circuit lineage leakage"):
        fold_assignment_records(rows, folds)


def test_lineage_identifier_is_forbidden_as_a_predictor():
    from ml_training import feature_matrix

    rows, _, _, _ = _fixture()
    with pytest.raises(ValueError, match="circuit_lineage_id"):
        feature_matrix(rows, ["circuit_lineage_id"])


def test_preflight_rejects_missing_metadata_bad_values_and_scenario_conflicts():
    rows, folds, provenance, generation_hashes = _fixture()
    rows[0]["circuit_family_id"] = None
    rows[1]["total_variation_distance"] = float("inf")
    rows[6]["hardware_condition_id"] = "incorrect-normal-condition"
    rows[3]["_metadata_fields_present"] = [field for field in REQUIRED_METADATA if field != "run_id"]
    dispositions = [
        {"disposition": "COMPLETED", "evaluation_eligibility": "ELIGIBLE"}
        for _ in rows
    ]
    result = _preflight_summary(
        rows, dispositions, folds,
        dataset_fingerprint_sha256="dataset-hash",
        source_hashes={},
        dataset_provenance=provenance,
        expected_generation_source_hashes=generation_hashes,
    )

    assert result["ready"] is False
    assert result["counts"]["ineligible"] == 4
    assert any("circuit_family_id" in reason for reason in result["ineligible_row_reasons"])
    assert any("total_variation_distance" in reason for reason in result["ineligible_row_reasons"])
    assert any("NORMAL row" in reason for reason in result["ineligible_row_reasons"])
    assert any("run_id" in reason for reason in result["ineligible_row_reasons"])
    assert result["folds"] == []


def test_preflight_counts_failed_unsupported_and_ineligible_separately():
    rows, folds, provenance, generation_hashes = _fixture()
    dispositions = [
        {"disposition": "COMPLETED", "evaluation_eligibility": "ELIGIBLE"},
        {"disposition": "FAILED", "evaluation_eligibility": "INELIGIBLE"},
        {"disposition": "UNSUPPORTED", "evaluation_eligibility": "INELIGIBLE"},
        {"disposition": "PARSING_FAILURE", "evaluation_eligibility": "INELIGIBLE"},
    ]
    result = _preflight_summary(
        rows, dispositions, folds,
        dataset_fingerprint_sha256="dataset-hash",
        source_hashes={},
        dataset_provenance=provenance,
        expected_generation_source_hashes=generation_hashes,
    )

    assert result["counts"]["failed"] == 1
    assert result["counts"]["unsupported"] == 1
    assert result["counts"]["ineligible"] == 1


def test_preflight_blocks_missing_or_mismatched_generation_provenance():
    rows, folds, provenance, generation_hashes = _fixture()
    dispositions = [
        {"disposition": "COMPLETED", "evaluation_eligibility": "ELIGIBLE"}
        for _ in rows
    ]

    missing = _preflight_summary(
        rows, dispositions, folds,
        dataset_fingerprint_sha256="dataset-hash",
        source_hashes={},
        dataset_provenance=None,
        expected_generation_source_hashes=generation_hashes,
    )
    assert missing["ready"] is False
    assert "dataset manifest is missing generation_provenance" in missing["dataset_provenance_blockers"]

    rows[0]["_source_sha256"] = {"generator.py": "b" * 64}
    mismatched = _preflight_summary(
        rows, dispositions, folds,
        dataset_fingerprint_sha256="dataset-hash",
        source_hashes={},
        dataset_provenance=provenance,
        expected_generation_source_hashes=generation_hashes,
    )
    assert mismatched["ready"] is False
    assert any("do not match the dataset generation source hashes" in reason for reason in mismatched["dataset_provenance_blockers"])

    rows, folds, provenance, generation_hashes = _fixture()
    stale_provenance = {
        "generation_provenance": {
            **provenance["generation_provenance"],
            "source_sha256": {"generator.py": "c" * 64},
        }
    }
    stale_code = _preflight_summary(
        rows, dispositions, folds,
        dataset_fingerprint_sha256="dataset-hash",
        source_hashes={},
        dataset_provenance=stale_provenance,
        expected_generation_source_hashes=generation_hashes,
    )
    assert stale_code["ready"] is False
    assert any("do not match current generation code" in reason for reason in stale_code["dataset_provenance_blockers"])

    old_runtime = {
        "generation_provenance": {
            **provenance["generation_provenance"],
            "runtime": {"python_version": "0.0.0", "numpy_version": "0.0.0"},
        }
    }
    runtime_mismatch = _preflight_summary(
        rows, dispositions, folds,
        dataset_fingerprint_sha256="dataset-hash",
        source_hashes={},
        dataset_provenance=old_runtime,
        expected_generation_source_hashes=generation_hashes,
    )
    assert runtime_mismatch["ready"] is False
    assert any("runtime does not exactly match" in reason for reason in runtime_mismatch["dataset_provenance_blockers"])


def test_training_entrypoint_stops_before_loading_or_fitting_when_preflight_fails(monkeypatch):
    import ml_training

    monkeypatch.setattr(ml_training, "run_preflight", lambda: {"ready": False, "blockers": ["source provenance mismatch"]})
    monkeypatch.setattr(ml_training, "load_dataset", lambda *args, **kwargs: pytest.fail("must block before loading training rows"))

    assert main(preflight_only=False) == 2

