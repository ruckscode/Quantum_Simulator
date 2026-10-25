import json
import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

from ml_evaluation import LABELS, load_report_dispositions
import ml_training
from ml_training import (
    CALIBRATION,
    PRODUCTION_OBSERVABLES,
    WITH_PROGRAM_COMPARISON,
    _metric_summary,
    _prediction_rows,
    _preflight_summary,
    _validate_feature_sets,
    build_evaluation_outputs,
    estimator_configs,
    evaluation_dataset_fingerprint,
    family_cluster_bootstrap_intervals,
    feature_matrix,
    feature_sets,
    fit_predict_fold,
    fold_assignment_records,
    prepare_new_output_directory,
    models,
    OUT,
    validate_three_way_family_partitions,
)


def _rows():
    rows = []
    schemas = feature_sets()
    for family_number in range(3):
        for class_number, label in enumerate(LABELS):
            row = {
                "run_id": f"run-{family_number}-{class_number}",
                "circuit_family_id": f"family-{family_number}",
                "circuit_lineage_id": f"lineage-{family_number}",
                "experimental_group_id": f"group-{family_number}-{class_number}",
                "ground_truth_label": label,
                "program_fault_id": f"fault-{family_number}" if class_number == 1 else None,
                "hardware_condition_id": f"hardware-{family_number}" if class_number == 2 else None,
                "program_fault_description": "synthetic metadata",
                "program_fault_operation_index": family_number,
                "hardware_affected_component": "synthetic metadata",
            }
            row.update({name: float(family_number * 3 + class_number) for name in schemas["PRODUCTION_FIVE_OBSERVABLES"]})
            row.update({name: float(class_number) for name in schemas["OBSERVABLE_ONLY"] if name not in row})
            row["gate_mismatch_fraction"] = float(class_number == 1)
            rows.append(row)
    return rows


def test_feature_schemas_keep_production_observables_separate_from_calibration_and_mismatch():
    schemas = feature_sets()
    _validate_feature_sets(schemas)

    assert tuple(schemas["PRODUCTION_FIVE_OBSERVABLES"]) == PRODUCTION_OBSERVABLES
    assert len(PRODUCTION_OBSERVABLES) == 5
    assert not {"calibration_readout_error_max", "calibration_t1_min", "calibration_t2_min", "gate_mismatch_fraction"}.intersection(PRODUCTION_OBSERVABLES)
    assert "gate_mismatch_fraction" in schemas["WITH_PROGRAM_COMPARISON"]
    assert CALIBRATION.isdisjoint(schemas["PRODUCTION_FIVE_OBSERVABLES"])
    assert CALIBRATION.issubset(schemas["OBSERVABLE_ONLY"])
    assert set(LABELS).isdisjoint({feature.upper() for features in schemas.values() for feature in features})
    with pytest.raises(ValueError, match="restricted to the explicitly diagnostic"):
        _validate_feature_sets({
            **schemas,
            "UNREVIEWED_PRIMARY": [*PRODUCTION_OBSERVABLES, "gate_mismatch_fraction"],
        })


def test_feature_schema_roles_mark_program_comparison_as_diagnostic_only():
    schemas = feature_sets()
    rows = _rows()
    folds = [
        (np.arange(3, 9), np.arange(0, 3)),
        (np.concatenate((np.arange(0, 3), np.arange(6, 9))), np.arange(3, 6)),
        (np.arange(0, 6), np.arange(6, 9)),
    ]
    outputs = build_evaluation_outputs(
        rows=rows, folds=folds, model_results={}, source_dispositions=[],
        schemas=schemas, configurations={}, source_hashes={},
        dataset_fingerprint_sha256="a" * 64,
    )
    dispositions = outputs["manifest"]["feature_schema_dispositions"]
    assert dispositions["PRODUCTION_FIVE_OBSERVABLES"]["role"] == "primary_output_observable_only"
    assert dispositions["WITH_PROGRAM_COMPARISON"]["role"] == "diagnostic_leakage_ablation_not_primary_evaluation"
    assert "gate_mismatch_fraction can reveal the synthetic program-fault construction" in dispositions["WITH_PROGRAM_COMPARISON"]["limitations"]


def test_model_export_blocks_before_loading_or_fitting_ineligible_data(tmp_path, monkeypatch):
    import experiments.export_observable_model as exporter
    import ml_training

    monkeypatch.setattr(ml_training, "run_preflight", lambda: {"ready": False, "blockers": ["missing adjudicated lineage"]})
    monkeypatch.setattr(exporter, "load_dataset", lambda: (_ for _ in ()).throw(AssertionError("loaded before preflight")))
    destination = tmp_path / "must-not-exist.json"
    with pytest.raises(RuntimeError, match="missing adjudicated lineage"):
        exporter.export_model(destination)
    assert not destination.exists()


def test_feature_validation_rejects_label_fields_in_schema():
    with pytest.raises(ValueError, match="Forbidden metadata/label fields"):
        _validate_feature_sets({
            "PRODUCTION_FIVE_OBSERVABLES": [*PRODUCTION_OBSERVABLES, "ground_truth_label"],
            "WITH_PROGRAM_COMPARISON": [*WITH_PROGRAM_COMPARISON],
        })


def test_fold_assignments_record_family_and_exactly_once_outer_test_membership():
    rows = _rows()
    folds = [
        (np.arange(3, 9), np.arange(0, 3)),
        (np.concatenate((np.arange(0, 3), np.arange(6, 9))), np.arange(3, 6)),
        (np.arange(0, 6), np.arange(6, 9)),
    ]

    assignments = fold_assignment_records(rows, folds)

    assert [row["outer_fold"] for row in assignments] == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert [row["circuit_family_id"] for row in assignments] == [row["circuit_family_id"] for row in rows]
    assert all(row["inner_fold"] is None for row in assignments)


def test_fold_assignments_reject_duplicate_or_missing_oof_membership():
    rows = _rows()
    complement = (np.arange(3, 9), np.arange(0, 3))
    duplicate = [complement, complement]
    missing = [complement]

    with pytest.raises(ValueError, match="multiple outer test folds"):
        fold_assignment_records(rows, duplicate)
    with pytest.raises(ValueError, match="exactly one"):
        fold_assignment_records(rows, missing)


def test_oof_prediction_rows_preserve_identity_fold_probabilities_and_error_status():
    rows = _rows()
    folds = [
        (np.arange(3, 9), np.arange(0, 3)),
        (np.concatenate((np.arange(0, 3), np.arange(6, 9))), np.arange(3, 6)),
        (np.arange(0, 6), np.arange(6, 9)),
    ]
    assignments = fold_assignment_records(rows, folds)
    predictions = [row["ground_truth_label"] for row in rows]
    predictions[1] = "FAILED"
    probabilities = [{label: 1 / 3 for label in LABELS} for _ in rows]
    errors = [None, "ValueError: synthetic failure", *([None] * (len(rows) - 2))]

    output = _prediction_rows(rows, assignments, "PRODUCTION_FIVE_OBSERVABLES", "Logistic Regression", predictions, probabilities, errors)

    assert len(output) == len(rows)
    assert len({row["run_id"] for row in output}) == len(rows)
    assert output[1]["prediction_status"] == "FAILED"
    assert output[1]["prediction"] == "FAILED"
    assert output[1]["error"] == errors[1]
    assert output[0]["probability_NORMAL"] == pytest.approx(1 / 3)


def test_metrics_normalize_integrated_diagnosis_outcomes_and_keep_abstention_visible():
    summary = _metric_summary(
        ["NORMAL", "PROGRAM_FAULT", "HARDWARE_ANOMALY", "NORMAL"],
        ["NO_ANOMALY", "PROGRAM_FAULT", "AMBIGUOUS / INSUFFICIENT_EVIDENCE", "FAILED"],
    )

    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["coverage_count"] == 2
    assert summary["coverage"] == pytest.approx(0.5)
    assert summary["selective_accuracy"] == pytest.approx(1.0)
    assert summary["selective_error"] == pytest.approx(0.0)
    assert summary["abstention_count"] == 1
    assert summary["other_prediction_count"] == 0
    matrix = summary["confusion_matrix_true_rows_predicted_columns"]
    assert matrix["NORMAL"]["NORMAL"] == 1
    assert matrix["HARDWARE_ANOMALY"]["AMBIGUOUS"] == 1
    assert matrix["NORMAL"]["FAILED"] == 1


def test_metrics_reject_missing_predictions_and_all_resolved_has_full_coverage():
    with pytest.raises(ValueError, match="truth/prediction counts differ"):
        _metric_summary(["NORMAL", "PROGRAM_FAULT"], ["NORMAL"])

    summary = _metric_summary(["NORMAL", "PROGRAM_FAULT"], ["NO_ANOMALY", "NORMAL"])
    assert summary["coverage_count"] == 2
    assert summary["coverage"] == 1.0
    assert summary["selective_accuracy"] == 0.5
    assert summary["selective_error"] == 0.5


def test_lineage_bootstrap_resamples_complete_clusters(monkeypatch):
    rows = [
        {"circuit_lineage_id": "lineage-a"},
        {"circuit_lineage_id": "lineage-a"},
        {"circuit_lineage_id": "lineage-b"},
    ]

    class FixedClusterDraw:
        def choice(self, candidates, *, size, replace):
            assert list(candidates) == ["lineage-a", "lineage-b"]
            assert size == 2
            assert replace is True
            return np.asarray(["lineage-a", "lineage-b"])

    monkeypatch.setattr(ml_training.np.random, "default_rng", lambda _seed: FixedClusterDraw())
    result = family_cluster_bootstrap_intervals(
        rows,
        ["NORMAL", "PROGRAM_FAULT", "HARDWARE_ANOMALY"],
        ["NORMAL", "PROGRAM_FAULT", "AMBIGUOUS"],
        repetitions=2,
        seed=3,
    )

    assert result["independent_lineage_count"] == 2
    assert result["intervals_95"]["coverage"] == pytest.approx([2 / 3, 2 / 3])
    assert result["intervals_95"]["selective_error"] == [0.0, 0.0]


def test_bootstrap_withholds_selective_error_interval_when_replicates_have_no_resolved_rows():
    rows = _rows()
    truth = [row["ground_truth_label"] for row in rows]
    result = family_cluster_bootstrap_intervals(
        rows,
        truth,
        ["AMBIGUOUS"] * len(rows),
        repetitions=25,
        seed=11,
    )

    assert result["metric_replicates"]["coverage"] == 25
    assert result["intervals_95"]["coverage"] == [0.0, 0.0]
    assert result["metric_replicates"]["selective_error"] == 0
    assert "selective_error" not in result["intervals_95"]


def test_metrics_keep_abstention_failure_and_unsupported_outcomes_visible():
    summary = _metric_summary(
        [LABELS[0], LABELS[1], LABELS[2], LABELS[0]],
        [LABELS[0], "AMBIGUOUS", "UNSUPPORTED", "FAILED"],
    )

    assert summary["evaluated_case_count"] == 4
    assert summary["abstention_count"] == 1
    assert summary["failure_count"] == 1
    assert summary["unsupported_operation_count"] == 1
    assert summary["confusion_matrix_true_rows_predicted_columns"][LABELS[1]]["AMBIGUOUS"] == 1
    assert summary["confusion_matrix_true_rows_predicted_columns"][LABELS[2]]["UNSUPPORTED"] == 1
    assert summary["per_class"][LABELS[0]]["support"] == 2
    assert summary["coverage_count"] == 1
    assert summary["coverage"] == pytest.approx(0.25)
    assert summary["selective_accuracy"] == pytest.approx(1.0)
    assert summary["selective_error"] == pytest.approx(0.0)
    assert summary["nonclassification_count"] == 3


def test_metrics_report_undefined_selective_error_when_every_prediction_abstains():
    summary = _metric_summary([LABELS[0], LABELS[1]], ["AMBIGUOUS", "AMBIGUOUS"])

    assert summary["accuracy"] == 0.0
    assert summary["coverage_count"] == 0
    assert summary["coverage"] == 0.0
    assert summary["selective_accuracy"] is None
    assert summary["selective_error"] is None
    assert summary["abstention_count"] == 2


def test_metrics_count_out_of_schema_predictions_as_other():
    summary = _metric_summary([LABELS[0]], ["unexpected"])
    assert summary["other_prediction_count"] == 1
    assert summary["confusion_matrix_true_rows_predicted_columns"][LABELS[0]]["OTHER"] == 1


def test_source_dispositions_preserve_completed_failed_and_unsupported_cases(monkeypatch):
    payloads = {
        "completed.json": {"status": "COMPLETED", "experiment_metadata": {"run_id": "r1", "ground_truth_label": "NORMAL"}},
        "failed.json": {"status": "FAILED", "experiment_metadata": {"run_id": "r2"}, "errors": ["execution error"]},
        "unsupported.json": {"status": "FAILED", "experiment_metadata": {"run_id": "r3"}, "errors": ["Unsupported operation in circuit"]},
        "ineligible.json": {"status": "COMPLETED", "experiment_metadata": {"run_id": "r4"}},
    }
    paths = [Path(name) for name in payloads]
    monkeypatch.setattr(Path, "rglob", lambda self, pattern: iter(paths))
    monkeypatch.setattr(Path, "read_text", lambda self, encoding=None: json.dumps(payloads[self.name]))

    rows = load_report_dispositions(Path("synthetic-reports"))

    by_run = {row["run_id"]: row for row in rows}
    assert by_run["r1"]["disposition"] == "COMPLETED"
    assert by_run["r2"]["disposition"] == "FAILED"
    assert by_run["r3"]["disposition"] == "UNSUPPORTED"
    assert by_run["r4"]["disposition"] == "COMPLETED"
    assert by_run["r4"]["evaluation_eligibility"] == "INELIGIBLE"


class _SpyClassifier(ClassifierMixin, BaseEstimator):
    """Mock estimator records fit inputs and emits predetermined predictions."""

    def __init__(self, outputs):
        self.outputs = list(outputs)

    def fit(self, X, y):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.classes_ = np.asarray(LABELS)
        return self

    def predict(self, X):
        return np.asarray(self.outputs[:len(X)])

    def predict_proba(self, X):
        result = np.zeros((len(X), len(LABELS)))
        for index, value in enumerate(self.predict(X)):
            result[index, LABELS.index(value)] = 1.0
        return result


def _synthetic_folds():
    return [
        (np.arange(3, 9), np.arange(0, 3)),
        (np.concatenate((np.arange(0, 3), np.arange(6, 9))), np.arange(3, 6)),
        (np.arange(0, 6), np.arange(6, 9)),
    ]


def test_feature_matrix_ignores_label_family_and_scenario_metadata():
    rows = _rows()
    schema = feature_sets()["PRODUCTION_FIVE_OBSERVABLES"]
    before = feature_matrix(rows, schema)
    altered = copy.deepcopy(rows)
    for row in altered:
        row["ground_truth_label"] = "PROGRAM_FAULT"
        row["circuit_family_id"] = "same-family"
        row["program_fault_id"] = "changed-fault-id"
        row["hardware_condition_id"] = "changed-hardware-id"
        row["program_fault_operation_index"] = 999
        row["hardware_affected_component"] = "changed-component"

    np.testing.assert_array_equal(feature_matrix(altered, schema), before)
    with pytest.raises(ValueError, match="Forbidden metadata/label fields"):
        feature_matrix(rows, [*schema, "ground_truth_label"])


def test_feature_matrix_defaults_to_primary_and_requires_registered_ablation_name():
    rows = _rows()
    with pytest.raises(ValueError, match="does not match registered schema"):
        feature_matrix(rows, WITH_PROGRAM_COMPARISON)
    matrix = feature_matrix(rows, WITH_PROGRAM_COMPARISON, schema_name="WITH_PROGRAM_COMPARISON")
    assert matrix.shape == (len(rows), len(WITH_PROGRAM_COMPARISON))


def test_feature_matrix_defaults_to_primary_and_requires_registered_ablation_name():
    rows = _rows()
    with pytest.raises(ValueError, match="does not match registered schema"):
        feature_matrix(rows, WITH_PROGRAM_COMPARISON)
    matrix = feature_matrix(rows, WITH_PROGRAM_COMPARISON, schema_name="WITH_PROGRAM_COMPARISON")
    assert matrix.shape == (len(rows), len(WITH_PROGRAM_COMPARISON))


def test_three_way_partitions_require_complete_family_and_lineage_separation():
    rows = _rows()
    partitions = {
        "train": [0, 1, 2],
        "calibration": [3, 4, 5],
        "test": [6, 7, 8],
    }
    result = validate_three_way_family_partitions(rows, partitions)
    assert result["structurally_valid"] is True
    assert result["partitions"]["train"]["lineage_ids"] == ["lineage-0"]

    related = copy.deepcopy(rows)
    related[6]["circuit_lineage_id"] = related[0]["circuit_lineage_id"]
    with pytest.raises(ValueError, match="circuit_lineage_id leakage"):
        validate_three_way_family_partitions(related, partitions)
    with pytest.raises(ValueError, match="every row exactly once"):
        validate_three_way_family_partitions(rows, {**partitions, "test": [6, 7]})


def test_three_way_partitions_reject_duplicate_fingerprints_across_partitions():
    rows = _rows()
    rows[0]["circuit_fingerprint"] = "same-circuit"
    rows[3]["circuit_fingerprint"] = "same-circuit"
    with pytest.raises(ValueError, match="duplicate circuit fingerprint"):
        validate_three_way_family_partitions(rows, {
            "train": [0, 1, 2], "calibration": [3, 4, 5], "test": [6, 7, 8],
        })


def test_three_way_partitions_reject_noninteger_indices_and_invalid_labels():
    rows = _rows()
    partitions = {"train": [0, 1, 2], "calibration": [3, 4, 5], "test": [6, 7, 8]}
    with pytest.raises(ValueError, match="indices must be integers"):
        validate_three_way_family_partitions(rows, {**partitions, "train": [0, 1.5, 2]})
    rows[0]["ground_truth_label"] = "UNKNOWN"
    with pytest.raises(ValueError, match="unsupported ground_truth_label"):
        validate_three_way_family_partitions(rows, partitions)


def test_family_bootstrap_resamples_lineages_deterministically_and_withholds_unsupported_ci():
    rows = _rows()
    truth = [row["ground_truth_label"] for row in rows]
    prediction = list(truth)
    first = family_cluster_bootstrap_intervals(rows, truth, prediction, repetitions=200, seed=9)
    second = family_cluster_bootstrap_intervals(rows, truth, prediction, repetitions=200, seed=9)
    assert first == second
    assert first["available"] is True
    assert first["independent_lineage_count"] == 3
    assert first["metric_replicates"]["all_classes_supported"] == 200
    assert first["intervals_95"]["accuracy"] == [1.0, 1.0]

    sparse = [dict(row) for row in rows[:3]]
    sparse[0]["circuit_lineage_id"] = "lineage-a"
    sparse[1]["circuit_lineage_id"] = "lineage-b"
    sparse[2]["circuit_lineage_id"] = "lineage-c"
    result = family_cluster_bootstrap_intervals(sparse, truth[:3], prediction[:3], repetitions=200, seed=9)
    assert "macro_f1" not in result["intervals_95"]
    assert "balanced_accuracy" not in result["intervals_95"]
    assert "class-wise intervals withheld" in result["reason"]


def test_main_rejects_family_leaking_outer_fold_before_building_models(monkeypatch, tmp_path):
    base = _rows()
    rows = []
    for family_number in range(30):
        for replicate in range(3):
            for class_number, label in enumerate(LABELS):
                row = copy.deepcopy(base[class_number])
                row.update({
                    "run_id": f"run-{family_number}-{replicate}-{class_number}",
                    "circuit_family_id": f"family-{family_number}",
                    "circuit_lineage_id": f"lineage-{family_number}",
                    "ground_truth_label": label,
                })
                row.pop("circuit_fingerprint", None)
                rows.append(row)

    train = np.asarray([i for i in range(len(rows)) if i % 2 == 0])
    test = np.asarray([i for i in range(len(rows)) if i % 2 == 1])

    class LeakingSplitter:
        def __init__(self, **_kwargs):
            pass

        def split(self, *_args):
            return iter([(train, test)])

    monkeypatch.setattr(ml_training, "run_preflight", lambda: {"ready": True})
    monkeypatch.setattr(ml_training, "prepare_new_output_directory", lambda _path: None)
    monkeypatch.setattr(ml_training, "OUT", tmp_path / "new-results")
    monkeypatch.setattr(ml_training, "load_dataset", lambda: rows)
    monkeypatch.setattr(ml_training, "StratifiedGroupKFold", LeakingSplitter)
    monkeypatch.setattr(ml_training, "models", lambda: pytest.fail("models constructed before split validation"))

    with pytest.raises(ValueError, match="circuit family leakage"):
        ml_training.main()


def test_comparison_models_include_dummy_and_scaled_margin_baselines_with_bounded_workers():
    configs = estimator_configs()
    configured = models()
    assert set(configured) == set(configs)
    assert "DummyClassifier(strategy=most_frequent)" == configs["Majority baseline"]["estimator"]
    assert "DummyClassifier(strategy=stratified" in configs["Stratified dummy baseline"]["estimator"]
    assert configured["Linear SVM"][-1].get_params()["dual"] == "auto"
    assert configured["Random Forest"][-1].get_params()["n_jobs"] == 1


def test_pipeline_preprocessing_fits_training_partition_only_and_builds_reproducible_outputs():
    rows = _rows()
    schema_name = "PRODUCTION_FIVE_OBSERVABLES"
    schemas = feature_sets()
    X = feature_matrix(rows, schemas[schema_name])
    y = [row["ground_truth_label"] for row in rows]
    folds = _synthetic_folds()
    predictions = [None] * len(rows)
    probabilities = [{} for _ in rows]
    errors = [None] * len(rows)

    for fold_number, (train, test) in enumerate(folds, 1):
        outputs = [y[index] for index in test]
        classifier = _SpyClassifier(outputs)
        pipeline = Pipeline([("preprocess", StandardScaler()), ("classifier", classifier)])
        fold_predictions, fold_probabilities = fit_predict_fold(pipeline, X, y, train, test)
        np.testing.assert_allclose(pipeline.named_steps["preprocess"].mean_, X[train].mean(axis=0))
        if fold_number == 1:
            assert not np.allclose(pipeline.named_steps["preprocess"].mean_, X.mean(axis=0))
        assert len(classifier.fit_X) == len(train)
        np.testing.assert_array_equal(classifier.fit_y, np.asarray(y)[train])
        for offset, index in enumerate(test):
            predictions[index] = str(fold_predictions[offset])
            probabilities[index] = fold_probabilities[offset]

    dispositions = [
        {"disposition": "COMPLETED", "evaluation_eligibility": "ELIGIBLE"},
        {"disposition": "FAILED", "evaluation_eligibility": "INELIGIBLE"},
        {"disposition": "UNSUPPORTED", "evaluation_eligibility": "INELIGIBLE"},
        {"disposition": "COMPLETED", "evaluation_eligibility": "INELIGIBLE"},
    ]
    hashes = {"synthetic_source.py": "source-hash"}
    options = {
        "schemas": schemas,
        "configurations": estimator_configs(),
        "source_hashes": hashes,
        "dataset_fingerprint_sha256": "synthetic-dataset-hash",
        "split_seed": 123,
        "splitter_name": "SyntheticFamilyHoldout",
        "split_shuffle": False,
        "runtime": {"python": "synthetic", "numpy": "synthetic", "scikit_learn": "synthetic"},
    }
    model_results = {
        (schema_name, "Logistic Regression"): {
            "predictions": predictions,
            "probabilities": probabilities,
            "errors": errors,
        }
    }
    first = build_evaluation_outputs(rows, folds, model_results, dispositions, **options)
    second = build_evaluation_outputs(rows, folds, model_results, dispositions, **options)

    assert first == second
    manifest = first["manifest"]
    oof = first["predictions"]
    assert len(oof) == len(rows)
    assert len({record["run_id"] for record in oof}) == len(rows)
    assert manifest["out_of_fold_predictions"]["row_count"] == len(oof)
    assert manifest["evaluated_configurations"] == [{"feature_set": schema_name, "model": "Logistic Regression"}]
    assert manifest["splitter"] == {"class": "SyntheticFamilyHoldout", "n_splits": 3, "shuffle": False, "random_state": 123, "group_field": "circuit_lineage_id"}
    assert manifest["dataset"]["eligible_report_count"] == len(rows)
    comparison = first["comparison_rows"][0]
    assert comparison["dataset_fingerprint_sha256"] == manifest["dataset"]["fingerprint_sha256"]
    assert comparison["dataset_fingerprint_scheme"] == manifest["dataset"]["fingerprint_scheme"]
    assert comparison["estimator_configuration"] == json.dumps(
        manifest["estimators"][comparison["model"]], sort_keys=True, separators=(",", ":")
    )
    assert comparison["split_seed"] == manifest["split_seed"] == 123
    assert comparison["protocol_version"] == manifest["protocol_version"]
    assert comparison["feature_schema_id"] == comparison["feature_set"] == schema_name
    assert json.loads(comparison["feature_names"]) == manifest["feature_schemas"][schema_name]
    for prediction in first["predictions"]:
        assert prediction["dataset_fingerprint_sha256"] == comparison["dataset_fingerprint_sha256"]
        assert prediction["dataset_fingerprint_scheme"] == comparison["dataset_fingerprint_scheme"]
        assert prediction["estimator_configuration"] == comparison["estimator_configuration"]
        assert prediction["split_seed"] == comparison["split_seed"]
        assert prediction["protocol_version"] == comparison["protocol_version"]
        assert prediction["feature_schema_id"] == comparison["feature_schema_id"]
    assert all(
        assignment["dataset_fingerprint_sha256"] == manifest["dataset"]["fingerprint_sha256"]
        and assignment["dataset_fingerprint_scheme"] == manifest["dataset"]["fingerprint_scheme"]
        and assignment["split_seed"] == manifest["split_seed"]
        and assignment["protocol_version"] == manifest["protocol_version"]
        for assignment in first["fold_assignments"]
    )
    for fold in manifest["outer_fold_membership"]:
        assert not set(fold["training_families"]) & set(fold["test_families"])
        assert not set(fold["training_lineages"]) & set(fold["test_lineages"])
        assert set(fold["training_run_ids"]) | set(fold["test_run_ids"]) == {row["run_id"] for row in rows}
    for prediction in oof:
        assert prediction["protocol_version"] == manifest["protocol_version"]
        assert prediction["split_seed"] == manifest["split_seed"] == 123
        assert json.loads(prediction["feature_schema"]) == manifest["feature_schemas"][schema_name]
        assert json.loads(prediction["estimator_configuration"]) == manifest["estimators"][prediction["model"]]
        assert prediction["dataset_fingerprint_sha256"] == manifest["dataset"]["fingerprint_sha256"]
        assert prediction["dataset_fingerprint_scheme"] == manifest["dataset"]["fingerprint_scheme"]
        assert prediction["feature_schema_id"] == schema_name
        assert prediction["source_hash_sha256"] == hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    metrics = first["metrics"][f"{schema_name}::Logistic Regression"]
    matrix = metrics["confusion_matrix_true_rows_predicted_columns"]
    total = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[label][label] for label in LABELS)
    assert total == metrics["evaluated_case_count"] == len(oof)
    assert metrics["accuracy"] == pytest.approx(correct / total)
    f1_values, recall_values = [], []
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(count for predicted, count in matrix[label].items() if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        assert metrics["per_class"][label]["precision"] == pytest.approx(precision)
        assert metrics["per_class"][label]["recall"] == pytest.approx(recall)
        assert metrics["per_class"][label]["f1"] == pytest.approx(f1)
        f1_values.append(f1)
        recall_values.append(recall)
    assert metrics["macro_f1"] == pytest.approx(sum(f1_values) / len(LABELS))
    assert metrics["balanced_accuracy"] == pytest.approx(sum(recall_values) / len(LABELS))
    assert metrics["source_dispositions"] == {"COMPLETED": 2, "FAILED": 1, "UNSUPPORTED": 1}
    assert metrics["source_eligibility"] == {"ELIGIBLE": 1, "INELIGIBLE": 3}


def test_default_training_outputs_are_outside_historical_reports():
    assert OUT.parent.name == "ml_results"
    assert OUT.name == "grouped_cv_audit_v3"
    assert "reports" not in OUT.parts


def test_evaluation_output_directory_refuses_to_overwrite_existing_evidence(tmp_path):
    output = tmp_path / "comparison"
    output.mkdir()
    artifact = output / "evaluation_manifest.json"
    artifact.write_text("historical evidence", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_new_output_directory(output)
    assert artifact.read_text(encoding="utf-8") == "historical evidence"


def test_fingerprint_used_by_preflight_and_evaluation_is_identical(tmp_path):
    source = tmp_path / "reports"
    source.mkdir()
    output = tmp_path / "ml_results" / "grouped_cv_audit_v1"
    output.mkdir(parents=True)
    (source / "dataset_manifest.json").write_text('{"source":"synthetic"}', encoding="utf-8")
    (source / "run-b.json").write_text('{"status":"COMPLETED"}', encoding="utf-8")
    (source / "run-a.json").write_text('{"status":"COMPLETED"}', encoding="utf-8")
    fingerprint = evaluation_dataset_fingerprint(source, output_dir=output)
    folds = _synthetic_folds()
    rows = _rows()
    y = [row["ground_truth_label"] for row in rows]
    predictions = list(y)
    probabilities = [{label: 1 / 3 for label in LABELS} for _ in rows]
    errors = [None] * len(rows)
    built = build_evaluation_outputs(
        rows,
        folds,
        {("PRODUCTION_FIVE_OBSERVABLES", "Logistic Regression"): {
            "predictions": predictions,
            "probabilities": probabilities,
            "errors": errors,
        }},
        [],
        dataset_fingerprint_sha256=fingerprint,
    )

    preflight_fingerprint = evaluation_dataset_fingerprint(source, output_dir=output)
    assert preflight_fingerprint == fingerprint
    preflight_payload = _preflight_summary(
        rows,
        [],
        folds,
        dataset_fingerprint_sha256=preflight_fingerprint,
        source_hashes={"synthetic": "hash"},
        output_dir=output,
    )
    assert preflight_payload["dataset_fingerprint_sha256"] == built["manifest"]["dataset"]["fingerprint_sha256"]
    assert preflight_payload["dataset_fingerprint_scheme"] == built["manifest"]["dataset"]["fingerprint_scheme"]
    assert built["manifest"]["dataset"]["fingerprint_sha256"] == preflight_fingerprint
    assert all(row["dataset_fingerprint_sha256"] == preflight_fingerprint for row in built["fold_assignments"])
    assert all(row["dataset_fingerprint_sha256"] == preflight_fingerprint for row in built["predictions"])
    assert built["comparison_rows"][0]["dataset_fingerprint_sha256"] == preflight_fingerprint


def test_fold_assignment_rejects_missing_lineage_and_cross_fold_related_families():
    rows = _rows()
    folds = _synthetic_folds()
    rows[0].pop("circuit_lineage_id")
    with pytest.raises(ValueError, match="nonempty circuit_lineage_id"):
        fold_assignment_records(rows, folds)

    rows = _rows()
    for row in rows:
        if row["circuit_family_id"] == "family-1":
            row["circuit_lineage_id"] = "lineage-0"
    with pytest.raises(ValueError, match="circuit lineage leakage"):
        fold_assignment_records(rows, folds)


def test_dataset_fingerprint_is_order_independent_sensitive_and_excludes_generated_output(tmp_path, monkeypatch):
    from ml_evaluation import dataset_fingerprint

    left = tmp_path / "left"
    right = tmp_path / "right"
    for directory in (left, right):
        (directory / "outputs").mkdir(parents=True)
    (left / "b.json").write_text('{"v":2}', encoding="utf-8")
    (left / "a.json").write_text('{"v":1}', encoding="utf-8")
    (right / "a.json").write_text('{"v":1}', encoding="utf-8")
    (right / "b.json").write_text('{"v":2}', encoding="utf-8")
    (left / "outputs" / "evaluation_manifest.json").write_text('{"generated":1}', encoding="utf-8")
    (right / "outputs" / "evaluation_manifest.json").write_text('{"generated":999}', encoding="utf-8")

    original_rglob = Path.rglob

    def reversed_left_traversal(path, pattern):
        paths = list(original_rglob(path, pattern))
        return iter(reversed(paths)) if path.resolve() == left.resolve() else iter(paths)

    monkeypatch.setattr(Path, "rglob", reversed_left_traversal)
    left_fingerprint = dataset_fingerprint(left, excluded_dirs=(left / "outputs",))
    right_fingerprint = dataset_fingerprint(right, excluded_dirs=(right / "outputs",))
    assert left_fingerprint == right_fingerprint
    (right / "a.json").write_text('{"v":changed}', encoding="utf-8")
    assert dataset_fingerprint(right, excluded_dirs=(right / "outputs",)) != left_fingerprint


def test_output_construction_does_not_mutate_existing_artifact_files(tmp_path):
    historical = tmp_path / "historical"
    historical.mkdir()
    manifest_path = historical / "evaluation_manifest.json"
    comparison_path = historical / "model_comparison.csv"
    manifest_path.write_bytes(b"historical manifest bytes")
    comparison_path.write_bytes(b"historical comparison bytes")
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in historical.iterdir()}

    rows = _rows()
    model_results = {("PRODUCTION_FIVE_OBSERVABLES", "Logistic Regression"): {
        "predictions": [row["ground_truth_label"] for row in rows],
        "probabilities": [{label: 1 / 3 for label in LABELS} for _ in rows],
        "errors": [None] * len(rows),
    }}
    build_evaluation_outputs(
        rows, _synthetic_folds(), model_results, [],
        dataset_fingerprint_sha256="synthetic-fingerprint",
    )

    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in historical.iterdir()}
    assert after == before


def test_v3_destination_preserves_v1_v2_and_excludes_generated_outputs_from_fingerprint(tmp_path):
    from ml_evaluation import dataset_fingerprint

    assert OUT == Path(__file__).resolve().parents[1] / "ml_results" / "grouped_cv_audit_v3"
    source = tmp_path / "source_reports"
    source.mkdir()
    (source / "run-1.json").write_bytes(b'{"status":"COMPLETED","source":1}')
    historical_v1 = tmp_path / "ml_results" / "grouped_cv_audit_v1"
    selected_v3 = source / "generated" / "ml_results" / "grouped_cv_audit_v3"
    historical_v1.mkdir(parents=True)
    historical_artifact = historical_v1 / "evaluation_manifest.json"
    historical_artifact.write_bytes(b"preserve v1 evaluation")
    historical_before = hashlib.sha256(historical_artifact.read_bytes()).hexdigest()

    historical_v2 = tmp_path / "ml_results" / "grouped_cv_audit_v2"
    historical_v2.mkdir()
    v2_artifact = historical_v2 / "evaluation_manifest.json"
    v2_artifact.write_bytes(b"preserve v2 evidence")
    v2_before = hashlib.sha256(v2_artifact.read_bytes()).hexdigest()
    fingerprint_before = evaluation_dataset_fingerprint(source, output_dir=selected_v3)
    selected_v3.mkdir(parents=True)
    (selected_v3 / "evaluation_manifest.json").write_bytes(b'{"generated":true}')
    (selected_v3 / "nested").mkdir()
    (selected_v3 / "nested" / "generated.json").write_bytes(b'{"generated":true}')
    fingerprint_after = dataset_fingerprint(source, excluded_dirs=(selected_v3,))

    assert fingerprint_after == fingerprint_before
    assert hashlib.sha256(historical_artifact.read_bytes()).hexdigest() == historical_before
    assert historical_artifact.read_bytes() == b"preserve v1 evaluation"
    assert hashlib.sha256(v2_artifact.read_bytes()).hexdigest() == v2_before
    assert v2_artifact.read_bytes() == b"preserve v2 evidence"


def test_every_comparison_row_carries_matching_manifest_and_oof_provenance():
    rows = _rows()
    model_results = {
        ("PRODUCTION_FIVE_OBSERVABLES", model): {
            "predictions": [row["ground_truth_label"] for row in rows],
            "probabilities": [{label: 1 / 3 for label in LABELS} for _ in rows],
            "errors": [None] * len(rows),
        }
        for model in ("Logistic Regression", "Decision Tree")
    }
    outputs = build_evaluation_outputs(
        rows,
        _synthetic_folds(),
        model_results,
        [],
        dataset_fingerprint_sha256="synthetic-shared-fingerprint",
        split_seed=77,
    )
    manifest = outputs["manifest"]
    assert manifest["evidence_scope"] == "SIMULATOR_SYNTHETIC_ONLY"
    assert manifest["independent_generalization_claim_eligible"] is False
    assert manifest["real_hardware_validation"] is False
    assert manifest["calibration"]["partition"] is None
    assert manifest["calibration"]["performed"] is False
    assert manifest["calibration"]["metrics_emitted"] is False
    assert "not calibrated probabilities" in manifest["calibration"]["probability_semantics"]
    interval = outputs["metrics"]["PRODUCTION_FIVE_OBSERVABLES::Logistic Regression"]["family_cluster_bootstrap"]
    assert interval["independent_lineage_count"] == 3
    assert "accuracy" in interval["intervals_95"]

    # Coverage is always defined; selective error is withheld if any bootstrap
    # replicate contains no resolved prediction.
    selective_predictions = ["AMBIGUOUS"] * len(rows)
    selective_predictions[:3] = [row["ground_truth_label"] for row in rows[:3]]
    selective_interval = family_cluster_bootstrap_intervals(
        rows,
        [row["ground_truth_label"] for row in rows],
        selective_predictions,
        repetitions=100,
        seed=77,
    )
    assert "coverage" in selective_interval["intervals_95"]
    assert selective_interval["metric_replicates"]["coverage"] == 100
    if selective_interval["metric_replicates"]["selective_error"] < 100:
        assert "selective_error" not in selective_interval["intervals_95"]
    predictions_by_model = {
        model: [row for row in outputs["predictions"] if row["model"] == model]
        for model in ("Logistic Regression", "Decision Tree")
    }

    assert len(outputs["comparison_rows"]) == len(model_results)
    for comparison in outputs["comparison_rows"]:
        model = comparison["model"]
        assert comparison["dataset_fingerprint_sha256"] == manifest["dataset"]["fingerprint_sha256"]
        assert comparison["dataset_fingerprint_scheme"] == manifest["dataset"]["fingerprint_scheme"]
        assert json.loads(comparison["estimator_configuration"]) == manifest["estimators"][model]
        assert comparison["split_seed"] == manifest["split_seed"] == 77
        assert comparison["protocol_version"] == manifest["protocol_version"]
        assert comparison["feature_schema_id"] in manifest["feature_schemas"]
        assert json.loads(comparison["feature_names"]) == manifest["feature_schemas"][comparison["feature_schema_id"]]
        for prediction in predictions_by_model[model]:
            assert comparison["dataset_fingerprint_sha256"] == prediction["dataset_fingerprint_sha256"]
            assert comparison["estimator_configuration"] == prediction["estimator_configuration"]
            assert comparison["split_seed"] == prediction["split_seed"]
            assert comparison["protocol_version"] == prediction["protocol_version"]
            assert comparison["feature_schema_id"] == prediction["feature_schema_id"]
