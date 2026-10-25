"""Family-grouped, fixed-configuration cross-validation for the audited dataset.

This is an evaluation/comparison stage, not a model-selection or deployment
step. Feature sets and estimator configurations are fixed in source. Any choice
among the reported alternatives requires a separate nested evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from ml_evaluation import (
    LABELS,
    OBSERVABLE_ONLY,
    WITH_PROGRAM_COMPARISON,
    EXPLICITLY_EXCLUDED,
    REQUIRED_METADATA,
    REPORT_DIR,
    DATASET_FINGERPRINT_DEFINITION,
    DATASET_FINGERPRINT_SCHEME,
    dataset_fingerprint,
    load_dataset,
    load_report_dispositions,
)
from quantum_simulator.ml_diagnosis import FEATURE_NAMES as ADAPTER_FEATURES


ROOT = Path(__file__).resolve().parent
# Keep new audited outputs separate from any historical/user result directories.
OUT = ROOT / "ml_results" / "grouped_cv_audit_v3"
PROTOCOL_VERSION = "grouped_stratified_model_comparison_v2"
N_SPLITS = 5
SPLIT_SEED = 42
CALIBRATION = {"calibration_readout_error_max", "calibration_t1_min", "calibration_t2_min"}
PRODUCTION_OBSERVABLES = tuple(OBSERVABLE_ONLY[:5])
FEATURE_SCHEMA_DISPOSITIONS = {
    "PRODUCTION_FIVE_OBSERVABLES": {
        "role": "primary_output_observable_only",
        "eligible_tasks": ["anomaly_detection", "cause_disposition_if_supported_by_independent_labels"],
        "limitations": ["cannot localize source/qubit/gate", "requires independent labels for performance claims"],
    },
    "OBSERVABLE_ONLY": {
        "role": "secondary_legacy_output_plus_calibration_comparison",
        "eligible_tasks": ["separate calibration-evidence ablation only"],
        "limitations": ["includes simulator configuration-derived calibration values", "not the primary observable-only schema"],
    },
    "OBSERVABLE_ONLY_NO_CALIBRATION": {
        "role": "secondary_output_observable_ablation",
        "eligible_tasks": ["anomaly_detection", "cause_disposition_if_supported_by_independent_labels"],
        "limitations": ["duplicative of the primary five-feature schema", "not independently validated"],
    },
    "OBSERVABLE_ONLY_DEDUP_CALIBRATION": {
        "role": "secondary_calibration_ablation",
        "eligible_tasks": ["separate calibration-evidence ablation only"],
        "limitations": ["includes configuration-derived calibration values", "not primary observable-only evidence"],
    },
    "WITH_PROGRAM_COMPARISON": {
        "role": "diagnostic_leakage_ablation_not_primary_evaluation",
        "eligible_tasks": ["program-versus-source comparison research only"],
        "limitations": ["gate_mismatch_fraction can reveal the synthetic program-fault construction", "requires source and submitted circuits", "does not localize by itself"],
    },
}
FORBIDDEN = set(EXPLICITLY_EXCLUDED) | {
    "validation_label", "program_fault_description", "program_fault_operation_index",
    "program_fault_qubits", "hardware_affected_component", "ground_truth",
    "label", "target", "y",
}
PREDICTION_LABELS = (*LABELS, "AMBIGUOUS", "FAILED", "UNSUPPORTED", "OTHER")
PREDICTION_ALIASES = {
    # Names emitted by the integrated diagnosis engine, normalized to the
    # training/evaluation label vocabulary. Keep abstentions distinct from
    # malformed or otherwise out-of-schema predictions.
    "NO_ANOMALY": "NORMAL",
    "AMBIGUOUS / INSUFFICIENT_EVIDENCE": "AMBIGUOUS",
}


def evaluation_dataset_fingerprint(report_dir: Path = REPORT_DIR, *, output_dir: Path = OUT) -> str:
    """Shared fingerprint entry point used by both preflight and evaluation."""
    return dataset_fingerprint(report_dir, excluded_dirs=(output_dir,))


def _preflight_summary(
    rows: Sequence[dict[str, Any]],
    dispositions: Sequence[dict[str, Any]],
    folds,
    *,
    dataset_fingerprint_sha256: str,
    source_hashes: dict[str, str],
    dataset_provenance: dict[str, Any] | None = None,
    expected_generation_source_hashes: dict[str, str] | None = None,
    output_dir: Path = OUT,
) -> dict[str, Any]:
    """Validate real-data and grouped-fold readiness without fitting or writing."""
    blockers: list[str] = []
    schema = feature_sets()["PRODUCTION_FIVE_OBSERVABLES"]
    if tuple(schema) != tuple(ADAPTER_FEATURES):
        blockers.append("production feature order differs from the diagnosis adapter schema")
    if len(schema) != 5:
        blockers.append("production feature schema must contain exactly five features")

    provenance_blockers: list[str] = []
    provenance = dataset_provenance.get("generation_provenance") if isinstance(dataset_provenance, dict) else None
    if not isinstance(provenance, dict):
        provenance_blockers.append("dataset manifest is missing generation_provenance")
        declared_sources: dict[str, str] = {}
    else:
        declared_sources = provenance.get("source_sha256") or {}
        runtime = provenance.get("runtime") or {}
        if not isinstance(declared_sources, dict) or not declared_sources:
            provenance_blockers.append("dataset generation source_sha256 map is missing or invalid")
            declared_sources = {}
        else:
            malformed = [
                name for name, digest in declared_sources.items()
                if not isinstance(name, str) or not isinstance(digest, str)
                or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest.lower())
            ]
            if malformed:
                provenance_blockers.append(f"dataset generation source hashes are invalid: {sorted(map(str, malformed))}")
        if not isinstance(runtime, dict) or not isinstance(runtime.get("python_version"), str) or not runtime.get("python_version") or not isinstance(runtime.get("numpy_version"), str) or not runtime.get("numpy_version"):
            provenance_blockers.append("dataset generation runtime must record Python and NumPy versions")
        elif runtime.get("python_version") != platform.python_version() or runtime.get("numpy_version") != np.__version__:
            provenance_blockers.append(
                "dataset generation runtime does not exactly match the active Python/NumPy versions"
            )
    if expected_generation_source_hashes is not None:
        stale = sorted(
            name for name, digest in expected_generation_source_hashes.items()
            if declared_sources.get(name) != digest
        )
        unexpected = sorted(set(declared_sources) - set(expected_generation_source_hashes))
        if stale or unexpected:
            provenance_blockers.append(
                "dataset generator source hashes do not match current generation code"
                + (f" (missing/changed: {stale})" if stale else "")
                + (f" (unexpected: {unexpected})" if unexpected else "")
            )
    rows_without_source = [row for row in rows if not isinstance(row.get("_source_sha256"), dict)]
    rows_with_source_mismatch = [
        row for row in rows
        if isinstance(row.get("_source_sha256"), dict) and row.get("_source_sha256") != declared_sources
    ]
    if rows_without_source:
        provenance_blockers.append(f"{len(rows_without_source)} report rows lack per-run source_sha256 provenance")
    if rows_with_source_mismatch:
        provenance_blockers.append(f"{len(rows_with_source_mismatch)} report rows do not match the dataset generation source hashes")
    blockers.extend(provenance_blockers)

    source_counts = Counter(item.get("disposition", "OTHER") for item in dispositions)
    eligible_rows: list[dict[str, Any]] = []
    invalid_row_reasons: Counter[str] = Counter()
    invalid_row_count = 0
    for row in rows:
        reasons: list[str] = []
        present = set(row.get("_metadata_fields_present", REQUIRED_METADATA))
        missing_fields = set(REQUIRED_METADATA) - present
        if missing_fields:
            reasons.append(f"missing metadata fields: {', '.join(sorted(missing_fields))}")
        for key in ("run_id", "circuit_id", "circuit_family_id", "circuit_lineage_id", "experimental_group_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                reasons.append(f"missing {key}")
        if row.get("ground_truth_label") not in LABELS:
            reasons.append("missing or unsupported ground_truth_label")
        if row.get("validation_label") != "SYNTHETIC_VALIDATION":
            reasons.append("unexpected validation_label")
        if row.get("execution_kind") != "SYNTHETIC_VALIDATION" or row.get("observation_source") != "SEEDED_LOCAL_SIMULATOR_SHOTS_WITH_HARDWARE_MODEL":
            reasons.append("execution provenance is not the expected local simulator source")
        if row.get("hardware_model_applied") is not True:
            reasons.append("hardware model application is missing or false")
        label = row.get("ground_truth_label")
        if label == "NORMAL" and any(row.get(key) is not None for key in ("program_fault_id", "hardware_condition_id")):
            reasons.append("NORMAL row has injected fault metadata")
        elif label == "PROGRAM_FAULT" and (not row.get("program_fault_id") or not row.get("program_fault_description") or row.get("hardware_condition_id") is not None):
            reasons.append("PROGRAM_FAULT scenario metadata is incomplete or contradictory")
        elif label == "HARDWARE_ANOMALY" and (not row.get("hardware_condition_id") or not row.get("hardware_affected_component") or row.get("program_fault_id") is not None):
            reasons.append("HARDWARE_ANOMALY scenario metadata is incomplete or contradictory")
        for feature in schema:
            value = row.get(feature)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(float(value))):
                reasons.append(f"invalid numeric feature {feature}")
        if reasons:
            invalid_row_count += 1
            invalid_row_reasons.update(reasons)
        else:
            eligible_rows.append(row)

    ids = [row["run_id"] for row in eligible_rows]
    if len(ids) != len(set(ids)):
        blockers.append("duplicate run_id values among otherwise eligible rows")
    fingerprints = [row.get("report_fingerprint") for row in eligible_rows]
    if any(not value for value in fingerprints):
        blockers.append("one or more eligible rows lack a report fingerprint")
    if len(fingerprints) != len(set(fingerprints)):
        blockers.append("duplicate report fingerprints among otherwise eligible rows")
    family_labels: dict[str, set[str]] = {}
    family_circuits: dict[str, set[str]] = {}
    experimental_groups: dict[str, set[tuple[str, str]]] = {}
    group_repetitions: dict[str, list[Any]] = {}
    for row in eligible_rows:
        family_labels.setdefault(row["circuit_family_id"], set()).add(row["ground_truth_label"])
        family_circuits.setdefault(row["circuit_family_id"], set()).add(row["circuit_id"])
        experimental_groups.setdefault(row["experimental_group_id"], set()).add((row["circuit_family_id"], row["ground_truth_label"]))
        group_repetitions.setdefault(row["experimental_group_id"], []).append(row.get("repetition_index"))
    if any(labels != set(LABELS) for labels in family_labels.values()):
        blockers.append("at least one circuit family does not contain all three scenario labels")
    if any(len(circuits) != 1 for circuits in family_circuits.values()):
        blockers.append("a circuit family maps to multiple circuit_id values")
    if any(len(values) != 1 for values in experimental_groups.values()):
        blockers.append("an experimental_group_id spans multiple family/label conditions")
    if any(None in values or len(values) != len(set(values)) for values in group_repetitions.values()):
        blockers.append("an experimental group has missing or duplicate repetition indices")
    if eligible_rows:
        for feature in schema:
            if not any(row.get(feature) is not None for row in eligible_rows):
                blockers.append(f"production feature {feature} is entirely missing")

    fold_summaries = []
    if not eligible_rows:
        blockers.append("no eligible rows are available for grouped folds")
    elif len(folds) != N_SPLITS:
        blockers.append(f"expected {N_SPLITS} outer folds, found {len(folds)}")
    else:
        try:
            assignments = fold_assignment_records(eligible_rows, folds)
            for number, (train, test) in enumerate(folds, 1):
                y_train = {eligible_rows[int(index)]["ground_truth_label"] for index in train}
                y_test = {eligible_rows[int(index)]["ground_truth_label"] for index in test}
                if y_train != set(LABELS) or y_test != set(LABELS):
                    blockers.append(f"outer fold {number} does not contain all classes in train and test")
                train_families = {eligible_rows[int(index)]["circuit_family_id"] for index in train}
                test_families = {eligible_rows[int(index)]["circuit_family_id"] for index in test}
                train_lineages = {eligible_rows[int(index)]["circuit_lineage_id"] for index in train}
                test_lineages = {eligible_rows[int(index)]["circuit_lineage_id"] for index in test}
                if train_families & test_families:
                    blockers.append(f"outer fold {number} leaks circuit families")
                if train_lineages & test_lineages:
                    blockers.append(f"outer fold {number} leaks circuit lineages")
                for feature in schema:
                    if not any(eligible_rows[int(index)].get(feature) is not None for index in train):
                        blockers.append(f"production feature {feature} is entirely missing in fold {number} training data")
                fold_summaries.append({
                    "outer_fold": number,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_families": len(train_families),
                    "test_families": len(test_families),
                    "family_overlap": sorted(train_families & test_families),
                    "lineage_overlap": sorted(train_lineages & test_lineages),
                    "train_lineages": len(train_lineages),
                    "test_lineages": len(test_lineages),
                    "train_class_counts": dict(Counter(eligible_rows[int(i)]["ground_truth_label"] for i in train)),
                    "test_class_counts": dict(Counter(eligible_rows[int(i)]["ground_truth_label"] for i in test)),
                })
            if len(assignments) != len(eligible_rows):
                blockers.append("fold assignment coverage differs from eligible row count")
        except (KeyError, ValueError) as exc:
            blockers.append(f"grouped fold validation failed: {type(exc).__name__}: {exc}")

    disposition_eligibility = Counter(item.get("evaluation_eligibility", "UNKNOWN") for item in dispositions)
    ineligible_source_only = sum(
        1 for item in dispositions
        if item.get("evaluation_eligibility") == "INELIGIBLE"
        and item.get("disposition") not in {"FAILED", "UNSUPPORTED"}
    )
    ineligible = ineligible_source_only + invalid_row_count
    return {
        "ready": not blockers,
        "blockers": sorted(set(blockers)),
        "counts": {
            "eligible": len(eligible_rows),
            "completed": source_counts.get("COMPLETED", 0),
            "failed": source_counts.get("FAILED", 0),
            "unsupported": source_counts.get("UNSUPPORTED", 0),
            "ineligible": ineligible,
            "other_source_dispositions": {key: value for key, value in source_counts.items() if key not in {"COMPLETED", "FAILED", "UNSUPPORTED"}},
        },
        "ineligible_row_reasons": dict(invalid_row_reasons),
        "feature_schema": schema,
        "class_order": list(LABELS),
        "label_semantics": "scenario-construction ground truth from controlled simulator reports; not independent external adjudication",
        "protocol_assumptions": [
            "circuit_lineage_id is the grouping unit for outer folds; all family conditions and recipe siblings stay together",
            "the dataset is a controlled local-simulator study, not physical-hardware validation",
            "scenario labels are assigned by construction metadata and require independent adjudication for external accuracy claims",
            "the five production observables are checked against the runtime diagnosis adapter order",
        ],
        "family_count": len(family_labels),
        "folds": fold_summaries,
        "dataset_fingerprint_sha256": dataset_fingerprint_sha256,
        "dataset_fingerprint_scheme": DATASET_FINGERPRINT_SCHEME,
        "dataset_fingerprint_definition": DATASET_FINGERPRINT_DEFINITION,
        "dataset_provenance": provenance if isinstance(provenance, dict) else None,
        "dataset_provenance_blockers": sorted(set(provenance_blockers)),
        "source_hashes": source_hashes,
        "proposed_output_directory": str(output_dir.resolve()),
        "output_is_outside_historical_reports": "reports" not in output_dir.parts,
        "writes_performed": False,
        "estimator_fitting_started": False,
        "prediction_started": False,
        "model_export_started": False,
    }


def run_preflight(report_dir: Path = REPORT_DIR, *, output_dir: Path = OUT) -> dict[str, Any]:
    """Read and validate the selected reports; never creates evaluation outputs."""
    rows = load_dataset(report_dir)
    dispositions = load_report_dispositions(report_dir)
    fingerprint = evaluation_dataset_fingerprint(report_dir, output_dir=output_dir)
    hashes = _source_hashes()
    manifest_path = Path(report_dir) / "dataset_manifest.json"
    try:
        dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        dataset_manifest = None
    from experiments.run_experiment import _source_fingerprints

    generation_hashes = _source_fingerprints()
    y = np.asarray([row["ground_truth_label"] for row in rows])
    groups = np.asarray([row["circuit_lineage_id"] for row in rows])
    folds = []
    if len(rows) >= N_SPLITS and all(row.get("circuit_lineage_id") for row in rows):
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
        try:
            folds = list(splitter.split(np.zeros((len(rows), 1)), y, groups))
        except ValueError:
            folds = []
    return _preflight_summary(
        rows,
        dispositions,
        folds,
        dataset_fingerprint_sha256=fingerprint,
        source_hashes=hashes,
        dataset_provenance=dataset_manifest,
        expected_generation_source_hashes=generation_hashes,
        output_dir=output_dir,
    )


def feature_sets() -> dict[str, list[str]]:
    """Return fixed schemas; only PRODUCTION_FIVE_OBSERVABLES is primary."""
    return {
        "PRODUCTION_FIVE_OBSERVABLES": list(PRODUCTION_OBSERVABLES),
        "OBSERVABLE_ONLY": list(OBSERVABLE_ONLY),
        "WITH_PROGRAM_COMPARISON": list(WITH_PROGRAM_COMPARISON),
        "OBSERVABLE_ONLY_NO_CALIBRATION": [f for f in OBSERVABLE_ONLY if f not in CALIBRATION],
        "OBSERVABLE_ONLY_DEDUP_CALIBRATION": [f for f in OBSERVABLE_ONLY if f != "calibration_t2_min"],
    }


def _validate_feature_sets(schemas: dict[str, Sequence[str]]) -> None:
    for name, features in schemas.items():
        forbidden = sorted(set(features) & FORBIDDEN)
        if forbidden:
            raise ValueError(f"Forbidden metadata/label fields in {name}: {forbidden}")
    if tuple(schemas["PRODUCTION_FIVE_OBSERVABLES"]) != PRODUCTION_OBSERVABLES:
        raise ValueError("production observable schema/order changed")
    if CALIBRATION.intersection(schemas["PRODUCTION_FIVE_OBSERVABLES"]):
        raise ValueError("production observable schema must exclude calibration-derived features")
    if "gate_mismatch_fraction" in schemas["PRODUCTION_FIVE_OBSERVABLES"]:
        raise ValueError("production observable schema must exclude circuit-comparison features")
    if "gate_mismatch_fraction" not in schemas["WITH_PROGRAM_COMPARISON"]:
        raise ValueError("program-comparison ablation must retain gate_mismatch_fraction")
    for name, features in schemas.items():
        if "gate_mismatch_fraction" in features and name != "WITH_PROGRAM_COMPARISON":
            raise ValueError("gate_mismatch_fraction is restricted to the explicitly diagnostic comparison ablation")


def estimator_configs() -> dict[str, dict[str, Any]]:
    """Serializable estimator configuration corresponding to :func:`models`."""
    return {
        "Majority baseline": {
            "pipeline": ["SimpleImputer(strategy=median)"],
            "estimator": "DummyClassifier(strategy=most_frequent)",
        },
        "Stratified dummy baseline": {
            "pipeline": ["SimpleImputer(strategy=median)"],
            "estimator": "DummyClassifier(strategy=stratified, random_state=42)",
        },
        "Logistic Regression": {
            "pipeline": ["SimpleImputer(strategy=median)", "StandardScaler()"],
            "estimator": "LogisticRegression(max_iter=2000, class_weight=balanced)",
        },
        "Linear SVM": {
            "pipeline": ["SimpleImputer(strategy=median)", "StandardScaler()"],
            "estimator": "LinearSVC(C=1.0, class_weight=balanced, random_state=42, dual=auto)",
            "probabilities": "not produced by the margin-only classifier",
        },
        "Decision Tree": {
            "pipeline": ["SimpleImputer(strategy=median)"],
            "estimator": "DecisionTreeClassifier(max_depth=6, class_weight=balanced, random_state=42)",
        },
        "Random Forest": {
            "pipeline": ["SimpleImputer(strategy=median)"],
            "estimator": "RandomForestClassifier(n_estimators=200, max_depth=8, class_weight=balanced, random_state=42, n_jobs=1)",
        },
    }


def models():
    return {
        "Majority baseline": make_pipeline(
            SimpleImputer(strategy="median"),
            DummyClassifier(strategy="most_frequent"),
        ),
        "Stratified dummy baseline": make_pipeline(
            SimpleImputer(strategy="median"),
            DummyClassifier(strategy="stratified", random_state=SPLIT_SEED),
        ),
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "Linear SVM": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LinearSVC(C=1.0, class_weight="balanced", random_state=SPLIT_SEED, dual="auto"),
        ),
        "Decision Tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=1),
        ),
    }


def validate_three_way_family_partitions(
    rows: Sequence[dict[str, Any]],
    partitions: dict[str, Sequence[int]],
) -> dict[str, Any]:
    """Require exhaustive train/calibration/test partitions with family isolation.

    This validates split structure only. It does not establish that lineages or
    labels are independently adjudicated, representative, or numerous enough.
    """
    required = ("train", "calibration", "test")
    if set(partitions) != set(required):
        raise ValueError(f"partitions must contain exactly {required}")
    if not rows:
        raise ValueError("cannot partition an empty dataset")
    expected = set(range(len(rows)))
    index_sets: dict[str, set[int]] = {}
    for name in required:
        raw_values = list(partitions[name])
        if any(isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) for index in raw_values):
            raise ValueError(f"{name} partition indices must be integers")
        values = [int(index) for index in raw_values]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate row index in {name} partition")
        indices = set(values)
        if not indices <= expected:
            raise ValueError(f"out-of-range row index in {name} partition")
        index_sets[name] = indices
    if set.union(*index_sets.values()) != expected:
        raise ValueError("three-way split must assign every row exactly once")
    for left_i, left in enumerate(required):
        for right in required[left_i + 1:]:
            if index_sets[left] & index_sets[right]:
                raise ValueError(f"row overlap between {left} and {right} partitions")

    fingerprints: dict[str, tuple[str, str]] = {}
    summary: dict[str, Any] = {}
    for name in required:
        subset = [rows[index] for index in sorted(index_sets[name])]
        for row in subset:
            for field in ("run_id", "circuit_family_id", "circuit_lineage_id"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"{name} partition row has missing {field}")
            if row.get("ground_truth_label") not in LABELS:
                raise ValueError(f"{name} partition row has missing or unsupported ground_truth_label")
            circuit_fingerprint = row.get("circuit_fingerprint")
            if circuit_fingerprint:
                identity = (str(row["circuit_lineage_id"]), name)
                previous = fingerprints.setdefault(str(circuit_fingerprint), identity)
                if previous[0] != identity[0]:
                    raise ValueError("duplicate circuit fingerprint is assigned to different lineage roots")
                if previous[1] != identity[1]:
                    raise ValueError("duplicate circuit fingerprint crosses family partitions")
        summary[name] = {
            "row_count": len(subset),
            "family_ids": sorted({row["circuit_family_id"] for row in subset}),
            "lineage_ids": sorted({row["circuit_lineage_id"] for row in subset}),
        }
    for field in ("circuit_family_id", "circuit_lineage_id"):
        membership = {name: {row[field] for row in (rows[i] for i in index_sets[name])} for name in required}
        for left_i, left in enumerate(required):
            for right in required[left_i + 1:]:
                if membership[left] & membership[right]:
                    raise ValueError(f"{field} leakage between {left} and {right} partitions")
    return {"structurally_valid": True, "partition_count": 3, "partitions": summary}


def prepare_new_output_directory(path: Path = OUT) -> None:
    """Refuse to overwrite any existing model-comparison evidence directory."""
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"refusing to overwrite nonempty evaluation output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def family_cluster_bootstrap_intervals(
    rows: Sequence[dict[str, Any]],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    repetitions: int = 1000,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Percentile intervals from resampling independent lineage clusters.

    No interval is returned for a metric unless every requested replicate has
    support for all three target classes. This avoids silently assigning zero
    performance to classes absent from a bootstrap sample.
    """
    if len(rows) != len(y_true) or len(rows) != len(y_pred):
        raise ValueError("rows, truth and predictions must have equal lengths")
    if isinstance(repetitions, bool) or not isinstance(repetitions, (int, np.integer)) or repetitions < 1:
        raise ValueError("bootstrap repetitions must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("bootstrap seed must be an integer")
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        lineage = row.get("circuit_lineage_id")
        if not isinstance(lineage, str) or not lineage.strip():
            raise ValueError("family bootstrap requires nonempty circuit_lineage_id for every row")
        groups.setdefault(lineage, []).append(index)
    group_names = sorted(groups)
    result: dict[str, Any] = {
        "independent_lineage_count": len(group_names),
        "requested_replicates": repetitions,
        "seed": seed,
        "method": "lineage-cluster percentile bootstrap; resample lineages, retain all rows within sampled lineage",
        "intervals_95": {},
        "metric_replicates": {},
    }
    if len(group_names) < 2:
        result["available"] = False
        result["reason"] = "fewer than two independent lineages; cluster uncertainty is not estimable"
        return result
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "accuracy": [], "macro_f1": [], "balanced_accuracy": [],
        "coverage": [], "selective_error": [],
        **{f"{label}_{metric}": [] for label in LABELS for metric in ("precision", "recall", "f1")},
    }
    complete_class_replicates = 0
    for _ in range(repetitions):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        indices = [index for name in sampled for index in groups[str(name)]]
        truth = [y_true[index] for index in indices]
        prediction = [y_pred[index] for index in indices]
        summary = _metric_summary(truth, prediction)
        values["accuracy"].append(float(summary["accuracy"]))
        values["coverage"].append(float(summary["coverage"]))
        if summary["selective_error"] is not None:
            values["selective_error"].append(float(summary["selective_error"]))
        if all(summary["per_class"][label]["support"] > 0 for label in LABELS):
            complete_class_replicates += 1
            values["macro_f1"].append(float(summary["macro_f1"]))
            values["balanced_accuracy"].append(float(summary["balanced_accuracy"]))
            for label in LABELS:
                for metric in ("precision", "recall", "f1"):
                    values[f"{label}_{metric}"].append(float(summary["per_class"][label][metric]))
    result["metric_replicates"] = {
        metric: len(samples) for metric, samples in values.items()
    }
    result["metric_replicates"]["all_classes_supported"] = complete_class_replicates
    for metric, samples in values.items():
        if len(samples) == repetitions:
            low, high = np.quantile(np.asarray(samples), [0.025, 0.975])
            result["intervals_95"][metric] = [float(low), float(high)]
    result["available"] = bool(result["intervals_95"])
    if len(values["macro_f1"]) != repetitions:
        result["reason"] = "class-wise intervals withheld because some cluster-bootstrap samples lack one or more classes"
    return result


def fold_assignment_records(rows: Sequence[dict[str, Any]], folds) -> list[dict[str, Any]]:
    """Validate family/lineage isolation and exact once-only outer test coverage."""
    if any(not isinstance(row.get("circuit_family_id"), str) or not row["circuit_family_id"].strip() for row in rows):
        raise ValueError("every evaluated row must have a nonempty circuit_family_id before fold assignment")
    if any(not isinstance(row.get("circuit_lineage_id"), str) or not row["circuit_lineage_id"].strip() for row in rows):
        raise ValueError("every evaluated row must have a nonempty circuit_lineage_id before fold assignment")
    assigned: list[int | None] = [None] * len(rows)
    fold_records = []
    for fold_number, (train, test) in enumerate(folds, 1):
        train_indices, test_indices = list(map(int, train)), list(map(int, test))
        if len(train_indices) != len(set(train_indices)) or len(test_indices) != len(set(test_indices)):
            raise ValueError(f"duplicate row index in outer fold {fold_number}")
        if set(train_indices) & set(test_indices):
            raise ValueError(f"training/test row overlap in outer fold {fold_number}")
        if set(train_indices) | set(test_indices) != set(range(len(rows))):
            raise ValueError(f"outer fold {fold_number} does not partition all evaluated rows")
        train_groups = {rows[int(index)]["circuit_family_id"] for index in train}
        test_groups = {rows[int(index)]["circuit_family_id"] for index in test}
        train_lineages = {rows[int(index)]["circuit_lineage_id"] for index in train}
        test_lineages = {rows[int(index)]["circuit_lineage_id"] for index in test}
        if train_groups & test_groups:
            raise ValueError(f"circuit family leakage in outer fold {fold_number}")
        if train_lineages & test_lineages:
            raise ValueError(f"circuit lineage leakage in outer fold {fold_number}")
        if set(map(int, train)) & set(map(int, test)):
            raise ValueError(f"training/test row overlap in outer fold {fold_number}")
        for index in test:
            index = int(index)
            if assigned[index] is not None:
                raise ValueError(f"row {index} appears in multiple outer test folds")
            assigned[index] = fold_number
    if any(fold is None for fold in assigned):
        raise ValueError("each evaluated row must appear in exactly one outer test fold")
    for index, row in enumerate(rows):
        fold_records.append({
            "run_id": row["run_id"],
            "circuit_family_id": row["circuit_family_id"],
            "circuit_lineage_id": row["circuit_lineage_id"],
            "experimental_group_id": row.get("experimental_group_id"),
            "ground_truth_label": row["ground_truth_label"],
            "outer_fold": assigned[index],
            # This fixed-model protocol has no inner CV or tuning.
            "inner_fold": None,
        })
    return fold_records


def feature_matrix(
    rows: Sequence[dict[str, Any]],
    features: Sequence[str],
    *,
    schema_name: str | None = None,
) -> np.ndarray:
    """Build a matrix only from a registered schema.

    The default is the primary observable-only production schema. Every other
    schema, including circuit-comparison and calibration ablations, must be
    named explicitly so callers cannot accidentally turn a diagnostic field
    into an inference feature.
    """
    forbidden = sorted(set(features) & FORBIDDEN)
    if forbidden:
        raise ValueError(f"Forbidden metadata/label fields in feature matrix: {forbidden}")
    schemas = feature_sets()
    _validate_feature_sets(schemas)
    if schema_name is None:
        schema_name = "PRODUCTION_FIVE_OBSERVABLES"
    if schema_name not in schemas:
        raise ValueError(f"unregistered feature schema: {schema_name}")
    if tuple(features) != tuple(schemas[schema_name]):
        raise ValueError(f"feature list does not match registered schema {schema_name}")
    return np.asarray([
        [float(row.get(feature)) if row.get(feature) is not None else np.nan for feature in features]
        for row in rows
    ], dtype=float)


def fit_predict_fold(estimator, X: np.ndarray, y: Sequence[str], train, test):
    """Fit preprocessing/model on train indices only and predict the held-out fold."""
    train_indices, test_indices = np.asarray(train, dtype=int), np.asarray(test, dtype=int)
    if set(train_indices.tolist()) & set(test_indices.tolist()):
        raise ValueError("training/test row overlap")
    estimator.fit(X[train_indices], np.asarray(y)[train_indices])
    predictions = estimator.predict(X[test_indices])
    probabilities = estimator.predict_proba(X[test_indices]) if hasattr(estimator, "predict_proba") else None
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        classes = getattr(estimator[-1], "classes_", [])
    classes = list(classes)
    probability_rows = [
        {str(label): float(value) for label, value in zip(classes, row)}
        for row in probabilities
    ] if probabilities is not None else [{} for _ in test_indices]
    return predictions, probability_rows


def _metric_summary(y_true: Sequence[str], y_pred: Sequence[str], *, source_dispositions=None) -> dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("truth/prediction counts differ")
    normalized_predictions = [
        PREDICTION_ALIASES.get(pred, pred) if isinstance(pred, str) else "OTHER"
        for pred in y_pred
    ]
    matrix = {truth: {pred: 0 for pred in PREDICTION_LABELS} for truth in LABELS}
    for truth, pred in zip(y_true, normalized_predictions):
        if truth not in matrix:
            raise ValueError(f"unsupported ground-truth label: {truth}")
        normalized = pred if pred in PREDICTION_LABELS else "OTHER"
        matrix[truth][normalized] += 1
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(count for predicted, count in matrix[label].items() if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": sum(matrix[label].values()), "precision": precision, "recall": recall, "f1": f1}
    dispositions = list(source_dispositions or [])
    covered = sum(pred in LABELS for pred in normalized_predictions)
    covered_correct = sum(truth == pred for truth, pred in zip(y_true, normalized_predictions) if pred in LABELS)
    selective_accuracy = covered_correct / covered if covered else None
    return {
        "evaluated_case_count": len(y_true),
        "class_order": list(LABELS),
        "prediction_columns": list(PREDICTION_LABELS),
        "confusion_matrix_true_rows_predicted_columns": matrix,
        "per_class": per_class,
        "accuracy": accuracy_score(y_true, normalized_predictions) if y_true else 0.0,
        "macro_f1": sum(value["f1"] for value in per_class.values()) / len(LABELS),
        "balanced_accuracy": sum(value["recall"] for value in per_class.values()) / len(LABELS),
        # Coverage counts only resolved predictions in the registered three-class
        # label space. Ambiguous, failed, unsupported and out-of-schema outputs
        # remain non-covered; overall accuracy above still penalizes each one.
        "coverage_count": covered,
        "coverage": covered / len(y_true) if y_true else 0.0,
        "selective_accuracy": selective_accuracy,
        "selective_error": 1.0 - selective_accuracy if selective_accuracy is not None else None,
        "nonclassification_count": len(y_true) - covered,
        "abstention_count": sum(value == "AMBIGUOUS" for value in normalized_predictions),
        "failure_count": sum(value == "FAILED" for value in normalized_predictions) + sum(item["disposition"] == "FAILED" for item in dispositions),
        "unsupported_operation_count": sum(value == "UNSUPPORTED" for value in normalized_predictions) + sum(item["disposition"] == "UNSUPPORTED" for item in dispositions),
        "ineligible_source_case_count": sum(item.get("evaluation_eligibility") == "INELIGIBLE" for item in dispositions),
        "other_prediction_count": sum(value == "OTHER" or value not in PREDICTION_LABELS for value in normalized_predictions),
        "source_dispositions": dict(Counter(item["disposition"] for item in dispositions)),
        "source_eligibility": dict(Counter(item.get("evaluation_eligibility", "UNKNOWN") for item in dispositions)),
    }


def _source_hashes() -> dict[str, str]:
    paths = {
        "ml_training.py": ROOT / "ml_training.py",
        "ml_evaluation.py": ROOT / "ml_evaluation.py",
        "experiments/full_dataset.py": ROOT / "experiments" / "full_dataset.py",
        "experiments/run_experiment.py": ROOT / "experiments" / "run_experiment.py",
        "experiment_config.py": ROOT / "experiment_config.py",
        "src/quantum_simulator/statistics.py": ROOT / "src" / "quantum_simulator" / "statistics.py",
        "src/quantum_simulator/diagnosis_engine.py": ROOT / "src" / "quantum_simulator" / "diagnosis_engine.py",
        "src/quantum_simulator/quantum_simulator.py": ROOT / "src" / "quantum_simulator" / "quantum_simulator.py",
        "src/quantum_simulator/hardware_model.py": ROOT / "src" / "quantum_simulator" / "hardware_model.py",
    }
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prediction_rows(rows, fold_records, feature_set, model_name, predictions, probabilities, errors, *, provenance=None):
    if not (len(rows) == len(fold_records) == len(predictions) == len(probabilities) == len(errors)):
        raise ValueError("out-of-fold prediction data must cover each evaluated row exactly once")
    result = []
    for row, fold, prediction, probability, error in zip(rows, fold_records, predictions, probabilities, errors):
        if row["run_id"] != fold["run_id"]:
            raise ValueError("fold assignment does not match row identity")
        record = {
            "run_id": row["run_id"],
            "circuit_family_id": row["circuit_family_id"],
            "experimental_group_id": row.get("experimental_group_id"),
            "feature_set": feature_set,
            "model": model_name,
            "outer_fold": fold["outer_fold"],
            "ground_truth_label": row["ground_truth_label"],
            # An error status is authoritative even if a caller also supplied
            # a partial or stale class prediction for the failed fold.
            "prediction": "FAILED" if error else prediction,
            "prediction_status": "FAILED" if error else "COMPLETED",
            "error": error,
        }
        record.update({f"probability_{label}": probability.get(label) for label in LABELS})
        if provenance:
            record.update(provenance)
        result.append(record)
    keys = [(record["feature_set"], record["model"], record["run_id"]) for record in result]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate out-of-fold prediction for a case/configuration")
    return result


def build_evaluation_outputs(
    rows: Sequence[dict[str, Any]],
    folds,
    model_results: dict[tuple[str, str], dict[str, Any]],
    source_dispositions: Sequence[dict[str, Any]],
    *,
    schemas: dict[str, Sequence[str]] | None = None,
    configurations: dict[str, dict[str, Any]] | None = None,
    source_hashes: dict[str, str] | None = None,
    dataset_fingerprint_sha256: str,
    split_seed: int = SPLIT_SEED,
    splitter_name: str = "StratifiedGroupKFold",
    split_shuffle: bool = True,
    runtime: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the common, internally checked manifest and OOF output payload.

    The training entry point and synthetic tests use this same constructor. No
    estimator is fitted here; ``model_results`` contains fold-held-out outputs.
    """
    schemas = schemas or feature_sets()
    configurations = configurations or estimator_configs()
    source_hashes = source_hashes or _source_hashes()
    runtime = runtime or {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__}
    _validate_feature_sets(schemas)
    assignments = fold_assignment_records(rows, folds)
    for assignment in assignments:
        assignment.update({
            "dataset_fingerprint_sha256": dataset_fingerprint_sha256,
            "dataset_fingerprint_scheme": DATASET_FINGERPRINT_SCHEME,
            "split_seed": split_seed,
            "protocol_version": PROTOCOL_VERSION,
        })
    if len({record["run_id"] for record in assignments}) != len(rows):
        raise ValueError("run IDs must uniquely identify evaluated rows")

    source_hash_digest = hashlib.sha256(json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    predictions = []
    metrics_by_configuration = {}
    comparison_rows = []
    fold_membership = []
    for fold_number, (train, test) in enumerate(folds, 1):
        fold_membership.append({
            "outer_fold": fold_number,
            "training_run_ids": [rows[int(index)]["run_id"] for index in train],
            "test_run_ids": [rows[int(index)]["run_id"] for index in test],
            "training_families": sorted({rows[int(index)]["circuit_family_id"] for index in train}),
            "test_families": sorted({rows[int(index)]["circuit_family_id"] for index in test}),
            "training_lineages": sorted({rows[int(index)]["circuit_lineage_id"] for index in train}),
            "test_lineages": sorted({rows[int(index)]["circuit_lineage_id"] for index in test}),
            "inner_fold_assignments": None,
        })
    for (feature_set, model_name), result in model_results.items():
        if feature_set not in schemas:
            raise ValueError(f"unknown feature schema: {feature_set}")
        if model_name not in configurations:
            raise ValueError(f"unknown estimator configuration: {model_name}")
        provenance = {
            "protocol_version": PROTOCOL_VERSION,
            "split_seed": split_seed,
            "feature_schema": json.dumps(list(schemas[feature_set]), separators=(",", ":")),
            "feature_schema_id": feature_set,
            "estimator_configuration": json.dumps(configurations[model_name], sort_keys=True, separators=(",", ":")),
            "dataset_fingerprint_sha256": dataset_fingerprint_sha256,
            "dataset_fingerprint_scheme": DATASET_FINGERPRINT_SCHEME,
            "source_hash_sha256": source_hash_digest,
        }
        config_predictions = _prediction_rows(
            rows,
            assignments,
            feature_set,
            model_name,
            result["predictions"],
            result["probabilities"],
            result["errors"],
            provenance=provenance,
        )
        predictions.extend(config_predictions)
        key = f"{feature_set}::{model_name}"
        metric_summary = _metric_summary(
            [row["ground_truth_label"] for row in rows],
            [row["prediction"] for row in config_predictions],
            source_dispositions=source_dispositions,
        )
        metric_summary["family_cluster_bootstrap"] = family_cluster_bootstrap_intervals(
            rows,
            [row["ground_truth_label"] for row in rows],
            [row["prediction"] for row in config_predictions],
            seed=split_seed,
        )
        metrics_by_configuration[key] = metric_summary
        fold_scores = []
        for fold_number in range(1, len(folds) + 1):
            fold_rows = [row for row in config_predictions if row["outer_fold"] == fold_number]
            fold_scores.append(_metric_summary(
                [row["ground_truth_label"] for row in fold_rows],
                [row["prediction"] for row in fold_rows],
            ))
        comparison_rows.append({
            "feature_set": feature_set,
            "feature_schema_id": feature_set,
            "model": model_name,
            "feature_names": json.dumps(list(schemas[feature_set])),
            "estimator_configuration": json.dumps(configurations[model_name], sort_keys=True, separators=(",", ":")),
            "dataset_fingerprint_sha256": dataset_fingerprint_sha256,
            "dataset_fingerprint_scheme": DATASET_FINGERPRINT_SCHEME,
            "split_seed": split_seed,
            "protocol_version": PROTOCOL_VERSION,
            **{
                f"{metric}_mean": float(np.mean([score[metric] for score in fold_scores]))
                for metric in ("accuracy", "macro_f1", "balanced_accuracy")
            },
            **{
                f"{metric}_std": float(np.std([score[metric] for score in fold_scores], ddof=1)) if len(fold_scores) > 1 else 0.0
                for metric in ("accuracy", "macro_f1", "balanced_accuracy")
            },
            "oof_macro_f1": metric_summary["macro_f1"],
            "oof_balanced_accuracy": metric_summary["balanced_accuracy"],
        })

    expected = len(rows) * len(model_results)
    prediction_keys = [
        (row["feature_set"], row["model"], row["run_id"])
        for row in predictions
    ]
    if len(predictions) != expected or len(set(prediction_keys)) != expected:
        raise ValueError("every eligible case must receive exactly one OOF prediction per configuration")
    for record in predictions:
        if record["protocol_version"] != PROTOCOL_VERSION or record["split_seed"] != split_seed:
            raise ValueError("prediction provenance does not match evaluation manifest")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": {
            "eligible_report_count": len(rows),
            "fingerprint_sha256": dataset_fingerprint_sha256,
            "fingerprint_scheme": DATASET_FINGERPRINT_SCHEME,
            "fingerprint_definition": DATASET_FINGERPRINT_DEFINITION,
            "source_directory": "reports/experiments/full_dataset",
        },
        "splitter": {"class": splitter_name, "n_splits": len(folds), "shuffle": split_shuffle, "random_state": split_seed, "group_field": "circuit_lineage_id"},
        "outer_fold_membership": fold_membership,
        "inner_cv": None,
        "inner_fold_assignments": None,
        "selection_policy": "Fixed feature sets and estimator configurations; no model or hyperparameter selected from these outer-fold results. Use nested CV for selection.",
        "calibration": {
            "partition": None,
            "performed": False,
            "metrics_emitted": False,
            "probability_semantics": "raw estimator predict_proba outputs when available; not calibrated probabilities",
            "reason": "No eligible independent calibration families or labels are available in this dataset.",
        },
        "feature_schemas": {name: list(features) for name, features in schemas.items()},
        "feature_schema_dispositions": {
            name: FEATURE_SCHEMA_DISPOSITIONS.get(name, {
                "role": "unclassified_schema",
                "eligible_tasks": [],
                "limitations": ["schema has no reviewed role; do not use for claims"],
            }) for name in schemas
        },
        "production_observables": list(PRODUCTION_OBSERVABLES),
        "prediction_task": "three-class synthetic scenario disposition from output distributions: NORMAL vs PROGRAM_FAULT vs HARDWARE_ANOMALY",
        "label_origin": "controlled simulator scenario-construction labels; not independent adjudication",
        "evidence_scope": "SIMULATOR_SYNTHETIC_ONLY",
        "independent_generalization_claim_eligible": False,
        "real_hardware_validation": False,
        "calibration_features": sorted(CALIBRATION),
        "program_comparison_feature": "gate_mismatch_fraction",
        "estimators": configurations,
        "split_seed": split_seed,
        "source_dispositions": dict(Counter(item["disposition"] for item in source_dispositions)),
        "eligibility_counts": dict(Counter(item["evaluation_eligibility"] for item in source_dispositions)),
        "source_hashes": source_hashes,
        "source_hash_sha256": source_hash_digest,
        "runtime": runtime,
        "metrics": metrics_by_configuration,
        "evaluated_configurations": [
            {"feature_set": feature_set, "model": model_name}
            for feature_set, model_name in model_results
        ],
        "out_of_fold_predictions": {"file": "out_of_fold_predictions.csv", "exactly_once_per_eligible_case_and_configuration": True, "row_count": len(predictions)},
    }
    return {
        "manifest": manifest,
        "fold_assignments": assignments,
        "predictions": predictions,
        "comparison_rows": comparison_rows,
        "metrics": metrics_by_configuration,
    }


def main(*, preflight_only: bool = False) -> int:
    preflight_result = run_preflight()
    if preflight_only:
        print(json.dumps(preflight_result, indent=2, sort_keys=True))
        return 0 if preflight_result["ready"] else 1
    if not preflight_result["ready"]:
        print("Training blocked by dataset preflight:")
        print(json.dumps(preflight_result["blockers"], indent=2))
        return 2
    try:
        prepare_new_output_directory(OUT)
    except FileExistsError as exc:
        print(str(exc))
        return 2

    rows = load_dataset()
    if len(rows) != 270:
        raise ValueError(f"Expected audited full dataset (270 records), found {len(rows)}")
    if len({row["circuit_family_id"] for row in rows}) != 30:
        raise ValueError("Expected 30 independent circuit-family groups")
    if Counter(row["ground_truth_label"] for row in rows) != Counter({label: 90 for label in LABELS}):
        raise ValueError("Expected exactly 90 records in each class")

    schemas = feature_sets()
    _validate_feature_sets(schemas)
    y = np.asarray([row["ground_truth_label"] for row in rows])
    groups = np.asarray([row["circuit_lineage_id"] for row in rows])
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    folds = list(splitter.split(np.zeros((len(rows), 1)), y, groups))
    # Validate every outer split before any estimator is fitted or any feature
    # importance is computed. build_evaluation_outputs repeats this check when
    # serializing results, but that later guard alone would be too late.
    fold_assignment_records(rows, folds)
    source_dispositions = load_report_dispositions()

    print("FIXED FEATURE LISTS / LEAKAGE CHECK")
    for name, features in schemas.items():
        print(f"{name}: {features}")
    print("The production five observables exclude calibration and program-comparison fields.")
    print("Dummy baselines and fixed Logistic Regression, Linear SVM, Decision Tree, and Random Forest configurations are compared without hyperparameter tuning.")
    print("Feature sets and estimator configurations are fixed; reported rankings are descriptive, not model selection.")
    for fold_i, (train, valid) in enumerate(folds, 1):
        print(f"fold {fold_i}: train_groups={len(set(groups[train]))}, validation_groups={len(set(groups[valid]))}")

    model_results, importance_rows = {}, []
    configuration = estimator_configs()
    for set_name, features in schemas.items():
        X = feature_matrix(rows, features, schema_name=set_name)
        for model_name, estimator in models().items():
            predictions: list[str | None] = [None] * len(rows)
            probabilities: list[dict[str, float]] = [{} for _ in rows]
            prediction_errors: list[str | None] = [None] * len(rows)
            for fold_i, (train, valid) in enumerate(folds, 1):
                fold_error = None
                try:
                    # Imputation and scaling are pipeline steps and fit only on this fold's training rows.
                    fold_predictions, fold_probability_rows = fit_predict_fold(estimator, X, y, train, valid)
                except Exception as exc:
                    fold_error = f"{type(exc).__name__}: {exc}"
                    for index in valid:
                        index = int(index)
                        predictions[index] = "FAILED"
                        prediction_errors[index] = fold_error
                    fold_predictions = ["FAILED"] * len(valid)
                else:
                    for offset, index in enumerate(valid):
                        index = int(index)
                        predictions[index] = str(fold_predictions[offset])
                        probabilities[index] = fold_probability_rows[offset]
                    if model_name == "Random Forest":
                        forest = estimator[-1]
                        for feature, value in zip(features, forest.feature_importances_):
                            importance_rows.append({"feature_set": set_name, "fold": fold_i, "importance_type": "impurity", "feature": feature, "importance_mean": float(value), "importance_std": 0.0})
                        perm = permutation_importance(estimator, X[valid], y[valid], scoring="f1_macro", n_repeats=5, random_state=SPLIT_SEED, n_jobs=1)
                        for feature, mean, std in zip(features, perm.importances_mean, perm.importances_std):
                            importance_rows.append({"feature_set": set_name, "fold": fold_i, "importance_type": "permutation_macro_f1", "feature": feature, "importance_mean": float(mean), "importance_std": float(std)})
                for index in valid:
                    if predictions[int(index)] is None:
                        raise ValueError(f"missing out-of-fold prediction for row {int(index)}")

            model_results[(set_name, model_name)] = {
                "predictions": predictions,
                "probabilities": probabilities,
                "errors": prediction_errors,
            }

    dataset_fingerprint_sha256 = evaluation_dataset_fingerprint(REPORT_DIR, output_dir=OUT)
    outputs = build_evaluation_outputs(
        rows,
        folds,
        model_results,
        source_dispositions,
        schemas=schemas,
        configurations=configuration,
        source_hashes=_source_hashes(),
        dataset_fingerprint_sha256=dataset_fingerprint_sha256,
        split_seed=SPLIT_SEED,
    )
    _write_csv(OUT / "source_case_dispositions.csv", source_dispositions)
    _write_csv(OUT / "fold_assignments.csv", outputs["fold_assignments"])
    _write_csv(OUT / "out_of_fold_predictions.csv", outputs["predictions"])
    _write_csv(OUT / "feature_importance.csv", importance_rows)
    _write_csv(OUT / "model_comparison.csv", outputs["comparison_rows"])
    for key, metrics in outputs["metrics"].items():
        feature_set, model_name = key.split("::", maxsplit=1)
        safe = f"{feature_set}__{model_name.lower().replace(' ', '_')}"
        (OUT / f"{safe}__metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        matrix = metrics["confusion_matrix_true_rows_predicted_columns"]
        matrix_rows = [[matrix[truth][predicted] for predicted in PREDICTION_LABELS] for truth in LABELS]
        np.savetxt(OUT / f"{safe}__confusion_matrix.csv", np.asarray(matrix_rows), delimiter=",", fmt="%d", header=",".join(PREDICTION_LABELS), comments="")
    (OUT / "evaluation_manifest.json").write_text(json.dumps(outputs["manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote auditable fixed-protocol evaluation outputs to {OUT}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="audit source data and grouped folds, then stop before fitting/prediction/export",
    )
    raise SystemExit(main(preflight_only=parser.parse_args().preflight_only))
