"""Run a small, family-held-out, rule-only diagnosis evaluation.

This deliberately reuses the controlled scenario generator without invoking its
full-dataset writer. Each evaluation gets a new output directory and records
its deterministic inputs, source hashes, predictions, and metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from experiment_config import ExperimentConfig
from experiments.full_dataset import (
    DESIGN_SEED_BASE,
    EXECUTION_SEED_BASE,
    SHOTS,
    _family_name,
    build_family_configs,
)
from experiments.run_experiment import run_experiment


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "controlled_rule_pipeline_family_holdout_v1"
FAMILY_INDICES = (0, 4, 8, 12, 16, 20)
LABELS = ("NORMAL", "PROGRAM_FAULT", "HARDWARE_ANOMALY")
PREDICTIONS = (*LABELS, "AMBIGUOUS", "FAILED", "UNSUPPORTED", "OTHER")
SOURCE_FILES = (
    "experiments/evaluate_controlled_rule_pipeline.py",
    "experiments/full_dataset.py",
    "experiments/run_experiment.py",
    "experiment_config.py",
    "src/quantum_simulator/diagnosis_engine.py",
    "src/quantum_simulator/fault_localizer.py",
    "src/quantum_simulator/hardware_fault_localizer.py",
    "src/quantum_simulator/quantum_simulator.py",
    "src/quantum_simulator/hardware_model.py",
    "src/quantum_simulator/statistics.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction(result: Any) -> str:
    if result.status != "COMPLETED":
        error_text = " ".join(result.errors).lower()
        return "UNSUPPORTED" if "unsupported operation" in error_text else "FAILED"
    if result.diagnosis is None:
        return "OTHER"
    value = getattr(result.diagnosis.category, "value", result.diagnosis.category)
    if value == "NO_ANOMALY":
        return "NORMAL"
    if value in LABELS:
        return value
    if value == "AMBIGUOUS / INSUFFICIENT_EVIDENCE":
        return "AMBIGUOUS"
    return "OTHER"


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = {label: {pred: 0 for pred in PREDICTIONS} for label in LABELS}
    for row in records:
        matrix[row["ground_truth"]][row["prediction"]] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[truth][label] for truth in LABELS if truth != label)
        fn = sum(matrix[label][pred] for pred in PREDICTIONS if pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": sum(matrix[label].values()), "precision": precision, "recall": recall, "f1": f1}
    macro_f1 = sum(row["f1"] for row in per_class.values()) / len(LABELS)
    balanced_accuracy = sum(row["recall"] for row in per_class.values()) / len(LABELS)
    ambiguous = sum(row["prediction"] == "AMBIGUOUS" for row in records)
    missing_evidence = sum(bool(row["missing_evidence"]) for row in records)
    failed = sum(row["prediction"] == "FAILED" for row in records)
    unsupported = sum(row["prediction"] == "UNSUPPORTED" for row in records)
    program_rows = [row for row in records if row["ground_truth"] == "PROGRAM_FAULT"]
    localized = sum(row["localization_match"] is True for row in program_rows)
    return {
        "class_order": list(LABELS),
        "prediction_columns": list(PREDICTIONS),
        "confusion_matrix_true_rows_predicted_columns": matrix,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "abstention_count": ambiguous,
        "abstention_rate": ambiguous / len(records) if records else 0.0,
        "missing_evidence_count": missing_evidence,
        "missing_evidence_rate": missing_evidence / len(records) if records else 0.0,
        "failed_count": failed,
        "unsupported_count": unsupported,
        "other_count": sum(row["prediction"] == "OTHER" for row in records),
        "program_localization": {
            "matched": localized,
            "eligible": len(program_rows),
            "rate": localized / len(program_rows) if program_rows else None,
            "criterion": "candidate operation index and qubits both equal explicit generator ground truth",
        },
    }


def evaluate(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    cases: list[tuple[ExperimentConfig, int]] = []
    for family_index in FAMILY_INDICES:
        configs = build_family_configs(family_index)
        cases.extend((config, family_index) for config in configs if config.repetition_index == 0)

    rows: list[dict[str, Any]] = []
    for config, family_index in cases:
        target = output_dir / f"{config.run_id}.json"
        result = run_experiment(config, output_path=target)
        diagnosis = result.diagnosis
        candidates = diagnosis.program_result.candidates if diagnosis and diagnosis.program_result else []
        localization_match = None
        if config.ground_truth_label == "PROGRAM_FAULT":
            localization_match = any(
                candidate.index == config.program_fault_operation_index
                and list(candidate.qubits) == list(config.program_fault_qubits or [])
                for candidate in candidates
            )
        rows.append({
            "family_index": family_index,
            "circuit_family_id": config.circuit_family_id,
            "circuit_id": config.circuit_id,
            "experimental_group_id": config.experimental_group_id,
            "run_id": config.run_id,
            "repetition_index": config.repetition_index,
            "ground_truth": config.ground_truth_label,
            "prediction": _prediction(result),
            "status": result.status,
            "errors": result.errors,
            "random_seed": config.random_seed,
            "shots": config.shots,
            "threshold": config.anomaly_threshold,
            "program_fault_id": config.program_fault_id,
            "program_fault_operation_index": config.program_fault_operation_index,
            "program_fault_qubits": config.program_fault_qubits,
            "localization_match": localization_match,
            "missing_evidence": diagnosis.missing_evidence if diagnosis else [],
            "anomaly_metric": (
                diagnosis.statistics_result.total_variation_distance
                if diagnosis and diagnosis.statistics_result else None
            ),
            "result_file": target.name,
        })

    source_hashes = {relative: _sha256(ROOT / relative) for relative in SOURCE_FILES}
    summary = {
        "protocol": PROTOCOL,
        "evaluation_scope": "18 seeded local simulator executions: repetition 0 from 6 fixed circuit families, with all 3 labels per family",
        "dataset_source": "experiments/full_dataset.py::build_family_configs; only selected family indices are constructed; no saved 270-report corpus is scanned",
        "ground_truth_source": "ExperimentConfig.ground_truth_label and explicit injected fault/profile metadata, set by the generator before execution; label is not passed to diagnosis",
        "family_indices": list(FAMILY_INDICES),
        "circuit_family_ids": [_family_name(index) for index in FAMILY_INDICES],
        "design_seed_base": DESIGN_SEED_BASE,
        "execution_seed_base": EXECUTION_SEED_BASE,
        "repetitions_per_scenario": 1,
        "shots_per_scenario": SHOTS,
        "threshold": 0.05,
        "execution": "LOCAL_SIMULATOR; seeded shot simulation with synthetic hardware profiles; no IBM hardware",
        "training_or_model_selection": False,
        "rule_based_diagnosis_only": True,
        "source_sha256": source_hashes,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "case_count": len(rows),
        "completed_count": sum(row["status"] == "COMPLETED" for row in rows),
        "metrics": _metrics(rows),
        "cases": rows,
        "limitations": [
            "Small deterministic synthetic sample; metrics are descriptive and not population estimates.",
            "Labels identify controlled simulator interventions, not independently adjudicated physical faults.",
            "The rule engine receives circuit and calibration evidence that the five-observable ML model does not; this is not a direct model comparison.",
            "Six family IDs are reserved as a prospective held-out set; no ML evaluation was performed here.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="New, nonexistent directory for run JSON and summary")
    args = parser.parse_args()
    summary = evaluate(args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "case_count": summary["case_count"], "completed_count": summary["completed_count"], "metrics": summary["metrics"]}, indent=2, sort_keys=True))
    return 0 if summary["completed_count"] == summary["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
