"""Checksum and split audit for the selected local grouped classical screen."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/experiments/local_grouped_classical_suite_eligible_20261029"
CONFIG_PATH = ROOT / "reports/experiments/qml_source_preprocessing_20261029/local_classical_suite_config.json"
RUNNER_PATH = ROOT / "experiments/run_grouped_classical_suite.py"
CONTRACTS = {
    "bugs4q_core": "bugs4q_core_source_state",
    "bugs4q_na": "bugs4q_na_source_state",
    "bugs4q_pooled": "bugs4q_joint_source_state",
    "qolumbina_ideal": "qolumbina_ideal_detection",
    "bugs4q_robust_execution": "bugs4q_robust_execution",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def calculate_metrics(y: list[int], pred: list[int]) -> dict:
    per = {}
    recalls, f1s = [], []
    for cls in (0, 1):
        tp = sum(a == cls and b == cls for a, b in zip(y, pred))
        fp = sum(a != cls and b == cls for a, b in zip(y, pred))
        fn = sum(a == cls and b != cls for a, b in zip(y, pred))
        support = sum(a == cls for a in y)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[str(cls)] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        recalls.append(recall); f1s.append(f1)
    return {"accuracy": sum(a == b for a, b in zip(y, pred)) / len(y),
            "balanced_accuracy": sum(recalls) / 2, "macro_f1": sum(f1s) / 2,
            "confusion_matrix": [[sum(a == i and b == j for a, b in zip(y, pred)) for j in (0, 1)] for i in (0, 1)]}


def main() -> None:
    issues = []
    task_audits = []
    input_hashes = {}
    for name, contract_name in CONTRACTS.items():
        contract_path = ROOT / f"experiments/source_task_contracts/{contract_name}.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        input_path = ROOT / contract["input_path"]
        input_hashes[str(input_path.relative_to(ROOT))] = sha256(input_path)
        with input_path.open(encoding="utf-8-sig", newline="") as stream:
            source_rows = list(csv.DictReader(stream))
        rid_col = contract["record_id_column"]
        group_col = contract["group_column"]
        label_col = contract["label_column"]
        source_by_id = {r[rid_col]: r for r in source_rows}
        if len(source_by_id) != len(source_rows):
            issues.append(f"{name}: duplicate source IDs")
        predictions_by_model = {}
        fold_by_record = {}
        task_dir = OUT / name
        pred_files = sorted(task_dir.glob("*_oof_predictions.csv"))
        if len(pred_files) != 20:
            issues.append(f"{name}: expected 20 completed model OOF files, found {len(pred_files)}")
        for pred_path in pred_files:
            with pred_path.open(encoding="utf-8-sig", newline="") as stream:
                pred_rows = list(csv.DictReader(stream))
            model = pred_path.name.removesuffix("_oof_predictions.csv")
            ids = [row["record_id"] for row in pred_rows]
            if len(ids) != len(set(ids)) or set(ids) != set(source_by_id):
                issues.append(f"{name}/{model}: OOF IDs do not exactly cover the source once")
            for row in pred_rows:
                source = source_by_id.get(row["record_id"])
                if source is None:
                    continue
                if int(row["y_true"]) != int(source[label_col]) or row["bug_id"] != source[group_col]:
                    issues.append(f"{name}/{model}: target/group mismatch at {row['record_id']}")
                fold = int(row["fold"])
                prior_fold = fold_by_record.setdefault(row["record_id"], fold)
                if prior_fold != fold:
                    issues.append(f"{name}/{model}: model OOF fold differs across model outputs")
            predictions_by_model[model] = pred_rows
        groups_to_folds = defaultdict(set)
        for rid, fold in fold_by_record.items():
            if rid in source_by_id:
                groups_to_folds[source_by_id[rid][group_col]].add(fold)
        if len(fold_by_record) != len(source_by_id):
            issues.append(f"{name}: common OOF fold assignments do not cover all source records")
        if any(len(folds) != 1 for folds in groups_to_folds.values()):
            issues.append(f"{name}: a source group crosses validation folds")
        if set(fold_by_record.values()) != {1, 2, 3, 4, 5}:
            issues.append(f"{name}: expected folds 1 through 5")
        summary_path = task_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        if summary.get("rows") != len(source_rows) or summary.get("groups") != len(groups_to_folds):
            issues.append(f"{name}: source totals disagree with summary")
        input_y = [int(r[label_col]) for r in source_rows]
        for model, pred_rows in predictions_by_model.items():
            sorted_pred = sorted(pred_rows, key=lambda r: r["record_id"])
            recalc = calculate_metrics([int(r["y_true"]) for r in sorted_pred],
                                       [int(r["y_pred"]) for r in sorted_pred])
            results_path = OUT / "classical_suite_results.csv"
            with results_path.open(encoding="utf-8-sig", newline="") as stream:
                result_rows = [r for r in csv.DictReader(stream) if r["task"] == name and r["model"] == model]
            if len(result_rows) != 1:
                issues.append(f"{name}/{model}: missing/duplicate aggregate metrics row")
            else:
                result = result_rows[0]
                for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                    if abs(float(result[metric]) - recalc[metric]) > 1e-12:
                        issues.append(f"{name}/{model}: stored {metric} disagrees with prediction replay")
        task_audits.append({"task": name, "rows": len(source_rows), "groups": len(groups_to_folds),
            "folds": sorted(set(fold_by_record.values())), "models": len(predictions_by_model),
            "source_sha256": input_hashes[str(input_path.relative_to(ROOT))],
            "runtime_seconds": summary.get("runtime_seconds"),
            "max_observed_process_rss_mib": summary.get("max_observed_process_rss_mib"),
            "summary_sha256": sha256(summary_path) if summary_path.is_file() else None})
    results_path = OUT / "classical_suite_results.csv"
    with results_path.open(encoding="utf-8-sig", newline="") as stream:
        results = list(csv.DictReader(stream))
    fold_metrics_path = OUT / "classical_suite_fold_metrics.csv"
    with fold_metrics_path.open(encoding="utf-8-sig", newline="") as stream:
        fold_metrics = list(csv.DictReader(stream))
    if len(results) != 100: issues.append(f"Expected 100 aggregate model/task rows, found {len(results)}")
    if len(fold_metrics) != 500: issues.append(f"Expected 500 fold/model metrics, found {len(fold_metrics)}")
    files = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name not in {"audit_manifest.json", "report.md"})
    artifact_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in files}
    report = {"schema": "local-grouped-classical-suite-audit-v1",
        "status": "VERIFIED" if not issues else "REVIEW_REQUIRED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_scope": "Five semantically eligible source tasks; independent GroupKFold classical screen, not a QNN-matched feature comparison.",
        "task_count": len(task_audits), "model_task_result_rows": len(results),
        "fold_model_metric_rows": len(fold_metrics), "tasks": task_audits,
        "input_sha256": input_hashes, "artifact_sha256": artifact_hashes,
        "execution_config": str(CONFIG_PATH.relative_to(ROOT)),
        "execution_config_sha256": sha256(CONFIG_PATH),
        "runner_sha256": sha256(RUNNER_PATH),
        "method": {"grouping": "GroupKFold by task-specific source issue/program group", "folds": 5,
            "text_features": "training-fold TF-IDF (1-2 grams, max_features=512), train-only TruncatedSVD to 4 components and MinMax scaling",
            "estimators_per_task": 20, "imbalance": "class-weighted estimators plus random over/under-sampling and continuous-space SMOTE fitted only on training folds",
            "models": sorted({row["model"] for row in results}), "frozen_test_accessed": False,
            "production_model_modified": False}, "issues": issues}
    (OUT / "audit_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Local grouped classical suite audit", "", f"Status: **{report['status']}**. Five tasks; 20 models per task; 100 aggregate rows; 500 fold/model rows.",
        "", "These are separate classical screens using training-only TF-IDF, SVD, scaling, and sampling. Their folds use GroupKFold by the source issue/program group. Results are not a matched QNN comparison because the classical representation differs from the pilot's hashed features.",
        "", "| Task | Rows | Groups | Runtime (s) | Max sampled RSS (MiB) |", "|---|---:|---:|---:|---:|"]
    for item in task_audits:
        lines.append(f"| {item['task']} | {item['rows']} | {item['groups']} | {item['runtime_seconds']:.2f} | {item['max_observed_process_rss_mib']:.1f} |")
    lines += ["", "All 20 OOF prediction files per task were checked for complete unique IDs, target/group agreement, group-exclusive folds, and aggregate metric replay.", ""]
    if issues:
        lines += ["## Issues", ""] + [f"- {x}" for x in issues]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "tasks": len(task_audits), "model_task_rows": len(results),
                      "fold_metric_rows": len(fold_metrics), "issues": issues,
                      "output": str(OUT.relative_to(ROOT))}, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
