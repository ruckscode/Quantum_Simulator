"""Replay metrics and verify grouped OOF integrity for a configured classical suite."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("QML_CLASSICAL_CONFIG_PATH",
    "reports/experiments/qml_source_preprocessing_20261029/classical_nine_source_v2_inputs/classical_nine_source_v2_config.json"))
if not CONFIG_PATH.is_absolute():
    CONFIG_PATH = ROOT / CONFIG_PATH
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUT = Path(os.environ.get("QML_CLASSICAL_OUTPUT", CONFIG["output"]))
if not OUT.is_absolute():
    OUT = ROOT / OUT


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def calculate_metrics(y: list[int], pred: list[int]) -> dict:
    per, recalls, f1s = {}, [], []
    for cls in (0, 1):
        tp = sum(a == cls and b == cls for a, b in zip(y, pred))
        fp = sum(a != cls and b == cls for a, b in zip(y, pred))
        support = sum(a == cls for a in y)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[str(cls)] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        recalls.append(recall); f1s.append(f1)
    if not y:
        raise ValueError("Cannot calculate metrics on an empty prediction set")
    return {"accuracy": sum(a == b for a, b in zip(y, pred)) / len(y),
        "balanced_accuracy": sum(recalls) / 2, "macro_f1": sum(f1s) / 2,
        "confusion_matrix": [[sum(a == i and b == j for a, b in zip(y, pred)) for j in (0, 1)] for i in (0, 1)],
        "per_class": per}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    issues, tasks, source_hashes = [], [], {}
    result_path, fold_path = OUT / "classical_suite_results.csv", OUT / "classical_suite_fold_metrics.csv"
    result_rows, fold_rows = read_csv(result_path), read_csv(fold_path)
    for task in CONFIG["tasks"]:
        name = task["name"]
        input_path = Path(task["input"])
        if not input_path.is_absolute(): input_path = ROOT / input_path
        source_hashes[str(input_path.relative_to(ROOT))] = sha256(input_path)
        if task.get("sha256") and source_hashes[str(input_path.relative_to(ROOT))] != task["sha256"]:
            issues.append(f"{name}: adapter hash mismatch")
        source_rows = read_csv(input_path)
        source_by_id = {row["record_id"]: row for row in source_rows}
        if len(source_by_id) != len(source_rows): issues.append(f"{name}: duplicate source IDs")
        pred_files = sorted((OUT / name).glob("*_oof_predictions.csv"))
        if len(pred_files) != 20: issues.append(f"{name}: expected 20 OOF files, found {len(pred_files)}")
        fold_by_id = {}
        task_model_count = 0
        for pred_path in pred_files:
            model = pred_path.name.removesuffix("_oof_predictions.csv")
            predictions = read_csv(pred_path)
            ids = [row["record_id"] for row in predictions]
            if len(ids) != len(set(ids)) or set(ids) != set(source_by_id):
                issues.append(f"{name}/{model}: OOF IDs not covered exactly once")
            task_model_count += 1
            for row in predictions:
                original = source_by_id.get(row["record_id"])
                if original is None: continue
                if int(row["y_true"]) != int(original["label"]) or row["bug_id"] != original["bug_id"]:
                    issues.append(f"{name}/{model}: label or group mismatch for {row['record_id']}")
                fold = int(row["fold"])
                previous = fold_by_id.setdefault(row["record_id"], fold)
                if previous != fold: issues.append(f"{name}/{model}: common folds differ across outputs")
            aggregate = [row for row in result_rows if row["task"] == name and row["model"] == model]
            if len(aggregate) != 1: issues.append(f"{name}/{model}: missing/duplicate aggregate row")
            else:
                score = calculate_metrics([int(row["y_true"]) for row in predictions],
                                          [int(row["y_pred"]) for row in predictions])
                for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                    if abs(float(aggregate[0][metric]) - score[metric]) > 1e-12:
                        issues.append(f"{name}/{model}: aggregate {metric} replay mismatch")
            for fold in range(1, 6):
                fold_pred = [row for row in predictions if int(row["fold"]) == fold]
                saved = [row for row in fold_rows if row["task"] == name and row["model"] == model and int(row["fold"]) == fold]
                if len(saved) != 1: issues.append(f"{name}/{model}/fold{fold}: missing/duplicate fold metric")
                elif fold_pred:
                    score = calculate_metrics([int(row["y_true"]) for row in fold_pred],
                                              [int(row["y_pred"]) for row in fold_pred])
                    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                        if abs(float(saved[0][metric]) - score[metric]) > 1e-12:
                            issues.append(f"{name}/{model}/fold{fold}: {metric} replay mismatch")
        groups_to_folds = defaultdict(set)
        for rid, fold in fold_by_id.items():
            if rid in source_by_id: groups_to_folds[source_by_id[rid]["bug_id"]].add(fold)
        if set(fold_by_id) != set(source_by_id): issues.append(f"{name}: common fold assignment coverage mismatch")
        if any(len(folds) != 1 for folds in groups_to_folds.values()): issues.append(f"{name}: group crosses validation folds")
        if set(fold_by_id.values()) != {1, 2, 3, 4, 5}: issues.append(f"{name}: expected folds 1..5")
        matching_results = [row for row in result_rows if row["task"] == name]
        if len(matching_results) != 20: issues.append(f"{name}: expected 20 aggregate result rows, got {len(matching_results)}")
        matching_folds = [row for row in fold_rows if row["task"] == name]
        if len(matching_folds) != 100: issues.append(f"{name}: expected 100 fold metric rows, got {len(matching_folds)}")
        task_summary = OUT / name / "summary.json"
        summary = json.loads(task_summary.read_text(encoding="utf-8")) if task_summary.is_file() else {}
        if summary.get("rows") != len(source_rows) or summary.get("groups") != len(groups_to_folds):
            issues.append(f"{name}: task summary count mismatch")
        tasks.append({"task": name, "rows": len(source_rows), "groups": len(groups_to_folds),
            "models": task_model_count, "folds": sorted(set(fold_by_id.values())),
            "input_sha256": source_hashes[str(input_path.relative_to(ROOT))],
            "summary_sha256": sha256(task_summary) if task_summary.is_file() else None,
            "runtime_seconds": summary.get("runtime_seconds"),
            "max_observed_process_rss_mib": summary.get("max_observed_process_rss_mib")})
    if len(result_rows) != len(CONFIG["tasks"]) * 20: issues.append("aggregate total row count mismatch")
    if len(fold_rows) != len(CONFIG["tasks"]) * 20 * 5: issues.append("fold total row count mismatch")
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(OUT.rglob("*"))
              if path.is_file() and path.name not in {"audit_manifest.json", "report.md"}}
    report = {"schema": "nine-source-grouped-classical-audit-v1",
        "status": "VERIFIED" if not issues else "REVIEW_REQUIRED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_count": len(tasks), "model_task_result_rows": len(result_rows),
        "fold_model_metric_rows": len(fold_rows), "tasks": tasks,
        "input_sha256": source_hashes, "artifact_sha256": hashes,
        "config_path": str(CONFIG_PATH.relative_to(ROOT)), "config_sha256": sha256(CONFIG_PATH),
        "runner_sha256": sha256(ROOT / "experiments/run_grouped_classical_suite.py"),
        "evaluation_scope": "Independent grouped classical CV using 20 estimators and training-fold-only imbalance methods; not matched QNN features/splits.",
        "frozen_test_accessed": False, "issues": issues}
    target = OUT / "audit_manifest.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Nine-source grouped classical suite audit", "",
        f"Status: **{report['status']}**; {report['task_count']} tasks; {report['model_task_result_rows']} model/task summaries; {report['fold_model_metric_rows']} fold/model metrics.",
        "", "Each task's 20 OOF files were checked for exact unique ID coverage, label/group agreement, group-exclusive five-fold assignments, and aggregate plus per-fold metric replay.",
        "The suite uses independent GroupKFold assignments and is not a matched QNN comparison. LintQ and QSimBench remain exploratory under their documented source caveats.", "",
        "| Task | Rows | Groups | Models | Runtime (s) | Sampled peak RSS (MiB) |", "|---|---:|---:|---:|---:|---:|"]
    for item in tasks:
        runtime = "n/a" if item["runtime_seconds"] is None else f"{item['runtime_seconds']:.2f}"
        rss = "n/a" if item["max_observed_process_rss_mib"] is None else f"{item['max_observed_process_rss_mib']:.1f}"
        lines.append(f"| {item['task']} | {item['rows']} | {item['groups']} | {item['models']} | {runtime} | {rss} |")
    if issues: lines += ["", "## Issues", ""] + [f"- {issue}" for issue in issues]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "tasks": len(tasks),
        "aggregate_rows": len(result_rows), "fold_rows": len(fold_rows),
        "issues": issues, "output": str(OUT.relative_to(ROOT))}, indent=2))
    if issues: raise SystemExit(1)


if __name__ == "__main__":
    main()
