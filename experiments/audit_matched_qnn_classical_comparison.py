"""Verify matched hashed-feature QNN/classical comparisons and write a report."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "reports/experiments/qml_standalone_accuracy_optimization_20261028/remaining_task_sweeps"
QNN_AUDIT = SWEEP / "candidate_sweep_artifact_audit.json"
CLASSICAL_ROOT = SWEEP / "matched_classical_controls"
OUT_JSON = SWEEP / "matched_qnn_classical_comparison.json"
OUT_MD = SWEEP / "matched_qnn_classical_comparison.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def metric(y: list[int], pred: list[int]) -> dict:
    labels = sorted(set(y) | set(pred))
    f1, recall, per = [], [], {}
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y, pred))
        fp = sum(a != label and b == label for a, b in zip(y, pred))
        fn = sum(a == label and b != label for a, b in zip(y, pred))
        support = sum(a == label for a in y)
        precision = tp / (tp + fp) if tp + fp else 0.0
        r = tp / support if support else 0.0
        f = 2 * precision * r / (precision + r) if precision + r else 0.0
        per[str(label)] = {"precision": precision, "recall": r, "f1": f, "support": support}
        f1.append(f)
        recall.append(r)
    return {"accuracy": sum(a == b for a, b in zip(y, pred)) / len(y),
            "macro_f1": sum(f1) / len(labels), "balanced_accuracy": sum(recall) / len(labels),
            "per_class": per}


def confusion_by_group(rows: list[dict[str, str]], group_field: str) -> tuple[list[str], np.ndarray]:
    groups = sorted({str(row[group_field]) for row in rows})
    counts = np.zeros((len(groups), 4), dtype=np.int64)
    index = {group: i for i, group in enumerate(groups)}
    for row in rows:
        y, pred = int(row["y_true"]), int(row["y_pred"])
        slot = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}[(y, pred)]
        counts[index[str(row[group_field])], slot] += 1
    return groups, counts


def grouped_bootstrap_delta(left_rows: list[dict[str, str]], right_rows: list[dict[str, str]],
                            rng: np.random.Generator, replicates: int = 1000) -> list[float]:
    groups, left = confusion_by_group(left_rows, "bug_id")
    right_groups, right = confusion_by_group(right_rows, "group_id")
    if groups != right_groups:
        raise AssertionError("Paired group bootstrap requires identical group IDs")
    deltas = np.empty(replicates, dtype=np.float64)
    batch = max(1, min(100, 200_000 // max(1, len(groups))))
    cursor = 0
    while cursor < replicates:
        n = min(batch, replicates - cursor)
        draws = rng.integers(0, len(groups), size=(n, len(groups)))
        l = left[draws].sum(axis=1)
        r = right[draws].sum(axis=1)
        def macro_f1(cm):
            tn, fp, fn, tp = (cm[:, i] for i in range(4))
            f0 = np.divide(2*tn, 2*tn+fp+fn, out=np.zeros_like(tn, dtype=float), where=(2*tn+fp+fn)!=0)
            f1 = np.divide(2*tp, 2*tp+fp+fn, out=np.zeros_like(tp, dtype=float), where=(2*tp+fp+fn)!=0)
            return (f0 + f1) / 2
        deltas[cursor:cursor+n] = macro_f1(l) - macro_f1(r)
        cursor += n
    return [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))]


def main() -> None:
    qnn_audit = json.loads(QNN_AUDIT.read_text(encoding="utf-8-sig"))
    baseline_campaign = json.loads((CLASSICAL_ROOT / "campaign_summary.json").read_text(encoding="utf-8-sig"))
    if qnn_audit.get("status") != "verified" or baseline_campaign.get("status") != "completed_matched_classical_grouped_sweeps":
        raise ValueError("QNN or matched classical campaign is incomplete")
    qnn_by_task = {row["task"]: row for row in qnn_audit["tasks"]}
    task_labels = {
        "bugs4q_core": "Bugs4Q core", "bugs4q_na": "Bugs4Q-NA",
        "bugs4q_pooled": "Bugs4Q pooled", "bugs4q_core_only_transfer": "Bugs4Q core-only transfer",
        "bugs4q_na_only_transfer": "Bugs4Q-NA-only transfer", "lintq": "LintQ",
        "qbugs_hornbro_subset": "QBugs HornBro subset", "qolumbina_ideal": "Qolumbina ideal",
        "quantum_platform_comment_task": "Quantum platform comment task",
    }
    baseline_results = {}
    for row in baseline_campaign["results"]:
        baseline_results.setdefault(row["task"], []).append(row)

    task_results, comparison_rows = [], []
    rng = np.random.default_rng(20261029)
    for task_name, qnn_task in qnn_by_task.items():
        baseline_task_name = task_labels[task_name]
        task_dir = CLASSICAL_ROOT / task_name
        summary_path, pred_path, fold_path = task_dir / "summary.json", task_dir / "oof_predictions.csv", task_dir / "fold_metrics.csv"
        if not all(path.is_file() for path in (summary_path, pred_path, fold_path)):
            raise FileNotFoundError(f"Missing matched classical artifacts for {task_name}")
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        predictions, folds = read_csv(pred_path), read_csv(fold_path)
        if summary.get("frozen_test_accessed") is not False or summary.get("folds") != 5:
            raise AssertionError(f"Unexpected evaluation scope for {task_name}")
        expected_ids = {row["record_id"] for row in read_csv(SWEEP / task_name / "oof_predictions.csv")}
        actual_ids = {row["record_id"] for row in predictions}
        if actual_ids != expected_ids:
            raise AssertionError(f"Matched baseline record IDs differ for {task_name}")
        group_folds: dict[str, set[int]] = {}
        for row in predictions:
            group_folds.setdefault(str(row["group_id"]), set()).add(int(row["fold"]))
        if any(len(value) != 1 for value in group_folds.values()):
            raise AssertionError(f"Matched classical group leakage in {task_name}")
        task_models = []
        qnn_task_result = []
        qnn_names = {4: "angle_reupload_4q_depth2_lr0035", 8: "amplitude_3q_depth1_lr004"}
        for width in (4, 8):
            qnn = next(model for model in qnn_task["models"] if model["model"] == qnn_names[width])
            controls = []
            for baseline in baseline_results[baseline_task_name]:
                if int(baseline["feature_count"]) != width:
                    continue
                model_name = baseline["model"]
                rows = [row for row in predictions if row["model"] == model_name and int(row["feature_count"]) == width]
                if len(rows) != len(expected_ids) or len({row["record_id"] for row in rows}) != len(expected_ids):
                    raise AssertionError(f"Baseline OOF row coverage mismatch: {task_name}/{width}/{model_name}")
                recalculated = metric([int(row["y_true"]) for row in rows], [int(row["y_pred"]) for row in rows])
                for key in ("accuracy", "macro_f1", "balanced_accuracy"):
                    if abs(recalculated[key] - float(baseline[key])) > 1e-12:
                        raise AssertionError(f"Baseline metric mismatch: {task_name}/{width}/{model_name}/{key}")
                fold_rows = [row for row in folds if row["model"] == model_name and int(row["feature_count"]) == width]
                if len(fold_rows) != 5:
                    raise AssertionError(f"Expected 5 baseline folds: {task_name}/{width}/{model_name}")
                macro_values = [float(row["macro_f1"]) for row in fold_rows]
                control = {"model": model_name, **recalculated,
                           "fold_macro_f1_mean": sum(macro_values) / len(macro_values),
                           "fold_macro_f1_sd": (sum((x - sum(macro_values)/len(macro_values))**2 for x in macro_values) / len(macro_values))**0.5,
                           "folds": 5}
                controls.append(control)
            if len(controls) != 5:
                raise AssertionError(f"Expected five controls for {task_name}/{width} features")
            qnn_rows = read_csv(SWEEP / task_name / "oof_predictions.csv")
            qnn_rows = [row for row in qnn_rows if row["model"] == qnn_names[width]]
            if len(qnn_rows) != len(expected_ids) or len({row["record_id"] for row in qnn_rows}) != len(expected_ids):
                raise AssertionError(f"QNN OOF alignment failure for {task_name}/{width}")
            for control in controls:
                right_rows = [row for row in predictions if row["model"] == control["model"] and int(row["feature_count"]) == width]
                control["paired_group_bootstrap_macro_f1_delta_95ci"] = grouped_bootstrap_delta(qnn_rows, right_rows, rng)
                control["bootstrap_replicates"] = 1000
            qnn_controls = {"task": baseline_task_name, "feature_count": width,
                            "qnn_model": qnn_names[width], "qnn_metrics": qnn["metrics"],
                            "qnn_minus_controls_macro_f1": [
                                {"control": control["model"],
                                 "delta": qnn["metrics"]["macro_f1"] - control["macro_f1"]}
                                for control in controls],
                            "classical_controls": controls}
            qnn_task_result.append(qnn_controls)
            comparison_rows.append(qnn_controls)
            task_models.extend(controls)
        task_results.append({"task": baseline_task_name, "records": qnn_task["records"],
                             "groups": qnn_task["groups"], "comparisons": qnn_task_result,
                             "baseline_prediction_sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest(),
                             "baseline_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest()})

    result = {"schema": "matched-qnn-classical-hash-feature-comparison-v1", "status": "verified",
              "task_count": len(task_results), "tasks": task_results,
              "primary_metric": "pooled grouped OOF macro F1; five outer group folds",
              "inputs": "All candidate/control models use the same deterministic 8-bin BLAKE2b token counts and fold-local min-max scaling; angle QNN is compared to 4-feature controls, amplitude QNN to 8-feature controls.",
              "classical_models": ["majority_dummy", "logistic_balanced", "decision_tree_balanced", "random_forest_balanced", "extra_trees_balanced"],
              "selection_protocol": "Fixed model configurations; no hyperparameter tuning on validation folds. Reporting the complete control set; no deployment model selected.",
              "limitations": ["Development OOF only; no independent confirmation set.",
                             "Four-epoch QNN candidate training is exploratory and uses a compact hashed-text representation.",
                             "Qolumbina is a seeded-mutant study, LintQ is warning-level TP/FP triage, and the quantum-platform task is issue-comment annotation; targets are not interchangeable."],
              "frozen_test_accessed": False, "production_model_modified": False}
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Matched QNN and classical comparison — 2026-10-29", "",
             "All entries are five-fold grouped development OOF results. Each classical model used the same fixed hashed-text feature representation, fold assignments, and eligible training rows as its QNN counterpart. No frozen test was accessed.", "",
             "| Task | QNN | Features | QNN macro F1 | Best fixed classical macro F1 | QNN minus control (pp) |",
             "|---|---|---:|---:|---:|---:|"]
    for comparison in comparison_rows:
        controls = comparison["classical_controls"]
        best = max(controls, key=lambda item: item["macro_f1"])
        delta = comparison["qnn_metrics"]["macro_f1"] - best["macro_f1"]
        lines.append(f"| {comparison['task']} | {comparison['qnn_model']} | {comparison['feature_count']} | {comparison['qnn_metrics']['macro_f1']:.3f} | {best['model']} {best['macro_f1']:.3f} | {delta*100:+.2f} |")
    lines += ["", "The ‘best fixed classical’ column is the highest OOF score among five predeclared controls and is shown descriptively; it is not an independently selected final model. Per-class metrics, fold-level scores, confusion matrices, and each control result are in `matched_qnn_classical_comparison.json`.", "",
              "The QNN configurations are 3-qubit depth-1 amplitude encoding of eight hashed features and 4-qubit depth-2 angle re-uploading of four features. Both use deterministic statevector simulation with four training epochs and fixed Adam settings. Classical controls are majority dummy, balanced Logistic Regression, balanced depth-4 Decision Tree, balanced Random Forest, and balanced Extra Trees.", "",
              "These small source-specific tasks support a development comparison only. Qolumbina labels seeded mutation outcomes; LintQ labels warning triage; QSimBench elsewhere in the project labels backend identity. None implies physical hardware fault attribution.", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "verified", "tasks": len(task_results),
                      "comparison_rows": len(comparison_rows), "json": str(OUT_JSON.relative_to(ROOT)),
                      "markdown": str(OUT_MD.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
