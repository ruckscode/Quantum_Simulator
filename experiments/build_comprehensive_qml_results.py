"""Recompute leakage-safe OOF metrics and build auditable campaign summary tables.

Uses Python's standard library only so the registry can be built below the model
training memory floor. It reads development OOF files and never opens frozen-test
prediction files.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import struct
import zipfile
from array import array
import ast
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028"
THREE = ROOT / "reports/experiments/qmutbench_three_class_qnn_campaign_20261027"

RUNS = [
    ("Bugs4Q core", "source-state fixed vs buggy", CAMPAIGN / "bugs4q_standalone/qnn_gradient_fix/oof_predictions.csv", CAMPAIGN / "bugs4q_standalone/qnn_gradient_fix/summary.json", "bug_id", "issue-group 5-fold"),
    ("Bugs4Q core classical text/context", "source-state fixed vs buggy", CAMPAIGN / "bugs4q_standalone/oof_predictions.csv", CAMPAIGN / "bugs4q_standalone/standalone_summary.json", "bug_id", "issue-group 5-fold"),
    ("Bugs4Q-NA", "source-state fixed vs buggy", CAMPAIGN / "bugs4q_na/qnn/oof_predictions.csv", CAMPAIGN / "bugs4q_na/qnn/summary.json", "bug_id", "issue-group 5-fold"),
    ("Bugs4Q pooled", "source-state fixed vs buggy", CAMPAIGN / "bugs4q_joint/qnn_hybrid/oof_predictions.csv", CAMPAIGN / "bugs4q_joint/qnn_hybrid/summary.json", "bug_id", "shared issue-group 5-fold"),
    ("Bugs4Q core-only transfer", "source-state fixed vs buggy", CAMPAIGN / "bugs4q_joint/qnn_train_core_complete/oof_predictions.csv", CAMPAIGN / "bugs4q_joint/qnn_train_core_complete/summary.json", "bug_id", "shared issue-group 5-fold; training restricted to core"),
    ("Bugs4Q-NA-only transfer", "source-state fixed vs buggy", CAMPAIGN / "bugs4q_joint/qnn_train_na_complete/oof_predictions.csv", CAMPAIGN / "bugs4q_joint/qnn_train_na_complete/summary.json", "bug_id", "shared issue-group 5-fold; training restricted to NA"),
    ("QBugs HornBro subset", "correct vs buggy circuit", CAMPAIGN / "qbugs/qnn_hybrid/oof_predictions.csv", CAMPAIGN / "qbugs/qnn_hybrid/summary.json", "bug_id", "paired circuit 5-fold"),
    ("Quantum platform comment task", "bug comment vs false-positive comment", CAMPAIGN / "quantum_platform_223_comment_only/qnn_hybrid/oof_predictions.csv", CAMPAIGN / "quantum_platform_223_comment_only/qnn_hybrid/summary.json", "bug_id", "commit-group 5-fold"),
    ("Qolumbina ideal", "seeded mutant detected vs not detected", CAMPAIGN / "qolumbina_detection_ideal/qnn/oof_predictions.csv", CAMPAIGN / "qolumbina_detection_ideal/qnn/summary.json", "bug_id", "program-group 5-fold"),
    ("LintQ", "manual TP vs FP warning", CAMPAIGN / "lintq_warning_triage/qnn/oof_predictions.csv", CAMPAIGN / "lintq_warning_triage/qnn/summary.json", "bug_id", "issue/file-group 5-fold"),
    ("Bugs4Q-NA", "source-state fixed vs buggy; equal probability ensemble diagnostics", CAMPAIGN / "bugs4q_na/qnn/ensemble_oof_predictions.csv", CAMPAIGN / "bugs4q_na/qnn/ensemble_summary.json", "bug_id", "issue-group 5-fold OOF diagnostics"),
    ("Bugs4Q pooled", "source-state fixed vs buggy; equal probability ensemble diagnostics", CAMPAIGN / "bugs4q_joint/qnn_hybrid/ensemble_oof_predictions.csv", CAMPAIGN / "bugs4q_joint/qnn_hybrid/ensemble_summary.json", "bug_id", "shared issue-group 5-fold OOF diagnostics"),
    ("Bugs4Q core-only transfer", "source-state fixed vs buggy; equal probability ensemble diagnostics", CAMPAIGN / "bugs4q_joint/qnn_train_core_complete/ensemble_oof_predictions.csv", CAMPAIGN / "bugs4q_joint/qnn_train_core_complete/ensemble_summary.json", "bug_id", "shared issue-group 5-fold OOF diagnostics"),
    ("Bugs4Q-NA-only transfer", "source-state fixed vs buggy; equal probability ensemble diagnostics", CAMPAIGN / "bugs4q_joint/qnn_train_na_complete/ensemble_oof_predictions.csv", CAMPAIGN / "bugs4q_joint/qnn_train_na_complete/ensemble_summary.json", "bug_id", "shared issue-group 5-fold OOF diagnostics"),
    ("Qolumbina ideal", "seeded mutant detected vs not detected; equal probability diagnostics", CAMPAIGN / "qolumbina_detection_ideal/qnn/ensemble_oof_predictions.csv", CAMPAIGN / "qolumbina_detection_ideal/qnn/ensemble_summary.json", "bug_id", "program-group 5-fold OOF diagnostics"),
    ("LintQ", "manual TP vs FP warning; equal probability diagnostics", CAMPAIGN / "lintq_warning_triage/qnn/ensemble_oof_predictions.csv", CAMPAIGN / "lintq_warning_triage/qnn/ensemble_summary.json", "bug_id", "issue/file-group 5-fold OOF diagnostics"),
    ("QMutBench three-class research", "Survivor vs WOO vs OPO", THREE / "predictions.csv.gz", THREE / "campaign_summary.json", "origin_program", "origin-group 3-fold"),
    ("QMutBench three-class research", "Survivor vs WOO vs OPO; amplitude encoding arm", THREE / "amplitude_encoding/predictions.csv.gz", THREE / "amplitude_encoding/campaign_summary.json", "origin_program", "origin-group 3-fold"),
    ("QMutBench three-class research", "Survivor vs WOO vs OPO; trainable readout arm", THREE / "trainable_readout/predictions.csv.gz", THREE / "trainable_readout/campaign_summary.json", "origin_program", "origin-group 3-fold"),
]


def open_csv(path: Path):
    return gzip.open(path, "rt", newline="", encoding="utf-8") if path.suffix == ".gz" else path.open("r", newline="", encoding="utf-8-sig")


def open_any(path: Path):
    with open_csv(path) as f:
        yield from csv.DictReader(f)


def metric_from_confusion(cm: dict[tuple[int, int], int], labels: list[int]) -> dict:
    n = sum(cm.values())
    correct = sum(cm[(c, c)] for c in labels)
    per = {}
    recalls = []
    f1s = []
    for c in labels:
        tp = cm[(c, c)]
        fp = sum(cm[(other, c)] for other in labels if other != c)
        fn = sum(cm[(c, other)] for other in labels if other != c)
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[str(c)] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        recalls.append(recall)
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(labels) if labels else 0.0
    balanced = sum(recalls) / len(labels) if labels else 0.0
    weighted = sum(per[str(c)]["f1"] * per[str(c)]["support"] for c in labels) / n if n else 0.0
    return {"records": n, "accuracy": correct / n if n else 0.0,
            "balanced_accuracy": balanced, "macro_f1": macro_f1,
            "weighted_f1": weighted, "per_class": per,
            "confusion_matrix_labels": labels,
            "confusion_matrix": [[cm[(a, b)] for b in labels] for a in labels]}


def family(model: str, declared: str = "") -> str:
    if "vqc" in model or "qnn" in model or "hybrid" in model:
        return "QNN/hybrid"
    if model.startswith("quantum_") or "ensemble" in model:
        return "Ensemble"
    if declared in {"classical", "quantum_or_hybrid"}:
        return "Classical" if declared == "classical" else "QNN/hybrid"
    return "Classical"


def feature_description(source: str) -> str:
    if "QMutBench binary" in source:
        return "23 mutation-aware source/QASM-difference features; fit per development protocol"
    if "QMutBench three-class" in source:
        return "Four fold-selected/scaled metadata and mutation-difference features"
    if "Bugs4Q core classical" in source:
        return "TF-IDF text and four-dimensional TF-IDF/SVD control arms; training-fold fitted"
    if source.startswith("Bugs4Q"):
        return "Comment/docstring-stripped source reduced to four scaled SVD features; fold-fitted"
    if "QBugs" in source:
        return "Circuit source plus derived circuit scalar features; pair-grouped"
    if "Quantum platform" in source:
        return "Comment text only with explicit label words masked; four-dimensional fold-fitted representation"
    if "Qolumbina" in source:
        return "Seeded mutant source with comments/docstrings stripped plus ideal backend token; four-dimensional fold-fitted representation"
    if "LintQ" in source:
        return "Warning rule, message, and local code context with comments/docstrings stripped; four-dimensional fold-fitted representation"
    return "See task_definition.json, normalized manifest, and run summary"


def model_configuration(model: str, summary: dict, ref: dict) -> str:
    detail = {"model": model}
    if "vqc" in model:
        detail.update({"encoding": "amplitude" if "amplitude" in model else "angle",
                       "steps": summary.get("optimizer_steps_per_run", 15),
                       "device": summary.get("device", "PennyLane analytic statevector")})
    elif model == "logistic_4d": detail.update({"C": 1.0, "class_weight": "balanced"})
    elif model in {"random_forest_4d", "extra_trees_4d"}: detail.update({"trees": 100, "max_depth": 6, "min_samples_leaf": 2, "class_weight": "balanced"})
    elif model == "mlp_4d": detail.update({"hidden_layers": [8], "activation": "tanh", "max_iter": 200})
    elif model == "ExtraTrees_100_mutation_aware_23f":
        try:
            metadata = json.loads((ROOT / "data/qmutbench_ml/master_optimization_v1/best_model_metadata.json").read_text(encoding="utf-8"))
            detail["parameters"] = metadata.get("model_parameters", {})
            detail["threshold"] = metadata.get("inference_threshold")
        except Exception: pass
    return json.dumps(detail, sort_keys=True)


def process_run(source: str, task: str, pred_path: Path, summary_path: Path,
                group_field: str, split: str) -> list[dict]:
    if not pred_path.is_file():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    summary_results = summary if isinstance(summary, list) else summary.get("results", [])
    meta_results = {str(x.get("model")): x for x in summary_results}
    declared = {}
    if source == "Bugs4Q core classical text/context":
        declared = {str(x.get("model")): "classical" for x in summary_results}
    # Streaming aggregation, grouped by model. OOF tables in this campaign have
    # a small bounded model count, so counts—not row-level features—are retained.
    stats = defaultdict(lambda: {"cm": Counter(), "groups": set(), "folds": set(),
                                 "records": 0, "targets": Counter(), "duplicates": 0,
                                 "seen": set(), "probability_metrics": None})
    columns = set()
    for row in open_any(pred_path):
        columns.update(row)
        model = str(row.get("model", "unknown"))
        st = stats[model]
        y, pred = int(row["y_true"]), int(row["y_pred"])
        st["cm"][(y, pred)] += 1
        st["records"] += 1
        st["targets"][y] += 1
        if row.get(group_field): st["groups"].add(row[group_field])
        if row.get("fold"): st["folds"].add(row["fold"])
        if row.get("record_id"):
            rid = str(row["record_id"])
            if rid in st["seen"]: st["duplicates"] += 1
            else: st["seen"].add(rid)
    out = []
    for model, st in stats.items():
        labels = sorted(set(st["targets"]) | {b for (a, b) in st["cm"]})
        calc = metric_from_confusion(st["cm"], labels)
        ref = meta_results.get(model, {})
        if not ref and isinstance(summary, dict):
            for x in summary.get("leaderboard", []):
                if str(x.get("model")) == model:
                    ref = x; break
        runtime = ref.get("measured_fit_runtime_seconds_sum", ref.get("runtime_seconds_total", ""))
        fold_path = pred_path.parent / "fold_metrics.json"
        fold_model_rows = []
        if runtime == "" and fold_path.exists():
            try:
                fr = json.loads(fold_path.read_text(encoding="utf-8"))
                if isinstance(fr, dict): fr = fr.get("fold_metrics", [])
                fold_model_rows = [x for x in fr if str(x.get("model")) == model]
                runtimes = [float(x.get("runtime_seconds", 0)) for x in fold_model_rows if x.get("runtime_seconds") is not None]
                if runtimes: runtime = sum(runtimes)
            except (ValueError, json.JSONDecodeError): pass
        elif fold_path.exists():
            try:
                fr = json.loads(fold_path.read_text(encoding="utf-8"))
                if isinstance(fr, dict): fr = fr.get("fold_metrics", [])
                fold_model_rows = [x for x in fr if str(x.get("model")) == model]
            except (ValueError, json.JSONDecodeError): pass
        fold_f1 = [float(x["macro_f1"]) for x in fold_model_rows if x.get("macro_f1") is not None]
        fold_acc = [float(x["accuracy"]) for x in fold_model_rows if x.get("accuracy") is not None]
        fam = family(model, declared.get(model, str(ref.get("family", ""))))
        path_rel = str(pred_path.relative_to(ROOT))
        summary_rel = str(summary_path.relative_to(ROOT)) if summary_path.is_file() else ""
        qnn_expected = int(summary.get("gradient_checks_expected", 0)) if isinstance(summary, dict) else 0
        qnn_passed = int(summary.get("gradient_checks_passed", 0)) if isinstance(summary, dict) else 0
        result = {
            "dataset_task": source, "task_semantics": task, "model": model, "model_family": fam,
            "feature_representation": feature_description(source),
            "model_configuration": model_configuration(model, summary if isinstance(summary, dict) else {}, ref),
            "validation": split, "group_unit": group_field,
            "records": st["records"], "unique_records": len(st["seen"]),
            "independent_groups": len(st["groups"]), "folds": len(st["folds"]),
            "class_counts": json.dumps({str(k): v for k, v in sorted(st["targets"].items())}, sort_keys=True),
            "accuracy": calc["accuracy"], "balanced_accuracy": calc["balanced_accuracy"],
            "macro_f1": calc["macro_f1"], "weighted_f1": calc["weighted_f1"],
            "per_class_metrics": json.dumps(calc["per_class"], sort_keys=True),
            "confusion_matrix": json.dumps(calc["confusion_matrix"]),
            "runtime_seconds": runtime,
            "fold_macro_f1_mean": sum(fold_f1) / len(fold_f1) if fold_f1 else "",
            "fold_macro_f1_sd": (math.sqrt(sum((x - sum(fold_f1) / len(fold_f1)) ** 2 for x in fold_f1) / (len(fold_f1) - 1)) if len(fold_f1) > 1 else ""),
            "fold_accuracy_mean": sum(fold_acc) / len(fold_acc) if fold_acc else "",
            "bootstrap_95ci_macro_f1": json.dumps(ref.get("bug_group_bootstrap_95ci_macro_f1", "")),
            "imbalance_method": "class_weight=balanced where recorded; otherwise see config/summary",
            "smote_status": "NOT_EVALUATED", "auc_probability_metrics": "not recomputed in this registry",
            "checkpoint_count": summary.get("resource_use", {}).get("checkpoint_count", "") if isinstance(summary, dict) else "",
            "sampled_peak_rss_mib": summary.get("resource_use", {}).get("sampled_peak_training_process_rss_mib", "") if isinstance(summary, dict) else "",
            "checkpoint_path": str((pred_path.parent / "fold_checkpoints.json").relative_to(ROOT)) if (pred_path.parent / "fold_checkpoints.json").exists() else ("data/qmutbench_ml/master_optimization_v1/best_model.joblib" if "QMutBench binary" in source else ""),
            "gradient_checks_passed": qnn_passed if fam == "QNN/hybrid" else "",
            "gradient_checks_expected": qnn_expected if fam == "QNN/hybrid" else "",
            "oof_prediction_path": path_rel, "summary_path": summary_rel,
            "source_version_license": "see dataset_provenance.json and verified_dataset_inventory.json",
            "status": ("OOF_ENSEMBLE_DIAGNOSTIC" if "ensemble" in source.lower() else
                       "COMPLETE_OOF" if isinstance(summary, dict) and summary.get("status") == "completed_grouped_supervised_qnn_benchmark" or source == "Bugs4Q core classical text/context" else "RECALCULATED_OOF"),
            "duplicate_prediction_rows": st["duplicates"],
            "metric_recalculation_agreement_macro_f1": (abs(calc["macro_f1"] - float(ref.get("macro_f1"))) if ref.get("macro_f1") is not None else ""),
        }
        out.append(result)
    return out


def read_npy_numeric(zf: zipfile.ZipFile, name: str, typecode: str, itemsize: int):
    with zf.open(name) as f:
        prefix = f.read(8)
        if prefix[:6] != b"\x93NUMPY":
            raise ValueError(f"Invalid NPY member {name}")
        major = prefix[6]
        if major == 1:
            header_len = struct.unpack("<H", f.read(2))[0]
        else:
            header_len = struct.unpack("<I", f.read(4))[0]
        header = ast.literal_eval(f.read(header_len).decode("latin1").strip())
        if header.get("descr") in {"|O", "<O"}:
            return None
        count = math.prod(header["shape"])
        raw = f.read(count * itemsize)
        values = array(typecode)
        values.frombytes(raw)
        return values


def qmut_binary_row() -> dict:
    path = ROOT / "data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/mutation_diff_oof.npz"
    summary_path = path.parent / "summary.json"
    d = json.loads(summary_path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(path) as zf:
        y = read_npy_numeric(zf, "y.npy", "b", 1)
        p = read_npy_numeric(zf, "probability.npy", "d", 8)
        fold = read_npy_numeric(zf, "fold.npy", "b", 1)
    threshold = float(d["threshold"])
    cm = Counter()
    classes = Counter()
    folds = set()
    brier_sum = 0.0
    for yi, pi, fi in zip(y, p, fold):
        yi, pi = int(yi), float(pi)
        pred = int(pi >= threshold)
        cm[(yi, pred)] += 1
        classes[yi] += 1
        folds.add(int(fi))
        brier_sum += (pi - yi) ** 2
    calc = metric_from_confusion(cm, [0, 1])
    ref = d["candidate_pooled_metrics"]
    return {
        "dataset_task": "QMutBench binary champion", "task_semantics": "Survivor vs WOO/OPO merged detection",
        "model": f"ExtraTrees_{d.get('trees', 100)}_mutation_aware_23f", "model_family": "Classical",
        "validation": "origin-group 5-fold development OOF", "group_unit": "origin_program",
        "records": len(y), "unique_records": len(y), "independent_groups": d.get("origins", 305), "folds": len(folds),
        "class_counts": json.dumps({str(k): v for k, v in sorted(classes.items())}, sort_keys=True),
        "accuracy": calc["accuracy"], "balanced_accuracy": calc["balanced_accuracy"],
        "macro_f1": calc["macro_f1"], "weighted_f1": calc["weighted_f1"],
        "per_class_metrics": json.dumps(calc["per_class"], sort_keys=True),
        "confusion_matrix": json.dumps(calc["confusion_matrix"]), "runtime_seconds": "",
        "bootstrap_95ci_macro_f1": "", "imbalance_method": "balanced_subsample class weights; threshold from development validation protocol",
        "smote_status": "NOT_EVALUATED", "auc_probability_metrics": json.dumps({"brier_score": brier_sum / len(y),
            "roc_auc_reported": ref["roc_auc"], "average_precision_reported": ref["average_precision"]}),
        "checkpoint_count": "model checkpoint exists; see best_model_metadata.json",
        "gradient_checks_passed": "", "gradient_checks_expected": "",
        "oof_prediction_path": str(path.relative_to(ROOT)), "summary_path": str(summary_path.relative_to(ROOT)),
        "source_version_license": "QMutBench source provenance recorded in campaign files; see model metadata and source hash",
        "status": "RECALCULATED_OOF", "duplicate_prediction_rows": 0,
        "metric_recalculation_agreement_macro_f1": abs(calc["macro_f1"] - float(ref["macro_f1"])),
    }


def main() -> None:
    results = []
    for run in RUNS:
        results.extend(process_run(*run))
    # Existing binary QMutBench champion OOF; only development predictions are read.
    qmut_binary = ROOT / "data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/mutation_diff_oof.npz"
    if qmut_binary.exists():
        results.append(qmut_binary_row())
    # Attach a concise configuration note by model family; exact source task
    # preparation and QNN settings remain traceable via the referenced summaries.
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    if results:
        with (CAMPAIGN / "comprehensive_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader(); writer.writerows(results)
    # Task-local winners only. Targets differ, so no cross-task score ranking.
    winners = []
    for task_name in sorted({x["dataset_task"] for x in results}):
        rows = [x for x in results if x["dataset_task"] == task_name]
        if not rows: continue
        best = max(rows, key=lambda x: x["macro_f1"])
        qnns = [x for x in rows if x["model_family"] == "QNN/hybrid"]
        classics = [x for x in rows if x["model_family"] == "Classical"]
        best_qnn = max(qnns, key=lambda x: x["macro_f1"]) if qnns else None
        best_classical = max(classics, key=lambda x: x["macro_f1"]) if classics else None
        winners.append({"dataset_task": task_name, "target_semantics": best["task_semantics"],
            "best_overall_model": best["model"], "best_overall_family": best["model_family"],
            "best_overall_macro_f1": best["macro_f1"], "best_overall_accuracy": best["accuracy"],
            "best_overall_balanced_accuracy": best["balanced_accuracy"],
            "best_qnn_model": best_qnn["model"] if best_qnn else "",
            "best_qnn_macro_f1": best_qnn["macro_f1"] if best_qnn else "",
            "best_classical_model": best_classical["model"] if best_classical else "",
            "best_classical_macro_f1": best_classical["macro_f1"] if best_classical else "",
            "qnn_minus_classical_macro_f1": (best_qnn["macro_f1"] - best_classical["macro_f1"] if best_qnn and best_classical else ""),
            "comparability": "within-task grouped OOF only; do not rank across different targets"})
    with (CAMPAIGN / "best_model_by_task.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(winners[0]) if winners else [])
        writer.writeheader(); writer.writerows(winners)
    paired = [x for x in winners if x["best_qnn_model"] and x["best_classical_model"]]
    with (CAMPAIGN / "classical_vs_qnn_by_task.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0]) if paired else [])
        writer.writeheader(); writer.writerows(paired)
    # Preserve fold-level evidence for every binary/text task with a saved file.
    fold_rows = []
    fold_paths = [
        CAMPAIGN / "bugs4q_standalone/qnn_gradient_fix/fold_metrics.json",
        CAMPAIGN / "bugs4q_standalone/fold_metrics.json",
        CAMPAIGN / "bugs4q_na/qnn/fold_metrics.json",
        CAMPAIGN / "bugs4q_joint/qnn_hybrid/fold_metrics.json",
        CAMPAIGN / "bugs4q_joint/qnn_train_core_complete/fold_metrics.json",
        CAMPAIGN / "bugs4q_joint/qnn_train_na_complete/fold_metrics.json",
        CAMPAIGN / "qbugs/qnn_hybrid/fold_metrics.json",
        CAMPAIGN / "quantum_platform_223_comment_only/qnn_hybrid/fold_metrics.json",
        CAMPAIGN / "qolumbina_detection_ideal/qnn/fold_metrics.json",
        CAMPAIGN / "lintq_warning_triage/qnn/fold_metrics.json",
        THREE / "fold_metrics.json",
    ]
    for path in fold_paths:
        if not path.exists(): continue
        try: data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError: continue
        if isinstance(data, dict): data = data.get("fold_metrics", [])
        for row in data:
            fold_rows.append({"run": str(path.parent.relative_to(ROOT)), **row})
    if fold_rows:
        with (CAMPAIGN / "fold_level_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in fold_rows for k in r}))
            writer.writeheader(); writer.writerows(fold_rows)
    transfer_rows = []
    transfer_dirs = {
        "pooled_core_plus_na": CAMPAIGN / "bugs4q_joint/qnn_hybrid",
        "train_core_only": CAMPAIGN / "bugs4q_joint/qnn_train_core_complete",
        "train_na_only": CAMPAIGN / "bugs4q_joint/qnn_train_na_complete",
    }
    for training, folder in transfer_dirs.items():
        path = folder / "source_performance.json"
        if not path.exists(): continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            transfer_rows.append({"training_source": training, "evaluation_source": row.get("source"),
                "model": row.get("model"), "records": row.get("records"), "groups": row.get("origins"),
                "accuracy": row.get("accuracy"), "balanced_accuracy": row.get("balanced_accuracy"),
                "macro_f1": row.get("macro_f1"),
                "confusion_matrix": json.dumps(row.get("confusion_matrix_labels_class0_class1")),
                "evaluation_protocol": "shared issue-group 5-fold OOF; source results reported separately"})
    if transfer_rows:
        with (CAMPAIGN / "cross_source_generalization.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(transfer_rows[0])); writer.writeheader(); writer.writerows(transfer_rows)
    manifest = {"status": "completed_streaming_oof_recalculation",
        "results_rows": len(results), "task_rows": len(winners), "fold_metric_rows": len(fold_rows),
        "metrics_recomputed_from": "saved development OOF predictions only",
        "frozen_test_accessed": False,
        "smote": "not evaluated in saved runs; training-fold-only resampling suite is bundled for external execution and remains unrun",
        "cross_source_rows": len(transfer_rows),
        "peak_memory": "not sampled by this standard-library streaming script",
        "limitations": ["AUC, Brier, calibration, fold SD, and inference runtime are not recomputed for every candidate here.",
            "Model configurations and feature protocols differ by task; see each run summary and task_definition.json.",
            "Only recorded candidates are listed; absent baselines are marked unrun in campaign_completion_matrix.csv."]}
    (CAMPAIGN / "comprehensive_results_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
