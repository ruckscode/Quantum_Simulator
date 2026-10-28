"""Build the campaign's source/task status table from verified local artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028"

TASKS = [
    ("Bugs4Q official framework", "bugs4q_standalone/qnn_gradient_fix", "source-state", "31 issues / 62 snapshots; five issue-group folds", "Additional 11 official framework issues were not included in this trained subset."),
    ("Bugs4Q-NA", "bugs4q_na/qnn", "source-state", "31 issues / 62 snapshots; five issue-group folds", "Nonoverlapping with the trained core subset."),
    ("Bugs4Q pooled", "bugs4q_joint/qnn_hybrid", "source-state integration", "62 issue groups / 124 snapshots; shared folds", "Matched pooled run; QNN macro F1 0.498 vs Extra Trees 0.524."),
    ("Bugs4Q core-only transfer", "bugs4q_joint/qnn_train_core_complete", "source-state transfer", "62 issue groups / 124 held-out predictions", "Training on core only; held-out NA performance is also recorded."),
    ("Bugs4Q-NA-only transfer", "bugs4q_joint/qnn_train_na_complete", "source-state transfer", "62 issue groups / 124 held-out predictions", "Training on NA only; held-out core performance is also recorded."),
    ("QBugs HornBro subset", "qbugs/qnn_hybrid", "correct/buggy circuit state", "750 correct/buggy pairs / 1,500 circuits; paired snapshots grouped", "Extracted QBugs subset; not the complete upstream QBugs corpus."),
    ("Quantum platform annotations", "quantum_platform_223_comment_only/qnn_hybrid", "issue-comment classification", "319 comments / 314 commit groups", "Metadata comment task. Full metadata candidate is invalid due to label leakage."),
    ("Qolumbina empirical study", "qolumbina_detection_ideal/qnn", "ideal-backend seeded-mutant detection", "182 mutants / 26 programs; five program-group folds", "Synthetic mutation task, not real-bug classification. Multi-backend 546-row run remains partial."),
    ("LintQ", "lintq_warning_triage/qnn", "manual warning triage TP vs FP", "290 warnings / 228 issue or file groups; five folds", "55 ambiguous NW labels excluded; not program-state classification."),
    ("Bugs4Q-Robust", "bugs4q_robust_execution/qnn", "test-harness pass/fail prediction", "2,220 observations / 37 issue groups; five issue-group folds", "Task manifest is valid; a training attempt hit the RAM floor before saving a QNN checkpoint."),
    ("QSimBench", "qsimbench_backend_fingerprint/qnn", "Aer vs fake_kyiv backend fingerprinting", "300 runs / 150 algorithm-size groups; five grouped folds", "Outcome-histogram task manifest is prepared; local training stopped after imports at the 500 MiB safety floor; included in external notebook."),
    ("QMutBench", None, "three-class mutation diagnosis", "205 records / 12 origins; three development folds", "OOF training exists; no recoverable QNN parameter checkpoint. Frozen test not opened."),
]

UNTRAINED = [
    ("Mute-Alg", "Blocked by legacy pickle deserialization", "169 pickle files", "Qiskit 1.4.5/2.5 compatibility errors; no trusted circuit labels loaded."),
    ("Bug-Pattern", "No validated supervised manifest", "20 git files", "Acquired; task and label mapping not validated."),
    ("QChecker", "No validated supervised manifest", "39 git files", "Acquired; task and label mapping not validated."),
    ("QML platform workbooks", "Schema/license normalization incomplete", "43 source archive members", "No training until labels, duplicates, and task semantics are verified."),
]


def best(summary: dict, family: str) -> dict | None:
    candidates = [x for x in summary.get("results", []) if x.get("family") == family]
    return max(candidates, key=lambda x: x.get("macro_f1", -1)) if candidates else None


def main() -> None:
    inventory = json.loads((CAMPAIGN / "verified_dataset_inventory.json").read_text(encoding="utf-8"))
    inv = {x["dataset"]: x for x in inventory["datasets"]}
    aliases = {"Bugs4Q official framework": "Bugs4Q framework",
               "Quantum platform annotations": "Quantum platform bug corpus",
               "QML platform workbooks": "QML platform issues"}
    rows = []
    for source, run, task, split, note in TASKS:
        item = inv.get(aliases.get(source, source), {})
        summary = {}
        if run:
            path = CAMPAIGN / run / "summary.json"
            if path.exists():
                summary = json.loads(path.read_text(encoding="utf-8"))
        qnn, classical = best(summary, "quantum_or_hybrid"), best(summary, "classical")
        rows.append({
            "dataset_source": source,
            "download_status": item.get("download_status", "COMPLETE" if item else "NOT_APPLICABLE"),
            "extraction_status": item.get("extraction_status", "COMPLETE" if item else "NOT_APPLICABLE"),
            "data_status": item.get("data_status", "PRESENT" if item else "DERIVED_TASK_MANIFEST"),
            "task": task,
            "training_status": "COMPLETE" if summary.get("status") == "completed_grouped_supervised_qnn_benchmark" else ("OOF_ONLY_NO_CHECKPOINT" if source == "QMutBench" else "PARTIAL"),
            "records": summary.get("records", ""),
            "groups": summary.get("origins", ""),
            "folds": summary.get("folds", ""),
            "best_qnn": qnn.get("model", "") if qnn else ("three_class_vqc" if source == "QMutBench" else ""),
            "qnn_macro_f1": qnn.get("macro_f1", "") if qnn else (0.2997533196871884 if source == "QMutBench" else ""),
            "best_classical": classical.get("model", "") if classical else ("extra_trees_4f" if source == "QMutBench" else ""),
            "classical_macro_f1": classical.get("macro_f1", "") if classical else (0.46688748084908643 if source == "QMutBench" else ""),
            "split_description": split,
            "notes": note,
        })
        if source == "QMutBench":
            rows[-1].update({"records": 205, "groups": 12, "folds": 3})
        if source == "Bugs4Q-Robust" and not summary:
            task_manifest = json.loads((CAMPAIGN / "bugs4q_robust_execution/task_definition.json").read_text(encoding="utf-8"))
            rows[-1].update({"records": task_manifest["observations"], "groups": task_manifest["issues"], "folds": 5})
    for source, task, count, note in UNTRAINED:
        item = inv.get(aliases.get(source, source), {})
        rows.append({"dataset_source": source,
                     "download_status": item.get("download_status", "COMPLETE" if item else "PARTIAL"),
                     "extraction_status": item.get("extraction_status", "COMPLETE" if item else "PARTIAL"),
                     "data_status": item.get("data_status", "PRESENT" if item else "PRESENT"),
                     "task": task, "training_status": "NOT_TRAINED", "records": count,
                     "groups": "", "folds": "", "best_qnn": "", "qnn_macro_f1": "",
                     "best_classical": "", "classical_macro_f1": "",
                     "split_description": "", "notes": note})
    # Keep the existing large binary QMutBench task separate from the research
    # three-class task; their labels, sample frames and scores are incomparable.
    binary_summary = json.loads((ROOT / "data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/summary.json").read_text(encoding="utf-8"))
    rows.append({"dataset_source":"QMutBench", "download_status":"COMPLETE", "extraction_status":"NOT_APPLICABLE",
        "data_status":"PRESENT", "task":"binary Survivor vs WOO/OPO merged detection (existing development champion)",
        "training_status":"EXISTING_CHAMPION_OOF", "records":578117, "groups":305, "folds":5,
        "best_qnn":"", "qnn_macro_f1":"", "best_classical":"ExtraTrees_100_mutation_aware_23f",
        "classical_macro_f1":0.7933135654357681, "split_description":"origin-grouped five-fold development OOF",
        "notes":"Production unchanged; no frozen-test access; separate task from three-class research."})
    rows.append({"dataset_source":"Nine binary source tasks", "download_status":"VERIFIED_INPUTS",
        "extraction_status":"NOT_APPLICABLE", "data_status":"DERIVED_TASK_MANIFESTS",
        "task":"Broad grouped classical models and training-fold-only imbalance methods",
        "training_status":"BUNDLE_READY_UNEXECUTED", "records":"see task manifests", "groups":"see task manifests",
        "folds":5, "best_qnn":"", "qnn_macro_f1":"", "best_classical":"", "classical_macro_f1":"",
        "split_description":"GroupKFold by task bug_id/algorithm-size group",
        "notes":"The 14-model baseline suite plus random over/under-sampling and continuous-space SMOTE have not run; external notebook still requires Kaggle execution."})
    fields = list(rows[0])
    target = CAMPAIGN / "campaign_completion_matrix.csv"
    with target.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tasks = [{k: v for k, v in row.items()} for row in rows]
    task_inventory = {"generated_from": "campaign_completion_matrix.csv and verified_dataset_inventory.json",
        "verified_source_count": len(inventory.get("datasets", [])), "task_count": len(tasks),
        "tasks": tasks, "frozen_test_accessed": False, "production_model_modified": False}
    (CAMPAIGN / "dataset_task_inventory.json").write_text(json.dumps(task_inventory, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(target.relative_to(ROOT)), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
