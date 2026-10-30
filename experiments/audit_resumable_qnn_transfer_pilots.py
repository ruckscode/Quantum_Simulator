"""Verify source-restricted Bugs4Q transfer pilots and reject mixed-source runs."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import run_resumable_qnn_pilot as pilot  # noqa: E402

TASKS = ["bugs4q_core_only_transfer_source_restricted_v2",
         "bugs4q_na_only_transfer_source_restricted_v2"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    issues, task_audits = [], []
    for task in TASKS:
        pilot.configure_task(task)
        spec = pilot.TASK_CONFIGS[task]
        rows, assignment, source_audit = pilot.read_task()
        output = ROOT / spec["output"]
        run = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        training_source, transfer_source = spec["training_source"], spec["transfer_target_source"]
        if run.get("status") != "completed" or run.get("frozen_test_accessed"):
            issues.append(f"{task}: run incomplete or frozen test accessed")
        if run.get("config", {}).get("training_source_filter") != training_source or run.get("training_source_filter") != training_source:
            issues.append(f"{task}: training-source filter is absent or mismatched")
        if run.get("transfer_target_source") != transfer_source:
            issues.append(f"{task}: transfer target source mismatch")
        ids = [row["record_id"] for row in rows]
        source_by_id = {row["record_id"]: str(row["source"]) for row in rows}
        qnn_rows = read_csv(output / "qnn_oof_predictions.csv")
        classical_rows = read_csv(output / "classical_oof_predictions.csv")
        key = lambda row: (row["record_id"], int(row["fold"]))
        qmap = {key(row): (row["group_id"], int(row["y_true"])) for row in qnn_rows}
        cmap = {key(row): (row["group_id"], int(row["y_true"])) for row in classical_rows}
        if set(qmap) != {(rid, fold) for rid, (fold, _) in assignment.items()} or qmap != cmap:
            issues.append(f"{task}: QNN/classical OOF IDs, folds, groups, or targets differ")
        if len(qnn_rows) != len(ids) or len(classical_rows) != len(ids):
            issues.append(f"{task}: OOF prediction count mismatch")
        target_qnn_rows = [row for row in qnn_rows if source_by_id.get(row["record_id"]) == transfer_source]
        target_classical_rows = [row for row in classical_rows if source_by_id.get(row["record_id"]) == transfer_source]
        transfer_bootstrap = pilot.paired_group_bootstrap(target_qnn_rows, target_classical_rows,
            seed=pilot.SEED, iterations=1000)
        fold_results = {int(item["fold"]): item for item in run.get("fold_results", [])}
        checkpoints = {int(item["fold"]): item for item in run.get("qnn", {}).get("checkpoints", [])}
        fold_audits = []
        for fold in range(1, 6):
            train, valid = pilot.build_fold_indices(rows, assignment, fold, training_source)
            train_ids = sorted(ids[i] for i in train)
            valid_ids = sorted(ids[i] for i in valid)
            fold_result = fold_results.get(fold, {})
            if fold_result.get("train_records") != len(train) or fold_result.get("validation_records") != len(valid):
                issues.append(f"{task}/fold{fold}: train/validation record counts disagree with source-restricted plan")
            if any(str(rows[i]["source"]) != training_source for i in train):
                issues.append(f"{task}/fold{fold}: forbidden source present in training rows")
            sidecar_path = output / "qnn" / pilot.MODEL_CONFIG["name"] / f"fold_{fold}" / "checkpoint_manifest.json"
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            provenance = sidecar.get("provenance", {})
            expected_train_hash = hashlib.sha256("\n".join(train_ids).encode()).hexdigest()
            expected_valid_hash = hashlib.sha256("\n".join(valid_ids).encode()).hexdigest()
            if provenance.get("train_record_ids_sha256") != expected_train_hash:
                issues.append(f"{task}/fold{fold}: checkpoint train ID receipt mismatch")
            if provenance.get("validation_record_ids_sha256") != expected_valid_hash:
                issues.append(f"{task}/fold{fold}: checkpoint validation ID receipt mismatch")
            checkpoint = sidecar_path.with_name("checkpoint.npz")
            history = sidecar_path.with_name("training_history.json")
            receipt = checkpoints.get(fold, {})
            if sha(checkpoint) != receipt.get("checkpoint_sha256") or sha(history) != receipt.get("history_sha256"):
                issues.append(f"{task}/fold{fold}: checkpoint/history hash mismatch")
            fold_audits.append({"fold": fold, "train_records": len(train), "train_groups": len({rows[i]["bug_id"] for i in train}),
                "validation_records": len(valid), "checkpoint_sha256": sha(checkpoint),
                "train_source_only": all(str(rows[i]["source"]) == training_source for i in train)})
        if metrics.get("qnn_metrics_by_source", {}).get(transfer_source) is None or metrics.get("classical_metrics_by_source", {}).get(transfer_source) is None:
            issues.append(f"{task}: separate transfer-target metrics missing")
        if run.get("resume_verification", {}).get("status") != "verified" or run.get("resume_verification", {}).get("resume_events") != 1:
            issues.append(f"{task}: resume verification incomplete")
        task_audits.append({"task": task, "training_source": training_source,
            "transfer_target_source": transfer_source, "rows": len(rows),
            "groups": source_audit["group_count"], "folds": fold_audits,
            "qnn_transfer_metrics": metrics["qnn_metrics_by_source"][transfer_source],
            "classical_transfer_metrics": metrics["classical_metrics_by_source"][transfer_source],
            "transfer_target_paired_group_bootstrap_95_percentile": transfer_bootstrap,
            "resume_status": run["resume_verification"]["status"],
            "frozen_test_accessed": run["frozen_test_accessed"],
            "run_manifest_sha256": sha(output / "run_manifest.json"),
            "metrics_sha256": sha(output / "metrics.json")})
    invalid_dir = ROOT / "reports/experiments/resumable_qnn_pilot_bugs4q_core_transfer_20261029"
    correction = json.loads((invalid_dir / "task_validity_correction.json").read_text(encoding="utf-8"))
    if correction.get("status") != "INVALID_FOR_TRANSFER_TASK" or correction.get("run_manifest_sha256") != sha(invalid_dir / "run_manifest.json"):
        issues.append("Mixed-source core transfer pilot was not correctly invalidated or its correction is stale")
    report = {"schema": "resumable-qnn-transfer-pilot-audit-v1",
        "status": "VERIFIED" if not issues else "REVIEW_REQUIRED",
        "tasks": task_audits, "invalidated_prior_run": {"status": correction.get("status"),
            "correction_path": str((invalid_dir / "task_validity_correction.json").relative_to(ROOT)),
            "reason": correction.get("reason")},
        "frozen_test_accessed": False, "issues": issues}
    target = ROOT / "reports/experiments/qml_source_preprocessing_20261029/bugs4q_transfer_pilot_audit.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "tasks": len(task_audits), "issues": issues,
        "output": str(target.relative_to(ROOT))}, indent=2))
    if issues: raise SystemExit(1)


if __name__ == "__main__":
    main()
