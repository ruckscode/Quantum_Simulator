"""Verify the recovered two-task QNN sweep and its matched classical controls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028"
TASKS = ("bugs4q_robust_execution", "qsimbench_backend_fingerprint")
OUT = CAMPAIGN / "qnn_hyperparameter_sweep_audit.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def audit_task(task: str) -> dict:
    import sys
    sys.path.insert(0, str(ROOT / "experiments"))
    import train_resource_light_qnn_tasks as q
    from run_resumable_qnn_hyperparameter_sweep import TASKS as SPECS, source_and_folds, CONFIGS, EPOCHS
    spec = SPECS[task]
    output = CAMPAIGN / task / "qnn_hyperparameter_sweep20_resumable_v2"
    run_path, oof_path = output / "run_manifest.json", output / "oof_predictions.csv"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    comp_path = output / "matched_classical_comparison.json"
    comparison = json.loads(comp_path.read_text(encoding="utf-8"))
    rows, y, groups, folds, receipt = source_and_folds(task, spec, q)
    issues = []
    if run.get("status") != "completed" or run.get("frozen_test_accessed"):
        issues.append("run incomplete or frozen-test access reported")
    if run.get("source_receipt") != receipt: issues.append("source/split receipt mismatch")
    if sha256(oof_path) != run.get("oof_predictions_sha256"): issues.append("QNN OOF hash mismatch")
    qnn_rows = read_csv(oof_path)
    expected_ids = {r["record_id"] for r in rows}
    if len(qnn_rows) != len(expected_ids) * len(CONFIGS): issues.append("QNN OOF row count mismatch")
    qnn_oof_sha = sha256(oof_path)
    config_by_name = {c["name"]: c for c in CONFIGS}
    source_by_id = {r["record_id"]: r for r in rows}
    fold_by_id = {r["record_id"]: fold for fold, (_, va) in folds.items() for r in [rows[int(i)] for i in va]}
    fold_result_by_key = {(r["model"], int(r["fold"])): r for r in run["fold_results"]}
    gap_receipts = {(x["model"], int(x["fold"])): x for x in run.get("training_history_gaps", [])}
    checkpoints_verified = 0
    inference_snapshots_reused = 0
    for cfg in CONFIGS:
        name = cfg["name"]
        model_rows = [r for r in qnn_rows if r["model"] == name]
        if {r["record_id"] for r in model_rows} != expected_ids: issues.append(f"OOF ID coverage mismatch: {name}")
        for r in model_rows:
            source = source_by_id.get(r["record_id"])
            if not source: issues.append(f"unknown OOF ID: {r['record_id']}"); continue
            if (r["group_id"] != source[spec["group"]] or int(r["y_true"]) != int(source["label"]) or
                    int(r["fold"]) != fold_by_id[r["record_id"]]):
                issues.append(f"OOF target/group/fold mismatch: {name}/{r['record_id']}")
        metrics = q.metrics([int(r["y_true"]) for r in model_rows], [int(r["y_pred"]) for r in model_rows])
        summary = next((x for x in run["results"] if x["model"] == name), None)
        if not summary: issues.append(f"missing summary: {name}")
        else:
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                if abs(metrics[metric] - summary[metric]) > 1e-12: issues.append(f"QNN metric replay mismatch {name}/{metric}")
        for fold in sorted(folds):
            fr = fold_result_by_key.get((name, fold))
            if not fr: issues.append(f"missing fold result {name}/{fold}"); continue
            if not fr.get("gradient_check", {}).get("passed"): issues.append(f"gradient check failed {name}/{fold}")
            if fr.get("checkpoint_kind") == "legacy_inference_only_checkpoint_reused":
                legacy = ROOT / fr["checkpoint_path"]
                folder = output / name / f"fold_{fold}"
                snap_path = folder / "inference_snapshot.npz"
                legacy_receipt = fr.get("legacy_checkpoint_receipt", {})
                if not legacy.is_file() or sha256(legacy) != legacy_receipt.get("sha256") or sha256(legacy) != fr.get("checkpoint_sha256"):
                    issues.append(f"legacy checkpoint hash mismatch {name}/{fold}")
                if not snap_path.is_file(): issues.append(f"missing inference snapshot {name}/{fold}")
                else:
                    with np.load(snap_path, allow_pickle=False) as d:
                        if int(d["epoch"]) != EPOCHS or str(d["source_sha256"].item()) != receipt["source_sha256"]:
                            issues.append(f"inference snapshot provenance mismatch {name}/{fold}")
                inference_snapshots_reused += 1
            elif fr.get("checkpoint_kind") == "resumable_adam_rng_checkpoint":
                checkpoint = ROOT / fr["checkpoint_path"]
                folder = output / name / f"fold_{fold}"
                history_path = folder / "training_history.json"
                if not checkpoint.is_file() or sha256(checkpoint) != fr.get("checkpoint_sha256"):
                    issues.append(f"resumable checkpoint hash mismatch {name}/{fold}"); continue
                with np.load(checkpoint, allow_pickle=False) as d:
                    required = {"theta", "head", "m_t", "v_t", "m_h", "v_h", "m_b", "v_b", "rng_state", "provenance", "config", "epoch"}
                    if not required.issubset(d.files): issues.append(f"optimizer/RNG state missing {name}/{fold}")
                    if int(d["epoch"]) != EPOCHS: issues.append(f"wrong checkpoint epoch {name}/{fold}")
                    provenance = json.loads(str(d["provenance"].item())) if "provenance" in d.files else {}
                    if (provenance.get("source_sha256") != receipt["source_sha256"] or
                            provenance.get("split_sha256") != receipt["group_fold_assignment_sha256"] or
                            provenance.get("fold") != fold or provenance.get("task") != task):
                        issues.append(f"checkpoint provenance mismatch {name}/{fold}")
                if not history_path.is_file(): issues.append(f"missing training history {name}/{fold}")
                else:
                    history = json.loads(history_path.read_text(encoding="utf-8"))
                    epochs = {int(x["epoch"]) for x in history if x.get("epoch") is not None}
                    gap = gap_receipts.get((name, fold), {})
                    gap_epochs = set(gap.get("missing_loss_epochs", []))
                    if epochs | gap_epochs != set(range(1, EPOCHS + 1)) or epochs & gap_epochs:
                        issues.append(f"history epoch coverage/gap mismatch {name}/{fold}")
                    if gap_epochs:
                        gap_path = ROOT / gap["path"]
                        if not gap_path.is_file(): issues.append(f"missing history gap receipt {name}/{fold}")
                checkpoints_verified += 1
            else:
                issues.append(f"unknown checkpoint kind {name}/{fold}")
    classical_metrics = {}
    for width in (4, 8):
        meta = comparison["classical_oof"][str(width)]
        path = ROOT / meta["path"]
        if not path.is_file() or sha256(path) != meta["sha256"]: issues.append(f"classical OOF hash mismatch width={width}"); continue
        crows = read_csv(path)
        if {r["record_id"] for r in crows} != expected_ids: issues.append(f"classical OOF ID coverage mismatch width={width}")
        for r in crows:
            src = source_by_id.get(r["record_id"])
            if not src or r["group_id"] != src[spec["group"]] or int(r["y_true"]) != int(src["label"]) or int(r["fold"]) != fold_by_id[r["record_id"]]:
                issues.append(f"classical OOF target/group/fold mismatch width={width}/{r['record_id']}")
        replay = q.metrics([int(r["y_true"]) for r in crows], [int(r["y_pred"]) for r in crows])
        saved = comparison["classical_metrics_by_feature_width"][str(width)]
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            if abs(replay[metric] - saved[metric]) > 1e-12: issues.append(f"classical metric replay mismatch width={width}/{metric}")
        classical_metrics[str(width)] = replay
    if comparison.get("frozen_test_accessed") is not False or run.get("evaluation_scope", "").find("no frozen test") < 0:
        issues.append("evaluation scope/frozen-test receipt invalid")
    return {"task": task, "status": "VERIFIED" if not issues else "REVIEW_REQUIRED", "issues": issues,
            "rows": len(rows), "groups": len(set(groups)), "folds": len(folds),
            "qnn_models": len(CONFIGS), "qnn_model_fold_results": len(fold_result_by_key),
            "resumable_checkpoints_verified": checkpoints_verified,
            "legacy_inference_snapshots_reused": inference_snapshots_reused,
            "classical_controls_verified": len(classical_metrics), "matched_comparisons": len(comparison.get("comparisons", [])),
            "qnn_oof_sha256": qnn_oof_sha, "run_manifest_sha256": sha256(run_path),
            "matched_comparison_sha256": sha256(comp_path), "runtime_seconds": run["runtime_seconds"],
            "memory_floor_mib": run["memory_floor_mib"], "frozen_test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    args = parser.parse_args()
    selected = TASKS if args.task == "all" else (args.task,)
    task_results = [audit_task(t) for t in selected]
    report = {"schema": "resumable-qnn-hyperparameter-sweep-audit-v1",
              "status": "VERIFIED" if all(x["status"] == "VERIFIED" for x in task_results) else "REVIEW_REQUIRED",
              "tasks": task_results, "frozen_test_accessed": False}
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "tasks": task_results,
                      "output": str(OUT.relative_to(ROOT))}, indent=2))
    if report["status"] != "VERIFIED": raise SystemExit(2)


if __name__ == "__main__":
    main()
