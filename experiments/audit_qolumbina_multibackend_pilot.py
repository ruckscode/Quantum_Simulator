"""Independently audit the grouped Qolumbina multi-backend pilot."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/qolumbina_detection_multibackend"
PROCESSED = ROOT / "data/processed/qolumbina/seeded_mutant_detection_multibackend"
RUN_DEFAULT = ROOT / "reports/experiments/resumable_qnn_pilot_qolumbina_multibackend_20261029"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(rows: list[dict], prediction_key: str) -> dict:
    y = [int(r["y_true"]) for r in rows]
    p = [int(r[prediction_key]) for r in rows]
    return {"rows": len(rows), "class_counts": {str(c): y.count(c) for c in sorted(set(y))},
            "accuracy": accuracy_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro", zero_division=0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN_DEFAULT)
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    out = BASE / ("pilot_audit_balanced.json" if "balanced" in run_dir.name else "pilot_audit.json")
    issues: list[str] = []
    processed = json.loads((PROCESSED / "manifest.json").read_text(encoding="utf-8"))
    contract_path = ROOT / "experiments/source_task_contracts/qolumbina_multibackend_detection.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    prepared = json.loads((BASE / "preparation_audit.json").read_text(encoding="utf-8"))
    run = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    source = ROOT / contract["input_path"]
    fold_input = ROOT / contract["fold_path"]
    if sha256(source) != processed["input_sha256"]: issues.append("input source hash mismatch")
    if sha256(fold_input) != processed["fold_sha256"]: issues.append("fold source hash mismatch")
    if sha256(contract_path) != run["config"]["contract_sha256"]: issues.append("contract hash mismatch")
    if sha256(PROCESSED / "manifest.json") != run["config"]["processed_manifest_sha256"]: issues.append("processed manifest hash mismatch")
    if sha256(PROCESSED / "manifests/group_folds.csv") != run["config"]["split_sha256"]: issues.append("processed split hash mismatch")
    for part in processed["partitions"]:
        if sha256(ROOT / part["path"]) != part["sha256"]: issues.append(f"partition hash mismatch: {part['path']}")
    rows = []
    for part in processed["partitions"]: rows.extend(read_csv(ROOT / part["path"]))
    splits = read_csv(PROCESSED / "manifests/group_folds.csv")
    split_by_id = {r["record_id"]: r for r in splits}
    qnn = read_csv(run_dir / "qnn_oof_predictions.csv")
    classical = read_csv(run_dir / "classical_oof_predictions.csv")
    row_by_id = {r["record_id"]: r for r in rows}
    if len(row_by_id) != len(rows) or len(split_by_id) != len(splits): issues.append("duplicate record ID in processed data or split")
    if set(row_by_id) != set(split_by_id) or set(row_by_id) != {r["record_id"] for r in qnn} or set(row_by_id) != {r["record_id"] for r in classical}:
        issues.append("data, split, QNN, and classical OOF IDs differ")
    folds_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = split_by_id.get(row["record_id"], {})
        if split.get("group_id") != row["bug_id"]: issues.append(f"split group mismatch: {row['record_id']}")
        folds_by_group[row["bug_id"]].add(split.get("fold", ""))
        if row["label"] != ("1" if float(row["detection_rate"]) >= 0.5 else "0"):
            issues.append(f"label rule mismatch: {row['record_id']}")
    if any(len(v) != 1 for v in folds_by_group.values()): issues.append("source program crosses folds")
    for pred_rows, pred_col, name in [(qnn, "y_pred", "QNN"), (classical, "y_pred", "classical")]:
        for pr in pred_rows:
            source_row = row_by_id.get(pr["record_id"], {})
            split = split_by_id.get(pr["record_id"], {})
            if int(pr["y_true"]) != int(source_row.get("label", -1)): issues.append(f"{name} target mismatch: {pr['record_id']}")
            if str(pr["group_id"]) != str(source_row.get("bug_id")): issues.append(f"{name} group mismatch: {pr['record_id']}")
            if str(pr["fold"]) != str(split.get("fold")): issues.append(f"{name} fold mismatch: {pr['record_id']}")
    if sha256(run_dir / "qnn_oof_predictions.csv") != run["qnn_oof_sha256"]: issues.append("QNN OOF hash mismatch")
    if sha256(run_dir / "classical_oof_predictions.csv") != run["classical_baseline_sha256"]: issues.append("classical OOF hash mismatch")
    if run.get("status") != "completed" or run.get("frozen_test_accessed") or metrics.get("frozen_test_accessed"):
        issues.append("run incomplete or frozen test access reported")
    parity = run.get("resume_verification", {}).get("uninterrupted_reference_parity", {})
    if run.get("resume_verification", {}).get("status") != "verified" or run.get("resume_verification", {}).get("resume_events") != 1 or not parity or not all(parity.values()):
        issues.append("resume parity receipt incomplete")
    checkpoint_receipts = {x["fold"]: x for x in metrics["qnn"].get("checkpoints", [])}
    checkpoint_checks = []
    for fold in range(1, 6):
        folder = run_dir / "qnn/amplitude_3q_depth1_lr004" / f"fold_{fold}"
        # The current runner stores checkpoints beneath per-fold directories.
        if not folder.exists():
            candidates = list(run_dir.glob(f"**/fold_{fold}/checkpoint.npz"))
            if candidates: folder = candidates[0].parent
        checkpoint = folder / "checkpoint.npz"
        history = folder / "training_history.json"
        if not checkpoint.exists() or not history.exists():
            issues.append(f"missing checkpoint/history for fold {fold}")
            continue
        c_hash, h_hash = sha256(checkpoint), sha256(history)
        receipt = checkpoint_receipts.get(fold, {})
        if c_hash != receipt.get("checkpoint_sha256") or h_hash != receipt.get("history_sha256"):
            issues.append(f"checkpoint/history receipt mismatch for fold {fold}")
        checkpoint_checks.append({"fold": fold, "checkpoint_sha256": c_hash, "history_sha256": h_hash})
    backend_metrics = {}
    q_by_backend: dict[str, list[dict]] = defaultdict(list)
    c_by_backend: dict[str, list[dict]] = defaultdict(list)
    for r in qnn: q_by_backend[row_by_id[r["record_id"]]["backend"]].append(r)
    for r in classical: c_by_backend[row_by_id[r["record_id"]]["backend"]].append(r)
    for backend in sorted(q_by_backend):
        backend_metrics[backend] = {"qnn": summarize(q_by_backend[backend], "y_pred"),
                                    "classical": summarize(c_by_backend[backend], "y_pred")}
    result = {"schema": "qolumbina-multibackend-pilot-audit-v1", "status": "VERIFIED" if not issues else "REVIEW_REQUIRED",
              "issues": issues, "rows": len(rows), "groups": len(folds_by_group), "folds": len(set(x["fold"] for x in splits)),
              "source_manifest_sha256": sha256(source), "source_preparation_status": processed["status"],
              "source_validation_pass": processed["validation_pass"], "feature_fields": contract["feature_fields"],
              "resume_status": run["resume_verification"], "frozen_test_accessed": False,
              "qnn_overall": summarize(qnn, "y_pred"), "classical_overall": summarize(classical, "y_pred"),
              "per_backend_metrics": backend_metrics, "checkpoint_receipts_verified": checkpoint_checks,
              "runtime": metrics["runtime"], "paired_group_bootstrap": metrics["paired_group_bootstrap_95_percentile"],
              "prepared_audit_sha256": sha256(BASE / "preparation_audit.json"),
              "run_manifest_sha256": sha256(run_dir / "run_manifest.json")}
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "issues": issues, "output": str(out.relative_to(ROOT)),
                      "qnn": result["qnn_overall"], "classical": result["classical_overall"],
                      "per_backend": backend_metrics}, indent=2))
    if issues: raise SystemExit(2)


if __name__ == "__main__":
    main()
