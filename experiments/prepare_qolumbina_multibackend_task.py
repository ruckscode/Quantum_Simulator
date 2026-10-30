"""Prepare a source-grouped, all-backend Qolumbina detection extension.

Reuses the established ideal-backend fold assignment at the original-program
level so every version/backend observation for a program remains in one fold.
The backend is included in the model feature policy by the pilot runner.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/qolumbina_detection"
IDEAL = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/qolumbina_detection_ideal"
OUT = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/qolumbina_detection_multibackend"
CONTRACT_PATH = ROOT / "experiments/source_task_contracts/qolumbina_multibackend_detection.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    source = BASE / "normalized_manifest.csv"
    ideal_oof = IDEAL / "qnn/oof_predictions.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with ideal_oof.open(encoding="utf-8-sig", newline="") as stream:
        ideal_rows = [r for r in csv.DictReader(stream) if r.get("model") == "angle_vqc"]
    if not rows or not ideal_rows:
        raise RuntimeError("Expected complete source manifest and verified ideal OOF assignment")
    groups: dict[str, set[int]] = defaultdict(set)
    for row in ideal_rows:
        groups[row["bug_id"]].add(int(row["fold"]))
    if any(len(folds) != 1 for folds in groups.values()):
        raise ValueError("Ideal pilot has source programs split across folds")
    group_fold = {group: next(iter(folds)) for group, folds in groups.items()}
    source_groups = {r["bug_id"] for r in rows}
    if source_groups != set(group_fold):
        raise ValueError(f"Full and ideal source group mismatch: full={len(source_groups)}, ideal={len(group_fold)}")
    if len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate multi-backend record identifiers")
    if any(r["label"] != ("1" if float(r["detection_rate"]) >= 0.5 else "0") for r in rows):
        raise ValueError("Recorded labels disagree with the source majority-detection rule")
    backends = sorted({r["backend"] for r in rows})
    if len(backends) < 2:
        raise ValueError("Expected multiple backends")
    OUT.mkdir(parents=True, exist_ok=True)
    folds_path = OUT / "group_folds.csv"
    with folds_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["record_id", "bug_id", "fold", "model"], lineterminator="\n")
        writer.writeheader()
        writer.writerows({"record_id": r["record_id"], "bug_id": r["bug_id"],
                          "fold": group_fold[r["bug_id"]], "model": "angle_vqc"} for r in sorted(rows, key=lambda x: x["record_id"]))
    task_def = json.loads((BASE / "task_definition.json").read_text(encoding="utf-8"))
    target = ("Predict whether DOSS detects a seeded Qolumbina mutant in a majority of available report seeds "
              "for one of ideal, fake_algiers, or fake_brooklyn. This is an empirical mutation-outcome target, "
              "not ground-truth repair or hardware diagnosis. Group all variants and backends by original program.")
    contract = {
        "source_id": "qolumbina", "task_id": "seeded_mutant_detection_multibackend",
        "target_semantics": target,
        "required": ["record_id", "bug_id", "program", "variant_version", "backend", "label", "detection_rate", "runs", "unique_random_seeds", "test_executions", "code"],
        "record_id_column": "record_id", "group_column": "bug_id", "label_column": "label", "feature_column": "code",
        "allowed_labels": ["0", "1"], "fold_record_column": "record_id", "fold_group_column": "bug_id",
        "fold_column": "fold", "fold_model_column": "model", "fold_model_value": "angle_vqc",
        "required_values": {}, "positive_integer_columns": ["runs", "unique_random_seeds", "test_executions"],
        "numeric_ranges": {"detection_rate": [0, 1]},
        "derived_label_rule": {"column": "detection_rate", "threshold": 0.5, "positive": "1", "negative": "0"},
        "input_path": str(source.relative_to(ROOT)).replace("\\", "/"),
        "fold_path": str(folds_path.relative_to(ROOT)).replace("\\", "/"),
        "output_path": "data/processed/qolumbina/seeded_mutant_detection_multibackend",
        "source_manifest_sha256": sha256(source), "fold_source_sha256": sha256(ideal_oof),
        "feature_fields": ["code", "backend"], "frozen_test_accessed": False,
    }
    CONTRACT_PATH.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    class_counts = Counter(r["label"] for r in rows)
    per_backend = {backend: dict(Counter(r["label"] for r in rows if r["backend"] == backend)) for backend in backends}
    report = {
        "schema": "qolumbina-multibackend-task-preparation-v1", "status": "PREPARED_FOR_SOURCE_ISOLATED_VALIDATION",
        "rows": len(rows), "program_groups": len(source_groups), "folds": len(set(group_fold.values())),
        "backends": backends, "class_counts": dict(class_counts), "per_backend_class_counts": per_backend,
        "source_manifest_sha256": sha256(source), "ideal_oof_assignment_source_sha256": sha256(ideal_oof),
        "group_fold_manifest_sha256": sha256(folds_path), "contract_sha256": sha256(CONTRACT_PATH),
        "split_policy": "reused established ideal-pilot 5-fold assignment by original program; every variant/backend remains grouped",
        "feature_policy": "mutant code plus pre-run backend token; no program identifier, report output, or detection-rate value",
        "label_policy": task_def["positive_label"] + "; retained target is empirical report outcome, not semantic bug correctness",
        "limitations": ["26 source-program groups limit interval precision", "extreme class imbalance for fake_brooklyn (one negative row)",
                        "seeded synthetic mutants are not real bug reports", "backend comparisons are exploratory, not hardware fault attribution"],
        "frozen_test_accessed": False,
    }
    (OUT / "preparation_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
