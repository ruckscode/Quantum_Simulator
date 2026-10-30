"""Run the omitted QNN-only candidate sweep for Qolumbina multi-backend outcomes.

Reuses the established two-architecture grouped sweep implementation while
pinning the task manifest, pilot fold source, and source-program split.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import train_remaining_standalone_qml_tasks as sweep

SOURCE = sweep.SOURCE
OUTPUT = ROOT / "reports/experiments/qolumbina_multibackend_candidate_sweep_20261029"
TASK_NAME = "Qolumbina multibackend"
TASK_SPEC = {
    "manifest": "qolumbina_detection/normalized_manifest.csv",
    "pred": "../resumable_qnn_pilot_qolumbina_multibackend_20261029/qnn_oof_predictions.csv",
    "model": "unused_no_model_column_in_source_oof",
    "text": ["code", "backend"],
}
EXPECTED_HASHES = {
    "manifest": "b580ad932945d713ad00ade2210e58ff980bdd66a45e67ad4d9b9283ca1c6898",
    "pilot_oof": "7dd9d7ea6b8f477bcc6d1b78c3692b053ce0b511accf7667bad63014f3ad6cfc",
    "split": "666945a7462cc2d68ff3029579d4bf2cff3bd836c4038795d0aedfc821840450",
}
SPLIT = ROOT / "data/processed/qolumbina/seeded_mutant_detection_multibackend/manifests/group_folds.csv"
TARGET_SEMANTICS = (
    "Predict whether DOSS detects a seeded Qolumbina mutant in a majority of available "
    "report seeds across ideal, fake_algiers, and fake_brooklyn. This empirical mutation "
    "outcome is not ground-truth repair or hardware diagnosis."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_inputs() -> dict:
    manifest_path = SOURCE / TASK_SPEC["manifest"]
    oof_path = SOURCE / TASK_SPEC["pred"]
    manifest_hash, oof_hash, split_hash = map(sha256, (manifest_path, oof_path, SPLIT))
    actual_hashes = {"manifest": manifest_hash, "pilot_oof": oof_hash, "split": split_hash}
    if actual_hashes != EXPECTED_HASHES:
        raise ValueError(f"Pinned Qolumbina source/split fingerprint mismatch: {actual_hashes}")

    records, pilot_oof, folds = map(read_csv, (manifest_path, oof_path, SPLIT))
    by_id = {row["record_id"]: row for row in records}
    pilot_by_id = {row["record_id"]: row for row in pilot_oof}
    split_by_id = {row["record_id"]: row for row in folds}
    if (len(by_id) != len(records) or len(pilot_by_id) != len(pilot_oof) or
            len(split_by_id) != len(folds) or set(by_id) != set(pilot_by_id) or
            set(by_id) != set(split_by_id)):
        raise ValueError("Manifest, pilot OOF, and frozen grouped split IDs are not unique and identical")

    group_folds: dict[str, set[int]] = {}
    for record_id, row in by_id.items():
        pilot, split = pilot_by_id[record_id], split_by_id[record_id]
        if int(row["label"]) != int(pilot["y_true"]):
            raise ValueError(f"Pilot OOF target differs from task manifest for {record_id}")
        if int(pilot["fold"]) != int(split["fold"]):
            raise ValueError(f"Pilot OOF fold differs from registered split for {record_id}")
        group_folds.setdefault(str(row["bug_id"]), set()).add(int(split["fold"]))
    crossing = [group for group, assigned in group_folds.items() if len(assigned) != 1]
    if crossing:
        raise ValueError(f"Registered split leaks original programs across folds: {crossing[:5]}")

    classes: dict[str, int] = {}
    for row in records:
        label = str(int(row["label"]))
        classes[label] = classes.get(label, 0) + 1
    return {
        "manifest_rows": len(records), "unique_record_ids": len(by_id),
        "groups": len(group_folds), "folds": len({int(r["fold"]) for r in folds}),
        "label_counts": classes, "cross_fold_groups": len(crossing),
        "hashes": actual_hashes, "target_semantics": TARGET_SEMANTICS,
        "frozen_test_accessed": False,
    }


def validate_outputs(input_audit: dict, task_output: Path) -> dict:
    summary_path = task_output / "qolumbina_multibackend" / "summary.json"
    oof_path = task_output / "qolumbina_multibackend" / "oof_predictions.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    predictions = read_csv(oof_path)
    models = [item["model"] for item in summary["results"]]
    expected = input_audit["manifest_rows"] * len(models)
    seen = {(row["record_id"], row["model"]) for row in predictions}
    records = read_csv(SOURCE / TASK_SPEC["manifest"])
    pilot_oof = read_csv(SOURCE / TASK_SPEC["pred"])
    expected_ids = {row["record_id"] for row in records}
    manifest_by_id = {row["record_id"]: row for row in records}
    folds_by_id = {row["record_id"]: int(row["fold"]) for row in pilot_oof}
    group_folds: dict[tuple[str, str], set[int]] = {}
    for row in predictions:
        rid, model = row["record_id"], row["model"]
        source = manifest_by_id.get(rid)
        if (source is None or model not in models or int(row["y_true"]) != int(source["label"]) or
                int(row["fold"]) != folds_by_id[rid] or str(row["bug_id"]) != str(source["bug_id"])):
            raise ValueError(f"OOF label, fold, or group mismatch for {rid}/{model}")
        group_folds.setdefault((model, str(row["bug_id"])), set()).add(int(row["fold"]))
    if any(len(assigned) != 1 for assigned in group_folds.values()):
        raise ValueError("Candidate sweep OOF output crosses an original-program group across folds")
    if (summary.get("status") != "completed_grouped_qnn_sweep" or len(models) != 2 or
            len(predictions) != expected or len(seen) != expected or
            {row["record_id"] for row in predictions} != expected_ids or
            summary.get("records") != len(records) or summary.get("groups") != input_audit["groups"] or
            summary.get("frozen_test_accessed") is not False):
        raise ValueError("Qolumbina candidate sweep output coverage or status is incomplete")
    checkpoint_hashes: dict[str, dict[str, str]] = {}
    for result in summary["results"]:
        model_hashes = {}
        for checkpoint in result["checkpoint_paths"]:
            path = ROOT / checkpoint
            if not path.is_file():
                raise ValueError(f"Missing candidate checkpoint: {checkpoint}")
            model_hashes[path.parent.name] = sha256(path)
        if len(model_hashes) != input_audit["folds"]:
            raise ValueError(f"Incomplete fold checkpoint set for {result['model']}")
        checkpoint_hashes[result["model"]] = model_hashes
    return {"summary": summary, "prediction_rows": len(predictions),
            "model_names": models, "oof_sha256": sha256(oof_path),
            "checkpoint_hashes": checkpoint_hashes,
            "fold_results_sha256": sha256(task_output / "qolumbina_multibackend" / "fold_results.csv"),
            "task_summary_sha256": sha256(summary_path)}


def main() -> None:
    input_audit = verify_inputs()
    if sweep.q.check_ram() < 700:
        raise MemoryError("Qolumbina candidate sweep requires 700 MiB available at launch")
    task_output = OUTPUT / re.sub(r"[^a-z0-9]+", "_", TASK_NAME.lower()).strip("_")
    if task_output.exists() and (task_output / "summary.json").exists():
        existing = json.loads((task_output / "summary.json").read_text(encoding="utf-8"))
        if (existing.get("source_manifest_sha256") == input_audit["hashes"]["manifest"] and
                existing.get("status") == "completed_grouped_qnn_sweep"):
            output_audit = validate_outputs(input_audit, OUTPUT)
            manifest_path = OUTPUT / "run_manifest.json"
            run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            run_manifest.update({
                "status": "VERIFIED", "prediction_rows": output_audit["prediction_rows"],
                "prediction_sha256": output_audit["oof_sha256"],
                "checkpoint_hashes": output_audit["checkpoint_hashes"],
                "fold_results_sha256": output_audit["fold_results_sha256"],
                "task_summary_sha256": output_audit["task_summary_sha256"],
                "integrity_audit": "passed labels, folds, groups, OOF coverage, and checkpoint existence/hashes",
            })
            manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
            print("Existing Qolumbina multi-backend candidate sweep is complete and validated.")
            return

    sweep.OUT = OUTPUT
    sweep.TASKS = {TASK_NAME: TASK_SPEC}
    start = time.perf_counter()
    try:
        sweep.run_campaign()
    except MemoryError as exc:
        partial = {
            "schema": "qolumbina-multibackend-qnn-candidate-sweep-v1",
            "status": "partial_stopped_by_ram_guard", "task": TASK_NAME,
            "source_hashes": input_audit["hashes"], "error": str(exc),
            "resume_command": ".venv\\Scripts\\python.exe experiments\\train_qolumbina_multibackend_candidate_sweep.py",
            "frozen_test_accessed": False,
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "run_manifest.json").write_text(json.dumps(partial, indent=2) + "\n",
                                                   encoding="utf-8")
        print(json.dumps(partial, indent=2))
        return
    output_audit = validate_outputs(input_audit, OUTPUT)
    run_manifest = {
        "schema": "qolumbina-multibackend-qnn-candidate-sweep-v1",
        "status": "VERIFIED", "task": "Qolumbina multi-backend seeded-mutant detection",
        "task_rows": input_audit["manifest_rows"], "groups": input_audit["groups"],
        "folds": input_audit["folds"], "label_counts": input_audit["label_counts"],
        "target_semantics": TARGET_SEMANTICS,
        "source_hashes": input_audit["hashes"], "split_cross_fold_groups": 0,
        "models": output_audit["model_names"], "epochs": sweep.EPOCHS,
        "batch_size": sweep.BATCH, "optimizer": "Adam", "seed": sweep.SEED,
        "feature_fields": TASK_SPEC["text"], "scaling": "fold-local minmax fitted on training rows only",
        "prediction_rows": output_audit["prediction_rows"],
        "prediction_sha256": output_audit["oof_sha256"],
        "checkpoint_hashes": output_audit["checkpoint_hashes"],
        "fold_results_sha256": output_audit["fold_results_sha256"],
        "task_summary_sha256": output_audit["task_summary_sha256"],
        "elapsed_seconds_this_invocation": time.perf_counter() - start,
        "available_ram_end_mib": round(sweep.q.check_ram(), 1),
        "frozen_test_accessed": False,
        "scope_note": "QNN-only grouped development sweep; no classical models trained or compared.",
        "limitations": ["40 negative vs 506 positive rows; results are exploratory under severe class imbalance.",
                        "Empirical seeded-mutant detection is not hardware diagnosis."],
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps({"status": "VERIFIED", "output": str(OUTPUT),
                      "models": run_manifest["models"],
                      "prediction_rows": run_manifest["prediction_rows"],
                      "elapsed_seconds": run_manifest["elapsed_seconds_this_invocation"],
                      "available_ram_end_mib": run_manifest["available_ram_end_mib"]}, indent=2))


if __name__ == "__main__":
    main()
