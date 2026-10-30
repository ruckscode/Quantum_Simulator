"""Reconcile OOF predictions, group assignments, gradients, and checkpoints."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import struct
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028"
SWEEP = ROOT / "reports/experiments/qml_standalone_accuracy_optimization_20261028/remaining_task_sweeps"
OUT = SWEEP / "candidate_sweep_artifact_audit.json"
TASKS = {
    "bugs4q_core": "bugs4q_standalone/normalized_manifest.csv",
    "bugs4q_na": "bugs4q_na/normalized_manifest.csv",
    "bugs4q_pooled": "bugs4q_joint/normalized_manifest.csv",
    "bugs4q_core_only_transfer": "bugs4q_joint/normalized_manifest.csv",
    "bugs4q_na_only_transfer": "bugs4q_joint/normalized_manifest.csv",
    "lintq": "lintq_warning_triage/normalized_manifest.csv",
    "qbugs_hornbro_subset": "qbugs/normalized_manifest.csv",
    "qolumbina_ideal": "qolumbina_detection_ideal/normalized_manifest.csv",
    "quantum_platform_comment_task": "quantum_platform_223_comment_only/normalized_manifest.csv",
}
REQUIRED_CHECKPOINT_ARRAYS = {
    "theta", "head", "bias", "config", "epoch", "m_t", "v_t", "m_h", "v_h", "m_b", "v_b", "rng_state"
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_npy_scalar(archive: zipfile.ZipFile, name: str):
    data = archive.read(name + ".npy")
    if data[:6] != b"\x93NUMPY":
        raise ValueError(f"Invalid NPY magic for {name}")
    major, minor = data[6], data[7]
    if major == 1:
        header_size = struct.unpack("<H", data[8:10])[0]
        offset = 10 + header_size
    elif major in (2, 3):
        header_size = struct.unpack("<I", data[8:12])[0]
        offset = 12 + header_size
    else:
        raise ValueError(f"Unsupported NPY format {major}.{minor}")
    header = data[10:offset].decode("latin1")
    if "shape': ()" not in header and 'shape\": ()' not in header:
        raise ValueError(f"Expected scalar NPY field {name}")
    dtype = re.search(r"descr':\s*'([^']+)'", header)
    if dtype is None:
        raise ValueError(f"Missing dtype for scalar NPY field {name}")
    code = dtype.group(1)
    payload = data[offset:]
    if code in ("<i4", "=i4"):
        return struct.unpack("<i", payload)[0]
    if code in ("<i8", "=i8"):
        return struct.unpack("<q", payload)[0]
    if code.startswith("<U"):
        return payload.decode("utf-32-le").rstrip("\x00")
    raise ValueError(f"Unsupported scalar dtype {code} for {name}")


def metric(y: list[int], pred: list[int]) -> dict:
    labels = sorted(set(y) | set(pred))
    f1s, recalls, per = [], [], {}
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y, pred))
        fp = sum(a != label and b == label for a, b in zip(y, pred))
        fn = sum(a == label and b != label for a, b in zip(y, pred))
        support = sum(a == label for a in y)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[str(label)] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1s.append(f1)
        recalls.append(recall)
    return {"accuracy": sum(a == b for a, b in zip(y, pred)) / len(y),
            "macro_f1": sum(f1s) / len(labels), "balanced_accuracy": sum(recalls) / len(labels),
            "per_class": per}


def main() -> None:
    tasks = []
    all_checkpoints = 0
    all_gradients = 0
    for task_name, manifest_rel in TASKS.items():
        folder = SWEEP / task_name
        summary_path, prediction_path = folder / "summary.json", folder / "oof_predictions.csv"
        if not summary_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"Missing complete task artifacts for {task_name}")
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        predictions = read_csv(prediction_path)
        manifest_path = SOURCE / manifest_rel
        manifest = read_csv(manifest_path)
        manifest_ids = {row["record_id"] for row in manifest}
        prediction_ids = {row["record_id"] for row in predictions}
        if manifest_ids != prediction_ids or len(predictions) != len(manifest) * len(summary["results"]):
            raise AssertionError(f"Record coverage mismatch for {task_name}")
        group_folds: dict[str, set[int]] = {}
        for row in predictions:
            group_folds.setdefault(str(row["bug_id"]), set()).add(int(row["fold"]))
        if any(len(folds) != 1 for folds in group_folds.values()):
            raise AssertionError(f"Group leakage in {task_name}")
        model_reports = []
        for model in summary["results"]:
            name = model["model"]
            rows = [row for row in predictions if row["model"] == name]
            if len(rows) != len(manifest) or len({row["record_id"] for row in rows}) != len(manifest):
                raise AssertionError(f"Per-model OOF row mismatch in {task_name}/{name}")
            recalculated = metric([int(row["y_true"]) for row in rows], [int(row["y_pred"]) for row in rows])
            for key in ("accuracy", "macro_f1", "balanced_accuracy"):
                if abs(recalculated[key] - float(model[key])) > 1e-12:
                    raise AssertionError(f"Metric mismatch in {task_name}/{name}/{key}")
            fold_rows = [row for row in summary["fold_results"] if row["model"] == name]
            if len(fold_rows) != 5 or not all(row["gradient_check"]["passed"] for row in fold_rows):
                raise AssertionError(f"Fold gradient checks incomplete in {task_name}/{name}")
            checkpoint_records = []
            if len(model["checkpoint_paths"]) != 5:
                raise AssertionError(f"Expected five checkpoints for {task_name}/{name}")
            for rel in model["checkpoint_paths"]:
                checkpoint = ROOT / rel.replace("\\", "/")
                if not zipfile.is_zipfile(checkpoint):
                    raise AssertionError(f"Invalid NPZ checkpoint: {checkpoint}")
                with zipfile.ZipFile(checkpoint) as archive:
                    if archive.testzip() is not None:
                        raise AssertionError(f"Corrupt NPZ checkpoint: {checkpoint}")
                    arrays = {item.removesuffix(".npy") for item in archive.namelist()}
                    if not REQUIRED_CHECKPOINT_ARRAYS <= arrays:
                        raise AssertionError(f"Missing model/optimizer/RNG state in {checkpoint}")
                    epoch = int(read_npy_scalar(archive, "epoch"))
                    configuration = json.loads(read_npy_scalar(archive, "config"))
                if epoch != 4:
                    raise AssertionError(f"Expected completed epoch 4, found {epoch}: {checkpoint}")
                checkpoint_records.append({"path": str(checkpoint.relative_to(ROOT)),
                                           "sha256": checksum(checkpoint), "epoch": epoch,
                                           "configuration": configuration})
            all_checkpoints += len(checkpoint_records)
            all_gradients += len(fold_rows)
            model_reports.append({"model": name, "records": len(rows), "metrics": recalculated,
                                  "gradient_checks_passed": len(fold_rows),
                                  "checkpoints": checkpoint_records})
        tasks.append({"task": task_name, "status": summary["status"], "records": len(manifest),
                      "groups": len(group_folds), "source_manifest_sha256": checksum(manifest_path),
                      "oof_prediction_sha256": checksum(prediction_path), "models": model_reports})
    result = {"schema": "qml-candidate-sweep-artifact-audit-v1", "status": "verified",
              "scope": "nine grouped development QNN candidate sweeps; fixed four-epoch configurations",
              "tasks": tasks, "task_count": len(tasks), "checkpoint_count": all_checkpoints,
              "gradient_checks_passed": all_gradients, "frozen_test_accessed": False,
              "production_model_modified": False,
              "matched_classical_controls": "Pending; historical scores use different hashed/SVD/metadata representations and are not substituted.",
              "out_of_scope": ["Full-data QMutBench binary QNN", "Qolumbina multi-backend incomplete task",
                               "Nine-task 14-model classical and imbalance-method bundle not executed"]}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    status_path = SWEEP / "remaining_task_sweep_status.json"
    matched_path = SWEEP / "matched_qnn_classical_comparison.json"
    matched = json.loads(matched_path.read_text(encoding="utf-8-sig")) if matched_path.is_file() else None
    status = {"status": "candidate_and_matched_classical_sweeps_complete" if matched and matched.get("status") == "verified"
              else "candidate_sweeps_complete_matched_classical_pending",
              "scope": "two fixed QNN candidate architectures across the nine listed source tasks; not the separate broad classical/imbalance suite",
              "completed_tasks": [{"task": row["task"], "records": row["records"], "groups": row["groups"],
                                   "models": [{"model": model["model"], "accuracy": model["metrics"]["accuracy"],
                                               "macro_f1": model["metrics"]["macro_f1"],
                                               "balanced_accuracy": model["metrics"]["balanced_accuracy"]}
                                              for model in row["models"]]} for row in tasks],
              "not_completed": ["registered broad 14-model classical and training-fold-only imbalance suite",
                                "Qolumbina multi-backend task", "QMutBench full-data binary QNN"],
              "matched_classical_comparison": {"status": matched.get("status") if matched else "pending",
                  "path": str(matched_path.relative_to(ROOT)) if matched else None,
                  "task_count": matched.get("task_count") if matched else 0},
              "matched_classical_first_attempt": {"status": "blocked_by_ram_preflight_then_completed_after_recovery",
                          "available_ram_mib_at_first_attempt": 507.2, "required_startup_mib": 600,
                          "in_run_ram_floor_mib": 500, "final_status": matched.get("status") if matched else "pending"},
              "frozen_test_accessed": False}
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(OUT.relative_to(ROOT)), "status": result["status"],
                      "tasks": len(tasks), "checkpoints": all_checkpoints,
                      "gradient_checks": all_gradients,
                      "matched_baselines": matched.get("status") if matched else "pending_memory"}, indent=2))


if __name__ == "__main__":
    main()
