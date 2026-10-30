"""Re-score saved OOF files and verify that QNN fold checkpoints reproduce them."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028"
RUNS = [
    "bugs4q_standalone/qnn_gradient_fix",
    "bugs4q_na/qnn",
    "qbugs/qnn_hybrid",
    "quantum_platform_223_comment_only/qnn_hybrid",
    "bugs4q_joint/qnn_hybrid",
    "bugs4q_joint/qnn_train_core_complete",
    "bugs4q_joint/qnn_train_na_complete",
    "qolumbina_detection_ideal/qnn",
    "lintq_warning_triage/qnn",
]


def score(y, pred):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=sorted(set(y)), zero_division=0)
    return {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "per_class": {str(c): {"precision": float(p[i]), "recall": float(r[i]),
                                   "f1": float(f[i]), "support": int(s[i])}
                          for i, c in enumerate(sorted(set(y)))}}


def verify_checkpoint_reproduction(run: Path, summary: dict) -> dict:
    import joblib
    import numpy as np
    import pandas as pd
    import pennylane as qml
    from pennylane import numpy as pnp
    from sklearn.model_selection import GroupKFold

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from experiments.bugs4q_qnn_grouped_benchmark import _make_qnn

    manifest_path = run.parent / "normalized_manifest.csv"
    if not manifest_path.exists():
        return {"status": "NOT_VERIFIED_NO_PARENT_MANIFEST"}
    df = pd.read_csv(manifest_path)
    y = df.label.to_numpy(dtype=int)
    groups = df.bug_id.to_numpy()
    checkpoint_path = run / "fold_checkpoints.json"
    pred_path = run / "oof_predictions.csv"
    if not checkpoint_path.exists() or not pred_path.exists():
        return {"status": "INCOMPLETE_MISSING_CHECKPOINT_OR_OOF"}
    checkpoints = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    pred = pd.read_csv(pred_path)
    qnn_names = {x["model"] for x in checkpoints}
    saved = pred[pred.model.isin(qnn_names)]
    split = GroupKFold(n_splits=5)
    valid_by_fold = {fold: valid for fold, (_, valid) in enumerate(split.split(df.code, y, groups), 1)}
    max_abs_error = 0.0
    checked = 0
    missing = []
    fold_group_overlap = {}
    for ck in checkpoints:
        fold = int(ck["fold"])
        model_name = ck["model"]
        valid = valid_by_fold[fold]
        prep_path = run / f"preprocessing_fold{fold}.joblib"
        if not prep_path.exists():
            missing.append({"model": model_name, "fold": fold, "problem": "missing preprocessing"})
            continue
        prep = joblib.load(prep_path)
        vec = prep["vectorizer"].transform(df.code.iloc[valid])
        z = prep["svd"].transform(vec)
        xva = prep["scaler"].transform(z)
        seed = int(ck["seed"])
        model = _make_qnn(model_name, seed, qml, pnp, np)
        w = pnp.array(ck["weights"], requires_grad=False)
        h = pnp.array(ck["readout_weights"], requires_grad=False) if ck["readout_weights"] is not None else None
        b = pnp.array(ck["readout_bias"], requires_grad=False) if ck["readout_bias"] is not None else None
        fold_saved = saved[(saved.fold == fold) & (saved.model == model_name)].set_index("record_id")
        for row_i, ix in enumerate(valid):
            rid = str(df.record_id.iloc[ix])
            if rid not in fold_saved.index:
                missing.append({"model": model_name, "fold": fold, "record_id": rid, "problem": "missing OOF row"})
                continue
            zq = np.asarray(model["circuit"](pnp.array(xva[row_i]), w), dtype=float)
            if model_name == "angle_hybrid_readout":
                logit = float(np.dot(np.asarray(h), zq) + float(b))
                p1 = float(1.0 / (1.0 + np.exp(-logit)))
            else:
                p1 = float(np.clip((1.0 - zq[0]) / 2.0, 0.0, 1.0))
            got = float(fold_saved.loc[rid, "prob_buggy"])
            max_abs_error = max(max_abs_error, abs(got - p1))
            checked += 1
        train_ids = set(map(str, prep.get("training_bug_ids", [])))
        valid_ids = set(map(str, groups[valid]))
        fold_group_overlap[str(fold)] = len(train_ids & valid_ids)
    return {"status": "PASS" if not missing and max_abs_error < 1e-8 and all(v == 0 for v in fold_group_overlap.values()) else "FAIL",
            "checkpoints": len(checkpoints), "checkpoint_predictions_recomputed": checked,
            "max_absolute_probability_error": max_abs_error, "missing_items": missing[:20],
            "fold_group_overlap_from_saved_preprocessors": fold_group_overlap}


def main() -> None:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

    reports = []
    started = time.perf_counter()
    for rel in RUNS:
        run = CAMPAIGN / rel
        summary_path, pred_path, metrics_path, ck_path = (run / x for x in
            ("summary.json", "oof_predictions.csv", "fold_metrics.json", "fold_checkpoints.json"))
        record = {"run": rel, "exists": run.exists()}
        if not all(x.exists() for x in (summary_path, pred_path, metrics_path, ck_path)):
            record["status"] = "PARTIAL_MISSING_REQUIRED_ARTIFACTS"
            reports.append(record)
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        pred = pd.read_csv(pred_path)
        checkpoints = json.loads(ck_path.read_text(encoding="utf-8"))
        fold_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        recalc = []
        for model, part in pred.groupby("model"):
            y = part.y_true.to_numpy(dtype=int)
            yh = part.y_pred.to_numpy(dtype=int)
            m = {"model": model, **score(y, yh)}
            if "prob_buggy" in part:
                m["mean_log_loss"] = float(-np.mean(y*np.log(np.clip(part.prob_buggy,1e-15,1))+
                    (1-y)*np.log(np.clip(part.prob_fixed,1e-15,1))))
            recalc.append(m)
        overlaps = [int(x.get("group_overlap", 0)) for x in fold_metrics if x.get("fold") is not None]
        gradient_checks = [x.get("gradient_check", {}).get("passed") for x in fold_metrics
                          if x.get("gradient_check") is not None]
        histories = [h.get("loss", []) for h in json.loads((run / "training_history.json").read_text(encoding="utf-8"))]
        updates = []
        for ck in checkpoints:
            w0, w1 = np.asarray(ck["initial_weights"]), np.asarray(ck["weights"])
            updates.append(float(np.linalg.norm(w1-w0)))
            if ck.get("readout_weights") is not None:
                updates.append(float(np.linalg.norm(np.asarray(ck["readout_weights"])-np.asarray(ck["initial_readout_weights"]))))
        record.update({"status": summary.get("status"), "rows": int(len(pred)),
            "models_in_predictions": sorted(pred.model.unique().tolist()),
            "checkpoint_count": len(checkpoints), "folds_present": sorted(pred.fold.unique().tolist()),
            "recalculated_oof_metrics": recalc,
            "zero_fold_group_overlap": bool(overlaps) and all(x == 0 for x in overlaps),
            "gradient_check_count": len(gradient_checks), "all_gradient_checks_passed": bool(gradient_checks) and all(gradient_checks),
            "all_checkpoints_updated": bool(updates) and all(x > 1e-8 for x in updates),
            "all_saved_loss_histories_decreased": bool(histories) and all(h and h[-1] < h[0] for h in histories),
            "checkpoint_reproduction": verify_checkpoint_reproduction(run, summary)})
        reports.append(record)
    # Audit the existing three-class QMutBench development campaign only. Do
    # not inspect any confirmation or frozen-test directory here.
    qmut_root = ROOT / "reports/experiments/qmutbench_three_class_qnn_campaign_20261027"
    qmut = {"status": "missing"}
    required = [qmut_root / "predictions.csv.gz", qmut_root / "candidate_leaderboard.csv",
                qmut_root / "grouped_split_manifest.csv", qmut_root / "checkpoint_manifest.json"]
    if all(p.exists() for p in required):
        oof = pd.read_csv(required[0])
        board = pd.read_csv(required[1])
        split = pd.read_csv(required[2])
        checkpoint_manifest = json.loads(required[3].read_text(encoding="utf-8"))
        recalculated = []
        for model, part in oof.groupby("model"):
            y, yh = part.y_true.to_numpy(dtype=int), part.y_pred.to_numpy(dtype=int)
            from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
            row = {"model": model, "macro_f1": float(f1_score(y, yh, average="macro", zero_division=0)),
                   "accuracy": float(accuracy_score(y, yh)),
                   "balanced_accuracy": float(balanced_accuracy_score(y, yh))}
            ref = board[board.model.eq(model)]
            if len(ref):
                row["leaderboard_macro_f1_abs_error"] = abs(row["macro_f1"]-float(ref.iloc[0].macro_f1))
                row["leaderboard_accuracy_abs_error"] = abs(row["accuracy"]-float(ref.iloc[0].accuracy))
            recalculated.append(row)
        val_split = split[split.role.eq("validation")]
        oof_key = set(zip(oof.record_id.astype(str), oof.fold.astype(int)))
        split_key = set(zip(val_split.record_id.astype(str), val_split.fold.astype(int)))
        qnn_names = sorted(set(oof.model) & {"three_class_vqc", "three_class_vqc_amplitude", "three_class_vqc_angle_linear"})
        qmut_summary = json.loads((qmut_root / "campaign_summary.json").read_text(encoding="utf-8"))
        qmut = {"status": "OOF_TRAINING_ARTIFACTS_PRESENT_NO_MODEL_CHECKPOINT",
                "records": int(oof.record_id.nunique()), "models": sorted(oof.model.unique().tolist()),
                "folds": sorted(map(int, oof.fold.unique())), "origin_groups": int(oof.origin_program.nunique()),
                "oof_records_match_grouped_validation_manifest": oof_key <= split_key,
                "fold_metrics_recalculated": recalculated,
                "gradient_and_optimizer_claims_from_saved_fold_metrics": qmut_summary.get("fold_metrics", []),
                "real_data_parameter_checkpoint_status": checkpoint_manifest.get("status"),
                "real_data_checkpoint_present": checkpoint_manifest.get("training_candidate_checkpoint") is not None,
                "qnn_candidate_count": len(qnn_names), "frozen_test_opened": False,
                "scope": "development OOF files only; confirmation and frozen-test directories were not read"}
    # A later bounded recovery run stores full optimizer/RNG checkpoints in a
    # separate versioned output; keep that distinct from the original archive.
    recovery_audit = CAMPAIGN / "qmutbench_threeclass_resumable_recovery_20261029/checkpoint_audit.json"
    if recovery_audit.exists():
        recovery = json.loads(recovery_audit.read_text(encoding="utf-8"))
        qmut["original_campaign_checkpoint_status"] = qmut.get("real_data_parameter_checkpoint_status")
        qmut["status"] = ("OOF_AND_VERSIONED_RESUMABLE_CHECKPOINT_RECOVERY_VERIFIED"
                          if recovery.get("status") == "VERIFIED"
                          else "OOF_PRESENT_RECOVERY_REQUIRES_REVIEW")
        qmut["resumable_checkpoint_recovery"] = {
            "audit_path": str(recovery_audit.relative_to(ROOT)),
            "status": recovery.get("status"),
            "issues": recovery.get("issues"),
            "records": recovery.get("records"), "origins": recovery.get("origins"),
            "folds": recovery.get("folds"), "models": recovery.get("models"),
            "resumable_checkpoints_verified": recovery.get("resumable_checkpoints_verified"),
            "replayed_oof_probabilities": recovery.get("replayed_oof_probabilities"),
            "matched_classical_controls": recovery.get("matched_classical_controls"),
            "frozen_test_accessed": recovery.get("frozen_test_accessed")}
    binary_progress_path = ROOT / "reports/experiments/qmutbench_binary_qnn_full_development_chunk16_20261029/progress_audit.json"
    qmut_binary = {"status": "not_started_or_no_verified_progress"}
    if binary_progress_path.exists():
        partial = json.loads(binary_progress_path.read_text(encoding="utf-8"))
        qmut_binary = {"status": "PARTIAL_RESUMABLE_PROGRESS_VERIFIED" if partial.get("status") == "VERIFIED" else "REVIEW_REQUIRED",
            "progress_audit_path": str(binary_progress_path.relative_to(ROOT)),
            "campaign_status": partial.get("campaign_status"),
            "active_fold": partial.get("active_fold"), "committed_chunks": partial.get("committed_chunks"),
            "processed_training_records": partial.get("processed_training_records"),
            "optimizer_updates": partial.get("optimizer_updates"),
            "cumulative_training_seconds": partial.get("cumulative_training_seconds"),
            "cumulative_circuit_executions": partial.get("cumulative_circuit_executions"),
            "estimated_total_optimizer_updates": partial.get("estimated_total_optimizer_updates"),
            "estimated_total_qnn_training_hours_at_observed_rate": partial.get("estimated_total_qnn_training_hours_at_observed_rate"),
            "minimum_available_ram_mib": partial.get("minimum_available_ram_mib"),
            "maximum_checkpointed_peak_rss_mib": partial.get("maximum_checkpointed_peak_rss_mib"),
            "frozen_test_accessed": partial.get("frozen_test_accessed")}
    result = {"status": "completed_artifact_audit", "runs": reports,
              "qmutbench_three_class": qmut,
              "qmutbench_binary_full_development": qmut_binary,
              "runtime_seconds": time.perf_counter()-started,
              "frozen_test_opened": False, "production_modified": False}
    target = CAMPAIGN / "training_artifact_audit.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"audit": str(target.relative_to(ROOT)), "runs": len(reports),
                      "checkpoint_results": [{"run": x["run"], "status": x.get("checkpoint_reproduction", {}).get("status"),
                                              "recomputed": x.get("checkpoint_reproduction", {}).get("checkpoint_predictions_recomputed"),
                                              "max_err": x.get("checkpoint_reproduction", {}).get("max_absolute_probability_error")}
                                             for x in reports]}, indent=2))


if __name__ == "__main__":
    main()
