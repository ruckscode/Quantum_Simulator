"""Verify resumable QMutBench three-class checkpoints and development OOF replay."""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028"
RUN = CAMPAIGN / "qmutbench_threeclass_resumable_recovery_20261029"
REFERENCE = ROOT / "reports/experiments/qmutbench_three_class_qnn_campaign_20261027"
sys.path.insert(0, str(ROOT / "experiments"))
import train_qmutbench_threeclass_light_qnn as trainer

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def read(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def main():
    data_path = trainer.CAMPAIGN / "qmutbench_threeclass_compact.csv"
    split_path = REFERENCE / "grouped_split_manifest.csv"
    feature_path = REFERENCE / "features_by_fold.json"
    rows = read(data_path); split = read(split_path)
    features = json.loads(feature_path.read_text(encoding="utf-8"))
    summary_path = RUN / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pred_rows = read(RUN / "oof_predictions.csv")
    by_pred = {(r["model"], r["record_id"]): r for r in pred_rows}
    labels = np.asarray([int(r["y_true"]) for r in rows])
    groups = [r["origin_program"] for r in rows]
    issues = []; checked = 0; ck_count = 0
    if summary.get("status") != "completed_development_oof_with_reloadable_checkpoints":
        issues.append("run_status_invalid")
    if len(rows) != 205 or len(set(groups)) != 12:
        issues.append("development_sample_count_mismatch")
    configs = {r["model"]: r["configuration"] for r in summary["results"]}
    for fold in (1, 2, 3):
        role = {int(r["record_id"]): r["role"] for r in split if int(r["fold"]) == fold}
        vi = np.asarray([i for i, r in enumerate(rows) if role.get(int(r["record_id"])) == "validation"])
        ti = np.asarray([i for i, r in enumerate(rows) if role.get(int(r["record_id"])) == "train"])
        if set(groups[i] for i in vi) & set(groups[i] for i in ti):
            issues.append(f"origin_leakage_fold_{fold}")
        if len(vi) == 0 or len(ti) == 0:
            issues.append(f"empty_fold_{fold}")
            continue
        xtr, xva = trainer.encode_saved_fold(rows, ti, vi, features[fold - 1]["features"])
        for name, cfg in configs.items():
            cp = RUN / name / f"fold_{fold}" / "checkpoint.npz"
            hist_path = cp.with_name("training_history.json")
            result_path = cp.with_name("fold_result.json")
            if not cp.is_file() or not hist_path.is_file() or not result_path.is_file():
                issues.append(f"missing_fold_artifact:{name}:{fold}"); continue
            with np.load(cp, allow_pickle=False) as z:
                required = {"theta", "head", "bias", "config", "epoch", "mt", "vt", "mh", "vh", "mb", "vb", "rng_state"}
                if not required.issubset(z.files):
                    issues.append(f"nonresumable_checkpoint:{name}:{fold}"); continue
                c = json.loads(str(z["config"].item()))
                if c != cfg or int(z["epoch"]) != 12:
                    issues.append(f"checkpoint_config_or_epoch_mismatch:{name}:{fold}")
                x = xva[:, :c["features"]]
                if c["encoding"] == "amplitude":
                    x = np.pad(x, ((0, 0), (0, max(0, 4 - x.shape[1]))))[:, :4]
                probs, _ = trainer.probs(x, z["theta"], z["head"], z["bias"], c)
            for j, i in enumerate(vi):
                rid = rows[i]["record_id"]; saved = by_pred.get((name, rid))
                if saved is None:
                    issues.append(f"oof_missing:{name}:{rid}"); continue
                if int(saved["y_true"]) != int(labels[i]) or int(saved["fold"]) != fold:
                    issues.append(f"oof_label_or_fold_mismatch:{name}:{rid}")
                for cls in range(3):
                    if abs(float(saved[f"prob_{cls}"]) - float(probs[j, cls])) > 2e-7:
                        issues.append(f"probability_replay_mismatch:{name}:{fold}:{rid}:{cls}")
                checked += 1
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
            if [int(r["epoch"]) for r in hist] != list(range(1, 13)):
                issues.append(f"history_epoch_gap:{name}:{fold}")
            ck_count += 1
    baseline_audit = {"status": "not_run"}
    baseline_path = RUN / "matched_classical_oof.csv"
    comparison_path = RUN / "matched_classical_comparison.json"
    if baseline_path.exists() and comparison_path.exists():
        baseline = read(baseline_path)
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        qnn_ref = {(r["record_id"], int(r["fold"]), r["origin_program"], int(r["y_true"])) for r in pred_rows}
        base_models = sorted({r["model"] for r in baseline})
        base_metrics = {}
        for model in base_models:
            part = [r for r in baseline if r["model"] == model]
            key = {(r["record_id"], int(r["fold"]), r["origin_program"], int(r["y_true"])) for r in part}
            if key != qnn_ref or len(part) != len(qnn_ref):
                issues.append(f"matched_control_fold_or_label_mismatch:{model}")
            base_metrics[model] = trainer.score(np.asarray([int(r["y_true"]) for r in part]),
                                                 np.asarray([int(r["y_pred"]) for r in part]))
            for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
                observed = float(comparison["classical_metrics"][model][metric])
                if abs(base_metrics[model][metric] - observed) > 1e-12:
                    issues.append(f"matched_control_metric_mismatch:{model}:{metric}")
        baseline_audit = {"status": "VERIFIED" if not any("matched_" in issue for issue in issues) else "FAILED",
            "rows_per_control": len(qnn_ref), "controls": base_models,
            "metrics_recalculated": base_metrics, "comparison_sha256": sha(comparison_path),
            "oof_sha256": sha(baseline_path), "frozen_test_accessed": False}
    report = {"schema": "qmutbench-threeclass-resumable-checkpoint-audit-v1",
        "status": "VERIFIED" if not issues else "FAILED", "issues": issues,
        "records": len(rows), "origins": len(set(groups)), "folds": 3,
        "models": len(configs), "resumable_checkpoints_verified": ck_count,
        "replayed_oof_probabilities": checked, "max_probability_tolerance": 2e-7,
        "matched_classical_controls": baseline_audit,
        "source_sha256": sha(data_path), "split_sha256": sha(split_path),
        "feature_manifest_sha256": sha(feature_path), "run_summary_sha256": sha(summary_path),
        "oof_sha256": sha(RUN / "oof_predictions.csv"), "frozen_test_accessed": False}
    (RUN / "checkpoint_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if issues: raise AssertionError(f"checkpoint audit failed: {issues[:8]}")

if __name__ == "__main__":
    main()
