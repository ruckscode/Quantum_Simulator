"""Validate saved Replace-specialist fold predictions and rebuild aggregates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from experiments.qmutbench_improvement_campaign import OUT as CACHE_OUT, _category_codes
from experiments.qmutbench_replace_specialist import OUT, metrics, grouped_bootstrap

ROOT = Path(__file__).resolve().parents[1]
OOF_PATH = ROOT / "data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/mutation_diff_oof.npz"


def main():
    oof = np.load(OOF_PATH, allow_pickle=True)
    ids, y = oof["record_id"].astype(np.int64), oof["y"].astype(np.int8)
    groups, folds = oof["origin_program"].astype(str), oof["fold"].astype(np.int8)
    p_base = oof["probability"].astype(np.float32)
    pred_base = (p_base >= .5).astype(np.int8)
    cat = np.load(CACHE_OUT / "categorical_tokens.npy", mmap_mode="r")
    replace_token = _category_codes(["Replace"])[0]
    is_replace = np.asarray(cat[ids, 1] == replace_token)
    prior = json.loads((OUT / "result_summary.json").read_text(encoding="utf-8"))
    parts, per_fold, checkpoints = [], {}, {}
    row_by_id = {int(record_id): i for i, record_id in enumerate(ids)}
    for fold in range(1, 6):
        path = OUT / f"fold{fold}_replace_predictions.csv.gz"
        frame = pd.read_csv(path)
        expected = np.flatnonzero((folds == fold) & is_replace)
        loc = np.fromiter((row_by_id[int(r)] for r in frame.record_id), dtype=np.int64, count=len(frame))
        if (frame.record_id.duplicated().any() or len(frame) != len(expected) or
                not np.array_equal(np.sort(frame.record_id.to_numpy(np.int64)), np.sort(ids[expected])) or
                not np.array_equal(y[loc], frame.y_true.to_numpy(np.int8)) or
                not np.array_equal(groups[loc], frame.origin_program.to_numpy(str)) or
                not np.array_equal(folds[loc], frame.fold.to_numpy(np.int8)) or
                not is_replace[loc].all() or
                not np.allclose(p_base[loc], frame.p_champion.to_numpy(np.float32), atol=1e-6)):
            raise ValueError(f"Fold {fold} checkpoint ID, label, origin, fold, operation, or baseline mismatch")
        base_fold = metrics(y[loc], pred_base[loc])
        specialist_fold = metrics(y[loc], frame.y_pred_specialist.to_numpy(np.int8))
        original = prior["per_fold"][str(fold)]["replace"]
        for key in ("macro_f1", "accuracy", "balanced_accuracy"):
            if abs(specialist_fold[key] - original["specialist_replace"][key]) > 1e-12:
                raise ValueError(f"Fold {fold} checkpoint differs from prior saved fit metric: {key}")
        checkpoints[str(fold)] = {
            "rows": len(frame), "origins": int(np.unique(groups[loc]).size),
            "train_replace_rows": original["train_replace_rows"],
            "train_replace_origins": original["train_replace_origins"],
            "baseline_replace": base_fold, "specialist_replace": specialist_fold,
            "fit_seconds": original["fit_seconds"],
            "available_ram_before_fit_mib": original["available_ram_before_fit_mib"],
            "available_ram_after_fit_mib": original["available_ram_after_fit_mib"],
            "checkpoint_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        }
        per_fold[str(fold)] = {"replace": checkpoints[str(fold)]}
        parts.append(frame)

    specialist = pd.concat(parts, ignore_index=True)
    if specialist.record_id.duplicated().any() or len(specialist) != int(is_replace.sum()):
        raise ValueError("Five fold checkpoints do not exactly cover the Replace slice")
    specialist.to_csv(OUT / "replace_oof_predictions.csv.gz", index=False, compression="gzip")
    loc = np.fromiter((row_by_id[int(r)] for r in specialist.record_id), dtype=np.int64,
                      count=len(specialist))
    pred_specialist = np.full(len(ids), -1, dtype=np.int8)
    pred_specialist[loc] = specialist.y_pred_specialist.to_numpy(np.int8)
    hybrid_prob = p_base.copy()
    hybrid_prob[loc] = specialist.p_specialist.to_numpy(np.float32)
    hybrid_pred = (hybrid_prob >= .5).astype(np.int8)

    # Fold-specific hybrid metrics use champion OOF predictions for every non-Replace case.
    for fold in range(1, 6):
        rows = folds == fold
        per_fold[str(fold)]["champion_all"] = metrics(y[rows], pred_base[rows])
        per_fold[str(fold)]["hybrid_all"] = metrics(y[rows], hybrid_pred[rows])
        per_fold[str(fold)]["overall_macro_f1_delta"] = (
            per_fold[str(fold)]["hybrid_all"]["macro_f1"] - per_fold[str(fold)]["champion_all"]["macro_f1"])
        per_fold[str(fold)]["replace"] = checkpoints[str(fold)]

    ci_all, n_all = grouped_bootstrap(y, pred_base, hybrid_pred, groups,
                                      np.ones(len(y), dtype=bool))
    ci_replace, n_replace = grouped_bootstrap(y, pred_base, hybrid_pred, groups, is_replace)
    origin_rows = []
    for origin in sorted(set(groups.tolist())):
        all_mask = groups == origin
        rep_mask = all_mask & is_replace
        origin_rows.append({
            "origin_program": origin,
            "rows": int(all_mask.sum()),
            "replace_rows": int(rep_mask.sum()),
            "replace_survivor_count": int((y[rep_mask] == 0).sum()),
            "replace_detected_count": int((y[rep_mask] == 1).sum()),
            "champion_accuracy": float(accuracy_score(y[all_mask], pred_base[all_mask])),
            "hybrid_accuracy": float(accuracy_score(y[all_mask], hybrid_pred[all_mask])),
            "champion_macro_f1": float(f1_score(y[all_mask], pred_base[all_mask], labels=[0, 1], average="macro", zero_division=0)),
            "hybrid_macro_f1": float(f1_score(y[all_mask], hybrid_pred[all_mask], labels=[0, 1], average="macro", zero_division=0)),
            "replace_champion_macro_f1": float(f1_score(y[rep_mask], pred_base[rep_mask], labels=[0, 1], average="macro", zero_division=0)) if rep_mask.any() else float("nan"),
            "replace_specialist_macro_f1": float(f1_score(y[rep_mask], pred_specialist[rep_mask], labels=[0, 1], average="macro", zero_division=0)) if rep_mask.any() else float("nan"),
        })
    pd.DataFrame(origin_rows).to_csv(OUT / "per_origin_metrics.csv", index=False)
    summary = {
        "schema": "qmutbench-replace-specialist-result-v1", "status": "complete_checkpoints_validated",
        "routing": "operation==Replace from documented model input; champion used on all non-Replace records",
        "configuration": prior["configuration"], "fold_count": 5,
        "replace_rows": int(is_replace.sum()), "replace_origins": int(np.unique(groups[is_replace]).size),
        "overall": {"champion": metrics(y, pred_base), "hybrid": metrics(y, hybrid_pred)},
        "replace_slice": {"champion": metrics(y[is_replace], pred_base[is_replace]),
                          "specialist": metrics(y[is_replace], pred_specialist[is_replace])},
        "overall_paired_origin_bootstrap_95ci_delta": ci_all, "overall_bootstrap_origins": n_all,
        "replace_slice_paired_origin_bootstrap_95ci_delta": ci_replace,
        "replace_bootstrap_origins": n_replace, "bootstrap_repetitions": 2000,
        "bootstrap_seed": 20261030, "per_fold": per_fold,
        "fold_delta_mean": float(np.mean([v["overall_macro_f1_delta"] for v in per_fold.values()])),
        "fold_delta_sd": float(np.std([v["overall_macro_f1_delta"] for v in per_fold.values()], ddof=1)),
        "runtime_seconds": float(sum(v["fit_seconds"] for v in checkpoints.values())),
        "checkpoint_validation": "all five fold files exactly match expected Replace record IDs, labels, origins, folds, operation cache, and baseline probabilities; specialist fold metrics match saved fit-time summary",
        "all_test_rows_excluded": True,
    }
    (OUT / "result_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    analysis_path = OUT / "baseline_slice_analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["status"] = "complete_checkpoints_validated"
    analysis["completed_folds"] = [1, 2, 3, 4, 5]
    analysis["resumed_from_saved_fold5_checkpoint"] = True
    analysis["verification_rerun_note"] = "A later rerun completed folds 1-4 but stopped before fold 5 at the 700 MiB fit-start guard. The already completed fold-5 checkpoint was preserved and validated with the other four outputs. No partial rerun result was substituted for the five-fold aggregate."
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"replace_baseline": summary["replace_slice"]["champion"]["macro_f1"],
                      "replace_specialist": summary["replace_slice"]["specialist"]["macro_f1"],
                      "champion_overall": summary["overall"]["champion"]["macro_f1"],
                      "hybrid_overall": summary["overall"]["hybrid"]["macro_f1"],
                      "overall_macro_f1_delta_ci": ci_all["macro_f1"],
                      "fold_deltas": [v["overall_macro_f1_delta"] for v in per_fold.values()],
                      "total_fit_seconds": summary["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
