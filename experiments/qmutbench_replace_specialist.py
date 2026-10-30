"""Grouped OOF screen for a small Replace-only mutation specialist.

The existing champion supplies predictions for every non-Replace record.
Replace is an explicit inference input in the champion schema, so routing is
performed only from that input category. All specialist preprocessing is fit
on Replace records from training origins inside each saved fold.
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import qmutbench_improvement_campaign as campaign

OUT = ROOT / "reports/experiments/qmutbench_replace_specialist_20261027"
OOF_PATH = ROOT / "data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/mutation_diff_oof.npz"
OOF_SHA = "bfc1affdd40ac1032b65e417548075ce066407297fc603085ba6519403f56367"
MIN_START = 700 * 1024**2
STOP_FLOOR = 500 * 1024**2
SEED = 20261030


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix_true_0_1_pred_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "per_class": {str(c): {"precision": float(pr[i]), "recall": float(rc[i]),
                               "f1": float(f1[i]), "support": int(sup[i])}
                      for i, c in enumerate([0, 1])},
    }


def score_matrix(cm: np.ndarray) -> np.ndarray:
    tn, fp, fn, tp = cm.ravel()
    f0 = 2 * tn / max(1, 2 * tn + fp + fn)
    f1 = 2 * tp / max(1, 2 * tp + fn + fp)
    return np.array([(tn + tp) / max(1, cm.sum()),
                     .5 * (tn / max(1, tn + fp) + tp / max(1, fn + tp)),
                     .5 * (f0 + f1)])


def grouped_bootstrap(y, p_base, p_hybrid, groups, mask, seed=SEED, reps=2000):
    names = sorted(set(groups[mask].tolist()))
    cm_base, cm_hybrid = [], []
    for name in names:
        ix = mask & (groups == name)
        cm_base.append(confusion_matrix(y[ix], p_base[ix], labels=[0, 1]))
        cm_hybrid.append(confusion_matrix(y[ix], p_hybrid[ix], labels=[0, 1]))
    base = np.stack(cm_base); hybrid = np.stack(cm_hybrid)
    rng = np.random.default_rng(seed)
    delta = np.empty((reps, 3), dtype=float)
    for i in range(reps):
        ix = rng.integers(0, len(names), len(names))
        delta[i] = score_matrix(hybrid[ix].sum(axis=0)) - score_matrix(base[ix].sum(axis=0))
    return {k: [float(np.percentile(delta[:, j], 2.5)), float(np.percentile(delta[:, j], 97.5))]
            for j, k in enumerate(["accuracy", "balanced_accuracy", "macro_f1"])}, len(names)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if campaign.available_ram() < MIN_START:
        raise MemoryError(f"Preflight stopped at {campaign.available_ram()/2**20:.0f} MiB; need 700 MiB")
    if hashlib.sha256(OOF_PATH.read_bytes()).hexdigest() != OOF_SHA:
        raise ValueError("Saved champion OOF hash differs from the audited manifest")
    oof = np.load(OOF_PATH, allow_pickle=True)
    ids = oof["record_id"].astype(np.int64)
    y = oof["y"].astype(np.int8)
    groups = oof["origin_program"].astype(str)
    folds = oof["fold"].astype(np.int8)
    p_base = oof["probability"].astype(np.float32)
    if len(ids) != 578117 or len(np.unique(ids)) != len(ids) or set(np.unique(folds)) != {1, 2, 3, 4, 5}:
        raise ValueError("OOF shape, IDs, or folds do not match the development protocol")
    if pd.DataFrame({"origin": groups, "fold": folds}).groupby("origin").fold.nunique().max() != 1:
        raise ValueError("Origin_program crosses saved folds")

    numeric = np.load(campaign.OUT / "numeric_features.npy", mmap_mode="r")
    categorical = np.load(campaign.OUT / "categorical_tokens.npy", mmap_mode="r")
    if numeric.shape != (int(ids.max()) + 1, len(campaign.BASE_NUMERIC)):
        raise ValueError("Numeric cache schema mismatch")
    if categorical.shape != (int(ids.max()) + 1, len(campaign.BASE_CATEGORICAL)):
        raise ValueError("Categorical cache schema mismatch")
    replace_token = campaign._category_codes(["Replace"])[0]
    is_replace = np.asarray(categorical[ids, 1] == replace_token)
    if not is_replace.any():
        raise ValueError("No Replace cases found through cached prediction-time operation field")

    # Cheap baseline slice audit is saved before fitting anything.
    pred_base = (p_base >= .5).astype(np.int8)
    replace_origin_rows = {}
    for name in sorted(set(groups[is_replace].tolist())):
        m = is_replace & (groups == name)
        replace_origin_rows[name] = {
            "rows": int(m.sum()), "class_0": int((y[m] == 0).sum()),
            "class_1": int((y[m] == 1).sum()),
            "errors": int((pred_base[m] != y[m]).sum()),
            "macro_f1": float(f1_score(y[m], pred_base[m], average="macro", zero_division=0)),
        }
    baseline_slices = {
        "all": metrics(y, pred_base),
        "replace": metrics(y[is_replace], pred_base[is_replace]),
        "non_replace": metrics(y[~is_replace], pred_base[~is_replace]),
    }
    operation_by_fold = {}
    for fold in range(1, 6):
        va = folds == fold
        rr = va & is_replace
        operation_by_fold[str(fold)] = {
            "replace_rows": int(rr.sum()), "replace_origins": int(len(set(groups[rr].tolist()))),
            "class_0": int((y[rr] == 0).sum()), "class_1": int((y[rr] == 1).sum()),
            "baseline_replace_metrics": metrics(y[rr], pred_base[rr]),
        }
    analysis = {
        "schema": "qmutbench-replace-specialist-v1", "status": "analysis_complete_fit_pending",
        "oof_sha256": OOF_SHA, "record_ids_unique": True, "origin_fold_isolation_verified": True,
        "replace_operation_token": int(replace_token), "replace_rows": int(is_replace.sum()),
        "replace_class_counts": {str(c): int((y[is_replace] == c).sum()) for c in [0, 1]},
        "replace_origin_count": len(replace_origin_rows),
        "replace_per_origin_distribution": {
            "rows_min": int(min(v["rows"] for v in replace_origin_rows.values())),
            "rows_median": float(np.median([v["rows"] for v in replace_origin_rows.values()])),
            "rows_max": int(max(v["rows"] for v in replace_origin_rows.values())),
            "origins_with_both_classes": int(sum(v["class_0"] > 0 and v["class_1"] > 0 for v in replace_origin_rows.values())),
        },
        "baseline_slices": baseline_slices, "folds": operation_by_fold,
        "routing_availability": "operation is a required raw model input in best_model_metadata.json; this pilot routes only from operation==Replace, never from labels",
        "feature_sources": {"numeric_cache": str(campaign.OUT / "numeric_features.npy"),
                            "categorical_cache": str(campaign.OUT / "categorical_tokens.npy")},
    }
    (OUT / "baseline_slice_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([{"origin_program": k, **v} for k, v in replace_origin_rows.items()]).to_csv(
        OUT / "replace_per_origin_baseline.csv", index=False)
    pd.DataFrame({"record_id": ids, "origin_program": groups, "fold": folds,
                  "y_true": y, "is_replace": is_replace}).to_csv(
        OUT / "fold_manifest.csv.gz", index=False, compression="gzip")
    print("Baseline Replace analysis:", json.dumps({k: analysis[k] for k in
          ["replace_rows", "replace_class_counts", "replace_origin_count", "replace_per_origin_distribution", "baseline_slices", "folds"]}, indent=2), flush=True)

    if campaign.available_ram() < MIN_START:
        analysis["status"] = "analysis_complete_fit_skipped_resource_guard"
        analysis["fit_skip_reason"] = f"Available RAM {campaign.available_ram()/2**20:.0f} MiB below 700 MiB start guard after audit"
        (OUT / "baseline_slice_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
        print(analysis["fit_skip_reason"], flush=True)
        return

    all_parts = []
    fold_results = {}
    config = {"family": "DecisionTreeClassifier", "max_depth": 8,
              "min_samples_leaf": 50, "class_weight": "balanced", "threshold": 0.5,
              "seed_rule": "20261030 + saved fold number"}
    for fold in range(1, 6):
        if campaign.available_ram() < MIN_START:
            analysis["status"] = "partial_fit_stopped_resource_guard"
            analysis["fit_skip_reason"] = f"Before fold {fold}: {campaign.available_ram()/2**20:.0f} MiB available"
            (OUT / "baseline_slice_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
            break
        train = (folds != fold) & is_replace
        valid = (folds == fold) & is_replace
        tr_idx, va_idx = np.flatnonzero(train), np.flatnonzero(valid)
        if set(groups[train]) & set(groups[valid]):
            raise ValueError(f"Fold {fold}: Replace specialist origin leakage")
        train_ids, valid_ids = ids[tr_idx], ids[va_idx]
        xtr_num = np.asarray(numeric[train_ids, :14], dtype=np.float32)
        xva_num = np.asarray(numeric[valid_ids, :14], dtype=np.float32)
        xtr_cat = np.asarray(categorical[train_ids], dtype=np.uint32)
        xva_cat = np.asarray(categorical[valid_ids], dtype=np.uint32)
        tr_cat, va_cat, vocab_sizes = campaign._fold_onehot(xtr_cat, xva_cat, sparse, np)
        del xtr_cat, xva_cat
        xtr = sparse.hstack([sparse.csr_matrix(xtr_num), tr_cat], format="csr", dtype=np.float32)
        xva = sparse.hstack([sparse.csr_matrix(xva_num), va_cat], format="csr", dtype=np.float32)
        del xtr_num, xva_num, tr_cat, va_cat
        gc.collect()
        before_fit = campaign.available_ram()
        if before_fit < MIN_START:
            analysis["status"] = "partial_fit_stopped_resource_guard"
            analysis["fit_skip_reason"] = f"Fold {fold} matrices ready with {before_fit/2**20:.0f} MiB; below 700 MiB fit guard"
            (OUT / "baseline_slice_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
            break
        model = DecisionTreeClassifier(max_depth=8, min_samples_leaf=50,
                                       class_weight="balanced", random_state=SEED + fold)
        started = time.perf_counter()
        model.fit(xtr, y[tr_idx])
        cls = int(np.flatnonzero(model.classes_ == 1)[0])
        p_specialist = model.predict_proba(xva)[:, cls].astype(np.float32)
        runtime = time.perf_counter() - started
        del model, xtr, xva
        gc.collect()
        after_fit = campaign.available_ram()
        if after_fit < STOP_FLOOR:
            raise MemoryError(f"Available RAM reached {after_fit/2**20:.0f} MiB after fold {fold}; below stop floor")
        frame = pd.DataFrame({"record_id": valid_ids, "origin_program": groups[va_idx],
                              "fold": np.full(len(va_idx), fold, dtype=np.int8),
                              "y_true": y[va_idx], "p_champion": p_base[va_idx],
                              "p_specialist": p_specialist,
                              "y_pred_champion": pred_base[va_idx],
                              "y_pred_specialist": (p_specialist >= .5).astype(np.int8)})
        frame.to_csv(OUT / f"fold{fold}_replace_predictions.csv.gz", index=False, compression="gzip")
        fold_results[str(fold)] = {
            "replace_rows": int(len(va_idx)), "replace_origins": int(len(set(groups[va_idx].tolist())),),
            "train_replace_rows": int(len(tr_idx)), "train_replace_origins": int(len(set(groups[tr_idx].tolist()))),
            "baseline_replace": metrics(y[va_idx], pred_base[va_idx]),
            "specialist_replace": metrics(y[va_idx], (p_specialist >= .5).astype(np.int8)),
            "fold_local_vocabulary_sizes": vocab_sizes,
            "fit_seconds": runtime, "available_ram_before_fit_mib": round(before_fit / 2**20, 1),
            "available_ram_after_fit_mib": round(after_fit / 2**20, 1),
        }
        all_parts.append(frame)
        print(f"fold {fold} Replace: baseline={fold_results[str(fold)]['baseline_replace']['macro_f1']:.4f}, "
              f"specialist={fold_results[str(fold)]['specialist_replace']['macro_f1']:.4f}, "
              f"fit={runtime:.1f}s, RAM={after_fit/2**20:.0f} MiB", flush=True)
        del frame, p_specialist
        gc.collect()

    if len(all_parts) != 5:
        analysis["completed_folds"] = fold_results
        analysis.setdefault("status", "partial_fit_stopped_resource_guard")
        (OUT / "baseline_slice_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
        return

    specialist = pd.concat(all_parts, ignore_index=True)
    if specialist.record_id.duplicated().any() or len(specialist) != int(is_replace.sum()):
        raise ValueError("Replace OOF predictions do not cover exact slice IDs")
    specialist.to_csv(OUT / "replace_oof_predictions.csv.gz", index=False, compression="gzip")
    hybrid = pred_base.copy()
    loc = {int(r): i for i, r in enumerate(ids)}
    row_idx = np.fromiter((loc[int(r)] for r in specialist.record_id), dtype=np.int64, count=len(specialist))
    if (not np.array_equal(y[row_idx], specialist.y_true.to_numpy(np.int8)) or
            not np.array_equal(groups[row_idx], specialist.origin_program.to_numpy(str)) or
            not np.array_equal(folds[row_idx], specialist.fold.to_numpy(np.int8))):
        raise ValueError("Specialist predictions do not align to OOF IDs, origins, labels, and folds")
    pred_specialist_oof = np.full(len(ids), -1, dtype=np.int8)
    pred_specialist_oof[row_idx] = (specialist.p_specialist.to_numpy(np.float32) >= .5).astype(np.int8)
    hybrid[row_idx] = specialist.p_specialist.to_numpy(np.float32)
    pred_hybrid = (hybrid >= .5).astype(np.int8)
    overall = {"champion": metrics(y, pred_base), "hybrid": metrics(y, pred_hybrid)}
    specialist_group_mask = is_replace
    ci_overall, ci_origins = grouped_bootstrap(y, pred_base, pred_hybrid, groups,
                                              np.ones(len(y), dtype=bool))
    ci_replace, replace_origins = grouped_bootstrap(y, pred_base, pred_hybrid, groups,
                                                    specialist_group_mask)
    per_fold = {}
    for fold in range(1, 6):
        ix = folds == fold
        per_fold[str(fold)] = {"champion": metrics(y[ix], pred_base[ix]),
                               "hybrid": metrics(y[ix], pred_hybrid[ix]),
                               "overall_macro_f1_delta": float(f1_score(y[ix], pred_hybrid[ix], average="macro") -
                                                                f1_score(y[ix], pred_base[ix], average="macro")),
                               "replace": fold_results[str(fold)]}
    summary = {
        "schema": "qmutbench-replace-specialist-result-v1", "status": "complete",
        "routing": "operation==Replace from documented model input; champion used on all non-Replace records",
        "configuration": config, "fold_count": 5, "replace_rows": int(is_replace.sum()),
        "replace_origins": len(set(groups[is_replace].tolist())),
        "overall": overall, "overall_paired_origin_bootstrap_95ci_delta": ci_overall,
        "replace_slice": {
            "champion": metrics(y[is_replace], pred_base[is_replace]),
            "specialist": metrics(y[is_replace], pred_specialist_oof[is_replace]),
        },
        "overall_bootstrap_origins": ci_origins,
        "replace_slice_paired_origin_bootstrap_95ci_delta": ci_replace,
        "replace_bootstrap_origins": replace_origins,
        "per_fold": per_fold,
        "fold_delta_mean": float(np.mean([v["overall_macro_f1_delta"] for v in per_fold.values()])),
        "fold_delta_sd": float(np.std([v["overall_macro_f1_delta"] for v in per_fold.values()], ddof=1)),
        "runtime_seconds": float(sum(v["fit_seconds"] for v in fold_results.values())),
        "all_test_rows_excluded": True,
    }
    (OUT / "result_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    analysis["status"] = "complete"
    analysis["specialist_configuration"] = config
    analysis["fold_results"] = fold_results
    (OUT / "baseline_slice_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print("FINAL", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
