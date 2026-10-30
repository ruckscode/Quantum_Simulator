"""Confirm the screened structure add-on without reusing screen-fold labels.

Every confirmation model excludes fold 1 from both training and validation.
The 23-feature champion and 31-feature add-on candidate use identical records,
fold-local category vocabularies, Extra Trees settings, and random seeds.
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import qmutbench_improvement_campaign as campaign


def metrics(actual, pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, pred, labels=[0, 1], zero_division=0)
    return {"accuracy": float(accuracy_score(actual, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(actual, pred)),
            "macro_f1": float(f1_score(actual, pred, average="macro", zero_division=0)),
            "confusion_matrix_labels_0_1": confusion_matrix(actual, pred, labels=[0, 1]).tolist(),
            "per_class": {str(label): {"precision": float(precision[i]), "recall": float(recall[i]),
                                       "f1": float(f1[i]), "support": int(support[i])}
                          for i, label in enumerate([0, 1])}}


def run():
    available = campaign.available_ram()
    if available < 700 * 1024 * 1024:
        raise MemoryError(f"Preflight stopped; only {available/2**20:.0f} MiB available")
    manifest_path = campaign.OUT / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"screen_complete", "confirmation_in_progress", "confirmation_paused_resource_guard"}:
        raise ValueError("Fold-1 screen did not authorize confirmation or no resume point exists")
    screen = manifest["experiments"][0]
    if screen.get("screen_decision") != "advance_to_full_five_fold_confirmation":
        raise ValueError("Screen result did not meet the materiality bar")

    oof = np.load(campaign.OOF_PATH, allow_pickle=True)
    ids, y = oof["record_id"].astype(np.int64), oof["y"].astype(np.int8)
    groups, folds = oof["origin_program"].astype(str), oof["fold"].astype(np.int8)
    if not np.array_equal(ids, np.unique(ids)):
        raise ValueError("Expected unique sorted OOF record IDs")
    numeric = np.load(campaign.OUT / "numeric_features.npy", mmap_mode="r")
    categorical = np.load(campaign.OUT / "categorical_tokens.npy", mmap_mode="r")
    if numeric.shape != (int(ids.max()) + 1, len(campaign.BASE_NUMERIC)):
        raise ValueError("Cached numeric feature matrix shape mismatch")
    if categorical.shape != (int(ids.max()) + 1, len(campaign.BASE_CATEGORICAL)):
        raise ValueError("Cached categorical token matrix shape mismatch")

    confirmation = manifest.setdefault("confirmation_folds", {})
    for fold in range(2, 6):
        prediction_path = campaign.OUT / f"fold{fold}_confirmation_predictions.csv.gz"
        if str(fold) in confirmation:
            entry = confirmation[str(fold)]
            if not prediction_path.exists() or hashlib.sha256(prediction_path.read_bytes()).hexdigest() != entry["prediction_sha256"]:
                raise ValueError(f"Fold {fold} checkpoint checksum mismatch")
            continue
        # Fold 1 selected whether to advance. Exclude all of it from this
        # confirmation fold's training and validation records.
        train_rows = np.flatnonzero((folds != 1) & (folds != fold))
        valid_rows = np.flatnonzero(folds == fold)
        if 1 in set(folds[train_rows]) and len(set(groups[train_rows]) & set(groups[np.flatnonzero(folds == 1)])):
            raise ValueError("Screen-fold origins leaked into confirmation training")
        if set(groups[train_rows]) & set(groups[valid_rows]):
            raise ValueError(f"Fold {fold} Origin_program overlap")
        train_ids, valid_ids = ids[train_rows], ids[valid_rows]
        x_train_num = np.asarray(numeric[train_ids], dtype=np.float32)
        x_valid_num = np.asarray(numeric[valid_ids], dtype=np.float32)
        if not np.isfinite(x_train_num).all() or not np.isfinite(x_valid_num).all():
            raise ValueError(f"Non-finite numeric values in fold {fold}")
        x_train_cat = np.asarray(categorical[train_ids], dtype=np.uint32)
        x_valid_cat = np.asarray(categorical[valid_ids], dtype=np.uint32)
        tr_cat, va_cat, vocab_sizes = campaign._fold_onehot(x_train_cat, x_valid_cat, sparse, np)
        del x_train_cat, x_valid_cat
        x_train_base = sparse.hstack([sparse.csr_matrix(x_train_num[:, :14]), tr_cat],
                                     format="csr", dtype=np.float32)
        x_valid_base = sparse.hstack([sparse.csr_matrix(x_valid_num[:, :14]), va_cat],
                                     format="csr", dtype=np.float32)
        x_train_candidate = sparse.hstack([sparse.csr_matrix(x_train_num), tr_cat],
                                          format="csr", dtype=np.float32)
        x_valid_candidate = sparse.hstack([sparse.csr_matrix(x_valid_num), va_cat],
                                          format="csr", dtype=np.float32)
        del x_train_num, x_valid_num, tr_cat, va_cat
        gc.collect()
        available = campaign.available_ram()
        print(f"fold {fold} matrices ready: train={x_train_base.shape}, valid={x_valid_base.shape}, "
              f"available={available/2**20:.0f} MiB", flush=True)
        if available < 700 * 1024 * 1024:
            manifest["status"] = "confirmation_paused_resource_guard"
            manifest["available_bytes_before_fold"] = available
            campaign.atomic_json(manifest_path, manifest)
            raise MemoryError("Insufficient headroom before matched fold fits; checkpoints preserved")

        start_wall = time.time()
        with threadpool_limits(limits=1):
            base_model = ExtraTreesClassifier(n_estimators=100, max_depth=22, min_samples_leaf=15,
                max_features="sqrt", class_weight="balanced_subsample", n_jobs=1,
                random_state=20261025 + fold)
            base_model.fit(x_train_base, y[train_rows])
            p_base = base_model.predict_proba(x_valid_base)[:, list(base_model.classes_).index(1)]
        del base_model, x_train_base, x_valid_base
        gc.collect()
        available = campaign.available_ram()
        if available < 600 * 1024 * 1024:
            manifest["status"] = "confirmation_paused_resource_guard"
            manifest["available_bytes_between_fits"] = available
            campaign.atomic_json(manifest_path, manifest)
            raise MemoryError("Insufficient headroom between matched model fits; fold not checkpointed")
        with threadpool_limits(limits=1):
            candidate_model = ExtraTreesClassifier(n_estimators=100, max_depth=22, min_samples_leaf=15,
                max_features="sqrt", class_weight="balanced_subsample", n_jobs=1,
                random_state=20261025 + fold)
            candidate_model.fit(x_train_candidate, y[train_rows])
            p_candidate = candidate_model.predict_proba(x_valid_candidate)[:, list(candidate_model.classes_).index(1)]
        del candidate_model, x_train_candidate, x_valid_candidate
        gc.collect()
        y_valid = y[valid_rows]
        pred_base, pred_candidate = (p_base >= .5).astype(np.int8), (p_candidate >= .5).astype(np.int8)
        output = pd.DataFrame({"record_id": valid_ids, "origin_program": groups[valid_rows],
            "fold": np.full(len(valid_ids), fold, dtype=np.int8), "y_true": y_valid,
            "p_champion_matched": p_base, "y_pred_champion_matched": pred_base,
            "p_candidate": p_candidate, "y_pred_candidate": pred_candidate})
        temp = prediction_path.with_suffix(prediction_path.suffix + ".tmp")
        output.to_csv(temp, index=False, compression="gzip")
        temp.replace(prediction_path)
        entry = {"rows": int(len(valid_rows)), "origins": int(np.unique(groups[valid_rows]).size),
            "train_rows": int(len(train_rows)), "train_origins": int(np.unique(groups[train_rows]).size),
            "candidate_metrics": metrics(y_valid, pred_candidate),
            "matched_champion_metrics": metrics(y_valid, pred_base),
            "macro_f1_delta": float(f1_score(y_valid, pred_candidate, average="macro")-
                                    f1_score(y_valid, pred_base, average="macro")),
            "category_vocabulary_sizes": vocab_sizes,
            "wall_seconds_two_fits": time.time()-start_wall,
            "available_ram_after_fold": campaign.available_ram(),
            "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest()}
        confirmation[str(fold)] = entry
        manifest["status"] = "confirmation_in_progress"
        campaign.atomic_json(manifest_path, manifest)
        print(f"fold {fold} done: delta={entry['macro_f1_delta']:+.4f}, "
              f"wall={entry['wall_seconds_two_fits']:.1f}s, "
              f"available={entry['available_ram_after_fold']/2**20:.0f} MiB", flush=True)
        if entry["available_ram_after_fold"] < campaign.RAM_STOP_BYTES:
            manifest["status"] = "confirmation_paused_resource_guard"
            campaign.atomic_json(manifest_path, manifest)
            raise MemoryError("Resource floor reached after checkpoint; resume safely from next fold")
        del output, p_base, p_candidate, pred_base, pred_candidate
        gc.collect()

    if set(confirmation) < {"2", "3", "4", "5"}:
        raise ValueError("Confirmation folds are incomplete")
    combined = pd.concat([pd.read_csv(campaign.OUT / f"fold{f}_confirmation_predictions.csv.gz")
                          for f in range(2, 6)], ignore_index=True)
    if combined.record_id.duplicated().any() or len(combined) != int(np.isin(folds, [2,3,4,5]).sum()):
        raise ValueError("Confirmation record coverage mismatch")
    combined.to_csv(campaign.OUT / "post_screen_confirmation_oof.csv.gz", index=False, compression="gzip")
    y_all = combined.y_true.to_numpy(np.int8)
    p_base_all = combined.y_pred_champion_matched.to_numpy(np.int8)
    p_cand_all = combined.y_pred_candidate.to_numpy(np.int8)

    # Origin-cluster paired bootstrap over the four folds untouched by screening.
    matrices = []
    for _, part in combined.groupby("origin_program", sort=False):
        yy = part.y_true.to_numpy(np.int8)
        matrices.append((confusion_matrix(yy, part.y_pred_champion_matched.to_numpy(np.int8), labels=[0,1]),
                         confusion_matrix(yy, part.y_pred_candidate.to_numpy(np.int8), labels=[0,1])))
    bm, cm = np.stack([x[0] for x in matrices]), np.stack([x[1] for x in matrices])
    rng = np.random.default_rng(20261027); boot = []
    for _ in range(2000):
        ix = rng.integers(0, len(matrices), len(matrices))
        b, c = bm[ix].sum(axis=0), cm[ix].sum(axis=0)
        def vals(matrix):
            tn, fp, fn, tp = matrix.ravel()
            f0 = 2*tn/max(1,2*tn+fp+fn); f1v = 2*tp/max(1,2*tp+fp+fn)
            return np.array([(tn+tp)/max(1,matrix.sum()),
                .5*(tn/max(1,tn+fp)+tp/max(1,fn+tp)), .5*(f0+f1v)])
        boot.append(vals(c)-vals(b))
    boot = np.asarray(boot)
    intervals = {name: [float(np.percentile(boot[:,i],2.5)),float(np.percentile(boot[:,i],97.5))]
                 for i,name in enumerate(["accuracy","balanced_accuracy","macro_f1"])}
    summary = {"schema":"qmutbench-structural-addition-confirmation-v1","status":"complete",
        "screen_fold_1_excluded_from_all_confirmation_training_and_validation":True,
        "confirmation_folds":[2,3,4,5],"rows":len(combined),"origins":len(matrices),
        "candidate_metrics":metrics(y_all,p_cand_all),"matched_champion_metrics":metrics(y_all,p_base_all),
        "macro_f1_delta":float(f1_score(y_all,p_cand_all,average="macro")-
                                 f1_score(y_all,p_base_all,average="macro")),
        "paired_origin_bootstrap_95_percentile_delta_intervals":intervals,
        "bootstrap_repetitions":2000,"bootstrap_seed":20261027,
        "fold_results":{f:confirmation[f] for f in sorted(confirmation)},
        "candidate_oof_predictions_path":"post_screen_confirmation_oof.csv.gz",
        "selection_note":"The fold-1 screen result was excluded from confirmatory estimate; all fits below exclude fold-1 origins from training."}
    (campaign.OUT/"confirmation_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    manifest["confirmation_summary"]=summary; manifest["status"]="confirmation_complete"
    campaign.atomic_json(manifest_path,manifest)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    run()
