"""Bounded, development-only ensemble audit for saved three-class OOF predictions.

Base probabilities were produced by the prior grouped campaign.  The outer
fold is kept out of every fitted meta-model below.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/experiments/qmutbench_three_class_qnn_campaign_20261027/predictions.csv.gz"
OUT = ROOT / "reports/experiments/qmutbench_multisource_ensemble_20261028"
CLASSICAL = ["extra_trees_4f", "logistic_4f", "rbf_svc_4f"]
QUANTUM = ["three_class_vqc", "three_class_vqc_amplitude", "three_class_vqc_angle_linear"]


def metrics(y, pred):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "per_class": {str(i): {"precision": float(p[i]), "recall": float(r[i]),
                               "f1": float(f[i]), "support": int(s[i])} for i in range(3)},
    }


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    campaign = SOURCE.parent
    inputs = [SOURCE, campaign / "amplitude_encoding/predictions.csv.gz",
              campaign / "trainable_readout/predictions.csv.gz"]
    raw = pd.concat([pd.read_csv(p) for p in inputs], ignore_index=True)
    expected = set(CLASSICAL + QUANTUM)
    raw = raw.drop_duplicates(["record_id", "model"], keep="first")
    if set(raw.model) != expected or raw.duplicated(["record_id", "model"]).any():
        raise ValueError("Saved candidate set or record/model uniqueness differs from the declared protocol")
    wide = raw.pivot(index=["record_id", "origin_program", "fold", "y_true"], columns="model",
                     values=["y_pred", "prob_0", "prob_1", "prob_2"]).reset_index()
    wide.columns = ["_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else col for col in wide.columns]
    # Pivot metadata columns also receive a trailing underscore.
    wide = wide.rename(columns={c: c.rstrip("_") for c in wide.columns})
    y = wide.y_true.to_numpy(dtype=int)
    folds = wide.fold.to_numpy(dtype=int)
    probs = {m: wide[[f"prob_{k}_{m}" for k in range(3)]].to_numpy(float) for m in expected}
    predictions = {m: wide[f"y_pred_{m}"].to_numpy(dtype=int) for m in sorted(expected)}
    records = []
    for m, pred in predictions.items():
        records.append({"candidate": m, "strategy": "base_oof", **metrics(y, pred)})

    # Prespecified broad family average and all-candidate equal average.
    for label, names in [("equal_classical_probability_mean", CLASSICAL),
                         ("equal_quantum_probability_mean", QUANTUM),
                         ("equal_all_probability_mean", CLASSICAL + QUANTUM)]:
        p = np.mean([probs[n] for n in names], axis=0)
        records.append({"candidate": label, "strategy": "fixed_probability_mean",
                        "members": names, **metrics(y, p.argmax(axis=1))})

    # Outer-fold logistic stacking: each meta-model sees only other-fold base OOF predictions.
    stack_pred = np.full((len(y),), -1, dtype=int)
    stack_prob = np.full((len(y), 3), np.nan)
    stack_meta = np.concatenate([probs[n] for n in CLASSICAL + QUANTUM], axis=1)
    for fold in sorted(np.unique(folds)):
        tr, te = folds != fold, folds == fold
        if len(np.unique(y[tr])) != 3 or len(np.unique(y[te])) != 3:
            raise ValueError(f"Fold {fold} lacks a class")
        model = LogisticRegression(C=0.1, max_iter=1000, random_state=20261028)
        model.fit(stack_meta[tr], y[tr])
        stack_pred[te] = model.predict(stack_meta[te])
        stack_prob[te] = model.predict_proba(stack_meta[te])[:, model.classes_.argsort()]
    if (stack_pred < 0).any() or not np.isfinite(stack_prob).all():
        raise ValueError("Incomplete outer-fold stacking predictions")
    records.append({"candidate": "all_six_logistic_stacker", "strategy": "outer_fold_held_out_stacking",
                    "members": CLASSICAL + QUANTUM, "meta_C": 0.1, **metrics(y, stack_pred)})
    pd.DataFrame({"record_id": wide.record_id, "origin_program": wide.origin_program,
                  "fold": folds, "y_true": y, "stack_pred": stack_pred,
                  **{f"stack_prob_{i}": stack_prob[:, i] for i in range(3)}}).to_csv(OUT / "stacking_oof_predictions.csv", index=False)
    pd.DataFrame(records).to_json(OUT / "candidate_metrics.json", orient="records", indent=2)

    # Origin bootstrap is descriptive only because all 12 origins already informed this exploration.
    rng = np.random.default_rng(20261028)
    origins = wide.origin_program.to_numpy()
    unique = np.array(sorted(pd.unique(origins)), dtype=object)
    by_origin = {g: np.flatnonzero(origins == g) for g in unique}
    preds_for_ci = {**predictions,
                    "equal_all_probability_mean": np.mean([probs[n] for n in CLASSICAL + QUANTUM], axis=0).argmax(1),
                    "all_six_logistic_stacker": stack_pred}
    boot = {name: [] for name in preds_for_ci}
    for _ in range(2000):
        draw = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([by_origin[g] for g in draw])
        for name, pred in preds_for_ci.items():
            boot[name].append(float(f1_score(y[ix], pred[ix], average="macro", zero_division=0)))
    ci = {name: [float(np.quantile(values, .025)), float(np.quantile(values, .975))]
          for name, values in boot.items()}
    (OUT / "origin_bootstrap_intervals.json").write_text(json.dumps({
        "unit": "origin_program", "origins": len(unique), "repetitions": 2000,
        "seed": 20261028, "intervals_macro_f1_95pct": ci,
        "warning": "descriptive exploratory uncertainty; candidate/blend inspection used the same 12 development origins"
    }, indent=2), encoding="utf-8")
    pd.DataFrame([{**r, "macro_f1_bootstrap_95ci_low": ci[r["candidate"]][0],
                   "macro_f1_bootstrap_95ci_high": ci[r["candidate"]][1]}
                  for r in records if r["candidate"] in ci]).to_csv(OUT / "candidate_leaderboard.csv", index=False)

    # Pairwise complementarity and per-origin performance for every serious base candidate.
    pairs = []
    ordered_models = sorted(expected)
    for i, a in enumerate(ordered_models):
        for b in ordered_models[i + 1:]:
            ca, cb = predictions[a] == y, predictions[b] == y
            pairs.append({"model_a": a, "model_b": b, "records": len(y),
                          "agreement": float(np.mean(predictions[a] == predictions[b])),
                          "both_correct": int(np.sum(ca & cb)), "a_only_correct": int(np.sum(ca & ~cb)),
                          "b_only_correct": int(np.sum(~ca & cb)), "both_wrong": int(np.sum(~ca & ~cb))})
    pd.DataFrame(pairs).to_csv(OUT / "pairwise_error_complementarity.csv", index=False)
    origin_rows = []
    for origin, idx in wide.groupby("origin_program", sort=True).indices.items():
        idx = np.asarray(idx)
        for m, pred in predictions.items():
            origin_rows.append({"origin_program": origin, "candidate": m, "records": len(idx),
                                **metrics(y[idx], pred[idx])})
        origin_rows.append({"origin_program": origin, "candidate": "all_six_logistic_stacker",
                            "records": len(idx), **metrics(y[idx], stack_pred[idx])})
    pd.DataFrame(origin_rows).to_json(OUT / "per_origin_metrics.json", orient="records", indent=2)
    # Save all six base OOF probabilities alongside IDs for future nested analysis.
    base = wide[["record_id", "origin_program", "fold", "y_true"]].copy()
    for m in ordered_models:
        for k in range(3):
            base[f"{m}_prob_{k}"] = probs[m][:, k]
        base[f"{m}_pred"] = predictions[m]
    base.to_csv(OUT / "base_oof_predictions.csv", index=False)
    result = {
        "status": "completed_development_only",
        "task": "QMutBench three-class research task: Survivor=0, WOO=1, OPO=2",
        "protocol": "205 saved OOF records from 12 origins; identical IDs and labels for all six candidates; three grouped folds; outer fold held out for stacking; no frozen-test access",
        "source_artifacts": {str(p.relative_to(ROOT)): __import__("hashlib").sha256(p.read_bytes()).hexdigest() for p in inputs},
        "records": int(len(y)), "origins": int(wide.origin_program.nunique()), "folds": int(len(np.unique(folds))),
        "class_counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()},
        "candidates": records, "origin_bootstrap_intervals": ci,
        "conditional_router": "not fitted: only 12 origins/205 records; no predeclared inference-time condition and nested routing would be underpowered",
        "source_transfer": "not run: available real-bug examples do not have compatible QMutBench mutation-outcome labels/features; no label remapping",
        "multitask": "not run: no source-specific labeled rows and shared features were validated for a joint representation task",
        "runtime_seconds": time.perf_counter() - started,
    }
    (OUT / "experiment_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
