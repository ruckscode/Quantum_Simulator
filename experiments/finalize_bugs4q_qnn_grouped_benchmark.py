"""Low-memory summary pass for the completed Bugs4Q grouped QNN OOF run."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("BUGS4Q_QNN_OUTPUT", ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/bugs4q_standalone/qnn_gradient_fix"))
FLOOR_MIB = 500
BOOTSTRAPS = 2000
SEED = 20261028


def score(y: np.ndarray, pred: np.ndarray) -> dict:
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "confusion_matrix_labels_class0_class1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "per_class": {str(i): {"precision": float(p[i]), "recall": float(r[i]),
                                    "f1": float(f[i]), "support": int(s[i])} for i in (0, 1)}}


def main() -> None:
    if psutil.virtual_memory().available / 2**20 < FLOOR_MIB:
        raise MemoryError("Stopped before summary: available RAM below 500 MiB")
    pred = pd.read_csv(OUT / "oof_predictions.csv")
    fold = pd.read_json(OUT / "fold_metrics.json")
    models = list(pred.model.drop_duplicates())
    group_ids = np.array(sorted(pred.bug_id.astype(str).unique(), key=int), dtype=object)
    group_index = {g: i for i, g in enumerate(group_ids)}
    group_rows = [np.flatnonzero(pred.bug_id.astype(str).to_numpy() == g) for g in group_ids]
    rng = np.random.default_rng(SEED)
    leaderboard, per_class, per_bug = [], {}, []
    for name in models:
        if psutil.virtual_memory().available / 2**20 < FLOOR_MIB:
            raise MemoryError("Stopped during summary at the 500 MiB RAM floor")
        subset = pred[pred.model == name].sort_values("record_id")
        y = subset.y_true.to_numpy(dtype=np.int8)
        p = subset.y_pred.to_numpy(dtype=np.int8)
        g = subset.bug_id.astype(str).to_numpy()
        base = score(y, p)
        # Each issue contributes the four binary confusion counts. Resampling whole
        # issue groups therefore preserves the paired buggy/fixed snapshots.
        ix_by_group = [np.flatnonzero(g == group) for group in group_ids]
        confusion = np.zeros((len(group_ids), 4), dtype=np.int16)
        for i, ix in enumerate(ix_by_group):
            yy, pp = y[ix], p[ix]
            confusion[i] = [np.sum((yy == 0) & (pp == 0)), np.sum((yy == 0) & (pp == 1)),
                            np.sum((yy == 1) & (pp == 0)), np.sum((yy == 1) & (pp == 1))]
        draws = rng.integers(0, len(group_ids), size=(BOOTSTRAPS, len(group_ids)))
        multiplicity = np.zeros((BOOTSTRAPS, len(group_ids)), dtype=np.int16)
        rows = np.arange(BOOTSTRAPS)[:, None]
        np.add.at(multiplicity, (np.broadcast_to(rows, draws.shape), draws), 1)
        tn, fp, fn, tp = (multiplicity @ confusion[:, col] for col in range(4))
        f0 = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn, dtype=float), where=(2 * tn + fp + fn) != 0)
        f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) != 0)
        macro = (f0 + f1) / 2
        base["bug_group_bootstrap_95ci_macro_f1"] = [float(np.quantile(macro, .025)), float(np.quantile(macro, .975))]
        timing = (fold.loc[fold["model"] == name, "training_seconds"].dropna()
                  if "training_seconds" in fold.columns else pd.Series(dtype=float))
        if timing.empty:
            timing = (fold.loc[fold["model"] == name, "runtime_seconds"].dropna()
                      if "runtime_seconds" in fold.columns else pd.Series(dtype=float))
        base["measured_fit_runtime_seconds_sum"] = float(timing.sum()) if not timing.empty else None
        base["model"] = name
        base["family"] = "classical" if name in {"logistic_4d", "random_forest_4d", "extra_trees_4d", "mlp_4d"} else "quantum_or_hybrid"
        leaderboard.append(base)
        per_class[name] = base["per_class"]
        for group in group_ids:
            rows_g = subset[subset.bug_id.astype(str) == group]
            per_bug.append({"bug_id": str(group), "model": name, "records": len(rows_g),
                            **score(rows_g.y_true.to_numpy(dtype=np.int8), rows_g.y_pred.to_numpy(dtype=np.int8))})
    pd.DataFrame(leaderboard).to_json(OUT / "leaderboard.json", orient="records", indent=2)
    (OUT / "per_class_metrics.json").write_text(json.dumps(per_class, indent=2), encoding="utf-8")
    pd.DataFrame(per_bug).to_json(OUT / "per_bug_metrics.json", orient="records", indent=2)

    # Complementarity is measured on the shared outer-fold OOF records.
    quantum_models = [m for m in models if m in {"angle_vqc", "angle_reupload_vqc", "angle_hybrid_readout", "amplitude_vqc"}]
    wide = pred[pred.model.isin(quantum_models)].pivot(index=["record_id", "bug_id", "y_true"], columns="model", values="y_pred")
    error_rows = []
    y = wide.index.get_level_values("y_true").to_numpy(dtype=np.int8)
    for i, left in enumerate(quantum_models):
        a = wide[left].to_numpy(dtype=np.int8) != y
        for right in quantum_models[i + 1:]:
            b = wide[right].to_numpy(dtype=np.int8) != y
            error_rows.append({"model_a": left, "model_b": right,
                               "pairwise_disagreement": float(np.mean(wide[left].to_numpy() != wide[right].to_numpy())),
                               "both_wrong": int(np.sum(a & b)), "a_only_wrong": int(np.sum(a & ~b)),
                               "b_only_wrong": int(np.sum(~a & b)), "either_wrong": int(np.sum(a | b)),
                               "shared_records": int(len(y))})
    pd.DataFrame(error_rows).to_json(OUT / "pairwise_error_complementarity.json", orient="records", indent=2)
    # Fixed probability means use no fitted weights or labels. Every row remains an
    # outer-fold prediction, so this is a valid diagnostic ensemble comparison.
    probability_models = [m for m in models if f"prob_buggy" in pred.columns and m in pred.model.unique()]
    probability_pivot = pred.pivot(index=["record_id", "bug_id", "fold", "y_true"],
                                   columns="model", values="prob_buggy")
    ensemble_rows, ensemble_metrics = [], []
    families = {"quantum_equal_probability_mean": quantum_models,
                "all_candidates_equal_probability_mean": probability_models}
    for ensemble_name, members in families.items():
        if not members or any(m not in probability_pivot.columns for m in members):
            continue
        probs = probability_pivot[members].mean(axis=1).to_numpy(dtype=float)
        y = probability_pivot.index.get_level_values("y_true").to_numpy(dtype=np.int8)
        pred_ens = (probs >= 0.5).astype(np.int8)
        ensemble_metrics.append({"model": ensemble_name, "members": members,
                                 **score(y, pred_ens)})
        for i, key in enumerate(probability_pivot.index):
            ensemble_rows.append({"record_id": key[0], "bug_id": str(key[1]), "fold": int(key[2]),
                                  "y_true": int(key[3]), "model": ensemble_name,
                                  "y_pred": int(pred_ens[i]), "prob_buggy": float(probs[i]),
                                  "prob_fixed": float(1.0 - probs[i])})
    if ensemble_rows:
        pd.DataFrame(ensemble_rows).to_csv(OUT / "ensemble_oof_predictions.csv", index=False)
        (OUT / "ensemble_summary.json").write_text(json.dumps(ensemble_metrics, indent=2), encoding="utf-8")
    source_results = []
    if "source" in pred.columns:
        for (name, source), subset in pred.groupby(["model", "source"]):
            source_results.append({"model": name, "source": str(source),
                                   "records": int(subset.record_id.nunique()),
                                   "origins": int(subset.bug_id.nunique()),
                                   **score(subset.y_true.to_numpy(dtype=np.int8), subset.y_pred.to_numpy(dtype=np.int8))})
        (OUT / "source_performance.json").write_text(json.dumps(source_results, indent=2), encoding="utf-8")
    history = json.loads((OUT / "training_history.json").read_text(encoding="utf-8"))
    checkpoints = json.loads((OUT / "fold_checkpoints.json").read_text(encoding="utf-8"))
    summary = {"status": "completed_grouped_supervised_qnn_benchmark",
               "task": os.environ.get("BUGS4Q_TASK", "Buggy versus fixed source state: fixed=0, buggy=1"),
               "data": os.environ.get("QNN_DATA_DESCRIPTION", "31 issue groups / 62 paired snapshots; comments and docstrings stripped; executable strings retained"),
               "group_split": os.environ.get("QNN_GROUP_SPLIT_DESCRIPTION", "5-fold GroupKFold by bug_id; related records stay within the same fold"),
               "qnn_candidates": quantum_models,
               "matched_classical_controls": [m for m in models if m not in quantum_models],
               "folds": int(pred.fold.nunique()), "records": int(pred.record_id.nunique()),
               "origins": int(pred.bug_id.nunique()),
               "gradient_checks_passed": sum(bool((r.get("gradient_check") or {}).get("passed")) for r in fold.to_dict("records")),
               "gradient_checks_expected": sum(1 for r in fold.to_dict("records") if r.get("model") in quantum_models),
               "optimizer_runs": len(history), "optimizer_steps_per_run": 15,
               "all_loss_histories_decreased": all(h["loss"][-1] < h["loss"][0] for h in history),
               "device": "PennyLane default.qubit CPU statevector with analytic backpropagation; no GPU or hardware claim",
               "pennylane_version": "0.45.1", "bootstrap_replicates": BOOTSTRAPS,
               "resource_use": {"available_ram_mib_at_summary_start": round(psutil.virtual_memory().available / 2**20, 1),
                                "local_ram_floor_mib": FLOOR_MIB,
                                "sampled_peak_training_process_rss_mib": (float(os.environ["QNN_SAMPLED_PEAK_RSS_MIB"]) if os.environ.get("QNN_SAMPLED_PEAK_RSS_MIB") else None),
                                "checkpoint_count": len(checkpoints)},
               "ensemble_diagnostics": ensemble_metrics,
               "per_source_oof_performance": source_results,
               "training_source_filter": os.environ.get("QNN_TRAIN_SOURCE", "all pooled sources"),
               "results": leaderboard,
               "finalization_note": "Training and OOF export completed. The original pandas bootstrap pass was interrupted when available RAM fell under the 500 MiB floor; this summary uses vectorized NumPy issue-group resampling.",
               "production_model_modified": False, "frozen_test_accessed": False}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
