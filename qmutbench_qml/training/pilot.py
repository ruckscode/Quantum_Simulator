"""Small, reproducible 3-fold group-disjoint pilot; never reads frozen-test labels."""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import psutil
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from qmutbench_qml.data.loader import FEATURES, load_development_records
from qmutbench_qml.features.reduction import ReducedFeatureEncoder, feature_parity_table
from qmutbench_qml.models.qsvc import QuantumSupportVectorClassifier
from qmutbench_qml.models.quantum_reservoir import QuantumReservoirClassifier
from qmutbench_qml.models.vqc import DataReuploadingClassifier, VariationalQuantumClassifier

SEED = 20261026
ORIGIN_COUNT = 9
ROWS_PER_ORIGIN = 8
FOLDS = 3
N_FEATURES = 4
VQC_STEPS = 20
N_QUBITS = 4
DEPTH = 2
MODELS = ["vqc", "qsvc", "extra_trees_4f", "logistic_4f", "rbf_svc_4f"]


def metrics(y, pred, score):
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "per_class": {str(label): {"precision": float(precision[i]), "recall": float(recall[i]),
                                    "f1": float(f1[i]), "support": int(support[i])}
                      for i, label in enumerate([0, 1])},
        "confusion_matrix_labels_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
    }


def _small_group_sample(frame, origin_count=ORIGIN_COUNT, rows_per_origin=ROWS_PER_ORIGIN, seed=SEED):
    rng = np.random.default_rng(seed)
    origins = np.array(sorted(frame.origin_program.unique()), dtype=object)
    chosen = np.sort(rng.choice(origins, size=origin_count, replace=False))
    parts = []
    for origin in chosen:
        group = frame.loc[frame.origin_program == origin]
        take = min(rows_per_origin, len(group))
        # Selection is random within origin and does not stratify on target.
        selected = group.iloc[np.sort(rng.choice(len(group), size=take, replace=False))]
        parts.append(selected)
    sampled = pd.concat(parts, ignore_index=True)
    if sampled.origin_program.nunique() != origin_count:
        raise AssertionError("Pilot origin count changed unexpectedly")
    return sampled, [str(x) for x in chosen]


def _choose_development_sample(root, origin_count, rows_per_origin, seed=SEED):
    oof = pd.read_csv(root / "data/qmutbench_ml/master_optimization_v1/oof_predictions.csv",
                      usecols=["record_id", "origin_program"])
    origins = np.array(sorted(oof.origin_program.astype(str).unique()), dtype=object)
    if origin_count > len(origins):
        raise ValueError("Requested more origins than the verified development split contains")
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(origins, size=origin_count, replace=False))
    selected_ids = []
    for origin in chosen:
        rows = oof.loc[oof.origin_program.astype(str) == origin, "record_id"].to_numpy()
        take = min(rows_per_origin, len(rows))
        selected_ids.extend(rows[np.sort(rng.choice(len(rows), size=take, replace=False))].astype(int).tolist())
    return chosen.astype(str).tolist(), selected_ids


def _scores(estimator, x, *, proba=True):
    if proba:
        return estimator.predict_proba(x)[:, 1]
    return estimator.decision_function(x)


def _classify_score(name, score):
    """Use probability midpoint or SVC's decision boundary, as appropriate."""
    threshold = 0.0 if name in {"qsvc", "rbf_svc_4f"} else 0.5
    return (np.asarray(score) >= threshold).astype(int)


def _validate_prediction_alignment(pred_frame, expected_models):
    """Reject missing, duplicate, or label/group-mismatched paired predictions."""
    keys = ["record_id", "origin_program", "y_true", "fold"]
    if pred_frame.duplicated(["record_id", "model"]).any():
        raise AssertionError("A record was predicted more than once by a model")
    consistency = pred_frame.groupby("record_id")[keys[1:]].nunique(dropna=False)
    if (consistency != 1).any().any():
        raise AssertionError("Models disagree on validation labels, origin, or fold for a record")
    sets = {name: set(group.record_id) for name, group in pred_frame.groupby("model")}
    if set(sets) != set(expected_models) or len({frozenset(ids) for ids in sets.values()}) != 1:
        raise AssertionError("Models do not share identical validation records")
    wide = pred_frame.pivot(index=keys, columns="model", values=["y_pred", "score"])
    if wide.isna().any().any():
        raise AssertionError("At least one model is missing a paired prediction")
    return wide


def run(root: Path, output_dir: Path | None = None, *, origin_count=ORIGIN_COUNT,
        rows_per_origin=ROWS_PER_ORIGIN, include_data_reuploading=False,
        include_quantum_reservoir=False) -> dict:
    root = root.resolve()
    output_dir = (output_dir or root / "qmutbench_qml/results").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    memory_before = process.memory_info().rss
    selected_origins, selected_ids = _choose_development_sample(root, origin_count, rows_per_origin)
    sampled, data_audit = load_development_records(root, origins=selected_origins, record_ids=selected_ids)
    model_names = MODELS + (["data_reupload"] if include_data_reuploading else [])
    if include_quantum_reservoir:
        model_names.append("quantum_reservoir")
    groups = sampled.origin_program.to_numpy()
    cv = GroupKFold(n_splits=FOLDS)
    predictions = []
    fold_records = []
    split_rows = []
    feature_records = []
    run_started = time.perf_counter()

    for fold, (train_ix, val_ix) in enumerate(cv.split(sampled, sampled.y, groups), start=1):
        train = sampled.iloc[train_ix].reset_index(drop=True)
        valid = sampled.iloc[val_ix].reset_index(drop=True)
        train_groups, val_groups = set(train.origin_program), set(valid.origin_program)
        if train_groups & val_groups:
            raise AssertionError("Origin leakage across grouped fold")
        split_rows.extend({"fold": fold, "record_id": int(row.record_id),
                           "origin_program": str(row.origin_program), "role": "train"}
                          for row in train.itertuples())
        split_rows.extend({"fold": fold, "record_id": int(row.record_id),
                           "origin_program": str(row.origin_program), "role": "validation"}
                          for row in valid.itertuples())
        if set(train.y) != {0, 1} or set(valid.y) != {0, 1}:
            raise ValueError(f"Fold {fold} does not contain both target classes")
        encoder = ReducedFeatureEncoder(n_features=N_FEATURES, seed=SEED + fold).fit(train[FEATURES], train.y)
        x_train = encoder.transform(train[FEATURES])
        x_valid = encoder.transform(valid[FEATURES])
        feature_records.append({"fold": fold, "selected_features": encoder.selected_names})
        y_train = train.y.to_numpy(dtype=int)
        y_valid = valid.y.to_numpy(dtype=int)
        ids = valid.record_id.to_numpy(dtype=int)
        origins = valid.origin_program.astype(str).to_numpy()
        fold_model_info = {"fold": fold, "train_records": len(train), "validation_records": len(valid),
                           "train_origins": len(train_groups), "validation_origins": len(val_groups),
                           "train_origin_ids": sorted(map(str, train_groups)),
                           "validation_origin_ids": sorted(map(str, val_groups)), "group_overlap": 0}

        qmodels = [
            ("vqc", VariationalQuantumClassifier(n_qubits=N_QUBITS, depth=DEPTH, steps=VQC_STEPS,
                                                 seed=SEED + fold), True),
            ("qsvc", QuantumSupportVectorClassifier(n_qubits=N_QUBITS, seed=SEED + fold), False),
        ]
        if include_data_reuploading:
            qmodels.append(("data_reupload", DataReuploadingClassifier(n_qubits=N_QUBITS, depth=DEPTH,
                                                                         steps=VQC_STEPS, seed=SEED + fold), True))
        if include_quantum_reservoir:
            qmodels.append(("quantum_reservoir", QuantumReservoirClassifier(n_qubits=N_QUBITS,
                                                                              seed=SEED + fold), True))
        for name, model, probability in qmodels:
            if name == "qsvc":
                model.fit_with_features(x_train, y_train)
                score = np.asarray(model.decision_function(x_valid))
                pred = _classify_score(name, score)
            else:
                model.fit(x_train, y_train)
                score = model.predict_proba(x_valid)[:, 1]
                pred = _classify_score(name, score)
            predictions.extend({"record_id": int(rid), "origin_program": origin, "y_true": int(y),
                                "model": name, "y_pred": int(p), "score": float(s), "fold": fold}
                               for rid, origin, y, p, s in zip(ids, origins, y_valid, pred, score))
            fold_model_info.setdefault("models", {})[name] = {
                "validation_metrics": metrics(y_valid, pred, score),
                "train_seconds": float(model.training_seconds),
                "inference_seconds": float(model.inference_seconds),
                "circuit_executions_train": int(model.training_executions),
                "circuit_executions_inference": int(model.inference_executions),
                "shots": None,
                "n_qubits": N_QUBITS,
                "depth": DEPTH if name in {"vqc", "data_reupload", "quantum_reservoir"} else 2,
                "loss_decreased": bool(len(model.loss_history) > 1 and model.loss_history[-1] < model.loss_history[0]) if name in {"vqc", "data_reupload"} else None,
                "loss_start": model.loss_history[0] if name in {"vqc", "data_reupload"} else None,
                "loss_end": model.loss_history[-1] if name in {"vqc", "data_reupload"} else None,
            }

        classical = [
            ("extra_trees_4f", ExtraTreesClassifier(n_estimators=100, min_samples_leaf=2,
                class_weight="balanced", random_state=SEED + fold, n_jobs=1), True),
            ("logistic_4f", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED + fold), True),
            ("rbf_svc_4f", SVC(C=1.0, kernel="rbf", class_weight="balanced",
                                random_state=SEED + fold), False),
        ]
        for name, model, probability in classical:
            started = time.perf_counter()
            model.fit(x_train, y_train)
            train_seconds = time.perf_counter() - started
            started = time.perf_counter()
            score = _scores(model, x_valid, proba=probability)
            infer_seconds = time.perf_counter() - started
            pred = _classify_score(name, score)
            predictions.extend({"record_id": int(rid), "origin_program": origin, "y_true": int(y),
                                "model": name, "y_pred": int(p), "score": float(s), "fold": fold}
                               for rid, origin, y, p, s in zip(ids, origins, y_valid, pred, score))
            fold_model_info.setdefault("models", {})[name] = {
                "validation_metrics": metrics(y_valid, pred, score),
                "train_seconds": float(train_seconds), "inference_seconds": float(infer_seconds),
                "circuit_executions_train": 0, "circuit_executions_inference": 0,
                "shots": None, "n_qubits": None, "depth": None, "fit_completed": True,
            }
        fold_records.append(fold_model_info)

    elapsed = time.perf_counter() - run_started
    pred_frame = pd.DataFrame(predictions)
    wide = _validate_prediction_alignment(pred_frame, model_names)
    summary = {model: metrics(group.y_true.to_numpy(), group.y_pred.to_numpy(), group.score.to_numpy())
               for model, group in pred_frame.groupby("model", sort=False)}
    for model, result in summary.items():
        fold_metrics = [entry["models"][model]["validation_metrics"] for entry in fold_records]
        result["fold_variability"] = {
            metric_name: {"mean": float(np.mean([fm[metric_name] for fm in fold_metrics])),
                          "sd": float(np.std([fm[metric_name] for fm in fold_metrics], ddof=1))}
            for metric_name in ("macro_f1", "accuracy", "balanced_accuracy")
        }

    # Origin-level paired bootstrap intervals for QML minus its matched reduced classical control.
    rng = np.random.default_rng(SEED)
    deltas = {}
    qml_candidates = ["vqc", "qsvc"] + (["data_reupload"] if include_data_reuploading else []) + (["quantum_reservoir"] if include_quantum_reservoir else [])
    for candidate in qml_candidates:
        deltas[candidate] = {}
        for baseline in ("extra_trees_4f", "logistic_4f", "rbf_svc_4f"):
            pair = pred_frame[pred_frame.model.isin([candidate, baseline])].pivot(
                index=["record_id", "origin_program", "y_true", "fold"], columns="model", values="y_pred").reset_index()
            origin_ids = pair.origin_program.unique()
            samples = {"accuracy": [], "balanced_accuracy": [], "macro_f1": []}
            for _ in range(500):
                draw = rng.choice(origin_ids, size=len(origin_ids), replace=True)
                blocks = [pair[pair.origin_program == origin].assign(_bootstrap_slot=i)
                          for i, origin in enumerate(draw)]
                boot = pd.concat(blocks, ignore_index=True)
                for metric_name, fn in (("accuracy", accuracy_score), ("balanced_accuracy", balanced_accuracy_score),
                                        ("macro_f1", lambda a,b:f1_score(a,b,average="macro",zero_division=0))):
                    samples[metric_name].append(float(fn(boot.y_true, boot[candidate]) - fn(boot.y_true, boot[baseline])))
            deltas[candidate][baseline] = {name: {"mean_difference": float(np.mean(vals)),
                                                  "ci95_percentile": [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))],
                                                  "resamples": 500, "unit": "sampled validation origin"}
                                          for name, vals in samples.items()}

    resource = {
        "process_rss_before_bytes": int(memory_before),
        "process_rss_after_bytes": int(process.memory_info().rss),
        "resource_scope": "process RSS snapshots; OS peaks unavailable from portable API",
    }
    config = {"seed": SEED, "selected_origin_count": origin_count, "rows_per_origin_cap": rows_per_origin,
              "grouped_cv": f"GroupKFold(3) on {origin_count} seeded sampled development origins",
              "selected_model_inputs": N_FEATURES, "feature_selector": "fold-local mutual information; train fold only",
              "vqc": {"framework": "PennyLane", "device": "default.qubit", "interface": "autograd",
                      "diff_method": "backprop", "n_qubits": N_QUBITS, "depth": DEPTH,
                      "shots": None, "optimizer": "PennyLane AdamOptimizer", "steps": VQC_STEPS,
                      "loss": "class-balanced binary cross-entropy of analytic Pauli-Z-derived detected probability"},
              "qsvc": {"framework": "PennyLane + scikit-learn SVC", "device": "default.qubit",
                       "kernel": "analytic statevector fidelity overlap with entangling RY/RZ map", "shots": None, "C": 1.0},
              "data_reupload": {"enabled": include_data_reuploading, "architecture": "RY angle encoding repeated before every trainable Rot layer plus ring CNOTs"},
              "quantum_reservoir": {"enabled": include_quantum_reservoir, "architecture": "fixed seeded random variational circuit with per-qubit Pauli-Z readout; only class-balanced logistic readout is trained"},
              "classical_controls": ["100-tree ExtraTrees", "balanced LogisticRegression", "balanced RBF SVC C=1"],
              "frozen_test_labels_used": False, "frozen_test_evaluation": False}
    cost_summary = {}
    for model in model_names:
        fold_costs = [fold["models"][model] for fold in fold_records]
        cost_summary[model] = {
            "training_seconds_total": float(sum(item["train_seconds"] for item in fold_costs)),
            "inference_seconds_total": float(sum(item["inference_seconds"] for item in fold_costs)),
            "circuit_executions_train_total": int(sum(item["circuit_executions_train"] for item in fold_costs)),
            "circuit_executions_inference_total": int(sum(item["circuit_executions_inference"] for item in fold_costs)),
            "qubits": N_QUBITS if model in {"vqc", "qsvc", "data_reupload", "quantum_reservoir"} else None,
            "depth": DEPTH if model in {"vqc", "data_reupload", "quantum_reservoir"} else (2 if model == "qsvc" else None),
            "shots": None if model in {"vqc", "qsvc", "data_reupload", "quantum_reservoir"} else None,
            "vqc_loss_decreased_folds": sum(bool(item["loss_decreased"]) for item in fold_costs) if model in {"vqc", "data_reupload"} else None,
        }
    manifest = {"status": "pilot_complete", "date": "2026-10-26", "config": config,
                "data_audit": data_audit, "selected_origins": selected_origins,
                "sampled_records": int(len(sampled)), "sampled_rows_by_origin": sampled.groupby("origin_program").size().to_dict(),
                "sampled_label_counts": {str(k): int(v) for k,v in sampled.y.value_counts().sort_index().items()},
                "folds": fold_records, "feature_selection_by_fold": feature_records,
                "metrics": summary, "cost_summary": cost_summary,
                "paired_origin_bootstrap_deltas_qml_minus_classical": deltas,
                "runtime": {"elapsed_seconds": elapsed, "memory": resource,
                            "python": sys.version.split()[0], "platform": platform.platform(),
                            "pennylane": qml.__version__, "scikit_learn": sklearn.__version__,
                            "numpy": np.__version__, "pandas": pd.__version__},
                "interpretation": f"Development-only pilot over {origin_count} of 305 development origins; not a full development or frozen-test evaluation."}
    comparison_rows = []
    for model, values in summary.items():
        comparison_rows.append({"model": model, "protocol": f"same {origin_count}-origin sampled 3-fold grouped pilot",
                                "records": len(sampled), "origins": origin_count,
                                "accuracy": values["accuracy"], "balanced_accuracy": values["balanced_accuracy"],
                                "macro_f1": values["macro_f1"], "roc_auc": values["roc_auc"],
                                **{f"fold_{metric}_{stat}": v for metric, stats in values["fold_variability"].items()
                                   for stat, v in stats.items()}})
    # Full-feature incumbent results are included for context only; protocol differs.
    research = json.loads((root / "reports/experiments/qmutbench_final_research_report_20261026.json").read_text(encoding="utf-8"))
    incumbent = research["baseline_and_feature_audit"]["reproduced_oof"]["mutation_aware_metrics"]
    nested = research["nested_grouped_evaluation"]["nested_selected_metrics"]
    comparison_rows.extend([
        {"model": "existing_mutation_aware_extra_trees_full23", "protocol": "saved 3-fold full-development OOF; not same pilot split",
         "records": 578117, "origins": 305, "accuracy": incumbent["accuracy"],
         "balanced_accuracy": incumbent["balanced_accuracy"], "macro_f1": incumbent["macro_f1"], "roc_auc": incumbent["roc_auc"]},
        {"model": "existing_nested_grouped_development", "protocol": "saved five-fold nested development; not same pilot split",
         "records": 578117, "origins": 305, "accuracy": nested["accuracy"],
         "balanced_accuracy": nested["balanced_accuracy"], "macro_f1": nested["macro_f1"], "roc_auc": nested["roc_auc"]},
    ])
    pd.DataFrame(comparison_rows).to_csv(output_dir / "model_comparison.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output_dir / "split_manifest.csv", index=False)
    diff_rows = []
    for candidate, controls in deltas.items():
        for baseline, metric_map in controls.items():
            for metric_name, value in metric_map.items():
                diff_rows.append({"qml_model": candidate, "matched_classical_model": baseline,
                                  "metric": metric_name, "difference_percentage_points": 100 * value["mean_difference"],
                                  "ci95_low_percentage_points": 100 * value["ci95_percentile"][0],
                                  "ci95_high_percentage_points": 100 * value["ci95_percentile"][1],
                                  "resamples": value["resamples"], "bootstrap_unit": value["unit"]})
    pd.DataFrame(diff_rows).to_csv(output_dir / "paired_differences.csv", index=False)
    pred_frame.to_csv(output_dir / "predictions.csv", index=False)
    class_rows = []
    for model, values in summary.items():
        for label, stats in values["per_class"].items():
            class_rows.append({"model": model, "class": int(label), **stats})
    pd.DataFrame(class_rows).to_csv(output_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(feature_parity_table()).to_csv(output_dir / "feature_parity.csv", index=False)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "experiment_summary.json").write_text(json.dumps({"metrics": summary, "costs": cost_summary,
        "deltas_qml_minus_classical": deltas, "sampled_records": len(sampled), "sampled_origins": origin_count}, indent=2), encoding="utf-8")
    _write_report(output_dir / "research_report.md", manifest, summary, deltas, cost_summary, incumbent, nested, elapsed)
    return manifest


def _write_report(path, manifest, summary, deltas, costs, incumbent, nested, elapsed):
    lines = [
        "# QMutBench QML reduced-feature pilot",
        "",
        "## Status and scope",
        "",
        "This is a completed small development-only pilot, not a full-scale model selection or deployment result. "
        "It uses the verified historical mutation-outcome target (WOO/OPO detected; Survivor undetected). The source CSV is scanned in bounded chunks to retain saved development IDs; frozen-test rows are not retained for features, labels, prediction, or evaluation.",
        "",
        f"- Sample: {manifest['sampled_records']} records from {manifest['data_audit']['development_rows']:,} development rows and {manifest['data_audit']['development_origins']} origins; pilot used {len(manifest['selected_origins'])} seeded origins.",
        f"- Evaluation: {FOLDS}-fold GroupKFold over {len(manifest['selected_origins'])} sampled origins; per-origin cap {manifest['config']['rows_per_origin_cap']}; no origin overlap between train and validation.",
        f"- Feature selection: four encoded scalar inputs chosen by mutual information within each training fold only; fold-selected names are in `run_manifest.json`.",
        "- Source data and cached QASM-derived features were read-only; source SHA-256 verified against the synchronized ML record.",
        f"- Wall time: {elapsed:.2f}s; simulator: PennyLane {manifest['runtime']['pennylane']} `default.qubit`, analytic statevector execution.",
        "",
        "## Results",
        "",
        "Accuracy, balanced accuracy, and macro F1 are computed on pooled out-of-fold pilot predictions. Each QML model is compared with classical controls trained on the same selected four inputs and the same fold records.",
        "",
        "| Model | Accuracy | Balanced accuracy | Macro F1 | Fold macro F1 mean ± SD | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, m in summary.items():
        auc = "n/a" if m["roc_auc"] is None else f"{100*m['roc_auc']:.1f}%"
        fv = m.get("fold_variability", {}).get("macro_f1", {"mean": float("nan"), "sd": float("nan")})
        lines.append(f"| {model} | {100*m['accuracy']:.1f}% | {100*m['balanced_accuracy']:.1f}% | {100*m['macro_f1']:.1f}% | {100*fv['mean']:.1f}% ± {100*fv['sd']:.1f}% | {auc} |")
    lines += ["", "**Pilot finding:** compare QML rows with the matched classical rows above. This development-only pilot is exploratory and does not establish quantum advantage."]
    lines += ["", "### Paired origin bootstrap: QML minus matched classical model", "",
              f"Differences use QML metric minus matched classical metric and are reported in percentage points (pp); intervals resample the {len(manifest['selected_origins'])} validation origins, 500 times.", "",
              "| QML model | Matched baseline | Metric | Difference (pp) | 95% percentile interval (pp) |", "|---|---|---|---:|---:|"]
    for candidate, controls in deltas.items():
        for baseline, entries in controls.items():
            for metric_name, value in entries.items():
                lo, hi = value["ci95_percentile"]
                lines.append(f"| {candidate} | {baseline} | {metric_name} | {100*value['mean_difference']:+.1f} | [{100*lo:+.1f}, {100*hi:+.1f}] |")
    lines += ["", "## Context from the established full-feature ML work", "",
              f"- Existing full 23-feature mutation-aware Extra Trees: accuracy {100*incumbent['accuracy']:.1f}%, balanced accuracy {100*incumbent['balanced_accuracy']:.1f}%, macro F1 {100*incumbent['macro_f1']:.1f}% on saved 3-fold grouped OOF over all 578,117 development records.",
              f"- Existing nested grouped development result: accuracy {100*nested['accuracy']:.1f}%, balanced accuracy {100*nested['balanced_accuracy']:.1f}%, macro F1 {100*nested['macro_f1']:.1f}%.",
              "- These larger full-feature figures use different data coverage/protocols. They are context only and are not a direct QML comparison. This pilot does not establish improvement or quantum advantage.",
              "",
              "## Costs, limits, and reproducibility", "",
              "- Circuit/training costs by model are listed below; execution counts are PennyLane device execution totals, not physical shots.", "",
              "| Model | Train time (s, total) | Inference time (s, total) | Train circuits | Inference circuits | Qubits |",
              "|---|---:|---:|---:|---:|---:|"]
    for model, cost in costs.items():
        lines.append(f"| {model} | {cost['training_seconds_total']:.2f} | {cost['inference_seconds_total']:.2f} | {cost['circuit_executions_train_total']} | {cost['circuit_executions_inference_total']} | {cost['qubits'] or '-'} |")
    lines += ["", "- VQC: 4 qubits, 2 variational layers, 20 Adam updates per fold, analytic shots (`None`), backpropagated simulator gradients. Loss decreased in "
              f"{costs['vqc']['vqc_loss_decreased_folds']}/{FOLDS} folds; this is not a formal convergence criterion.",
              "- QSVC: 4-qubit analytic fidelity kernel and classical precomputed-kernel SVC. No quantum hardware or finite-shot noise was used.",
              f"- Per-model timings cover estimator fit/inference, not feature loading, fold-local encoding, or selection; total elapsed time covers the run. Memory records process RSS before and after the run; they are snapshots, not peak RSS. Small sample size, only {len(manifest['selected_origins'])} origins, and noisy fold-local feature rankings limit conclusions.",
              "- The QASM alignment features inherit the documented common-prefix/common-suffix alignment limitations. Only features selected among the same 23 inputs are used; `feature_parity.csv` documents categorical/numeric treatment and the train-only reduction.",
              "- Frozen test was deliberately not evaluated because this is a pilot and no candidate was frozen. No test prediction/metric is claimed.",
              f"- Process RSS snapshots: {manifest['runtime']['memory']['process_rss_before_bytes'] / (1024**2):.1f} MiB before and {manifest['runtime']['memory']['process_rss_after_bytes'] / (1024**2):.1f} MiB after; these are not peak memory.",
              "- Reproduce in the isolated env: `python -m venv qmutbench_qml/.venv`, install `qmutbench_qml/requirements-lock.txt`, then run `qmutbench_qml/.venv/Scripts/python.exe -m qmutbench_qml.training.run_pilot` from the repository root.",
              "- Dependencies are isolated under `qmutbench_qml/.venv`; production ML, simulator, and historical reports were not modified.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
