"""Benchmark additional QML families on the exact completed nine-origin pilot records."""
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from qmutbench_qml.data.loader import FEATURES, load_development_records
from qmutbench_qml.features.reduction import ReducedFeatureEncoder, feature_parity_table
from qmutbench_qml.models.quantum_reservoir import QuantumReservoirClassifier
from qmutbench_qml.models.vqc import DataReuploadingClassifier
from qmutbench_qml.training.pilot import (DEPTH, N_FEATURES, N_QUBITS, ORIGIN_COUNT,
    ROWS_PER_ORIGIN, SEED, VQC_STEPS, _choose_development_sample,
    _classify_score, metrics)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PILOT = ROOT / "qmutbench_qml/results"
OUT = ROOT / "qmutbench_qml/results/family_extension_9origins_20261027"
MODELS = ["vqc", "qsvc", "data_reupload", "quantum_reservoir",
          "extra_trees_4f", "logistic_4f", "rbf_svc_4f"]


def _paired_bootstrap(predictions):
    rng = np.random.default_rng(SEED)
    qml_names = ["vqc", "qsvc", "data_reupload", "quantum_reservoir"]
    classical = ["extra_trees_4f", "logistic_4f", "rbf_svc_4f"]
    rows = []
    for qname in qml_names:
        for cname in classical:
            pair = predictions[predictions.model.isin([qname, cname])].pivot(
                index=["record_id", "origin_program", "y_true", "fold"], columns="model", values="y_pred").reset_index()
            origins = pair.origin_program.unique()
            samples = {name: [] for name in ("accuracy", "balanced_accuracy", "macro_f1")}
            for _ in range(500):
                draw = rng.choice(origins, size=len(origins), replace=True)
                boot = pd.concat([pair[pair.origin_program == origin].assign(_slot=i)
                                  for i, origin in enumerate(draw)], ignore_index=True)
                for name, fn in (("accuracy", accuracy_score), ("balanced_accuracy", balanced_accuracy_score),
                                 ("macro_f1", lambda a,b: f1_score(a,b,average="macro",zero_division=0))):
                    samples[name].append(float(fn(boot.y_true, boot[qname]) - fn(boot.y_true, boot[cname])))
            for metric, values in samples.items():
                rows.append({"qml_model": qname, "matched_classical_model": cname, "metric": metric,
                             "difference_percentage_points": 100 * float(np.mean(values)),
                             "ci95_low_percentage_points": 100 * float(np.quantile(values, .025)),
                             "ci95_high_percentage_points": 100 * float(np.quantile(values, .975)),
                             "resamples": 500, "bootstrap_unit": "sampled development origin"})
    return pd.DataFrame(rows)


def run():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    memory_before = psutil.Process().memory_info().rss
    origins, selected_ids = _choose_development_sample(ROOT, ORIGIN_COUNT, ROWS_PER_ORIGIN)
    sampled, audit = load_development_records(ROOT, origins=origins, record_ids=selected_ids)
    if len(sampled) != 72:
        raise ValueError("Shared pilot sample should contain exactly 72 development records")
    sampled = sampled.set_index("record_id").loc[selected_ids].reset_index()

    old_predictions = pd.read_csv(SOURCE_PILOT / "predictions.csv")
    old_split = pd.read_csv(SOURCE_PILOT / "split_manifest.csv")
    old_manifest = json.loads((SOURCE_PILOT / "run_manifest.json").read_text(encoding="utf-8"))
    if set(old_predictions.record_id) != set(selected_ids) or set(old_predictions.model.unique()) != set(MODELS[:2] + MODELS[4:]):
        raise ValueError("Completed pilot predictions do not match the selected rows and expected model set")

    fold_rows, new_predictions, feature_records, model_costs = [], [], [], {}
    expected_split = set(zip(old_split.fold.astype(int), old_split.record_id.astype(int), old_split.role))
    for fold, (train_ix, val_ix) in enumerate(GroupKFold(n_splits=3).split(sampled, sampled.y, sampled.origin_program), start=1):
        train, valid = sampled.iloc[train_ix].reset_index(drop=True), sampled.iloc[val_ix].reset_index(drop=True)
        if set(train.origin_program) & set(valid.origin_program):
            raise AssertionError("Origin overlap in shared pilot fold")
        observed = set((fold, int(r.record_id), "train") for r in train.itertuples())
        observed |= set((fold, int(r.record_id), "validation") for r in valid.itertuples())
        if observed != {r for r in expected_split if r[0] == fold}:
            raise ValueError(f"New family split differs from the completed pilot in fold {fold}")
        encoder = ReducedFeatureEncoder(n_features=N_FEATURES, seed=SEED + fold).fit(train[FEATURES], train.y)
        previous = next(item["selected_features"] for item in old_manifest["feature_selection_by_fold"] if item["fold"] == fold)
        if encoder.selected_names != previous:
            raise ValueError(f"Train-fold feature selection differs from the completed pilot in fold {fold}")
        feature_records.append({"fold": fold, "selected_features": encoder.selected_names})
        x_train, x_valid = encoder.transform(train[FEATURES]), encoder.transform(valid[FEATURES])
        y_train, y_valid = train.y.to_numpy(int), valid.y.to_numpy(int)
        ids, groups = valid.record_id.to_numpy(int), valid.origin_program.astype(str).to_numpy()
        candidates = [
            ("data_reupload", DataReuploadingClassifier(n_qubits=N_QUBITS, depth=DEPTH,
                                                        steps=VQC_STEPS, seed=SEED + fold)),
            ("quantum_reservoir", QuantumReservoirClassifier(n_qubits=N_QUBITS, seed=SEED + fold)),
        ]
        for name, model in candidates:
            model.fit(x_train, y_train)
            score = model.predict_proba(x_valid)[:, 1]
            pred = _classify_score(name, score)
            new_predictions.extend({"record_id": int(rid), "origin_program": str(group), "y_true": int(y),
                                    "model": name, "y_pred": int(p), "score": float(s), "fold": fold}
                                   for rid, group, y, p, s in zip(ids, groups, y_valid, pred, score))
            model_costs.setdefault(name, []).append({
                "fold": fold, "train_seconds": float(model.training_seconds),
                "inference_seconds": float(model.inference_seconds),
                "circuit_executions_train": int(model.training_executions),
                "circuit_executions_inference": int(model.inference_executions),
                "qubits": N_QUBITS, "depth": DEPTH, "shots": None,
            })

    combined = pd.concat([old_predictions, pd.DataFrame(new_predictions)], ignore_index=True)
    combined.to_csv(OUT / "predictions.csv", index=False)
    split = old_split.drop_duplicates(["fold", "record_id", "origin_program", "role"])
    split.to_csv(OUT / "split_manifest.csv", index=False)
    pd.DataFrame(feature_records).to_json(OUT / "feature_selection_by_fold.json", orient="records", indent=2)
    summary, fold_metrics = {}, []
    for name, group in combined.groupby("model", sort=False):
        summary[name] = metrics(group.y_true.to_numpy(), group.y_pred.to_numpy(), group.score.to_numpy())
        for fold, part in group.groupby("fold"):
            m = metrics(part.y_true.to_numpy(), part.y_pred.to_numpy(), part.score.to_numpy())
            fold_metrics.append({"model": name, "fold": int(fold), **m})
        vals = [r for r in fold_metrics if r["model"] == name]
        summary[name]["fold_variability"] = {
            key: {"mean": float(np.mean([row[key] for row in vals])), "sd": float(np.std([row[key] for row in vals], ddof=1))}
            for key in ("macro_f1", "accuracy", "balanced_accuracy")}
    pd.DataFrame(fold_metrics).to_json(OUT / "fold_metrics.json", orient="records", indent=2)
    class_rows = [{"model": name, "class": int(label), **values}
                  for name, m in summary.items() for label, values in m["per_class"].items()]
    pd.DataFrame(class_rows).to_csv(OUT / "per_class_metrics.csv", index=False)
    comparisons = []
    for name, m in summary.items():
        comparisons.append({"model": name, "protocol": "same nine-origin, 72-record, three-fold pilot; matched 4-feature inputs",
                            "records": 72, "origins": 9, "accuracy": m["accuracy"],
                            "balanced_accuracy": m["balanced_accuracy"], "macro_f1": m["macro_f1"],
                            "roc_auc": m["roc_auc"],
                            **{f"fold_{key}_{stat}": value for key, stats in m["fold_variability"].items()
                               for stat, value in stats.items()}})
    incumbent = json.loads((ROOT / "reports/experiments/qmutbench_final_research_report_20261026.json").read_text(encoding="utf-8"))
    for name, result, protocol in [
        ("existing_mutation_aware_extra_trees_full23", incumbent["baseline_and_feature_audit"]["reproduced_oof"]["mutation_aware_metrics"],
         "historical full-development 3-fold OOF; context only, not matched"),
        ("existing_nested_grouped_development", incumbent["nested_grouped_evaluation"]["nested_selected_metrics"],
         "historical five-fold full-development nested result; context only, not matched"),
    ]:
        comparisons.append({"model": name, "protocol": protocol, "records": 578117, "origins": 305,
                            "accuracy": result["accuracy"], "balanced_accuracy": result["balanced_accuracy"],
                            "macro_f1": result["macro_f1"], "roc_auc": result["roc_auc"]})
    pd.DataFrame(comparisons).to_csv(OUT / "model_comparison.csv", index=False)
    differences = _paired_bootstrap(combined)
    differences.to_csv(OUT / "paired_differences.csv", index=False)
    pd.DataFrame(feature_parity_table()).to_csv(OUT / "feature_parity.csv", index=False)
    cost_summary = {name: old_manifest["cost_summary"][name] for name in MODELS[:2] + MODELS[4:]}
    for name, entries in model_costs.items():
        cost_summary[name] = {
            "training_seconds_total": float(sum(item["train_seconds"] for item in entries)),
            "inference_seconds_total": float(sum(item["inference_seconds"] for item in entries)),
            "circuit_executions_train_total": int(sum(item["circuit_executions_train"] for item in entries)),
            "circuit_executions_inference_total": int(sum(item["circuit_executions_inference"] for item in entries)),
            "qubits": N_QUBITS, "depth": DEPTH, "shots": None,
        }
    manifest = {"status": "completed_on_shared_pilot_sample", "date": "2026-10-27",
                "sample_size": len(sampled), "origins": origins, "record_ids": selected_ids,
                "source_audit": audit, "split_source": "completed qmutbench_qml/results nine-origin pilot; fold assignments verified identical",
                "feature_selection_by_fold": feature_records, "metrics": summary,
                "cost_summary": cost_summary,
                "additional_family_costs": model_costs,
                "reused_pilot_model_costs": {name: old_manifest["cost_summary"][name]
                                               for name in MODELS[:2] + MODELS[4:]},
                "reused_completed_pilot_models": ["vqc", "qsvc", "extra_trees_4f", "logistic_4f", "rbf_svc_4f"],
                "smoke_test_results": {"data_reuploading": "passed", "quantum_reservoir": "passed"},
                "larger_24_origin_attempt": "stopped when available system memory fell below 450 MiB; no fold results were written",
                "resource": {"rss_before_bytes": int(memory_before), "rss_after_bytes": int(psutil.Process().memory_info().rss)},
                "runtime": {"elapsed_seconds": time.perf_counter() - started, "python": sys.version.split()[0],
                            "platform": platform.platform(), "pennylane": qml.__version__},
                "frozen_test_labels_used": False, "frozen_test_evaluation": False,
                "interpretation": "Small matched development-only pilot; no QML advantage claim or QML model selected for frozen-test evaluation."}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = ["# QMutBench additional QML family pilot", "",
              "This extension reuses the completed VQC, QSVC, and matched classical predictions from the nine-origin pilot, then evaluates data re-uploading and quantum reservoir classifiers on the identical 72 records, origin assignments, fold assignments, and fold-local four-feature inputs. Frozen-test records and labels were not read.", "",
              "| Model | Accuracy | Balanced accuracy | Macro F1 | Fold macro F1 mean +/- SD | ROC-AUC |", "|---|---:|---:|---:|---:|---:|"]
    for name, m in summary.items():
        f = m["fold_variability"]["macro_f1"]
        report.append(f"| {name} | {100*m['accuracy']:.1f}% | {100*m['balanced_accuracy']:.1f}% | {100*m['macro_f1']:.1f}% | {100*f['mean']:.1f}% +/- {100*f['sd']:.1f}% | {100*m['roc_auc']:.1f}% |")
    report += ["", "## Family accounting", "",
               "VQC is the trainable QNN in this environment, so it is not counted twice as another independent family. QSVC is the quantum-kernel family. Data re-uploading is a distinct trainable-circuit architecture. Quantum reservoir computing uses a fixed quantum map and a trained classical readout.", "",
               "The larger 24-origin expansion was stopped after system available memory dropped below 450 MiB; its interrupted attempt wrote no fold metrics. This report instead extends the already completed nine-origin pilot using exactly matched records and folds. Results are exploratory and do not establish quantum advantage. The 23-feature historical Extra Trees and nested grouped metrics in `model_comparison.csv` are context only because their coverage and protocol differ.", "",
               "| Model | Total train s | Total inference s | Train circuit executions | Inference circuit executions | Qubits | Depth |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, cost in cost_summary.items():
        report.append(f"| {name} | {cost['training_seconds_total']:.2f} | {cost['inference_seconds_total']:.2f} | {cost['circuit_executions_train_total']} | {cost['circuit_executions_inference_total']} | {cost['qubits'] or '-'} | {cost['depth'] or '-'} |")
    report += ["",
               f"Elapsed extension runtime: {manifest['runtime']['elapsed_seconds']:.2f}s. Process RSS before/after: {memory_before/1024**2:.1f}/{manifest['resource']['rss_after_bytes']/1024**2:.1f} MiB. Circuit evaluations, qubit count, depth, fold metrics, per-class scores, and origin-bootstrap intervals are in the adjacent CSV/JSON outputs.", ""]
    (OUT / "research_report.md").write_text("\n".join(report), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
