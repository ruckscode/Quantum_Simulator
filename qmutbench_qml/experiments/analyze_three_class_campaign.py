"""Reproducibly consolidate development-only three-class QNN campaign artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, log_loss, precision_recall_fscore_support)
from qmutbench_qml.models.multiclass_vqc import ThreeClassVQC

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/experiments/qmutbench_three_class_qnn_campaign_20261027"
ENCODINGS = {
    "angle_reupload": (OUT, "three_class_vqc"),
    "amplitude": (OUT / "amplitude_encoding", "three_class_vqc_amplitude"),
    "angle_linear": (OUT / "trainable_readout", "three_class_vqc_angle_linear"),
}
SEED = 20261028


def _metric(y, pred):
    return {"macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred))}


def _bootstrap_delta(a, b, groups, reps=2000):
    rng = np.random.default_rng(SEED)
    group_ids = np.array(sorted(pd.unique(groups)), dtype=object)
    parts = {g: np.flatnonzero(groups == g) for g in group_ids}
    deltas = []
    for _ in range(reps):
        chosen = rng.choice(group_ids, size=len(group_ids), replace=True)
        idx = np.concatenate([parts[g] for g in chosen])
        y = a.y_true.to_numpy()[idx]
        d_a = f1_score(y, a.y_pred.to_numpy()[idx], labels=[0, 1, 2], average="macro", zero_division=0)
        d_b = f1_score(y, b.y_pred.to_numpy()[idx], labels=[0, 1, 2], average="macro", zero_division=0)
        deltas.append(d_a - d_b)
    return {"delta_macro_f1": float(_metric(a.y_true, a.y_pred)["macro_f1"] -
                                      _metric(b.y_true, b.y_pred)["macro_f1"]),
            "origin_bootstrap_95_ci": [float(x) for x in np.quantile(deltas, [.025, .975])],
            "bootstrap_replicates": reps, "origin_count": len(group_ids)}


def main():
    predictions = {}
    summaries = {}
    fold_rows, per_origin, curves, config_rows = [], [], [], []
    for encoding, (directory, qnn_name) in ENCODINGS.items():
        summary = json.loads((directory / "campaign_summary.json").read_text(encoding="utf-8"))
        pred = pd.read_csv(directory / "predictions.csv.gz")
        predictions[encoding] = pred
        summaries[encoding] = summary
        qnn = pred[pred.model == qnn_name].copy().sort_values("record_id").reset_index(drop=True)
        if not qnn.y_true.isin([0, 1, 2]).all() or qnn.record_id.duplicated().any():
            raise AssertionError(f"Invalid QNN predictions for {encoding}")
        for model, group in pred.groupby("model"):
            for origin, og in group.groupby("origin_program"):
                per_origin.append({"encoding": encoding, "model": model, "origin_program": origin,
                    "records": len(og), **_metric(og.y_true, og.y_pred)})
            fold_metrics = pd.read_json(directory / "fold_metrics.json")
            for row in fold_metrics[fold_metrics.model == model].to_dict("records"):
                fold_rows.append({"encoding": encoding, **row})
        curves.append(pd.read_csv(directory / "training_curves.csv").assign(encoding=encoding))
        config = summary["qnn_configuration"]
        model = ThreeClassVQC(n_qubits=config["qubits"], depth=config["depth"],
            encoding="amplitude" if encoding == "amplitude" else "angle",
            data_reuploading=(encoding != "amplitude"), trainable_readout=(encoding == "angle_linear"))
        specs = qml.specs(model.circuit)(np.array([.11, .27, .63, .89]),
            np.zeros((2, config["qubits"], 3)))
        if encoding == "amplitude":
            specs = qml.specs(qml.transforms.decompose(model.circuit))(
                np.array([.11, .27, .63, .89]), np.zeros((2, config["qubits"], 3)))
        config_rows.append({"variant": encoding, "ansatz_layers": config["depth"],
            "logical_qubits": config["qubits"], "trainable_parameters": config["trainable_parameters"],
            "encoding_description": config["encoding"], "readout_description": config["readout"],
            "compiled_circuit_depth": int(specs.resources.depth),
            "compiled_gate_count": int(sum(specs.resources.gate_types.values())),
            "compiled_gate_types": specs.resources.gate_types,
            "state_preparation_decomposed": encoding == "amplitude", "shots": None,
            "simulator": "PennyLane default.qubit; analytic statevector"})

    # All controls are expected to be identical because the sampling seed and grouped splitter are fixed.
    base = predictions["angle_reupload"]
    for encoding, pred in predictions.items():
        keys = ["record_id", "origin_program", "fold", "y_true"]
        expected = base[base.model == "extra_trees_4f"][keys].sort_values("record_id").reset_index(drop=True)
        actual = pred[pred.model == "extra_trees_4f"][keys].sort_values("record_id").reset_index(drop=True)
        if not expected.equals(actual):
            raise AssertionError(f"{encoding} does not share identical held-out rows/folds")

    comparisons = []
    reference = base[base.model == "extra_trees_4f"].sort_values("record_id").reset_index(drop=True)
    for encoding, (_, name) in ENCODINGS.items():
        qnn = predictions[encoding].query("model == @name").sort_values("record_id").reset_index(drop=True)
        if not qnn[["record_id", "origin_program", "fold", "y_true"]].equals(
                reference[["record_id", "origin_program", "fold", "y_true"]]):
            raise AssertionError("QNN and matched controls are not paired")
        for control in ("extra_trees_4f", "logistic_4f", "rbf_svc_4f"):
            other = predictions[encoding].query("model == @control").sort_values("record_id").reset_index(drop=True)
            delta = _bootstrap_delta(qnn, other, qnn.origin_program.to_numpy())
            comparisons.append({"qnn": name, "control": control, **delta})

    all_qnn = []
    for encoding, (_, name) in ENCODINGS.items():
        qnn = predictions[encoding].query("model == @name").copy()
        pooled = _metric(qnn.y_true, qnn.y_pred)
        prec, rec, f1, sup = precision_recall_fscore_support(qnn.y_true, qnn.y_pred,
            labels=[0, 1, 2], zero_division=0)
        qnn_metrics = {"encoding": encoding, "model": name, **pooled,
            "per_class": {str(c): {"precision": float(prec[c]), "recall": float(rec[c]),
                                   "f1": float(f1[c]), "support": int(sup[c])} for c in range(3)},
            "confusion_matrix": confusion_matrix(qnn.y_true, qnn.y_pred, labels=[0, 1, 2]).tolist(),
            "validation_log_loss": float(log_loss(qnn.y_true,
                qnn[["prob_0", "prob_1", "prob_2"]].to_numpy(), labels=[0, 1, 2]))}
        all_qnn.append(qnn_metrics)

    # Collapse multiclass predictions to the incumbent's binary target for same-record context only.
    champion_path = ROOT / "data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/mutation_diff_oof.npz"
    champion_data = np.load(champion_path, allow_pickle=True)
    champion = pd.DataFrame({"record_id": champion_data["record_id"], "y_binary": champion_data["y"],
        "origin_program": champion_data["origin_program"], "champion_fold": champion_data["fold"],
        "probability": champion_data["probability"]})
    sample = predictions["angle_reupload"].query("model == 'extra_trees_4f'")[["record_id", "origin_program", "y_true"]].drop_duplicates()
    matched = sample.merge(champion, on="record_id", validate="one_to_one", suffixes=("_sample", "_champion"))
    if len(matched) != len(sample) or not (matched.y_binary == (matched.y_true != 0).astype(int)).all():
        raise AssertionError("Saved binary champion OOF predictions do not align to sampled three-class records")
    incumbent_pred = (matched.probability.to_numpy() >= .5).astype(int)
    incumbent_metrics = _metric_binary(matched.y_binary.to_numpy(), incumbent_pred)
    collapsed = []
    for encoding, (_, name) in ENCODINGS.items():
        qnn = predictions[encoding].query("model == @name").sort_values("record_id")
        yy = (qnn.y_true.to_numpy() != 0).astype(int)
        pp = (qnn.y_pred.to_numpy() != 0).astype(int)
        collapsed.append({"encoding": encoding, **_metric_binary(yy, pp)})

    resources = {encoding: {"elapsed_seconds": summary["elapsed_seconds"],
        "available_before_mib": summary["memory"]["available_ram_before"] / 2**20,
        "available_after_mib": summary["memory"]["available_ram_after"] / 2**20,
        "rss_before_mib": summary["memory"]["rss_before"] / 2**20,
        "qnn_folds": [{"fold": r["fold"], "runtime_seconds": r["runtime_seconds"],
                       "train_executions": r["circuit_train_evaluations"],
                       "validation_executions": r["circuit_validation_evaluations"],
                       "qubits": r["qubits"], "depth": r["depth"], "parameters": r["parameters"]}
                      for r in summary["fold_metrics"] if r["model"].startswith("three_class_vqc_") or r["model"] == "three_class_vqc"]}
        for encoding, (_, _name) in ENCODINGS.items() for summary in [summaries[encoding]]}

    pd.DataFrame(fold_rows).to_csv(OUT / "all_fold_metrics.csv", index=False)
    pd.DataFrame(per_origin).to_csv(OUT / "per_origin_metrics.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(OUT / "all_training_curves.csv", index=False)
    pd.DataFrame(config_rows).to_json(OUT / "encoding_and_circuit_config.json", orient="records", indent=2)
    pd.DataFrame(comparisons).to_csv(OUT / "paired_origin_bootstrap.csv", index=False)
    pd.DataFrame(all_qnn).to_json(OUT / "qnn_pooled_metrics.json", orient="records", indent=2)

    leaderboard = []
    for model_name, group in base.groupby("model"):
        if model_name not in {"extra_trees_4f", "logistic_4f", "rbf_svc_4f"}:
            continue
        metrics = _metric(group.y_true, group.y_pred)
        folds_for_model = pd.read_json(OUT / "fold_metrics.json").query("model == @model_name")
        per_class = precision_recall_fscore_support(group.y_true, group.y_pred,
            labels=[0, 1, 2], zero_division=0)
        leaderboard.append({"model": model_name, "family": "classical_control", **metrics,
            "fold_macro_f1_mean": float(folds_for_model.macro_f1.mean()),
            "fold_macro_f1_sd": float(folds_for_model.macro_f1.std(ddof=1)),
            "per_class_f1": json.dumps({str(i): float(per_class[2][i]) for i in range(3)}),
            "runtime_seconds_total": float(folds_for_model.runtime_seconds.sum())})
    all_fold_frame = pd.read_csv(OUT / "all_fold_metrics.csv")
    for item in all_qnn:
        folds_for_model = all_fold_frame.loc[all_fold_frame.model == item["model"]]
        leaderboard.append({"model": item["model"], "family": "supervised_quantum_neural_network",
            "macro_f1": item["macro_f1"], "accuracy": item["accuracy"],
            "balanced_accuracy": item["balanced_accuracy"],
            "fold_macro_f1_mean": float(folds_for_model.macro_f1.mean()),
            "fold_macro_f1_sd": float(folds_for_model.macro_f1.std(ddof=1)),
            "per_class_f1": json.dumps({k: v["f1"] for k, v in item["per_class"].items()}),
            "runtime_seconds_total": float(folds_for_model.runtime_seconds.sum())})
    pd.DataFrame(leaderboard).to_csv(OUT / "candidate_leaderboard.csv", index=False)
    validation_rows = []
    for encoding, pred in predictions.items():
        for (model_name, fold), group in pred.groupby(["model", "fold"]):
            validation_rows.append({"encoding": encoding, "model": model_name, "fold": int(fold),
                "validation_log_loss": float(log_loss(group.y_true,
                    group[["prob_0", "prob_1", "prob_2"]].to_numpy(), labels=[0, 1, 2]))})
    pd.DataFrame(validation_rows).to_csv(OUT / "validation_loss_by_fold.csv", index=False)

    result = {"status": "completed_no_qml_candidate_approached_matched_controls",
        "label_map": {"Survivor": 0, "WOO": 1, "OPO": 2}, "qnn_results": all_qnn,
        "paired_origin_bootstrap": comparisons, "matched_controls":
            base.groupby("model").apply(lambda g: _metric(g.y_true, g.y_pred), include_groups=False).to_dict(),
        "binary_champion_same_record_context": {"metrics": incumbent_metrics,
            "records": len(matched), "origins": int(matched.origin_program_sample.nunique()),
            "paired_qnn_collapsed": collapsed,
            "comparison_note": "Saved champion is a different binary target, trained on far more records and existing five-fold OOF splits; same records are shown only as protocol context, not a matched three-class comparison."},
        "sample": summaries["angle_reupload"]["sample"], "resources": resources,
        "confirmation_attempt": {"status": "stopped_at_memory_safety_floor",
        "reason": "The single fit using the 12-origin training sample and 36 reserved OPO-bearing development origins crossed below 500 MiB available RAM before completing; no predictions or metrics were written and no retry was made.",
            "checkpoint_saved": False,
            "memory_optimized_followup": "not launched on resume; raw RAM was 853 MiB before framework imports, so 700 MiB post-import headroom could not be established"},
        "test_integrity": {"frozen_test_accessed": False, "production_model_modified": False,
                           "source_and_split_manifests_modified": False},
        "next_search_decision": "Do not scale. Three distinct encoding/readout hypotheses failed to beat matched classical controls on the conditional class-balanced sample. A single-candidate holdout fit using the remaining 36 OPO-bearing development origins hit the 500 MiB safety floor and stopped before metrics; further training requires a larger-memory host."}
    (OUT / "consolidated_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_report(result, comparisons, resources, config_rows)
    _write_campaign_management_artifacts(result, resources, config_rows, summaries)
    return result


def _write_campaign_management_artifacts(result, resources, configs, summaries):
    foundation_checks = summaries["angle_reupload"].get("foundations", [])
    toy = summaries["angle_reupload"]["toy_learning_check"]
    max_state_error = max(row.get("state_max_abs_error", 0.0) for row in foundation_checks)
    max_gate_error = max(abs(row.get("z_expectation", row.get("bell_state_norm", 1.0)) -
                             row.get("expected_z", row.get("bell_state_norm", 1.0)))
                         for row in foundation_checks)
    foundation_report = [
        "# Quantum foundations and circuit validation", "",
        "## State and measurement", "",
        "A normalized qubit is |psi> = alpha|0> + beta|1>, with |alpha|^2 + |beta|^2 = 1. Computational-basis measurement gives probabilities |alpha|^2 and |beta|^2. The RY(theta) gate is unitary and produces RY(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>. Thus P(0)=cos^2(theta/2), P(1)=sin^2(theta/2), <Z>=cos(theta), and d<Z>/dtheta=-sin(theta). At theta=pi/2 it creates equal superposition. Hadamard followed by CNOT creates the Bell state (|00>+|11>)/sqrt(2).",
        "", "The PennyLane `default.qubit` one- and two-qubit statevectors, norms, basis probabilities, Pauli-Z expectations, and autodiff gradients were compared against those closed-form values at theta=0, pi/3, and pi/2. Bell-state amplitudes and probabilities were also checked. Maximum state-amplitude absolute error was {:.3g}; gradient and expectation errors were below numerical tolerance in the tests.".format(max_state_error),
        "", "## Data encoding and supervised readout", "",
        "The angle QNN uses fold-local four-feature selection/scaling, then RY(pi*x_i) on four wires. Re-uploading repeats this encoding before each of two trainable Rot layers, with nearest-neighbor and wraparound CNOTs. The measured Z expectations on wires 0, 1, and 2 become three logits; softmax makes valid probabilities and class-weighted cross-entropy trains the circuit parameters.",
        "", "The amplitude alternative normalizes the four selected features into a two-qubit state, then applies a depth-two trainable circuit. It discards the feature-vector norm and requires explicit state preparation; PennyLane decomposition counted its preparation gates in the compiled depth. Its readout uses Z0, Z1, and Z0Z1 expectations.",
        "", "The trainable-readout hybrid jointly learns a 3x3 affine matrix and bias on the three quantum expectations. This tests whether a restricted fixed expectation-to-logit mapping caused weak class separation. The 24 circuit parameters plus 12 head parameters are optimized end-to-end by Adam and autodiff.",
        "", "## Optimization and validation evidence", "",
        f"The controlled 12-row toy task reduced weighted cross-entropy from {toy['initial_loss']:.4f} to {toy['final_loss']:.4f} and predicted {toy['correct_of_12']}/12 correctly. Circuit parameters changed: {toy['parameters_changed']}; probabilities finite and normalized: {toy['probabilities_valid']}; training used {toy['circuit_executions']} analytic circuit executions. Real-data fold training curves and final validation losses are separate CSV artifacts.",
        "", "Circuit resource summary is in `encoding_and_circuit_config.json`: angle circuits are 4-qubit, depth 12 after PennyLane resource expansion (two ansatz layers), 24 or 36 trainable parameters; amplitude is 2-qubit, decomposed depth 8, 12 trainable parameters. This is ideal analytic simulation; shots, noise, and hardware were not used. For N independent shots, a Pauli-Z sample mean has standard error sqrt((1-<Z>^2)/N); this campaign did not run finite-shot estimates.",
        "", "## Scope limits", "", "Basis encoding was excluded because it would discretize fold-selected continuous/dummy values and discard magnitude information. No claim is made that these small designs exhaust all QNN architectures. The actual development evaluation sample is class-balanced and conditional on OPO-bearing origins; results do not estimate natural-prevalence performance. The Initial held-out confirmation stopped at the 500 MiB floor. Exact chunked full-batch gradients then matched the ordinary toy update within 2.4e-12, but safe post-import RAM headroom was not established for a real-data retry."
    ]
    (OUT / "foundations_and_circuit_validation.md").write_text("\n".join(foundation_report) + "\n", encoding="utf-8")

    ledger = [
        {"id": "QNN-01", "hypothesis": "Verify state preparation, unitary evolution, entangling gate, measurement, and gradient independently.", "status": "passed", "evidence": "Foundations analytic checks and focused tests."},
        {"id": "QNN-02", "hypothesis": "A supervised variational circuit can learn a controlled three-class nonlinear toy task.", "status": "passed", "evidence": f"Toy loss {toy['initial_loss']:.4f} -> {toy['final_loss']:.4f}; {toy['correct_of_12']}/12 correct."},
        {"id": "QNN-03", "hypothesis": "Four-feature repeated angle embedding with trainable VQC learns QMutBench labels on unseen origins.", "status": "completed_no_advantage", "evidence": "12 origins / 205 stratified records / 3 grouped folds; macro F1 0.2998 vs Extra Trees 0.4669."},
        {"id": "QNN-04", "hypothesis": "Amplitude encoding may preserve four reduced inputs with fewer wires and improve QNN learning.", "status": "completed_rejected", "evidence": "Macro F1 0.3086; 2 qubits, decomposed depth 8, preparation included; no matched-control gain."},
        {"id": "QNN-05", "hypothesis": "A trainable affine measurement head may remedy low separation from bounded fixed expectation logits.", "status": "completed_rejected", "evidence": "Macro F1 0.3218; still below matched LR 0.4407 and ET 0.4669."},
        {"id": "QNN-06", "hypothesis": "Confirm the selected QNN once on the remaining OPO-bearing origin groups.", "status": "stopped_resource_floor", "evidence": "Training crossed below 500 MiB available RAM before fit completion; no predictions/metrics/checkpoint were written and no retry was made."},
        {"id": "QNN-07", "hypothesis": "Scale to all 578,117 development records or 305 origins.", "status": "not_run", "evidence": "All tested QNNs trail matched controls by 10.9-16.7 pp; held-out confirmation hit RAM floor; scale-up unjustified."},
        {"id": "QNN-08", "hypothesis": "Use the historic frozen test for final QNN evaluation.", "status": "prohibited_not_run", "evidence": "Historical test previously reused; frozen test remained untouched."}
        ,{"id": "QNN-09", "hypothesis": "Distinct three-class quantum-kernel or fixed-reservoir families may complement supervised QNN.", "status": "not_run", "evidence": "Existing kernel/reservoir implementations address the binary target and underperformed their matched controls. Three-class QNN variants also trail matched classical controls; resources are constrained. No three-class extension was justified."}
        ,{"id": "QNN-10", "hypothesis": "Accumulate exact full-batch gradients in small chunks to lower peak memory without changing optimizer updates.", "status": "implemented_and_smoke_validated", "evidence": "One-step, six-row comparison: chunk size 2 matched full-batch loss within 2.3e-16, circuit weights within 2.4e-12, readout within 7e-18; both used six circuit executions. The 36-origin confirmation itself was not launched because current available RAM was 663 MiB, below 700 MiB preflight."}
    ]
    (OUT / "experiment_ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    checkpoint_manifest = {"status": "no_real_data_qnn_parameter_checkpoint",
        "why": "The initial real-data final-candidate fit stopped at the 500 MiB safety floor. Exact-gradient chunking was only smoke-validated on toy data; the real-data confirmation was not launched because current RAM was below its 700 MiB preflight. No real-data weights are fabricated or recovered from metrics.",
        "saved": ["predictions.csv.gz", "amplitude_encoding/predictions.csv.gz",
                  "trainable_readout/predictions.csv.gz", "grouped_split_manifest.csv",
                  "sample_manifest.csv", "features_by_fold.json", "training_curves.csv",
                  "amplitude_encoding/training_curves.csv", "trainable_readout/training_curves.csv"],
        "training_candidate_checkpoint": None, "frozen_test_accessed": False}
    (OUT / "checkpoint_manifest.json").write_text(json.dumps(checkpoint_manifest, indent=2), encoding="utf-8")

    progress = {"status": "completed_exploration_confirmation_blocked_by_memory",
        "objective": "Test an end-to-end supervised QNN for three-class QMutBench labels with matched grouped controls.",
        "stages": {"source_label_and_task_audit": "complete; 3 raw classes vs incumbent binary mapping resolved",
            "quantum_foundations": "complete; analytic statevector/gate/measurement/gradient validation",
            "toy_supervised_qnn": "complete; loss and accuracy improved",
            "angle_reupload_qnn": "complete; 3 grouped folds",
            "amplitude_qnn": "complete; 3 grouped folds",
            "trainable_readout_qnn": "complete; 3 grouped folds",
            "matched_classical_controls": "complete; same features, records, origins, folds",
            "paired_origin_uncertainty": "complete; 2,000 origin bootstrap resamples",
            "reserved_origin_confirmation": "initial fit stopped below 500 MiB; no results",
            "memory_optimized_confirmation": "exact full-batch gradient chunking implemented and toy-equivalence checked; not launched below 700 MiB preflight",
            "frozen_test": "untouched"},
        "decision": "No QNN promoted; all three variants trail matched classical controls. No further QML training on this host.",
        "next_requirements": ["Larger memory capacity", "Predeclared origin-representative sample", "Multiple seeds", "Untouched development-origin confirmation", "Natural-prevalence evaluation"],
        "report": "research_report.md", "summary": "consolidated_summary.json",
        "ledger": "experiment_ledger.json", "checkpoint_manifest": "checkpoint_manifest.json",
        "resource_log": "resource_log.json", "leaderboard": "candidate_leaderboard.csv"}
    (OUT / "progress_manifest.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")

    runs = {}
    for encoding, summary in summaries.items():
        qnn_rows = [row for row in summary["fold_metrics"] if row["model"].startswith("three_class_vqc") or row["model"] == "three_class_vqc"]
        runs[encoding] = {"elapsed_seconds": summary["elapsed_seconds"],
            "rss_at_start_bytes": summary["memory"]["rss_before"],
            "available_before_bytes": summary["memory"]["available_ram_before"],
            "available_after_bytes": summary["memory"]["available_ram_after"],
            "qnn_fold_runtime_seconds": [row["runtime_seconds"] for row in qnn_rows],
            "qnn_train_circuit_evaluations": [row["circuit_train_evaluations"] for row in qnn_rows],
            "qnn_validation_circuit_evaluations": [row["circuit_validation_evaluations"] for row in qnn_rows],
            "qnn_qubits": 2 if encoding == "amplitude" else 4,
            "compiled_circuit_depth": next(row["compiled_circuit_depth"] for row in configs if row["variant"] == encoding),
            "trainable_parameters": next(row["trainable_parameters"] for row in configs if row["variant"] == encoding),
            "simulator": "PennyLane default.qubit analytic CPU statevector"}
    resource = {"environment": {"python": "3.12.10", "pennylane": "0.45.1",
            "scikit_learn": "1.9.1", "hardware": "CPU only", "device_total_ram_mib": 7488},
        "runs": runs, "observed_available_ram_min_mib": 547,
        "confirmation": {"status": "initial_fit_stopped; optimized_fit_not_launched",
            "stop_floor_mib": 500, "optimized_preflight_requirement_mib": 700,
            "latest_available_ram_mib": 663,
            "reason": "Initial fit crossed the 500 MiB floor; the single-pass loader and exact chunked-gradient followup was not launched below 700 MiB preflight."},
        "chunked_gradient_smoke": {"records": 6, "full_batch_vs_chunk_size": 2,
            "steps": 1, "loss_abs_difference": 2.3e-16, "circuit_weight_max_abs_difference": 2.4e-12,
            "readout_max_abs_difference": 7e-18, "circuit_executions_each": 6,
            "real_qmutbench_fit": False},
        "earlier_24_origin_attempt": "Stopped below 450 MiB before producing fold metrics.",
        "process_peak_rss": "not measured", "full_scale_qnn_training": "not attempted"}
    (OUT / "resource_log.json").write_text(json.dumps(resource, indent=2), encoding="utf-8")

    progress["reports"] = sorted([p.name for p in OUT.iterdir() if p.is_file()])
    (OUT / "progress_manifest.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")


def _metric_binary(y, pred):
    return {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "confusion_matrix_labels_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def _write_report(result, comparisons, resources, config_rows):
    class_names = {"0": "Survivor", "1": "WOO", "2": "OPO"}
    lines = [
        "# QMutBench three-class supervised QNN campaign — 2026-10-27", "",
        "## Executive result", "",
        "A genuine end-to-end supervised three-class QNN was implemented, mathematically checked, trained on real development records, and tested against matched classical controls. It learned the toy task, but none of the three QNN encoding/readout variants beat the matched four-feature classical models. The outcome supports no quantum advantage claim and does not establish that all QML designs fail.", "",
        "The project champion predicts a different binary target (WOO/OPO merged versus Survivor). It remains unchanged. Its 79.33% five-fold OOF macro F1 is not directly comparable to this three-class study. The raw source contains three labels, but the established production target intentionally merges WOO and OPO. This campaign follows the directive’s separate three-class research objective without changing the incumbent.", "",
        "## Verified labels and development sample", "",
        f"Raw development label distribution: {result['sample']['development_label_counts']}. Mapping: Survivor=0, WOO=1, OPO=2. OPO occurs in only {result['sample']['development_origins_with_OPO']} of the 305 development origins.",
        f"For class coverage, a seeded sample selected 12 OPO-bearing origins and up to six records per class within each origin. It contains {result['sample']['sample_label_counts']} across 205 rows. This deliberately class-balanced, rare-origin sample is not prevalence-representative and estimates transfer only among these OPO-bearing origins. It is not the earlier nine-origin QML pilot.", "",
        "Three-fold StratifiedGroupKFold split the 12 origins into disjoint groups; each validation fold covered all three labels. Feature selection and scaling were fitted separately inside each training fold. All QNNs and classical controls used identical transformed four-dimensional inputs and held-out record IDs/origins/folds. No frozen-test rows were used.", "",
        "## Supervised QNN foundations and learning loop", "",
        "For a one-qubit state |0>, RY(theta) produces cos(theta/2)|0> + sin(theta/2)|1>. Therefore P(0)=cos²(theta/2), P(1)=sin²(theta/2), <Z>=cos(theta), and d<Z>/dtheta=-sin(theta). PennyLane statevectors, probabilities, expectation values, and autodiff gradients matched these analytic identities at theta=0, pi/3, and pi/2. Hadamard/CNOT entanglement behavior is covered by the circuit validation tests.", "",
        "The QNN applies angle or amplitude feature encoding, trainable Rot gates and CNOT entanglers, measures three Pauli expectations as logits, applies softmax, and optimizes weighted three-class cross-entropy with PennyLane Adam/backprop. The hybrid readout variant jointly learns an affine 3x3 head and bias. Circuit parameters and, where enabled, head parameters demonstrably changed. The toy task loss fell from 1.304 to 0.716 and classified 12/12 examples.", "",
        "All reported real-data circuits use ideal analytic PennyLane `default.qubit` statevector simulation with no finite shots or noise. This says nothing about shot noise, noisy devices, hardware performance, or computational speedup. A finite-shot Pauli estimate would have standard error sqrt((1-<Z>²)/N); it was not simulated.", "",
        "## Matched development results", "",
        "Macro F1 is primary; pooled metrics are computed across identical paired validation records. Fold standard deviations are across three grouped folds.", "",
        "| Model | Pooled macro F1 | Accuracy | Balanced accuracy | Fold macro F1 mean +/- SD |",
        "|---|---:|---:|---:|---:|",
    ]
    base = pd.read_csv(OUT / "predictions.csv.gz")
    for encoding, (folder, model_name) in ENCODINGS.items():
        pred = pd.read_csv(folder / "predictions.csv.gz").query("model == @model_name")
        folds = pd.read_json(folder / "fold_metrics.json").query("model == @model_name")
        m = _metric(pred.y_true, pred.y_pred)
        lines.append(f"| {model_name} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | {folds.macro_f1.mean():.4f} +/- {folds.macro_f1.std(ddof=1):.4f} |")
    for model in ("extra_trees_4f", "logistic_4f", "rbf_svc_4f"):
        pred = base.query("model == @model")
        folds = pd.read_json(OUT / "fold_metrics.json").query("model == @model")
        m = _metric(pred.y_true, pred.y_pred)
        lines.append(f"| {model} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | {folds.macro_f1.mean():.4f} +/- {folds.macro_f1.std(ddof=1):.4f} |")
    lines += ["", "### Per-class results and errors", "", f"QNN per-class labels are {class_names}. Pooled QNN confusions, per-class precision/recall/F1/support, validation log loss, and per-origin metrics are saved in `qnn_pooled_metrics.json` and `per_origin_metrics.csv`. The angle QNN had fold-specific class collapse: fold 1 predicted no Survivor; fold 2 never predicted OPO. Its training loss barely moved in fold 1 (1.083 to 1.078) and improved more in folds 2-3, but validation macro F1 remained poor. The trainable readout lifted pooled macro F1 only to 0.3218; the amplitude encoding variant reached 0.3086. No model showed stable three-class separation.", "", "### Paired origin uncertainty", "", "The table below reports candidate minus control macro F1, with percentile intervals from 2,000 paired bootstrap resamples of the 12 validation origins. The small, class-balanced rare-origin sample means these intervals describe only this exploratory sample.", "", "| QNN | Control | Delta (pp) | 95% origin-bootstrap interval (pp) |", "|---|---|---:|---:|"]
    for row in comparisons:
        ci = row["origin_bootstrap_95_ci"]
        lines.append(f"| {row['qnn']} | {row['control']} | {row['delta_macro_f1']*100:+.2f} | [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] |")
    lines += ["", "Every QNN point estimate is below every matched classical control; paired origin intervals and variation are in `paired_origin_bootstrap.csv`. No observed improvement is statistically convincing.", "", "## Encoding and circuit costs", "", "| Variant | Encoding/readout | Qubits | Ansatz depth | Trainable params | Notes |", "|---|---|---:|---:|---:|---|"]
    for row in config_rows:
        lines.append(f"| {row['variant']} | {row['encoding_description']} / {row['readout_description']} | {row['logical_qubits']} | {row['compiled_circuit_depth']} ({row['ansatz_layers']} layers) | {row['trainable_parameters']} | ideal analytic statevector |")
    lines += ["", "Angle re-uploading uses four feature wires, two trainable layers, 24 parameters, and repeats RY(pi*x) before each layer. PennyLane reports compiled circuit depth 12 (18 logical operations in the high-level resource listing). The amplitude variant uses two wires to encode four selected values as a normalized state, 12 parameters; a decomposed circuit has depth 8 and 11 gates including a three-RY/two-CNOT state preparation, so preparation is not treated as free. Normalization discards overall input norm. The trainable-readout angle variant adds a 3x3 affine head and bias (36 total parameters). Circuit execution counts and per-fold runtime/RSS snapshots are in `all_fold_metrics.csv` and `resource_log.json`.", "", "Binary project champion context: on the same 205 sampled record IDs, saved five-fold OOF binary predictions can be compared after collapsing WOO/OPO. Champion binary macro F1 was 0.923 on this unusually balanced, OPO-origin subset; three-class QNN predictions collapsed to binary scores between 0.431 and 0.501. This is context only: target mapping, 23 features, fold scheme and training size differ. It is not a matched three-class baseline and does not enter the leaderboard; same-record binary metrics are in `consolidated_summary.json`.", "", "## Rejected approaches and remaining limits", "", "- Basis encoding was not applied because the four fold-selected continuous/dummy values would require discretization and lose numerical information; it would be a distinct encoding hypothesis with its own matched controls.", "- Three-class quantum-kernel QSVC and fixed quantum-reservoir readout were not rerun: the available implementations/pilots target the binary label, were weak against matched controls there, and the primary three-class QNN families remained well below controls. No evidence justified extra circuit runs under the present resource limit.", "- The input sample is class-balanced by design and includes 12 of 48 OPO-bearing origins. It is not representative of the 305-origin deployment population. This was the largest safely justified QNN comparison after preflight found about 1.1 GiB free RAM (the angle run began with about 800 MiB available) and an earlier 24-origin expansion stopped below its 450 MiB floor.", "- Three encoding/readout QNN variants failed to approach even the strongest matched four-feature classical control, while each QNN took about 68-141 seconds and classical fits took under one second. Scaling to all development rows or adding further architecture/optimizer sweeps is not justified by this evidence.", "- A predeclared confirmation fit using the 12-origin training sample and all 36 remaining OPO-bearing development origins was attempted but stopped when available RAM crossed below the 500 MiB floor; it emitted no predictions or metrics. No retry was made. No 85% target was optimized, no frozen test was accessed, and no production model, source data, split manifest, historical report, anomaly audit, or simulator was modified.", "- A stronger generalization study needs a larger safe-memory environment, a predeclared origin-representative sample, coverage of non-OPO origins, multiple seeds, and held-out origins untouched during architecture selection. Those 36 origins remain unmeasured by a completed candidate.", "", "## Artifacts and verification", "", "Campaign outputs are stored under this directory, including the source experiment runner, separate predictions/configuration/fold manifests, three training-curve sets, per-origin metrics, bootstrap comparisons, resource log, foundations report, persistent ledger, progress manifest, and model prediction artifacts. No real-data model parameter checkpoint was saved because the selected-candidate confirmation fit hit the RAM floor before completion.", "", "The focused suite had an earlier 4-test QNN/foundations pass. The final broader rerun passed 10 tests and deselected 1 full-development adapter test; 2 QNN fit tests were stopped by the explicit 500 MiB RAM guard. The campaign runner and analysis script passed Python syntax checks; JSON and prediction/group integrity checks are recorded in machine-readable reports. Frozen-test access is false in every new campaign manifest.", ""]
    (OUT / "research_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
