"""Validate and summarize the saved QMutBench QML pilots without training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
QML = ROOT / "qmutbench_qml"
RESULTS = QML / "results"
OUT = ROOT / "reports/experiments/qmutbench_qml_advantage_campaign_20261027"
OOF = ROOT / "data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/mutation_diff_oof.npz"
CHAMPION_SHA = "7532be26b6959e7e62277220f0c0b78e648d723a0be35d2b693d68e1c9ebf085"
TARGET_MAP = {"WOO": 1, "OPO": 1, "Survivor": 0}
QML_NAMES = ["vqc", "qsvc", "data_reupload", "quantum_reservoir"]
CLASSICAL_NAMES = ["logistic_4f", "rbf_svc_4f", "extra_trees_4f"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics(y, pred):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {"rows": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
            "confusion_matrix_labels_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "per_class": {str(c): {"precision": float(p[i]), "recall": float(r[i]),
                                   "f1": float(f[i]), "support": int(s[i])}
                          for i, c in enumerate([0, 1])}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pilot_manifest = json.loads((RESULTS / "run_manifest.json").read_text(encoding="utf-8"))
    family_dir = RESULTS / "family_extension_9origins_20261027"
    family_manifest = json.loads((family_dir / "run_manifest.json").read_text(encoding="utf-8"))
    pilot_summary = json.loads((RESULTS / "experiment_summary.json").read_text(encoding="utf-8"))
    champion_metadata = json.loads((ROOT / "data/qmutbench_ml/mutation_aware_v1/best_model_metadata.json").read_text(encoding="utf-8"))
    if pilot_manifest["data_audit"]["target_map"] != TARGET_MAP or champion_metadata["target_map"] != TARGET_MAP:
        raise ValueError("QML and champion target maps disagree")
    if family_manifest.get("frozen_test_labels_used") or family_manifest.get("frozen_test_evaluation"):
        raise ValueError("Existing QML family pilot accessed frozen-test labels or evaluation")
    if family_manifest.get("status") != "completed_on_shared_pilot_sample":
        raise ValueError("Existing family extension is not marked complete")
    if champion_metadata["model_sha256"] != CHAMPION_SHA:
        raise ValueError("The champion model bundle differs from the audited reference")

    # Check the saved pilot sample against the authoritative development-only grouped OOF records.
    oof = np.load(OOF, allow_pickle=True)
    dev = pd.DataFrame({"record_id": oof["record_id"].astype(np.int64),
                        "origin_program": oof["origin_program"].astype(str),
                        "y_true": oof["y"].astype(np.int8)})
    if len(dev) != 578117 or dev.record_id.duplicated().any() or dev.origin_program.nunique() != 305:
        raise ValueError("Authoritative development OOF record manifest failed schema checks")
    dev_lookup = dev.set_index("record_id")
    split = pd.read_csv(family_dir / "split_manifest.csv")
    pred = pd.read_csv(family_dir / "predictions.csv")
    if split.record_id.nunique() != 72 or split.origin_program.nunique() != 9:
        raise ValueError("Existing QML pilot does not match expected 72-record/9-origin sample")
    if set(pred.model.unique()) != set(QML_NAMES + CLASSICAL_NAMES):
        raise ValueError("Saved family results have missing or unexpected model families")
    validation = split[split.role == "validation"]
    if validation.record_id.nunique() != 72 or validation.groupby("record_id").size().max() != 1:
        raise ValueError("Validation records are duplicated or incomplete")
    for fold, block in split.groupby("fold"):
        train_origins = set(block.loc[block.role == "train", "origin_program"].astype(str))
        valid_origins = set(block.loc[block.role == "validation", "origin_program"].astype(str))
        if train_origins & valid_origins:
            raise ValueError(f"Origin overlap in saved QML fold {fold}")
    valids = validation[["record_id", "origin_program", "fold"]].copy()
    for name, part in pred.groupby("model"):
        if len(part) != 72 or part.record_id.nunique() != 72:
            raise ValueError(f"Saved predictions for {name} do not cover 72 unique validation IDs")
        aligned = part.merge(valids, on=["record_id", "origin_program", "fold"], how="outer", indicator=True)
        if not (aligned._merge == "both").all():
            raise ValueError(f"Saved predictions for {name} do not match exact held-out records/folds")
        auth = dev_lookup.reindex(part.record_id.astype(np.int64))
        if auth.isna().any().any() or not np.array_equal(auth.origin_program.to_numpy(str), part.origin_program.to_numpy(str)):
            raise ValueError(f"Saved {name} record IDs/groups are not development-manifest aligned")
        if not np.array_equal(auth.y_true.to_numpy(np.int8), part.y_true.to_numpy(np.int8)):
            raise ValueError(f"Saved {name} labels disagree with authoritative development OOF labels")
    if pred.groupby("model").y_true.nunique().max() != 2:
        raise ValueError("Pilot validation does not contain both documented binary classes")

    qml_metrics = {}
    for name, part in pred.groupby("model", sort=False):
        qml_metrics[name] = metrics(part.y_true.to_numpy(np.int8), part.y_pred.to_numpy(np.int8))
    per_origin = []
    for (name, origin), part in pred.groupby(["model", "origin_program"], sort=False):
        cm = confusion_matrix(part.y_true, part.y_pred, labels=[0, 1])
        per_origin.append({"model": name, "origin_program": str(origin), "rows": int(len(part)),
                           "support_0": int((part.y_true == 0).sum()),
                           "support_1": int((part.y_true == 1).sum()),
                           "accuracy": float(accuracy_score(part.y_true, part.y_pred)),
                           "macro_f1": float(f1_score(part.y_true, part.y_pred, labels=[0, 1], average="macro", zero_division=0)),
                           "f1_class_0": float(f1_score(part.y_true, part.y_pred, labels=[0], average="macro", zero_division=0)),
                           "f1_class_1": float(f1_score(part.y_true, part.y_pred, labels=[1], average="macro", zero_division=0)),
                           "false_positives": int(cm[0, 1]), "false_negatives": int(cm[1, 0])})

    # One fixed equal-weight re-uploading/Logistic blend is evaluated as an
    # exploratory complementarity diagnostic; its weight and threshold are
    # not selected from these validation outcomes.
    reupload = pred[pred.model == "data_reupload"].copy()
    logistic = pred[pred.model == "logistic_4f"].copy()
    pair = reupload.merge(logistic, on=["record_id", "origin_program", "y_true", "fold"],
                          suffixes=("_qml", "_lr"), validate="one_to_one")
    pair["blend_score"] = .5 * (pair.score_qml + pair.score_lr)
    pair["blend_pred"] = (pair.blend_score >= .5).astype(np.int8)
    blend_metrics = metrics(pair.y_true.to_numpy(np.int8), pair.blend_pred.to_numpy(np.int8))
    blend_by_fold = {}
    for fold, block in pair.groupby("fold"):
        blend_by_fold[str(fold)] = metrics(block.y_true.to_numpy(np.int8), block.blend_pred.to_numpy(np.int8))
    origins = sorted(pair.origin_program.unique())
    cm_lr, cm_blend = [], []
    for origin in origins:
        block = pair[pair.origin_program == origin]
        cm_lr.append(confusion_matrix(block.y_true, block.y_pred_lr, labels=[0, 1]))
        cm_blend.append(confusion_matrix(block.y_true, block.blend_pred, labels=[0, 1]))
    cm_lr, cm_blend = np.stack(cm_lr), np.stack(cm_blend)
    rng = np.random.default_rng(20261027)
    boot = []
    for _ in range(2000):
        sampled = rng.integers(0, len(origins), len(origins))
        a, b = cm_lr[sampled].sum(axis=0), cm_blend[sampled].sum(axis=0)
        def macro_from_cm(cm):
            tn, fp, fn, tp = cm.ravel()
            f0 = 2 * tn / max(1, 2 * tn + fp + fn)
            f1 = 2 * tp / max(1, 2 * tp + fn + fp)
            return .5 * (f0 + f1)
        boot.append(macro_from_cm(b) - macro_from_cm(a))
    blend_ci = [float(np.percentile(boot, 2.5) * 100), float(np.percentile(boot, 97.5) * 100)]
    pair[["record_id", "origin_program", "fold", "y_true", "score_qml", "score_lr",
          "blend_score", "blend_pred"]].to_csv(OUT / "exploratory_reupload_logistic_blend.csv.gz",
                                                  index=False, compression="gzip")
    blend_origin_rows = []
    for origin, block in pair.groupby("origin_program", sort=False):
        cm = confusion_matrix(block.y_true, block.blend_pred, labels=[0, 1])
        blend_origin_rows.append({"model": "data_reupload_50_50_logistic_blend", "origin_program": str(origin),
                                  "rows": int(len(block)), "support_0": int((block.y_true == 0).sum()),
                                  "support_1": int((block.y_true == 1).sum()),
                                  "accuracy": float(accuracy_score(block.y_true, block.blend_pred)),
                                  "macro_f1": float(f1_score(block.y_true, block.blend_pred, labels=[0, 1], average="macro", zero_division=0)),
                                  "f1_class_0": float(f1_score(block.y_true, block.blend_pred, labels=[0], average="macro", zero_division=0)),
                                  "f1_class_1": float(f1_score(block.y_true, block.blend_pred, labels=[1], average="macro", zero_division=0)),
                                  "false_positives": int(cm[0, 1]), "false_negatives": int(cm[1, 0])})
    pd.DataFrame([*per_origin, *blend_origin_rows]).to_csv(OUT / "per_origin_metrics.csv", index=False)
    # Use the same 72 IDs for historical full-feature champion context, while
    # explicitly not treating its larger training set/five-fold allocation as matched.
    champ_prob = pd.Series(oof["probability"], index=oof["record_id"].astype(np.int64))
    pilot_ids = validation.record_id.astype(np.int64).to_numpy()
    auth_y = dev_lookup.reindex(pilot_ids).y_true.to_numpy(np.int8)
    full_champion_context = metrics(auth_y, (champ_prob.reindex(pilot_ids).to_numpy() >= .5).astype(np.int8))

    overlap = []
    wide = pred.pivot(index="record_id", columns="model", values="y_pred")
    truth = pred.drop_duplicates("record_id").set_index("record_id").y_true
    for qname in QML_NAMES:
        for cname in CLASSICAL_NAMES:
            qcorrect = wide[qname].to_numpy() == truth.reindex(wide.index).to_numpy()
            ccorrect = wide[cname].to_numpy() == truth.reindex(wide.index).to_numpy()
            overlap.append({"qml_model": qname, "classical_model": cname,
                            "records": len(wide), "both_correct": int((qcorrect & ccorrect).sum()),
                            "qml_only_correct": int((qcorrect & ~ccorrect).sum()),
                            "classical_only_correct": int((~qcorrect & ccorrect).sum()),
                            "both_wrong": int((~qcorrect & ~ccorrect).sum()),
                            "prediction_agreement": float((wide[qname].to_numpy() == wide[cname].to_numpy()).mean())})
    pd.DataFrame(overlap).to_csv(OUT / "matched_error_overlap.csv", index=False)

    paired = pd.read_csv(family_dir / "paired_differences.csv")
    best_matched = {}
    for qname in QML_NAMES:
        metric = qml_metrics[qname]["macro_f1"]
        controls = {c: qml_metrics[c]["macro_f1"] for c in CLASSICAL_NAMES}
        strongest = max(controls, key=controls.get)
        ci = paired[(paired.qml_model == qname) & (paired.matched_classical_model == strongest) &
                    (paired.metric == "macro_f1")]
        best_matched[qname] = {"strongest_matched_classical": strongest,
                               "strongest_classical_macro_f1": controls[strongest],
                               "qml_macro_f1": metric,
                               "difference_percentage_points": 100 * (metric - controls[strongest]),
                               "paired_origin_bootstrap_95ci_percentage_points":
                                   [float(ci.iloc[0].ci95_low_percentage_points),
                                    float(ci.iloc[0].ci95_high_percentage_points)] if len(ci) else None}

    # Read existing family-level timing/circuit measures and preserve all source artifact hashes.
    source_files = [RESULTS / "run_manifest.json", RESULTS / "predictions.csv", RESULTS / "split_manifest.csv",
                    family_dir / "run_manifest.json", family_dir / "predictions.csv",
                    family_dir / "split_manifest.csv", family_dir / "paired_differences.csv"]
    model_sources = [QML / "models/vqc.py", QML / "models/qsvc.py",
                     QML / "models/quantum_reservoir.py", QML / "features/reduction.py",
                     QML / "training/pilot.py", QML / "training/run_family_extension.py",
                     QML / "requirements-lock.txt"]
    output = {
        "schema": "qmutbench-qml-campaign-audit-v1", "status": "saved_pilots_validated_no_new_training",
        "task": {"target_map": TARGET_MAP, "task_type": "documented binary mutation outcome",
                 "development_rows": len(dev), "development_origins": int(dev.origin_program.nunique()),
                 "pilot_rows": int(validation.record_id.nunique()), "pilot_origins": int(validation.origin_program.nunique()),
                 "pilot_labels": {str(k): int(v) for k,v in pred[pred.model == "vqc"].drop_duplicates("record_id").y_true.value_counts().sort_index().items()}},
        "integrity": {"record_ids_unique_in_development": True, "all_models_same_validation_ids_origins_folds": True,
                      "group_overlap_per_fold": 0, "label_alignment_verified": True,
                      "target_map_matches_classical_champion": True, "frozen_test_accessed": False,
                      "model_bundle_sha256": champion_metadata["model_sha256"],
                      "files": {str(p.relative_to(ROOT)): sha(p) for p in source_files}},
        "pilot_metrics": qml_metrics,
        "per_origin_metrics_path": "per_origin_metrics.csv",
        "configuration": {
            "pilot": pilot_manifest["config"],
            "family_extension": {"models": ["data_reupload", "quantum_reservoir"],
                                  "n_qubits": 4, "depth": 2, "vqc_steps": 20,
                                  "seed_rule": "20261026 + fold number",
                                  "source_of_reused_predictions": "completed nine-origin pilot predictions; no retraining of VQC, QSVC, or controls"},
            "selected_fold_inputs": json.loads((family_dir / "feature_selection_by_fold.json").read_text(encoding="utf-8")),
        },
        "implementation_source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in model_sources},
        "strongest_matched_classical_comparisons": best_matched,
        "exploratory_fixed_blend": {
            "hypothesis": "A fixed 50:50 mean of re-uploading and matched Logistic probabilities may exploit their comparable but nonidentical errors.",
            "weight_selection": "fixed 0.5, not fitted or tuned; threshold fixed at 0.5",
            "blend_metrics": blend_metrics,
            "matched_logistic_metrics": qml_metrics["logistic_4f"],
            "macro_f1_difference_percentage_points": 100 * (blend_metrics["macro_f1"] - qml_metrics["logistic_4f"]["macro_f1"]),
            "paired_origin_bootstrap_95ci_macro_f1_delta_percentage_points": blend_ci,
            "bootstrap_repetitions": 2000, "bootstrap_seed": 20261027,
            "fold_metrics": blend_by_fold,
            "status": "exploratory_same_pilot_sample_not_independent_confirmation",
        },
        "full_feature_champion_same_records_context_only": {
            **full_champion_context,
            "training_protocol_warning": "These predictions cover the same 72 record IDs but use the saved five-fold model trained on a much larger set and a different fold assignment; descriptive context only, not a matched training-data comparison."},
        "model_families": {
            "vqc_trainable_qnn": "same trainable VQC family; do not count twice",
            "qsvc": "quantum kernel classifier",
            "data_reupload": "distinct re-encoding trainable circuit architecture",
            "quantum_reservoir": "fixed quantum feature map plus classical Logistic Regression readout",
            "excluded": {"Torch/Keras QNN wrappers": "Torch and TensorFlow absent from isolated lock; duplicates the VQC/QNN family rather than a distinct method",
                         "hardware/finite-shot": "only local analytic simulator present; no authorized hardware or credentials",
                         "quantum convolutional/deeper statevector models": "new architectures with no evidence supporting cost under current resource limit",
                         "hybrid stacking": "not evaluated; tiny pilot, no QML win over best matched control, and no nested grouped predictions to fit a safe combiner"}},
        "runtime_circuit_costs": family_manifest["cost_summary"],
        "runtime": family_manifest["runtime"], "memory": family_manifest["resource"],
        "incomplete": {"24_origin_attempt": family_manifest.get("larger_24_origin_attempt"),
                       "new_full_five_fold_qml": "not run: poor matched pilot, previous 24-origin memory stop, and current available RAM below 1 GiB"},
        "classical_champion": {"macro_f1": 0.7933135654357681, "accuracy": 0.7933290320125511,
                               "balanced_accuracy": 0.7940874828150084,
                               "comparison_note": "QML pilot uses 72 records/9 origins and four fold-selected features; do not directly rank against champion's full 578,117-record/305-origin OOF."},
    }
    (OUT / "integrity_and_metrics.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "pilot_metrics": {k: v["macro_f1"] for k,v in qml_metrics.items()},
                      "strongest_matched": best_matched,
                      "full_champion_same_records_context": full_champion_context}, indent=2))


if __name__ == "__main__":
    main()
