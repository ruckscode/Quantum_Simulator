"""One-shot untouched-development-origin check of the selected QNN hybrid."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.svm import SVC

from qmutbench_qml.data.loader import FEATURES, load_development_records
from qmutbench_qml.features.reduction import ReducedFeatureEncoder
from qmutbench_qml.models.multiclass_vqc import ThreeClassVQC
from qmutbench_qml.training.run_three_class_campaign import (
    LABEL_MAP, RAM_STOP_MIB, SEED, _sample_manifest, metrics,
)


def run(root: Path, output: Path):
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    available_before = psutil.virtual_memory().available
    if available_before / 2**20 < RAM_STOP_MIB + 200:
        raise MemoryError("Insufficient pre-run RAM headroom")

    train_origins, train_ids, train_manifest, audit = _sample_manifest(root)
    reserved = pd.DataFrame(audit["reserved_confirmation_sample"])
    if not audit["reserved_confirmation_origins"] or reserved.empty:
        raise ValueError("No reserved OPO-bearing origins remain for confirmation")
    if set(train_origins) & set(audit["reserved_confirmation_origins"]):
        raise AssertionError("Confirmation origins leaked into exploratory training origins")
    valid_ids = reserved.record_id.astype(int).tolist()
    all_ids = train_ids + valid_ids
    all_origins = train_origins + audit["reserved_confirmation_origins"]
    combined, combined_audit = load_development_records(root, origins=all_origins, record_ids=all_ids)
    combined["y3"] = combined.Killed.astype(str).map(LABEL_MAP).astype(int)
    train_set = set(train_ids)
    train = combined[combined.record_id.isin(train_set)].copy().reset_index(drop=True)
    valid = combined[~combined.record_id.isin(train_set)].copy().reset_index(drop=True)
    if set(train.y3) != {0, 1, 2} or set(valid.y3) != {0, 1, 2}:
        raise AssertionError("Confirmation split is missing a target class")
    if set(train.origin_program.astype(str)) & set(valid.origin_program.astype(str)):
        raise AssertionError("Group leakage between QNN fit and confirmation origins")
    expected = reserved.set_index("record_id")
    if not np.array_equal(valid.record_id.map(expected.y_true).to_numpy(), valid.y3.to_numpy()):
        raise AssertionError("Confirmation labels are misaligned")

    encoder = ReducedFeatureEncoder(n_features=4, seed=SEED + 100).fit(train[FEATURES], train.y3)
    x_train, x_valid = encoder.transform(train[FEATURES]), encoder.transform(valid[FEATURES])
    y_train, y_valid = train.y3.to_numpy(), valid.y3.to_numpy()
    models = [
        ("three_class_vqc_angle_linear", ThreeClassVQC(steps=10, learning_rate=.04,
            seed=SEED + 101, data_reuploading=True, trainable_readout=True)),
        ("logistic_4f", LogisticRegression(C=1.0, class_weight="balanced", max_iter=500,
                                             solver="lbfgs", random_state=SEED + 101)),
        ("rbf_svc_4f", SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True,
                            random_state=SEED + 101)),
        ("extra_trees_4f", ExtraTreesClassifier(n_estimators=100, max_depth=12,
            min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=1,
            random_state=SEED + 101)),
    ]
    results, prediction_rows = [], []
    started = time.perf_counter()
    qnn_checkpoint = None
    for name, model in models:
        if psutil.virtual_memory().available / 2**20 < RAM_STOP_MIB:
            raise MemoryError("Available RAM fell below 500 MiB before confirmation fit")
        t0 = time.perf_counter()
        if name.startswith("three_class_vqc"):
            model.fit(x_train, y_train, gradient_batch_size=24)
            validation_probability_parts = []
            inference_seconds = 0.0
            inference_executions = 0
            for start in range(0, len(x_valid), 24):
                validation_probability_parts.append(model.predict_proba(x_valid[start:start + 24]))
                inference_seconds += model.inference_seconds
                inference_executions += model.inference_executions
            probs = np.vstack(validation_probability_parts)
            pred = probs.argmax(axis=1)
            fit_seconds = model.training_seconds
            qmeta = {"qubits": model.n_qubits, "compiled_ansatz_depth": 12,
                "ansatz_layers": model.depth, "trainable_parameters": model.parameter_count,
                "train_circuit_executions": model.training_executions,
                "validation_circuit_executions": inference_executions}
            qnn_checkpoint = output / "selected_qnn_parameters.npz"
            np.savez_compressed(qnn_checkpoint, circuit_weights=np.asarray(model.weights),
                readout_weights=np.asarray(model.readout_weights), readout_bias=np.asarray(model.readout_bias),
                initial_circuit_weights=model.initial_weights, initial_readout_weights=model.initial_readout,
                selected_feature_names=np.asarray(encoder.selected_names),
                train_record_ids=train.record_id.to_numpy(dtype=np.int64),
                train_origin_ids=np.asarray(sorted(train.origin_program.astype(str).unique())),
                validation_record_ids=valid.record_id.to_numpy(dtype=np.int64),
                validation_origin_ids=np.asarray(sorted(valid.origin_program.astype(str).unique())),
                y_mapping=np.asarray(["Survivor=0", "WOO=1", "OPO=2"]), seed=np.asarray(SEED + 101),
                configuration=np.asarray(json.dumps({"qubits": 4, "layers": 2, "steps": 10,
                    "learning_rate": .04, "encoding": "four-fold-local MI selected features; RY(pi*x) re-uploaded",
                    "readout": "trainable affine expectation readout + softmax",
                    "simulator": "PennyLane default.qubit analytic statevector"})))
        else:
            model.fit(x_train, y_train)
            probs = model.predict_proba(x_valid)
            pred = probs.argmax(axis=1)
            fit_seconds, inference_seconds = time.perf_counter() - t0, 0.0
            qmeta = {"qubits": None, "compiled_ansatz_depth": None, "ansatz_layers": None,
                "trainable_parameters": None, "train_circuit_executions": 0,
                "validation_circuit_executions": 0}
        metric = metrics(y_valid, pred)
        results.append({"model": name, "fit_seconds": fit_seconds, "inference_seconds": inference_seconds,
            "training_records": len(train), "validation_records": len(valid),
            "training_origins": train.origin_program.nunique(),
            "validation_origins": valid.origin_program.nunique(), **qmeta, **metric})
        for i, row in valid.iterrows():
            prediction_rows.append({"record_id": int(row.record_id), "origin_program": str(row.origin_program),
                "y_true": int(y_valid[i]), "model": name, "y_pred": int(pred[i]),
                "prob_0": float(probs[i, 0]), "prob_1": float(probs[i, 1]), "prob_2": float(probs[i, 2])})
    pd.DataFrame(results).to_json(output / "confirmation_metrics.json", orient="records", indent=2)
    pd.DataFrame(prediction_rows).to_csv(output / "confirmation_predictions.csv", index=False)
    pd.DataFrame([{**row, "role": "train", "split": "reserved_36_origin_confirmation"}
                  for row in train_manifest.to_dict("records")] +
                 [{**row, "role": "validation", "split": "reserved_36_origin_confirmation"}
                  for row in reserved.to_dict("records")]).to_csv(output / "confirmation_split_manifest.csv", index=False)
    report = {"status": "completed_single_confirmation_no_tuning", "selection_protocol":
        "Trainable-readout angle QNN selected as highest QNN macro F1 in the preceding 12-origin exploration; model design frozen before using the other 36 OPO-bearing development origins.",
        "frozen_test_accessed": False, "reserved_origins": audit["reserved_confirmation_origins"],
        "sample_label_counts": {str(k): int(v) for k, v in valid.y3.value_counts().sort_index().items()},
        "training_origins": train_origins, "selected_features": encoder.selected_names,
        "metrics": results, "elapsed_seconds": time.perf_counter() - started,
        "ram_available_before_bytes": available_before,
        "ram_available_after_bytes": psutil.virtual_memory().available,
        "input_selection_fitted_on_training_only": True,
        "qnn_parameter_checkpoint": str(qnn_checkpoint.relative_to(root))}
    (output / "confirmation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    project = Path.cwd()
    destination = project / "reports/experiments/qmutbench_three_class_qnn_campaign_20261027/confirmation"
    result = run(project, destination)
    print(json.dumps({"status": result["status"], "origins": result["reserved_origins"],
        "metrics": result["metrics"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2))
