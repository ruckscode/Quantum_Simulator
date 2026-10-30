"""Run the frozen-design development-origin QNN confirmation and matched controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LABELS = [0, 1, 2]
MIN_RAM_MIB = 500


def metrics(y, pred):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
        f1_score, precision_recall_fscore_support)
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    return {"macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "per_class": {str(k): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                               "support": int(support[i])} for i, k in enumerate(LABELS)},
        "confusion_matrix_labels_0_1_2": confusion_matrix(y, pred, labels=LABELS).tolist()}


def _sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def paired_origin_bootstrap(frame, models, seed=20261129, repetitions=2000):
    import numpy as np
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    origins = np.asarray(sorted(frame.origin_program.astype(str).unique()), dtype=object)
    if len(origins) < 2:
        raise ValueError("At least two confirmation origins are required for uncertainty estimates")
    rows = []
    for comparator in models:
        if comparator == "three_class_vqc_angle_trainable_readout":
            continue
        deltas = []
        for _ in range(repetitions):
            sampled = rng.choice(origins, size=len(origins), replace=True)
            true_parts, qnn_parts, other_parts = [], [], []
            for origin in sampled:
                group = frame[frame.origin_program.astype(str) == origin]
                true_parts.append(group.y_true.to_numpy())
                qnn_parts.append(group.loc[group.model == "three_class_vqc_angle_trainable_readout", "y_pred"].to_numpy())
                other_parts.append(group.loc[group.model == comparator, "y_pred"].to_numpy())
            y = np.concatenate(true_parts)
            qpred, cpred = np.concatenate(qnn_parts), np.concatenate(other_parts)
            deltas.append(float(f1_score(y, qpred, labels=LABELS, average="macro", zero_division=0)
                                - f1_score(y, cpred, labels=LABELS, average="macro", zero_division=0)))
        rows.append({"comparison": f"QNN - {comparator}", "bootstrap_seed": seed,
            "repetitions": repetitions, "origin_count": len(origins),
            "mean_delta_macro_f1": float(np.mean(deltas)),
            "ci_2_5": float(np.quantile(deltas, .025)), "ci_97_5": float(np.quantile(deltas, .975)),
            "probability_delta_positive": float(np.mean(np.asarray(deltas) > 0))})
    return rows


def run(data_dir: Path, output: Path):
    import psutil
    data_dir, output = data_dir.resolve(), output.resolve()
    available_start = int(psutil.virtual_memory().available)
    if available_start / 2**20 < MIN_RAM_MIB + 200:
        raise MemoryError("At least 700 MiB available RAM is required before starting the confirmation")
    rss_start = int(psutil.Process().memory_info().rss)
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from qmutbench_qml.features.reduction import ReducedFeatureEncoder
    from qmutbench_qml.models.multiclass_vqc import ThreeClassVQC
    if psutil.virtual_memory().available / 2**20 < MIN_RAM_MIB + 200:
        raise MemoryError("Post-import headroom is below 700 MiB; no confirmation fit was started")
    process = psutil.Process()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((data_dir / "data_manifest.json").read_text(encoding="utf-8"))
    train = pd.read_csv(data_dir / "train.csv")
    valid = pd.read_csv(data_dir / "validation.csv")
    if _sha(data_dir / "train.csv") != manifest["train_sha256"] or _sha(data_dir / "validation.csv") != manifest["validation_sha256"]:
        raise ValueError("Input data checksum does not match the export manifest")
    if not manifest.get("development_only") or manifest.get("frozen_test_rows") != 0:
        raise ValueError("Input manifest does not certify development-only data")
    if train.record_id.duplicated().any() or valid.record_id.duplicated().any():
        raise ValueError("Duplicate record IDs")
    if set(train.origin_program.astype(str)) != set(manifest["training_origins"]):
        raise ValueError("Training origins differ from the predeclared manifest")
    if set(valid.origin_program.astype(str)) != set(manifest["validation_origins"]):
        raise ValueError("Validation origins differ from the predeclared manifest")
    if set(train.origin_program.astype(str)) & set(valid.origin_program.astype(str)):
        raise ValueError("Group leakage between fit and confirmation origins")
    if set(train.y.astype(int)) != set(LABELS) or set(valid.y.astype(int)) != set(LABELS):
        raise ValueError("Both splits must contain the predeclared three classes")
    features = manifest["features"]
    if any(name not in train or name not in valid for name in features):
        raise ValueError("Exported feature schema is incomplete")

    device = os.environ.get("QMB_DEVICE", "default.qubit")
    diff_method = "adjoint" if device == "lightning.gpu" else "backprop"
    smoke = subprocess.run([sys.executable, str(Path(__file__).with_name("backend_smoke.py"))],
        capture_output=True, text=True, env={**os.environ, "QMB_DEVICE": device}, check=False)
    (output / "backend_smoke_stdout.json").write_text(smoke.stdout, encoding="utf-8")
    (output / "backend_smoke_stderr.log").write_text(smoke.stderr, encoding="utf-8")
    if smoke.returncode != 0:
        raise RuntimeError(f"Requested backend {device!r} failed smoke validation; see backend_smoke_stderr.log")
    smoke_result = json.loads(smoke.stdout)
    encoder = ReducedFeatureEncoder(n_features=4, seed=20261127).fit(train[features], train.y.astype(int))
    x_train, x_valid = encoder.transform(train[features]), encoder.transform(valid[features])
    y_train, y_valid = train.y.to_numpy(dtype=int), valid.y.to_numpy(dtype=int)
    models = [
        ("three_class_vqc_angle_trainable_readout", ThreeClassVQC(steps=10, learning_rate=.04,
            seed=20261128, data_reuploading=True, trainable_readout=True,
            device_name=device, diff_method=diff_method)),
        ("logistic_4f", LogisticRegression(C=1.0, class_weight="balanced", max_iter=500,
            solver="lbfgs", random_state=20261128)),
        ("rbf_svc_4f", SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True,
            random_state=20261128)),
        ("extra_trees_4f", ExtraTreesClassifier(n_estimators=100, max_depth=12,
            min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=1,
            random_state=20261128)),
    ]
    result_rows, prediction_rows = [], []
    start_all = time.perf_counter()
    for name, model in models:
        if psutil.virtual_memory().available / 2**20 < MIN_RAM_MIB:
            raise MemoryError("Available RAM fell below 500 MiB during fitting")
        started = time.perf_counter()
        if name.startswith("three_class_vqc"):
            model.fit(x_train, y_train, gradient_batch_size=24)
            parts = []
            inference_seconds = 0.0
            inference_evaluations = 0
            for i in range(0, len(x_valid), 24):
                parts.append(model.predict_proba(x_valid[i:i + 24]))
                inference_seconds += model.inference_seconds
                inference_evaluations += model.inference_executions
            prob = np.vstack(parts)
            qmeta = {"device": device, "qubits": model.n_qubits, "ansatz_depth": 12,
                "trainable_parameters": model.parameter_count,
                "training_circuit_evaluations": model.training_executions,
                "validation_circuit_evaluations": inference_evaluations,
                "training_seconds": model.training_seconds,
                "inference_seconds": float(inference_seconds)}
            np.savez_compressed(output / "selected_qnn_parameters.npz",
                circuit_weights=np.asarray(model.weights), readout_weights=np.asarray(model.readout_weights),
                readout_bias=np.asarray(model.readout_bias), initial_circuit_weights=model.initial_weights,
                initial_readout_weights=model.initial_readout,
                selected_feature_names=np.asarray(encoder.selected_names), seed=np.asarray(20261128),
                device=np.asarray(device))
            qmeta["training_loss"] = model.loss_history
        else:
            model.fit(x_train, y_train)
            prob = model.predict_proba(x_valid)
            qmeta = {"device": None, "qubits": None, "ansatz_depth": None,
                "trainable_parameters": None, "training_circuit_evaluations": 0,
                "validation_circuit_evaluations": 0, "training_seconds": time.perf_counter() - started,
                "inference_seconds": 0.0}
        pred = prob.argmax(axis=1)
        result_rows.append({"model": name, "n_train": len(train), "n_validation": len(valid),
            "training_origins": train.origin_program.nunique(), "validation_origins": valid.origin_program.nunique(),
            **qmeta, **metrics(y_valid, pred)})
        for i, row in valid.reset_index(drop=True).iterrows():
            prediction_rows.append({"record_id": int(row.record_id), "origin_program": str(row.origin_program),
                "y_true": int(y_valid[i]), "model": name, "y_pred": int(pred[i]),
                "prob_0": float(prob[i, 0]), "prob_1": float(prob[i, 1]), "prob_2": float(prob[i, 2])})
    pd.DataFrame(result_rows).to_json(output / "confirmation_metrics.json", orient="records", indent=2)
    pd.DataFrame(prediction_rows).to_csv(output / "confirmation_predictions.csv", index=False)
    predictions = pd.DataFrame(prediction_rows)
    bootstrap = paired_origin_bootstrap(predictions, [row["model"] for row in result_rows])
    (output / "paired_origin_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    pd.DataFrame({"record_id": train.record_id, "origin_program": train.origin_program,
                  "role": "train"}).to_csv(output / "fit_manifest.csv", index=False)
    pd.DataFrame({"record_id": valid.record_id, "origin_program": valid.origin_program,
                  "role": "confirmation"}).to_csv(output / "confirmation_manifest.csv", index=False)
    summary = {"status": "completed_single_predeclared_development_origin_confirmation",
        "frozen_test_accessed": False, "data_manifest_sha256": _sha(data_dir / "data_manifest.json"),
        "selected_features": encoder.selected_names, "selected_device": device,
        "backend_smoke": smoke_result, "available_ram_before_bytes": available_start,
        "available_ram_after_bytes": int(psutil.virtual_memory().available),
        "process_rss_before_bytes": rss_start, "process_rss_after_bytes": int(process.memory_info().rss),
        "elapsed_seconds": time.perf_counter() - start_all, "metrics": result_rows,
        "selection_warning": "Do not use confirmation scores for model/hyperparameter selection."}
    (output / "confirmation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.output), indent=2))
