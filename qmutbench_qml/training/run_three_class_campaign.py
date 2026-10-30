"""Run one resource-bounded, development-only three-class QNN comparison."""
from __future__ import annotations

import hashlib
import json
import time
import csv
from pathlib import Path
import os

import psutil

SEED = 20261027
ORIGINS = 12
PER_CLASS_PER_ORIGIN = 6
LABEL_MAP = {"Survivor": 0, "WOO": 1, "OPO": 2}
MODEL_NAMES = ["three_class_vqc", "logistic_4f", "rbf_svc_4f", "extra_trees_4f"]
FOLD_COUNT = 3
# Keep at least 1 GiB available throughout fitting, with 200 MiB of additional
# headroom before imports, data loading, and simulator initialization begin.
RAM_STOP_MIB = 1024
RAM_START_HEADROOM_MIB = 200


def _load_training_dependencies():
    """Import data/scikit-learn/QML stacks only after the RAM preflight passes."""
    global np, pd, ExtraTreesClassifier, LogisticRegression
    global accuracy_score, balanced_accuracy_score, confusion_matrix
    global f1_score, precision_recall_fscore_support, StratifiedGroupKFold, SVC
    global FEATURES, load_development_records, ReducedFeatureEncoder
    global validate_foundations, ThreeClassVQC

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                 f1_score, precision_recall_fscore_support)
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.svm import SVC

    from qmutbench_qml.data.loader import FEATURES, load_development_records
    from qmutbench_qml.features.reduction import ReducedFeatureEncoder
    from qmutbench_qml.foundations import validate_foundations
    from qmutbench_qml.models.multiclass_vqc import ThreeClassVQC


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _fold_fingerprint(train, valid, adapter_audit, encoding):
    digest = hashlib.sha256()
    metadata = {
        "schema": "three-class-fold-checkpoint-v1", "seed": SEED,
        "fold_count": FOLD_COUNT, "encoding": encoding, "features": FEATURES,
        "labels": LABEL_MAP, "source_sha256": adapter_audit["source_sha256"],
        "qasm_manifest_sha256": adapter_audit["qasm_cache_manifest_sha256"],
        "train": sorted(zip(train.record_id.astype(int), train.y3.astype(int),
                             train.origin_program.astype(str))),
        "valid": sorted(zip(valid.record_id.astype(int), valid.y3.astype(int),
                             valid.origin_program.astype(str))),
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for frame in (train, valid):
        feature_hashes = pd.util.hash_pandas_object(frame[FEATURES], index=False).to_numpy()
        digest.update(feature_hashes.tobytes())
    return digest.hexdigest()


def _save_fold_checkpoint(path: Path, payload: dict, qnn_state: dict) -> None:
    npz_path = path.with_suffix(".npz")
    temporary_npz = npz_path.with_name(npz_path.name + ".tmp")
    with temporary_npz.open("wb") as stream:
        np.savez_compressed(stream, **qnn_state)
    os.replace(temporary_npz, npz_path)
    payload["qnn_state_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    _atomic_json(path, payload)


def _load_fold_checkpoint(path: Path, fingerprint: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != fingerprint:
        raise ValueError(f"Stale fold checkpoint rejected: {path}")
    npz_path = path.with_suffix(".npz")
    if not npz_path.is_file() or hashlib.sha256(npz_path.read_bytes()).hexdigest() != payload.get("qnn_state_sha256"):
        raise ValueError(f"Missing or corrupt QNN state checkpoint: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as state:
        required = {"weights", "readout_weights", "readout_bias", "loss_history", "configuration"}
        if not required <= set(state.files) or any(not np.isfinite(state[key]).all()
                                                   for key in required - {"configuration"}):
            raise ValueError(f"Incomplete or non-finite QNN state checkpoint: {npz_path}")
    return payload


def metrics(y, pred):
    labels = [0, 1, 2]
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=labels, zero_division=0
    )
    return {
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "per_class": {str(label): {"precision": float(precision[i]), "recall": float(recall[i]),
                                    "f1": float(f1[i]), "support": int(support[i])}
                      for i, label in enumerate(labels)},
        "confusion_matrix_labels_0_1_2": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def _sample_manifest(root: Path):
    oof_path = root / "data/qmutbench_ml/master_optimization_v1/oof_predictions.csv"
    raw_path = root / "data/qmutbench_ingestion/raw/official_all_data_20261015.csv"
    oof = pd.read_csv(oof_path, usecols=["record_id", "origin_program"])
    oof["record_id"] = oof.record_id.astype("int64")
    dev_ids = set(oof.record_id.astype(int))
    expected_origins = dict(zip(oof.record_id.astype(int), oof.origin_program.astype(str)))
    source_rows = []
    # Parse the identifier for each source record, but do not read target/group
    # values unless its ID is in the saved development-only OOF manifest.
    with raw_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        id_i = header.index("") if header and header[0] == "" else header.index("Unnamed: 0")
        origin_i, label_i = header.index("Origin_program"), header.index("Killed")
        for row in reader:
            record_id = int(row[id_i])
            if record_id not in dev_ids:
                continue
            origin = row[origin_i]
            if origin != expected_origins[record_id]:
                raise AssertionError(f"Development source group mismatch for record {record_id}")
            source_rows.append({"record_id": record_id, "origin_program": origin, "Killed": row[label_i]})
    labels = pd.DataFrame(source_rows)
    if labels.record_id.duplicated().any() or len(labels) != len(oof):
        raise AssertionError("Development source labels do not align one-to-one with saved OOF IDs")
    if set(labels.Killed.dropna().unique()) != set(LABEL_MAP):
        raise ValueError(f"Unexpected raw target labels: {sorted(labels.Killed.dropna().unique())}")
    counts = labels.Killed.value_counts().to_dict()
    per_origin = labels.groupby(["origin_program", "Killed"]).size().unstack(fill_value=0)
    rare_origins = sorted(per_origin.index[per_origin.get("OPO", pd.Series(0, index=per_origin.index)) > 0].astype(str))
    rng = np.random.default_rng(SEED)
    chosen = sorted(rng.choice(np.asarray(rare_origins, dtype=object), size=ORIGINS, replace=False).tolist())
    chosen_ids = []
    sample_rows = []
    for origin in chosen:
        group = labels.loc[labels.origin_program.astype(str) == origin]
        for label in ("Survivor", "WOO", "OPO"):
            available = group.loc[group.Killed == label, "record_id"].to_numpy(dtype=np.int64)
            take = min(PER_CLASS_PER_ORIGIN, len(available))
            if take:
                ids = np.sort(rng.choice(available, size=take, replace=False))
                chosen_ids.extend(ids.tolist())
                sample_rows.extend({"record_id": int(i), "origin_program": origin, "label": label}
                                   for i in ids)
    if len(set(chosen_ids)) != len(chosen_ids):
        raise AssertionError("Stratified sample duplicated record IDs")
    selected = pd.DataFrame(sample_rows)
    selected["y_true"] = selected.label.map(LABEL_MAP).astype(int)
    if set(selected.y_true) != {0, 1, 2}:
        raise ValueError("Three-class sample lost one or more classes")
    reserved_origins = sorted(set(rare_origins) - set(chosen))
    reserved_ids, reserved_rows = [], []
    for origin in reserved_origins:
        group = labels.loc[labels.origin_program.astype(str) == origin]
        for label in ("Survivor", "WOO", "OPO"):
            available = group.loc[group.Killed == label, "record_id"].to_numpy(dtype=np.int64)
            take = min(PER_CLASS_PER_ORIGIN, len(available))
            if take:
                ids = np.sort(rng.choice(available, size=take, replace=False))
                reserved_ids.extend(ids.tolist())
                reserved_rows.extend({"record_id": int(i), "origin_program": origin, "label": label}
                                     for i in ids)
    reserved = pd.DataFrame(reserved_rows)
    reserved["y_true"] = reserved.label.map(LABEL_MAP).astype(int)
    return chosen, chosen_ids, selected, {"development_label_counts": counts,
        "development_origins_with_OPO": len(rare_origins), "selected_origins": chosen,
        "reserved_confirmation_origins": reserved_origins,
        "reserved_confirmation_sample": reserved.to_dict("records"),
        "sample_label_counts": {str(k): int(v) for k, v in selected.y_true.value_counts().sort_index().items()},
        "sampling": "Seeded within-origin and within-label subsample; up to six records per class per OPO-bearing origin. Exploratory balanced rare-class-origin sample, not prevalence representative."}


def _toy_check():
    """Check parameter update, gradient sensitivity, and learning on a toy task."""
    rng = np.random.default_rng(41)
    centers = np.array([[.12, .2, .25, .15], [.82, .72, .68, .88], [.2, .82, .85, .3]])
    x = np.vstack([np.clip(center + rng.normal(0, .035, (4, 4)), 0, 1) for center in centers])
    y = np.repeat(np.arange(3), 4)
    model = ThreeClassVQC(steps=24, learning_rate=.08, seed=11)
    model.fit(x, y)
    probs = model.predict_proba(x)
    correct = int((probs.argmax(axis=1) == y).sum())
    return {"rows": len(y), "initial_loss": model.loss_history[0], "final_loss": model.loss_history[-1],
            "correct_of_12": correct, "parameters_changed": bool(not np.allclose(model.initial_weights, model.weights)),
            "probabilities_valid": bool(np.isfinite(probs).all() and np.allclose(probs.sum(axis=1), 1.0)),
            "runtime_seconds": model.training_seconds, "circuit_executions": model.training_executions}


def run(root: Path, output: Path, encoding="angle_reupload"):
    root, output = root.resolve(), output.resolve()
    if encoding not in {"angle_reupload", "amplitude", "angle_linear"}:
        raise ValueError("encoding must be angle_reupload, amplitude, or angle_linear")
    checkpoint_dir = output / "fold_checkpoints"
    existing_results = ([p for p in output.iterdir() if p != checkpoint_dir]
                        if output.exists() else [])
    has_fold_checkpoints = (checkpoint_dir.is_dir() and
                            any(checkpoint_dir.glob("fold_*.json")))
    if existing_results and not has_fold_checkpoints:
        raise FileExistsError(f"Refusing to overwrite existing campaign outputs without resumable checkpoints: {output}")
    process = psutil.Process()
    memory_start = psutil.virtual_memory().available
    minimum_start_mib = RAM_STOP_MIB + RAM_START_HEADROOM_MIB
    if memory_start / 2**20 < minimum_start_mib:
        raise MemoryError(
            f"Only {memory_start / 2**20:.0f} MiB available before run; "
            f"need at least {minimum_start_mib} MiB to preserve the {RAM_STOP_MIB} MiB safety floor"
        )

    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    _load_training_dependencies()
    foundation = validate_foundations()
    toy = _toy_check()
    if not toy["parameters_changed"] or not toy["probabilities_valid"] or toy["final_loss"] >= toy["initial_loss"]:
        raise AssertionError("Toy QNN failed parameter, probability, or loss-improvement checks")

    origins, ids, sample_manifest, sample_audit = _sample_manifest(root)
    data, adapter_audit = load_development_records(root, origins=origins, record_ids=ids)
    data["y3"] = data.Killed.astype(str).map(LABEL_MAP).astype(int)
    if not data.record_id.is_unique or set(data.y3.unique()) != {0, 1, 2}:
        raise AssertionError("Loaded development records failed three-class alignment checks")
    expected = sample_manifest.set_index("record_id")
    aligned_origin = data.record_id.map(expected.origin_program)
    aligned_y = data.record_id.map(expected.y_true)
    if aligned_origin.isna().any() or not np.array_equal(
            data.origin_program.astype(str).to_numpy(), aligned_origin.astype(str).to_numpy()):
        raise AssertionError("Loaded origins differ from precomputed class-aware sample manifest")
    if aligned_y.isna().any() or not np.array_equal(data.y3.to_numpy(), aligned_y.to_numpy()):
        raise AssertionError("Loaded source labels differ from the sampled labels")

    cv = StratifiedGroupKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=SEED)
    predictions, fold_rows, curve_rows, split_rows, feature_rows = [], [], [], [], []
    started = time.perf_counter()
    for fold, (train_i, valid_i) in enumerate(cv.split(data, data.y3, data.origin_program), start=1):
        if psutil.virtual_memory().available / 2**20 < RAM_STOP_MIB:
            raise MemoryError(f"Available RAM crossed the {RAM_STOP_MIB} MiB safety floor before fold {fold}")
        train, valid = data.iloc[train_i].reset_index(drop=True), data.iloc[valid_i].reset_index(drop=True)
        train_groups, valid_groups = set(train.origin_program.astype(str)), set(valid.origin_program.astype(str))
        if train_groups & valid_groups or set(train.y3) != {0, 1, 2} or set(valid.y3) != {0, 1, 2}:
            raise AssertionError(f"Fold {fold} lacks class coverage or has group leakage")
        checkpoint_path = checkpoint_dir / f"fold_{fold}.json"
        fingerprint = _fold_fingerprint(train, valid, adapter_audit, encoding)
        if checkpoint_path.exists():
            saved = _load_fold_checkpoint(checkpoint_path, fingerprint)
            predictions.extend(saved["predictions"])
            fold_rows.extend(saved["fold_rows"])
            curve_rows.extend(saved["curve_rows"])
            split_rows.extend(saved["split_rows"])
            feature_rows.extend(saved["feature_rows"])
            print(f"[fold {fold}/{FOLD_COUNT}] resumed verified checkpoint", flush=True)
            continue
        print(f"[fold {fold}/{FOLD_COUNT}] fitting {len(train)} train / {len(valid)} validation rows; "
              f"encoding={encoding}", flush=True)
        starts = {"predictions": len(predictions), "fold_rows": len(fold_rows),
                  "curve_rows": len(curve_rows), "split_rows": len(split_rows),
                  "feature_rows": len(feature_rows)}
        encoder = ReducedFeatureEncoder(n_features=4, seed=SEED + fold).fit(train[FEATURES], train.y3)
        x_train, x_valid = encoder.transform(train[FEATURES]), encoder.transform(valid[FEATURES])
        feature_rows.append({"fold": fold, "features": encoder.selected_names})
        y_train, y_valid = train.y3.to_numpy(), valid.y3.to_numpy()
        models = [
            (f"three_class_vqc_{encoding}", ThreeClassVQC(
                n_qubits=2 if encoding == "amplitude" else 4,
                steps=10, learning_rate=.04, seed=SEED + fold,
                data_reuploading=(encoding != "amplitude"),
                encoding="amplitude" if encoding == "amplitude" else "angle",
                trainable_readout=(encoding == "angle_linear"))),
            ("logistic_4f", LogisticRegression(C=1.0, class_weight="balanced", max_iter=500,
                                                solver="lbfgs", random_state=SEED + fold)),
            ("rbf_svc_4f", SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True,
                                random_state=SEED + fold)),
            ("extra_trees_4f", ExtraTreesClassifier(n_estimators=100, max_depth=12,
                                min_samples_leaf=2, max_features="sqrt", class_weight="balanced",
                                n_jobs=1, random_state=SEED + fold)),
        ]
        qnn_state = None
        for name, model in models:
            before = time.perf_counter()
            if name.startswith("three_class_vqc_"):
                model.fit(x_train, y_train)
                probs = model.predict_proba(x_valid)
                pred = probs.argmax(axis=1)
                runtime = model.training_seconds + model.inference_seconds
                circuit_train = model.training_executions
                circuit_valid = model.inference_executions
                qnn_state = {"weights": np.asarray(model.weights),
                    "readout_weights": np.asarray(model.readout_weights),
                    "readout_bias": np.asarray(model.readout_bias),
                    "loss_history": np.asarray(model.loss_history),
                    "configuration": np.asarray(json.dumps({"encoding": encoding, "fold": fold,
                        "seed": SEED + fold, "qubits": model.n_qubits, "depth": model.depth,
                        "steps": model.steps, "learning_rate": model.learning_rate,
                        "trainable_readout": model.trainable_readout,
                        "device": model.device_name, "diff_method": model.diff_method}))}
                for epoch, loss in enumerate(model.loss_history, start=1):
                    curve_rows.append({"fold": fold, "epoch": epoch, "training_loss": loss})
                qmeta = {"qubits": model.n_qubits, "depth": model.depth,
                         "parameters": model.parameter_count, "shots": None,
                         "statevector_dimension": 2**model.n_qubits}
            else:
                model.fit(x_train, y_train)
                probs = model.predict_proba(x_valid)
                pred = probs.argmax(axis=1)
                runtime = time.perf_counter() - before
                circuit_train = circuit_valid = 0
                qmeta = {"qubits": None, "depth": None, "parameters": None, "shots": None}
            result = metrics(y_valid, pred)
            for i, row in valid.iterrows():
                predictions.append({"record_id": int(row.record_id), "origin_program": str(row.origin_program),
                    "fold": fold, "y_true": int(y_valid[i]), "model": name, "y_pred": int(pred[i]),
                    "prob_0": float(probs[i, 0]), "prob_1": float(probs[i, 1]), "prob_2": float(probs[i, 2])})
            fold_rows.append({"fold": fold, "model": name, "train_records": len(train),
                "validation_records": len(valid), "train_origins": len(train_groups),
                "validation_origins": len(valid_groups), "runtime_seconds": runtime,
                "circuit_train_evaluations": circuit_train, "circuit_validation_evaluations": circuit_valid,
                **qmeta, **result})
        for idx in train_i:
            row = data.iloc[idx]
            split_rows.append({"record_id": int(row.record_id), "origin_program": str(row.origin_program),
                               "fold": fold, "role": "train"})
        for idx in valid_i:
            row = data.iloc[idx]
            split_rows.append({"record_id": int(row.record_id), "origin_program": str(row.origin_program),
                               "fold": fold, "role": "validation"})
        if qnn_state is None:
            raise AssertionError("Fold completed without a QNN state checkpoint")
        payload = {"fingerprint": fingerprint,
            "predictions": predictions[starts["predictions"]:],
            "fold_rows": fold_rows[starts["fold_rows"]:],
            "curve_rows": curve_rows[starts["curve_rows"]:],
            "split_rows": split_rows[starts["split_rows"]:],
            "feature_rows": feature_rows[starts["feature_rows"]:]}
        _save_fold_checkpoint(checkpoint_path, payload, qnn_state)
        print(f"[fold {fold}/{FOLD_COUNT}] checkpoint saved; available RAM "
              f"{psutil.virtual_memory().available / 2**20:.0f} MiB", flush=True)
        if psutil.virtual_memory().available / 2**20 < RAM_STOP_MIB:
            raise MemoryError(f"Available RAM fell below {RAM_STOP_MIB} MiB after fold {fold}")

    pred = pd.DataFrame(predictions)
    if pred.duplicated(["record_id", "model"]).any():
        raise AssertionError("Duplicate predictions")
    for _, group in pred.groupby("record_id"):
        if group.y_true.nunique() != 1 or group.origin_program.nunique() != 1 or group.fold.nunique() != 1:
            raise AssertionError("Prediction records are not paired on label, origin, and fold")
    fold_frame = pd.DataFrame(fold_rows)
    summary = []
    for name, group in fold_frame.groupby("model"):
        summary.append({"model": name, "fold_macro_f1_mean": group.macro_f1.mean(),
            "fold_macro_f1_sd": group.macro_f1.std(ddof=1), "fold_accuracy_mean": group.accuracy.mean(),
            "fold_balanced_accuracy_mean": group.balanced_accuracy.mean(),
            "pooled_macro_f1": f1_score(group_truth := pred.loc[pred.model == name, "y_true"],
                                        pred.loc[pred.model == name, "y_pred"], average="macro"),
            "pooled_accuracy": accuracy_score(group_truth, pred.loc[pred.model == name, "y_pred"]),
            "pooled_balanced_accuracy": balanced_accuracy_score(group_truth, pred.loc[pred.model == name, "y_pred"]),
            "runtime_seconds_total": group.runtime_seconds.sum()})

    pred.to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    fold_frame.to_json(output / "fold_metrics.json", orient="records", indent=2)
    pd.DataFrame(summary).to_csv(output / "candidate_leaderboard.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output / "training_curves.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output / "grouped_split_manifest.csv", index=False)
    pd.DataFrame(feature_rows).to_json(output / "features_by_fold.json", orient="records", indent=2)
    sample_manifest.to_csv(output / "sample_manifest.csv", index=False)
    report = {
        "schema": "qmutbench-three-class-qnn-campaign-v1", "status": "completed",
        "objective": "Exploratory three-class task distinct from incumbent's binary target.",
        "label_map": LABEL_MAP, "sample": sample_audit, "data_adapter": adapter_audit,
        "foundations": foundation, "toy_learning_check": toy,
        "rows": int(len(data)), "origins": int(data.origin_program.nunique()),
        "fold_count": FOLD_COUNT, "fold_metrics": fold_rows, "leaderboard": summary,
        "qnn_configuration": {"class": "ThreeClassVQC", "qubits": 2 if encoding == "amplitude" else 4,
            "depth": 2, "trainable_parameters": (12 if encoding == "amplitude" else
                                                        36 if encoding == "angle_linear" else 24),
            "steps": 10, "optimizer": "PennyLane Adam; full batch",
            "encoding": ("normalized four-value amplitude state; global input norm discarded; state-preparation gates costed separately as an explicit limitation"
                         if encoding == "amplitude" else "RY(pi*x) per qubit and repeated before each trainable layer"),
            "readout": ("Z on each wire plus ZZ expectation as three softmax logits" if encoding == "amplitude"
                        else "three Pauli-Z expectations; " + ("trainable affine 3x3 readout and bias then softmax" if encoding == "angle_linear"
                                                               else "fixed expectations as softmax logits")),
            "simulation": "ideal analytic PennyLane default.qubit statevector; no shots or noise"},
        "elapsed_seconds": time.perf_counter() - started,
        "memory": {"rss_before": process.memory_info().rss, "available_ram_before": memory_start,
                   "available_ram_after": psutil.virtual_memory().available,
                   "stop_floor_mib": RAM_STOP_MIB},
        "existing_binary_champion": {"macro_f1": 0.7933135654357681,
            "comparison": "not comparable: predicts merged WOO/OPO vs Survivor; do not rank against this three-class result"},
        "frozen_test_accessed": False,
        "files": ["predictions.csv.gz", "fold_metrics.json", "candidate_leaderboard.csv",
            "training_curves.csv", "grouped_split_manifest.csv", "features_by_fold.json", "sample_manifest.csv"]}
    (output / "campaign_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "progress_manifest.json").write_text(json.dumps({"status": "completed_exploratory_three_class_qnn",
        "frozen_test_accessed": False, "development_only": True, "next_step": "Assess fold results and resources before any scaling.",
        "completed_files": report["files"] + ["campaign_summary.json", "progress_manifest.json"]}, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding", choices=["angle_reupload", "amplitude", "angle_linear"], default="angle_reupload")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path.cwd()
    out = args.output or root / "reports/experiments/qmutbench_three_class_qnn_campaign_20261027"
    result = run(root, out, encoding=args.encoding)
    print(json.dumps({"status": result["status"], "rows": result["rows"], "origins": result["origins"],
                      "elapsed_seconds": result["elapsed_seconds"], "leaderboard": result["leaderboard"]}, indent=2))
