"""Bounded interaction-feature screen on the saved QMutBench grouped folds.

All target encoding, global lookup tables, origin identifiers, and frozen-test
labels are excluded. OneHotEncoder is fit independently within each training
fold. Outputs are checkpointed under the ignored QMutBench artifact directory.
"""
from __future__ import annotations

import gc
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from experiments.qmutbench_master_optimization import _nested_threshold_report, _sha_ids
from experiments.qmutbench_outcome_ml import (DEFAULT_DATA, EXPECTED_SHA256, FEATURES,
    GROUP_COLUMN, TARGET_COLUMN, TARGET_MAP, group_holdout, load_dataset, metrics,
    prepare_xy, sha256_file)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/qmutbench_ml/master_optimization_v1"
OUT = ROOT / "data/qmutbench_ml/performance_phase_v3"
FULL_OUT = ROOT / "data/qmutbench_ml/performance_phase_full_v1"
FINAL_MODEL_DIR = ROOT / "data/qmutbench_ml/position_fraction_v1"
SEED = 20261025
SCREEN_TREES = 60
MIN_AVAILABLE_BYTES = 450 * 1024**2
VARIANTS = {
    "baseline": (),
    "algorithm_operation": (("algorithm", "operation", "algorithm_operation"),),
    "gate_operation": (("gate", "operation", "gate_operation"),),
    "algorithm_gate": (("algorithm", "gate", "algorithm_gate"),),
    "position_bin": (),
    "position_fraction": (),
}


def add_features(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Add one deterministic categorical interaction; never uses labels/groups."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown feature variant {variant!r}")
    result = frame.loc[:, list(FEATURES)].copy()
    if variant == "position_bin":
        if "position" not in frame:
            raise ValueError("Source-defined position category is required for this variant")
        result["position"] = frame["position"].astype(str)
        return result
    if variant == "position_fraction":
        if "position" not in frame:
            raise ValueError("Source-defined position category is required for this variant")
        pieces = frame["position"].astype(str).str.extract(r"^(\d+)-(\d+)%$").astype(float)
        if pieces.isna().any().any():
            bad = frame.loc[pieces.isna().any(axis=1), "position"].astype(str).unique().tolist()
            raise ValueError(f"Unsupported source position categories: {bad[:5]}")
        result["position_fraction"] = pieces.mean(axis=1) / 100.0
        return result
    for left, right, name in VARIANTS[variant]:
        result[name] = result[left].astype(str) + "::" + result[right].astype(str)
    return result


def _pipeline(columns: list[str], seed: int) -> Pipeline:
    categorical = [name for name in columns if name in {
        "algorithm", "operation", "gate", "dominant_state", "algorithm_operation",
        "gate_operation", "algorithm_gate", "position"}]
    numeric = [name for name in columns if name not in categorical]
    transforms = []
    if numeric:
        transforms.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transforms.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                                     dtype=np.float32)),
        ]), categorical))
    prep = ColumnTransformer(transforms, remainder="drop", sparse_threshold=1.0)
    estimator = ExtraTreesClassifier(n_estimators=SCREEN_TREES, max_depth=22,
        min_samples_leaf=15, max_features="sqrt", class_weight="balanced_subsample",
        n_jobs=2, random_state=seed)
    return Pipeline([("preprocess", prep), ("model", estimator)])


def _metric_record(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    result = metrics(y, (p >= threshold).astype(np.int8))
    result.update({"roc_auc": None, "average_precision": None,
                   "brier_score": float(np.mean((p - y) ** 2))})
    from sklearn.metrics import average_precision_score, roc_auc_score
    if len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, p))
        result["average_precision"] = float(average_precision_score(y, p))
    return result


def _nested_thresholds(oof: pd.DataFrame) -> list[dict]:
    report = _nested_threshold_report(oof.rename(columns={"y": "y_true", "probability": "p_detected"}), seed=SEED)
    return [{"held_fold": row["held_fold"],
        "threshold_fit_on_other_folds": row["nested_threshold_metrics"]["macro_f1"]["threshold_fit_on_other_folds"],
        "threshold_selection_scope": "other two development folds only",
        "held_macro_f1": row["nested_threshold_metrics"]["macro_f1"]["macro_f1"],
        "held_balanced_accuracy": row["nested_threshold_metrics"]["macro_f1"]["balanced_accuracy"]}
        for row in report["folds"]]


def _record_sha(values) -> str:
    return _sha_ids(values)


def _comparison_slices(frame: pd.DataFrame, column: str, *, minimum_support: int = 100) -> list[dict]:
    rows = []
    for value, part in frame.groupby(column, observed=True, dropna=False):
        if len(part) < minimum_support or part.y_candidate.nunique() < 2:
            continue
        y = part.y_candidate.to_numpy(np.int8)
        candidate = _metric_record(y, part.probability.to_numpy(float), .5)
        baseline = _metric_record(y, part.baseline_probability.to_numpy(float), .5)
        rows.append({"slice": str(value), "support": int(len(part)),
            "candidate_macro_f1": candidate["macro_f1"], "baseline_macro_f1": baseline["macro_f1"],
            "macro_f1_delta": candidate["macro_f1"] - baseline["macro_f1"],
            "candidate_balanced_accuracy": candidate["balanced_accuracy"],
            "baseline_balanced_accuracy": baseline["balanced_accuracy"],
            "detected_recall_delta": candidate["per_class"]["Detected by WOO/OPO (1)"]["recall"] - baseline["per_class"]["Detected by WOO/OPO (1)"]["recall"],
            "false_positive_delta": candidate["confusion_matrix_labels_0_1"][0][1] - baseline["confusion_matrix_labels_0_1"][0][1],
            "false_negative_delta": candidate["confusion_matrix_labels_0_1"][1][0] - baseline["confusion_matrix_labels_0_1"][1][0]})
    return rows


def _calibration_summary(y: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> dict:
    frame = pd.DataFrame({"probability": probabilities, "y": y})
    frame["bin"] = pd.qcut(frame.probability, q=bins, duplicates="drop")
    rows = [{"bin": str(interval), "rows": int(len(part)),
        "mean_predicted": float(part.probability.mean()), "observed_rate": float(part.y.mean()),
        "absolute_gap": float(abs(part.probability.mean() - part.y.mean()))}
        for interval, part in frame.groupby("bin", observed=True)]
    ece = float(sum(row["rows"] / len(frame) * row["absolute_gap"] for row in rows))
    return {"equal_frequency_bins": rows, "expected_calibration_error_approx": ece,
        "max_bin_absolute_gap": max((row["absolute_gap"] for row in rows), default=0.0)}


def run(data: Path = DEFAULT_DATA, output: Path = OUT, *, variants: tuple[str, ...] | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(data)
    if source_hash != EXPECTED_SHA256:
        raise ValueError("Dataset checksum does not match the verified QMutBench source")
    base_manifest = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    frame = load_dataset(data)
    x, y, groups = prepare_xy(frame)
    dev_idx, frozen_idx = group_holdout(y, groups, seed=SEED)
    dev = frame.iloc[dev_idx].reset_index(drop=True)
    del frame, x, y, groups
    gc.collect()
    x_dev, y_dev, g_dev = prepare_xy(dev)
    ids_dev = dev["Unnamed: 0"].to_numpy(dtype=np.int64)
    fold_definitions = base_manifest["config"]["folds"]
    for f in fold_definitions:
        fit, val = set(f["fit_groups"]), set(f["validation_groups"])
        if fit & val or (fit | val) != set(g_dev):
            raise ValueError("Saved grouped fold definitions do not partition development origins")
        if _record_sha(ids_dev[np.isin(g_dev, list(val))]) != f["validation_ids_sha256"]:
            raise ValueError("Development row membership differs from saved grouped fold manifest")
    representative = set(base_manifest["selector_memberships"]["representative_200000"]["groups"])
    if not representative <= set(g_dev):
        raise ValueError("Representative screen includes a non-development origin")
    mask = np.isin(g_dev, list(representative))
    feature_source = x_dev.copy()
    feature_source["position"] = dev["position"].astype(str).to_numpy()
    requested = tuple(variants or VARIANTS.keys())
    if any(name not in VARIANTS for name in requested):
        raise ValueError("Requested unknown feature variant")
    prior_path = output / "manifest.json"
    manifest = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else {
        "schema": "qmutbench-performance-phase-v1", "source_sha256": source_hash,
        "base_manifest_sha256": sha256_file(BASE / "manifest.json"), "seed": SEED,
        "training_selection_rows": int(mask.sum()), "training_selection_groups": len(representative),
        "validation_rows": int(len(dev)), "validation_groups": int(len(set(g_dev))),
        "selection_protocol": "representative origins restrict each training fold, matching the existing feature-screen protocol; all held-out origins and rows are evaluated in each saved fixed fold",
        "status": "running", "stages": {}}
    if manifest.get("source_sha256") != source_hash or manifest.get("base_manifest_sha256") != sha256_file(BASE / "manifest.json"):
        raise ValueError("Existing performance checkpoints have incompatible provenance")
    pred_dir = output / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for variant in requested:
        augmented = add_features(feature_source, variant)
        columns = list(augmented.columns)
        for fold_id, fold_info in enumerate(fold_definitions, 1):
            stage = f"{variant}_fold{fold_id}"
            result_path = output / f"{stage}.json"
            prediction_path = pred_dir / f"{stage}.npz"
            cached = manifest["stages"].get(stage, {})
            if (result_path.is_file() and prediction_path.is_file() and
                cached.get("result_sha256") == sha256_file(result_path) and
                cached.get("prediction_sha256") == sha256_file(prediction_path)):
                continue
            available = int(psutil.virtual_memory().available)
            if available < MIN_AVAILABLE_BYTES:
                manifest["stages"][stage] = {"status": "skipped_resource",
                    "available_ram_bytes": available, "minimum_required_bytes": MIN_AVAILABLE_BYTES}
                result_path.write_text(json.dumps(manifest["stages"][stage], indent=2) + "\n", encoding="utf-8")
                continue
            fit_groups = set(fold_info["fit_groups"]) & representative
            val_groups = set(fold_info["validation_groups"])
            fit = np.isin(g_dev, list(fit_groups))
            val = np.isin(g_dev, list(val_groups))
            if not fit.any() or not val.any() or (set(g_dev[fit]) & set(g_dev[val])):
                raise ValueError(f"Invalid group split for {stage}")
            seed = SEED + fold_id
            model = _pipeline(columns, seed)
            started = time.perf_counter()
            with threadpool_limits(limits=2):
                model.fit(augmented.loc[fit], y_dev[fit])
                probability = model.predict_proba(augmented.loc[val])[:, 1]
            result = {"status": "complete", "variant": variant, "fold": fold_id,
                "rows": int(val.sum()), "training_rows": int(fit.sum()),
                "training_groups": len(fit_groups), "validation_groups": len(val_groups),
                "validation_group_sha256": _record_sha(np.unique(g_dev[val])),
                "validation_ids_sha256": _record_sha(ids_dev[val]),
                "feature_columns": columns, "model_parameters": model.named_steps["model"].get_params(),
                "seed": seed, "fit_seconds": float(time.perf_counter() - started),
                "available_ram_before_fit_bytes": available,
                "metrics_at_0_5": _metric_record(y_dev[val], probability, .5)}
            tmp = prediction_path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, record_id=ids_dev[val], y=y_dev[val],
                origin_program=g_dev[val].astype(str), fold=np.full(int(val.sum()), fold_id, dtype=np.int8),
                probability=probability)
            os.replace(tmp, prediction_path)
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest["stages"][stage] = {"status": "complete",
                "result_sha256": sha256_file(result_path), "prediction_sha256": sha256_file(prediction_path),
                "result_path": str(result_path), "prediction_path": str(prediction_path)}
            manifest["status"] = "running"
            prior_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            del model
            gc.collect()
        # Merge fold predictions and calculate paired metrics and nested cutoffs.
        if all(manifest["stages"].get(f"{variant}_fold{i}", {}).get("status") == "complete" for i in range(1, 4)):
            arrays = [np.load(pred_dir / f"{variant}_fold{i}.npz") for i in range(1, 4)]
            oof = pd.DataFrame({key: np.concatenate([a[key] for a in arrays])
                for key in ("record_id", "y", "origin_program", "fold", "probability")})
            if (oof.record_id.duplicated().any() or len(oof) != len(dev)
                    or set(oof.record_id) != set(ids_dev)):
                raise ValueError(f"OOF coverage failure in {variant}")
            if set(oof.origin_program) != set(g_dev):
                raise ValueError(f"OOF origin coverage failure in {variant}")
            predictions = oof.probability.to_numpy(float)
            labels = oof.y.to_numpy(np.int8)
            result = {"variant": variant, "rows": len(oof), "groups": len(set(g_dev)),
                "training_selection_rows": int(mask.sum()), "training_selection_groups": len(representative),
                "metrics_at_0_5": _metric_record(labels, predictions, .5),
                "metrics_at_0_485": _metric_record(labels, predictions, .485),
                "fold_metrics_at_0_5": [json.loads((output / f"{variant}_fold{i}.json").read_text())["metrics_at_0_5"] for i in range(1, 4)],
                "nested_thresholds_macro_f1": _nested_thresholds(oof),
                "prediction_sha256": hashlib.sha256(b"".join(sha256_file(pred_dir / f"{variant}_fold{i}.npz").encode() for i in range(1, 4))).hexdigest()}
            result["nested_macro_f1_mean"] = float(np.mean([r["held_macro_f1"] for r in result["nested_thresholds_macro_f1"]]))
            result["nested_macro_f1_sd"] = float(np.std([r["held_macro_f1"] for r in result["nested_thresholds_macro_f1"]]))
            (output / f"{variant}_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest.setdefault("summaries", {})[variant] = result
            prior_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            for a in arrays:
                a.close()
            del oof, arrays
            gc.collect()
        del augmented
        gc.collect()
    del feature_source
    gc.collect()
    statuses = [v.get("status") for k, v in manifest["stages"].items() if any(k.startswith(f"{n}_") for n in requested)]
    manifest["status"] = "complete" if statuses and all(s == "complete" for s in statuses) else "complete_with_resource_skips"
    prior_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def run_full_position_finalist(data: Path = DEFAULT_DATA, output: Path = FULL_OUT) -> dict:
    """Evaluate the promising percent-position feature on all saved full-dev folds."""
    output.mkdir(parents=True, exist_ok=True)
    base_manifest = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(data) != EXPECTED_SHA256:
        raise ValueError("Dataset checksum does not match the verified QMutBench source")
    frame = load_dataset(data)
    x, y, groups = prepare_xy(frame)
    dev_idx, _frozen_idx = group_holdout(y, groups, seed=SEED)
    dev = frame.iloc[dev_idx].reset_index(drop=True)
    del frame, x, y, groups
    gc.collect()
    x_dev, y_dev, g_dev = prepare_xy(dev)
    ids_dev = dev["Unnamed: 0"].to_numpy(dtype=np.int64)
    base_oof_path = BASE / "oof_predictions.csv"
    base_oof = pd.read_csv(base_oof_path)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "schema": "qmutbench-full-position-finalist-v1", "source_sha256": EXPECTED_SHA256,
        "base_oof_sha256": sha256_file(base_oof_path),
        "base_manifest_sha256": sha256_file(BASE / "manifest.json"),
        "feature_variant": "position_fraction",
        "feature_definition": "midpoint of the source Position (Circuit %) category, divided by 100",
        "model_parameters": {"n_estimators": 100, "max_depth": 22, "min_samples_leaf": 15,
            "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": 2},
        "seed": SEED, "status": "running", "stages": {}}
    if manifest["source_sha256"] != EXPECTED_SHA256 or manifest["base_oof_sha256"] != sha256_file(base_oof_path):
        raise ValueError("Existing full-finalist checkpoint provenance mismatch")
    feature_source = x_dev.copy()
    feature_source["position"] = dev["position"].astype(str).to_numpy()
    augmented = add_features(feature_source, "position_fraction")
    columns = list(augmented.columns)
    fold_definitions = base_manifest["config"]["folds"]
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for fold_id, fold_info in enumerate(fold_definitions, 1):
        stage = f"position_fraction_fold{fold_id}"
        result_path = output / f"{stage}.json"
        prediction_path = prediction_dir / f"{stage}.npz"
        cached = manifest["stages"].get(stage, {})
        if (result_path.is_file() and prediction_path.is_file() and
            cached.get("result_sha256") == sha256_file(result_path) and
            cached.get("prediction_sha256") == sha256_file(prediction_path)):
            continue
        available = int(psutil.virtual_memory().available)
        if available < MIN_AVAILABLE_BYTES:
            manifest["stages"][stage] = {"status": "skipped_resource", "available_ram_bytes": available,
                "minimum_required_bytes": MIN_AVAILABLE_BYTES}
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            continue
        fit_groups, val_groups = set(fold_info["fit_groups"]), set(fold_info["validation_groups"])
        if fit_groups & val_groups or (fit_groups | val_groups) != set(g_dev):
            raise ValueError(f"Saved group split invalid at finalist fold {fold_id}")
        fit, val = np.isin(g_dev, list(fit_groups)), np.isin(g_dev, list(val_groups))
        if _record_sha(ids_dev[val]) != fold_info["validation_ids_sha256"]:
            raise ValueError(f"Finalist fold {fold_id} rows mismatch saved manifest")
        model = _pipeline(columns, SEED + fold_id)
        model.named_steps["model"].set_params(n_estimators=100)
        start = time.perf_counter()
        with threadpool_limits(limits=2):
            model.fit(augmented.loc[fit], y_dev[fit])
            probability = model.predict_proba(augmented.loc[val])[:, 1]
        result = {"status": "complete", "fold": fold_id, "training_rows": int(fit.sum()),
            "validation_rows": int(val.sum()), "training_groups": len(fit_groups),
            "validation_groups": len(val_groups), "validation_ids_sha256": _record_sha(ids_dev[val]),
            "validation_group_sha256": _record_sha(np.unique(g_dev[val])), "feature_columns": columns,
            "seed": SEED + fold_id, "model_parameters": model.named_steps["model"].get_params(),
            "fit_seconds": float(time.perf_counter() - start),
            "available_ram_before_fit_bytes": available,
            "metrics_at_0_5": _metric_record(y_dev[val], probability, .5)}
        tmp = prediction_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, record_id=ids_dev[val], y=y_dev[val], origin_program=g_dev[val].astype(str),
            fold=np.full(int(val.sum()), fold_id, dtype=np.int8), probability=probability)
        os.replace(tmp, prediction_path)
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["stages"][stage] = {"status": "complete", "result_sha256": sha256_file(result_path),
            "prediction_sha256": sha256_file(prediction_path)}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        del model, probability
        gc.collect()
    if all(manifest["stages"].get(f"position_fraction_fold{i}", {}).get("status") == "complete" for i in range(1, 4)):
        arrays = [np.load(prediction_dir / f"position_fraction_fold{i}.npz") for i in range(1, 4)]
        candidate = pd.DataFrame({key: np.concatenate([a[key] for a in arrays]) for key in
            ("record_id", "y", "origin_program", "fold", "probability")})
        if candidate.record_id.duplicated().any() or set(candidate.record_id) != set(ids_dev):
            raise ValueError("Full finalist OOF predictions do not exactly cover development row ids")
        baseline = base_oof.rename(columns={"y_true": "y", "p_detected": "baseline_probability"})
        joined = candidate.merge(baseline[["record_id", "y", "origin_program", "fold", "baseline_probability"]],
            on="record_id", how="inner", validate="one_to_one", suffixes=("_candidate", "_baseline"))
        if len(joined) != len(dev) or not np.array_equal(joined.y_candidate, joined.y_baseline):
            raise ValueError("Finalist and baseline OOF rows or targets do not align")
        if not np.array_equal(joined.fold_candidate, joined.fold_baseline):
            raise ValueError("Finalist OOF fold assignments differ from baseline")
        diagnostic_columns = ["Unnamed: 0", GROUP_COLUMN, "algorithm", "operation", "gate",
            "dominant_state", "qubits", "gates", "Position_int", "position"]
        diagnostic = dev[diagnostic_columns].rename(columns={"Unnamed: 0": "record_id",
            GROUP_COLUMN: "source_origin"})
        joined = joined.merge(diagnostic, on="record_id", how="left", validate="one_to_one")
        if joined.source_origin.isna().any() or not np.array_equal(
                joined.origin_program_candidate.astype(str), joined.source_origin.astype(str)):
            raise ValueError("OOF origin keys failed to match source diagnostics")
        y_true = joined.y_candidate.to_numpy(np.int8)
        candidate_p = joined.probability.to_numpy(float)
        baseline_p = joined.baseline_probability.to_numpy(float)
        nested = _nested_threshold_report(pd.DataFrame({"y_true": y_true,
            "p_detected": candidate_p, "origin_program": joined.origin_program_candidate,
            "fold": joined.fold_candidate}), seed=SEED)
        baseline_nested = _nested_threshold_report(pd.DataFrame({"y_true": y_true,
            "p_detected": baseline_p, "origin_program": joined.origin_program_candidate,
            "fold": joined.fold_candidate}), seed=SEED)
        threshold_grid = np.linspace(0, 1, 201)
        threshold_scores = [f1_score(y_true, candidate_p >= t, average="macro", zero_division=0)
                            for t in threshold_grid]
        selected_idx = max(range(len(threshold_scores)), key=lambda i: (threshold_scores[i], -abs(threshold_grid[i] - .5)))
        fold_results = []
        for fold_id in (1, 2, 3):
            part = joined.fold_candidate.to_numpy() == fold_id
            c = _metric_record(y_true[part], candidate_p[part], .5)
            b = _metric_record(y_true[part], baseline_p[part], .5)
            fold_results.append({"fold": fold_id, "groups": int(joined.loc[part, "origin_program_candidate"].nunique()),
                "candidate": c, "baseline": b, "macro_f1_delta": c["macro_f1"] - b["macro_f1"],
                "balanced_accuracy_delta": c["balanced_accuracy"] - b["balanced_accuracy"]})
        group_stats = []
        for group, rows in joined.groupby("origin_program_candidate", observed=True):
            cm = _metric_record(rows.y_candidate.to_numpy(np.int8), rows.probability.to_numpy(float), .5)
            bm = _metric_record(rows.y_baseline.to_numpy(np.int8), rows.baseline_probability.to_numpy(float), .5)
            group_stats.append({"origin_program": str(group), "rows": len(rows),
                "candidate_macro_f1": cm["macro_f1"], "baseline_macro_f1": bm["macro_f1"],
                "macro_f1_delta": cm["macro_f1"] - bm["macro_f1"]})
        slice_frame = joined.copy()
        slice_frame["qubit_bin"] = pd.cut(slice_frame.qubits, [0, 4, 8, 16, 24, np.inf], include_lowest=True)
        slice_frame["gate_count_bin"] = pd.cut(slice_frame.gates, [0, 25, 50, 100, 250, 500, 1000, np.inf], include_lowest=True)
        slice_frame["position_index_bin"] = pd.cut(slice_frame.Position_int, [-1, 5, 10, 25, 50, 100, 250, np.inf], include_lowest=True)
        slice_frame["gate_operation"] = slice_frame.gate.astype(str) + "::" + slice_frame.operation.astype(str)
        slice_frame["algorithm_operation"] = slice_frame.algorithm.astype(str) + "::" + slice_frame.operation.astype(str)
        slice_comparisons = {column: _comparison_slices(slice_frame, column) for column in
            ["algorithm", "operation", "gate", "dominant_state", "qubit_bin", "gate_count_bin",
             "position", "position_index_bin", "gate_operation", "algorithm_operation"]}
        candidate_pred = candidate_p >= .5
        errors = candidate_pred != y_true
        high_confidence = np.maximum(candidate_p, 1 - candidate_p) >= .9
        candidate_calibration = _calibration_summary(y_true, candidate_p)
        baseline_calibration = _calibration_summary(y_true, baseline_p)
        candidate_metrics = _metric_record(y_true, candidate_p, .5)
        baseline_metrics = _metric_record(y_true, baseline_p, .5)
        candidate_metrics_0485 = _metric_record(y_true, candidate_p, .485)
        baseline_metrics_0485 = _metric_record(y_true, baseline_p, .485)
        summary = {"schema": "qmutbench-full-position-finalist-results-v1",
            "dataset_sha256": EXPECTED_SHA256, "baseline_oof_sha256": sha256_file(base_oof_path),
            "candidate_oof_sha256": hashlib.sha256(b"".join(sha256_file(prediction_dir / f"position_fraction_fold{i}.npz").encode() for i in (1,2,3))).hexdigest(),
            "rows": len(joined), "groups": int(joined.origin_program_candidate.nunique()),
            "candidate": {"feature": "position_fraction", "definition": "source Position (Circuit %) category midpoint / 100",
                "features": columns, "training_scope": "all development rows in each fit fold; group holdout respected"},
            "baseline_metrics_at_0_5": baseline_metrics, "candidate_metrics_at_0_5": candidate_metrics,
            "baseline_metrics_at_0_485": baseline_metrics_0485,
            "candidate_metrics_at_0_485": candidate_metrics_0485,
            "macro_f1_delta": candidate_metrics["macro_f1"] - baseline_metrics["macro_f1"],
            "balanced_accuracy_delta": candidate_metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"],
            "accuracy_delta": candidate_metrics["accuracy"] - baseline_metrics["accuracy"],
            "fold_results": fold_results,
            "fold_delta_mean": float(np.mean([r["macro_f1_delta"] for r in fold_results])),
            "fold_delta_sd": float(np.std([r["macro_f1_delta"] for r in fold_results], ddof=1)),
            "fold_wins": int(sum(r["macro_f1_delta"] > 0 for r in fold_results)),
            "nested_thresholds": nested,
            "nested_macro_f1_mean": float(np.mean([f["nested_threshold_metrics"]["macro_f1"]["macro_f1"] for f in nested["folds"]])),
            "baseline_nested_thresholds": baseline_nested,
            "baseline_nested_macro_f1_mean": float(np.mean([f["nested_threshold_metrics"]["macro_f1"]["macro_f1"] for f in baseline_nested["folds"]])),
            "nested_macro_f1_delta": float(np.mean([f["nested_threshold_metrics"]["macro_f1"]["macro_f1"] for f in nested["folds"]]) -
                np.mean([f["nested_threshold_metrics"]["macro_f1"]["macro_f1"] for f in baseline_nested["folds"]])),
            "candidate_default_fold_macro_f1_mean": float(np.mean([f["default_threshold"]["macro_f1"] for f in nested["folds"]])),
            "full_oof_selected_threshold_descriptive": {"objective": "macro F1", "threshold": float(threshold_grid[selected_idx]),
                "score": float(threshold_scores[selected_idx]), "selection_bias_note": "same OOF predictions were used to choose and score this threshold; nested fold results are the less-biased assessment"},
            "origin_performance": {"baseline_mean_macro_f1": float(np.mean([r["baseline_macro_f1"] for r in group_stats])),
                "candidate_mean_macro_f1": float(np.mean([r["candidate_macro_f1"] for r in group_stats])),
                "mean_origin_macro_f1_delta": float(np.mean([r["macro_f1_delta"] for r in group_stats])),
                "median_origin_macro_f1_delta": float(np.median([r["macro_f1_delta"] for r in group_stats])),
                "origins_improved": int(sum(r["macro_f1_delta"] > 0 for r in group_stats)),
                "origins_total": len(group_stats)},
            "error_analysis": {"candidate_false_positives": candidate_metrics["confusion_matrix_labels_0_1"][0][1],
                "baseline_false_positives": baseline_metrics["confusion_matrix_labels_0_1"][0][1],
                "candidate_false_negatives": candidate_metrics["confusion_matrix_labels_0_1"][1][0],
                "baseline_false_negatives": baseline_metrics["confusion_matrix_labels_0_1"][1][0],
                "wrong_predictions_at_probability_confidence_ge_0_9": int(np.sum(errors & high_confidence)),
                "high_confidence_wrong_prediction_share_of_all_rows": float(np.mean(errors & high_confidence)),
                "candidate_calibration": candidate_calibration,
                "baseline_calibration": baseline_calibration,
                "minimum_slice_support": 100, "slice_comparisons": slice_comparisons,
                "lowest_macro_f1_slices": {name: sorted(rows, key=lambda row: (row["candidate_macro_f1"], -row["support"]))[:10]
                    for name, rows in slice_comparisons.items()},
                "largest_slice_gains": {name: sorted(rows, key=lambda row: row["macro_f1_delta"], reverse=True)[:5]
                    for name, rows in slice_comparisons.items()}},
            "selection_criterion": "at least +0.5 percentage points grouped macro F1 on comparable full-development OOF, no material deterioration in balanced accuracy or either class recall, and positive nested grouped-fold evidence",
            "historical_frozen_test": "not used"}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["summary"] = summary
        manifest["status"] = "complete"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for a in arrays:
            a.close()
    elif any(stage.get("status") == "skipped_resource" for stage in manifest["stages"].values()):
        manifest["status"] = "complete_with_resource_skips"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def fit_final_position_model(data: Path = DEFAULT_DATA, output: Path = FINAL_MODEL_DIR) -> dict:
    """Fit the selected full-development model to a new versioned artifact path."""
    summary_path = FULL_OUT / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Full grouped validation must complete before final model fitting")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("macro_f1_delta", 0) < .005 or summary.get("fold_wins", 0) != 3:
        raise ValueError("Position-fraction candidate did not meet the promotion criterion")
    model_path = output / "best_model.joblib"
    metadata_path = output / "best_model_metadata.json"
    if model_path.exists() or metadata_path.exists():
        raise FileExistsError("Versioned final-model path already exists; preserving existing artifacts")
    if sha256_file(data) != EXPECTED_SHA256:
        raise ValueError("Dataset checksum does not match verified source")
    available = int(psutil.virtual_memory().available)
    if available < MIN_AVAILABLE_BYTES:
        raise MemoryError(f"Only {available} bytes RAM available before full-development fit")
    frame = load_dataset(data)
    x, y, groups = prepare_xy(frame)
    dev_idx, _frozen_idx = group_holdout(y, groups, seed=SEED)
    dev = frame.iloc[dev_idx].reset_index(drop=True)
    del frame, x, y, groups
    gc.collect()
    x_dev, y_dev, groups_dev = prepare_xy(dev)
    source = x_dev.copy()
    source["position"] = dev["position"].astype(str).to_numpy()
    features = add_features(source, "position_fraction")
    model = _pipeline(list(features.columns), SEED)
    model.named_steps["model"].set_params(n_estimators=100)
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        model.fit(features, y_dev)
    elapsed = float(time.perf_counter() - started)
    output.mkdir(parents=True, exist_ok=False)
    tmp_model = output / "best_model.joblib.tmp"
    joblib.dump(model, tmp_model, compress=3)
    os.replace(tmp_model, model_path)
    position_map = {f"{lo}-{hi}%": (lo + hi) / 200.0 for lo, hi in
        [(0, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60),
         (61, 70), (71, 80), (81, 90), (91, 100)]}
    metadata = {"schema": "qmutbench-inference-model-v1", "model_version": "qmutbench-position-fraction-v1",
        "model_path": model_path.relative_to(ROOT).as_posix(), "model_sha256": sha256_file(model_path),
        "feature_schema": list(features.columns), "input_feature_schema": [*FEATURES, "position"],
        "feature_derivations": {"position_fraction": {"source_column": "position", "mapping": position_map,
            "definition": "midpoint of source Position (Circuit %) category divided by 100"}},
        "target_map": TARGET_MAP, "inference_threshold": 0.5,
        "threshold_selection": "fixed 0.5 retained; nested grouped threshold selection did not improve held-fold macro F1 over 0.5",
        "training_data_sha256": EXPECTED_SHA256, "training_rows": int(len(dev)),
        "training_origin_groups": int(len(set(groups_dev))),
        "training_scope": "all development rows only; historical frozen-test origins excluded",
        "validation_manifest_sha256": sha256_file(FULL_OUT / "manifest.json"),
        "validation_oof_sha256": summary["candidate_oof_sha256"],
        "model_parameters": model.named_steps["model"].get_params(),
        "fit_seconds": elapsed, "python_version": os.sys.version.split()[0],
        "scikit_learn_version": sklearn.__version__,
        "note": "Probability is an uncalibrated Extra Trees probability; the historical frozen-test set was not used."}
    tmp_meta = metadata_path.with_suffix(".json.tmp")
    tmp_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_meta, metadata_path)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-position-finalist", action="store_true",
        help="Run the screened circuit-percent feature on all saved grouped development folds")
    parser.add_argument("--fit-final-model", action="store_true",
        help="Fit and save the promoted full-development model under its new versioned artifact directory")
    args = parser.parse_args()
    if args.fit_final_model:
        result = fit_final_position_model()
    elif args.full_position_finalist:
        result = run_full_position_finalist()
    else:
        result = run()
    print(json.dumps({"status": result.get("status", "saved"), "summary": result.get("summary", result.get("summaries")),
        "model_path": result.get("model_path"),
        "manifest": str((FULL_OUT if args.full_position_finalist else OUT) / "manifest.json")}, indent=2))
