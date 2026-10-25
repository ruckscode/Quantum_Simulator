"""Compare QMutBench predictors with syntax-only features on saved grouped folds."""
from __future__ import annotations

import gc
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from experiments.qmutbench_qasm_features import (BASELINE_OOF, DEFAULT_OUTPUT, FEATURE_COLUMNS,
    COUNT_GATES, ROOT, SOURCE_CSV, sha256_file)

PREDICTIONS = ROOT / "data/qmutbench_ml/performance_phase_full_v1/predictions"
OUT = ROOT / "data/qmutbench_ml/qasm_model_comparison_v3"
RUNS = OUT / "folds"
BASELINE_RUNS = ROOT / "data/qmutbench_ml/qasm_model_comparison_v1/folds"
SEED = 20261025
TREES = 60
MEMORY_GUARD_BYTES = 450 * 1024**2
BASE_CATEGORICAL = ["algorithm", "operation", "gate", "dominant_state"]
BASE_NUMERIC = ["qubits", "gates", "Position_int", "position_fraction"]
BASE_FEATURES = BASE_CATEGORICAL + BASE_NUMERIC
STRUCTURE_FEATURES = [
    "qasm_mut_gate_count", "qasm_mut_depth", "qasm_mut_depth_per_qubit",
    "qasm_mut_single_qubit_count", "qasm_mut_two_qubit_count", "qasm_mut_three_plus_count",
    "qasm_mut_two_plus_fraction", "qasm_mut_distinct_gate_types",
    "qasm_orig_gate_count", "qasm_orig_depth", "qasm_orig_depth_per_qubit",
    "qasm_orig_single_qubit_count", "qasm_orig_two_qubit_count", "qasm_orig_three_plus_count",
    "qasm_orig_two_plus_fraction", "qasm_orig_distinct_gate_types",
    "qasm_delta_gate_count", "qasm_delta_depth", "qasm_delta_two_qubit_fraction",
]
COMPOSITION_FEATURES = [name for name in FEATURE_COLUMNS if name.endswith("_fraction") and "qasm_mut_" in name]
ORIGINAL_COMPOSITION_COLUMNS = [f"qasm_orig_{name}_fraction" for name in COUNT_GATES]
DELTA_COMPOSITION_FEATURES = [f"qasm_delta_{name}_fraction" for name in COUNT_GATES]
VARIANTS = {
    "baseline": [],
    "baseline_random_forest": [],
    "structure_only": STRUCTURE_FEATURES,
    "mutant_gate_composition": COMPOSITION_FEATURES,
    "all_qasm": FEATURE_COLUMNS,
}
MODELS = {"extra_trees": ExtraTreesClassifier, "random_forest": RandomForestClassifier}


def assert_group_fold_isolation(oof: pd.DataFrame) -> None:
    required = {"record_id", "origin_program", "fold", "y"}
    if not required <= set(oof.columns):
        raise ValueError(f"OOF split table is missing {sorted(required - set(oof.columns))}")
    if oof.record_id.duplicated().any():
        raise ValueError("OOF record IDs must be unique")
    if oof.groupby("origin_program", sort=False).fold.nunique().max() != 1:
        raise ValueError("An Origin_program is split across validation folds")


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(np.int8)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "per_class": {"survivor_0": {"precision": float(precision[0]), "recall": float(recall[0]),
                                        "f1": float(f1[0]), "support": int(support[0])},
                      "detected_1": {"precision": float(precision[1]), "recall": float(recall[1]),
                                     "f1": float(f1[1]), "support": int(support[1])}},
        "confusion_matrix_labels_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(np.mean((p - y) ** 2)),
    }


def add_delta_composition(qasm: pd.DataFrame) -> pd.DataFrame:
    missing = (set(ORIGINAL_COMPOSITION_COLUMNS) | set(COMPOSITION_FEATURES)) - set(qasm.columns)
    if missing:
        raise ValueError(f"QASM table lacks composition inputs: {sorted(missing)}")
    for gate, feature in zip(COUNT_GATES, DELTA_COMPOSITION_FEATURES):
        qasm[feature] = (qasm[f"qasm_mut_{gate}_fraction"] - qasm[f"qasm_orig_{gate}_fraction"]).astype(np.float32)
    return qasm


def _load_verified_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    oof_parts = [np.load(PREDICTIONS / f"position_fraction_fold{i}.npz", allow_pickle=False)
                 for i in (1, 2, 3)]
    oof = pd.DataFrame({key: np.concatenate([part[key] for part in oof_parts])
                        for key in ("record_id", "y", "origin_program", "fold", "probability")})
    if len(oof) != 578117 or oof.record_id.duplicated().any() or set(oof.fold) != {1, 2, 3}:
        raise ValueError("Saved position-fraction grouped OOF rows failed their manifest contract")
    assert_group_fold_isolation(oof)

    dev_ids = set(oof.record_id.astype(int))
    base_cols = ["Unnamed: 0", "algorithm", "operation", "gate", "dominant_state",
                 "qubits", "gates", "Position_int", "position"]
    # Reading features only. The outcome column is not loaded from the full source CSV.
    raw = pd.read_csv(SOURCE_CSV, usecols=base_cols, low_memory=False)
    raw = raw[raw["Unnamed: 0"].isin(dev_ids)].rename(columns={"Unnamed: 0": "record_id"})
    raw = raw.set_index("record_id").loc[oof.record_id.to_numpy()].reset_index()
    if len(raw) != len(oof) or raw.record_id.duplicated().any():
        raise ValueError("Source prediction inputs do not align exactly with development OOF IDs")
    pieces = raw.position.astype(str).str.extract(r"^(\d+)-(\d+)%$").astype(np.float32)
    if pieces.isna().any().any():
        raise ValueError("Unsupported Position band in development inputs")
    raw["position_fraction"] = pieces.mean(axis=1).div(100).astype(np.float32)
    for name in BASE_CATEGORICAL:
        raw[name] = raw[name].astype("category")
    for name in BASE_NUMERIC:
        raw[name] = raw[name].astype(np.float32)
    if {"record_id", "Origin_program", "Killed", "y_true"}.intersection(BASE_FEATURES):
        raise AssertionError("Target, group, or row ID is present in the registered model inputs")

    chunks = []
    qasm_columns = ["record_id", *FEATURE_COLUMNS, *ORIGINAL_COMPOSITION_COLUMNS]
    for path in sorted(DEFAULT_OUTPUT.glob("features_*.csv.gz")):
        part = pd.read_csv(path, usecols=qasm_columns,
                           dtype={c: np.float32 for c in qasm_columns if c != "record_id"})
        chunks.append(part)
    qasm = pd.concat(chunks, ignore_index=True)
    if len(qasm) != len(oof) or qasm.record_id.duplicated().any() or set(qasm.record_id) != dev_ids:
        raise ValueError("Extracted QASM features do not exactly cover the development OOF row IDs")
    qasm = qasm.set_index("record_id").loc[oof.record_id.to_numpy()].reset_index()
    qasm = add_delta_composition(qasm)
    x = raw.merge(qasm, on="record_id", how="inner", validate="one_to_one")
    if len(x) != len(oof) or x.isna().any().any():
        raise ValueError("Development model matrix has missing rows or values")
    return x, oof


def _model(model_family: str, feature_names: list[str], seed: int, trees: int = TREES):
    categorical = [name for name in feature_names if name in BASE_CATEGORICAL]
    numeric = [name for name in feature_names if name not in categorical]
    transformations = []
    if numeric:
        transformations.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformations.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
        ]), categorical))
    prep = ColumnTransformer(transformations, remainder="drop", sparse_threshold=1.0)
    estimator_type = MODELS[model_family]
    estimator = estimator_type(n_estimators=trees, max_depth=22, min_samples_leaf=15,
                               max_features="sqrt", class_weight="balanced_subsample",
                               n_jobs=1, random_state=seed)
    return Pipeline([("preprocess", prep), ("model", estimator)])


def run_screen(resume: bool = True, selected_variants: list[str] | None = None) -> dict:
    x, oof = _load_verified_data()
    RUNS.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    provenance = {"schema": "qmutbench-qasm-model-screen-v1",
                  "dataset_sha256": "3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778",
                  "qasm_features_manifest_sha256": sha256_file(DEFAULT_OUTPUT / "manifest.json"),
                  "candidate_baseline_oof_sha256": hashlib.sha256(b"".join(sha256_file(PREDICTIONS / f"position_fraction_fold{i}.npz").encode() for i in (1, 2, 3))).hexdigest(),
                  "baseline_source_oof_sha256": sha256_file(BASELINE_OOF),
                  "folds": {str(f): {"rows": int((oof.fold == f).sum()),
                                     "groups": int(oof.loc[oof.fold == f, "origin_program"].nunique())}
                            for f in (1, 2, 3)},
                  "seed": SEED, "trees": TREES, "feature_variants": VARIANTS,
                  "models": ["extra_trees", "random_forest"], "threshold": 0.5}
    config_key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() and resume else {
        **provenance, "config_key": config_key, "stages": {}, "status": "running"}
    if manifest.get("config_key") != config_key:
        raise ValueError("Existing QASM model-screen checkpoint has stale provenance/configuration")
    selected = selected_variants or list(VARIANTS)
    if any(name not in VARIANTS for name in selected):
        raise ValueError(f"Unknown QASM feature variant in {selected}")
    manifest["execution_scope"] = selected
    for variant in selected:
        add = VARIANTS[variant]
        names = BASE_FEATURES + list(add)
        families = (("random_forest",) if variant == "baseline_random_forest" else
                    (("extra_trees", "random_forest") if variant == "all_qasm" else ("extra_trees",)))
        for model_family in families:
            for fold in (1, 2, 3):
                stage = f"{model_family}_{variant}_fold{fold}"
                pred_path = RUNS / f"{stage}.npz"
                result_path = RUNS / f"{stage}.json"
                if (stage in manifest["stages"] and pred_path.exists() and result_path.exists()
                        and manifest["stages"][stage]["prediction_sha256"] == sha256_file(pred_path)
                        and manifest["stages"][stage]["result_sha256"] == sha256_file(result_path)):
                    continue
                available = int(psutil.virtual_memory().available)
                if available < MEMORY_GUARD_BYTES:
                    manifest["status"] = "paused_resource_guard"
                    manifest["available_ram_bytes"] = available
                    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    return manifest
                val = oof.fold.to_numpy() == fold
                train = ~val
                if set(oof.loc[train, "origin_program"]) & set(oof.loc[val, "origin_program"]):
                    raise ValueError(f"Origin_program leakage in fold {fold}")
                model = _model(model_family, names, SEED + fold)
                start = time.perf_counter()
                with threadpool_limits(limits=1):
                    model.fit(x.loc[train, names], oof.loc[train, "y"].to_numpy(np.int8))
                    p = model.predict_proba(x.loc[val, names])[:, 1]
                y = oof.loc[val, "y"].to_numpy(np.int8)
                ids = oof.loc[val, "record_id"].to_numpy(np.int64)
                result = {"status": "complete", "stage": stage, "model_family": model_family,
                          "variant": variant, "fold": fold, "training_rows": int(train.sum()),
                          "validation_rows": int(val.sum()), "training_origins": int(oof.loc[train, "origin_program"].nunique()),
                          "validation_origins": int(oof.loc[val, "origin_program"].nunique()),
                          "validation_record_id_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                          "feature_names": names, "seed": SEED + fold, "fit_and_predict_seconds": time.perf_counter() - start,
                          "available_ram_before_fit": available, "metrics_at_threshold_0_5": metrics(y, p)}
                tmp = pred_path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, record_id=ids, y=y, origin_program=oof.loc[val, "origin_program"].astype(str).to_numpy(dtype=str),
                                    fold=np.full(int(val.sum()), fold, np.int8), probability=p)
                tmp.replace(pred_path)
                result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                manifest["stages"][stage] = {"prediction_sha256": sha256_file(pred_path),
                                             "result_sha256": sha256_file(result_path), "status": "complete"}
                manifest["status"] = "running"
                manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                print(f"{stage}: macroF1={result['metrics_at_threshold_0_5']['macro_f1']:.5f}; fold rows={len(y)}", flush=True)
                del model, p
                gc.collect()
    summaries = {}
    for variant, add in VARIANTS.items():
        families = (("random_forest",) if variant == "baseline_random_forest" else
                    (("extra_trees", "random_forest") if variant == "all_qasm" else ("extra_trees",)))
        for family in families:
            if not all((RUNS / f"{family}_{variant}_fold{i}.npz").is_file() and
                       (RUNS / f"{family}_{variant}_fold{i}.json").is_file() for i in (1, 2, 3)):
                continue
            pred = pd.concat([pd.DataFrame({k: np.load(RUNS / f"{family}_{variant}_fold{i}.npz")[k]
                                            for k in ("record_id", "y", "origin_program", "fold", "probability")})
                              for i in (1, 2, 3)], ignore_index=True)
            pred = pred.sort_values("record_id").reset_index(drop=True)
            if len(pred) != len(oof) or not np.array_equal(pred.record_id, oof.sort_values("record_id").record_id):
                raise ValueError(f"OOF coverage mismatch for {family}/{variant}")
            fold_scores = [json.loads((RUNS / f"{family}_{variant}_fold{i}.json").read_text()) for i in (1, 2, 3)]
            summaries[f"{family}/{variant}"] = {
                "pooled_oof_metrics_at_0_5": metrics(pred.y.to_numpy(np.int8), pred.probability.to_numpy(float)),
                "fold_metrics_at_0_5": [r["metrics_at_threshold_0_5"] for r in fold_scores],
                "mean_fold_macro_f1": float(np.mean([r["metrics_at_threshold_0_5"]["macro_f1"] for r in fold_scores])),
                "fold_macro_f1_sd_sample": float(np.std([r["metrics_at_threshold_0_5"]["macro_f1"] for r in fold_scores], ddof=1)),
                "fit_seconds_by_fold": [r["fit_and_predict_seconds"] for r in fold_scores],
            }
            manifest["summary"] = summaries
    expected_stages = {f"{family}_{variant}_fold{fold}" for variant, add in VARIANTS.items()
                       for family in (("random_forest",) if variant == "baseline_random_forest" else
                                      (("extra_trees", "random_forest") if variant == "all_qasm" else ("extra_trees",)))
                       for fold in (1, 2, 3)}
    manifest["status"] = "complete" if expected_stages <= set(manifest["stages"]) else "partial_screen"
    manifest["summary"] = summaries
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def run_delta_composition_screen() -> dict:
    """Test signed mutant-minus-original gate-mix shifts on identical saved folds."""
    x, oof = _load_verified_data()
    RUNS.mkdir(parents=True, exist_ok=True)
    base_names = BASE_FEATURES
    names = base_names + DELTA_COMPOSITION_FEATURES
    results = []
    for fold in (1, 2, 3):
        stage = f"delta_gate_composition_fold{fold}"
        pred_path = RUNS / f"{stage}.npz"
        result_path = RUNS / f"{stage}.json"
        if pred_path.is_file() and result_path.is_file():
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        val = oof.fold.to_numpy() == fold
        train = ~val
        if set(oof.loc[train, "origin_program"]) & set(oof.loc[val, "origin_program"]):
            raise ValueError(f"Origin_program leakage in delta-composition fold {fold}")
        model = _model("extra_trees", names, SEED + fold)
        start = time.perf_counter()
        with threadpool_limits(limits=1):
            model.fit(x.loc[train, names], oof.loc[train, "y"].to_numpy(np.int8))
            p = model.predict_proba(x.loc[val, names])[:, 1]
        y = oof.loc[val, "y"].to_numpy(np.int8)
        ids = oof.loc[val, "record_id"].to_numpy(np.int64)
        result = {"status": "complete", "stage": stage, "model_family": "extra_trees",
                  "trees": TREES, "feature_names": names, "fold": fold,
                  "training_rows": int(train.sum()), "validation_rows": int(val.sum()),
                  "training_origins": int(oof.loc[train, "origin_program"].nunique()),
                  "validation_origins": int(oof.loc[val, "origin_program"].nunique()),
                  "threshold": 0.5, "fit_and_predict_seconds": time.perf_counter() - start,
                  "metrics_at_threshold_0_5": metrics(y, p)}
        tmp = pred_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, record_id=ids, y=y,
                            origin_program=oof.loc[val, "origin_program"].astype(str).to_numpy(dtype=str),
                            fold=np.full(int(val.sum()), fold, np.int8), probability=p)
        tmp.replace(pred_path)
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        results.append(result)
        print(f"{stage}: macroF1={result['metrics_at_threshold_0_5']['macro_f1']:.5f}", flush=True)
        del model, p
        gc.collect()
    pred = pd.concat([pd.DataFrame({key: np.load(RUNS / f"delta_gate_composition_fold{i}.npz", allow_pickle=False)[key]
                                    for key in ("record_id", "y", "origin_program", "fold", "probability")})
                      for i in (1, 2, 3)], ignore_index=True).sort_values("record_id")
    baseline = pd.concat([pd.DataFrame({key: np.load(BASELINE_RUNS / f"extra_trees_baseline_fold{i}.npz", allow_pickle=False)[key]
                                        for key in ("record_id", "y", "origin_program", "fold", "probability")})
                          for i in (1, 2, 3)], ignore_index=True).sort_values("record_id")
    if not np.array_equal(pred.record_id, baseline.record_id) or not np.array_equal(pred.y, baseline.y):
        raise ValueError("Delta-composition screen does not align with matched baseline OOF rows")
    score = metrics(pred.y.to_numpy(np.int8), pred.probability.to_numpy(float))
    baseline_score = metrics(baseline.y.to_numpy(np.int8), baseline.probability.to_numpy(float))
    report = {"definition": "per-gate mutant QASM instruction-fraction minus original QASM instruction-fraction",
              "features": DELTA_COMPOSITION_FEATURES, "rows": len(pred),
              "threshold": 0.5, "pooled_oof_metrics": score,
              "matched_60_tree_baseline_pooled_oof_metrics": baseline_score,
              "pooled_macro_f1_delta": score["macro_f1"] - baseline_score["macro_f1"],
              "fold_macro_f1": [r["metrics_at_threshold_0_5"]["macro_f1"] for r in results],
              "baseline_fold_macro_f1": [json.loads((BASELINE_RUNS / f"extra_trees_baseline_fold{i}.json").read_text())["metrics_at_threshold_0_5"]["macro_f1"] for i in (1, 2, 3)],
              "fold_macro_f1_delta": [r["metrics_at_threshold_0_5"]["macro_f1"] - json.loads((BASELINE_RUNS / f"extra_trees_baseline_fold{i}.json").read_text())["metrics_at_threshold_0_5"]["macro_f1"] for i, r in enumerate(results, 1)],
              "fit_seconds_by_fold": [r["fit_and_predict_seconds"] for r in results],
              "status": "complete"}
    report_path = OUT / "delta_composition_summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_candidate_confirmation() -> dict:
    """Confirm only mutant gate composition with the selected model's 100 trees."""
    x, oof = _load_verified_data()
    RUNS.mkdir(parents=True, exist_ok=True)
    names = BASE_FEATURES + COMPOSITION_FEATURES
    results = []
    for fold in (1, 2, 3):
        available = int(psutil.virtual_memory().available)
        if available < MEMORY_GUARD_BYTES:
            raise MemoryError(f"Need >= {MEMORY_GUARD_BYTES} bytes for fold confirmation; available={available}")
        stage = f"extra_trees_gate_composition_100_fold{fold}"
        val = oof.fold.to_numpy() == fold
        train = ~val
        if set(oof.loc[train, "origin_program"]) & set(oof.loc[val, "origin_program"]):
            raise ValueError(f"Origin_program leakage in confirmation fold {fold}")
        model = _model("extra_trees", names, SEED + fold, trees=100)
        start = time.perf_counter()
        with threadpool_limits(limits=1):
            model.fit(x.loc[train, names], oof.loc[train, "y"].to_numpy(np.int8))
            p = model.predict_proba(x.loc[val, names])[:, 1]
        y = oof.loc[val, "y"].to_numpy(np.int8)
        ids = oof.loc[val, "record_id"].to_numpy(np.int64)
        result = {"status": "complete", "stage": stage, "model_family": "extra_trees",
                  "trees": 100, "features": names, "fold": fold,
                  "training_rows": int(train.sum()), "validation_rows": int(val.sum()),
                  "training_origins": int(oof.loc[train, "origin_program"].nunique()),
                  "validation_origins": int(oof.loc[val, "origin_program"].nunique()),
                  "threshold": 0.5, "fit_and_predict_seconds": time.perf_counter() - start,
                  "metrics_at_threshold_0_5": metrics(y, p)}
        pred_path, result_path = RUNS / f"{stage}.npz", RUNS / f"{stage}.json"
        temp = pred_path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, record_id=ids, y=y,
                            origin_program=oof.loc[val, "origin_program"].astype(str).to_numpy(dtype=str),
                            fold=np.full(len(y), fold, np.int8), probability=p)
        temp.replace(pred_path)
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        results.append(result)
        print(f"{stage}: macroF1={result['metrics_at_threshold_0_5']['macro_f1']:.5f}", flush=True)
        del model, p
        gc.collect()
    candidate = pd.concat([pd.DataFrame({key: np.load(RUNS / f"extra_trees_gate_composition_100_fold{i}.npz", allow_pickle=False)[key]
                                         for key in ("record_id", "y", "origin_program", "fold", "probability")})
                           for i in (1, 2, 3)], ignore_index=True).sort_values("record_id")
    baseline = pd.concat([pd.DataFrame({key: np.load(PREDICTIONS / f"position_fraction_fold{i}.npz", allow_pickle=False)[key]
                                        for key in ("record_id", "y", "origin_program", "fold", "probability")})
                          for i in (1, 2, 3)], ignore_index=True).sort_values("record_id")
    if not np.array_equal(candidate.record_id, baseline.record_id) or not np.array_equal(candidate.y, baseline.y):
        raise ValueError("Confirmation OOF candidate does not align exactly with the saved 100-tree baseline")
    if not np.array_equal(candidate.fold, baseline.fold):
        raise ValueError("Confirmation candidate changed saved grouped folds")
    candidate_score = metrics(candidate.y.to_numpy(np.int8), candidate.probability.to_numpy(float))
    baseline_score = metrics(baseline.y.to_numpy(np.int8), baseline.probability.to_numpy(float))
    report = {"schema": "qmutbench-qasm-candidate-confirmation-v1", "status": "complete",
              "feature_set": "current position_fraction model inputs plus 12 mutant QASM native gate-family fractions",
              "rows": len(candidate), "origins": int(candidate.origin_program.nunique()),
              "folds": [r["metrics_at_threshold_0_5"] for r in results],
              "baseline_fold_metrics": [metrics(baseline.loc[baseline.fold.eq(f), "y"].to_numpy(np.int8),
                                                       baseline.loc[baseline.fold.eq(f), "probability"].to_numpy(float)) for f in (1, 2, 3)],
              "candidate_pooled_oof_metrics": candidate_score,
              "baseline_pooled_oof_metrics": baseline_score,
              "pooled_macro_f1_delta": candidate_score["macro_f1"] - baseline_score["macro_f1"],
              "fold_macro_f1_deltas": [results[f-1]["metrics_at_threshold_0_5"]["macro_f1"] -
                                       metrics(baseline.loc[baseline.fold.eq(f), "y"].to_numpy(np.int8),
                                               baseline.loc[baseline.fold.eq(f), "probability"].to_numpy(float))["macro_f1"] for f in (1,2,3)],
              "runtime_seconds_by_fold": [r["fit_and_predict_seconds"] for r in results],
              "selection_rule": "require >=0.5 pp macro-F1 gain and supportive gains on all grouped folds; threshold fixed at 0.5"}
    path = OUT / "candidate_confirmation.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=None,
                        help="Optional bounded subset for staged screening; cached fold fits are reused")
    parser.add_argument("--delta-composition", action="store_true",
                        help="Screen signed mutant-minus-original gate-mix features on the same saved folds")
    parser.add_argument("--confirm-gate-composition", action="store_true",
                        help="Confirm the strongest screen candidate with the selected model's 100 trees")
    args = parser.parse_args()
    if args.delta_composition:
        report = run_delta_composition_screen()
        print(json.dumps(report, indent=2))
        raise SystemExit(0)
    if args.confirm_gate_composition:
        report = run_candidate_confirmation()
        print(json.dumps(report, indent=2))
        raise SystemExit(0)
    report = run_screen(selected_variants=args.variants)
    print(json.dumps({"status": report["status"], "summary": report.get("summary", {})}, indent=2))
