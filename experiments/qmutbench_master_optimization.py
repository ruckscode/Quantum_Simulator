"""Evidence-driven, resumable QMutBench grouped model optimization.

All model/feature/selection decisions use the 305 development origins. The
historically inspected 70-origin test is intentionally not opened here.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from experiments.qmutbench_outcome_ml import (
    CATEGORICAL, DEFAULT_DATA, DTYPES, EXPECTED_SHA256, FEATURES,
    GROUP_COLUMN, LABEL_ORDER, NUMERIC, SEED, TARGET_COLUMN, TARGET_MAP,
    evaluate_saved_predictions, group_holdout, load_dataset, make_pipeline,
    metrics, prepare_xy, sha256_file,
)
from experiments.qmutbench_selected_pipeline import _load_selected

try:
    import psutil
except ImportError:  # Keep the runner usable without an optional monitor package.
    psutil = None

SCHEMA = "qmutbench-master-optimization-v1"
OUT = Path("data/qmutbench_ml/master_optimization_v1")
LEARNING_DIR = Path("data/qmutbench_ml/learning_curve_v1")
SELECTED_DIR = Path("data/qmutbench_ml/selected_v3")
BEST_ET_PARAMS = {"n_estimators": 100, "max_depth": 22, "min_samples_leaf": 15,
                  "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": 2}
SCREEN_ET_PARAMS = {**BEST_ET_PARAMS, "n_estimators": 60}
PRACTICAL_WITHIN_PP = 1.0
MIN_AVAILABLE_BYTES = 450 * 1024**2
FOLDS = 3
HIST_MAX_FIT_ROWS = 150_000


def _sha_ids(values) -> str:
    payload = "\n".join(str(x) for x in sorted(values)).encode()
    return hashlib.sha256(payload).hexdigest()


def _available_ram_bytes() -> int:
    if psutil is not None:
        return int(psutil.virtual_memory().available)
    if os.name == "nt":
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.avail_phys)
    return 0


def _process_rss_bytes() -> int:
    if psutil is not None:
        return int(psutil.Process().memory_info().rss)
    return 0

FEATURE_SETS = {
    "full": tuple(FEATURES),
    "numeric_only": tuple(NUMERIC),
    "categorical_only": tuple(CATEGORICAL),
    "no_algorithm": tuple(c for c in FEATURES if c != "algorithm"),
    "no_operation": tuple(c for c in FEATURES if c != "operation"),
    "no_gate": tuple(c for c in FEATURES if c != "gate"),
    "no_dominant_state": tuple(c for c in FEATURES if c != "dominant_state"),
    "add_algorithm_group": tuple([*FEATURES, "algorithm_group"]),
    "add_position_bin": tuple([*FEATURES, "position"]),
}


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        frame.to_csv(f, index=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _onehot_pipeline(columns: tuple[str, ...], model, *, linear: bool = False) -> Pipeline:
    categorical = [c for c in columns if c in CATEGORICAL or c in ("algorithm_group", "position")]
    numeric = [c for c in columns if c in NUMERIC]
    transforms = []
    if numeric:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if linear:
            numeric_steps.append(("scale", StandardScaler()))
        transforms.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transforms.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True,
                                     dtype=np.float32)),
        ]), categorical))
    return Pipeline([("preprocess", ColumnTransformer(transforms, remainder="drop",
        sparse_threshold=1.0)), ("model", model)])


def _model(name: str, seed: int, feature_set: str = "full", params: dict | None = None):
    columns = FEATURE_SETS[feature_set]
    params = dict(params or {})
    if name == "historical_rf":
        est = RandomForestClassifier(n_estimators=40, max_depth=18, min_samples_leaf=30,
            class_weight="balanced_subsample", max_features="sqrt", n_jobs=2, random_state=seed)
    elif name == "rf_unweighted":
        est = RandomForestClassifier(n_estimators=40, max_depth=18, min_samples_leaf=30,
            class_weight=None, max_features="sqrt", n_jobs=2, random_state=seed)
    elif name == "extra_trees_best":
        est = ExtraTreesClassifier(**{**BEST_ET_PARAMS, **params, "random_state": seed})
    elif name == "extra_trees_screen":
        est = ExtraTreesClassifier(**{**SCREEN_ET_PARAMS, **params, "random_state": seed})
    elif name == "logistic_regression":
        est = LogisticRegression(class_weight="balanced", max_iter=500, solver="liblinear",
                                 random_state=seed)
        return _onehot_pipeline(columns, est, linear=True)
    elif name == "dummy":
        est = DummyClassifier(strategy="most_frequent")
    elif name == "et_unweighted_tuned":
        est = ExtraTreesClassifier(**{**BEST_ET_PARAMS, "class_weight": None,
                                      **params, "random_state": seed})
    else:
        raise ValueError(f"Unknown candidate {name}")
    return _onehot_pipeline(columns, est)


def _metric_record(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    result = metrics(y_true, y_pred)
    return result


def _resource_snapshot() -> dict:
    return {"available_ram_bytes": _available_ram_bytes(),
            "total_ram_bytes": int(psutil.virtual_memory().total) if psutil is not None else None,
            "free_disk_bytes": int(shutil.disk_usage(OUT if OUT.exists() else Path.cwd()).free),
            "cpu_count": int(os.cpu_count() or 1)}


def _profile_groups(dev: pd.DataFrame) -> tuple[dict, dict]:
    rows = {}
    features = {}
    for group, part in dev.groupby(GROUP_COLUMN, observed=True, sort=False):
        g = str(group)
        y = part[TARGET_COLUMN].astype(str).map(TARGET_MAP).to_numpy(dtype=np.int8)
        rows[g] = {"n": len(part), "positive": int(y.sum()), "rate": float(y.mean())}
        values = set()
        for col in ("algorithm_group", "algorithm", "operation", "gate", "dominant_state"):
            values.update((col, str(x)) for x in part[col].unique())
        values.update(("qubits", str(x)) for x in part.qubits.unique())
        for col, bins in (("gates", [0, 10, 25, 50, 100, 250, 500, 1000, 5000, float("inf")]),
                          ("Position_int", [0, 5, 10, 25, 50, 100, 250, 500, float("inf")])):
            cuts = pd.cut(part[col], bins=bins, include_lowest=True)
            values.update((col, str(x)) for x in cuts.astype(str).unique())
        features[g] = values
    return rows, features


def _make_selectors(dev: pd.DataFrame, representative_groups: list[str], *, seed: int) -> dict[str, list[str]]:
    row_stats, group_features = _profile_groups(dev)
    available = sorted(row_stats)
    target = 200_000
    selectors = {"representative_200000": list(representative_groups)}
    # Plain whole-origin random baseline. Complete origins are kept together.
    rng = np.random.default_rng(seed + 113)
    order = list(rng.permutation(available))
    chosen, rows = [], 0
    for group in order:
        if rows >= target:
            break
        chosen.append(group)
        rows += row_stats[group]["n"]
    selectors["random_group_200000"] = chosen

    # Coverage plus explicit prevalence control. Rarity is softened with
    # inverse-square-root frequencies; candidates are scored after adding the
    # whole origin, including current row-level class-prevalence deviation.
    freq = {}
    for group in available:
        for feature in group_features[group]:
            freq[feature] = freq.get(feature, 0) + 1
    full_rate = sum(v["positive"] for v in row_stats.values()) / sum(v["n"] for v in row_stats.values())
    rng = np.random.default_rng(seed + 211)
    ties = dict(zip(available, rng.random(len(available))))
    remaining = set(available)
    chosen, seen, n, positive = [], set(), 0, 0
    while remaining and n < target:
        winner, winner_score = None, -1e30
        for group in remaining:
            stat = row_stats[group]
            fresh = group_features[group] - seen
            novelty = sum(1 / np.sqrt(freq[f]) for f in fresh)
            post_rate = (positive + stat["positive"]) / (n + stat["n"])
            class_cost = abs(post_rate - full_rate)
            overshoot = max(0, n + stat["n"] - target) / target
            size_cost = stat["n"] / target
            score = novelty / np.sqrt(stat["n"]) - 10.0 * class_cost - 0.8 * overshoot - 0.03 * size_cost + ties[group] * 1e-7
            if score > winner_score:
                winner, winner_score = group, score
        stat = row_stats[winner]
        chosen.append(winner)
        remaining.remove(winner)
        seen.update(group_features[winner])
        n += stat["n"]
        positive += stat["positive"]
    selectors["prevalence_coverage_200000"] = chosen
    for name, values in selectors.items():
        if len(values) != len(set(values)):
            raise AssertionError(f"duplicate groups in selector {name}")
    return selectors


def _validate_membership(train_groups, val_groups, frozen_groups) -> None:
    if set(train_groups) & set(val_groups):
        raise AssertionError("Origin_program leakage between fit and validation")
    if set(train_groups) & set(frozen_groups) or set(val_groups) & set(frozen_groups):
        raise AssertionError("Frozen test origin entered development experiment")


def _run_fold(out: Path, manifest: dict, config_key: str, *, stage: str,
              model_name: str, feature_set: str, x_train: pd.DataFrame,
              y_train: np.ndarray, x_val: pd.DataFrame, y_val: np.ndarray,
              val_ids: np.ndarray, val_groups: np.ndarray,
              model_params: dict | None = None, seed: int = SEED,
              save_probabilities: bool = False) -> dict:
    result_path = out / "fold_results" / f"{stage}.json"
    pred_path = out / "fold_predictions" / f"{stage}.csv"
    saved = manifest["stages"].get(stage, {})
    if result_path.is_file() and saved.get("result_sha256") == sha256_file(result_path):
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if (cached.get("config_key") == config_key and
                (not save_probabilities or (pred_path.is_file() and
                 saved.get("predictions_sha256") == sha256_file(pred_path)))):
            return cached
    available_now = _available_ram_bytes()
    if available_now < MIN_AVAILABLE_BYTES:
        skip = {"status": "skipped_resource", "reason": "available RAM below 450 MiB",
                "available_before_bytes": available_now}
        manifest["stages"][stage] = skip
        _atomic_json(out / "manifest.json", manifest)
        return skip
    start = time.perf_counter()
    model = _model(model_name, seed, feature_set, model_params)
    available = _available_ram_bytes()
    peak = _process_rss_bytes()
    with threadpool_limits(limits=2):
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        prob = model.predict_proba(x_val)[:, 1] if save_probabilities else None
    peak = max(peak, _process_rss_bytes())
    result = {"status": "complete", "config_key": config_key, "stage": stage,
              "model": model_name, "feature_set": feature_set,
              "features": list(FEATURE_SETS[feature_set]), "model_params": model_params or {},
              "train_rows": int(len(y_train)), "validation_rows": int(len(y_val)),
              "train_groups": int(x_train.attrs.get("n_groups", -1)),
              "validation_groups": int(len(np.unique(val_groups))),
              "validation_group_sha256": _sha_ids(np.unique(val_groups)),
              "validation_record_ids_sha256": _sha_ids(val_ids),
              "fit_seconds": float(time.perf_counter() - start),
              "available_before_bytes": available,
              "peak_process_rss_bytes": int(peak), **_metric_record(y_val, pred)}
    if save_probabilities:
        fold_id = int(stage.rsplit("fold", 1)[1])
        predictions = pd.DataFrame({"record_id": val_ids, "origin_program": val_groups,
                                    "y_true": y_val, "y_pred": pred, "p_detected": prob,
                                    "fold": fold_id})
        _atomic_csv(predictions, pred_path)
        manifest["stages"][stage] = {"status": "complete", "result_path": str(result_path),
            "result_sha256": "", "predictions_path": str(pred_path),
            "predictions_sha256": sha256_file(pred_path)}
    _atomic_json(result_path, result)
    if not save_probabilities:
        manifest["stages"][stage] = {"status": "complete", "result_path": str(result_path),
                                     "result_sha256": sha256_file(result_path)}
    else:
        manifest["stages"][stage]["result_sha256"] = sha256_file(result_path)
    manifest["failed_stages"].pop(stage, None)
    _atomic_json(out / "manifest.json", manifest)
    del model, pred
    gc.collect()
    return result


def _feature_set_experiments(out: Path, manifest: dict, config_key: str,
                             x_dev: pd.DataFrame, raw_dev: pd.DataFrame,
                             y_dev: np.ndarray, groups_dev: np.ndarray,
                             ids_dev: np.ndarray, folds, selector_groups: list[str], *, seed: int) -> dict:
    results = {}
    selected = set(selector_groups)
    for feature_set in FEATURE_SETS:
        for fold_idx, (fit_idx, val_idx) in enumerate(folds, 1):
            fit_group_set = set(groups_dev[fit_idx]) & selected
            _validate_membership(fit_group_set, set(groups_dev[val_idx]), set(manifest["config"]["frozen_test_groups"]))
            train_mask = np.isin(groups_dev, list(fit_group_set))
            columns = FEATURE_SETS[feature_set]
            source = x_dev.copy()
            if "algorithm_group" in columns:
                source["algorithm_group"] = raw_dev["algorithm_group"].to_numpy()
            if "position" in columns:
                source["position"] = raw_dev["Position_int"].to_numpy()
            x = source.loc[:, list(columns)].copy()
            train_x = x.loc[train_mask]
            train_x.attrs["n_groups"] = len(fit_group_set)
            val_x = x.iloc[val_idx]
            stage = f"feature_{feature_set}_rep200k_fold{fold_idx}"
            result = _run_fold(out, manifest, config_key, stage=stage,
                model_name="extra_trees_screen", feature_set=feature_set,
                x_train=train_x, y_train=y_dev[train_mask], x_val=val_x,
                y_val=y_dev[val_idx], val_ids=ids_dev[val_idx],
                val_groups=groups_dev[val_idx], seed=seed + fold_idx)
            results.setdefault(feature_set, []).append(result)
            print(f"{stage}: {result.get('status')} macro_f1={result.get('macro_f1', float('nan')):.4f}", flush=True)
            del train_x, val_x, x
            gc.collect()
    return results


def _candidate_selector_experiments(out: Path, manifest: dict, config_key: str,
                                    x_dev: pd.DataFrame, y_dev: np.ndarray,
                                    groups_dev: np.ndarray, ids_dev: np.ndarray, folds,
                                    selectors: dict[str, list[str]], *, seed: int) -> dict:
    results = {}
    for selector in ("random_group_200000", "prevalence_coverage_200000"):
        group_set = set(selectors[selector])
        for fold_idx, (fit_idx, val_idx) in enumerate(folds, 1):
            fit_group_set = set(groups_dev[fit_idx]) & group_set
            _validate_membership(fit_group_set, set(groups_dev[val_idx]), set(manifest["config"]["frozen_test_groups"]))
            train_mask = np.isin(groups_dev, list(fit_group_set))
            x_train = x_dev.loc[train_mask].copy()
            x_train.attrs["n_groups"] = len(fit_group_set)
            result = _run_fold(out, manifest, config_key,
                stage=f"selector_{selector}_fold{fold_idx}", model_name="extra_trees_screen",
                feature_set="full", x_train=x_train, y_train=y_dev[train_mask],
                x_val=x_dev.iloc[val_idx], y_val=y_dev[val_idx], val_ids=ids_dev[val_idx],
                val_groups=groups_dev[val_idx], seed=seed + fold_idx)
            results.setdefault(selector, []).append(result)
            print(f"selector {selector} fold {fold_idx}: {result.get('status')} macro_f1={result.get('macro_f1', float('nan')):.4f}", flush=True)
            del x_train
            gc.collect()
    return results


def _oof_and_thresholds(out: Path, manifest: dict, config_key: str,
                        x_dev: pd.DataFrame, raw_dev: pd.DataFrame,
                        y_dev: np.ndarray, groups_dev: np.ndarray,
                        ids_dev: np.ndarray, folds, *, seed: int) -> dict:
    predictions = []
    for fold_idx, (fit_idx, val_idx) in enumerate(folds, 1):
        x_train = x_dev.iloc[fit_idx].copy()
        x_train.attrs["n_groups"] = len(set(groups_dev[fit_idx]))
        stage = f"oof_extra_trees_best_fold{fold_idx}"
        result = _run_fold(out, manifest, config_key, stage=stage,
            model_name="extra_trees_best", feature_set="full", x_train=x_train,
            y_train=y_dev[fit_idx], x_val=x_dev.iloc[val_idx], y_val=y_dev[val_idx],
            val_ids=ids_dev[val_idx], val_groups=groups_dev[val_idx],
            model_params=BEST_ET_PARAMS, seed=seed + fold_idx, save_probabilities=True)
        pred_file = out / "fold_predictions" / f"{stage}.csv"
        predictions.append(pd.read_csv(pred_file))
        print(f"OOF fold {fold_idx}: {result.get('status')} fit={result.get('fit_seconds',0):.1f}s", flush=True)
        del x_train
        gc.collect()
    oof = pd.concat(predictions, ignore_index=True).sort_values("record_id").reset_index(drop=True)
    if oof.record_id.duplicated().any() or len(oof) != len(ids_dev):
        raise ValueError("OOF prediction set must cover each development record once")
    _atomic_csv(oof, out / "oof_predictions.csv")
    threshold_report = _nested_threshold_report(oof, seed=seed)
    error_report = _error_analysis(oof, dev_frame=raw_dev, seed=seed)
    result = {"oof_rows": len(oof), "oof_groups": int(oof.origin_program.nunique()),
              "default_threshold_metrics": metrics(oof.y_true.to_numpy(), oof.y_pred.to_numpy()),
              "nested_thresholds": threshold_report,
              "error_analysis": error_report,
              "predictions_sha256": sha256_file(out / "oof_predictions.csv")}
    _atomic_json(out / "oof_analysis.json", result)
    manifest["oof_analysis"] = {"path": str(out / "oof_analysis.json"),
                                 "sha256": sha256_file(out / "oof_analysis.json")}
    _atomic_json(out / "manifest.json", manifest)
    return result


def _nested_threshold_report(oof: pd.DataFrame, *, seed: int) -> dict:
    y = oof.y_true.to_numpy(dtype=np.int8)
    p = oof.p_detected.to_numpy(dtype=float)
    groups = oof.origin_program.astype(str).to_numpy()
    fold_key = oof.fold.to_numpy(dtype=np.int8) if "fold" in oof else None
    if fold_key is None:
        raise ValueError("OOF rows must retain their validation fold")
    rows = []
    # A bounded grid avoids quadratic work over hundreds of thousands of
    # unique probabilities while still resolving thresholds to 0.005.
    candidates = np.linspace(0.0, 1.0, 201)

    def score_grid(mask):
        truth, probability = y[mask], p[mask]
        positive = np.sort(probability[truth == 1])
        negative = np.sort(probability[truth == 0])
        tp = len(positive) - np.searchsorted(positive, candidates, side="left")
        fp = len(negative) - np.searchsorted(negative, candidates, side="left")
        fn, tn = len(positive) - tp, len(negative) - fp
        recall_pos = tp / max(1, len(positive))
        recall_neg = tn / max(1, len(negative))
        precision_pos = tp / np.maximum(1, tp + fp)
        precision_neg = tn / np.maximum(1, tn + fn)
        f1_pos = 2 * precision_pos * recall_pos / np.maximum(1e-15, precision_pos + recall_pos)
        f1_neg = 2 * precision_neg * recall_neg / np.maximum(1e-15, precision_neg + recall_neg)
        accuracy = (tp + tn) / max(1, len(truth))
        return {"accuracy": accuracy, "balanced_accuracy": (recall_pos + recall_neg) / 2,
                "macro_f1": (f1_pos + f1_neg) / 2}

    for held_fold in np.unique(fold_key):
        fit_mask, eval_mask = fold_key != held_fold, fold_key == held_fold
        thresholds = {}
        fitted_scores = score_grid(fit_mask)
        evaluated_scores = score_grid(eval_mask)
        default_scores = score_grid(eval_mask)
        for metric_name in ("macro_f1", "balanced_accuracy"):
            best = max(range(len(candidates)), key=lambda i: (
                fitted_scores[metric_name][i], -abs(candidates[i] - .5)))
            threshold = float(candidates[best])
            thresholds[metric_name] = {"threshold_fit_on_other_folds": threshold,
                                       **{k: float(v[best]) for k, v in evaluated_scores.items()}}
        rows.append({"held_fold": int(held_fold), "validation_groups": int(len(np.unique(groups[eval_mask]))),
                     "default_threshold": {"threshold": 0.5, **{k: float(v[100]) for k, v in default_scores.items()}},
                     "nested_threshold_metrics": thresholds})
    # Final training threshold is estimated only from development OOF rows and
    # is reported separately from the nested estimate.
    final = {}
    for metric_name in ("macro_f1", "balanced_accuracy"):
        final_scores = score_grid(np.ones(len(y), dtype=bool))
        best = max(range(len(candidates)), key=lambda i: (
            final_scores[metric_name][i], -abs(candidates[i] - .5)))
        threshold = float(candidates[best])
        final[metric_name] = {"development_oof_threshold": threshold,
                              **{k: float(v[best]) for k, v in final_scores.items()}}
    return {"objective": "Nested leave-one-development-fold-out threshold selection; each fold threshold fitted only on the other two folds",
            "folds": rows, "development_oof_final_thresholds_for_future_fit": final,
            "note": "Final OOF thresholds are for a future deployment fit only; same-OOF metrics are descriptive and not unbiased performance estimates."}


def _error_analysis(oof: pd.DataFrame, dev_frame: pd.DataFrame, *, seed: int) -> dict:
    # The features have already been assembled in dev_frame and remain separate
    # from model inputs. Join only by exact record order established from source IDs.
    frame = dev_frame.copy().reset_index(drop=True)
    if len(frame) != len(oof):
        raise ValueError("OOF/error-analysis row counts differ")
    if "Unnamed: 0" in frame:
        aligned = frame.set_index("Unnamed: 0").loc[oof.record_id].reset_index()
    else:
        raise ValueError("Source row IDs needed only for analysis joins")
    aligned["y_true"] = oof.y_true.to_numpy()
    aligned["y_pred"] = oof.y_pred.to_numpy()
    aligned["fold"] = oof.fold.to_numpy()
    aligned["error"] = aligned.y_true.ne(aligned.y_pred)
    overall = metrics(aligned.y_true.to_numpy(), aligned.y_pred.to_numpy())
    group = aligned.groupby(GROUP_COLUMN, observed=True).agg(rows=("error", "size"),
        accuracy=("error", lambda s: 1 - float(s.mean())),
        errors=("error", "sum"), detected_rate=("y_true", "mean"))
    group_summary = {"n_origins": int(len(group)),
        "mean_accuracy": float(group.accuracy.mean()),
        "median_accuracy": float(group.accuracy.median()),
        "minimum_accuracy": float(group.accuracy.min()),
        "maximum_accuracy": float(group.accuracy.max()),
        "lowest_supported_origins": group[group.rows >= 100].sort_values("accuracy").head(10).reset_index().to_dict("records")}
    subgroup_specs = {
        "algorithm": aligned.algorithm.astype(str),
        "operation": aligned.operation.astype(str),
        "gate": aligned.gate.astype(str),
        "qubit_bin": pd.cut(aligned.qubits, [0, 4, 8, 16, 24, 31], include_lowest=True).astype(str),
        "gate_count_bin": pd.cut(aligned.gates, [0, 25, 50, 100, 250, 500, 1000, float("inf")], include_lowest=True).astype(str),
        "dominant_state": aligned.dominant_state.astype(str),
        "mutation_position_bin": pd.cut(aligned.Position_int, [0, 5, 10, 25, 50, 100, 250, 500, float("inf")], include_lowest=True).astype(str),
    }
    subgroups = {}
    for name, values in subgroup_specs.items():
        tmp = aligned.assign(_subgroup=values).groupby("_subgroup", observed=True).agg(
            support=("error", "size"), errors=("error", "sum"),
            accuracy=("error", lambda s: 1 - float(s.mean())),
            detected_support=("y_true", "sum"))
        tmp = tmp[tmp.support >= 100].sort_values("accuracy")
        subgroups[name] = {"min_support": 100, "groups": tmp.reset_index().to_dict("records")}
    return {"overall_oof": overall, "origin_performance": group_summary,
            "subgroups": subgroups,
            "note": "Subgroup slices are overlapping development OOF diagnostics, not independent estimates; supports <100 omitted."}


def run(data: Path = DEFAULT_DATA, out: Path = OUT, *, seed: int = SEED,
        phase: str = "all") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if sha256_file(data) != EXPECTED_SHA256:
        raise ValueError("Official source checksum mismatch")
    prior = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").is_file() else {}
    frame = load_dataset(data)
    x, y, groups = prepare_xy(frame)
    ids = frame["Unnamed: 0"].to_numpy(dtype=np.int64)
    train_idx, test_idx = group_holdout(y, groups, seed=seed)
    frozen = set(groups[test_idx].astype(str))
    dev = frame.iloc[train_idx].reset_index(drop=True)
    x_dev, y_dev, g_dev = prepare_xy(dev)
    ids_dev = dev["Unnamed: 0"].to_numpy(dtype=np.int64)
    dev_groups = set(g_dev)
    if dev_groups & frozen:
        raise AssertionError("Frozen-origin leakage")
    folds = list(StratifiedGroupKFold(n_splits=FOLDS, shuffle=True,
        random_state=seed + 1).split(np.zeros(len(y_dev)), y_dev, g_dev))
    fold_manifest = []
    for fit_idx, val_idx in folds:
        fit_groups, val_groups = set(g_dev[fit_idx]), set(g_dev[val_idx])
        _validate_membership(fit_groups, val_groups, frozen)
        fold_manifest.append({"fit_groups": sorted(fit_groups), "validation_groups": sorted(val_groups),
            "fit_group_sha256": _sha_ids(fit_groups), "validation_group_sha256": _sha_ids(val_groups),
            "fit_ids_sha256": _sha_ids(ids_dev[fit_idx]), "validation_ids_sha256": _sha_ids(ids_dev[val_idx]),
            "fit_rows": len(fit_idx), "validation_rows": len(val_idx)})
    learn_manifest = json.loads((LEARNING_DIR / "manifest.json").read_text())
    selected_manifest = json.loads((SELECTED_DIR / "manifest.json").read_text())
    if learn_manifest["config"]["source_sha256"] != EXPECTED_SHA256 or selected_manifest["source_sha256"] != EXPECTED_SHA256:
        raise ValueError("Prior artifacts do not match official source")
    regimes = {name: list(groups_) for name, groups_ in learn_manifest["config"]["regime_groups"].items()}
    regimes["selected_v3"] = list(selected_manifest["selected_groups"])
    for name, groups_ in regimes.items():
        if not set(groups_) <= dev_groups:
            raise ValueError(f"Regime {name} contains non-development origins")
    config = {"schema": SCHEMA, "source_sha256": EXPECTED_SHA256, "source_bytes": data.stat().st_size,
        "source_rows": len(frame), "seed": seed, "git_commit": os.popen("git rev-parse HEAD").read().strip(),
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "scikit_learn": sklearn.__version__, "cpu_count": os.cpu_count(),
        "frozen_test_groups": sorted(frozen), "frozen_test_group_sha256": _sha_ids(frozen),
        "development_groups_sha256": _sha_ids(dev_groups), "development_rows": len(dev),
        "features": list(FEATURES), "feature_sets": {k: list(v) for k,v in FEATURE_SETS.items()},
        "folds": fold_manifest, "regime_groups": regimes,
        "selected_v3_manifest_sha256": sha256_file(SELECTED_DIR / "manifest.json"),
        "learning_curve_manifest_sha256": sha256_file(LEARNING_DIR / "manifest.json"),
        "models": {"base_reused": ["dummy", "logistic_regression", "historical_rf", "extra_trees", "hist_gradient"],
                   "new": {"et_unweighted_tuned": {**BEST_ET_PARAMS, "class_weight": None},
                           "rf_unweighted": {"n_estimators": 40, "max_depth": 18,
                               "min_samples_leaf": 30, "class_weight": None,
                               "max_features": "sqrt", "n_jobs": 2},
                           "extra_trees_best": BEST_ET_PARAMS}},
        "resource_limits": {"min_available_ram_bytes": MIN_AVAILABLE_BYTES, "parallel_threads": 2,
                            "hist_gradient_max_fit_rows": HIST_MAX_FIT_ROWS}}
    config["runner_sha256"] = sha256_file(Path(__file__).resolve())
    # Resume only when the experiment definition is unchanged. Implementation
    # fixes to bounded postprocessing may reuse validated fit checkpoints.
    if prior and prior.get("config", {}).get("source_sha256") == EXPECTED_SHA256:
        config["runner_sha256"] = prior["config"].get("runner_sha256", config["runner_sha256"])
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest_path = out / "manifest.json"
    if prior and prior.get("config_key") != key:
        raise ValueError("Existing master manifest is incompatible; choose another output directory")
    manifest = prior or {"schema": SCHEMA, "config_key": key, "config": config,
                         "status": "prepared", "stages": {}, "failed_stages": {}}
    manifest["preflight"] = {"resources": _resource_snapshot(),
        "source": {"path": str(data.resolve()), "sha256": EXPECTED_SHA256,
                   "rows": len(frame), "groups": len(set(groups))},
        "original_baseline_artifact_loadable": _baseline_artifact_check(),
        "selected_v3_chunks_verified": _verified_selected_chunks(),
        "learning_curve_stage_count": len(learn_manifest["stages"]),
        "learning_curve_manifest_status": learn_manifest.get("status")}
    manifest["status"] = "running"
    _atomic_json(manifest_path, manifest)

    # Preserve previous evidence as inputs to the master report; do not rerun
    # these candidates on the frozen test.
    _write_reused_evidence(out, manifest, learn_manifest, selected_manifest)

    selectors = _make_selectors(dev, regimes["representative_200000"], seed=seed)
    group_stats, _ = _profile_groups(dev)
    selector_summary = {}
    for name, chosen in selectors.items():
        total_n = sum(group_stats[g]["n"] for g in chosen)
        positives = sum(group_stats[g]["positive"] for g in chosen)
        selector_summary[name] = {"groups": chosen, "rows": total_n,
                                  "positive_count": positives,
                                  "detected_rate": positives / total_n,
                                  "group_sha256": _sha_ids(chosen)}
    manifest["selector_memberships"] = selector_summary
    _atomic_json(manifest_path, manifest)

    # Use selected-v3 saved chunks directly; this validates and reuses them.
    selected_frame = _load_selected(SELECTED_DIR)
    if len(selected_frame) != 125013 or not set(selected_frame[GROUP_COLUMN].astype(str)) <= dev_groups:
        raise ValueError("Verified selected-v3 chunk contents do not match the recorded subset")
    manifest["selected_v3_reused"] = {"rows": len(selected_frame),
        "groups": int(selected_frame[GROUP_COLUMN].nunique()),
        "chunks_verified": len(selected_manifest.get("chunks", {})),
        "source_sha256": selected_manifest["source_sha256"]}
    _atomic_json(manifest_path, manifest)

    # Evaluate selection alternatives and feature ablations on representative
    # 200k using identical full-development validation origins.
    if phase in ("all", "selection", "features"):
        manifest["status"] = "feature_and_selection_screen"
        _atomic_json(manifest_path, manifest)
        manifest["feature_screen"] = _feature_set_experiments(out, manifest, key,
            x_dev, dev, y_dev, g_dev, ids_dev, folds,
            regimes["representative_200000"], seed=seed)
        _atomic_json(manifest_path, manifest)
        manifest["selector_screen"] = _candidate_selector_experiments(out, manifest, key,
            x_dev, y_dev, g_dev, ids_dev, folds, selectors, seed=seed)
        _atomic_json(manifest_path, manifest)

    # Model/weight comparisons on all development folds plus tuned selected-v3.
    if phase in ("all", "models"):
        model_runs = {}
        experiments = [
            ("et_unweighted_tuned", "all_development", BEST_ET_PARAMS),
            ("rf_unweighted", "all_development", None),
            ("extra_trees_best", "selected_v3", BEST_ET_PARAMS),
        ]
        for model_name, regime_name, params in experiments:
            group_set = set(regimes[regime_name])
            for fold_idx, (fit_idx, val_idx) in enumerate(folds, 1):
                fit_groups = set(g_dev[fit_idx]) & group_set
                _validate_membership(fit_groups, set(g_dev[val_idx]), frozen)
                mask = np.isin(g_dev, list(fit_groups))
                xx = x_dev.loc[:, list(FEATURES)].copy()
                xx.attrs["n_groups"] = len(fit_groups)
                result = _run_fold(out, manifest, key,
                    stage=f"model_{model_name}_{regime_name}_fold{fold_idx}",
                    model_name=model_name, feature_set="full",
                    x_train=xx.loc[mask], y_train=y_dev[mask], x_val=xx.iloc[val_idx],
                    y_val=y_dev[val_idx], val_ids=ids_dev[val_idx],
                    val_groups=g_dev[val_idx], model_params=params,
                    seed=seed+fold_idx)
                model_runs.setdefault(f"{model_name}:{regime_name}", []).append(result)
                print(f"model {model_name}/{regime_name} fold {fold_idx}: {result.get('status')} macro_f1={result.get('macro_f1', float('nan')):.4f}", flush=True)
                del xx
                gc.collect()
        manifest["new_model_comparisons"] = model_runs
        _atomic_json(manifest_path, manifest)

    # Full-development out-of-fold predictions support error analysis and a
    # nested threshold check. This phase never reads frozen test outcomes.
    if phase in ("all", "oof"):
        manifest["status"] = "development_oof"
        _atomic_json(manifest_path, manifest)
        oof = _oof_and_thresholds(out, manifest, key, x_dev, dev, y_dev, g_dev,
            ids_dev, folds, seed=seed)
        manifest["oof_summary"] = oof
        _atomic_json(manifest_path, manifest)

    model_metadata = _prepare_final_model(data, out, seed=seed)
    summary = _aggregate_master(out, manifest, learn_manifest, selected_manifest,
                                selector_summary)
    summary["best_model_metadata"] = model_metadata
    summary["inference_command"] = (
        "python -m experiments.qmutbench_master_optimization predict "
        "--input INPUT.csv --output PREDICTIONS.csv")
    _atomic_json(out / "metrics.json", summary)
    manifest["summary"] = summary
    manifest["status"] = "complete_with_resource_skips" if any(
        stage.get("status", "").startswith("skipped") for stage in manifest["stages"].values()) else "complete"
    manifest["metrics_sha256"] = sha256_file(out / "metrics.json")
    _atomic_json(manifest_path, manifest)
    return summary


def _baseline_artifact_check() -> dict:
    model_path = Path("data/qmutbench_ml/outcome_model_v1/qmutbench_outcome_model.joblib")
    prediction_path = Path("data/qmutbench_ml/outcome_model_v1/heldout_predictions.csv")
    metrics_path = Path("data/qmutbench_ml/outcome_model_v1/metrics.json")
    result = {"model_exists": model_path.is_file(), "predictions_exist": prediction_path.is_file(),
              "metrics_exist": metrics_path.is_file()}
    if model_path.is_file():
        model = joblib.load(model_path)
        result["model_loadable"] = hasattr(model, "predict")
        result["model_sha256"] = sha256_file(model_path)
    if prediction_path.is_file() and metrics_path.is_file():
        recorded = json.loads(metrics_path.read_text())["final_test"]
        measured = evaluate_saved_predictions(prediction_path)["metrics"]
        result["saved_prediction_metrics_match"] = all(abs(measured[k]-recorded[k]) < 1e-12
            for k in ("accuracy", "balanced_accuracy", "macro_f1"))
        result["saved_prediction_sha256"] = sha256_file(prediction_path)
    return result


def _verified_selected_chunks() -> dict:
    m = json.loads((SELECTED_DIR / "manifest.json").read_text())
    complete = 0
    bad = []
    for name, entry in m.get("chunks", {}).items():
        p = SELECTED_DIR / "chunks" / name
        if p.is_file() and sha256_file(p) == entry.get("sha256"):
            complete += 1
        else:
            bad.append(name)
    return {"completed_verified": complete, "failed": bad, "rows": m.get("selected_rows"),
            "groups": m.get("selected_origin_groups"), "source_sha256": m.get("source_sha256")}


def _write_reused_evidence(out: Path, manifest: dict, learning: dict, selected: dict) -> None:
    # Write compact, provenance-linked extracts instead of copying giant files.
    baseline = json.loads(Path("data/qmutbench_ml/outcome_model_v1/metrics.json").read_text())
    selected_metrics = json.loads((SELECTED_DIR / "experiment" / "metrics.json").read_text())
    learning_metrics = json.loads((LEARNING_DIR / "metrics.json").read_text())
    evidence = {"baseline_metrics_sha256": sha256_file(Path("data/qmutbench_ml/outcome_model_v1/metrics.json")),
        "baseline_metrics": baseline, "selected_v3_metrics_sha256": sha256_file(SELECTED_DIR / "experiment" / "metrics.json"),
        "selected_v3_metrics": selected_metrics, "learning_curve_metrics_sha256": sha256_file(LEARNING_DIR / "metrics.json"),
        "learning_curve_metrics": learning_metrics,
        "learning_curve_manifest_sha256": sha256_file(LEARNING_DIR / "manifest.json"),
        "selected_v3_manifest_sha256": sha256_file(SELECTED_DIR / "manifest.json")}
    _atomic_json(out / "reused_historical_evidence.json", evidence)
    manifest["reused_evidence"] = {"path": str(out / "reused_historical_evidence.json"),
                                   "sha256": sha256_file(out / "reused_historical_evidence.json")}
    _atomic_json(out / "manifest.json", manifest)


def _aggregate_master(out: Path, manifest: dict, learning: dict,
                      selected: dict, selector_summary: dict) -> dict:
    selected_metrics = json.loads((SELECTED_DIR / "experiment" / "metrics.json").read_text())
    metrics_path = out / "metrics.json"
    existing = {}
    for name, group_set in selector_summary.items():
        existing[name] = {"rows": group_set["rows"], "groups": len(group_set["groups"]),
                          "detected_rate": group_set["detected_rate"]}
    return {"schema": SCHEMA, "source_sha256": EXPECTED_SHA256,
        "config_key": manifest["config_key"], "validation_scope": "fixed 3-fold StratifiedGroupKFold over the 305 development origins; frozen 70 test origins excluded",
        "data_regime_comparison": learning["summary"],
        "historical_selected_v3": {"record_count": 125013, "origin_groups": 86,
            "own_split_cv": selected_metrics["validation"][selected_metrics["selected_model"]],
            "common_fold_cv": learning["summary"]["regimes"]["selected_v3"]},
        "selector_comparison": existing,
        "new_feature_screen": manifest.get("feature_screen", {}),
        "selector_validation": manifest.get("selector_screen", {}),
        "new_model_comparisons": manifest.get("new_model_comparisons", {}),
        "development_oof_analysis": manifest.get("oof_summary", {}),
        "frozen_test_evaluation": "not performed in this experiment; prior frozen-test metrics remain exploratory and unchanged",
        "model_path": str((out / "best_model.joblib").resolve()) if (out / "best_model.joblib").is_file() else None,
        "artifact_manifest": str((out / "manifest.json").resolve())}


def _prepare_final_model(data: Path, out: Path, *, seed: int = SEED) -> dict:
    """Fit or reuse the selected full-development model without test access."""
    prior_metrics = json.loads((LEARNING_DIR / "metrics.json").read_text())
    prior_model = Path(prior_metrics["final_development_model"]["path"])
    if not prior_model.is_file() or sha256_file(prior_model) != prior_metrics["final_development_model"]["sha256"]:
        raise ValueError("Prior development-selected best model is missing or checksum-invalid")
    model = joblib.load(prior_model)
    if not hasattr(model, "predict_proba") or not hasattr(model, "named_steps"):
        raise ValueError("Prior winner is not a prediction pipeline")
    dest = out / "best_model.joblib"
    if not dest.is_file() or sha256_file(dest) != sha256_file(prior_model):
        tmp = dest.with_name(dest.name + f".{os.getpid()}.tmp")
        shutil.copyfile(prior_model, tmp)
        os.replace(tmp, dest)
    metadata = {"source_model_path": str(prior_model), "source_model_sha256": sha256_file(prior_model),
        "model_path": str(dest.resolve()), "model_sha256": sha256_file(dest),
        "feature_schema": list(FEATURES), "target_map": TARGET_MAP,
        "training_data": "all 578,117 records across the 305 development Origin_program groups",
        "model_parameters": model.named_steps["model"].get_params(),
        "inference_threshold": 0.485,
        "note": "Reused the already fitted development-only winner from the completed learning-curve run; no frozen test used.",
        "sklearn_version": sklearn.__version__}
    _atomic_json(out / "best_model_metadata.json", metadata)
    return metadata


def predict(input_path: Path, output_path: Path, *, model_path: Path = OUT / "best_model.joblib",
            threshold: float | None = None) -> Path:
    frame = pd.read_csv(input_path, low_memory=False)
    if TARGET_COLUMN in frame.columns:
        raise ValueError("Prediction input must not contain the Killed target")
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Input missing required training features: {missing}")
    x = frame.loc[:, list(FEATURES)].copy()
    for col in NUMERIC:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    model = joblib.load(model_path)
    probability = model.predict_proba(x)[:, 1]
    metadata_path = Path(model_path).with_name("best_model_metadata.json")
    if threshold is None and metadata_path.is_file():
        threshold = float(json.loads(metadata_path.read_text(encoding="utf-8")).get("inference_threshold", 0.5))
    cut = 0.5 if threshold is None else threshold
    labels = (probability >= cut).astype(np.int8)
    result = pd.DataFrame({"predicted_outcome": np.where(labels == 1, "Detected_WOO_or_OPO", "Survivor"),
                           "probability_detected": probability})
    if GROUP_COLUMN in frame:
        result.insert(0, GROUP_COLUMN, frame[GROUP_COLUMN].astype(str))
    _atomic_csv(result, output_path)
    return output_path


def status(out: Path = OUT) -> dict:
    manifest = json.loads((out / "manifest.json").read_text())
    complete, invalid = 0, []
    for name, stage in manifest.get("stages", {}).items():
        rp = Path(stage.get("result_path", ""))
        if rp.is_file() and stage.get("result_sha256") == sha256_file(rp):
            pp = stage.get("predictions_path")
            if pp and (not Path(pp).is_file() or stage.get("predictions_sha256") != sha256_file(Path(pp))):
                invalid.append(name)
            else:
                complete += 1
        elif stage.get("status") == "complete":
            invalid.append(name)
    return {"status": manifest.get("status"), "verified_completed_stages": complete,
        "invalid_or_missing_stages": invalid, "failed_stages": manifest.get("failed_stages", {}),
        "manifest": str((out / "manifest.json").resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "predict"))
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--phase", choices=("all", "features", "selection", "models", "oof"), default="all")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    if args.command == "run":
        result = run(args.data, args.output_dir, phase=args.phase)
    elif args.command == "status":
        result = status(args.output_dir)
    else:
        if args.input is None or args.output is None:
            parser.error("predict requires --input and --output")
        result = {"predictions": str(predict(args.input, args.output,
            model_path=args.model or (args.output_dir / "best_model.joblib"), threshold=args.threshold))}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
