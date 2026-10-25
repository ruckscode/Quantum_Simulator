"""Resumable development-only QMutBench grouped learning-curve experiment."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from experiments.qmutbench_outcome_ml import (
    CATEGORICAL, DEFAULT_DATA, DTYPES, EXPECTED_SHA256, FEATURES, GROUP_COLUMN,
    NUMERIC, SEED, TARGET_COLUMN, TARGET_MAP, group_holdout, load_dataset,
    make_pipeline, metrics, prepare_xy, sha256_file, group_bootstrap_intervals,
)
from experiments.qmutbench_selected_pipeline import _model as selected_model

SCHEMA = "qmutbench-learning-curve-v1"
OUT = Path("data/qmutbench_ml/learning_curve_v1")
SELECTED_V3 = Path("data/qmutbench_ml/selected_v3")
FOLDS = 3
SIZES = (125_000, 200_000, 300_000, 578_117)
MIN_AVAILABLE_BYTES = 450 * 1024**2
HIST_MAX_FIT_ROWS = 150_000
MODEL_CONFIGS = {
    "random_forest_original": {"n_estimators": 40, "max_depth": 18,
        "min_samples_leaf": 30, "class_weight": "balanced_subsample",
        "max_features": "sqrt", "n_jobs": 2},
    "extra_trees": {"n_estimators": 100, "max_depth": 18,
        "min_samples_leaf": 30, "class_weight": "balanced_subsample",
        "max_features": "sqrt", "n_jobs": 2},
    "hist_gradient": {"max_iter": 100, "learning_rate": 0.1,
        "max_leaf_nodes": 31, "l2_regularization": 1.0},
}
TUNING_CONFIGS = {
    "et_deeper_smaller_leaf": ("extra_trees", {"n_estimators": 100,
        "max_depth": 22, "min_samples_leaf": 15, "max_features": "sqrt",
        "class_weight": "balanced_subsample", "n_jobs": 2}),
    "et_unweighted_baseline_shape": ("extra_trees", {"n_estimators": 100,
        "max_depth": 18, "min_samples_leaf": 30, "max_features": "sqrt",
        "class_weight": None, "n_jobs": 2}),
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


def _atomic_npz(path: Path, **arrays) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    return sha256_file(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _sha_ids(values) -> str:
    payload = "\n".join(str(x) for x in sorted(values)).encode()
    return hashlib.sha256(payload).hexdigest()


def _group_order(dev: pd.DataFrame, seed: int) -> list[str]:
    """Round-robin algorithm/prevalence/size strata, then seeded order within cells."""
    rows = []
    for group, frame in dev.groupby(GROUP_COLUMN, observed=True, sort=False):
        labels = frame[TARGET_COLUMN].astype(str).map(TARGET_MAP).to_numpy(dtype=np.int8)
        rows.append({"group": str(group), "rows": len(frame),
                     "rate": float(labels.mean()),
                     "algorithm_group": str(frame["algorithm_group"].mode().iloc[0])})
    profile = pd.DataFrame(rows)
    profile["prevalence_bin"] = pd.qcut(profile.rate.rank(method="first"), 5, labels=False)
    profile["size_bin"] = pd.qcut(profile.rows.rank(method="first"), 4, labels=False)
    rng = np.random.default_rng(seed)
    profile["tie"] = rng.random(len(profile))
    cells = {}
    for key, frame in profile.groupby(["algorithm_group", "prevalence_bin", "size_bin"], sort=True):
        cells[key] = frame.sort_values(["tie", "group"]).group.tolist()
    order = []
    keys = list(cells)
    depth = 0
    while any(depth < len(cells[key]) for key in keys):
        for key in keys:
            if depth < len(cells[key]):
                order.append(cells[key][depth])
        depth += 1
    if len(order) != profile["group"].nunique():
        raise AssertionError("Representative group ordering lost or duplicated origins")
    return order


def _prefix_groups(order: list[str], sizes: list[int], row_counts: dict[str, int]) -> dict[str, list[str]]:
    result = {}
    running, rows, pos = [], 0, 0
    for target in sizes:
        while rows < target and pos < len(order):
            group = order[pos]
            running.append(group)
            rows += row_counts[group]
            pos += 1
        result[f"representative_{target}"] = list(running)
    result["all_development"] = list(order)
    return result


def _make_model(candidate: str, seed: int, params: dict | None = None):
    params = dict(params or {})
    if candidate == "random_forest_original":
        model = make_pipeline("random_forest", seed=seed)
        model.named_steps["model"].set_params(**params)
        return model
    if candidate == "extra_trees":
        model = selected_model("extra_trees", seed)
        model.named_steps["model"].set_params(**params)
        return model
    if candidate == "hist_gradient":
        numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                     dtype=np.float32)),
        ])
        pre = ColumnTransformer([("numeric", numeric, NUMERIC),
                                 ("categorical", categorical, CATEGORICAL)],
                                remainder="drop", sparse_threshold=0.0)
        estimator = HistGradientBoostingClassifier(**{**MODEL_CONFIGS["hist_gradient"],
                                                        **params, "random_state": seed})
        return Pipeline([("preprocess", pre), ("model", estimator)])
    raise ValueError(f"Unknown candidate: {candidate}")


class _PeakRSS:
    def __init__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.running = False

    def _sample(self):
        while self.running:
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(0.05)

    def __enter__(self):
        self.running = True
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.running = False
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def _regime_profile(dev: pd.DataFrame, selected_groups: list[str]) -> dict:
    selected = dev[dev[GROUP_COLUMN].astype(str).isin(selected_groups)]
    y = selected[TARGET_COLUMN].astype(str).map(TARGET_MAP).to_numpy(dtype=np.int8)
    sizes = selected.groupby(GROUP_COLUMN, observed=True).size()
    per_group_pos = selected.assign(_y=y).groupby(GROUP_COLUMN, observed=True)._y.mean()
    return {
        "pool_rows": int(len(selected)), "pool_groups": int(sizes.size),
        "binary_class_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "detected_rate": float(y.mean()),
        "origin_size_quantiles": {str(q): float(sizes.quantile(q)) for q in (0, .25, .5, .75, 1)},
        "origin_detected_rate_quantiles": {str(q): float(per_group_pos.quantile(q)) for q in (0, .25, .5, .75, 1)},
        "origins_with_both_classes": int(((per_group_pos > 0) & (per_group_pos < 1)).sum()),
        "category_counts": {col: int(selected[col].nunique()) for col in CATEGORICAL},
        "numeric_quantiles": {col: {str(q): float(selected[col].quantile(q))
            for q in (0, .25, .5, .75, 1)} for col in NUMERIC},
    }


def _metric_summary(folds: list[dict]) -> dict:
    values = ("accuracy", "balanced_accuracy", "macro_f1")
    return {f"mean_{key}": float(np.mean([row[key] for row in folds])) for key in values}


def _fast_group_bootstrap(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray,
                          *, seed: int, repetitions: int = 1000) -> dict:
    """Bootstrap grouped confusion counts without rebuilding row arrays per replicate."""
    unique = np.unique(groups)
    per_group = np.zeros((len(unique), 4), dtype=np.int64)
    for index, group in enumerate(unique):
        mask = groups == group
        actual, pred = y_true[mask], y_pred[mask]
        per_group[index] = [np.sum((actual == 0) & (pred == 0)),
                            np.sum((actual == 0) & (pred == 1)),
                            np.sum((actual == 1) & (pred == 0)),
                            np.sum((actual == 1) & (pred == 1))]
    rng = np.random.default_rng(seed)
    sampled = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)),
                              size=repetitions)
    tn, fp, fn, tp = (sampled @ per_group).T.astype(np.float64)
    total = tn + fp + fn + tp
    recall_0 = tn / np.maximum(tn + fp, 1)
    recall_1 = tp / np.maximum(tp + fn, 1)
    precision_0 = tn / np.maximum(tn + fn, 1)
    precision_1 = tp / np.maximum(tp + fp, 1)
    f1_0 = 2 * precision_0 * recall_0 / np.maximum(precision_0 + recall_0, 1e-15)
    f1_1 = 2 * precision_1 * recall_1 / np.maximum(precision_1 + recall_1, 1e-15)
    samples = {"accuracy": (tn + tp) / np.maximum(total, 1),
               "balanced_accuracy": (recall_0 + recall_1) / 2,
               "macro_f1": (f1_0 + f1_1) / 2}
    return {key: [float(np.quantile(values, .025)), float(np.quantile(values, .975))]
            for key, values in samples.items()}


def _fit_one(candidate: str, x_train: pd.DataFrame, y_train: np.ndarray,
             x_val: pd.DataFrame, y_val: np.ndarray, *, seed: int,
             model_params: dict | None = None) -> tuple[dict, object]:
    before_available = psutil.virtual_memory().available
    if before_available < MIN_AVAILABLE_BYTES:
        return {"status": "skipped_resource", "reason": "available memory below 450 MiB",
                "available_before_bytes": before_available}, None
    start = time.perf_counter()
    model = _make_model(candidate, seed, model_params)
    with _PeakRSS() as memory:
        with threadpool_limits(limits=2):
            model.fit(x_train, y_train)
            pred = model.predict(x_val)
    result = {"status": "complete", "candidate": candidate,
              "model_params": dict(model_params or MODEL_CONFIGS[candidate]),
              "fit_seconds": time.perf_counter() - start,
              "peak_process_rss_bytes": int(memory.peak),
              "available_before_bytes": int(before_available),
              **metrics(y_val, pred)}
    del pred
    return result, model


def _folds_for_dev(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int):
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed + 1)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def run(data: Path = DEFAULT_DATA, out: Path = OUT, *, seed: int = SEED,
        selected_v3_dir: Path = SELECTED_V3, max_regime: str = "all") -> dict:
    start = time.perf_counter()
    dataset_hash = sha256_file(data)
    if dataset_hash != EXPECTED_SHA256:
        raise ValueError("Official source hash does not match the pinned source")
    selected_manifest_path = selected_v3_dir / "manifest.json"
    selected_manifest_hash = sha256_file(selected_manifest_path)
    selected_manifest = json.loads(selected_manifest_path.read_text(encoding="utf-8"))
    manifest_path = out / "manifest.json"
    prior = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if (max_regime == "all" and prior.get("status") in {"complete", "complete_with_resource_skips"}
            and prior.get("config", {}).get("source_sha256") == dataset_hash
            and prior.get("config", {}).get("seed") == seed
            and prior.get("config", {}).get("runner_sha256") == sha256_file(Path(__file__).resolve())
            and prior.get("config", {}).get("selected_v3_manifest_sha256") == selected_manifest_hash):
        artifacts = prior.get("final_artifacts", {})
        paths = [Path(artifacts.get(k, {}).get("path", "")) for k in
                 ("best_model", "test_diagnostic", "metrics")]
        if all(p.is_file() and artifacts[k]["sha256"] == sha256_file(p)
               for p, k in zip(paths, ("best_model", "test_diagnostic", "metrics"))):
            return json.loads(paths[2].read_text(encoding="utf-8"))
    frame = load_dataset(data, expected_source=False)
    x, y, groups = prepare_xy(frame)
    train_idx, test_idx = group_holdout(y, groups, seed=seed)
    test_groups = set(groups[test_idx])
    dev = frame.iloc[train_idx].reset_index(drop=True)
    x_dev, y_dev, groups_dev = prepare_xy(dev)
    dev_groups = set(groups_dev)
    if dev_groups & test_groups:
        raise AssertionError("Frozen test origins entered development data")
    fold_pairs = _folds_for_dev(x_dev, y_dev, groups_dev, seed)
    fold_groups = []
    for fit_idx, val_idx in fold_pairs:
        fit_groups, val_groups = set(groups_dev[fit_idx]), set(groups_dev[val_idx])
        if fit_groups & val_groups or (fit_groups | val_groups) & test_groups:
            raise AssertionError("Leakage across fixed development folds or frozen test")
        fold_groups.append({"fit": sorted(fit_groups), "validation": sorted(val_groups),
                            "fit_rows": int(len(fit_idx)), "validation_rows": int(len(val_idx))})

    if selected_manifest.get("source_sha256") != dataset_hash:
        raise ValueError("selected-v3 artifact was generated from a different source")
    v3_groups = set(selected_manifest["selected_groups"])
    if not v3_groups <= dev_groups:
        raise ValueError("selected-v3 contains origins outside frozen development partition")
    representative_order = _group_order(dev, seed)
    row_counts = dev.groupby(GROUP_COLUMN, observed=True).size().astype(int).to_dict()
    row_counts = {str(k): int(v) for k, v in row_counts.items()}
    representative = _prefix_groups(representative_order, list(SIZES[:-1]), row_counts)
    regime_groups = {"selected_v3": sorted(v3_groups), **representative}
    # Full source development pool is 578,117 rows; regime size refers to the
    # nested complete-origin pool before each fold's fixed validation groups.
    regime_sizes = {name: int(sum(row_counts[g] for g in set(gs)))
                    for name, gs in regime_groups.items()}
    selected_prefixes = [set(regime_groups[f"representative_{n}"]) for n in SIZES[:-1]]
    if not all(a <= b for a, b in zip(selected_prefixes, selected_prefixes[1:])):
        raise AssertionError("Representative learning-curve samples are not nested")

    config = {"schema": SCHEMA, "source_sha256": dataset_hash, "seed": seed,
              "runner_sha256": sha256_file(Path(__file__).resolve()),
              "selected_v3_manifest_sha256": selected_manifest_hash,
              "feature_schema": FEATURES, "target_map": TARGET_MAP,
              "frozen_test_groups_sha256": _sha_ids(test_groups),
              "development_groups_sha256": _sha_ids(dev_groups),
              "folds": fold_groups, "regime_groups": {k: sorted(v) for k, v in regime_groups.items()},
              "regime_pool_rows": regime_sizes,
              "candidates": ["random_forest_original", "extra_trees", "hist_gradient"],
              "model_configs": MODEL_CONFIGS, "tuning_configs": TUNING_CONFIGS,
              "preprocessing": "median numeric imputation; most-frequent categorical imputation; one-hot handle_unknown=ignore; fold-train fit only",
              "hist_gradient_max_fit_rows": HIST_MAX_FIT_ROWS}
    config_key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if prior and prior.get("config_key") != config_key:
        raise ValueError("Existing learning-curve manifest has incompatible source, splits, or configuration")
    manifest = prior or {"schema": SCHEMA, "config_key": config_key, "config": config,
                         "status": "prepared", "stages": {}, "failed_stages": {}}
    out.mkdir(parents=True, exist_ok=True)

    # Persist exact source row IDs and groups for each fixed fold/regime.
    membership_dir = out / "membership"
    global_source_ids = dev["Unnamed: 0"].to_numpy(dtype=np.int64)
    for regime, selected in regime_groups.items():
        for fold_number, (fit_idx, val_idx) in enumerate(fold_pairs, 1):
            fit_group_set, val_group_set = set(groups_dev[fit_idx]), set(groups_dev[val_idx])
            regime_train_groups = set(selected) & fit_group_set
            if regime_train_groups & val_group_set or regime_train_groups & test_groups:
                raise AssertionError("Regime membership leaks across folds")
            train_mask = np.isin(groups_dev, list(regime_train_groups))
            val_mask = np.isin(groups_dev, list(val_group_set))
            dest = membership_dir / f"{regime}_fold_{fold_number}.npz"
            expected = {"train_ids_sha256": _sha_ids(global_source_ids[train_mask]),
                        "validation_ids_sha256": _sha_ids(global_source_ids[val_mask]),
                        "train_groups": sorted(regime_train_groups),
                        "validation_groups": sorted(val_group_set),
                        "train_rows": int(train_mask.sum()), "validation_rows": int(val_mask.sum())}
            key = f"membership:{regime}:fold_{fold_number}"
            saved = manifest["stages"].get(key, {})
            if dest.is_file() and saved.get("sha256") == sha256_file(dest):
                continue
            digest = _atomic_npz(dest, train_record_ids=global_source_ids[train_mask],
                validation_record_ids=global_source_ids[val_mask],
                train_groups=np.array(sorted(regime_train_groups), dtype=str),
                validation_groups=np.array(sorted(val_group_set), dtype=str))
            manifest["stages"][key] = {"status": "complete", "path": str(dest),
                                       "sha256": digest, **expected}
            _atomic_json(manifest_path, manifest)

    profiles = {name: _regime_profile(dev, list(selected))
                for name, selected in regime_groups.items()}
    manifest["data_profiles"] = profiles
    manifest["status"] = "running"
    manifest["resource_preflight"] = {
        "available_ram_bytes_after_dataset_load": int(psutil.virtual_memory().available),
        "total_ram_bytes": int(psutil.virtual_memory().total),
        "free_disk_bytes": int(__import__("shutil").disk_usage(out).free),
        "hist_gradient_policy": f"run only when fold training rows <= {HIST_MAX_FIT_ROWS}; dense one-hot pipeline",
    }
    _atomic_json(manifest_path, manifest)

    run_regime_names = [name for name in regime_groups
                        if max_regime == "all" or name == max_regime or name == "selected_v3"]
    total_stages = 0
    for regime in run_regime_names:
        selected = set(regime_groups[regime])
        for fit_idx, _ in fold_pairs:
            fit_groups = set(groups_dev[fit_idx]) & selected
            fit_rows = int(np.isin(groups_dev, list(fit_groups)).sum())
            total_stages += 2 + int(fit_rows <= HIST_MAX_FIT_ROWS)
    done_times = []
    for key, stage in manifest["stages"].items():
        if not (key.startswith("fit:") and stage.get("status") == "complete"
                and key.split(":")[1] in run_regime_names):
            continue
        result_file = Path(stage.get("path", ""))
        if result_file.is_file() and stage.get("sha256") == sha256_file(result_file):
            done_times.append(stage.get("fit_seconds", 0))
    stage_count = len(done_times)
    for regime_index, (regime, selected) in enumerate(regime_groups.items()):
        if max_regime != "all" and regime != max_regime and regime != "selected_v3":
            continue
        for fold_number, (fit_idx, val_idx) in enumerate(fold_pairs, 1):
            fit_groups = set(groups_dev[fit_idx]) & set(selected)
            if not fit_groups:
                raise ValueError(f"No training origins for {regime}, fold {fold_number}")
            train_mask = np.isin(groups_dev, list(fit_groups))
            x_train, y_train = x_dev.loc[train_mask], y_dev[train_mask]
            x_val, y_val = x_dev.iloc[val_idx], y_dev[val_idx]
            if len(np.unique(y_train)) < 2:
                raise ValueError(f"One-class fit partition in {regime}, fold {fold_number}")
            candidate_names = ["random_forest_original", "extra_trees"]
            if len(train_mask.nonzero()[0]) <= HIST_MAX_FIT_ROWS:
                candidate_names.append("hist_gradient")
            else:
                for candidate in ("hist_gradient",):
                    key = f"fit:{regime}:fold_{fold_number}:{candidate}"
                    if key not in manifest["stages"]:
                        manifest["stages"][key] = {"status": "skipped_resource",
                            "reason": f"dense gradient boosting capped at {HIST_MAX_FIT_ROWS} fit rows",
                            "training_rows": int(train_mask.sum())}
            for candidate in candidate_names:
                key = f"fit:{regime}:fold_{fold_number}:{candidate}"
                result_path = out / "fold_results" / f"{regime}_fold_{fold_number}_{candidate}.json"
                saved = manifest["stages"].get(key, {})
                if result_path.is_file():
                    try:
                        cached = json.loads(result_path.read_text(encoding="utf-8"))
                        checksum_matches = saved.get("sha256") == sha256_file(result_path)
                        compatible_orphan = cached.get("config_key") == config_key and cached.get("status") == "complete"
                        if checksum_matches or compatible_orphan:
                            if not checksum_matches:
                                manifest["stages"][key] = {"status": "complete", "path": str(result_path),
                                    "sha256": sha256_file(result_path), "fit_seconds": cached["fit_seconds"],
                                    "recovered_orphan_checkpoint": True}
                                manifest["failed_stages"].pop(key, None)
                                done_times.append(cached["fit_seconds"])
                                stage_count += 1
                                _atomic_json(manifest_path, manifest)
                            continue
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
                stage_count += 1
                average = float(np.mean(done_times)) if done_times else 8.0
                remain = max(total_stages - stage_count, 0)
                print(f"[{stage_count}/{total_stages}] {regime} fold {fold_number}/{FOLDS} {candidate}; estimated remaining {remain*average:.0f}s", flush=True)
                try:
                    result, fitted = _fit_one(candidate, x_train, y_train, x_val, y_val,
                                              seed=seed + fold_number)
                    result.update({"config_key": config_key, "regime": regime,
                        "candidate": candidate, "fold": fold_number,
                        "train_rows": int(len(y_train)), "validation_rows": int(len(y_val)),
                        "train_groups": int(len(fit_groups)),
                        "validation_groups": int(len(set(groups_dev[val_idx]))),
                        "train_group_sha256": _sha_ids(fit_groups),
                        "validation_group_sha256": _sha_ids(set(groups_dev[val_idx])),
                        "train_record_ids_sha256": _sha_ids(global_source_ids[train_mask]),
                        "validation_record_ids_sha256": _sha_ids(global_source_ids[val_idx]),
                        "features": FEATURES})
                    if result["status"] == "complete":
                        _atomic_json(result_path, result)
                        manifest["stages"][key] = {"status": "complete", "path": str(result_path),
                                                    "sha256": sha256_file(result_path),
                                                    "fit_seconds": result["fit_seconds"]}
                        done_times.append(result["fit_seconds"])
                    else:
                        manifest["stages"][key] = result
                    manifest["failed_stages"].pop(key, None)
                    del fitted
                    gc.collect()
                except Exception as exc:
                    manifest["failed_stages"][key] = {"type": type(exc).__name__, "message": str(exc)}
                    manifest["stages"][key] = {"status": "failed", **manifest["failed_stages"][key]}
                _atomic_json(manifest_path, manifest)
    result_summary = _aggregate(out, manifest, regime_groups.keys())
    manifest["summary"] = result_summary
    if max_regime == "all" and not manifest.get("failed_stages"):
        return _tune_and_finalize(out, manifest, config_key, regime_groups, fold_pairs,
            x_dev, y_dev, groups_dev, dev, frame, test_idx, result_summary,
            seed=seed, run_started=start)
    manifest["status"] = "partial"
    manifest["elapsed_seconds"] = time.perf_counter() - start
    _atomic_json(out / "metrics.json", result_summary)
    _atomic_json(manifest_path, manifest)
    return result_summary


def _aggregate(out: Path, manifest: dict, regimes) -> dict:
    result = {"schema": SCHEMA, "source_sha256": manifest["config"]["source_sha256"],
              "config_key": manifest["config_key"], "seed": manifest["config"]["seed"],
              "pool_rows": manifest["config"]["regime_pool_rows"], "profiles": manifest.get("data_profiles", {}),
              "fold_groups": manifest["config"]["folds"], "regimes": {}}
    for regime in regimes:
        candidates = set()
        for fold in range(1, FOLDS + 1):
            prefix = f"fit:{regime}:fold_{fold}:"
            candidates.update(k[len(prefix):] for k in manifest["stages"] if k.startswith(prefix))
        result["regimes"][regime] = {}
        for candidate in sorted(candidates):
            rows = []
            for fold in range(1, FOLDS + 1):
                path = out / "fold_results" / f"{regime}_fold_{fold}_{candidate}.json"
                if path.is_file():
                    rows.append(json.loads(path.read_text(encoding="utf-8")))
            if rows and len(rows) == FOLDS:
                result["regimes"][regime][candidate] = {
                    **_metric_summary(rows), "folds": rows,
                    "total_fit_seconds": float(sum(r["fit_seconds"] for r in rows)),
                    "max_peak_process_rss_bytes": int(max(r["peak_process_rss_bytes"] for r in rows)),
                }
            else:
                result["regimes"][regime][candidate] = {"status": "incomplete",
                    "completed_folds": len(rows),
                    "skipped_or_failed": [manifest["stages"].get(f"fit:{regime}:fold_{fold}:{candidate}", {})
                                            for fold in range(1, FOLDS+1) if not (out / "fold_results" / f"{regime}_fold_{fold}_{candidate}.json").is_file()]}
    return result


def _tune_and_finalize(out: Path, manifest: dict, config_key: str,
                       regime_groups: dict[str, list[str]], fold_pairs,
                       x_dev: pd.DataFrame, y_dev: np.ndarray, groups_dev: np.ndarray,
                       dev: pd.DataFrame, full_frame: pd.DataFrame, test_idx: np.ndarray,
                       base_summary: dict, *, seed: int, run_started: float) -> dict:
    """Tune the top two model families on validation, then evaluate one final model once."""
    best_by_family = {}
    for regime, candidates in base_summary["regimes"].items():
        for family, summary in candidates.items():
            if "mean_macro_f1" not in summary:
                continue
            current = best_by_family.get(family)
            score = (summary["mean_macro_f1"], summary["mean_balanced_accuracy"])
            if current is None or score > current["score"]:
                best_by_family[family] = {"regime": regime, "score": score,
                                          "summary": summary}
    tunable_families = {family for family, _ in TUNING_CONFIGS.values()}
    top_families = sorted((family for family in best_by_family if family in tunable_families),
        key=lambda f: best_by_family[f]["score"], reverse=True)[:2]
    tuning = {}
    total = sum(len([name for name, (family, _) in TUNING_CONFIGS.items() if family == f]) * FOLDS
                for f in top_families)
    completed_tuning = 0
    for family in top_families:
        regime = best_by_family[family]["regime"]
        selected = set(regime_groups[regime])
        variant_names = [name for name, (owner, _) in TUNING_CONFIGS.items() if owner == family]
        for variant in variant_names:
            rows = []
            model_family, params = TUNING_CONFIGS[variant]
            for fold_number, (fit_idx, val_idx) in enumerate(fold_pairs, 1):
                fit_groups = set(groups_dev[fit_idx]) & selected
                train_mask = np.isin(groups_dev, list(fit_groups))
                x_train, y_train = x_dev.loc[train_mask], y_dev[train_mask]
                x_val, y_val = x_dev.iloc[val_idx], y_dev[val_idx]
                key = f"tune:{regime}:fold_{fold_number}:{variant}"
                path = out / "tuning_results" / f"{regime}_fold_{fold_number}_{variant}.json"
                saved = manifest["stages"].get(key, {})
                cached = None
                if path.is_file():
                    try:
                        cached = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        cached = None
                compatible = cached and cached.get("config_key") == config_key and cached.get("model_params") == params
                if compatible and (saved.get("sha256") == sha256_file(path) or saved.get("status") != "complete"):
                    rows.append(cached)
                    completed_tuning += 1
                    if saved.get("status") != "complete":
                        manifest["stages"][key] = {"status": "complete", "path": str(path),
                            "sha256": sha256_file(path), "fit_seconds": cached["fit_seconds"],
                            "recovered_orphan_checkpoint": True}
                        _atomic_json(out / "manifest.json", manifest)
                    continue
                completed_tuning += 1
                print(f"[tune {completed_tuning}/{total}] {regime} {variant} fold {fold_number}/{FOLDS}; validation only", flush=True)
                try:
                    result, fitted = _fit_one(model_family, x_train, y_train, x_val, y_val,
                                              seed=seed + fold_number, model_params=params)
                    result.update({"config_key": config_key, "variant": variant,
                        "regime": regime, "fold": fold_number, "estimator_family": model_family,
                        "model_params": params, "train_rows": int(len(y_train)),
                        "validation_rows": int(len(y_val)), "train_groups": len(fit_groups),
                        "validation_groups": len(set(groups_dev[val_idx])),
                        "train_group_sha256": _sha_ids(fit_groups),
                        "validation_group_sha256": _sha_ids(set(groups_dev[val_idx])),
                        "train_record_ids_sha256": _sha_ids(dev.loc[train_mask, "Unnamed: 0"]),
                        "validation_record_ids_sha256": _sha_ids(dev.iloc[val_idx]["Unnamed: 0"]),
                        "features": FEATURES})
                    if result["status"] == "complete":
                        _atomic_json(path, result)
                        manifest["stages"][key] = {"status": "complete", "path": str(path),
                            "sha256": sha256_file(path), "fit_seconds": result["fit_seconds"]}
                        rows.append(result)
                    else:
                        manifest["stages"][key] = result
                    del fitted
                    gc.collect()
                except Exception as exc:
                    manifest["failed_stages"][key] = {"type": type(exc).__name__, "message": str(exc)}
                    manifest["stages"][key] = {"status": "failed", **manifest["failed_stages"][key]}
                _atomic_json(out / "manifest.json", manifest)
            if len(rows) == FOLDS:
                tuning[variant] = {"family": family, "regime": regime,
                    **_metric_summary(rows), "folds": rows,
                    "total_fit_seconds": float(sum(r["fit_seconds"] for r in rows)),
                    "max_peak_process_rss_bytes": int(max(r["peak_process_rss_bytes"] for r in rows))}

    choices = []
    for family, best in best_by_family.items():
        choices.append({"name": family, "family": family, "regime": best["regime"],
                        "params": MODEL_CONFIGS[family],
                        "mean_macro_f1": best["score"][0],
                        "mean_balanced_accuracy": best["score"][1]})
    for variant, result in tuning.items():
        family = result["family"]
        params = TUNING_CONFIGS[variant][1]
        choices.append({"name": variant, "family": family, "regime": result["regime"],
                        "params": params, "mean_macro_f1": result["mean_macro_f1"],
                        "mean_balanced_accuracy": result["mean_balanced_accuracy"]})
    if not choices:
        raise RuntimeError("No completed development validation candidate is available")
    selected_choice = max(choices, key=lambda c: (c["mean_macro_f1"], c["mean_balanced_accuracy"]))
    regime = selected_choice["regime"]
    fit_group_set = set(regime_groups[regime])
    fit_mask = groups_dev.astype(str)
    train_mask = np.isin(fit_mask, list(fit_group_set))
    x_train, y_train = x_dev.loc[train_mask], y_dev[train_mask]
    if selected_choice["family"] == "hist_gradient" and len(y_train) > HIST_MAX_FIT_ROWS * 1.7:
        choices = [c for c in choices if c["family"] != "hist_gradient"]
        selected_choice = max(choices, key=lambda c: (c["mean_macro_f1"], c["mean_balanced_accuracy"]))
        regime = selected_choice["regime"]
        fit_group_set = set(regime_groups[regime])
        train_mask = np.isin(groups_dev.astype(str), list(fit_group_set))
        x_train, y_train = x_dev.loc[train_mask], y_dev[train_mask]
    selection = {"purpose": "development validation only; frozen test excluded",
        "selected": selected_choice, "validation_candidates": choices,
        "tuning": tuning, "tuned_families": top_families}
    _atomic_json(out / "model_selection.json", selection)
    manifest["development_selection"] = selection
    _atomic_json(out / "manifest.json", manifest)

    model_path = out / "best_development_model.joblib"
    model_params = selected_choice["params"]
    final_key = hashlib.sha256(json.dumps({"config_key": config_key,
        "selected": selected_choice}, sort_keys=True).encode()).hexdigest()
    saved_final = manifest.get("final_fit", {})
    model = None
    if (model_path.is_file() and saved_final.get("config_key") == final_key
            and saved_final.get("sha256") == sha256_file(model_path)):
        model = joblib.load(model_path)
    else:
        if psutil.virtual_memory().available < MIN_AVAILABLE_BYTES:
            raise MemoryError("Insufficient available memory to fit final development-selected model")
        final_fit_start = time.perf_counter()
        model = _make_model(selected_choice["family"], seed, model_params)
        with _PeakRSS() as memory, threadpool_limits(limits=2):
            model.fit(x_train, y_train)
        final_fit_seconds = time.perf_counter() - final_fit_start
        tmp = model_path.with_name(model_path.name + f".{os.getpid()}.tmp")
        joblib.dump(model, tmp, compress=3)
        digest = sha256_file(tmp)
        os.replace(tmp, model_path)
        manifest["final_fit"] = {"config_key": final_key, "sha256": digest,
            "path": str(model_path), "train_rows": int(len(y_train)),
            "train_groups": int(len(fit_group_set)),
            "train_group_sha256": _sha_ids(fit_group_set),
            "estimator_parameters": model.named_steps["model"].get_params(),
            "scikit_learn_version": sklearn.__version__,
            "fit_seconds": final_fit_seconds,
            "peak_process_rss_bytes": int(memory.peak), "status": "complete"}
        _atomic_json(out / "manifest.json", manifest)

    # The test is evaluated only after all development-only choices are frozen.
    # A sidecar binds cached predictions to the chosen model and split.
    diagnostic_path = out / "test_diagnostic.json"
    prediction_path = out / "test_predictions.csv"
    model_sha = sha256_file(model_path)
    sidecar_path = out / "test_predictions_manifest.json"
    test_id = full_frame.iloc[test_idx]["Unnamed: 0"].to_numpy(dtype=np.int64)
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else {}
    if (diagnostic_path.is_file() and prediction_path.is_file()
            and sidecar.get("final_model_sha256") == model_sha
            and sidecar.get("test_record_ids_sha256") == _sha_ids(test_id)
            and sidecar.get("predictions_sha256") == sha256_file(prediction_path)):
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    else:
        if (prediction_path.is_file() and sidecar.get("final_model_sha256") == model_sha
                and sidecar.get("test_record_ids_sha256") == _sha_ids(test_id)
                and sidecar.get("predictions_sha256") == sha256_file(prediction_path)
                and sidecar.get("rows") == len(test_id)):
            predictions = pd.read_csv(prediction_path)
            if (len(predictions) != len(test_id)
                    or not np.array_equal(predictions.record_id.to_numpy(dtype=np.int64), test_id)):
                raise ValueError("Saved exploratory test predictions do not match frozen test record IDs")
            tg = predictions.origin_program.astype(str).to_numpy()
            ty = predictions.y_true.to_numpy(dtype=np.int8)
            pred = predictions.y_pred.to_numpy(dtype=np.int8)
            test_pred_sha = sha256_file(prediction_path)
        else:
            test_frame = full_frame.iloc[test_idx]
            tx, ty, tg = prepare_xy(test_frame)
            pred = model.predict(tx)
            predictions = pd.DataFrame({"record_id": test_frame["Unnamed: 0"].to_numpy(),
                "origin_program": tg, "y_true": ty, "y_pred": pred})
            _atomic_csv(predictions, prediction_path)
            test_pred_sha = sha256_file(prediction_path)
            sidecar_value = {"config_key": config_key, "final_model_sha256": model_sha,
                "test_record_ids_sha256": _sha_ids(test_id), "predictions_sha256": test_pred_sha,
                "rows": len(predictions), "groups": len(set(tg))}
            _atomic_json(sidecar_path, sidecar_value)
        group_accuracy = [float(np.mean(predictions.loc[predictions.origin_program == g, "y_true"].to_numpy()
                            == predictions.loc[predictions.origin_program == g, "y_pred"].to_numpy()))
                          for g in np.unique(tg)]
        diagnostic = {"status": "exploratory_diagnostic_only", "used_for_selection": False,
            "reason": "Frozen test has been reused in historical experiments; this run evaluates only the final development-selected candidate once.",
            "rows": len(predictions), "groups": len(np.unique(tg)),
            "group_overlap_with_development": len(set(tg) & set(groups_dev)),
            **metrics(ty, pred),
            "group_bootstrap_95_percentile_intervals": _fast_group_bootstrap(ty, pred, tg, seed=seed),
            "groupwise_accuracy": {"mean": float(np.mean(group_accuracy)),
                "median": float(np.median(group_accuracy)), "minimum": float(np.min(group_accuracy)),
                "maximum": float(np.max(group_accuracy))},
            "model": selected_choice, "model_sha256": model_sha,
            "predictions_sha256": test_pred_sha}
        _atomic_json(diagnostic_path, diagnostic)
    manifest["final_artifacts"] = {
        "best_model": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "test_diagnostic": {"path": str(diagnostic_path), "sha256": sha256_file(diagnostic_path)},
        "metrics": {"path": str(out / "metrics.json"), "sha256": ""}}
    total_fit_seconds = sum(v.get("total_fit_seconds", 0)
        for regimes in base_summary["regimes"].values() for v in regimes.values()
        if v.get("status") != "incomplete") + sum(v.get("total_fit_seconds", 0) for v in tuning.values())
    if manifest.get("final_fit"):
        total_fit_seconds += manifest["final_fit"].get("fit_seconds", 0)
    result = {**base_summary, "development_selection": selection,
              "final_development_model": manifest["final_fit"],
              "test_diagnostic": diagnostic,
              "runtime_seconds_this_invocation": float(time.perf_counter() - run_started),
              "summed_model_fit_seconds": float(total_fit_seconds)}
    _atomic_json(out / "metrics.json", result)
    manifest["final_artifacts"]["metrics"]["sha256"] = sha256_file(out / "metrics.json")
    manifest["status"] = "complete_with_resource_skips" if any(
        stage.get("status", "").startswith("skipped") for stage in manifest["stages"].values()) else "complete"
    manifest["elapsed_seconds"] = float(time.perf_counter() - run_started)
    _atomic_json(out / "manifest.json", manifest)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status"))
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--selected-v3-dir", type=Path, default=SELECTED_V3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-regime", default="all", help="Run all regimes or selected_v3/representative_200000, etc.")
    args = parser.parse_args()
    if args.command == "run":
        result = run(args.data, args.output_dir, seed=args.seed,
                     selected_v3_dir=args.selected_v3_dir, max_regime=args.max_regime)
    else:
        manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
        result = {"status": manifest.get("status"), "completed_stages": sum(v.get("status") == "complete" for v in manifest.get("stages", {}).values()),
                  "failed_stages": manifest.get("failed_stages", {}), "output_dir": str(args.output_dir.resolve())}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
