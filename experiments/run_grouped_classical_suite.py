"""Broad matched classical and imbalance-method screen for binary task manifests.

Every transform and sampler is fit only within each training fold. Inputs are
compact task manifests with columns record_id, bug_id, label, and code.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ["QML_CLASSICAL_CONFIG_PATH"]) if os.environ.get("QML_CLASSICAL_CONFIG_PATH") else None
if CONFIG_PATH:
    if not CONFIG_PATH.is_absolute():
        CONFIG_PATH = ROOT / CONFIG_PATH
    EXECUTION_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    TASKS = EXECUTION_CONFIG["tasks"]
    configured_output = os.environ.get("QML_CLASSICAL_OUTPUT", EXECUTION_CONFIG["output"])
    OUT = Path(configured_output)
    if not OUT.is_absolute():
        OUT = ROOT / OUT
else:
    EXECUTION_CONFIG = {}
    TASKS = json.loads(os.environ["QML_CLASSICAL_TASKS_JSON"])
    OUT = Path(os.environ["QML_CLASSICAL_OUTPUT"])
FLOOR_MIB = 500


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def atomic_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader(); writer.writerows(rows); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return sha256_file(path)


def smote_numpy(X, y, seed, strategy="smote"):
    """Random over/under and continuous-space SMOTE without external packages."""
    import numpy as np
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = int(counts.max()) if strategy != "random_under" else int(counts.min())
    out_x, out_y = [], []
    for cls, count in zip(classes, counts):
        ix = np.flatnonzero(y == cls)
        if strategy == "random_under":
            chosen = rng.choice(ix, size=target, replace=False)
            out_x.extend(X[chosen]); out_y.extend(y[chosen]); continue
        if strategy == "random_over":
            chosen = rng.choice(ix, size=target-count, replace=True) if target > count else np.array([], dtype=int)
            out_x.extend(X[ix]); out_y.extend(y[ix]); out_x.extend(X[chosen]); out_y.extend(y[chosen]); continue
        out_x.extend(X[ix]); out_y.extend(y[ix])
        if target <= count: continue
        if count < 2:
            raise ValueError(f"SMOTE needs at least two minority samples; class {cls} has {count}")
        minority = X[ix]
        dist = ((minority[:, None, :] - minority[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(dist, np.inf)
        k = min(5, count - 1)
        neighbors = np.argpartition(dist, kth=k-1, axis=1)[:, :k]
        for _ in range(target-count):
            base_i = int(rng.integers(count))
            near_i = int(neighbors[base_i, int(rng.integers(k))])
            fraction = float(rng.random())
            out_x.append(minority[base_i] + fraction * (minority[near_i] - minority[base_i]))
            out_y.append(cls)
    return np.asarray(out_x, dtype=float), np.asarray(out_y, dtype=int)


def metrics(y, pred):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "per_class": {str(c): {"precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c]), "support": int(s[c])} for c in (0, 1)}}


def estimator_suite(seed):
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                                  HistGradientBoostingClassifier, RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import LinearSVC, SVC
    from sklearn.tree import DecisionTreeClassifier
    return [
        ("dummy_prior", "none", DummyClassifier(strategy="prior")),
        ("dummy_stratified", "none", DummyClassifier(strategy="stratified", random_state=seed)),
        ("logistic_unweighted", "none", LogisticRegression(C=1.0, max_iter=500, random_state=seed)),
        ("logistic_weighted", "class_weight", LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=seed)),
        ("linear_svm_weighted", "class_weight", LinearSVC(C=1.0, class_weight="balanced", random_state=seed)),
        ("rbf_svm_weighted", "class_weight", SVC(C=1.0, kernel="rbf", class_weight="balanced", random_state=seed)),
        ("decision_tree_weighted", "class_weight", DecisionTreeClassifier(max_depth=6, min_samples_leaf=2, class_weight="balanced", random_state=seed)),
        ("random_forest_weighted", "class_weight", RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=2, class_weight="balanced", n_jobs=1, random_state=seed)),
        ("extra_trees_weighted", "class_weight", ExtraTreesClassifier(n_estimators=100, max_depth=8, min_samples_leaf=2, class_weight="balanced", n_jobs=1, random_state=seed)),
        ("gradient_boosting", "none", GradientBoostingClassifier(n_estimators=75, max_depth=2, learning_rate=.05, random_state=seed)),
        ("hist_gradient_boosting", "none", HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, l2_regularization=1.0, random_state=seed)),
        ("knn_5", "none", KNeighborsClassifier(n_neighbors=5, weights="distance")),
        ("gaussian_nb", "none", GaussianNB()),
        ("mlp", "class_weight_via_resampling", MLPClassifier(hidden_layer_sizes=(16,), alpha=.01, max_iter=200, early_stopping=False, random_state=seed)),
    ]


def main():
    if psutil.virtual_memory().available / 2**20 < 1200:
        raise MemoryError("Classical suite preflight needs 1,200 MiB available; floor is unchanged at 500 MiB")
    import gc
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import MinMaxScaler
    if psutil.virtual_memory().available / 2**20 < 500:
        raise MemoryError("Post-import available memory below 500 MiB")
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Classical suite output already exists at {OUT}; refusing to overwrite it")
    OUT.mkdir(parents=True, exist_ok=True)
    rows, fold_rows = [], []
    for task_i, task in enumerate(TASKS):
        if psutil.virtual_memory().available / 2**20 < FLOOR_MIB:
            raise MemoryError("Available RAM crossed the 500 MiB floor before a task")
        data_path = Path(task["input"])
        if task.get("sha256") and sha256_file(data_path) != task["sha256"]:
            raise ValueError(f"Task source hash changed for {task['name']}: {data_path}")
        df = pd.read_csv(data_path)
        required = {"record_id", "bug_id", "label", "code"}
        if not required.issubset(df.columns): raise ValueError(f"{data_path} missing {required-set(df.columns)}")
        if df.record_id.duplicated().any() or set(df.label.unique()) != {0, 1}:
            raise ValueError(f"Invalid duplicated IDs or class coverage in {data_path}")
        y = df.label.to_numpy(dtype=int)
        groups = df.bug_id.astype(str).to_numpy()
        splitter = GroupKFold(n_splits=5)
        predictions = {}
        started_task = time.perf_counter()
        process = psutil.Process(os.getpid())
        peak_rss_mib = process.memory_info().rss / 2**20
        for fold, (tr, va) in enumerate(splitter.split(df.code, y, groups), 1):
            if set(groups[tr]) & set(groups[va]): raise AssertionError("Group leakage")
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=512, min_df=1,
                sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")
            xtr0 = vectorizer.fit_transform(df.code.iloc[tr])
            xva0 = vectorizer.transform(df.code.iloc[va])
            svd = TruncatedSVD(n_components=4, n_iter=2, random_state=20261028+fold)
            xtr = svd.fit_transform(xtr0)
            xva = svd.transform(xva0)
            scaler = MinMaxScaler().fit(xtr)
            xtr, xva = scaler.transform(xtr), scaler.transform(xva)
            models = estimator_suite(20261028 + fold)
            for name, balance, model in models:
                if name not in predictions: predictions[name] = []
                before = time.perf_counter(); model.fit(xtr, y[tr])
                fit_seconds = time.perf_counter() - before
                before = time.perf_counter(); pred = model.predict(xva)
                inference_seconds = time.perf_counter() - before
                peak_rss_mib = max(peak_rss_mib, process.memory_info().rss / 2**20)
                pred = np.asarray(pred, dtype=int)
                predictions[name].extend({"record_id": str(df.record_id.iloc[ix]), "bug_id": str(groups[ix]),
                    "fold": fold, "y_true": int(y[ix]), "y_pred": int(pred[j])}
                    for j, ix in enumerate(va))
                fold_rows.append({"task": task["name"], "model": name, "fold": fold,
                    "balance_method": balance, "train_groups": len(set(groups[tr])),
                    "validation_groups": len(set(groups[va])), "train_records": len(tr),
                    "validation_records": len(va), "fit_seconds": fit_seconds,
                    "inference_seconds": inference_seconds,
                    **metrics(y[va], pred)})
            # Binary logistic and Extra Trees with training-only resampling methods.
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import ExtraTreesClassifier
            for sampler in ("random_over", "random_under", "smote"):
                sx, sy = smote_numpy(xtr, y[tr], 20261028 + fold, sampler)
                for name, estimator in (("logistic", LogisticRegression(C=1.0, max_iter=500, random_state=20261028+fold)),
                    ("extra_trees", ExtraTreesClassifier(n_estimators=100, max_depth=8, min_samples_leaf=2, n_jobs=1, random_state=20261028+fold))):
                    model_id = f"{name}_{sampler}"
                    model = estimator
                    before = time.perf_counter(); model.fit(sx, sy)
                    fit_seconds = time.perf_counter() - before
                    before = time.perf_counter(); pred = np.asarray(model.predict(xva), dtype=int)
                    inference_seconds = time.perf_counter() - before
                    peak_rss_mib = max(peak_rss_mib, process.memory_info().rss / 2**20)
                    predictions.setdefault(model_id, []).extend({"record_id": str(df.record_id.iloc[ix]), "bug_id": str(groups[ix]),
                        "fold": fold, "y_true": int(y[ix]), "y_pred": int(pred[j])}
                        for j, ix in enumerate(va))
                    fold_rows.append({"task": task["name"], "model": model_id, "fold": fold,
                        "balance_method": sampler, "train_groups": len(set(groups[tr])),
                        "validation_groups": len(set(groups[va])), "train_records": len(tr),
                        "validation_records": len(va), "resampled_train_records": len(sy),
                        "fit_seconds": fit_seconds, "inference_seconds":inference_seconds,
                        **metrics(y[va], pred)})
            # Save after each fold; interrupted campaigns retain real fold work.
            out = OUT / task["name"]
            out.mkdir(parents=True, exist_ok=True)
            for model_name, pred_rows in predictions.items():
                path = out / f"{model_name}_oof_partial.csv"
                with path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=["record_id","bug_id","fold","y_true","y_pred"])
                    w.writeheader(); w.writerows(pred_rows)
            if psutil.virtual_memory().available / 2**20 < FLOOR_MIB:
                raise MemoryError("Available RAM crossed 500 MiB after fold; partial predictions are saved")
            del xtr0, xva0, xtr, xva, sx, sy
            gc.collect()
        out = OUT / task["name"]
        with (out / "fold_metrics.json").open("w", encoding="utf-8") as f:
            json.dump([x for x in fold_rows if x["task"] == task["name"]], f, indent=2)
        for model_name, pred_rows in predictions.items():
            score = metrics(np.asarray([x["y_true"] for x in pred_rows]), np.asarray([x["y_pred"] for x in pred_rows]))
            rng = np.random.default_rng(20261028)
            group_map = {}
            for i, row in enumerate(pred_rows): group_map.setdefault(row["bug_id"], []).append(i)
            group_names = list(group_map)
            y_all = np.asarray([x["y_true"] for x in pred_rows]); p_all = np.asarray([x["y_pred"] for x in pred_rows])
            boot = []
            for _ in range(500):
                sampled = rng.choice(group_names, size=len(group_names), replace=True)
                ix = np.asarray([j for group in sampled for j in group_map[group]], dtype=int)
                boot.append(metrics(y_all[ix], p_all[ix])["macro_f1"])
            score["group_bootstrap_95ci_macro_f1"] = [float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]
            rows.append({"task": task["name"], "model": model_name, "records": len(pred_rows),
                "groups": len({x["bug_id"] for x in pred_rows}), "folds": len({x["fold"] for x in pred_rows}),
                "runtime_seconds": sum(x["fit_seconds"] for x in fold_rows if x["task"] == task["name"] and x["model"] == model_name),
                "inference_seconds": sum(x["inference_seconds"] for x in fold_rows if x["task"] == task["name"] and x["model"] == model_name),
                "balance_methods": sorted({x["balance_method"] for x in fold_rows if x["task"] == task["name"] and x["model"] == model_name}),
                **score})
            with (out / f"{model_name}_oof_predictions.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["record_id","bug_id","fold","y_true","y_pred"]); w.writeheader(); w.writerows(pred_rows)
        (out / "summary.json").write_text(json.dumps({"status":"completed_grouped_classical_suite",
            "task":task, "rows":len(df), "groups":int(df.bug_id.nunique()), "folds":5,
            "runtime_seconds":time.perf_counter()-started_task,
            "sampling_policy":"all sampling fitted only inside each training fold, after SVD and scaling",
            "max_observed_process_rss_mib":peak_rss_mib,
            "memory_measurement_note":"RSS sampled after model fits; maximum observed, not guaranteed OS peak",
            "production_model_modified":False,
            "frozen_test_accessed":False}, indent=2), encoding="utf-8")
        del df, predictions
        gc.collect()
    with (OUT / "classical_suite_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (OUT / "classical_suite_fold_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in fold_rows for k in r})); w.writeheader(); w.writerows(fold_rows)
    print(json.dumps({"status":"completed", "tasks":len(TASKS), "results":len(rows), "fold_rows":len(fold_rows),
        "available_ram_mib":psutil.virtual_memory().available/2**20, "floor_mib":FLOOR_MIB}, indent=2))


if __name__ == "__main__":
    main()
