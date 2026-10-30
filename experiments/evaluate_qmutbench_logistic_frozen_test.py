"""One-time frozen-test evaluation after development tuning is locked."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from experiments.qmutbench_outcome_ml import CATEGORICAL, DEFAULT_DATA, EXPECTED_SHA256, FEATURES, NUMERIC, TARGET_MAP, sha256_file
from experiments.tune_qmutbench_logistic import OUT as TUNING_OUT, SEED, make_preprocessor
from qmutbench_qml.data.loader import sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = TUNING_OUT / "final_frozen_test"


def metrics(y, pred):
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "per_class": {str(label): {"precision": float(precision[i]), "recall": float(recall[i]),
                                        "f1": float(f1[i]), "support": int(support[i])}
                          for i, label in enumerate([0, 1])}}


def _bootstrap(y, predictions, groups, seed=SEED, repetitions=1000):
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(pd.unique(groups)), dtype=object)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    samples = {name: {metric: [] for metric in ("accuracy", "balanced_accuracy", "macro_f1")}
               for name in predictions}
    deltas = {metric: [] for metric in ("accuracy", "balanced_accuracy", "macro_f1")}
    for _ in range(repetitions):
        draw = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([members[g] for g in draw])
        for name, pred in predictions.items():
            result = metrics(y[idx], pred[idx])
            for metric in samples[name]:
                samples[name][metric].append(result[metric])
        names = list(predictions)
        a = metrics(y[idx], predictions[names[1]][idx])
        b = metrics(y[idx], predictions[names[0]][idx])
        for metric in deltas:
            deltas[metric].append(a[metric] - b[metric])
    interval = lambda values: [float(np.quantile(values, .025)), float(np.quantile(values, .975))]
    return ({name: {metric: interval(values) for metric, values in by_metric.items()} for name, by_metric in samples.items()},
            {metric: interval(values) for metric, values in deltas.items()})


def run():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (TUNING_OUT / "run_manifest.json").exists():
        raise FileNotFoundError("Development Logistic Regression tuning must be completed before frozen-test evaluation")
    tuning = json.loads((TUNING_OUT / "run_manifest.json").read_text(encoding="utf-8"))
    if tuning.get("frozen_test_read"):
        raise ValueError("Development selection manifest indicates a prior frozen-test read")
    source_hash = sha256_file(DEFAULT_DATA)
    if source_hash != EXPECTED_SHA256:
        raise ValueError("Source dataset checksum changed")
    oof = pd.read_csv(ROOT / "data/qmutbench_ml/master_optimization_v1/oof_predictions.csv",
                      usecols=["record_id", "origin_program", "y_true", "fold"])
    development_ids = set(oof.record_id.astype("int64"))
    dtypes = {"Origin_program": "category", "Killed": "category", **{c: "category" for c in CATEGORICAL},
              "qubits": "int16", "gates": "int32", "Position_int": "int32"}
    columns = ["Unnamed: 0", "Origin_program", "Killed", *FEATURES]
    development_parts, test_parts = [], []
    # This test-only feature/label read occurs only after the development grid has selected its candidate.
    for chunk in pd.read_csv(DEFAULT_DATA, usecols=columns, dtype=dtypes, chunksize=50000):
        chunk = chunk.rename(columns={"Unnamed: 0": "record_id", "Origin_program": "origin_program"})
        is_development = chunk.record_id.isin(development_ids)
        dev = chunk[is_development]
        test = chunk[~is_development]
        if not dev.empty:
            development_parts.append(dev)
        if not test.empty:
            test_parts.append(test)
    development = pd.concat(development_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    if len(development) != len(oof) or len(test) + len(development) != 723079:
        raise ValueError("Development/frozen-test row counts differ from the verified source split")
    dev_groups = set(development.origin_program.astype(str))
    test_groups = set(test.origin_program.astype(str))
    if dev_groups & test_groups or len(test_groups) != 70:
        raise ValueError("Frozen-test origin manifest is inconsistent or overlaps development origins")
    if set(test.Killed.dropna().unique()) - set(TARGET_MAP):
        raise ValueError("Unexpected frozen-test target label")
    y_train = development.Killed.map(TARGET_MAP).to_numpy(dtype=np.int8)
    y_test = test.Killed.map(TARGET_MAP).to_numpy(dtype=np.int8)
    if not np.array_equal(development.record_id.to_numpy(), oof.record_id.to_numpy()):
        # Chunk order must align exactly after sorting, before using saved development labels.
        development = development.sort_values("record_id").reset_index(drop=True)
        oof = oof.sort_values("record_id").reset_index(drop=True)
        y_train = development.Killed.map(TARGET_MAP).to_numpy(dtype=np.int8)
        if not np.array_equal(development.record_id.to_numpy(), oof.record_id.to_numpy()):
            raise ValueError("Development data failed record alignment to saved OOF manifest")
    if not np.array_equal(y_train, oof.y_true.to_numpy(dtype=np.int8)):
        raise ValueError("Development source labels disagree with the verified OOF manifest")
    test = test.reset_index(drop=True)
    y_test = test.Killed.map(TARGET_MAP).to_numpy(dtype=np.int8)
    preprocessor = make_preprocessor()
    with threadpool_limits(limits=1):
        x_train = preprocessor.fit_transform(development[FEATURES], y_train)
        x_test = preprocessor.transform(test[FEATURES])
    best = tuning["best_config"]
    configurations = [
        {"name": "baseline_liblinear_C1_balanced", "C": 1.0, "solver": "liblinear", "class_weight": "balanced"},
        {"name": "tuned_development_selected", "C": float(best["C"]), "solver": best["solver"],
         "class_weight": best["class_weight"]},
    ]
    predictions, summaries = {}, {}
    for config in configurations:
        model = LogisticRegression(C=config["C"], solver=config["solver"], class_weight=config["class_weight"],
                                   max_iter=500, tol=1e-4, random_state=SEED)
        with threadpool_limits(limits=1):
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
        predictions[config["name"]] = pred
        summaries[config["name"]] = metrics(y_test, pred)
        joblib.dump({"preprocessor": preprocessor, "model": model, "features": FEATURES,
                     "config": config, "target_map": TARGET_MAP}, OUT / f"{config['name']}.joblib")
    prediction_frame = test[["record_id", "origin_program"]].copy()
    prediction_frame["y_true"] = y_test
    for name, pred in predictions.items():
        prediction_frame[name] = pred
    prediction_frame.to_csv(OUT / "frozen_test_predictions.csv", index=False)
    intervals, delta_intervals = _bootstrap(y_test, predictions, test.origin_program.astype(str).to_numpy())
    report = {"status": "completed", "protocol": "one-time final evaluation after development selection; train on all 578,117 development records, frozen test on 70 disjoint origins",
              "development_selection_metric": tuning["selection_metric"], "development_selected_config": best,
              "baseline_config": configurations[0], "test_rows": len(test), "test_origins": len(test_groups),
              "target_map": TARGET_MAP, "source_sha256": source_hash, "development_split_sha256": sha256(ROOT / "data/qmutbench_ml/master_optimization_v1/oof_predictions.csv"),
              "features": FEATURES, "preprocessing": "fit on full development records only; numeric standard scaling and categorical one-hot encoding; unknown test categories ignored",
              "threshold": 0.5, "metrics": summaries, "origin_bootstrap_95pct_intervals": intervals,
              "paired_tuned_minus_baseline_95pct_intervals": delta_intervals, "bootstrap_repetitions": 1000,
              "bootstrap_unit": "frozen-test Origin_program", "test_used_for_selection": False,
              "elapsed_seconds": time.perf_counter() - started}
    (OUT / "frozen_test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
