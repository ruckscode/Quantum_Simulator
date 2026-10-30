"""Grouped development-fold tuning of the QMutBench Logistic Regression baseline."""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from experiments.qmutbench_outcome_ml import CATEGORICAL, DEFAULT_DATA, EXPECTED_SHA256, FEATURES, NUMERIC, TARGET_MAP, sha256_file
from qmutbench_qml.data.loader import sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/experiments/qmutbench_logistic_tuning_20261026"
OOF = ROOT / "data/qmutbench_ml/master_optimization_v1/oof_predictions.csv"
SEED = 20261026


def make_preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32))]), CATEGORICAL),
    ], sparse_threshold=1.0)


def metric_row(y, p):
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=[0, 1], zero_division=0)
    return {"accuracy": float(accuracy_score(y, p)), "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "per_class": {str(k): {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])}
                          for i, k in enumerate([0, 1])}}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(OOF, usecols=["record_id", "origin_program", "y_true", "fold"])
    keep = ["Unnamed: 0", "Origin_program", "Killed", *FEATURES]
    oof = oof.set_index("record_id")
    pieces = []
    dtypes = {"Origin_program": "category", "Killed": "category", **{c: "category" for c in CATEGORICAL},
              "qubits": "int16", "gates": "int32", "Position_int": "int32"}
    for chunk in pd.read_csv(DEFAULT_DATA, usecols=keep, dtype=dtypes, chunksize=50_000):
        chunk = chunk.rename(columns={"Unnamed: 0": "record_id", "Origin_program": "origin_program"})
        sub = chunk[chunk.record_id.isin(oof.index)]
        if not sub.empty:
            metadata = oof.reindex(sub.record_id.to_numpy())
            if not np.array_equal(sub.origin_program.astype(str).to_numpy(), metadata.origin_program.astype(str).to_numpy()):
                raise ValueError("Source origin differs from saved development manifest")
            sub["y_true"] = metadata.y_true.to_numpy(np.int8)
            sub["fold"] = metadata.fold.to_numpy(np.int8)
            pieces.append(sub)
    data = pd.concat(pieces, ignore_index=True)
    if len(data) != len(oof) or sha256(DEFAULT_DATA) != EXPECTED_SHA256:
        raise ValueError("Verified development rows or source checksum mismatch")
    if not np.array_equal(data.Killed.map(TARGET_MAP).to_numpy(), data.y_true.to_numpy()):
        raise ValueError("Source target mapping differs from saved development OOF labels")
    data[NUMERIC] = data[NUMERIC].apply(pd.to_numeric)
    y = data.y_true.to_numpy(np.int8)
    folds = sorted(data.fold.unique())
    c_grid = np.logspace(-3, 3, 5)
    candidates = list(ParameterGrid({"C": c_grid, "solver": ["liblinear", "lbfgs", "saga"],
                                      "class_weight": ["balanced", None]}))
    # Existing baseline is re-run on the exact same 7 inputs and saved 3-fold records.
    configs = [{"name": "baseline_liblinear_C1_balanced", "C": 1.0, "solver": "liblinear", "class_weight": "balanced"}]
    configs += [{"name": f"tuned_{i:02d}", **p} for i, p in enumerate(candidates, 1)]
    fold_summaries = []
    start = time.perf_counter()
    # Fit preprocessing once per fold, on that fold's training rows only.
    for fold in folds:
        train = data.fold != fold
        valid = ~train
        prep = make_preprocessor()
        with threadpool_limits(limits=1):
            xtr = prep.fit_transform(data.loc[train, FEATURES], y[train])
            xva = prep.transform(data.loc[valid, FEATURES])
        if set(data.loc[train, "origin_program"]) & set(data.loc[valid, "origin_program"]):
            raise ValueError(f"Origin overlap in saved fold {fold}")
        for config in configs:
            clf = LogisticRegression(C=config["C"], solver=config["solver"], class_weight=config["class_weight"],
                                     max_iter=500, tol=1e-4, random_state=SEED)
            with threadpool_limits(limits=1):
                clf.fit(xtr, y[train])
                pred = clf.predict(xva)
            row = {"model": config["name"], "fold": int(fold), **metric_row(y[valid], pred)}
            row["iterations_max"] = int(np.max(clf.n_iter_))
            row["hit_max_iter"] = bool(row["iterations_max"] >= 500)
            fold_summaries.append(row)
            if len(fold_summaries) % 10 == 0:
                print(f"Fold {fold}: completed {len(fold_summaries)} configurations", flush=True)
        print(f"Completed fold {fold}/{len(folds)}: train={int(train.sum()):,}, validation={int(valid.sum()):,}, configs={len(configs)}", flush=True)
        del xtr, xva, prep
    fold_frame = pd.DataFrame(fold_summaries)
    summary = []
    for name, g in fold_frame.groupby("model", sort=False):
        summary.append({"model": name, "accuracy": float(g.accuracy.mean()),
                       "balanced_accuracy": float(g.balanced_accuracy.mean()), "macro_f1": float(g.macro_f1.mean()),
                       "fold_macro_f1_mean": float(g.macro_f1.mean()),
                       "fold_macro_f1_sd": float(g.macro_f1.std(ddof=1)), "folds": len(g)})
    baseline_name = configs[0]["name"]
    best_name = max((r for r in summary if r["model"] != baseline_name),
                    key=lambda r: (r["fold_macro_f1_mean"], r["balanced_accuracy"]))["model"]
    best_config = next(c for c in configs if c["name"] == best_name)
    predictions = []
    for fold in folds:
        train = data.fold != fold
        valid = ~train
        prep = make_preprocessor()
        xtr = prep.fit_transform(data.loc[train, FEATURES], y[train])
        xva = prep.transform(data.loc[valid, FEATURES])
        ids_valid = data.loc[valid, ["record_id", "origin_program"]].reset_index(drop=True)
        for config in (configs[0], best_config):
            clf = LogisticRegression(C=config["C"], solver=config["solver"], class_weight=config["class_weight"],
                                     max_iter=500, tol=1e-4, random_state=SEED)
            with threadpool_limits(limits=1):
                clf.fit(xtr, y[train])
                pred = clf.predict(xva)
            predictions.extend({"record_id": int(rid), "origin_program": str(group), "fold": int(fold),
                                "model": config["name"], "y_true": int(yt), "y_pred": int(yp)}
                               for rid, group, yt, yp in zip(ids_valid.record_id, ids_valid.origin_program, y[valid], pred))
        del xtr, xva, prep
    pred_df = pd.DataFrame(predictions)
    for result in summary:
        matched = pred_df[pred_df.model == result["model"]]
        if not matched.empty:
            result.update(metric_row(matched.y_true.to_numpy(), matched.y_pred.to_numpy()))
    pd.DataFrame(summary).to_csv(OUT / "model_summary.csv", index=False)
    fold_frame.to_json(OUT / "fold_metrics.json", orient="records", indent=2)
    pd.DataFrame([c for c in configs]).to_csv(OUT / "configurations.csv", index=False)
    pred_df.to_csv(OUT / "oof_predictions.csv", index=False)
    data[["record_id", "origin_program", "y_true", "fold"]].to_csv(OUT / "split_manifest.csv", index=False)
    manifest = {"status": "completed", "seed": SEED, "source_sha256": sha256_file(DEFAULT_DATA),
                "split_manifest_sha256": sha256(OOF), "python": sys.version.split()[0],
                "scikit_learn": sklearn.__version__, "platform": platform.platform(),
                "split_source": str(OOF.relative_to(ROOT)), "rows": len(data), "origins": int(data.origin_program.nunique()),
                "folds": {str(k): {"rows": int((data.fold == k).sum()), "origins": int(data.loc[data.fold == k, "origin_program"].nunique())} for k in folds},
                "features": FEATURES, "target_map": TARGET_MAP, "selection_metric": "mean macro F1 across fixed saved 3-fold grouped development OOF folds",
                "solver_max_iter": 500, "solver_tolerance": 1e-4,
                "grid": {"C": list(map(float, c_grid)), "solver": ["liblinear", "lbfgs", "saga"], "class_weight": ["balanced", None], "max_iter": 500},
                "preprocessing": "fold-local median imputation; standard scaling on three numeric features; fold-local one-hot encoding of four categorical features",
                "baseline": configs[0], "best_config": best_config, "best_metrics": next(r for r in summary if r["model"] == best_name),
                "frozen_test_read": False, "elapsed_seconds": time.perf_counter() - start}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    base = next(r for r in summary if r["model"] == baseline_name)
    selected = manifest["best_metrics"]
    report = [
        "# QMutBench Logistic Regression development tuning", "",
        "The target maps WOO/OPO to detected (1) and Survivor to undetected (0). Tuning used only the verified 578,117 development records across the saved three-fold origin-disjoint OOF split. The frozen test was not read.", "",
        "The existing balanced `liblinear`, C=1 baseline and all candidates use the same seven historical features, same records, and same folds. Numeric scaling, imputation, and one-hot category vocabularies were fit within each training fold.", "",
        f"Grid: C={', '.join(f'{x:g}' for x in manifest['grid']['C'])}; solvers={', '.join(manifest['grid']['solver'])}; class weights={manifest['grid']['class_weight']}; max_iter={manifest['solver_max_iter']}; tol={manifest['solver_tolerance']}. Selection used mean fold macro F1.", "",
        "| Model | Accuracy | Balanced accuracy | Macro F1 (pooled OOF) | Fold macro F1 mean +/- SD |", "|---|---:|---:|---:|---:|",
    ]
    for result in (base, selected):
        report.append(f"| {result['model']} | {100*result['accuracy']:.2f}% | {100*result['balanced_accuracy']:.2f}% | {100*result['macro_f1']:.2f}% | {100*result['fold_macro_f1_mean']:.2f}% +/- {100*result['fold_macro_f1_sd']:.2f}% |")
    report += ["", f"Selected configuration: C={best_config['C']:g}, solver={best_config['solver']}, class_weight={best_config['class_weight']}. The selected minus baseline pooled OOF macro-F1 difference is {100*(selected['macro_f1']-base['macro_f1']):+.3f} percentage points. This is a small development difference; the selected configuration was frozen before any final-test evaluation.", "",
               "`fold_metrics.json` includes per-class metrics and convergence iteration counts for every configuration and fold. `oof_predictions.csv` contains paired row predictions for the baseline and selected candidate. The full grid and fold assignment are in `configurations.csv` and `split_manifest.csv`.", "",
               f"Runtime: {manifest['elapsed_seconds']:.2f}s. Source SHA-256: `{manifest['source_sha256']}`. Split manifest SHA-256: `{manifest['split_manifest_sha256']}`.", ""]
    (OUT / "research_report.md").write_text("\n".join(report), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
