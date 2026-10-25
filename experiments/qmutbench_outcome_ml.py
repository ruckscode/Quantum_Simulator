"""Group-aware ML for the recorded QMutBench mutation-detection outcome.

This predicts the benchmark's historical oracle outcome from pre-outcome
mutation/circuit metadata. It is not a hardware diagnosis model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/qmutbench_ingestion/raw/official_all_data_20261015.csv"
DEFAULT_OUT = ROOT / "data/qmutbench_ml/outcome_model_v1"
EXPECTED_SHA256 = "3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778"
EXPECTED_BYTES = 59_913_911
EXPECTED_ROWS = 723_079
SEED = 20261025
GROUP_COLUMN = "Origin_program"
TARGET_COLUMN = "Killed"
TARGET_MAP = {"WOO": 1, "OPO": 1, "Survivor": 0}
CATEGORICAL = ["algorithm", "operation", "gate", "dominant_state"]
NUMERIC = ["qubits", "gates", "Position_int"]
FEATURES = CATEGORICAL + NUMERIC
REQUIRED = {
    "Unnamed: 0", GROUP_COLUMN, "algorithm", "qubits", "gates", "operation",
    "Position_int", TARGET_COLUMN, "gate", "algorithm_group", "dominant_state", "position",
}
DTYPES = {
    "Origin_program": "category", "algorithm": "category", "operation": "category",
    "Killed": "category", "gate": "category", "algorithm_group": "category",
    "dominant_state": "category", "position": "category",
    "qubits": "int16", "gates": "int32", "Position_int": "int32",
}
LABEL_ORDER = [0, 1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path, *, expected_source: bool = True, nrows: int | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_source and nrows is None:
        if path.stat().st_size != EXPECTED_BYTES:
            raise ValueError(f"QMutBench source byte-size mismatch: {path.stat().st_size}")
        actual_hash = sha256_file(path)
        if actual_hash != EXPECTED_SHA256:
            raise ValueError(f"QMutBench source SHA-256 mismatch: {actual_hash}")
    frame = pd.read_csv(path, dtype=DTYPES, nrows=nrows, low_memory=False)
    missing = sorted(REQUIRED - set(frame.columns))
    extra = sorted(set(frame.columns) - REQUIRED)
    if missing or extra:
        raise ValueError(f"QMutBench schema mismatch; missing={missing}; extra={extra}")
    if frame.isna().any().any():
        missing_counts = frame.isna().sum()
        raise ValueError(f"Missing values are not supported: {missing_counts[missing_counts.gt(0)].to_dict()}")
    raw_labels = set(frame[TARGET_COLUMN].astype(str).unique())
    if not raw_labels <= set(TARGET_MAP):
        raise ValueError(f"Unexpected {TARGET_COLUMN} values: {sorted(raw_labels - set(TARGET_MAP))}")
    if frame[GROUP_COLUMN].nunique() < 2:
        raise ValueError("At least two Origin_program groups are needed")
    return frame


def prepare_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return fixed, pre-outcome features, binary recorded outcome, and groups."""
    if not set(FEATURES) <= set(frame.columns):
        raise ValueError("Required model features are missing")
    if TARGET_COLUMN in FEATURES or GROUP_COLUMN in FEATURES or "Unnamed: 0" in FEATURES:
        raise AssertionError("Target/index/group leakage in registered feature schema")
    x = frame[FEATURES].copy()
    y = frame[TARGET_COLUMN].astype(str).map(TARGET_MAP).to_numpy(dtype=np.int8)
    groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    if len(x) != len(y) or len(y) != len(groups):
        raise ValueError("Feature, target, and group row counts differ")
    return x, y, groups


def group_holdout(y: np.ndarray, groups: np.ndarray, *, seed: int = SEED, folds: int = 5):
    """Choose a deterministic stratified whole-origin fold nearest a 20% test set."""
    if len(np.unique(groups)) < folds:
        raise ValueError(f"Need at least {folds} distinct groups; found {len(np.unique(groups))}")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    candidates = list(splitter.split(np.zeros(len(y)), y, groups))
    target_rate = float(np.mean(y))
    def score(pair):
        _, test_idx = pair
        rate = float(np.mean(y[test_idx]))
        size_error = abs(len(test_idx) / len(y) - 0.20)
        prevalence_error = abs(rate - target_rate)
        return size_error + prevalence_error
    train_idx, test_idx = min(candidates, key=score)
    if set(groups[train_idx]) & set(groups[test_idx]):
        raise AssertionError("Origin_program leakage across group holdout")
    if len(np.unique(y[train_idx])) != 2 or len(np.unique(y[test_idx])) != 2:
        raise ValueError("Both classes must occur in train and held-out groups")
    return train_idx, test_idx


def make_pipeline(model_name: str, *, seed: int = SEED) -> Pipeline:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "logistic_regression":
        numeric_steps.append(("scale", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ], remainder="drop", sparse_threshold=1.0)
    estimators = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=400, solver="liblinear", random_state=seed,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=18, min_samples_leaf=25, class_weight="balanced", random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=40, max_depth=18, min_samples_leaf=30,
            class_weight="balanced_subsample", max_features="sqrt",
            n_jobs=2, random_state=seed,
        ),
    }
    if model_name not in estimators:
        raise ValueError(f"Unknown model {model_name!r}; choices: {sorted(estimators)}")
    return Pipeline([("preprocess", preprocessor), ("model", estimators[model_name])])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_ORDER, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {
            "Survivor (0)": {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0]), "support": int(support[0])},
            "Detected by WOO/OPO (1)": {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1]), "support": int(support[1])},
        },
        "confusion_matrix_labels_0_1": confusion_matrix(y_true, y_pred, labels=LABEL_ORDER).tolist(),
    }


def group_bootstrap_intervals(y_true, y_pred, groups, *, seed: int = SEED, repetitions: int = 1000):
    """Percentile intervals resampling parent-circuit groups, not individual rows."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    values = {"accuracy": [], "macro_f1": [], "balanced_accuracy": []}
    for _ in range(repetitions):
        sampled_groups = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([members[group] for group in sampled_groups])
        result = metrics(y_true[idx], y_pred[idx])
        for key in values:
            values[key].append(result[key])
    return {
        key: [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]
        for key, samples in values.items()
    }


def _fold_cv(x, y, groups, train_idx, *, seed=SEED, folds=3, cache_dir=None, cache_key=""):
    outer_x, outer_y, outer_groups = x.iloc[train_idx], y[train_idx], groups[train_idx]
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + 1)
    results = {}
    for model_name in ("majority", "logistic_regression", "decision_tree", "random_forest"):
        fold_rows = []
        for fold, (fit_local, val_local) in enumerate(splitter.split(outer_x, outer_y, outer_groups), 1):
            if set(outer_groups[fit_local]) & set(outer_groups[val_local]):
                raise AssertionError("Origin_program leakage in grouped validation fold")
            cache_path = cache_dir / f"{model_name}_fold_{fold}.json" if cache_dir else None
            if cache_path and cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("cache_key") == cache_key:
                    fold_rows.append(cached["result"])
                    print(f"CV {model_name} fold {fold}/{folds}: resumed", flush=True)
                    continue
            estimator = make_pipeline(model_name, seed=seed + fold)
            estimator.fit(outer_x.iloc[fit_local], outer_y[fit_local])
            pred = estimator.predict(outer_x.iloc[val_local])
            fold_result = {
                "fold": fold,
                "validation_groups": int(len(np.unique(outer_groups[val_local]))),
                **metrics(outer_y[val_local], pred),
            }
            fold_rows.append(fold_result)
            if cache_path:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({"cache_key": cache_key, "result": fold_result}, indent=2), encoding="utf-8")
            print(f"CV {model_name} fold {fold}/{folds}: done", flush=True)
        results[model_name] = {
            "folds": fold_rows,
            "mean_accuracy": float(np.mean([row["accuracy"] for row in fold_rows])),
            "mean_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in fold_rows])),
            "mean_macro_f1": float(np.mean([row["macro_f1"] for row in fold_rows])),
        }
    return results


def run_smoke(path: Path, *, limit: int = 30_000, output_dir: Path | None = None) -> dict:
    frame = load_dataset(path, expected_source=False, nrows=limit)
    x, y, groups = prepare_xy(frame)
    train_idx, test_idx = group_holdout(y, groups, folds=2)
    model = make_pipeline("logistic_regression")
    model.fit(x.iloc[train_idx], y[train_idx])
    pred = model.predict(x.iloc[test_idx])
    if set(groups[train_idx]) & set(groups[test_idx]):
        raise AssertionError("smoke split leaked parent groups")
    result = {
        "status": "pass",
        "rows": len(frame),
        "groups": int(len(np.unique(groups))),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_groups": int(len(np.unique(groups[train_idx]))),
        "test_groups": int(len(np.unique(groups[test_idx]))),
        "metrics": metrics(y[test_idx], pred),
        "purpose": "pipeline/split/save-load smoke only; not an evaluation result",
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "smoke_model.joblib"
        joblib.dump(model, model_path)
        reloaded = joblib.load(model_path)
        if not np.array_equal(pred, reloaded.predict(x.iloc[test_idx])):
            raise AssertionError("save/load changed deterministic predictions")
        model_path.unlink()
        (output_dir / "smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_saved_model(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    model = joblib.load(path)
    if not hasattr(model, "predict"):
        raise ValueError("Saved artifact is not a prediction pipeline")
    return model


def predict_metadata(input_path: Path, model_path: Path, output_path: Path) -> Path:
    frame = pd.read_csv(input_path, low_memory=False)
    if TARGET_COLUMN in frame.columns:
        raise ValueError("Prediction input must not contain the Killed target column")
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction input is missing model features: {missing}")
    x = frame[FEATURES].copy()
    for column in NUMERIC:
        x[column] = pd.to_numeric(x[column], errors="coerce")
    model = load_saved_model(model_path)
    predicted = model.predict(x)
    output = pd.DataFrame({
        "predicted_recorded_outcome": np.where(predicted == 1, "Detected_WOO_or_OPO", "Survivor"),
    })
    if GROUP_COLUMN in frame:
        output.insert(0, GROUP_COLUMN, frame[GROUP_COLUMN].astype(str))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output_path


def evaluate_saved_predictions(path: Path, *, seed: int = SEED) -> dict:
    frame = pd.read_csv(path)
    required = {"origin_program", "y_true", "y_pred"}
    if not required <= set(frame.columns):
        raise ValueError(f"Held-out prediction file missing columns: {sorted(required-set(frame.columns))}")
    y_true = frame["y_true"].to_numpy(dtype=np.int8)
    y_pred = frame["y_pred"].to_numpy(dtype=np.int8)
    groups = frame["origin_program"].astype(str).to_numpy()
    return {
        "metrics": metrics(y_true, y_pred),
        "group_bootstrap_95_percentile_intervals": group_bootstrap_intervals(y_true, y_pred, groups, seed=seed),
        "rows": int(len(frame)),
        "origin_program_groups": int(len(np.unique(groups))),
        "purpose": "recalculate metrics from the frozen saved final-test predictions; does not refit or retune",
    }


def run_training(path: Path, output_dir: Path, *, seed: int = SEED, cv_folds: int = 3) -> dict:
    frame = load_dataset(path)
    x, y, groups = prepare_xy(frame)
    train_idx, test_idx = group_holdout(y, groups, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_hash = sha256_file(path)
    cache_key = hashlib.sha256(json.dumps({
        "dataset_sha256": dataset_hash, "seed": seed, "cv_folds": cv_folds,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "train_groups": sorted(set(groups[train_idx])), "features": FEATURES,
        "models": ["majority", "logistic_regression", "decision_tree", "random_forest"],
    }, sort_keys=True).encode()).hexdigest()
    cv = _fold_cv(x, y, groups, train_idx, seed=seed, folds=cv_folds,
                  cache_dir=output_dir / "cv_checkpoints", cache_key=cache_key)
    selected = max(cv, key=lambda name: (cv[name]["mean_macro_f1"], cv[name]["mean_balanced_accuracy"]))
    final_model = make_pipeline(selected, seed=seed)
    final_model.fit(x.iloc[train_idx], y[train_idx])
    y_pred = final_model.predict(x.iloc[test_idx])
    test_metrics = metrics(y[test_idx], y_pred)
    test_groups = groups[test_idx]
    intervals = group_bootstrap_intervals(y[test_idx], y_pred, test_groups, seed=seed)

    # Random-row validation is a development-only diagnostic of sibling leakage;
    # it does not choose a model and is kept separate from the locked group test.
    dev_local = np.arange(len(train_idx))
    rand_fit, rand_val = train_test_split(
        dev_local, test_size=0.2, random_state=seed, stratify=y[train_idx],
    )
    random_diagnostic = make_pipeline(selected, seed=seed)
    random_diagnostic.fit(x.iloc[train_idx[rand_fit]], y[train_idx[rand_fit]])
    random_pred = random_diagnostic.predict(x.iloc[train_idx[rand_val]])

    model_path = output_dir / "qmutbench_outcome_model.joblib"
    joblib.dump(final_model, model_path, compress=3)
    round_trip = joblib.load(model_path)
    if not np.array_equal(y_pred, round_trip.predict(x.iloc[test_idx])):
        raise AssertionError("Model persistence round-trip changed final test predictions")
    pd.DataFrame({
        "row_index": frame.iloc[test_idx]["Unnamed: 0"].to_numpy(),
        "origin_program": groups[test_idx],
        "y_true": y[test_idx],
        "y_pred": y_pred,
    }).to_csv(output_dir / "heldout_predictions.csv", index=False)

    result = {
        "schema": "qmutbench-outcome-ml-results-v1",
        "dataset": {
            "path": str(path.resolve()), "sha256": sha256_file(path),
            "bytes": path.stat().st_size, "rows": int(len(frame)),
            "origin_program_groups": int(len(np.unique(groups))),
            "raw_label_counts": {str(k): int(v) for k, v in frame[TARGET_COLUMN].value_counts().items()},
            "binary_target_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        },
        "task": {
            "target_column": TARGET_COLUMN,
            "target_definition": "1 if raw Killed is WOO or OPO; 0 if Survivor",
            "intended_use": "predict the recorded historical benchmark oracle outcome from mutation/circuit metadata",
            "not_claimed": ["source-level localization", "behavioral anomaly diagnosis", "hardware-fault attribution", "noise robustness"],
        },
        "features": FEATURES,
        "excluded_columns": sorted(REQUIRED - set(FEATURES) - {TARGET_COLUMN}),
        "split": {
            "strategy": "StratifiedGroupKFold by Origin_program; one deterministic fold isolated as final test",
            "seed": seed, "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
            "train_groups": int(len(np.unique(groups[train_idx]))), "test_groups": int(len(np.unique(groups[test_idx]))),
            "group_overlap": sorted(set(groups[train_idx]) & set(groups[test_idx])),
            "train_origin_programs_sha256": hashlib.sha256("\n".join(sorted(set(groups[train_idx]))).encode()).hexdigest(),
            "test_origin_programs_sha256": hashlib.sha256("\n".join(sorted(set(groups[test_idx]))).encode()).hexdigest(),
        },
        "selection": {"metric": "mean group-CV macro F1", "cv_folds": cv_folds, "cv": cv, "selected_model": selected},
        "random_row_split_development_diagnostic": {
            **metrics(y[train_idx[rand_val]], random_pred),
            "overlapping_origin_groups": int(len(set(groups[train_idx[rand_fit]]) & set(groups[train_idx[rand_val]]))),
            "use": "diagnostic only; not used for model selection or final-test claims",
        },
        "final_test": {
            **test_metrics,
            "group_bootstrap_95_percentile_intervals": intervals,
            "groups": int(len(np.unique(test_groups))),
            "accuracy_target_85_percent_achieved": bool(test_metrics["accuracy"] >= 0.85),
        },
        "model_parameters": str(final_model.named_steps["model"].get_params()),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "model_path": str(model_path.resolve()),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(
        confusion_matrix(y[test_idx], y_pred, labels=LABEL_ORDER),
        index=["actual_survivor", "actual_detected"], columns=["pred_survivor", "pred_detected"],
    ).to_csv(output_dir / "confusion_matrix.csv")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "smoke", "train", "evaluate", "predict", "load-model"))
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--limit", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cv-folds", type=int, default=3)
    args = parser.parse_args()
    if args.command == "audit":
        from experiments.qmutbench_outcome_ml_audit import audit_dataset
        result = audit_dataset(args.data)
    elif args.command == "smoke":
        result = run_smoke(args.data, limit=args.limit, output_dir=args.output_dir)
    elif args.command == "train":
        result = run_training(args.data, args.output_dir, seed=args.seed, cv_folds=args.cv_folds)
    elif args.command == "evaluate":
        predictions = args.input or (args.output_dir / "heldout_predictions.csv")
        result = evaluate_saved_predictions(predictions, seed=args.seed)
    elif args.command == "load-model":
        model_path = args.model or (args.output_dir / "qmutbench_outcome_model.joblib")
        model = load_saved_model(model_path)
        result = {"loaded": True, "model_path": str(model_path.resolve()), "estimator": str(model.named_steps["model"]), "classes": model.classes_.tolist()}
    else:
        if args.input is None or args.output is None:
            parser.error("predict requires --input and --output")
        model_path = args.model or (args.output_dir / "qmutbench_outcome_model.joblib")
        result = {"predictions": str(predict_metadata(args.input, model_path, args.output))}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
