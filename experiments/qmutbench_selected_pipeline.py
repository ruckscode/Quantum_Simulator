"""Group-aware, chunk-checkpointed QMutBench data selection and training.

The frozen test is defined by the original full-source group split. Selection
uses only development groups, then picks complete origins for broad metadata
coverage. Generated data and models belong under ignored data/qmutbench_ml/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold

from experiments.qmutbench_outcome_ml import (
    DEFAULT_DATA, EXPECTED_SHA256, FEATURES, GROUP_COLUMN, TARGET_COLUMN,
    TARGET_MAP, SEED, DTYPES, prepare_xy, group_holdout, make_pipeline, metrics,
    group_bootstrap_intervals,
    sha256_file,
)

SCHEMA = "qmutbench-selected-chunks-v2"
EXPERIMENT_SCHEMA = "qmutbench-selected-experiment-v2"
OUT = Path("data/qmutbench_ml/selected_pilot_v2")
CHUNK_ROWS = 50_000
DEFAULT_TARGET_ROWS = 50_000
PROFILE = ["algorithm_group", "algorithm", "operation", "gate", "dominant_state",
           "qubits", "Position_int", "_gates_bin"]


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


def _source_identity(path: Path) -> tuple[str, int]:
    digest = sha256_file(path)
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Source hash mismatch: {digest}")
    return digest, path.stat().st_size


def _read_labels_groups(path: Path, chunksize: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids, labels, groups = [], [], []
    for part in pd.read_csv(path, usecols=["Unnamed: 0", GROUP_COLUMN, TARGET_COLUMN],
                            dtype={GROUP_COLUMN: "category", TARGET_COLUMN: "category"}, chunksize=chunksize):
        ids.append(part["Unnamed: 0"].to_numpy(dtype=np.int64))
        labels.append(part[TARGET_COLUMN].astype(str).map(TARGET_MAP).to_numpy(dtype=np.int8))
        groups.append(part[GROUP_COLUMN].astype(str).to_numpy())
    return np.concatenate(ids), np.concatenate(labels), np.concatenate(groups)


def _group_profiles(path: Path, chunksize: int, dev_groups: set[str]) -> tuple[dict, dict]:
    """Summarize only development groups using bounded CSV reads."""
    profiles: dict[str, dict] = {}
    feature_group_frequency: dict[tuple, int] = {}
    usecols = ["Unnamed: 0", GROUP_COLUMN, TARGET_COLUMN,
               *[col for col in PROFILE if col != "_gates_bin"], "gates"]
    for part in pd.read_csv(path, usecols=usecols, dtype=DTYPES, chunksize=chunksize, low_memory=False):
        part = part[part[GROUP_COLUMN].astype(str).isin(dev_groups)]
        if part.empty:
            continue
        part["_y"] = part[TARGET_COLUMN].astype(str).map(TARGET_MAP).astype("int8")
        part["_gates_bin"] = pd.cut(part["gates"],
            bins=[-1, 10, 25, 50, 100, 250, 500, 1000, 5000, 10000, float("inf")],
            labels=["<=10", "11-25", "26-50", "51-100", "101-250", "251-500",
                    "501-1000", "1001-5000", "5001-10000", ">10000"]).astype(str)
        for group, frame in part.groupby(GROUP_COLUMN, observed=True, sort=False):
            key = str(group)
            item = profiles.setdefault(key, {"n": 0, "positive": 0, "features": set(), "combos": set()})
            item["n"] += len(frame)
            item["positive"] += int(frame["_y"].sum())
            item["features"].update((col, str(v)) for col in PROFILE for v in frame[col].unique())
            combos = frame[["algorithm", "operation", "gate", "dominant_state", "_y"]].drop_duplicates()
            vals = {tuple(map(str, row[:-1])) + (int(row[-1]),) for row in combos.itertuples(index=False, name=None)}
            item["combos"].update(vals)
    for item in profiles.values():
        for feature in item["features"]:
            feature_group_frequency[feature] = feature_group_frequency.get(feature, 0) + 1
        for combo in item["combos"]:
            feature_group_frequency[("combo", *combo)] = feature_group_frequency.get(("combo", *combo), 0) + 1
    return profiles, feature_group_frequency


def _select_groups(profiles: dict, frequencies: dict, target_rows: int, seed: int) -> list[str]:
    if target_rows < 1:
        raise ValueError("target_rows must be positive")
    rng = np.random.default_rng(seed)
    tie = {g: float(rng.random()) for g in profiles}
    remaining = set(profiles)
    selected: list[str] = []
    seen_features: set[tuple] = set()
    seen_combos: set[tuple] = set()
    rows = pos = 0
    total_rows = sum(p["n"] for p in profiles.values())
    target_rate = sum(p["positive"] for p in profiles.values()) / total_rows
    while remaining and rows < target_rows:
        best, best_score = None, -1e30
        for group in remaining:
            p = profiles[group]
            fresh_features = p["features"] - seen_features
            fresh_combos = p["combos"] - seen_combos
            novelty = sum(1 / frequencies[v] for v in fresh_features)
            novelty += 3.0 * sum(1 / frequencies[("combo", *v)] for v in fresh_combos)
            new_rate = (pos + p["positive"]) / (rows + p["n"])
            balance = abs(new_rate - target_rate)
            # Reward rare coverage and useful class balance; mild size penalty
            # avoids letting one huge parent dominate while retaining groups whole.
            score = novelty / (p["n"] ** 0.35) - 8.0 * balance + tie[group] * 1e-6
            if score > best_score:
                best, best_score = group, score
        p = profiles[best]
        selected.append(best)
        remaining.remove(best)
        seen_features.update(p["features"])
        seen_combos.update(p["combos"])
        rows += p["n"]
        pos += p["positive"]
    return selected


def prepare(path: Path = DEFAULT_DATA, out: Path = OUT, *, target_rows: int = DEFAULT_TARGET_ROWS,
            chunksize: int = CHUNK_ROWS, seed: int = SEED) -> dict:
    source_hash, source_bytes = _source_identity(path)
    ids, y, groups = _read_labels_groups(path, chunksize)
    if len(np.unique(ids)) != len(ids) or not len(ids):
        raise ValueError("Source record IDs must be present and unique")
    tr, te = group_holdout(y, groups, seed=seed)
    test_groups = set(groups[te])
    dev_groups = set(groups[tr])
    profiles, frequencies = _group_profiles(path, chunksize, dev_groups)
    chosen = _select_groups(profiles, frequencies, target_rows, seed)
    chosen_set = set(chosen)
    assignments = pd.DataFrame({"record_id": ids, "origin_program": groups,
                                "split": np.where(np.isin(groups, list(test_groups)), "frozen_test",
                                         np.where(np.isin(groups, list(chosen_set)), "selected_development", "unselected_development"))})
    assignment_path = out / "selection_assignments.csv"
    _atomic_csv(assignments, assignment_path)
    identity = {"schema": SCHEMA, "source_sha256": source_hash, "source_bytes": source_bytes,
                "source_rows": int(len(ids)), "seed": seed, "chunk_rows": chunksize, "target_rows": target_rows,
                "selection_strategy": "deterministic greedy whole-origin coverage; rare algorithm/algorithm_group, operation, gate, dominant_state, qubit count, mutation position, and gate-count bins plus (algorithm,operation,gate,dominant_state,target) combinations; class prevalence regularizer",
                "features": FEATURES, "group_column": GROUP_COLUMN, "target_map": TARGET_MAP,
                "assignment_sha256": sha256_file(assignment_path),
                "frozen_test_groups": sorted(test_groups), "selected_groups": chosen,
                "test_definition": "original 5-fold StratifiedGroupKFold holdout over all source rows; unchanged seed/fold scoring",
                "group_overlap_test_selected": sorted(test_groups & chosen_set)}
    manifest_path = out / "manifest.json"
    prior = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for key in ("schema", "source_sha256", "seed", "chunk_rows", "target_rows", "features", "selected_groups"):
        if key in prior and prior[key] != identity[key]:
            raise ValueError(f"Existing manifest is incompatible at {key}; choose a new output directory")
    if prior.get("experiment"):
        identity["experiment"] = prior["experiment"]
    out.mkdir(parents=True, exist_ok=True)
    chunks = prior.get("chunks", {})
    failed_chunks = prior.get("failed_chunks", {})
    identity["failed_chunks"] = failed_chunks
    # Re-read source and atomically persist each selected source range. Empty
    # chunks are recorded too, so progress is explicit and resume is bounded.
    for index, part in enumerate(pd.read_csv(path, dtype=DTYPES, chunksize=chunksize, low_memory=False)):
        name = f"chunk_{index:05d}.csv"
        final = out / "chunks" / name
        saved = chunks.get(name)
        if saved and final.is_file() and sha256_file(final) == saved.get("sha256"):
            continue
        try:
            selected_part = part[part[GROUP_COLUMN].astype(str).isin(chosen_set)].copy()
            _atomic_csv(selected_part, final)
            digest = sha256_file(final)
            check = pd.read_csv(final, dtype=DTYPES, low_memory=False)
            if (len(check) != len(selected_part) or check["Unnamed: 0"].duplicated().any()
                    or not set(check[GROUP_COLUMN].astype(str)) <= chosen_set):
                raise IOError(f"Chunk validation failed: {name}")
            raw = check[TARGET_COLUMN].astype(str).value_counts().to_dict()
        except Exception as exc:
            failed_chunks[name] = {"index": index, "status": "failed",
                                   "error_type": type(exc).__name__, "error": str(exc)}
            identity["chunks"] = chunks
            identity["failed_chunks"] = failed_chunks
            identity["status"] = "preparation_failed"
            _atomic_json(manifest_path, identity)
            raise
        chunks[name] = {"schema": SCHEMA, "index": index, "rows": len(check),
                        "record_id_min": int(check["Unnamed: 0"].min()) if len(check) else None,
                        "record_id_max": int(check["Unnamed: 0"].max()) if len(check) else None,
                        "raw_class_counts": raw,
                        "sha256": digest, "status": "complete", "verified": True}
        failed_chunks.pop(name, None)
        identity["chunks"] = chunks
        identity["failed_chunks"] = failed_chunks
        identity["completed_chunks"] = len(chunks)
        identity["progress"] = f"{index + 1} source chunks scanned"
        _atomic_json(manifest_path, identity)
    identity["chunks"] = chunks
    identity["failed_chunks"] = failed_chunks
    identity["completed_chunks"] = len(chunks)
    identity["selected_rows"] = int(sum(v["rows"] for v in chunks.values()))
    identity["selected_origin_groups"] = len(chosen)
    identity["status"] = ("complete" if prior.get("experiment", {}).get("status") == "complete"
                           else "prepared")
    _atomic_json(manifest_path, identity)
    return identity


def status(out: Path = OUT) -> dict:
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    complete = 0
    corrupt = []
    for name, info in manifest.get("chunks", {}).items():
        path = out / "chunks" / name
        if path.is_file() and sha256_file(path) == info["sha256"]:
            complete += 1
        else:
            corrupt.append(name)
    source_rows = manifest.get("source_rows", 723_079)
    expected = (source_rows + manifest["chunk_rows"] - 1) // manifest["chunk_rows"]
    known_indices = {int(value["index"]) for value in manifest.get("chunks", {}).values()}
    corrupt.extend(f"chunk_{index:05d}.csv" for index in range(expected) if index not in known_indices)
    return {"status": manifest.get("status", "preparing"), "completed_chunks_verified": complete,
            "remaining_or_corrupt_chunks": corrupt, "selected_rows": manifest.get("selected_rows"),
            "selected_origin_groups": manifest.get("selected_origin_groups"),
            "failed_chunks": manifest.get("failed_chunks", {}),
            "current_experiment": manifest.get("experiment", {}).get("status", "not_started"),
            "manifest": str((out / "manifest.json").resolve())}


def _load_selected(out: Path) -> pd.DataFrame:
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    frames = []
    for name, info in sorted(manifest["chunks"].items()):
        file = out / "chunks" / name
        if not file.is_file() or sha256_file(file) != info["sha256"]:
            raise ValueError(f"Missing or corrupt completed chunk: {name}; rerun prepare")
        if info["rows"]:
            frames.append(pd.read_csv(file, dtype=DTYPES, low_memory=False))
    frame = pd.concat(frames, ignore_index=True)
    if frame["Unnamed: 0"].duplicated().any():
        raise ValueError("Duplicate source record IDs across chunks")
    return frame


def _model(name: str, seed: int):
    pipe = make_pipeline("random_forest", seed=seed)
    if name == "extra_trees":
        pipe.set_params(model=ExtraTreesClassifier(n_estimators=100, max_depth=18,
            min_samples_leaf=30, class_weight="balanced_subsample", max_features="sqrt",
            n_jobs=2, random_state=seed))
    return pipe


def train(out: Path = OUT, data: Path = DEFAULT_DATA, *, seed: int = SEED, folds: int = 3) -> dict:
    start = time.perf_counter()
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source_sha256"] != _source_identity(data)[0]:
        raise ValueError("Training source differs from prepared source")
    frame = _load_selected(out)
    x, y, groups = prepare_xy(frame)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + 1)
    splits = list(splitter.split(x, y, groups))
    if any(set(groups[a]) & set(groups[b]) for a, b in splits):
        raise AssertionError("Selected validation fold leaked origin groups")
    exp_key = hashlib.sha256(json.dumps({"experiment_schema": EXPERIMENT_SCHEMA,
        "source": manifest["source_sha256"], "seed": seed,
        "folds": folds, "selection": manifest["selected_groups"], "features": FEATURES,
        "models": ["random_forest", "extra_trees"]}, sort_keys=True).encode()).hexdigest()
    expdir = out / "experiment"
    expdir.mkdir(parents=True, exist_ok=True)
    state_path = expdir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"key": exp_key, "folds": {}}
    if state.get("key") != exp_key:
        raise ValueError("Experiment state belongs to a different dataset/configuration")
    cached_metrics = expdir / "metrics.json"
    cached_model = expdir / "best_validation_model.joblib"
    if (state.get("status") == "complete" and cached_metrics.is_file() and cached_model.is_file()
            and state.get("metrics_sha256") == sha256_file(cached_metrics)
            and state.get("best_model_sha256") == sha256_file(cached_model)):
        result = json.loads(cached_metrics.read_text(encoding="utf-8"))
        if "group_bootstrap_95_percentile_intervals" not in result["frozen_test"]:
            return complete_test_analysis(out, data, seed=seed)
        manifest["status"] = "complete"
        manifest["experiment"] = {"status": "complete", "config_key": exp_key,
                                   "model_path": str(cached_model)}
        _atomic_json(manifest_path, manifest)
        return result
    for model_name in ("random_forest", "extra_trees"):
        for fold, (fit_idx, val_idx) in enumerate(splits, 1):
            key = f"{model_name}_fold_{fold}"
            if key in state["folds"]:
                saved_model = expdir / f"{key}.joblib"
                if (saved_model.is_file() and state["folds"][key].get("model_sha256")
                        == sha256_file(saved_model)):
                    continue
                del state["folds"][key]
            t0 = time.perf_counter()
            try:
                model = _model(model_name, seed + fold)
                model.fit(x.iloc[fit_idx], y[fit_idx])
                prediction = model.predict(x.iloc[val_idx])
                row = {"model": model_name, "fold": fold, "train_rows": len(fit_idx),
                       "validation_rows": len(val_idx), "train_groups": len(np.unique(groups[fit_idx])),
                       "validation_groups": len(np.unique(groups[val_idx])), "seconds": time.perf_counter()-t0,
                       **metrics(y[val_idx], prediction)}
                model_path = expdir / f"{key}.joblib"
                tmp = model_path.with_suffix(".tmp")
                joblib.dump(model, tmp, compress=2)
                os.replace(tmp, model_path)
                row["model_sha256"] = sha256_file(model_path)
                state["folds"][key] = row
                state["status"] = "in_progress"
                state.pop("error", None)
                _atomic_json(state_path, state)
            except Exception as exc:
                state["status"] = "failed"
                state["error"] = {"candidate_fold": key, "type": type(exc).__name__, "message": str(exc)}
                _atomic_json(state_path, state)
                raise
    summaries = {}
    for model_name in ("random_forest", "extra_trees"):
        rows = [state["folds"][f"{model_name}_fold_{n}"] for n in range(1, folds + 1)]
        summaries[model_name] = {m: float(np.mean([r[m] for r in rows])) for m in ("accuracy", "balanced_accuracy", "macro_f1")}
        summaries[model_name]["folds"] = rows
    selected = max(summaries, key=lambda n: (summaries[n]["macro_f1"], summaries[n]["balanced_accuracy"]))
    model = _model(selected, seed)
    model.fit(x, y)
    model_path = expdir / "best_validation_model.joblib"
    tmp = model_path.with_suffix(".tmp")
    joblib.dump(model, tmp, compress=3)
    os.replace(tmp, model_path)
    # Evaluate exactly once on the frozen test groups. They were neither selected
    # nor used for CV/model choice. Test material is streamed from the pinned CSV.
    frozen = set(manifest["frozen_test_groups"])
    test_parts = []
    for part in pd.read_csv(data, dtype=DTYPES, chunksize=CHUNK_ROWS, low_memory=False):
        part = part[part[GROUP_COLUMN].astype(str).isin(frozen)]
        if len(part):
            test_parts.append(part)
    test = pd.concat(test_parts, ignore_index=True)
    tx, ty, tg = prepare_xy(test)
    test_pred = model.predict(tx)
    if set(groups) & frozen:
        raise AssertionError("Selected records overlap frozen test groups")
    result = {"schema": EXPERIMENT_SCHEMA, "dataset_sha256": manifest["source_sha256"],
        "selection_rows": len(frame), "selection_groups": len(np.unique(groups)), "seed": seed,
        "validation": summaries, "selected_model": selected, "validation_model_sha256": sha256_file(model_path),
        "frozen_test": {**metrics(ty, test_pred), "rows": len(test), "groups": len(np.unique(tg)),
                        "group_overlap_with_training": 0},
        "training_seconds_including_resumed_prior_stages": time.perf_counter()-start,
        "train_records": len(frame), "train_origin_groups": len(np.unique(groups)),
        "model_path": str(model_path.resolve()), "status": "complete"}
    _atomic_json(expdir / "metrics.json", result)
    state.update({"status": "complete", "selected_model": selected,
                  "metrics_sha256": sha256_file(expdir / "metrics.json"),
                  "best_model_sha256": sha256_file(model_path)})
    _atomic_json(state_path, state)
    manifest["status"] = "complete"
    manifest["experiment"] = {"status": "complete", "config_key": exp_key, "model_path": str(model_path)}
    _atomic_json(manifest_path, manifest)
    return complete_test_analysis(out, data, seed=seed)


def complete_test_analysis(out: Path = OUT, data: Path = DEFAULT_DATA, *, seed: int = SEED,
                           bootstrap_repetitions: int = 1000) -> dict:
    """Persist held-out predictions and add group-wise/cluster-bootstrap metrics."""
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _source_identity(data)[0] != manifest["source_sha256"]:
        raise ValueError("Evaluation source differs from prepared source")
    expdir = out / "experiment"
    metrics_path = expdir / "metrics.json"
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    pred_path = expdir / "frozen_test_predictions.csv"
    if pred_path.is_file() and result["frozen_test"].get("predictions_sha256") == sha256_file(pred_path):
        return result
    model_path = expdir / "best_validation_model.joblib"
    model = joblib.load(model_path)
    frozen = set(manifest["frozen_test_groups"])
    records = []
    for part in pd.read_csv(data, dtype=DTYPES, chunksize=CHUNK_ROWS, low_memory=False):
        part = part[part[GROUP_COLUMN].astype(str).isin(frozen)]
        if part.empty:
            continue
        tx, ty, tg = prepare_xy(part)
        prediction = model.predict(tx)
        records.append(pd.DataFrame({"record_id": part["Unnamed: 0"].to_numpy(),
            "origin_program": tg, "y_true": ty, "y_pred": prediction}))
    predictions = pd.concat(records, ignore_index=True)
    if predictions["record_id"].duplicated().any():
        raise ValueError("Frozen-test prediction record IDs are duplicated")
    _atomic_csv(predictions, pred_path)
    yt = predictions.y_true.to_numpy(dtype=np.int8)
    yp = predictions.y_pred.to_numpy(dtype=np.int8)
    groups = predictions.origin_program.astype(str).to_numpy()
    scores = np.array([float((predictions.loc[predictions.origin_program == g, "y_true"].to_numpy()
                              == predictions.loc[predictions.origin_program == g, "y_pred"].to_numpy()).mean())
                       for g in np.unique(groups)])
    result["frozen_test"].update({
        "group_bootstrap_95_percentile_intervals": group_bootstrap_intervals(
            yt, yp, groups, seed=seed, repetitions=bootstrap_repetitions),
        "groupwise_accuracy": {"mean": float(scores.mean()), "median": float(np.median(scores)),
                               "minimum": float(scores.min()), "maximum": float(scores.max()),
                               "groups": int(len(scores))},
        "predictions_sha256": sha256_file(pred_path),
        "predictions_path": str(pred_path.resolve()),
        "bootstrap_repetitions": bootstrap_repetitions,
    })
    _atomic_json(metrics_path, result)
    state_path = expdir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["metrics_sha256"] = sha256_file(metrics_path)
    _atomic_json(state_path, state)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "run", "status", "evaluate"))
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--target-rows", type=int, default=DEFAULT_TARGET_ROWS)
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cv-folds", type=int, default=3)
    args = parser.parse_args()
    if args.command in ("prepare", "run"):
        result = prepare(args.data, args.output_dir, target_rows=args.target_rows,
                         chunksize=args.chunk_rows, seed=args.seed)
    if args.command in ("train", "run"):
        result = train(args.output_dir, args.data, seed=args.seed, folds=args.cv_folds)
    if args.command == "status":
        result = status(args.output_dir)
    if args.command == "evaluate":
        result = complete_test_analysis(args.output_dir, args.data, seed=args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
