"""Bounded, checkpointed retrospective frozen-test comparison for QMutBench.

The development-selected mutation-aware model and 0.5 threshold are fixed.
This runner reads the previously saved full-development baseline predictions,
extracts mutation features by record ID in small streaming batches, and never
fits or selects a model on frozen-test data.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def available_ram_bytes() -> int:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.ullAvailPhys)


MIN_AVAILABLE_BYTES = 500 * 1024 * 1024


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_ids(values) -> str:
    return hashlib.sha256("\n".join(str(x) for x in sorted(values)).encode()).hexdigest()


def run(smoke: bool = False) -> None:
    # Fail closed before importing the large scientific stack.
    before = available_ram_bytes()
    if before < 650 * 1024 * 1024:
        raise RuntimeError(f"Resource stop before imports: {before / 2**20:.0f} MiB available")

    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                 f1_score, precision_recall_fscore_support)
    from threadpoolctl import threadpool_limits

    from experiments.qmutbench_qasm_features import (EXPECTED_DATA_SHA, EXPECTED_REPO_COMMIT,
        OFFICIAL_CHECKOUT, QASM_ROOT, SOURCE_CSV, sha256_file as source_sha256_file)
    from experiments.qmutbench_qasm_mutation_diff import _ops, _safe_paths, FEATURE_COLUMNS, USECOLS
    from qmutbench_ml.qasm_mutation_diff import mutation_diff_features

    baseline_path = ROOT / "data/qmutbench_ml/learning_curve_v1/test_predictions.csv"
    baseline_manifest_path = ROOT / "data/qmutbench_ml/learning_curve_v1/test_predictions_manifest.json"
    baseline_model_path = ROOT / "data/qmutbench_ml/learning_curve_v1/best_development_model.joblib"
    candidate_path = ROOT / "data/qmutbench_ml/mutation_aware_v1/best_model.joblib"
    candidate_meta_path = ROOT / "data/qmutbench_ml/mutation_aware_v1/best_model_metadata.json"
    out = ROOT / "reports/experiments/qmutbench_frozen_comparison_bounded_20261027"
    out.mkdir(parents=True, exist_ok=True)
    work = out / ("smoke" if smoke else "candidate_feature_cache")
    work.mkdir(parents=True, exist_ok=True)

    if source_sha256_file(SOURCE_CSV) != EXPECTED_DATA_SHA:
        raise ValueError("Official source CSV checksum mismatch")
    commit = subprocess.run(["git", "-C", str(OFFICIAL_CHECKOUT), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(OFFICIAL_CHECKOUT), "status", "--porcelain"],
                           check=True, capture_output=True, text=True).stdout.strip()
    if commit != EXPECTED_REPO_COMMIT or dirty:
        raise ValueError("Pinned official QMutBench checkout is not clean at expected commit")

    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    baseline_sha = sha256_file(baseline_path)
    if baseline_sha != baseline_manifest["predictions_sha256"]:
        raise ValueError("Saved baseline predictions do not match their manifest")
    if sha256_file(baseline_model_path) != baseline_manifest["final_model_sha256"]:
        raise ValueError("Saved full-development baseline model does not match its manifest")
    baseline = pd.read_csv(baseline_path, dtype={"origin_program": str})
    if len(baseline) != 144962 or baseline.record_id.duplicated().any():
        raise ValueError("Baseline frozen predictions violate row/unique-ID contract")
    if sha_ids(baseline.record_id.astype("int64").tolist()) != baseline_manifest["test_record_ids_sha256"]:
        raise ValueError("Frozen-test record-ID checksum mismatch")
    target_counts = baseline.y_true.value_counts().to_dict()
    if target_counts != {1: 75076, 0: 69886}:
        raise ValueError(f"Frozen-test label counts differ from locked contract: {target_counts}")

    metadata = json.loads(candidate_meta_path.read_text(encoding="utf-8"))
    candidate_sha = sha256_file(candidate_path)
    if candidate_sha != metadata["model_sha256"] or metadata["inference_threshold"] != 0.5:
        raise ValueError("Candidate model hash or fixed threshold differs from selected metadata")
    feature_names = list(metadata["feature_schema"])
    expected_qasm_features = [name for name in feature_names if name.startswith("qasm_diff_")]
    if len(feature_names) != 23 or len(expected_qasm_features) != 15:
        raise ValueError("Unexpected selected candidate feature schema")
    baseline_ids = set(baseline.record_id.astype("int64"))
    del baseline
    gc.collect()

    original_map = {"0-10%": .05, "11-20%": .155, "21-30%": .255, "31-40%": .355,
                    "41-50%": .455, "51-60%": .555, "61-70%": .655, "71-80%": .755,
                    "81-90%": .855, "91-100%": .955}
    raw_columns = ["Unnamed: 0", "algorithm", "qubits", "operation", "gate", "Position_int",
                   "gates", "dominant_state", "position"]
    tracked_raw = subprocess.run(["git", "-C", str(OFFICIAL_CHECKOUT), "ls-files", "-z", "--",
                                  "data/Origin_programs", "data/Mutated_programs"],
                                 check=True, capture_output=True).stdout
    tracked = {os.fsdecode(path).removeprefix("data/") for path in tracked_raw.split(b"\0")
               if path.endswith(b".qasm")}
    original_cache_path = work / "original_signatures.json"
    if original_cache_path.exists():
        raw_cache = json.loads(original_cache_path.read_text(encoding="utf-8"))
        originals = {name: [(gate, tuple(wires)) for gate, wires in ops]
                     for name, ops in raw_cache.items()}
    else:
        originals = {}

    model = joblib.load(candidate_path)
    if tuple(model.feature_names_in_) != tuple(feature_names):
        raise ValueError("Loaded candidate feature order differs from frozen metadata")
    available = available_ram_bytes()
    if available < MIN_AVAILABLE_BYTES:
        raise RuntimeError(f"Resource stop after model load: {available / 2**20:.0f} MiB available")
    print(f"start: {available / 2**20:.0f} MiB available; candidate loaded; smoke={smoke}", flush=True)

    cache_key_payload = {
        "source_sha256": EXPECTED_DATA_SHA, "official_commit": commit,
        "baseline_prediction_sha256": baseline_sha, "baseline_model_sha256": baseline_manifest["final_model_sha256"],
        "test_record_ids_sha256": baseline_manifest["test_record_ids_sha256"],
        "candidate_model_sha256": candidate_sha, "candidate_features": feature_names,
        "threshold": .5, "parser_sha256": source_sha256_file(ROOT / "qmutbench_validation.py"),
        "runtime_feature_sha256": source_sha256_file(ROOT / "qmutbench_ml/qasm_mutation_diff.py"),
        "extractor_sha256": source_sha256_file(Path(__file__)),
    }
    cache_key = hashlib.sha256(json.dumps(cache_key_payload, sort_keys=True).encode()).hexdigest()
    manifest_path = work / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cache_key") != cache_key:
            raise ValueError("Existing bounded feature cache has a different provenance key")
    else:
        manifest = {"schema": "qmutbench-bounded-frozen-feature-cache-v1", "cache_key": cache_key,
                    "key": cache_key_payload, "status": "in_progress", "completed": {},
                    "stop_floor_bytes": MIN_AVAILABLE_BYTES, "batch_rows": 1000}
        atomic_json(manifest_path, manifest)

    # A smoke run uses a separate directory, only the first 64 frozen IDs, and
    # performs parity checks without producing any comparison metrics.
    smoke_ids = set(sorted(baseline_ids)[:64]) if smoke else None
    row_cap = 64 if smoke else None
    processed = 0
    started = time.perf_counter()
    source_chunk_size = 10000
    with threadpool_limits(limits=1):
        for source_chunk_index, source_chunk in enumerate(pd.read_csv(
                SOURCE_CSV, usecols=raw_columns, chunksize=source_chunk_size,
                dtype={"algorithm": str, "operation": str, "gate": str,
                       "dominant_state": str, "position": str})):
            subset = source_chunk[source_chunk["Unnamed: 0"].astype("int64").isin(
                smoke_ids if smoke else baseline_ids)]
            if subset.empty:
                continue
            # For resumability, full runs store row batches keyed by source chunk
            # and within-chunk batch. Existing verified batches are skipped.
            rows = subset.to_dict("records")
            for batch_index, batch_start in enumerate(range(0, len(rows), 1000)):
                batch_key = f"{source_chunk_index:05d}_{batch_index:03d}"
                if not smoke and batch_key in manifest["completed"]:
                    entry = manifest["completed"][batch_key]
                    for key in ("features_file", "predictions_file"):
                        fpath = work / entry[key]
                        if not fpath.exists() or sha256_file(fpath) != entry[key.replace("_file", "_sha256")]:
                            raise ValueError(f"Checkpoint checksum mismatch: {fpath}")
                    continue
                part = rows[batch_start:batch_start + 1000]
                records = []
                for row in part:
                    rid = int(row["Unnamed: 0"])
                    _, original_rel, mutant_rel = _safe_paths(row, tracked)
                    if original_rel not in originals:
                        originals[original_rel] = _ops(QASM_ROOT / original_rel)
                    mutant_ops = _ops(QASM_ROOT / mutant_rel)
                    qasm_features = mutation_diff_features(originals[original_rel], mutant_ops,
                                                            row["operation"], row["gate"])
                    position = str(row["position"])
                    if position not in original_map:
                        raise ValueError(f"Unknown position category on test record {rid}: {position}")
                    record = {
                        "record_id": rid, "algorithm": row["algorithm"], "operation": row["operation"],
                        "gate": row["gate"], "dominant_state": row["dominant_state"],
                        "qubits": int(row["qubits"]), "gates": int(row["gates"]),
                        "Position_int": int(row["Position_int"]), "position_fraction": original_map[position],
                    }
                    record.update({name: qasm_features[name] for name in expected_qasm_features})
                    records.append(record)
                frame = pd.DataFrame(records)
                if frame.record_id.duplicated().any() or list(frame.columns[1:]) != feature_names:
                    raise ValueError("Extracted feature schema/order or row uniqueness mismatch")
                probabilities = model.predict_proba(frame[feature_names])[:, list(model.classes_).index(1)]
                predicted = (probabilities >= .5).astype("int8")
                if smoke:
                    print(f"smoke rows={len(frame)} feature_schema={len(feature_names)} "
                          f"probability_range=({probabilities.min():.4f},{probabilities.max():.4f}) "
                          f"available={available_ram_bytes()/2**20:.0f} MiB", flush=True)
                else:
                    feature_file = f"features_{batch_key}.csv.gz"
                    pred_file = f"predictions_{batch_key}.csv"
                    feature_path, pred_path = work / feature_file, work / pred_file
                    temp_feature = feature_path.with_suffix(feature_path.suffix + ".tmp")
                    frame.to_csv(temp_feature, index=False, compression="gzip")
                    os.replace(temp_feature, feature_path)
                    prediction_frame = pd.DataFrame({"record_id": frame.record_id.astype("int64"),
                                                     "p_candidate": probabilities,
                                                     "y_pred_candidate": predicted})
                    temp_pred = pred_path.with_suffix(pred_path.suffix + ".tmp")
                    prediction_frame.to_csv(temp_pred, index=False)
                    os.replace(temp_pred, pred_path)
                    manifest["completed"][batch_key] = {
                        "rows": len(frame), "features_file": feature_file,
                        "features_sha256": sha256_file(feature_path),
                        "predictions_file": pred_file,
                        "predictions_sha256": sha256_file(pred_path),
                    }
                    manifest["completed_rows"] = sum(x["rows"] for x in manifest["completed"].values())
                    atomic_json(manifest_path, manifest)
                    del prediction_frame
                processed += len(frame)
                del records, frame, probabilities, predicted
                if processed % 5000 < 1000:
                    gc.collect()
                    available = available_ram_bytes()
                    print(f"progress rows_this_run={processed} source_chunk={source_chunk_index} "
                          f"available={available/2**20:.0f} MiB elapsed={time.perf_counter()-started:.0f}s",
                          flush=True)
                    if available < MIN_AVAILABLE_BYTES:
                        manifest["status"] = "paused_resource_floor"
                        manifest["last_available_bytes"] = available
                        atomic_json(manifest_path, manifest)
                        raise RuntimeError(f"Paused at resource floor: {available / 2**20:.0f} MiB available")
                if smoke and processed >= row_cap:
                    break
            # Persist the bounded original circuit cache after each source chunk.
            cache_temp = original_cache_path.with_suffix(".json.tmp")
            cache_temp.write_text(json.dumps({name: [[g, list(w)] for g, w in ops]
                                              for name, ops in originals.items()}, separators=(",", ":")),
                                  encoding="utf-8")
            os.replace(cache_temp, original_cache_path)
            if smoke and processed >= row_cap:
                break
    if smoke:
        print(f"smoke complete: {processed} rows; no metrics computed; elapsed={time.perf_counter()-started:.1f}s",
              flush=True)
        return

    manifest["status"] = "complete" if manifest.get("completed_rows") == len(baseline_ids) else "incomplete"
    manifest["elapsed_seconds_this_run"] = time.perf_counter() - started
    manifest["last_available_bytes"] = available_ram_bytes()
    atomic_json(manifest_path, manifest)
    if manifest["status"] != "complete":
        raise RuntimeError(f"Incomplete frozen feature cache: {manifest.get('completed_rows')} rows")

    # Final scoring occurs only after all feature and prediction checkpoints pass
    # checksum/ID coverage checks; no model selection or tuning is performed.
    pred_parts = [pd.read_csv(work / entry["predictions_file"])
                  for entry in manifest["completed"].values()]
    candidate_predictions = pd.concat(pred_parts, ignore_index=True)
    if candidate_predictions.record_id.duplicated().any() or set(candidate_predictions.record_id) != baseline_ids:
        raise ValueError("Candidate predictions do not exactly cover the locked frozen-test IDs")
    base = pd.read_csv(baseline_path, dtype={"origin_program": str})
    joined = base.merge(candidate_predictions, on="record_id", validate="one_to_one", sort=False)
    if len(joined) != len(base) or joined.y_true.isna().any() or joined.origin_program.isna().any():
        raise ValueError("Prediction/label/origin alignment failed")
    if int((joined.y_pred_candidate == joined.y_pred).sum()) < 0:
        raise AssertionError("Unreachable alignment guard")
    joined["candidate_threshold"] = .5
    joined.to_csv(out / "paired_frozen_test_predictions.csv.gz", index=False, compression="gzip")

    def metrics(y, pred):
        precision, recall, f1, support = precision_recall_fscore_support(
            y, pred, labels=[0, 1], zero_division=0)
        return {"accuracy": float(accuracy_score(y, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                "confusion_matrix_labels_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                "per_class": {str(c): {"precision": float(precision[i]), "recall": float(recall[i]),
                                       "f1": float(f1[i]), "support": int(support[i])}
                              for i, c in enumerate([0, 1])}}

    y = joined.y_true.to_numpy(dtype="int8")
    pred_base = joined.y_pred.to_numpy(dtype="int8")
    pred_candidate = joined.y_pred_candidate.to_numpy(dtype="int8")
    group_confusions = []
    for _, group in joined.groupby("origin_program", sort=False):
        gy = group.y_true.to_numpy(dtype="int8")
        group_confusions.append((confusion_matrix(gy, group.y_pred.to_numpy(dtype="int8"), labels=[0, 1]),
                                 confusion_matrix(gy, group.y_pred_candidate.to_numpy(dtype="int8"), labels=[0, 1])))
    rng = np.random.default_rng(20261027)
    boot = {name: [] for name in ("baseline", "candidate", "candidate_minus_baseline")}
    for _ in range(1000):
        chosen = rng.integers(0, len(group_confusions), size=len(group_confusions))
        matrices = [sum((group_confusions[i][j] for i in chosen), start=np.zeros((2, 2), dtype=np.int64))
                    for j in (0, 1)]
        vals = []
        for cm in matrices:
            tn, fp, fn, tp = cm.ravel()
            f1_negative = 2 * tn / max(1, 2 * tn + fp + fn)
            f1_positive = 2 * tp / max(1, 2 * tp + fp + fn)
            vals.append((float((tn + tp) / max(1, cm.sum())),
                         float(.5 * (tn / max(1, tn + fp) + tp / max(1, fn + tp))),
                         float(.5 * (f1_negative + f1_positive))))
        boot["baseline"].append(vals[0]); boot["candidate"].append(vals[1])
        boot["candidate_minus_baseline"].append(tuple(b-a for a, b in zip(vals[0], vals[1])))
    metric_names = ["accuracy", "balanced_accuracy", "macro_f1"]
    intervals = {key: {metric_names[i]: [float(np.percentile(np.asarray(values)[:, i], 2.5)),
                                          float(np.percentile(np.asarray(values)[:, i], 97.5))]
                       for i in range(3)} for key, values in boot.items()}
    results = {
        "schema": "qmutbench-mutation-aware-frozen-test-comparison-v1",
        "status": "complete_retrospective_comparison",
        "test_reuse_caveat": "Frozen test has prior historical reuse; retrospective comparison, not pristine independent validation.",
        "protocol": "candidate and baseline trained on all 578117 development rows; fixed 0.5 thresholds; no test tuning",
        "rows": len(joined), "origins": int(joined.origin_program.nunique()),
        "label_counts": {str(k): int(v) for k, v in joined.y_true.value_counts().to_dict().items()},
        "record_ids_sha256": sha_ids(joined.record_id.astype("int64").tolist()),
        "baseline": {"model": "full-development eight-feature Extra Trees", "sha256": baseline_manifest["final_model_sha256"],
                     "predictions_sha256": baseline_sha, "metrics": metrics(y, pred_base)},
        "candidate": {"model": "mutation-aware 23-feature Extra Trees", "sha256": candidate_sha,
                      "threshold": .5, "metrics": metrics(y, pred_candidate)},
        "group_bootstrap_95_percentile_intervals": intervals,
        "bootstrap_repetitions": 1000, "bootstrap_unit": "Origin_program, paired for differences",
        "paired_predictions_path": str((out / "paired_frozen_test_predictions.csv.gz").relative_to(ROOT)),
        "feature_cache_manifest": str(manifest_path.relative_to(ROOT)),
        "elapsed_seconds": time.perf_counter() - started,
        "available_ram_bytes_at_finish": available_ram_bytes(),
    }
    atomic_json(out / "frozen_test_comparison.json", results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="extract and predict 64 held-out rows only")
    args = parser.parse_args()
    run(smoke=args.smoke)
