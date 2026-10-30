"""Resource-bounded development screen for additive circuit-depth features.

Uses the completed five-fold mutation-aware OOF assignments. The initial screen
fits only outer fold 1; a predeclared material gain is required before any
full five-fold confirmation. No frozen-test rows, labels, or metrics are read.
"""
from __future__ import annotations

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

STRUCTURE_ADD_ON = [
    "qasm_orig_depth", "qasm_mut_depth", "qasm_delta_depth",
    "qasm_orig_depth_per_qubit", "qasm_mut_depth_per_qubit",
    "qasm_mut_two_plus_fraction", "qasm_delta_two_qubit_fraction",
    "qasm_delta_gate_count",
]
BASE_CATEGORICAL = ["algorithm", "operation", "gate", "dominant_state",
                    "qasm_diff_removed_gate", "qasm_diff_added_gate",
                    "qasm_diff_alignment_class", "qasm_diff_local_prev_gate",
                    "qasm_diff_local_next_gate"]
BASE_NUMERIC = ["qubits", "gates", "Position_int", "position_fraction",
    "qasm_diff_orig_span", "qasm_diff_mut_span", "qasm_diff_total_span",
    "qasm_diff_expected_gate_match", "qasm_diff_qubit_set_equal", "qasm_diff_qubits_touched",
    "qasm_diff_removed_arity", "qasm_diff_added_arity", "qasm_diff_exact_single_edit",
    "qasm_diff_is_local"] + STRUCTURE_ADD_ON
DIFF_CATEGORICAL = BASE_CATEGORICAL[4:]
SOURCE_NUMERIC = ["qubits", "gates", "Position_int"]
SOURCE_CATEGORICAL = ["algorithm", "operation", "gate", "dominant_state"]
DIFF_NUMERIC = [n for n in BASE_NUMERIC if n.startswith("qasm_diff_")]

SOURCE = ROOT / "data/qmutbench_ingestion/raw/official_all_data_20261015.csv"
OOF_PATH = ROOT / "data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/mutation_diff_oof.npz"
DIFF_CACHE = ROOT / "data/qmutbench_ml/qasm_mutation_diff_v1"
STRUCTURE_CACHE = ROOT / "data/qmutbench_ml/qasm_structural_v4"
OUT = ROOT / "reports/experiments/qmutbench_improvement_campaign_20261027"
RAM_STOP_BYTES = 500 * 1024 * 1024
MATERIALITY_BAR = 0.005


def available_ram() -> int:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                    ("virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                    ("extended", ctypes.c_ulonglong)]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.available)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _category_codes(values):
    # Stable, label-independent tokenization. The fold-specific one-hot
    # vocabulary below is learned exclusively from that fold's training rows.
    from sklearn.utils import murmurhash3_32
    return [murmurhash3_32(str(value), positive=True) for value in values]


def _build_cache(out: Path, manifest: dict, ids, id_to_row, max_id: int, pd, np):
    from experiments.qmutbench_qasm_features import EXPECTED_DATA_SHA

    numeric_path = out / "numeric_features.npy"
    category_path = out / "categorical_tokens.npy"
    if manifest.get("cache_status") == "complete" and numeric_path.exists() and category_path.exists():
        if np.load(numeric_path, mmap_mode="r").shape == (max_id + 1, len(BASE_NUMERIC)) \
                and np.load(category_path, mmap_mode="r").shape == (max_id + 1, len(BASE_CATEGORICAL)):
            return np.load(numeric_path, mmap_mode="r"), np.load(category_path, mmap_mode="r")

    numeric = np.lib.format.open_memmap(numeric_path, mode="w+", dtype=np.float32,
                                        shape=(max_id + 1, len(BASE_NUMERIC)))
    categorical = np.lib.format.open_memmap(category_path, mode="w+", dtype=np.uint32,
                                            shape=(max_id + 1, len(BASE_CATEGORICAL)))
    source_seen = np.zeros(max_id + 1, dtype=np.uint8)
    diff_seen = np.zeros(max_id + 1, dtype=np.uint8)
    structure_seen = np.zeros(max_id + 1, dtype=np.uint8)
    position_map = {"0-10%": .05, "11-20%": .155, "21-30%": .255, "31-40%": .355,
                    "41-50%": .455, "51-60%": .555, "61-70%": .655, "71-80%": .755,
                    "81-90%": .855, "91-100%": .955}
    development_ids = np.zeros(max_id + 1, dtype=np.uint8)
    development_ids[ids] = 1
    source_columns = ["Unnamed: 0", *SOURCE_CATEGORICAL, *SOURCE_NUMERIC, "position"]
    for chunk in pd.read_csv(SOURCE, usecols=source_columns, chunksize=25000,
                             dtype={c: str for c in SOURCE_CATEGORICAL + ["position"]}):
        record_ids = chunk["Unnamed: 0"].to_numpy(dtype=np.int64)
        mask = development_ids[record_ids].astype(bool)
        if not mask.any():
            continue
        part = chunk.loc[mask]
        rid = part["Unnamed: 0"].to_numpy(dtype=np.int64)
        if np.any(source_seen[rid]):
            raise ValueError("Duplicate development IDs while reading source metadata")
        bands = part["position"].map(position_map)
        if bands.isna().any():
            raise ValueError("Unknown mutation position band in development features")
        for j, col in enumerate(SOURCE_NUMERIC):
            numeric[rid, j] = part[col].to_numpy(dtype=np.float32)
        numeric[rid, 3] = bands.to_numpy(dtype=np.float32)
        for j, col in enumerate(SOURCE_CATEGORICAL):
            categorical[rid, j] = _category_codes(part[col].astype(str).tolist())
        source_seen[rid] = 1
        del chunk, part
        if available_ram() < RAM_STOP_BYTES:
            raise MemoryError("Memory floor reached during source metadata staging")
    if int(source_seen[ids].sum()) != len(ids):
        raise ValueError("Source metadata does not cover every development row")
    numeric.flush(); categorical.flush()

    diff_numeric_names = DIFF_NUMERIC
    diff_categorical_names = DIFF_CATEGORICAL
    for path in sorted(DIFF_CACHE.glob("diff_*.csv.gz")):
        part = pd.read_csv(path, compression="gzip", usecols=["record_id", *diff_numeric_names,
                                                                *diff_categorical_names])
        rid = part.record_id.to_numpy(dtype=np.int64)
        mask = development_ids[rid].astype(bool)
        part, rid = part.loc[mask], rid[mask]
        if np.any(diff_seen[rid]):
            raise ValueError(f"Duplicate IDs in mutation-diff cache: {path.name}")
        for j, col in enumerate(diff_numeric_names, start=4):
            numeric[rid, j] = part[col].to_numpy(dtype=np.float32)
        for j, col in enumerate(diff_categorical_names, start=4):
            categorical[rid, j] = _category_codes(part[col].astype(str).tolist())
        diff_seen[rid] = 1
        del part
        if available_ram() < RAM_STOP_BYTES:
            raise MemoryError("Memory floor reached while reading mutation-diff features")
    numeric.flush(); categorical.flush()

    structure_columns = ["record_id", *STRUCTURE_ADD_ON]
    for path in sorted(STRUCTURE_CACHE.glob("features_*.csv.gz")):
        part = pd.read_csv(path, compression="gzip", usecols=structure_columns)
        rid = part.record_id.to_numpy(dtype=np.int64)
        mask = development_ids[rid].astype(bool)
        part, rid = part.loc[mask], rid[mask]
        if np.any(structure_seen[rid]):
            raise ValueError(f"Duplicate IDs in structural cache: {path.name}")
        for j, col in enumerate(STRUCTURE_ADD_ON, start=4 + len(DIFF_NUMERIC)):
            numeric[rid, j] = part[col].to_numpy(dtype=np.float32)
        structure_seen[rid] = 1
        del part
        if available_ram() < RAM_STOP_BYTES:
            raise MemoryError("Memory floor reached while reading structural features")
    numeric.flush(); categorical.flush()
    coverage = {"source": int(source_seen[ids].sum()), "mutation_diff": int(diff_seen[ids].sum()),
                "structure": int(structure_seen[ids].sum())}
    if any(n != len(ids) for n in coverage.values()):
        raise ValueError(f"Feature cache coverage mismatch: {coverage}")
    numeric_view, categorical_view = numeric, categorical
    del numeric, categorical
    gc.collect()
    manifest["cache_status"] = "complete"
    manifest["cache_coverage"] = coverage
    atomic_json(out / "run_manifest.json", manifest)
    return numeric_view, categorical_view


def _fold_onehot(train_cat, valid_cat, scipy_sparse, np):
    train_blocks, valid_blocks, vocab_manifest = [], [], []
    for j in range(train_cat.shape[1]):
        vocabulary = np.unique(train_cat[:, j])
        # All-zero token is the hash of a real category too; no sentinel value
        # is reserved. Vocabulary is explicitly fit from training rows only.
        train_codes = np.searchsorted(vocabulary, train_cat[:, j])
        valid_codes = np.searchsorted(vocabulary, valid_cat[:, j])
        known = valid_codes < len(vocabulary)
        known &= vocabulary[np.minimum(valid_codes, len(vocabulary) - 1)] == valid_cat[:, j]
        tr = scipy_sparse.csr_matrix((np.ones(len(train_codes), dtype=np.float32),
                                      (np.arange(len(train_codes)), train_codes)),
                                     shape=(len(train_codes), len(vocabulary)))
        valid_rows = np.flatnonzero(known)
        va = scipy_sparse.csr_matrix((np.ones(len(valid_rows), dtype=np.float32),
                                      (valid_rows, valid_codes[known])),
                                     shape=(len(valid_codes), len(vocabulary)))
        train_blocks.append(tr); valid_blocks.append(va)
        vocab_manifest.append(int(len(vocabulary)))
    return scipy_sparse.hstack(train_blocks, format="csr"), scipy_sparse.hstack(valid_blocks, format="csr"), vocab_manifest


def run() -> None:
    if available_ram() < 700 * 1024 * 1024:
        raise MemoryError(f"Preflight stopped: only {available_ram()/2**20:.0f} MiB available")
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                 f1_score, precision_recall_fscore_support)
    from threadpoolctl import threadpool_limits
    from experiments.qmutbench_qasm_features import EXPECTED_DATA_SHA

    oof = np.load(OOF_PATH, allow_pickle=True)
    ids = oof["record_id"].astype(np.int64)
    y = oof["y"].astype(np.int8)
    groups = oof["origin_program"].astype(str)
    folds = oof["fold"].astype(np.int8)
    if len(ids) != 578117 or len(np.unique(ids)) != len(ids) or set(np.unique(folds)) != {1, 2, 3, 4, 5}:
        raise ValueError("Five-fold development OOF contract mismatch")
    if len(np.unique(groups)) != 305 or pd.DataFrame({"g": groups, "f": folds}).groupby("g").f.nunique().max() != 1:
        raise ValueError("Development Origin_program groups are not fold-isolated")
    if sha256_file(Path("data/qmutbench_ingestion/raw/official_all_data_20261015.csv")) != EXPECTED_DATA_SHA:
        raise ValueError("Source checksum mismatch")
    for cache, expected_rows in ((DIFF_CACHE, 578117), (STRUCTURE_CACHE, 578117)):
        meta = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        if meta.get("status") != "complete" or meta.get("feature_rows", meta.get("records_requested")) != expected_rows:
            raise ValueError(f"Feature cache not complete: {cache}")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "run_manifest.json"
    cache_key_data = {
        "source_sha256": EXPECTED_DATA_SHA,
        "five_fold_oof_sha256": sha256_file(OOF_PATH),
        "mutation_cache_manifest_sha256": sha256_file(DIFF_CACHE / "manifest.json"),
        "structure_cache_manifest_sha256": sha256_file(STRUCTURE_CACHE / "manifest.json"),
        "structure_features": STRUCTURE_ADD_ON, "model": "ExtraTrees 100 trees depth22 minleaf15 sqrt balanced_subsample",
        "initial_screen_fold": 1, "materiality_bar_macro_f1": MATERIALITY_BAR,
    }
    cache_key = hashlib.sha256(json.dumps(cache_key_data, sort_keys=True).encode()).hexdigest()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cache_key") != cache_key:
            raise ValueError("Campaign manifest provenance mismatch")
    else:
        manifest = {"schema": "qmutbench-improvement-campaign-v1", "status": "screen_in_progress",
                    "cache_key": cache_key, "provenance": cache_key_data,
                    "champion": "mutation-aware 23-feature Extra Trees, five-fold OOF",
                    "hypothesis": "circuit depth and global two-qubit complexity add signal beyond local mutation alignment",
                    "selected_structure_add_on": STRUCTURE_ADD_ON,
                    "fold_protocol": "reuse saved five-fold grouped development assignments; fold 1 screen only",
                    "threshold": 0.5, "seed": 20261025 + 1,
                    "resource_guard_mib": RAM_STOP_BYTES // 2**20,
                    "experiments": []}
        atomic_json(manifest_path, manifest)

    id_map = np.zeros(int(ids.max()) + 1, dtype=np.uint8)
    id_map[ids] = 1
    numeric, categorical = _build_cache(OUT, manifest, ids, id_map, int(ids.max()), pd, np)
    if available_ram() < RAM_STOP_BYTES:
        raise MemoryError("Resource floor reached before model screen")
    fold = 1
    train_ids, valid_ids = ids[folds != fold], ids[folds == fold]
    train_rows, valid_rows = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    if set(groups[train_rows]) & set(groups[valid_rows]):
        raise ValueError("Fold 1 origin leakage")
    x_train_num = np.asarray(numeric[train_ids], dtype=np.float32)
    x_valid_num = np.asarray(numeric[valid_ids], dtype=np.float32)
    if not np.isfinite(x_train_num).all() or not np.isfinite(x_valid_num).all():
        raise ValueError("Non-finite numeric values in candidate feature matrix")
    x_train_cat = np.asarray(categorical[train_ids], dtype=np.uint32)
    x_valid_cat = np.asarray(categorical[valid_ids], dtype=np.uint32)
    tr_cat, va_cat, category_counts = _fold_onehot(x_train_cat, x_valid_cat, sparse, np)
    del x_train_cat, x_valid_cat
    x_train = sparse.hstack([sparse.csr_matrix(x_train_num), tr_cat], format="csr", dtype=np.float32)
    x_valid = sparse.hstack([sparse.csr_matrix(x_valid_num), va_cat], format="csr", dtype=np.float32)
    del x_train_num, x_valid_num, tr_cat, va_cat
    gc.collect()
    available = available_ram()
    print(f"fold1 matrix ready: train={x_train.shape}, valid={x_valid.shape}, "
          f"available={available/2**20:.0f} MiB", flush=True)
    if available < 700 * 1024 * 1024:
        manifest["status"] = "stopped_resource_floor_before_fit"
        manifest["available_bytes_before_fit"] = available
        atomic_json(manifest_path, manifest)
        raise MemoryError("Not enough headroom for an Extra Trees fit; stopped before training")

    start = time.perf_counter()
    model = ExtraTreesClassifier(n_estimators=100, max_depth=22, min_samples_leaf=15,
        max_features="sqrt", class_weight="balanced_subsample", n_jobs=1, random_state=20261026)
    with threadpool_limits(limits=1):
        model.fit(x_train, y[train_rows])
        probabilities = model.predict_proba(x_valid)[:, list(model.classes_).index(1)]
    predicted = (probabilities >= .5).astype(np.int8)
    baseline_oof = oof["probability"][valid_rows]

    def metrics(actual, pred):
        precision, recall, f1, support = precision_recall_fscore_support(actual, pred, labels=[0, 1], zero_division=0)
        return {"accuracy": float(accuracy_score(actual, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(actual, pred)),
                "macro_f1": float(f1_score(actual, pred, average="macro", zero_division=0)),
                "confusion_matrix_labels_0_1": confusion_matrix(actual, pred, labels=[0, 1]).tolist(),
                "per_class": {str(label): {"precision": float(precision[i]), "recall": float(recall[i]),
                                           "f1": float(f1[i]), "support": int(support[i])}
                              for i, label in enumerate([0, 1])}}

    current = metrics(y[valid_rows], (baseline_oof >= .5).astype(np.int8))
    candidate_metrics = metrics(y[valid_rows], predicted)
    delta = candidate_metrics["macro_f1"] - current["macro_f1"]
    output_predictions = pd.DataFrame({"record_id": valid_ids, "origin_program": groups[valid_rows],
        "y_true": y[valid_rows], "p_champion": baseline_oof, "y_pred_champion": (baseline_oof >= .5).astype(np.int8),
        "p_candidate": probabilities, "y_pred_candidate": predicted})
    output_predictions.to_csv(OUT / "fold1_screen_predictions.csv.gz", index=False, compression="gzip")
    experiment = {
        "name": "mutation-aware plus eight predeclared circuit complexity features",
        "fold": 1, "train_rows": int(len(train_rows)), "validation_rows": int(len(valid_rows)),
        "train_origins": int(len(set(groups[train_rows]))), "validation_origins": int(len(set(groups[valid_rows]))),
        "candidate_config": {"model": "ExtraTreesClassifier", "n_estimators": 100, "max_depth": 22,
            "min_samples_leaf": 15, "max_features": "sqrt", "class_weight": "balanced_subsample",
            "n_jobs": 1, "random_state": 20261026, "threshold": .5,
            "feature_schema": [*BASE_CATEGORICAL, *BASE_NUMERIC],
            "categorical_hashing": "MurmurHash3 32-bit fixed tokenization; fold-specific one-hot vocabulary"},
        "champion_metrics_fold1": current, "candidate_metrics_fold1": candidate_metrics,
        "macro_f1_delta": float(delta), "materiality_bar": MATERIALITY_BAR,
        "screen_decision": "advance_to_full_five_fold_confirmation" if delta >= MATERIALITY_BAR else "reject_additive_structure_hypothesis",
        "category_vocabulary_sizes_fold1": category_counts,
        "fit_and_predict_seconds": time.perf_counter() - start,
        "available_ram_bytes_after_fit": available_ram(),
        "predictions_path": "fold1_screen_predictions.csv.gz",
    }
    manifest["experiments"].append(experiment)
    manifest["status"] = "screen_complete"
    manifest["next_step"] = experiment["screen_decision"]
    atomic_json(manifest_path, manifest)
    print(json.dumps(experiment, indent=2), flush=True)


if __name__ == "__main__":
    run()
