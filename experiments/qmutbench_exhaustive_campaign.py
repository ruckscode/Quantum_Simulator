"""Evaluate target-blind QASM mutation-alignment features on saved grouped folds."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

from experiments.qmutbench_qasm_features import BASELINE_OOF, SOURCE_CSV, ROOT, sha256_file
from experiments.qmutbench_qasm_model_comparison import (BASE_CATEGORICAL, BASE_NUMERIC, BASE_FEATURES,
    metrics, assert_group_fold_isolation, SEED, MEMORY_GUARD_BYTES)

CACHE = ROOT / "data/qmutbench_ml/qasm_mutation_diff_v1"
OUT = ROOT / "data/qmutbench_ml/exhaustive_campaign_v1"
RUNS = OUT / "mutation_diff_folds_matched"
BASELINE = ROOT / "data/qmutbench_ml/qasm_model_comparison_v1/folds"
FEATURES = ["qasm_diff_orig_span", "qasm_diff_mut_span", "qasm_diff_total_span",
            "qasm_diff_expected_gate_match", "qasm_diff_qubit_set_equal", "qasm_diff_qubits_touched",
            "qasm_diff_removed_arity", "qasm_diff_added_arity", "qasm_diff_exact_single_edit",
            "qasm_diff_is_local", "qasm_diff_removed_gate", "qasm_diff_added_gate",
            "qasm_diff_alignment_class", "qasm_diff_local_prev_gate", "qasm_diff_local_next_gate"]
CATEGORICAL = ["qasm_diff_removed_gate", "qasm_diff_added_gate", "qasm_diff_alignment_class",
               "qasm_diff_local_prev_gate", "qasm_diff_local_next_gate"]


def load_data():
    if json.loads((CACHE / "manifest.json").read_text()).get("status") != "complete":
        raise ValueError("Mutation-diff extraction is not complete")
    parts = [pd.read_csv(p, compression="gzip") for p in sorted(CACHE.glob("diff_*.csv.gz"))]
    d = pd.concat(parts, ignore_index=True)
    oof_parts = [np.load(ROOT / f"data/qmutbench_ml/performance_phase_full_v1/predictions/position_fraction_fold{i}.npz", allow_pickle=False) for i in (1,2,3)]
    oof = pd.DataFrame({k: np.concatenate([x[k] for x in oof_parts]) for k in ("record_id","y","origin_program","fold")})
    if len(d) != len(oof) or d.record_id.duplicated().any() or set(d.record_id) != set(oof.record_id):
        raise ValueError("Mutation-diff feature cache does not exactly match development OOF records")
    d = d.set_index("record_id").loc[oof.record_id.to_numpy()].reset_index()
    base_cols = ["Unnamed: 0", "algorithm", "operation", "gate", "dominant_state", "qubits", "gates", "Position_int", "position"]
    raw = pd.read_csv(SOURCE_CSV, usecols=base_cols, low_memory=False)
    raw = raw[raw["Unnamed: 0"].isin(set(oof.record_id))].rename(columns={"Unnamed: 0":"record_id"})
    raw = raw.set_index("record_id").loc[oof.record_id.to_numpy()].reset_index()
    bands = raw.position.astype(str).str.extract(r"^(\d+)-(\d+)%$").astype(np.float32)
    raw["position_fraction"] = bands.mean(axis=1).div(100).astype(np.float32)
    x = raw.merge(d, on="record_id", validate="one_to_one")
    for col in BASE_CATEGORICAL + CATEGORICAL:
        x[col] = x[col].astype("category")
    for col in BASE_NUMERIC + [c for c in FEATURES if c not in CATEGORICAL]:
        x[col] = x[col].astype(np.float32)
    if len(x) != len(oof) or x[BASE_FEATURES + FEATURES].isna().any().any():
        raise ValueError("Model feature matrix has missing values")
    assert_group_fold_isolation(oof.rename(columns={"origin_program":"origin_program"}))
    return x, oof


def model(names, fold, trees):
    cat = [n for n in names if n in BASE_CATEGORICAL + CATEGORICAL]
    num = [n for n in names if n not in cat]
    prep = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), num),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32))]), cat),
    ], sparse_threshold=1.0)
    est = ExtraTreesClassifier(n_estimators=trees, max_depth=22, min_samples_leaf=15, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=1, random_state=SEED + fold)
    return Pipeline([("prep", prep), ("model", est)])


def run(trees=60):
    global RUNS
    RUNS = OUT / f"mutation_diff_{trees}_trees"
    x, oof = load_data()
    RUNS.mkdir(parents=True, exist_ok=True)
    names = BASE_FEATURES + FEATURES
    results=[]
    for fold in (1,2,3):
        existing_json, existing_npz = RUNS/f"fold{fold}.json", RUNS/f"fold{fold}.npz"
        if existing_json.is_file() and existing_npz.is_file():
            results.append(json.loads(existing_json.read_text()))
            continue
        if psutil.virtual_memory().available < MEMORY_GUARD_BYTES:
            raise MemoryError("Memory guard reached before a fold fit")
        tr=oof.fold.to_numpy()!=fold; va=~tr
        if set(oof.loc[tr,"origin_program"]) & set(oof.loc[va,"origin_program"]): raise ValueError("Origin leakage")
        start=time.perf_counter(); m=model(names,fold,trees)
        with threadpool_limits(limits=1):
            m.fit(x.loc[tr,names],oof.loc[tr,"y"].to_numpy(np.int8))
            p=m.predict_proba(x.loc[va,names])[:,1]
        y=oof.loc[va,"y"].to_numpy(np.int8); ids=oof.loc[va,"record_id"].to_numpy(np.int64)
        np.savez_compressed(RUNS/f"fold{fold}.npz",record_id=ids,y=y,origin_program=oof.loc[va,"origin_program"].astype(str).to_numpy(dtype=str),fold=np.full(len(y),fold,np.int8),probability=p)
        r={"fold":fold,"rows":int(va.sum()),"origins":int(oof.loc[va,"origin_program"].nunique()),"seconds":time.perf_counter()-start,"metrics":metrics(y,p)}
        (RUNS/f"fold{fold}.json").write_text(json.dumps(r,indent=2)+"\n")
        results.append(r); print(f"fold {fold}: macroF1={r['metrics']['macro_f1']:.6f}",flush=True)
        del m,p; gc.collect()
    manifest={"schema":"qmutbench-mutation-diff-screen-v1","status":"complete","rows":len(oof),"origins":int(oof.origin_program.nunique()),"trees":trees,"seed":"20261025+fold","base_features":BASE_FEATURES,"mutation_diff_features":FEATURES,"folds":results,"dataset_sha256":sha256_file(SOURCE_CSV),"diff_manifest_sha256":sha256_file(CACHE/"manifest.json"),"baseline_manifest_sha256":sha256_file(BASELINE_OOF)}
    pred=pd.concat([pd.DataFrame({k:np.load(RUNS/f"fold{i}.npz",allow_pickle=True)[k] for k in ("record_id","y","origin_program","fold","probability")}) for i in (1,2,3)],ignore_index=True).sort_values("record_id")
    base_dir=(ROOT / "data/qmutbench_ml/performance_phase_full_v1/predictions") if trees==100 else BASELINE
    base_name=(lambda i:f"position_fraction_fold{i}.npz") if trees==100 else (lambda i:f"extra_trees_baseline_fold{i}.npz")
    base=pd.concat([pd.DataFrame({k:np.load(base_dir/base_name(i),allow_pickle=False)[k] for k in ("record_id","y","origin_program","fold","probability")}) for i in (1,2,3)],ignore_index=True).sort_values("record_id")
    if not np.array_equal(pred.record_id,base.record_id) or not np.array_equal(pred.y,base.y) or not np.array_equal(pred.fold,base.fold): raise ValueError("Baseline rows/folds do not match")
    manifest["pooled_oof_metrics"]=metrics(pred.y.to_numpy(np.int8),pred.probability.to_numpy(float))
    manifest["matched_baseline_pooled_oof_metrics"]=metrics(base.y.to_numpy(np.int8),base.probability.to_numpy(float))
    manifest["pooled_macro_f1_delta"]=manifest["pooled_oof_metrics"]["macro_f1"]-manifest["matched_baseline_pooled_oof_metrics"]["macro_f1"]
    manifest["fold_macro_f1_deltas"]=[results[i-1]["metrics"]["macro_f1"]-(metrics(base.loc[base.fold.eq(i),"y"].to_numpy(np.int8),base.loc[base.fold.eq(i),"probability"].to_numpy(float))["macro_f1"]) for i in (1,2,3)]
    (OUT/"mutation_diff_screen.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:manifest[k] for k in ("pooled_oof_metrics","matched_baseline_pooled_oof_metrics","pooled_macro_f1_delta","fold_macro_f1_deltas")},indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--trees",type=int,choices=[60,100],default=60)
    run(parser.parse_args().trees)
