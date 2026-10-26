"""Five-fold grouped confirmation of baseline vs mutation-aware features."""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from experiments.qmutbench_exhaustive_campaign import (BASE_FEATURES, FEATURES, OUT, load_data, model)
from experiments.qmutbench_qasm_model_comparison import MEMORY_GUARD_BYTES, assert_group_fold_isolation, metrics, SEED
from experiments.qmutbench_qasm_features import sha256_file, SOURCE_CSV

RUNS = OUT / "five_fold_confirmation"
TREES = 100


def run() -> dict:
    x, oof = load_data()
    RUNS.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261026)
    splits = list(splitter.split(np.zeros(len(oof)), oof.y.to_numpy(np.int8), oof.origin_program.to_numpy()))
    fold_contract = {}
    for i, (tr, va) in enumerate(splits, 1):
        train_groups = set(oof.origin_program.iloc[tr]); val_groups = set(oof.origin_program.iloc[va])
        if train_groups & val_groups:
            raise ValueError(f"Origin leakage in fold {i}")
        fold_contract[str(i)] = {"train_rows": len(tr), "validation_rows": len(va),
                                 "train_origins": len(train_groups), "validation_origins": len(val_groups),
                                 "validation_record_id_sha256": __import__("hashlib").sha256(oof.record_id.iloc[va].to_numpy(np.int64).tobytes()).hexdigest()}
    predictions = {}
    for variant, names in (("baseline", BASE_FEATURES), ("mutation_diff", BASE_FEATURES + FEATURES)):
        pred_path = RUNS / f"{variant}_oof.npz"
        if pred_path.is_file():
            z = np.load(pred_path, allow_pickle=False)
            predictions[variant] = pd.DataFrame({k: z[k] for k in ("record_id", "y", "origin_program", "fold", "probability")}).sort_values("record_id").reset_index(drop=True)
            continue
        out = []
        for fold, (tr, va) in enumerate(splits, 1):
            if psutil.virtual_memory().available < MEMORY_GUARD_BYTES:
                raise MemoryError("Memory guard reached before five-fold model fit")
            start = time.perf_counter()
            estimator = model(names, fold, TREES)
            with threadpool_limits(limits=1):
                estimator.fit(x.iloc[tr][names], oof.y.iloc[tr].to_numpy(np.int8))
                probability = estimator.predict_proba(x.iloc[va][names])[:, 1]
            y = oof.y.iloc[va].to_numpy(np.int8)
            out.append(pd.DataFrame({"record_id": oof.record_id.iloc[va].to_numpy(np.int64), "y": y,
                "origin_program": oof.origin_program.iloc[va].astype(str).to_numpy(dtype=str), "fold": np.full(len(va), fold, np.int8),
                "probability": probability}))
            print(f"{variant} fold {fold}: macroF1={metrics(y, probability)['macro_f1']:.6f}; seconds={time.perf_counter()-start:.1f}", flush=True)
            del estimator, probability
            gc.collect()
        frame = pd.concat(out, ignore_index=True).sort_values("record_id").reset_index(drop=True)
        np.savez_compressed(pred_path, **{c: frame[c].to_numpy() for c in frame.columns})
        predictions[variant] = frame
    baseline, candidate = predictions["baseline"], predictions["mutation_diff"]
    if not np.array_equal(baseline.record_id, candidate.record_id) or not np.array_equal(baseline.y, candidate.y) or not np.array_equal(baseline.fold, candidate.fold):
        raise ValueError("Five-fold candidates are not compared on identical records and folds")
    fold_metrics = {}
    for variant, frame in predictions.items():
        fold_metrics[variant] = [metrics(frame.loc[frame.fold.eq(f), "y"].to_numpy(np.int8), frame.loc[frame.fold.eq(f), "probability"].to_numpy(float)) for f in range(1, 6)]
    summary = {"schema":"qmutbench-five-fold-grouped-confirmation-v1", "status":"complete", "rows":len(oof),
        "origins":int(oof.origin_program.nunique()), "splitter":"StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261026)",
        "trees":TREES,"model_seed":"20261025+fold","threshold":0.5,"fold_contract":fold_contract,
        "baseline_pooled_metrics":metrics(baseline.y.to_numpy(np.int8),baseline.probability.to_numpy(float)),
        "candidate_pooled_metrics":metrics(candidate.y.to_numpy(np.int8),candidate.probability.to_numpy(float)),
        "pooled_macro_f1_delta":metrics(candidate.y.to_numpy(np.int8),candidate.probability.to_numpy(float))["macro_f1"]-metrics(baseline.y.to_numpy(np.int8),baseline.probability.to_numpy(float))["macro_f1"],
        "baseline_fold_metrics":fold_metrics["baseline"],"candidate_fold_metrics":fold_metrics["mutation_diff"],
        "fold_macro_f1_deltas":[fold_metrics["mutation_diff"][i]["macro_f1"]-fold_metrics["baseline"][i]["macro_f1"] for i in range(5)],
        "source_sha256":sha256_file(SOURCE_CSV),"interpretation":"additional grouped stability confirmation on the development dataset; not an independent external test because feature selection used development OOF evidence"}
    (RUNS / "summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(),indent=2))
