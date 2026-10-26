"""Ablate QASM gate identity/alignment category block on the fixed three folds."""
from __future__ import annotations

import gc, json, time
from pathlib import Path
import numpy as np, pandas as pd, psutil
from threadpoolctl import threadpool_limits

from experiments.qmutbench_exhaustive_campaign import BASE_FEATURES, FEATURES, CATEGORICAL, OUT, load_data, model
from experiments.qmutbench_qasm_model_comparison import MEMORY_GUARD_BYTES, metrics
from experiments.qmutbench_qasm_features import sha256_file

RUNS=OUT/"feature_ablation_no_gate_context"
BASELINE=Path("data/qmutbench_ml/qasm_model_comparison_v1/folds")

def run():
    x,oof=load_data(); RUNS.mkdir(parents=True,exist_ok=True)
    removed=["qasm_diff_removed_gate","qasm_diff_added_gate","qasm_diff_alignment_class","qasm_diff_local_prev_gate","qasm_diff_local_next_gate"]
    names=BASE_FEATURES+[c for c in FEATURES if c not in removed]
    fold_results=[]
    for fold in (1,2,3):
        rp=RUNS/f"fold{fold}.json"; pp=RUNS/f"fold{fold}.npz"
        if rp.exists() and pp.exists(): fold_results.append(json.loads(rp.read_text())); continue
        if psutil.virtual_memory().available<MEMORY_GUARD_BYTES: raise MemoryError("Memory guard reached before ablation fit")
        tr=oof.fold.to_numpy()!=fold; va=~tr
        start=time.perf_counter(); estimator=model(names,fold,60)
        with threadpool_limits(limits=1):
            estimator.fit(x.loc[tr,names],oof.loc[tr,"y"].to_numpy(np.int8)); p=estimator.predict_proba(x.loc[va,names])[:,1]
        y=oof.loc[va,"y"].to_numpy(np.int8); ids=oof.loc[va,"record_id"].to_numpy(np.int64)
        result={"fold":fold,"rows":len(y),"origins":int(oof.loc[va,"origin_program"].nunique()),"seconds":time.perf_counter()-start,"metrics":metrics(y,p)}
        np.savez_compressed(pp,record_id=ids,y=y,origin_program=oof.loc[va,"origin_program"].astype(str).to_numpy(dtype=str),fold=np.full(len(y),fold,np.int8),probability=p)
        rp.write_text(json.dumps(result,indent=2)+"\n"); fold_results.append(result)
        print(f"ablation fold {fold}: macroF1={result['metrics']['macro_f1']:.6f}",flush=True)
        del estimator,p;gc.collect()
    pred=pd.concat([pd.DataFrame({k:np.load(RUNS/f"fold{i}.npz",allow_pickle=False)[k] for k in ("record_id","y","origin_program","fold","probability")}) for i in (1,2,3)],ignore_index=True).sort_values("record_id")
    baseline=pd.concat([pd.DataFrame({k:np.load(BASELINE/f"extra_trees_baseline_fold{i}.npz",allow_pickle=False)[k] for k in ("record_id","y","origin_program","fold","probability")}) for i in (1,2,3)],ignore_index=True).sort_values("record_id")
    if not np.array_equal(pred.record_id,baseline.record_id) or not np.array_equal(pred.y,baseline.y) or not np.array_equal(pred.fold,baseline.fold): raise ValueError("Ablation does not match baseline rows/folds")
    full60=[json.loads((OUT/"mutation_diff_folds_matched"/f"fold{i}.json").read_text())["metrics"]["macro_f1"] for i in (1,2,3)]
    report={"schema":"qmutbench-feature-ablation-v1","status":"complete","trees":60,"features":names,"excluded_features":removed,"pooled_metrics":metrics(pred.y.to_numpy(np.int8),pred.probability.to_numpy(float)),"matched_60_tree_baseline_metrics":metrics(baseline.y.to_numpy(np.int8),baseline.probability.to_numpy(float)),"pooled_macro_f1_delta_vs_baseline":metrics(pred.y.to_numpy(np.int8),pred.probability.to_numpy(float))["macro_f1"]-metrics(baseline.y.to_numpy(np.int8),baseline.probability.to_numpy(float))["macro_f1"],"fold_macro_f1":[r["metrics"]["macro_f1"] for r in fold_results],"fold_macro_f1_delta_vs_baseline":[fold_results[i-1]["metrics"]["macro_f1"]-json.loads((BASELINE/f"extra_trees_baseline_fold{i}.json").read_text())["metrics_at_threshold_0_5"]["macro_f1"] for i in (1,2,3)],"full_candidate_60_tree_macro_f1":0.7885594340039961,"full_candidate_60_tree_fold_macro_f1":full60,"source_sha256":sha256_file(Path("data/qmutbench_ingestion/raw/official_all_data_20261015.csv")),"runtime_seconds_by_fold":[r["seconds"] for r in fold_results],"interpretation":"Group ablation preserves numeric edit geometry and flags while removing all five learned gate/alignment category fields; compare on exact same grouped folds and threshold."}
    (RUNS/"summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report

if __name__=="__main__":print(json.dumps(run(),indent=2))
