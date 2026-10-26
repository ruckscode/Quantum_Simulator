"""Fold-local native-categorical histogram boosting comparison."""
from __future__ import annotations

import gc, json, time
from pathlib import Path
import numpy as np, pandas as pd, psutil
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from experiments.qmutbench_exhaustive_campaign import BASE_FEATURES, FEATURES, CATEGORICAL, OUT, load_data
from experiments.qmutbench_qasm_model_comparison import MEMORY_GUARD_BYTES, metrics

RUNS=OUT/"histogram_boosting_screen"
BASE_CATS=["algorithm","operation","gate","dominant_state"]
CAT=BASE_CATS+CATEGORICAL

def run():
    x,oof=load_data(); RUNS.mkdir(parents=True,exist_ok=True)
    summary={"status":"running","model":"HistGradientBoostingClassifier","params":{"max_iter":100,"max_leaf_nodes":31,"min_samples_leaf":30,"learning_rate":0.1,"l2_regularization":1.0,"class_weight":"balanced","early_stopping":False,"categorical_features":"from_dtype"},"folds":{}}
    (RUNS/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    for variant,names in (("baseline",BASE_FEATURES),("mutation_diff",BASE_FEATURES+FEATURES)):
        for fold in (1,2,3):
            stage=f"{variant}_fold{fold}"; rp=RUNS/f"{stage}.json"; pp=RUNS/f"{stage}.npz"
            if rp.exists() and pp.exists():
                summary["folds"][stage]=json.loads(rp.read_text()); continue
            if psutil.virtual_memory().available<MEMORY_GUARD_BYTES: raise MemoryError("Memory guard reached")
            tr=oof.fold.to_numpy()!=fold; va=~tr
            train=x.loc[tr,names].copy(); valid=x.loc[va,names].copy()
            # Learn category vocabulary only from this training partition.
            for col in [c for c in CAT if c in names]:
                categories=pd.Index(train[col].astype(str).unique())
                train[col]=pd.Categorical(train[col].astype(str),categories=categories)
                valid[col]=pd.Categorical(valid[col].astype(str),categories=categories)
            est=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=31,min_samples_leaf=30,learning_rate=.1,l2_regularization=1.0,class_weight="balanced",early_stopping=False,categorical_features="from_dtype",random_state=20261025+fold)
            start=time.perf_counter()
            with threadpool_limits(limits=1):
                est.fit(train,oof.loc[tr,"y"].to_numpy(np.int8)); p=est.predict_proba(valid)[:,1]
            y=oof.loc[va,"y"].to_numpy(np.int8); ids=oof.loc[va,"record_id"].to_numpy(np.int64)
            result={"stage":stage,"fold":fold,"rows":len(y),"origins":int(oof.loc[va,"origin_program"].nunique()),"seconds":time.perf_counter()-start,"metrics":metrics(y,p)}
            np.savez_compressed(pp,record_id=ids,y=y,origin_program=oof.loc[va,"origin_program"].astype(str).to_numpy(dtype=str),fold=np.full(len(y),fold,np.int8),probability=p)
            rp.write_text(json.dumps(result,indent=2)+"\n"); summary["folds"][stage]=result
            (RUNS/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
            print(f"{stage}: macroF1={result['metrics']['macro_f1']:.6f}; seconds={result['seconds']:.1f}",flush=True)
            del est,train,valid,p; gc.collect()
    report={"status":"complete","model":summary["model"],"params":summary["params"],"folds":summary["folds"]}
    for v in ("baseline","mutation_diff"):
        frames=[np.load(RUNS/f"{v}_fold{i}.npz",allow_pickle=False) for i in (1,2,3)]
        pred=pd.DataFrame({k:np.concatenate([p[k] for p in frames]) for k in ("record_id","y","origin_program","fold","probability")}).sort_values("record_id")
        report[v+"_pooled_metrics"]=metrics(pred.y.to_numpy(np.int8),pred.probability.to_numpy(float))
        report[v+"_fold_macro_f1"]=[summary["folds"][f"{v}_fold{i}"]["metrics"]["macro_f1"] for i in (1,2,3)]
    report["pooled_macro_f1_delta"]=report["mutation_diff_pooled_metrics"]["macro_f1"]-report["baseline_pooled_metrics"]["macro_f1"]
    report["fold_macro_f1_deltas"]=[report["mutation_diff_fold_macro_f1"][i]-report["baseline_fold_macro_f1"][i] for i in range(3)]
    (RUNS/"summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report

if __name__=="__main__": print(json.dumps(run(),indent=2))
