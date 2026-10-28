"""Replay OOF predictions from the selected newly trained NumPy QNN weights."""
from __future__ import annotations
import csv,json,os,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import train_resource_light_qnn_tasks as binary
import train_qmutbench_threeclass_light_qnn as qmut
C=binary.CAMPAIGN
OUT=ROOT/"reports/experiments/qml_standalone_accuracy_optimization_20261028"

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))

def replay_binary(task,run_name,seed):
    manifest=read(C/task/"normalized_manifest.csv")
    y=np.asarray([int(r["label"]) for r in manifest]);groups=[r["bug_id"] for r in manifest]
    X=np.asarray([binary.features_from_text(r["code"],8) for r in manifest],dtype=float)
    preds=read(C/task/run_name/"oof_predictions.csv")
    saved={(r["model"],r["record_id"]):r for r in preds}
    models=sorted({r["model"] for r in preds})
    errors=[];count=0
    for fold,(tr,va) in enumerate(binary.grouped_folds(y,groups),1):
        xtr,xva,_=binary.minmax_fit_apply(X[tr].tolist(),X[va].tolist())
        for model in models:
            folder=C/task/run_name/model/f"fold_{fold}"
            with np.load(folder/"checkpoint.npz",allow_pickle=False) as z:
                theta=z["theta"];head=z["head"];bias=float(z["bias"]);cfg=json.loads(str(z["config"].item()))
            tr_x=np.asarray(xtr)[:,:cfg["features"]];va_x=np.asarray(xva)[:,:cfg["features"]]
            if cfg["encoding"]=="amplitude":
                dim=2**cfg["qubits"]
                tr_x=np.pad(tr_x,((0,0),(0,max(0,dim-tr_x.shape[1]))))[:,:dim]
                va_x=np.pad(va_x,((0,0),(0,max(0,dim-va_x.shape[1]))))[:,:dim]
            p=binary.predict_prob(va_x,theta,head,bias,cfg)
            for j,i in enumerate(va):
                r=saved[(model,manifest[i]["record_id"])]
                if int(r["y_true"])!=int(y[i]) or int(r["fold"])!=fold:errors.append(f"label/fold mismatch {task} {model} {i}")
                if abs(float(r["probability"])-float(p[j]))>2e-7:errors.append(f"probability replay mismatch {task} {model} fold{fold} id={manifest[i]['record_id']}")
                count+=1
    return {"task":task,"run":run_name,"seed":seed,"replayed_predictions":count,"models":len(models),"folds":5,"errors":errors}

def replay_qmut(run_name):
    rows=read(C/"qmutbench_threeclass_compact.csv")
    split=read(ROOT/"reports/experiments/qmutbench_three_class_qnn_campaign_20261027/grouped_split_manifest.csv")
    features=json.loads((ROOT/"reports/experiments/qmutbench_three_class_qnn_campaign_20261027/features_by_fold.json").read_text(encoding="utf-8"))
    saved=read(C/run_name/"oof_predictions.csv")
    model="amplitude_hybrid_2q_d2_lr008"
    pred={(r["record_id"]):r for r in saved if r["model"]==model}
    y=np.asarray([int(r["y_true"]) for r in rows]);errors=[];count=0
    for fold in (1,2,3):
        role={int(r["record_id"]):r["role"] for r in split if int(r["fold"])==fold}
        vi=np.asarray([i for i,r in enumerate(rows) if role[int(r["record_id"])]=="validation"])
        ti=np.asarray([i for i,r in enumerate(rows) if role[int(r["record_id"])]=="train"])
        xt,xv=qmut.encode_saved_fold(rows,ti,vi,features[fold-1]["features"])
        folder=C/run_name/model/f"fold_{fold}"
        with np.load(folder/"checkpoint.npz",allow_pickle=False) as z:
            theta=z["theta"];head=z["head"];bias=z["bias"];cfg=json.loads(str(z["config"].item()))
        p,_=qmut.probs(xv[:,:cfg["features"]],theta,head,bias,cfg)
        for j,i in enumerate(vi):
            rid=rows[i]["record_id"];r=pred[rid]
            if int(r["y_true"])!=int(y[i]) or int(r["fold"])!=fold:errors.append(f"QMut label/fold mismatch {rid}")
            for c in range(3):
                if abs(float(r[f"prob_{c}"])-float(p[j,c]))>2e-7:errors.append(f"QMut probability replay mismatch fold{fold} id={rid} class{c}")
            count+=1
    return {"task":"QMutBench three-class research","run":run_name,"model":model,"replayed_predictions":count,"folds":3,"errors":errors}

def main():
    runs=[]
    for task in ("bugs4q_robust_execution","qsimbench_backend_fingerprint"):
        for run,seed in (("standalone_optimization_sweep4",28102026),("standalone_replicate_seed4321",4321),("standalone_replicate_seed2027",2027)):
            if (C/task/run/"summary.json").exists():runs.append(replay_binary(task,run,seed))
    for run in ("qmutbench_threeclass_qnn_sweep24","qmutbench_threeclass_standalone_seed4321"):
        if (C/run/"summary.json").exists():runs.append(replay_qmut(run))
    errors=[e for r in runs for e in r["errors"]]
    report={"status":"verified" if not errors else "failed","runs":runs,"total_replayed_predictions":sum(r["replayed_predictions"] for r in runs),
            "max_probability_tolerance":2e-7,"errors":errors,"frozen_test_accessed":False}
    (OUT/"checkpoint_replay_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    if errors:raise AssertionError(f"checkpoint replay errors: {errors[:5]}")
    print(json.dumps({"status":report["status"],"runs":len(runs),"replayed_predictions":report["total_replayed_predictions"],"frozen_test_accessed":False},indent=2))

if __name__=="__main__":main()
