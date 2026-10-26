"""Nested grouped validation and inner-only calibration/threshold selection."""
from __future__ import annotations

import gc, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from experiments.qmutbench_exhaustive_campaign import (BASE_FEATURES, FEATURES, CATEGORICAL,
    model as et_model, load_data)
from experiments.qmutbench_qasm_model_comparison import MEMORY_GUARD_BYTES

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data/qmutbench_ml/final_research_v1/nested_outer"
SEED=20261026
THRESHOLDS=np.round(np.arange(.30,.701,.025),3)
CANDIDATES={"et_base":BASE_FEATURES,"et_mutation":BASE_FEATURES+FEATURES,"hgb_mutation":BASE_FEATURES+FEATURES}

def score(y,p,t=.5):
    pred=(p>=t).astype(np.int8)
    return {"macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
      "balanced_accuracy":float(balanced_accuracy_score(y,pred)),"accuracy":float(accuracy_score(y,pred)),
      "precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),
      "brier":float(brier_score_loss(y,p)),"roc_auc":float(roc_auc_score(y,p)),
      "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist()}

def ece(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); total=0.
    for i in range(bins):
        mask=(p>=edges[i])&(p<(edges[i+1] if i<bins-1 else edges[i+1]+1e-12))
        if mask.any(): total+=mask.mean()*abs(float(y[mask].mean())-float(p[mask].mean()))
    return float(total)

def fit_hgb(x,y,names,seed):
    d=x[names].copy(); cats=[c for c in names if c in ("algorithm","operation","gate","dominant_state")+tuple(CATEGORICAL)]
    vocab={}
    for c in cats:
        vocab[c]=pd.Index(d[c].astype(str).unique())
        d[c]=pd.Categorical(d[c].astype(str),categories=vocab[c])
    m=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=31,min_samples_leaf=30,learning_rate=.1,
        l2_regularization=1.,class_weight="balanced",early_stopping=False,categorical_features="from_dtype",random_state=seed)
    m.fit(d,y)
    return m,vocab

def hgb_predict(m,x,names,vocab):
    d=x[names].copy()
    for c,cats in vocab.items():
        values=d[c].astype(str)
        d[c]=pd.Categorical(values.where(values.isin(cats),np.nan),categories=cats)
    return m.predict_proba(d)[:,1]

def fit_predict(key,xtr,ytr,xva,names,seed,trees):
    if key.startswith("et_"):
        m=et_model(names,seed,trees)
        m.fit(xtr[names],ytr); return m.predict_proba(xva[names])[:,1]
    m,v=fit_hgb(xtr,ytr,names,seed); p=hgb_predict(m,xva,names,v); del m; return p

def calibrator(kind,p,y):
    if kind=="raw": return None
    if kind=="platt":
        z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))).reshape(-1,1)
        return LogisticRegression(C=1.,solver="lbfgs",max_iter=200).fit(z,y)
    return IsotonicRegression(out_of_bounds="clip").fit(p,y)

def apply_cal(kind,model,p):
    if kind=="raw": return p
    if kind=="platt":
        q=np.clip(p,1e-6,1-1e-6); z=np.log(q/(1-q)).reshape(-1,1)
        return model.predict_proba(z)[:,1]
    return model.predict(p)

def inner_calibration(p,y,fold):
    out={}
    # Cross-fit calibrators: calibrator is fitted on the opposite inner fold.
    for kind in ("raw","platt","isotonic"):
        q=np.empty(len(y),float)
        for f in (0,1):
            tr=fold!=f; va=fold==f
            if kind=="raw": q[va]=p[va]
            else:
                m=calibrator(kind,p[tr],y[tr]); q[va]=apply_cal(kind,m,p[va])
        out[kind]=q
    return out

def run():
    x,oof=load_data(); OUT.mkdir(parents=True,exist_ok=True)
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    splits=list(splitter.split(np.zeros(len(oof)),oof.y.to_numpy(np.int8),oof.origin_program.to_numpy()))
    all_rows=[]; summaries=[]
    for outer,(tr,va) in enumerate(splits,1):
        jp,npz=OUT/f"fold{outer}.json",OUT/f"fold{outer}.npz"
        if jp.exists() and npz.exists():
            r=json.loads(jp.read_text()); z=np.load(npz,allow_pickle=False)
            if r.get("config")!= "nested-v1-seed-20261026-5x2": raise ValueError("Stale fold configuration")
            summaries.append(r); all_rows.append({k:z[k] for k in z.files}); continue
        if psutil.virtual_memory().available<MEMORY_GUARD_BYTES: raise MemoryError("Memory guard before nested outer fold")
        groups=oof.origin_program.to_numpy()
        if set(groups[tr])&set(groups[va]): raise ValueError("Outer origin leakage")
        ytr=oof.y.to_numpy(np.int8)[tr]; yva=oof.y.to_numpy(np.int8)[va]
        inner=StratifiedGroupKFold(n_splits=2,shuffle=True,random_state=SEED+outer)
        inner_splits=list(inner.split(np.zeros(len(tr)),ytr,groups[tr]))
        ip={k:np.full(len(tr),np.nan,float) for k in CANDIDATES}; ifold=np.full(len(tr),-1,np.int8)
        t0=time.perf_counter()
        for fi,(it,iv) in enumerate(inner_splits):
            if set(groups[tr][it])&set(groups[tr][iv]): raise ValueError("Inner origin leakage")
            ifold[iv]=fi
            for key,names in CANDIDATES.items():
                if psutil.virtual_memory().available<MEMORY_GUARD_BYTES: raise MemoryError("Memory guard before inner fit")
                with threadpool_limits(limits=1):
                    ip[key][iv]=fit_predict(key,x.iloc[tr[it]],ytr[it],x.iloc[tr[iv]],names,SEED+outer*10+fi,30)
                print(f"outer {outer}/5 inner {fi+1}/2 {key} complete",flush=True)
        if any(np.isnan(v).any() for v in ip.values()) or (ifold<0).any(): raise ValueError("Inner OOF incomplete")
        choices=[]
        for key in CANDIDATES:
            calibrated=inner_calibration(ip[key],ytr,ifold)
            for kind,p in calibrated.items():
                for threshold in THRESHOLDS:
                    val=float(f1_score(ytr,(p>=threshold).astype(np.int8),average="macro",zero_division=0))
                    # Stable tie order favors simpler model, raw calibration, then threshold near .5.
                    rank=(list(CANDIDATES).index(key),("raw","platt","isotonic").index(kind),abs(float(threshold)-.5),float(threshold))
                    choices.append((val,rank,key,kind,float(threshold)))
        best=max(choices,key=lambda c:(c[0],tuple(-v for v in c[1][:3]),-c[1][3]))
        _,_,selected,cal_kind,threshold=best
        # Train outer models. Retain raw fixed-threshold comparisons and selected model.
        outerp={}; outer_raw={}
        outer_keys=["et_base","et_mutation"]+([selected] if selected=="hgb_mutation" else [])
        for key in outer_keys:
            if psutil.virtual_memory().available<MEMORY_GUARD_BYTES: raise MemoryError("Memory guard before outer fit")
            with threadpool_limits(limits=1): outer_raw[key]=fit_predict(key,x.iloc[tr],ytr,x.iloc[va],names,SEED+100+outer,100)
        # Fit final selected calibration on all inner OOF predictions; no outer-label access.
        cm=calibrator(cal_kind,ip[selected],ytr)
        outerp=apply_cal(cal_kind,cm,outer_raw[selected])
        ids=oof.record_id.to_numpy()[va].astype(np.int64)
        origin=groups[va].astype(str)
        row={"record_id":ids,"y":yva,"origin_program":origin,"outer_fold":np.full(len(va),outer,np.int8),
             "base_p":outer_raw["et_base"],"mutation_p":outer_raw["et_mutation"],"selected_p":outerp,
             "selected_raw_p":outer_raw[selected],"selected_threshold":np.full(len(va),threshold),
             "selected_model":np.full(len(va),selected,dtype="U32"),"selected_calibration":np.full(len(va),cal_kind,dtype="U16")}
        np.savez_compressed(npz,**row)
        result={"outer_fold":outer,"config":"nested-v1-seed-20261026-5x2","train_rows":len(tr),"test_rows":len(va),
          "train_origins":int(len(set(groups[tr]))),"test_origins":int(len(set(groups[va]))),"selected_model":selected,
          "selected_calibration":cal_kind,"selected_threshold":threshold,"inner_macro_f1":best[0],
          "metrics":{"baseline_et_raw_0_5":score(yva,outer_raw["et_base"]),"mutation_et_raw_0_5":score(yva,outer_raw["et_mutation"]),
            "nested_selected":score(yva,outerp,threshold),"selected_raw_0_5":score(yva,outer_raw[selected])},
          "seconds":time.perf_counter()-t0}
        jp.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        summaries.append(result); all_rows.append(row); print(f"outer {outer}/5 selected {selected}/{cal_kind}/{threshold:.3f}; nested macro-F1={result['metrics']['nested_selected']['macro_f1']:.6f}",flush=True)
        del ip,outer_raw,outerp,cm; gc.collect()
    pred={k:np.concatenate([r[k] for r in all_rows]) for k in all_rows[0]}
    order=np.argsort(pred["record_id"]); pred={k:v[order] for k,v in pred.items()}
    y=pred["y"].astype(np.int8); p=pred["selected_p"].astype(float); th=pred["selected_threshold"].astype(float)
    selected_pred=(p>=th).astype(np.int8)
    nested={"macro_f1":float(f1_score(y,selected_pred,average="macro",zero_division=0)),"balanced_accuracy":float(balanced_accuracy_score(y,selected_pred)),
      "accuracy":float(accuracy_score(y,selected_pred)),"precision":float(precision_score(y,selected_pred,zero_division=0)),
      "recall":float(recall_score(y,selected_pred,zero_division=0)),"brier":float(brier_score_loss(y,p)),"ece_10":ece(y,p),
      "roc_auc":float(roc_auc_score(y,p)),"confusion_matrix":confusion_matrix(y,selected_pred,labels=[0,1]).tolist()}
    report={"schema":"qmutbench-nested-grouped-research-v1","status":"complete","outer_folds":5,"inner_folds":2,
      "threshold_grid":THRESHOLDS.tolist(),"candidates":list(CANDIDATES),"outer_selection_isolation":"outer labels used only for final metrics",
      "historical_selection_caveat":"Mutation-aware feature design and candidate pipeline were informed by prior use of this same development corpus; nested estimates are development-only and not independent external evidence.",
      "nested_selected_metrics":nested,"fixed_baseline_et_raw_0_5":score(y,pred["base_p"].astype(float)),
      "fixed_mutation_et_raw_0_5":score(y,pred["mutation_p"].astype(float)),"folds":summaries,
      "selected_model_counts":pd.Series([s["selected_model"] for s in summaries]).value_counts().to_dict(),
      "selected_calibration_counts":pd.Series([s["selected_calibration"] for s in summaries]).value_counts().to_dict(),
      "thresholds_by_fold":[s["selected_threshold"] for s in summaries],"row_count":len(y),"origin_count":len(set(pred["origin_program"]))}
    (OUT/"summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("nested_selected_metrics","fixed_baseline_et_raw_0_5","fixed_mutation_et_raw_0_5","selected_model_counts","selected_calibration_counts","thresholds_by_fold")},indent=2),flush=True)

if __name__=="__main__": run()
