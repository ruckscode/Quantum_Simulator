"""Matched classical OOF benchmark on QMutBench development folds only."""
from __future__ import annotations
import csv, json, time
from pathlib import Path
import sys
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.svm import LinearSVC

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/"reports/experiments/qmutbench_classical_development_matched_20261030"
TMP=OUT/"work"
# Match the incumbent QNN's fold-local encoder seed exactly (config seed + fold).
SEED=20261028

def models():
    return {
        "dummy_prior":DummyClassifier(strategy="prior"),
        "logistic":LogisticRegression(max_iter=300,solver="lbfgs",random_state=SEED),
        "logistic_balanced":LogisticRegression(max_iter=300,solver="lbfgs",class_weight="balanced",random_state=SEED),
        "linear_svc":LinearSVC(C=1.0,dual="auto",random_state=SEED),
        "sgd_logistic":SGDClassifier(loss="log_loss",alpha=1e-4,max_iter=1000,tol=1e-3,random_state=SEED),
        "extra_trees":ExtraTreesClassifier(n_estimators=64,max_depth=16,min_samples_leaf=3,max_features="sqrt",n_jobs=-1,random_state=SEED),
        "random_forest":RandomForestClassifier(n_estimators=64,max_depth=16,min_samples_leaf=3,max_features="sqrt",n_jobs=-1,random_state=SEED),
        "hist_gradient_boosting":HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=31,l2_regularization=1.0,random_state=SEED),
    }

def scores(y,p):
    return {"records":int(len(y)),"accuracy":float(accuracy_score(y,p)),"macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),"confusion_matrix_labels_0_1":confusion_matrix(y,p,labels=[0,1]).tolist()}

def main():
    from qmutbench_qml.training.run_full_streaming import _fit_encoder,_write_memmaps,_close_memmaps
    from qmutbench_qml.data.streaming import DevelopmentStream
    OUT.mkdir(parents=True,exist_ok=True); TMP.mkdir(exist_ok=True)
    index=TMP/"development_index.sqlite"
    if index.exists(): index.unlink()
    stream=DevelopmentStream(ROOT,index,archive_authoritative=True)
    audit=stream.build_index()
    (OUT/"stream_audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    manifest={"status":"running","dataset":"QMutBench binary development split","target":{"0":"survived","1":"killed"},"rows":578117,"origins":305,"folds":5,"frozen_test_accessed":False,"feature_protocol":"Same fold-local ReducedFeatureEncoder as QNN; fit on up to 5,000 balanced training-fold rows only; four MI-selected features; same saved origin-disjoint folds.","feature_names_by_fold":{},"models":list(models()),"seed":SEED,"fold_results":{}}
    (OUT/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    for fold in range(1,6):
        t0=time.time(); encoder,fit_count=_fit_encoder(stream,fold,4,SEED+fold,5000)
        manifest["feature_names_by_fold"][str(fold)]=encoder.selected_names
        ntr=sum(audit["fold_record_counts"][str(f)] for f in range(1,6) if f!=fold); nv=audit["fold_record_counts"][str(fold)]
        tr=_write_memmaps(stream,fold,"training",encoder,TMP,f"f{fold}_train",ntr,4); va=_write_memmaps(stream,fold,"validation",encoder,TMP,f"f{fold}_valid",nv,4)
        Xtr=np.memmap(tr[0],mode="r",dtype=np.float64,shape=(ntr,4)); ytr=np.memmap(tr[1],mode="r",dtype=np.int64,shape=(ntr,))
        Xva=np.memmap(va[0],mode="r",dtype=np.float64,shape=(nv,4)); yva=np.memmap(va[1],mode="r",dtype=np.int64,shape=(nv,))
        ids=np.memmap(va[2],mode="r",dtype=np.int64,shape=(nv,)); org=np.memmap(va[3],mode="r",dtype=np.int16,shape=(nv,))
        predictions={}
        for name,model in models().items():
            model.fit(Xtr,ytr); predictions[name]=model.predict(Xva).astype(np.int8)
        path=OUT/f"fold_{fold}_predictions.csv"
        with path.open("w",encoding="utf-8",newline="") as f:
            w=csv.writer(f); w.writerow(["record_id","origin_code","fold","y_true","model","y_pred"])
            for name,pred in predictions.items():
                for rid,oid,actual,guess in zip(ids,org,yva,pred): w.writerow([int(rid),int(oid),fold,int(actual),name,int(guess)])
        _close_memmaps(Xtr,ytr,Xva,yva,ids,org)
        for p in (*tr[:4],*va[:4]):
            try: Path(p).unlink()
            except FileNotFoundError: pass
        manifest["fold_results"][str(fold)]={"status":"complete","records":nv,"training_records":ntr,"encoder_fit_records":fit_count,"feature_names":encoder.selected_names,"prediction_file":path.name,"elapsed_seconds":round(time.time()-t0,2)}
        manifest["completed_folds"]=fold
        (OUT/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
        print(f"fold {fold}/5 complete: train={ntr:,}, validation={nv:,}; features={encoder.selected_names}; elapsed={time.time()-t0:.1f}s",flush=True)
    rows=[]
    for name in models():
        ys=[]; ps=[]
        for fold in range(1,6):
            with (OUT/f"fold_{fold}_predictions.csv").open(encoding="utf-8",newline="") as f:
                for row in csv.DictReader(f):
                    if row["model"]==name: ys.append(int(row["y_true"])); ps.append(int(row["y_pred"]))
        if len(ys)!=578117: raise ValueError(f"Incomplete OOF predictions for {name}: {len(ys)}")
        rows.append({"model":name,**scores(np.asarray(ys),np.asarray(ps))})
    rows.sort(key=lambda x:(-x["accuracy"],-x["macro_f1"]))
    (OUT/"pooled_oof_metrics.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
    with (OUT/"pooled_oof_metrics.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["model","records","accuracy","macro_f1","balanced_accuracy","confusion_matrix_labels_0_1"]); w.writeheader(); w.writerows(rows)
    manifest["status"]="complete"; manifest["pooled_oof_metrics"]="pooled_oof_metrics.json"
    (OUT/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(rows,indent=2),flush=True)

if __name__=="__main__": main()
