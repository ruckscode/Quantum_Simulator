"""Matched fold-local classical controls for the 205-row QMutBench recovery run."""
from __future__ import annotations
import csv, json
import time
from pathlib import Path
import numpy as np
import psutil
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import sys
ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028"
RUN = CAMPAIGN / "qmutbench_threeclass_resumable_recovery_20261029"
REFERENCE = ROOT / "reports/experiments/qmutbench_three_class_qnn_campaign_20261027"
sys.path.insert(0, str(ROOT / "experiments"))
import train_qmutbench_threeclass_light_qnn as trainer

def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))

def metrics(rows, pred):
    y=np.asarray([int(r["y_true"]) for r in rows]); p=np.asarray(pred)
    return {"accuracy":float(accuracy_score(y,p)),"macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
            "balanced_accuracy":float(balanced_accuracy_score(y,p))}

def main():
    started=time.perf_counter(); process=psutil.Process(); rss_start=process.memory_info().rss
    available_start=psutil.virtual_memory().available
    source=read(trainer.CAMPAIGN/"qmutbench_threeclass_compact.csv")
    split=read(REFERENCE/"grouped_split_manifest.csv")
    feature_manifest=json.loads((REFERENCE/"features_by_fold.json").read_text(encoding="utf-8"))
    y=np.asarray([int(r["y_true"]) for r in source])
    predictions={name:[] for name in ("logistic_4f","rbf_svc_4f","extra_trees_4f")}
    fit_seconds={name:0. for name in predictions}; rss_peak=rss_start
    for fold in (1,2,3):
        roles={int(r["record_id"]):r["role"] for r in split if int(r["fold"])==fold}
        ti=np.asarray([i for i,r in enumerate(source) if roles.get(int(r["record_id"]))=="train"])
        vi=np.asarray([i for i,r in enumerate(source) if roles.get(int(r["record_id"]))=="validation"])
        xtr,xva=trainer.encode_saved_fold(source,ti,vi,feature_manifest[fold-1]["features"])
        xtr=xtr[:,:4]; xva=xva[:,:4]
        models=[("logistic_4f",LogisticRegression(C=1.,class_weight="balanced",max_iter=500,solver="lbfgs",random_state=20261028+fold)),
                ("rbf_svc_4f",SVC(C=1.,kernel="rbf",class_weight="balanced",probability=True,random_state=20261028+fold)),
                ("extra_trees_4f",ExtraTreesClassifier(n_estimators=100,max_depth=12,min_samples_leaf=2,max_features="sqrt",class_weight="balanced",n_jobs=1,random_state=20261028+fold))]
        for name,model in models:
            fit_started=time.perf_counter(); model.fit(xtr,y[ti]); fit_seconds[name]+=time.perf_counter()-fit_started
            p=model.predict(xva); rss_peak=max(rss_peak,process.memory_info().rss)
            for j,i in enumerate(vi):
                predictions[name].append({"record_id":source[i]["record_id"],"origin_program":source[i]["origin_program"],"fold":fold,"model":name,"y_true":int(y[i]),"y_pred":int(p[j])})
    baseline_rows=[r for values in predictions.values() for r in values]
    with (RUN/"matched_classical_oof.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(baseline_rows[0]));w.writeheader();w.writerows(baseline_rows)
    qnn=read(RUN/"oof_predictions.csv")
    groups=sorted({r["origin_program"] for r in qnn}); rng=np.random.default_rng(20261029)
    intervals={}
    def group_confusions(rows, groups):
        mats=[]
        for group in groups:
            part=[r for r in rows if r["origin_program"]==group]
            cm=np.zeros((3,3),dtype=np.int64)
            for r in part: cm[int(r["y_true"]),int(r["y_pred"])]+=1
            mats.append(cm)
        return np.asarray(mats)
    def bootstrap_macro_f1_delta(qrows,brows,groups):
        qcm=group_confusions(qrows,groups); bcm=group_confusions(brows,groups)
        weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
        qsum=np.einsum('bg,gij->bij',weights,qcm); bsum=np.einsum('bg,gij->bij',weights,bcm)
        def macro(cm):
            tp=np.diagonal(cm,axis1=1,axis2=2); den=cm.sum(axis=2)+cm.sum(axis=1)
            f=np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0)
            return f.mean(axis=1)
        return macro(qsum)-macro(bsum)
    for qnn_name in sorted({r["model"] for r in qnn}):
        qrows=[r for r in qnn if r["model"]==qnn_name]
        intervals[qnn_name]={}
        for base_name,base_rows in predictions.items():
            # Paired cluster bootstrap: resample origins, keeping each model's
            # records together; duplicate sampled groups are repeated as clusters.
            diffs=bootstrap_macro_f1_delta(qrows,base_rows,groups)
            intervals[qnn_name][base_name]={"macro_f1_delta":metrics(qrows,[int(r["y_pred"]) for r in qrows])["macro_f1"]-metrics(base_rows,[int(r["y_pred"]) for r in base_rows])["macro_f1"],
                "group_bootstrap_95_ci":[float(np.quantile(diffs,.025)),float(np.quantile(diffs,.975))],"bootstrap_draws":2000,"groups":len(groups)}
    report={"schema":"qmutbench-threeclass-resumable-matched-classical-v1","status":"COMPLETED","rows":len(source),"origins":len(groups),"folds":3,
        "feature_contract":"same saved origin folds, same reconstructed fold-selected inputs and train-only numeric scaling as the QNN recovery run; four inputs; no identifiers",
        "resource_usage":{"runtime_seconds":time.perf_counter()-started,"classical_fit_seconds_by_model":fit_seconds,
            "available_ram_start_mib":available_start/2**20,"available_ram_end_mib":psutil.virtual_memory().available/2**20,
            "process_rss_start_mib":rss_start/2**20,"process_rss_sampled_peak_mib":rss_peak/2**20},
        "classical_metrics":{name:metrics(values,[int(r["y_pred"]) for r in values]) for name,values in predictions.items()},
        "qnn_metrics":{name:metrics([r for r in qnn if r["model"]==name],[int(r["y_pred"]) for r in qnn if r["model"]==name]) for name in sorted({r["model"] for r in qnn})},
        "qnn_minus_classical_macro_f1":intervals,"frozen_test_accessed":False}
    (RUN/"matched_classical_comparison.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
