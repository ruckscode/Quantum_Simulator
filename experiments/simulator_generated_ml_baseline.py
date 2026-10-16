"""Controlled simulator-generated ML baseline; not official benchmark or IBM accuracy."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from quantum_simulator import QuantumSimulator
from quantum_simulator.hardware_model import HardwareModel

LABELS = ["NORMAL", "PROGRAM_FAULT", "SIMULATED_HARDWARE_NOISE"]
FAMILIES = [
 [("h",0),("measure",[0])], [("x",0),("measure",[0])],
 [("ry",0,0.7),("measure",[0])], [("h",0),("x",0),("measure",[0])],
 [("x",0),("h",0),("measure",[0])], [("h",0),("rz",0,0.8),("measure",[0])],
 [("h",0),("cx",0,1),("measure",[0,1])],
 [("x",0),("h",1),("cx",0,1),("measure",[0,1])],
 [("h",0),("h",1),("cx",0,1),("measure",[0,1])],
 [("ry",0,0.7),("cx",0,1),("measure",[0,1])],
 [("h",0),("cx",0,1),("x",1),("measure",[0,1])],
 [("x",0),("x",1),("ccx",0,1,2),("measure",[0,1,2])],
]

def qubit_count(ops):
    q = []
    for op in ops:
        if op[0] == "measure" and len(op)>1 and isinstance(op[1], (list,tuple)): q += list(op[1])
        else: q += [v for v in op[1:] if isinstance(v,int) and not isinstance(v,bool)]
    return max(q, default=0)+1

def mutate(ops):
    replacements = {"h":"x","x":"h","ry":"rx","rz":"rx","cx":"cz"}
    out = [tuple(x) for x in ops]
    for i, op in enumerate(out):
        if op[0] == "measure": continue
        if op[0] == "ccx": out.pop(i)
        elif op[0] in replacements: out[i] = (replacements[op[0]],)+op[1:]
        else: out.pop(i)
        return out
    return out

def run_counts(ops, n, shots, seed, hardware=None):
    rng=np.random.default_rng(seed); counts={}
    for _ in range(shots):
        sim=QuantumSimulator(n, seed=int(rng.integers(0,2**31-1)))
        sim.run_circuit(ops, hardware_model=hardware)
        outcome=str(sim.measurement_results[-1]) if sim.measurement_results else "0"*n
        counts[outcome]=counts.get(outcome,0)+1
    return counts

def dist(counts,n):
    total=max(1,sum(counts.values()))
    return np.array([counts.get(format(i,f"0{n}b"),0)/total for i in range(2**n)])

def entropy(p):
    p=p[p>0]
    return float(-np.sum(p*np.log2(p))) if len(p) else 0.0

def generate(repetitions=12, shots=128, seed=42):
    rng=np.random.default_rng(seed); rows=[]
    for fi, expected in enumerate(FAMILIES):
        n=qubit_count(expected)
        reference=dist(run_counts(expected,n,max(1024,shots*4),seed+fi*1000),n)
        broken=mutate(expected)
        hw=HardwareModel(n,backend_name="SIMULATED_READOUT_NOISE")
        for q in range(n): hw.set_qubit_properties(q,readout_error=0.25 if q==0 else 0.0)
        for label, ops, model in [("NORMAL",expected,None),("PROGRAM_FAULT",broken,None),("SIMULATED_HARDWARE_NOISE",expected,hw)]:
            for rep in range(repetitions):
                counts=run_counts(ops,n,shots,int(rng.integers(0,2**31-1)),model)
                observed=dist(counts,n)
                row={"family_id":f"family_{fi:02d}","label":label,"qubits":n,"depth":len(ops),
                     "gate_count":sum(x[0]!="measure" for x in ops),"shots":shots,
                     "tvd_from_reference":float(0.5*np.abs(reference-observed).sum()),
                     "observed_entropy":entropy(observed),"reference_entropy":entropy(reference),
                     "distinct_outcomes":int(np.count_nonzero(observed))}
                row.update({f"p_{i:0{n}b}":float(p) for i,p in enumerate(observed)})
                rows.append(row)
    return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",default="reports/simulator_generated_ml")
    parser.add_argument("--repetitions",type=int,default=12)
    parser.add_argument("--shots",type=int,default=128)
    parser.add_argument("--seed",type=int,default=42)
    a=parser.parse_args()
    if a.repetitions<2 or a.shots<16: parser.error("repetitions must be >=2 and shots >=16")
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    df=generate(a.repetitions,a.shots,a.seed)
    features=["qubits","depth","gate_count","shots","tvd_from_reference","observed_entropy","reference_entropy","distinct_outcomes"]
    features += sorted(c for c in df.columns if c.startswith("p_"))
    X=df[features].fillna(0).astype(float); y=df.label; groups=df.family_id
    tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42).split(X,y,groups))
    models={
      "Logistic Regression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42)),
      "Random Forest":RandomForestClassifier(n_estimators=250,class_weight="balanced",min_samples_leaf=2,random_state=42)
    }
    metrics=[]; preds={}
    for name,model in models.items():
        model.fit(X.iloc[tr],y.iloc[tr]); pred=model.predict(X.iloc[te]); preds[name]=pred
        metrics.append({"model":name,"accuracy":accuracy_score(y.iloc[te],pred),
          "macro_precision":precision_score(y.iloc[te],pred,average="macro",zero_division=0),
          "macro_recall":recall_score(y.iloc[te],pred,average="macro",zero_division=0),
          "macro_f1":f1_score(y.iloc[te],pred,average="macro",zero_division=0),
          "train_rows":len(tr),"test_rows":len(te),"train_families":groups.iloc[tr].nunique(),"test_families":groups.iloc[te].nunique()})
        print("\n===",name,"===\n",classification_report(y.iloc[te],pred,labels=LABELS,zero_division=0))
    met=pd.DataFrame(metrics); df.to_csv(out/"simulator_generated_ml_dataset.csv",index=False); met.to_csv(out/"simulator_generated_ml_metrics.csv",index=False)
    ax=met.set_index("model")[["accuracy","macro_f1"]].plot(kind="bar",ylim=(0,1),rot=0)
    ax.set_ylabel("Score"); ax.set_title("Simulator-generated controlled experiment — not benchmark accuracy"); ax.grid(axis="y",alpha=.25)
    plt.tight_layout(); plt.savefig(out/"simulator_generated_ml_metrics.png",dpi=160); plt.close()
    fig,ax=plt.subplots(figsize=(7,6))
    ConfusionMatrixDisplay.from_predictions(y.iloc[te],preds["Random Forest"],labels=LABELS,display_labels=LABELS,xticks_rotation=25,ax=ax,colorbar=False)
    ax.set_title("Random Forest — held-out circuit families"); plt.tight_layout(); plt.savefig(out/"simulator_generated_ml_confusion_matrix.png",dpi=160); plt.close()
    summary={"dataset_type":"controlled simulator-generated observations","rows":len(df),"circuit_families":int(groups.nunique()),
      "split":"GroupShuffleSplit by circuit family","metrics":metrics,"official_benchmark_accuracy":False,"physical_ibm_hardware":False,
      "warning":"Labels are known from controlled interventions. Scores describe simulator-generated scenarios only, not official QMutBench/Bugs4Q or IBM hardware accuracy."}
    (out/"simulator_generated_ml_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("\nMETRICS\n",met.to_string(index=False)); print("\nSaved CSVs, JSON summary and charts to",out); print(summary["warning"])
if __name__=="__main__": main()
