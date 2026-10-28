"""Run bounded local NumPy statevector QNN refinements on remaining binary tasks.

Uses each task's existing grouped OOF fold assignment, trains each task
independently, and never reads a frozen-test artifact.
"""
from __future__ import annotations
import csv,hashlib,json,re,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import train_resource_light_qnn_tasks as q
SOURCE=q.CAMPAIGN
OUT=ROOT/"reports/experiments/qml_standalone_accuracy_optimization_20261028/remaining_task_sweeps"
EPOCHS=4
BATCH=64
SEED=28102026
CURRENT_TASK=""
COMPLETED_TASKS=[]

TASKS={
 "Bugs4Q core": {"manifest":"bugs4q_standalone/normalized_manifest.csv","pred":"bugs4q_standalone/qnn_gradient_fix/oof_predictions.csv","model":"angle_reupload_vqc","text":["code"]},
 "Bugs4Q-NA": {"manifest":"bugs4q_na/normalized_manifest.csv","pred":"bugs4q_na/qnn/ensemble_oof_predictions.csv","model":"quantum_equal_probability_mean","text":["code"]},
 "Bugs4Q pooled": {"manifest":"bugs4q_joint/normalized_manifest.csv","pred":"bugs4q_joint/qnn_hybrid/oof_predictions.csv","model":"angle_hybrid_readout","text":["code"]},
 "Bugs4Q core-only transfer": {"manifest":"bugs4q_joint/normalized_manifest.csv","pred":"bugs4q_joint/qnn_train_core_complete/oof_predictions.csv","model":"angle_hybrid_readout","text":["code"],"train_source":"Bugs4Q_core"},
 "Bugs4Q-NA-only transfer": {"manifest":"bugs4q_joint/normalized_manifest.csv","pred":"bugs4q_joint/qnn_train_na_complete/oof_predictions.csv","model":"angle_hybrid_readout","text":["code"],"train_source":"Bugs4Q_NA_additional"},
 "LintQ": {"manifest":"lintq_warning_triage/normalized_manifest.csv","pred":"lintq_warning_triage/qnn/oof_predictions.csv","model":"angle_hybrid_readout","text":["warning_message"]},
 "QBugs HornBro subset": {"manifest":"qbugs/normalized_manifest.csv","pred":"qbugs/qnn_hybrid/oof_predictions.csv","model":"angle_hybrid_readout","text":["code","scenario","circuit_qubits","circuit_depth","instruction_count"]},
 "Qolumbina ideal": {"manifest":"qolumbina_detection_ideal/normalized_manifest.csv","pred":"qolumbina_detection_ideal/qnn/oof_predictions.csv","model":"angle_vqc","text":["program","variant_version","backend","code"]},
 "Quantum platform comment task": {"manifest":"quantum_platform_223_comment_only/normalized_manifest.csv","pred":"quantum_platform_223_comment_only/qnn_hybrid/oof_predictions.csv","model":"angle_hybrid_readout","text":["code"],"mask_comment_labels":True},
}
CONFIGS=[
 {"name":"amplitude_3q_depth1_lr004","qubits":3,"features":8,"encoding":"amplitude","depth":1,"reupload":False,"learning_rate":.04},
 {"name":"angle_reupload_4q_depth2_lr0035","qubits":4,"features":4,"encoding":"angle","depth":2,"reupload":True,"learning_rate":.035},
]

def read(path):
 with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))

def text_for(row,spec):
 parts=[]
 for key in spec["text"]:
  value=str(row.get(key,""))
  if key=="code" and spec.get("mask_comment_labels"):
   value=re.sub(r"(?i)\b(bugs?|buggy|fix(?:ed|es)?|defects?|false\s+positive|true\s+positive)\b"," ",value)
  parts.append(value)
 return " ".join(parts)

def metric(y,p):
 return q.metrics(y,p)

def run_campaign():
 global CURRENT_TASK,COMPLETED_TASKS
 if q.check_ram()<700:raise MemoryError("remaining-task sweep preflight requires 700 MiB available RAM")
 reports=[]
 for task,spec in TASKS.items():
  CURRENT_TASK=task
  if q.check_ram()<600:raise MemoryError(f"stopping before task {task}: available RAM below 600 MiB launch margin")
  manifest=read(SOURCE/spec["manifest"]);oof=read(SOURCE/spec["pred"])
  if oof and "model" in oof[0]:oof=[r for r in oof if r.get("model")==spec["model"]]
  oof_by_id={r["record_id"]:r for r in oof}
  ids=[r["record_id"] for r in manifest]
  missing=set(ids)-set(oof_by_id);extra=set(oof_by_id)-set(ids)
  if missing or extra:raise ValueError(f"OOF/data id mismatch for {task}: missing={len(missing)}, extra={len(extra)}")
  labels=np.asarray([int(r["label"]) for r in manifest]); groups=[str(r["bug_id"]) for r in manifest]
  folds=np.asarray([int(oof_by_id[rid]["fold"]) for rid in ids])
  for group in set(groups):
   if len(set(folds[i] for i,g in enumerate(groups) if g==group))!=1:raise AssertionError(f"group split leakage in task {task}: {group}")
  X=np.asarray([q.features_from_text(text_for(r,spec),8) for r in manifest],dtype=float)
  task_dir=OUT/re.sub(r"[^a-z0-9]+","_",task.lower()).strip("_")
  task_dir.mkdir(parents=True,exist_ok=True)
  all_pred=[];fold_rows=[];start_task=time.perf_counter()
  for config in CONFIGS:
   for fold in sorted(set(folds)):
    valid=np.flatnonzero(folds==fold);train=np.flatnonzero(folds!=fold)
    if spec.get("train_source"):
     train=np.asarray([i for i in train if manifest[i].get("source")==spec["train_source"]],dtype=int)
    if len(set(labels[train]))<2 or len(set(labels[valid]))<2:raise ValueError(f"class missing from split for {task}, fold {fold}")
    xt,xv,scale=q.minmax_fit_apply(X[train].tolist(),X[valid].tolist())
    xt=np.asarray(xt)[:,:config["features"]];xv=np.asarray(xv)[:,:config["features"]]
    if config["encoding"]=="amplitude":
     dim=2**config["qubits"]
     xt=np.pad(xt,((0,0),(0,max(0,dim-xt.shape[1]))))[:,:dim]
     xv=np.pad(xv,((0,0),(0,max(0,dim-xv.shape[1]))))[:,:dim]
    gc=q.gradient_check(config)
    if not gc["passed"]:raise ArithmeticError(f"gradient check failed {task} {config['name']}: {gc}")
    folder=task_dir/config["name"]/f"fold_{fold}";folder.mkdir(parents=True,exist_ok=True)
    # The shared runner's epoch count and minibatch size are fixed in module state.
    q.STEPS=EPOCHS;q.BATCH=BATCH
    theta,head,bias,history,secs=q.fit_model(xt,labels[train].astype(float),config,SEED+int(fold),folder/"checkpoint.npz",folder/"training_history.json")
    prob=q.predict_prob(xv,theta,head,bias,config);pred=(prob>=.5).astype(int)
    m=metric(labels[valid],pred)
    fr={"task":task,"model":config["name"],"fold":int(fold),"records":len(valid),"groups":len(set(groups[i] for i in valid)),
        "train_records":len(train),"training_seconds":secs,"available_ram_mib":q.check_ram(),"gradient_check":gc,"metrics":m}
    fold_rows.append(fr)
    np.savez(folder/"preprocessing.npz",minimum=np.asarray(scale["min"]),maximum=np.asarray(scale["max"]),
             task=task,manifest_sha256=hashlib.sha256((SOURCE/spec["manifest"]).read_bytes()).hexdigest(),
             fold=int(fold),feature_fields=json.dumps(spec["text"]),seed=SEED+int(fold))
    (folder/"fold_result.json").write_text(json.dumps(fr,indent=2),encoding="utf-8")
    for j,i in enumerate(valid):
     all_pred.append({"record_id":ids[i],"bug_id":groups[i],"fold":int(fold),"model":config["name"],"y_true":int(labels[i]),"probability":float(prob[j]),"y_pred":int(pred[j])})
   model_pred=[r for r in all_pred if r["model"]==config["name"]]
   m=metric([r["y_true"] for r in model_pred],[r["y_pred"] for r in model_pred])
   model_dir=task_dir/config["name"]
   model_dir.mkdir(parents=True,exist_ok=True)
   (model_dir/"summary.json").write_text(json.dumps({"model":config["name"],"configuration":config,**m,
       "folds":len(set(folds)),"records":len(model_pred),"training_seconds":sum(x["training_seconds"] for x in fold_rows if x["model"]==config["name"]),
       "split":"reused existing selected-model grouped OOF fold assignments","group_unit":"bug_id","epochs":EPOCHS,"batch_size":BATCH,
       "optimizer":"Adam","text_fields":spec["text"],"train_source_restriction":spec.get("train_source"),"frozen_test_accessed":False},indent=2),encoding="utf-8")
   with (model_dir/"oof_predictions.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(model_pred[0]));w.writeheader();w.writerows(model_pred)
  with (task_dir/"fold_results.csv").open("w",newline="",encoding="utf-8") as f:
   w=csv.DictWriter(f,fieldnames=list(fold_rows[0]));w.writeheader();w.writerows(fold_rows)
  summary={"status":"completed_grouped_qnn_sweep","task":task,"records":len(manifest),"groups":len(set(groups)),
           "label_distribution":{str(c):int(sum(labels==c)) for c in sorted(set(labels))},"results":[],"fold_results":fold_rows,
           "runtime_seconds":time.perf_counter()-start_task,"frozen_test_accessed":False,
           "source_manifest_sha256":hashlib.sha256((SOURCE/spec["manifest"]).read_bytes()).hexdigest()}
  for config in CONFIGS:
   model_pred=[r for r in all_pred if r["model"]==config["name"]];m=metric([r["y_true"] for r in model_pred],[r["y_pred"] for r in model_pred])
   summary["results"].append({"model":config["name"],"configuration":config,**m,"records":len(model_pred),"folds":len(set(folds)),
       "training_seconds":sum(x["training_seconds"] for x in fold_rows if x["model"]==config["name"]),
       "checkpoint_paths":[str((task_dir/config["name"]/f"fold_{int(f)}"/"checkpoint.npz").relative_to(ROOT)) for f in sorted(set(folds))]})
  (task_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
  with (task_dir/"oof_predictions.csv").open("w",newline="",encoding="utf-8") as f:
   w=csv.DictWriter(f,fieldnames=list(all_pred[0]));w.writeheader();w.writerows(all_pred)
  reports.append({"task":task,"status":summary["status"],"results":[{"model":r["model"],"accuracy_percent":round(r["accuracy"]*100,4),"macro_f1_percent":round(r["macro_f1"]*100,4),"balanced_accuracy_percent":round(r["balanced_accuracy"]*100,4)} for r in summary["results"]]})
  COMPLETED_TASKS=list(reports)
  (OUT/"remaining_task_sweep_status.json").write_text(json.dumps({"status":"running","completed_tasks":COMPLETED_TASKS,"current_task":CURRENT_TASK,"frozen_test_accessed":False},indent=2),encoding="utf-8")
  print(json.dumps(reports[-1]),flush=True)
 if len(reports)!=len(TASKS):raise AssertionError(f"not all planned tasks finished: {len(reports)}/{len(TASKS)}")
 (OUT/"remaining_task_sweep_status.json").write_text(json.dumps({"status":"all_remaining_tasks_completed","completed_tasks":reports,"current_task":None,"frozen_test_accessed":False},indent=2),encoding="utf-8")
 print(json.dumps({"status":"all_remaining_tasks_completed","tasks":reports,"frozen_test_accessed":False},indent=2))

def main():
 try:
  run_campaign()
 except MemoryError as exc:
  OUT.mkdir(parents=True,exist_ok=True)
  (OUT/"remaining_task_sweep_status.json").write_text(json.dumps({"status":"partial_stopped_by_ram_guard","completed_tasks":COMPLETED_TASKS,
      "blocked_task":CURRENT_TASK,"reason":str(exc),"partial_checkpoint_note":"Any task without summary.json remains incomplete and must be safely resumed before scoring.",
      "frozen_test_accessed":False},indent=2),encoding="utf-8")
  print(json.dumps({"status":"partial_stopped_by_ram_guard","completed_tasks":len(COMPLETED_TASKS),"blocked_task":CURRENT_TASK,"reason":str(exc)},indent=2))

if __name__=="__main__":main()
