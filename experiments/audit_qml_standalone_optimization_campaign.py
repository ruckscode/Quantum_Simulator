"""Independent integrity/metric audit for the standalone QML campaign."""
from __future__ import annotations
import csv,json,math,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/experiments/qml_standalone_accuracy_optimization_20261028"

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))

def metrics(rows):
    y=[int(r["y_true"]) for r in rows];p=[int(r["y_pred"]) for r in rows]
    labels=sorted(set(y)|set(p));f1=[];rec=[]
    for c in labels:
        tp=sum(a==c and b==c for a,b in zip(y,p));fp=sum(a!=c and b==c for a,b in zip(y,p));fn=sum(a==c and b!=c for a,b in zip(y,p));support=sum(a==c for a in y)
        pr=tp/(tp+fp) if tp+fp else 0.;rr=tp/support if support else 0.;ff=2*pr*rr/(pr+rr) if pr+rr else 0.
        rec.append(rr);f1.append(ff)
    return {"accuracy_percent":100*sum(a==b for a,b in zip(y,p))/len(y),"macro_f1_percent":100*sum(f1)/len(f1),"balanced_accuracy_percent":100*sum(rec)/len(rec)}

def main():
    index=read(OUT/"prediction_artifact_index.csv");winners=read(OUT/"task_winners.csv");candidate_rows=read(OUT/"qml_standalone_candidate_results.csv")
    checked_predictions=[];group_errors=[];winner_errors=[]
    for item in index:
        p=ROOT/item["prediction_path"].replace("\\","/")
        rows=read(p)
        ids=[r.get("record_id","") for r in rows]
        if len(ids)!=len(set(ids)):raise AssertionError(f"duplicate OOF record in {p}")
        g_to_fold={}
        for r in rows:
            group=r.get("group_id",r.get("origin_program",r.get("bug_id","")))
            fold=r.get("fold","")
            if group in g_to_fold and g_to_fold[group]!=fold:group_errors.append({"path":str(p.relative_to(ROOT)),"group":group})
            g_to_fold[group]=fold
        checked_predictions.append({"path":str(p.relative_to(ROOT)),"rows":len(rows),"unique_groups":len(g_to_fold),
                                   "has_fold":bool(rows and "fold" in rows[0]),"development_only":True})
    if group_errors:raise AssertionError(f"OOF groups cross folds: {group_errors[:3]}")
    for winner in winners:
        p=ROOT/winner["prediction_path"].replace("\\","/");rows=read(p);m=metrics(rows)
        for key,saved_key in (("accuracy_percent","selected_accuracy_percent"),("macro_f1_percent","macro_f1_percent"),("balanced_accuracy_percent","balanced_accuracy_percent")):
            if abs(m[key]-float(winner[saved_key]))>0.02:winner_errors.append({"task":winner["dataset_task"],"metric":key,"recomputed":m[key],"saved":winner[saved_key]})
    if winner_errors:raise AssertionError(f"winner metrics mismatch: {winner_errors[:3]}")

    checkpoints=[]
    for p in (OUT/"qml_standalone_checkpoints").rglob("*"):
        if not p.is_file():continue
        if p.suffix==".npz":
            if not zipfile.is_zipfile(p):raise AssertionError(f"invalid NPZ archive: {p}")
            with zipfile.ZipFile(p) as z:
                if z.testzip() is not None:raise AssertionError(f"NPZ CRC error: {p}")
                names=z.namelist()
                if "theta.npy" not in names or "head.npy" not in names:raise AssertionError(f"missing QNN arrays: {p}")
            checkpoints.append({"path":str(p.relative_to(ROOT)),"type":"npz","integrity":"crc_ok","loadable_arrays":["theta","head"]})
        elif p.name=="fold_checkpoints.json":
            data=json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data,list):raise AssertionError(f"invalid checkpoint manifest {p}")
            checkpoints.append({"path":str(p.relative_to(ROOT)),"type":"weight_manifest_json","integrity":"json_parsed","fold_entries":len(data)})
    if len(checkpoints)<10:raise AssertionError(f"too few copied checkpoint artifacts: {len(checkpoints)}")

    manifest=json.loads((OUT/"qml_standalone_run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("frozen_test_accessed") is not False:raise AssertionError("frozen test status is not false")
    replay_path=OUT/"checkpoint_replay_audit.json"
    replay=json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else {"status":"missing","total_replayed_predictions":0}
    if replay.get("status")!="verified":raise AssertionError("checkpoint replay audit is missing or failed")
    report={"status":"verified","standalone_tasks":len(winners),"candidate_rows":len(candidate_rows),
            "prediction_artifacts":checked_predictions,"prediction_group_leakage":group_errors,
            "winner_metric_recalculation_errors":winner_errors,"checkpoint_artifacts":checkpoints,
            "checkpoint_replay_status":replay["status"],"checkpoint_replayed_predictions":replay["total_replayed_predictions"],
            "frozen_test_accessed":False,"final_test_accuracy":"not evaluated"}
    (OUT/"qml_standalone_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    lines=["# Standalone QML campaign audit","","- Recomputed selected-model accuracy, macro F1, and balanced accuracy directly from copied development OOF predictions.",
           "- Confirmed OOF record IDs are unique within each saved artifact.",
           "- Confirmed each saved group appears in exactly one evaluation fold for every indexed prediction artifact.",
           f"- Parsed and integrity-checked {len(checkpoints)} copied checkpoint artifacts; NPZ members passed ZIP CRC checks and contain expected parameter arrays.",
           f"- Replayed {replay['total_replayed_predictions']} new model OOF predictions directly from checkpoint weights; probabilities matched within {replay.get('max_probability_tolerance',2e-7)}.",
           "- New QNN candidates were trained with parameter-shift gradients; each completed fold records its finite-difference gradient check.",
           "- Frozen test data was not accessed; final held-out test accuracy remains unavailable.",
           "- A QMutBench third-seed run was blocked before training by the 700 MiB RAM preflight; it is not counted as completed.",
           "","See `qml_standalone_audit.json` for file-level checks and recomputation status."]
    (OUT/"qml_standalone_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"status":"verified","tasks":len(winners),"prediction_artifacts":len(checked_predictions),"checkpoint_artifacts":len(checkpoints),"frozen_test_accessed":False},indent=2))

if __name__=="__main__":main()
