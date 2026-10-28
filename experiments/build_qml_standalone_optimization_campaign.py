"""Build a standalone QML-only optimization record from development OOF runs.

This script intentionally reads only saved development prediction artifacts. It
does not inspect any frozen-test manifest or label file.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028"
OUT = ROOT / "reports/experiments/qml_standalone_accuracy_optimization_20261028"


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pct(value):
    return round(float(value) * 100, 4) if value not in (None, "") else ""


def score(y, pred):
    labels=sorted(set(y)|set(pred)); per={};f1=[];recall=[]
    for c in labels:
        tp=sum(a==c and b==c for a,b in zip(y,pred));fp=sum(a!=c and b==c for a,b in zip(y,pred));fn=sum(a==c and b!=c for a,b in zip(y,pred));sup=sum(a==c for a in y)
        pr=tp/(tp+fp) if tp+fp else 0.;re=tp/sup if sup else 0.;ff=2*pr*re/(pr+re) if pr+re else 0.
        per[str(c)]={"precision":pr,"recall":re,"f1":ff,"support":sup};f1.append(ff);recall.append(re)
    return {"accuracy":sum(a==b for a,b in zip(y,pred))/len(y),"macro_f1":sum(f1)/len(f1),
            "balanced_accuracy":sum(recall)/len(recall),"per_class_metrics":json.dumps(per,sort_keys=True)}


def copy_prediction(task, candidate, source_path, model_filter=None):
    src = ROOT / source_path.replace("\\", "/")
    if not src.exists():
        return ""
    dst = OUT / "qml_standalone_predictions" / task / candidate / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower()==".csv":
        data=read_csv(src)
        filt=model_filter or candidate
        if data and "model" in data[0] and any(r.get("model")==filt for r in data):
            data=[r for r in data if r.get("model")==filt]
            write_csv(dst,data)
        else:
            shutil.copy2(src,dst)
    else:
        shutil.copy2(src, dst)
    return str(dst.relative_to(ROOT))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qml_standalone_predictions").mkdir(exist_ok=True)
    (OUT / "qml_standalone_checkpoints").mkdir(exist_ok=True)
    historical = read_csv(SOURCE / "qml_model_accuracy_benchmark.csv")
    task_winners = {r["dataset_task"]: r for r in read_csv(SOURCE / "best_qml_model_by_task.csv")}
    fold_level = read_csv(SOURCE / "fold_level_results.csv")
    manifest = json.loads((SOURCE / "verified_dataset_inventory.json").read_text(encoding="utf-8"))
    rows = []
    winner_rows = []
    copied_predictions = []
    candidate_dirs = defaultdict(set)

    # Copy every existing task winner's OOF predictions and checkpoint metadata.
    for task, winner in task_winners.items():
        pred_copy = copy_prediction(task, "existing_best", winner.get("oof_prediction_path", ""), winner["model"])
        copied_predictions.append({"task": task, "candidate": winner["model"], "prediction_path": pred_copy,
                                   "evaluation": "development grouped OOF", "frozen_test": False})
        winner_rows.append({"dataset_task": task, "target":winner.get("target",""),
                            "grouping_strategy":winner.get("validation_protocol",""),
                            "selected_model": winner["model"],
                            "original_best_accuracy_percent": winner["validation_accuracy_percent"],
                            "selected_accuracy_percent": winner["validation_accuracy_percent"],
                            "macro_f1_percent": winner["macro_f1_percent"],
                            "balanced_accuracy_percent": winner["balanced_accuracy_percent"],
                            "per_class_metrics": winner.get("per_class_metrics", ""),
                            "training_seconds":winner.get("training_seconds",""),
                            "lowest_observed_available_ram_mib":winner.get("lowest_observed_available_ram_mib",""),
                            "selection_status": "historical OOF incumbent; unchanged pending repeatable optimization",
                            "checkpoint_path": winner.get("checkpoint_path", ""),
                            "prediction_path": pred_copy})
        cp = winner.get("checkpoint_path", "")
        parent = (ROOT / cp.replace("\\", "/")).parent if cp else None
        target = OUT / "qml_standalone_checkpoints" / task / "existing_best"
        if parent and parent.exists() and parent.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            # fold_checkpoints.json contains learned circuit/readout weights for
            # the prior PennyLane runs; preprocessing objects are fold-specific.
            for p in [*parent.glob("fold_checkpoints.json"), *parent.glob("preprocessing_fold*.joblib")]:
                shutil.copy2(p, target / p.name)
            for p in [*parent.glob("checkpoint.npz")]:
                shutil.copy2(p, target / p.name)

    # Historical candidate summaries from the already-completed grouped runs.
    for r in historical:
        task, model = r["dataset_task"], r["model"]
        pred = r.get("oof_prediction_path", "")
        run_dir = str(Path(pred.replace("\\", "/")).parent) if pred else ""
        rows.append({"dataset_task": task, "candidate": model, "seed": "historical saved run",
                     "fold": "OOF aggregate", "accuracy_percent": r["validation_accuracy_percent"],
                     "macro_f1_percent": r["macro_f1_percent"], "balanced_accuracy_percent": r["balanced_accuracy_percent"],
                     "records": r.get("records", ""), "groups": r.get("groups", ""), "folds": r.get("folds", ""),
                     "training_seconds": r.get("training_seconds", ""), "configuration": r.get("configuration", ""),
                     "completion_status": "completed_historical_grouped_oof", "checkpoint_path": r.get("checkpoint_path", ""),
                     "prediction_path": pred, "per_class_metrics": r.get("per_class_metrics", ""), "validation_only": True})
        if run_dir:
            candidate_dirs[(task, model)].add(run_dir)

    # Add historical fold-level QNN rows when the recorded run/model match.
    for f in fold_level:
        model = f.get("model", "")
        for (task, candidate), dirs in candidate_dirs.items():
            if candidate != model or f.get("run", "").replace("\\", "/") not in dirs:
                continue
            rows.append({"dataset_task": task, "candidate": candidate, "seed": "historical saved run",
                         "fold": f.get("fold", ""), "accuracy_percent": pct(f.get("accuracy")),
                         "macro_f1_percent": pct(f.get("macro_f1")),
                         "balanced_accuracy_percent": pct(f.get("balanced_accuracy")),
                         "records": f.get("validation_records", ""), "groups": f.get("validation_origins", ""),
                         "training_seconds": f.get("training_seconds", f.get("runtime_seconds", "")),
                         "configuration": json.dumps({k: f.get(k) for k in ("encoding", "qubits", "depth", "features", "data_reuploading", "optimizer_batch_rows") if f.get(k)}, sort_keys=True),
                         "completion_status": "completed_historical_fold", "validation_only": True})

    # New four-epoch local grouped sweeps and three-seed finalist replications.
    new_runs = []
    run_specs = []
    for task in ("bugs4q_robust_execution", "qsimbench_backend_fingerprint"):
        run_specs.append((task, "standalone_optimization_sweep4", "28102026"))
        run_specs.append((task, "standalone_replicate_seed4321", "4321"))
        run_specs.append((task, "standalone_replicate_seed2027", "2027"))
    for task, dirname, seed in run_specs:
        folder = SOURCE / task / dirname
        summary_path = folder / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        pred_path = folder / "oof_predictions.csv"
        pred_rel = str(pred_path.relative_to(ROOT)) if pred_path.exists() else ""
        for name, result in summary["results"].items():
            prediction_copy=copy_prediction(task,f"{dirname}_{name}",pred_rel,name)
            copied_predictions.append({"task":task,"candidate":f"{dirname}_{name}","prediction_path":prediction_copy,
                                       "evaluation":"development grouped OOF","frozen_test":False})
            cp = ";".join(result.get("checkpoint_paths", []))
            agg = {"dataset_task": task, "candidate": name, "seed": seed, "fold": "OOF aggregate",
                   "accuracy_percent": pct(result["accuracy"]), "macro_f1_percent": pct(result["macro_f1"]),
                   "balanced_accuracy_percent": pct(result["balanced_accuracy"]), "records": result.get("records", ""),
                   "groups": summary.get("groups", ""), "folds": result.get("folds", ""),
                   "training_seconds": result.get("training_seconds", ""),
                   "configuration": json.dumps(result.get("architecture", {}), sort_keys=True),
                   "completion_status": "completed_grouped_oof", "checkpoint_path": cp,
                   "prediction_path": prediction_copy, "validation_only": True}
            rows.append(agg)
            new_runs.append({**agg, "run_directory": str(folder.relative_to(ROOT))})
            for fold in summary.get("fold_results", []):
                if fold["model"] != name:
                    continue
                fm = fold.get("metrics", fold)
                rows.append({"dataset_task": task, "candidate": name, "seed": seed, "fold": fold["fold"],
                             "accuracy_percent": pct(fm["accuracy"]), "macro_f1_percent": pct(fm["macro_f1"]),
                             "balanced_accuracy_percent": pct(fm["balanced_accuracy"]), "records": fold.get("records", ""),
                             "groups": fold.get("groups", ""), "training_seconds": fold.get("training_seconds", ""),
                             "configuration": json.dumps(result.get("architecture", {}), sort_keys=True),
                             "completion_status": "completed_grouped_fold", "checkpoint_path": cp,
                             "prediction_path": prediction_copy, "gradient_check": fold.get("gradient_check", {}).get("passed", False),
                             "available_ram_mib": fold.get("available_ram_mib", ""), "validation_only": True})
            # Copy each NPZ checkpoint into the dedicated campaign directory.
            ckpt_dir = OUT / "qml_standalone_checkpoints" / task / f"{name}_seed{seed}"
            for cp_path in result.get("checkpoint_paths", []):
                src = ROOT / cp_path.replace("\\", "/")
                if src.exists():
                    dst = ckpt_dir / src.parent.name / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

    # QMutBench finalist reruns use the saved three-fold origin split and only
    # its previously selected amplitude/QASM feature family.
    qmut_specs=[("qmutbench_threeclass_qnn_sweep24","28102026"),
                ("qmutbench_threeclass_standalone_seed4321","4321")]
    qmut_runs=[]
    ensemble_summaries=[]
    for dirname,seed in qmut_specs:
        folder=SOURCE/dirname
        summary_path=folder/"summary.json"
        if not summary_path.exists(): continue
        summary=json.loads(summary_path.read_text(encoding="utf-8"))
        pred_path=folder/"oof_predictions.csv"
        pred_rel=str(pred_path.relative_to(ROOT))
        for result in summary.get("results",[]):
            if result["model"]!="amplitude_hybrid_2q_d2_lr008": continue
            model=result["model"]
            prediction_copy=copy_prediction("QMutBench three-class research",dirname,pred_rel,model)
            copied_predictions.append({"task":"QMutBench three-class research","candidate":dirname,
                                       "prediction_path":prediction_copy,"evaluation":"development origin-group OOF","frozen_test":False})
            checkpoints=[str((folder/model/f"fold_{f}"/"checkpoint.npz").relative_to(ROOT)) for f in range(1,4)]
            cp=";".join(checkpoints)
            agg={"dataset_task":"QMutBench three-class research","candidate":model,"seed":seed,"fold":"OOF aggregate",
                 "accuracy_percent":pct(result["accuracy"]),"macro_f1_percent":pct(result["macro_f1"]),
                 "balanced_accuracy_percent":pct(result["balanced_accuracy"]),"records":result.get("records",205),
                 "groups":summary.get("origins",12),"folds":result.get("folds",3),
                 "training_seconds":result.get("training_seconds",""),"configuration":json.dumps(result.get("configuration",{}),sort_keys=True),
                 "completion_status":"completed_grouped_oof","checkpoint_path":cp,"prediction_path":prediction_copy,"validation_only":True}
            rows.append(agg);qmut_runs.append({**agg,"run_directory":str(folder.relative_to(ROOT))})
            for fold in summary.get("fold_results",[]):
                if fold.get("model")!=model:continue
                rows.append({"dataset_task":"QMutBench three-class research","candidate":model,"seed":seed,"fold":fold["fold"],
                             "accuracy_percent":pct(fold["accuracy"]),"macro_f1_percent":pct(fold["macro_f1"]),
                             "balanced_accuracy_percent":pct(fold["balanced_accuracy"]),"records":fold.get("records",0),
                             "groups":fold.get("origins",0),"training_seconds":fold.get("training_seconds",""),
                             "configuration":json.dumps(result.get("configuration",{}),sort_keys=True),"completion_status":"completed_grouped_fold",
                             "checkpoint_path":cp,"prediction_path":prediction_copy,
                             "gradient_check":fold.get("gradient_check",{}).get("passed",False),"validation_only":True})
            new_runs.append({**agg,"run_directory":str(folder.relative_to(ROOT))})
            ckpt_dir=OUT/"qml_standalone_checkpoints"/"QMutBench three-class research"/f"{model}_seed{seed}"
            for cp_path in checkpoints:
                src=ROOT/cp_path.replace("\\","/")
                if src.exists():
                    dst=ckpt_dir/src.parent.name/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)

    if qmut_runs:
        aligned={}
        for run in qmut_runs:
            pred=list(csv.DictReader((ROOT/run["run_directory"]/"oof_predictions.csv").open(encoding="utf-8-sig",newline="")))
            for row in pred:
                if row["model"]!="amplitude_hybrid_2q_d2_lr008":continue
                rid=row["record_id"]
                entry=aligned.setdefault(rid,{"record_id":rid,"origin_program":row["origin_program"],"fold":row["fold"],"y_true":row["y_true"],"probs":[]})
                if entry["y_true"]!=row["y_true"] or entry["fold"]!=row["fold"] or entry["origin_program"]!=row["origin_program"]:
                    raise AssertionError(f"QMutBench seed OOF mismatch for {rid}")
                entry["probs"].append([float(row[f"prob_{c}"]) for c in range(3)])
        ens=[]
        for rid in sorted(aligned,key=lambda v:(0,int(v)) if str(v).isdigit() else (1,str(v))):
            entry=aligned[rid]
            if len(entry["probs"])!=len(qmut_runs):raise AssertionError("incomplete QMutBench seed coverage")
            probs=[sum(x[c] for x in entry["probs"])/len(qmut_runs) for c in range(3)]
            ens.append({"record_id":rid,"origin_program":entry["origin_program"],"fold":entry["fold"],"y_true":entry["y_true"],
                        "y_pred":max(range(3),key=lambda c:probs[c]),"member_seed_count":len(qmut_runs),**{f"prob_{c}":probs[c] for c in range(3)}})
        em=score([int(r["y_true"]) for r in ens],[int(r["y_pred"]) for r in ens])
        ensemble_path=OUT/"qml_standalone_predictions"/"QMutBench three-class research"/"multi_seed_probability_ensemble_oof.csv"
        write_csv(ensemble_path,ens);ensemble_rel=str(ensemble_path.relative_to(ROOT))
        ensemble_summaries.append({"dataset_task":"QMutBench three-class research","candidate":"multi_seed_probability_ensemble_amplitude_lr008",
                                   "seed_count":len(qmut_runs),"accuracy_percent":pct(em["accuracy"]),"macro_f1_percent":pct(em["macro_f1"]),
                                   "balanced_accuracy_percent":pct(em["balanced_accuracy"]),"prediction_path":ensemble_rel,
                                   "per_class_metrics":em["per_class_metrics"]})
        rows.append({"dataset_task":"QMutBench three-class research","candidate":"multi_seed_probability_ensemble_amplitude_lr008",
                     "seed":";".join(x["seed"] for x in qmut_runs),"fold":"OOF aggregate","accuracy_percent":pct(em["accuracy"]),
                     "macro_f1_percent":pct(em["macro_f1"]),"balanced_accuracy_percent":pct(em["balanced_accuracy"]),"records":len(ens),"folds":3,
                     "configuration":json.dumps({"members":len(qmut_runs),"architecture":"2q amplitude hybrid depth2","learning_rate":.08,
                                                  "aggregation":"equal probability mean"},sort_keys=True),
                     "completion_status":"completed_grouped_oof_ensemble","prediction_path":ensemble_rel,"validation_only":True})
        copied_predictions.append({"task":"QMutBench three-class research","candidate":"multi_seed_probability_ensemble_amplitude_lr008",
                                   "prediction_path":ensemble_rel,"evaluation":"development origin-group OOF","frozen_test":False})
        incumbent=task_winners["QMutBench three-class research"]
        if float(pct(em["accuracy"])) > float(incumbent["validation_accuracy_percent"]):
            for winner_row in winner_rows:
                if winner_row["dataset_task"]=="QMutBench three-class research":
                    winner_row.update({"selected_model":"multi_seed_probability_ensemble_amplitude_lr008",
                                       "selected_accuracy_percent":pct(em["accuracy"]),
                                       "macro_f1_percent":pct(em["macro_f1"]),
                                       "balanced_accuracy_percent":pct(em["balanced_accuracy"]),
                                       "per_class_metrics":em["per_class_metrics"],
                                       "training_seconds":sum(float(x.get("training_seconds",0) or 0) for x in qmut_runs),
                                       "selection_status":"selected by higher grouped development OOF accuracy; two-seed ensemble",
                                       "checkpoint_path":";".join(x["checkpoint_path"] for x in qmut_runs),
                                       "prediction_path":ensemble_rel})

    # Three-seed means for the LR=.07 finalist. Do not select on an individual seed.
    finalist = defaultdict(list)
    for r in new_runs:
        if r["candidate"] == "angle_reupload_4q_depth2_lr007":
            finalist[r["dataset_task"]].append(r)
    seed_summaries = []
    for task, group in finalist.items():
        vals = [float(x["accuracy_percent"]) for x in group]
        mean = sum(vals) / len(vals)
        variance = sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)
        seed_summaries.append({"dataset_task": task, "candidate": group[0]["candidate"], "seed_count": len(vals),
                               "seed_accuracy_percent": ";".join(f"{v:.4f}" for v in vals),
                               "mean_accuracy_percent": round(mean, 4), "sd_accuracy_percentage_points": round(variance ** .5, 4),
                               "baseline_accuracy_percent": task_winners[task]["validation_accuracy_percent"],
                               "selected_as_best": False if task == "qsimbench_backend_fingerprint" else "review_required_for_stability",
                               "selection_note": "seed variability checked; individual-seed best is not used for selection"})

        # Equal-probability averaging across seeds; all seeds share the same
        # fixed development folds and each member predicts its own held-out fold.
        aligned={}
        for run in group:
            run_dir=ROOT/run["run_directory"]
            pred=list(csv.DictReader((run_dir/"oof_predictions.csv").open(encoding="utf-8-sig",newline="")))
            for row in pred:
                if row["model"] != run["candidate"]: continue
                rid=row["record_id"]
                entry=aligned.setdefault(rid,{"record_id":rid,"group_id":row.get("group_id",""),"fold":row["fold"],"y_true":row["y_true"],"probabilities":[]})
                if entry["y_true"]!=row["y_true"] or entry["fold"]!=row["fold"] or entry["group_id"]!=row.get("group_id",""):
                    raise AssertionError(f"seed OOF mismatch for {task} record {rid}")
                entry["probabilities"].append(float(row["probability"]))
        ens=[]
        for rid in sorted(aligned,key=lambda v:(0,int(v)) if str(v).isdigit() else (1,str(v))):
            e=aligned[rid]
            if len(e["probabilities"]) != len(group): raise AssertionError("incomplete seed OOF coverage")
            probability=sum(e["probabilities"])/len(group)
            ens.append({"record_id":rid,"group_id":e["group_id"],"fold":e["fold"],"y_true":e["y_true"],
                        "probability":probability,"y_pred":int(probability>=.5),"member_seed_count":len(group)})
        em=score([int(r["y_true"]) for r in ens],[r["y_pred"] for r in ens])
        ensemble_path=OUT/"qml_standalone_predictions"/task/"three_seed_probability_ensemble_oof.csv"
        write_csv(ensemble_path,ens)
        ensemble_rel=str(ensemble_path.relative_to(ROOT))
        ensemble_summaries.append({"dataset_task":task,"candidate":"three_seed_probability_ensemble_lr007",
                                   "seed_count":len(group),"accuracy_percent":pct(em["accuracy"]),
                                   "macro_f1_percent":pct(em["macro_f1"]),"balanced_accuracy_percent":pct(em["balanced_accuracy"]),
                                   "prediction_path":ensemble_rel,"per_class_metrics":em["per_class_metrics"]})
        rows.append({"dataset_task":task,"candidate":"three_seed_probability_ensemble_lr007","seed":"4321;2027;28102026",
                     "fold":"OOF aggregate","accuracy_percent":pct(em["accuracy"]),"macro_f1_percent":pct(em["macro_f1"]),
                     "balanced_accuracy_percent":pct(em["balanced_accuracy"]),"records":len(ens),"folds":5,
                     "configuration":json.dumps({"members":len(group),"architecture":"4q angle reupload depth2","learning_rate":0.07,"aggregation":"equal probability mean"},sort_keys=True),
                     "completion_status":"completed_grouped_oof_ensemble","prediction_path":ensemble_rel,"validation_only":True})
        copied_predictions.append({"task":task,"candidate":"three_seed_probability_ensemble_lr007","prediction_path":ensemble_rel,
                                   "evaluation":"development grouped OOF","frozen_test":False})

    # Manifest validation: task counts/label distributions come from the saved
    # development OOF prediction artifacts only.
    task_info = []
    for task, winner in task_winners.items():
        p = ROOT / winner["oof_prediction_path"].replace("\\", "/")
        labels = Counter()
        groups = set()
        with p.open(encoding="utf-8-sig", newline="") as f:
            rr = list(csv.DictReader(f))
        if rr and "model" in rr[0]:
            rr=[row for row in rr if row.get("model")==winner["model"]]
        for row in rr:
            labels[row.get("y_true", row.get("label", "unknown"))] += 1
            groups.add(row.get("group_id", row.get("origin_program", row.get("bug_id", ""))))
        task_info.append({"dataset_task": task, "target":winner.get("target",""), "records_in_oof": len(rr), "group_count_observed": len(groups),
                          "label_distribution": dict(labels), "group_unit": winner.get("group_unit", ""),
                          "validation_protocol": winner.get("validation_protocol", ""),
                          "frozen_test_accessed": False})

    write_csv(OUT / "qml_standalone_candidate_results.csv", rows)
    write_csv(OUT / "task_winners.csv", winner_rows)
    write_csv(OUT / "multiseed_finalist_summary.csv", seed_summaries)
    write_csv(OUT / "multiseed_ensemble_summary.csv", ensemble_summaries)
    write_csv(OUT / "prediction_artifact_index.csv", copied_predictions)
    run_manifest = {"status": "local_development_optimization_completed_with_scope_blockers",
                    "created_by": "experiments/build_qml_standalone_optimization_campaign.py",
                    "source_campaign": str(SOURCE.relative_to(ROOT)),
                    "campaign_output": str(OUT.relative_to(ROOT)),
                    "tasks": task_info,
                    "new_local_runs": new_runs,
                    "multiseed_finalists": seed_summaries,
                    "multiseed_ensembles": ensemble_summaries,
                    "blocked_run": {"task":"QMutBench three-class research","seed":2027,"status":"blocked_preflight",
                                    "reason":"700 MiB RAM preflight failed; no training started and no checkpoint was written"},
                    "software": {"python":sys.version.split()[0],"platform":platform.platform(),"numpy":__import__("numpy").__version__},
                    "compute": {"backend": "NumPy statevector CPU", "cpu_logical_count":os.cpu_count(),"gpu": "not detected/available",
                                "available_ram_mib_at_campaign_start": 781.2,
                                "safety_floor_mib": 500, "external_compute_used": False},
                    "frozen_test_accessed": False,
                    "limitations": ["Standalone tasks without compatible local raw feature manifests were not retrained in this optimization pass; their existing validated grouped QNN OOF runs are preserved as incumbents.",
                                    "The QNN environment has NumPy and psutil but no PennyLane, Torch, or Qiskit; new runs use the verified NumPy statevector runner.",
                                    "New binary sweep uses deterministic 8-bin hashed token features and one four-epoch seed per candidate; finalists received two additional seeds.",
                                    "Historical fold checkpoints are JSON weight manifests plus fold-specific preprocessing objects; new statevector checkpoints are NPZ archives."]}
    (OUT / "qml_standalone_run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    by_task_info={x["dataset_task"]:x for x in task_info}
    by_task_winner={x["dataset_task"]:x for x in winner_rows}
    lines=["# Standalone QML Accuracy Optimization Campaign", "",
           "**Evaluation scope:** grouped development out-of-fold (OOF) predictions only. The frozen test was not read or used for selection; final held-out test accuracy is unavailable and intentionally blank.", "",
           "The campaign preserves prior grouped QNN runs as incumbents and adds local, CPU-only statevector optimization for Bugs4Q-Robust and QSimBench, plus a QMutBench finalist seed rerun. Scores are task-specific and should not be compared across different targets.", "",
           "## Per-task results", "",
           "| Task | Prediction target | Records / groups | Grouped split | Label distribution | Original best QML accuracy | Selected development accuracy | Macro F1 | Balanced accuracy | Training seconds | Change | Selected model | Final test accuracy |",
           "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for task in sorted(by_task_info):
        info=by_task_info[task]; win=by_task_winner[task]
        old=float(win["original_best_accuracy_percent"]);new=float(win["selected_accuracy_percent"])
        lines.append(f"| {task} | {info['target']} | {info['records_in_oof']} / {info['group_count_observed']} | {info['validation_protocol']} ({info['group_unit']}) | `{json.dumps(info['label_distribution'],sort_keys=True)}` | {old:.2f}% | {new:.2f}% | {win['macro_f1_percent']}% | {win['balanced_accuracy_percent']}% | {win.get('training_seconds','')} | {new-old:+.2f} pp | {win['selected_model']} | unavailable |")
    lines += ["", "## Per-class selected-model results", "",
              "| Task | Per-class precision / recall / F1 / support |",
              "|---|---|"]
    for task in sorted(by_task_winner):
        metric=by_task_winner[task].get("per_class_metrics","")
        lines.append(f"| {task} | `{metric}` |")
    lines += ["", "## Executed optimization", "",
              "- **Bugs4Q-Robust:** six 4-epoch QNN configurations were trained on five issue-grouped folds; the angle re-upload, 4-qubit, depth-2, learning-rate 0.07 candidate was repeated with two additional seeds. Candidate OOF accuracies were 55.41%, 48.65%, and 54.05% (mean 52.70%, sample SD 3.58 percentage points). The fixed three-seed probability ensemble reached 50.00%. The previous 51.35% QNN remains selected because the apparent gain is seed-unstable and the ensemble is lower.",
              "- **QSimBench:** six 4-epoch configurations were trained on five algorithm-size grouped folds. The same finalist across three seeds reached 65.00%, 62.67%, and 63.33% (mean 63.67%, sample SD 1.20 pp); the probability ensemble reached 64.33%. The prior 66.00% candidate remains selected.",
              "- **QMutBench three-class:** the prior bounded six-configuration, 24-epoch QNN sweep was followed by a 24-epoch rerun of its strongest single model (2-qubit amplitude hybrid, depth 2, learning rate 0.08), on the saved three-fold origin split. The extra seed scored 46.34%; the two-seed probability ensemble reached 47.32% accuracy, 41.89% macro F1, and 46.06% balanced accuracy, up 0.49 pp from the previous 46.83% QNN ensemble. The additional seed 2027 was attempted twice but failed the runner's 700 MiB preflight; no training began for that seed. The two-seed ensemble is selected provisionally on development OOF accuracy.",
              "- **Other nine standalone tasks:** their existing grouped QNN configuration comparisons and OOF predictions are retained as the incumbents. They were not retrained in this local pass because the remaining environment has NumPy/psutil but no PennyLane, Torch, or Qiskit, and the resource-light runner only supports the two binary manifests. No external execution was used.",
              "", "## Protocol and resources", "",
              "The new binary runs used identical deterministic five-fold group-disjoint partitions, training-fold min-max scaling, supervised cross-entropy, Adam, and validated parameter-shift gradients. The QMutBench runs reused the saved origin-group folds and fold-specific selected features with training-only scaling. OOF probabilities were recomputed for seed ensembles by aligned sample ID, label, group, and fold. No frozen-test data, classical model tuning, cross-source training, or cloud compute was used.",
              "", "Training used the local NumPy CPU statevector implementation. No GPU was detected. Available system RAM fluctuated around the safety floor; the QMutBench third-seed attempts stopped at the explicit 700 MiB preflight. The prior broader binary sweep that fell below 500 MiB remains partial and unscored.",
              "", "## Artifacts", "",
              "- Candidate and fold-level rows: `qml_standalone_candidate_results.csv`.",
              "- Selected per-task development model: `task_winners.csv`.",
              "- Seed variability and probability-ensemble metrics: `multiseed_finalist_summary.csv`, `multiseed_ensemble_summary.csv`.",
              "- Development OOF predictions: `qml_standalone_predictions/`.",
              "- Fold checkpoints and preprocessing metadata: `qml_standalone_checkpoints/`.",
              "- Configuration, task labels/groups, and blocked-run status: `qml_standalone_run_manifest.json`.",
              "- Independent checks: `qml_standalone_audit.md`.",
              "", "**Status:** local optimization improved the QMutBench development ensemble by 0.49 percentage points. The new binary searches did not produce a more reliable system than the prior per-task incumbents. Final test results remain unavailable; the campaign does not establish unseen-test accuracy."]
    (OUT/"qml_standalone_accuracy_benchmark.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"out": str(OUT.relative_to(ROOT)), "candidate_rows": len(rows),
                      "tasks": len(task_info), "new_completed_runs": len(new_runs),
                      "prediction_artifacts": len(copied_predictions), "frozen_test_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
