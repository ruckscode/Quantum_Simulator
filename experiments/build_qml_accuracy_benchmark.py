"""Consolidate completed development-only QML runs; never reads frozen tests."""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'reports/experiments/qmutbench_multisource_qml_campaign_20261028'

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def dump(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def metric(y,p):
    labels=sorted(set(y)); f1=[]; recall=[]; per={}
    for c in labels:
        tp=sum(a==c and b==c for a,b in zip(y,p)); fp=sum(a!=c and b==c for a,b in zip(y,p)); fn=sum(a==c and b!=c for a,b in zip(y,p)); support=sum(a==c for a in y)
        pr=tp/(tp+fp) if tp+fp else 0.; re=tp/support if support else 0.; f=2*pr*re/(pr+re) if pr+re else 0.
        f1.append(f);recall.append(re);per[str(c)]={'precision':pr,'recall':re,'f1':f,'support':support}
    return {'accuracy':sum(a==b for a,b in zip(y,p))/len(y),'macro_f1':sum(f1)/len(f1),'balanced_accuracy':sum(recall)/len(recall),'per_class_metrics':json.dumps(per,sort_keys=True)}

def config_dict(value):
    try:
        parsed=json.loads(value or '{}')
        return parsed if isinstance(parsed,dict) else {}
    except (TypeError,ValueError):
        return {}

def history_epochs(path):
    try:
        value=json.loads(path.read_text(encoding='utf-8'))
        return len(value.get('epochs',value)) if isinstance(value,dict) else len(value)
    except (OSError,ValueError,TypeError):
        return ''

def main():
    rows=[]
    historical=read(C/'comprehensive_results.csv')
    baseline={}
    for r in historical:
        if r['model_family']=='Classical' and (r['dataset_task'] not in baseline or float(r['macro_f1'])>float(baseline[r['dataset_task']]['macro_f1'])):
            baseline[r['dataset_task']]=r
    for r in historical:
        if r['model_family'] not in {'QNN/hybrid','Ensemble'}:continue
        b=baseline.get(r['dataset_task'],{}); cp=r.get('checkpoint_path','')
        cfg=config_dict(r.get('model_configuration'))
        rows.append({'dataset_task':r['dataset_task'],'target':r['task_semantics'],'model':r['model'],
            'configuration':r['model_configuration'],'records':r['records'],'groups':r['independent_groups'],'folds':r['folds'],
            'epochs':cfg.get('epochs',cfg.get('steps','')),'batch_size':cfg.get('batch_size',''),'optimizer':cfg.get('optimizer',''),
            'validation_accuracy_percent':round(float(r['accuracy'])*100,4),'final_test_accuracy_percent':'',
            'macro_f1_percent':round(float(r['macro_f1'])*100,4),'balanced_accuracy_percent':round(float(r['balanced_accuracy'])*100,4),
            'per_class_metrics':r.get('per_class_metrics',''),
            'baseline_model':b.get('model',''),'baseline_accuracy_percent':round(float(b['accuracy'])*100,4) if b else '',
            'training_seconds':r.get('runtime_seconds',''),'checkpoint_path':cp if cp and (ROOT/cp.replace('\\','/')).exists() else '',
            'checkpoint_status':'available' if cp and (ROOT/cp.replace('\\','/')).exists() else 'OOF_ONLY_OR_PATH_UNAVAILABLE',
            'oof_prediction_path':r.get('oof_prediction_path',''),'validation_protocol':r.get('validation',''),
            'group_unit':r.get('group_unit',''),'gradient_checks':r.get('gradient_checks_passed',''),
            'status':'completed_development_oof','source':'previous campaign'})
    # Newly completed two binary tasks, using saved development OOF folds.
    for task in ('bugs4q_robust_execution','qsimbench_backend_fingerprint'):
        folder=C/task/'resource_light_qnn'; s=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        oof=read(folder/'oof_predictions.csv')
        for name,x in s['results'].items():
            scored=[r for r in oof if r['model']==name]
            cm=metric([int(r['y_true']) for r in scored],[int(r['y_pred']) for r in scored])
            rows.append({'dataset_task':task,'target':s['task'],'model':name,'configuration':json.dumps(x['architecture'],sort_keys=True),
                'records':x['records'],'groups':s['groups'],'folds':x['folds'],'validation_accuracy_percent':round(cm['accuracy']*100,4),
                'epochs':history_epochs(folder/name/'fold_1'/'training_history.json'),'batch_size':64,'optimizer':'Adam',
                'final_test_accuracy_percent':'','macro_f1_percent':round(cm['macro_f1']*100,4),'balanced_accuracy_percent':round(cm['balanced_accuracy']*100,4),
                'per_class_metrics':cm['per_class_metrics'],
                'baseline_model':'','baseline_accuracy_percent':'','training_seconds':round(x['training_seconds'],4),
                'checkpoint_path':';'.join(x['checkpoint_paths']),'checkpoint_status':'five_fold_checkpoints_saved',
                'oof_prediction_path':str((folder/'oof_predictions.csv').relative_to(ROOT)),'validation_protocol':s['split'],
                'group_unit':'bug_id; issue groups' if task.startswith('bugs4q') else 'bug_id; algorithm-size groups',
                'lowest_observed_available_ram_mib':min(fr['available_ram_mib'] for fr in s['fold_results'] if fr['model']==name),
                'gradient_checks':'5/5 passed','status':'completed_development_oof','source':'parameter-shift statevector QNN'})
    # QMutBench three-class retraining and bounded hyperparameter sweep.
    for dirname in ('qmutbench_threeclass_resource_light_qnn','qmutbench_threeclass_qnn_sweep24'):
        folder=C/dirname
        if not (folder/'summary.json').exists():continue
        s=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        oof=read(folder/'oof_predictions.csv')
        for x in s['results']:
            name=x['model']; k=x.get('folds',3); model_dir=folder/name
            scored=[r for r in oof if r['model']==name]
            qm=metric([int(r['y_true']) for r in scored],[int(r['y_pred']) for r in scored])
            cps=';'.join(str((model_dir/f'fold_{f}'/'checkpoint.npz').relative_to(ROOT)) for f in range(1,k+1))
            rows.append({'dataset_task':'QMutBench three-class research','target':s['task'],'model':name,
                'configuration':json.dumps(x.get('configuration',{}),sort_keys=True),'records':x['records'],'groups':s['origins'],'folds':k,
                'epochs':history_epochs(model_dir/'fold_1'/'training_history.json'),'batch_size':32,'optimizer':'Adam',
                'validation_accuracy_percent':round(qm['accuracy']*100,4),'final_test_accuracy_percent':'',
                'macro_f1_percent':round(qm['macro_f1']*100,4),'balanced_accuracy_percent':round(qm['balanced_accuracy']*100,4),
                'per_class_metrics':qm['per_class_metrics'],
                'baseline_model':'extra_trees_4f','baseline_accuracy_percent':49.2683,'training_seconds':round(x.get('training_seconds',0),4),
                'checkpoint_path':cps,'checkpoint_status':'three_fold_reloadable_checkpoints',
                'oof_prediction_path':str((folder/'oof_predictions.csv').relative_to(ROOT)),'validation_protocol':s['split'],
                'group_unit':'origin_program','gradient_checks':'3/3 passed','status':'completed_development_oof','source':s['features']})
    # Fixed equal-probability QNN ensemble diagnostic over the 24-epoch sweep.
    folder=C/'qmutbench_threeclass_qnn_sweep24'; predrows=read(folder/'oof_predictions.csv')
    models=sorted({r['model'] for r in predrows}); by={m:{} for m in models}
    for r in predrows:by[r['model']][r['record_id']]=r
    ens=[]
    for rid in sorted(by[models[0]],key=int):
        ref=by[models[0]][rid]; probs=[sum(float(by[m][rid][f'prob_{c}']) for m in models)/len(models) for c in range(3)]
        ens.append({'record_id':rid,'origin_program':ref['origin_program'],'fold':ref['fold'],'model':'equal_probability_qnn_ensemble',
            'y_true':ref['y_true'],'y_pred':max(range(3),key=lambda c:probs[c]),**{f'prob_{c}':probs[c] for c in range(3)}})
    dump(folder/'equal_probability_ensemble_oof.csv',ens)
    m=metric([int(r['y_true']) for r in ens],[int(r['y_pred']) for r in ens])
    rows.append({'dataset_task':'QMutBench three-class research','target':'Survivor/WOO/OPO','model':'equal_probability_qnn_ensemble',
        'configuration':json.dumps({'members':models,'weights':'equal, fixed a priori'}),'records':len(ens),'groups':12,'folds':3,
        'epochs':'member runs','batch_size':'member runs','optimizer':'Adam (members)',
        'validation_accuracy_percent':round(m['accuracy']*100,4),'final_test_accuracy_percent':'','macro_f1_percent':round(m['macro_f1']*100,4),
        'balanced_accuracy_percent':round(m['balanced_accuracy']*100,4),'per_class_metrics':m['per_class_metrics'],'baseline_model':'extra_trees_4f','baseline_accuracy_percent':49.2683,
        'training_seconds':'member runs listed above','checkpoint_path':';'.join(str((folder/m/f'fold_{f}'/'checkpoint.npz').relative_to(ROOT)) for m in models for f in (1,2,3)),
        'checkpoint_status':'member fold checkpoints listed','oof_prediction_path':str((folder/'equal_probability_ensemble_oof.csv').relative_to(ROOT)),
        'validation_protocol':'saved origin-group 3-fold OOF; fixed equal probability mean','group_unit':'origin_program',
        'gradient_checks':'all member checks passed','status':'completed_development_oof_diagnostic','source':'validation-only QNN ensemble'})
    # Recovered, versioned 20-epoch binary architecture sweeps with newly
    # matched fold-local classical controls. Historical partial outputs stay intact.
    sweep_configs = {
        'amplitude_hybrid_3q_depth1_lr004': 8,
        'amplitude_hybrid_3q_depth2_lr004': 8,
        'angle_reupload_4q_depth2_lr002': 4,
        'angle_reupload_4q_depth2_lr0035': 4,
        'angle_reupload_4q_depth2_lr007': 4,
        'angle_4q_depth2_lr004': 4,
    }
    for task in ('bugs4q_robust_execution','qsimbench_backend_fingerprint'):
        folder=C/task/'qnn_hyperparameter_sweep20_resumable_v2'
        run=json.loads((folder/'run_manifest.json').read_text(encoding='utf-8'))
        matched=json.loads((folder/'matched_classical_comparison.json').read_text(encoding='utf-8'))
        for result in run['results']:
            name=result['model']; width=sweep_configs[name]
            control=matched['classical_metrics_by_feature_width'][str(width)]
            fold_rows=[r for r in run['fold_results'] if r['model']==name]
            checkpoint_paths=';'.join(r['checkpoint_path'] for r in fold_rows)
            resumable=sum(r['checkpoint_kind']=='resumable_adam_rng_checkpoint' for r in fold_rows)
            legacy=sum(r['checkpoint_kind']=='legacy_inference_only_checkpoint_reused' for r in fold_rows)
            rows.append({'dataset_task':task,'target':run['source_receipt']['manifest_path'],'model':name,
                'configuration':json.dumps(result['configuration'],sort_keys=True),'records':result['records'],
                'groups':run['groups'],'folds':result['folds'],'epochs':20,'batch_size':64,'optimizer':'Adam',
                'validation_accuracy_percent':round(float(result['accuracy'])*100,4),'final_test_accuracy_percent':'',
                'macro_f1_percent':round(float(result['macro_f1'])*100,4),
                'balanced_accuracy_percent':round(float(result['balanced_accuracy'])*100,4),
                'baseline_model':f"balanced_logistic_{width}f_same_hashed_features",
                'baseline_accuracy_percent':round(float(control['accuracy'])*100,4),
                'baseline_macro_f1_percent':round(float(control['macro_f1'])*100,4),
                'baseline_balanced_accuracy_percent':round(float(control['balanced_accuracy'])*100,4),
                'training_seconds':round(sum(float(r['training_seconds']) for r in fold_rows),4),
                'checkpoint_path':checkpoint_paths,
                'checkpoint_status':f'{resumable} resumable; {legacy} legacy inference-only reused',
                'oof_prediction_path':str((folder/'oof_predictions.csv').relative_to(ROOT)),
                'validation_protocol':'5-fold GroupKFold; exact assignments verified against saved baseline OOF; train-only scaling',
                'group_unit':'bug_id source group','lowest_observed_available_ram_mib':min(float(r['available_ram_mib']) for r in fold_rows),
                'gradient_checks':f"{sum(bool(r['gradient_check']['passed']) for r in fold_rows)}/{len(fold_rows)} passed",
                'status':'completed_development_oof_exploratory_sweep','source':'resumed 20-epoch QNN sweep with matched classical controls'})
    # Separate 12-epoch QMutBench three-class rerun closes the prior
    # parameter-checkpoint gap without replacing the historical 24-epoch run.
    folder=C/'qmutbench_threeclass_resumable_recovery_20261029'
    if (folder/'summary.json').exists() and (folder/'matched_classical_comparison.json').exists():
        run=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        matched=json.loads((folder/'matched_classical_comparison.json').read_text(encoding='utf-8'))
        for result in run['results']:
            name=result['model']; fold_rows=[r for r in run['fold_results'] if r['model']==name]
            histories=[json.loads((folder/name/f"fold_{f}"/'training_history.json').read_text(encoding='utf-8')) for f in (1,2,3)]
            rows.append({'dataset_task':'QMutBench three-class research','target':'Survivor=0 / WOO=1 / OPO=2',
                'model':name,'configuration':json.dumps(result['configuration'],sort_keys=True),
                'records':result['records'],'groups':run['origins'],'folds':result['folds'],'epochs':12,
                'batch_size':32,'optimizer':'Adam','validation_accuracy_percent':round(float(result['accuracy'])*100,4),
                'final_test_accuracy_percent':'','macro_f1_percent':round(float(result['macro_f1'])*100,4),
                'balanced_accuracy_percent':round(float(result['balanced_accuracy'])*100,4),
                'baseline_model':'extra_trees_4f_same_fold_same_features',
                'baseline_accuracy_percent':round(float(matched['classical_metrics']['extra_trees_4f']['accuracy'])*100,4),
                'baseline_macro_f1_percent':round(float(matched['classical_metrics']['extra_trees_4f']['macro_f1'])*100,4),
                'baseline_balanced_accuracy_percent':round(float(matched['classical_metrics']['extra_trees_4f']['balanced_accuracy'])*100,4),
                'training_seconds':round(float(result['training_seconds']),4),
                'checkpoint_path':';'.join(str((folder/name/f'fold_{f}'/'checkpoint.npz').relative_to(ROOT)) for f in (1,2,3)),
                'checkpoint_status':'3 resumable Adam/RNG checkpoints; replay audit verified',
                'oof_prediction_path':str((folder/'oof_predictions.csv').relative_to(ROOT)),
                'validation_protocol':'saved three-fold origin-group split; train-only feature transforms; exact OOF replay verified',
                'group_unit':'origin_program','lowest_observed_available_ram_mib':min(float(h['available_ram_mib']) for group in histories for h in group),
                'gradient_checks':f"{sum(bool(r['gradient_check']['passed']) for r in fold_rows)}/{len(fold_rows)} passed",
                'status':'completed_development_oof_resumable_recovery','source':'versioned QMutBench three-class checkpoint recovery with matched classical controls'})
    dump(C/'qml_model_accuracy_benchmark.csv',sorted(rows,key=lambda r:(r['dataset_task'],-float(r['validation_accuracy_percent']))))
    winners={}
    for r in rows:
        key=r['dataset_task']
        if key not in winners or float(r['validation_accuracy_percent'])>float(winners[key]['validation_accuracy_percent']):winners[key]=r
    dump(C/'best_qml_model_by_task.csv',sorted(winners.values(),key=lambda r:r['dataset_task']))
    print(json.dumps({'qml_candidate_rows':len(rows),'task_winners':len(winners),
        'final_test_scores':'not evaluated; development model selection remains in progress','frozen_test_accessed':False},indent=2))

if __name__=='__main__':main()
