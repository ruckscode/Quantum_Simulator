"""Lightweight integrity and OOF-metric audit for newly trained QNN artifacts."""
from __future__ import annotations
import csv,json,math,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'reports/experiments/qmutbench_multisource_qml_campaign_20261028'

def read(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def score(rows,nclasses):
 labels=range(nclasses);y=[int(r['y_true']) for r in rows];p=[int(r['y_pred']) for r in rows];per={};f=[];rec=[]
 for c in labels:
  tp=sum(a==c and b==c for a,b in zip(y,p));fp=sum(a!=c and b==c for a,b in zip(y,p));fn=sum(a==c and b!=c for a,b in zip(y,p));sup=sum(a==c for a in y)
  pr=tp/(tp+fp) if tp+fp else 0.;rr=tp/sup if sup else 0.;ff=2*pr*rr/(pr+rr) if pr+rr else 0.
  per[str(c)]={'precision':pr,'recall':rr,'f1':ff,'support':sup};f.append(ff);rec.append(rr)
 return {'accuracy':sum(a==b for a,b in zip(y,p))/len(y),'macro_f1':sum(f)/nclasses,'balanced_accuracy':sum(rec)/nclasses,'per_class':per}

def audit_run(name,folder,nclasses,groups_field='group_id'):
 s=json.loads((folder/'summary.json').read_text(encoding='utf-8'));pred=read(folder/'oof_predictions.csv')
 report={'task':name,'summary_status':s['status'],'records':s['records'],'groups':s['groups'],'models':{}}
 for model,summary in s['results'].items():
  rows=[r for r in pred if r['model']==model]
  if len(rows)!=summary['records'] or len({r['record_id'] for r in rows})!=len(rows):raise AssertionError(f'{name}/{model}: OOF row or ID mismatch')
  grouped={}
  for r in rows:grouped.setdefault(r[groups_field],set()).add(r['fold'])
  if any(len(v)!=1 for v in grouped.values()):raise AssertionError(f'{name}/{model}: group crosses folds')
  m=score(rows,nclasses)
  for key in ('accuracy','macro_f1','balanced_accuracy'):
   if abs(m[key]-summary[key])>1e-12:raise AssertionError(f'{name}/{model}: {key} does not match summary')
  checkpoints=summary.get('checkpoint_paths',[])
  if len(checkpoints)!=summary['folds']:raise AssertionError(f'{name}/{model}: missing fold checkpoint path')
  for cp in checkpoints:
   path=ROOT/cp
   if not zipfile.is_zipfile(path):raise AssertionError(f'invalid npz {path}')
   with zipfile.ZipFile(path) as z:
    if z.testzip() is not None:raise AssertionError(f'corrupt checkpoint {path}')
    names=set(z.namelist())
    if not {'theta.npy','head.npy','bias.npy','config.npy','epoch.npy'}<=names:raise AssertionError(f'incomplete arrays in {path}')
  fold_results=[x for x in s['fold_results'] if x['model']==model]
  if len(fold_results)!=summary['folds'] or not all(x['gradient_check']['passed'] for x in fold_results):raise AssertionError(f'{name}/{model}: fold or gradient audit failed')
  report['models'][model]={'oof_rows':len(rows),'metrics_recomputed':m,'metrics_match_summary':True,'checkpoint_files':len(checkpoints),'checkpoint_archives_valid':True,'gradient_checks_passed':len(fold_results)}
 return report

def main():
 reports=[]
 for task in ('bugs4q_robust_execution','qsimbench_backend_fingerprint'):
  reports.append(audit_run(task,C/task/'resource_light_qnn',2))
 for folder_name in ('qmutbench_threeclass_resource_light_qnn','qmutbench_threeclass_qnn_sweep24'):
  folder=C/folder_name;s=json.loads((folder/'summary.json').read_text(encoding='utf-8'));pred=read(folder/'oof_predictions.csv')
  result={'task':folder_name,'summary_status':s['status'],'records':s['records'],'groups':s['origins'],'models':{}}
  for model,x in ((r['model'],r) for r in s['results']):
   rows=[r for r in pred if r['model']==model];m=score(rows,3)
   for key in ('accuracy','macro_f1','balanced_accuracy'):
    if abs(m[key]-x[key])>1e-12:raise AssertionError(f'{folder_name}/{model}: {key} mismatch')
   cps=[folder/model/f'fold_{f}'/'checkpoint.npz' for f in range(1,4)]
   for cp in cps:
    if not zipfile.is_zipfile(cp):raise AssertionError(f'invalid checkpoint {cp}')
    with zipfile.ZipFile(cp) as z:
     if z.testzip() is not None or not {'theta.npy','head.npy','bias.npy','config.npy','epoch.npy','class_weights.npy'}<=set(z.namelist()):raise AssertionError(f'incomplete checkpoint {cp}')
   fr=[r for r in s['fold_results'] if r['model']==model]
   if len(fr)!=3 or not all(r['gradient_check']['passed'] for r in fr):raise AssertionError(f'gradient/fold failure {model}')
   group_folds={}
   for r in rows:group_folds.setdefault(r['origin_program'],set()).add(r['fold'])
   if any(len(v)!=1 for v in group_folds.values()):raise AssertionError(f'origin leakage in {model}')
   result['models'][model]={'oof_rows':len(rows),'metrics_recomputed':m,'metrics_match_summary':True,'checkpoint_files':len(cps),'checkpoint_archives_valid':True,'gradient_checks_passed':3}
  reports.append(result)
 out={'status':'verified','runs':reports,'frozen_test_accessed':False,'test_accuracy':'not evaluated; these are development OOF metrics'}
 (C/'completed_qnn_artifact_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
