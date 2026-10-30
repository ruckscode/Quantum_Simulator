"""Nested group-CV search of text representations/classifiers per source."""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
from sklearn.model_selection import GridSearchCV,GroupKFold,ParameterGrid
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion,Pipeline
from sklearn.svm import LinearSVC

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/experiments/classical_ml_benchmarks_20261030/inputs'
CONFIG=BASE/'classical_nine_source_v2_config.json';SEED=20261030
def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def grid_spec():
 word=TfidfVectorizer(analyzer='word',ngram_range=(1,3),max_features=30000,min_df=1,sublinear_tf=True,strip_accents='unicode',token_pattern=r'(?u)\b\w+\b',dtype=np.float32)
 char=TfidfVectorizer(analyzer='char',ngram_range=(2,5),max_features=50000,min_df=1,sublinear_tf=True,strip_accents='unicode',dtype=np.float32)
 wb=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),max_features=50000,min_df=1,sublinear_tf=True,strip_accents='unicode',dtype=np.float32)
 combo=FeatureUnion([('word',word),('char',char)],n_jobs=1)
 grids=[]
 for vec in (word,char,wb,combo):
  grids += [
   {'features':[vec],'clf':[LinearSVC(random_state=SEED,max_iter=100000)],'clf__C':[.03,.3,3,30],'clf__class_weight':[None,'balanced']},
   {'features':[vec],'clf':[LogisticRegression(max_iter=1500,solver='liblinear',random_state=SEED)],'clf__C':[.03,.3,3,30],'clf__class_weight':[None,'balanced']},
   {'features':[vec],'clf':[SGDClassifier(loss='modified_huber',max_iter=5000,tol=1e-4,random_state=SEED)],'clf__alpha':[1e-5,1e-4,1e-3],'clf__class_weight':[None,'balanced']},
   {'features':[vec],'clf':[ComplementNB()],'clf__alpha':[.05,.2,1,5]},
  ]
 return grids
def metrics(y,p):
 return {'records':int(len(y)),'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,p)),'confusion_matrix_labels_0_1':confusion_matrix(y,p,labels=[0,1]).tolist()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--input',default=None);a=ap.parse_args()
 cfg=json.loads(CONFIG.read_text(encoding='utf-8'));spec=next((t.copy() for t in cfg['tasks'] if t['name']==a.task),None)
 if spec is None:
  if not a.input:raise ValueError(f'unknown task {a.task}; provide --input')
  spec={'name':a.task,'input':a.input}
 if a.input:spec['input']=a.input
 src=Path(spec['input']);src=src if src.is_absolute() else ROOT/src
 if spec.get('sha256') and sha256(src)!=spec['sha256']:raise ValueError('source SHA256 mismatch')
 df=pd.read_csv(src).fillna({'code':''})
 if df.record_id.duplicated().any() or set(df.label.astype(int).unique())!={0,1}:raise ValueError('invalid record IDs/labels')
 eval_X=df.code.astype(str).to_numpy();eval_y=df.label.to_numpy(np.int8);eval_groups=df.bug_id.astype(str).to_numpy()
 # Robust-execution records repeat identical source snippets many times within
 # an issue. Fit on each exact snippet once with its multiplicity as weight;
 # continue scoring every held-out execution record below.
 deduplicated=(a.task=='bugs4q_robust_execution')
 if deduplicated:
  if df.groupby('code').label.nunique().max()!=1 or df.groupby('code').bug_id.nunique().max()!=1:
   raise ValueError('Robust-execution duplicate code crosses label/group; refusing deduplication')
  unique=df.drop_duplicates('code',keep='first').reset_index(drop=True)
  X=unique.code.astype(str).to_numpy();y=unique.label.to_numpy(np.int8);groups=unique.bug_id.astype(str).to_numpy()
  weights=df.groupby('code',sort=False).size().reindex(unique.code).to_numpy(dtype=float)
 else:
  X=eval_X;y=eval_y;groups=eval_groups;weights=None
 if len(np.unique(groups))<5:raise ValueError('fewer than 5 groups')
 out=ROOT/'reports/experiments/classical_nested_accuracy_search_20261030'/a.task;out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()):raise FileExistsError(f'refusing to overwrite {out}')
 grids=grid_spec();n_candidates=sum(len(ParameterGrid(g)) for g in grids)
 manifest={'status':'running','task':a.task,'input':str(src.relative_to(ROOT)),'input_sha256':sha256(src),'records':len(df),'groups':int(df.bug_id.nunique()),'fit_rows':len(X),'exact_duplicate_compression':deduplicated,'class_counts':{str(int(k)):int(v) for k,v in df.label.value_counts().items()},'outer_split':'GroupKFold(5)','inner_split':'GroupKFold(3) within each outer training partition','selection_metric':'inner-CV accuracy (macro-F1 and balanced accuracy also reported)','search_candidates':n_candidates,'representations':['word TF-IDF (1-3 grams)','character TF-IDF (2-5 grams)','character-boundary TF-IDF (3-5 grams)','combined word + character TF-IDF'],'classifiers':['LinearSVC','LogisticRegression','SGD modified-huber','ComplementNB'],'frozen_test_accessed':False,'folds':[]}
 (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
 fold_results=[];pooled=[]
 for fold,(tr,va) in enumerate(GroupKFold(5).split(X,y,groups),1):
  if set(groups[tr])&set(groups[va]):raise AssertionError('group leakage')
  inner=GroupKFold(3).split(X[tr],y[tr],groups[tr]);pipe=Pipeline([('features',TfidfVectorizer()),('clf',LinearSVC(max_iter=100000))])
  search=GridSearchCV(pipe,grids,scoring='accuracy',cv=inner,n_jobs=1,refit=True,error_score='raise',return_train_score=False)
  t=time.perf_counter()
  if deduplicated:search.fit(X[tr],y[tr],clf__sample_weight=weights[tr])
  else:search.fit(X[tr],y[tr])
  elapsed=time.perf_counter()-t
  if deduplicated:
   held_groups=set(groups[va]);full_va=np.flatnonzero(np.isin(eval_groups,list(held_groups)))
   pred=search.predict(eval_X[full_va]).astype(np.int8);y_eval=eval_y[full_va];eval_indices=full_va
  else:
   eval_indices=va;y_eval=y[va];pred=search.predict(X[va]).astype(np.int8)
  score=metrics(y_eval,pred)
  path=out/f'fold_{fold}_oof.csv'
  with path.open('w',encoding='utf-8',newline='') as f:
   w=csv.writer(f);w.writerow(['record_id','group_id','fold','y_true','y_pred']);w.writerows(zip(df.record_id.iloc[eval_indices].astype(str),eval_groups[eval_indices],np.full(len(eval_indices),fold),y_eval,pred))
  pooled.extend((int(i),int(y0),int(p0)) for i,y0,p0 in zip(eval_indices,y_eval,pred))
  rec={'fold':fold,'train_records':int(weights[tr].sum()) if deduplicated else len(tr),'fit_rows':len(tr),'validation_records':len(eval_indices),'train_groups':len(set(groups[tr])),'validation_groups':len(set(groups[va])),'selected_params':search.best_params_,'best_inner_accuracy':float(search.best_score_),'outer_metrics':score,'fit_seconds':round(elapsed,3),'prediction_file':path.name}
  fold_results.append(rec);manifest['folds']=fold_results;manifest['completed_folds']=fold
  (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n',encoding='utf-8')
  print(f"{a.task} fold {fold}/5 accuracy={score['accuracy']:.4f} macroF1={score['macro_f1']:.4f} inner_accuracy={search.best_score_:.4f} secs={elapsed:.1f}",flush=True)
 pooled.sort();
 if len(pooled)!=len(df) or len({r[0] for r in pooled})!=len(df):raise ValueError('OOF coverage incomplete')
 yy=np.array([r[1] for r in pooled]);pp=np.array([r[2] for r in pooled]);summary=metrics(yy,pp)
 summary.update({'outer_fold_accuracy_mean':float(np.mean([r['outer_metrics']['accuracy'] for r in fold_results])),'outer_fold_macro_f1_mean':float(np.mean([r['outer_metrics']['macro_f1'] for r in fold_results])),'outer_fold_macro_f1_sd':float(np.std([r['outer_metrics']['macro_f1'] for r in fold_results],ddof=1))})
 prior=int(np.bincount(y).argmax());summary['majority_accuracy_baseline']=float(np.mean(y==prior));summary['majority_macro_f1_baseline']=float(f1_score(y,np.full(len(y),prior),average='macro',zero_division=0))
 (out/'pooled_oof_metrics.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
 manifest.update({'status':'complete','completed_folds':5,'runtime_seconds':round(sum(r['fit_seconds'] for r in fold_results),3),'pooled_metrics':summary})
 (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n',encoding='utf-8')
 print(json.dumps({'task':a.task,'status':'complete','records':len(df),'groups':len(set(groups)),'search_candidates':n_candidates,'pooled_metrics':summary},indent=2),flush=True)
if __name__=='__main__':main()
