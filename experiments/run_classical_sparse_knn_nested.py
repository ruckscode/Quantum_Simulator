"""Nested grouped sparse-KNN screen, comparable to the text-model search."""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
from sklearn.model_selection import GridSearchCV,GroupKFold,ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion,Pipeline

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/experiments/classical_ml_benchmarks_20261030/inputs'
CFG=BASE/'classical_nine_source_v2_config.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def metric(y,p):
 return {'records':int(len(y)),'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,p)),'confusion_matrix_labels_0_1':confusion_matrix(y,p,labels=[0,1]).tolist()}
def grids():
 word=TfidfVectorizer(analyzer='word',ngram_range=(1,3),max_features=30000,min_df=1,sublinear_tf=True,strip_accents='unicode',token_pattern=r'(?u)\b\w+\b',dtype=np.float32)
 char=TfidfVectorizer(analyzer='char',ngram_range=(2,5),max_features=50000,min_df=1,sublinear_tf=True,strip_accents='unicode',dtype=np.float32)
 wb=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),max_features=50000,min_df=1,sublinear_tf=True,strip_accents='unicode',dtype=np.float32)
 combo=FeatureUnion([('word',word),('char',char)],n_jobs=1)
 return [{'features':[v],'knn':[KNeighborsClassifier(metric='cosine',algorithm='brute',n_jobs=1)],'knn__n_neighbors':[3,5,9,15],'knn__weights':['uniform','distance']} for v in (word,char,wb,combo)]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--input',default=None);a=ap.parse_args()
 config=json.loads(CFG.read_text(encoding='utf-8'));spec=next((t.copy() for t in config['tasks'] if t['name']==a.task),None)
 if spec is None:
  if not a.input:raise ValueError('Unknown task needs --input')
  spec={'name':a.task,'input':a.input}
 if a.input:spec['input']=a.input
 src=Path(spec['input']);src=src if src.is_absolute() else ROOT/src
 if spec.get('sha256') and sha(src)!=spec['sha256']:raise ValueError('source hash mismatch')
 df=pd.read_csv(src).fillna({'code':''});evx=df.code.astype(str).to_numpy();evy=df.label.to_numpy(np.int8);evg=df.bug_id.astype(str).to_numpy()
 compressed=a.task=='bugs4q_robust_execution'
 if compressed:
  if df.groupby('code').label.nunique().max()!=1 or df.groupby('code').bug_id.nunique().max()!=1:raise ValueError('duplicate code crosses labels/groups')
  unique=df.drop_duplicates('code').reset_index(drop=True);X=unique.code.astype(str).to_numpy();y=unique.label.to_numpy(np.int8);g=unique.bug_id.astype(str).to_numpy()
 else:X,y,g=evx,evy,evg
 out=ROOT/'reports/experiments/classical_sparse_knn_nested_20261030'/a.task;out.mkdir(parents=True,exist_ok=True)
 if any(out.iterdir()):raise FileExistsError(f'refusing to overwrite {out}')
 gs=grids();n=sum(len(ParameterGrid(x)) for x in gs)
 manifest={'status':'running','task':a.task,'records':len(df),'fit_rows':len(X),'groups':int(df.bug_id.nunique()),'input':str(src.relative_to(ROOT)),'input_sha256':sha(src),'exact_duplicate_compression':compressed,'search_candidates':n,'outer_split':'GroupKFold(5)','inner_split':'GroupKFold(3) within outer training groups','selection_metric':'inner-CV accuracy','feature_families':['word TF-IDF','character TF-IDF','character-boundary TF-IDF','combined word+character TF-IDF'],'classifier':'cosine KNN; k=3,5,9,15; uniform or distance weights','frozen_test_accessed':False,'folds':[]}
 (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
 folds=[];oof=[]
 for fold,(tr,va) in enumerate(GroupKFold(5).split(X,y,g),1):
  if set(g[tr])&set(g[va]):raise AssertionError('group leakage')
  search=GridSearchCV(Pipeline([('features',TfidfVectorizer()),('knn',KNeighborsClassifier(metric='cosine',algorithm='brute'))]),gs,scoring='accuracy',cv=GroupKFold(3).split(X[tr],y[tr],g[tr]),n_jobs=1,refit=True,error_score='raise')
  t=time.perf_counter();search.fit(X[tr],y[tr]);seconds=time.perf_counter()-t
  if compressed:
   held=set(g[va]);ix=np.flatnonzero(np.isin(evg,list(held)));yt=evy[ix];pred=search.predict(evx[ix]).astype(np.int8)
  else:ix=va;yt=evy[ix];pred=search.predict(evx[ix]).astype(np.int8)
  score=metric(yt,pred);path=out/f'fold_{fold}_oof.csv'
  with path.open('w',encoding='utf-8',newline='') as f:
   w=csv.writer(f);w.writerow(['record_id','group_id','fold','y_true','y_pred']);w.writerows(zip(df.record_id.iloc[ix].astype(str),evg[ix],np.full(len(ix),fold),yt,pred))
  oof.extend((int(i),int(y0),int(p0)) for i,y0,p0 in zip(ix,yt,pred))
  row={'fold':fold,'train_groups':len(set(g[tr])),'validation_groups':len(set(g[va])),'fit_rows':len(tr),'validation_records':len(ix),'selected_params':search.best_params_,'best_inner_accuracy':float(search.best_score_),'outer_metrics':score,'fit_seconds':round(seconds,3),'prediction_file':path.name}
  folds.append(row);manifest['folds']=folds;manifest['completed_folds']=fold
  (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n',encoding='utf-8')
  print(f"{a.task} KNN fold {fold}/5 accuracy={score['accuracy']:.4f} macroF1={score['macro_f1']:.4f} inner_accuracy={search.best_score_:.4f} seconds={seconds:.1f}",flush=True)
 oof.sort();yy=np.array([r[1] for r in oof]);pp=np.array([r[2] for r in oof]);summary=metric(yy,pp);summary['outer_fold_macro_f1_sd']=float(np.std([f['outer_metrics']['macro_f1'] for f in folds],ddof=1));
 (out/'pooled_oof_metrics.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
 manifest.update({'status':'complete','pooled_metrics':summary,'runtime_seconds':round(sum(f['fit_seconds'] for f in folds),3),'completed_folds':5})
 (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n',encoding='utf-8')
 print(json.dumps({'task':a.task,'status':'complete','search_candidates':n,'pooled_metrics':summary},indent=2),flush=True)
if __name__=='__main__':main()
