"""Recreate reloadable QMutBench three-class QNN weights on saved dev folds."""
from __future__ import annotations
import csv,json,math,time,sys,os
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from train_resource_light_qnn_tasks import (CAMPAIGN,FLOOR_MIB,features_from_text,
    minmax_fit_apply,q_expectations,check_ram)
EPOCHS=int(os.environ.get('QUT_EPOCHS','12'));BATCH=32;SEED=28102026
NUMERIC=['qubits','gates','Position_int','position_fraction','qasm_diff_orig_span','qasm_diff_mut_span','qasm_diff_total_span','qasm_diff_expected_gate_match','qasm_diff_qubit_set_equal','qasm_diff_qubits_touched','qasm_diff_removed_arity','qasm_diff_added_arity','qasm_diff_exact_single_edit','qasm_diff_is_local']
CATEGORICAL=['algorithm','operation','gate','dominant_state','qasm_diff_removed_gate','qasm_diff_added_gate','qasm_diff_alignment_class','qasm_diff_local_prev_gate','qasm_diff_local_next_gate']

def encode_saved_fold(records,train_idx,valid_idx,feature_names):
 import numpy as np
 raw=[json.loads(r['features_json']) for r in records]
 train_raw=[raw[i] for i in train_idx]
 encoded={}
 for name in NUMERIC:
  values=[]
  for row in raw:
   try: values.append(float(row.get(name)))
   except (TypeError,ValueError): values.append(float('nan'))
  finite=[values[i] for i in train_idx if math.isfinite(values[i])]
  lo=min(finite) if finite else 0.;hi=max(finite) if finite else 0.
  encoded['numeric__'+name]=[0.0 if not math.isfinite(v) or hi<=lo else (v-lo)/(hi-lo) for v in values]
 for name in CATEGORICAL:
  cats=sorted({str(row.get(name) if row.get(name) is not None else 'nan') for row in train_raw})
  for cat in cats:
   encoded[f'categorical__{name}_{cat}']=[1.0 if str(row.get(name) if row.get(name) is not None else 'nan')==cat else 0.0 for row in raw]
 missing=set(feature_names)-set(encoded)
 if missing: raise ValueError('Saved fold feature names could not be rebuilt: '+repr(sorted(missing)))
 matrix=np.column_stack([encoded[name] for name in feature_names])
 tr=matrix[train_idx];va=matrix[valid_idx]
 mins=tr.min(axis=0);maxs=tr.max(axis=0);span=maxs-mins;span[span==0]=1.
 return np.clip((tr-mins)/span,0,1),np.clip((va-mins)/span,0,1)

def softmax(z):
 import numpy as np
 z=z-z.max(axis=1,keepdims=True); e=np.exp(z); return e/e.sum(axis=1,keepdims=True)

def score(y,pred):
 import numpy as np
 labels=sorted(set(map(int,y))|set(map(int,pred))); per={}; f1=[]; recall=[]
 for c in labels:
  tp=int(np.sum((y==c)&(pred==c)));fp=int(np.sum((y!=c)&(pred==c)));fn=int(np.sum((y==c)&(pred!=c)));sup=int(np.sum(y==c))
  pr=tp/(tp+fp) if tp+fp else 0.;re=tp/sup if sup else 0.;ff=2*pr*re/(pr+re) if pr+re else 0.
  per[str(c)]={'precision':pr,'recall':re,'f1':ff,'support':sup};f1.append(ff);recall.append(re)
 return {'accuracy':float(np.mean(y==pred)),'macro_f1':float(np.mean(f1)),'balanced_accuracy':float(np.mean(recall)),
  'per_class':per,'confusion_matrix':[[int(np.sum((y==a)&(pred==b))) for b in labels] for a in labels]}

def probs(x,theta,head,bias,cfg):
 import numpy as np
 z=q_expectations(x,theta,cfg['qubits'],cfg['encoding'],cfg['depth'],cfg['reupload'])
 if cfg['qubits']==2: z=np.column_stack([z,z[:,0]*z[:,1]])
 else: z=z[:,:3]
 return softmax(z@head.T+bias),z

def check_gradient(cfg):
 import numpy as np
 n=cfg['qubits'];d=cfg['depth'];k=3
 x=np.full((3,cfg['features']),.31); y=np.array([0,1,2]);theta=np.linspace(-.2,.2,n*d);head=np.arange(k*(3 if n==2 else 3),dtype=float).reshape(k,3)*.03;bias=np.zeros(k)
 def loss(t):
  p,_=probs(x,t,head,bias,cfg);return float(-np.log(p[np.arange(k),y]).mean())
 plus=theta.copy();minus=theta.copy();plus[0]+=math.pi/2;minus[0]-=math.pi/2
 p,z=probs(x,theta,head,bias,cfg);err=p-np.eye(3)[y];dz=err@head
 zp=q_expectations(x,plus,n,cfg['encoding'],d,cfg['reupload']);zm=q_expectations(x,minus,n,cfg['encoding'],d,cfg['reupload'])
 if n==2: zp=np.column_stack([zp,zp[:,0]*zp[:,1]]);zm=np.column_stack([zm,zm[:,0]*zm[:,1]])
 else: zp=zp[:,:3];zm=zm[:,:3]
 shift=float(np.sum(dz*(zp-zm)/2)/k);eps=1e-5
 fd=(loss(plus*0+theta+np.eye(1,len(theta),0)[0]*eps)-loss(theta-np.eye(1,len(theta),0)[0]*eps))/(2*eps)
 return {'parameter_shift':shift,'finite_difference':fd,'absolute_error':abs(shift-fd),'passed':abs(shift-fd)<3e-4}

def train(x,y,cfg,seed,checkpoint,history_path):
 import numpy as np
 n=cfg['qubits'];outdims=3;featout=3;paramn=n*cfg['depth'];rng=np.random.default_rng(seed)
 theta=rng.normal(0,.12,paramn);head=rng.normal(0,.15,(outdims,featout));bias=np.zeros(outdims)
 mt=np.zeros_like(theta);vt=np.zeros_like(theta);mh=np.zeros_like(head);vh=np.zeros_like(head);mb=np.zeros_like(bias);vb=np.zeros_like(bias)
 counts=np.bincount(y,minlength=3);weights=len(y)/(3*np.maximum(counts,1));sw=weights[y];lr=cfg['learning_rate'];hist=[];start=time.perf_counter()
 start_epoch=1
 if os.environ.get('QUT_RESUME')=='1' and checkpoint.exists():
  saved=np.load(checkpoint,allow_pickle=False);required={'mt','vt','mh','vh','mb','vb','rng_state'}
  if not required.issubset(saved.files):raise ValueError(f'Checkpoint {checkpoint} lacks resumable optimizer state')
  theta=saved['theta'];head=saved['head'];bias=saved['bias'];mt=saved['mt'];vt=saved['vt'];mh=saved['mh'];vh=saved['vh'];mb=saved['mb'];vb=saved['vb']
  start_epoch=int(saved['epoch'])+1;rng.bit_generator.state=json.loads(str(saved['rng_state'].item()))
  if history_path.exists():hist=json.loads(history_path.read_text(encoding='utf-8'))
 for epoch in range(start_epoch,EPOCHS+1):
  order=rng.permutation(len(y));losses=[]
  for bi,off in enumerate(range(0,len(y),BATCH),1):
   check_ram();ix=order[off:off+BATCH];xb=x[ix];yb=y[ix];wb=sw[ix];bs=len(ix)
   p,z=probs(xb,theta,head,bias,cfg);err=(p-np.eye(3)[yb])*wb[:,None]/bs
   gh=err.T@z;gb=err.sum(axis=0);dz=err@head;gt=np.zeros_like(theta)
   for j in range(paramn):
    tp=theta.copy();tm=theta.copy();tp[j]+=math.pi/2;tm[j]-=math.pi/2
    zp=q_expectations(xb,tp,n,cfg['encoding'],cfg['depth'],cfg['reupload']);zm=q_expectations(xb,tm,n,cfg['encoding'],cfg['depth'],cfg['reupload'])
    if n==2:zp=np.column_stack([zp,zp[:,0]*zp[:,1]]);zm=np.column_stack([zm,zm[:,0]*zm[:,1]])
    else:zp=zp[:,:3];zm=zm[:,:3]
    gt[j]=float(np.sum(dz*(zp-zm)/2))
   t=(epoch-1)*math.ceil(len(y)/BATCH)+bi;b1=.9;b2=.999
   mt=b1*mt+(1-b1)*gt;vt=b2*vt+(1-b2)*gt*gt;mh=b1*mh+(1-b1)*gh;vh=b2*vh+(1-b2)*gh*gh;mb=b1*mb+(1-b1)*gb;vb=b2*vb+(1-b2)*gb*gb
   theta-=lr*(mt/(1-b1**t))/(np.sqrt(vt/(1-b2**t))+1e-8);head-=lr*(mh/(1-b1**t))/(np.sqrt(vh/(1-b2**t))+1e-8);bias-=lr*(mb/(1-b1**t))/(np.sqrt(vb/(1-b2**t))+1e-8)
   losses.append(float(-np.mean(wb*np.log(p[np.arange(bs),yb]+1e-12))))
  hist.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'elapsed_seconds':time.perf_counter()-start,'available_ram_mib':check_ram()})
  np.savez(checkpoint,theta=theta,head=head,bias=bias,config=json.dumps(cfg),epoch=epoch,class_weights=weights,
   mt=mt,vt=vt,mh=mh,vh=vh,mb=mb,vb=vb,rng_state=json.dumps(rng.bit_generator.state))
  history_path.write_text(json.dumps(hist,indent=2),encoding='utf-8')
 return theta,head,bias,hist,time.perf_counter()-start

def main():
 if check_ram()<700:raise MemoryError('preflight requires 700 MiB available')
 import numpy as np
 if check_ram()<650:raise MemoryError('NumPy import left less than 650 MiB; stopped before training')
 path=CAMPAIGN/'qmutbench_threeclass_compact.csv';rows=list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')))
 y=np.asarray([int(r['y_true']) for r in rows]);groups=np.asarray([r['origin_program'] for r in rows])
 if os.environ.get('QUT_SWEEP')=='1':
  configs=[{'name':'amplitude_hybrid_2q_d2_lr004','qubits':2,'features':4,'encoding':'amplitude','depth':2,'reupload':False,'learning_rate':.04},
   {'name':'amplitude_hybrid_2q_d2_lr008','qubits':2,'features':4,'encoding':'amplitude','depth':2,'reupload':False,'learning_rate':.08},
   {'name':'angle_reupload_4q_d2_lr002','qubits':4,'features':4,'encoding':'angle','depth':2,'reupload':True,'learning_rate':.02},
   {'name':'angle_reupload_4q_d2_lr0035','qubits':4,'features':4,'encoding':'angle','depth':2,'reupload':True,'learning_rate':.035},
   {'name':'angle_reupload_4q_d2_lr006','qubits':4,'features':4,'encoding':'angle','depth':2,'reupload':True,'learning_rate':.06},
   {'name':'angle_4q_d2_lr004','qubits':4,'features':4,'encoding':'angle','depth':2,'reupload':False,'learning_rate':.04}]
 else:
  configs=[{'name':'amplitude_hybrid_2q_depth2','qubits':2,'features':4,'encoding':'amplitude','depth':2,'reupload':False,'learning_rate':.04},
   {'name':'angle_hybrid_4q_depth2','qubits':4,'features':4,'encoding':'angle','depth':2,'reupload':False,'learning_rate':.04},
   {'name':'angle_reupload_hybrid_4q_depth2','qubits':4,'features':4,'encoding':'angle','depth':2,'reupload':True,'learning_rate':.035}]
 out=CAMPAIGN/os.environ.get('QUT_OUT_DIR','qmutbench_threeclass_resource_light_qnn');out.mkdir(parents=True,exist_ok=True);predictions=[];foldrows=[];started=time.perf_counter()
 split=list(csv.DictReader((CAMPAIGN.parent/'qmutbench_three_class_qnn_campaign_20261027/grouped_split_manifest.csv').open(encoding='utf-8-sig',newline='')))
 fold_features=json.loads((CAMPAIGN.parent/'qmutbench_three_class_qnn_campaign_20261027/features_by_fold.json').read_text(encoding='utf-8'))
 for fold in sorted({int(r['fold']) for r in split}):
  role={int(r['record_id']):r['role'] for r in split if int(r['fold'])==fold}
  valid=np.asarray([i for i,r in enumerate(rows) if role[int(r['record_id'])]=='validation']);trainix=np.asarray([i for i,r in enumerate(rows) if role[int(r['record_id'])]=='train'])
  if set(groups[valid])&set(groups[trainix]):raise AssertionError('origin leakage')
  if set(y[trainix])!={0,1,2} or set(y[valid])!={0,1,2}:raise AssertionError('fold misses class')
  selected=fold_features[fold-1]['features'];xx,xv=encode_saved_fold(rows,trainix,valid,selected)
  for cfg in configs:
   xtr=xx[:,:cfg['features']];xva=xv[:,:cfg['features']]
   if cfg['encoding']=='amplitude':xtr=np.pad(xtr,((0,0),(0,4-xtr.shape[1])))[:,:4];xva=np.pad(xva,((0,0),(0,4-xva.shape[1])))[:,:4]
   gc=check_gradient(cfg)
   if not gc['passed']:raise ArithmeticError(f"gradient check failed {cfg['name']}: {gc}")
   folder=out/cfg['name']/f'fold_{fold}';folder.mkdir(parents=True,exist_ok=True)
   th,h,b,hist,secs=train(xtr,y[trainix],cfg,SEED+fold,folder/'checkpoint.npz',folder/'training_history.json')
   pp,_=probs(xva,th,h,b,cfg);pd=pp.argmax(axis=1);m=score(y[valid],pd)
   result={'fold':fold,'model':cfg['name'],'records':len(valid),'origins':len(set(groups[valid])),'training_seconds':secs,'gradient_check':gc,**m}
   foldrows.append(result);(folder/'fold_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
   for j,i in enumerate(valid):predictions.append({'record_id':rows[i]['record_id'],'origin_program':rows[i]['origin_program'],'fold':fold,'model':cfg['name'],'y_true':int(y[i]),'y_pred':int(pd[j]),'prob_0':float(pp[j,0]),'prob_1':float(pp[j,1]),'prob_2':float(pp[j,2])})
 summary=[]
 for cfg in configs:
  sub=[r for r in predictions if r['model']==cfg['name']];m=score(np.array([r['y_true'] for r in sub]),np.array([r['y_pred'] for r in sub]))
  summary.append({'model':cfg['name'],**m,'folds':3,'records':len(sub),'training_seconds':sum(r['training_seconds'] for r in foldrows if r['model']==cfg['name']),'configuration':cfg})
 with (out/'oof_predictions.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(predictions[0]));w.writeheader();w.writerows(predictions)
 report={'status':'completed_development_oof_with_reloadable_checkpoints','task':'QMutBench Survivor/WOO/OPO three-class exploratory pilot','records':len(rows),'origins':len(set(groups)),'folds':3,'features':'reconstructed original four fold-selected metadata/QASM-diff features from train-fold-only selection manifests; numeric scaling fit on training fold','split':'reused saved three-fold origin-group validation manifest','results':summary,'fold_results':foldrows,'runtime_seconds':time.perf_counter()-started,'frozen_test_accessed':False}
 (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
