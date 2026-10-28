"""Export only the saved development-only QMutBench three-class pilot inputs."""
from __future__ import annotations

import csv, gzip, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/"reports/experiments/qmutbench_three_class_qnn_campaign_20261027"
RAW=ROOT/"data/qmutbench_ingestion/raw/official_all_data_20261015.csv"
CACHE=ROOT/"data/qmutbench_ml/qasm_mutation_diff_v1"
OUT=ROOT/"reports/experiments/qmutbench_multisource_qml_campaign_20261028/qmutbench_threeclass_compact.csv"
META=["algorithm","operation","gate","dominant_state","qubits","gates","Position_int","position"]
DIFF_NUM=["qasm_diff_orig_span","qasm_diff_mut_span","qasm_diff_total_span","qasm_diff_expected_gate_match","qasm_diff_qubit_set_equal","qasm_diff_qubits_touched","qasm_diff_removed_arity","qasm_diff_added_arity","qasm_diff_exact_single_edit","qasm_diff_is_local"]
DIFF_CAT=["qasm_diff_removed_gate","qasm_diff_added_gate","qasm_diff_alignment_class","qasm_diff_local_prev_gate","qasm_diff_local_next_gate"]

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 sample=list(csv.DictReader((CAMPAIGN/'sample_manifest.csv').open(encoding='utf-8-sig',newline='')))
 wanted={int(r['record_id']):r for r in sample}
 split=list(csv.DictReader((CAMPAIGN/'grouped_split_manifest.csv').open(encoding='utf-8-sig',newline='')))
 validation_fold={int(r['record_id']):int(r['fold']) for r in split if r['role']=='validation'}
 if set(wanted)!=set(validation_fold): raise ValueError('Sample and saved validation split IDs differ')
 raw_sha=sha(RAW)
 cache_manifest=json.loads((CACHE/'manifest.json').read_text(encoding='utf-8'))
 if raw_sha!='3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778' or cache_manifest.get('source_sha256')!=raw_sha:
  raise ValueError('QMutBench source/cache digest mismatch')
 selected={}; allowed=set(wanted)
 with RAW.open(encoding='utf-8-sig',newline='') as stream:
  reader=csv.reader(stream); header=next(reader); id_i=header.index('') if header[0]=='' else header.index('Unnamed: 0')
  fields=[*META,'Killed']; indices={name:header.index(name) for name in fields}
  for row in reader:
   rid=int(row[id_i])
   if rid in allowed:
    selected[rid]={name:row[ix] for name,ix in indices.items()}
 if set(selected)!=allowed: raise ValueError('Missing selected development rows in source CSV')
 diff={}; required=['record_id',*DIFF_NUM,*DIFF_CAT]
 for chunk in sorted(CACHE.glob('diff_*.csv.gz')):
  with gzip.open(chunk,'rt',encoding='utf-8',newline='') as f:
   reader=csv.DictReader(f)
   for row in reader:
    rid=int(row['record_id'])
    if rid in allowed: diff[rid]={k:row[k] for k in required[1:]}
 if set(diff)!=allowed: raise ValueError('QASM feature cache does not align to selected development records')
 rows=[]
 for rid,meta in wanted.items():
  if meta['label']!=selected[rid]['Killed']: raise ValueError(f'label mismatch for selected record {rid}')
  feature_values={**selected[rid],**diff[rid]}
  feature_values['position_fraction']={'0-10%':.05,'11-20%':.155,'21-30%':.255,'31-40%':.355,'41-50%':.455,'51-60%':.555,'61-70%':.655,'71-80%':.755,'81-90%':.855,'91-100%':.955}.get(selected[rid].get('position'))
  text=' '.join(str(selected[rid].get(k,'')) for k in META)+' '+' '.join(str(diff[rid].get(k,'')) for k in [*DIFF_NUM,*DIFF_CAT])
  rows.append({'record_id':rid,'origin_program':meta['origin_program'],'label':meta['label'],'y_true':int(meta['y_true']),
               'fold':validation_fold[rid],'code':text,'features_json':json.dumps(feature_values,sort_keys=True)})
 with OUT.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 report={'status':'completed_development_only_export','records':len(rows),'origins':len(set(r['origin_program'] for r in rows)),
   'folds':{str(f):sum(r['fold']==f for r in rows) for f in sorted(set(r['fold'] for r in rows))},
   'class_counts':{str(c):sum(r['y_true']==c for r in rows) for c in (0,1,2)},'raw_sha256':raw_sha,
   'cache_source_sha256':cache_manifest['source_sha256'],'feature_fields':len(META)+len(DIFF_NUM)+len(DIFF_CAT),
   'records_included_only_when_id_in_saved_development_sample':True,'frozen_test_accessed':False,
   'output':str(OUT.relative_to(ROOT)),'output_sha256':sha(OUT)}
 (OUT.with_suffix('.json')).write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
