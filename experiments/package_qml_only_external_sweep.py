"""Package only the unfinished QNN hyperparameter sweeps for Kaggle CPU."""
from __future__ import annotations
import base64,hashlib,json,subprocess,zipfile
from importlib.metadata import version
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'reports/experiments/qmutbench_multisource_qml_campaign_20261028'

def main():
    sources={
      C/'bugs4q_robust_execution/normalized_manifest.csv':'workspace/reports/experiments/qmutbench_multisource_qml_campaign_20261028/bugs4q_robust_execution/normalized_manifest.csv',
      C/'qsimbench_backend_fingerprint/normalized_manifest.csv':'workspace/reports/experiments/qmutbench_multisource_qml_campaign_20261028/qsimbench_backend_fingerprint/normalized_manifest.csv',
      C/'qmutbench_threeclass_compact.csv':'workspace/reports/experiments/qmutbench_multisource_qml_campaign_20261028/qmutbench_threeclass_compact.csv',
      ROOT/'reports/experiments/qmutbench_three_class_qnn_campaign_20261027/grouped_split_manifest.csv':'workspace/reports/experiments/qmutbench_three_class_qnn_campaign_20261027/grouped_split_manifest.csv',
      ROOT/'reports/experiments/qmutbench_three_class_qnn_campaign_20261027/features_by_fold.json':'workspace/reports/experiments/qmutbench_three_class_qnn_campaign_20261027/features_by_fold.json',
      ROOT/'experiments/train_resource_light_qnn_tasks.py':'workspace/experiments/train_resource_light_qnn_tasks.py',
      ROOT/'experiments/train_qmutbench_threeclass_light_qnn.py':'workspace/experiments/train_qmutbench_threeclass_light_qnn.py',
    }
    venv=ROOT/'.venv/Scripts/python.exe'
    versions=json.loads(subprocess.check_output([str(venv),'-c',"import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in ['numpy','psutil']}))"],text=True)) if venv.exists() else {n:version(n) for n in ('numpy','psutil')}
    env={'python':'3.12','packages':versions,'backend':'NumPy analytic statevector simulator; CPU only; no GPU/PennyLane dependency required',
         'source_hashes':{name:hashlib.sha256(path.read_bytes()).hexdigest() for path,name in sources.items()}}
    archive=C/'qml_only_external_sweep_bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path,name in sources.items():z.write(path,name)
        z.writestr('workspace/environment.json',json.dumps(env,indent=2))
    payload=base64.b64encode(archive.read_bytes()).decode('ascii')
    cell='''import base64, hashlib, json, os, pathlib, subprocess, sys, zipfile
payload = """PAYLOAD"""
archive=pathlib.Path('/kaggle/working/qml_only_external_sweep_bundle.zip');archive.write_bytes(base64.b64decode(payload))
base=pathlib.Path('/kaggle/working/qml_sweep_bundle')
with zipfile.ZipFile(archive) as z:z.extractall(base)
root=base/'workspace';env=json.loads((root/'environment.json').read_text())
for rel,want in env['source_hashes'].items():
    if hashlib.sha256((base/rel).read_bytes()).hexdigest()!=want:raise RuntimeError('Input hash mismatch: '+rel)
import numpy, psutil
print('QML CPU runtime:',numpy.__version__,psutil.__version__)
def run(command,log,environment):
    result=subprocess.run(command,cwd=root,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    log.parent.mkdir(parents=True,exist_ok=True);log.write_text(result.stdout,encoding='utf-8');print(result.stdout[-5000:]);return result.returncode
status={'binary_sweep':{},'qmutbench_threeclass_sweep':{},'frozen_test_accessed':False}
binary_env=os.environ.copy();binary_env.update({'LIGHT_QNN_SWEEP':'1','LIGHT_QNN_EPOCHS':'20','LIGHT_QNN_OUT_DIR':'qnn_external_sweep20'})
code=run([sys.executable,'experiments/train_resource_light_qnn_tasks.py'],pathlib.Path('/kaggle/working/binary_qnn_sweep.log'),binary_env)
status['binary_sweep']={'status':'completed' if code==0 else 'failed_or_stopped','return_code':code}
qmut_env=os.environ.copy();qmut_env.update({'QUT_SWEEP':'1','QUT_EPOCHS':'24','QUT_OUT_DIR':'qmutbench_threeclass_qnn_external_sweep'})
code2=run([sys.executable,'experiments/train_qmutbench_threeclass_light_qnn.py'],pathlib.Path('/kaggle/working/qmutbench_threeclass_sweep.log'),qmut_env)
status['qmutbench_threeclass_sweep']={'status':'completed' if code2==0 else 'failed_or_stopped','return_code':code2}
status_path=pathlib.Path('/kaggle/working/qml_sweep_execution_status.json');status_path.write_text(json.dumps(status,indent=2))
result=pathlib.Path('/kaggle/working/qml_only_sweep_results.zip')
with zipfile.ZipFile(result,'w',zipfile.ZIP_DEFLATED) as z:
    for name in ('bugs4q_robust_execution','qsimbench_backend_fingerprint'):
        out=root/'reports/experiments/qmutbench_multisource_qml_campaign_20261028'/name/'qnn_external_sweep20'
        if out.exists():
            for f in out.rglob('*'):
                if f.is_file():z.write(f,f.relative_to(root))
    out=root/'reports/experiments/qmutbench_multisource_qml_campaign_20261028/qmutbench_threeclass_qnn_external_sweep'
    if out.exists():
        for f in out.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(root))
    for log in (pathlib.Path('/kaggle/working/binary_qnn_sweep.log'),pathlib.Path('/kaggle/working/qmutbench_threeclass_sweep.log'),status_path):
        if log.exists():z.write(log,log.name)
print('Result archive:',result);print(status_path.read_text())
'''.replace('PAYLOAD',payload)
    notebook={'cells':[{'cell_type':'markdown','metadata':{},'source':['# QML-only QNN optimization sweeps\n','CPU statevector QNN training only: grouped Bugs4Q-Robust and QSimBench architecture/learning-rate sweeps, plus QMutBench three-class retraining. No classical suite and no frozen-test records are included. Click Run All, then download qml_only_sweep_results.zip.\n']},
      {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':cell.splitlines(keepends=True)}],
      'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12.0'}},'nbformat':4,'nbformat_minor':5}
    nb=C/'QML_Only_External_Sweeps.ipynb';nb.write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    print(json.dumps({'notebook':str(nb.relative_to(ROOT)),'bundle':str(archive.relative_to(ROOT)),'bytes':archive.stat().st_size,'environment':env},indent=2))

if __name__=='__main__':main()
