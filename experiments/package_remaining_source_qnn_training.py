"""Create one Kaggle notebook for remaining QNNs and broad classical CV."""
from __future__ import annotations

import base64
import hashlib
import json
from importlib.metadata import version
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028"


def main() -> None:
    items = {
        CAMPAIGN / "bugs4q_standalone/normalized_manifest.csv": "workspace/reports/bugs4q_standalone/normalized_manifest.csv",
        CAMPAIGN / "bugs4q_na/normalized_manifest.csv": "workspace/reports/bugs4q_na/normalized_manifest.csv",
        CAMPAIGN / "bugs4q_joint/normalized_manifest.csv": "workspace/reports/bugs4q_joint/normalized_manifest.csv",
        CAMPAIGN / "qbugs/normalized_manifest.csv": "workspace/reports/qbugs/normalized_manifest.csv",
        CAMPAIGN / "quantum_platform_223_comment_only/normalized_manifest.csv": "workspace/reports/quantum_platform_223_comment_only/normalized_manifest.csv",
        CAMPAIGN / "qolumbina_detection_ideal/normalized_manifest.csv": "workspace/reports/qolumbina_detection_ideal/normalized_manifest.csv",
        CAMPAIGN / "lintq_warning_triage/normalized_manifest.csv": "workspace/reports/lintq_warning_triage/normalized_manifest.csv",
        CAMPAIGN / "bugs4q_robust_execution/normalized_manifest.csv": "workspace/reports/bugs4q_robust_execution/normalized_manifest.csv",
        CAMPAIGN / "bugs4q_robust_execution/task_definition.json": "workspace/reports/bugs4q_robust_execution/task_definition.json",
        CAMPAIGN / "qsimbench_backend_fingerprint/normalized_manifest.csv": "workspace/reports/qsimbench_backend_fingerprint/normalized_manifest.csv",
        CAMPAIGN / "qsimbench_backend_fingerprint/task_definition.json": "workspace/reports/qsimbench_backend_fingerprint/task_definition.json",
        ROOT / "experiments/bugs4q_qnn_grouped_benchmark.py": "workspace/experiments/bugs4q_qnn_grouped_benchmark.py",
        ROOT / "experiments/finalize_bugs4q_qnn_grouped_benchmark.py": "workspace/experiments/finalize_bugs4q_qnn_grouped_benchmark.py",
        ROOT / "experiments/run_grouped_classical_suite.py": "workspace/experiments/run_grouped_classical_suite.py",
    }
    venv_python = ROOT / ".venv/Scripts/python.exe"
    if venv_python.exists():
        versions = json.loads(subprocess.check_output([str(venv_python), "-c",
            "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ['numpy','pandas','pennylane','psutil','scikit-learn']}))"], text=True))
    else:
        versions = {name: version(name) for name in ("numpy", "pandas", "pennylane", "psutil", "scikit-learn")}
    env = {"python": "3.12", "packages": {**versions},
           "backend": "CPU PennyLane default.qubit analytic backpropagation; GPU not required",
           "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for path, name in items.items()}}
    archive = CAMPAIGN / "remaining_source_qnn_training_bundle.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path, name in items.items():
            z.write(path, name)
        z.writestr("workspace/environment.json", json.dumps(env, indent=2))
    payload = base64.b64encode(archive.read_bytes()).decode("ascii")
    cell = '''import base64, hashlib, json, os, pathlib, subprocess, zipfile
payload = """PAYLOAD"""
archive = pathlib.Path("/kaggle/working/remaining_source_qnn_training_bundle.zip")
archive.write_bytes(base64.b64decode(payload))
root_dir = pathlib.Path("/kaggle/working/remaining_source_qnn_bundle")
with zipfile.ZipFile(archive) as z: z.extractall(root_dir)
root = root_dir / "workspace"
env = json.loads((root / "environment.json").read_text())
for relative_path, expected in env["sha256"].items():
    actual = hashlib.sha256((root_dir / relative_path).read_bytes()).hexdigest()
    if actual != expected: raise RuntimeError("Bundle input hash mismatch: " + relative_path)
p = env["packages"]
subprocess.check_call(["python", "-m", "pip", "install", f"numpy=={p['numpy']}", f"pennylane=={p['pennylane']}",
    f"pandas=={p['pandas']}", f"scikit-learn=={p['scikit-learn']}", f"psutil=={p['psutil']}"])
def run_logged(command, path):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log: log.write(result.stdout)
    print(result.stdout[-6000:])
    return result.returncode
tasks = [
 {"name":"qsimbench_backend_fingerprint", "input":"qsimbench_backend_fingerprint/normalized_manifest.csv",
  "task":"QSimBench Aer versus fake_kyiv output fingerprinting", "data":"300 runs from 150 algorithm-size groups",
  "split":"5-fold GroupKFold by algorithm-size setting"},
 {"name":"bugs4q_robust_execution", "input":"bugs4q_robust_execution/normalized_manifest.csv",
  "task":"Bugs4Q-Robust test pass/fail from program and harness", "data":"2,220 observations from 37 issues",
  "split":"5-fold GroupKFold by issue id; all repeated runs and states grouped"}
]
outputs = []
execution_status = {"qnn_tasks":[],"classical_suite":"not_started"}
for task in tasks:
    out = pathlib.Path("/kaggle/working") / task["name"] / "qnn"
    os.environ.update({"BUGS4Q_QNN_INPUT":str(root / "reports" / task["input"]),
      "BUGS4Q_QNN_OUTPUT":str(out), "BUGS4Q_TASK":task["task"],
      "QNN_DATA_DESCRIPTION":task["data"], "QNN_GROUP_SPLIT_DESCRIPTION":task["split"],
      "QNN_BATCH_ROWS":"1", "QNN_VALIDATION_TRACK_ROWS":"8",
      "QNN_TFIDF_MAX_FEATURES":"512", "QNN_SVD_ITERATIONS":"2", "PYTHONWARNINGS":"ignore"})
    log_path = out / "execution.log"
    code = run_logged(["python",str(root / "experiments/bugs4q_qnn_grouped_benchmark.py")], log_path)
    if code == 0: code = run_logged(["python",str(root / "experiments/finalize_bugs4q_qnn_grouped_benchmark.py")], log_path)
    execution_status["qnn_tasks"].append({"task":task["name"],"status":"completed" if code == 0 else "failed_or_stopped",
        "return_code":code,"log":str(log_path)})
    outputs.append(out)
classical_tasks = [
 {"name":"bugs4q_core", "input":str(root / "reports/bugs4q_standalone/normalized_manifest.csv")},
 {"name":"bugs4q_na", "input":str(root / "reports/bugs4q_na/normalized_manifest.csv")},
 {"name":"bugs4q_pooled", "input":str(root / "reports/bugs4q_joint/normalized_manifest.csv")},
 {"name":"qbugs", "input":str(root / "reports/qbugs/normalized_manifest.csv")},
 {"name":"quantum_platform_comments", "input":str(root / "reports/quantum_platform_223_comment_only/normalized_manifest.csv")},
 {"name":"qolumbina_ideal", "input":str(root / "reports/qolumbina_detection_ideal/normalized_manifest.csv")},
 {"name":"lintq_warning_triage", "input":str(root / "reports/lintq_warning_triage/normalized_manifest.csv")},
 {"name":"bugs4q_robust_execution", "input":str(root / "reports/bugs4q_robust_execution/normalized_manifest.csv")},
 {"name":"qsimbench_backend_fingerprint", "input":str(root / "reports/qsimbench_backend_fingerprint/normalized_manifest.csv")}
]
classical_out = pathlib.Path("/kaggle/working/classical_suite")
os.environ["QML_CLASSICAL_TASKS_JSON"] = json.dumps(classical_tasks)
os.environ["QML_CLASSICAL_OUTPUT"] = str(classical_out)
classical_log = classical_out / "execution.log"
classical_code = run_logged(["python",str(root / "experiments/run_grouped_classical_suite.py")], classical_log)
execution_status["classical_suite"] = "completed" if classical_code == 0 else {"status":"failed_or_stopped","return_code":classical_code,"log":str(classical_log)}
outputs.append(classical_out)
status_path = pathlib.Path("/kaggle/working/execution_status.json")
status_path.write_text(json.dumps(execution_status,indent=2))
result = pathlib.Path("/kaggle/working/remaining_source_qnn_results.zip")
with zipfile.ZipFile(result,"w",zipfile.ZIP_DEFLATED) as z:
    for out in outputs:
        for f in out.rglob("*"):
            if f.is_file(): z.write(f,f.relative_to(pathlib.Path("/kaggle/working")))
    z.write(status_path,status_path.name)
print("Completed; download:",result)
print(status_path.read_text())
for out in outputs:
    summary = out / "summary.json"
    if summary.exists(): print(summary.read_text())
    else: print("No completed summary:", out)
'''.replace("PAYLOAD", payload)
    notebook = {"cells": [
        {"cell_type":"markdown", "metadata":{}, "source":[
            "# Remaining grouped QNN and classical benchmark\n",
            "Open this notebook in Kaggle and choose Run All once. It trains the two pending QNN tasks and runs a broad matched classical model and imbalance-method suite over nine binary task manifests. The classical suite uses training-fold-only SVD, oversampling, undersampling, and SMOTE in a continuous four-feature space. No QMutBench frozen-test records are included.\n"]},
        {"cell_type":"code", "execution_count":None, "metadata":{}, "outputs":[], "source":cell.splitlines(keepends=True)}],
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                    "language_info":{"name":"python","version":"3.12.0"}}, "nbformat":4,"nbformat_minor":5}
    path = CAMPAIGN / "Remaining_Source_QNN_Training.ipynb"
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(json.dumps({"notebook":str(path.relative_to(ROOT)),"bundle":str(archive.relative_to(ROOT)),
        "bundle_bytes":archive.stat().st_size,"environment":env},indent=2))


if __name__ == "__main__":
    main()
