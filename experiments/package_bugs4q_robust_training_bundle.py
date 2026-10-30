"""Package the verified Bugs4Q-Robust task for an optional high-memory runner."""
from __future__ import annotations

import json
import base64
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028"
TASK = CAMPAIGN / "bugs4q_robust_execution"
BUNDLE = CAMPAIGN / "external_training_bundle_bugs4q_robust"


def main() -> None:
    import numpy, pandas, pennylane, psutil, sklearn
    BUNDLE.mkdir(parents=True, exist_ok=True)
    files = {
        TASK / "normalized_manifest.csv": "workspace/reports/bugs4q_robust_execution/normalized_manifest.csv",
        TASK / "task_definition.json": "workspace/reports/bugs4q_robust_execution/task_definition.json",
        CAMPAIGN / "qsimbench_backend_fingerprint/normalized_manifest.csv": "workspace/reports/qsimbench_backend_fingerprint/normalized_manifest.csv",
        CAMPAIGN / "qsimbench_backend_fingerprint/task_definition.json": "workspace/reports/qsimbench_backend_fingerprint/task_definition.json",
        ROOT / "experiments/bugs4q_qnn_grouped_benchmark.py": "workspace/experiments/bugs4q_qnn_grouped_benchmark.py",
        ROOT / "experiments/finalize_bugs4q_qnn_grouped_benchmark.py": "workspace/experiments/finalize_bugs4q_qnn_grouped_benchmark.py",
    }
    environment = {
        "python_target": ">=3.10,<3.13",
        "packages": {"numpy": numpy.__version__, "pandas": pandas.__version__,
                     "pennylane": pennylane.__version__, "psutil": psutil.__version__,
                     "scikit-learn": sklearn.__version__},
        "backend": "PennyLane default.qubit CPU statevector with analytic backpropagation",
        "device": "CPU; no GPU is required or used",
    }
    (BUNDLE / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    commands = r"""# Run from the extracted bundle root on Linux (Kaggle or equivalent).
python -m pip install pennylane=={pennylane} scikit-learn=={sklearn} pandas=={pandas} psutil=={psutil}
export BUGS4Q_QNN_INPUT="$PWD/workspace/reports/bugs4q_robust_execution/normalized_manifest.csv"
export BUGS4Q_QNN_OUTPUT="$PWD/output/bugs4q_robust_execution/qnn"
export BUGS4Q_TASK="Bugs4Q-Robust pass/fail prediction from the test harness and program snapshot"
export QNN_DATA_DESCRIPTION="2,220 program-run observations from 37 issue groups"
export QNN_GROUP_SPLIT_DESCRIPTION="5-fold GroupKFold by issue id; all snapshots and runs for an issue stay together"
export QNN_BATCH_ROWS=1 QNN_VALIDATION_TRACK_ROWS=16 QNN_TFIDF_MAX_FEATURES=512 QNN_SVD_ITERATIONS=2
mkdir -p "$BUGS4Q_QNN_OUTPUT"
python workspace/experiments/bugs4q_qnn_grouped_benchmark.py
python workspace/experiments/finalize_bugs4q_qnn_grouped_benchmark.py
"""
    commands = commands.format(pennylane=pennylane.__version__, sklearn=sklearn.__version__,
                               pandas=pandas.__version__, psutil=psutil.__version__)
    (BUNDLE / "RUN.md").write_text(commands, encoding="utf-8")
    archive = CAMPAIGN / "remaining_source_qnn_training_bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path, relative in files.items():
            z.write(path, relative)
        z.write(BUNDLE / "environment.json", "workspace/environment.json")
        z.write(BUNDLE / "RUN.md", "workspace/RUN.md")
    payload = base64.b64encode(archive.read_bytes()).decode("ascii")
    code = f'''import base64, os, pathlib, subprocess, zipfile
payload = """{payload}"""
bundle = pathlib.Path("/kaggle/working/remaining_source_qnn_training_bundle.zip")
bundle.write_bytes(base64.b64decode(payload))
with zipfile.ZipFile(bundle) as z:
    z.extractall("/kaggle/working/remaining_source_qnn_bundle")
root = pathlib.Path("/kaggle/working/remaining_source_qnn_bundle/workspace")
environment = __import__("json").loads((root / "environment.json").read_text())
packages = environment["packages"]
subprocess.check_call(["python", "-m", "pip", "install",
    f"pennylane=={{packages['pennylane']}}", f"scikit-learn=={{packages['scikit-learn']}}",
    f"pandas=={{packages['pandas']}}", f"psutil=={{packages['psutil']}}"])
os.environ.update({{
    "BUGS4Q_QNN_INPUT": str(root / "reports/bugs4q_robust_execution/normalized_manifest.csv"),
    "BUGS4Q_QNN_OUTPUT": "/kaggle/working/bugs4q_robust_execution/qnn",
    "BUGS4Q_TASK": "Bugs4Q-Robust pass/fail prediction from the test harness and program snapshot",
    "QNN_DATA_DESCRIPTION": "2,220 program-run observations from 37 issue groups",
    "QNN_GROUP_SPLIT_DESCRIPTION": "5-fold GroupKFold by issue id; all snapshots and runs for an issue stay together",
    "QNN_BATCH_ROWS": "1", "QNN_VALIDATION_TRACK_ROWS": "16",
    "QNN_TFIDF_MAX_FEATURES": "512", "QNN_SVD_ITERATIONS": "2",
    "PYTHONWARNINGS": "ignore"}})
subprocess.check_call(["python", str(root / "experiments/bugs4q_qnn_grouped_benchmark.py")])
subprocess.check_call(["python", str(root / "experiments/finalize_bugs4q_qnn_grouped_benchmark.py")])
out = pathlib.Path(os.environ["BUGS4Q_QNN_OUTPUT"])
bundle_out = pathlib.Path("/kaggle/working/bugs4q_robust_execution_results.zip")
with zipfile.ZipFile(bundle_out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in out.rglob("*"):
        if p.is_file(): z.write(p, p.relative_to(out.parent))
print("Training completed. Download artifact:", bundle_out)
print((out / "summary.json").read_text())
'''
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# Bugs4Q-Robust supervised QNN training\n",
                "Run this notebook with **Run All**. It embeds the verified task bundle, installs the recorded CPU backend dependencies, trains five issue-group folds, and writes a downloadable ZIP of checkpoints and evaluation artifacts.\n",
                "The frozen QMutBench test is not used."
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12.0"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path = CAMPAIGN / "Bugs4Q_Robust_Kaggle_Training.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(json.dumps({"bundle": str(archive.relative_to(ROOT)), "bytes": archive.stat().st_size,
                      "notebook": str(notebook_path.relative_to(ROOT)),
                      "environment": environment}, indent=2))


if __name__ == "__main__":
    main()
