# QMutBench three-class QNN confirmation (Kaggle-ready bundle)

This is a reproducible, development-only export and confirmation workflow. It does not include source rows, credentials, frozen-test records, or API tokens. The local source repository declares GPL-3.0; respect that license when preparing or sharing any derived dataset.

## Data and roles

From the repository root, use `qmutbench_qml/training/export_confirmation_data.py --output <private-output-directory>` when the machine has sufficient memory. It emits sampled development-only `train.csv`, `validation.csv`, and an integrity manifest. The fixed design uses the existing 12-origin exploration sample for fitting and the 36 predeclared OPO-bearing development origins for confirmation. Feature selection and scaling are fit on training rows only. The export includes only the 23 existing model inputs, the three-class label, record ID, and group ID. Do not upload the full source dataset or frozen-test data.

Create a **private Kaggle Dataset** from those three export files and a separate code dataset containing this package and the `qmutbench_qml` source package. In a Kaggle notebook, attach both datasets, select a GPU accelerator if available, then run:

```bash
python /kaggle/input/<code-dataset>/qmutbench_qml/kaggle_confirmation/backend_smoke.py
python /kaggle/input/<code-dataset>/qmutbench_qml/kaggle_confirmation/run_confirmation.py \
  --data /kaggle/input/<private-data-dataset> --output /kaggle/working/qmutbench-confirmation
```

Install `qmutbench_qml/requirements-lock.txt` in the notebook if PennyLane is not already present. Kaggle Internet access may be needed for pip installation. No Kaggle API key is needed for the notebook UI workflow. Uploading private data and enabling an accelerator are account/UI steps and are not performed by these scripts.

## Backend choice and interpretation

`backend_smoke.py` compares the same deterministic forward pass and parameter gradient on `default.qubit`, `lightning.qubit`, and `lightning.gpu` when each plugin is available. It records timing, peak process RSS, device details, and numerical agreement. The GPU smoke uses the plugin's adjoint differentiation path; the selected backend is usable only if both forward and gradient checks pass. PennyLane documents Lightning-GPU as a cuQuantum-backed NVIDIA simulator and supports circuit gradients through adjoint differentiation; installation depends on the Kaggle CUDA/cuQuantum environment, so follow the [official installation guide](https://docs.pennylane.ai/projects/lightning/en/stable/dev/installation.html). `run_confirmation.py` uses `QMB_DEVICE` only if the smoke report validates that backend; otherwise it refuses to fit. For GPU, set `QMB_DEVICE=lightning.gpu` after the smoke passes. A successful GPU execution is not evidence of quantum advantage; this is simulator execution.

The confirmation sample is balanced within rare-class-bearing origins and is not representative of natural prevalence. This confirmation was predeclared after the 12-origin exploration; do not tune on its scores or use it repeatedly to select models. It is not the frozen test set.
