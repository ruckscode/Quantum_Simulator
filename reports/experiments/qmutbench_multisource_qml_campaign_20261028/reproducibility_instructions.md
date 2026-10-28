# Reproduction

Run the Bugs4Q classical standalone stage:

```powershell
.\.venv\Scripts\python.exe experiments/bugs4q_standalone_benchmark.py
```

It reads the checked-in local Bugs4Q framework extraction, keeps only Qiskit issues with paired buggy/fixed source, issue metadata, and both corresponding trigger-test files, strips comments and docstrings while retaining executable string literals, and executes five-fold `GroupKFold` by issue ID. TF-IDF, SVD, and scaling are fitted only inside each training fold. It saves OOF predictions, group metrics, and a checkpoint of the strongest grouped-OOF candidate, retrained on all eligible cases after CV.

The development-only QMutBench ensemble audit can be reproduced with:

```powershell
.\.venv\Scripts\python.exe experiments/qmutbench_multisource_ensemble_audit.py
```

The QMutBench 36-origin exporter is intentionally gated at 900 MiB pre-import headroom; do not lower that guard. It stopped at 792 MiB and loaded no source records. Bugs4Q QNN fitting is separately gated at 700 MiB and stopped at 684 MiB. Resume either only after materially higher available memory or use a verified remote environment with the same source and split manifests.

The Bugs4Q-Robust replication commit is recorded in `dataset_provenance.json`. Its result CSVs can be independently checked using Python's standard `csv` module. Subject-program source terms remain upstream; do not redistribute those files under the replication package's MIT license.
