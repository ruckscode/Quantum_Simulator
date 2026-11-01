# Repository structure and contribution guide

This repository contains the simulator product, classical ML research, and a separate QML research track. The simulator package is the default user-facing product. Research models and datasets do not silently enter the simulator diagnosis path.

## Main folders

```text
.
├── src/quantum_simulator/       Installable simulator and diagnosis package
├── examples/                    Small runnable simulator examples
├── ui/                          Optional local Streamlit user interface
├── tests/                       Simulator and integration checks
├── qmutbench_ml/                Classical ML training, evaluation, and inference
├── qmutbench_qml/               Separate experimental QML code and confirmations
├── data/
│   ├── classical_ml_v1/         Tracked source/task registry and grouped OOF scores
│   └── curated_qml_training_v1/ Small curation bundle and recorded benchmark summaries
├── reports/                     Reviewed reports and project documentation
│   └── experiments/             Local generated run artifacts; not committed by default
├── docs/                        Architecture and repository navigation
└── *.py                         Legacy compatibility entry points and scripts
```

## Where to start

- Simulator API and installation: [`README.md`](../README.md).
- End-to-end module/model-use map: [`project_architecture.md`](project_architecture.md).
- Diagnosis equations and actual algorithm stages: [`algorithm_reference.md`](algorithm_reference.md).
- Classical source/task provenance and scores: [`data/classical_ml_v1/README.md`](../data/classical_ml_v1/README.md).
- Classical QMutBench workflow and model use: [`qmutbench_ml/README.md`](../qmutbench_ml/README.md).
- QML result scope, comparisons, and limitations: [`data/curated_qml_training_v1/README.md`](../data/curated_qml_training_v1/README.md) and [`matched_qml_classical_comparison.md`](../data/curated_qml_training_v1/matched_qml_classical_comparison.md).
- Research status, evidence, and source-specific reports: [`reports/`](../reports/).

## Data and experiment files

Downloaded source archives, full raw datasets, processed caches, per-run predictions, and training checkpoints remain local. `.gitignore` excludes `data/external_quantum_datasets/`, `data/processed/`, and new files under `reports/experiments/`. This avoids large or incomplete run outputs overwhelming the source repository. Small curated datasets and summaries may be committed when their provenance, task definition, and limitations are documented.

For a reproducible experiment, commit the code, configuration, compact input manifest, dependency/version details, and a concise validated summary. Keep full checkpoints and large row-level dumps in the experiment workspace or an explicitly versioned artifact store. Do not present a sampled development result as frozen-test performance.

## Integration boundary

`qmutbench_ml/` and `qmutbench_qml/` are research tracks. Their models are not automatically invoked by `diagnose_circuit()`. Integration requires a versioned feature adapter and a separate end-to-end evaluation against diagnosis labels; benchmark scores on mutation-outcome datasets do not establish circuit-diagnosis accuracy.

## Tracked summaries vs local training inputs

`data/classical_ml_v1/` preserves the source/task registry and outer-fold
metrics for the completed classical multi-source search. The raw data,
processed tables, per-row predictions, and experiment logs remain local and
ignored. A collaborator can understand what was trained from the registry, but
must reacquire permitted inputs to reproduce training. QML summaries and its
small curated data bundle are in `data/curated_qml_training_v1/`; the long
campaign and GPU run limitations are recorded there.
