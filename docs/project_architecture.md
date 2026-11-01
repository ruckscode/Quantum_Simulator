# Project architecture and model-use map

## Project purpose

The simulator accepts a quantum circuit, runs the ideal statevector model or a
configured noise model, compares expected and observed outcomes, then reports
normal behavior, a supported program fault, a supported hardware anomaly, or an
abstention when the evidence is incomplete/conflicting. An optional IBM path
uses user-provided credentials for a real backend. Classical ML and QML are
research tracks; they are part of the project, but are not automatically wired
into this diagnosis path unless they have compatible labels, features, saved
artifacts, and end-to-end validation.

## Structure

| Path | Role | Importance |
|---|---|---|
| `src/quantum_simulator/` | Installable statevector, circuit, statistical comparison, program and hardware localization, optional ML hook | Product runtime; default diagnosis code |
| `ibm_validation.py` | Optional IBM Runtime preflight, calibration capture, transpilation, execution, and count diagnosis | Real-hardware adapter; requires optional dependencies, key, backend and access |
| `ui/` | Optional local Streamlit interface | Normal and fault scenarios plus masked, explicit IBM key entry; not required by the Python API |
| `data/classical_ml_v1/` | Tracked compact source/task registry and completed classical OOF scores | Lets collaborators understand source semantics and compare task-appropriate results |
| `data/curated_qml_training_v1/` | Tracked small QML/classical task data and benchmark summaries | Reproducible research evidence for bounded QML comparisons |
| `qmutbench_ml/` | Historical mutation-outcome model training and inference code | Separate model task: predicts mutation survival/outcome |
| `qmutbench_qml/` | Quantum neural network implementations, feature preparation and training workflows | Experimental QML track; not used by simulator diagnosis |
| `data/external_quantum_datasets/` | Local-only third-party archives, Git checkouts, and inventory | Source provenance; generally excluded from Git due size/licensing |
| `data/processed/` | Local-only cleaned task tables, manifests, folds and partitions | Rebuild/training inputs; excluded from Git by design |
| `experiments/` | Experiment drivers and reproducible benchmark runners | Connects research procedures to result records |
| `docs/` | Algorithm equations, module roles, limitations and navigation | Explains what runs, when, and why |
| `reports/experiments/` | Local run-level logs, OOF predictions, checkpoints, and incomplete campaigns | Audit detail retained locally; ignored so partial/large output is not accidentally committed |

## What happens during a normal simulator call

1. `QuantumCircuit` or an operation list describes the circuit.
2. `diagnose_circuit()` samples the reference ideally. If there is no separate
   reference, the submitted circuit is the reference (a smoke-style
   self-comparison; it cannot detect a wrong intended algorithm).
3. The submitted circuit is sampled ideally or with the explicitly supplied
   `HardwareModel`.
4. `count_statistics.py` tests finite counts using permutation/bootstrap when
   counts are available. `statistics.py` supplies TVD, Hellinger, JSD and the
   legacy finite-shot threshold for probability-only calls.
5. `fault_localizer.py` compares normalized source operations and measured
   output impact. `hardware_fault_localizer.py` checks circuit-relevant
   calibration evidence.
6. `diagnosis_engine.py` combines the independent evidence and can abstain.
7. `explain_diagnosis()` returns the algorithm, equation, and run-specific
   values for a UI/report.

With no hardware model, hardware cause attribution is limited. A synthetic
model exercises the simplified simulator noise model; it does not validate a
physical device. A real IBM job runs only through the explicit IBM adapter and
may consume backend quota.

## Which learned model is used at which point?

| Model/data track | Learned target | Used by default `diagnose_circuit()`? | Proper current role |
|---|---|---:|---|
| Source-specific classical ML (nine adapters + compatible three-source pooled task) | Each dataset's own code/comment/detector/harness/backend target | No | Reported grouped nested-OOF research baselines; not a common diagnosis model |
| QMutBench classical ExtraTrees | Mutation Survivor vs WOO/OPO outcome | No | Mutation-outcome benchmark; inference weights are absent from this checkout |
| QMutBench/QBugs/Qolumbina QNNs | Mutation outcome, circuit snapshot correctness, or seeded-mutant detector outcome depending on task | No | QML research comparisons; no deployable arbitrary-circuit diagnosis model |
| Observable Logistic Regression candidate | Three-way diagnosis label from five distribution observables | No, only if caller passes `classifier=` | Optional development candidate, uncalibrated, deployment validation not performed |
| Rule/statistics/localization pipeline | Statistical anomaly plus source/calibration evidence | Yes | Current operational diagnosis path; accuracy claim limited to the documented synthetic/control benchmark |

Scores from different targets must not be combined into a single overall
accuracy. The source-specific classical report is `data/classical_ml_v1/`.
The QML result definitions and caveats are in
`data/curated_qml_training_v1/README.md` and `training_results.json`.
The wheel built from `pyproject.toml` packages `quantum_simulator*` and the
optional root IBM adapter; `qmutbench_ml/` and `qmutbench_qml/` remain research
code in the repository and are not installed as part of the simulator wheel.

## Classical ML status and preservation

The source-specific nested search produced outer grouped OOF metrics for each
task. Its concise result table is tracked in
`data/classical_ml_v1/benchmark_results.csv`; task counts/labels and data
availability are in `source_registry.csv`. The input rows and per-row
predictions remain local because third-party data rights and size make raw
republication inappropriate. For rebuilding, download the appropriate upstream
corpus and follow local manifests/experiment configuration; a fresh clone will
not contain these raw inputs or ignored experiment predictions.

QMutBench has a stronger large development benchmark (reported 79.33% accuracy,
79.33% macro-F1, 79.41% balanced accuracy on 578,117 development rows grouped
by 305 origins). It is not diagnosis accuracy, not frozen-test performance,
and its inference model bundle is absent from the current checkout. The smaller
four-feature QMutBench result is a feature-matched QNN comparator, not the full
classical result.

## QML role and boundaries

The project retains QNN implementations, trained sample/OOF outputs and matched
classical comparators. The bounded QMutBench four-feature sample had QNN
accuracy around 57.7% (2,560 rows) and 59.25% (10,240 rows), versus matched
ExtraTrees 57.11% and 59.35%. Separate QBugs and Qolumbina tasks have their own
targets and metrics. The long original QNN campaign is recorded incomplete;
the GPU sample is a research result with an audit caveat. These artifacts are
useful for research comparison, but no QNN is a diagnosis plug-in. Integration
would require a diagnosis-labeled training set, an inference feature contract,
saved portable weights, grouped external validation, and evidence-fusion rules.

The recorded synthetic diagnosis control reports 80/80 generated cases across
four injected categories and two small circuit templates. This shows that the
controlled pipeline behaved as expected on those constructed cases; it is not
100% real-world accuracy and does not establish physical-hardware accuracy.

## IBM credential boundary

`execute_on_ibm_backend(..., token=...)` receives a token explicitly and the
adapter does not call an account-save function. Credentials are not part of
calibration snapshots or benchmark registries. Any optional UI must keep the
token only in process memory, mask it, avoid files/environment/URLs/logs, and
offer explicit clearing. A remote hosted UI necessarily receives the secret
on its server; use a trusted local deployment for private API keys.

## Reproducibility and interpretation

- Keep task-specific target meanings intact; never merge outcomes solely
  because the labels look binary.
- Use group-safe outer folds and fit all learned preprocessing in training
  folds; keep frozen test data untouched.
- Preserve exact configurations, seed, package versions, source revision,
  source hashes, and row/group counts with every new result.
- Treat simulated synthetic fault accuracy as a controlled pipeline check,
  not independent proof of real-hardware accuracy.
- Review the equations and assumptions in
  [`algorithm_reference.md`](algorithm_reference.md).
