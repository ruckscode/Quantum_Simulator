# QMutBench outcome prediction: grouped ML experiment

**Status:** bounded full-metadata training/evaluation completed on 2026-10-25. The 85% test-accuracy target was **not achieved**. This is an outcome-prediction experiment, not hardware diagnosis.

## Data audit and target

The selected source is the read-only public QMutBench metadata CSV `data/qmutbench_ingestion/raw/official_all_data_20261015.csv`, retrieved from the official endpoint and associated in the local source manifest with QMutBench repository commit `01f5e7700c051fe715fc368b307deaa7e6869ba8`. It is 59,913,911 bytes, SHA-256 `3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778`, and contains 723,079 rows, 12 columns, 375 `Origin_program` groups, 28 algorithms, 12 algorithm groups, and 2–30 qubits. The source repository is GPL-3.0; its cited commit was reported unsigned in the acquisition record. The official QMutBench paper describes 375 originals and 723,079 mutants ([paper](https://arxiv.org/abs/2604.15870), [repository](https://github.com/EnautMendi/QMutBench)).

There are no missing values. Full-row duplicates: 0. Duplicate rows ignoring source index and parent ID: 0. Repeated exact profiles across the registered seven model inputs: 0, including no profile crossing parent groups and no profile with conflicting labels. Near-identical circuit structures cannot be audited for most rows because the local QASM acquisition is only a 24-file, one-original-group subset; the CSV contains no mutant file path or mutant identifier. The complete metadata table is locally available; the complete source/mutant circuit corpus is not.

The raw `Killed` values are `WOO` (368,568), `OPO` (6,212), and `Survivor` (348,299). The prediction target is **recorded mutant detection by either oracle versus survival**: WOO and OPO map to detected (1), and Survivor maps to not detected (0). The original mutation-study definitions describe WOO as wrong-output detection, OPO as detection of a probability-distribution change using a chi-square test at significance level 0.01, and a mutant as surviving if neither oracle detects it ([primary study](https://link.springer.com/article/10.1007/s10664-025-10643-z)). This is a prediction of the historical benchmark outcome. It does not predict hardware noise, source-level localization, or behavior under the current simulator. OPO and WOO are combined only because both mean “detected” for this binary task; no three-class claim is made.

## Features and leakage controls

Registered input features, all available as circuit/mutation metadata before the recorded oracle outcome:

- Categorical: `algorithm`, `operation`, `gate`, `dominant_state`.
- Numeric: `qubits`, `gates`, `Position_int`.

`Killed` is the target and never enters the feature transformer. `Origin_program` is used only as the split group. `Unnamed: 0` is an acquisition row index. `algorithm_group` duplicates a coarser algorithm grouping and `position` bins `Position_int`; both are excluded. No simulator-derived observations or prior diagnosis fields are used. Mutation operator is intentionally an input because this task estimates historical detection outcome for a described mutation; the result must not be represented as inference from circuit behavior alone.

## Split and model comparison

Seed: `20261025`. A deterministic `StratifiedGroupKFold` partition holds out 70 whole `Origin_program` groups (144,962 rows); 305 groups (578,117 rows) are used for model selection and fitting. The train/test group intersection is empty. The remaining 305 development groups are evaluated with three stratified group folds. Preprocessing (median numeric imputation, categorical most-frequent imputation, one-hot encoding, and logistic scaling) is inside each sklearn pipeline and is fit on each fold's training partition only. No hyperparameter search or final-test threshold selection was performed. Candidate selection uses mean grouped-CV macro F1.

| Model | Mean group-CV accuracy | Mean balanced accuracy | Mean macro F1 |
|---|---:|---:|---:|
| Majority baseline | 51.84% | 50.00% | 34.14% |
| Logistic regression, class-weighted | 69.26% | 69.35% | 69.26% |
| Decision tree, fixed depth/leaf limits | 70.80% | 70.76% | 70.62% |
| Random forest, 40 trees, fixed depth/leaf limits | **73.41%** | **73.57%** | **73.40%** |

Random forest was selected by grouped-CV macro F1. A development-only random-row split scored 79.50% accuracy / 79.50% macro F1, but all 305 parent IDs appeared on both sides. It is a leakage-prone diagnostic, not validation evidence. The fixed group-held-out test is lower and has zero parent-ID overlap. The random and grouped estimates have different split composition and training sizes, so their difference is descriptive rather than a pure causal estimate of leakage.

The held-out group test accuracy is **77.65%**, balanced accuracy **77.78%**, and macro F1 **77.64%**. The 85% target was not met. Parent-group bootstrap 95% percentile intervals are 74.64–80.72% for accuracy, 74.83–80.68% for balanced accuracy, and 74.58–80.58% for macro F1 (1,000 resamples of the 70 test groups). These groups are `Origin_program` IDs; independent MQT recipe-root ancestry is unresolved, so this is not a verified unseen-recipe-family test.

| Actual \ predicted | Survivor | Detected (WOO/OPO) |
|---|---:|---:|
| Survivor | 57,023 | 12,863 |
| Detected (WOO/OPO) | 19,543 | 55,533 |

Held-out per-class metrics:

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Survivor | 69,886 | 74.48% | 81.59% | 77.87% |
| Detected (WOO/OPO) | 75,076 | 81.19% | 73.97% | 77.41% |

Mean unweighted per-`Origin_program` accuracy across the 70 held-out origins was 80.37% (median 82.13%, range 58.79–98.76%; 10th–90th percentiles 62.63–94.45%). It is descriptive and weights each parent equally, unlike the row-pooled headline metrics.

![Grouped cross-validation model comparison](grouped_cv_macro_f1.svg)

![Held-out test confusion matrix](heldout_confusion_matrix.svg)

## Reproduction and artifacts

From the repository root with the existing environment:

```powershell
.\.venv\Scripts\python.exe -m experiments.qmutbench_outcome_ml_audit
.\.venv\Scripts\python.exe -m experiments.qmutbench_outcome_ml smoke --limit 30000 --output-dir data/qmutbench_ml/outcome_model_v1
.\.venv\Scripts\python.exe -m experiments.qmutbench_outcome_ml train --output-dir data/qmutbench_ml/outcome_model_v1 --cv-folds 3
.\.venv\Scripts\python.exe -m experiments.qmutbench_outcome_ml evaluate --output-dir data/qmutbench_ml/outcome_model_v1
.\.venv\Scripts\python.exe -m experiments.qmutbench_outcome_ml load-model --output-dir data/qmutbench_ml/outcome_model_v1
.\.venv\Scripts\python.exe -m experiments.qmutbench_outcome_ml predict --input path\to\new_metadata.csv --output data\qmutbench_ml\outcome_model_v1\predictions.csv --output-dir data\qmutbench_ml\outcome_model_v1
.\.venv\Scripts\python.exe -m experiments.render_qmutbench_outcome_ml_figures --metrics data/qmutbench_ml/outcome_model_v1/metrics.json --output-dir reports/experiments/qmutbench_outcome_ml
.\.venv\Scripts\python.exe -m pytest tests/test_qmutbench_outcome_ml.py -q
```

Grouped-CV fold summaries are cached under `data/qmutbench_ml/outcome_model_v1/cv_checkpoints`; matching completed folds are reused. The full model, held-out row predictions, and complete metrics manifest are stored under `data/qmutbench_ml/outcome_model_v1/`, which is git-ignored. The committed sanitized metrics snapshot is `metrics.json`; it records the source hash, split seed and group hashes, features, target, CV results, parameters, Python/pandas/NumPy/scikit-learn/joblib versions, test metrics, and model SHA-256 `5638e554328ba18b0f741e1af2fdcb78d58ea158419073e15b47fbf50c6141c7`. The model binary itself is not committed.

The bounded smoke test used 30,000 rows and 36 parent groups, completed a group-disjoint fit/evaluation, and checked deterministic joblib save/load prediction equality. It is a pipeline check, not a metric result. The full metadata training took approximately four minutes on this host; the category-coded dataframe used about 19.6 MB, and observed process working set samples were around 260–310 MB. Available physical memory before training was 2.67 GiB. No QASM simulation, full-corpus re-ingestion, IBM job, or added dependency was used.

## Limitations and next work

- QMutBench's 375 origins are not independently verified MQT recipe roots. The result establishes generalization to held-out origin identifiers only.
- The metadata outcome is from historical oracle evaluation, not independent behavioral adjudication. It cannot establish fault attribution or noise robustness.
- The public corpus metadata is complete locally, but QASM sources are not. The model intentionally predicts from metadata rather than claiming circuit-semantic understanding.
- A further study should hold out verified recipe families, compare richer circuit representations only after source-lineage acquisition, and pre-register a new test split. The current held-out split has already been inspected and must not be reused as a future confirmation set.
- Improving beyond 85% is future work. Do not tune against this final test or reinterpret the target.
