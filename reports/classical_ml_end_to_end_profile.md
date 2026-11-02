# Classical Machine Learning in the Quantum Simulator Project

**Report draft**

**Evidence cut-off:** 2 November 2026

**Scope:** Classical machine learning only. QML methods and results are outside this report.

**Purpose:** Explain what was trained, what it predicts, where it is used, what its numbers mean, and what it does not establish.

## Abstract

This project uses classical machine learning in three distinct ways: (1) as a source-by-source research benchmark on several quantum-software datasets, (2) as a QMutBench mutation-outcome predictor, and (3) as an explicit, optional candidate for classifying circuit-execution observations in the simulator. These activities do not form one model and do not share one target label. Source benchmark models predict labels such as buggy-versus-fixed source state, issue-comment annotation, test-harness outcome, or backend identity. The QMutBench model predicts whether a generated mutation survived or was killed. The optional simulator diagnosis candidate predicts NORMAL, PROGRAM_FAULT, or HARDWARE_ANOMALY from measured-output statistics.

The current simulator does **not** depend on these source classifiers for ordinary execution or diagnosis. Its default path performs circuit simulation, statistical comparison, operation-mismatch analysis, counterfactual output checks, and—when supplied—calibration-evidence analysis. The observable-only Logistic Regression candidate can be supplied explicitly to the diagnosis API, but it is marked **not validated for deployment**. It cannot localize a faulty gate by itself. The source benchmark models and the QMutBench model are separate research products and are not called by `diagnose_circuit()`.

The completed source-specific nested grouped OOF accuracies range from 49.19% to 94.04%, depending on each task. These values cannot be averaged into a single simulator accuracy because the datasets have different targets. The full-feature QMutBench ExtraTrees model achieved 79.33% accuracy, 79.33% macro-F1, and 79.41% balanced accuracy on five-fold origin-grouped development OOF data; it did not reach the 85% goal and was not evaluated on an untouched frozen test set. The optional diagnosis candidate was evaluated on a synthetic grid: raw single-cause accuracy was 80.0%, while the predeclared overall acceptance failed because program-fault recall was 70%, below the required 80%. Consequently, classical ML currently contributes research evidence, comparative baselines, and a guarded experimental option; it is not yet a validated general-purpose fault-diagnosis model.

## 1. System objective and the role of classical ML

The simulator’s primary user flow is to run a quantum circuit and distinguish among normal behavior, a program fault, and a hardware anomaly when the evidence supports that distinction. The mathematical simulator and evidence logic remain usable without a trained model.

Classical ML contributes to this project in three bounded roles:

1. **Benchmarking quantum-software datasets.** It tests whether text, circuit descriptors, or execution records contain predictive signal for the label provided by each dataset. This provides per-task baselines and shows where the current data is too small, imbalanced, or inconsistent for strong generalization.
2. **Predicting QMutBench mutation outcomes.** It predicts the dataset’s historical mutation result from mutation metadata and circuit-difference information. This is useful as a mutation-testing research task and as a comparison baseline for a separate QML experiment. It is not runtime circuit diagnosis.
3. **Optional execution-level diagnosis classification.** A small Logistic Regression model predicts a broad class from differences between expected and observed measurement distributions. The simulator can optionally receive this classifier, but it does not load it by default. Independent statistical, circuit, and calibration evidence still gates the final diagnosis.

There is no source-selection mechanism that silently chooses the most accurate external dataset model during a circuit run. The multi-source models predict incompatible labels, so no scientifically valid universal “best source model” or averaged “modal accuracy” exists.

## 2. Three classical-ML components and their integration status

| Component | Input and target | Reported evaluation | Called during normal simulator use? | Current status |
|---|---|---|---|---|
| Multi-source text/descriptor benchmark | Each prepared source task has its own feature and label definition | Five-fold outer GroupKFold OOF, with inner grouped model selection | **No** | Research benchmarks; source-specific estimators are not wired into diagnosis |
| QMutBench classical mutation predictor | Mutation metadata and original/mutant QASM-derived features; survived vs. killed | Five-fold origin-grouped development OOF | **No** | Separate mutation-outcome task; current model bundle is absent from the public checkout |
| Observable simulator-diagnosis candidate | Five statistics of expected vs. observed output distributions; NORMAL / PROGRAM_FAULT / HARDWARE_ANOMALY | Synthetic development/evaluation grid; predeclared acceptance failed | **No by default.** Optional explicit `classifier=` argument exists | Development candidate only; not deployment validated |

### What happens when a user runs a circuit

By default, `QuantumCircuit.run()` performs simulation and sampling. `diagnose_circuit()` can compare a submitted circuit against an optional reference circuit, compute count-based statistical evidence, compare expected and observed operations, test whether a program difference changes modeled measured outputs, and examine relevant hardware calibration evidence if a `HardwareModel` is supplied. It does not automatically load QMutBench or any source-specific classifier.

When the caller explicitly provides an `ObservableClassifier` through `diagnose_circuit(..., classifier=...)`, the optional path extracts five output-distribution features and records the model’s predicted class and uncalibrated probabilities in the result. A predicted fault class still requires independent supporting evidence. A direct disagreement with a definitive evidence-led result causes ambiguity; combined program and hardware evidence is not forced into a single-cause label. The classifier does not receive source code as a learned feature representation and does not itself name or localize a gate.

The adapter contract is defined in `src/quantum_simulator/ml_diagnosis.py`; the public optional call is in `src/quantum_simulator/diagnosis_engine.py`. The bundled portable Logistic Regression JSON is a development artifact. The simulator’s default statistical and evidence-based path does not require it.

## 3. Multi-source classical benchmark

### 3.1 Tasks and pooled outer-OOF results

The following numbers are pooled predictions for held-out groups across the five outer folds. Each row refers only to that task’s label.

| Task | Rows / independent groups | Accuracy | Macro-F1 | Balanced accuracy | Meaning and major caveat |
|---|---:|---:|---:|---:|---|
| QBugs QASM (`qbugs_qasm_v2`) | 1,496 / 748 | **92.71%** | 92.69% | 92.71% | Correct vs. buggy QASM snapshots; not runtime fault-diagnosis accuracy |
| Quantum platform comments | 319 / 304 | **94.04%** | 92.67% | 91.29% | Predicts the curated comment annotation label only |
| Three-source bug correctness | 1,620 / 810 | **89.63%** | 89.60% | 89.63% | Compatible correct/fixed vs. buggy target from Bugs4Q core, Bugs4Q-NA, and QBugs; below QBugs QASM alone |
| QSimBench exploratory | 300 / 150 | **82.33%** | 82.33% | 82.33% | Backend identity inferred from output descriptors; recorded descriptor collisions make it exploratory |
| LintQ conflict-excluded context | 287 / 216 | **79.44%** | 67.92% | 65.91% | Warning true-positive vs. false-positive task; low balanced accuracy shows minority weakness |
| Qolumbina ideal | 182 / 26 | **84.07%** | 54.16% | 54.69% | Seeded-mutant detector outcome; majority-class accuracy alone is 82.42% |
| Bugs4Q robust execution | 2,220 / 37 | **56.76%** | 56.73% | 56.76% | Test harness pass/fail; 2,220 repeated rows represent only 37 issue groups |
| Bugs4Q core source state | 62 / 31 | **51.61%** | 51.56% | 51.61% | Small buggy-vs-fixed source-state task; noisy estimate near chance |
| Bugs4Q-NA source state | 62 / 31 | **50.00%** | 48.92% | 50.00% | Same small source-state target; near chance |
| Bugs4Q core + NA source state | 124 / 62 | **49.19%** | 47.97% | 49.19% | Pooling these two sources did not improve this task |

The best standalone task accuracy is the 94.04% comment-label result, but it is not evidence that the simulator detects 94.04% of circuit faults. Likewise, the 92.71% QASM result concerns paired correct/buggy circuit snapshots, not live execution anomalies. The three-source correctness merge scored 89.63%, lower than QBugs QASM alone. Different targets—comment annotation, backend identity, warning triage, harness outcome, and source state—were not mixed into one aggregate score.

### 3.2 Source inputs and their meaning

The benchmark uses prepared adapters from several upstream sources: QBugs/Bugs4Q circuit or source pairs; Quantum Platform Bugs issue annotations; QSimBench execution descriptors; LintQ warning context; Qolumbina seeded-mutant outcomes; and a Bugs4Q robust-execution package. The individual provenance, acquisition URLs, versions, and expected local paths are documented in `CLASSICAL_DATA_SOURCES.md` and `data/classical_ml_v1/source_acquisition_manifest.csv`.

Raw data and many prepared tables are not committed because of size, third-party terms, and source-specific redistribution restrictions. The metrics and protocols are tracked, but a fresh checkout does not contain every raw corpus or local intermediate needed for an exact rerun. The source-specific models are explicitly marked as not used for diagnosis inference in `data/classical_ml_v1/source_registry.csv`.

### 3.3 Text representation and candidate models

For each source task whose model input is text, the search represents the text using four TF-IDF feature families:

- word n-grams of lengths 1–3 (up to 30,000 features);
- character n-grams of lengths 2–5 (up to 50,000 features);
- character-within-word n-grams of lengths 3–5 (up to 50,000 features);
- a combined word plus character representation.

The vectorizers use sublinear term frequency and Unicode accent stripping. The search compared four classifier families over these representations: LinearSVC, Logistic Regression, SGD with modified-Huber loss, and Complement Naive Bayes. The grid contains 104 text-model configurations. A separate grid contains 32 sparse cosine KNN configurations: four representations, neighbor counts 3/5/9/15, and uniform or distance weighting. Thus, up to 136 candidate configurations were available inside each outer training partition.

The models selected most often were linear text models for QBugs QASM, platform comments, the pooled correctness task, QSimBench, and Qolumbina. Sparse cosine KNN was selected more often for the smallest Bugs4Q source-state tasks and most LintQ folds. Selection varies by fold; there is not one universally winning classifier.

For the Bugs4Q robust-execution task, exact duplicate snippets with the same label and issue group were compressed for fitting and weighted by their multiplicity. All held-out execution records were still scored. This avoids pretending duplicated rows are independent training examples while retaining the record-level test metric.

## 4. Evaluation design and metric definitions

### 4.1 Nested group-disjoint cross-validation

A random row split would leak related records: two versions of the same issue, circuit, or execution group could land in both training and validation data. The benchmark therefore groups by source-defined issue/program groups.

For each of five outer folds, one group partition is held out for final scoring. The remaining groups are used in three-fold inner GroupKFold to choose features and model settings. The winning setting is refit on the outer training groups, then predicts the untouched outer validation groups once. Every record receives one outer out-of-fold prediction. No frozen test set was accessed.

In notation, for outer fold (k), model selection is

\[
\theta_k^* = \arg\max_{\theta \in \Theta}
\frac{1}{3}\sum_{j=1}^{3}
\operatorname{Accuracy}(D_{k,j}^{\text{inner-val}};\theta),
\]

where \(\Theta\) is the configuration grid and each inner split is group-disjoint. The selected model then predicts the outer held-out groups. Hyperparameter selection never sees those outer labels. Pooled OOF metrics are calculated from all outer predictions, not by taking the best outer fold.

This protocol reduces leakage from related groups and selection bias. It does not guarantee transfer to a new source, new label policy, new circuit family, or hardware platform.

### 4.2 Metrics

For \(N\) labeled predictions, accuracy is

\[
\operatorname{Accuracy}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\hat y_i=y_i].
\]

For class \(c\), precision is \(TP_c/(TP_c+FP_c)\), recall is \(TP_c/(TP_c+FN_c)\), and

\[
F1_c=2\frac{\operatorname{Precision}_c\operatorname{Recall}_c}
{\operatorname{Precision}_c+\operatorname{Recall}_c}.
\]

Macro-F1 is the unweighted mean of class F1 scores. Balanced accuracy is the unweighted mean of per-class recall. These latter metrics reveal failures hidden by class imbalance. For example, Qolumbina accuracy of 84.07% is only slightly above an 82.42% majority baseline, with 54.16% macro-F1 and 54.69% balanced accuracy.

### 4.3 TF-IDF and model objectives

For token/ngram \(t\) in document \(d\), the used sublinear term frequency is approximately \(1+\log f_{t,d}\) for positive count \(f_{t,d}\). Smoothed inverse-document frequency is

\[
\operatorname{IDF}(t)=\log\left(\frac{1+N_d}{1+df_t}\right)+1,
\qquad
x_{t,d}=\operatorname{TF}(t,d)\operatorname{IDF}(t),
\]

where \(N_d\) is the number of training documents and \(df_t\) counts training documents containing \(t\). Each vector is L2-normalized by the vectorizer. Vectorizers are fitted within the model-selection pipeline on training partitions, not on the held-out fold.

The main classifier objectives are:

- **Linear SVM:** fit a separating hyperplane by minimizing a regularized hinge-loss objective, conventionally \(\frac12\|w\|^2+C\sum_i\max(0,1-y_i(w^Tx_i+b))\). The searched \(C\) values were 0.03, 0.3, 3, and 30; class weights were either unweighted or balanced.
- **Logistic Regression:** estimate class probabilities by minimizing regularized log loss, e.g. \(\sum_i\log(1+e^{-y_i(w^Tx_i+b)})+\lambda\|w\|^2/2\). The search varied \(C\) over 0.03, 0.3, 3, and 30, using the liblinear solver and optional balanced class weights.
- **SGD modified-Huber:** learn a linear boundary by stochastic-gradient updates. For margin \(z=yf(x)\), modified-Huber loss is 0 for \(z>1\), \((1-z)^2\) for \(-1\le z\le1\), and \(-4z\) for \(z<-1\). The grid used \(\alpha\in\{10^{-5},10^{-4},10^{-3}\}\) and optional balanced class weights.
- **Complement Naive Bayes:** estimate smoothed token likelihoods from the complement of each class. With vocabulary count \(N_{\bar c,t}\) and smoothing \(\alpha\), a representative weight is \(\log((N_{\bar c,t}+\alpha)/(\sum_tN_{\bar c,t}+\alpha V))\), where \(V\) is vocabulary size. It is often useful for sparse, imbalanced text; smoothing values 0.05, 0.2, 1, and 5 were searched.
- **Cosine KNN:** compare a held-out TF-IDF vector \(x\) with training vectors using \(\operatorname{sim}(x,x_i)=x^Tx_i/(\|x\|\|x_i\|)\). The labels of the \(k\) most similar examples are voted uniformly or with distance weighting, with \(k\in\{3,5,9,15\}\). It is a retrieval-style baseline, not a parametric model.

The regularization and model selection happen inside each training fold. Test groups are not used to choose model type or hyperparameters.

## 5. QMutBench classical mutation-outcome work

QMutBench is a separate task. The binary target is **survived (0)** versus **killed (1)** according to the dataset’s recorded mutation-testing outcome. The model does not decide whether a submitted simulator circuit is a hardware anomaly, nor does it locate an erroneous gate in a live execution.

The full-feature mutation-aware ExtraTrees model reported **79.33% accuracy, 79.33% macro-F1, and 79.41% balanced accuracy** on 578,117 development records grouped across 305 original program/circuit origins. The evaluation used five-fold origin-grouped OOF. It did not access a frozen test set, and the 85% target was not met. This is a development estimate, not external validation.

The inference package’s input contract includes algorithm, mutation operation, gate, dominant state, qubit and gate counts, mutation position metadata, and original and mutant OpenQASM text. It parses the QASM pair to derive mutation alignment and local-context information; measurement and barrier instructions are excluded from the alignment. Source labels and identifiers are not accepted as predictive features. The feature documentation and runtime contract are in `qmutbench_ml/README.md`.

A second experiment was deliberately feature-matched to a narrow QNN comparison. It used the same saved origin-disjoint folds and a fold-local four-feature selector/encoder. The encoder was fit on up to 5,000 balanced examples from the training fold only; the selected four features differed by fold. The strongest classical estimator in this narrow representation was ExtraTrees:

| Matched four-feature estimator | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| ExtraTrees | **57.00%** | 56.61% | 56.73% |
| HistGradientBoosting | 56.93% | 56.48% | 56.64% |
| RandomForest | 56.79% | 56.46% | 56.54% |
| SGD Logistic | 55.83% | 55.39% | 55.54% |
| Logistic Regression | 55.73% | 55.00% | 55.34% |
| LinearSVC | 55.69% | 54.95% | 55.30% |
| Balanced Logistic Regression | 55.47% | 55.41% | 55.68% |
| Prior-only dummy | 51.84% | 34.14% | 50.00% |

The 79.33% full-feature score and 57.00% matched-feature score answer different questions; the narrower result is the appropriate classical comparator for that four-feature experiment. Neither is simulator diagnosis accuracy. The trained QMutBench inference bundle is not included in the current public checkout, so these benchmark scores do not mean QMutBench inference is currently available from a fresh installation of the simulator package.

### ExtraTrees model in brief

ExtraTrees grows many randomized decision trees. At each node, candidate thresholds/features are randomized; the selected split reduces class impurity. For tree \(m\), a leaf predicts a class distribution \(p_m(c\mid x)\). The ensemble averages these distributions:

\[
\hat p(c\mid x)=\frac{1}{M}\sum_{m=1}^{M}p_m(c\mid x),
\qquad \hat y=\arg\max_c\hat p(c\mid x).
\]

This captures feature interactions without assuming a linear boundary, but it can still learn source-specific correlations, does not guarantee causal interpretation, and can generalize poorly when the held-out origin distribution differs from training data.

## 6. Optional simulator-diagnosis Logistic Regression

### 6.1 Features and classifier

The bundled optional candidate uses five features computed from expected and observed output distributions \(P\) and \(Q\):

1. total variation distance (TVD);
2. Jensen–Shannon divergence (JSD);
3. Hellinger distance;
4. Shannon entropy of the observed distribution;
5. the maximum probability of any observed outcome.

For distributions over outcomes \(x\),

\[
\operatorname{TVD}(P,Q)=\frac12\sum_x|P(x)-Q(x)|,
\]

\[
\operatorname{JSD}(P,Q)=\frac12D_{KL}(P\|M)+\frac12D_{KL}(Q\|M),
\quad M=\frac12(P+Q),
\]

\[
Hellinger(P,Q)=\frac{1}{\sqrt2}\sqrt{\sum_x(\sqrt{P(x)}-\sqrt{Q(x)})^2},
\]

\[
H(Q)=-\sum_x Q(x)\log_2 Q(x),\qquad Q_{max}=\max_xQ(x).
\]

The feature vector is therefore \(\mathbf{x}=[TVD,JSD,Hellinger,H(Q),Q_{max}]\). A multiclass Logistic Regression model maps this vector to scores/probabilities for NORMAL, PROGRAM_FAULT, and HARDWARE_ANOMALY. Its 0.60 maximum-probability rule is an operational abstention threshold, not a claim that a 0.60 probability is 60% empirically correct. The probabilities have not been calibrated.

The model was fitted on 330 synthetic rows from only two four-qubit development families: F1 GHZ chain and F2 Toffoli compute. The evaluation grid contains 900 configurations (60 scenarios, five seeds, three shot counts, five circuit families); the saved results from F1/F2 and the later families have already been examined. Therefore F3/F4/F5 cannot now be described as an untouched locked holdout.

### 6.2 Evidence gate and gate localization

The ML model predicts only a broad class. It does not consume gate-token text and cannot identify which gate caused a fault. The independent simulator pipeline supplies localization:

- a program candidate comes from an expected/observed operation mismatch;
- a program fault requires that the source difference can change measured outputs and that execution evidence supports an anomaly;
- a hardware candidate must be relevant to the executed circuit and supported by supplied calibration evidence;
- simultaneous credible program and hardware evidence remains ambiguous instead of being collapsed into one label;
- unsupported or conflicting evidence may result in abstention or insufficient evidence.

The broad ML class is retained as supporting information. It cannot silently replace the evidence gate. If no classifier is explicitly passed, this ML path is not run.

### 6.3 Synthetic benchmark results and acceptance

The classifier/evidence-gate rescore of 900 saved synthetic cases reported:

| Measure | Result | Correct interpretation |
|---|---:|---|
| Raw ML accuracy | **80.0% (540/675)** | Single-cause rows only; class-imbalanced; not physical-hardware performance |
| Evidence-adjudicated macro-F1 | **0.9363** | The same 675 single-cause rows after applying the evidence-gating workflow; not raw model accuracy |
| Program-fault precision / recall / F1 | 1.000 / **0.700** / 0.8235 | Recall failed the predeclared 0.80 requirement |
| Hardware-anomaly precision / recall / F1 | 1.000 / 0.9978 / 0.9989 | Synthetic grid only, with relevant calibration evidence |
| Normal precision / recall / F1 | 1.000 / 0.9733 / 0.9865 | Synthetic grid only |
| Exact labeled-target localization | **599/600 (99.83%)** | Performance of the integrated evidence/localization system on the benchmark’s labeled targets, not localization by ML alone |
| Mixed program+hardware ambiguity recall | **71/75 (94.67%)** | The system left competing-cause cases ambiguous for these rows |
| Overall predeclared acceptance | **FAIL: 6/7 criteria passed** | Program-fault recall was below 80% |

The separate output-change heuristic used \(TVD>\max(0.05,1/\sqrt{shots})\). Its sensitivity/specificity were 98.10%/95.00% at 256 shots and 100%/100% at 1,024 and 4,096 shots in the defined synthetic grid. This is an output-deviation heuristic, not the ML classifier’s accuracy, not a p-value, and not a confidence interval. In the public diagnosis API, count-based comparisons use a conditional permutation test plus a practical effect threshold.

Program-fault recall was highly family-dependent: F3 swap routing had 0%, F5 phase layer 50%, while other listed family recalls varied. This is direct evidence against a broad arbitrary-circuit accuracy claim. The benchmark is synthetic, the family set has already been examined, and the per-case rescore report does not include all raw records in the public branch.

## 7. Why results vary or are low

Low or uneven scores are not fixed by simply combining all rows. They reflect distinct statistical and data problems:

- **Different targets:** comment labels, source state, mutation survival, harness success, warning correctness, and backend identity are not the same target. Combining them would create label noise rather than more examples for one task.
- **Few independent groups:** Bugs4Q core and NA each have only 31 issue groups. The robust-execution set has 37 groups even though it has 2,220 execution records. Group-safe validation properly leaves few independent examples per fold.
- **Class imbalance:** Qolumbina’s 84.07% accuracy is near its 82.42% majority baseline; its minority-aware metrics are only about 54–55%.
- **Source shift:** Training and validation groups come from the same prepared source family. This does not prove transfer to a new SDK, backend, code style, or label policy.
- **Ambiguous annotations and collisions:** LintQ had conflicting context labels that were excluded; QSimBench descriptors collide for some settings.
- **Finite observations:** Circuit-output statistics depend on shot count. Close distributions may be indistinguishable at finite shots, and distinct causes can generate the same observed distribution.
- **Selection and model limits:** Nested selection reduces the optimism from choosing a winner, but small groups still make scores unstable. Text models exploit statistical associations, not quantum-causal structure.
- **Synthetic-to-real gap:** Synthetic faults are generated from a finite set of designed circuit families and severities. A model can exploit those construction patterns and fail on real devices or unseen programs.

## 8. What value classical ML currently brings

### Demonstrated value

- A repeatable method for comparing multiple conventional estimators under group-safe validation.
- Per-source evidence about which target types currently contain learnable signal and which do not.
- A robust warning against relying on accuracy alone, illustrated by Qolumbina and LintQ.
- A strong task-specific QASM and comment benchmark, without overstating those scores as diagnosis accuracy.
- A large QMutBench mutation-outcome baseline and a narrower feature-matched ExtraTrees control for research comparison.
- A portable, optional observable-feature classifier interface in the simulator, plus explicit evidence gating and ML metadata in diagnosis results.
- A controlled synthetic evaluation identifying the weakest diagnosis class (program faults) and family-specific failures before any default deployment claim.

### Value not yet established

- No source-specific benchmark estimator is called from the live simulator.
- The QMutBench model does not classify live program faults, hardware anomalies, or normal circuit execution.
- The optional diagnosis model has failed its predeclared release gate and is not enabled by default.
- There is no verified real-device accuracy result for the classical diagnosis model.
- There is no single pooled “classical ML accuracy” for the project.

The honest project conclusion is that classical ML currently provides **evaluated research baselines and a guarded experimental diagnosis option**, while dependable runtime diagnosis remains based on the simulator’s statistical and evidence-localization pipeline. The work shows where ML may assist and what further validation is required; it does not yet establish that classical ML improves user-facing simulator reliability across arbitrary circuits.

## 9. Reproducibility and release limitations

Tracked summaries include `CLASSICAL_ML_RESULTS.md`, `data/classical_ml_v1/benchmark_results.csv`, `data/classical_ml_v1/source_registry.csv`, the source-acquisition manifest, and the report artifacts referenced by the README. Download links, pinned revisions, extraction destinations, checksums, and license notes are in `CLASSICAL_DATA_SOURCES.md`.

Raw corpora, processed task tables, per-run intermediates, and some model bundles are excluded from Git because they are large, local-only, or covered by separate third-party terms. The nine-source adapter builder validates the manifests and partitions it consumes, but a fresh clone does not currently contain the full prepared input bundle required to regenerate every result end to end. The published metrics are traceable to manifests and reports but are not a one-command reproduction package. The QMutBench inference bundle is also absent from the current checkout.

## 10. Recommended report wording

> Classical machine learning was evaluated in three separate roles: source-specific quantum-software benchmarks, QMutBench mutation-outcome prediction, and an optional synthetic execution-diagnosis candidate. The source-specific text/descriptor models were evaluated using nested five-fold group-disjoint cross-validation and achieved task-dependent pooled OOF accuracies from 49.19% to 94.04%; these values predict different labels and are not a unified simulator-accuracy score. The full-feature QMutBench ExtraTrees model achieved 79.33% accuracy on grouped development OOF data, while the feature-matched four-feature control achieved 57.00%; neither is a circuit-diagnosis score or an untouched-test result. The optional diagnosis Logistic Regression candidate was trained on 330 synthetic cases and evaluated on a 900-case synthetic grid. Its raw single-cause accuracy was 80.0%, and the evidence-adjudicated system achieved 0.9363 macro-F1 on those rows, but the overall release gate failed because program-fault recall was 70%, below the 80% requirement. Accordingly, no classical ML model is enabled by default for simulator diagnosis. The current runtime’s primary decision remains based on statistical output comparison, circuit-operation/counterfactual evidence, and relevant hardware calibration evidence; the ML candidate is opt-in and not validated for deployment.

## 11. Files for review

- `CLASSICAL_ML_RESULTS.md` — concise results update.
- `data/classical_ml_v1/benchmark_results.csv` — per-task metrics and evaluation protocol.
- `data/classical_ml_v1/source_registry.csv` — dataset task semantics and diagnosis-inference status.
- `CLASSICAL_DATA_SOURCES.md` — acquisition links, revisions, paths, and reproduction limits.
- `reports/ml_diagnosis_integration_report.md` — classifier integration boundary and status.
- `reports/classical_ml_sim_diagnosis_benchmark_20261031.md` — synthetic diagnosis scorecard and release gate.
- `qmutbench_ml/README.md` — QMutBench inference input contract.
- `experiments/run_classical_source_nested_search.py` and `experiments/run_classical_sparse_knn_nested.py` — nested source-search implementations.
