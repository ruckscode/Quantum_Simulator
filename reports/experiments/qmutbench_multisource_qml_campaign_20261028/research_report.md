# Multi-dataset quantum ML campaign - 2026-10-28

## Executive result

The QMutBench binary production model remains the strongest validated system: mutation-aware Extra Trees, macro F1 **0.793314**, accuracy **0.793329**, balanced accuracy **0.794087**, five grouped folds across 578,117 development rows and 305 origins. Its checkpoint SHA-256 is unchanged at `8cd332a309e27ef09cc46c5bf8a8788da35c05aa6ed25ed5de21d26ced076175`.

No cross-source or quantum model was promoted. In the separate exploratory three-class QMutBench study, the six-model equal-probability blend reached macro F1 0.4919 on 205 sampled rows from 12 origins, versus 0.4669 for the strongest single model. The blend has not been evaluated on untouched origins. Its planned 36-origin confirmation export stopped before reading data because available RAM was 792 MiB, below the exporter’s 900 MiB pre-import gate.

Bugs4Q supports a distinct source-state task. A new group-held-out classical benchmark on 31 bug issues (62 paired source snapshots) reached 0.5312 macro F1 with a Random Forest using a four-dimensional fold-local projection. The Bugs4Q QNN was skipped at its 700 MiB preflight guard after the environment reported 684 MiB. This dataset-specific result is not comparable with either QMutBench task.

## Data sources, versions, and label semantics

The detailed source inventory and provenance are in `source_inventory.csv` and `dataset_provenance.json`.

- **QMutBench:** the pinned local metadata fingerprint is `3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778`. The production task maps WOO/OPO to detected and Survivor to undetected. The three-class research task keeps Survivor, WOO, and OPO distinct.
- **Bugs4Q:** the local official database ZIP is checksum verified, and the framework README documents buggy/fixed source versions and triggering tests. The README describes the Qiskit collection as 42 manually curated bug cases; only 31 had paired buggy/fixed files, metadata, and both triggering-test files in the local extraction. Its Apache-2.0 framework license is preserved. [Official Bugs4Q repository](https://github.com/Z-928/Bugs4Q) documents the `Buggy` and `Fixed` source-state fields and describes the source as a real-world Qiskit bug benchmark.
- **Bugs4Q-Robust:** the official replication repository was cloned at commit `c7048476e5e76e95412b3640a3fa5651f51c01a4`. It contains 1,110 paired repeated-run rows across 37 artifacts (30 runs per artifact), and published aggregated result tables. The replication package states its scripts/configuration are MIT and generated logs/results are CC BY 4.0, while subject programs remain third-party and retain upstream rights. I audited the published data only; I did not retrain on repeated runs or treat them as independent samples. [Replication-package documentation](https://github.com/kusumotolab/quantum-defect-reproducibility) explains those license boundaries and the snapshot layout.
- **QBugs:** the located 2021 paper is a proposal describing a planned catalog and supporting infrastructure, not a locally available row-level dataset; no compatible examples were downloaded. [The paper](https://jose.github.io/assets/pdfs/Q-SE2021-shortpaper.pdf) describes the catalog as a development objective.
- **Qolumbina:** current documentation describes a benchmark of scalable test-ready quantum programs with execution utilities and unit tests. It provides program-level test subjects rather than mutation-outcome labels, and no local data was present. [Official v0.1.0 documentation](https://qolumbinadoc.netlify.app/) describes the program and test infrastructure.
- **Ecosystem defect reports:** the paper reports 32,296 issues across 123 repositories, but says supporting data can be shared upon request. No public row-level artifact was verified or downloaded, and no author was contacted. [Data availability statement](https://arxiv.org/abs/2512.24656).
- **Recent verified-bug collection:** a current preprint describes 410 verified bugs from ten quantum repositories and 56 bug families, but a row-level release and reuse license were not identified. It remains an acquisition lead, not a training source. [Preprint](https://arxiv.org/abs/2610.22089).
- **Kaggle:** search identified a community “Quantum” competition with a separate 800-row training target for simulated circuit-run stability/crash and a CC0 listing. No Kaggle CLI or logged-in data access was available, so I did not download it or use its competition test records. [Kaggle competition page](https://www.kaggle.com/competitions/quantum/data).

No source label was converted into a QMutBench label. Source-specific supervised tasks are registered in `task_and_label_registry.json`.

## Standalone benchmarks

### QMutBench binary production task

The incumbent binary model remains best supported at macro F1 0.793314 (95% paired-origin bootstrap difference vs legacy 8-feature model: +1.13 to +2.98 pp). Per-class and fold summaries are in `standalone_model_leaderboards/qmutbench_binary.json` and the earlier classical campaign report. No production artifact was changed.

### QMutBench three-class research task

The existing fixed sample contains 205 records from 12 OPO-bearing development origins, deliberately class-balanced (Survivor 72, WOO 68, OPO 65). Three-fold GroupKFold held out origins. Classical models received four fold-selected features; QNN variants used supervised losses after fold-local reduction and scaling. The best matched classical candidate was Extra Trees at macro F1 0.4669. The strongest QNN/hybrid was angle VQC with trainable readout at 0.3218. Angle VQC, amplitude VQC, and trainable-readout VQC scores remain far below the matched classical baseline. All circuits ran on analytic PennyLane `default.qubit` CPU statevectors; earlier work validated analytic gates, measurements, gradients, and optimizer updates. No GPU or real quantum hardware was used.

The saved-prediction integration screen tested fixed probability averages and outer-fold-held-out stacking. The six-model blend improved the 12-origin exploratory score by 2.50 pp over the best individual model, while stacking scored 0.4212. The blend and stacking analysis used the same exploratory origins that had already informed model inspection, so neither result supports promotion. Pairwise error overlaps, per-class results, origin metrics, and bootstrap intervals are in the preceding ensemble campaign directory. A weighted search and conditional router were not attempted because the set is too small and no independent tuning folds remain.

### Bugs4Q buggy/fixed source-state task

The task predicts the upstream source-state label: fixed=0, buggy=1. The independent group is the bug/issue ID; both versions remain together. Features come from sanitized source text with comments and docstrings stripped; executable string literals are retained because they encode gate names and other source semantics. A first audit found that removing every string made one buggy/fixed pair feature-identical, so the sanitizer was corrected and the full benchmark rerun; the final feature corpus has no exact duplicate bodies. TF-IDF vocabulary, dimensionality reduction, and scaling were fit within each fold. The same four-dimensional projection fed matched classical controls and was a candidate input for a quantum model.

| Classical candidate | Macro F1 | Accuracy | Balanced accuracy | Group-bootstrap 95% macro-F1 interval |
|---|---:|---:|---:|---:|
| Random Forest, 4D projection | **0.5312** | **0.5323** | **0.5323** | 0.4637?0.5941 |
| Logistic Regression, 4D projection | 0.5222 | 0.5323 | 0.5323 | 0.4689?0.5767 |
| RBF SVC, TF-IDF | 0.5156 | 0.5161 | 0.5161 | 0.4837?0.5479 |
| Small MLP, 4D projection | 0.5156 | 0.5161 | 0.5161 | 0.4891?0.5484 |
| Extra Trees, 4D projection | 0.4999 | 0.5000 | 0.5000 | 0.4353?0.5635 |
| Linear SVC, TF-IDF | 0.4999 | 0.5000 | 0.5000 | 0.4342?0.5635 |
| Decision Tree, 4D projection | 0.4701 | 0.4839 | 0.4839 | 0.4126?0.4988 |
| Histogram Gradient Boosting, 4D projection | 0.4676 | 0.4677 | 0.4677 | 0.4018?0.5161 |
| Logistic Regression, TF-IDF | 0.4676 | 0.4677 | 0.4677 | 0.4139?0.4999 |
| Majority baseline | 0.3333 | 0.5000 | 0.5000 | 0.3333?0.3333 |

The selected Random Forest checkpoint, including TF-IDF, SVD, and scaling, was fitted on all eligible paired cases after OOF evaluation. The scores are near chance, and the 31-group sample is small. The five-fold split keeps each bug’s variants together, but does not claim generalization to other repositories or SDKs. The actual trigger tests were required to exist for inclusion, but were not re-executed in this standalone ML stage. Per-fold metrics, predictions, per-bug results, resource data, and checkpoint are in `bugs4q_standalone/`.

### Bugs4Q-Robust reproducibility evidence

The archived result table has 37 artifact IDs and 30 repeats per artifact. Across paired test outcomes, the buggy version failed 990/1,110 runs and passed 120/1,110; the fixed version passed 990/1,110 and failed 120/1,110. These are execution-oracle results from repeated runs, not 1,110 independent ML examples. The summary table classifies 29 artifacts as complete success, four partial success, and four failures. This is useful reliability evidence, but it is a different task from QMutBench mutation prediction and was not merged into the training matrix.

## Integration, transfer, and ensemble decision

The source/task integration matrix is in `transfer_learning_results.json` and `integration_leaderboard.csv`.

- **Compatible-source joint training:** none. The only source with QMutBench WOO/OPO/Survivor mutation labels is QMutBench.
- **Transfer learning and multi-task learning:** not run because Bugs4Q bug/fix state, Bugs4Q-Robust trigger reproducibility, Qolumbina test programs, and issue-taxonomy labels do not share QMutBench’s target or validated input schema. No label remapping was made.
- **Feature integration:** existing QMutBench mutation-aware feature ablations and structural add-on confirmation remain as previously reported; the structure add-on did not show a reliable improvement. Bugs4Q uses its own source-code representation.
- **Model-level ensembles:** fixed equal probability blend and fold-held-out stacking were evaluated on the 12-origin QMutBench pilot; errors and origins are saved. Six-model blend confirmation on the reserved 36 origins did not run. Therefore the answer to whether the blend improves untouched origins is **unknown**, not yes.
- **Conditional routing:** not run; the available pilot has 12 origins, and no predeclared inference-time route can be evaluated credibly.

No combination is promoted. Existing full-feature binary Extra Trees remains the best validated production-aligned system. The binary and three-class scores are kept separate.

## Compute, confirmation, and safeguards

- The local machine has 7,488 MiB RAM and a 500 MiB safety floor. Bugs4Q classical grouped CV completed in about 18.4 seconds; available RAM after imports was 1,055.9 MiB and after the run was 1,329.4 MiB. Peak RAM was not measured.
- Bugs4Q QNN fitting stopped before PennyLane/model import or training when available RAM was 684 MiB after the shared scikit-learn stack had loaded, below its 700 MiB fit gate. No QNN optimizer update or checkpoint is claimed for Bugs4Q.
- The QMutBench reserved-origin exporter stopped before importing the data loader, scanning rows, or creating output when available RAM was 792 MiB, below its 900 MiB pre-import requirement. The 36-origin confirmation did not run.
- QMutBench QNN work already completed on the 12-origin exploratory sample did not establish an advantage. Previous confirmation attempts stopped at the 500 MiB in-fit safety floor; no confirmation model checkpoint exists.
- CPU analytic `default.qubit` forward/gradient checks were completed in the prior QNN campaign. The lightning GPU plugin was absent, the optional CPU lightning binary was blocked by Windows policy, and no Kaggle session or GPU was available. GPU equivalence and Kaggle training were not verified.
- Frozen QMutBench test records and labels were not loaded for this campaign. The frozen evaluator was not run. A prior QNN report records that an earlier adapter parsed raw target columns before filtering, although no frozen values were retained or used for model selection. The current confirmation exporter stops before any data read when its memory gate fails.

## Verification and next experiment

The Bugs4Q benchmark script completed grouped OOF evaluation, emitted per-class and per-bug metrics, and saved a source-specific Random Forest checkpoint. The QMutBench ensemble artifact audit and this campaign’s JSON/ID validations and syntax checks are listed in `test_manifest.json`. A pytest suite was not run during this continuation. The production model SHA-256 is verified unchanged in `checkpoint_manifest.json`.

The highest-value next experiment is the predeclared six-model blend evaluation on the reserved 36 QMutBench development origins, compared against the matched classical Extra Trees, under the existing protocol. It requires materially more memory or a verified remote environment. Run it once, report uncertainty across the 36 origin groups, and do not tune using its outcomes. Separately, any Bugs4Q QNN follow-up requires a stable headroom above its 700 MiB preflight gate and should be treated as a standalone source-state task.
