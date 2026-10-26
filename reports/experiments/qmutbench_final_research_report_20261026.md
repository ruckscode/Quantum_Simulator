# QMutBench final research phase — 2026-10-26

## Evidence and decision

The selected development candidate remains the mutation-aware Extra Trees artifact at `data/qmutbench_ml/mutation_aware_v1/`, using the existing fixed 0.5 inference threshold. The previous matched three-fold OOF result reproduces at macro F1 **0.7889** versus **0.7682** for the incumbent (a 2.07 percentage-point increase). These are historical, selection-informed development results.

The new five-fold outer grouped evaluation scores **578,117** rows across **305** origins. It chooses model, calibration method, and threshold within two-fold grouped inner CV; each outer fold's labels are reserved for scoring. The nested-selected aggregate is macro F1 **0.7928**, balanced accuracy **0.7927**, accuracy **0.7931**, ROC AUC **0.8822**, Brier **0.1430**, and 10-bin ECE **0.0491**. Fixed mutation-aware Extra Trees at raw threshold 0.5 scores macro F1 **0.7934**, balanced accuracy **0.7942**, and Brier **0.1437**. The tuned strategy did not improve on fixed 0.5, so no threshold or calibration change is justified.

Selected fold strategies: `{'et_mutation': 4, 'hgb_mutation': 1}`; calibration: `{'platt': 1, 'raw': 4}`; thresholds: `[0.475, 0.475, 0.525, 0.425, 0.5]`. These nested results are still development-only: the mutation feature set and candidate family were designed after examining this corpus. They are not independent external validation.

## Feature and inference audit

The cache covers 578,117 development rows with zero parse errors. A reproducible 20-record check found zero training/inference feature mismatches. The extractor reads no target or origin-group fields. Measurement and barrier instructions are parsed then excluded from the gate/wire sequence; conditions remain unsupported by the QMutBench OpenQASM 2 parser, malformed input is rejected, unsupported gates now raise a named error, and inference rejects original/mutant pairs with differing declared qubit counts. Repeated or reordered instructions can make common-prefix/suffix alignment ambiguous; this limitation remains.

## External data and errors

The [official QMutBench paper](https://arxiv.org/abs/2604.15870) and [repository](https://github.com/EnautMendi/QMutBench) describe 375 original circuits and 723,079 mutants; the locked corpus contains that release. Public metadata for [QSurvDiag](https://zenodo.org/records/23480917) does not establish matching outcome labels or disjoint origins. [QUMUG](https://zenodo.org/records/21718539) uses a different mutant selection and label semantics. Neither was ingested. No compatible independent external dataset was verified. The prior exhaustive error and subgroup analysis is retained at `qmutbench_exhaustive_error_analysis_20261026.json`.

## Validation and reproducibility

Nested results and fold predictions are local under `data/qmutbench_ml/final_research_v1/nested_outer/` (git-ignored). The baseline/feature audit is `qmutbench_final_research_baseline_audit_20261026.json`; this report's full machine-readable form is `qmutbench_final_research_report_20261026.json`. The QMutBench suite passed **90 tests** (808 unrelated tests deselected); focused QASM and inference tests passed **17 tests**. Real-record single/batch inference and the 20-row train/inference feature check passed. The model binaries and QASM caches remain local.
