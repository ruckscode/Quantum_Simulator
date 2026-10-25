# ML data recovery and generalization design checkpoint — 2026-10-15

## Decision

No existing local dataset is eligible for the project's primary diagnosis-generalization evaluation. The historical 270-row dataset remains rejected by strict preflight (270/270 ineligible); the lineage inventory is truncated; and other found datasets either have synthetic scenario labels, address a different target, or lack compatible provenance. No model was fit, no historical score was reproduced, and no full dataset was generated.

## Historical 270-row record recovery

The record-level table is `ml_historical_record_recovery_20261015_v2.json` (the earlier same-date table is preserved). It contains 270 stable run/artifact IDs, current artifact SHA-256 integrity hashes, exact fields read from each saved report, per-field `RECORDED` / `INFERRED` / `MISSING` status, and eligibility decisions. The bounded rebuild confirms the row values match all 270 current reports and the dataset manifest.

Fields directly preserved per report include run/circuit/family/group IDs, repetition index, random seed, shot count, reference and submitted circuit configurations, hardware/noise configuration, Python and NumPy versions, the generic simulator string, label, and saved run fingerprint. The dataset manifest records design/execution seed bases, 30 nominal families, three repetitions, 1,024 shots, and 270 rows. All labels are controlled simulator scenario-construction labels.

The reports do not record per-run source hashes or recipe-root/ancestry lineage. The manifest has no `generation_provenance`, source hash map, exact runtime/build record, or family ancestry. `simulator_version: project-statevector-current` is not a source identity. Current Git history has only one commit that touches the generator path and does not link that source state to the report records. Recomputed artifact/configuration hashes verify only the saved bytes as they exist now; they do not establish which code generated them. Therefore no row is eligible for training, development/model selection, calibration, or held-out evaluation. The rows remain usable only as explicitly synthetic software fixtures.

## Other local data sources inspected

| Source | What is established | Eligibility for primary diagnosis generalization |
|---|---|---|
| Nine-row ML data pilot | Simulator-derived synthetic scenarios; three recipe lineages; previous integrity smoke check; zero independent adjudications. | Engineering fixture only; it has already been inspected and is not fresh validation evidence. |
| Phase 5 F1/F2 training JSONL (330 rows) | Earlier simulator-derived synthetic data for a restricted scenario subset. | Not independent families or adjudicated truth; not eligible for primary claims. |
| QMutBench ML artifacts | Manifest records a 10,000-candidate selection; output audit records 9,792 successful mutant records and 211 original records; 208 mutant execution failures are separately recorded. The task is mutant/original behavioral deviation and uses five trace-difference features, not the project's three-way simulator diagnosis target. All 211 originals have zero features; this is a target-construction signal that requires task-specific treatment. | Not eligible for three-way cause diagnosis or hardware claims. The manifest's expected source CSV SHA-256 is `3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778`; the local 60,636,991-byte `all_data.csv` hashes to `d8adabf58173f1ae7c28954832e1b9607bc315eb1269a9cb57237343dfa95934`. The mismatch and absent generator-source fingerprint prevent reuse of the saved derived features as reproducible current evaluation data. |
| QMutBench full source tree | Local Git history reports one grafted commit; the checked-in data are mutation benchmark inputs, not independent hardware diagnosis labels. | Could support a separately specified mutant-behavior task after source/version reconciliation; not the primary target. |
| Bugs4Q archive and extracted source | Local archive and extracted repository are present; extracted Git history reports 40 commits. Existing project audit covers 42 official-source cases, 19 translated/executed, and zero metric-eligible cases. No compatible ML table of output-observable features and independently adjudicated diagnosis labels was found in the reviewed artifacts. | Not eligible for the current ML target; benchmark parsing/truth compatibility remains separately blocked. |
| Saved grouped-CV v1/v2 artifacts | Historical synthetic-label evaluation artifacts; source/runtime manifests identify evaluator snapshots, not the generator that produced each historical report. | Historical descriptive evidence only; not current model results. |
| Finite-shot calibration outputs | Simulator-generated evidence under the preserved NO-GO decision. | Not independent ML validation; do not reuse for tuning or confirmation. |

The bounded prior-lineage inventory covered 108 roots but stopped at 20,000 files / 128 MiB and remains marked truncated. It reports 5,366 circuit-like records, 5,186 unknown lineage records, and zero eligible fresh roots. All five oversized entries previously skipped by that scan were inspected by metadata and streamed through SHA-256 in 1 MiB chunks under a 320 MiB aggregate cap (315,670,588 bytes total); this verifies their local byte identities but does not make opaque Git pack/index contents eligible evidence or complete the truncated inventory.

| Oversized inventory entry | Bytes | Streamed SHA-256 |
|---|---:|---|
| `data/Bugs4Q_official/Bugs4Q-Database.zip` | 29,276,112 | `e06a0fdf258c7c8596782cabdb2e1846d51c8866ed8a12129375fe07ea7ed9b1` |
| `data/Bugs4Q_official/extracted/Zenodo-Database/Bugs4Q-Database/.git/objects/pack/pack-a8a067c54dcd1079fd41c9e82902238b480a8ad0.pack` | 29,005,661 | `1bb9d792376e05ae60eec2018068788c2e8b75174308fa0557d0138b162f5d12` |
| `data/qmutbench_full/.git/index` | 104,672,040 | `0bfaca183b0bd77fc968df795d3c2a7f9a0f44a7d2bf847b0ba27418473f223b` |
| `data/qmutbench_full/.git/objects/pack/pack-c620372d323793afd58ff1944a38f5e8345ed0af.pack` | 92,079,784 | `007423c375d3881195240570185c1524248e99fe53aea75b5e472b55f64dfb51` |
| `data/qmutbench_full/data/all_data.csv` | 60,636,991 | `d8adabf58173f1ae7c28954832e1b9607bc315eb1269a9cb57237343dfa95934` |

## Pipeline improvements in this checkpoint

- The existing future simulator generator already records transitive source hashes and runtime provenance, per-run configuration/seed/shot/noise metadata, and explicit `circuit_lineage_id` values; its generator regression tests check the manifest against the current source. The current design cycle declares only six `design_recipe_*` roots and assigns synthetic intervention labels, so this is a reproducible synthetic-development path, not an independent family frame. No regeneration was run.
- Feature matrix construction defaults to the registered five-observable production schema. Calibration and circuit-comparison schemas must be named explicitly; label/post-outcome fields remain forbidden.
- The fixed comparison catalog now includes majority and stratified dummy baselines, Logistic Regression, a scaled fixed Linear SVM, Decision Tree, and Random Forest. This is a descriptive fixed comparison; it does not tune or select a model. Random Forest and permutation-importance workers are capped at one.
- A train/calibration/test structural validator enforces exhaustive row assignment, valid labels, nonempty IDs, exact fingerprint separation, and family plus lineage-root disjointness. It reports partition composition but makes no claim about independent truth or adequacy.
- Evaluation manifests identify the target as synthetic simulator scenario disposition, label origin, synthetic-only evidence scope, and explicitly deny independent-generalization and real-hardware claims.
- Evaluation metrics now include deterministic lineage-cluster percentile bootstrap intervals for accuracy, balanced accuracy, macro-F1, and class-wise precision/recall/F1 only when bootstrap replicates support every class. Unsupported class-wise intervals are withheld. This is an engineering reporting capability, not evidence of adequate sample size.
- New comparison outputs use `ml_results/grouped_cv_audit_v3`; existing v1/v2 artifacts are retained, and a nonempty output directory is rejected before data loading/fitting.
- The record-level recovery audit reads only bounded report files and refuses count mismatches or caps; the existing report is reproducible from current saved inputs.

The current model loader/preflight intentionally accepts only the controlled synthetic validation row contract. There is not yet a reviewed ingestion adapter that joins independent adjudications to observation-only rows; adding a declaration to a manifest would not establish reviewer independence. That adapter requires a specified adjudication record schema and independent review rules. Real-hardware records remain a separate source type and are not accepted as simulator scenario labels.

## Evaluation design to use when eligible families exist

**Primary estimand candidate:** detection of an observable output deviation from the supplied reference distribution, at the time an execution's observed output distribution is available and before diagnostic metadata is consulted. This uses the five registered output-distribution observables. Cause disposition (normal/program fault/hardware-related anomaly) is a separate, harder target and requires independent cause adjudication; observationally equivalent or insufficient-evidence cases must remain ambiguous/abstain. Fault subtype and source/qubit/gate localization are not supported by these five features.

**Independent unit:** a genuinely independent circuit-family/recipe root, including every mutation, template, seed, configuration, and repetition descended from that root. Related items must remain in one partition. Previously inspected calibration/validation templates and their descendants are excluded from new validation. The current family inventory does not identify enough eligible new roots to sample this frame.

**Label source:** simulator intervention labels may support software-mechanism development and must be marked synthetic. Primary generalization needs blinded independent adjudication against intended program semantics and execution evidence. Hardware-cause labels additionally need device identity, calibration/run logs, backend conditions, and reviewers able to distinguish program from device causes. Unknown/equivalent cases need an adjudicated ambiguous disposition. No independent adjudicators or such evidence are currently available.

**Partitions and selection:** new families are allocated before outcomes are examined to development/training, calibration, and sealed test sets. Model or hyperparameter comparison uses family-grouped nested validation wholly inside training/development families. Probability calibration and any operating threshold/abstention policy use calibration families only. The sealed test is used once after model, features, criteria, and analysis are frozen. No test-family information may enter feature selection or model choice.

**Comparators and measures:** majority and stratified dummy baselines; fixed Logistic Regression, scaled Linear SVM, Decision Tree, and bounded Random Forest; and the existing statistical detector on identical eligible observations and execution budgets. Report detection sensitivity and false-positive rate, class-wise precision/recall/F1, balanced accuracy, macro-F1, confusion matrix/support, family-level uncertainty, and coverage/selective error plus abstention cost where a policy is evaluated. Probability calibration measures are reported only for calibrated probability outputs using calibration families. These comparisons cannot establish hardware performance from simulator observations.

**Power and acceptance inputs still required:** target population/use; per-class prevalence; smallest scientifically meaningful sensitivity/FPR difference or required interval width; family-cluster intraclass correlation/effect heterogeneity; family sampling frame; multiplicity and model-selection plan; calibration method; abstention/error cost; resource cap; and independent review. Without these assumptions there is no defensible numeric family count, power result, performance floor, or acceptance criterion. No sample-size claim is made here.

## Current validation boundary

The current preflight remains `ready=false`, zero eligible historical rows. The three-way validator, schema control, family bootstrap, model catalog, recovery-table reader, and no-overwrite guard are exercised only on small software fixtures. No model was fit or evaluated on the 270 historical rows, the nine-row pilot, QMutBench, or hardware observations. The frozen diagnosis acceptance remains unchanged at 7/45 mismatches; this ML checkpoint does not alter or retune it.
