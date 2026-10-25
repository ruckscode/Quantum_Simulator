# Phase 5.7 — Blocker Resolution for the Reduced Simulator Benchmark

**Review result: NO-GO for implementing the benchmark-specific runner until the approvals listed below are recorded.** This report resolves the source-based parts of the Phase 5.6 blockers. It does not authorize or perform benchmark execution, model training, or production changes. Phase 5.4 remains authoritative; neither Phase 5.4 nor Phase 5.6 was edited.

## 1. Output-anomaly acceptance denominator

### Evidence and distinction of eligibility

Three things must remain separate:

1. **Constructed output truth.** From the predeclared intervention and reviewed exact probability equations, `D+` means the exact intervention distribution differs from the expected-circuit reference; `D−` means it is equal. This is fixed before predictions and samples. Phase 5.3 gives exact TVD formulae by intervention/family.
2. **Diagnostic-input availability.** Output-anomaly evaluation needs both `expected_distribution` and `observed_distribution` plus the configured `shots`. It does not require a program mismatch or a hardware calibration candidate: anomaly comparison is independent of cause classification/localization. Under the retained custom-runner protocol these distribution inputs are available for each successful configuration. Missing/malformed distributions or shot metadata make the row invalid and must be reported as an execution/input failure, not silently removed from the denominator.
3. **Observed finite-shot outcome.** For each seed/shot row, compute sampled TVD and the existing strict decision `TVD_observed > max(0.05, 1/sqrt(shots))`. A failed crossing is a false negative for an eligible D+ row; it never changes D truth or removes the row. Exact expected TVD is used only for pre-run eligibility, never a model prediction or an observed result.

The Phase 5 design defines hardware strength `.15` as moderate and `.30` as severe; `.06` is near-threshold and `.02` subtle. Phase 5.4's acceptance wording restricts the sensitivity target to identifiable moderate/severe D+ cases. Applying those predeclared strength names and the same fixed detector tolerance to the Phase 5.3 exact TVDs gives this proposed acceptance set:

| Eligible templates | Count | Exact truth and fixed-rule identifiability |
|---|---:|---|
| R15/R30 across F1–F5 | 10 | D+, exact TVD `.15`/`.30`; greater than `.0625` at 256 shots and `.05` at 1024/4096 |
| G15/G30 in F1, F2, F4 | 6 | D+; TVD is `p/2` in F1/F4 (`.075`/`.15`) and `p` in F2 (`.15`/`.30`); all exceed the fixed guard at all three shot counts |
| C15 across F1–F5 | 5 | D+, exact TVD `.15`; the output-detection metric is separate from its combined-cause metric |
| **Proposed primary acceptance set** | **21** | Exact D+ and moderate/severe; exact TVD clears the unchanged shot-specific guard |

For every eligible template, each of the five seeds at each shot level is an observation: **105 decisions per shot stratum** (`21 × 5`) and **315 total** (`21 × 5 × 3`). Preserve separate numerator/denominator by shot, family, and template. The corresponding `>=0.80` target is the Phase 5.4 target; the denominator rule adds no threshold.

This eligibility rule is independent of predictions and sampled outcomes. It resolves the F5 issue: G15/G30 there are constructed D+ hardware interventions, but Phase 5.3 gives exact TVD `((2-√2)/4) × p`, approximately `.02197`/`.04393`. Neither clears the existing `.05` threshold even at 4096 shots. Keep them in the complete D+ descriptive detection table, but exclude them from the *identifiable moderate/severe acceptance denominator*. Likewise, exclude F3 G cases because they are D−, and exclude p=.02/.06 because they are not in the predeclared moderate/severe levels. Program P1/P2 have no strength level in the ledger; report their detection performance in separate program-fault strata, not in this hardware-strength acceptance denominator.

Specificity truth remains exact D−, regardless of intervention label. N0 and the explicitly masked program cases (F3 P1/P2, F5 P2) are the D− controls named in Phase 5.4. F3 G02/G06/G15/G30 are also exact D− despite hardware intervention and must be reported as output-equivalent hardware cases. **Human approval required:** the contract does not state whether F3 G rows join the formal specificity acceptance denominator or remain a separate D− stratum. Proposed resolution: include them in specificity, because output-detection truth is distribution equality independent of cause, while reporting their results separately. C15 inclusion in the sensitivity denominator is proposed for the same reason: output detection is separate from combined-cause scoring. Approve or reject those two denominator membership choices before implementation freeze.

**Aggregation decision also requiring approval:** Phase 5.4 asks for stratification by shot/family but does not say whether `>=0.80` is applied to the pooled 315 eligible observations, separately at each shot level, or to another predeclared aggregate. Proposed report is all stratum numerators and denominators, with the target evaluated separately at each shot level (105 rows/shot); do not use test results to choose the aggregation. Approval is required because this is not explicitly fixed by Phase 5.4.

## 2. Learned component and inference schema

### Is learning required?

For this specification, **yes**. Phase 5.4's final split table assigns F1–F2 to “Fit learned component and development transformations”; it then reserves F3 for probability calibration “if needed.” Earlier Phase 5 design language allows a rule-only path, but the Phase 5.4 final contract is later and more specific. The benchmark runner should therefore use the existing learned diagnosis component for its single-cause class prediction and keep the existing evidence/localization engine for circuit, hardware, and combined-fault adjudication. C15 is not a fourth ML class: its truth remains multi-label, and its categorical result is scored through the evidence-based ambiguous disposition, not by forcing the ML three-class output to be correct.

Evidence for a compatible interface exists in `src/quantum_simulator/ml_diagnosis.py`: `FEATURE_NAMES` is fixed and ordered; `extract_observable_features` computes it from expected/observed distributions; `diagnose_with_ml` accepts that classifier and then delegates to the rule diagnosis/localization path. `ml_training.py` defines `PRODUCTION_FIVE_OBSERVABLES` as the same first five observables and checks its order against the diagnosis adapter. This schema uses no labels, intervention fields, family identity, seed, or result artifacts.

### Ordered feature schema to freeze

Use exactly the existing five names in this order. Do not add calibration, circuit-signature, shot, or identifier features to the learned vector; circuits and calibration remain separate inputs to the evidence-based diagnosis/localizers.

| Order | Feature | Existing transformation | Required inputs and missing-value policy |
|---:|---|---|---|
| 1 | `total_variation_distance` | Existing `compare_distributions(expected, observed)` TVD: half the L1 distance over aligned outcomes | Expected reference and observed normalized distribution required; invalid/missing input blocks the row |
| 2 | `js_divergence` | Jensen–Shannon divergence returned by the same distribution comparison implementation | Same required distributions; finite result required |
| 3 | `hellinger_distance` | Hellinger distance returned by the same comparison implementation | Same required distributions; finite result required |
| 4 | `observed_entropy_bits` | `-sum(p * log2(p))` over positive probabilities in the normalized observed distribution | Observed distribution required; zero probabilities contribute zero; nonempty valid distribution required |
| 5 | `observed_max_probability` | Maximum normalized observed outcome probability | Observed distribution required; nonempty valid distribution required |

Feature computation uses the Phase 5.4 exact expected distribution and finite-shot observed distribution. It does not use `shots`, `threshold`, circuits, or `hardware_model` as feature values. `shots` and fixed threshold still go to rule-based statistics; expected/observed circuits and calibration still go to rule-based diagnosis/localization. The test-family label, case and family IDs, split, intervention/target/strength, expected D status, seed, file path, and predictions are prohibited from the learned input and remain only in the segregated audit ledger.

The current training pipeline's production LR path is `SimpleImputer(strategy="median")` then `StandardScaler()` then `LogisticRegression(max_iter=2000, class_weight="balanced")`; the portable observable adapter requires a Logistic Regression artifact with this exact five-feature order and applies the stored imputer/scaler statistics. Proposed training transform is fitted on F1–F2 only and frozen. Since all five values are derivable from required valid distributions, benchmark preflight should reject missing or nonfinite features; median imputation must be a no-op for benchmark rows, not a way to conceal missing evidence. No feature scaling or statistic may be fit on F3/F4/F5. F3 may fit a probability calibration mapping only if needed, never a detection threshold.

### Remaining learner decision

The feature names/transforms are specified by existing code, but there is **no genuine model artifact** at the adapter's configured path, and the ML integration report says no trained backend/artifact is integrated. `UnavailableObservableClassifier` raises when no classifier is supplied. The existing adapter is specifically Logistic Regression; this makes that the only already-defined export/runtime pairing, but no Phase 5-trained weights exist. The integration also has an existing `MIN_CONFIDENCE=0.60` operational abstention cutoff. Phase 5.4 does not explicitly adopt that cutoff or describe how it interacts with C15 ambiguous recall.

**Human approval required before runner implementation:** approve the learner route as (a) fit the existing LR pipeline using only F1–F2 and export a genuine artifact; use the current `.60` confidence cutoff unchanged; use F3 only for probability calibration if needed; and send predictions through existing evidence adjudication, or (b) clarify that the “Fit learned component” Phase 5.4 row was not intended to require an ML component and formally amend scope outside this report. This report recommends (a) to honor the authoritative contract. No training or artifact generation was performed. The configured default artifact path was checked and is absent.

## 3. Calibration preflight for both circuits

### Complete profile the reports require

A benchmark-specific preflight can be fully specified from existing `ExperimentConfig.hardware_parameters`, `HardwareModel`, and simulator fields. For each retained case, preflight must run **before** creating either hardware model or sampling:

- Require a non-null object with explicit q0–q3 records. Each must explicitly contain `readout_error`, `t1`, `t2`, `status`, and `available`; require baseline values `0.0`, `1e6`, `1e6`, `"available"`, `true`, except the named readout override on the R target for R cases or C15.
- Require one explicit gate calibration per canonical gate and **ordered operand tuple in the union of the expected and observed circuits**. Each record must explicitly contain `error`, `duration`, and `supported`; baseline `0.0`, `0.0`, `true`, except the named target gate-skip probability for G cases. This includes both sides of substitutions and all non-mutated gates. Canonicalize known aliases before checking exact tuple coverage and duplicates.
- Require `coupling_map=[]` and `edges=[]`. Empty means no edge calibration; it is not an ideal or healthy topology assertion. No edge feature or localization is eligible.
- Allow only the exact case override: R only its target readout p; G only its target gate/tuple error p; C15 only P1 observed-circuit mutation plus target readout `.15`. Every other required value remains explicit baseline. Gate `error` is the simulator's operation skip probability, not depolarizing noise.
- Validate identifiers, gate arity and tuple order, q indices in `[0,3]`, uniqueness after alias normalization, numeric finiteness, probability bounds `[0,1]`, nonnegative durations/T1/T2, exact booleans for `supported`/`available`, and allowed status values. Reject null, absent, malformed, duplicate-conflicting, extra, or unapproved fields/overrides. Do not let Python truthiness or constructor defaults coerce invalid values.
- Reconstruct/check the canonical union from both frozen circuit lists independently. Require the explicit calibration table to cover that union. Do not depend on `diagnosis_engine._calibration_gaps` to do this check.

If any required simulator calibration field is absent/invalid, mark that configuration preflight-failed and do not run it; do not pass an ideal default. If calibration is truly unavailable in another backend protocol, the Phase 5.4 contract says use `hardware_model=None`, preserve anomaly/program evidence where available, and mark hardware classification/localization unsupported. External backends are excluded from this Phase 5 run. `hardware_parameters=None` is not a retained hidden-calibration simulator condition because the same profile controls injection.

### Current code supplies only defaults/partial records, not the required guarantee

`ExperimentConfig` checks only that `hardware_parameters` is a dictionary or null. `experiments/run_experiment.py::_build_hardware_model` defaults missing `coupling_map`, `qubits`, `gates`, and `edges` to empty values; `HardwareModel` pre-populates all qubits with ideal T1/T2/readout/status/availability defaults; gate insertion defaults missing error/duration/support to `0.0/0.0/true`. Therefore current source can construct a plausible ideal-looking object without explicit calibration fields. Existing runner tests use partial profiles and do not test the Phase 5 complete-profile rule.

The model supports the required data: qubit setters store the named fields; gate calibration records preserve a gate and exact ordered tuple; edge records can remain empty. Thus both expected and observed circuit calibration coverage **can** be checked using current configuration fields without production changes. The existing diagnosis gap helper inspects only `observed_localizer_circuit or expected_localizer_circuit`, not their union, so the new preflight must inspect both circuits before calling the ordinary runner. The current generic runner does not do that and is not safe as the Phase 5 case generator without this preflight.

## 4. Blocker register and decision

| Blocker | Evidence | Proposed resolution | Human approval |
|---|---|---|---|
| Sensitivity denominator | Phase 5.4 defines moderate/severe D+ sensitivity; Phase 5.1 defines `.15/.30` as moderate/severe; Phase 5.3 supplies exact D truth/TVD; statistics code supplies strict fixed finite-shot guard | Freeze the 21-template positive denominator above; rows pass eligibility only from pre-run exact truth, declared severity, available distributions, and exact TVD exceeding the existing shot guard. Count every seed/shot outcome; misses remain FNs. Report per-shot strata. | Yes: approve C15 membership, F3-G specificity membership, and whether the `.80` acceptance test is applied per-shot or to a pooled result. |
| Learned component | Phase 5.4 split says “Fit learned component”; existing `ml_diagnosis` provides ordered five-feature path and rule-localizer integration; model report says no artifact exists | Use the five `FEATURE_NAMES` and existing LR preprocess/export adapter; train only on F1–F2; exclude C15 from 3-class fit; let evidence adjudication produce C15 ambiguity. | Yes: approve LR training route and the existing `.60` confidence cutoff; otherwise formally change the Phase 5.4 learner requirement. |
| Calibration validation | Phase 5.4 baseline field list; current config/model supports all named values but silently defaults omissions; diagnosis gap helper checks one side only | Add pure benchmark preflight over exact expected/observed circuit union before model construction; missing/invalid synthetic calibration blocks that configuration. Keep `None` only for unavailable external evidence, which remains out of scope. | No additional policy decision needed; profile values and fail-closed behavior are already stated in Phase 5.4. |

**Recommendation: NO-GO for implementing the benchmark-specific runner until the two denominator and learner decisions above are approved.** After approval, the implementation can use the frozen profile checks above without production-logic changes. This recommendation concerns runner implementation; execution still requires separate authorization and successful preflight.

## 5. Sources inspected and validation boundary

Inspected only the relevant source/report material: Phase 5.4 contract; Phase 5.6 execution specification; Phase 5.1 readiness ledger; Phase 5.3 exact-distribution/evidence contract; Phase 5 design wording on learned-vs-rule pipelines; `src/quantum_simulator/statistics.py`; `src/quantum_simulator/ml_diagnosis.py`; `src/quantum_simulator/observable_model_adapter.py`; `ml_training.py`; `ml_evaluation.py` feature schema definitions only; `experiment_config.py`; `experiments/run_experiment.py`; `src/quantum_simulator/hardware_model.py`; `src/quantum_simulator/diagnosis_engine.py`; `tests/test_experiment_runner.py` and the feature/schema test definitions in `tests/test_ml_training_artifacts.py`; and `reports/ml_diagnosis_integration_report.md`. The configured portable LR model-artifact path and new report path were checked; both were absent at inspection time.

Changed: created this new report only. Did not modify Phase 5.4, Phase 5.6, production code, tests, configs, or historical results. No simulations, dataset scans, training, benchmarks, timing, tests, IBM access, or hardware jobs were run. Validation is limited to lightweight report structure/content/whitespace checks and `git diff --check`.
