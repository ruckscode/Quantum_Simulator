# Phase 5.6 — Reduced Independent Simulator Benchmark Execution Specification

**Status: NO-GO for execution.** This is a read-only-reviewed specification. It authorizes no benchmark run. The Phase 5.4 contract remains authoritative. The 60 case IDs and their intervention/truth/eligibility mapping can be reconstructed from the Phase 5.1 family ledger and Phase 5.3 evidence review without adding cases. The current runner does not enforce the required calibration preflight, and the feature/acceptance implementation gaps listed below must be closed before any later execution authorization.

## 1. Authority, scope, and execution gate

Phase 5.4 says **GO only for a later, separately authorized reduced custom-simulator benchmark**. It excludes physical backends, unknown-cause/evidence-withheld diagnosis (U1/U2), coupling-edge localization, and deployment claims. Execution additionally requires a frozen family split, source-visible circuit inputs, the explicit full calibration profile for every retained case, fixed thresholds, exclusions, and a separate explicit execution authorization. This report does not supply that authorization.

Retain exactly the 12 suffixes `N0`, `P1`, `P2`, `R02`, `R06`, `R15`, `R30`, `G02`, `G06`, `G15`, `G30`, `C15` in each of five Phase 5 families. U1 and U2 are not cases in this run. The scope is the existing four-qubit custom simulator and diagnosis API. The hardware profile is synthetic, simulator-visible calibration shared with injection; it is privileged evidence, not independent measurement. No physical topology, coupling localization, backend, or broad population claim is supported.

## 2. Exact 60-case intervention and scoring ledger

The following family rows enumerate the complete case ID set. In each row, all 12 listed IDs are distinct cases. Each case is completed by joining (a) its family-specific program and hardware targets below with (b) the suffix contract in the next table. This is the exact Cartesian product supported by the Phase 5.1 ledger and Phase 5.4 retained-suffix rule; it adds no new intervention. Case IDs are formed exactly as family_id, a hyphen, then suffix; for readability, each family row writes the full ID for N0 and abbreviates each remaining ID as -suffix with that row family prefix. For example, F2_TOFFOLI_COMPUTE-P1 is the second F2 case.

| Family | Cases (12) |
|---|---|
| `F1_GHZ_CHAIN` | `F1_GHZ_CHAIN-N0`, `-P1`, `-P2`, `-R02`, `-R06`, `-R15`, `-R30`, `-G02`, `-G06`, `-G15`, `-G30`, `-C15` |
| `F2_TOFFOLI_COMPUTE` | `F2_TOFFOLI_COMPUTE-N0`, `-P1`, `-P2`, `-R02`, `-R06`, `-R15`, `-R30`, `-G02`, `-G06`, `-G15`, `-G30`, `-C15` |
| `F3_SWAP_ROUTING` | `F3_SWAP_ROUTING-N0`, `-P1`, `-P2`, `-R02`, `-R06`, `-R15`, `-R30`, `-G02`, `-G06`, `-G15`, `-G30`, `-C15` |
| `F4_PARITY_TREE` | `F4_PARITY_TREE-N0`, `-P1`, `-P2`, `-R02`, `-R06`, `-R15`, `-R30`, `-G02`, `-G06`, `-G15`, `-G30`, `-C15` |
| `F5_PHASE_LAYER` | `F5_PHASE_LAYER-N0`, `-P1`, `-P2`, `-R02`, `-R06`, `-R15`, `-R30`, `-G02`, `-G06`, `-G15`, `-G30`, `-C15` |

Every expected circuit is the listed family circuit, measured on q0–q3. The family-specific intervention map is:

| Family | Expected (reference) circuit | P1 observed-program intervention | P2 observed-program intervention | R target qubit | G target gate and ordered tuple |
|---|---|---|---|---:|---|
| F1 | `H(0), CX(0,1), CX(1,2), CX(2,3)` | Delete `CX(1,2)` | Substitute `CZ(1,2)` for `CX(1,2)` | q2 | `CX(2,3)` |
| F2 | `X(0), X(1), CCX(0,1,2), CX(2,3)` | Delete `CCX(0,1,2)` | Change CCX target: `CCX(0,1,2)` → `CCX(0,1,3)` | q2 | `CCX(0,1,2)` |
| F3 | `X(0), X(2), SWAP(0,1), SWAP(1,2), SWAP(2,3)` | Delete `SWAP(1,2)` | Substitute `I(1)` for `SWAP(1,2)` | q1 | `SWAP(1,2)` |
| F4 | `X(0), H(1), CX(0,1), CX(1,2), CX(1,3), H(1)` | Delete `CX(1,3)` | Substitute `CZ(1,3)` for `CX(1,3)` | q3 | `CX(1,3)` |
| F5 | `H(0), T(0), H(0), CX(0,1), H(2), T(2), H(2)` | Delete `T(2)` | Substitute `Tdg(2)` for `T(2)` | q1 | `T(2)` |

`R02/R06/R15/R30` mean readout probability `p=.02/.06/.15/.30` at that family’s R target, with no program mutation. `G02/G06/G15/G30` mean gate-skip probability `p=.02/.06/.15/.30` at that family’s exact G gate/tuple, with no program mutation. `C15` means the family’s P1 deletion **plus** readout `p=.15` on that family’s R target. In N0/P1/P2/R/G, the opposite intervention channel is absent. All cases retain the same explicit baseline profile except these declared overrides.

| Suffix | Constructed truth (ledger truth, not prediction or sampled result) | Evidence available to diagnosis | Eligible metrics and case-specific qualifications |
|---|---|---|---|
| N0 | `NORMAL`; no intervention | Expected and observed circuit are the same; exact ideal reference; sampled observed counts/distribution and actual shots; explicit baseline calibration profile | Negative-control output specificity; normal-class metrics; no localization; no abstention positive |
| P1 | `PROGRAM_FAULT`; family P1 deletion only | Expected and observed circuits; exact reference and sampled observed counts/distribution; explicit baseline calibration profile | Output anomaly against exact D truth; program-vs-hardware/normal classification; program operation/signature localization. F3 P1 is D− and signature-only, not output-causal. |
| P2 | `PROGRAM_FAULT`; family P2 substitution only | Expected and observed circuits; exact reference and sampled observed counts/distribution; explicit baseline calibration profile | Same as P1. F3 P2 (SWAP→I) and F5 P2 (T→Tdg) are D− signature-only program controls, never positive output-anomaly cases. |
| R02 | `HARDWARE_ANOMALY`; readout p=.02 at family R target | Same circuit pair; reference and sampled observed counts/distribution; complete hardware model exposing target readout calibration p=.02 | Output detection and constructed hardware-class metrics, descriptive at p=.02; no qubit localization (strict source cutoff `> .05`); no abstention |
| R06 | `HARDWARE_ANOMALY`; readout p=.06 at family R target | Same circuit pair; reference and sampled observed counts/distribution; complete hardware model exposing target readout calibration p=.06 | Output detection and hardware-class metrics; supported qubit localization; no abstention |
| R15 | `HARDWARE_ANOMALY`; readout p=.15 at family R target | Same circuit pair; reference and sampled observed counts/distribution; complete hardware model exposing target readout calibration p=.15 | Output detection and hardware-class metrics; supported qubit localization; no abstention |
| R30 | `HARDWARE_ANOMALY`; readout p=.30 at family R target | Same circuit pair; reference and sampled observed counts/distribution; complete hardware model exposing target readout calibration p=.30 | Output detection and hardware-class metrics; supported qubit localization; no abstention |
| G02 | `HARDWARE_ANOMALY`; gate skip p=.02 at family G target | Same circuit pair; reference and sampled observed counts/distribution; complete hardware model exposing target gate name, exact ordered tuple, and error p=.02 | Output detection and constructed hardware-class metrics, descriptive at p=.02; no gate localization (strict `> .05`); no coupling localization |
| G06 | `HARDWARE_ANOMALY`; gate skip p=.06 at family G target | Same circuit pair; reference and sampled observed counts/distribution; complete hardware model exposing target gate name, exact ordered tuple, and error p=.06 | Output detection and hardware-class metrics; supported gate-tuple localization from calibration. F2 CCX target is the full `(q0,q1,q2)` base calibration candidate only; no trace-assisted CCX claim. |
| G15 | `HARDWARE_ANOMALY`; gate skip p=.15 at family G target | Same circuit pair; reference and sampled observed counts/distribution; complete hardware model exposing target gate name, exact ordered tuple, and error p=.15 | Same as G06 |
| G30 | `HARDWARE_ANOMALY`; gate skip p=.30 at family G target | Same circuit pair; reference and sampled observed counts/distribution; complete hardware model exposing target gate name, exact ordered tuple, and error p=.30 | Same as G06 |
| C15 | Multi-label truth `{PROGRAM_FAULT, HARDWARE_ANOMALY}`; P1 deletion plus readout p=.15 | Expected/observed circuit pair; reference and sampled observed counts/distribution; complete profile exposing readout p=.15; both program and hardware evidence channels remain visible | Output anomaly; multi-label recovery of both injected causes/targets when the corresponding API candidates exist; known-combined categorical `AMBIGUOUS` recall and coverage. A single-cause prediction is not correct combined recovery. Never call this unknown-cause abstention. No coupling target. |

**Exact-distribution and detection truth:** `D+` means the ideal intervention output distribution differs from the expected-circuit reference. `D−` means exact equality. This is constructed probability truth, not the finite-shot detection result. Phase 5.3 specifies R cases as D+ for p>0; gate cases as D+ for p>0 except F3 G cases, which are D−; F1/F2/F4/F5 P1/P2 as D+ except F3 P1/P2 and F5 P2, which are D−; and all C15 cases as D+. It also records that F1/F4 gate exact TVD is `p/2`, F2 is `p`, F3 is zero, F5 is `((2-sqrt(2))/4)*p`, and C15 is .15. These formulas are not empirical detection outcomes.

**Explicit denominator caution:** the Phase 5.4 phrase “identifiable moderate/severe D+ cases” does not freeze a complete case-by-shot denominator. Phase 5.3 says F5 gate cases remain below the fixed .05 threshold even at p=.30; F1/F4 G06 have exact TVD .03; and at 256 shots the effective threshold is .0625. The evaluator must not silently count these as threshold-detectable positives or remove them after seeing outcomes. Before implementation is frozen, define the eligibility rule directly from the existing exact distributions and fixed threshold policy, then publish the resulting case-by-shot denominator. This is a specification blocker; it does not authorize altering the threshold or acceptance target.

### Split assignment and mechanical checks

| Partition | Complete family set | Eligible use |
|---|---|---|
| Development/train | `{F1_GHZ_CHAIN, F2_TOFFOLI_COMPUTE}` | Fit any learned component and development-only transformations |
| Calibration | `{F3_SWAP_ROUTING}` | Probability calibration mapping only, if used; no threshold search |
| Locked test | `{F4_PARITY_TREE, F5_PHASE_LAYER}` | One final evaluation after schema/model/calibration freeze; report each family separately |

The generator/preflight must mechanically assert: (1) the case ID set equals the 60 IDs in the family table, with exactly 12 unique suffixes per family; (2) every case has exactly one immutable family key; (3) `family_to_partition` is the mapping above; (4) the sets of family IDs in each pair of partitions have empty intersection and their union is exactly F1–F5; (5) every case, mutation, strength, seed, shot level, and pair stays under its one family; and (6) no F4/F5 outputs are read for model, feature, calibration-map, or threshold selection. Fail closed on any mismatch. F3 is calibration only for a probability map, not threshold selection.

## 3. Calibration contract and current implementation audit

### Required input profile and validation

Every retained synthetic case must provide an explicit `hardware_parameters` object before either simulator injection or diagnosis model construction:

1. Exactly four qubit records, q0–q3. Each explicitly contains `readout_error`, `t1`, `t2`, `status`, and `available`. Baseline values are `0.0`, `1e6`, `1e6`, `"available"`, and `true`, respectively. Validate probability range `[0,1]`; finite nonnegative T1/T2; exact allowed status and boolean availability; and all four expected indices. Reject absent, null, nonfinite, mistyped, or out-of-range values.
2. An explicit calibration for every canonical gate and exact ordered operand tuple in the **union** of the expected and observed circuits. Each record explicitly includes `error`, `duration`, and `supported`, with baseline `0.0`, `0.0`, `true`; validate error `[0,1]`, finite nonnegative duration, boolean support, valid gate/operand arity and in-range operands. Canonical aliases must be normalized consistently (`cnot`→`cx`, `id`/`identity`→`i`, `toffoli`→`ccx`) before duplicate/missing-entry checks. Require both sides of substitutions (e.g. F1/F4 CX and CZ, F2 CCX target q2 and q3, F3 SWAP and I, F5 T and Tdg). Measurement/barrier instructions do not require gate calibration.
3. Fault overrides are exact and exclusive: R cases override only the declared target qubit readout probability; G cases override only the declared target canonical gate/ordered tuple error (simulator interpretation: gate-skip probability); C15 combines the family's P1 observed-circuit mutation with only its target readout `.15`. All other fields remain explicit baseline. No injected target/strength metadata is passed as an extra model feature.
4. Explicit `coupling_map=[]` and `edges=[]` for every case. This denotes absent edge-calibration evidence, not a healthy topology. No coupling localization may be scored. Do not infer edges from two-qubit operands or add default edges.
5. Fail before constructing either model or running any shot if a required field is absent/invalid, if an unexpected override occurs, or if expected/observed circuit calibration coverage is incomplete. Missing external calibration remains `None` only in an external adapter; external backends are outside this run. Never substitute a default ideal profile for missing evidence.

### Source comparison: what exists today

`experiments/run_experiment.py::_build_hardware_model` accepts `hardware_parameters=None` as no model, but when given a partial object it uses `parameters.get(..., {})` or defaults. It constructs `HardwareModel` with `coupling_map` defaulting to `[]`; that constructor pre-populates all qubits with ideal T1/T2/readout/status/availability defaults. Gate records are optional, and `add_gate_calibration` defaults error/duration/support to `0.0/0.0/true`. Thus absent required profile values can silently become ideal/default values. `ExperimentConfig` validates that `hardware_parameters` is a dictionary or null, but does not validate this nested profile. Existing `test_experiment_runner.py` uses partial one-qubit profiles and tests generic execution, not this Phase 5 profile contract.

The runner separately builds a diagnosis `HardwareModel` and a simulator `SimulatorHardwareModel` from the same `hardware_parameters`, so the calibration is inference-visible and correlated with injection as the report discloses. It converts observed counts to normalized probabilities and passes distributions, circuit-localization views, model, threshold, and shots to `diagnose_execution`. However, `diagnosis_engine._calibration_gaps` is called on `observed_localizer_circuit or expected_localizer_circuit`, so it checks only one side of a program substitution. The required independent union check for both circuits is absent. The ordinary runner has no case ledger generator, preflight, or locked split enforcement. The existing code therefore **does not meet** this section until benchmark-specific fail-closed validation is added.

## 4. Frozen inference schema

This benchmark does not define or authorize a separate learned classifier or an engineered ML feature matrix. The only defensible frozen schema derived from Phase 5.4 and the current diagnosis call is the ordered native `diagnose_execution` input record below. Preserve each as a typed structured input; do not flatten variable-length distributions/circuits/calibrations into guessed columns. If implementation requires a learned feature vector, that vector remains **unresolved** and execution is NO-GO until its exact columns and transforms are separately reviewed and frozen.

| Order | Native input | Frozen transform/representation | Availability at inference and missing-value rule |
|---:|---|---|---|
| 1 | `expected_distribution` | Exact ideal statevector probabilities for the expected program, marginalized to measured qubits; sorted outcome keys and normalized probabilities. This is computed from the declared reference circuit, without noise or sampling. | Available by the custom-runner contract. Missing/non-normalized distribution or key/width mismatch blocks the case. Do not derive it from the observed samples or truth label. |
| 2 | `observed_distribution` | `observed_counts[k] / sum(counts)` for every observed outcome, with zero probabilities inserted for reference outcomes absent from counts; sorted common outcome keys. | Available from the seeded custom simulator at the configured shots. Counts are sampled outcomes, not truth. Missing counts, nonpositive total, invalid counts, or wrong shot total blocks the case. |
| 3 | `expected_circuit` | Expected operation sequence passed through the runner's measurement-localization view (measurements become barriers; gate names/ordered operands remain available to localizers). | Available in this custom simulator protocol. Missing expected program blocks classification/localization and the retained case. No family/case/target metadata may be appended. |
| 4 | `observed_circuit` | Same representation for the configured submitted/observed sequence. It equals expected for N0/R/G and is the exact frozen P1/P2 mutation for P/C15. | Available in this custom-runner contract. Missing observed program blocks the case; never reconstruct it from the truth ledger after predictions. |
| 5 | `hardware_model` | Structured model with q0–q3 required calibration records, only circuit-relevant canonical gate/ordered-tuple records, and empty edge/topology records. Preserve absent evidence as absent; for this simulator benchmark, required baseline records are explicit. | Available but privileged: same synthetic profile drives injection and diagnosis. Missing/incomplete profile blocks this simulator case; `None` is not a synthetic ideal substitute. Disclose the direct calibration channel in results. |
| 6 | `threshold` | Fixed numeric `0.05`, identical for all cases and not selected on test. | Protocol constant available to diagnosis; missing or case-varying value blocks the run. |
| 7 | `shots` | Positive integer among exactly `{256,1024,4096}`; preserve actual configured value. | Available per observation and required to calculate finite-shot guard. Missing or other value blocks the case. It is grouping/protocol metadata, not a case label or feature for a learned classifier. |

Forbidden from inference features: case/family/split IDs or names; truth class/cause; intervention kind, target, mutation, or injected strength; affected-target/expected-detection annotations; pair IDs; random seed; file path; result/evidence artifacts; prediction-derived or truth-derived encodings. These values may reside in a separate audit ledger and join only after predictions are frozen. The runner's serialized config/results include metadata fields; the benchmark adapter must construct the API payload from the allowlist rather than pass the whole result/config record.

**Feature-schema limit:** this freezes the API input contract and its transforms, not a numeric learned-feature vector. Phase 5.4 permits deterministic transformations but names none. No feature selection, aggregation, imputation, scaling, or model may be invented under this spec. The existing work is rule-based diagnosis; report those direct API predictions. Any learned component requires a new exact feature schema and separate freeze before calibration/test use.

## 5. Scoring contract: truth, observations, and predictions

Maintain three independent records per case/seed/shot: (a) predeclared intervention truth from the sealed ledger; (b) exact distribution truth `D+`/`D−` from reviewed circuit/noise probability equations; and (c) observed sampled counts, measured statistics, and API predictions. Do not relabel injected hardware/program truth based on whether a sample happened to cross a threshold. Do not use predictions to create labels.

1. **Output anomaly detection:** compare the exact expected distribution to the sampled observed distribution with TVD. Store exact `D+`/`D−`, TVD, fixed/effective threshold, and `anomaly_detected` separately. Report sensitivity, specificity, balanced accuracy by family, fault strength, and shots. N0 is negative; F3 P1/P2 and F5 P2 are masked D− negatives despite program truth; F3 G cases are D− despite hardware intervention truth. Do not count program-class success as anomaly success. Apply the denominator blocker in section 2 before computing acceptance.
2. **Cause classification:** the single-cause ledger classes are NORMAL, PROGRAM_FAULT, HARDWARE_ANOMALY for N0/P/R/G cases. C15 truth is the two-label set `{PROGRAM_FAULT,HARDWARE_ANOMALY}`, not a single cause. Report per-class precision/recall/F1 and confusion for single-cause identifiable cases, with p=.02 hardware cases shown separately. Map diagnosis `NO_ANOMALY`, `PROGRAM_FAULT`, `HARDWARE_ANOMALY` directly; preserve `AMBIGUOUS / INSUFFICIENT_EVIDENCE` and insufficient evidence as abstention dispositions, never coerce them to a single-cause class. Report calibration-visible evidence in all hardware results.
3. **Evidence-supported localization:** score only target types actually supplied by visible evidence. Program operation/signature localization uses the expected/observed circuit pair; masked F3 P1/P2 and F5 P2 support signature/index only, not output-causal localization. Qubit localization is eligible only R06/R15/R30. Gate tuple localization is eligible only G06/G15/G30; F2 CCX scores full ordered `(q0,q1,q2)` base calibration candidate, with no trace-assisted claim. For C15, score the program target and readout qubit only if both supported candidates are returned. Exclude p=.02, unsupported/absent candidates, and every edge/coupling target. Report eligible denominator and per-family numerators.
4. **Abstention:** report coverage and selective risk; score categorical `AMBIGUOUS` on known C15 cases as combined-fault abstention only. Report incorrect confident single-cause rate on C15. Do not make U1/U2 or unknown-cause abstention claims.
5. **C15 combined recovery:** separately report multi-label cause recovery and target-set recovery. Both program and hardware truths must be recovered for exact combined success; a single-cause prediction is a failure for combined recovery even if that cause is correct. Also report the single-label disposition, where `AMBIGUOUS` is the predeclared desired result. Combined truth and AMBIGUOUS prediction are not the same field.

## 6. Acceptance and fixed threshold policy

Transcribe and preserve the Phase 5.4 criteria; do not tune thresholds, change strata, or relax targets after outputs are seen.

| Measure | Phase 5.4 acceptance criterion |
|---|---|
| Output anomaly | Sensitivity `>=0.80` on identifiable moderate/severe D+ cases; specificity `>=0.90` on N0 and masked D− controls. p=.02 and below-threshold cases are descriptive, not pass/fail. |
| Single-cause classification | Macro-F1 `>=0.75` on identifiable single-cause classes; program-only and supported hardware-only recall `>=0.80`. p=.02 is a separate descriptive stratum. |
| Combined cause | Report multi-label target recovery and categorical result separately. No numeric multi-label threshold is stated; do not invent one or count a single-cause prediction as correct. |
| Localization | Exact target accuracy `>=0.70` on supported, identifiable cases. Exclude p=.02, masked output localization, and coupling localization. |
| Abstention | C15 `AMBIGUOUS` recall `>=0.80`; report coverage. No unknown-cause/evidence-withheld acceptance claim. |
| Reproducibility | Same frozen inputs and seed reproduce identical artifacts; primary moderate/severe metric seed SD `<=0.10` where denominator permits. Report per-seed scores, mean/SD/range, paired-control differences, and config/reference hashes. |

Primary statistical threshold is fixed at `0.05`. The implementation's empirical anomaly rule is strict `TVD > max(0.05, 1/sqrt(shots))`: guard `.0625` for 256 and `.05` for 1024/4096. Equality does not count as detection. This tolerance is applied to sampled outcomes; exact D truth is separate. Do not use F3 or locked-test results to choose threshold; F3 may fit a probability-calibration map only. If fixed thresholds fail any acceptance target, report failure. Do not retune.

**Acceptance blocker:** the report has no exact row-level definition for “identifiable moderate/severe D+” or a named primary metric for seed SD beyond that phrase. Phase 5.3's F5 G formula makes exact TVD below .05 at every declared strength; F1/F4 G06 have TVD .03. The implementation spec must freeze the acceptance denominator from exact distributions and the fixed rule before execution. Preserve the agreed numerical targets; if a stratum is not estimable under this rule, report it as not estimable rather than creating a substitute threshold. No Phase 5 acceptance decision is valid until denominator selection is reviewed and frozen.

Overall **GO for execution** requires all of: exact ledger/preflight and both-circuit calibration validation implemented; no unresolved schema or acceptance-denominator items; all mechanical split/schema checks pass; all metric denominators/eligibility frozen before test access; artifacts/hashes and reproducibility policy defined; and separate explicit execution authorization. Any failure or unresolved item is **NO-GO**. This specification's current result is NO-GO.

## 7. Execution accounting

The exact ledger table contains 5 families × 12 retained templates = **60 cases**. Each case is paired with 5 seeds (`20263001`–`20263005`) and 3 shot counts (`256`, `1024`, `4096`):

- Configurations: `60 × 5 × 3 = 900`.
- Shots per configuration across one shot pass: `256 + 1024 + 4096 = 5,376`.
- Runner passes per configuration: one ideal sampled pass plus one observed sampled pass = 2.
- Shot-circuit executions: `60 × 5 × 2 × 5,376 = 3,225,600`.
- Per-case executions: `5 × 2 × 5,376 = 53,760`.

This matches Phase 5.4 exactly. It excludes the ideal statevector/reference computation and diagnosis/tracing work. It assumes all 60 cases pass the design preflight and no case is silently dropped. A failed/unsupported case must be reported separately; the actual completed count must not be presented as the planned 60-case grid. No runtime estimate is supported by prior timing evidence.

## 8. Minimum implementation and test gap register (not implemented here)

| Gap | Minimum benchmark-specific addition | Required lightweight tests before execution |
|---|---|---|
| Frozen ledger materialization | Pure-data generator for exactly 60 cases from family/program/target/suffix tables; separate sealed truth ledger and allowlisted inference payload | Exact ID set/count; family-specific circuits/mutations/targets; labels and D flags; no U1/U2; expected and observed circuit hashes |
| Calibration fail-closed preflight | Validate explicit q0–q3 fields and exact canonical gate/ordered-tuple union across both circuits before model creation; enforce only named overrides and empty topology | Missing each field, bad type/range, missing expected-side or observed-side substitution calibration, duplicate/alias collision, unexpected override, wrong target all reject; full profile accepted |
| Family split guard | Immutable family-to-partition map and no cross-partition case/variant/seed/shot joins | Exact three family sets, pairwise disjointness, exhaustive coverage, each case maps once, perturbation that leaks family rejected |
| Schema/input adapter | Build only the seven ordered API inputs in section 4; separate any truth/audit IDs; no generic config/result pass-through; no invented learned features | Assert input order/types, count normalization, ideal distribution source, metadata leakage rejection, missing-input rejection, calibration visibility disclosure |
| Metric and truth evaluator | Keep intervention truth, exact D truth, sampled outcomes, predictions in separate columns; implement eligibility and C15 multilabel/categorical scoring | D− masked controls never positive truth; sampled miss does not relabel truth; C15 single-cause not combined success; unsupported localization excluded; exact per-case denominators |
| Acceptance/threshold evaluator | Fixed threshold/guard, pre-frozen moderate/severe denominator, per-family/test reporting, no test-based selection | 256/1024/4096 guard values and strict comparator; fixed threshold cannot vary by case; acceptance pass/fail and not-estimable behavior; seed SD where denominator permits |
| Reproducibility/artifact manifest | Persist case/config/reference/schema hashes, seeds/shots, predictions, counts, denominators and per-seed summaries | Same-input same-seed deterministic artifact check; changed input changes hash; report keeps IDs out of features |

Do not implement production logic or historical result changes as part of this specification task. The table is a minimum future-work register, not authorization to run or build the benchmark.

## 9. Read-only source basis and unresolved items

Reviewed sources: the Phase 5.4 final contract; Phase 5.1 readiness ledger for exact family circuits, P1/P2 mutations, and R/G targets; Phase 5.3 reference/evidence contract for D+/D− formulas and score eligibility; `experiment_config.py`; `experiments/run_experiment.py`; `src/quantum_simulator/hardware_model.py`; `src/quantum_simulator/diagnosis_engine.py`; `src/quantum_simulator/statistics.py`; and `tests/test_experiment_runner.py`. No simulations, tests, training, scans, timing, benchmark execution, or hardware jobs were run for this report.

Resolved: exact retained IDs and interventions; all five family targets; label/evidence/metric mapping including masked program cases and C15; profile field contract; source defaults/gaps; family partition and mechanical checks; API-level input allowlist; fixed thresholds and guard; planned run counts and arithmetic.

Unresolved and blocking: (1) exact benchmark-specific fail-closed preflight and ledger/scorer do not exist; (2) whether the run uses only the existing rule-based API or adds a learned component has not been frozen—if learned, exact feature columns/transforms remain unspecified; (3) row-level denominator for “identifiable moderate/severe D+” and the primary seed-SD metric is not defined, including treatment of F5 gate cases whose exact TVD stays below .05; and (4) acceptance/reproducibility artifacts are not yet implemented. Do not guess these items. **NO-GO until resolved, reviewed, and separately authorized.**

