# Phase 4 — End-to-End Diagnosis Reliability Audit

## Scope and evidence

This read-only audit traced the current canonical implementation and reviewed saved Phase 1, Phase 1.3, Phase 1.4, and Phase 2 results. No simulations, datasets, or historical reports were regenerated. All experiment profiles are simulator-side calibration inputs; none of these results is physical-hardware validation.

Primary implementation inspected:

- `experiments/run_experiment.py` — execution orchestration, ideal and hardware-model shot paths, distributions, diagnosis invocation, and saved result construction.
- `src/quantum_simulator/diagnosis_engine.py` — category selection, anomaly result, missing evidence, and candidate filtering.
- `src/quantum_simulator/statistics.py` — distribution comparison and finite-shot guard.
- `src/quantum_simulator/fault_localizer.py` — direct operation mismatch and simulator trace localization.
- `src/quantum_simulator/hardware_fault_localizer.py` and `src/quantum_simulator/hardware_model.py` — calibration scoring, candidate relevance, and simulated hardware behavior.
- `experiment_config.py`, `experiments/fault_isolation_phase1.py`, and `experiments/fault_isolation_phase2.py` — scenario metadata, independently declared scenario truth, and result aggregation.

Saved evidence inspected:

- `reports/experiments/controlled_fault_isolation_phase1/REPORT.md` and its `phase1_results.json` / case JSON files.
- `reports/experiments/controlled_fault_isolation_phase1_3/REPORT.md` and `phase1_results.json`.
- `reports/experiments/controlled_fault_isolation_phase1_4/REPORT.md`, `scenario_seed_results.csv`, and `scenario_variability.csv`.
- `reports/experiments/independent_generalization_phase2/REPORT.md`, `aggregate_metrics.json`, `scenario_manifest.json`, `scenario_seed_results.csv`, and case JSON files.

## Pipeline and evidence flow

1. `run_experiment()` resolves the expected circuit and optional observed circuit. It builds one calibration model for diagnosis and a simulator hardware model for execution.
2. The ideal statevector and ideal seeded shots are generated from the expected circuit. Observed shots use the observed circuit and optional simulated hardware model, unless `observed_counts` explicitly supplies a controlled observation fixture.
3. Counts are normalized into expected and observed distributions. `diagnose_execution()` receives those distributions, the expected and observed circuits, hardware calibration, threshold, and shot count.
4. `compare_distributions()` determines the statistical anomaly flag using the distributions and finite-shot guard.
5. `localize_program_fault()` compares the operation lists and may separately run ideal simulator traces. Direct operation differences are program evidence; a trace-only deviation is not promoted to a program fault by itself.
6. `localize_hardware_fault()` scores calibration entries and can add trace-associated locations. The diagnosis engine promotes only candidates with recognized calibration-fault evidence that also concern components used by the circuit.
7. The final category combines statistical anomaly status with direct program evidence and relevant calibration evidence. If an execution anomaly has neither cause, the result is insufficient evidence. If both causes are supported while anomalous, the result is ambiguous.

The scenario `ground_truth_label` and related identifiers are present in configuration/result metadata and are used by scenario validation/grouping. Inspection of the call to `diagnose_execution()` shows these fields are not passed to the diagnosis engine. The saved Phase 1/Phase 2 generators declare truth from scenario construction before execution rather than from predictions. This is independent of the prediction, but it remains simulator scenario truth, not external adjudication. A future ML extractor must continue to exclude ground-truth and label-derived metadata.

Seeds are passed to both ideal and observed shot simulators. Controlled phase repetitions fix shot count and vary the seed. The localization trace itself is deterministic ideal simulation, not a replay of the sampled hardware trajectory; its location should therefore be interpreted as a structural clue, not independent causal proof. Calibration evidence and execution-distribution evidence are distinguishable in the diagnosis result, but aggregate class metrics can obscure that distinction.

## Findings and classification

| Finding | Classification | Evidence and impact |
|---|---|---|
| Distribution comparison, direct mismatch, calibration scoring, relevance filtering, and uncertainty branches are implemented. | Implemented and verified | Focused diagnosis/localization tests below pass. Code paths are in `src/quantum_simulator/diagnosis_engine.py`, `fault_localizer.py`, `hardware_fault_localizer.py`, and `statistics.py`. |
| Unsupported operations skip simulation-based trace evidence and produce missing-evidence notes; direct supported mismatches can remain visible. | Implemented and verified | `tests/test_diagnosis_engine.py` and `tests/test_hardware_fault_localizer.py` cover the skip and no-unsupported-attribution boundary. |
| Long/short absolute gate duration alone is not hardware fault evidence in the current canonical localizer. | Implemented and verified | `_score_gate()` records duration as metadata but no longer uses an absolute duration threshold; focused tests cover long/short durations and real calibration error. |
| Calibration can classify an anomaly-free execution as `HARDWARE_ANOMALY` when a relevant calibration fault is independently present. | Implemented behavior; interpretation risk | Phase 2 `hardware_readout_q3_near_threshold` and `hardware_rzz_error_near_threshold` show `anomaly_detected=false` while calibration evidence supports the hardware class. This is not an output-distribution detection success. Consumers must read `anomaly_detected` separately from category. |
| Phase 1 healthy coherence control was originally misclassified. | Confirmed historical defect; corrected in later saved result | Phase 1 reports `coherence_healthy_control` as HARDWARE_ANOMALY with `anomaly_detected=false`, citing identity duration 2.0. Phase 1.3 `phase1_results.json` records it as NORMAL; Phase 1.4 reports 5/5 correct NORMAL outcomes for that control. |
| Phase 1.3 report prose still describes the old false positive despite corrected Phase 1.3 case/aggregate data. | Confirmed documentation inconsistency | `controlled_fault_isolation_phase1_3/REPORT.md` reproduces the Phase 1 warning and its old confusion matrix while `phase1_results.json` has the coherence case as NORMAL and aggregate accuracy 1.0. This audit preserves both files; the report was not rewritten. |
| Phase 2 exposes finite-shot sensitivity for subtle hardware profiles. | Confirmed limitation / observed misses | Across 36 scenario-seed cases: 22/30 positive detections, 8 false negatives, and 0/6 false positives. The report gives classification accuracy 0.917 and macro-F1 0.922; those high category metrics partly reflect calibration evidence when the distribution flag is false. |
| Phase 2 readout `0.04` and `0.055` cases are not reliably detected from outputs. | Confirmed failure pattern | `hardware_readout_q3_subthreshold` has anomaly false on all three seeds; predicted NORMAL twice and ambiguous once, with localization NOT_JUSTIFIED all three times. `hardware_readout_q3_near_threshold` has anomaly false on two of three seeds, while calibration supports the HWA category; it has one NOT_JUSTIFIED localization. |
| Subtle RZZ calibration error is classified from calibration despite no output anomaly flag. | Confirmed evidence distinction | `hardware_rzz_error_near_threshold` has anomaly false on all three seeds but HWA category and correct calibration-target localization in the saved Phase 2 table. The result supports detection of a configured calibration condition, not observable manifestation. |
| Phase 2 has family concentration and shared intervention structure. | Known limitation | The 36 results cover six declared families, but several variants share families. Per-family inspection found all three classification misses and all three NOT_JUSTIFIED localization outcomes in `phase2_cx_q1_q3_readout`; the related `phase2_rzz_phase_q0_q2` family has four detection misses across its mixed healthy/program/hardware cases. Repeated seeds are not independent circuits or generalization samples. |
| A separate `AMBIGUOUS` vs `INSUFFICIENT_EVIDENCE` serialized value is unavailable. | Known API limitation | In `DiagnosisCategory`, both names alias the existing string `AMBIGUOUS / INSUFFICIENT_EVIDENCE`. Reports can distinguish reasons through summary/evidence, but not through distinct serialized categories without a public format change. |
| Physical-hardware behavior and calibration fidelity are not established. | Known limitation | All reviewed cases use the local simulator and constructed profiles; Phase 3.5 live preflight was blocked pending an explicit backend name. No IBM access or execution occurred in this audit. |

### Historical result details

- **Phase 1:** 9 cases; accuracy 0.889. The false positive is the healthy coherence control. Its saved execution reports `anomaly_detected=false`, so this is a classification/candidate-scoring false positive rather than distribution detection.
- **Phase 1.3:** the saved `phase1_results.json` shows 9/9 correct, including coherence control NORMAL. The Markdown report appears stale/copied from Phase 1 and must not be used as the authoritative Phase 1.3 case result.
- **Phase 1.4:** 45 runs across five seeds, all classifications correct and 30/30 declared target-localizations correct. These are repeated runs of the same nine scenarios and one circuit family, not new generalization evidence.
- **Phase 2:** 36 runs, 12 scenarios × 3 seeds, with 6 circuit families. Class errors are two HARDWARE_ANOMALY→NORMAL and one HARDWARE_ANOMALY→AMBIGUOUS; the 8 detection false negatives are distinct from class errors. There were 24 correct localizations, 3 NOT_JUSTIFIED target localizations, 0 incorrect localizations, and 7 abstentions. All failure/unsupported counts were zero.

The saved Phase 2 `REPORT.md` tabulates one representative seed per scenario in the scenario table; `scenario_seed_results.csv` and `aggregate_metrics.json` are the sources for all-seed counts. Do not infer that all three repeated rows share the displayed first-seed outcome.

## Localization and uncertainty assessment

Direct operation mismatch evidence is separated from trace-only deviation. Unsupported operations are not treated as direct proof of program fault; absent distributions leave anomaly status unknown and lead to insufficient evidence. Hardware diagnosis filters calibration candidates for circuit relevance and for calibration markers such as elevated error, short coherence, unavailable qubit, or unsupported calibration status. Trace-associated hardware locations without calibration-backed evidence are not included in `hardware_evidence_candidates` used to choose a definitive hardware category.

In the saved Phase 2 outcomes, no wrong localization was reported. The three `NOT_JUSTIFIED` cases are correct uncertainty behavior: two/three subthreshold readout-seed outcomes without a justified localization and one remaining seed where the observed/evidence boundary did not justify localization. The summaries report correct localization only when a declared target is supported; these small synthetic results do not establish that candidate ranking is calibrated for real devices.

## Concrete defects and code changes

No new production-code defect was demonstrated in this audit, so no source or test files were changed. A known past defect (absolute duration treated as fault evidence) is already corrected in the current canonical `src/quantum_simulator/hardware_fault_localizer.py` implementation and protected by regression tests. Historical Phase 1 output remains unchanged.

The confirmed Phase 1.3 report inconsistency is not corrected here because this task explicitly preserves prior experiment reports/results. It is recorded as an outstanding evidence-handling issue.

## Verification

Command run using the project `.venv` and a fresh repository-local pytest temp directory:

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_diagnosis_engine.py tests/test_fault_localizer.py tests/test_hardware_fault_localizer.py tests/test_operation_trace_localization_integration.py --basetemp=.pytest-tmp-batch-20261012/phase4_20261012
```

Result: **70 passed in 1.05s**.

Command:

```powershell
git diff --check
```

Result: exit code 0. Git emitted only existing line-ending conversion warnings (LF to CRLF) for modified files; no whitespace errors were reported. The full test suite was not run.

## Remaining risks and prioritized next steps

1. **Highest priority — reconcile the stale Phase 1.3 report** in a separately authorized documentation-only follow-up using its saved `phase1_results.json` and case files. Preserve the original Phase 1 false positive as historical evidence and clearly state that Phase 1.3 is a later corrected implementation run; do not rerun either experiment.
2. Establish a predeclared, independently designed simulator evaluation over more circuit families, with matched seeds/shots and separate output-detection, calibration-condition, classification, and localization metrics. Include ambiguous program/hardware pairs and calibration-only controls.
3. For small calibration deviations, report calibration suspicion separately from observed-output detection and predeclare how the user-facing category should explain that distinction. Do not tune the existing threshold against Phase 2.
4. Keep a machine-readable evidence disposition for missing, stale, unsupported, and conflicting calibration so candidate explanations remain auditable.
5. If physical claims are required, first resolve the explicit backend selection blocker for read-only preflight, then obtain independent hardware labels and calibration snapshots under a separately approved protocol. Simulator results alone cannot satisfy this.

**Overall assessment:** the current diagnosis path has useful evidence separation and uncertainty behavior, and focused tests pass. Reliability is not established beyond the small constructed simulator cases. Phase 2 reveals missed subtle output deviations and the Phase 1.3 Markdown report conflicts with its saved corrected result artifact. No physical-hardware validation has been performed.

