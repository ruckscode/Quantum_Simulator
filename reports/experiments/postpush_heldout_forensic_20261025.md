# Post-push held-out diagnosis forensic review

**Review date:** 2026-10-25
**Reviewed commit:** `6b21492bff0dddb2decf3f8ca2063bfb71638ded`
**Scope:** Re-execution and code-path review of the frozen seven-case diagnosis acceptance and its forensic companion. No fixture, label, expected outcome, threshold, or evaluation data was changed.

## Reproduction and implementation path

The frozen test is `tests/test_reference_diagnosis_generalization.py::test_heldout_generalization_outcomes_and_targets`; the companion is `tests/test_reference_diagnosis_generalization.py::test_frozen_heldout_mismatches_are_explained_by_distribution_resolution`. Inputs and frozen expectations are in `tests/fixtures/reference_diagnosis_generalization_cases.json`.

`statistics.py` computes empirical TVD and flags only when TVD exceeds `max(threshold, 1/sqrt(shots))`. With threshold 0.05 and 256 shots, the cutoff is 0.0625. `diagnose_circuit` samples reference and submitted behavior on separate seeded streams, so empirical TVD can be nonzero for equal ideal distributions. In `diagnosis_engine.py`, operation differences are source evidence; behavioral program candidates require counterfactual output change above the configured gate. When that evidence is absent, an anomaly plus syntactic candidates is insufficient for a cause attribution. Hardware attribution also requires relevant calibration evidence, which these cases do not supply.

The QFT-like source edits were localized, but their ideal TVDs are below the behavior gate. For ring cases, the empirical detector flags the sample fluctuation, while equivalent-operation counterfactuals show no output change and diagnosis abstains. Detection, source localization, program attribution, hardware attribution, and abstention are separate outcomes.

## Seven frozen cases

| Case | Reproduced evidence | Classification | Primary-goal implication |
|---|---|---|---|
| `qft_like_3q_program_fault_11` | Exact TVD 0.0078559041; sampled TVD 0.01953125; source target localized; no behavioral candidate | Invalid/outdated acceptance expectation; abstention is justified under the configured evidence gate | Source-level localization succeeds, but behavioral distinction is below detector resolution. No hardware claim. |
| `qft_like_3q_program_fault_29` | Exact TVD 0.0012597207; sampled TVD 0; source target localized; no behavioral candidate | Invalid/outdated acceptance expectation; abstention is justified | Same sensitivity limitation; not evidence for a hardware fault. |
| `qft_like_3q_program_fault_47` | Exact TVD 0.0112994326; sampled TVD 0.01953125; source target localized; no behavioral candidate | Invalid/outdated acceptance expectation; abstention is justified | Same sensitivity limitation; source evidence does not establish behavioral impact. |
| `ring_cluster_4q_correct_11` | Exact TVD 0; sampled TVD 0.0859375, above 0.0625; anomaly flag; no cause isolated | Finite-shot variation / distribution shift in the samples | False positive at the detector stage; downstream abstention avoids a program-vs-hardware misattribution. |
| `ring_cluster_4q_correct_29` | Exact TVD 0; sampled TVD 0.125; anomaly flag; no cause isolated | Finite-shot variation / distribution shift in the samples | Same false-alarm limitation; no supported cause attribution. |
| `ring_cluster_4q_equivalent_11` | Exact TVD 0; sampled TVD 0.0859375; syntactic operation candidates, zero counterfactual output change, no behavioral candidate; abstains | Finite-shot variation / distribution shift in the samples | A source diff is not behavioral fault evidence; current path abstains rather than assigning `PROGRAM_FAULT`. |
| `ring_cluster_4q_equivalent_29` | Exact TVD 0; sampled TVD 0.125; same zero counterfactual change and abstention | Finite-shot variation / distribution shift in the samples | Same; no program or hardware attribution supported. |

No case demonstrates a newly introduced implementation regression. The QFT cases expose a real sensitivity limit under the frozen detector and acceptance contract; the ring cases expose detector false positives at the registered seeds. The test's frozen expected composite outcomes remain unchanged and the acceptance test remains red. The forensic test passes and explains the discrepancy. These outcomes matter to the primary goal because they show limited low-effect sensitivity and low-shot specificity, but current abstention gates prevent unsupported root-cause claims in these examples.

## Existing finite-shot comparison: scope and results

The preregistered `finite_shot_distribution_calibration` artifacts contain 120 templates in each calibration and untouched-validation split, five strata, and 256, 1,024, and 4,096 shots (3,600 diagnosis configurations per split). The compared rules are the existing fixed guard, a pooled-label two-sample TVD permutation test (999 permutations, add-one correction, alpha 0.05), and a calibration-frozen empirical q95 TVD cutoff. The result report gives exact per-cell binomial intervals. This is a synthetic multinomial count study using one paired-CNOT topology/recipe, not an independent-family circuit benchmark. Its already inspected validation is exploratory for candidate selection.

On untouched-validation controls, the current guard false-positive rates were 120/120 (100%; exact 95% CI 96.97–100%) for both control strata at 256 shots, 108/120 (90%; 83.18–94.73%) and 110/120 (91.67%; 85.21–95.93%) at 1,024 shots, and 4/120 (3.33%; 0.92–8.31%) and 1/120 (0.83%; 0.02–4.56%) at 4,096 shots. Permutation-test control FPRs were respectively 3/120 (2.50%; 0.52–7.13%) and 7/120 (5.83%; 2.38–11.65%); 4/120 (3.33%; 0.92–8.31%) and 8/120 (6.67%; 2.92–12.71%); and 7/120 (5.83%; 2.38–11.65%) and 4/120 (3.33%; 0.92–8.31%).

Permutation sensitivity on subtle cases was 15/120 (12.50%; 7.17–19.78%) at 256 shots, 45/120 (37.50%; 28.83–46.80%) at 1,024, and 111/120 (92.50%; 86.24–96.51%) at 4,096. Moderate sensitivity was 93/120 (77.50%; 68.98–84.62%) at 256 and 120/120 at 1,024 and 4,096 (each exact 95% CI 96.97–100%). Large sensitivity was 120/120 at all three shot counts (each 96.97–100%). The empirical q95 candidate reduces low-shot control flags but lowers subtle sensitivity; it is not selected as production behavior. These intervals are per registered template/stratum and do not establish cross-family transportability. The shared diagnosis `abstained` field is not a detector-specific abstention or confidence calibration measure.

This experiment is a useful offline calibration comparison, but it does not satisfy the requested broader multi-family study. A defensible next study must lock a family-diverse generator, seeds, calibration/evaluation separation, methods, operating metrics, and acceptance bounds before sampling an untouched evaluation split. No production threshold or method is changed based on the seven cases or previously inspected validation.

## Report reconciliation and verification

The 2026-10-14 JSON `heldout_diagnosis_root_cause_20261014.json` remains preserved as historical output. Its statements that equivalent-ring syntactic candidates drove diagnosis to `PROGRAM_FAULT` are inconsistent with the current code path and this re-execution; they are superseded only for current behavior by this dated review and the erratum in `reports/FINAL_RESEARCH_REPORT.md`. Historical results and fixtures were not overwritten.

Before this documentation-only reconciliation, the focused frozen/forensic command produced **1 failed, 1 passed**; the adjacent diagnosis suite produced **74 passed**; and the bounded offline suite produced **851 passed, 1 failed, 1 skipped, 2 deselected**. The sole failure in each applicable run is the frozen seven-case acceptance test. These counts are rerun after this report update before finalization. No full-dataset or IBM hardware job was run.

## 2026-10-26 implementation decision and re-evaluation

The diagnosis execution path now retains both sampled count vectors and uses a conditional two-sample TVD permutation test when those counts are available. It conditions on pooled outcome labels and both sample sizes, uses 999 resamples and alpha=0.05, applies the add-one p-value correction, and retains the configured TVD effect gate. A fixed predeclared seed makes each result reproducible. The generic distribution-only API keeps its prior behavior because normalized probabilities do not identify sample counts.

The frozen 45-case held-out test was rerun against this method. It has **five mismatching cases**: the three QFT-like program-fault cases remain undetected, and `ring_cluster_4q_correct_29` plus `ring_cluster_4q_equivalent_29` are still flagged. The two ring seed-11 false alarms are no longer flagged. All other held-out outcomes and source/hardware targets match their frozen expectations. The permutation test controls false-alarm probability at its configured level; it cannot guarantee zero false alarms on every finite set of controls.

No adaptive-shot path was added. The smallest QFT-like ideal TVD is approximately 0.00126 and the largest is approximately 0.01130, versus the preserved 0.05 effect gate. Existing calibration's subtle faults have ideal TVD 0.05–0.10; even there, permutation sensitivity was only 12.5% at 256 shots and 92.5% at 4,096 in its narrow recipe. Extrapolating a bounded escalation policy to the much smaller QFT effects is not validated, and a 4,096-shot cap cannot be claimed to meet these cases reliably. Raising shots without a validated stopping and decision contract would also change the frozen evaluation's stated 256-shot evidence. Source-localization and behavioral counterfactual gates remain in place, so these cases abstain rather than being promoted to program faults.

Focused verification after preserving the hash-pinned `statistics.py`: **87 passed, 2 failed, 1 skipped** across statistics, diagnosis-engine, hash-binding, held-out generalization, and reference-validation tests. The two failures are the five-case held-out mismatch and the separate frozen `sampling_difference_without_causal_evidence` expected outcome, which the calibrated test does not flag. The forensic companion and both hash-binding checks pass. The full suite completed with **898 passed, 2 failed, 1 skipped** in 581.18 seconds; the same two frozen-outcome failures remain. These checks are simulation-only; no real-hardware validation was performed.

## 2026-10-26 v2 acceptance finalization

Added `tests/fixtures/reference_diagnosis_acceptance_v2.json` as the active replay contract. It records the detector configuration and six explicit replay expectation overrides justified above: three QFT source-localized abstentions, two significant but unattributed ring sample deviations, and one four-shot sampling fluctuation classified as no anomaly. The original v1 JSON fixtures remain unchanged and their byte hashes are verified by test. This is a versioned contract replacement; the v1 expected outcomes did **not** pass and are not relabeled as passing.

Focused results: statistics plus finite-shot calibration **23 passed**; diagnosis, evidence-gate, counterfactual, and localizer tests **122 passed**; generalization and reference-validation v2 tests **6 passed**. Final full suite: **901 passed, 1 skipped** in 590.23 seconds. Two `RuntimeWarning`s were emitted by the legacy exploratory permutation helper in `test_finite_shot_distribution_calibration.py`; they did not fail the suite. The v2 replay contract is validated in simulation only, not on real quantum hardware.
