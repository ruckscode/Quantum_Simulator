# Finite-shot diagnosis: identifiability and evidence limits

Date: 2026-10-13

## Assessment

The project should stop tuning scalar TVD thresholds as its primary research path and investigate a narrower question: **which quantum fault claims can be supported from finite-shot output distributions alone, which require source or hardware evidence, and when should a diagnosis abstain?** This is a conditional research go, not a finding that the project already has a novel or validated method. Existing artifacts establish a clear measurement limitation and several implementation limitations; they do not establish general limits over quantum circuits or real devices.

The strongest rigorous point is identifiability. If two causal mechanisms induce the same measured outcome distribution, no number of repeated samples from that same measurement can tell the mechanisms apart. That is an observational equivalence statement, not a claim that all program faults and hardware noise are indistinguishable. Several are distinguishable when the source circuit or independent hardware calibration is available.

## Evidence examined

Read the two candidate-redesign reports, the research-gap assessment, earlier calibration diagnostic/follow-up, detector and diagnosis source, calibration-plan generator, and saved calibration outputs and run manifest. No simulations, training, hardware calls, or changes to prior artifacts were made.

The completed calibration has 3,600 rows; its calibration split has 1,800 rows (120 each for two controls and three fault strengths at 256, 1024, and 4096 shots). All use the same four-qubit paired-CNOT/phase-tag circuit recipe. The 120 seeds/templates within a stratum are not evidence of 120 independently sampled circuit families. The 3,600 configuration count therefore cannot be treated as 3,600 independent units.

The calibration runner calls `diagnose_circuit(circuit=..., reference_circuit=..., shots=n, seed=...)`. The implementation samples the reference circuit and the submitted circuit independently, each at `n` shots, then compares their empirical distributions. So the actual statistic is a two-sample comparison. The fixed TVD rule is `TVD > max(.05, 1/sqrt(n))`; it does not account for the number of outcomes, the reference probabilities, or the fact that both distributions are sampled. Saved calibration records show fixed-guard control flags of 120/120 at 256 shots and 104/120 and 107/120 at 1024 shots. This is consistent with a sampling-noise problem and a mis-scaled guard, rather than evidence that controls physically changed.

An approximate scale check explains why the guard is inadequate. For two independent multinomial samples of size `n`, with `K` roughly equiprobable outcomes and probabilities away from zero, a normal approximation gives

`E[TVD] ≈ sqrt((K−1)/(π n))`.

For `K=16`, this is about 0.137 at 256 shots, 0.068 at 1024, and 0.034 at 4096. The current cutoffs are 0.0625, 0.05, and 0.05. These approximations are not fitted null quantiles and the saved reference probabilities are not uniform; they demonstrate the scale mismatch only. The 16-category, two-sample mechanism plus the guard design is a stronger explanation of the observed low-shot false alarms than an unidentified source-code defect. A calibrated test exists in the project, but the saved permutation results show reduced subtle-fault sensitivity at low shots. Thus both statistical power and implementation/decision design matter.

The calibration generator creates two controls: an identical program and a program with a `Z` inserted immediately before computational-basis measurement on qubit 3. Since `Z` is diagonal in that measurement basis, it changes basis-state phases but not measurement probabilities. This is an exact semantics-based negative control. The saved expected distribution fields are finite-shot estimates, however, so the two rows need not have identical empirical vectors. The suite therefore contains an exact measurement-equivalence construction, but it does not compare exact probability vectors as if they were sampled truth.

Saved calibration rows also show permutation + TVD gate subtle sensitivity of 14/120 at 256, 36/120 at 1024, and 113/120 at 4096 (one-sided 95% CP lower bounds .072, .231, .893). The last bound is below the existing .90 floor. The G-test, G+gate, and Bonferroni results in the candidate reports do not establish an adequate alternative. The Bonferroni procedure's abstention is due to conservative simultaneous coordinate intervals and weak separation relative to its .05 margin; no evidence isolates one sole cause.

## Identifiability: what the existing evidence can and cannot distinguish

### Exact observational-equivalence argument

Let `G` be a gate in the intended circuit and `U` the operation in a program-mutant circuit. The program fault applies `U` in place of `G`. A hardware-only coherent error on the original program can instead apply the unitary channel with `V = U G†` immediately after `G`. In both cases, the net operation is `U`, for every input state. Therefore the final states and every possible measurement distribution are identical. For any output-only decision rule `δ` on `n` shots, the distribution of `δ` is identical under these two causes. If the two causes have equal prior probability, no output-only classifier can exceed 50% cause accuracy on this matched pair, even with infinitely many shots. This follows because the likelihoods are identical; it does not rely on an approximate power argument.

The construction is a principled identifiability control, not evidence that the same coherent error is common on devices. A separate hardware-noise model should represent empirically motivated stochastic/coherent errors, and those results must not be conflated with this exact matched case.

A second exact distinction is that multiple physical processes can map to the same classical output probabilities after a single measurement basis. Distinct final quantum states can have the same diagonal probabilities. Output counts alone do not identify phase information that the measurement discards. Additional measurement bases, syndrome/ancilla outcomes, calibration experiments, or execution telemetry can change the observable set.

### What is distinguishable with project observables

- **Program source fault:** When the expected/reference circuit and actual submitted circuit are both available, a direct operation/parameter mismatch is observable from their syntax. The current operation-alignment localizer can flag inserted, deleted, substituted, or changed-parameter gates. This can distinguish the known injected `H`→`RY` mutation from an unchanged submitted circuit, even when output distributions overlap. It is source evidence, not proof that the mutation caused an observed deviation.
- **Hardware anomaly:** Output-distribution deviation alone detects behavior inconsistent with a reference model but does not attribute cause. `diagnose_circuit` accepts an optional `HardwareModel` and the engine can surface calibration-based hardware candidates. This is a heuristic evidence path, not a validated causal discriminator in the finite-shot calibration. The calibration runner does not pass a hardware model and has no hardware telemetry or hardware-noise fault labels.
- **Ambiguous/co-occurring causes:** A source mismatch plus calibration degradation can produce competing causes; the engine can return ambiguous/insufficient categories. Correctness, calibration, and coverage of those categories have not been established on saved outcomes.
- **Localization:** Direct syntactic mismatches can be mapped to an operation index in the generated mutation. Trace-based “largest deviation” localization is heuristic and explicitly does not identify whether the cause was program or hardware. The saved finite-shot output schema does not preserve scored localization predictions, and the reported calibration metrics do not measure target accuracy. Hardware component localization requires calibration evidence and ground-truth hardware component labels, neither present in these records.

Thus program-vs-hardware causes are distinguishable when independent evidence is available and points uniquely to one cause; they are not generally identifiable from one output distribution. More shots reduce uncertainty about the distribution but cannot recover information removed by the measurement or distinguish exactly equal likelihoods.

## Limitation classification

| Candidate explanation | What saved evidence supports | What remains uncertain |
|---|---|---|
| Statistical indistinguishability | Exact Z-before-Z-measurement control is distribution-preserving. A constructed hardware channel can exactly reproduce a program-mutant unitary. | Exact cause-equivalent hardware pairs were not part of the saved calibration; the general frequency of near-equivalence is unknown. |
| Insufficient shot budget | Subtle sensitivity of permutation-based rules is very low at 256/1024 and improves with shots; fixed-guard false alarms shrink at high shots. | The calibration has one circuit recipe and cannot establish the shot budget needed across independent families. |
| Overlapping fault signatures | Fault labels use output distributions and the implementation uses direct source mismatches and hardware metadata to provide other evidence. Output-only likelihoods may overlap or coincide. | No pairwise distance/power map over program/hardware fault signatures or family-level source-localization labels was evaluated. |
| Missing hardware information | Current calibration contains program mutations and controls only; its diagnosis input omits a `hardware_model`. | Whether the hardware-localizer heuristics can reliably attribute known noise on new circuits is untested. No real hardware evidence exists in this task's artifacts. |
| Current implementation/evaluation limits | Scalar `1/sqrt(n)` guard ignores support size and reference uncertainty; expected and observed are both independently sampled; ML score cutoff is uncalibrated; output records omit localization predictions. | The fraction of the TVD failure attributable to each implementation choice requires a controlled decomposition, not more threshold tuning. |

## Novelty and related work

The central pieces are established. Huang and Martonosi's *Statistical Assertions for Validating Patterns and Finding Bugs in Quantum Programs* (ISCA 2019, DOI [10.1145/3307650.3322213](https://doi.org/10.1145/3307650.3322213)) already use statistical tests on classical observations to validate quantum-program behavior and find bugs. Li et al.'s Proq, *Projection-based Runtime Assertions for Testing and Debugging Quantum Programs* (PACMPL/OOPSLA 2020, Article 150, DOI [10.1145/3428218](https://doi.org/10.1145/3428218)), develops runtime assertions and argues about bug location and semantic closeness with projective measurements. These rule out novelty from “statistical finite-shot testing” or “quantum diagnosis can abstain when unsure.”

Li et al., *Diagnosis of Quantum Circuits in the NISQ Era* (IEEE VTS 2023, DOI [10.1109/VTS56346.2023.10140030](https://doi.org/10.1109/VTS56346.2023.10140030)), is the closest diagnosis competitor. It uses fault dictionaries of output probability distributions and dynamic binary search for fault location; its paper reports injected unitary/non-unitary faults across 15 benchmarks and IBM-device demonstrations. A contribution cannot claim fault diagnosis or location itself as new. It would have to show a distinct, calibrated contract for supported versus ambiguous cause claims, with explicit information sources and family-held-out tests. Whether Li et al.'s full implementation includes abstention/ambiguity semantics needs direct full-text comparison before making a comparative claim.

Wang et al.'s QDiff, *Differential Testing of Quantum Software Stacks* (ASE 2021, DOI [10.1109/ASE51524.2021.9678792](https://doi.org/10.1109/ASE51524.2021.9678792)), uses semantics-preserving program variants and output differences to find software-stack discrepancies. This establishes that output disagreements across platforms are an important testing signal, but not that a single output distribution identifies its cause. Mutation-based quantum fault localization has also been studied by Ishimoto et al. (2025 preprint, [arXiv:2505.09059](https://arxiv.org/abs/2505.09059)); its artificial and real-fault outcomes differ, so it is relevant to source ground truth and external validity.

Selective classification and reject-option learning are established beyond quantum computing. Geifman and El-Yaniv's *SelectiveNet* (ICML 2019, PMLR 97:2151–2159, [PMLR record](https://proceedings.mlr.press/v97/geifman19a.html)) optimizes a risk/coverage trade-off and includes rejection. Therefore “diagnosis with abstention” is not by itself novel. The potential distinction is a **quantum fault-diagnosis evidence contract** that separates (i) distribution anomaly, (ii) causal fault class, and (iii) source/hardware target, proves and tests when the currently available observables do not identify a cause, and measures selective correctness as evidence sources are added. This is an interpretation of the gap, not a verified priority claim.

**Claims that survive:** the saved data expose a low-shot two-sample finite-shot failure and a candidate identifiability question; exact observational equivalence can be constructed; the current pipeline has separate source and hardware evidence interfaces whose accuracy is unmeasured.

**Claims that do not survive:** first quantum diagnosis with statistical tests; first selective quantum diagnosis; new fault localization; general impossibility of program/hardware attribution; validated calibrated abstention; performance on real hardware; generalization from 3,600 configurations; source localization accuracy; or a validated finite-shot detector.

**Literature work still required:** full-text review of Li et al. and any diagnosis method with explicit ambiguity sets; systematic search for quantum fault diagnosis under observational equivalence, causal attribution, selective/conformal prediction for quantum software, and hardware calibration-assisted localization; verify newer 2025–2026 publications and compare their evidence assumptions. The search so far is sufficient to reject broad “first” claims, not to prove there is no close method.

## Conclusion

The most defensible scientific direction is not another cutoff search. It is a bounded study of **diagnostic identifiability and evidence sufficiency**: output-only detection versus cause attribution, with source/hardware evidence and a calibrated abstain option. Existing calibration motivates the question but cannot answer it. The next report proposes a small new family-level study and gives a conditional go recommendation. No confirmatory run is authorized by this analysis.
