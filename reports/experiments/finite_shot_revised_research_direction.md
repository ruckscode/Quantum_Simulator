# Revised research direction: finite-shot fault diagnosis and identifiability

Date: 2026-10-13

## Proposed paper direction

**Working title:** *What Can Finite-Shot Quantum Outputs Diagnose? Identifiability, Evidence, and Abstention for Program and Hardware Faults*

**Research question:** Under explicitly stated circuit, channel, and measurement assumptions, which fault claims (anomaly, program-vs-hardware cause, and source/component location) are identifiable from finite-shot output data alone, and how much does independent source or hardware-calibration evidence improve supported diagnosis without forcing a cause when the evidence is ambiguous?

### Primary hypothesis

For a controlled set of independent circuit families, an evidence-tiered diagnostic framework that separates anomaly detection from cause and localization will (a) abstain on output-only pairs whose program-fault and hardware-fault likelihoods are equal by construction, and (b) achieve a preregistered lower bound on correct supported cause-and-target diagnosis when the appropriate independent evidence is provided, while controlling false diagnoses on negative controls. It will outperform output-only baselines on **supported cause/location success at a nontrivial coverage**, not necessarily on anomaly detection alone.

The exact-equivalence part follows from the likelihood model; the performance part is empirical and remains untested. The intended distinction is an explicit evidence/identifiability contract and its evaluation, not a new generic statistical test or a new reject option.

## Proposed method contract

The diagnosis output has three evidence levels:

1. **Behavior:** `anomaly`, `consistent with reference`, or `unknown`, with a sampling-aware test and an explicit null/alternative. Non-rejection is not proof of equivalence.
2. **Cause:** `program`, `hardware`, `multiple/ambiguous`, or `unsupported`. A cause is asserted only when cause-specific evidence is present and no competing explanation remains. A distribution deviation alone cannot label program or hardware.
3. **Target:** operation/source location or hardware component, with ground-truth-defined top-1/top-k scoring. Output-distribution trace saliency by itself supports only a “suspicious region” candidate, not a causal location.

The abstention policy is not selected yet. It must be calibrated on development families only and frozen before the test split. The existing fixed TVD and permutation procedures remain unchanged baselines. No production threshold or registered criterion changes are proposed.

## Minimal decisive evaluation

### Unit and split

The independent unit is a **circuit family**: a distinct source/circuit skeleton and algorithmic structure, not a seed, a mutant, a shot repetition, or a repeated execution. Source families and all derived mutants are grouped together. The existing fixed-topology family and all fingerprints/seeds in prior calibration/validation are excluded. The study must obtain a new, versioned corpus of family skeletons with auditable provenance and ground truth; seeded variants of the old 4-qubit recipe do not count as new independent families.

Use a family-disjoint development/calibration set to select any non-theorem decision parameters, then freeze code, parameters, data hashes, and decision contract. A separate final test set contains 252 new independent families. The 252 is a planning count for the single primary operating point described below, not a claim that the current 120 templates are independent or adequate. If no adequate new family corpus is available, the experiment is a no-go; do not relabel seeds as families.

Each test family gets four controlled conditions:

- identical negative control;
- measurement-equivalent negative control (for example the existing final Z before computational-basis measurement, where valid for the circuit);
- known program fault with source-level ground truth and target operation;
- hardware-channel fault with ground-truth channel/component and unmodified intended source circuit.

Include a matched program/hardware pair for each family where the hardware-only channel after intended gate `G` is `V=UG†`, and the program mutant substitutes `U` for `G`. These induce the same ideal unitary at that location and therefore identical output distributions. Do not pass intervention labels or simulator internals to the diagnostic method. The output-only ablation must abstain on the matched cause pair; a source/hardware-evidence condition may distinguish it only using the declared permitted evidence. This is a deliberately constructed identifiability case; add realistic stochastic hardware-error cases as a separate stratum and do not generalize the exact equivalence result to them.

**Ground truth:** immutable source AST/circuit diff for program fault and operation index; hardware intervention manifest hidden from methods plus independently exposed calibration measurements for allowed hardware evidence; exact/simulator reference probabilities for the analytic oracle; shot samples from independent seeds; family IDs and all variants stored in a lineage manifest. The evaluation process must score only predictions written before the evaluator reveals labels.

### Baselines and ablations

- unchanged fixed-guard TVD detector;
- unchanged pooled permutation-TVD detector;
- current diagnosis engine, including its existing operation-mismatch and hardware-model evidence paths;
- output-only fault dictionary / nearest distribution baseline inspired by Li et al. (implement a faithful comparable version or state why not comparable);
- proposed evidence-tiered rule;
- ablations: output-only, source-diff only, hardware-calibration only, and full evidence. Keep the method and baselines frozen.

A baseline comparison is paired by family. Scores must be computed from the same sampled records. The outputs for the three shot counts are repeated within family, not independent sample units.

### Primary endpoint, thresholds, and power

Primary shot budget: **4096 shots**. Lower shot counts 256 and 1024 are prespecified secondary operating points with stratified intervals and no independent pass claim in this minimal design. A later confirmatory multi-shot claim would need multiplicity-adjusted power/sample size.

Primary endpoints by stratum:

- supported correct diagnosis: correct `program` or `hardware` class **and** correct target, with a supported evidence record; abstentions and wrong answers count as failure for this unconditional endpoint;
- control false diagnosis rate separately for each of the two controls;
- coverage and abstention reported separately; show wrong confident cause/target rate among accepted diagnoses and a risk–coverage curve;
- anomaly sensitivity/FPR, cause confusion, exact/top-k source localization, and calibration/coverage diagnostics where the method emits a score;
- per-family paired differences versus baselines; use family-cluster bootstrap or paired family-level inference, never per-shot pseudo-replication.

Planning criteria for the *new study* (not modifications to any existing registry): one-sided lower CP bound at least .80 for supported correct diagnosis in each of the two fault-cause strata; one-sided upper CP bound at most .05 for FPR in each control stratum; family-wise one-sided alpha .05 across those four gates by Bonferroni (alpha=.0125 each). Abstentions count as unsuccessful in supported-correct diagnosis, so these gates cannot be met by abstaining on everything. Coverage and abstention remain reported separately. Since supported-correct diagnoses are a subset of accepted diagnoses, a lower bound of .80 on unconditional supported success also rules out near-total abstention; a separate coverage gate is optional and must be fixed before test outcomes are seen.

The exact binomial power calculation assumes independent family-level Bernoulli indicators within each stratum. With `n=252`, a sensitivity gate `L_CP(.0125) ≥ .80` requires at least 216/252 correct supported diagnoses; a control gate `U_CP(1−.0125) ≤ .05` permits at most 5/252 false diagnoses. If true supported sensitivity is .90 and true control FPR is .01, the exact binomial pass probabilities are 0.9883 and 0.9575, respectively. Across two sensitivity and two control gates, the union bound gives joint pass probability at least `1 − 2(1−.9883) − 2(1−.9575) = .8915`, regardless of dependence between gates, provided independent families justify each stratum's binomial model. These are planning assumptions, not predictions about the proposed method. If true FPR is .02 or sensitivity .85, power is lower and must be recalculated before preregistration.

The calculation is reproducible with SciPy: `beta.ppf(alpha,x,n−x+1)` for the lower CP bound, `beta.ppf(1−alpha,y+1,n−y)` for the upper, and exact binomial tail probabilities under prespecified true rates. No finite-shot samples were generated for this report.

### Resource estimate

Primary test at one shot budget: `252 families × 4 conditions = 1,008` family-condition executions and `1,008 × 4,096 = 4,128,768` submitted-circuit shots. The current API also samples the reference independently at the same shot count, so the full diagnosis call consumes `8,257,536` shot-circuit executions in total. If the full 256/1024/4096 curve is retained, that becomes 3,024 family-condition-shot configurations and `252 × 4 × (256+1,024+4,096) = 5,419,008` submitted-circuit shots, plus the same number of reference shots: `10,838,016` total shot-circuit executions. Running a baseline on the same count records adds analysis CPU but no quantum/simulator shots. Additional calibration circuits for hardware telemetry are not included; their count depends on the calibration protocol and must be budgeted separately.

For 16 outcomes stored as 64-bit counts, a single observed count vector is 128 bytes; 1,008 vectors are about 126 KiB (expected and observed vectors together about 252 KiB), before JSON metadata and manifests. Count storage is small. Peak simulator memory depends on maximum family width and state-vector implementation; it cannot be stated until the family corpus and width are frozen. A state vector alone requires `16 × 2^q` bytes for complex128 at `q` qubits, per active state vector, excluding runtime overhead.

The saved calibration manifest reports 3,600 configurations and 12,902,400 shot-circuit executions. Its `runtime_seconds` is 3,192.3, while the manifest's UTC start-to-finish timestamps span about 8.16 hours; these do not give a clean, reproducible throughput estimate (and the prior run experienced lock-allocation and OneDrive contention). Naive scaling from those two different clocks gives approximately 34 minutes to 5.2 hours for 8.26 million total executions, before data contention or failed/retried work. This is only a very rough planning range, not a runtime promise. A future execution plan needs a small separately authorized pilot, safe resume, atomic per-configuration output, local staging or a single manifest writer, bounded workers, and a lock-allocation preflight before scheduling the full study.

## Failure and no-go criteria

Do not execute if any of these hold:

- fewer than 252 genuinely independent, eligible new circuit families with traceable provenance and a disjoint split;
- program and hardware ground truth is leaked to the method, or allowed hardware evidence is not independently calibrated;
- the method cannot distinguish “distribution anomaly” from a cause/location claim in its output contract;
- development-family calibration cannot attain useful supported-correct coverage while meeting the proposed FPR and sensitivity gates;
- a matched exact-equivalence case is confidently assigned to the wrong cause using output-only data;
- final test family or seed overlap with prior calibration/validation cannot be ruled out;
- a baseline is not implemented faithfully enough for a fair comparison;
- runtime/storage preflight cannot safely manage the prior Windows lock issue and OneDrive contention.

On held-out data, the primary contribution fails if any of the four family-wise gates fail, if the method attains the gates only by near-total abstention (the unconditional supported-success gate prevents that), or if it does not improve supported cause/location success over baselines at a matched FPR/coverage operating point. Report negative results rather than alter margins or criteria.

## Novelty audit and recommendation

A rejection/abstention action and risk–coverage analysis are established selective-classification ideas (for example Geifman & El-Yaniv, ICML 2019). Quantum statistical assertions and projection-based debugging already exist (Huang & Martonosi 2019; Li et al. 2020). Fault dictionaries and dynamic location search exist in quantum circuit diagnosis (Li et al. 2023); quantum mutation-based localization is also an active area. Therefore the paper must not claim novelty for generic test statistics, an abstention label, fault localization, or a simple fusion of source and hardware signals.

The potentially defensible contribution is a controlled, auditable characterization of **fault-cause identifiability under explicit observable sets**, paired with an evidence-qualified diagnosis contract and family-level risk/coverage/localization evaluation. The exact-equivalence construction gives a sound negative control; it does not by itself establish novelty. A stronger claim requires literature verification of whether existing quantum diagnosis work already analyzes observational equivalence and evidence-dependent abstention, and experimental evidence that the proposed method improves supported decisions at matched operating points on independent families. This literature audit is incomplete.

**Recommendation: conditional go for the revised research question; no-go for another detector-tuning run and no-go for a confirmatory simulation now.** The exact identifiability argument and source-code audit make the question worth pursuing. Existing calibration cannot support the claimed framework, and the method's unique contribution is not yet established. The next step is a focused full-text literature comparison and data/provenance feasibility check for 252 truly distinct families and ground-truth hardware interventions. If the family corpus or independent hardware evidence is unavailable, narrow the paper to a formal identifiability/negative-result study using explicitly constructed channels, and do not claim hardware realism or performance generalization.

## Evidence and literature cited

- Huang & Martonosi, “Statistical Assertions for Validating Patterns and Finding Bugs in Quantum Programs,” ISCA 2019, [DOI: 10.1145/3307650.3322213](https://doi.org/10.1145/3307650.3322213).
- Li et al., “Projection-based Runtime Assertions for Testing and Debugging Quantum Programs,” PACMPL 4(OOPSLA), Article 150, 2020, [DOI: 10.1145/3428218](https://doi.org/10.1145/3428218).
- Wang et al., “QDiff: Differential Testing of Quantum Software Stacks,” ASE 2021, [DOI: 10.1109/ASE51524.2021.9678792](https://doi.org/10.1109/ASE51524.2021.9678792).
- Li et al., “Diagnosis of Quantum Circuits in the NISQ Era,” IEEE VTS 2023, [DOI: 10.1109/VTS56346.2023.10140030](https://doi.org/10.1109/VTS56346.2023.10140030).
- Geifman & El-Yaniv, “SelectiveNet: A Deep Neural Network with an Integrated Reject Option,” ICML 2019, PMLR 97:2151–2159, [PMLR record](https://proceedings.mlr.press/v97/geifman19a.html).
- Ishimoto et al., “Evaluating Mutation-based Fault Localization for Quantum Programs,” 2025 preprint, [arXiv:2505.09059](https://arxiv.org/abs/2505.09059).


