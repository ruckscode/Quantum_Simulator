# Phase 5.3 — Reference Distributions and Evidence Contracts

**Review type:** deterministic mathematics and source inspection only. No simulator execution, sampling, diagnosis call, dataset generation, timing run, model training, or threshold tuning was performed.

## Decision

# NO-GO for the original full benchmark

The retained 60-case simulator grid has source-derived input contracts and exact ideal reference distributions. However, two required dimensions from the original Phase 5 objective are not represented by the frozen ledger: (1) unknown-cause / evidence-withheld cases U1 and U2 remain excluded, and (2) no case declares a coupling-edge calibration fault, so coupling localization cannot be evaluated. The existing combined case C15 supports a narrow, predeclared multi-fault ambiguity/abstention measurement, but it does not replace unknown-cause abstention. The benchmark could proceed only as a revised, narrower scope after those lost claims are removed from its objectives/acceptance criteria and that revised scope is approved. This report does not authorize execution.

## Reference-distribution method and source basis

All bit strings below are written in qubit-index order `(q0,q1,q2,q3)`, independent of any backend display convention. The reference means the ideal probability distribution of the **expected** circuit. It is distinct from finite-shot counts. Exact probabilities are analytically derivable for the frozen circuits using the listed gate matrices and ideal basis-state transformations; no reference case in this ledger requires a simulator execution. This review did not execute `_statevector_probabilities` or use a diagnosis result.

Source/method anchors:

- Gate operations and matrices: `src/quantum_simulator/gates.py` (H, CX, CZ, SWAP, CCX, T, Tdg).
- Simulator gate-noise semantics: `src/quantum_simulator/quantum_simulator.py::run_circuit`: readout error flips measured bits with its configured probability; calibrated gate error is a probability the named operation is skipped. It is not depolarizing or coherent rotation noise.
- Runner reference path: `experiments/run_experiment.py::_statevector_probabilities`, `_marginal_ideal_distribution`, and `run_experiment`. These establish the ideal expected-circuit distribution independently of `diagnose_execution`; their values were derived here by gate algebra, not by invoking that code.
- Anomaly decision: `src/quantum_simulator/statistics.py::compare_distributions` compares the observed finite-shot distribution to the expected distribution using TVD and effective tolerance `max(0.05, 1/sqrt(shots))`.

Let `a=(2-√2)/4 ≈ 0.1464466` and `b=(2+√2)/4 ≈ 0.8535534`. Let `D` denote the ideal reference distribution in the family table, `D[-g]` the ideal distribution after omitting the named calibrated gate, and `Flip_q,p(D)` the exact readout-flip transformation: `P(y)=(1-p)D(y)+pD(y xor e_q)`. Since the simulator gate model skips a gate with probability p, gate-noise output probabilities are exactly `(1-p)D+pD[-g]`. These formulas specify exact probabilities; they do not predict finite-shot counts.

### Exact ideal and program-fault output distributions

| Family | N0 ideal reference D | P1 distribution | P2 distribution | Exact TVD from D (P1 / P2) |
|---|---|---|---|---:|
| `F1_GHZ_CHAIN` | 0000: 1/2, 1111: 1/2 | 0000: 1/2, 1100: 1/2 | 0000: 1/2, 1100: 1/2 | 1/2 / 1/2 |
| `F2_TOFFOLI_COMPUTE` | 1111: 1 | 1100: 1 | 1101: 1 | 1 / 1 |
| `F3_SWAP_ROUTING` | 0101: 1 | 0101: 1 | 0101: 1 | 0 / 0 |
| `F4_PARITY_TREE` | 1000, 1100, 1011, 1111: each 1/4 | 1000, 1100, 1010, 1110: each 1/4 | 1000, 1100, 1010, 1110: each 1/4 | 1/2 / 1/2 |
| `F5_PHASE_LAYER` | 0000:b², 0010:ab, 1000:ab, 1010:a² | 0000:b, 1000:a | Same as D | a / 0 |

Important corrections from direct calculation:

- F3 begins with q1=q2=1 before SWAP(1,2), so that selected SWAP is inert. Both its program deletion and SWAP→I substitution preserve the exact output. Circuit signatures remain different and are available to diagnosis, so these remain constructed `PROGRAM_FAULT` classification/localization cases under the source-visible program contract. They are **negative output-anomaly** cases and must not be counted as positive distribution-detection cases.
- F5 T→Tdg changes the observed operation signature but not computational-basis probabilities: `H T H|0>` and `H Tdg H|0>` have the same outcome probabilities. F5 P2 is a program-fault case identifiable from the available circuit signature, but it is a **negative output-anomaly** case.
- F1 P1 and P2 have the same output distribution as each other, but not as the reference. F4 P1 and P2 likewise share a distribution. Both causes remain distinguishable from the expected circuit in the permitted program-input contract; counts alone do not distinguish which mutation occurred.

### Exact noisy and combined distributions

| Scenario(s) | Expected output distribution / exact TVD from ideal reference | Anomaly-truth interpretation |
|---|---|---|
| `R02/R06/R15/R30` in any family | `Flip_q,p(D)` for the declared target q. In all five families the target bit's flip support is disjoint from D, so exact TVD is p. | Distribution differs for all p>0. At p=.02, exact divergence is below the fixed .05 threshold. Finite-shot outcomes are still random. |
| `G02/G06/G15/G30` F1 CX(2,3) | `(1-p){0000:1/2,1111:1/2}+p{0000:1/2,1110:1/2}`; TVD=p/2. | Positive exact distribution change for p>0; p=.02 yields TVD=.01. |
| Same, F2 CCX(0,1,2) | `(1-p){1111:1}+p{1100:1}`; TVD=p. | Positive exact distribution change for p>0; p=.02 yields TVD=.02. |
| Same, F3 SWAP(1,2) | D for every p, because omitting the target SWAP has the same measured outcome; TVD=0. | Gate intervention is present in constructed truth/calibration but has no output-distribution signature in this circuit. Negative output-anomaly case at all strengths. |
| Same, F4 CX(1,3) | `(1-p)D+p{1000,1100,1010,1110: each 1/4}`; TVD=p/2. | Positive exact distribution change for p>0; p=.02 yields TVD=.01. |
| Same, F5 T(q2) | `(1-p)D+p{0000:b,1000:a}`; TVD=a·p. | Positive exact distribution change for p>0, but TVD ranges only about .00293–.04393 across p=.02–.30, below .05. Calibration evidence can still exist at p>.05. |
| `C15` any family | Apply `Flip_q,.15` to that family's P1 distribution. For all five families the target q bit is fixed in the P1 output support, so exact TVD from D is .15. | Positive output anomaly; constructed truth is multi-cause `PROGRAM_FAULT + HARDWARE_ANOMALY`. Score combined/multi-label recovery and whether a single-cause answer is unjustified. Predeclare `AMBIGUOUS` as the categorical outcome for the combined single-label task; this is justified by the two injected causes, not by prediction. |

At 256 shots, the decision tolerance is .0625; at 1024 and 4096 it is .05. Thus an exact TVD above .05 does not guarantee an empirical finite-shot statistic above threshold on every seed. F1/F4 gate cases at p=.06 have exact TVD .03; F5 gate cases remain below .05 even at p=.30. F3 gate cases have zero exact output deviation. These distinctions must be reported by case and strength; injection truth, exact probability change, and sampled anomaly decision are separate columns.

## Inference-time evidence contract

`diagnose_execution` takes `expected_distribution`, `observed_distribution`, `expected_circuit`, `observed_circuit`, `hardware_model`, `threshold`, and `shots`; the ordinary scenario runner passes all seven. It receives normalized distributions, not raw counts. The runner retains raw counts as artifacts and computes ideal reference from the expected circuit. Its `hardware_parameters` are used both to inject simulator behavior and to build the `hardware_model` visible to diagnosis.

| Scenario family/template | Circuit and measurement evidence available | Calibration / topology actually available in the proposed local run | Withheld fields | Permitted claims and limits |
|---|---|---|---|---|
| All `N0`, `P1`, `P2` | Expected and observed operation lists (including measurement-all added by runner); expected ideal distribution; observed counts converted to distribution; shots 256/1024/4096 | A synthetic baseline hardware model is passed. Baseline is intended to have readout/gate error 0 and long T1/T2. Exact per-gate baseline entries are not enumerated by the frozen case ledger and must be frozen before any run. Logical indices are the only declared qubit mapping. | Truth labels, case/family IDs, mutation ID/target annotation, pair/split IDs, seed as feature, and intervention descriptions are not passed to diagnosis. | Program-fault class and operation localization may use expected/observed circuit mismatch. Distribution detection uses only distributions. No physical-backend calibration or physical-qubit localization claim. |
| `R02/R06/R15/R30` | Same circuits/distributions/counts/shots as above | Synthetic `hardware_model` includes target-qubit `readout_error=p`; other readout values intended 0. T1/T2 and gate entries are synthetic baseline. This value is also the simulator injection setting, so it is privileged simulated calibration evidence, not an independent sensor measurement. | Separate `injected_strength`, fault label, target annotation, scenario ID are withheld; the numeric calibration itself is visible as permitted evidence. | Output detection/classification can be scored. Qubit localization only for p>.05 under current localizer. Do not describe these calibration fields as independently measured backend evidence. |
| `G02/G06/G15/G30` | Same circuits/distributions/counts/shots as above | Synthetic `hardware_model` includes target gate name + ordered tuple + error p; other calibration values intended baseline. The same p drives skip injection and is visible to diagnosis. No explicit edge calibration is declared. | Separate injection metadata/labels/target ID withheld; target tuple and error remain visible in the hardware model. | Gate-target localization only at p>.05; F2 CCX only via fully calibrated `(q0,q1,q2)` base candidate. No coupling-edge localization. Gate and observed-circuit evidence are logically available in this simulator but are not an independently calibrated physical record. |
| `C15` | Expected circuit; mutated observed circuit; reference and noisy observed distributions; shots | Synthetic target readout calibration .15 plus baseline gate/qubit data; calibration is shared with the simulator injection | Label, case ID, explicit combined-fault flag/target annotation withheld; raw calibration remains visible | Multi-label/combined diagnosis and cautious AMBIGUOUS single-label score. A positive abstention result here generalizes only to known simultaneous faults with both evidence channels supplied. |
| `U1/U2` | No executable case contract | No valid evidence view / no competing-cause calibrations defined | Unresolved by design | Excluded from every metric. |

**Topology/mapping boundary:** The frozen families specify logical operation pairs and the configuration supports a `coupling_map`, but Phase 5.1 does not assign a physical backend mapping or any `edges` calibration entries. Gate calibration tuples identify the simulated gate target; they do not establish a physical coupling-edge diagnosis. Consequently, qubit localization is eligible where supported, gate-tuple localization is eligible as noted, and coupling localization is **not estimable** for this ledger. Add a separately reviewed edge calibration case before claiming coupling localization.

**Real-backend distinction:** all calibration rows described above are synthetic simulator settings that also cause the injected noise. They are available to the local diagnosis API by construction, but are not independently observed hardware calibrations. This protocol can support claims about diagnosis under those supplied synthetic profiles only. It cannot support claims that a real backend would provide those values, their accuracy, or the same fault signal. If deployment-realistic generalization is claimed, a separately specified evidence source/provenance audit is still required.

## Scoring eligibility by retained case

`D+` below means the exact output distribution differs from the ideal reference; it does not assert every finite-shot run crosses the existing threshold. `D−` means exact output distributions are equal. Program classification/localization uses the visible expected/observed program pair, so it can remain eligible for a D− case.

| Retained suffix | Output anomaly detection | Program-vs-hardware classification | Qubit / gate / coupling localization | Abstention / insufficient evidence |
|---|---|---|---|---|
| `N0` | Negative control (D−) | NORMAL control | No target; localization N/A | No |
| `P1`, `P2` F1 | Positive distribution anomaly (D+) | PROGRAM_FAULT | Program operation index eligible | No |
| `P1`, `P2` F2 | Positive (D+) | PROGRAM_FAULT | Program operation index eligible | No |
| `P1`, `P2` F3 | Negative output anomaly (D−); retain as masked-fault control | PROGRAM_FAULT via circuit mismatch | Program operation index eligible; not output-causal localization | No |
| `P1`, `P2` F4 | Positive (D+) | PROGRAM_FAULT | Program operation index eligible | No |
| `P1` F5 | Positive (D+) | PROGRAM_FAULT | Program operation index eligible | No |
| `P2` F5 (T→Tdg) | Negative output anomaly (D−) | PROGRAM_FAULT via circuit mismatch | Program operation index eligible; signature-only | No |
| All `R02/R06/R15/R30` | D+ for all p>0; measure observed detection by seeds/shots | HARDWARE_ANOMALY intervention | Qubit target only for .06/.15/.30; exclude .02 | No |
| `G02/G06/G15/G30` F1, F2, F4, F5 | D+ for p>0, with family-specific exact TVD formula above | HARDWARE_ANOMALY intervention | Gate tuple for p>.05; F2 CCX only full calibrated target tuple; no edge localization | No |
| `G02/G06/G15/G30` F3 | D− for every p | HARDWARE_ANOMALY intervention remains calibration-identifiable | Gate tuple for p>.05, despite no output localization signal; no coupling localization | No |
| `C15` all families | D+ | COMBINED two-cause truth; do not score as a correct single-cause answer | Program target + qubit target for readout p=.15; no coupling target | Yes, for known combined-fault AMBIGUOUS outcome only |
| all U1/U2 | Excluded | Excluded | Excluded | Excluded; no abstention score |

At p=.02, readout `_score_qubit` and gate `_score_gate` use strict `>0.05`; no hardware-fault candidate is emitted from the p=.02 calibration. Keep p=.02 out of localization scoring. No threshold has been altered.

## Uncertainty disposition and objective impact

- **U1: `EXCLUDED_PENDING_DEFINITION`.** The intended hidden/available fields were not specified, and the runner cannot separate a simulation-only fault profile from the model given to diagnosis. There is no independently justified expected `INSUFFICIENT_EVIDENCE` output.
- **U2: `EXCLUDED_PENDING_DEFINITION`.** No concrete causes or distributions were in the frozen design. No observational equivalence claim is made. Exact equality must be proved over distributions and all permitted circuit/calibration inputs; empirical similarity at a shot count is not equivalence.
- **Objective impact:** excluding U1/U2 removes evaluation of unknown-cause and evidence-withheld abstention. C15 retains only the known-combined-fault ambiguity task. Do not describe C15 results as unknown-cause abstention or satisfy an acceptance criterion that requires U1/U2.
- **Split preserved without change:** development/train `{F1_GHZ_CHAIN,F2_TOFFOLI_COMPUTE}`; calibration `{F3_SWAP_ROUTING}`; locked test `{F4_PARITY_TREE,F5_PHASE_LAYER}`. Family groups remain disjoint; all their variants/seeds/shots stay together.

## Resource-count reconciliation

The frozen 14 templates per family are N0 (1), P1/P2 (2), readout strengths (4), gate strengths (4), C15 (1), U1/U2 (2). Excluding U1/U2 leaves 12 × 5 = **60 cases**. With five seeds and three shot counts, this is **900 configurations**. `run_experiment` makes one ideal and one observed shot pass, so total shot-circuit executions are `60 × 5 × 2 × (256+1024+4096) = 3,225,600`. This excludes statevector/reference and diagnosis-trace work. At the Phase 2 mean JSON artifact size (~10.7 KB), 900 files would be ~9.6 MB; this remains an extrapolation, not an observed Phase 5 size. No historical per-run wall times were found; runtime remains unquantified. No timing experiment was run.

## Remaining items and final run gate

1. Freeze exact synthetic baseline hardware-profile fields and clarify that simulated calibration values are visible evidence, not independent hardware measurements.
2. If a coupling-localization claim remains required, add and audit an explicit edge-calibration scenario and mapping; current case set cannot support that metric.
3. Decide whether the benchmark objective is formally narrowed to known faults plus C15 combined-fault ambiguity. U1/U2 cannot be counted toward uncertain-case acceptance.
4. Complete an independent exact-distribution review as part of the preregistration record, including the formulas/tables above, then freeze/identify the ledger and metrics.
5. Only after the scope/acceptance update and a separate explicit authorization may a benchmark run be considered.

Until items 1–4 are closed, **NO-GO**. This decision concerns the original Phase 5 scope; it does not prohibit a later authorized, explicitly narrowed study.
