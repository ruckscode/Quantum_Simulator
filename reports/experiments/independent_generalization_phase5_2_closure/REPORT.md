# Phase 5.2 — Benchmark Design Closure Review

**Scope:** source/design review only. No simulations, statevector calculation, dataset generation, model training, or threshold tuning was performed. The prior frozen Phase 5.1 report remains unchanged.

## Decision

# NO-GO for Phase 5 execution

The executable single-fault/combined-fault grid is specified, but the full Phase 5 objective includes a valid uncertainty/abstention evaluation. Both proposed uncertainty rows lack a defensible evidence contract or an explicit competing-cause equivalence proof. They are therefore `EXCLUDED_PENDING_DEFINITION`, not relabeled as expected abstentions. Do not start the benchmark under the current full-objective specification.

## Evidence-withheld and ambiguous case dispositions

The existing runner uses the same `hardware_parameters` input to build (a) the simulator execution profile and (b) the `HardwareModel` passed to `diagnose_execution`. It also passes both the expected and observed circuit sequences, expected distribution, observed distribution, shots, and threshold to diagnosis. It has no separate simulation-only calibration profile versus inference-visible calibration profile. Passing `observed_counts` selects a controlled observation fixture and bypasses simulated observed counts; it does not create a hidden-cause simulator case.

| Case template | Intended evidence withholding / remaining evidence | Independent expected result | Disposition and metric eligibility |
|---|---|---|---|
| `*-U1` | Proposed anomaly with its causal calibration and program-comparison evidence withheld. The earlier design did not define exactly which of the shared hardware profile, expected circuit, observed circuit, expected distribution, and observed distribution remains visible. Under the existing runner, injected hardware calibration and both circuits are all passed into diagnosis; removing them separately requires an interface path that does not exist. | `INSUFFICIENT_EVIDENCE` would be justified only if a predeclared inference view retained a measured anomaly but omitted all cause-identifying evidence. That view is not implementable through the current ordinary simulator-run config. A counts fixture would not justify a simulator-generated case. No diagnosis prediction can repair this missing definition. | **EXCLUDED_PENDING_DEFINITION.** No class, anomaly, localization, or abstention score. Revisit only after a separate simulation-profile / inference-evidence contract is designed and independently reviewed. |
| `*-U2` | Proposed pair of competing causes with “identical” allowed evidence. Phase 5.1 specified no actual pair of circuits, injected profiles, target location, or evidence view. | No expected distribution pair exists to compare. Consequently there is no evidence for exact observational equivalence, no distinction between exact equality and finite-shot similarity, and no justified `AMBIGUOUS` label. | **EXCLUDED_PENDING_DEFINITION.** No classification, detection, localization, or abstention score. A future case must specify both causes, the measurements and inference fields, and prove equality analytically before assigning ambiguity. |

`AMBIGUOUS` is reserved for multiple supported causes present together or competing causes that remain indistinguishable under the frozen permitted evidence. `INSUFFICIENT_EVIDENCE` is reserved for a detected deviation with cause-identifying evidence deliberately absent. Neither label is assigned merely because the diagnosis output abstains. Labels must be written from a reviewed intervention/evidence ledger before any predictions are generated.

There is no “current shot count equivalence” claim for U2: it has no two expected distributions to compare. A future pair with equal ideal probabilities but different calibration or circuit inputs would not be observationally equivalent under the permitted full API. A pair with different exact distributions but close empirical counts is finite-shot overlap, not exact equivalence; it must be scored against its distinct constructed truths and stratified by shots.

## 0.02 strength disposition

The exact source conditions are in `src/quantum_simulator/hardware_fault_localizer.py`:

- `_score_qubit` adds readout-fault evidence only for `props.readout_error > 0.05`.
- `_score_gate` adds gate-fault evidence only for `best.error > 0.05`.
- Candidate collection drops scores `<= 0.05`.

At p=.02, the simulator can apply a readout bit flip or skip the calibrated gate with probability .02, but the hardware localizer does not emit a readout/gate degradation evidence candidate from that calibration value. The strict inequality also means exactly .05 does not cross the cutoff. The p=.02 `R02` and `G02` cases remain eligible for detection and class-prediction evaluation as constructed HARDWARE_ANOMALY interventions; report misses/insufficient-evidence outcomes as performance, not as re-labeled NORMAL cases. They are **ineligible for hardware localization scoring** because no target candidate is expected from the current rule. This report does not change the 0.05 threshold or tune against benchmark cases.

## CCX support and localization

Source-level path inspection:

| Stage | Evidence in source | Disposition |
|---|---|---|
| Parsing/arity | `gates.py` registers `ccx` and `toffoli` with arity 3. `fault_localizer._normalize_operation` obtains arity from `get_gate_qubit_count`. The experiment runner accepts tuple/list operation sequences and infers qubit width from configured width and operands. | **SUPPORTED** |
| Ordinary simulation | `quantum_simulator.py::run_circuit` has an explicit `ccx`/`toffoli` branch, validates three operands, and calls `apply_toffoli`. The gate matrix implementation is registered. | **SUPPORTED** |
| Operation tracing | `run_circuit_traced` selects the first three operands for CCX-family gates, records operation/error metadata, and executes each noisy instruction through `run_circuit`. | **SUPPORTED by source**; no runtime validation was run under this design-only constraint. |
| Program-fault localization | Operation signatures preserve all three qubits; direct delete/substitute comparison can identify CCX target mutation. | **SUPPORTED by source** |
| Hardware localization | Base calibration candidates preserve the complete `entry.qubits` tuple, so a calibrated CCX candidate can identify `(q0,q1,q2)`. However, `_relevant_gate_candidates` uses only `qubits[:2]` for trace-impact matching; it will not match a three-qubit CCX calibration tuple there. | **SUPPORTED only as calibration-target localization** for p>.05; do not claim trace-assisted CCX localization. |

Thus F2 `G06/G15/G30` can contribute exact hardware localization only when the diagnosis receives its declared CCX calibration and the base candidate's full tuple is scored. `G02` is not localization-eligible. If future source changes remove this base-candidate path, exclude CCX localization rather than infer it from the trace.

## Circuit distinguishability and score eligibility

No circuit execution or statevector calculation was made. The listed program variants were reviewed as sequences and operand changes only. Their intended causal patterns are:

- F1 H/CX ladder: deleting or replacing the selected middle CX changes propagation to measured q2/q3; P2's CZ differs from CX in its action on the populated control/target branches.
- F2 known `q0=q1=1` CCX controls: deleting CCX or redirecting its target from q2 to q3 changes the measured bit pattern, including after CX(2,3).
- F3 nonuniform input pattern and routing chain: removing or replacing SWAP(1,2) changes which measured wires carry the initialized bits.
- F4 parity fanout: removing/replacing the selected CX changes the measured parity/data relationship.
- F5 H–T–H framing: T and Tdg have opposite phase signs whose exact output probabilities differ after the final H; deleting T also changes probabilities.

These are source-informed expected-effect arguments, **not independently calculated expected distributions**. Therefore do not record program-fault localization/detection eligibility as empirically validated until a separate design-only exact-probability review enumerates the relevant distributions without invoking diagnosis. For the planned ledger they remain eligible by construction, with a pre-execution closure gate requiring that exact calculation. Any mutation whose exact measured distribution equals the reference must be relabeled as a direct syntactic program mismatch task only, or removed from output-anomaly evaluation; do not infer a behavioral anomaly label from syntax.

Readout/gate fault distributions are stochastic. Their ideal reference can be independently obtained from the ideal expected circuit. Their finite-shot anomaly outcome is not guaranteed by injection truth. Detection labels refer to the injected-fault construct; measured anomaly statistics are a separate outcome.

## Final permitted input schema and leakage review

Current `diagnose_execution` receives exactly: `expected_distribution`, `observed_distribution`, `expected_circuit`, `observed_circuit`, `hardware_model`, `threshold`, and `shots` (plus optional `precomputed_trace`, which is not part of the ordinary `run_experiment` call described here). It does **not** receive raw counts. The runner's `observed_distribution` is normalized from observed counts. Exact ideal probabilities come from the expected circuit statevector.

For a learned component, the frozen permitted schema is:

1. `expected_distribution` and `observed_distribution` (or deterministic features derived only from these distributions), plus `shots` to describe sampling precision.
2. `expected_circuit` and `observed_circuit` only if source-code comparison is genuinely available in the intended deployment contract.
3. Calibration fields actually present in the case's inference-visible `hardware_model`: per-qubit readout/T1/T2/status/availability, gate and edge calibration error/duration/support, and their target tuples. Missing calibration stays missing; do not impute the injected profile from the intervention ledger.
4. The fixed threshold value may be part of the frozen protocol, but must not be varied per case or exposed as a case label.

The synthetic runner constructs calibration values from the configured hardware profile, which is also used to inject simulation faults. These values can reveal the intervention. They are permissible only as explicitly declared inference-time calibration evidence; they must not be passed through a separate feature column named after the injected fault or copied from truth metadata. If the intended deployed system would not have these calibration values, do not use the current full-profile cases to claim deployment-realistic classification. Calibration-only cases are used only where their scenario explicitly supplies calibration; no-cause-evidence cases are excluded pending an interface that can actually withhold it.

Forbidden: case ID, family ID/name, split, scenario/description, label/cause, intervention type, injected probability/strength, target annotations, expected-detection flag, pair ID, seed, file/path, result/evidence artifact, or features derived from any of those. IDs/seeds can remain in the evaluation ledger for grouping and reproducibility, never in model features. The data join from prediction to truth happens only after predictions are frozen.

## Split and scenario-count reconciliation

Family disjointness remains: development/train = `{F1_GHZ_CHAIN, F2_TOFFOLI_COMPUTE}`; calibration = `{F3_SWAP_ROUTING}`; locked test = `{F4_PARITY_TREE, F5_PHASE_LAYER}`. All variants and repetitions stay within family; there are no cross-split family IDs.

The frozen Phase 5.1 ledger defines **14 templates per family**: N0 (1), P1/P2 (2), readout levels (4), gate levels (4), C15 (1), U1/U2 (2); total 14. Across five families that is 70 nominal case rows. U1/U2 are now `EXCLUDED_PENDING_DEFINITION`, not run. The executable candidate set is therefore **12 cases per family × 5 = 60 cases**.

With five seeds and three shot counts:

- Configurations: 60 × 5 × 3 = **900**.
- Shot executions per configuration: ideal pass + observed pass = 2 × shot count.
- Per case over all seeds/shot levels: 5 × 2 × (256+1024+4096) = **53,760** shot-circuit executions.
- Total: 60 × 53,760 = **3,225,600** shot-circuit executions.
- If U1/U2 were later defined and included: 70 × 15 = **1,050** configurations and 70 × 53,760 = **3,763,200** shot-circuit executions.

These counts exclude one statevector/reference pass and diagnosis trace work per configuration. They assume `run_experiment` continues to perform both ideal and observed shot passes at each configured shot count and one saved result per configuration. The previous ~9.6 MB artifact estimate extrapolates Phase 2's observed mean JSON size (about 10.7 KB) to 900 outputs; it is an approximate storage allowance, not a measured Phase 5 result. No historical wall-clock timing was found, and no timing benchmark is authorized; runtime remains unquantified. No threshold adjustment is included.

## Dispositions and later-run gate

- **Retain:** N0, P1, P2, R02/R06/R15/R30, G02/G06/G15/G30, C15 as intervention-defined cases, subject to the pre-run exact-distribution distinguishability review for the program variants.
- **Exclude from localization metric:** R02 and G02 because their calibration evidence lies below the existing strict cutoff; CCX G06/G15/G30 may be localized from their full calibration target tuple only, with no trace-assisted claim.
- **Exclude pending definition:** every family’s U1 and U2. No abstention/uncertainty performance claim may be made for this grid.
- **No-go remains** until exact reference distributions for P1/P2 are independently checked and all feature availability assumptions are frozen. Full Phase 5 additionally needs at least one independently justified uncertainty case if abstention quality remains a required objective.

## Single next action

Complete a **design-only reference/evidence review**: calculate and record exact ideal measured distributions for each family’s P1/P2 variants without calling diagnosis; approve whether the expected/observed circuit and calibration fields are truly available at inference; then decide whether to define a supported U1 evidence-control path and a concrete U2 competing-cause equivalence pair. Do not execute simulations, generate datasets, train, or tune thresholds.
