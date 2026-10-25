# Phase 5.1 — Benchmark Compatibility and Execution-Readiness Audit

**Status:** audit/specification only. No simulator run, dataset generation, model training, or threshold tuning was performed. The Phase 5 design report is the source proposal; this report freezes an audited 4-qubit version and records unresolved rows explicitly.

## Decision summary

**Current decision: NO-GO for benchmark execution.** Gate and readout injections are representable, but evidence-withheld uncertainty cannot be produced as an ordinary simulator case using the existing runner interface. The 0.02 injected hardware level is below the hardware candidate evidence cutoff. Also, five families × 14 scenarios × five seeds × three shot counts is a large workload without historical wall-time records. Complete the design-only next action stated at the end before seeking separate execution authorization.

## Compatibility audit

The inspected implementation is `src/quantum_simulator/gates.py`, `quantum_simulator.py`, `hardware_model.py`, `fault_localizer.py`, `hardware_fault_localizer.py`, `diagnosis_engine.py`, and `experiments/run_experiment.py`. The circuit runner accepts list/tuple operations and recognizes measurement. The gate registry contains H, X, Y, Z, S/Sdg, T/Tdg, CX/CNOT, SWAP, CCX/Toffoli and aliases. The simulator has explicit execution branches for these gates. It accepts readout probabilities and calibrated gate-error probabilities in [0,1]; each gate error represents operation skip probability, not depolarizing noise. Existing Phase 2 results show four-qubit cases complete. No hard maximum qubit constant was found; memory and operation cost grow exponentially with width. This specification therefore fixes all cases at **four qubits**, within observed prior use, and does not claim larger widths have been audited.

| Scenario template (instantiated once per family unless noted) | Parser/simulator | Fault mechanism and independent output reference | Identifiability / metric eligibility | Compatibility |
|---|---|---|---|---|
| `N0` normal control | Four-qubit gate sequences + measure; registered operations | Ideal statevector probabilities can be computed by `_statevector_probabilities` on the reference circuit without diagnosis | NORMAL for classification; negative detection control; no localization target; not abstention-positive | **SUPPORTED** |
| `P1` program deletion | Remove a named CX/SWAP/CCX/T operation from observed program | Program mutation is a circuit-level intervention; ideal reference distribution is statevector-derived from expected circuit | Classification and program localization only when output/mutation affects measured outputs; each listed deletion is chosen on a causal path | **SUPPORTED** for cases defined below |
| `P2` program substitution | CX→CZ, SWAP→I on a selected target, or T→Tdg as specified below | Both operations are registered. The expected/observed signatures are compared independently from the output distribution | Classification/localization eligible when the selected input and measurement make the changed operation observable; exact location is declared in ledger | **SUPPORTED** for cases defined below |
| `R(p)` readout fault, p∈{0.02, 0.06, 0.15, 0.30} | Readout probability is validated as a probability and applied on measured qubit | Supported stochastic bit-flip readout; ideal reference independent of diagnosis | Detection/classification eligible. At p=.02 current hardware localizer gives no readout candidate because it tests `readout_error > .05`; localization not accepted at that level. At .06/.15/.30 candidate evidence is available, but output anomaly remains sample-dependent. | **SUPPORTED** injection; **UNSUPPORTED** localization pass criterion at .02 |
| `G(p)` gate fault, p∈{0.02, 0.06, 0.15, 0.30} | Calibration error is applied by simulator as probability the calibrated gate is skipped | Supported mechanism for CX, SWAP, CCX, and T operation calibrations; not a continuous over-rotation/channel | Detection/classification eligible. Hardware localizer also requires gate error `> .05`; no hardware candidate at .02. For CCX, source supports 3-qubit gate calibration and candidate representation, but no saved end-to-end CCX-localization case exists: mark its localization result exploratory until validated. | **SUPPORTED** injection; CCX localization **NEEDS_CLARIFICATION** |
| `C` program + readout combined | Uses listed P1 plus R(.15) | Two independently declared interventions; compare against reference statevector | Combined truth is multi-label. A forced single-cause class is not a valid success; evaluate combined detection and abstention. | **SUPPORTED** |
| `U1` unknown-cause/evidence-withheld simulator scenario | Would require suppressing hardware profile and expected/observed program evidence while retaining a simulated intervention | `run_experiment` passes the hardware model and both circuits directly to diagnosis. Withholding these requires changing the runner/interface; setting `observed_counts` creates a fixture and bypasses a simulator observation. | Uncertain label can be valid only if a frozen allowed-evidence view makes causes non-identifiable. Current normal-run interface cannot establish that condition. | **NEEDS_CLARIFICATION**; do not run as simulator scenario |
| `U2` observationally ambiguous natural pair | Two distinct interventions deliberately produce identical measured distributions at selected outputs | Statevector probabilities can establish equality in principle for ideal circuits; noisy intervention equivalence still needs independent analytic proof | Abstention eligible only after proof that permitted evidence is identical. Current proposed cases do not supply that proof. | **NEEDS_CLARIFICATION**; omit until specified |

A scenario remains injection-supported even when subtle. “SUPPORTED” does not mean the current diagnosis logic will detect it. The strict cutoff is visible in `_score_qubit` / `_score_gate` (`>0.05`), and the simulator does not implement general depolarizing or coherent gate-noise channels. The Phase 5 strengths .02/.06/.15/.30 are representable values; accepted model evidence differs by threshold as indicated.

## Frozen case ledger

### Circuit families and case construction

Use only four-qubit indices q0–q3. Each family below has **14** cases: `N0`, `P1`, `P2`, `R02`, `R06`, `R15`, `R30`, `G02`, `G06`, `G15`, `G30`, `C15`, `U1`, `U2`. Stable case ID is `{family_id}-{suffix}`; the suffix values are exact and the same template definitions apply to every family. This produces 70 ledger rows before execution. `U1` and `U2` are retained as blocked design rows and excluded from run-count until resolved, so the currently executable candidate subset is 60 cases. No generated case IDs, targets, or profile values may enter model features.

| Family ID | Four-qubit circuit specification (expected program; measure all at end) | `P1` deletion | `P2` substitution | `R` target | `G` target |
|---|---|---|---|---|---|
| `F1_GHZ_CHAIN` | H(q0), CX(0,1), CX(1,2), CX(2,3) | Delete CX(1,2) | Replace CX(1,2) with CZ(1,2) | q2 | CX(2,3) |
| `F2_TOFFOLI_COMPUTE` | X(q0), X(q1), CCX(0,1,2), CX(2,3) | Delete CCX(0,1,2) | Change CCX target from q2 to q3 | q2 | CCX(0,1,2) |
| `F3_SWAP_ROUTING` | X(q0), X(q2), SWAP(0,1), SWAP(1,2), SWAP(2,3) | Delete SWAP(1,2) | Replace SWAP(1,2) with I(q1) | q1 | SWAP(1,2) |
| `F4_PARITY_TREE` | X(q0), H(q1), CX(0,1), CX(1,2), CX(1,3), H(q1) | Delete CX(1,3) | Replace CX(1,3) with CZ(1,3) | q3 | CX(1,3) |
| `F5_PHASE_LAYER` | H(q0), T(q0), H(q0), CX(0,1), H(q2), T(q2), H(q2) | Delete T(q2) | Replace T(q2) with Tdg(q2) | q1 | T(q2) |

The circuits are structurally different from the Phase 2 Bell, rotation/RZZ, and H/CX pair cases, though CX remains a shared primitive. The F5 H–T–H framing makes the phase operation affect computational-basis output probabilities. F2's initial controls set a known CCX input; the CCX target mutation changes the output bit. F3 starts from a nonuniform bit pattern and uses asymmetric routing so the selected SWAP is behaviorally relevant. F4 creates a measured parity dependency. Before run authorization, independently verify the listed P1/P2 measured-output effects from exact statevector probabilities. A syntactic mutation alone is not evidence of behavioral distinguishability.

### Ledger rule, targets, labels, and eligibility

Each family template has identical preregistered metadata fields: `case_id`, `family_id`, `program_expected`, `program_observed`, `intervention_kind`, `target`, `strength`, `label`, `seed`, `shots`, `evidence_available`, and metric flags. The following suffix rules freeze the planned values:

| Suffix | Intervention/strength/target | Constructed truth | Required allowed evidence | Eligible metrics |
|---|---|---|---|---|
| `N0` | No intervention | NORMAL | Expected circuit/reference; ideal/no-fault calibration; counts | Classification, negative detection; no localization |
| `P1` | Family-specific deletion above; no hardware intervention | PROGRAM_FAULT | Expected and observed circuit; measured counts/reference | Detection, program-vs-hardware classification, operation localization |
| `P2` | Family-specific substitution above; no hardware intervention | PROGRAM_FAULT | Expected and observed circuit; measured counts/reference | Same as P1 |
| `R02/R06/R15/R30` | Readout p=.02/.06/.15/.30 on family target q | HARDWARE_ANOMALY | Expected circuit/reference; calibration interface for qubit readout; counts | Detection and classification all levels; localization only R06/R15/R30 |
| `G02/G06/G15/G30` | Gate skip probability p=.02/.06/.15/.30 on family target gate/edge | HARDWARE_ANOMALY | Expected circuit/reference; gate calibration interface with gate+ordered qubit tuple; counts | Detection and classification all levels; localization only G06/G15/G30; G* on CCX localization remains exploratory |
| `C15` | P1 deletion plus readout p=.15 on the family's R target | COMBINED (multi-label program+hardware) | Expected/observed circuit; readout calibration; counts/reference | Combined detection; multi-label class; both targets if jointly identifiable; abstention behavior, but not single-cause accuracy |
| `U1` | Proposed source-hidden anomaly without calibration/program evidence | UNCERTAIN / INSUFFICIENT_EVIDENCE | Requires evidence suppression unavailable in current runner | Abstention only; excluded until interface resolves |
| `U2` | Proposed observationally indistinguishable distinct causes | UNCERTAIN / INSUFFICIENT_EVIDENCE | Requires independent equality proof and identical allowed evidence | Abstention only; excluded until proof and exact case are frozen |

Constructed labels are assigned from the intervention ledger **before** any execution and without prediction fields. Normal means no injected intervention; program-only and hardware-only reflect their declared injection; combined retains both truths; uncertain is assigned only to a pre-proven evidence-equivalent case. Do not convert a fault to NORMAL if its finite-shot sample is close to reference. Separately report the count deviation statistic against the ideal statevector distribution, with the predeclared existing 0.05 threshold; this output statistic is not the label source. The “exact reference” is computed from the expected circuit by the statevector calculation in `run_experiment.py`, independent of the diagnosis result.

### Seeds and shots

For every executable case use seeds **20263001, 20263002, 20263003, 20263004, 20263005** and shot counts **256, 1024, 4096**. Paired normal/fault cases use identical seed and shot values. Primary anomaly threshold remains frozen at 0.05. No threshold tuning is authorized by this specification.

## Split manifest and leakage controls

| Split | Family IDs | Allowed use |
|---|---|---|
| Development/train | `F1_GHZ_CHAIN`, `F2_TOFFOLI_COMPUTE` | Fit learned component and development-only feature transformations |
| Calibration | `F3_SWAP_ROUTING` | Select calibration map/decision threshold if needed; report its metrics separately from train |
| Locked test | `F4_PARITY_TREE`, `F5_PHASE_LAYER` | One final evaluation after all model, features, calibration, and thresholds are frozen |

Exact family sets are disjoint (2/1/2 members; no shared ID). All variants, strengths, mutations, controls, seeds, and shot levels stay with their family. Do not split by row, case, seed, or shot count. This protocol does not estimate test performance well at only two test families; report each family separately.

**Permitted inference-time diagnosis-input schema (exact current API):** `expected_circuit`, `observed_circuit`, `expected_distribution` (exact ideal probabilities from the reference circuit), `observed_distribution` (normalized counts), `shots`, `threshold`, and `hardware_model` calibration fields: qubit `readout_error`, `T1`, `T2`, status/availability; gate/edge error, duration, support, and target tuples. Raw counts are retained in result artifacts but `diagnose_execution` receives normalized distributions rather than an `observed_counts` argument. A separate learned model may use only features deterministically derived from these permitted fields; its feature transformation must be frozen before calibration/test evaluation. This schema describes existing diagnosis inputs, not a claim that every field is available in deployment; restrict fields to those declared available for the intended use case before training.

**Forbidden feature fields:** `case_id`, family ID/name, split, scenario description, ground-truth label/cause, intervention kind/target/strength, affected target set, expected-detection flag, pair ID, seed, file path, result/evidence artifact, and any label-derived encoding. Seed and case/family identifiers may be retained in the audit ledger for grouping only, then removed before feature construction. Calibration values in the permitted schema may expose injected strengths; if that calibration is not genuinely available at inference time, remove it from model inputs and rerun the design review. Keep the truth ledger physically separate and join only after predictions are frozen.

## Resource estimate

The historical Phase 2 aggregate contains 36 case JSON outputs totaling about 384,732 bytes (mean about 10.7 KB per output); Phase 1.4 records no duration fields in the inspected case results. Searches of the saved Phase 1.4/Phase 2 cases and reports found no per-run wall-time timings, so a defensible wall-clock estimate cannot be calculated without a timing run, which is disallowed here.

If only the 60 executable rows are retained: **900 run configurations** (60×5 seeds×3 shot levels). Existing `run_experiment` makes an ideal shot pass and a hardware/observed shot pass; at each scenario seed the total shot executions across shot levels are 2×(256+1024+4096)=10,752. That gives **3,225,600 shot-circuit executions** across the matrix, plus statevector/reference and diagnosis-trace work. If U1/U2 are later resolved into executable cases, the 70-row grid becomes 1,050 configurations and 3,763,200 shot-circuit executions. At the historical mean output size, the 900 result files would be roughly **9.6 MB** (or ~11.2 MB for 1,050), excluding any duplicated traces, manifests, and logs. Reserve **15–25 MB** for primary JSON/CSV artifacts as a planning allowance, not a measured bound. Statevector width is four qubits (16 amplitudes); the per-shot Python loop is likely the main runtime cost. Wall time remains **unquantified** pending an authorized development-only pilot or a valid historical timing source.

## Unresolved ambiguities and explicit later-run criteria

- `U1` cannot be implemented as evidence-withheld simulator input through the current runner; do not use a synthetic counts fixture as simulator truth.
- `U2` has no exact intervention pair or analytical equality proof yet. Omit until specified.
- p=.02 readout/gate injections execute, but hardware localization is below the localizer's strict 0.05 cutoff. Keep them for descriptive detection/classification only; no localization acceptance claim.
- CCX hardware injection is source-supported, but no saved end-to-end CCX localization precedent was found. Its fault localization is exploratory until the support audit is closed.
- Current runner makes scenario-related circuits and hardware calibration visible to diagnosis. This is valid only if both are explicitly part of the target inference contract. If not, an input-ablation design must be authorized and specified before benchmark execution.
- “Ambiguous case” acceptance cannot be evaluated until U1/U2 are resolved. Do not silently remove their planned share while claiming a balanced uncertainty benchmark.
- Runtime cannot be credibly stated from current artifacts. A design-only workload count is provided; no timing extrapolation is fabricated.

A later benchmark run is **GO** only when all are true: (1) a read-only compatibility/support audit confirms every retained gate, circuit mutation, calibration target and metric mapping, including CCX localization if claimed; (2) F1–F5 case outputs are analytically distinguishable where labels require it; (3) U1/U2 are either specified with source-supported evidence control/proof or formally removed from the intended objective with updated acceptance criteria; (4) inference-time availability of both circuits and calibration fields is confirmed; (5) the case ledger, family split, feature schema, threshold 0.05, seeds, shot counts, and resource cap are frozen and hashed; (6) a run-time/storage cap is approved based on an authorized development-only estimate; and (7) a separate explicit authorization is given to execute. Otherwise **NO-GO**.

## Single next action

Perform a **design-only closure review**: prove the P1/P2 measured-output effects from ideal statevector calculations without invoking diagnosis, decide whether the interface's program and calibration evidence is permitted at inference time, and resolve/omit U1/U2 with revised criteria. Record those decisions and hashes; do not execute benchmark simulations or train a model.

