# Phase 5 — Independent Simulator Generalization Benchmark Design

**Status:** design only; no simulations, model training, dataset generation, or threshold changes were performed.

## Purpose and interpretation

This protocol tests whether the diagnosis pipeline transfers to circuit families and fault cases held out by family. It is a local simulator benchmark, not a claim about physical hardware. All truth labels below are constructed from a sealed intervention ledger. They are not independent real-world ground truth, even though the ledger is created before execution and independently of the diagnosis output.

## Audit of prior evaluation coverage

| Phase | Circuit structures | Faults and locations | Strengths / execution settings |
|---|---|---|---|
| 1 | One two-qubit X then measure q0/q1; a matched idle/coherence circuit | Readout q0 and q1; X-gate calibration error q0; remove X at operation 0; T1/T2 degradation q0; combined remove-X plus q0 readout; uncertain observed-count fixture with causal evidence withheld | Readout 0.40; X gate error 0.40; degraded coherence profile T1=2, T2=1 versus healthy ~1e6; 512 shots; seed 20261013; threshold 0.05 |
| 1.3 | Same nine scenarios and circuit structures as Phase 1 | Same fault types and q0/q1 locations | Same strengths; 512 shots; seed 20261013; threshold 0.05. Report records correction to duration-only hardware evidence behavior; historical Phase 1 false positive remains intact. |
| 1.4 | Same nine scenarios; no new circuit family or fault location | Same fault types/locations and strengths | 512 shots; seeds 20261014–20261018 (plus Phase 1 baseline seed 20261013 in prior report). This varies sampling seed, not shot count. |
| 2 | Six families: Bell preparation on q2–q3; RX/RY/RZZ phase circuit q0–q2; H/X/RY rotation-readout circuit; H/CX readout q1–q3; H/CX coupling edge q1–q3; one-qubit RY unknown-cause observation q3 | RY→RZ at operation 2 on q2; remove RZZ at operation 2 on q0–q2; readout q2 inversion and q3 degradation; RZZ error q0–q2; CX error/coupling q1–q3; combined RY→RZ plus q2 readout; unknown-cause counts | Readout: 0.04, 0.055, 0.20, 1.0; RZZ gate error 0.055; CX edge case uses simulator-applied CX error plus edge metadata for localization. 512 shots; seeds 20262001–20262003; threshold 0.05. |

The Phase 1 family is only two qubits. Phase 2 uses four-register configurations but only six named structures, with several strength variants sharing a family. These are saved, scenario-derived evaluations; their labels describe injected configurations, not independently adjudicated failures. Seed repeats do not create new families. The saved Phase 1/1.3/1.4 reports and JSON results, and Phase 2 `REPORT.md`, `scenario_manifest.json`, and case results are the inventory sources.

## Proposed scenario matrix

Use five new 4–5 qubit families. Do not reuse Bell-only, the rotation/RZZ phase sequence, the Phase 2 RY→RZ circuit, or the H/CX readout/coupler circuits. Keep all parameterized variants and faults of a base family under the same family key.

| Family (proposed structure) | Representative structure | Program interventions | Hardware interventions and target locations |
|---|---|---|---|
| F1: GHZ ladder with verification | H(q0), CX(q0,q1), CX(q1,q2), CX(q2,q3), measure all | Delete one ladder CX; substitute one ladder CX with CZ | Readout on q1/q3; CX error on q0–q1 or q2–q3 |
| F2: Toffoli compute/uncompute | Prepare controls, CCX(q0,q1,q2), uncompute with CX/H, measure | Delete CCX; alter target operand | CCX error q0–q1–q2; readout q2/q4 |
| F3: SWAP-routing circuit | Alternating SWAP network over a line with local X/H preparation, measure permuted outputs | Omit one SWAP; reverse one selected operand order where semantically distinct | SWAP error on internal versus boundary edge; readout on q0/q4 |
| F4: parity-check network | Prepare a nonuniform bit pattern; fanout parity with CX tree; uncompute/check; measure checks and data | Omit one parity CX; insert/remove an X before a check | CX calibration error on a check edge; readout on check/data qubits |
| F5: layered non-Clifford control | Local H/T/S and controlled phase-free CX network, repeated in two layers; measure all | Replace T with Tdg; delete one local phase gate | T or CX gate error on an interior location; readout on an interior qubit |

Each family has these scenario classes: (a) normal controls, (b) program-only single faults, (c) hardware-only single faults, (d) combined program+hardware faults, (e) uncertain cases. For each hardware mechanism use strengths 0.02 (subtle), 0.06 (near the existing 0.05 evidence threshold), and 0.15 (moderate); include 0.30 only as a severe reference. For program faults include one visibly consequential and one potentially masked intervention. The matrix must include at least one normal paired control for every intervention, and matched pairs hold circuit family, seed, shot count, and all other interventions constant.

**Feasibility boundary:** the local simulator applies readout probability and gate calibration error as a chance that an operation is skipped; this is not a depolarizing channel or a general coherent over-rotation. It also models T1/T2 relaxation over gate durations. The report's proposed faults therefore use supported gate operations, readout error, gate error, and optionally T1/T2. Do not describe skip probability as depolarizing noise. Confirm each gate/operand form against `src/quantum_simulator/gates.py` and the runner's circuit handling before a future execution. If the runner/localizer cannot handle CCX, SWAP, T/Tdg, or one proposed fault mutation end-to-end, mark that cell unsupported and document it; do not substitute a different mechanism post hoc. Do not attach edge-only metadata and claim it injects stochastic noise: Phase 2 documents that edge metadata aids localization, while the simulator-applied gate error creates the perturbation.

### Execution grid

Predeclare seeds **20263001, 20263002, 20263003, 20263004, 20263005**. Run each retained scenario at **256, 1024, and 4096 shots** for a shot-count curve. Use identical seed values in paired conditions. This yields 15 seed/shot observations per scenario; run counts depend on the supported scenario cells after a design-only support audit, which must be frozen before execution. Keep the threshold fixed at 0.05 for the primary benchmark; evaluate any alternative thresholds only as a secondary, explicitly non-primary analysis selected from calibration data.

## Label generation independent of diagnosis

1. Freeze a scenario ledger before any benchmark execution. Record the ideal reference circuit, the predeclared program mutation, the injected hardware profile, affected physical indices/edges, intervention strength, and label rules. A reviewer or deterministic generator creates labels from these inputs without reading predictions, evidence rankings, or diagnosis categories.
2. For each case assign an intervention truth set: `NORMAL` if no program or hardware intervention; `PROGRAM_FAULT` if only a program mutation; `HARDWARE_ANOMALY` if only a hardware intervention; `COMBINED` if both are injected; and `UNCERTAIN` if the causal intervention is deliberately not disclosed/identifiable from allowed evidence. Preserve multi-label truth for combined cases. Do not relabel a fault as normal because its sampled output happens to look nominal.
3. Independently compute each ideal expected distribution from the declared reference circuit using an exact statevector/probability calculation with noise disabled, or a reviewed exact reference table. Compare observed counts to that fixed reference only for the separately reported output-deviation statistic. This computation is not the diagnosis result. A predeclared statistical test and correction policy must be frozen before execution; keep injected-cause truth distinct from whether finite-shot output crossed the detection rule.
4. Uncertain controls include (i) an injected deviation with calibration and program-comparison evidence withheld from the diagnosis input, and (ii) paired mechanisms selected in advance to be observationally non-identifiable at the chosen measurements. Declare expected abstention before execution. Do not use an arbitrary counts vector as if it were a physical intervention; if a synthetic fixture is used, label it explicitly as such and exclude it from simulator-generated detection-rate estimates.
5. Record `UNSUPPORTED`, runtime failure, and simulator failure separately; never count them as diagnosis successes or remove them silently.

## Leakage safeguards and family split

Use immutable family keys F1–F5 and split by the entire family, including all strengths, mutations, seeds, shot counts, paired controls, and repeated variants. Proposed allocation: **development/training F1–F2; calibration/threshold selection F3; locked test F4–F5**. This leaves two unseen test families and one unseen calibration family. A learned component may fit only development families. Any preprocessing, feature selection, model selection, and threshold choice use development and calibration data only. Freeze the model, feature schema, calibration mapping, and thresholds before opening test predictions. If the rule pipeline has no learned component, retain the same family separation for threshold calibration and report the locked test without retuning.

Model inputs may include only declared available diagnostic evidence: observed counts/statistics, the expected circuit/reference, and hardware calibration evidence that would genuinely be available at diagnosis time. Exclude scenario ID/name, family ID, split, truth class/cause, mutation ID, intervention type/strength, target indices, expected detection, file path, seed, and shot-count labels. Do not include a simulator's hidden injected profile as a sensor readout unless that value is explicitly designated as available calibration evidence; if calibration is supplied, the input must be the same calibration interface and fields available in a real diagnosis. Store truth ledger separately, join only after predictions are frozen, and strip identifiers before feature construction. Audit feature columns and serialized input payloads before execution.

## Metrics and predeclared acceptance criteria

Report bootstrap 95% intervals resampling at the **family** level where estimable, plus per-seed results and exact numerators/denominators. With only two locked test families, family-level intervals will be coarse; state this limitation and do not imply population-level certainty.

| Measurement | Primary reporting | Predeclared acceptance on locked test |
|---|---|---|
| Output anomaly detection | Sensitivity, specificity, balanced accuracy, AUROC/AUPRC when scores exist; calibration and detection rates by strength, shots, and family; count-based deviation statistic reported separately | Sensitivity ≥0.80 on moderate/severe injected cases; specificity ≥0.90 on normal controls; report subtle/near-threshold cases without a forced pass threshold |
| Program vs hardware classification | Confusion matrix and macro-F1 for normal/program/hardware/combined/uncertain; one-vs-rest metrics; combined detection measured separately from single-cause assignment | Macro-F1 ≥0.75 over identifiable classes; ≥0.80 recall for program-only and hardware-only; combined cases must not be falsely scored as a correct single-cause diagnosis |
| Qubit/coupling localization | Exact target-set accuracy, component precision/recall/F1, top-k recall; per-target and per-family counts | ≥0.70 exact target-set accuracy on supported, identifiable single-fault cases; no fabricated localization for uncertain cases |
| Abstention and insufficient evidence | Coverage, selective risk, risk–coverage curve, uncertain-case abstention recall, incorrect confident diagnosis rate, failure/unsupported rates | ≥0.80 abstention recall on predeclared uncertain cases; confident error rate ≤0.10 among covered cases; report coverage alongside all selective metrics |
| Stratified performance | All above by family, intervention strength, fault type/location, and shots (256/1024/4096) | No pooled score may hide a test family with moderate/severe balanced accuracy below 0.65; subtle strata remain descriptive |
| Reproducibility | Per-seed scores, mean/SD, range, seed-wise paired differences, runtime, and deterministic input/config hashes | Same frozen inputs and seed reproduce identical outputs; moderate/severe primary metric SD across five seeds ≤0.10 where denominator permits |

Acceptance thresholds are design targets, not claims about current performance. No threshold is selected from F4/F5. If a metric has too few cases or an unsupported operation, report it as not estimable/unsupported rather than relaxing criteria after observing results.

## Compute and runtime risks

The proposed 4–5 qubit circuits are small statevectors, but the scenario × strength × location × class × five seeds × three shot levels can create hundreds of runs. Counts scale approximately linearly with shots and operations. Traced execution/diagnosis may dominate statevector sampling. Freeze a maximum cell count and estimated wall time after a design-only support audit; record host, package versions, and timeout policy. Do not begin execution as part of this task. A pilot, if later authorized, must use development-only families and cannot be used to alter locked test thresholds after test access.

## Exact next authorized step

**Next step:** perform a design-only compatibility audit of F1–F5 against the existing circuit parser, simulator, hardware profile, and diagnosis/localization interfaces; write down supported/unsupported matrix cells and freeze the scenario ledger, split assignments, input schema, and run-count/runtime estimate. This step must not execute simulations, train models, generate datasets, or inspect any held-out predictions. Benchmark execution requires a separate explicit authorization after that frozen design is reviewed.
