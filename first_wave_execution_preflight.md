# First-wave execution preflight

**Decision: STOP — execution preflight failed. No reports were generated.** Independent checks confirmed the matrix, splits, seed schedule, and circuit observability, but two execution/data-integrity requirements are unresolved. Per the requested decision gate, the experiment must not start until they are addressed.

## Verified specification identity

SHA-256 hashes of the exact reviewed files:

| Artifact | SHA-256 |
|---|---|
| `data/qmutbench_ml/final/first_wave_experiment_spec_v3.md` | `CB6D1FE0031036483234367288D31C6CE4CF4F9A4E71206B651E6BA15F1AD52F` |
| `data/qmutbench_ml/final/first_wave_experiment_spec_v3.json` | `0B01CEF12E3C1D54ACDD383B782A118277C99DBC14910625F1D8167A9CD47A55` |
| `data/qmutbench_ml/final/first_wave_spec_v3_review.md` | `AA285971A56FADCE49A1BD36133DCFD78616B0EA42B7A4B9C885A4BCA92743A9` |
| `data/qmutbench_ml/final/first_wave_spec_v3_review.json` | `9BD45B930EFAE7C85D83176003257F5D9C4069C725F3DAA0671B7E14AC192788` |

These hashes identify the files reviewed here. They do not make an incomplete execution protocol complete.

## Preflight checks

| Check | Result | Evidence |
|---|---|---|
| Supported interventions | PASS | `HardwareModel` validates probabilities in `[0,1]` (`src/quantum_simulator/hardware_model.py::_validate_probability`, line 23); gate error is looked up by gate/qubit (`gate_error`, line 244). `QuantumSimulator.run_circuit`, line 344, skips on `rng.random() < gate_error`. Measurement applies configured per-qubit readout error. `run_experiment.py::_build_hardware_model` consumes `hardware_parameters`; `run_experiment` records `hardware_model_applied` and `noise_parameters_applied=false` (lines 436–437). |
| T1/T2 excluded | PASS | V3 marks T1/T2 `BLOCKED_AND_EXCLUDED`; no relaxation arm is in the matrix. RX duration is zero. No relaxation or real-hardware claim is made. |
| RX sampling semantics | PASS, with blocker below | `run_shots`, line 485, iterates shots; `run_circuit` samples once per calibrated operation. One RX invocation × 1,024 shots means 1,024 skip decisions. The value is an operation-skip probability per invocation. |
| Observable circuits | PASS, static analysis | Every family has `RY(q0,pi/3)`, one `RX(q0,theta)`, measures q0, and spectator operations are only on q1+. With theta in `(0,pi/2)`, the exact q0 marginal depends on RX application; all combined readout levels are `<0.5`, preserving sensitivity. The specified mutations also act on measured q0. This establishes exact-distribution sensitivity, not finite-shot classifier performance. |
| Matrix arithmetic | PASS | 16×6 control rows = 96; 12 cells×4 families×3 repetitions = 144; total 240. Classes 48/48/144. Every cell maps to four families and 12 reports. |
| Split isolation | PASS | F01–F08 train, F09–F12 validation, F13–F16 exploratory test. Independently checked each family has one split and each cell spans 2/1/1 families across train/validation/test. |
| Seed schedule | PASS | Independently recalculated all 48 values as `uint32_be(SHA256("qsim-first-wave-v3|Fxx|r")[:4])`, mapping zero to one. 48/48 matched; all are nonzero and unique. |
| Output isolation | PASS for proposed location | `data/qmutbench_ml/final/first_wave_v3_reports` did not exist at preflight. This location would not overwrite existing reports. No directory was created. |

## Blocking issues

### 1. Realized RX skip counts cannot be captured by the specified report execution path

V3 explicitly says the report format does not expose per-shot gate skip events (v3 spec, RX semantics and metadata sections). The implementation confirms this: `QuantumSimulator.run_shots` at `src/quantum_simulator/quantum_simulator.py:471–495` aggregates only measurement outcomes and returns the counts dictionary. Although `run_circuit` computes `gate_failed` at line 344, it does not return or persist that value. `run_experiment.py` serializes distributions/counts and application flags, not a realized skip count.

`QuantumSimulator.run_circuit_traced` includes a `gate_failed` field (same simulator file, around lines 498 and 554–592), but it executes a separate traced path, not the 1,024-shot `run_shots` stream used to produce the saved report. Its result cannot be represented as the realized skip count for those report shots without proving and implementing a trace that observes the exact same execution stream. Re-running a separate trace would produce unrelated stochastic events and would misstate the count. A design count of 1,024 is not the realized number of skips.

**Required correction:** define and review a supported way to capture the gate-skip event for each actual shot in the same execution that supplies the report's observed distribution, without changing production diagnosis behavior. Alternatively, revise the required deliverable and acceptance criteria to explicitly treat realized skip count as unavailable; that would require owner approval because this task requests realized noise counts where applicable. Do not infer the realized count from the observed bitstring distribution.

### 2. Generation control protocol is not specified

The v3 documents freeze scenarios and seeds, but do not define the requested execution controls: progress checkpoints, failure classification/handling, safe resume behavior, duplicate detection, maximum retry policy, resource monitoring, or a stopping threshold. The available `run_experiment` writes directly to the requested output path (`experiments/run_experiment.py:447–451`) and overwrites an existing same-name path; `run_config_batch` repeats each configuration for reproducibility (`:463–472`), so it is not the desired 240-row resumable runner. `experiments/full_dataset.py::generate_dataset` removes existing `run-*.json` files before generation (`:262–268`) and is unsafe for this wave. No execution should use either path as a bulk runner without a separately reviewed safe orchestration protocol.

**Required correction:** predeclare an isolated-runner protocol with deterministic scenario IDs, exclusive output directory creation, atomic per-row writes, completion/failure status, progress log, skip-completed-on-resume checks keyed by spec hash and scenario ID, no silent reruns/replacements, bounded retry policy (or no retries), resource/failure stopping threshold, and an explicit all-rows/stop condition. Specify final reconciliation of planned, completed, failed, and skipped rows.

## Disposition and outputs

The current v3 circuit and scenario design passed static observability and arithmetic checks. The unresolved items above are implementation/data-integrity blockers for this requested execution, not evidence that the underlying interventions are unsupported. T1/T2 remains excluded. No simulations, reports, ML models, packages, or real hardware were used. The destination directory remains absent.

Only this preflight report and its JSON companion were created. The generation manifest, run log, and data-validation report are deferred because the decision gate failed before generation.

**BLOCKED — no reports generated.**
