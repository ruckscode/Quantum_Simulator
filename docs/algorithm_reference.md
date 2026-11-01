# Simulator algorithms and mathematical explanation

This document follows one execution from input to diagnosis. The runtime
explanation API, `quantum_simulator.explain_diagnosis(result)`, returns the
same algorithm stages with values from a particular run. It explains computed
evidence; it does not turn a score into a correctness guarantee.

## 1. Circuit state evolution

For `n` qubits the simulator stores a complex state vector of size `2^n`:

\[
|\psi_0\rangle=|0\rangle^{\otimes n},\qquad
|\psi_{t+1}\rangle=U_t|\psi_t\rangle.
\]

Each supported gate is a unitary matrix applied to its target amplitudes. A
computational-basis measurement outcome `x` has Born probability
\[
p(x)=|\langle x|\psi\rangle|^2,
\]
and a finite run of `N` shots samples counts
\[
(c_1,\ldots,c_K)\sim\operatorname{Multinomial}(N;p_1,\ldots,p_K),
\quad \hat p_i=c_i/N.
\]

This is a statevector simulator: memory and runtime grow exponentially with
qubit count. It is not a stabilizer simulator and does not claim hardware
equivalence.

## 2. Optional hardware-noise simulation

`HardwareModel` supplies synthetic or captured calibration values. During a
gate, the current simplified model skips that gate with probability `e_g` and
applies a relaxation/dephasing step over its configured duration `t`.
For positive `T1`,
\[
\gamma_1=1-e^{-t/T_1}.
\]
The pure-dephasing rate used by the implementation is
\[
\Gamma_\phi=\max(0,1/T_2-1/(2T_1)),\qquad
p_Z=\tfrac12(1-e^{-t\Gamma_\phi}).
\]
At measurement, each qubit's configured readout error is applied as a bit-flip
probability. This compact stochastic model omits coherent over-rotation,
correlated multi-qubit error, crosstalk, leakage, drift, and queue/runtime
effects. A synthetic fixture is a controlled test case, not “real-like” proof.
The IBM adapter normalizes reported T1/T2 and gate durations to microseconds
before combining them in this model; the original reported duration/unit stays
in calibration-capture metadata.

## 3. Distribution comparison

The primary distance is total variation distance:
\[
\operatorname{TVD}(p,q)=\tfrac12\sum_x|p(x)-q(x)|.
\]
The engine also reports Hellinger distance
\[
H(p,q)=\sqrt{\tfrac12\sum_x(\sqrt{p(x)}-\sqrt{q(x)})^2}
\]
and Jensen–Shannon divergence using base-2 KL divergence
\[
\operatorname{JSD}(p,q)=\tfrac12D_{KL}(p\|m)+\tfrac12D_{KL}(q\|m),
\quad m=(p+q)/2.
\]
These are descriptive distances; TVD drives the legacy probability-only
threshold.

### Finite-shot rule

When the engine receives normalized distributions and no count samples, the
effective threshold is
\[
\tau=\max(\tau_{user},1/\sqrt{N}),\qquad
\text{anomaly}=[\operatorname{TVD}>\tau].
\]
This guard is a simple tolerance, not a formal confidence interval.

### Count-aware tests

When both expected and observed samples are counts, the engine pools outcomes,
randomly permutes the pooled labels while preserving both sample sizes, and
estimates
\[
\hat p=(1+\#\{TVD_b\ge TVD_{obs}\})/(B+1).
\]
It reports an anomaly only when `p <= alpha` **and** observed TVD exceeds the
practical effect threshold. When the reference is an exact ideal distribution
and only observed counts are sampled, a seeded multinomial parametric bootstrap
is used instead. Both methods use an add-one estimate to avoid a zero p-value.
The default permutation/bootstrap seeds and iteration counts live in
`src/quantum_simulator/count_statistics.py`.

## 4. Program-fault localization

Expected and observed operations are normalized to gate signatures. Alignment
uses the standard edit-distance recurrence
\[
D(i,j)=\min\{D(i-1,j)+1,D(i,j-1)+1,D(i-1,j-1)+[a_i\ne b_j]\}.
\]
It supplies operation-mismatch candidates. A mismatch alone is not a diagnosed
fault: the engine also checks whether the counterfactual output distribution
changes beyond the configured threshold and whether the candidate can affect
measured outputs. This prevents behaviorally equivalent edits from being
reported as proven execution faults.

## 5. Hardware-fault localization

Calibration candidates are filtered for relevance to used gates, qubits, and
coupling edges. The localizer scores evidence such as readout error, `T1/T2`,
gate/edge error, unsupported or unavailable resources, and circuit context.
For a qubit, the implementation uses
\[
S_q=1.5I_{unavailable}+4e_{ro}I(e_{ro}>.05)
+\max(0,(100-T_1)/100)I(T_1<100)
+\max(0,(100-T_2)/100)I(T_2<100)+.5\min(n_q,3).
\]
For a gate and edge it uses `S_g=1.5 I_unsupported + 10 e_g I(e_g>.05)` and
`S_edge=1.5 I_unsupported + 8 e_edge I(e_edge>.05) + .75 min(n_edge,3)`.
These are heuristic candidate ranks, not probabilities. Threshold and
weighting constants live in `hardware_fault_localizer.py`; they are not a
learned probability and do not have a universal accuracy interpretation. For reported calibration, a caller
must set a maximum age to establish freshness. Stale/unknown calibration is
retained as context but excluded from causal attribution. Synthetic calibration
is marked synthetic and makes no physical freshness claim.

## 6. Final decision and abstention

The final layer combines statistical anomaly status with separate program and
hardware evidence. It can return `NO_ANOMALY`, `PROGRAM_FAULT`,
`HARDWARE_ANOMALY`, or `AMBIGUOUS / INSUFFICIENT_EVIDENCE`. A program diagnosis
requires a behavior-changing source candidate. A hardware diagnosis requires
relevant calibration evidence. Competing causes or missing evidence cause
abstention rather than forced attribution.

## 7. Optional classical diagnosis model

`diagnose_circuit(..., classifier=...)` is the only explicit hook for an
observable-feature classifier. No classifier is loaded implicitly. The bundled
candidate computes five distribution features, imputes missing values with
training medians, standardizes them, and applies multiclass Logistic Regression:
\[
z=(x-\mu)/s,\quad
P(y=k|x)=\frac{e^{w_k^Tz+b_k}}{\sum_j e^{w_j^Tz+b_j}}.
\]
Its probabilities are uncalibrated; metadata says deployment validation was
not performed. It is a research candidate and the independent evidence engine
remains authoritative. QMutBench and the cross-source models have incompatible
targets/features, so they are not called here.

## 8. Real IBM execution

The optional IBM adapter converts the user circuit, obtains backend and
calibration information, checks preflight/freshness, transpiles/maps, submits
the job only after preflight succeeds, then compares returned counts with the
ideal reference and available calibration. The API token is supplied explicitly
to the call. The adapter does not call IBM account-save functions. Real hardware
execution is still not an independent ground-truth diagnosis label.
