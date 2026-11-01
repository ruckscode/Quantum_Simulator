# Project mathematical reference: simulator, classical ML, and QML

This document states the mathematics implemented or evaluated in the project,
the purpose of each process, where it runs, what is heuristic, and what remains
unvalidated. It complements the [algorithm reference](algorithm_reference.md),
[classical benchmark report](../reports/experiments/classical_ml_final_results_20261030.md),
and [matched QML/classical report](../data/curated_qml_training_v1/matched_qml_classical_comparison.md).

## 0. Status and notation

| Label | Meaning |
|---|---|
| Runtime | Runs when the simulator is used. |
| Benchmark | Used in an offline training or evaluation experiment; not automatically used for diagnosis. |
| Heuristic | A manually chosen score/threshold, not a calibrated probability. |
| Not established | No validated result in this repository supports the claim. |

Use n for qubits, x for an input feature vector, y for a class, N for a number
of shots or examples as indicated, p for an ideal distribution, and p-hat for
an empirical distribution. Gate angles are in radians. A circuit is an ordered
sequence; the rightmost matrix acts first on a column state vector.

## 1. Project map: what computes what

| Stage | Mathematics | Purpose and current role |
|---|---|---|
| Circuit execution | State-vector unitary evolution, measurement, sampling | Runtime ideal simulator |
| Hardware emulation | Gate omission, T1/T2 trajectories, readout flips | Optional synthetic or captured-calibration simulation |
| Anomaly detection | TVD, Hellinger, JSD, count tests | Runtime evidence |
| Program localization | Operation edit distance and counterfactual output comparison | Runtime when an intended/reference circuit is supplied |
| Hardware localization | Calibration thresholds and weighted component scores | Runtime heuristic evidence ranking |
| Source classical ML | TF-IDF and linear/NB/KNN/tree classifiers | Offline source-specific benchmarks |
| QMutBench ML | Extra Trees on mutation metadata/QASM-difference features | Offline mutation-outcome prediction |
| Observable Logistic Regression | Five distribution features, imputation/scaling, softmax | Optional adapter; development candidate, not deployment validated |
| QML | Variational circuit, fidelity kernel, reservoir feature map | Separate research experiments; not called by diagnose_circuit() |
| IBM integration | Calibration extraction/conversion, transpilation and execution | Optional external path; not live-fidelity validated |

The data-source ML scores do not measure end-to-end simulator diagnosis
accuracy. Their labels (comments, mutation outcome, source state, backend
identity, execution result) are not one shared independently verified
program-fault/hardware-anomaly label.

## 2. Quantum state and circuit mathematics

### 2.1 State, gates, and probabilities

The simulator stores a pure state of 2^n complex amplitudes:

\[
|\psi\rangle=\sum_{x=0}^{2^n-1}\alpha_x|x\rangle,
\qquad \sum_x|\alpha_x|^2=1.
\]

The initial state is |0> tensor n. For operations U_1,...,U_m:

\[
|\psi_m\rangle=U_mU_{m-1}\cdots U_1|\psi_0\rangle.
\]

Each gate matrix acts on its specified qubit axes. Internal storage uses q0 as
the leftmost tensor axis; public statevector/count display uses Qiskit order:
q0 is the least-significant state bit and the highest classical-bit index is
printed leftmost. Implicit measurements map q[i] to c[i].

### 2.2 One-qubit gate matrices

\[
I=\begin{bmatrix}1&0\\0&1\end{bmatrix},\quad
X=\begin{bmatrix}0&1\\1&0\end{bmatrix},\quad
Y=\begin{bmatrix}0&-i\\i&0\end{bmatrix},\quad
Z=\begin{bmatrix}1&0\\0&-1\end{bmatrix}.
\]

\[
H=\frac1{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix},\quad
S=\begin{bmatrix}1&0\\0&i\end{bmatrix},\quad
S^\dagger=\begin{bmatrix}1&0\\0&-i\end{bmatrix}.
\]

\[
T=\operatorname{diag}(1,e^{i\pi/4}),\quad
T^\dagger=\operatorname{diag}(1,e^{-i\pi/4}),\quad
\sqrt X=\frac12\begin{bmatrix}1+i&1-i\\1-i&1+i\end{bmatrix}.
\]

The sxdg gate is the conjugate transpose of square-root-X. Aliases are:
i/id/identity = I; h/hadamard = H; sdg = S-dagger; tdg = T-dagger;
sxdg = square-root-X dagger.

Parameterized rotations, with c=cos(theta/2), s=sin(theta/2)

\[
R_X(\theta)=\begin{bmatrix}c&-is\\-is&c\end{bmatrix},\quad
R_Y(\theta)=\begin{bmatrix}c&-s\\s&c\end{bmatrix},\quad
R_Z(\theta)=\operatorname{diag}(e^{-i\theta/2},e^{i\theta/2}).
\]

\[
P(\lambda)=\operatorname{diag}(1,e^{i\lambda}),\qquad U_1(\lambda)=P(\lambda).
\]

\[
U_2(\phi,\lambda)=\frac1{\sqrt2}
\begin{bmatrix}1&-e^{i\lambda}\\e^{i\phi}&e^{i(\phi+\lambda)}\end{bmatrix}.
\]

\[
U_3(\theta,\phi,\lambda)=U(\theta,\phi,\lambda)=
\begin{bmatrix}
\cos(\theta/2)&-e^{i\lambda}\sin(\theta/2)\\
e^{i\phi}\sin(\theta/2)&e^{i(\phi+\lambda)}\cos(\theta/2)
\end{bmatrix}.
\]

### 2.3 Two-qubit gates

For controlled gates the first operand is the control and the second is the
target. In the local |control,target> basis:

\[
CX=\begin{bmatrix}1&0&0&0\\0&1&0&0\\0&0&0&1\\0&0&1&0\end{bmatrix},
\quad CZ=\operatorname{diag}(1,1,1,-1),
\quad SWAP=\begin{bmatrix}1&0&0&0\\0&0&1&0\\0&1&0&0\\0&0&0&1\end{bmatrix}.
\]

\[
CH=\begin{bmatrix}
1&0&0&0\\0&1&0&0\\0&0&1/\sqrt2&1/\sqrt2\\0&0&1/\sqrt2&-1/\sqrt2
\end{bmatrix},
\quad CP(\theta)=\operatorname{diag}(1,1,1,e^{i\theta}).
\]

With c=cos(theta/2), s=sin(theta/2):

\[
CR_X(\theta)=\begin{bmatrix}1&0&0&0\\0&1&0&0\\0&0&c&-is\\0&0&-is&c\end{bmatrix},
\quad
CR_Y(\theta)=\begin{bmatrix}1&0&0&0\\0&1&0&0\\0&0&c&-s\\0&0&s&c\end{bmatrix},
\]

\[
CR_Z(\theta)=\operatorname{diag}(1,1,e^{-i\theta/2},e^{i\theta/2}).
\]

\[
R_{XX}(\theta)=e^{-i\theta(X\otimes X)/2}
=\begin{bmatrix}c&0&0&-is\\0&c&-is&0\\0&-is&c&0\\-is&0&0&c\end{bmatrix},
\]

\[
R_{ZZ}(\theta)=e^{-i\theta(Z\otimes Z)/2}
=\operatorname{diag}(e^{-i\theta/2},e^{i\theta/2},e^{i\theta/2},e^{-i\theta/2}).
\]

### 2.4 Three-qubit gates

CCX/Toffoli implement the Boolean map

\[
|a,b,c\rangle\mapsto|a,b,c\oplus(ab)\rangle.
\]

CSWAP/Fredkin implement

\[
|a,b,c\rangle\mapsto
\begin{cases}|a,c,b\rangle,&a=1,\\|a,b,c\rangle,&a=0.\end{cases}
\]

The implemented relative-phase Toffoli rccx acts as

\[
|101\rangle\mapsto-|101\rangle,\quad
|110\rangle\mapsto i|111\rangle,\quad
|111\rangle\mapsto-i|110\rangle,
\]

and leaves other basis states unchanged.

### 2.5 Measurement, reset, barriers, and sampling

Born's rule for measurement outcome x is

\[
P(x)=|\langle x|\psi\rangle|^2=|\alpha_x|^2.
\]

The projector and conditioned post-measurement state are

\[
\Pi_x=|x\rangle\langle x|,\qquad
|\psi_x\rangle=\Pi_x|\psi\rangle/\sqrt{P(x)}.
\]

With N_s shots and outcome probabilities p_1,...,p_K:

\[
(C_1,...,C_K)\sim\operatorname{Multinomial}(N_s;p_1,...,p_K),
\qquad \hat p_j=C_j/N_s.
\]

Repeated measurements collapse at each step and write into one flat
classical-bit array; a later measurement overwrites its destination bit.
Barriers do not change amplitudes. Reset implements

\[
\mathcal R(\rho)=K_0\rho K_0^\dagger+K_1\rho K_1^\dagger,
\quad K_0=|0\rangle\langle0|,\quad K_1=|0\rangle\langle1|.
\]

The simulator samples a branch and normalizes its pure-state trajectory;
averaging trajectories reproduces the mixed channel.

State memory is O(2^n). A complex128 state alone requires 16*2^n bytes. The
optional memory budget estimates principal arrays; it is not a total process
memory guarantee.

## 3. Hardware model and synthetic anomalies

### 3.1 Gate skip

Only when a hardware model is supplied, a gate with calibration error e_g
samples

\[
F_g\sim\operatorname{Bernoulli}(e_g).
\]

If F_g=1, the ideal gate is skipped. Relaxation/dephasing still accrues during
the gate duration. This is a simple gate-skip approximation, not depolarizing
noise or a general quantum channel.

### 3.2 T1 relaxation

For duration t and positive T1:

\[
\gamma_1=1-e^{-t/T_1}.
\]

The amplitude-damping channel is

\[
\mathcal E_{T_1}(\rho)=K_0\rho K_0^\dagger+K_1\rho K_1^\dagger,
\quad K_0=|0\rangle\langle0|+\sqrt{1-\gamma_1}|1\rangle\langle1|,
\quad K_1=\sqrt{\gamma_1}|0\rangle\langle1|.
\]

The code samples equivalent jump/no-jump trajectories.

### 3.3 T2 pure dephasing

The model removes coherence decay already caused by T1:

\[
\Gamma_\phi=\max(0,1/T_2-1/(2T_1)),\qquad
p_Z=\tfrac12(1-e^{-t\Gamma_\phi}).
\]

It applies Z with probability p_Z:

\[
\mathcal E_\phi(\rho)=(1-p_Z)\rho+p_ZZ\rho Z.
\]

Missing, nonfinite, or nonpositive calibration times suppress the corresponding
term.

### 3.4 Readout and synthetic calibration

For true bit b and readout error e_r:

\[
P(\tilde b\ne b)=e_r,\qquad P(\tilde b=b)=1-e_r.
\]

Physical collapse uses the true bit; only its classical report can flip.
Synthetic values in HardwareModel are explicit local test assumptions, not
values inferred from IBM. With a configured 0.35 readout error, for example,
the synthetic assumption is a 35% independent reporting flip per measured
bit. No API key means no current IBM calibration is retrieved.

### 3.5 IBM time units

The IBM adapter retains original values/units in capture metadata and converts
the model's times to microseconds:

\[
t_{\mu s}=10^6t_s=10^{-3}t_{ns}=10^3t_{ms}.
\]

That makes t/T1 and t/T2 dimensionless. Fixture validation does not establish
live IBM fidelity.

## 4. Distribution metrics and anomaly tests

Let p(x) be expected and q(x) observed probabilities over the union of
outcomes, assigning zero to absent outcomes.

### 4.1 Distances

\[
\operatorname{TVD}(p,q)=\frac12\sum_x|p(x)-q(x)|.
\]

\[
H(p,q)=\sqrt{\frac12\sum_x(\sqrt{p(x)}-\sqrt{q(x)})^2}.
\]

For m=(p+q)/2:

\[
\operatorname{JSD}(p,q)=\tfrac12D_{KL}(p\|m)+\tfrac12D_{KL}(q\|m),
\quad D_{KL}(p\|q)=\sum_{x:p(x)>0}p(x)\ln(p(x)/q(x)).
\]

The implementation uses the natural logarithm, so JSD is in nats. JSD and
Hellinger are reported descriptive metrics; TVD primarily drives decisions.

### 4.2 Probability-only guard

For user tolerance tau_u and shots N_s:

\[
\tau_{eff}=\max(\tau_u,1/\sqrt{N_s}),\qquad
\text{flag iff TVD}(p,q)>\tau_{eff}.
\]

The 1/sqrt(N_s) guard is a rough scale tolerance, not a formal confidence
interval.

### 4.3 Two independent count samples

The conditional permutation test pools categorical observations and randomly
reassigns them while preserving sample sizes. For B permutations:

\[
\hat p=\frac{1+\#\{b:\operatorname{TVD}_b\geq\operatorname{TVD}_{obs}\}}{B+1}.
\]

It reports an anomaly only if both

\[
\hat p\leq\alpha\quad\text{and}\quad\operatorname{TVD}_{obs}>\tau_{effect}.
\]

The add-one correction avoids a zero Monte Carlo p-value.

### 4.4 Exact probabilities versus observed counts

For exact reference p and observed N_s shots:

\[
C^{(b)}\sim\operatorname{Multinomial}(N_s,p),\quad
TVD_b=\tfrac12\lVert C^{(b)}/N_s-p\rVert_1.
\]

The bootstrap tail p-value uses the same add-one rule and practical-effect
gate. Thus a tiny finite-shot difference is not automatically an anomaly.

### 4.5 Five optional observable-classifier features

For observed outcome probabilities q_j:

\[
f_1=TVD(p,q),\quad f_2=JSD(p,q),\quad f_3=H(p,q),
\]

\[
f_4=-\sum_{j:q_j>0}q_j\log_2q_j,\qquad f_5=\max_jq_j.
\]

These encode distribution change/concentration, not a fault location.

## 5. Program-fault localization

### 5.1 Reference and intent

A circuit lists operations; it does not state what the programmer intended.
Detecting a missing gate requires an intended circuit, a formal output
invariant, or an expected observable. The engine does not infer missing intent.
Given intended program A=(a_1,...,a_m) and submitted program B=(b_1,...,b_n),
it compares their operation signatures and modeled outputs.

### 5.2 Signature mismatch score and alignment

The simple signature mismatch score is

\[
s(a,b)=0.5I[g_a\ne g_b]+0.25I[q_a\ne q_b]+0.25I[\theta_a\ne\theta_b].
\]

Insertions/deletions have priority score 1.0. These scores are not calibrated
fault probabilities.

The sequence alignment uses dynamic-programming edit distance:

\[
D(i,j)=\min\begin{cases}
D(i-1,j)+1 & \text{delete},\\
D(i,j-1)+1 & \text{insert},\\
D(i-1,j-1)+I[a_i\ne b_j] & \text{match/substitute}.
\end{cases}
\]

### 5.3 Behavioral-impact gate

The simulator computes ideal terminal distributions p_A and p_B from the
expected/submitted gates and compares

\[
\Delta_{out}=\operatorname{TVD}(p_A,p_B).
\]

A source mismatch is promoted only when it can affect measured outputs and
counterfactual output change exceeds the configured threshold. Identity
omission or an unmeasured, behaviorally irrelevant edit is not established as
a behavior-changing fault. Unsupported operations limit trace-based
localization; an unavailable trace placeholder is not evidence of zero
deviation.

## 6. Hardware-fault localization

Hardware component scores are manually weighted ranks, not probabilities.

For qubit q:

\[
S_q=1.5I_{unavailable}+4e_{ro}I(e_{ro}>0.05)
+\max(0,(100-T_1)/100)I(T_1<100)
+\max(0,(100-T_2)/100)I(T_2<100)+0.5\min(n_q,3).
\]

For gate g and edge e:

\[
S_g=1.5I_{unsupported}+10e_gI(e_g>0.05),
\]

\[
S_e=1.5I_{unsupported}+8e_eI(e_e>0.05)+0.75\min(n_e,3).
\]

Here n_q/n_e count relevant uses and T1/T2 are in microseconds. Thresholds
and coefficients are engineering heuristics; they are not fitted against a
representative real-hardware fault-label dataset. Absolute duration is used
for simulated relaxation, but is not independently considered a fault without
an expected-duration reference.

## 7. Diagnosis decision policy

| Result | Current rule/evidence requirement |
|---|---|
| NO_ANOMALY | No statistical output anomaly and no contradictory direct-program or relevant calibration evidence |
| PROGRAM_FAULT | Source mismatch changes modeled measured outcomes and no competing hardware cause dominates |
| HARDWARE_ANOMALY | Statistical output deviation with relevant calibration evidence, or relevant hardware evidence remains despite matching counts |
| AMBIGUOUS / INSUFFICIENT_EVIDENCE | Competing causes, unsupported operations, missing evidence, or anomaly without localized cause |

This is a rule/statistics evidence policy, not an automatically trained
three-class classifier. If two distinct causes have identical measured output
distributions under the available measurements, observational data cannot
identify which cause occurred.

## 8. Classical ML: feature and classifier mathematics

### 8.1 TF-IDF source tasks

Text/code/QASM is converted to sparse word/character n-gram features. A
simplified sublinear-TF, smoothed-IDF weight is

\[
tfidf(t,d)=\left(1+\ln count(t,d)\right)
\left(1+\ln\frac{1+N_d}{1+df(t)}\right).
\]

The vectorizer L2-normalizes document vectors:

\[
x_d\leftarrow x_d/\lVert x_d\rVert_2.
\]

For linear score f_c(x)=w_c^T x+b_c, binary logistic probability is

\[
P(y=1\mid x)=\sigma(w^Tx+b),\quad \sigma(a)=1/(1+e^{-a}),
\]

and multiclass logistic uses

\[
P(y=c\mid x)=\frac{e^{w_c^Tx+b_c}}{\sum_j e^{w_j^Tx+b_j}},
\quad L_{CE}=-\sum_i\ln P(y_i\mid x_i).
\]

The fitted coefficients minimize regularized loss, typically
\[
\arg\min_{W,b}\;L_{CE}(W,b)+\lambda\lVert W\rVert_2^2,
\]
with the selected regularization parameter chosen inside the grouped inner
folds. For binary labels y in {-1,+1}, LinearSVC minimizes the hinge-loss
equivalent objective
\[
\tfrac12\lVert w\rVert_2^2+C\sum_i\max(0,1-y_i(w^Tx_i+b)).
\]
This is the same maximum-margin principle as the slack-variable form above.

Linear SVM selects a maximum-margin boundary:

\[
\min_{w,b,\xi}\tfrac12\lVert w\rVert^2+C\sum_i\xi_i,
\quad y_i(w^Tx_i+b)\geq1-\xi_i,\quad\xi_i\geq0.
\]

Modified-Huber SGD uses margin z=yf(x):

\[
\ell_{MH}(z)=
\begin{cases}0,&z\geq1,\\(1-z)^2,&-1\leq z<1,\\-4z,&z<-1.\end{cases}
\]

Complement Naive Bayes estimates smoothed token probability from the
complement of class c:

\[
\widehat P(t\mid\neg c)=\frac{N_{t,\neg c}+\alpha}{N_{\neg c}+\alpha V}.
\]

This uses a conditional-independence/bag-of-features approximation. Sparse
cosine KNN uses

\[
sim(x,x_i)=\frac{x^Tx_i}{\lVert x\rVert\lVert x_i\rVert}
\]

and votes among the nearest training documents. The source search compared
104 linear/NB configurations and 32 cosine-KNN configurations under nested
grouped folds.

The other tabular QMutBench controls were:

- **Decision trees / Extra Trees / Random Forest:** a node selects feature j
  and threshold t to reduce class impurity. For class proportions p_k,
  \(G=1-\sum_kp_k^2\) (Gini) and \(H=-\sum_kp_k\ln p_k\) (entropy).
  Extra Trees averages randomized tree votes; Random Forest averages
  bootstrap-tree votes. Neither vote fraction is a calibrated probability
  unless separately calibrated.
- **Histogram Gradient Boosting:** additive stagewise fitting
  \(F_m(x)=F_{m-1}(x)+\eta h_m(x)\), where each shallow tree approximates
  the negative gradient of the selected loss over binned features.
- **RBF SVC:** \(K(x,x')=\exp(-\gamma\lVert x-x'\rVert_2^2)\), then the
  maximum-margin dual objective above with K replacing the dot product.
- **Dummy prior:** predicts the most common training class; it is a
  prevalence baseline, not a learned circuit model.

The four-feature matched comparison contains eight controls (Extra Trees,
histogram gradient boosting, Random Forest, SGD logistic, Logistic Regression,
LinearSVC, class-balanced Logistic Regression, and Dummy prior). The full
mutation-aware benchmark is a separate 23-feature Extra Trees model; its
77.34% matched full-feature baseline and 77.77% HistGradientBoosting screen
are protocol-specific reported values, while the selected full-feature Extra
Trees OOF result is 79.33%.

### 8.2 QMutBench features and Extra Trees

Target mapping:

\[
y=0\text{ for Survivor},\qquad y=1\text{ for WOO or OPO}.
\]

Metadata inputs include algorithm, operation, gate, dominant state,
qubit/gate counts, and mutation position. A categorical position band is
mapped to its midpoint and divided by 100. Original and mutant QASM are parsed
into gate/wire signatures; measurement and barrier operations are excluded.
The mutation-diff extractor finds common prefix length a and suffix length s:

\[
L_o=|O[a:|O|-s]|,\quad L_m=|M[a:|M|-s]|,\quad L_{diff}=L_o+L_m.
\]

Other derived values include prefix/suffix fractions, changed position,
expected-gate match, touched-qubit count, arity changes, local/single-edit
flags, and removed/added-gate context. Complex alignments receive unknown
local-context values instead of guessed context. The full-feature run used
23 inputs; this checkout does not contain the full trained bundle and raw
training source, so that full fit is not reproducible from a fresh clone alone.

For 100 Extra Trees, each tree gives the class proportions in the reached
leaf; the forest probability averages those leaf distributions:

\[
\hat P(y=c\mid x)=\frac1{100}\sum_{t=1}^{100}
\frac{N_{t,leaf(x),c}}{N_{t,leaf(x)}},
\qquad\hat y=\arg\max_c\hat P(y=c\mid x).
\]

This estimates a historical mutation-outcome class, not causal probability of
fault in an arbitrary submitted circuit.

### 8.3 Optional observable-only diagnosis LR

The five inputs are TVD, JSD, Hellinger, observed entropy, and observed maximum
probability. Missing values are median-imputed and standardized:

\[
\tilde x_j=\begin{cases}x_j,&x_j\text{ present},\\m_j,&x_j\text{ missing},\end{cases}
\qquad z_j=(\tilde x_j-\mu_j)/s_j.
\]

The three-class output is softmax:

\[
P(y=k\mid z)=\frac{e^{w_k^Tz+b_k}}{\sum_j e^{w_j^Tz+b_j}}.
\]

The probabilities are uncalibrated; the 0.60 threshold is an abstention rule,
not a 60%-correct guarantee. A fault category requires independent
localization evidence. Artifact metadata says deployment validation was not
performed. The model is not loaded implicitly by circuit diagnosis.

### 8.4 Source targets and scores

All source results below are development OOF metrics, not frozen-test results.
Tasks differ and scores must not be averaged into a project-wide accuracy.

| Source/task | Rows / groups | Target | Selected family across outer folds | Accuracy / macro-F1 / balanced accuracy |
|---|---:|---|---|---:|
| QBugs QASM v2 | 1,496 / 748 | Correct vs buggy snapshots | Linear text models (5/5) | 92.71 / 92.69 / 92.71% |
| Quantum-platform comments | 319 / 304 | Curated comment annotation | Linear text models (5/5) | 94.04 / 92.67 / 91.29% |
| Bugs4Q Core + NA + QBugs | 1,620 / 810 | Correct/fixed vs buggy | Linear text models (5/5) | 89.63 / 89.60 / 89.63% |
| QSimBench | 300 / 150 | Aer vs fake-backend descriptor identity | Linear text models (5/5) | 82.33 / 82.33 / 82.33% |
| LintQ context | 287 / 216 | Warning true-positive vs false-positive | Cosine KNN (4), linear (1) | 79.44 / 67.92 / 65.91% |
| Qolumbina ideal | 182 / 26 | Detector outcome for seeded mutants | Linear text models (5/5) | 84.07 / 54.16 / 54.69% |
| Bugs4Q robust execution | 2,220 / 37 | Test harness pass vs fail | Cosine KNN (1), linear (4) | 56.76 / 56.73 / 56.76% |
| Bugs4Q Core | 62 / 31 | Source-state task | Cosine KNN (5) | 51.61 / 51.56 / 51.61% |
| Bugs4Q NA | 62 / 31 | Source-state task | Cosine KNN (2), linear (3) | 50.00 / 48.92 / 50.00% |
| Bugs4Q Core + NA | 124 / 62 | Source-state task | Cosine KNN (5) | 49.19 / 47.97 / 49.19% |
| QMutBench full feature | 578,117 / 305 | Survivor vs WOO/OPO | Extra Trees, 23 inputs | 79.33 / 79.33 / 79.41% |

Text models used outer five-fold GroupKFold and inner three-fold grouped model
selection. Repeated identical robust-execution records were compressed for
fitting within issue group using multiplicity weights and restored for scoring.
The compatible three-source correctness merge scored 89.63%, below QBugs alone
at 92.71%; adding data is not guaranteed to help if P(X|Y,source) shifts.
Qolumbina's majority baseline is 82.42%, so 84.07% accuracy with 54.16%
macro-F1 indicates weak minority-class performance.

For a confusion matrix with true positives TP, false positives FP, true
negatives TN, and false negatives FN, the reported metrics are
\[
Accuracy=\frac{TP+TN}{TP+FP+TN+FN},\quad
Precision=\frac{TP}{TP+FP},\quad Recall=\frac{TP}{TP+FN},
\]
\[
F1=\frac{2\,Precision\,Recall}{Precision+Recall},\quad
MacroF1=\frac1K\sum_{k=1}^{K}F1_k,\quad
BalancedAccuracy=\frac1K\sum_{k=1}^{K}\frac{TP_k}{TP_k+FN_k}.
\]
Pooled OOF metrics are calculated over each record's single held-out
prediction, not by averaging fold percentages. Outer GroupKFold keeps the
same issue/origin group out of train and validation; each fold's vectorizer,
scaler, feature selection, and model selection are fitted using training
groups only. Inner GroupKFold chooses the model using accuracy. These scores
estimate performance on unseen groups from these source task distributions;
they do not estimate all circuits, future devices, or arbitrary fault types.

## 9. QML model mathematics

### 9.1 Binary VQC/QNN

The matched QMutBench Binary VQC uses four features and wires, depth two, and
angle re-upload encoding

\[
E(x)=\bigotimes_{j=0}^{3}R_Y(\pi x_j).
\]

Each layer applies encoding, trainable Rot gates, and a ring of CNOTs. In
state-update order:

\[
|\psi_l\rangle=C_{ring}R(\theta_l)E(x)|\psi_{l-1}\rangle,
\qquad z_0=\langle\psi_d|Z_0|\psi_d\rangle.
\]

Readout:

\[
\ell(x)=a z_0+b,\qquad p_1(x)=\sigma(\ell(x)).
\]

For d=2, n_q=4, there are 3*d*n_q=24 circuit rotation parameters plus readout
scale and bias, 26 total. With training-fold class counts M_0 and M_1, sample
weight r_i=M/(2M_{y_i}). The class-weighted binary cross-entropy is:

\[
L=-\frac1M\sum_{i=1}^{M}r_i
\left[y_i\ln p_i+(1-y_i)\ln(1-p_i)\right].
\]

PennyLane analytic statevector backprop computes gradients. Adam updates
parameters using

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\]

\[
\theta_{t+1}=\theta_t-\eta\widehat m_t/(\sqrt{\widehat v_t}+\epsilon).
\]

The matched QNN used eight epochs and no finite shots. It is simulated QML,
not a run on quantum hardware.

Features are ranked by training-fold
mutual information, four are selected, and each selected feature is scaled
with training-fold min/max to \([0,1]\); validation rows use those frozen
training-fold values. The implementation's predicted detected probability is
\(p_1=\sigma(a z_0+b)\). The separate plain expectation-to-probability
mapping \(p_1=(1-z_0)/2\) belongs to another circuit/readout configuration
and must not be substituted when describing this matched VQC.

For discrete feature bins, the ranking statistic is
\[
I(X_j;Y)=\sum_{x_j,y}P(x_j,y)\ln\frac{P(x_j,y)}{P(x_j)P(y)}.
\]
The fold-local min/max transform is
\[
x'_j=\frac{x_j-\min_{train}x_j}{\max_{train}x_j-\min_{train}x_j},
\]
with constant columns handled by the preprocessing implementation. These
selection and scaling operations are refit for each fold; validation data do
not choose features or set scaling bounds.

### 9.2 QSVC

For quantum feature state |phi(x)>=U_phi(x)|0>:

\[
K(x,x')=|\langle\phi(x)|\phi(x')\rangle|^2.
\]

A classical SVC optimizes the dual:

\[
\max_{\alpha}\sum_i\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_jy_iy_jK(x_i,x_j),
\quad0\leq\alpha_i\leq C,\quad\sum_i\alpha_iy_i=0.
\]

Here the kernel circuits are evaluated on a classical statevector simulator.
In the implemented QSVC feature map,
\[
U_\phi(x)=\left(\prod_{j=0}^{n_q-1}R_Z^{(j)}(\pi x_j)\right)
 C_{ring}\left(\prod_{j=0}^{n_q-1}R_Y^{(j)}(\pi x_j)\right),
\]
where \(C_{ring}\) applies nearest-neighbor CNOTs and then the last-to-first
CNOT. The overlap circuit applies \(U_\phi(x)^\dagger U_\phi(x')\);
the probability of measuring all zero is \(|\langle\phi(x)|\phi(x')\rangle|^2\).

### 9.3 Data re-uploading

Re-encoding before every variational layer gives

\[
|\psi_d(x)\rangle=
\prod_{l=1}^{d}(C_{ring}R(\theta_l)E(x))|0\rangle.
\]

It tests repeated input encoding; it increases work and does not guarantee
generalization gain.

### 9.4 Quantum reservoir

A fixed seeded quantum map draws each trainable-gate angle uniformly from
\([-\pi,\pi]\), applies RY(pi*x), two rounds of per-wire Rot and ring
CNOTs, then gives expectation features z_j(x)=<Z_j>. A balanced classical
Logistic Regression readout is trained:

\[
P(y=k\mid x)=softmax_k(Wz(x)+b).
\]

Circuit parameters stay fixed; only W and b are trained.

### 9.5 Three-class QNN

For angle encoding, the three expectation features are
z=(<Z0>,<Z1>,<Z2>); for amplitude encoding they are
z=(<Z0>,<Z1>,<Z0 Z1>). A trainable readout uses

\[
\ell=Wz+b,\qquad P(y=k\mid x)=\frac{e^{\ell_k}}{\sum_{j=0}^{2}e^{\ell_j}},
\quad L=-\frac1M\sum_i r_i\ln P(y_i\mid x_i).
\]

The angle variant repeats RY(pi*x); the amplitude variant uses normalized
amplitudes x/||x||_2, thereby discarding the original input norm.

## 10. Matched comparisons and their scope

### 10.1 Binary QML versus classical, same sampled data

Two seeds used the same origin-grouped five-fold split, same sample per fold,
same four features, and fold-local preprocessing. Per fold, there were 1,024
training rows per class and 1,536 validation rows per class; each model had
15,360 pooled OOF validation predictions per seed.

| Model | Mean accuracy | Mean macro-F1 | Mean balanced accuracy |
|---|---:|---:|---:|
| Binary VQC | 58.37% | 56.04% | 58.37% |
| Extra Trees, same four features | 58.64% | 56.45% | 58.64% |
| Logistic Regression, same four features | 58.41% | 55.03% | 58.41% |

The sample is balanced, so accuracy and balanced accuracy coincide. These
means are descriptive across two sampled development OOF runs, not independent
test estimates. Data/split/features are matched; compute is not: VQC took about
47 minutes per five-fold run, versus under a second for the classical controls.
No QNN advantage is demonstrated.

The later Kaggle GPU-backed matched summary reported VQC 58.49% accuracy /
58.25% macro-F1, Extra Trees 59.85% / 58.11%, and Logistic Regression
58.88% / 55.80% on a different 2,048-training-examples-per-class sampled
protocol. The report flags this output for artifact audit: fold 2 has identical
metrics/confusion matrices across all models, and summed VQC fit time exceeds
the reported whole-cell time. The row-level OOF prediction artifact is not in
this repository, so these GPU-run values are documented as provisional and
must not be treated as a validated comparison or pooled with the two earlier
seeds.

The 79.33% full-data classical score uses 578,117 rows and 23 features; it is
not a fair direct comparator to this sampled four-feature QNN experiment.

### 10.2 Nine-origin pilot and three-class experiment

The 72-row nine-origin pilot reported VQC 50.0% accuracy / 48.6% macro-F1,
QSVC 43.1% / 42.2%, data re-uploading 48.6% / 46.9%, quantum reservoir
37.5% / 34.7%, logistic control 50.0% / 49.4%. Its small origin count makes
all estimates exploratory.

The 205-row, 12-OPO-origin three-class study reported:

| Model | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Angle VQC | 32.20% | 29.98% | 31.80% |
| Amplitude VQC | 33.66% | 30.86% | 34.45% |
| Angle VQC + trainable readout | 35.61% | 32.18% | 36.49% |
| Extra Trees, matched 4 features | 49.27% | 46.69% | 48.46% |
| Logistic, matched 4 features | 46.34% | 44.07% | 45.72% |
| RBF SVC, matched 4 features | 43.90% | 43.06% | 43.48% |

The origin sample was deliberately enriched for rare OPO-bearing origins and
is not representative of the 305-origin population. QNN variants were below
the matched classical controls in this experiment.

## 11. Why the equations limit conclusions

1. **Intent is underdetermined from circuit text alone.** The submitted U
   generates its own ideal distribution. Without U_spec, a test invariant, or
   trusted reference, the simulator cannot know whether a gate was omitted
   accidentally. This is a missing-specification problem, not a classifier
   weakness.
2. **Finite-shot noise:** for one outcome with probability p, frequency
   standard error is approximately sqrt(p(1-p)/N_s). Rare outputs need more
   shots; the simple 1/sqrt(N_s) guard is not a formal confidence bound.
3. **Independent sample size is group-limited.** Repeated runs within one issue
   or circuit family are correlated. Row count can overstate effective sample
   size; several tasks have only 26-37 independent groups.
4. **Accuracy hides imbalance:** if prevalence is pi and class recalls are r1,
   r0, then Accuracy=pi*r1+(1-pi)*r0, while BalancedAccuracy=(r1+r0)/2.
   A majority classifier may have high accuracy and zero minority recall.
5. **Four-feature reduction loses possible information.** For deterministic
   Z=T(X), the data-processing inequality gives I(Y;Z)<=I(Y;X). Four
   dimensions make the QNN feasible but do not guarantee preservation of label
   signal.
6. **Source shift:** pooled performance may degrade when P(X|Y,source) or
   P(Y|X,source) differs by source. The three-source correctness result below
   QBugs alone demonstrates that pooling is not automatically beneficial.
7. **Classical model assumptions:** TF-IDF treats tokens/n-grams as features;
   linear classifiers use linear boundaries in that space; ComplementNB uses
   conditional-independence approximations; KNN needs a meaningful local
   cosine geometry; trees depend on the available tabular feature signal.
   None implies causal fault detection.
8. **QNN capacity/training:** the benchmark circuit has four wires, depth two,
   26 trainable parameters and eight epochs. This is a bounded model search;
   it does not prove all QML designs fail, but measured results do not beat the
   matched controls.
9. **Analytic QML omits shot noise:** for a Pauli-Z estimate,
   SE(z-hat)=sqrt((1-z^2)/N_s). Headline VQC runs used analytic expectation
   values rather than finite-shot samples; no hardware or finite-shot accuracy
   claim follows.
10. **Classical statevector simulation is exponential:** exact amplitudes scale
    as 2^n. Four-qubit experiments establish neither scalability nor quantum
    speedup.
11. **Noise-model mismatch:** the HardwareModel's gate-skip/T1/T2/readout
    channels are narrower than full Aer or real devices. If modeled channel
    E-hat differs from physical E, then modeled output p-hat(x) can differ
    from real p(x) even for a correct program.
12. **Hardware ranks are not calibrated probabilities:** without a fitted
    P(fault|features), score 2 is not twice as likely as score 1. No universal
    accuracy follows from the weights.
13. **Compute was not equalized:** the QML comparison matches data and folds,
    not wall time, energy, FLOPs, or tuning budget.
14. **Incomplete/restricted artifacts:** full QML development did not produce
    full-data OOF metrics; a later GPU summary is flagged for fold/runtime
    audit. Raw external datasets and the full QMutBench fitted bundle are not
    present in this checkout, limiting fresh-clone reproduction.
15. **No common end-to-end diagnosis benchmark:** the source tasks have
    incompatible labels. There is no valid equation that combines these
    accuracies into one “simulator accuracy.” A new independently labeled,
    group-separated diagnosis evaluation is required.

## 12. Project decisions still needed for real ML/QML integration

No extra explanation is needed to read these equations. Operational model
integration still requires:

1. A precise target definition: detection, cause type, affected gate/qubit,
   severity, and policy for mixed or ambiguous causes.
2. An intended-program contract: trusted reference, mathematical invariant,
   expected observable, or specification for arbitrary input circuits.
3. A hardware truth contract: synthetic scenario testing, current-calibration
   ranking, or independently labeled real-device diagnosis.
4. A deployment role: whether ML/QML may recommend, filter, or override
   evidence. Current code does not silently load either research family.
5. Redistributable raw data, pinned split/source hashes, and fitted model
   artifacts for experiments whose inputs/bundles are not included. Never
   include user credentials.

The accurate current description is: a mathematical statevector simulator,
statistical/rule-based diagnosis, heuristic hardware localization,
source-specific classical development benchmarks, and separate QML research.
No ML/QML model is yet validated as a production diagnosis model for arbitrary
program faults or real hardware anomalies.

## 13. Source map

- Gate matrices/aliases: src/quantum_simulator/gates.py
- State execution, measurement, reset, noise: src/quantum_simulator/quantum_simulator.py
- Circuit builder: src/quantum_simulator/quantum_circuit.py
- Distribution distances: src/quantum_simulator/statistics.py
- Count permutation/bootstrap tests: src/quantum_simulator/count_statistics.py
- Program edit alignment: src/quantum_simulator/fault_localizer.py
- Hardware candidate scores: src/quantum_simulator/hardware_fault_localizer.py
- Diagnosis policy: src/quantum_simulator/diagnosis_engine.py
- Optional observable classifier: src/quantum_simulator/ml_diagnosis.py and src/quantum_simulator/observable_model_adapter.py
- Source ML: ml_training.py and experiments/run_classical_source_nested_search.py
- QMutBench QASM features: qmutbench_ml/qasm_mutation_diff.py
- Binary QNN: qmutbench_qml/models/binary_vqc.py
- Quantum kernel: qmutbench_qml/models/qsvc.py
- Quantum reservoir: qmutbench_qml/models/quantum_reservoir.py
- Three-class QNN: qmutbench_qml/models/multiclass_vqc.py
- Results: reports/experiments/classical_ml_final_results_20261030.md,
  data/curated_qml_training_v1/matched_qml_classical_comparison.md, and
  reports/experiments/qmutbench_three_class_qnn_campaign_20261027/research_report.md
