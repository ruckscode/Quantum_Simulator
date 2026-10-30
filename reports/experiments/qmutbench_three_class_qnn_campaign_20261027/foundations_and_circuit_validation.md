# Quantum foundations and circuit validation

## State and measurement

A normalized qubit is |psi> = alpha|0> + beta|1>, with |alpha|^2 + |beta|^2 = 1. Computational-basis measurement gives probabilities |alpha|^2 and |beta|^2. The RY(theta) gate is unitary and produces RY(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>. Thus P(0)=cos^2(theta/2), P(1)=sin^2(theta/2), <Z>=cos(theta), and d<Z>/dtheta=-sin(theta). At theta=pi/2 it creates equal superposition. Hadamard followed by CNOT creates the Bell state (|00>+|11>)/sqrt(2).

The PennyLane `default.qubit` one- and two-qubit statevectors, norms, basis probabilities, Pauli-Z expectations, and autodiff gradients were compared against those closed-form values at theta=0, pi/3, and pi/2. Bell-state amplitudes and probabilities were also checked. Maximum state-amplitude absolute error was 0; gradient and expectation errors were below numerical tolerance in the tests.

## Data encoding and supervised readout

The angle QNN uses fold-local four-feature selection/scaling, then RY(pi*x_i) on four wires. Re-uploading repeats this encoding before each of two trainable Rot layers, with nearest-neighbor and wraparound CNOTs. The measured Z expectations on wires 0, 1, and 2 become three logits; softmax makes valid probabilities and class-weighted cross-entropy trains the circuit parameters.

The amplitude alternative normalizes the four selected features into a two-qubit state, then applies a depth-two trainable circuit. It discards the feature-vector norm and requires explicit state preparation; PennyLane decomposition counted its preparation gates in the compiled depth. Its readout uses Z0, Z1, and Z0Z1 expectations.

The trainable-readout hybrid jointly learns a 3x3 affine matrix and bias on the three quantum expectations. This tests whether a restricted fixed expectation-to-logit mapping caused weak class separation. The 24 circuit parameters plus 12 head parameters are optimized end-to-end by Adam and autodiff.

## Optimization and validation evidence

The controlled 12-row toy task reduced weighted cross-entropy from 1.3040 to 0.7156 and predicted 12/12 correctly. Circuit parameters changed: True; probabilities finite and normalized: True; training used 288 analytic circuit executions. Real-data fold training curves and final validation losses are separate CSV artifacts.

Circuit resource summary is in `encoding_and_circuit_config.json`: angle circuits are 4-qubit, depth 12 after PennyLane resource expansion (two ansatz layers), 24 or 36 trainable parameters; amplitude is 2-qubit, decomposed depth 8, 12 trainable parameters. This is ideal analytic simulation; shots, noise, and hardware were not used. For N independent shots, a Pauli-Z sample mean has standard error sqrt((1-<Z>^2)/N); this campaign did not run finite-shot estimates.

## Scope limits

Basis encoding was excluded because it would discretize fold-selected continuous/dummy values and discard magnitude information. No claim is made that these small designs exhaust all QNN architectures. The actual development evaluation sample is class-balanced and conditional on OPO-bearing origins; results do not estimate natural-prevalence performance. The held-out development confirmation run stopped at the 500 MiB memory safety floor before producing a fitted checkpoint or predictions.


## Memory-efficient exact-gradient update

To reduce peak memory without switching to stochastic mini-batch optimization, the confirmation runner loads training and reserved-origin records in one pass and accumulates per-chunk gradients over the entire training set before a single Adam update. A six-record toy check with chunk size two matched the full-batch update to 2.4e-12 maximum circuit-parameter error, 7e-18 readout error, and 2.2e-16 loss difference, using six circuit evaluations in either mode. This optimization was not run on QMutBench because safe post-import RAM headroom could not be established on resume.