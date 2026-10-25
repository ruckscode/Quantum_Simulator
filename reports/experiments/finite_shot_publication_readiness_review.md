# Finite-shot publication readiness review

**Review type:** read-only analytical assessment  
**Verdict: CONDITIONAL GO**  
**Execution status:** no new experiment is authorized by this review. The proposed 252-family protocol is not sufficiently specified to execute as registered.

## Executive assessment

The project has a sound, limited identifiability demonstration: output data alone cannot distinguish two causes when those causes induce the same effective channel. This is a useful constraint on diagnostic claims, but it is not by itself a new method or a publishable empirical contribution. Existing work already covers quantum software fault diagnosis, hardware/software attribution, selective prediction, and measurement-dependent fault equivalence. A defensible paper would need a precise, experimentally validated contribution beyond combining these ideas.

The saved finite-shot calibration demonstrates that the current fixed-guard TVD rule has severe low-shot false positives for its configured controls, and that the permutation detector trades sensitivity for fewer false positives. It does not contain hardware-fault cases, independently calibrated device evidence, or a family-diverse evaluation. It therefore cannot validate hardware attribution, causal localization, or generalization to unseen circuit families.

The proposed 252-family, four-condition study has arithmetically plausible sample-size gates under optimistic assumed rates, but its protocol has unresolved ground-truth, leakage, estimand, and novelty issues. Those are blockers, not reasons to merely scale the sample size. Resolve them and establish a distinctive method before committing the projected execution cost.

## 1. Identifiability audit

Let the input state be \(\rho\), intended gate be \(G\), and a program mutant replace it with \(U\). The mutant yields

\[
  \rho_{\rm mutant}=U\rho U^\dagger.
\]

Now let the correct program apply \(G\), followed immediately by a hardware unitary channel \(V=UG^\dagger\). Then

\[
 V(G\rho G^\dagger)V^\dagger
 = (UG^\dagger)G\rho G^\dagger(GU^\dagger)
 = U\rho U^\dagger.
\]

Thus the two constructions produce the same state at that point. If the rest of the circuit and measurement process are the same, they produce the same distribution for every subsequent measurement. With output-only evidence and equal prior probabilities, no classifier can exceed 50% cause accuracy on this exactly paired case; more shots do not resolve equal likelihoods.

This is a constructed equivalence, not evidence that a naturally occurring hardware channel commonly equals \(UG^\dagger\). It assumes that \(G\) and \(U\) are unitaries on the same subsystem, the hardware channel is the specified unitary immediately after \(G\), and subsequent operations, noise, and measurements match. Timing, leakage, non-Markovian effects, cross-talk, a channel acting on a different subsystem, or other context-dependent noise may invalidate this particular equality. More generally, a nonunitary channel can sometimes match a program fault on a restricted input/state/measurement set without being operation-equivalent.

Three notions must remain distinct:

1. **Effective-operation equivalence:** the same channel on all input states. The construction above establishes this for the local composed operation \(VG=U\).
2. **Measured-distribution equivalence:** equal outcome probabilities for a specified circuit, input, and measurement. This is weaker and may hold even when operations differ. For example, a terminal Z phase before computational-basis measurement is invisible in ideal probabilities.
3. **Cause distinguishability with auxiliary evidence:** a source diff can establish that code differs, while independent device calibration can support a hardware-error hypothesis. Those observations add evidence beyond output counts. A source diff alone does not establish that the changed operation caused the output deviation. Calibration is useful only if collected independently, matched to relevant device/time/component conditions, and not exposed as an oracle label or duplicated model input.

The available calibration is a software simulator experiment using known circuit variants; it contains no real hardware fault samples or independent calibration evidence. Hence the equivalence example supports a limit on output-only attribution, not a measured estimate of how often attribution is impossible in practice.

## 2. Statistical and artifact audit

For two independently sampled multinomial histograms with common probabilities \(p_i\) and \(n\) shots each, the difference in category proportions has approximate variance \(2p_i(1-p_i)/n\). For roughly uniform \(K=16\) outcomes, summing expected absolute normal deviations gives the approximate null scale

\[
  E[\operatorname{TVD}] \approx \tfrac12\sum_i\sqrt{\tfrac{2}{\pi}\tfrac{2p_i(1-p_i)}{n}}
  \approx \sqrt{\tfrac{K-1}{\pi n}}.
\]

The latter evaluates to about **0.1366 at 256 shots** and **0.0683 at 1,024 shots**. These are approximate null means under an interior, roughly uniform multinomial model. They are not quantiles, p-values, false-positive guarantees, or calibrated cutoffs. Sparsity, nonuniform probabilities, estimated references, and finite-sample discreteness change the distribution.

The inspected diagnosis implementation samples the reference circuit and submitted circuit separately, each with the requested shot count and separate RNG streams, then compares the empirical distributions. The detector is therefore a two-sample comparison. The fixed TVD decision uses `TVD > max(0.05, 1/sqrt(shots))`; it is not adjusted to the number of outcomes or the reference probability vector and is not a formal error-controlled test.

Saved calibration evidence: 3,600 valid configurations in one fixed four-qubit topology/phase-tag recipe, split evenly between calibration and validation. At 256 shots, the fixed-guard TVD flagged 120/120 in each registered control scenario. At 1,024 it flagged 104/120 and 107/120. The permutation baseline reduced false positives but detected only 14/120 subtle positives at 256 and 36/120 at 1,024. At 4,096 shots, permutation plus its effect gate detected 113/120 subtle positives, with the saved one-sided lower bound below the registered .90 floor. These outcomes describe the saved design and detector, not general quantum programs or hardware.

The calibration runner supplied no hardware model to its run calls. The saved results therefore do not establish hardware attribution. Source localization includes operation-syntax matching and a largest-deviation heuristic; the latter is not causal evidence. Scored localization ground truth/results are absent from the inspected calibration summary. Output-only results also cannot adjudicate exact observationally equivalent causes.

## 3. Novelty assessment

### Verified related work

| Work | Established formulation/method | Consequence for this project |
|---|---|---|
| Huang & Martonosi, ISCA 2019, [DOI](https://doi.org/10.1145/3307650.3322213) | Statistical tests on classical observations to detect quantum program bugs under finite measurement outcomes. | Finite-shot statistical bug detection is established; the paper cannot claim that framing alone as novel. |
| Li et al., IEEE VTS 2023, [DOI](https://doi.org/10.1109/VTS56346.2023.10140030) | Quantum circuit fault diagnosis using a fault dictionary of output probability distributions and dynamic binary search; benchmark and IBM-device demonstrations. | Probability-distribution diagnosis and circuit fault localization have direct precedents. |
| Geifman & El-Yaniv, SelectiveNet, ICML 2019, [PMLR](https://proceedings.mlr.press/v97/geifman19a.html) | Prediction with a reject option optimized/evaluated through risk and coverage. | Abstention and risk/coverage are established general methods, not novel simply because applied here. |
| Virani et al., IEEE QCE 2025, [DOI](https://doi.org/10.1109/QCE65121.2025.00096), [paper](https://arxiv.org/html/2507.20475v1) | Statistical distinction between quantum software bugs and hardware noise, with four bug/noise conditions and output-statistic classification. | This is a direct competitor to a generic software-versus-hardware attribution claim. Its assumptions and scope should be compared in full before making a narrower claim. |
| Hassan & Kaabouch, arXiv:2602.21253v4 (2026), [preprint](https://arxiv.org/html/2602.21253v4) | Hardware error attribution using device-calibrated Bhattacharyya evidence and an ANFIS model; reports cross-device evaluation and discusses indistinguishability. | A current, close preprint further narrows hardware attribution and calibration-aware novelty. It is not treated as peer-reviewed evidence. |
| Maidl et al., DDECS 2025, [publication record](https://www.dfki.de/web/forschung/projekte-publikationen/publikation/15756) | Automated circuit debugging with slicing, repeated measurements, and equivalence classes of faults indistinguishable under measurements. | Measurement-dependent equivalence and localization are already addressed. |
| Calderon-Vargas et al., *Quantum* 7, 921 (2023), [DOI](https://doi.org/10.22331/q-2023-02-09-921) | Layer sensitivity/localization using local inversions, including IBM hardware evaluation. | Hardware-tested layer localization has precedent. |

Bibliographic details above were checked against the linked publication or primary paper pages where available. The Hassan & Kaabouch item is explicitly a preprint. The prior assessment's claim that novelty might lie in combining calibrated uncertainty, selective diagnosis, and source-localization limits is not sufficient: Virani et al. and the 2026 preprint overlap the attribution framing, while Maidl et al. overlaps measurement-dependent equivalence. A narrower contribution could still survive if it defines and validates a new *operational protocol* for calibrated evidence provenance and selective diagnosis under controlled identifiability limits, but that distinction remains a hypothesis requiring full methods-level comparison and experimental evidence.

## 4. Proposed experiment audit

### Four conditions and projected work

The existing proposal defines: (1) identical negative controls; (2) measurement-equivalent negative controls, including terminal Z before computational measurement; (3) known program faults with source/operation truth; and (4) hardware-channel faults with channel/component truth and unmodified source. It also discusses realistic stochastic hardware noise as another stratum; this is not reconciled with the four-condition count and should not silently be treated as included.

At 252 family units per condition and one primary shot setting of 4,096, the plan implies 1,008 configurations. Submitted and reference circuits are both sampled at 4,096 shots, yielding 8,257,536 shot-circuit executions. The arithmetic is correct. A three-setting grid at 256, 1,024, and 4,096 would imply 3,024 configurations and 10,838,016 shot-circuit executions. Counts of family units must refer to independent, held-out circuit families; multiple templates, fault variants, shots, or repeated configurations from one family are clustered observations, not new independent units.

The calibration evidence comes from one topology/recipe, not 252 demonstrated independent families. The proposed number therefore does not itself establish the independence or representativeness of its sampling frame. The 252-family calculation concerns binomial endpoint gates, not paired superiority against baselines and not a generalization guarantee across a family population.

### Reconstructed power calculation

For a one-sided Clopper–Pearson lower sensitivity bound, with \(x\) successes among \(n\),

\[
 L=\operatorname{Beta}^{-1}(\alpha;x,n-x+1),
\]

and for an upper FPR bound with \(y\) false positives,

\[
 U=\operatorname{Beta}^{-1}(1-\alpha;y+1,n-y).
\]

With \(n=252\), per-bound \(\alpha=.0125\) (Bonferroni for four one-sided gates), minimum sensitivity .80 and maximum FPR .05, the gates are at least 216 detections and at most 5 false positives. The calculated bounds are approximately 0.80046 and 0.04988. If true sensitivity is .90, the probability of passing that sensitivity gate is 0.9883; if true FPR is .01, probability of passing the FPR gate is 0.9575. With two positive and two control gates, a conservative union-bound lower bound on joint pass probability is 0.8915 under those assumed true rates and independent family sampling per endpoint. At true sensitivity .85 and FPR .02, respective gate powers fall to about 0.417 and 0.609, so the optimistic assumptions matter substantially.

These are marginal-binomial gate powers, not a powered paired detector comparison. The values .90 and .01 are assumptions, not established rates from the existing calibration, and the calculation does not account for family heterogeneity, family-level correlation, or multiplicity across additional metrics/subgroups. Dependence between criteria is unknown; a joint probability cannot be inferred by multiplying marginal powers. A union bound is defensible but conservative under specified marginal powers; realistic joint power needs pilot-informed dependence or a prespecified joint simulation/analytical model using independent family units.

### Ground truth, leakage, and outcomes still needed

The source-level truth must be encoded from the controlled source mutation and reviewed independently. Hardware truth must be acquired from a separately logged, blinded or held-out device calibration/telemetry procedure matched to device, time window, and component. The diagnosis model must not receive simulator-injected channel identity or any calibration feature that deterministically encodes the assigned condition. Keep source diffs, calibration labels, and adjudication records in separate provenance channels; predefine what the method may observe at inference. Current plan does not establish acquisition, blinding, time matching, or leakage controls.

Before execution the protocol needs one primary estimand, e.g. family-level selective error at a prespecified minimum coverage, with separate sensitivity/FPR gates. Define whether one prediction is made per family or how repeated variants aggregate. Include unchanged fixed-guard TVD and permutation baselines; any candidate method must be frozen using calibration families disjoint by family identity and seed from held-out evaluation. Use paired family-level differences with a prespecified interval/test, one-sided exact binomial intervals for sensitivity/FPR, and intervals for coverage/selective risk clustered at family. Freeze multiplicity handling for the primary gates and secondary metrics. Define explicit failure as any primary bound/coverage gate failing, plus what happens if ground truth is indeterminate or a condition cannot be constructed.

### Resource interpretation

The manifest reports 3,600 configurations, 12,902,400 shot-circuit executions, and 3,192.3 seconds recorded runtime, while the timestamps span roughly 8.16 hours. Straight-line scaling to 8.26 million executions varies from about 34 minutes (runtime field) to roughly 5.2 hours (timestamp span), so neither is a reliable forecast. Windows lock-allocation failure and OneDrive manifest contention are known operational risks; the run design needs safe checkpointing, bounded worker/lock allocation, and local or atomic manifest writes. The count-vector storage for 1,008 expected/observed 16-category histograms is small (about 252 KiB at 64-bit counts), but this excludes simulator state, per-worker objects, serialization, and process overhead. Resource feasibility does not resolve the protocol's scientific blockers.

## 5. Blocking issues and decision

**CONDITIONAL GO** for refining the research question; **NO execution** of the present 252-family plan until the following are resolved:

1. **Novelty not established.** Direct 2025 peer-reviewed attribution work, a close 2026 preprint, and 2025 debugging work overlap the proposed claims. Required evidence: a methods-level comparison table against full texts, then an explicit operational distinction and a measurable hypothesis not already tested by these methods.
2. **Four-condition design is internally ambiguous.** Exact unitary channel equivalence and realistic stochastic hardware noise are both mentioned without a settled condition/stratum map. Required evidence: a frozen condition table with one ground-truth mechanism and admissible evidence source per unit, reconciled with the stated count and primary estimand.
3. **Ground truth and leakage controls are unspecified.** No independent hardware fault evidence is present in calibration. Required evidence: executable provenance/blinding protocol for source mutations and independent device calibration, with a data-flow audit proving injected labels do not enter model inputs.
4. **Independent family sampling and paired estimand are not demonstrated.** The prior calibration uses repeated templates/variants within a narrow recipe; configuration count is not independent family count. Required evidence: a deduplicated family sampling frame, disjoint calibration/evaluation identities and seeds, clustering unit, paired baseline comparison, and power analysis for that paired estimand under defensible rate assumptions.
5. **Primary outcomes and decision rules are incomplete.** Marginal sensitivity/FPR gates do not specify coverage, selective risk, localization scoring, multiplicity, indeterminate labels, or failure handling. Required evidence: frozen endpoint definitions, intervals, multiplicity policy, acceptance/failure rules, and an analysis script specification, all finalized before held-out outcomes are observed.

The next step is a protocol-and-novelty resolution review using the closest papers and a concrete family/ground-truth data-flow specification. If no distinct methodological hypothesis remains after that comparison, narrow or abandon the attribution framing rather than spending the projected compute budget. No threshold retuning on the inspected validation outcomes is justified.

## Evidence provenance and limitations

This assessment used the existing reports, diagnosis/statistics implementation, calibration plan, manifest, and saved calibration records. Literature links were accessed during this review; the two 2025 publications and the linked primary pages were verified, and the Hassan & Kaabouch source is a preprint. Calculations are analytical reconstructions of the stated model and binomial assumptions. No tests, simulations, training, benchmarks, or hardware jobs were run. Source artifacts and existing results were not changed.
