# Finite-shot protocol blocker resolution

**Assessment date:** 2026-10-14  
**Scope:** Read-only resolution review of the five blockers in `finite_shot_experiment_protocol_gaps.json` for the proposed 252-independent-family, four-condition experiment.  
**Verdict: NO-GO** for the current experiment to answer the intended full research question. The project may continue protocol and novelty work; this is not authorization to execute.

## Decision summary

| Original blocker ID | Severity | Status | Blocks execution? |
|---|---|---|---|
| `novelty_not_established` | critical | PARTIALLY RESOLVED | Yes |
| `condition_definition_ambiguous` | high | PARTIALLY RESOLVED | Yes |
| `ground_truth_and_leakage_protocol_missing` | critical | UNRESOLVED | Yes |
| `independent_family_unit_and_paired_estimand_unverified` | high | PARTIALLY RESOLVED | Yes |
| `primary_endpoints_and_decision_rules_incomplete` | high | PARTIALLY RESOLVED | Yes |

None is RESOLVED for the specific 252-family proposal. Other project documents contain useful protocol work, but they are proposals, specifications for other studies, or analyses that do not instantiate and freeze the missing evidence for this design. The JSON companion preserves the exact original IDs, severity, evidence, and resolution requirements.

## Blocker-by-blocker findings

### 1. `novelty_not_established` — critical — PARTIALLY RESOLVED

**Original evidence:** “Virani et al. (IEEE QCE 2025) directly study statistical distinction between quantum software bugs and hardware noise; Hassan and Kaabouch (2026 arXiv preprint) study hardware error attribution with device calibration and cross-device evaluation; Maidl et al. (DDECS 2025) address measurement-dependent fault equivalence and localization. These overlap central proposed claims.”

**Original requirement:** “Complete a methods-level comparison against the full papers and specify one operational distinction and measurable hypothesis that those methods do not already establish.”

**Existing evidence and verification:** The readiness review and `finite_shot_candidate_selection_protocol.md` identify close prior work. Web verification of primary/publication sources confirms that Virani et al. present a statistical approach to distinguishing quantum software bugs from hardware noise ([arXiv paper](https://arxiv.org/abs/2507.20475); IEEE DOI `10.1109/QCE65121.2025.00096`), while Maidl et al. use circuit slicing, repeated measurement, and measurement-relative gate equivalence classes for fault localization ([DFKI publication record](https://www.dfki.de/web/forschung/projekte-publikationen/publikation/15756); [full paper PDF](https://hets.dfki.de/doc/konf/DDECS2025_AM.pdf)). The Hassan–Kaabouch work is an arXiv preprint, not peer-reviewed evidence; its abstract describes calibrated, physics-informed software/noise attribution and explicitly notes a single-basis blind spot ([arXiv preprint](https://arxiv.org/abs/2602.21253)). These sources substantiate overlap and make generic attribution, finite-shot diagnosis, localization, or observational-equivalence novelty claims untenable without a narrower comparison.

**Why only partial:** The repository does not contain a completed methods-by-method comparison against all relevant full texts, nor a precise operational distinction with a measurable hypothesis frozen for the 252-family study. Existing candidate/research-direction reports label a possible contribution as a hypothesis; they do not establish novelty. The source review does not establish novelty, and this review makes no novelty claim.

**Smallest next artifact:** A cited comparison table with method, input evidence, fault/noise assumptions, experimental unit, output/abstention rule, ground truth, and evaluation scope for the full closest papers; followed by one explicit, falsifiable operational difference and a claim boundary. Record the conclusion even if no distinct claim survives. No sample-size or threshold change follows from this blocker.

### 2. `condition_definition_ambiguous` — high — PARTIALLY RESOLVED

**Original evidence:** “The four stated conditions are identical controls, measurement-equivalent controls, known program faults, and hardware-channel faults. The proposal also mentions realistic stochastic hardware noise as a separate stratum without reconciling that with the four-condition design or its resource count.”

**Original requirement:** “Freeze a condition table specifying mechanism, ground-truth label, construction, evidence source, and count for each unit; explicitly decide whether realistic stochastic noise is in scope.”

**Existing evidence:** The readiness review names the four intended conditions and gives a constructed local unitary-equivalence example. `finite_shot_identifiability_analysis.md` distinguishes effective-operation equivalence from measurement-distribution equivalence and explains that a terminal Z phase may be invisible to computational-basis measurement. Existing controlled-fault artifacts include simulated software and hardware-noise scenarios. These clarify concepts and possible constructions.

**Why only partial:** No frozen condition table for this 252-family design reconciles a hardware-channel fault with the separate realistic stochastic-noise stratum, assigns each family exactly once, and specifies its ground-truth evidence source and count. The constructed unitary (V=UG^\dagger) is an exact illustrative equivalence under its stated assumptions; it is not evidence of a naturally occurring device fault. No selection between “stochastic noise excluded,” “included as a fifth condition,” or “included as a prespecified hardware stratum” is recorded for this proposal. The count/resource arithmetic depends on that choice.

**Smallest next decision/artifact:** One dated, versioned condition table for the existing four cells, with one row per cell and columns for mechanism, construction, label source, admissible model input, unit/count, and equivalence relation; explicitly mark realistic stochastic noise in or out of scope. Do not add a fifth cell or change counts until that decision is made.

### 3. `ground_truth_and_leakage_protocol_missing` — critical — UNRESOLVED

**Original evidence:** “Saved calibration has no hardware-fault samples or independent calibration evidence. The diagnosis interface can accept a HardwareModel; using simulator-injected condition identity or correlated calibration data as inference evidence risks label leakage.”

**Original requirement:** “Specify independent source-mutation adjudication and device calibration/telemetry acquisition, blinding and time/component matching, admissible inference inputs, and a data-flow audit demonstrating labels cannot leak into model inputs.”

**Existing evidence:** The saved finite-shot distribution calibration is a simulator calibration with known software variants; its runner did not supply a hardware model. `diagnosis_engine.py` accepts `hardware_model` and uses it to generate/interpret hardware evidence, and `fault_localizer.py` performs source-operation comparison/heuristics. Readiness and revised-proposal reports correctly distinguish source-diff evidence from evidence that a source change caused the output shift, and distinguish independent device telemetry from injected simulator identity.

**Status rationale:** No artifact provides device-acquired, independently logged hardware fault evidence matched to device, time window, and component; there is no blinding/adjudication procedure or completed input data-flow audit for the proposed experiment. Existing simulation labels cannot substitute for independent device evidence. Since the primary hardware-attribution claim requires such evidence, this is unresolved, not a documentation-only omission.

**Smallest next artifact:** A protocol and provenance schema naming the independently acquired telemetry/calibration source, timestamp and component matching rule, label adjudicator/blinding, held-out assignment, and exact allowed inference fields; plus a data-flow table (or audit) showing injected condition and adjudication labels are excluded from model inputs. If external device evidence is unavailable, explicitly remove hardware attribution as a primary claim before any execution decision.

### 4. `independent_family_unit_and_paired_estimand_unverified` — high — PARTIALLY RESOLVED

**Original evidence:** “The saved calibration uses repeated templates and variants in a single narrow topology/recipe. Its configuration count does not establish independent family count. The n=252 calculation powers marginal binomial gates, not paired superiority between detectors.”

**Original requirement:** “Provide a deduplicated family sampling frame, disjoint calibration/evaluation family identities and seeds, family-level clustering/aggregation rules, a paired primary estimand, and power analysis for that estimand using defensible rate and dependence assumptions.”

**Existing evidence:** `finite_shot_statistical_power_analysis.md`, `finite_shot_candidate_selection_protocol.md`, and `finite_shot_shot_aware_heldout_plan.md` discuss family identity, structural deduplication, disjoint splits, clustering, paired outcomes, and the need for cluster-aware inference. The analysis explicitly says the current 3,600 calibration configurations arise from one narrow recipe and that the n=252 gates are marginal binomial gates, not paired-comparison power. The revised proposal presents a different 20-family × 12-template design and calls its ICC/power assumptions planning assumptions.

**Why only partial:** These reports do not provide the deduplicated sampling frame or evidence that the proposed 252 units are 252 independent, representative held-out circuit families. The later 20-family proposal is a different design, and clustering twelve templates under twenty families does not establish 252 independent family units. The paired estimand and defensible discordance/dependence assumptions are not frozen for this 252-family proposal. Historical calibration and validation outcomes cannot be relabeled as new held-out families.

**Smallest next artifact:** A machine-readable family frame with canonical structure hashes, family IDs, provenance, and explicit disjointness against all calibration/development artifacts; state the independent sampling unit and one aggregation rule. Then specify one paired estimand and calculate its operating power under declared family-level discordance/dependence assumptions. If 252 independent families cannot be sourced, label 252 as configuration/template count and recalculate inference; do not inflate counts by adding repeats.

### 5. `primary_endpoints_and_decision_rules_incomplete` — high — PARTIALLY RESOLVED

**Original evidence:** “The reconstructed sensitivity/FPR gates do not fully define coverage, selective risk, localization scoring, multiplicity, indeterminate labels, paired comparisons, or failure handling. Joint power relies on assumed rates and only a conservative union bound is available without criterion dependence data.”

**Original requirement:** “Freeze endpoint definitions, confidence intervals, multiplicity policy, coverage/selective-risk and localization rules, indeterminate-case handling, paired analysis, and explicit acceptance/failure criteria before held-out outcomes are observed.”

**Existing evidence:** `finite_shot_selective_diagnosis_revised_proposal.md`, `finite_shot_candidate_selection_protocol.md`, and the held-out plan specify proposed selective-risk/coverage, sensitivity/FPR, paired family resampling, localization only when valid truth exists, multiplicity, and failure handling. The power analysis distinguishes marginal gate calculations from paired power and records the independence assumptions. This is material progress in documenting candidate endpoint choices.

**Why only partial:** These are proposed successor-study specifications and contain unresolved choices, including primary estimand/coverage matching, whether classification is per family or repeated variant, and what claims are valid when independently adjudicated target truth is absent. They do not freeze a single analysis contract for the four-condition 252-family experiment. The original 252-family sensitivity/FPR gates do not define a complete selective classifier endpoint or an outcome for unidentifiable/indeterminate cases. No analysis-script specification tied to frozen endpoints is present for this design.

**Smallest next artifact:** A signed/dated analysis contract for this design that fixes one primary estimand, unit and aggregation, positive/negative denominators, abstention and indeterminate handling, all interval and multiplicity rules, paired comparison, localization scoring eligibility, and explicit pass/fail outcomes. Include the handling of an impossible-to-identify cause as an expected ambiguity/abstention outcome, not an incorrect forced label.

## Can the 252-family, four-condition design support the intended claims?

**Not as currently specified.** The proposed 252 count and four conditions can at most support descriptive anomaly-detection comparisons on the actual sampled synthetic families once units, labels, and analysis are frozen. The existing power reconstruction is conditional on assumed marginal rates and independent Bernoulli units; it is not evidence that the 252-family frame exists, that the conditions are constructed as intended, that paired superiority is powered, or that the population of circuit families is represented.

* **Hardware attribution:** Not validated. Current calibration is simulator-generated and has no independent device fault samples or time/component-matched telemetry. A `HardwareModel` interface or injected channel truth is not independent hardware evidence. Without a separate provenance channel, the experiment can test behavior under declared synthetic channels only, not real hardware/software attribution.
* **Source localization:** Not validated by the finite-shot calibration. Operation-syntax matching and largest-deviation heuristics are not causal localization, and the calibration summary has no independently adjudicated source/target localization scores. Injected mutation locations can supply benchmark truth for controlled synthetic cases if kept blinded from inference, but that protocol is not yet specified here.
* **Observational identifiability:** Output counts cannot distinguish causes that induce identical measured distributions. The exact unitary construction establishes a mathematical possibility under its assumptions, not its prevalence on hardware. More shots do not separate exactly equal likelihoods. The design can report the limit or assess whether an auxiliary evidence channel resolves a case; it cannot claim output-only causal identification in such cases.
* **Novelty and generalization:** The cited literature already overlaps generic statistical bug/noise distinction, calibrated attribution, and measurement-dependent localization. A narrower new operational claim remains possible but unestablished. The saved one-recipe calibration does not support cross-family generalization, and 252 is not a generalization guarantee without a defined family population and sampling frame.

The design therefore cannot answer the intended joint question—finite-shot detection plus defensible cause/target attribution across independent families—without first resolving the named blockers. This supports **NO-GO**, rather than conditional execution: unresolved evidence currently prevents the proposed experiment from adjudicating its central claims. A narrower anomaly-detection study could be reconsidered under a separately frozen scope and analysis contract.

## Evidence inspected and limits

Primary inputs: `finite_shot_publication_readiness_review.md` and `finite_shot_experiment_protocol_gaps.json`. Related local evidence: `finite_shot_identifiability_analysis.md`, `finite_shot_statistical_power_analysis.md`, `finite_shot_selective_diagnosis_revised_proposal.md`, `finite_shot_candidate_selection_protocol.md`, `finite_shot_shot_aware_heldout_plan.md`, `finite_shot_distribution_calibration_run/summary.json`, `finite_shot_distribution_calibration_run/run_manifest.json`, `src/quantum_simulator/diagnosis_engine.py`, and `src/quantum_simulator/fault_localizer.py`. The readiness review's four-condition and execution arithmetic were treated as proposal content, not as proof of actual units or hardware evidence.

Literature verification was limited to the cited primary/publication records for Virani et al., Hassan & Kaabouch, and Maidl et al. It verifies the overlap statements above; it does not complete a systematic search or methods-level novelty comparison. No novelty claim is made.

No simulations, tests, training, benchmarks, or hardware jobs were run. No source code, datasets, preregistrations, thresholds, acceptance criteria, or existing artifacts were changed.
