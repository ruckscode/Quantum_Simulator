# Phase 5.8 — Owner Decision Memo: Reduced Independent Simulator Benchmark

**Current status: NO-GO for benchmark-runner implementation pending owner decisions below.** Phase 5.4 remains the contract; this memo summarizes the open choices recorded in Phases 5.6 and 5.7. It does not authorize implementation, training, or execution.

## 1. Detection denominator

Phase 5.7 proposes **21 templates** for the Phase 5.4 sensitivity target:

- **10 readout cases:** R15 and R30 in all five families.
- **6 gate cases:** G15 and G30 in F1, F2, and F4.
- **5 combined cases:** C15 in all five families.

These are the moderate/severe strengths named in the earlier design. Each has exact D+ truth and exact TVD above the existing finite-shot guard at all shot counts. With five seeds, this yields **105 observations per shot level** and **315 total**. Determine eligibility before predictions and samples; a sampled miss remains a false negative and cannot be removed from the denominator.

The exclusions are deliberate. G15/G30 in F5 are D+ interventions but their exact TVDs (`((2-√2)/4) × p`) remain below .05, so they are descriptive rather than identifiable positives for this fixed-threshold acceptance target. F3 G cases are D−. The .02 subtle and .06 near-threshold strengths are descriptive. Program P1/P2 cases have no declared strength tier and should be reported in separate detection strata.

The reports document alternatives or unresolved membership choices: C15 is scored separately for combined-fault recovery and ambiguous disposition, while 5.7 proposes also including it in output sensitivity because detection is cause-independent. F3 G cases can be included in specificity as D− or reported in a separate output-equivalent hardware stratum. Phase 5.4 requires shot/family stratification but does not say whether the .80 target applies separately per shot or to a pooled result; 5.7 recommends evaluating it per shot.

## 2. Learner route

Phase 5.4 assigns F1–F2 to “Fit learned component and development transformations,” so the final contract requires a learner. Phase 5.7 traces a supported route already present in the code: the ordered observable features are `total_variation_distance`, `js_divergence`, `hellinger_distance`, `observed_entropy_bits`, and `observed_max_probability`. They derive only from expected and observed distributions. The existing LR pipeline applies median imputation, then `StandardScaler`, then `LogisticRegression(max_iter=2000, class_weight="balanced")`; preprocessing statistics are fitted on training data and stored in the portable adapter. No labels, intervention metadata, family IDs, seeds, circuits, or calibration values enter these five model features. Circuit and calibration evidence remain inputs to the separate rule/localization path. C15 remains combined truth and is not forced into the learner’s three single-cause classes.

An **existing trained artifact** could be loaded by the portable adapter only if it existed and matched this schema and provenance. The expected artifact is absent, and the integration report says no genuine trained backend/artifact is integrated. Therefore the proposed route requires **training and exporting a new Phase 5 artifact** on F1–F2; this is distinct from reusing an existing artifact. No training was done for this memo.

## 3. Confidence threshold

The existing ML integration sets `MIN_CONFIDENCE = 0.60`. It controls whether a classifier prediction proceeds: below .60, `diagnose_with_ml` returns `INSUFFICIENT_EVIDENCE` before rule-based localization. At or above .60, the existing evidence/localization path can run and may still yield `AMBIGUOUS` for C15.

Lowering the cutoff would let more lower-confidence predictions reach evidence adjudication; raising it would produce more early insufficient-evidence outcomes and could reduce C15 ambiguous recall because those cases would not reach localization. The Phase 5.4 contract does not explicitly adopt this cutoff. No replacement value is recommended here.

## 4. Execution prerequisites

Before any separate execution authorization, the project needs:

- The exact 60-case ledger, declared seeds and shots, and separate sealed truth records.
- A fail-closed calibration preflight before model construction or sampling: explicit q0–q3 fields; explicit canonical gate calibrations for the ordered-tuple union of expected and observed circuits; only named fault overrides; empty `coupling_map`/`edges`; reject missing, invalid, duplicate, or silently defaulted fields.
- Mechanical family isolation: F1–F2 development, F3 calibration only, F4–F5 locked test; no test access for fitting or threshold selection.
- The approved feature pipeline and a genuine model artifact; predictions passed through the evidence-based diagnosis/localization path.
- Scoring that keeps intervention truth, exact D truth, sampled measurements, and predictions separate; implements the approved 21-template denominator, shot/family reporting, supported localization, and C15 multi-label plus categorical ambiguity metrics.
- Lightweight checks for exact ledger membership, profile validation on both circuits, split disjointness, feature order/leakage, threshold behavior, scoring denominators, and reproducibility artifacts. These are implementation prerequisites, not runs performed here.
- Separate explicit execution authorization. The fixed statistical threshold remains .05 with the existing guard `max(.05, 1/sqrt(shots))`; no threshold selection or tuning is allowed on locked test.

## 5. Minimum owner decisions

1. **Approve the detection denominator and aggregation.** Recommended: approve the 21 templates; include C15 in output-detection sensitivity while scoring its cause separately; include F3 G cases in specificity as exact D− while still reporting them separately; evaluate the .80 sensitivity target separately at each shot level (105 observations per level) and show family strata. Available evidence cannot establish these membership and aggregation choices uniquely because Phase 5.4 does not state them.
2. **Approve the learner route and .60 cutoff.** Recommended: train/export the existing five-feature Logistic Regression route using only F1–F2; keep the .60 cutoff unchanged; use F3 only for probability calibration if needed; pass predictions through existing evidence adjudication. The existing model artifact is absent, so the owner must authorize training a new Phase 5 artifact. The reports do not establish a different confidence threshold or a rationale to change .60.
3. **Confirm the implementation gate.** Once decisions 1–2 are recorded, authorize benchmark-specific runner and lightweight validation work. This does not authorize benchmark execution; execution still needs its own explicit authorization after the prerequisites pass.

**Recommendation:** record decisions 1 and 2 before implementation. Until then, **NO-GO for runner implementation**. No simulations, scans, training, benchmarks, timing, full tests, IBM access, or hardware jobs were run.

## Sources and change record

Read the Phase 5.4 contract, Phase 5.6 execution specification, and Phase 5.7 blocker-resolution report. Created this new memo only; did not alter those reports, production logic, model artifacts, or historical results. Validation was limited to lightweight memo-content/whitespace checks and `git diff --check`.

## Subsequent owner decision — 2026-10-13

The owner approved the Phase 5 specificity denominator on 2026-10-13. This is a decision made now; the preceding sections record that approval was pending when this memo was written.

Formal specificity uses the eight-template denominator: all five N0 controls plus the masked D− controls F3-P1, F3-P2, and F5-P2. With five seeds, the denominator is 40 observations at each shot level. The four F3 G cases (G02, G06, G15, and G30) are excluded from formal specificity and reported separately as reserved-family diagnostics, each with five observations per shot level.

F3 remains reserved and is excluded from training, feature selection, and threshold tuning. F4–F5 remain locked for final testing. This decision resolves the denominator choice only; it does not authorize benchmark execution.
