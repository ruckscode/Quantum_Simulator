# QMutBench Bounded Data Pipeline Milestone

**Date:** 2026-10-15  
**Status:** implemented and smoke-tested on a bounded benchmark subset. No full ingestion, model fitting, confirmatory evaluation, or hardware run was performed.  
**Frozen baseline:** unchanged; the 45-case fixture, labels, thresholds, and acceptance test were not edited.

## Dataset acquisition and integrity

The metadata source is the [official QMutBench repository](https://github.com/EnautMendi/QMutBench) and its [published data endpoint](https://enautmendi.github.io/QMutBench/data/all_data.csv). The pinned repository commit is [01f5e7700c051fe715fc368b307deaa7e6869ba8](https://github.com/EnautMendi/QMutBench/commit/01f5e7700c051fe715fc368b307deaa7e6869ba8); GitHub reports that commit as unsigned. The metadata CSV was retrieved on 2026-10-15 (HTTP Date observed 13:22:59 UTC), is 59,913,911 bytes, and has SHA-256 3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778. Its published Last-Modified header was 2026-03-10 15:37:10 UTC and ETag was 69b03aa6-39236b7.

The previously observed hash/size discrepancy is resolved. data/qmutbench_full/data/all_data.csv is 60,636,991 bytes with SHA-256 d8adabf58173f1ae7c28954832e1b9607bc315eb1269a9cb57237343dfa95934. It has the same 12 parsed columns and all 723,079 parsed data rows in the same order as the official copy. The 723,080-byte difference is exactly one newline byte per line: the official copy has 723,080 LF line endings; the workspace copy has 723,080 CRLF line endings. No parsed row difference was found. The downloaded authoritative bytes are stored separately; the existing copy was not overwritten.

The complete metadata CSV has 723,079 records, 375 distinct Origin_program IDs, 28 algorithms, and a 2–30 qubit range. The schema is the CSV blank index column plus Origin_program, algorithm, qubits, gates, operation, Position_int, Killed, gate, algorithm_group, dominant_state, and position. The raw Killed values are WOO (368,568), Survivor (348,299), and OPO (6,212). They are retained as metadata; this pipeline does not interpret them as diagnosis labels or independently adjudicated fault truth.

| Field | Meaning used / limits |
|---|---|
| Blank first column | Preserved source index; checked against CSV row order and used with the source hash for stable IDs. |
| Origin_program | Published original-circuit identifier; links each mutant to the official parent group, but does not by itself prove recipe-root independence. |
| algorithm | Algorithm name shown by the public interface. |
| qubits | Circuit qubit count shown by the public interface; values above the simulator's 12-qubit cap are rejected. |
| gates | Raw gate-count metadata; preserved, not used as a label or primary feature. |
| operation | Mutation operator; public docs list Add, Remove, and Replace. Metadata only. |
| Position_int | Raw integer mutation-position metadata. |
| position | Mutation-position percentage band shown by the public interface. |
| Killed | Raw outcome/survival field. Exact meanings of WOO and OPO are unresolved; never used as ground truth here. |
| gate | Gate characteristic shown by the public interface; metadata only. |
| algorithm_group | Preserved raw field; exact semantics are not established by reviewed release documentation. |
| dominant_state | Preserved raw field; exact semantics are not established by reviewed release documentation. |

QMutBench documentation describes original-to-mutant directory grouping and acknowledges MQT Bench as an initial original-circuit source. It supports parent association for the corpus but does not attest that all 375 IDs are independent recipe roots. Thirteen previously staged GHZ files were byte-compared with the pinned repository; eleven additional mutant files for the selected smoke range were fetched from that pinned commit. The resulting local source bundle contains one original plus 23 mutant QASM files, each with a SHA-256 in data/qmutbench_ingestion/source_manifest.json. Only one original group was used in the smoke run.

## Pipeline and contract

Entry point: experiments/qmutbench_corpus_pipeline.py.

~~~powershell
.\.venv\Scripts\python.exe experiments\qmutbench_corpus_pipeline.py --start-row-index 47223 --limit 12 --shots 256 --seed-base 1701 --output-dir data\qmutbench_ingestion\runs\smoke_ghz_first12_20261015_v6
~~~

The pipeline verifies the metadata CSV against its source manifest (hash, size, schema), checks each QASM file against a pinned commit/hash, derives stable row IDs from the source digest and source row index, preserves original-to-mutant association and all raw metadata, and processes records as a stream. It has a bounded simulator working-array cap (12 qubits and 64 MiB per circuit), explicit parse/unsupported/resource/source/execution failure statuses, and resumable JSONL output using atomic checkpoints with source/config/code hashes and byte offsets. The source CSV, normalized records, observations, and rejections/failures are separate artifacts. Only eight original operation caches are retained; completed normalized rows are streamed through preflight rather than loaded as a corpus.

The generated observations use the versioned qmutbench-simulated-observation-v1 schema and contain original and mutant QASM hashes, source-row hash/index, pinned source commit, simulator/parser/pipeline code hashes, Python/NumPy/platform versions, seed, 256-shot count, ideal/no-noise configuration, computational-basis distributions and counts, TVD, and an explicit interpretation limit. “Equivalent” means only indistinguishable in the exact computational-basis output distribution at tolerance 1e-12; it does not establish full program equivalence. No hardware behavior is modeled. The run manifest records field lists for normalized, observation, and failure schemas.

The separate preflight blocks fitting for missing source hashes, unlocked/overlapping lineage-group partitions, incompatible task labels, label-leaking primary features, missing primary feature schema, missing preregistered minimum family count, unverified independent roots, and non-independent labels. It does not authorize fitting even when syntactic checks pass unless family and label attestations are supplied.

## Smoke result

Artifact: data/qmutbench_ingestion/runs/smoke_ghz_first12_20261015_v6/run_manifest.json.

| Measure | Result |
|---|---:|
| Full source metadata rows / original IDs | 723,079 / 375 |
| Selected QMutBench mutant records | 12 |
| Original groups represented | 1 (ghz_indep_qiskit_2; independence unverified) |
| Parsed and supported | 12 / 12 |
| Simulated original/mutant pair runs | 12 pairs (24 circuit sampling calls) |
| Rejected or failed records | 0 |
| Exact computational-basis distribution changed | 4 |
| Equivalent/indistinguishable in that basis | 8 |
| Maximum estimated simulator working-array size | 192 bytes per circuit (does not include Python/process overhead) |
| ML preflight | Blocked; 0 eligible independent roots |

This is an ingestion smoke test, not a mutant-detection performance evaluation. The behavior counts are simulator-derived observations for one recipe group, not independent labels, generalization evidence, or proof that Killed indicates observable change.

## Validation and baseline status

Focused command:

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests\test_qmutbench_corpus_pipeline.py -q --tb=short --basetemp .tqmb-focus-finalcheck-20261015
# 9 passed
~~~

The focused tests cover source hash and schema rejection, parent lineage and stable identifiers, malformed and unsupported QASM, equivalent observation handling, interrupted/resumed ingestion, split overlap, label leakage, and missing provenance/label compatibility.

Bounded suite command, preserving both registered finite-shot exclusions:

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests -q --tb=short -k "not bounded_recipe_comparison_120_templates_and_finite_shot_subset and not independent_calibration_fixture_and_permutation_comparison" --basetemp .tqmbi-finalcheck-20261015
# 803 passed, 1 failed, 2 deselected
~~~

The sole failure is test_heldout_generalization_outcomes_and_targets, the existing frozen 45-case acceptance test. It still reports the same seven mismatches: three subtle QFT-like program faults and four ring-cluster correct/equivalent controls. This reproduces the frozen baseline failure; no frozen artifact or threshold was changed. Both finite-shot NO-GO tests remain deselected.

git diff --check and JSON checks were run after implementation. No packages were installed. The run was limited to 12 records; the full 700,000-plus mutant dataset was not ingested.

## Eligibility, unresolved evidence, and next action

The corpus now has a reproducible source-byte and observation path for benchmark-derived mutant-artifact records. The 375 published origin IDs are not yet verified independent circuit-family/recipe roots. The small acquired QASM subset does not represent the whole metadata corpus. The Killed field's values are preserved without assuming their semantics, and mutant/original artifact roles are not independently adjudicated behavioral fault labels. Consequently this does not resolve the independent-family or independent-ground-truth blocker, does not support three-way hardware-aware labels, and does not establish ML generalization.

**Smallest next action:** obtain an upstream, versioned origin/recipe-lineage attestation mapping the complete QMutBench roots and shared MQT Bench ancestry, then independently reconcile that map against the project's prior-artifact inventory. Until that review is complete and an independent labeling and partition protocol is approved, do not launch full ingestion or model fitting.
