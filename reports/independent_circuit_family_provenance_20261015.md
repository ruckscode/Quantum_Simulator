# Independent Circuit-Family Provenance Acquisition Report

**Status:** no eligible independent sampling frame identified. Acquisition and provenance review only; no experiment, model fitting, or source-code change was performed. The frozen 45-case evaluation and its artifacts are unchanged.

## Decision

None of the currently available or reviewed source candidates meets all eligibility gates for a future diagnosis evaluation. Public datasets document useful circuit collections and some parent/mutation relationships, but published metadata is not local verification of hashes, recipe lineage, observations, or independent labels. The workspace currently has **zero eligible independent roots**. Do not use any candidate below as a development, calibration, or held-out family until its missing evidence is supplied and independently reviewed.

The best near-term acquisition lead is QMutBench because its public release documents original-to-mutant grouping. It is not an eligible frame as currently staged: only one local origin family is present, its originals are reported to derive initially from MQT Bench, and local audit results show no diagnosis-compatible ground truth. QMutBench and MQT Bench roots must therefore be reconciled rather than counted as separate sources by name.

## Minimum eligibility evidence

These requirements follow `reports/experiments/independent_diagnosis_protocol_package_20261015.json`, `experiments/diagnosis_evidence_gates.py`, and the inventory/reconciliation records:

1. **Traceable source identity:** immutable source release/commit and custodian; source URI; original circuit and configuration bytes; per-artifact cryptographic hashes; canonical circuit/configuration hashes; generator and adapter versions/hashes where applicable.
2. **Attested family and recipe lineage:** a stable root ID, documented recipe/generator lineage, parent IDs for every derived representation, mutation, seed, size variant, and repetition, plus an externally verified inventory attestation. IDs, filenames, seeds, mutation appearance, or byte similarity alone do not establish ancestry. Missing ancestry is ineligible. All descendants of one root remain in one partition.
3. **Deduplication and frame completeness:** reconciliation against the prior artifact inventory and across every proposed source; exact and near-duplicate/recipe-root merges recorded; an independent attestation that the inventory is complete. Present-byte hashes establish only current byte integrity, not historical provenance or independent roots.
4. **Fault and control truth:** controls and fault-positive examples both required. Each case needs a source-grounded condition and independently adjudicated outcome relevant to the diagnosis endpoint. Dataset labels such as “mutant”, mutation operator, buggy/fixed status, or synthetic scenario labels are not by themselves diagnosis ground truth. At least two blinded independent judgments and a separate disagreement-resolution record are required by the project protocol.
5. **Observation provenance:** immutable observations and their hashes tied to the exact circuit/reference/configuration, execution ID, shots, seeds, simulator/noise settings, runtime/library versions, and (if applicable) device/session metadata. Preserve raw counts/logs and execution failures. Labels must be supported by evidence separate from predictors.
6. **Partition and authorization:** freeze the protocol before outcomes are inspected; split whole lineage roots into development, calibration, and sealed test partitions; keep the frozen 45 and all related roots out. Establish a defensible sample-size/power and resource plan from the eligible frame before collection. Existing 8-development/8-validation targets are not proof that enough roots exist and do not create an ML test partition.

The evidence-gate validator checks structure but does not authenticate identities, signatures, or truth. An external reviewer must verify attestations and custodian independence.

## Locally available evidence

| Candidate / record | Locally verified status | Eligibility decision and missing evidence |
|---|---|---|
| Sampling frame and prior inventory | `diagnosis_family_sampling_frame_20261015.json` records `blocked_incomplete_prior_inventory`, zero eligible roots, no selected development/validation roots, 5,366 prior-artifact circuit records, and 5,186 unknown or unreconciled lineage records. The reconciliation says the saved inventory was truncated at 20,000 artifacts / 124,638,914 bytes and no new filesystem discovery was performed. | **Ineligible.** Complete independently attested inventory, root/ancestor reconciliation, deduplication, and verified source attestations are absent. |
| Local QMutBench sample | Static audit records file hashes for the local GHZ origin and 12 mutants (one origin family); QMutBench portion of the audit covers 12 cases. Two saved result snapshots disagree on parse/diagnosis support (11/12 versus 12/12); both show zero reliable ground-truth and zero metric-eligible cases. | **Ineligible.** One root cannot support disjoint partitions. No independently adjudicated diagnosis truth or compatible observation provenance; resolve result snapshot discrepancy only through a future provenance audit, not by choosing one. Local hashes establish current files only. |
| QMutBench full-data metadata files | `data/qmutbench_full/README.md` describes a much larger public collection. Local `all_data.csv` (60,636,991 bytes) and `grouped_data_SR.csv` (4,006,447 bytes) were only header/sample-inspected; they are not verified complete circuit/lineage packages. | **Ineligible.** No complete local archive, immutable release manifest/checksums, or per-record lineage/custodian attestation verified. Do not infer local availability or family count from README claims. |
| Historical ML records | Recovery report covers 270 records; source/generator hashes and recipe ancestry are missing; labels are construction metadata rather than independent truth; no physical observations. | **Ineligible for primary train, calibration, development/model selection, or held-out evaluation.** Permitted only as clearly caveated software-engineering fixtures per the recovery record. |
| Bugs4Q integration records | Local audit reports no dataset in one validation snapshot; another saved integration snapshot records 42 cases, 41 parsed, 19 translated/executable/diagnosed, 22 unsupported, and zero labeled/metric-eligible cases. | **Ineligible.** Snapshot disagreement and absent compatible labels/observation provenance; software/SDK defect records are not automatically independent circuit-recipe families. |
| MQT Bench, QASMBench, VeriQBench | No local corpus verified in the project inventory. | **Not locally eligible.** Public circuit assets do not supply a verified local sampling frame, diagnosis labels, or the required execution evidence. |

These counts and statuses are from the saved reports; this task did not rerun corpus discovery or benchmark execution. Historical source references and project evidence are listed below.

## Public source assessment

“Published” below describes the source’s public documentation. It does not mean the corresponding release was downloaded, hash-verified, or lineage-attested in this workspace.

| Source | Potential contribution | Assessment against required evidence | Decision |
|---|---|---|---|
| [QMutBench repository](https://github.com/EnautMendi/QMutBench), [paper](https://arxiv.org/abs/2604.15870) | Public docs describe original programs grouped with their generated mutants and mutation metadata; docs identify MQT Bench among the initial original-circuit sources. Useful lead for paired source/mutation artifacts. | Parent grouping and mutation labels are useful metadata, but do not independently establish all recipe roots, source hashes for the exact local release, behavioral diagnosis truth, or observation provenance. Its MQT-derived roots cannot be counted as independent of MQT without root-level reconciliation. Locally only one family is verified. | **Candidate only; not eligible.** Acquire a pinned release and custodian lineage/label attestations, then reconcile roots and observations. |
| [MQT Bench repository](https://github.com/munich-quantum-toolkit/bench), [project documentation](https://www.cda.cit.tum.de/mqtbench/description), [paper](https://arxiv.org/abs/2204.13719) | Versioned benchmark algorithms and related circuit representations across abstraction/target levels; potential traceable originals and generator lineage. | Multiple target mappings, abstractions, and size variants are related descendants, not independent families. No fault/control diagnosis labels or observations established here. QMutBench reports MQT Bench as an original-circuit source. | **Candidate source of controls/origins only; not eligible alone.** Need root mapping and independently labeled positive cases. |
| [QASMBench repository](https://github.com/pnnl/QASMBench), [paper](https://arxiv.org/abs/2005.13018) | Public OpenQASM benchmark circuits with descriptive metadata; potential original controls. | Public circuit identity is not complete recipe-lineage proof; no paired fault positives, independent diagnosis labels, or compatible observation manifests were verified. No local corpus. | **Candidate source of controls only; not eligible alone.** |
| [VeriQBench repository](https://github.com/Veri-Q/Benchmark), [paper](https://arxiv.org/abs/2206.10880) | Public benchmark examples and scripts for generating circuits at multiple scales/types; potential originals and generation recipes. | Generated scales/instances are not independent by default. No independently labeled fault/control pairs or observation provenance established; no local corpus. | **Candidate source only; not eligible alone.** |
| [Bugs4Q repository](https://github.com/z-928/bugs4q), [Zenodo release](https://zenodo.org/records/8148982), [paper](https://www.sciencedirect.com/science/article/am/pii/S0164121223002005) | Publicly described real quantum software bugs with buggy/fixed artifacts and test cases; possible separate software-defect study. | Defect records concern SDK/software behavior and are not necessarily independent circuit recipe roots. Local evidence is incomplete and has conflicting compatibility snapshots; buggy/fixed status and tests do not substitute for independently adjudicated labels for this diagnosis endpoint. Runtime/library-version reproducibility matters. | **Not eligible for this frame.** Could support a separately specified SDK-defect evaluation after acquisition and protocol review. |

No reviewed public candidate, singly or in combination, currently supports a defensible development, calibration, and sealed held-out split. Combining public names or repositories does not create independence; shared origins and recipe ancestry must be merged before partitioning.

## Acquisition checklist

1. **Name the source custodian(s).** For each selected corpus, obtain the release maintainer or dataset author as custodian. The custodian supplies a signed or otherwise externally verifiable statement identifying the release, source revision, generation process, known parent datasets, and completeness of the delivered artifacts. The project evaluator must not attest to its own source inventory.
2. **Acquire immutable source artifacts.** Obtain pinned release archives/commits, license and inclusion terms, original QASM/circuit and configuration files, metadata, generator source/version, adapter code/version, and authoritative checksums. Preserve the original archive and record local receipt hashes; do not substitute regenerated files for source artifacts.
3. **Obtain root/recipe lineage.** Request a machine-readable manifest mapping every circuit and derivative to stable origin-family and recipe-root IDs, parent IDs, generator/recipe revision, and all known cross-corpus sources. Require the custodian to explain mutation/variant creation and certify known omissions. For QMutBench, explicitly resolve links from each origin to MQT Bench and any other upstream source.
4. **Reconcile the complete frame.** An independent inventory reviewer reconciles all supplied roots and artifacts against the existing prior-artifact inventory, including the 5,186 unknown/unreconciled records; records exact/near-duplicate and shared-recipe decisions; and signs/attests the completeness result. If the original inventory remains truncated, first obtain a complete inventory snapshot and its independent completeness attestation. Do not promote candidates to eligible while this gate is open.
5. **Acquire diagnosis labels and observations.** A separate label custodian supplies source-grounded fault/control definitions, the target endpoint, independent supporting evidence, raw observations/logs/counts, and their hashes with full execution configuration and version metadata. At least two blinded reviewers make independent judgments; a separately appointed resolver records disagreements. Obtain external verification of reviewer independence and attestations. Keep labels and label-derived fields out of predictors.
6. **Review before partitioning.** An independent protocol reviewer verifies hashes, lineage, source/label/observation attestations, endpoint compatibility, exclusions, and all gates. Only then freeze root-level development/calibration/test assignments, preregistration, power/resource caps, and access controls. Keep the historical 45 cases and their ancestor/descendant roots excluded. If eligible roots or labels remain insufficient, record the frame as blocked; do not fit a model or open a confirmatory evaluation.

## Smallest next action

Request from the custodian of one candidate corpus (QMutBench is the most direct lead) a **pinned-release, complete origin/recipe-lineage manifest with immutable artifact checksums, upstream-source links, and a completeness attestation**. Have an independent reviewer reconcile that manifest against the existing prior-artifact inventory before acquiring or evaluating new cases. If the custodian cannot provide it, retain the current zero-eligible-roots status and stop acquisition planning for that source.

## Evidence and source references

Local records inspected:

- `reports/independent_research_handoff_20261015.md`
- `reports/experiments/diagnosis_family_sampling_frame_20261015.json`
- `reports/experiments/diagnosis_sampling_frame_reconciliation_20261015.json`
- `reports/experiments/diagnosis_prior_lineage_inventory_20261015.json`
- `reports/experiments/independent_diagnosis_protocol_package_20261015.json`
- `reports/experiments/benchmark_provenance_static_audit_20261014.json`
- `reports/experiments/ml_historical_record_recovery_20261015_v2.json`
- `reports/experiments/ml_current_pipeline_audit_20261014_release.json`
- `experiments/diagnosis_evidence_gates.py`
- `data/qmutbench/` and `data/qmutbench_full/` (metadata/corpus presence as described above)

Public primary sources:

- QMutBench: [repository](https://github.com/EnautMendi/QMutBench), [paper](https://arxiv.org/abs/2604.15870)
- MQT Bench: [repository](https://github.com/munich-quantum-toolkit/bench), [documentation](https://www.cda.cit.tum.de/mqtbench/description), [paper](https://arxiv.org/abs/2204.13719)
- QASMBench: [repository](https://github.com/pnnl/QASMBench), [paper](https://arxiv.org/abs/2005.13018)
- VeriQBench: [repository](https://github.com/Veri-Q/Benchmark), [paper](https://arxiv.org/abs/2206.10880)
- Bugs4Q: [repository](https://github.com/z-928/bugs4q), [Zenodo record](https://zenodo.org/records/8148982), [paper](https://www.sciencedirect.com/science/article/am/pii/S0164121223002005)

## Addendum — QMutBench source acquisition (2026-10-15)

After this report's initial audit, a separately authorized bounded acquisition obtained the official QMutBench metadata and a small pinned QASM subset. The source CSV is 59,913,911 bytes, SHA-256 3f4547868673c583f047cee19d27c7bc97c64f15bb0586098ada778a3a0ff778, from repository commit 01f5e7700c051fe715fc368b307deaa7e6869ba8 (reported unsigned by GitHub). Its schema and all 723,079 rows match the prior workspace CSV under parsed row-by-row comparison. The prior byte-size/hash difference is explained by LF in the downloaded official copy and CRLF in data/qmutbench_full/data/all_data.csv.

The official metadata enumerates 375 Origin_program IDs, but the available public origin/mutant grouping does not attest that these are independent recipe roots, and the corpus acknowledges MQT Bench as an upstream source. Thirteen previously staged GHZ files matched the pinned official commit byte-for-byte; the bounded pipeline acquired additional official files for its 12-record smoke. The smoke represents one origin group and remains ineligible for an independent sampling frame. It does not resolve source-root independence, compatible independent labels, or observation provenance for the remaining corpus.

Updated acquisition and pipeline evidence is in reports/qmutbench_data_pipeline_20261015.md; the official-source hashes and QASM file list are in data/qmutbench_ingestion/source_manifest.json. This addendum changes the source-verification status only: no circuit family is promoted to eligible.
