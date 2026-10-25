# Request for QMutBench Source, Lineage, and Evaluation Evidence

**Purpose:** prepare a source-grounded, family-separated program-mutation detection evaluation. This is a draft request only; it has not been sent.

Dear QMutBench maintainer / research collaborator,

We are assessing whether QMutBench can support a reproducible evaluation of output-observable program-mutation detection. We currently regard the dataset as having **zero verified independent recipe roots**. We would appreciate the following evidence for a pinned release, or a clear statement where an item is unavailable.

1. **Pinned source release and custody.** Please identify the repository revision or immutable archive, release date, custodian, source URL, retrieval/accession method, and any signed release record. Provide an archive checksum and a complete per-file checksum manifest (SHA-256 and byte size), including the metadata and all original and mutant circuit files.

2. **Complete origin and mutant map.** Provide a machine-readable row for every original and mutant artifact, with stable artifact IDs, exact file paths and hashes, original-to-mutant parent edges, mutation operator/parameters, generation order, and any other parent or derivative relationships. State the total expected coverage and explain exclusions, unavailable files, and known gaps.

3. **MQT Bench ancestry and generation configuration.** For each original derived from MQT Bench, identify the exact MQT Bench version/commit and benchmark/design ID; abstraction level; circuit size; target/gateset; optimization and mapping settings; generator/dependency versions; adapter/transformation code; seeds and parameters; and hashes of the resulting source artifacts. For origins without known MQT ancestry, please distinguish “none” from “unknown” and explain the basis.

4. **Completeness and reproducibility.** Provide a release completeness statement, the generation scripts/configuration and dependency lockfiles needed to reproduce the mapping and source artifacts, and a checksum for each delivered evidence file. Please identify all known sources, transformations, duplicates, and shared recipe ancestry.

5. **Behavioral labels and evidence.** Describe how mutation outcomes, survival-rate categories, and any “killed” or equivalent fields were generated. Provide the operational definitions, tools and versions, test/oracle conditions, raw supporting evidence, and case-level linkages. Please clarify what can and cannot be concluded about behavioral change or equivalence for an output-only detection endpoint. Dataset documentation, mutation IDs, and survival values alone will not be treated as independent behavioral ground truth.

6. **Execution and measurement provenance.** If execution results are available, provide immutable records linking every observation to circuit/reference hashes, measurement basis, shots, seeds, simulator or device and noise settings, software/runtime versions, execution IDs, raw counts/logs, failures, and timestamps. Identify which configuration fields were fixed or varied.

7. **Independent reconciliation and review.** Please identify a custodian who can answer provenance questions and permit an independent reviewer to reconcile the complete mapping against source artifacts and a prior-artifact inventory. We need reviewable completeness and lineage findings, reviewer independence/conflict disclosures, and externally verifiable attestations. We will not treat a project-authored checklist or an unverified signature claim as independent verification.

The goal is to preserve uncertainty accurately. An original circuit ID or directory name establishes a useful dataset grouping, but does not by itself establish a recipe root or independence from other circuits. Likewise, published dataset descriptions are valuable context but do not substitute for artifact hashes, complete lineage, execution provenance, and independently adjudicated labels.

If some requested evidence does not exist, a precise statement of what was recorded, what is unknown, and the coverage limits would still be useful. No evaluation or model training will proceed until the relevant provenance, label, observation, and independent review gates are satisfied.

Thank you for your time.

Sincerely,  
Quantum Simulator research team
