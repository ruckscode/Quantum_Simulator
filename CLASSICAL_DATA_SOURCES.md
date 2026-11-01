# Classical ML data sources: download links and local destinations

This guide answers two practical questions: **which outside data was used for each published classical-ML task, and where does it go in a fresh clone?** The links below point to the original public repositories or dataset records. Raw data is not mirrored in this GitHub repository; download it from those upstream locations and follow each source's terms before use.

The benchmark reports are already committed under `data/classical_ml_v1/benchmark_results.csv` and `reports/`. Those are evaluated results. Getting the original corpora is a separate step; obtaining a raw corpus alone does not recreate the processed training table or the reported score.

## Sources used by the classical multi-source benchmark

| Benchmark task | Upstream dataset / download page | Version used in the recorded acquisition | Local raw-data destination | What the task predicts |
|---|---|---|---|---|
| `bugs4q_core` | [Bugs4Q-Framework (GitHub)](https://github.com/Z-928/Bugs4Q-Framework) and [Bugs4Q dataset record (Zenodo)](https://zenodo.org/records/8148982) | `e596a251ec723e58a042557277d584d56ffa1781` | `data/external_quantum_datasets/bugs4q_full/official_framework/` | Buggy vs fixed source snapshot, grouped by issue |
| `bugs4q_na` | [Bugs4Q-NA (GitHub)](https://github.com/Z-928/Bugs4Q-NA) | `bb853b888c752f90f2a6515c6776e020e2e59d01` | `data/external_quantum_datasets/bugs4q_na/` | Buggy vs fixed source snapshot, grouped by issue |
| `bugs4q_pooled` | [Bugs4Q-Framework](https://github.com/Z-928/Bugs4Q-Framework) + [Bugs4Q-NA](https://github.com/Z-928/Bugs4Q-NA) | The two snapshots above | The two destinations above | Pooled source-state task; it does not add a new independent source |
| `qbugs_qasm_v2` | [QBugs dataset record (Zenodo)](https://zenodo.org/records/14288140) and [HornBro project (GitHub)](https://github.com/JanusQ/HornBro) | HornBro Zenodo v3, dated 2024-12-06; archive MD5 is recorded in the source manifest | `data/external_quantum_datasets/qbugs_hornbro/` | Correct vs buggy QASM circuit snapshots, grouped by pair/scenario |
| `bug_correctness_three_source_pooled` | The Bugs4Q core, Bugs4Q-NA, and QBugs sources above | Same pinned source versions above | The three source destinations above | A pooled correct/fixed vs buggy task; labels are not hardware diagnoses |
| `quantum_platform_comment_v2` | [Quantum Platform Bugs (GitHub)](https://github.com/MattePalte/Bugs-Quantum-Computing-Platforms) and [paper (arXiv)](https://arxiv.org/abs/2110.14560) | `0c1c805fd5dfce465a8955ee3faf81037023a23e` | `data/external_quantum_datasets/quantum_platform_bugs_223/` | The dataset's issue-comment annotation label |
| `bugs4q_robust_execution` | [Quantum Defect Reproducibility package (GitHub)](https://github.com/kusumotolab/quantum-defect-reproducibility) | `c7048476e5e76e95412b3640a3fa5651f51c01a4` | `data/Bugs4Q_Robust_replication/` | Harness pass vs fail in grouped execution records |
| `qolumbina_ideal` | [Qolumbina empirical-study artifact (Zenodo)](https://zenodo.org/records/20987445) | Zenodo v1, 2026-06-28; archive SHA-256 is in the source manifest | `data/external_quantum_datasets/qolumbina/extracted/qolumbina-empirical-study-main/` | Detection of seeded mutants in the study artifact, not naturally occurring defects |
| `lintq_context_v2_conflict_excluded` | [LintQ code (GitHub)](https://github.com/sola-st/LintQ) and [dataset deposit (Figshare)](https://doi.org/10.6084/m9.figshare.22350592) | `481ca24dfabf032eb93cdf728369242b25a5216e`; dataset archive SHA-256 is in the source manifest | `data/external_quantum_datasets/lintq/` | Exploratory warning-context label; conflicting-label cases were excluded |
| `qsimbench_exploratory` | [QSimBench code (GitHub)](https://github.com/GBisi/qsimbench) and [QSimBench data (GitHub)](https://github.com/GBisi/qsimbench-dataset) | `fb2056243df7fd45a37a5d4bc6bc01b84db9e932` | `data/external_quantum_datasets/qsimbench/dataset/` | Exploratory backend-identity prediction from execution traces |

The acquisition registry [`data/classical_ml_v1/source_acquisition_manifest.csv`](data/classical_ml_v1/source_acquisition_manifest.csv) contains the full source URLs, pinned snapshot IDs, archive checksums, license notes, local paths, and acquisition status. It also lists datasets that were investigated but are **not** inputs to the classical benchmark above; do not mistake an inventory entry for a trained task.

## Download and place Git repositories

Run from the root of a fresh project clone. These commands retrieve the pinned Git snapshots listed above. They do not download the Zenodo or Figshare archives; use the linked dataset record pages for those.

```bash
mkdir -p data/external_quantum_datasets/bugs4q_full data/external_quantum_datasets data/Bugs4Q_Robust_replication data/external_quantum_datasets/qsimbench
git lfs install

git clone https://github.com/Z-928/Bugs4Q-Framework.git data/external_quantum_datasets/bugs4q_full/official_framework
git -C data/external_quantum_datasets/bugs4q_full/official_framework checkout e596a251ec723e58a042557277d584d56ffa1781

git clone https://github.com/Z-928/Bugs4Q-NA.git data/external_quantum_datasets/bugs4q_na
git -C data/external_quantum_datasets/bugs4q_na checkout bb853b888c752f90f2a6515c6776e020e2e59d01

git clone https://github.com/MattePalte/Bugs-Quantum-Computing-Platforms.git data/external_quantum_datasets/quantum_platform_bugs_223
git -C data/external_quantum_datasets/quantum_platform_bugs_223 checkout 0c1c805fd5dfce465a8955ee3faf81037023a23e

git clone https://github.com/kusumotolab/quantum-defect-reproducibility.git data/Bugs4Q_Robust_replication
git -C data/Bugs4Q_Robust_replication checkout c7048476e5e76e95412b3640a3fa5651f51c01a4

git clone https://github.com/GBisi/qsimbench-dataset.git data/external_quantum_datasets/qsimbench/dataset
git -C data/external_quantum_datasets/qsimbench/dataset checkout fb2056243df7fd45a37a5d4bc6bc01b84db9e932
git -C data/external_quantum_datasets/qsimbench/dataset lfs pull
```

The GitHub repositories contain the upstream source layout; this clone command does **not** manufacture the processed CSVs consumed by the benchmark runners. For Zenodo/Figshare entries, open the linked record, select its downloadable archive, extract it under the destination in the table, and verify its checksum against the manifest before processing. Some sources are multi-gigabyte.

## Rebuilding the processed benchmark inputs

The published task tables and fold predictions are local experiment artifacts and are intentionally not committed. Source-specific normalization and validation are described by `experiments/source_task_contracts/*.json` and the preparation/audit scripts under `experiments/`. The nine-source adapter builder is:

```bash
python experiments/prepare_nine_source_classical_v2.py
```

It expects the already-created, hash-verified `data/processed/**` partitions and manifests, plus its referenced prior experiment manifests. A fresh download must first be normalized into those contracts; the builder refuses to proceed when files, checksums, partitions, or row counts are missing or invalid. The source-specific nested search runner is `experiments/run_classical_source_nested_search.py`. It expects task adapter CSVs and a config under ignored local experiment-output paths. Consult the source contract and corresponding source-specific preparation scripts before running it.

**Reproduction status:** this repository publishes the metrics, protocol, code, and provenance, but does not currently provide a one-command downloader/normalizer that regenerates every local processed input from an empty checkout. Therefore the benchmark table is transparent and reviewable, but an exact fresh-clone rerun still requires source-specific data preparation. Do not claim exact reproduction from the download commands alone.

## License and target cautions

Check the upstream license and terms before downloading, processing, or redistributing any corpus. In particular, QSimBench is AGPL-3.0; QML-platform issue data has no located repository license; and some research archives contain third-party program fragments with their own conditions. The task labels differ across sources (source-state, issue annotation, harness outcome, seeded-mutant detection, warning triage, backend identity). Their accuracy scores are task-specific and cannot be pooled as a single simulator diagnosis accuracy.
