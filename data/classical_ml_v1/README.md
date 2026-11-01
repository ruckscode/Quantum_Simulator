# Classical ML source and benchmark registry (v1)

This small tracked registry preserves source/task provenance and completed
grouped nested-OOF metrics. It intentionally does not copy raw quantum source,
issue text, QASM, or multi-gigabyte execution traces into Git. Raw downloads and
processed task tables remain local/ignored; acquisition, task manifests, and
checksums are recorded in `data/external_quantum_datasets/` and local
`data/processed/**/manifest.json` files.

## Files

- `source_registry.csv`: task meaning, target, source family, and limits.
- `benchmark_results.csv`: pooled outer-fold metrics and validation method.
  Scores belong to their listed tasks; they are not interchangeable estimates
  of simulator diagnosis accuracy.

## Evaluation protocol

The source-specific benchmark searched word/character TF-IDF with LinearSVC,
LogisticRegression, SGD modified-huber, ComplementNB and sparse cosine KNN.
For each outer GroupKFold fold, configuration selection used inner GroupKFold
accuracy; the selected model predicted held-out groups once. Pooled outer
predictions generated accuracy, macro-F1, and balanced accuracy. Grouping
prevents records from the same issue/program group crossing train/test folds.
It does not remove source shift or label-construction limits.

Ten rows include nine source adapters plus the three-source same-target merge.
QMutBench is a separate mutation-outcome task. The current QMutBench inference
bundle is not preserved in this checkout, so its benchmark is not a runnable
inference model here.

## Diagnosis integration status

None of these source-specific classifiers is called from `diagnose_circuit()`.
Their labels differ (code snapshot state, comments, detector outcome, harness
pass/fail, backend identity). Combining the scores would have no single target
meaning. A diagnosis classifier can only be used through the explicit optional
`classifier=` argument; it consumes five observable distribution features,
and the bundled Logistic Regression candidate is marked not validated for
deployment. The evidence/statistical diagnosis engine remains the default path.

## Reproduce / obtain the data

Use source URLs, snapshot commits, hashes, schema notes, and license review in
`data/external_quantum_datasets/master_dataset_inventory.csv`. Do not
redistribute third-party corpora until source-specific terms permit it. The
compact registry makes the work understandable; it does not make absent local
input files available to a fresh clone.

## Fresh-clone reproducibility and source access

`source_acquisition_manifest.csv` is the Git-tracked acquisition registry for the external datasets. It records upstream URLs, pinned snapshots where known, data/task descriptions, license and redistribution cautions, local extraction destinations, and available archive or repository checksums. The source corpora, processed tables, and per-run outputs are deliberately not committed: they are large, some contain third-party source text or code under different terms, and several sources are not labels for the same task. A collaborator should obtain each corpus from its listed upstream source, review its terms, and place it in the documented local path before running preparation. The manifest is provenance guidance; it does not itself distribute those datasets.

The nine-source grouped benchmark adapters were generated from validated local manifests and partitions by `experiments/prepare_nine_source_classical_v2.py`. That script checks the source hashes, partition hashes, row counts, IDs, labels, and source validation flags before producing adapters. The nested search runner is `experiments/run_classical_source_nested_search.py`; its source inputs are configured by `reports/experiments/classical_ml_benchmarks_20261030/inputs/classical_nine_source_v2_config.json`, whose referenced adapters are intentionally local experiment artifacts. Reproducing the exact reported scores therefore requires recreating those inputs from the upstream sources and the corresponding source-specific preparation contracts; the benchmark CSV contains the published metrics and validation protocol, not raw training rows.

The source-specific scores must not be presented as one pooled simulator diagnosis accuracy. Each dataset has its own target semantics and limitations, and the registry marks these source models as not used by diagnosis inference. See `../../reports/classical_ml_sim_diagnosis_benchmark_20261031.md` and `../../reports/ml_diagnosis_integration_report.md` for the separate, evidence-gated diagnosis candidate and its acceptance status.
