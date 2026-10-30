# QML accuracy optimization v3 — continuation update

Current branch commit: `a2d51be22fd985bbdd10655f7e31dc77b9d19ed6`. The user-provided latest commit `8965d3cfa9d40a53f841e99268d72eedd50fbe56` is not in this checkout’s Git object database; current local HEAD is unchanged.

Development QASM extraction is complete and independently rehashed; see `feature_extraction_audit.json`. Grouped OOF contains 578,117 records / 305 origins, three development folds, with zero parser errors or label mapping mismatches.

The existing five-fold ExtraTrees champion remains 79.33% accuracy / 79.33% macro F1 / 79.41% balanced accuracy. Same-campaign QASM feature screens on three folds showed modest aggregate gains for mutant gate composition (+0.48 pp macro F1 over its matched metadata baseline), but improvement was not consistent in every fold; structure-only and delta-composition screens regressed. The complete all-QASM candidate has only one fold. None are promoted, and none demonstrate improvement over the five-fold champion because their fold contracts differ. Full details are in `qasm_feature_ablation_report.md`.

The correct existing binary label map is WOO=1, OPO=1, Survivor=0 and was verified against every development OOF record. `source_gate_count_match` remains excluded pending an ablation.

40 relevant tests passed after directing temp files to the writable workspace; see `test_report.md`. No QNN training occurred: current available RAM was 487.5 MiB at report generation, below the 1 GiB guard. Frozen-test labels were not accessed. The v3 directory remains partial.
