# QMutBench standalone stage

The existing binary standalone benchmark is the production-aligned task: WOO/OPO -> detected, Survivor -> undetected. Mutation-aware Extra Trees (23 features) remains best at macro F1 0.793314 across five grouped folds (578,117 rows and 305 origins). The model hash was 8cd332a309e27ef09cc46c5bf8a8788da35c05aa6ed25ed5de21d26ced076175. No production files were changed.

The independent research task preserves all three labels (Survivor=0, WOO=1, OPO=2). Existing matched 4-feature controls and QNN variants were evaluated on the same 205 rows / 12 origins / three grouped folds. Extra Trees scored 0.4669 macro F1; the strongest angle QNN/hybrid scored 0.3218. The reserved 36-origin confirmation has not run. Its data exporter stopped before reading any records at 792 MiB available RAM, below a 900 MiB gate.

See [the prior classical campaign](../../qmutbench_final_classical_campaign_20261027.md), [the QNN standalone report](../../qmutbench_three_class_qnn_campaign_20261027/research_report.md), and [the current ensemble campaign](../../qmutbench_multisource_ensemble_20261028/research_report.md).
