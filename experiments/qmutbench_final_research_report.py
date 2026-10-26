"""Assemble the final research-phase machine-readable and Markdown reports."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/"reports/experiments"
AUDIT=REPORTS/"qmutbench_final_research_baseline_audit_20261026.json"
NESTED=ROOT/"data/qmutbench_ml/final_research_v1/nested_outer/summary.json"
OUT=REPORTS/"qmutbench_final_research_report_20261026"

def main():
    audit=json.loads(AUDIT.read_text(encoding="utf-8")); nested=json.loads(NESTED.read_text(encoding="utf-8"))
    preds=[np.load(ROOT/f"data/qmutbench_ml/final_research_v1/nested_outer/fold{i}.npz",allow_pickle=False) for i in range(1,6)]
    err=json.loads((REPORTS/"qmutbench_exhaustive_error_analysis_20261026.json").read_text(encoding="utf-8"))
    result={"schema":"qmutbench-final-research-report-v1","date":"2026-10-26","status":"complete",
      "baseline_and_feature_audit":audit,"nested_grouped_evaluation":nested,
      "feature_contract_findings":audit["feature_contract"],"external_data_search":audit["external_data_search"],
      "historical_error_analysis_artifact":"qmutbench_exhaustive_error_analysis_20261026.json",
      "historical_error_analysis_summary":err,
      "validation":{"qmutbench_tests":"90 passed, 808 deselected in 19.43s","focused_qasm_and_inference_tests":"17 passed (prior focused run)","nested_script_compiled":True,"real_record_single_and_batch_inference":"passed","train_inference_feature_equivalence":"20 sampled rows, zero mismatches"},
      "nested_prediction_artifacts":{"path":"data/qmutbench_ml/final_research_v1/nested_outer/","fold_files":5,
         "rows":int(sum(len(p["y"]) for p in preds)),"all_ids_unique":bool(len(np.concatenate([p["record_id"] for p in preds]))==len(set(np.concatenate([p["record_id"] for p in preds]).tolist()))),
         "all_outer_groups_disjoint":all(not (set(preds[i]["origin_program"].astype(str)) & set(preds[j]["origin_program"].astype(str))) for i in range(5) for j in range(i+1,5))},
      "decision":{"packaged_model":"mutation_aware_v1 remains the development candidate","threshold":0.5,
         "reason":"Nested selected threshold/calibration strategy did not outperform fixed mutation-aware raw 0.5; calibration selected in only one of five folds and thresholds varied.",
         "independent_external_validation":False,"deployment_claim":"development evidence only; no independent generalization claim"},
      "limitations":["Mutation-aware feature discovery used prior labels from this development corpus, so nested estimates remain selection-informed.",
         "No verified compatible independent dataset was available.","Nested 2-fold inner selection is intentionally small for runtime and is noisy.",
         "Historical test set was not loaded or used.","QASM common-prefix/suffix alignment has ambiguity for repeated or reordered instructions.",
         "ECE is the fixed-bin 10-bin estimate and depends on binning."]}
    OUT.with_suffix(".json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    b=audit["reproduced_oof"]["baseline_metrics"]; c=audit["reproduced_oof"]["mutation_aware_metrics"]
    nm=nested["nested_selected_metrics"]; fixed=nested["fixed_mutation_et_raw_0_5"]
    md=f'''# QMutBench final research phase — 2026-10-26

## Evidence and decision

The selected development candidate remains the mutation-aware Extra Trees artifact at `data/qmutbench_ml/mutation_aware_v1/`, using the existing fixed 0.5 inference threshold. The previous matched three-fold OOF result reproduces at macro F1 **{c['macro_f1']:.4f}** versus **{b['macro_f1']:.4f}** for the incumbent (a {100*(c['macro_f1']-b['macro_f1']):.2f} percentage-point increase). These are historical, selection-informed development results.

The new five-fold outer grouped evaluation scores **{nested['row_count']:,}** rows across **{nested['origin_count']}** origins. It chooses model, calibration method, and threshold within two-fold grouped inner CV; each outer fold's labels are reserved for scoring. The nested-selected aggregate is macro F1 **{nm['macro_f1']:.4f}**, balanced accuracy **{nm['balanced_accuracy']:.4f}**, accuracy **{nm['accuracy']:.4f}**, ROC AUC **{nm['roc_auc']:.4f}**, Brier **{nm['brier']:.4f}**, and 10-bin ECE **{nm['ece_10']:.4f}**. Fixed mutation-aware Extra Trees at raw threshold 0.5 scores macro F1 **{fixed['macro_f1']:.4f}**, balanced accuracy **{fixed['balanced_accuracy']:.4f}**, and Brier **{fixed['brier']:.4f}**. The tuned strategy did not improve on fixed 0.5, so no threshold or calibration change is justified.

Selected fold strategies: `{nested['selected_model_counts']}`; calibration: `{nested['selected_calibration_counts']}`; thresholds: `{nested['thresholds_by_fold']}`. These nested results are still development-only: the mutation feature set and candidate family were designed after examining this corpus. They are not independent external validation.

## Feature and inference audit

The cache covers 578,117 development rows with zero parse errors. A reproducible 20-record check found zero training/inference feature mismatches. The extractor reads no target or origin-group fields. Measurement and barrier instructions are parsed then excluded from the gate/wire sequence; conditions remain unsupported by the QMutBench OpenQASM 2 parser, malformed input is rejected, unsupported gates now raise a named error, and inference rejects original/mutant pairs with differing declared qubit counts. Repeated or reordered instructions can make common-prefix/suffix alignment ambiguous; this limitation remains.

## External data and errors

The [official QMutBench paper](https://arxiv.org/abs/2604.15870) and [repository](https://github.com/EnautMendi/QMutBench) describe 375 original circuits and 723,079 mutants; the locked corpus contains that release. Public metadata for [QSurvDiag](https://zenodo.org/records/23480917) does not establish matching outcome labels or disjoint origins. [QUMUG](https://zenodo.org/records/21718539) uses a different mutant selection and label semantics. Neither was ingested. No compatible independent external dataset was verified. The prior exhaustive error and subgroup analysis is retained at `qmutbench_exhaustive_error_analysis_20261026.json`.

## Validation and reproducibility

Nested results and fold predictions are local under `data/qmutbench_ml/final_research_v1/nested_outer/` (git-ignored). The baseline/feature audit is `qmutbench_final_research_baseline_audit_20261026.json`; this report's full machine-readable form is `qmutbench_final_research_report_20261026.json`. The QMutBench suite passed **90 tests** (808 unrelated tests deselected); focused QASM and inference tests passed **17 tests**. Real-record single/batch inference and the 20-row train/inference feature check passed. The model binaries and QASM caches remain local.

'''
    OUT.with_suffix(".md").write_text(md.rstrip()+"\n",encoding="utf-8")
    print(f"Wrote {OUT.with_suffix('.md')} and {OUT.with_suffix('.json')}")
if __name__=="__main__": main()
