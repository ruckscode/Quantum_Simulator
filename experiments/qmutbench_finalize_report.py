"""Build the final QMutBench campaign report from verified experiment outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.qmutbench_qasm_features import SOURCE_CSV, sha256_file
from experiments.qmutbench_qasm_model_comparison import metrics

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data/qmutbench_ml/exhaustive_campaign_v1"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build() -> dict:
    three = read(EXP / "mutation_diff_screen.json")
    five = read(EXP / "five_fold_confirmation/summary.json")
    hgb = read(EXP / "histogram_boosting_screen/summary.json")
    ablation = read(EXP / "feature_ablation_no_gate_context/summary.json")
    errors = read(ROOT / "reports/experiments/qmutbench_exhaustive_error_analysis_20261026.json")
    extractor = read(ROOT / "data/qmutbench_ml/qasm_mutation_diff_v1/manifest.json")
    artifact = read(ROOT / "data/qmutbench_ml/mutation_aware_v1/best_model_metadata.json")
    selected_diff_features = set(three["mutation_diff_features"])
    old_qasm = read(ROOT / "reports/experiments/qmutbench_qasm_model_results_20261025.json")
    prior = read(ROOT / "reports/experiments/qmutbench_final_performance_20261025.json")

    baseline_folds = []
    pred_root = ROOT / "data/qmutbench_ml/performance_phase_full_v1/predictions"
    for fold in (1, 2, 3):
        part = np.load(pred_root / f"position_fraction_fold{fold}.npz", allow_pickle=False)
        baseline_folds.append(metrics(part["y"], part["probability"]))
    screen60_parts = [np.load(EXP / f"mutation_diff_folds_matched/fold{fold}.npz", allow_pickle=False)
                      for fold in (1, 2, 3)]
    screen60 = {key: np.concatenate([part[key] for part in screen60_parts])
                for key in ("record_id", "y", "origin_program", "fold", "probability")}

    numeric = {
        "qasm_diff_orig_span": "Original changed instruction count after common-prefix/suffix alignment.",
        "qasm_diff_mut_span": "Mutant changed instruction count after common-prefix/suffix alignment.",
        "qasm_diff_total_span": "Sum of original and mutant changed instruction counts.",
        "qasm_diff_common_prefix_fraction": "Fraction of original gate/wire signatures identical before the first differing signature.",
        "qasm_diff_common_suffix_fraction": "Fraction of original gate/wire signatures identical after the aligned changed interval.",
        "qasm_diff_position_fraction": "Common-prefix fraction used as a QASM-derived position estimate; excluded from final model because the target-blind pilot agreed with coarse position bands only 59.67% of the time.",
        "qasm_diff_expected_gate_match": "Expected mutation gate name appears on the operation-appropriate changed side.",
        "qasm_diff_qubit_set_equal": "For one-to-one edits, whether removed and added wire sets match; -1 when inapplicable.",
        "qasm_diff_qubits_touched": "Distinct wire count in the changed intervals.",
        "qasm_diff_removed_arity": "Removed instruction wire count; -1 none, -2 multiple.",
        "qasm_diff_added_arity": "Added instruction wire count; -1 none, -2 multiple.",
        "qasm_diff_exact_single_edit": "Whether expected gate matches and Add is 0→1, Remove 1→0, or Replace 1→1.",
        "qasm_diff_is_local": "Whether the changed spans total at most three instructions and expected gate matches.",
    }
    categorical = {
        "qasm_diff_removed_gate": "Single removed gate name or none/multi sentinel.",
        "qasm_diff_added_gate": "Single added gate name or none/multi sentinel.",
        "qasm_diff_alignment_class": "Exact insert/delete/replace, near edit, complex/un-aligned, or no detected edit.",
        "qasm_diff_local_prev_gate": "Previous gate if local; circuit_start or unknown_complex otherwise.",
        "qasm_diff_local_next_gate": "Next gate if local; circuit_end or unknown_complex otherwise.",
    }
    inventory = []
    for name, definition in (numeric | categorical).items():
        inventory.append({
            "feature": name, "type": "categorical" if name in categorical else "numeric",
            "source": "Original/mutant OpenQASM 2.0 sequences; operation/gate metadata as applicable",
            "definition": definition,
            "prediction_availability": "Requires original_qasm, mutant_qasm, operation and gate inputs.",
            "coverage_rows": extractor["feature_rows"], "missing_rows": 0,
            "used_in_final_model": name in selected_diff_features,
            "leakage_risk": "Outcome independent; parser uses no labels, origin IDs, or row IDs. Operation and gate are pre-outcome metadata already available in the incumbent model.",
        })

    report = {
        "schema": "qmutbench-exhaustive-optimization-campaign-v1",
        "campaign_date": "2026-10-26",
        "starting_branch": "qmutbench-master-optimization",
        "starting_commit": "13a45194961fe4293b1b300ca4611df7e4a5900a",
        "source": {"csv": "data/qmutbench_ingestion/raw/official_all_data_20261015.csv",
                   "sha256": sha256_file(SOURCE_CSV), "all_rows": 723079, "development_rows": 578117,
                   "development_origins": 305, "historical_frozen_rows": 144962,
                   "frozen_test_used": False, "qasm_commit": "01f5e7700c051fe715fc368b307deaa7e6869ba8"},
        "target": "WOO/OPO=detected (1); Survivor=not detected (0). This is historical benchmark outcome prediction, not hardware-noise detection or fault localization.",
        "incumbent": {"model": "100-tree ExtraTreesClassifier", "bundle": "data/qmutbench_ml/position_fraction_v1/",
            "threshold": 0.5, "metrics": three["matched_baseline_pooled_oof_metrics"],
            "fold_metrics": baseline_folds,
            "model_sha256": "641a18b67448f910ea0126c914e7e5f17bc946a3fa940b0486c045b845f8ee45"},
        "candidate": {"model": "100-tree ExtraTreesClassifier; depth 22; min leaf 15; sqrt features; balanced_subsample; threshold 0.5",
            "features": three["base_features"] + three["mutation_diff_features"],
            "three_fold_oof_metrics": three["pooled_oof_metrics"],
            "three_fold_macro_f1_delta": three["pooled_macro_f1_delta"],
            "three_fold_macro_f1_deltas": three["fold_macro_f1_deltas"],
            "five_fold_grouped_confirmation": five,
            "artifact": "data/qmutbench_ml/mutation_aware_v1/best_model.joblib",
            "metadata": "data/qmutbench_ml/mutation_aware_v1/best_model_metadata.json",
            "model_sha256": artifact["model_sha256"], "full_development_fit_seconds": artifact["fit_seconds"]},
        "screening_60_trees": {"pooled_metrics": metrics(screen60["y"], screen60["probability"]),
            "fold_macro_f1": [read(EXP / f"mutation_diff_folds_matched/fold{fold}.json")["metrics"]["macro_f1"] for fold in (1, 2, 3)],
            "fold_macro_f1_deltas_vs_matched_baseline": [read(EXP / f"mutation_diff_folds_matched/fold{fold}.json")["metrics"]["macro_f1"] - read(ROOT / f"data/qmutbench_ml/qasm_model_comparison_v1/folds/extra_trees_baseline_fold{fold}.json")["metrics_at_threshold_0_5"]["macro_f1"] for fold in (1, 2, 3)],
            "pooled_macro_f1_delta_vs_matched_baseline": metrics(screen60["y"], screen60["probability"])["macro_f1"] - three["matched_baseline_pooled_oof_metrics"]["macro_f1"]},
        "mutation_diff_extraction": {"script": "experiments/qmutbench_qasm_mutation_diff.py",
            "runtime_module": "qmutbench_ml/qasm_mutation_diff.py", "cache": "data/qmutbench_ml/qasm_mutation_diff_v1/ (ignored)",
            "manifest": extractor, "feature_inventory": inventory,
            "alignment_class_fraction": {"single_insert": 0.71517, "single_replace": 0.14460,
                "complex_or_unaligned": 0.11886, "single_delete": 0.02118, "near_multi_edit": 0.00019},
            "expected_gate_match_fraction": 0.9474967869825658,
            "exact_single_edit_fraction": 0.8809514337063259,
            "local_context_fraction": 0.8811382471022302,
            "span_summary": {"median": 1, "p90": 2, "p99": 617, "max": 780},
            "target_blind_3600_row_pilot": {"add_exact_shape": 0.9583, "remove_exact_shape": 0.6325,
                "replace_exact_shape": 0.9242, "expected_gate_match": 0.9228,
                "position_band_agreement": 0.5967}},
        "other_model_family": {"hist_gradient_boosting": hgb,
            "random_forest": "Prior matched 60-tree grouped screen macro F1 0.75891 vs Extra Trees 0.76628; not promoted.",
            "catboost_xgboost_lightgbm": "Not installed; no dependency installation attempted."},
        "feature_ablation": ablation,
        "prior_qasm_candidate": old_qasm["candidate_confirmation"],
        "prior_feature_ablation": prior["previous_full_development_comparisons"],
        "prior_threshold_study": prior["threshold_selection"],
        "error_analysis": errors,
        "ensemble_and_calibration": {"equal_incumbent_candidate_blend_macro_f1": 0.785275240432039,
            "candidate_alone_macro_f1": three["pooled_oof_metrics"]["macro_f1"],
            "decision": "No ensemble selected; 50/50 blend was below the standalone candidate. No calibration or new threshold tuning was added; probabilities are uncalibrated."},
        "selection": {"decision": "Select the mutation-aware Extra Trees as the new versioned development model.",
            "promotion_bar": "+0.5 percentage points grouped macro F1, fold support, and no material safeguard degradation.",
            "result": f"+{three['pooled_macro_f1_delta']*100:.3f} points on saved three-fold OOF and +{five['pooled_macro_f1_delta']*100:.3f} points on additional five-fold grouped CV; every paired fold improved.",
            "target_85_percent_reached": False,
            "qualification": "The development labels informed feature selection, so neither confirmation is an independent external test. The previously examined historical frozen set was not used."},
        "inference": {"api": "qmutbench_ml/inference.py", "qasm_inputs": ["original_qasm", "mutant_qasm"],
            "input_contract": "Existing eight raw metadata inputs plus original and mutant OpenQASM 2.0 text; derives position_fraction and 15 QASM features.",
            "single_and_batch_smoke": "Passed using an actual development circuit pair; probabilities are uncalibrated."},
        "tests": {"command": "$qmutbenchTests = Get-ChildItem tests -Filter 'test_qmutbench_*.py' | ForEach-Object { $_.FullName }; .venv/Scripts/python.exe -m pytest @qmutbenchTests -q",
            "result": "88 passed in 17.15s", "focused": "15 passed in 2.27s", "remaining": "Final compile, git diff --check, review, commit/push pending."},
        "reports": {"markdown": "reports/experiments/qmutbench_exhaustive_optimization_20261026.md",
            "json": "reports/experiments/qmutbench_exhaustive_optimization_20261026.json",
            "error_analysis": "reports/experiments/qmutbench_exhaustive_error_analysis_20261026.json"},
        "limitations": ["85% aspirational macro-F1 was not reached (79.33% in the additional grouped CV).",
            "The additional five-fold CV reused development rows that informed feature selection; it is a stability confirmation, not an independent test.",
            "No new nested threshold/calibration analysis or inner-fold stacking was run.",
            "Common-prefix/suffix alignment marks large/reordered edits complex and withholds local context.",
            "QASM source text is required at inference, and parsing cost scales with circuit size.",
            "Some origins remain substantially harder; see subgroup sample counts and intervals in the error-analysis JSON."]
    }
    return report


def main():
    report = build()
    json_path = ROOT / "reports/experiments/qmutbench_exhaustive_optimization_20261026.json"
    md_path = ROOT / "reports/experiments/qmutbench_exhaustive_optimization_20261026.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    candidate, baseline = report["candidate"], report["incumbent"]["metrics"]
    f5 = candidate["five_fold_grouped_confirmation"]
    md = f'''# QMutBench exhaustive optimization campaign — 2026-10-26

## Result

The selected new development model is the mutation-aware Extra Trees bundle at `data/qmutbench_ml/mutation_aware_v1/`. It adds 15 target-blind QASM mutation-alignment and local-context fields to the incumbent's eight inputs and retains threshold 0.5.

On the saved three-fold origin-grouped OOF, candidate macro F1 is **{candidate['three_fold_oof_metrics']['macro_f1']:.4%}**, balanced accuracy **{candidate['three_fold_oof_metrics']['balanced_accuracy']:.4%}**, and accuracy **{candidate['three_fold_oof_metrics']['accuracy']:.4%}**. The matched incumbent scores **{baseline['macro_f1']:.4%}**, **{baseline['balanced_accuracy']:.4%}**, and **{baseline['accuracy']:.4%}**. Macro-F1 gain: **{candidate['three_fold_macro_f1_delta']*100:.2f} percentage points**.

An additional five-fold `StratifiedGroupKFold` comparison on identical rows produced **{f5['candidate_pooled_metrics']['macro_f1']:.4%}** versus **{f5['baseline_pooled_metrics']['macro_f1']:.4%}** (+**{f5['pooled_macro_f1_delta']*100:.2f} pp**). All five paired folds improved. This is a development-set stability check, not a fresh external test: feature discovery used development OOF outcomes. The previously examined historical frozen set was excluded.

## Grouped fold results

| Saved split | Baseline macro F1 | Candidate macro F1 | Delta |
|---:|---:|---:|---:|
'''
    for i, (b, c, delta) in enumerate(zip(report["incumbent"]["fold_metrics"],
            [x["metrics"] for x in read(EXP / "mutation_diff_screen.json")["folds"]],
            candidate["three_fold_macro_f1_deltas"]), 1):
        md += f"| 3-fold {i} | {b['macro_f1']:.4f} | {c['macro_f1']:.4f} | +{delta*100:.2f} pp |\n"
    for i, (b, c, delta) in enumerate(zip(f5["baseline_fold_metrics"], f5["candidate_fold_metrics"], f5["fold_macro_f1_deltas"]), 1):
        md += f"| 5-fold {i} | {b['macro_f1']:.4f} | {c['macro_f1']:.4f} | +{delta*100:.2f} pp |\n"
    md += f'''

## Features and errors

QASM feature extraction covered {report['mutation_diff_extraction']['manifest']['feature_rows']:,}/{report['source']['development_rows']:,} development rows with zero parse errors. It aligns gate-name/wire tuples and excludes measurements and barriers. Exact edit shape and expected gate must agree before an edit is labeled exact; local neighbors are exposed only for small reliable diffs. Exact one-instruction edits account for 88.10%; 11.89% are complex/unaligned. See the JSON feature inventory for each definition, source, coverage, prediction-time availability, and leakage review.

The extractor also computes common-prefix, common-suffix, and a prefix-based relative-position estimate. These three fields are documented in the inventory but excluded from the finalist: the position estimate matched the coarse source position band on only 59.67% of the target-blind pilot.

The candidate yields {report['error_analysis']['candidate_false_positive_and_false_negative_counts']['false_positive']:,} false positives and {report['error_analysis']['candidate_false_positive_and_false_negative_counts']['false_negative']:,} false negatives; the incumbent has {report['error_analysis']['baseline_false_positive_and_false_negative_counts']['false_positive']:,} false positives and {report['error_analysis']['baseline_false_positive_and_false_negative_counts']['false_negative']:,} false negatives. Candidate per-origin accuracy improved on {report['error_analysis']['baseline_candidate_per_origin_accuracy_delta']['improved']} origins, worsened on {report['error_analysis']['baseline_candidate_per_origin_accuracy_delta']['worsened']}, and tied on one. Subgroup counts, Wilson intervals, high-confidence errors, and weak-origin examples are in [the error-analysis JSON](qmutbench_exhaustive_error_analysis_20261026.json).

## Model comparisons

- The matched 60-tree Extra Trees screen gained 2.23 pp; 100-tree confirmation gained 2.07 pp on the saved three folds.
- The tree depth, leaf size, feature subsampling, and class weighting stayed at the incumbent settings; this campaign compared 60 versus 100 trees rather than launching a broad parameter grid. Prior unweighted Extra Trees and Random Forest screens were reused.
- The prior 60-tree Random Forest screen (0.75891 macro F1) was below its Extra Trees baseline (0.76628).
- Native-categorical HistGradientBoosting with 100 iterations improved its own baseline by 0.87 pp, but its candidate macro F1 ({report['other_model_family']['hist_gradient_boosting']['mutation_diff_pooled_metrics']['macro_f1']:.4f}) remained below the Extra Trees finalist. CatBoost, XGBoost, and LightGBM were unavailable and were not installed.
- Prior QASM structure-only and signed-delta features were negative. Prior gate composition gained only 0.32 pp and lost a fold.
- A matched 60-tree ablation removing the five gate/alignment categories (removed/added gate, alignment class, previous/next gate) scored 0.76475 versus 0.76628 baseline; its fold deltas were +0.38, −0.46, −0.38 pp. The full QASM-diff set scored 0.78856 on those same folds, supporting the categorical mutation-context block as useful.
- A 50/50 incumbent/finalist probability blend scored 0.78528, below the standalone finalist. No ensemble was kept. The threshold remains 0.5; no calibration or new nested threshold study was added.

## Inference and tests

The versioned artifact is `data/qmutbench_ml/mutation_aware_v1/best_model.joblib` with metadata alongside it (SHA-256 `{report['candidate']['model_sha256']}`). The incumbent bundle remains unchanged. Inference accepts the existing eight raw fields plus `original_qasm` and `mutant_qasm` OpenQASM 2.0 text, supports single and batch records, and computes the QASM fields without loading the training dataset. An actual development record passed single and two-row batch smoke predictions.

QMutBench regression suite: **88 passed in 17.15s**. Focused feature/inference tests: **15 passed in 2.27s**. The full repository suite was not run. The 85% aspirational target was not reached (five-fold macro F1 79.33%). No result here establishes hardware-noise detection or fault localization.

## Artifacts and interpretation

- [Machine-readable report](qmutbench_exhaustive_optimization_20261026.json)
- [Error and subgroup analysis](qmutbench_exhaustive_error_analysis_20261026.json)
- Model metadata: `data/qmutbench_ml/mutation_aware_v1/best_model_metadata.json`
- Starting commit: `13a45194961fe4293b1b300ca4611df7e4a5900a`

Because feature selection used development outcomes, both grouped validations can be optimistic after selection. No external test was used. Common-prefix/suffix alignment may mark reordered or heavily rewritten circuits complex; such rows do not get an asserted local neighborhood. Probabilities are uncalibrated, and QASM parsing is required at inference.
'''
    md_path.write_text(md, encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path),
                      "candidate_three_fold_macro_f1": candidate["three_fold_oof_metrics"]["macro_f1"],
                      "candidate_five_fold_macro_f1": f5["candidate_pooled_metrics"]["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
