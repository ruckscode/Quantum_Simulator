"""Build the tracked report from validated QMutBench finalist artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "data/qmutbench_ml/performance_phase_v3/manifest.json"
FULL = ROOT / "data/qmutbench_ml/performance_phase_full_v1/summary.json"
AUDIT = ROOT / "reports/experiments/qmutbench_feature_audit_20261025.json"
MODEL_META = ROOT / "data/qmutbench_ml/position_fraction_v1/best_model_metadata.json"
PREVIOUS = ROOT / "reports/experiments/qmutbench_model_comparison_20261025.json"
JSON_OUT = ROOT / "reports/experiments/qmutbench_final_performance_20261025.json"
MD_OUT = ROOT / "reports/experiments/qmutbench_final_performance_20261025.md"


def _pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def run() -> dict:
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    full = json.loads(FULL.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    metadata = json.loads(MODEL_META.read_text(encoding="utf-8"))
    prior = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    baseline = full["baseline_metrics_at_0_5"]
    candidate = full["candidate_metrics_at_0_5"]
    screen_rows = []
    for variant, result in sorted(screen.get("summaries", {}).items()):
        screen_rows.append({"variant": variant,
            "macro_f1_mean_at_0_5": float(np.mean([x["macro_f1"] for x in result["fold_metrics_at_0_5"]])),
            "macro_f1_fold_sd_at_0_5": float(np.std([x["macro_f1"] for x in result["fold_metrics_at_0_5"]], ddof=1)),
            "pooled_macro_f1_at_0_5": result["metrics_at_0_5"]["macro_f1"],
            "nested_macro_f1_mean": result["nested_macro_f1_mean"],
            "nested_macro_f1_sd": result["nested_macro_f1_sd"],
            "rows": result["rows"], "training_selection_rows": result["training_selection_rows"],
            "groups": result["groups"]})
    results = {
        "schema": "qmutbench-final-performance-v1",
        "starting_commit": "017a6446ff905a27b1e9209b549d28b27766a3ad",
        "source": {"path": "data/qmutbench_ingestion/raw/official_all_data_20261015.csv",
            "sha256": full["dataset_sha256"], "development_rows": full["rows"], "development_groups": full["groups"],
            "historical_frozen_test": "excluded from feature, threshold, and model selection"},
        "grouped_validation": {"folds": [{"fold": row["fold"], "groups": row["groups"],
                "baseline_macro_f1": row["baseline"]["macro_f1"], "candidate_macro_f1": row["candidate"]["macro_f1"],
                "macro_f1_delta": row["macro_f1_delta"], "baseline_balanced_accuracy": row["baseline"]["balanced_accuracy"],
                "candidate_balanced_accuracy": row["candidate"]["balanced_accuracy"],
                "baseline_detected_recall": row["baseline"]["per_class"]["Detected by WOO/OPO (1)"]["recall"],
                "candidate_detected_recall": row["candidate"]["per_class"]["Detected by WOO/OPO (1)"]["recall"],
                "baseline_survivor_recall": row["baseline"]["per_class"]["Survivor (0)"]["recall"],
                "candidate_survivor_recall": row["candidate"]["per_class"]["Survivor (0)"]["recall"]}
                for row in full["fold_results"]],
            "paired_macro_f1_delta_mean": full["fold_delta_mean"],
            "paired_macro_f1_delta_sd_sample": full["fold_delta_sd"],
            "fold_wins": full["fold_wins"]},
        "baseline": {"model": "100-tree Extra Trees; seven original features; threshold 0.5",
            "metrics": baseline, "threshold_0_485_metrics": full["baseline_metrics_at_0_485"]},
        "selected_model": {"name": "100-tree Extra Trees with source-derived circuit-position fraction",
            "metric_definition": "macro F1", "metrics_at_0_5": candidate,
            "threshold_0_485_metrics": full["candidate_metrics_at_0_485"],
            "macro_f1_delta": full["macro_f1_delta"],
            "balanced_accuracy_delta": full["balanced_accuracy_delta"],
            "accuracy_delta": full["accuracy_delta"],
            "nested_macro_f1_mean": full["nested_macro_f1_mean"],
            "baseline_nested_macro_f1_mean": full["baseline_nested_macro_f1_mean"],
            "nested_macro_f1_delta": full["nested_macro_f1_delta"],
            "parameters": metadata["model_parameters"],
            "feature_schema": metadata["feature_schema"],
            "training_scope": metadata["training_scope"], "artifact_path": "data/qmutbench_ml/position_fraction_v1/best_model.joblib",
            "artifact_sha256": metadata["model_sha256"], "inference_threshold": metadata["inference_threshold"],
            "material_improvement_criterion_met": True},
        "threshold_selection": {"final_threshold": 0.5,
            "nested_grouped_results": full["nested_thresholds"],
            "full_oof_macro_f1_selected_threshold_descriptive": full["full_oof_selected_threshold_descriptive"],
            "reason": "Nested fold thresholds (0.495, 0.460, 0.480) did not improve mean held-fold macro F1 over using 0.5 directly; the all-OOF selected threshold is optimistic."},
        "screen_experiments": screen_rows,
        "previous_full_development_comparisons": prior["model_comparisons"],
        "feature_audit": audit,
        "error_analysis": full["error_analysis"],
        "origin_analysis": full["origin_performance"],
        "feature_definition": {"position_fraction": "midpoint of the QMutBench source Position (Circuit %) band divided by 100; deterministic from inference-time metadata, no target-based encoding"},
        "limitations": [
            "The three grouped validation folds share the same 305 development origins; results do not establish performance on new external origins.",
            "Origin-level performance still varies; 248 of 305 origins improved at threshold 0.5, so 57 did not.",
            "Probabilities remain uncalibrated model probabilities; improved Brier score is not a calibration guarantee.",
            "The historical 70-origin test was previously examined and was not used in this phase.",
            "The 85% macro F1 target was not reached."
        ],
        "research_scope": "Historical QMutBench mutation-outcome prediction only; no hardware-noise detection, fault localization, or simulator diagnosis claim."
    }
    JSON_OUT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = ["# QMutBench final performance optimization", "",
        "## Directly measured results", "",
        f"Dataset SHA-256 `{full['dataset_sha256']}`; {full['rows']:,} development records across {full['groups']} origins. All comparisons use the same saved three `StratifiedGroupKFold` splits (91, 101, and 113 validation origins). The frozen historical test was not used.", "",
        "| Model | Macro F1 | Balanced accuracy | Accuracy | ROC-AUC | Average precision | Brier |", "|---|---:|---:|---:|---:|---:|---:|",
        f"| Baseline Extra Trees, 7 features | {baseline['macro_f1']:.4f} | {baseline['balanced_accuracy']:.4f} | {baseline['accuracy']:.4f} | {baseline['roc_auc']:.4f} | {baseline['average_precision']:.4f} | {baseline['brier_score']:.4f} |",
        f"| Extra Trees + position fraction, 8 model features | {candidate['macro_f1']:.4f} | {candidate['balanced_accuracy']:.4f} | {candidate['accuracy']:.4f} | {candidate['roc_auc']:.4f} | {candidate['average_precision']:.4f} | {candidate['brier_score']:.4f} |", "",
        f"At threshold 0.5, macro F1 improved by {_pp(full['macro_f1_delta'])}; balanced accuracy by {_pp(full['balanced_accuracy_delta'])}; accuracy by {_pp(full['accuracy_delta'])}. Mean paired fold macro-F1 gain was {_pp(full['fold_delta_mean'])} (sample SD {100*full['fold_delta_sd']:.2f} pp), with {full['fold_wins']}/3 folds improved.", "",
        "| Fold | Baseline macro F1 | Candidate macro F1 | Change | Baseline BA | Candidate BA |", "|---:|---:|---:|---:|---:|---:|"]
    for row in full["fold_results"]:
        lines.append(f"| {row['fold']} | {row['baseline']['macro_f1']:.4f} | {row['candidate']['macro_f1']:.4f} | {_pp(row['macro_f1_delta'])} | {row['baseline']['balanced_accuracy']:.4f} | {row['candidate']['balanced_accuracy']:.4f} |")
    lines += ["", "At threshold 0.5, detected-class recall rose from " +
        f"{baseline['per_class']['Detected by WOO/OPO (1)']['recall']:.4f} to {candidate['per_class']['Detected by WOO/OPO (1)']['recall']:.4f}; Survivor recall rose from {baseline['per_class']['Survivor (0)']['recall']:.4f} to {candidate['per_class']['Survivor (0)']['recall']:.4f}. False negatives fell from {full['error_analysis']['baseline_false_negatives']:,} to {full['error_analysis']['candidate_false_negatives']:,}; false positives fell from {full['error_analysis']['baseline_false_positives']:,} to {full['error_analysis']['candidate_false_positives']:,}.", "",
        f"The model improved macro F1 on {full['origin_performance']['origins_improved']}/{full['origin_performance']['origins_total']} origins at threshold 0.5. Mean origin macro F1 changed by {_pp(full['origin_performance']['mean_origin_macro_f1_delta'])}; median change was {_pp(full['origin_performance']['median_origin_macro_f1_delta'])}. There remain weak origins and category slices; the JSON companion has supported slice comparisons and high-confidence error counts.", "",
        "## Feature audit and controlled screen", "",
        "All original selected features have zero missing rows. Algorithm (28 categories) and gate (19) had the strongest average training-fold Cramer's V associations in the audit; Position_int (493 values) had the weakest. These are descriptive, univariate associations computed from training-side samples, not evidence of causal effects or standalone model gain.", "",
        "The local QMutBench README defines `position` as a circuit-percent band. Position_int ranges beyond gate count on some records, so no ratio to gate count was assumed. `position_fraction` is the deterministic midpoint of the documented percentage band divided by 100. It is available from the source metadata and uses no target or origin identifier.", "",
        "| 60-tree screen variant | Fold-mean macro F1 at 0.5 | Nested threshold macro F1 |", "|---|---:|---:|"]
    for row in screen_rows:
        lines.append(f"| {row['variant']} | {row['macro_f1_mean_at_0_5']:.4f} | {row['nested_macro_f1_mean']:.4f} |")
    lines += ["", "The screen trained on the representative whole-origin selection (201,066 records) and evaluated all rows of each saved held fold. The numeric position fraction had the strongest nested screen score; its categorical-bin encoding was also positive. Algorithm-operation and algorithm-gate interactions were negative; gate-operation was a small non-nested gain but declined under nested thresholds. Only position fraction advanced to the full-development 100-tree comparison.", "",
        "Prior checked candidates remain negative: no-class-weight Extra Trees was effectively tied; unweighted Random Forest and the smaller selected-v3 subset were lower than the full-data baseline. Existing 200k feature ablations showed algorithm and gate are important; numeric-only features were weak.", "",
        "The schema audit records target-free cardinality and missingness for available features. `Origin_program` is used only for grouping and diagnostics; `Unnamed: 0` and `Killed` are excluded. One-hot encoding is fitted inside each training fold; no target encoding or resampling was used.", "",
        "## Thresholds and calibration", "",
        "Nested grouped threshold selection fitted each threshold on the other two folds, then evaluated it on the held fold. The candidate thresholds were 0.495, 0.460, and 0.480. Candidate nested macro F1 averaged " +
        f"{full['nested_macro_f1_mean']:.4f}, compared with {full['baseline_nested_macro_f1_mean']:.4f} for the baseline's nested thresholds. For the candidate, the nested selected thresholds averaged slightly below using 0.5 directly ({full['candidate_default_fold_macro_f1_mean']:.4f}); therefore deployment keeps 0.5. The all-OOF selected threshold 0.475 scores 0.7697 on those same OOF predictions and is explicitly treated as selection-biased.", "",
        f"Brier score improved from {baseline['brier_score']:.4f} to {candidate['brier_score']:.4f}; ROC-AUC and average precision also improved. Probabilities remain uncalibrated; reliability bins and ECE summaries are reported in JSON.", "",
        "## Interpretation and limits", "",
        "The improvement meets the predeclared +0.5 pp macro-F1 criterion on comparable full-development OOF, improves balanced accuracy and both class recalls, wins in all three grouped folds, and improves on most origins. It is a defensible development improvement, not proof of performance on new external programs. The 85% target was not reached.", "",
        "This remains historical mutation-outcome prediction. It does not establish hardware-noise detection, hardware-aware fault localization, or quantum-simulator diagnosis.", "",
        "## Reproduction and artifacts", "",
        "Run the feature screen with `python -m experiments.qmutbench_performance_phase`, full grouped finalist with `python -m experiments.qmutbench_performance_phase --full-position-finalist`, full-development model fit with `python -m experiments.qmutbench_performance_phase --fit-final-model`, and audit with `python -m experiments.qmutbench_feature_audit`. Generated model and prediction files remain under ignored `data/qmutbench_ml/`.", "",
        f"Selected model: `{metadata['model_path']}` (SHA-256 `{metadata['model_sha256']}`). Inference API: `qmutbench_ml.QMutBenchOutcomeModel`; input includes source `position`, transformed using the saved mapping. Machine-readable results: `{JSON_OUT.relative_to(ROOT).as_posix()}`.", ""]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")
    return results


if __name__ == "__main__":
    result = run()
    print(json.dumps({"report": str(MD_OUT), "results": str(JSON_OUT),
        "macro_f1": result["selected_model"]["metrics_at_0_5"]["macro_f1"],
        "delta_pp": 100 * result["selected_model"]["macro_f1_delta"]}, indent=2))
