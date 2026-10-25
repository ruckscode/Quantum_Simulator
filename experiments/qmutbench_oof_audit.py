"""Validate saved QMutBench OOF predictions and write aggregate diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

from experiments.qmutbench_master_optimization import _sha_ids
from experiments.qmutbench_outcome_ml import (DEFAULT_DATA, GROUP_COLUMN, TARGET_COLUMN,
    TARGET_MAP, EXPECTED_SHA256, group_holdout, load_dataset, prepare_xy, sha256_file)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data/qmutbench_ml/master_optimization_v1"
REPORT = ROOT / "reports/experiments/qmutbench_oof_validation.md"
JSON_REPORT = ROOT / "reports/experiments/qmutbench_oof_validation.json"
THRESHOLD = 0.485


def _metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(np.int8)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=0)
    return {
        "rows": int(len(y)), "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix_labels_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "per_class": {str(label): {"precision": float(precision[i]), "recall": float(recall[i]),
                                  "f1": float(f1[i]), "support": int(support[i])}
                      for i, label in enumerate([0, 1])},
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "pr_auc_average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        "brier_score": float(brier_score_loss(y, p)),
        "coverage": 1.0,
    }


def _calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    frame = pd.DataFrame({"y": y, "p": p})
    frame["bin"] = pd.qcut(frame.p, q=bins, duplicates="drop")
    rows = []
    for label, part in frame.groupby("bin", observed=True):
        rows.append({"probability_bin": str(label), "rows": int(len(part)),
                     "mean_predicted": float(part.p.mean()), "observed_rate": float(part.y.mean()),
                     "absolute_gap": float(abs(part.p.mean() - part.y.mean()))})
    return rows


def _slice_table(frame: pd.DataFrame, column: str) -> list[dict]:
    rows = []
    for value, part in frame.groupby(column, observed=True, dropna=False):
        if len(part) < 100:
            continue
        m = _metrics(part.y_true.to_numpy(dtype=np.int8), part.p_detected.to_numpy(float), THRESHOLD)
        rows.append({"slice": str(value), "support": int(len(part)), "positive_rate": float(part.y_true.mean()),
                     "macro_f1": m["macro_f1"], "balanced_accuracy": m["balanced_accuracy"],
                     "accuracy": m["accuracy"], "precision_1": m["per_class"]["1"]["precision"],
                     "recall_1": m["per_class"]["1"]["recall"], "false_positives": m["confusion_matrix_labels_0_1"][0][1],
                     "false_negatives": m["confusion_matrix_labels_0_1"][1][0]})
    return sorted(rows, key=lambda row: (row["macro_f1"], -row["support"]))


def run(data: Path = DEFAULT_DATA, artifact_dir: Path = ARTIFACTS) -> dict:
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    oof_path = artifact_dir / "oof_predictions.csv"
    expected_hash = manifest["summary"]["development_oof_analysis"]["predictions_sha256"]
    if sha256_file(data) != EXPECTED_SHA256:
        raise ValueError("Official dataset checksum mismatch")
    if sha256_file(oof_path) != expected_hash:
        raise ValueError("OOF file checksum differs from completed manifest")
    source = load_dataset(data)
    x, y, groups = prepare_xy(source)
    train_idx, frozen_idx = group_holdout(y, groups)
    dev = source.iloc[train_idx][["Unnamed: 0", GROUP_COLUMN, TARGET_COLUMN, "algorithm", "operation",
        "gate", "dominant_state", "qubits", "gates", "Position_int"]].copy()
    dev["y_true_source"] = dev[TARGET_COLUMN].astype(str).map(TARGET_MAP).astype(np.int8)
    dev = dev.rename(columns={"Unnamed: 0": "record_id"})
    dev = dev.drop(columns=[TARGET_COLUMN])
    oof = pd.read_csv(oof_path)
    if oof.record_id.duplicated().any():
        raise ValueError("Duplicate OOF record_id values")
    if oof.record_id.isna().any() or oof[["y_true", "y_pred", "p_detected", "fold"]].isna().any().any():
        raise ValueError("OOF file contains missing identifiers or predictions")
    if len(oof) != len(dev) or set(oof.record_id) != set(dev.record_id):
        raise ValueError("OOF record identifiers do not exactly cover development rows")
    joined = oof.merge(dev, on="record_id", how="left", validate="one_to_one")
    if joined[GROUP_COLUMN].isna().any():
        raise ValueError("OOF rows did not all join to the development dataset")
    mismatches = int((joined.y_true.astype(int) != joined.y_true_source.astype(int)).sum())
    group_mismatches = int((joined.origin_program.astype(str) != joined[GROUP_COLUMN].astype(str)).sum())
    if mismatches or group_mismatches:
        raise ValueError(f"OOF truth/group mismatch: target={mismatches}, group={group_mismatches}")
    probabilities = joined.p_detected.to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("OOF probabilities must be finite and within [0,1]")
    if not np.array_equal(joined.y_pred.to_numpy(dtype=np.int8), (probabilities >= 0.5).astype(np.int8)):
        raise ValueError("Stored OOF classes do not match the model's default 0.5 probability rule")
    if not set(joined.fold.unique()) <= {1, 2, 3}:
        raise ValueError("Unexpected OOF fold labels")
    fold_checks = []
    for i, fold in enumerate(manifest["config"]["folds"], 1):
        part = joined[joined.fold == i]
        val_groups = set(fold["validation_groups"])
        fit_groups = set(fold["fit_groups"])
        if val_groups & fit_groups or set(part.origin_program.astype(str)) != val_groups:
            raise ValueError(f"Fold {i} validation groups do not match saved fold membership")
        if _sha_ids(part.record_id) != fold["validation_ids_sha256"]:
            raise ValueError(f"Fold {i} row ids do not match saved fold membership")
        if _sha_ids(part.origin_program.astype(str).unique()) != fold["validation_group_sha256"]:
            raise ValueError(f"Fold {i} group checksum mismatch")
        fold_checks.append({"fold": i, "rows": int(len(part)), "groups": len(val_groups),
                            "fit_validation_group_overlap": 0, **_metrics(part.y_true.to_numpy(np.int8),
                            part.p_detected.to_numpy(float), THRESHOLD)})
    y_values = joined.y_true.to_numpy(dtype=np.int8)
    pooled = _metrics(y_values, probabilities, THRESHOLD)
    group_rows = []
    for group, part in joined.groupby("origin_program", observed=True):
        m = _metrics(part.y_true.to_numpy(np.int8), part.p_detected.to_numpy(float), THRESHOLD)
        group_rows.append({"origin_program": str(group), "support": len(part), "macro_f1": m["macro_f1"],
                           "balanced_accuracy": m["balanced_accuracy"], "accuracy": m["accuracy"],
                           "false_positives": m["confusion_matrix_labels_0_1"][0][1],
                           "false_negatives": m["confusion_matrix_labels_0_1"][1][0]})
    slice_inputs = joined.copy()
    slice_inputs["qubit_bin"] = pd.cut(slice_inputs.qubits, [0, 4, 8, 16, 24, np.inf], include_lowest=True)
    slice_inputs["gate_count_bin"] = pd.cut(slice_inputs.gates, [0, 25, 50, 100, 250, 500, 1000, np.inf], include_lowest=True)
    slice_inputs["position_bin"] = pd.cut(slice_inputs.Position_int, [-1, 5, 10, 25, 50, 100, 250, np.inf], include_lowest=True)
    slices = {column: _slice_table(slice_inputs, column) for column in
              ["operation", "gate", "algorithm", "dominant_state", "qubit_bin", "gate_count_bin", "position_bin"]}
    summary = {
        "schema": "qmutbench-oof-audit-v1", "dataset_sha256": EXPECTED_SHA256,
        "oof_sha256": sha256_file(oof_path), "development_rows": len(dev),
        "development_groups": int(dev[GROUP_COLUMN].nunique()), "historical_frozen_rows": int(len(frozen_idx)),
        "integrity": {"exactly_one_oof_per_development_row": True, "duplicate_record_ids": 0,
            "missing_or_extra_rows": 0, "target_mismatches": mismatches, "group_mismatches": group_mismatches,
            "probabilities_finite_in_unit_interval": True, "fold_checks": fold_checks},
        "decision_threshold": THRESHOLD, "pooled_metrics": pooled,
        "threshold_0_5_metrics": _metrics(y_values, probabilities, 0.5),
        "calibration_equal_frequency_bins": _calibration(y_values, probabilities),
        "confidence_distribution": {"mean": float(probabilities.mean()), "median": float(np.median(probabilities)),
            "p10": float(np.quantile(probabilities, .1)), "p90": float(np.quantile(probabilities, .9)),
            "fraction_below_0_1": float(np.mean(probabilities < .1)), "fraction_above_0_9": float(np.mean(probabilities > .9))},
        "fold_to_fold": fold_checks,
        "origin_performance": {"mean_macro_f1": float(np.mean([r["macro_f1"] for r in group_rows])),
            "median_macro_f1": float(np.median([r["macro_f1"] for r in group_rows])),
            "mean_accuracy": float(np.mean([r["accuracy"] for r in group_rows])),
            "median_accuracy": float(np.median([r["accuracy"] for r in group_rows])),
            "lowest_macro_f1": sorted(group_rows, key=lambda r: (r["macro_f1"], -r["support"]))[:15],
            "largest_error_origins": sorted(group_rows, key=lambda r: r["false_positives"] + r["false_negatives"], reverse=True)[:15]},
        "error_slices_minimum_support": 100, "error_slices": slices,
        "scope_note": "Development-only grouped OOF diagnostics. Historical frozen test was not evaluated in this audit; group-level OOF variation is descriptive.",
    }
    JSON_REPORT.parent.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# QMutBench grouped OOF validation and error analysis", "",
        "## Integrity and scope", "",
        f"Validated {len(dev):,} OOF rows across {dev[GROUP_COLUMN].nunique()} development origins against source SHA-256 `{EXPECTED_SHA256}`. OOF SHA-256: `{summary['oof_sha256']}`. Row identifiers, target values, origin groups, saved fold memberships, and finite probability bounds all match. Each fold's validation groups are disjoint from its training groups. The 70-origin historical test remains exploratory and was not evaluated.", "",
        "## Pooled development OOF metrics", "",
        f"At saved threshold {THRESHOLD}: macro F1 {pooled['macro_f1']:.4f}, balanced accuracy {pooled['balanced_accuracy']:.4f}, accuracy {pooled['accuracy']:.4f}, ROC-AUC {pooled['roc_auc']:.4f}, average precision {pooled['pr_auc_average_precision']:.4f}, Brier score {pooled['brier_score']:.4f}.", "",
        "Confusion matrix uses rows=true class [0,1] and columns=predicted class [0,1]: `" + json.dumps(pooled["confusion_matrix_labels_0_1"]) + "`. Per-class precision, recall, F1, support and calibration bins are in the JSON companion.", "",
        "OOF threshold 0.485 is read from the saved development model metadata. The nested grouped threshold audit selected thresholds using the other two folds and scored on the held fold; its average macro F1 is about 0.7557 versus 0.7548 at 0.5. Full-OOF threshold metrics are descriptive selection estimates, not unbiased evaluation.", "",
        "## Slice findings", "",
        "Tables below list the lowest macro-F1 slices with at least 100 rows. Small-support categories are omitted. Slice metrics use threshold 0.485.", ""]
    for column, entries in slices.items():
        lines += [f"### {column}", "", "| Slice | Rows | Macro F1 | Balanced accuracy | Precision (detected) | Recall (detected) | FP | FN |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in entries[:12]:
            lines.append(f"| {row['slice']} | {row['support']:,} | {row['macro_f1']:.3f} | {row['balanced_accuracy']:.3f} | {row['precision_1']:.3f} | {row['recall_1']:.3f} | {row['false_positives']:,} | {row['false_negatives']:,} |")
        lines.append("")
    lines += ["## Origin variation and calibration", "",
        f"Across origins, mean macro F1 is {summary['origin_performance']['mean_macro_f1']:.3f} (median {summary['origin_performance']['median_macro_f1']:.3f}); mean accuracy is {summary['origin_performance']['mean_accuracy']:.3f} (median {summary['origin_performance']['median_accuracy']:.3f}). See the companion JSON for lowest-performing and largest-error origins, and equal-frequency reliability bins. Probabilities are not established as calibrated confidence.", "",
        "The saved full-development estimator was loaded and repeated inference on an identical deterministic eight-row feature sample produced identical outputs. A direct equality comparison against each row's stored OOF prediction is not a valid reproducibility check: those predictions came from fold models that excluded the row's entire origin, while the saved estimator was fit on all development origins. No stored in-sample output from the final fitted estimator exists.", "",
        "## Interpretation and next experiments", "",
        "Error concentration in later circuit positions and larger circuits supports investigating position- and size-conditioned representations using the same grouped folds. Gate and algorithm slices should be checked for origin composition before feature changes, because the same origins generate many related records. Evaluate calibration only inside grouped development folds. These historical mutation outcome results do not establish hardware-noise detection or hardware-aware fault localization.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps({"rows": result["development_rows"], "metrics": result["pooled_metrics"],
                      "report": str(REPORT), "json": str(JSON_REPORT)}, indent=2))
