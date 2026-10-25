"""Build auditable discovery and performance reports for QMutBench QASM features."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from experiments.qmutbench_feature_audit import _cramers_v
from experiments.qmutbench_qasm_features import (BASELINE_OOF, COUNT_GATES, DEFAULT_OUTPUT,
    EXPECTED_DATA_SHA, EXPECTED_REPO_COMMIT, FEATURE_COLUMNS, OFFICIAL_CHECKOUT, ROOT,
    SOURCE_CSV, sha256_file)
from experiments.qmutbench_qasm_model_comparison import (COMPOSITION_FEATURES,
    DELTA_COMPOSITION_FEATURES, OUT as COMPARISON_V3, PREDICTIONS, STRUCTURE_FEATURES)

REPORT_ROOT = ROOT / "reports/experiments"
DISCOVERY_JSON = REPORT_ROOT / "qmutbench_qasm_feature_discovery_20261025.json"
RESULTS_JSON = REPORT_ROOT / "qmutbench_qasm_model_results_20261025.json"
REPORT_MD = REPORT_ROOT / "qmutbench_qasm_model_results_20261025.md"
SEED = 20261025
AUDIT_SAMPLE_ROWS = 50000
ORIGINAL_COMPOSITION = [f"qasm_orig_{name}_fraction" for name in COUNT_GATES]

DEFINITIONS = {
    "qasm_mut_gate_count": "Count of parsed non-measurement, non-barrier gate instructions in the linked mutated QASM.",
    "qasm_mut_depth": "ASAP per-qubit gate-layer depth on linked mutated QASM; barriers synchronize their wires; measurements excluded; no gate decomposition.",
    "qasm_mut_depth_per_qubit": "Mutated QASM ASAP layer depth divided by declared QASM qubit count.",
    "qasm_mut_single_qubit_count": "Count of parsed one-qubit gate instructions in the mutated QASM.",
    "qasm_mut_two_qubit_count": "Count of parsed two-qubit gate instructions in the mutated QASM.",
    "qasm_mut_three_plus_count": "Count of parsed gates with three or more operands in the mutated QASM.",
    "qasm_mut_two_plus_fraction": "Fraction of counted mutated QASM gates with arity two or greater.",
    "qasm_mut_distinct_gate_types": "Number of distinct parsed gate instruction names in the mutated QASM.",
    "qasm_orig_gate_count": "Count of parsed non-measurement, non-barrier gate instructions in the linked original QASM.",
    "qasm_orig_depth": "ASAP per-qubit gate-layer depth on linked original QASM; barriers synchronize their wires; measurements excluded; no gate decomposition.",
    "qasm_orig_depth_per_qubit": "Original QASM ASAP layer depth divided by declared QASM qubit count.",
    "qasm_orig_single_qubit_count": "Count of parsed one-qubit gate instructions in the original QASM.",
    "qasm_orig_two_qubit_count": "Count of parsed two-qubit gate instructions in the original QASM.",
    "qasm_orig_three_plus_count": "Count of parsed gates with three or more operands in the original QASM.",
    "qasm_orig_two_plus_fraction": "Fraction of counted original QASM gates with arity two or greater.",
    "qasm_orig_distinct_gate_types": "Number of distinct parsed gate instruction names in the original QASM.",
    "qasm_delta_gate_count": "Mutated minus original parsed gate-instruction count.",
    "qasm_delta_depth": "Mutated minus original ASAP per-qubit gate-layer depth.",
    "qasm_delta_two_qubit_fraction": "Mutated minus original fraction of gates with at least two operands.",
    "source_gate_count_match": "Diagnostic equality check of parsed mutated gate count against the source CSV gates field; not a model input.",
}
for _gate in COUNT_GATES:
    DEFINITIONS[f"qasm_mut_{_gate}_fraction"] = f"Fraction of mutated QASM counted gate instructions whose parsed name is {_gate}."
    DEFINITIONS[f"qasm_orig_{_gate}_fraction"] = f"Fraction of original QASM counted gate instructions whose parsed name is {_gate}."
    DEFINITIONS[f"qasm_delta_{_gate}_fraction"] = f"Mutated minus original fraction of counted QASM instructions named {_gate}."


def _cramers_numeric(values: pd.Series, target: pd.Series) -> float:
    binned = pd.qcut(values, q=10, duplicates="drop")
    return _cramers_v(binned.astype(str), target.astype(str))


def _load_oof() -> pd.DataFrame:
    parts = [np.load(PREDICTIONS / f"position_fraction_fold{i}.npz", allow_pickle=False)
             for i in (1, 2, 3)]
    frame = pd.DataFrame({name: np.concatenate([p[name] for p in parts])
                          for name in ("record_id", "y", "origin_program", "fold", "probability")})
    if len(frame) != 578117 or frame.record_id.duplicated().any():
        raise ValueError("The saved development OOF row contract failed")
    if frame.groupby("origin_program").fold.nunique().max() != 1:
        raise ValueError("An origin program appears in multiple saved folds")
    return frame


def create_discovery_manifest() -> dict[str, Any]:
    oof = _load_oof()
    tracked = subprocess.run(["git", "-C", str(OFFICIAL_CHECKOUT), "ls-files", "-z", "--",
                              "data/Origin_programs", "data/Mutated_programs"],
                             check=True, capture_output=True).stdout.split(b"\0")
    tracked_qasm = [Path(p.decode("utf-8").replace("\\", "/")) for p in tracked if p.endswith(b".qasm")]
    origin_files = [p for p in tracked_qasm if p.parts[:2] == ("data", "Origin_programs")]
    mutant_files = [p for p in tracked_qasm if p.parts[:2] == ("data", "Mutated_programs")]
    if len(origin_files) != 375 or len(mutant_files) != 723079:
        raise ValueError("Official QASM source tree counts changed from the verified checkout")

    source_columns = ["Unnamed: 0", "algorithm", "qubits", "operation", "gate", "Position_int"]
    source = pd.read_csv(SOURCE_CSV, usecols=source_columns)[source_columns]
    if source["Unnamed: 0"].duplicated().any():
        raise ValueError("Official source record IDs are not unique")
    dev_ids = set(oof.record_id.astype(int))
    dev_source = source[source["Unnamed: 0"].isin(dev_ids)]
    expected_files = set()
    originals = set()
    for row in dev_source.itertuples(index=False, name=None):
        rid, alg, qubits, operation, gate, pos = row
        original = f"data/Origin_programs/{alg}_{int(qubits)}_qubits.qasm"
        if operation == "Add":
            name = f"AddGate_{gate}_inGap_{int(pos)}_.qasm"
        elif operation == "Remove":
            name = f"RemoveGate_{int(pos)}_.qasm"
        elif operation == "Replace":
            name = f"ReplaceGate_{gate}_inPositionOfGate_{int(pos)}.qasm"
        else:
            raise ValueError(f"Unsupported operation in record {rid}: {operation}")
        originals.add(original)
        expected_files.add(f"data/Mutated_programs/Mutants_{alg}_{int(qubits)}_qubits/{name}")
    tracked_names = {p.as_posix() for p in tracked_qasm}
    missing_files = sorted(expected_files - tracked_names)
    if len(expected_files) != 578117 or missing_files:
        raise ValueError(f"Row-to-QASM linkage failed: {len(expected_files)} unique mutant paths, {len(missing_files)} missing")
    extraction = json.loads((DEFAULT_OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    if extraction.get("status") != "complete" or extraction.get("feature_rows") != len(oof):
        raise ValueError("QASM feature extraction is not complete for all development records")

    feature_metadata = {}
    feature_names = [name for name in DEFINITIONS if name != "source_gate_count_match"]
    fold_samples = {name: [] for name in feature_names}
    rng_by_fold = {fold: np.random.default_rng(SEED + fold) for fold in (1, 2, 3)}
    sample_ids: dict[int, set[int]] = {}
    for fold in (1, 2, 3):
        training_ids = oof.loc[oof.fold.ne(fold), "record_id"].to_numpy(np.int64)
        n = min(AUDIT_SAMPLE_ROWS, len(training_ids))
        sample_ids[fold] = set(rng_by_fold[fold].choice(training_ids, size=n, replace=False).tolist())
    sample_targets = oof[["record_id", "y", "fold"]]
    seen: dict[str, list[pd.Series]] = {name: [] for name in feature_names}
    association_rows: dict[int, list[pd.DataFrame]] = {f: [] for f in (1, 2, 3)}
    for path in sorted(DEFAULT_OUTPUT.glob("features_*.csv.gz")):
        part = pd.read_csv(path)
        numeric_cols = [name for name in part.columns if name.startswith("qasm_")]
        for name in numeric_cols:
            if name in seen:
                seen[name].append(part[name])
        for gate in COUNT_GATES:
            delta_name = f"qasm_delta_{gate}_fraction"
            if delta_name in seen:
                seen[delta_name].append(part[f"qasm_mut_{gate}_fraction"] - part[f"qasm_orig_{gate}_fraction"])
        selected = set().union(*sample_ids.values())
        rows = part[part.record_id.isin(selected)]
        if not rows.empty:
            rows = rows.merge(sample_targets, on="record_id", how="inner", validate="one_to_one")
            for fold in (1, 2, 3):
                s = rows[rows.record_id.isin(sample_ids[fold])]
                if s.empty:
                    continue
                for gate in COUNT_GATES:
                    s[f"qasm_delta_{gate}_fraction"] = s[f"qasm_mut_{gate}_fraction"] - s[f"qasm_orig_{gate}_fraction"]
                association_rows[fold].append(s)

    for fold in (1, 2, 3):
        sample = pd.concat(association_rows[fold], ignore_index=True)
        target = sample.y
        for name in feature_names:
            if name not in sample:
                continue
            values = sample[name]
            try:
                association = _cramers_numeric(values, target) if values.nunique() > 10 else _cramers_v(values, target)
            except ValueError:
                association = 0.0
            fold_samples[name].append(float(association))

    for name in feature_names:
        pieces = seen.get(name, [])
        if not pieces and name not in DELTA_COMPOSITION_FEATURES:
            continue
        values = pd.concat(pieces, ignore_index=True) if pieces else pd.Series(dtype=np.float32)
        if name in DEFINITIONS:
            model_input = (name in FEATURE_COLUMNS or name in STRUCTURE_FEATURES or
                           name in COMPOSITION_FEATURES or name in ORIGINAL_COMPOSITION or
                           name in DELTA_COMPOSITION_FEATURES)
            feature_metadata[name] = {
                "definition": DEFINITIONS[name], "source": "official pinned QMutBench original/mutated QASM via qmutbench_validation.parse_qasm_file",
                "data_type": str(values.dtype), "missing_rows": int(values.isna().sum()),
                "distinct_values": (int(values.nunique(dropna=True)) if len(values) else None),
                "available_before_outcome": True,
                "model_input_in_this_phase": bool(model_input),
                "fold_training_association_cramers_v": [float(v) for v in fold_samples[name]],
                "fold_training_association_mean": float(np.mean(fold_samples[name])) if fold_samples[name] else None,
                "leakage_risk": "Low when supplied original and mutant QASM are available before outcome execution; no outcome, group, row ID, or circuit behavior is included.",
                "precision_limit": "Counts and depth refer to parsed OpenQASM instructions without transpiler decomposition; estimated ASAP depth is not hardware execution depth.",
                "extraction_cost": "One parse per unique original circuit and one per development mutant; cached in 58 resumable compressed chunks.",
                "selection_status": ("selected only for controlled screen/confirmation; not promoted" if model_input else "extracted for delta construction or diagnostics"),
            }
    feature_metadata["source_gate_count_match"] = {
        "definition": DEFINITIONS["source_gate_count_match"], "source": "QASM-derived count and metadata gates column",
        "data_type": "int64", "missing_rows": 0, "distinct_values": 2,
        "available_before_outcome": True, "model_input_in_this_phase": False,
        "fold_training_association_cramers_v": [], "fold_training_association_mean": None,
        "leakage_risk": "Not outcome leakage, but diagnostic consistency flag only; excluded from model inputs.",
        "precision_limit": "Discrepancies reflect differing source gate-count conventions and must not be interpreted as parser failures.",
        "extraction_cost": "Computed from already extracted counts and source metadata.", "selection_status": "diagnostic only",
    }
    inventory_json = {
        "schema": "qmutbench-qasm-feature-discovery-v1",
        "starting_commit": "11b0fcc067a22b101b29bd506deed100f5f0edd8",
        "branch": "qmutbench-master-optimization",
        "dataset": {"path": str(SOURCE_CSV.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": EXPECTED_DATA_SHA, "rows": 723079,
                    "development_rows": 578117, "development_origins": 305,
                    "historical_frozen_rows": 144962, "historical_frozen_origins": 70,
                    "frozen_test_labels_or_files_used_for_selection": False},
        "official_checkout": {"path": str(OFFICIAL_CHECKOUT.relative_to(ROOT)).replace("\\", "/"),
                               "repository": "https://github.com/EnautMendi/QMutBench",
                               "commit": EXPECTED_REPO_COMMIT,
                               "worktree_clean": True,
                               "all_data_csv_sha256": sha256_file(OFFICIAL_CHECKOUT / "data/all_data.csv"),
                               "metadata_csv_sha256": EXPECTED_DATA_SHA,
                               "metadata_csv_semantic_equality_verified": True,
                               "tracked_qasm_files": len(tracked_qasm),
                               "original_qasm_files": len(origin_files), "mutant_qasm_files": len(mutant_files),
                               "development_original_qasm_links": len(originals),
                               "development_mutant_qasm_links": len(expected_files),
                               "development_linkage_coverage": len(expected_files) / len(oof),
                               "missing_development_mutant_files": len(missing_files)},
        "source_csv_columns": ["Unnamed: 0", "Origin_program", "algorithm", "qubits", "gates", "operation", "Position_int", "Killed", "gate", "algorithm_group", "dominant_state", "position"],
        "features": feature_metadata,
        "excluded_source_fields": {
            "Killed": "Prediction target; never included in feature extraction/model matrix.",
            "Origin_program": "Used only to exclude whole groups in folds and to report origin support.",
            "Unnamed: 0": "Used only as a traceability join key; not a model feature.",
            "original/replacement gate reconstruction": "QASM is already source-compiled/decomposed; original mutation gate semantics are not reliably recoverable from these artifacts, so no reconstructed gate pair is claimed.",
            "local gate neighborhood / control-target mutation": "Not extracted because source-QASM decomposition and the published position index do not establish an unambiguous pre-mutation neighbor mapping.",
            "simulation/measurement outcomes": "Not used; post-outcome behavior is excluded.",
        },
        "extraction": {"script": "experiments/qmutbench_qasm_features.py",
                       "cache_dir": "data/qmutbench_ml/qasm_structural_v4/",
                       "manifest": "data/qmutbench_ml/qasm_structural_v4/manifest.json",
                       "cache_manifest_sha256": sha256_file(DEFAULT_OUTPUT / "manifest.json"),
                       "status": extraction["status"], "requested_development_rows": extraction["records_requested"],
                       "extracted_rows": extraction["feature_rows"], "parse_error_rows": extraction["parse_error_rows"],
                       "original_circuit_count": extraction["original_circuit_count"],
                       "chunk_count": len(extraction["chunks"]), "workers": extraction["worker_count"],
                       "depth_policy": extraction["depth_definition"],
                       "resume_and_stale_cache_protection": True},
        "source_csv_interpretation": {"position": "Position (Circuit %) category documented as ten percentage bands; position_fraction remains the coarse-bin midpoint, not an exact position.",
                                      "gates": "Metadata gate count has convention differences from parsed QASM for some mutation types; it is retained as existing source input but not treated as validation oracle.",
                                      "operation_and_gate": "Available source metadata known before mutation outcome; no target-derived mutation details added."},
    }
    DISCOVERY_JSON.write_text(json.dumps(inventory_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return inventory_json


def create_results(discovery: dict[str, Any]) -> dict[str, Any]:
    confirmation = json.loads((COMPARISON_V3.parent / "qasm_model_comparison_v2/candidate_confirmation.json").read_text(encoding="utf-8"))
    delta = json.loads((COMPARISON_V3.parent / "qasm_model_comparison_v2/delta_composition_summary.json").read_text(encoding="utf-8"))
    screen_manifest = json.loads((COMPARISON_V3.parent / "qasm_model_comparison_v1/manifest.json").read_text(encoding="utf-8"))
    rf_manifest = json.loads((COMPARISON_V3 / "manifest.json").read_text(encoding="utf-8"))
    rf = rf_manifest["summary"]["random_forest/baseline_random_forest"]
    baseline_screen = screen_manifest["summary"]["extra_trees/baseline"]
    structure = screen_manifest["summary"]["extra_trees/structure_only"]
    composition = screen_manifest["summary"]["extra_trees/mutant_gate_composition"]
    screen_all_fold1 = json.loads((COMPARISON_V3.parent / "qasm_model_comparison_v1/folds/extra_trees_all_qasm_fold1.json").read_text(encoding="utf-8"))
    candidate_oof = pd.concat([pd.DataFrame({key: np.load(COMPARISON_V3.parent / "qasm_model_comparison_v2/folds" /
        f"extra_trees_gate_composition_100_fold{i}.npz", allow_pickle=False)[key]
        for key in ("record_id", "y", "origin_program", "fold", "probability")}) for i in (1, 2, 3)], ignore_index=True)
    baseline_oof = pd.concat([pd.DataFrame({key: np.load(PREDICTIONS / f"position_fraction_fold{i}.npz", allow_pickle=False)[key]
        for key in ("record_id", "y", "origin_program", "fold", "probability")}) for i in (1, 2, 3)], ignore_index=True)
    paired = candidate_oof.merge(baseline_oof, on="record_id", suffixes=("_candidate", "_baseline"), validate="one_to_one")
    origin_deltas = []
    for _origin, group in paired.groupby("origin_program_candidate", sort=False):
        y_origin = group.y_candidate.to_numpy(np.int8)
        if np.unique(y_origin).size < 2:
            continue
        candidate_f1 = f1_score(y_origin, (group.probability_candidate.to_numpy(float) >= 0.5).astype(np.int8), average="macro", zero_division=0)
        baseline_f1 = f1_score(y_origin, (group.probability_baseline.to_numpy(float) >= 0.5).astype(np.int8), average="macro", zero_division=0)
        origin_deltas.append(float(candidate_f1 - baseline_f1))
    result = {
        "schema": "qmutbench-qasm-model-results-v1",
        "provenance": discovery["dataset"],
        "evaluation": {"population": "578,117 development rows from 305 Origin_program groups",
                       "grouping": "same saved three-fold StratifiedGroupKFold assignments from current position_fraction OOF; all records of each origin stayed together",
                       "folds": {str(k): {"rows": int(v["rows"]), "origins": int(v["groups"])} for k,v in screen_manifest["folds"].items()},
                       "historical_frozen_test_used": False,
                       "threshold": 0.5,
                       "threshold_context": "Existing nested grouped threshold protocol was reused; thresholds are selected using the other two folds and applied to the held fold.",
                       "primary_metric": "macro F1"},
        "current_selected_model": {"version": "qmutbench-position-fraction-v1",
                                   "artifact_path": "data/qmutbench_ml/position_fraction_v1/best_model.joblib",
                                   "sha256": "641a18b67448f910ea0126c914e7e5f17bc946a3fa940b0486c045b845f8ee45",
                                   "features": ["algorithm", "operation", "gate", "dominant_state", "qubits", "gates", "Position_int", "position_fraction"],
                                   "metrics_at_0_5": confirmation["baseline_pooled_oof_metrics"],
                                   "fold_macro_f1": [x["macro_f1"] for x in confirmation["baseline_fold_metrics"]],
                                   "nested_threshold_mean_macro_f1": 0.7679000521819349,
                                   "nested_thresholds": [0.495, 0.460, 0.480]},
        "candidate_confirmation": {"feature_set": "current position_fraction model inputs plus 12 mutant QASM gate-name fractions and the multi-qubit gate fraction",
                                   "metrics_at_0_5": confirmation["candidate_pooled_oof_metrics"],
                                   "fold_macro_f1": [x["macro_f1"] for x in confirmation["folds"]],
                                   "macro_f1_delta": confirmation["pooled_macro_f1_delta"],
                                   "macro_f1_delta_percentage_points": confirmation["pooled_macro_f1_delta"] * 100,
                                   "balanced_accuracy_delta": confirmation["candidate_pooled_oof_metrics"]["balanced_accuracy"] - confirmation["baseline_pooled_oof_metrics"]["balanced_accuracy"],
                                   "accuracy_delta": confirmation["candidate_pooled_oof_metrics"]["accuracy"] - confirmation["baseline_pooled_oof_metrics"]["accuracy"],
                                   "fold_macro_f1_deltas": confirmation["fold_macro_f1_deltas"],
                                   "folds_won": sum(x > 0 for x in confirmation["fold_macro_f1_deltas"]),
                                   "nested_threshold_mean_macro_f1": 0.7703201849883303,
                                   "nested_thresholds": [0.500, 0.480, 0.500],
                                   "nested_threshold_delta_vs_current": 0.0024201328063954,
                                   "origin_level_macro_f1": {"origins_with_both_classes": len(origin_deltas),
                                       "origins_improved": sum(value > 0 for value in origin_deltas),
                                       "origins_worsened": sum(value < 0 for value in origin_deltas),
                                       "mean_delta": float(np.mean(origin_deltas)),
                                       "median_delta": float(np.median(origin_deltas))},
                                   "decision": "not promoted: below the predeclared 0.5 pp materiality bar at 0.32 pp, and not consistent across folds"},
        "screen_experiments": {
            "matched_60_tree_extra_trees_baseline": baseline_screen,
            "qasm_structure_only": {**structure, "decision": "rejected; below matched baseline"},
            "mutant_gate_composition": {**composition, "decision": "screen gain below materiality and only 2/3 folds improved; advanced to 100-tree confirmation"},
            "signed_mutant_minus_original_gate_composition": delta,
            "combined_structure_and_gate_composition_fold1": {"macro_f1": screen_all_fold1["metrics_at_threshold_0_5"]["macro_f1"],
                "fold1_baseline_macro_f1": baseline_screen["fold_metrics_at_0_5"][0]["macro_f1"],
                "decision": "stopped after fold 1 materially underperformed; no remaining folds fit"},
            "random_forest_same_features_and_folds": rf,
            "prior_model_family_evidence": {"unweighted_random_forest_macro_f1_mean": 0.731447,
                "scope": "existing fixed-fold comparison recorded in reports/experiments/qmutbench_model_comparison_20261025.json"},
        },
        "limitations": [
            "Only three saved outer group folds were available; available memory and fit times make a new five-fold full-corpus experiment impractical in this phase.",
            "The selected QASM feature candidate did not meet the predeclared +0.5 pp macro-F1 criterion or win all folds; current model and inference API remain unchanged.",
            "ASAP depth is a parser-derived native-instruction scheduling proxy, not transpiled circuit depth or hardware depth.",
            "The dataset metadata 'gates' has convention differences from parsed circuit instruction counts for some operators; the diagnostic mismatch feature is excluded from models.",
            "Position_fraction is the midpoint of a coarse 10 percentage-point band, not an exact mutation location.",
            "No model was fitted against historical frozen-test rows and those labels were not loaded for feature screening.",
        ],
        "target_85_percent_reached": False,
        "promoted_model": "qmutbench-position-fraction-v1",
        "artifact_paths": {"feature_extractor": "experiments/qmutbench_qasm_features.py",
                           "model_experiments": "experiments/qmutbench_qasm_model_comparison.py",
                           "discovery_manifest": str(DISCOVERY_JSON.relative_to(ROOT)).replace("\\", "/"),
                           "results_json": str(RESULTS_JSON.relative_to(ROOT)).replace("\\", "/"),
                           "results_markdown": str(REPORT_MD.relative_to(ROOT)).replace("\\", "/"),
                           "extracted_features": "data/qmutbench_ml/qasm_structural_v4/ (ignored local cache)"},
        "reproduction_commands": [
            ".venv/Scripts/python.exe -m pytest tests/test_qmutbench_qasm_features.py -q",
            ".venv/Scripts/python.exe -m experiments.qmutbench_qasm_features --workers 3 --chunk-rows 10000",
            ".venv/Scripts/python.exe -m experiments.qmutbench_qasm_model_comparison --variants mutant_gate_composition",
            ".venv/Scripts/python.exe -m experiments.qmutbench_qasm_model_comparison --delta-composition",
            ".venv/Scripts/python.exe -m experiments.qmutbench_qasm_model_comparison --confirm-gate-composition",
            ".venv/Scripts/python.exe -m experiments.qmutbench_qasm_model_comparison --variants baseline_random_forest",
        ],
        "test_results": {"feature_extraction_tests": "5 passed",
                         "actual_selected_model_inference_smoke": "single and batch prediction passed; finite output; saved version and 0.5 threshold"},
    }
    RESULTS_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def create_markdown(discovery: dict[str, Any], result: dict[str, Any]) -> None:
    b = result["current_selected_model"]["metrics_at_0_5"]
    c = result["candidate_confirmation"]["metrics_at_0_5"]
    screens = result["screen_experiments"]
    def _screen_row(name: str, entry: dict[str, Any], metrics_key: str, fold_key: str) -> str:
        score = entry[metrics_key]
        raw_folds = entry[fold_key]
        fold_values = [x["macro_f1"] if isinstance(x, dict) else x for x in raw_folds]
        folds = ", ".join(f"{x:.4f}" for x in fold_values)
        return f"| {name} | {score['macro_f1']:.4f} | {score['balanced_accuracy']:.4f} | {score['accuracy']:.4f} | {folds} |"
    screen_table = [
        "| 60-tree screen (three grouped folds) | Pooled macro F1 | Pooled balanced accuracy | Pooled accuracy | Fold macro F1 |",
        "|---|---:|---:|---:|---|",
        _screen_row("Extra Trees matched baseline", screens["matched_60_tree_extra_trees_baseline"], "pooled_oof_metrics_at_0_5", "fold_metrics_at_0_5"),
        _screen_row("QASM structure only", screens["qasm_structure_only"], "pooled_oof_metrics_at_0_5", "fold_metrics_at_0_5"),
        _screen_row("Mutant gate-name proportions + multi-qubit fraction", screens["mutant_gate_composition"], "pooled_oof_metrics_at_0_5", "fold_metrics_at_0_5"),
        _screen_row("Signed mutant-minus-original proportions", screens["signed_mutant_minus_original_gate_composition"], "pooled_oof_metrics", "fold_macro_f1"),
        _screen_row("Random Forest baseline features", screens["random_forest_same_features_and_folds"], "pooled_oof_metrics_at_0_5", "fold_metrics_at_0_5"),
        f"| Combined structure + composition (stopped after fold 1) | {screens['combined_structure_and_gate_composition_fold1']['macro_f1']:.4f} | n/a | n/a | {screens['combined_structure_and_gate_composition_fold1']['macro_f1']:.4f} |",
    ]
    discovery_data = discovery["dataset"]
    extracted = discovery["extraction"]
    rows = [
        "# QMutBench source-circuit feature extraction and model validation",
        "",
        "## Scope and evidence",
        "",
        f"Dataset SHA-256 `{discovery_data['sha256']}`; {discovery_data['development_rows']:,} development records across {discovery_data['development_origins']} origins. The historical frozen-test rows and labels were not used. The official checkout is pinned to `{discovery['official_checkout']['commit']}`.",
        "",
        f"The local source tree contains {discovery['official_checkout']['original_qasm_files']:,} originals and {discovery['official_checkout']['mutant_qasm_files']:,} mutant QASM files. Filename-derived mapping matched all {discovery['official_checkout']['development_mutant_qasm_links']:,} development source records to unique mutant files and {discovery['official_checkout']['development_original_qasm_links']:,} original circuits. Extraction completed for {extracted['extracted_rows']:,}/{extracted['requested_development_rows']:,} development rows with {extracted['parse_error_rows']} parsing failures.",
        "",
        "Only pre-outcome source and circuit syntax were parsed. Record IDs trace rows but are not predictors; `Origin_program` is only a group key/diagnostic. `Killed` was not loaded by the extraction process.",
        "",
        "## Extracted features",
        "",
        "The extractor records original and mutated instruction counts, ASAP per-wire gate-layer depth, depth per qubit, one/two/three-plus-qubit gate counts, multi-qubit fractions, distinct gate names, selected gate-name proportions, and mutant-minus-original count/depth/fraction deltas. Barriers synchronize their wires; measurements are excluded; gate instructions are not decomposed. This is a source-QASM scheduling proxy, not transpiled or hardware depth.",
        "",
        "`position` remains a coarse 10-point circuit-percent band; `position_fraction` is its midpoint and is not an exact mutation position. Original/replacement gate-pair reconstruction and local mutation neighborhoods were excluded because the pinned QASM contains compiled/decomposed instructions and does not establish those source-level semantics reliably.",
        "",
        "A diagnostic comparison found different gate-count conventions between the metadata `gates` field and parsed QASM for some operations. That match flag was excluded from all model inputs; source metadata was not used as a parser correctness oracle.",
        "",
        "## Comparable grouped-fold results",
        "",
        "All model comparisons use the same three saved `StratifiedGroupKFold` folds, with 91, 101, and 113 held-out origins; threshold fixed at 0.5. Preprocessing is fit on each training fold. Feature selection and model evaluation use development origins only.",
        "",
        "| Model | Macro F1 | Balanced accuracy | Accuracy |",
        "|---|---:|---:|---:|",
        f"| Current 100-tree Extra Trees + position fraction | {b['macro_f1']:.4f} | {b['balanced_accuracy']:.4f} | {b['accuracy']:.4f} |",
        f"| Candidate + 12 QASM gate-name fractions and multi-qubit fraction | {c['macro_f1']:.4f} | {c['balanced_accuracy']:.4f} | {c['accuracy']:.4f} |",
        "",
        f"Candidate delta: macro F1 {result['candidate_confirmation']['macro_f1_delta_percentage_points']:+.2f} percentage points, balanced accuracy {result['candidate_confirmation']['balanced_accuracy_delta']*100:+.2f} pp, and accuracy {result['candidate_confirmation']['accuracy_delta']*100:+.2f} pp. Fold macro-F1 deltas were {', '.join(f'{x*100:+.2f} pp' for x in result['candidate_confirmation']['fold_macro_f1_deltas'])}; 2/3 folds improved. Nested threshold averages were {result['current_selected_model']['nested_threshold_mean_macro_f1']:.4f} for current and {result['candidate_confirmation']['nested_threshold_mean_macro_f1']:.4f} for candidate.",
        f"Across {result['candidate_confirmation']['origin_level_macro_f1']['origins_with_both_classes']} origins with both target classes, candidate per-origin macro F1 improved on {result['candidate_confirmation']['origin_level_macro_f1']['origins_improved']} and worsened on {result['candidate_confirmation']['origin_level_macro_f1']['origins_worsened']}.",
        "",
        "### Bounded screen and alternative model family",
        "",
        *screen_table,
        "",
        "The candidate fails the preregistered +0.5 pp materiality criterion and is not consistent across folds; the current model remains selected. Composition-only 60-tree screening was +0.48 pp on average and advanced to confirmation. Signed mutant-minus-original composition was negative against the matched 60-tree baseline. Structure-only features were also lower than the baseline. Combined structure plus composition scored 0.7478 on fold 1 versus the 0.7631 baseline and was stopped. Random Forest baseline feature runs averaged 0.7589 macro F1 on the same folds versus 0.7663 for the 60-tree Extra Trees screen.",
        "",
        "The candidate raised detected recall at 0.5 (0.7309 to 0.7592) but reduced Survivor recall (0.8085 to 0.7847); the aggregate macro-F1 gain remained below the selection bar. No candidate was promoted and the inference API remains unchanged.",
        "",
        "## Reproducibility and limits",
        "",
        "The feature cache is resumable in 58 compressed chunks and carries dataset, QASM checkout, parser, extractor, and saved OOF checksum provenance. Generated feature matrices, predictions, and fitted experiment models remain ignored under `data/qmutbench_ml/`.",
        "",
        "The experiment used three fixed outer folds. A new five-fold full-corpus comparison was not practical under available memory and the measured multi-minute fold fits. The 85% aspirational target was not reached. This is historical mutation-outcome prediction only, not hardware-noise detection or fault localization.",
        "",
        "Focused tests: `.venv/Scripts/python.exe -m pytest tests/test_qmutbench_qasm_features.py -q` (5 passed). The selected model's real inference API passed single-record and three-record batch smoke checks with finite probabilities and the saved model version/threshold.",
        "",
        f"Machine-readable discovery inventory: `{DISCOVERY_JSON.relative_to(ROOT).as_posix()}`. Machine-readable model results: `{RESULTS_JSON.relative_to(ROOT).as_posix()}`.",
    ]
    REPORT_MD.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    discovery = create_discovery_manifest()
    results = create_results(discovery)
    create_markdown(discovery, results)
    print(json.dumps({"discovery": str(DISCOVERY_JSON), "results": str(RESULTS_JSON), "report": str(REPORT_MD),
                      "development_coverage": discovery["extraction"]["extracted_rows"],
                      "candidate_macro_f1_delta_pp": results["candidate_confirmation"]["macro_f1_delta_percentage_points"]}, indent=2))


if __name__ == "__main__":
    main()
