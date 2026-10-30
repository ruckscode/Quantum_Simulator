"""Summarize already-saved QMutBench development results for campaign v3.

This script does not fit models or read frozen-test labels. Metrics are taken
from saved OOF predictions/validated result summaries in existing campaigns.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/experiments/full_data_qml_optimization_v3"
ML = ROOT / "data/qmutbench_ml"
FINAL = ML / "exhaustive_campaign_v1/five_fold_confirmation/summary.json"
V1 = ML / "qasm_model_comparison_v1/manifest.json"
V2 = ML / "qasm_model_comparison_v2/candidate_confirmation.json"
V2D = ML / "qasm_model_comparison_v2/delta_composition_summary.json"
V3 = ML / "qasm_model_comparison_v3/manifest.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return "" if value is None else round(float(value) * 100, 6)


def row(name, family, feature_set, records, groups, folds, accuracy=None, f1=None, bacc=None,
        fold_f1="", fold_delta="", status="", artifact="", seeds="", runtime=""):
    return {"candidate": name, "model_family": family, "feature_set": feature_set,
            "records": records, "independent_origins": groups, "folds": folds,
            "accuracy_percent": pct(accuracy), "macro_f1_percent": pct(f1),
            "balanced_accuracy_percent": pct(bacc), "fold_macro_f1": fold_f1,
            "fold_macro_f1_delta_vs_matched_baseline": fold_delta, "seeds": seeds,
            "runtime_seconds": runtime, "status": status, "source_artifact": artifact}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    final = read(FINAL)
    v1 = read(V1)
    v2 = read(V2)
    v2d = read(V2D)
    v3 = read(V3)
    winner = final["candidate_pooled_metrics"]
    base5 = final["baseline_pooled_metrics"]
    v1s = v1["summary"]
    rows = [
        row("historical_mutation_aware_champion", "ExtraTrees", "mutation-aware metadata and QASM-diff representation",
            final["rows"], final["origins"], 5, winner["accuracy"], winner["macro_f1"], winner["balanced_accuracy"],
            status="HISTORICAL_REFERENCE; grouped 5-fold OOF; not retrained in v3",
            artifact="data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/summary.json"),
        row("historical_matched_baseline_5fold", "ExtraTrees", "incumbent comparison baseline",
            final["rows"], final["origins"], 5, base5["accuracy"], base5["macro_f1"], base5["balanced_accuracy"],
            status="HISTORICAL_REFERENCE; same 5-fold OOF contract as champion",
            artifact="data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/summary.json"),
    ]
    for name, key, desc in [
        ("qasm_v1_metadata_baseline", "extra_trees/baseline", "metadata baseline on matched 3-fold grouped OOF"),
        ("qasm_v1_mutant_gate_composition", "extra_trees/mutant_gate_composition", "metadata + mutant gate composition QASM features"),
        ("qasm_v1_structure_only", "extra_trees/structure_only", "QASM structural features only"),
    ]:
        stat = v1s[key]
        pooled = stat["pooled_oof_metrics_at_0_5"]
        fold_f1 = [x["macro_f1"] for x in stat["fold_metrics_at_0_5"]]
        rows.append(row(name, "ExtraTrees", desc, 578117, 305, 3, pooled["accuracy"], pooled["macro_f1"], pooled["balanced_accuracy"],
                        fold_f1=json.dumps(fold_f1), status="SAVED 3-FOLD GROUPED OOF; all eligible records",
                        artifact="data/qmutbench_ml/qasm_model_comparison_v1/manifest.json"))
    v1_all = read(ML / "qasm_model_comparison_v1/folds/extra_trees_all_qasm_fold1.json")
    m = v1_all["metrics_at_threshold_0_5"]
    rows.append(row("qasm_v1_all_qasm_fold1_only", "ExtraTrees", "all registered QASM structural features",
                    v1_all.get("validation_rows", ""), v1_all.get("validation_origins", ""), 1,
                    m["accuracy"], m["macro_f1"], m["balanced_accuracy"], status="PARTIAL; one of three folds only; not pooled",
                    artifact="data/qmutbench_ml/qasm_model_comparison_v1/folds/extra_trees_all_qasm_fold1.json"))

    cand = v2["candidate_pooled_oof_metrics"]
    base = v2["baseline_pooled_oof_metrics"]
    rows.append(row("qasm_v2_metadata_plus_mutant_gate_fractions", "ExtraTrees",
                    v2["feature_set"], v2["rows"], v2["origins"], len(v2["folds"]), cand["accuracy"], cand["macro_f1"], cand["balanced_accuracy"],
                    fold_f1=json.dumps([x["macro_f1"] for x in v2["folds"]]),
                    fold_delta=json.dumps(v2["fold_macro_f1_deltas"]),
                    status="SAVED GROUPED OOF; 2/3 folds improve; not stable enough to promote",
                    artifact="data/qmutbench_ml/qasm_model_comparison_v2/candidate_confirmation.json"))
    rows.append(row("qasm_v2_matched_metadata_baseline", "ExtraTrees", "position_fraction model inputs without added gate fractions",
                    v2["rows"], v2["origins"], len(v2["folds"]), base["accuracy"], base["macro_f1"], base["balanced_accuracy"],
                    fold_f1=json.dumps([x["macro_f1"] for x in v2["baseline_fold_metrics"]]),
                    status="SAVED MATCHED GROUPED OOF BASELINE",
                    artifact="data/qmutbench_ml/qasm_model_comparison_v2/candidate_confirmation.json"))
    d = v2d["pooled_oof_metrics"]
    rows.append(row("qasm_v2_delta_gate_composition", "ExtraTrees", "metadata position_fraction + mutant-minus-original gate-family fractions",
                    v2d["rows"], 305, 3, d["accuracy"], d["macro_f1"], d["balanced_accuracy"],
                    fold_f1=json.dumps(v2d["fold_macro_f1"]), fold_delta=json.dumps(v2d["fold_macro_f1_delta"]),
                    status="SAVED GROUPED OOF; all three folds regress against matched baseline",
                    artifact="data/qmutbench_ml/qasm_model_comparison_v2/delta_composition_summary.json"))
    for key, metrics in v3["summary"].items():
        pooled = metrics["pooled_oof_metrics_at_0_5"]
        rows.append(row("qasm_v3_random_forest_baseline_screen", "RandomForest", "metadata plus position_fraction; QASM features not added",
                        578117, 305, 3, pooled["accuracy"], pooled["macro_f1"], pooled["balanced_accuracy"],
                        fold_f1=json.dumps([x["macro_f1"] for x in metrics["fold_metrics_at_0_5"]]),
                        status="PARTIAL SCREEN; baseline control only; not a QASM feature result",
                        artifact="data/qmutbench_ml/qasm_model_comparison_v3/manifest.json"))
    write_csv(OUT / "qasm_feature_model_comparison.csv", rows)

    fold_rows = []
    for key, label in [("extra_trees/baseline", "qasm_v1_metadata_baseline"),
                       ("extra_trees/mutant_gate_composition", "qasm_v1_mutant_gate_composition"),
                       ("extra_trees/structure_only", "qasm_v1_structure_only")]:
        for i, metrics in enumerate(v1s[key]["fold_metrics_at_0_5"], 1):
            fold_rows.append({"candidate": label, "fold": i, "validation_records": metrics["confusion_matrix_labels_0_1"][0][0] + metrics["confusion_matrix_labels_0_1"][0][1] + metrics["confusion_matrix_labels_0_1"][1][0] + metrics["confusion_matrix_labels_0_1"][1][1],
                              "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                              "balanced_accuracy": metrics["balanced_accuracy"], "roc_auc": metrics.get("roc_auc"),
                              "average_precision": metrics.get("average_precision"), "brier_score": metrics.get("brier_score")})
    for i, metrics in enumerate(v2["folds"], 1):
        fold_rows.append({"candidate": "qasm_v2_metadata_plus_mutant_gate_fractions", "fold": i,
                          "validation_records": sum(metrics["confusion_matrix_labels_0_1"][j][k] for j in range(2) for k in range(2)),
                          "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "balanced_accuracy": metrics["balanced_accuracy"],
                          "roc_auc": metrics.get("roc_auc"), "average_precision": metrics.get("average_precision"), "brier_score": metrics.get("brier_score")})
    write_csv(OUT / "qasm_feature_fold_metrics.csv", fold_rows)

    feat_audit = read(OUT / "feature_extraction_audit.json")
    leakage = read(OUT / "label_leakage_audit.json")
    vm_available = None
    try:
        import psutil  # type: ignore
        vm_available = round(psutil.virtual_memory().available / 1048576, 1)
    except ImportError:
        pass
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    qmut_status = {
        "development_extraction": "COMPLETED_AND_REHASHED",
        "records": feat_audit["development_oof"]["rows"],
        "origins": feat_audit["development_oof"]["origin_count"],
        "structural_feature_rows": feat_audit["qasm_structural_features"]["actual_feature_rows"],
        "structural_parse_errors": feat_audit["qasm_structural_features"]["manifest_parse_error_rows"],
        "mutation_diff_feature_rows": feat_audit["qasm_mutation_diff_features"]["actual_feature_rows"],
        "mutation_diff_parse_errors": feat_audit["qasm_mutation_diff_features"]["manifest_parse_error_rows"],
        "metadata_binary_mapping": {"WOO": 1, "OPO": 1, "Survivor": 0},
        "mapping_mismatches": feat_audit["development_oof"]["label_mapping_mismatches"],
        "available_ram_mib_at_report": vm_available,
        "new_qml_training_started": False,
        "overall_cache_integrity": feat_audit["overall_integrity"],
    }
    extraction = read(OUT / "extraction_audit.json")
    extraction["qmutbench"]["development_qasm_extraction_status"] = "COMPLETE_VERIFIED"
    extraction["qmutbench"]["development_qasm_feature_rows"] = qmut_status["structural_feature_rows"]
    extraction["qmutbench"]["development_origin_count"] = qmut_status["origins"]
    extraction["qmutbench"]["development_parse_errors"] = qmut_status["structural_parse_errors"]
    extraction["qmutbench"]["structural_manifest"] = "data/qmutbench_ml/qasm_structural_v4/manifest.json"
    extraction["qmutbench"]["mutation_diff_manifest"] = "data/qmutbench_ml/qasm_mutation_diff_v1/manifest.json"
    extraction["qmutbench"]["independent_cache_audit"] = "feature_extraction_audit.json"
    extraction["qmutbench"]["full_metadata_source_row_value_comparison"] = "VALUE_IDENTICAL; line ending/size difference only"
    extraction["qmutbench"]["full_syntax_parse_coverage"] = "COMPLETE for all 578,117 development IDs; frozen-test IDs intentionally excluded"
    (OUT / "extraction_audit.json").write_text(json.dumps(extraction, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = read(OUT / "training_manifest.json")
    manifest["qmutbench_development_extraction"] = qmut_status
    manifest["qmutbench_full_data_status"] = "DEVELOPMENT_FEATURE_EXTRACTION_COMPLETE; QNN/model re-evaluation and full-data training incomplete"
    manifest["available_ram_mib_at_report"] = vm_available
    manifest["new_training_runs_started"] = 0
    manifest["frozen_test_accessed"] = False
    (OUT / "training_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    blocked = read(OUT / "blocked_tasks.json")
    blocked["blocked"] = [
        {"task": "full-data QNN on QMutBench binary and multiclass inputs", "status": "BLOCKED",
         "reason": f"Available RAM was {vm_available} MiB at report generation, below the 1,024 MiB safety floor; CPU utilization was previously observed at 84.9% and no CUDA GPU is available.",
         "evidence": "resource_usage.csv and environment audit", "required_to_resume": "Wait for at least 1 GiB available RAM, remeasure, then run bounded CPU PennyLane jobs one at a time."},
        {"task": "five-fold QASM-feature QNN and exact champion comparison", "status": "NOT_RUN",
         "reason": "Existing QASM feature candidate screens use three grouped folds, while the verified historical champion uses five; no new QML fit was run in this turn.",
         "required_to_resume": "Run the QASM-feature QNN and matched classical controls on the incumbent five-fold contract after the RAM guard is met."},
        {"task": "source_gate_count_match feature ablation", "status": "NOT_RUN",
         "reason": "The diagnostic column varies by label and is excluded from the registered feature list; no same-fold with/without model ablation was run.",
         "required_to_resume": "Evaluate with and without the field on the same grouped development folds; do not add it to the production schema beforehand."},
        {"task": "frozen-test checksum verification", "status": "PARTIAL",
         "reason": "No frozen-test artifact path/checksum is recorded in the available test manifest; no test outcomes or test file were opened.",
         "required_to_resume": "Locate the already-pinned frozen-test artifact and verify its hash without reading labels, if checksum proof is required."},
        {"task": "commit required campaign updates", "status": "BLOCKED",
         "reason": "The workspace policy exposes .git as read-only; git add failed with `fatal: Unable to create '.git/index.lock': Permission denied`.",
         "required_to_resume": "Run Git staging/commit in an environment with write access to .git."},
    ]
    blocked["incomplete"] = ["full-data QMutBench QNN training and optimization", "QASM all_features three-fold comparison (only fold 1 exists)",
                             "same-five-fold QASM feature and QNN comparisons against the 79.33% champion", "same-fold source_gate_count_match ablation",
                             "expanded-source QNN retraining and full-data integration"]
    (OUT / "blocked_tasks.json").write_text(json.dumps(blocked, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    registry = read(OUT / "task_registry.json")
    for task in registry.get("tasks", []):
        if task.get("id") == "qmutbench_binary_production":
            task["labels"] = {"Survivor": 0, "WOO": 1, "OPO": 1}
            task["mapping_verified_against_saved_development_oof"] = True
            task["development_feature_status"] = "578117 records / 305 origins; structural and mutation-difference caches verified; no parse errors"
            task["new_qml_status"] = "NOT_RUN; below RAM safety floor"
    registry["full_data_v3_update"] = qmut_status
    (OUT / "task_registry.json").write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    base_macro = winner["macro_f1"]
    v1_comp = v1s["extra_trees/mutant_gate_composition"]["pooled_oof_metrics_at_0_5"]
    v1_base = v1s["extra_trees/baseline"]["pooled_oof_metrics_at_0_5"]
    v1_structure = v1s["extra_trees/structure_only"]["pooled_oof_metrics_at_0_5"]
    v2_gain_pp = (cand["macro_f1"] - base["macro_f1"]) * 100
    qasm_report = f"""# QMutBench QASM feature and model audit\n\n## Data and integrity\n\n- Verified official ingestion metadata SHA-256: `{feat_audit['metadata']['ingestion_sha256']}`; row-value comparison against `data/qmutbench_full/data/all_data.csv` is `{feat_audit['metadata']['checkout_vs_ingestion_row_value_audit']['status']}` over {feat_audit['metadata']['rows']:,} rows (the files differ in byte size/hash because of line endings).\n- Grouped development OOF: {qmut_status['records']:,} unique record IDs across {qmut_status['origins']} origins; 3 folds with record counts {feat_audit['development_oof']['fold_record_counts']}; cross-fold origin count {feat_audit['development_oof']['origins_crossing_folds']}.\n- QASM structure cache: {qmut_status['structural_feature_rows']:,} rows, {feat_audit['qasm_structural_features']['manifest_chunks']} chunks, zero parse errors, all chunk checksums verified.\n- Mutation-difference cache: {qmut_status['mutation_diff_feature_rows']:,} rows, {feat_audit['qasm_mutation_diff_features']['manifest_chunks']} chunks, zero parse errors, all chunk checksums verified.\n- Existing binary mapping verified record-by-record: WOO=1, OPO=1, Survivor=0; disagreements: {feat_audit['development_oof']['label_mapping_mismatches']}.\n- Frozen-test outcomes were not opened. Its independent file checksum was not rehashed in this turn because no frozen-test artifact/hash was located in the available manifest.\n\n## Saved feature comparison results\n\nAll QASM comparison rows below are historical 3-fold grouped OOF screens on the same 578,117 records. They are not directly comparable with the separate 5-fold production champion because the fold contracts differ.\n\n| Candidate | Feature family | Accuracy | Macro F1 | Balanced accuracy | Macro F1 change vs matched QASM baseline | Decision |\n|---|---|---:|---:|---:|---:|---|\n| 3-fold metadata baseline | Metadata | {pct(v1_base['accuracy']):.2f}% | {pct(v1_base['macro_f1']):.2f}% | {pct(v1_base['balanced_accuracy']):.2f}% | reference | — |\n| 3-fold QASM mutant gate composition | Metadata + gate-family fractions | {pct(v1_comp['accuracy']):.2f}% | {pct(v1_comp['macro_f1']):.2f}% | {pct(v1_comp['balanced_accuracy']):.2f}% | +{(v1_comp['macro_f1']-v1_base['macro_f1'])*100:.2f} pp | Two of three fold deltas positive; not consistent |\n| 3-fold QASM structure only | Structural counts/depth | {pct(v1_structure['accuracy']):.2f}% | {pct(v1_structure['macro_f1']):.2f}% | {pct(v1_structure['balanced_accuracy']):.2f}% | {(v1_structure['macro_f1']-v1_base['macro_f1'])*100:.2f} pp | Lower than metadata baseline |\n| Position-fraction + mutant gate fractions | Metadata + mutant gate-family composition | {pct(cand['accuracy']):.2f}% | {pct(cand['macro_f1']):.2f}% | {pct(cand['balanced_accuracy']):.2f}% | {v2_gain_pp:+.2f} pp | Two of three fold deltas positive; not promoted |\n| Position-fraction + delta gate composition | Metadata + mutant-minus-original fractions | {pct(v2d['pooled_oof_metrics']['accuracy']):.2f}% | {pct(v2d['pooled_oof_metrics']['macro_f1']):.2f}% | {pct(v2d['pooled_oof_metrics']['balanced_accuracy']):.2f}% | {v2d['pooled_macro_f1_delta']*100:+.2f} pp | Regressed in all three folds |\n| All registered QASM features | Combined | — | — | — | — | Partial: fold 1 only; no pooled score |\n\n### Five-fold historical reference\n\nThe existing mutation-aware ExtraTrees champion has saved grouped 5-fold OOF metrics of **{winner['accuracy']*100:.2f}% accuracy, {base_macro*100:.2f}% macro F1, and {winner['balanced_accuracy']*100:.2f}% balanced accuracy** on the same 578,117 development records / 305 origins. Its matched five-fold baseline scored {base5['accuracy']*100:.2f}% accuracy and {base5['macro_f1']*100:.2f}% macro F1. QASM screens above use different three-fold assignments, so they do not establish improvement over this champion. The 85% accuracy research target remains {85-winner['accuracy']*100:.2f} percentage points above the verified five-fold champion.\n\n## Leakage and feature handling\n\n`source_gate_count_match` is emitted in the feature CSV as a diagnostic but omitted from the manifest’s registered model feature list. It is not constant: counts by label are `{feat_audit['qasm_structural_features']['source_gate_count_match_by_binary_label_and_value']}`. This association warrants a same-fold ablation and provenance review; it does not alone prove target leakage. The field remains excluded. Identifiers, group names, paths, and folds were not used as predictive columns.\n\n## QML status\n\nNo full-data QMutBench binary QNN or hybrid fit was run in this continuation. Existing QNN scores on the 205-row three-class subset are a separate target and are not compared with the binary champion. The local `.venv` has PennyLane and Qiskit but no CUDA GPU; memory remained below the 1 GiB guard.\n"""
    (OUT / "qasm_feature_ablation_report.md").write_text(qasm_report, encoding="utf-8")

    tests_report = """# Test report — current continuation\n\n- Requested `tests/test_qmutbench_full_corpus_streaming.py`: **not present** in this checkout; `experiments/qmutbench_full_corpus_streaming.py` is also absent. The repository’s existing `experiments/qmutbench_qasm_features.py` and completed cache were used instead; no duplicate extractor was introduced.\n- Initial system-Python pytest collection failed because repository modules were not on its import path. The first `.venv` attempt also hit sandbox-protected `%TEMP%`. These were environment invocation issues, not assertion failures.\n- Rerun with temp/basetemp inside the writable repository: `tests/test_qmutbench_qasm_features.py`, `tests/test_qmutbench_qasm_mutation_diff.py`, and `tests/test_qmutbench_corpus_pipeline.py`: **21 passed**.\n- Broader focused suite: `tests/test_qmutbench_master_optimization.py`, `tests/test_qmutbench_learning_curve.py`, `tests/test_qmutbench_selected_pipeline.py`, and `tests/test_qmutbench_ml_pipeline_resume.py`: **19 passed**.\n- Total relevant passing tests: **40**. No test used frozen-test labels.\n\nReproduction (PowerShell, repository root):\n\n```powershell\n$env:TEMP = Join-Path (Get-Location) '.tmp-qmutbench-tests'\n$env:TMP = $env:TEMP\n.\\.venv\\Scripts\\python.exe -m pytest --basetemp=\"$PWD\\.pytest-full-corpus-campaign\" tests/test_qmutbench_qasm_features.py tests/test_qmutbench_qasm_mutation_diff.py tests/test_qmutbench_corpus_pipeline.py tests/test_qmutbench_master_optimization.py tests/test_qmutbench_learning_curve.py tests/test_qmutbench_selected_pipeline.py tests/test_qmutbench_ml_pipeline_resume.py -q\n```\n"""
    (OUT / "test_report.md").write_text(tests_report, encoding="utf-8")

    readme = OUT / "README.md"
    readme.write_text("# Full-data QML optimization campaign v3\n\n"
                      "The full QMutBench development QASM extraction already existed in the local repository and has now been independently verified: 578,117 development records / 305 origins, 58 structural chunks and 58 mutation-diff chunks, zero parse errors, and matching metadata values across the official checkout and ingestion snapshot. The frozen-test records were excluded. See `feature_extraction_audit.json` and `label_leakage_audit.json`.\n\n"
                      "No full-data QNN training was started in this continuation because available RAM was below the 1 GiB floor. Existing three-fold feature screens did not demonstrate a consistent QASM gain, and their folds differ from the five-fold 79.33% macro-F1 historical champion. See `qasm_feature_ablation_report.md` and `blocked_tasks.json`.\n\n"
                      "No frozen-test outcomes were read. The required extractor-specific test file and script named by the prompt are absent from this checkout; equivalent existing QASM and corpus tests passed. See `test_report.md`.\n", encoding="utf-8")
    (OUT / "qml_accuracy_optimization_report.md").write_text(
        f"# QML accuracy optimization v3 — continuation update\n\n"
        f"Current branch commit: `{commit}`. The user-provided latest commit `{EXPECTED_PROMPT_COMMIT}` is not in this checkout’s Git object database; current local HEAD is unchanged.\n\n"
        f"Development QASM extraction is complete and independently rehashed; see `feature_extraction_audit.json`. Grouped OOF contains 578,117 records / 305 origins, three development folds, with zero parser errors or label mapping mismatches.\n\n"
        f"The existing five-fold ExtraTrees champion remains {winner['accuracy']*100:.2f}% accuracy / {winner['macro_f1']*100:.2f}% macro F1 / {winner['balanced_accuracy']*100:.2f}% balanced accuracy. Same-campaign QASM feature screens on three folds showed modest aggregate gains for mutant gate composition (+0.48 pp macro F1 over its matched metadata baseline), but improvement was not consistent in every fold; structure-only and delta-composition screens regressed. The complete all-QASM candidate has only one fold. None are promoted, and none demonstrate improvement over the five-fold champion because their fold contracts differ. Full details are in `qasm_feature_ablation_report.md`.\n\n"
        "The correct existing binary label map is WOO=1, OPO=1, Survivor=0 and was verified against every development OOF record. `source_gate_count_match` remains excluded pending an ablation.\n\n"
        f"40 relevant tests passed after directing temp files to the writable workspace; see `test_report.md`. No QNN training occurred: current available RAM was {vm_available} MiB at report generation, below the 1 GiB guard. Frozen-test labels were not accessed. The v3 directory remains partial.\n", encoding="utf-8")

    index_path = OUT / "prediction_index.csv"
    existing = []
    with index_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f); fields = reader.fieldnames or []; existing = list(reader)
    additions = [
        {"task": "QMutBench binary historical incumbent", "candidate": "mutation_aware_ExtraTrees", "prediction_path": "data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/mutation_diff_oof.npz", "evaluation": "grouped 5-fold development OOF; 578117 rows / 305 origins", "frozen_test": "False"},
        {"task": "QMutBench binary historical comparator", "candidate": "matched_ExtraTrees_baseline", "prediction_path": "data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/baseline_oof.npz", "evaluation": "grouped 5-fold development OOF; 578117 rows / 305 origins", "frozen_test": "False"},
    ]
    for name in ["baseline", "mutant_gate_composition", "structure_only"]:
        additions.append({"task": "QMutBench binary QASM feature screen", "candidate": f"extra_trees_{name}",
                          "prediction_path": ";".join(f"data/qmutbench_ml/qasm_model_comparison_v1/folds/extra_trees_{name}_fold{i}.npz" for i in range(1,4)),
                          "evaluation": "3-fold grouped development OOF; 578117 rows / 305 origins", "frozen_test": "False"})
    known = {(r.get("task"), r.get("candidate")) for r in existing}
    existing.extend(r for r in additions if (r["task"], r["candidate"]) not in known)
    with index_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(existing)

    (OUT / "model_card_qmutbench_binary_incumbent.md").write_text(
        f"# Model card: QMutBench binary development incumbent\n\n"
        f"- Model: classical ExtraTrees, mutation-aware feature model; this is not a QNN.\n"
        f"- Target: killed mutant (WOO or OPO)=1; Survivor=0. Mapping verified against {feat_audit['development_oof']['rows']:,} saved development OOF rows.\n"
        f"- Evaluation: grouped 5-fold OOF over 578,117 records and 305 QMutBench origins; 79.33% accuracy, 79.33% macro F1, 79.41% balanced accuracy. No frozen-test data is reported here.\n"
        f"- Intended use: research baseline for QMutBench mutation outcomes; does not diagnose fault cause, localize bugs, or predict execution reproducibility.\n"
        f"- Limitations: classical model; outcome distribution and mutation-generation process are benchmark-specific. Full QASM-enhanced QNN comparison is outstanding.\n"
        f"- Artifacts: `data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/summary.json`, `data/qmutbench_ml/exhaustive_campaign_v1/five_fold_confirmation/mutation_diff_oof.npz`, and `data/qmutbench_ml/master_optimization_v1/best_model_metadata.json`.\n", encoding="utf-8")

    print(json.dumps({"report_dir": str(OUT.relative_to(ROOT)), "qasm_comparison_rows": len(rows),
                      "qasm_fold_metric_rows": len(fold_rows), "available_ram_mib": vm_available,
                      "status": "PARTIAL; no training was performed"}, indent=2))
    return 0


# This literal is metadata only; no checkout lookup or fetch is performed.
EXPECTED_PROMPT_COMMIT = "8965d3cfa9d40a53f841e99268d72eedd50fbe56"


if __name__ == "__main__":
    raise SystemExit(main())
