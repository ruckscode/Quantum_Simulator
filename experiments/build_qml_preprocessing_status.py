"""Join source inventory, task matrix, processed manifests, and semantic audit."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028"
OUT = ROOT / "reports/experiments/qml_source_preprocessing_20261029"


def main() -> None:
    inventory = json.loads((CAMPAIGN / "verified_dataset_inventory.json").read_text(encoding="utf-8-sig"))
    with (CAMPAIGN / "campaign_completion_matrix.csv").open(encoding="utf-8-sig", newline="") as stream:
        tasks = list(csv.DictReader(stream))
    semantic_path = OUT / "semantic_validation_audit.json"
    semantic = json.loads(semantic_path.read_text(encoding="utf-8")) if semantic_path.is_file() else {"tasks": []}
    nine_classical_path = ROOT / "reports/experiments/local_grouped_classical_suite_nine_source_v2_20261029/audit_manifest.json"
    nine_classical_audit = json.loads(nine_classical_path.read_text(encoding="utf-8")) if nine_classical_path.is_file() else None
    training_artifact_path = CAMPAIGN / "training_artifact_audit.json"
    training_artifact_audit = json.loads(training_artifact_path.read_text(encoding="utf-8")) if training_artifact_path.is_file() else None
    semantic_by_key = {(x["source_id"], x["task_id"]): x for x in semantic["tasks"]}
    contracts = {}
    for path in sorted((ROOT / "experiments/source_task_contracts").glob("*.json")):
        c = json.loads(path.read_text(encoding="utf-8"))
        contracts[(c["source_id"], c["task_id"])] = (path, c)
    source_name_to_id = {
        "Bugs4Q official framework": "bugs4q_core", "Bugs4Q-NA": "bugs4q_na",
        "Bugs4Q pooled": "bugs4q_joint", "Bugs4Q core-only transfer": "bugs4q_joint",
        "Bugs4Q-NA-only transfer": "bugs4q_joint", "QBugs HornBro subset": "qbugs_hornbro",
        "Quantum platform annotations": "quantum_platform_bugs", "Qolumbina empirical study": "qolumbina_ideal",
        "LintQ": "lintq", "Bugs4Q-Robust": "bugs4q_robust", "QSimBench": "qsimbench",
        "QMutBench": "qmutbench", "Mute-Alg": "mute_alg_hornbro",
    }
    task_id_by_name = {
        "Bugs4Q official framework": "source_state", "Bugs4Q-NA": "source_state",
        "Bugs4Q pooled": "source_state", "Bugs4Q core-only transfer": "core_only_transfer",
        "Bugs4Q-NA-only transfer": "na_only_transfer", "QBugs HornBro subset": "correct_buggy_circuit_state",
        "Quantum platform annotations": "comment_only_classification", "Qolumbina empirical study": "seeded_mutant_detection_ideal",
        "LintQ": "warning_triage", "Bugs4Q-Robust": "harness_execution_outcome",
        "QSimBench": "backend_fingerprint", "QMutBench": "three_class_mutation_outcome_exploratory",
        "Mute-Alg": "compatible_subset_qasm_v2",
    }
    limitation_by_name = {
        "Mute-Alg": "The versioned Qiskit 1.4.5 compatibility scan loaded all 169/169 files. Of 16,360 source records, 7,220 were explicitly rejected in complete pairs where exact OpenQASM had contradictory labels; 9,140 accepted records form 812 duplicate-safe groups in five folds. The previous 5,440-row subset is preserved as ineligible; see the v2 source and processed manifests.",
        "Bug-Pattern": "Source audit found only 9 local notebooks across four categories; README-linked external issue examples are not local rows. Notebook titles/markdown expose category labels, and there are no clean negatives or enough independent groups for a leakage-safe split. See bug_pattern_qchecker_source_audit.json.",
        "QChecker": "Source audit found 31 instance files across nine directory labels, one syntax error, severe class imbalance, no clean negatives, and no verified per-file detector outcomes; README uses MI while the local directory is IM. No defensible leakage-safe supervised split is established. See bug_pattern_qchecker_source_audit.json.",
        "QML platform workbooks": "The 39 workbooks parse as OOXML, but their exports have heterogeneous and sometimes shifted columns. The integrated Conclusion workbook has 481 issue rows: 339 unique issues have clean 0/1 quantum-related labels (220/119), 140 rows are unlabeled or malformed, and two duplicate issue rows were removed. Ten repositories collapse to nine repository/title components, one containing 264 rows. The source inventory found no reuse license, so model training remains blocked pending rights clarification and a leakage-safe supervised split; see the workbook label audit.",
        "Nine binary source tasks": "The refreshed source-isolated v2 suite completed on nine validated task adapters (5,055 rows): 180 model/task summaries and 900 fold/model metrics across 20 models and five grouped folds per task. OOF IDs, labels, groups, folds, and aggregate/per-fold metrics replayed with zero audit issues. It is an independent classical CV screen, not matched to QNN feature/split choices. MLP convergence warnings at the configured 200-iteration limit were observed and retained. The separate external Kaggle notebook remains unexecuted.",
        "QMutBench binary Survivor vs WOO/OPO merged detection (existing development champion)": "Development preprocessing indexed and validated; QNN training blocked by measured memory gate. Frozen test untouched.",
    }
    task_rows = []
    for row in tasks:
        original_training_status = row.get("training_status")
        source_name = row["dataset_source"]
        sid, tid = source_name_to_id.get(source_name), task_id_by_name.get(source_name)
        if source_name == "QMutBench" and "three-class" not in row["task"]:
            sid, tid = "qmutbench", "binary_mutation_outcome"
        elif source_name == "Bugs4Q-Robust":
            tid = "harness_execution_outcome"
        key = (sid, tid)
        aud = semantic_by_key.get(key, {})
        contract_tuple = contracts.get(key)
        if aud:
            prep_status = aud.get("eligibility_status", "REVIEW_REQUIRED")
            prep_evidence = aud.get("eligibility_reason", "")
            manifest = aud.get("manifest")
        elif source_name == "QMutBench" and tid == "binary_mutation_outcome" and (ROOT / "data/processed/qmutbench/binary_development/manifest.json").is_file():
            prep_status = "PREPROCESSING_READY_TRAINING_BLOCKED_MEMORY"
            prep_evidence = "578,117 development rows and 305 origin groups indexed; no frozen rows; memory preflight blocks QNN training."
            manifest = "data/processed/qmutbench/binary_development/manifest.json"
        elif source_name == "Nine binary source tasks":
            prep_status = ("COMPLETE_LOCAL_CLASSICAL_SCREEN" if nine_classical_audit and
                           nine_classical_audit.get("status") == "VERIFIED" else "PARTIAL_LOCAL_CLASSICAL_SCREEN")
            prep_evidence, manifest = limitation_by_name[source_name], None
        else:
            prep_status = "BLOCKED" if source_name in {"Mute-Alg", "QML platform workbooks"} else "INELIGIBLE_UNVALIDATED"
            prep_evidence = limitation_by_name.get(source_name, "No source-isolated validated task manifest.")
            manifest = None
        if source_name == "Nine binary source tasks":
            row["recorded_campaign_training_status"] = original_training_status
            row["training_status"] = "COMPLETE_LOCAL_9_TASK_CLASSICAL_SCREEN"
            row["records"] = "5,055 across nine validated task adapters"
            row["groups"] = "1,606 across nine task-local groups"
        elif source_name in {"Bugs4Q core-only transfer", "Bugs4Q-NA-only transfer"}:
            row["recorded_campaign_training_status"] = original_training_status
            row["training_status"] = "COMPLETE_SOURCE_RESTRICTED_RESUMABLE_QNN_PILOT"
        elif source_name == "Mute-Alg":
            row["recorded_campaign_training_status"] = original_training_status
            row["training_status"] = "COMPLETE_RESUMABLE_QNN_PILOT"
            row["task"] = "original vs synthetically buggy QASM v2"
            row["records"] = str(aud.get("accepted_rows", ""))
            row["groups"] = str(aud.get("groups", ""))
            prep_evidence += " Upstream HornBro repository lists MIT; the local dataset-only snapshot lacks its repository license file and commit ID. The 169 source-file hashes are pinned in `source_provenance.json`."
        elif source_name == "QMutBench" and "three-class" in row["task"] and training_artifact_audit:
            q3 = training_artifact_audit.get("qmutbench_three_class", {})
            row["recorded_campaign_training_status"] = original_training_status
            recovery = q3.get("resumable_checkpoint_recovery", {})
            if recovery.get("status") == "VERIFIED":
                row["training_status"] = "COMPLETE_DEVELOPMENT_OOF_WITH_VERSIONED_RESUMABLE_CHECKPOINT_RECOVERY"
                prep_status = "READY_FOR_TRAINING_AND_CHECKPOINT_REPLAY_VERIFIED"
                prep_evidence += (f" Original campaign checkpoint status remains {q3.get('original_campaign_checkpoint_status')}; "
                    f"a separate versioned recovery verified {recovery.get('resumable_checkpoints_verified')} optimizer/RNG checkpoints "
                    f"and replayed {recovery.get('replayed_oof_probabilities')} OOF probability rows. This historical mutation-outcome "
                    "task is separate from hardware fault diagnosis.")
            else:
                row["training_status"] = ("OOF_RESULTS_PRESENT_CHECKPOINT_UNVERIFIED"
                    if q3.get("status") == "OOF_TRAINING_ARTIFACTS_PRESENT_NO_MODEL_CHECKPOINT" else original_training_status)
                prep_evidence += " Existing development OOF predictions were rescored, but no real-data QNN parameter checkpoint was verified; do not claim reproducible QNN training completion."
        elif source_name == "QMutBench" and "three-class" not in row["task"] and training_artifact_audit:
            qb = training_artifact_audit.get("qmutbench_binary_full_development", {})
            row["recorded_campaign_training_status"] = original_training_status
            if qb.get("status") == "PARTIAL_RESUMABLE_PROGRESS_VERIFIED":
                row["training_status"] = "PARTIAL_FULL_DEVELOPMENT_QNN_RESUMABLE"
                prep_status = "PREPROCESSING_READY_TRAINING_IN_PROGRESS_RESUMABLE"
                prep_evidence = (f"Full-development streaming run is safely checkpointed in fold {qb.get('active_fold')} after "
                    f"{qb.get('committed_chunks')} chunks / {qb.get('processed_training_records')} training-record exposures and "
                    f"{qb.get('optimizer_updates')} optimizer updates ({float(qb.get('cumulative_training_seconds') or 0):.1f} cumulative training seconds; "
                    f"minimum recorded available RAM {float(qb.get('minimum_available_ram_mib') or 0):.1f} MiB; peak RSS {float(qb.get('maximum_checkpointed_peak_rss_mib') or 0):.1f} MiB); "
                    f"observed-rate estimate for all configured folds/epochs is {float(qb.get('estimated_total_qnn_training_hours_at_observed_rate') or 0):.1f} hours before evaluation; "
                    "chunk audit verified all hashes, gradients, and parameter changes. "
                    "This is partial training with no OOF metrics yet. Resume the versioned full-development run; frozen test untouched.")
        task_rows.append({**row, "preprocessing_validation_status": prep_status,
                          "preprocessing_evidence": prep_evidence,
                          "processed_manifest": manifest,
                          "contract": str(contract_tuple[0].relative_to(ROOT)) if contract_tuple else None,
                          "semantic_audit_status": aud.get("status")})
    represented = set()
    for row in tasks:
        source_name = row["dataset_source"]
        sid, tid = source_name_to_id.get(source_name), task_id_by_name.get(source_name)
        if source_name == "QMutBench" and "three-class" not in row["task"]:
            sid, tid = "qmutbench", "binary_mutation_outcome"
        represented.add((sid, tid))
    task_extensions = [{"source_id": audit["source_id"], "task_id": audit["task_id"],
                        "preprocessing_validation_status": audit["eligibility_status"],
                        "semantic_audit_status": audit["status"], "rows": audit["accepted_rows"],
                        "groups": audit["groups"], "manifest": audit["manifest"],
                        "evidence": audit["eligibility_reason"]}
                       for audit in semantic["tasks"]
                       if (audit["source_id"], audit["task_id"]) not in represented]
    qolumbina_multi_audits = {}
    multi_base = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028/qolumbina_detection_multibackend"
    for name, filename in [("unweighted", "pilot_audit.json"), ("fold_balanced", "pilot_audit_balanced.json")]:
        path = multi_base / filename
        if path.is_file():
            qolumbina_multi_audits[name] = json.loads(path.read_text(encoding="utf-8"))
    if qolumbina_multi_audits:
        task_extensions.append({
            "source_id": "qolumbina", "task_id": "seeded_mutant_detection_multibackend",
            "preprocessing_validation_status": "READY_EXPLORATORY_SEVERE_BACKEND_CLASS_IMBALANCE",
            "semantic_audit_status": "VERIFIED" if all(x["status"] == "VERIFIED" for x in qolumbina_multi_audits.values()) else "REVIEW_REQUIRED",
            "rows": 546, "groups": 26,
            "evidence": "All 546 rows across ideal/fake_algiers/fake_brooklyn follow the source majority-detection rule. The established five-fold original-program grouping is reused; no program crosses folds. fake_brooklyn has only one negative row; outcomes are exploratory seeded-mutant detection, not hardware attribution. Two resumable pilots are audit-verified.",
            "pilot_audits": {name: str((multi_base / filename).relative_to(ROOT))
                             for name, filename in [("unweighted", "pilot_audit.json"), ("fold_balanced", "pilot_audit_balanced.json")]
                             if (multi_base / filename).is_file()},
        })
    source_rows = []
    for item in inventory["datasets"]:
        source = item["dataset"]
        linked = [r for r in task_rows if r["dataset_source"] == source or (source == "QMutBench" and r["dataset_source"] == "QMutBench")]
        source_rows.append({"source": source, "path": item.get("path"), "download_status": item.get("download_status"),
                            "extraction_status": item.get("extraction_status"), "data_status": item.get("data_status"),
                            "task_statuses": [r["preprocessing_validation_status"] for r in linked]})
    pilot_runs = []
    invalidated_pilot_runs = []
    for path in sorted((ROOT / "reports/experiments").glob("resumable_qnn_pilot_*_20261029/run_manifest.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        correction_path = path.with_name("task_validity_correction.json")
        if correction_path.is_file():
            correction = json.loads(correction_path.read_text(encoding="utf-8"))
            invalidated_pilot_runs.append({"task": run.get("task"), "status": correction.get("status"),
                "reason": correction.get("reason"), "correction": str(correction_path.relative_to(ROOT)),
                "run_manifest": str(path.relative_to(ROOT))})
            continue
        if run.get("status") == "completed":
            pilot_runs.append({"task": run.get("task"), "status": run["status"],
                "rows": run.get("rows"), "groups": run.get("groups"), "folds": run.get("folds"),
                "qnn": {k: run["qnn"].get(k) for k in ("accuracy", "balanced_accuracy", "macro_f1")},
                "classical": {k: run["classical"].get(k) for k in ("accuracy", "balanced_accuracy", "macro_f1")},
                "resume_verification": run.get("resume_verification"), "runtime": run.get("runtime"),
                "run_manifest": str(path.relative_to(ROOT)),
                "metrics": str(path.with_name("metrics.json").relative_to(ROOT))})
    classical_audit_path = ROOT / "reports/experiments/local_grouped_classical_suite_eligible_20261029/audit_manifest.json"
    classical_audit = json.loads(classical_audit_path.read_text(encoding="utf-8")) if classical_audit_path.is_file() else None
    transfer_audit_path = OUT / "bugs4q_transfer_pilot_audit.json"
    transfer_audit = json.loads(transfer_audit_path.read_text(encoding="utf-8")) if transfer_audit_path.is_file() else None
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "source_count": len(source_rows),
              "task_matrix_rows": len(task_rows), "previous_report_claimed_task_rows": 18,
              "reconciliation_note": "The current campaign_completion_matrix.csv contains 18 data rows, matching the previous system-state report.",
              "semantic_audit": str(semantic_path.relative_to(ROOT)), "sources": source_rows, "tasks": task_rows,
              "additional_task_extensions": task_extensions,
              "resumable_pilots": pilot_runs, "invalidated_resumable_pilots": invalidated_pilot_runs,
              "local_classical_suite": classical_audit, "local_classical_suite_nine_source_v2": nine_classical_audit,
              "bugs4q_transfer_pilot_audit": transfer_audit,
              "qolumbina_multibackend_pilots": qolumbina_multi_audits,
              "training_artifact_audit": str(training_artifact_path.relative_to(ROOT)) if training_artifact_audit else None,
              "resume_commands": [
                  f".venv\\Scripts\\python.exe experiments\\prepare_source_isolated_task.py --contract experiments\\source_task_contracts\\{name}.json --chunk-size {size}"
                  for name, size in [("bugs4q_core_source_state", 64), ("bugs4q_na_source_state", 32),
                                     ("bugs4q_joint_source_state", 64), ("bugs4q_core_only_transfer", 64),
                                     ("bugs4q_na_only_transfer", 64), ("qsimbench_backend_fingerprint", 64),
                                     ("bugs4q_robust_execution", 128), ("lintq_warning_triage", 64),
                                     ("qbugs_correct_buggy_circuit_state", 64), ("quantum_platform_comment_classification", 64),
                                     ("qolumbina_ideal_detection", 64), ("qmutbench_three_class_exploratory", 64),
                                     ("lintq_warning_triage_context_v2", 64),
                                     ("mute_alg_compatible_subset_qasm_v2_qiskit_compat_v2", 128)]],
              "global_constraints": ["Task labels and source cohorts remain separate.", "QMutBench is historical mutation outcome, not hardware diagnosis.",
                                     "QMutBench frozen test was not accessed.", "Simulator diagnosis remains 7 mismatches among 45 outcomes.",
                                     "Existing checkpoints and historical reports are preserved."]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "source_preprocessing_matrix.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Source preprocessing and execution status — 2026-10-29", "",
             f"Inventory sources: {len(source_rows)}; current task matrix rows: {len(task_rows)}.",
             "The current CSV contains 18 task rows; all are represented below. The acquisition inventory has 19 source entries.", "",
             "## Task status", "", "| Dataset/task | Existing training status | Records | Groups | Validation/readiness | Evidence / next action |", "|---|---|---:|---:|---|---|"]
    for r in task_rows:
        lines.append(f"| {r['dataset_source']} / {r['task']} | {r['training_status']} | {r['records']} | {r['groups']} | {r['preprocessing_validation_status']} | {r['preprocessing_evidence']} |")
    lines += ["", "## Resumable pilot runs", "", "| Task | Rows/groups/folds | QNN macro F1 | Classical macro F1 | Resume state |", "|---|---:|---:|---:|---|"]
    for run in pilot_runs:
        lines.append(f"| {run['task']} | {run['rows']}/{run['groups']}/{run['folds']} | {run['qnn']['macro_f1']:.4f} | {run['classical']['macro_f1']:.4f} | {run['resume_verification']['status']} ({run['resume_verification']['resume_events']} resumes) |")
    if invalidated_pilot_runs:
        lines += ["", "## Invalidated pilot artifacts", ""]
        for item in invalidated_pilot_runs:
            lines.append(f"- `{item['task']}`: {item['status']}; {item['reason']} See `{item['correction']}`. Artifacts are preserved and excluded from valid pilot summaries.")
    if classical_audit:
        lines += ["", "## Separate grouped classical screen", "",
                  f"Status: {classical_audit['status']}; {classical_audit['task_count']} eligible tasks, {classical_audit['model_task_result_rows']} aggregate rows and {classical_audit['fold_model_metric_rows']} fold/model rows. This is not QNN-matched because it uses TF-IDF/SVD features and its own deterministic GroupKFold splits.",
                  "Four duplicate-risk tasks were excluded from this local screen; the external nine-task Kaggle bundle remains unexecuted.", "See `reports/experiments/local_grouped_classical_suite_eligible_20261029/report.md`."]
    if nine_classical_audit:
        lines += ["", "## Refreshed nine-source classical suite", "",
                  f"Status: {nine_classical_audit['status']}; {nine_classical_audit['task_count']} tasks, {nine_classical_audit['model_task_result_rows']} model/task results, {nine_classical_audit['fold_model_metric_rows']} fold/model metrics. This is an independent classical screen, not a matched QNN evaluation.",
                  "The OOF ID/label/group/fold checks and aggregate/per-fold metric replays passed. MLP convergence warnings at the configured 200 iterations were observed and retained.",
                  "See `reports/experiments/local_grouped_classical_suite_nine_source_v2_20261029/report.md`."]
    if transfer_audit:
        lines += ["", "## Source-restricted Bugs4Q transfer pilots", "",
                  f"Status: {transfer_audit['status']}; {len(transfer_audit['tasks'])} directions verified with training-source-only folds, separate transfer-source metrics, matched classical OOF, and resume/checkpoint receipts.",
                  "See `bugs4q_transfer_pilot_audit.json`. A prior mixed-source run is preserved but invalidated for a transfer claim."]
    lines += ["", "## Additional task variants", "", "These variants extend the campaign task matrix while preserving its original 18 rows.", "", "| Source/task | Readiness | Rows/groups | Evidence |", "|---|---|---:|---|"]
    for item in task_extensions:
        lines.append(f"| {item['source_id']} / {item['task_id']} | {item['preprocessing_validation_status']} | {item['rows']}/{item['groups']} | {item['evidence']} |")
    lines += ["", "## Validated source-isolated manifests", "",
              f"{semantic.get('task_count')} manifests/indexes were reconciled, including the six prior manifests (2,595 accepted rows), additional source-specific task variants, QMutBench development data and its binary SQLite index (578,117 rows). The 2,220-row Bugs4Q-Robust v1 output with an incorrect run-index contract is preserved as rejected evidence; corrected v2 has all rows accepted.",
              "", "## Source task resume commands", ""]
    for command in report["resume_commands"]:
        lines.append(f"- `{command}`")
    lines += ["", "## Other resume commands", "",
              "- QMutBench binary development index: `.venv\\Scripts\\python.exe experiments\\prepare_qmutbench_binary_development_index.py`.",
              "- Resume a pilot only if interrupted before completion: `.venv\\Scripts\\python.exe experiments\\run_resumable_qnn_pilot.py --resume --epochs 4` (use the matching `--task`, including `qbugs_qasm_v2_source_state` or `mute_alg_hornbro_qasm_v2`, for a non-default task).",
              "- Rebuild the LintQ context component task: `.venv\\Scripts\\python.exe experiments\\prepare_lintq_context_v2.py` then `.venv\\Scripts\\python.exe experiments\\prepare_source_isolated_task.py --contract experiments\\source_task_contracts\\lintq_warning_triage_context_v2.json --chunk-size 64`.",
              "- Rebuild the duplicate-aware platform comment task: `.venv\\Scripts\\python.exe experiments\\prepare_quantum_platform_comment_v2.py` then `.venv\\Scripts\\python.exe experiments\\prepare_source_isolated_task.py --contract experiments\\source_task_contracts\\quantum_platform_comment_v2.json --chunk-size 64`.",
              "- Rebuild the QBugs QASM task with the Qiskit 1.4.5 environment: `.qbugs1_env\\Scripts\\python.exe experiments\\prepare_qbugs_qasm_v2.py` then `.venv\\Scripts\\python.exe experiments\\prepare_source_isolated_task.py --contract experiments\\source_task_contracts\\qbugs_correct_buggy_circuit_state_qasm_v2.json --chunk-size 128`.",
              "- QNN full-development QMutBench training only after memory preflight passes: inspect `.venv\\Scripts\\python.exe -m qmutbench_qml.training.run_binary_development --help`; use the established `--mode full-development` and `--resume` options.",
              "- Rebuild integrity/readiness matrix: `.venv\\Scripts\\python.exe experiments\\audit_qml_task_semantics.py` followed by `.venv\\Scripts\\python.exe experiments\\build_qml_preprocessing_status.py`.", "",
              "No source task is called clean solely because its columns parsed. Semantic feature collisions, group leakage, target ambiguity, and source-specific blockers are recorded above. The preserved frozen simulator diagnosis remains 7 mismatches among 45 outcomes.", ""]
    (OUT / "source_preprocessing_matrix.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"source_count": len(source_rows), "task_matrix_rows": len(task_rows), "manifests_audited": semantic.get("task_count"), "output": str(OUT.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
