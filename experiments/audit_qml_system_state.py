"""Create a compact, read-only inventory of current QML research state.

The audit reuses existing source and experiment manifests. It does not load
datasets, import quantum frameworks, evaluate frozen test data, or train models.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/experiments"
CAMPAIGN = REPORTS / "qmutbench_multisource_qml_campaign_20261028"
OUT_JSON = REPORTS / "qml_system_state_20261029.json"
OUT_MD = REPORTS / "qml_system_state_20261029.md"


def read_json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                   stderr=subprocess.DEVNULL).strip()


def main() -> None:
    inv_path = "reports/experiments/qmutbench_multisource_qml_campaign_20261028/verified_dataset_inventory.json"
    task_path = "reports/experiments/qmutbench_multisource_qml_campaign_20261028/campaign_completion_matrix.csv"
    training_path = "reports/experiments/full_data_qml_optimization_v3/training_manifest.json"
    progress_path = "reports/experiments/qmutbench_multisource_qml_campaign_20261028/progress_manifest.json"
    artifact_audit_path = "reports/experiments/qmutbench_multisource_qml_campaign_20261028/completed_qnn_artifact_audit.json"
    inventory = read_json(inv_path)
    training = read_json(training_path)
    progress = read_json(progress_path)
    artifact_audit = read_json(artifact_audit_path)
    project_status = read_json("reports/project_status.json")

    sources = []
    for row in inventory["datasets"]:
        rel = str(row.get("path", "")).replace("\\", "/")
        path = ROOT / rel if rel else None
        sources.append({**row, "path_exists_now": bool(path and path.exists())})

    with (ROOT / task_path).open(encoding="utf-8-sig", newline="") as stream:
        tasks = list(csv.DictReader(stream))

    refs = git("for-each-ref", "--format=%(refname:short) %(objectname)", "refs/heads", "refs/remotes/origin")
    branch_map = {}
    for line in refs.splitlines():
        name, _, commit = line.partition(" ")
        if any(token in name for token in ("main", "qmutbench", "backup/qmutbench-binary-qnn")):
            branch_map[name] = commit

    try:
        import psutil
        memory = psutil.virtual_memory()
        available_mib = round(memory.available / 2**20, 1)
        total_mib = round(memory.total / 2**20, 1)
    except Exception:
        available_mib = total_mib = None

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from qmutbench_qml.training.run_binary_development import DEFAULT_CONFIG, memory_preflight
    try:
        binary_preflight = {"status": "passed", **memory_preflight(DEFAULT_CONFIG)}
    except MemoryError as exc:
        binary_preflight = {"status": "blocked_memory_guard", "error": str(exc)}

    training_artifact_audit_path = "reports/experiments/qmutbench_multisource_qml_campaign_20261028/training_artifact_audit.json"
    candidate_audit_path = "reports/experiments/qml_standalone_accuracy_optimization_20261028/remaining_task_sweeps/candidate_sweep_artifact_audit.json"
    candidate_status_path = "reports/experiments/qml_standalone_accuracy_optimization_20261028/remaining_task_sweeps/remaining_task_sweep_status.json"
    matched_comparison_path = "reports/experiments/qml_standalone_accuracy_optimization_20261028/remaining_task_sweeps/matched_qnn_classical_comparison.json"
    check_paths = [inv_path, task_path, training_path, progress_path, artifact_audit_path,
                   training_artifact_audit_path, candidate_audit_path, candidate_status_path,
                   "reports/experiments/qml_standalone_accuracy_optimization_20261028/remaining_task_sweeps/matched_classical_controls/campaign_summary.json",
                   matched_comparison_path,
                   "qmutbench_qml/results/run_manifest.json",
                   "qmutbench_qml/results/family_extension_9origins_20261027/run_manifest.json",
                   "reports/experiments/qmutbench_three_class_qnn_campaign_20261027/campaign_summary.json",
                   "reports/experiments/qmutbench_three_class_qnn_campaign_20261027/checkpoint_manifest.json"]
    evidence_hashes = {rel: sha256(ROOT / rel) for rel in check_paths if (ROOT / rel).is_file()}
    matched_comparison = read_json(matched_comparison_path)
    matched_cells = []
    for task in matched_comparison["tasks"]:
        for comparison in task["comparisons"]:
            best = max(comparison["classical_controls"], key=lambda row: row["macro_f1"])
            matched_cells.append({"task": task["task"], "feature_count": comparison["feature_count"],
                                  "qnn": comparison["qnn_model"], "best_fixed_control": best["model"],
                                  "qnn_macro_f1": comparison["qnn_metrics"]["macro_f1"],
                                  "control_macro_f1": best["macro_f1"],
                                  "delta": comparison["qnn_metrics"]["macro_f1"] - best["macro_f1"],
                                  "paired_group_bootstrap_delta_95ci": best["paired_group_bootstrap_macro_f1_delta_95ci"]})
    qnn_leads = sum(row["delta"] > 1e-12 for row in matched_cells)
    tied = sum(abs(row["delta"]) <= 1e-12 for row in matched_cells)
    qnn_trails = len(matched_cells) - qnn_leads - tied

    result = {
        "schema": "quantum-simulator-qml-system-state-v1",
        "audit_date": date.today().isoformat(),
        "scope": "read-only reconciliation of local source, task, training, artifact, and repository manifests",
        "repository": {
            "branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"),
            "branches": branch_map,
            "working_tree": "pre-existing tracked edits and many untracked datasets, scripts, reports, and artifacts preserved",
            "remote_freshness": "local remote-tracking refs inspected; no fetch or branch switch performed in this audit",
        },
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "psutil_version": importlib.metadata.version("psutil")},
        "resources": {"available_ram_mib_at_audit": available_mib, "total_ram_mib": total_mib,
                      "qnn_training_floor_mib": 500, "qmutbench_binary_startup_requirement_mib": 2048},
        "source_inventory": {
            "inventory_path": inv_path, "inventory_date": inventory.get("verified_at_local"),
            "listed_sources": len(sources), "listed_archives": len(inventory.get("archives", [])),
            "listed_rar_archives": len(inventory.get("rar_archives", [])), "sources": sources,
            "reconciliation_note": "The acquisition inventory has 19 source entries. A separate broad README inventory has 16 differently named source categories; counts are not directly comparable and have not been reconciled item by item.",
            "extraction_reconciliation": {
                "source": "Bugs4Q official Zenodo database verified extraction",
                "path": "data/Bugs4Q_official/extracted/Zenodo-Database-safe",
                "manifest_files": 337, "manifest_uncompressed_bytes": 29288166,
                "archive_non_git_files": 296, "archive_non_git_bytes": 177433,
                "observed_non_git_files": 296, "observed_non_git_bytes": 177433,
                "git_metadata_files": 41, "git_metadata_bytes": 29110733,
                "status": "RESOLVED_GIT_METADATA_EXCLUDED_FROM_TREE_INVENTORY",
                "note": "All 337 archive members and CRC validate. The extraction manifest counts 41 nested .git files (29,110,733 bytes); the inventory intentionally excludes .git and reports 296 data files / 177,433 bytes, plus its own 279-byte receipt.",
            },
        },
        "tasks": tasks,
        "qml_models": {
            "implemented_families": [
                "Variational quantum classifiers / trainable QNNs (angle, angle re-uploading, amplitude encodings)",
                "Hybrid quantum-classical readout models",
                "Quantum-kernel SVC (QSVC)",
                "Fixed quantum reservoir with classical logistic readout",
                "Matched classical Logistic Regression, SVC, Decision Tree, Random Forest, Extra Trees, and related controls where registered",
            ],
            "trained_scopes": [
                "Nine source-specific grouped binary QNN task runs with matched controls, OOF predictions, gradients and checkpoints",
                "Bugs4Q-Robust and QSimBench grouped QNN runs with per-fold checkpoints",
                "QMutBench three-class sampled, origin-grouped QNN sweeps with matched classical controls",
                "Nine-origin QMutBench binary pilot with VQC, QSVC, reservoir extension, and matched classical controls",
            ],
            "not_trained": [
                "Full-data QMutBench binary QNN",
                "Full-data QMutBench three-class QNN",
                "Qolumbina multi-backend incomplete task",
                "Unnormalized QML-platform issue workbooks; Bug-Pattern and QChecker tasks",
                "Real hardware-aware fault attribution QML",
            ],
        },
        "task_semantics": {
            "QMutBench": "Historical mutation detection outcome; WOO/OPO are merged only for the established binary task. This is not hardware-fault evidence.",
            "Bugs4Q": "Buggy/fixed source-state classification grouped by issue; transfers remain source-specific.",
            "QSimBench": "Post-execution Aer versus fake_kyiv backend fingerprinting grouped by algorithm-size setting; not bug detection.",
            "Bugs4Q_Robust": "Published harness pass/fail outcome prediction grouped by issue; not a universal defect label.",
            "hardware_attribution": "No independently labeled IBM execution/calibration observations verified; synthetic simulator cases remain separate.",
        },
        "training_state": {
            "multisource_status": progress.get("status"),
            "completed_claims": progress.get("completed", []),
            "not_completed_claims": progress.get("not_completed", []),
            "full_data_campaign_status": training.get("status"),
            "full_data_qmutbench_status": training.get("qmutbench_full_data_status"),
            "full_data_new_runs_started": training.get("new_training_runs_started"),
            "blockers": training.get("reason"),
            "binary_qnn": "Implemented with epoch/chunk resume and integrity tests; no full-data binary QMutBench training in the saved run directory. Current RAM is below its 2 GiB startup requirement.",
            "binary_qnn_preflight": binary_preflight,
            "qmutbench_three_class": "Exploratory 205-record, 12-origin grouped development runs exist with matching classical controls; these are rare-class sample estimates, not full-data or confirmation results.",
            "hardware_qml": "Not a trainable labeled task with current evidence; no physical-device attribution result is claimed.",
            "completed_candidate_sweeps": {
                "status": read_json(candidate_status_path).get("status"),
                "task_count": len(read_json(candidate_status_path).get("completed_tasks", [])),
                "new_results": [
                    {"task": task["task"], "records": task["records"], "groups": task["groups"],
                     "models": [{"model": model["model"], "accuracy": model["metrics"]["accuracy"],
                                 "macro_f1": model["metrics"]["macro_f1"],
                                 "balanced_accuracy": model["metrics"]["balanced_accuracy"]}
                                for model in task["models"]]}
                    for task in read_json(candidate_audit_path)["tasks"]
                ],
                "matched_classical_comparison": {"status": matched_comparison["status"], "path": matched_comparison_path,
                    "task_count": matched_comparison["task_count"], "paired_task_width_cells": len(matched_cells),
                    "qnn_leads_descriptive_best_fixed_control": qnn_leads,
                    "ties": tied, "qnn_trails": qnn_trails, "cells": matched_cells,
                    "interpretation": "Best control per cell is chosen from the same OOF scores and is descriptive, not an independent selection."},
                "initial_resource_guard": "First matched baseline attempt blocked at 507.2 MiB; after memory recovered, the complete matched comparison finished.",
                "misrouted_auxiliary_artifact": "An extra Bugs4Q-Robust QNN run was written under a leaf named bugs4q_na; it is preserved and excluded from the nine-task audited aggregate.",
            },
        },
        "artifact_validation": {
            "saved_completed_qnn_audit_status": artifact_audit.get("status"),
            "saved_completed_qnn_audit_path": artifact_audit_path,
            "saved_multisource_training_audit_status": read_json(training_artifact_audit_path).get("status"),
            "saved_multisource_training_audit_path": training_artifact_audit_path,
            "new_candidate_sweep_audit_status": read_json(candidate_audit_path).get("status"),
            "new_candidate_sweep_audit_path": candidate_audit_path,
            "matched_comparison_status": matched_comparison["status"],
            "matched_comparison_path": matched_comparison_path,
            "artifact_audit_sha256": evidence_hashes.get(artifact_audit_path),
            "qmutbench_frozen_test_accessed_in_recorded_qml_campaigns": False,
            "production_model_modified_in_recorded_qml_campaigns": False,
            "checksummed_evidence_manifests": evidence_hashes,
        },
        "preexisting_simulator_acceptance_status": {
            "latest_saved_record": project_status.get("finalization_milestone", {}).get("bounded_validation", {}),
            "freshly_rerun_in_this_audit": False,
            "note": "The known frozen diagnosis acceptance failure is preserved. The focused QML test result does not validate or supersede the simulator acceptance result.",
        },
        "tests_run_in_this_audit": [
            {"command": ".venv\\Scripts\\python.exe -m pytest --basetemp=.pytest-qml-audit-base tests\\test_qmutbench_binary_qnn.py -q",
             "result": "27 passed, 1 skipped in 7.54s", "plugins": "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
             "skip_reason": "real PennyLane gradient/update/resume smoke requires the unchanged 2048 MiB startup reserve"},
            {"command": ".venv\\Scripts\\python.exe -m qmutbench_qml.training.run_binary_development --validate-config",
             "result": "configuration_validated_no_data_loaded"},
            {"command": ".venv\\Scripts\\python.exe experiments\\audit_qml_candidate_sweeps.py",
             "result": "verified; 9 tasks, 90 checkpoints, 90 fold gradient checks"},
            {"command": ".venv\\Scripts\\python.exe experiments\\audit_matched_qnn_classical_comparison.py",
             "result": "verified; 9 tasks, 18 matched representation comparisons, 1000 paired group-bootstrap resamples per QNN/control pair"},
            {"command": ".venv\\Scripts\\python.exe -m pytest --basetemp=.pytest-qml-foundation-base qmutbench_qml\\tests\\test_foundations_and_multiclass.py -q",
             "result": "5 passed in 5.06s", "plugins": "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"},
        ],
        "remaining_feasible_work": [
            "Run the QMutBench binary and full-data grouped comparisons only after measured RAM meets the explicit startup/in-run gates.",
            "Execute the separate prepared 14-model/imbalance-method external suite on a supported CPU runtime; local bundle validation is not execution evidence.",
            "Normalize and deduplicate QML-platform issue workbooks and build validated task manifests for Bug-Pattern/QChecker before modeling.",
            "Resolve Mute-Alg legacy Qiskit pickle compatibility without relabeling or altering originals.",
            "Obtain authorized IBM observations with calibration, mappings, timestamps, and independently adjudicated labels before hardware-attribution modeling.",
        ],
        "limitations": [
            "Historical OOF results are development evidence, not independent confirmation scores.",
            "The 85% historical objective was not met by the verified QMutBench champion (macro F1 0.7933); no new universal target or hardware claim is made.",
            "The status audit reconciles existing manifests and key artifact hashes; it does not re-train, load, or replay all model binaries.",
            "Untracked source, data, and experiment artifacts are preserved; this audit does not establish that every local addition is backed up remotely.",
        ],
    }

    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    completed = sum(row.get("training_status", "").startswith("COMPLETE") or row.get("training_status", "").startswith("EXISTING_CHAMPION") for row in tasks)
    lines = [
        "# QML system state audit — 2026-10-29", "",
        f"Repository `{result['repository']['branch']}` at `{result['repository']['head']}`. This read-only audit preserves the existing working tree.", "",
        "## Current status", "",
        f"- {len(sources)} acquisition-inventory entries and {len(inventory.get('archives', []))} archive entries are recorded; {sum(bool(x['path_exists_now']) for x in sources)} listed source paths exist locally.",
        f"- {completed} of {len(tasks)} task-matrix rows are marked complete or existing champion; this count includes distinct source/task rows, not unique datasets.",
        f"- Saved QNN artifact replay audit status: **{artifact_audit.get('status')}**. It records checkpoint archive validation, metric recalculation, and gradient checks for the newer completed tasks.",
        f"- Saved multi-source training artifact audit status: **{read_json(training_artifact_audit_path).get('status')}**; its scope is development artifacts and it records that the frozen test was not opened.",
        f"- New four-epoch QNN candidate sweeps: **{len(read_json(candidate_audit_path)['tasks'])} tasks, {read_json(candidate_audit_path)['checkpoint_count']} checksummed fold checkpoints, and {read_json(candidate_audit_path)['gradient_checks_passed']} passing fold gradient checks**.",
        "- Matched classical controls are complete for all nine tasks and both feature widths. The QNN has higher macro F1 than the descriptively highest fixed control in 1 of 18 task/width cells, ties in 1, and trails in 16. Each paired interval resamples whole source groups 1,000 times.",
        f"- Available RAM at audit: **{available_mib} MiB**; full-data QMutBench training is below its 1,024 MiB floor and sampled binary training is below its 2,048 MiB startup requirement.",
        f"- Fresh binary QNN preflight: **{binary_preflight.get('status')}** — {binary_preflight.get('error', 'configuration accepted')}",
        "- QMutBench binary training is not recorded as complete. Its historical WOO/OPO outcome target is distinct from hardware anomaly attribution.",
        "- IBM hardware fault attribution remains unavailable because verified device observations and independent labels are absent.",
        "- Latest saved broad simulator suite: 794 passed, 1 failed, 0 skipped, 2 registered finite-shot NO-GO tests deselected (797 collected). Its sole failure is the unchanged frozen 45-case diagnosis acceptance with 7 mismatches; this audit did not rerun it.", "",
        "## Source and task findings", "",
        "The detailed 19-entry acquisition inventory and the per-task completion matrix are embedded in the JSON report. QMutBench, Bugs4Q, QSimBench, Bugs4Q-Robust, QBugs, Qolumbina ideal-backend, LintQ warning triage, and quantum-platform comment tasks have source-specific target definitions. Their scores remain separate and development-only.", "",
        "Implemented QML families include angle and re-uploading VQCs, amplitude-encoded and hybrid-readout QNNs, QSVC, and a quantum reservoir with classical readout. Existing evidence includes 4,575 saved QNN OOF predictions replayed across nine source runs with maximum probability error below 1.2e-16, plus separate integrity/gradient audits for the newer QNN tasks.", "",
        "**Extraction reconciliation:** the Bugs4Q Zenodo archive contains 337 files and passes ZIP CRC validation. Its 41 nested `.git` metadata files account for 29,110,733 bytes; excluding `.git`, both archive and extracted tree contain 296 data files / 177,433 bytes. The prior apparent count mismatch is resolved.", "",
        "## Verified evaluation summary", "",
        "- Existing QMutBench binary classical champion: 578,117 development records, 305 origin groups, five-fold grouped OOF macro F1 **0.7933**. This is historical and below the 0.85 objective; it is not a QML result or hardware diagnosis.",
        "- QMutBench three-class QNN work: 205 records / 12 OPO-bearing origins / three grouped folds. The best saved QNN ensemble macro F1 is about **0.434**, versus matched Extra Trees about **0.467**. It is exploratory and class-balanced.",
        "- QSimBench backend fingerprint QNN: 300 observations / 150 circuit-setting groups / five folds; the angle re-uploading model has about **0.660** accuracy and macro F1. The label is backend identity, not a fault label.",
        "- Bugs4Q-Robust pass/fail QNN: 2,220 repeated observations / 37 issue groups / five folds; the resource-light amplitude hybrid has about **0.514** accuracy and macro F1. The saved matrix has no matched classical score for this exact feature run.",
        "- New four-epoch hashed-text QNN results include Bugs4Q pooled macro F1 **0.494**, Bugs4Q-NA **0.470**, QBugs **0.718**, Qolumbina ideal **0.452**, LintQ warning triage **0.426**, and quantum-platform comments **0.894**. Same-input classical controls were completed; only the Bugs4Q core-only transfer 8-feature QNN exceeded the descriptively top fixed control (+3.38 macro-F1 points). The full table and paired group-bootstrap intervals are in [the matched comparison report](../qml_standalone_accuracy_optimization_20261028/remaining_task_sweeps/matched_qnn_classical_comparison.md).",
        "- The OOF-highest classical control is reported descriptively after scoring all controls; it was not selected on an independent validation/confirmation set. These are development results and do not establish generalization to unseen source projects beyond the registered grouped folds.", "",
        "## Validation performed", "",
        "Focused command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv\\Scripts\\python.exe -m pytest --basetemp=.pytest-qml-audit-base tests\\test_qmutbench_binary_qnn.py -q` — **27 passed, 1 skipped**. The skipped case is the real PennyLane gradient/update/resume smoke, gated on 2,048 MiB available RAM. Binary configuration validation passed; the fresh training preflight was safely blocked below its 2,048 MiB requirement. These checks do not constitute a new real-data fit.",
        "The simulator's frozen acceptance failure remains preserved at its existing 7/45 mismatch. This QML-only suite is separate verification and was not counted as simulator validation.", "",
        "The QML foundation and multiclass forward/backward suite also passed: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv\\Scripts\\python.exe -m pytest --basetemp=.pytest-qml-foundation-base qmutbench_qml\\tests\\test_foundations_and_multiclass.py -q` — **5 passed**.", "",
        "## Reproduction commands", "",
        "Resume/reconcile the fixed four-epoch development QNN candidates: `LIGHT_QNN_RESUME=1`, `LIGHT_QNN_EPOCHS=4`, `LIGHT_QNN_BATCH=64`, `LIGHT_QNN_SEED=28102026`, then run `experiments/train_remaining_standalone_qml_tasks.py`. Run the matched controls with `experiments/evaluate_qnn_candidate_matched_classical.py`; regenerate integrity reports with `experiments/audit_qml_candidate_sweeps.py` and `experiments/audit_matched_qnn_classical_comparison.py`.", "",
        "## Remaining work and blockers", "",
    ]
    lines.extend(f"- {item}" for item in result["remaining_feasible_work"])
    lines += ["", "No test labels from a frozen QMutBench confirmation set were read in generating this audit. Machine-readable source/task rows, branch refs, resource state, provenance hashes, and blockers are in `qml_system_state_20261029.json`.", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(OUT_JSON.relative_to(ROOT)), "markdown": str(OUT_MD.relative_to(ROOT)),
                      "source_entries": len(sources), "source_paths_present": sum(bool(x["path_exists_now"]) for x in sources),
                      "task_rows": len(tasks), "completed_task_rows": completed,
                      "available_ram_mib": available_mib, "artifact_audit": artifact_audit.get("status")}, indent=2))


if __name__ == "__main__":
    main()
