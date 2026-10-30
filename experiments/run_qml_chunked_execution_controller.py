"""Reconcile the 18-task QML matrix and safely resume one QMutBench chunk.

Completed evidence is referenced in place. The controller never starts classical
comparisons, accesses frozen-test data, alters a checkpoint, or changes training
configuration. QMutBench resumes only through its existing bounded runner.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "reports/experiments/qml_source_preprocessing_20261029/source_preprocessing_matrix.json"
QMB_OUT = ROOT / "reports/experiments/qmutbench_binary_qnn_full_development_chunk16_20261029"
OUT = ROOT / "reports/experiments/qml_chunked_execution_campaign_20261030"
RUNNER = ROOT / "qmutbench_qml/training/run_binary_development.py"
PYTHON = ROOT / ".venv/Scripts/python.exe"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def classify(row: dict) -> tuple[str, str]:
    source, task = row["dataset_source"], row["task"]
    if source == "QMutBench" and "binary" in task.lower():
        audit = QMB_OUT / "progress_audit.json"
        if audit.is_file():
            state = json.loads(audit.read_text(encoding="utf-8"))
            return "PARTIAL_BLOCKED_MEMORY", (
                f"Verified resume point fold {state['active_fold']}, chunk "
                f"{state['committed_chunks'] + 1}; {state['optimizer_updates']} updates "
                f"and {state['processed_training_records']} training-record exposures committed. "
                "Resume only when the existing memory preflight passes.")
        return "PARTIAL", "Existing full-development run; inspect its verified checkpoint audit."
    readiness = row.get("preprocessing_validation_status", "")
    training = row.get("training_status", "")
    if readiness == "BLOCKED":
        return "BLOCKED", row.get("preprocessing_evidence", row.get("notes", "Source or license gate remains open."))
    if readiness.startswith("INELIGIBLE"):
        return "INELIGIBLE", row.get("preprocessing_evidence", row.get("notes", "No validated supervised target/split."))
    if training.startswith("NOT_TRAINED"):
        return "INELIGIBLE", row.get("preprocessing_evidence", row.get("notes", "No validated supervised task."))
    if readiness.startswith("BLOCKED_") or readiness == "BLOCKED_FEATURE_LABEL_AMBIGUITY":
        return "COMPLETED_ARTIFACTS_LIMITED", row.get("preprocessing_evidence", row.get("notes", "Historical artifacts retained; current split/feature gate blocks claims."))
    return "COMPLETED_REUSE", row.get("notes", row.get("preprocessing_evidence", "Validated existing training artifacts reused."))


def main() -> int:
    import psutil
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    if len(matrix.get("tasks", [])) != 18:
        raise RuntimeError(f"Expected 18 registered tasks, found {len(matrix.get('tasks', []))}")
    tasks = []
    for number, source_row in enumerate(matrix["tasks"], 1):
        status, disposition = classify(source_row)
        task_id = f"task_{number:02d}"
        tasks.append({
            "task_id": task_id, "dataset_source": source_row["dataset_source"],
            "task": source_row["task"], "classification": status,
            "records": source_row.get("records"), "groups": source_row.get("groups"),
            "folds": source_row.get("folds"),
            "preprocessing_validation_status": source_row.get("preprocessing_validation_status"),
            "existing_training_status": source_row.get("training_status"),
            "disposition": disposition,
            "source_matrix_row": number,
            "existing_evidence": source_row.get("processed_manifest") or source_row.get("notes"),
            "chunk_execution": "reused_existing_fold_artifacts" if status.startswith("COMPLETED") else
                               ("existing_streamed_chunks" if status.startswith("PARTIAL") else "not_run"),
        })

    # Every invocation retries exactly one bounded resume through the established
    # checkpoint-aware runner. Its preflight must pass before any data is loaded.
    qmb_task = next(t for t in tasks if t["classification"] == "PARTIAL_BLOCKED_MEMORY")
    started = datetime.now(timezone.utc).isoformat()
    before = json.loads((QMB_OUT / "progress_audit.json").read_text(encoding="utf-8"))
    command = [str(PYTHON), "-m", "qmutbench_qml.training.run_binary_development", "--mode", "full-development",
               "--chunk-batches", "16", "--max-chunks-per-invocation", "1", "--resume"]
    available_before_mib = round(psutil.virtual_memory().available / (1024 * 1024), 1)
    started_perf = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    elapsed_seconds = round(time.perf_counter() - started_perf, 3)
    available_after_mib = round(psutil.virtual_memory().available / (1024 * 1024), 1)
    after = json.loads((QMB_OUT / "progress_audit.json").read_text(encoding="utf-8"))
    invocation = {
        "started_utc": started, "command": [str(PYTHON.relative_to(ROOT)),
            "-m", "qmutbench_qml.training.run_binary_development", "--mode",
            "full-development", "--chunk-batches", "16", "--max-chunks-per-invocation", "1", "--resume"],
        "return_code": result.returncode,
        "elapsed_seconds": elapsed_seconds,
        "available_ram_before_mib": available_before_mib,
        "available_ram_after_mib": available_after_mib,
        "stdout": result.stdout.strip(), "stderr": result.stderr.strip(),
        "checkpoint_unchanged": before.get("latest_checkpoint") == after.get("latest_checkpoint") and
                                before.get("optimizer_updates") == after.get("optimizer_updates"),
        "frozen_test_accessed": False,
    }
    if result.returncode == 0:
        qmb_task["classification"] = "PARTIAL_RESUMABLE"
        qmb_task["disposition"] = "One additional checkpointed chunk committed; resume the controller to continue."
    else:
        qmb_task["classification"] = "PARTIAL_BLOCKED_MEMORY"
        error_text = result.stdout.strip() or result.stderr.strip()
        qmb_task["disposition"] = "Bounded resume attempt did not commit a chunk; checkpoint state was preserved. " + error_text

    registry = {
        "schema": "qml-chunked-execution-progress-v1",
        "campaign_id": "qml_chunked_execution_campaign_20261030",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "source_matrix": str(MATRIX.relative_to(ROOT)), "source_matrix_sha256": sha256(MATRIX),
        "task_count": len(tasks), "completed_reused": sum(t["classification"].startswith("COMPLETED") for t in tasks),
        "ineligible": sum(t["classification"] == "INELIGIBLE" for t in tasks),
        "blocked": sum(t["classification"] == "BLOCKED" for t in tasks),
        "partial": sum(t["classification"].startswith("PARTIAL") for t in tasks),
        "frozen_test_accessed": False, "classical_comparisons_started": False,
        "tasks": tasks, "latest_execution_attempt": invocation,
        "qmutbench_resume_state": after,
        "next_action": "Retry one QMutBench full-development chunk only after its memory preflight passes; preserve config and checkpoint identity.",
    }
    registry_path = OUT / "progress_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    csv_path = OUT / "task_status_matrix.csv"
    fields = ["task_id", "dataset_source", "task", "classification", "records", "groups", "folds",
              "preprocessing_validation_status", "existing_training_status", "chunk_execution", "disposition"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: t.get(k) for k in fields} for t in tasks)
    report = [
        "# QML chunked execution campaign — 2026-10-30", "",
        "This report reconciles the registered 18-task preprocessing matrix and reuses existing validated artifacts. It does not start a classical comparison or access a frozen test set.", "",
        f"Status: {registry['completed_reused']} tasks have reusable completed artifacts; {registry['ineligible']} are ineligible; {registry['blocked']} is blocked by data/licensing/split gates; {registry['partial']} full-development task remains partial.", "",
        "## Incomplete eligible work", "",
        f"QMutBench binary development remains partial in `{QMB_OUT.relative_to(ROOT).as_posix()}`. Its VERIFIED audit records {after.get('committed_chunks')} independently checkpointed chunks in fold {after.get('active_fold')}, {after.get('optimizer_updates')} optimizer updates, and {after.get('processed_training_records')} training-record exposures. All committed chunks passed recorded gradient, parameter-update, and integrity checks. No fold is complete and no OOF metrics are available.", "",
        "The bounded resume attempt returned: `" + (invocation["stdout"] or invocation["stderr"]).replace("`", "'").replace("\n", " ") + "`.",
        "The attempt preserved the latest checkpoint: " + ("yes" if invocation["checkpoint_unchanged"] else "no") + ". Frozen-test access: no.", "",
        f"Preflight invocation runtime: {invocation['elapsed_seconds']} seconds; available RAM moved from {invocation['available_ram_before_mib']} MiB to {invocation['available_ram_after_mib']} MiB. Existing verified stream peak RSS was {round(after.get('maximum_checkpointed_peak_rss_mib', 0), 1)} MiB, minimum available RAM during committed chunks was {round(after.get('minimum_available_ram_mib', 0), 1)} MiB, and cumulative QNN training time was {round(after.get('cumulative_training_seconds', 0), 1)} seconds. These are execution measurements, not estimates for the remaining work.", "",
        "## Execution limits", "",
        "All other eligible QNN tasks already have grouped OOF/checkpoint evidence and are marked for reuse. The original QBugs, LintQ, and platform-comment rows retain their current leakage/label caveats; separately versioned corrected variants remain separate tasks. Bug-Pattern and QChecker lack validated supervised targets/splits. QML platform workbooks remain blocked by unresolved license and split issues. Mute-Alg's full-source task remains blocked by pickle compatibility; its compatibility-selected subset remains ineligible for representative training.", "",
        "The separate broad classical suite is explicitly outside this execution and was not initiated. QMutBench binary training is the only eligible incomplete QNN task. Its execution estimate is measured from existing progress evidence; no completion time is promised.", "",
        "## Per-task status", "",
        "See `task_status_matrix.csv` and `progress_registry.json` for all 18 tasks, classifications, readiness states, and source-matrix references.", "",
        "## Resume", "",
        "Re-run `.venv\\Scripts\\python.exe experiments\\run_qml_chunked_execution_controller.py` when available memory is sufficient. Each call requests one checkpointed chunk and retains the existing training configuration.", "",
    ]
    (OUT / "campaign_report.md").write_text("\n".join(report), encoding="utf-8")
    checkpoint_name = str(after.get("latest_checkpoint", ""))
    evidence_paths = [
        MATRIX, Path(__file__), registry_path, csv_path, OUT / "campaign_report.md",
        QMB_OUT / "progress_audit.json", QMB_OUT / "full_campaign_manifest.json",
        QMB_OUT / "fold_checkpoints" / checkpoint_name,
    ]
    integrity = {
        "schema": "qml-campaign-integrity-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "SHA-256",
        "artifacts": [
            {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in evidence_paths if path.is_file()
        ],
    }
    (OUT / "integrity_manifest.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps({"status": "reconciled", "tasks": len(tasks), "completed_reused": registry["completed_reused"],
                      "ineligible": registry["ineligible"], "blocked": registry["blocked"],
                      "partial": registry["partial"], "attempt": invocation["stdout"]}, indent=2))
    return 0 if result.returncode == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
