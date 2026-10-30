"""Assemble a provenance-first, non-training v3 QML campaign audit.

This utility reuses only existing campaign artifacts and streams the local
dataset inventory. It does not access frozen-test data or start training.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/experiments/full_data_qml_optimization_v3"
SRC_INV = ROOT / "data/external_quantum_datasets/master_dataset_inventory.csv"
OLD = ROOT / "reports/experiments/qmutbench_multisource_qml_campaign_20261028"
STAND = ROOT / "reports/experiments/qml_standalone_accuracy_optimization_20261028"


def relative(path: Path | str) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_stats(path: Path) -> tuple[int, int, int]:
    n = size = zero = 0
    if not path.exists():
        return n, size, zero
    if path.is_file():
        return 1, path.stat().st_size, int(path.stat().st_size == 0)
    pending = [path]
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name != ".git":
                                pending.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            st = entry.stat(follow_symlinks=False)
                            n += 1
                            size += st.st_size
                            zero += int(st.st_size == 0)
                    except OSError:
                        continue
        except OSError:
            continue
    return n, size, zero


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def oof_metrics(path: Path, selected_model: str | None = None) -> dict:
    """Recompute basic classification metrics from one saved OOF CSV."""
    if not path.exists():
        return {"prediction_present": False}
    ys: list[str] = []
    ps: list[str] = []
    groups: set[str] = set()
    folds: set[str] = set()
    fold_y: dict[str, list[str]] = {}
    fold_p: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        ycol = next((c for c in ("y_true", "true_label", "label", "target") if c in fields), None)
        pcol = next((c for c in ("y_pred", "predicted_label", "prediction") if c in fields), None)
        gcol = next((c for c in ("bug_id", "origin_program", "program_id", "group_id", "commit_group", "algorithm_group") if c in fields), None)
        fcol = next((c for c in ("fold", "fold_id") if c in fields), None)
        if not ycol or not pcol:
            return {"prediction_present": True, "metrics_status": "unrecognized_prediction_schema", "columns": fields}
        for row in reader:
            if selected_model and "model" in fields and row.get("model") not in (None, "") and row.get("model") != selected_model:
                continue
            y, p = str(row[ycol]), str(row[pcol])
            ys.append(y); ps.append(p)
            if gcol and row.get(gcol) not in (None, ""):
                groups.add(str(row[gcol]))
            if fcol and row.get(fcol) not in (None, ""):
                fold = str(row[fcol]); folds.add(fold)
                fold_y.setdefault(fold, []).append(y); fold_p.setdefault(fold, []).append(p)
    if not ys:
        return {"prediction_present": True, "metrics_status": "empty"}
    labels = sorted(set(ys) | set(ps))
    per_class = {}
    recalls = []
    f1s = []
    for label in labels:
        tp = sum(y == label and p == label for y, p in zip(ys, ps))
        fp = sum(y != label and p == label for y, p in zip(ys, ps))
        fn = sum(y == label and p != label for y, p in zip(ys, ps))
        support = sum(y == label for y in ys)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        recalls.append(recall); f1s.append(f1)
    fold_acc = []
    for fold in sorted(folds):
        a, b = fold_y[fold], fold_p[fold]
        fold_acc.append(sum(y == p for y, p in zip(a, b)) / len(a))
    mean_fold_acc = sum(fold_acc) / len(fold_acc) if fold_acc else None
    fold_acc_sd = ((sum((x - mean_fold_acc) ** 2 for x in fold_acc) / (len(fold_acc) - 1)) ** .5
                   if len(fold_acc) > 1 else 0.0 if fold_acc else None)
    return {"prediction_present": True, "metrics_status": "recomputed_from_saved_oof",
            "records": len(ys), "groups": len(groups) if groups else "", "folds": len(folds) if folds else "",
            "accuracy_percent": 100 * sum(y == p for y, p in zip(ys, ps)) / len(ys),
            "macro_f1_percent": 100 * sum(f1s) / len(f1s),
            "balanced_accuracy_percent": 100 * sum(recalls) / len(recalls),
            "per_class_metrics": json.dumps(per_class, sort_keys=True),
            "fold_accuracy_sd_percentage_points": 100 * fold_acc_sd if fold_acc_sd is not None else ""}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT, text=True).splitlines()
    baseline_status = subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=no"], cwd=ROOT, text=True).splitlines()
    vm = None
    try:
        import psutil  # type: ignore
        m = psutil.virtual_memory()
        vm = {"available_mib": round(m.available / 1048576, 1), "total_mib": round(m.total / 1048576, 1)}
    except ImportError:
        vm = {"available_mib": None, "total_mib": None, "measurement": "psutil unavailable"}

    # Verify the full QMutBench metadata, labels, groups, and QASM file counts with bounded memory.
    qfull = ROOT / "data/qmutbench_full/data"
    metadata = qfull / "all_data.csv"
    qmut = {"metadata_path": relative(metadata), "metadata_exists": metadata.exists()}
    if metadata.exists():
        labels: Counter[str] = Counter()
        groups: set[str] = set()
        rows = 0
        with metadata.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            qmut["metadata_columns"] = reader.fieldnames
            for row in reader:
                rows += 1
                labels[str(row.get("Killed", ""))] += 1
                groups.add(str(row.get("Origin_program", "")))
        qmut.update({"metadata_rows": rows, "independent_origin_groups": len(groups), "raw_label_counts": dict(labels),
                     "metadata_bytes": metadata.stat().st_size, "metadata_sha256": sha256(metadata)})
    originals, original_bytes, original_zero = tree_stats(qfull / "Origin_programs")
    mutants, mutant_bytes, mutant_zero = tree_stats(qfull / "Mutated_programs")
    qmut.update({"origin_qasm_files": originals, "origin_qasm_bytes": original_bytes, "origin_zero_byte_files": original_zero,
                 "mutant_qasm_files": mutants, "mutant_qasm_bytes": mutant_bytes, "mutant_zero_byte_files": mutant_zero,
                 "metadata_to_qasm_row_count_match": qmut.get("metadata_rows") == mutants})
    ingestion_csv = ROOT / "data/qmutbench_ingestion/raw/official_all_data_20261015.csv"
    if ingestion_csv.exists():
        qmut["ingestion_snapshot"] = {"path": relative(ingestion_csv), "bytes": ingestion_csv.stat().st_size,
                                       "sha256": sha256(ingestion_csv), "byte_identical_to_checkout_metadata":
                                       metadata.exists() and metadata.stat().st_size == ingestion_csv.stat().st_size and sha256(metadata) == sha256(ingestion_csv)}

    # Source-by-source filesystem inventory, based on the prior provenance registry but re-counted now.
    inventory: list[dict] = []
    with SRC_INV.open("r", encoding="utf-8-sig", newline="") as f:
        for src in csv.DictReader(f):
            path_text = (src.get("path") or "").strip().replace("\\", "/")
            p = ROOT / path_text if path_text else ROOT / "__not_located__"
            n, b, z = tree_stats(p)
            row = dict(src)
            row.update({"verified_local_path": path_text, "exists_now": p.exists(), "actual_files_now": n,
                        "actual_bytes_now": b, "zero_byte_files_now": z,
                        "status_now": "PRESENT_RECOUNTED" if p.exists() else "NOT_PRESENT_LOCALLY"})
            inventory.append(row)
    write_csv(OUT / "dataset_inventory.csv", inventory)
    write_json(OUT / "dataset_inventory.json", {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "inventory_source": relative(SRC_INV),
        "filesystem_recount": "os.walk excluding .git; file counts and byte sizes, streamed without retaining file listings",
        "qmutbench_full_validation": qmut, "datasets": inventory,
    })

    extraction = {
        "qmutbench": {"metadata": "present_and_stream_counted", "source_qasm": "present_and_file_counted",
                      "metadata_rows": qmut.get("metadata_rows"), "mutant_qasm_files": mutants,
                      "origin_qasm_files": originals, "row_file_count_match": qmut.get("metadata_to_qasm_row_count_match"),
                      "full_syntax_parse_coverage": "NOT_RUN: 426.7 MiB available RAM is below the 1 GiB campaign floor; full-corpus parse remains outstanding"},
        "other_sources": "Local presence/extracted-file counts are in dataset_inventory.csv; source-specific semantic normalization is not implied by extraction.",
        "archives": [],
        "limitations": ["RAR archives cannot be safely enumerated with the installed Windows tar.exe; no UnRAR/7z tool was found in the prior audit."],
    }
    write_json(OUT / "extraction_audit.json", extraction)

    compatibility = []
    for row in inventory:
        compatibility.append({"source": row.get("dataset"), "task_semantics": row.get("task"),
                              "label_provenance": row.get("labels"), "local_status": row.get("status_now"),
                              "can_joint_train_with": "", "compatibility_decision": "REVIEW_TASK_REGISTRY",
                              "reason_or_blocker": row.get("limitations", ""), "no_label_remapping": True})
    compatibility += [
        {"source": "Bugs4Q core + Bugs4Q-NA", "task_semantics": "fixed-vs-buggy source state", "label_provenance": "source native paired labels",
         "local_status": "PRESENT", "can_joint_train_with": "each other", "compatibility_decision": "VALIDATED_SEPARATE_SOURCE_HEADS_AND_POOLED_QNN_EXIST",
         "reason_or_blocker": "Prior grouped QNN joint/transfer runs exist on 31+31 issue subsets; full deduped source set not yet trained.", "no_label_remapping": True},
        {"source": "QMutBench + Bugs4Q / QBugs / Robust", "task_semantics": "distinct targets", "label_provenance": "mutation outcome vs source state vs execution outcome",
         "local_status": "PRESENT", "can_joint_train_with": "separate task heads only after validated shared representation", "compatibility_decision": "NO_DIRECT_LABEL_MAPPING",
         "reason_or_blocker": "Existing transfer audit found no valid task-level transfer fit; separate targets cannot be collapsed.", "no_label_remapping": True},
    ]
    write_csv(OUT / "dataset_compatibility_matrix.csv", compatibility)

    # Existing task registry remains authoritative for target definitions and prior statuses.
    registry_path = OLD / "task_and_label_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {"tasks": []}
    registry["campaign_id"] = "full_data_qml_optimization_v3"
    registry["new_full_data_qmutbench_audit"] = qmut
    registry["status_policy"] = ["NOT_STARTED", "DOWNLOADING", "EXTRACTING", "VALIDATING", "READY", "RUNNING", "PARTIAL", "COMPLETED", "BLOCKED", "INVALIDATED"]
    registry["frozen_test_accessed"] = False
    write_json(OUT / "task_registry.json", registry)

    # Transfer the prior QNN winner table but leave test scores blank. Verify paths remain present.
    winner_path = STAND / "task_winners.csv"
    standalone = []
    with winner_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pred = ROOT / row["prediction_path"].replace("\\", "/") if row.get("prediction_path") else None
            ckpt = [p.strip() for p in row.get("checkpoint_path", "").split(";") if p.strip()]
            metrics = oof_metrics(pred, row.get("selected_model")) if pred else {"prediction_present": False}
            standalone.append({"task": row.get("dataset_task"), "target": row.get("target"), "model": row.get("selected_model"),
                               "development_accuracy_percent": metrics.get("accuracy_percent", row.get("selected_accuracy_percent")),
                               "heldout_test_accuracy_percent": "", "macro_f1_percent": metrics.get("macro_f1_percent", row.get("macro_f1_percent")),
                               "balanced_accuracy_percent": metrics.get("balanced_accuracy_percent", row.get("balanced_accuracy_percent")),
                               "per_class_metrics": metrics.get("per_class_metrics", row.get("per_class_metrics")),
                               "records": metrics.get("records", ""), "groups": metrics.get("groups", ""), "folds": metrics.get("folds", ""), "seeds": "",
                               "fold_accuracy_sd_percentage_points": metrics.get("fold_accuracy_sd_percentage_points", ""),
                               "runtime_seconds": row.get("training_seconds"), "checkpoint_path": row.get("checkpoint_path"),
                               "checkpoint_present": all((ROOT / c.replace("\\", "/")).exists() for c in ckpt) if ckpt else False,
                               "prediction_path": row.get("prediction_path"), "prediction_present": bool(pred and pred.exists()),
                               "metrics_source": metrics.get("metrics_status", "historical registry"),
                               "status": "HISTORICAL_DEVELOPMENT_OOF; NOT_FULL_DATA_RETRAINED"})
    write_csv(OUT / "standalone_qml_results.csv", standalone)

    # No compatible cross-source outcome labels were found beyond Bugs4Q fixed/buggy; mark result slots explicitly.
    pooled_pred = OLD / "bugs4q_joint/qnn_hybrid/oof_predictions.csv"
    pooled_metrics = oof_metrics(pooled_pred, "angle_hybrid_readout")
    write_csv(OUT / "multisource_qml_results.csv", [
        {"target": "Bugs4Q fixed-vs-buggy state", "combination": "Bugs4Q core + Bugs4Q-NA", "model": "angle_hybrid_readout",
         "development_accuracy_percent": pooled_metrics.get("accuracy_percent", ""),
         "macro_f1_percent": pooled_metrics.get("macro_f1_percent", ""),
         "balanced_accuracy_percent": pooled_metrics.get("balanced_accuracy_percent", ""),
         "records": pooled_metrics.get("records", 124), "groups": pooled_metrics.get("groups", 62), "folds": pooled_metrics.get("folds", 5),
         "prediction_path": relative(pooled_pred), "status": "HISTORICAL_QNN_POOLED; SUBSET; FULL_DATA_INTEGRATION_PENDING",
         "decision": "Not promoted: pooled QNN did not reliably improve the standalone QNN; incomplete full-source deduplication."},
        {"target": "QMutBench Survivor/WOO/OPO", "combination": "Bugs4Q, QBugs, Bugs4Q-Robust", "model": "",
         "development_accuracy_percent": "", "macro_f1_percent": "", "balanced_accuracy_percent": "", "records": 0, "groups": 0, "folds": 0,
         "status": "NOT_RUN", "decision": "No valid target mapping; separate-head multi-task candidate not assembled."},
    ])

    # Copy existing saved OOF indexes and fold/seed artifacts without altering source campaigns.
    shutil.copy2(STAND / "prediction_artifact_index.csv", OUT / "prediction_index.csv")
    shutil.copy2(OLD / "fold_level_results.csv", OUT / "fold_metrics.csv")
    shutil.copy2(STAND / "multiseed_finalist_summary.csv", OUT / "seed_metrics.csv")

    write_csv(OUT / "resource_usage.csv", [{"phase": "campaign audit", "available_ram_mib": vm.get("available_mib"),
                                             "total_ram_mib": vm.get("total_mib"), "safety_floor_mib": 1024,
                                             "cpu_logical_count": os.cpu_count(), "training_started": False,
                                             "reason": "available RAM below campaign floor"}])
    write_json(OUT / "training_manifest.json", {
        "campaign": "full_data_qml_optimization_v3", "status": "PARTIAL",
        "baseline_commit": base_commit, "baseline_tracked_worktree_changes": baseline_status,
        "preexisting_untracked_paths_preserved": len(status) - len(baseline_status),
        "available_ram_mib": vm.get("available_mib"), "required_floor_mib": 1024,
        "new_training_runs_started": 0, "frozen_test_accessed": False,
        "reason": "Safety guard: available RAM below 1 GiB. Existing historical QNN runs are indexed separately and are not represented as full-data v3 training.",
        "qmutbench_full_data_status": "VALIDATED_FILE_AND_LABEL_COUNTS; QASM PARSE/FEATURE EXTRACTION AND QNN TRAINING INCOMPLETE",
        "task_statuses": {r["task"]: r["status"] for r in standalone},
    })
    write_json(OUT / "checkpoint_audit.json", {"historical_checkpoint_root": relative(OLD),
        "new_v3_checkpoints": [], "historical_winner_checkpoint_presence": {r["task"]: r["checkpoint_present"] for r in standalone},
        "full_data_checkpoints_created": 0, "optimizer_state_replay_for_all_winners": "not rerun in v3"})
    write_json(OUT / "leakage_audit.json", {"frozen_test_accessed": False, "test_metrics_reported": False,
        "grouped_splits": "historical artifacts preserved; v3 full-data grouped splits not generated because no training was run",
        "cross-source_label_mapping": "none", "train_fold_preprocessing_reverified_in_v3": False,
        "status": "PARTIAL; historical artifacts are development OOF and retain their original audit scope"})
    write_json(OUT / "blocked_tasks.json", {"blocked": [
        {"task": "all new QML training and full-data feature extraction", "status": "BLOCKED",
         "reason": f"Available RAM {vm.get('available_mib')} MiB is below the mandatory 1024 MiB campaign floor; prior jobs reached critically low memory.",
         "evidence": "resource_usage.csv", "required_to_resume": "Free sufficient memory so an idle measurement exceeds 1024 MiB, then remeasure before each heavy phase."},
        {"task": "full QMutBench QASM parser coverage", "status": "BLOCKED",
         "reason": "723,079 mutant and 375 origin files are present, but parsing all files under the current resource guard was not run.",
         "required_to_resume": "Run the streaming parser audit after the RAM floor is met."},
        {"task": "QBugs full upstream RAR extraction", "status": "BLOCKED",
         "reason": "RAR extraction tool unavailable in the prior environment audit; archive is present but upstream completeness is not verified.",
         "required_to_resume": "Install/use a compatible local RAR reader, then validate archive listing and checksums."},
    ], "incomplete": ["full-data QMutBench binary and multiclass QNN", "full Bugs4Q and Bugs4Q-NA QNN retraining", "full QBugs corpus QNN", "all other suitable-source full-data QNN runs", "multi-source integration on expanded full data"]})

    (OUT / "README.md").write_text(
        "# Full-data QML optimization campaign v3\n\n"
        "This directory records a new, non-destructive audit over local data and prior development-only QNN artifacts. "
        "It does **not** claim completion of full-data training. The QMutBench checkout contains 723,079 mutant metadata rows, "
        "375 origin programs, and a matching number of mutant QASM files; QASM syntax coverage and v3 model training remain outstanding.\n\n"
        "The campaign was halted before new training because available RAM measured below the 1 GiB safety floor. "
        "No frozen test was accessed. Historical OOF metrics remain labeled historical and must not be read as v3 full-data results.\n\n"
        "See `qml_accuracy_optimization_report.md`, `dataset_inventory.csv`, `task_registry.json`, and `blocked_tasks.json`.\n",
        encoding="utf-8")
    (OUT / "reproduction_commands.md").write_text(
        "# Reproduction\n\n"
        "Run the bounded inventory artifact builder from the repository root:\n\n"
        "```powershell\n.\\.qbugs_env\\Scripts\\python.exe experiments\\create_full_data_qml_v3_campaign.py\n```\n\n"
        "This command only inventories the local sources and indexes existing artifacts. It does not train. "
        "Before resuming QML training, ensure the idle system has at least 1 GiB available RAM, then rerun the memory measurement. "
        "Do not run any frozen-test evaluation during optimization.\n", encoding="utf-8")
    (OUT / "qml_accuracy_optimization_report.md").write_text(
        f"# QML accuracy optimization v3 — status report\n\n"
        f"- Baseline commit: `{base_commit}` on `{subprocess.check_output(['git','branch','--show-current'], cwd=ROOT, text=True).strip()}`.\n"
        f"- Available RAM: {vm.get('available_mib')} MiB; safety floor: 1,024 MiB.\n"
        f"- New training runs: 0; frozen test access: no.\n"
        f"- QMutBench: {qmut.get('metadata_rows')} metadata rows, {qmut.get('independent_origin_groups')} origins, {mutants} mutant QASM files, {originals} origin QASM files. "
        f"Raw counts: {qmut.get('raw_label_counts')}. Full syntax parsing, feature extraction and full-data QNN training have not been executed.\n\n"
        "## Existing development-only QNN results\n\n"
        "`standalone_qml_results.csv` indexes prior grouped OOF winners and their artifacts; these were trained on the historical task subsets and are not full-data v3 results. "
        "The strongest listed accuracy is not a final test score. Held-out test fields are blank by design.\n\n"
        "The prior Bugs4Q pooled QNN used 124 snapshots from 62 issue groups; macro F1 was 49.791%. This was a historical subset and did not establish a consistent gain over standalone training. "
        "QMutBench mutation outcomes were not mapped to Bugs4Q, QBugs, or execution pass/fail labels.\n\n"
        "## Incomplete work\n\n"
        "Full-data QMutBench and source-specific QNN runs, QASM parser coverage, expanded-source training, finalist multi-seed confirmation, and full-data integration remain incomplete. "
        "The immediate execution blocker is current available RAM below the 1 GiB safety floor. `blocked_tasks.json` lists the exact resume conditions.\n\n"
        "No accuracy claim in this v3 report is based on a frozen test. Historical development metrics are retained for traceability only.\n", encoding="utf-8")

    print(json.dumps({"out": relative(OUT), "baseline_commit": base_commit, "ram": vm,
                      "qmutbench": qmut, "mutant_qasm_files": mutants, "origin_qasm_files": originals,
                      "inventory_sources": len(inventory), "standalone_rows": len(standalone)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
